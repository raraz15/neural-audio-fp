import os
import argparse
import essentia.standard as es
import numpy as np

SAMPLE_RATE = 8000

def cut_segments(audio, L, H):
    """ Cut the audio into consecutive segments of length L with hop size H.
    Discards the last segment."""

    # Calculate the number of segments that can be cut from the audio
    N_cut = int(np.floor((len(audio) - L + H) / H))
    assert N_cut > 0, "audio is too short"
    remainder = len(audio) - L - (N_cut-1)*H
    assert remainder < L, "remainder should be smaller than L"

    # Cut the signal into N_segments segments and discard the remainder
    start_indices = np.arange(N_cut)*H
    end_indices = start_indices + L
    assert np.all(end_indices < len(audio)), "end times are out of bounds"

    # Get the segments
    segments = []
    for start,end in zip(start_indices, end_indices):
        segment = audio[start:end]
        segments.append([start, end, segment])

    return segments

# TODO: augmentation?
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Extracts segments from audio files.')
    parser.add_argument('query_text', 
                        type=str,
                        help='Path to the text file containing test_query audio paths.')
    parser.add_argument('noise_text', 
                        type=str,
                        help='Path to the text file containing test_noise audio paths.')
    parser.add_argument('output_dir', 
                        type=str,
                        help='Path to the output directory. Taken frames will be written '
                         'here as .npy files. The directory structure will be '
                         'output_dir/test/split/audio_name[:2]/audio_name/segment_start_segment_end.npy')
    parser.add_argument('--segment_duration', 
                        type=float, 
                        default=1.0,
                        help='Duration of each segment in seconds.')
    parser.add_argument('--hop_duration', 
                        type=float, 
                        default=0.5,
                        help='Hop duration of segments in seconds.')
    args = parser.parse_args()

    # Calculate the number of samples for each segment and hop
    L = int(args.segment_duration * SAMPLE_RATE)
    H = int(args.hop_duration * SAMPLE_RATE)
    assert L > H, "segment duration should be larger than hop duration"

    # Read the text files
    with open(args.query_text, "r") as f:
        query_paths = [line.strip() for line in f.readlines()]
    with open(args.noise_text, "r") as f:
        noise_paths = [line.strip() for line in f.readlines()]

    # Cut segments from each audio file for each set and write them to disk
    for split, paths in zip(["query_clean", "noise"], [query_paths, noise_paths]):

        # Count the number of total segments
        counter = 0

        # Create the split directory
        split_dir = os.path.join(args.output_dir, "test", split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"Split Directory: {split_dir}")

        # Sample segments from each audio file
        for i, audio_path in enumerate(paths):

            # Print progress
            if (i+1) % 10000 == 0:
                print(f"{split}: [{i+1}/{len(paths)}]")

            # Create a directory for the audio segments following the same
            # directory structure as the original discotube dataset
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            segments_dir = os.path.join(split_dir, audio_name[:2], audio_name)
            os.makedirs(segments_dir, exist_ok=True)

            # Load the audio and downsample to SAMPLE_RATE, 
            # use the lowest quality for high speed downsampling
            audio = es.MonoLoader(filename=audio_path, 
                                sampleRate=SAMPLE_RATE, 
                                resampleQuality=4)()

            # Cut the audio into segments
            try:
                segments = cut_segments(audio, L, H)
            except AssertionError as e:
                print(f"Skipping {audio_path}")
                print(e)
                continue

            # Write the segments to disk
            for start,end, segment in segments:
                segment_path = os.path.join(segments_dir, 
                                            f"{start}_{end}.npy")
                with open(segment_path, "wb") as f:
                    np.save(f, segment)
                counter += 1

        print(f"{split}: [{i+1}/{len(paths)}]")
        print(f"Total number of successfully saved segments: {counter}")

    print("Done!")