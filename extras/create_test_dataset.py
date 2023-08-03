import os
import argparse
import essentia.standard as es
import numpy as np

def cut_segments(audio, L, H):
    """ Cut the audio into consecutive segments of length L with hop size H.
    Discards the last segment."""

    # Calculate the number of segments that can be cut from the audio
    N_cut = int(np.floor((len(audio) - L + H) / H))
    assert N_cut > 0, "audio is too short"
    remainder = len(audio) - L - (N_cut-1)*H
    assert remainder < L, "remainder should be smaller than L"

    # Cut the signal into segments and discard the remainder
    start_indices = np.arange(N_cut)*H
    end_indices = start_indices + L
    assert np.all(end_indices <= len(audio)), "end times are out of bounds"

    # Get the segments
    segments = []
    for start,end in zip(start_indices, end_indices):
        segment = audio[start:end]
        segments.append([start, end, segment])

    return segments

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
                        help='Path to the output directory. Frames will be written '
                         'here as in single .npy file. The directory structure will be '
                         'output_dir/test/<split>/audio_name[:2]/audio_name.npy')
    parser.add_argument('--segment_duration', 
                        type=float, 
                        default=1.0,
                        help='Duration of each segment in seconds.')
    parser.add_argument('--hop_duration', 
                        type=float, 
                        default=0.5,
                        help='Hop duration of segments in seconds.')
    parser.add_argument('--num_segments',
                        type=int,
                        default=120,
                        help='Number of segments to take from each file.')
    parser.add_argument("--sample_rate",
                        type=int,
                        default=8000,
                        help="Sample rate to use for audio files.")
    args = parser.parse_args()

    # Calculate the number of samples for each segment and hop
    L = int(args.segment_duration * args.sample_rate)
    H = int(args.hop_duration * args.sample_rate)
    assert L > H, "segment duration should be larger than hop duration"

    # Calculate the minimum number of samples required for each audio file
    min_samples = int((args.num_segments-1)*H + L)

    # Read the text files containing the audio paths
    with open(args.noise_text, "r") as f:
        noise_paths = [line.strip() for line in f.readlines()]
    with open(args.query_text, "r") as f:
        query_paths = [line.strip() for line in f.readlines()]

    # Cut segments from each audio file for each set and write them to disk
    for split, paths in zip(["noise", "query_clean"], [noise_paths, query_paths]):

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
            audio_dir = os.path.join(split_dir, audio_name[:2])
            os.makedirs(audio_dir, exist_ok=True)

            # Load the audio and downsample to args.sample_rate, 
            # use the lowest quality for high speed downsampling
            audio = es.MonoLoader(filename=audio_path, 
                                sampleRate=args.sample_rate, 
                                resampleQuality=4)()

            if len(audio) < min_samples:
                print(f"Skipping {audio_path}, duration is too short.")
                continue

            # Get a random min_samples segment from the audio
            start = np.random.randint(0, len(audio) - min_samples)
            audio = audio[start:start+min_samples]

            # Cut the audio into segments
            try:
                segments = np.array([s[2] for s in cut_segments(audio, L, H)])
            except AssertionError as e:
                print(f"Skipping {audio_path}")
                print(e)
                continue

            # Write the segments to disk as npy files
            segments_path = os.path.join(audio_dir, audio_name+".npy")
            with open(segments_path, "wb") as f:
                # Write as float16 to save disk space
                np.save(f, segments.astype(np.float16))
            counter += 1

        print(f"{split}: [{i+1}/{len(paths)}]")
        print(f"Total number of successfully segmented tracks: {counter}")

    print("Done!")