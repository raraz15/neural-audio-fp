import os
import argparse
import essentia.standard as es
import numpy as np

SAMPLE_RATE = 8000

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Extracts chunks from audio files.')
    parser.add_argument('train_text', type=str, 
                        help='Path to the text file containing train audio paths.')
    parser.add_argument('val_text', type=str, 
                        help='Path to the text file containing validation audio paths.')
    parser.add_argument('test_text', type=str,
                        help='Path to the text file containing test audio paths.')
    parser.add_argument('output_dir', type=str,
                        help='Path to the output directory. '
                        'The sampled frames will be written here.')
    parser.add_argument('--n_segments', type=int, default=61,
                        help='Number of segments to sample from each audio file.')
    parser.add_argument('--segment_duration', type=float, default=2.0,
                        help='Duration of each segment in seconds.')
    parser.add_argument('--seed', type=int, default=27,
                        help='Random seed.')
    args = parser.parse_args()

    # Read the text files
    with open(args.train_text, "r") as f:
        train_paths = [line.strip() for line in f.readlines()]
    with open(args.val_text, "r") as f:
        val_paths = [line.strip() for line in f.readlines()]
    with open(args.test_text, "r") as f:
        test_paths = [line.strip() for line in f.readlines()]

    # Create the output directories
    os.makedirs(args.output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    # TODO: normalize audio?
    T_min = args.segment_duration * args.n_segments
    # TODO: is ceil good?
    segment_length = np.ceil(args.segment_duration * SAMPLE_RATE)

    # Sample segments from each audio file for train and val sets
    for split, paths in zip(["train", "val", "test"], [train_paths, val_paths, test_paths]):

        for i,audio_path in enumerate(paths):

            if (i+1) % 10000 == 0:
                print(f"{split}: [{i+1}/{len(paths)}]")

            # Load the audio and downsample to SAMPLE_RATE, use the lowest quality
            audio = es.MonoLoader(filename=audio_path, 
                                sampleRate=SAMPLE_RATE,
                                sampleQuality=4)()

            # Check the audio duration
            T = len(audio) / SAMPLE_RATE
            if T < T_min:
                print(f"Warning: {audio_path} is too short ({T:.2f}) for Train and Validation sampling.")
                continue

            # Sample segments
            boundaries = np.linspace(0, T, args.n_segments+1)

        print(f"{split}: [{i+1}/{len(paths)}]")
    

    for i,audio_path in enumerate(test_paths):


