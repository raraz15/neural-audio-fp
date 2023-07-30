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
    # parser.add_argument('eval_text', type=str,
    #                     help='Path to the text file containing eval audio paths.')
    # parser.add_argument('noise_text', type=str,
    #                     help='Path to the text file containing noise audio paths.')
    parser.add_argument('output_dir', type=str,
                        help='Path to the output directory. '
                        'The sampled frames will be written here.')
    parser.add_argument('--n_segments', type=int, default=59,
                        help='Number of segments to sample from each audio file.')
    parser.add_argument('--segment_duration', type=float, default=2.0,
                        help='Duration of each segment in seconds.')
    parser.add_argument('--seed', type=int, default=27,
                        help='Random seed.')
    args = parser.parse_args()

    # Set the random seed
    np.random.seed(args.seed)

    # Read the text files
    with open(args.train_text, "r") as f:
        train_paths = [line.strip() for line in f.readlines()]
    with open(args.val_text, "r") as f:
        val_paths = [line.strip() for line in f.readlines()]
    # with open(args.test_text, "r") as f:
    #     test_paths = [line.strip() for line in f.readlines()]

    # Create the output directories
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # TODO: normalize audio?
    T_min = args.segment_duration * args.n_segments
    L0 = np.floor(args.segment_duration * SAMPLE_RATE).astype(int)

    # Sample segments from each audio file for train and val sets
    for split, paths in zip(["train", "val"], [train_paths, val_paths]):

        # Count the number of total segments
        counter = 0

        # Create the split directory
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"Split Directory: {split_dir}")

        # Sample segments from each audio file
        for i,audio_path in enumerate(paths):

            # Create a directory for the audio segments
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            segments_dir = os.path.join(split_dir, audio_name[:2], audio_name)
            os.makedirs(segments_dir, exist_ok=True)

            # Print progress
            if (i+1) % 10000 == 0:
                print(f"{split}: [{i+1}/{len(paths)}]")

            # Load the audio and downsample to SAMPLE_RATE, use the lowest quality
            audio = es.MonoLoader(filename=audio_path, 
                                sampleRate=SAMPLE_RATE, 
                                resampleQuality=4)()

            # Check the audio duration
            T = len(audio)/SAMPLE_RATE
            if T < T_min:
                print(f"Warning: {audio_path} is too short ({T:.2f}sec)"
                      " for Train and Validation sampling.")
                continue
            N = np.floor(T*SAMPLE_RATE).astype(int)

            # Determine the remainder and the number of samples per segment
            N_cut = (N // args.n_segments) * args.n_segments
            L = N_cut // args.n_segments
            remainder = N - N_cut
            assert remainder < L, "remainder is too large"

            # Cut the signal into N_segments segments
            boundaries = np.linspace(0, N_cut, args.n_segments+1).astype(int)
            assert np.all(np.diff(boundaries) == L), "boundaries are not equally spaced"

            # Half of the remainder goes to the first segment 
            # the other half goes to the last segment
            boundaries += np.floor(remainder//2).astype(int)

            # Sample subsegments from each segment
            segment_offsets = np.random.randint(0, L-1-L0, size=args.n_segments)
            start_indices = boundaries[:-1] + segment_offsets
            end_indices = start_indices + L0
            assert np.all(end_indices < boundaries[1:]), "end indices are out of bounds"

            # Write the segments to disk
            for start,end in zip(start_indices, end_indices):
                segment = audio[start:end]
                segment_path = os.path.join(segments_dir, f"{start}_{end}.npy")
                with open(segment_path, "wb") as f:
                    np.save(f, segment)
                counter += 1

        print(f"{split}: [{i+1}/{len(paths)}]")
        print(f"Total number of successfully saved segments: {counter}")

    print("Done!")

    # for i,audio_path in enumerate(test_paths):