import os
import argparse
import essentia.standard as es
import numpy as np

SAMPLE_RATE = 8000

SEED = 27
np.random.seed(SEED)

def cut_to_segments_and_sample(audio, T_min, L0, n_segments):
    """ First, cuts a given audio signal into n_segments non-overlapping, 
    consecutive segments of equal length. Then samples a random sub-segment 
    from each segment of length L0. The sub-segments are returned as a list, 
    together with their start and end indices in the original audio signal.
    
        Parameters
        ----------
        audio : np.array
            The audio signal to be cut and sampled.
        T_min : float
            The minimum duration of the audio signal in seconds.
        L0 : int
            The length of the sub-segments to be sampled in samples.
        n_segments : int
            The number of segments to cut the audio signal into.
        
        Returns
        -------
        segments : [ [start, end, segment], ... ]
            A list of lists containing the start and end indices of the
            sub-segments and the sub-segments themselves.

    """

    # Check the audio duration
    T = len(audio)/SAMPLE_RATE
    if T < T_min:
        raise ValueError(f"audio is too short ({T:.2f}sec) for Train and Validation sampling.")

    # Determine the remainder and the number of samples per segment
    N = np.floor(T*SAMPLE_RATE).astype(int)
    N_cut = (N // n_segments) * n_segments
    L = N_cut // n_segments
    remainder = N - N_cut
    assert remainder < L, "remainder is too large"

    # Cut the signal into N_segments non-overlapping, consecutive segments
    boundaries = np.linspace(0, N_cut, n_segments+1).astype(int)
    assert np.all(np.diff(boundaries) == L), "boundaries are not equally spaced"

    # Half of the remainder goes to the first segment and the other half goes to the last segment
    boundaries += np.floor(remainder//2).astype(int)

    # Sample n_segments, subsegments from each segment
    segment_offsets = np.random.randint(0, L-1-L0, size=n_segments)
    start_indices = boundaries[:-1] + segment_offsets
    end_indices = start_indices + L0
    assert np.all(end_indices <= boundaries[1:]), "end indices are out of bounds"

    # Get the segments and their start and end indices
    segments, boundaries = [], []
    for start,end in zip(start_indices, end_indices):
        segment = audio[start:end]
        segments.append(segment)
        boundaries.append([start, end])

    # Convert to numpy arrays
    segments = np.array(segments)
    boundaries = np.array(boundaries)

    return segments, boundaries

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Extracts chunks from audio files.')
    parser.add_argument('train_text', 
                        type=str, 
                        help='Path to the text file containing train audio paths.')
    parser.add_argument('val_text', 
                        type=str, 
                        help='Path to the text file containing validation audio paths.')
    parser.add_argument('output_dir', 
                        type=str,
                        help='Path to the output directory. Sampled segments will '
                        'be written here inside the corresponding partition. '
                        'The directory structure will be '
                        'output_dir/<split>/audio_name[:2]/audio_name/<segment_idx>.npz')
    parser.add_argument('--n_segments', 
                        type=int, 
                        default=59,
                        help='Number of segments to sample from each audio file.')
    parser.add_argument('--segment_duration', 
                        type=float, 
                        default=2.0,
                        help='Duration of a segment in seconds.')
    args = parser.parse_args()

    # Minimum audio duration for no overlap
    T_min = args.segment_duration * args.n_segments
    # Length of the sub-segments to be sampled
    L0 = np.floor(args.segment_duration * SAMPLE_RATE).astype(int)

    # Read the text files
    with open(args.train_text, "r") as f:
        train_paths = [line.strip() for line in f.readlines()]
    print(f"Number of training files: {len(train_paths)}")
    with open(args.val_text, "r") as f:
        val_paths = [line.strip() for line in f.readlines()]
    print(f"Number of validation files: {len(val_paths)}")

    # Sample segments from each audio file for train and val sets
    for split, paths in zip(["train", "val"], [train_paths, val_paths]):

        if len(paths) == 0:
            print(f"No files found for {split}, skipping...")
            continue

        # Count the number of total segments for this split
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

            try:
                # Load the audio and downsample to SAMPLE_RATE, use the lowest quality
                audio = es.MonoLoader(filename=audio_path, 
                                    sampleRate=SAMPLE_RATE, 
                                    resampleQuality=4)()
            except:
                print(f"Could not load the audio file. Skipping {audio_path}.")
                continue

            try:
                segments, boundaries = cut_to_segments_and_sample(audio, 
                                                                T_min, 
                                                                L0, 
                                                                args.n_segments)
            except ValueError as e:
                print(e)
                print(f"Skipping {audio_path}.")
                continue

            # Write the segments to disk as separate .npz files (needed for our training scheme)
            for j, (segment, boundary)  in enumerate(zip(segments, boundaries)):
                np.savez(os.path.join(segments_dir, f"{j}.npz"), 
                         segment=segment.astype(np.float16), # Write as float16 to save disk space
                         boundary=boundary.astype(np.int32) # Write the boundaries together with the segments
                         )
                counter += 1

            # Print progress
            if (i+1) % 10000 == 0 or (i+1) == len(paths) or i == 0:
                print(f"{split}: [{i+1}/{len(paths)}]")

        print(f"{split}: [{i+1}/{len(paths)}]")
        print(f"Total number of successfully saved segments: {counter}")

    print("Done!")