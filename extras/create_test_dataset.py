import os
import argparse

import numpy as np

import essentia.standard as es

SEED = 27
np.random.seed(SEED)

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
                        help='Path to the output directory. Segments will be written '
                         'here as in single .npz file. The directory structure will be '
                         'output_dir/test/<split>/audio_name[:2]/audio_name.npz')
    parser.add_argument('--noise_chunk_duration',
                        type=float,
                        default=150.,
                        help='Duration of a noise track chunk in seconds.')
    parser.add_argument('--query_chunk_duration',
                        type=float,
                        default=30.,
                        help='Duration of a query track chunk in seconds.')
    parser.add_argument("--sample_rate",
                        type=int,
                        default=8000,
                        help="Sample rate to use for audio files.")
    args = parser.parse_args()

    # Read the text files containing the audio paths
    noise_paths, query_paths = [], []
    if args.noise_text is not None:
        with open(args.noise_text, "r") as f:
            noise_paths = [line.strip() for line in f.readlines()]
    print(f"Number of noise files: {len(noise_paths)}")
    if args.query_text is not None:
        with open(args.query_text, "r") as f:
            query_paths = [line.strip() for line in f.readlines()]
    print(f"Number of query files: {len(query_paths)}")

    # Calculate the minimum number of samples required for each audio file
    assert args.noise_chunk_duration > 0, "noise_chunk_duration should be positive."
    min_noise_samples = int(args.noise_chunk_duration * args.sample_rate)
    assert args.query_chunk_duration > 0, "noise_chunk_duration should be positive."
    min_query_samples = int(args.query_chunk_duration * args.sample_rate)

    # Cut segments from each audio file for each set and write them to disk
    for split, paths, min_samples in zip(["query_clean", "noise"], [query_paths, noise_paths], [min_query_samples, min_noise_samples]):

        # Count the number of total segments for this split
        counter = 0

        # Determine the output directory for this split
        split_dir = os.path.join(args.output_dir, "test", split)
        print(f"Writing {split} segments to {split_dir}...")

        if len(paths) == 0:
            print(f"No files found for {split}, skipping...")
            continue

        # Sample segments from each audio file
        for i, audio_path in enumerate(paths):

            # Create a directory for the audio segments following the same
            # directory structure as the original discotube dataset
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            audio_dir = os.path.join(split_dir, audio_name[:2])
            os.makedirs(audio_dir, exist_ok=True)

            try:
                # Load the audio and downsample to args.sample_rate, 
                # use the lowest quality for high speed downsampling
                audio = es.MonoLoader(filename=audio_path, 
                                    sampleRate=args.sample_rate, 
                                    resampleQuality=4)()
            except:
                print(f"Could not load the audio file. Skipping {audio_path}.")
                continue

            if len(audio) < min_samples:
                print(f"Audio duration is too short. Skipping {audio_path}.")
                continue

            # Get a random min_samples chunk from the audio
            start = np.random.randint(0, len(audio) - min_samples)
            end = start + min_samples
            chunk = audio[start:end].astype(np.float16) # Store as float16 to save space
            boundary = np.array([start, end]).astype(np.int32)

            # Write the chunk to disk as a npy file
            save_path = os.path.join(audio_dir, audio_name+".npz")
            np.savez(save_path, 
                     segment=chunk,
                     boundary=boundary # Store the start and end indices of the chunk
                     )
            # Increment the counter for successfully segmented tracks
            counter += 1

            # Print progress
            if (i+1) % 10000 == 0 or (i+1) == len(paths) or i == 0:
                print(f"{split}: [{i+1}/{len(paths)}]")

        print(f"Total number of successfully taken chunks: {counter}")

    print("Done!")