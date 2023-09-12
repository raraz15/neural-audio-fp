import os
import sys
import argparse
import multiprocessing

import numpy as np

import essentia.standard as es

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.utils import max_normalize

SEED = 27
np.random.seed(SEED)


def main(paths, split_dir, sample_rate, min_samples, partition_idx):

    # Sample segments from each audio file
    for i, audio_path in enumerate(paths):

        # Create a directory for the audio segments following the same
        # directory structure as the original discotube dataset
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        audio_dir = os.path.join(split_dir, audio_name[:2])
        os.makedirs(audio_dir, exist_ok=True)

        # Create the path to save the segment
        save_path = os.path.join(audio_dir, audio_name+".npz")
        # Skip if the segment already exists
        if os.path.isfile(save_path):
            print(f"Segment already exists. Skipping {audio_path}.")
            continue

        try:
            # Load the audio and downsample to args.sample_rate, 
            # use the lowest quality for high speed downsampling
            audio = es.MonoLoader(filename=audio_path, 
                                sampleRate=sample_rate, 
                                resampleQuality=4)()
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print(f"Could not load the audio file. Skipping {audio_path}.")
            continue

        # Normalize the audio
        audio = max_normalize(audio)

        if len(audio) < min_samples:
            print(f"Audio duration is too short. Skipping {audio_path}.")
            continue

        # Get a random min_samples chunk from the audio
        start = np.random.randint(0, len(audio) - min_samples)
        end = start + min_samples
        chunk = audio[start:end].astype(np.float16) # Store as float16 to save space
        boundary = np.array([start, end]).astype(np.int32)

        # Write the chunk to disk as a npy file
        np.savez(save_path, 
                segment=chunk,
                boundary=boundary # Store the start and end indices of the chunk
                )

        # Print progress
        if (i+1) % 10000 == 0 or (i+1) == len(paths) or i == 0:
            print(f"Partition {partition_idx}: [{i+1}/{len(paths)}]")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Extracts segments from audio files.')
    parser.add_argument('--query_text', 
                        type=str,
                        default=None,
                        help='Path to the text file containing test_query audio paths.')
    parser.add_argument('--noise_text', 
                        type=str,
                        default=None,
                        help='Path to the text file containing test_noise audio paths.')
    parser.add_argument('--output_dir', 
                        type=str,
                        default="../data/",
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
    parser.add_argument("--num_workers",
                        type=int,
                        default=multiprocessing.cpu_count(),
                        help="Number of workers to use for parallel processing.")
    args = parser.parse_args()

    assert args.query_text is not None or args.noise_text is not None, \
        "At least one of --query_text or --noise_text should be provided."

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
    min_noise = int(args.noise_chunk_duration * args.sample_rate)
    assert args.query_chunk_duration > 0, "noise_chunk_duration should be positive."
    min_query = int(args.query_chunk_duration * args.sample_rate)

    # Cut segments from each audio file for each set and write them to disk
    for split, paths, min_samples in [("query_clean", query_paths, min_query), ("noise", noise_paths, min_noise)]:

        if len(paths) == 0:
            continue

        # Determine the output directory for this split
        split_dir = os.path.join(args.output_dir, "test", split)
        print(f"Writing {split} segments to {split_dir}...")

        # Split the paths for parallel processing
        split_size = len(paths) // args.num_workers
        processes = []
        for i in range(args.num_workers):
            start = i * split_size
            end = (i+1) * split_size if i < args.num_workers - 1 else len(paths)
            process = multiprocessing.Process(target=main, 
                                            args=(paths[start:end], 
                                                split_dir, 
                                                args.sample_rate, 
                                                min_samples,
                                                i))
            process.start()
            processes.append(process)

    print("Done!")