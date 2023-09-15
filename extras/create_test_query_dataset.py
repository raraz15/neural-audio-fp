import os
import sys
import argparse

import numpy as np

import essentia.standard as es

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.utils.audio_utils import max_normalize

SEED = 27
np.random.seed(SEED)

def main(paths, split_dir, sample_rate, min_samples):

    print(f"Writing query_clean segments to {split_dir}...")

    # Sample segments from each audio file
    for i, audio_path in enumerate(paths):

        # Create a directory for the audio segments following the same
        # directory structure as the original discotube dataset
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.join(split_dir, audio_name[:2])
        os.makedirs(output_dir, exist_ok=True)

        # Create the path to save the segment
        output_path = os.path.join(output_dir, audio_name+".wav")

        # Skip if the segment already exists
        if os.path.isfile(output_path):
            print(f"Segment already exists. Skipping {audio_path}.")
            continue

        try:
            # Load the audio and downsample to args.sample_rate, 
            # use the lowest quality for high speed downsampling
            audio = es.MonoLoader(filename=audio_path, 
                                sampleRate=sample_rate, 
                                resampleQuality=4)()
        except KeyboardInterrupt:
            sys.exit()
        except:
            print(f"Could not load the audio file. Skipping {audio_path}.")
            continue

        if len(audio) < min_samples:
            print(f"Audio duration is too short. Skipping {audio_path}.")
            continue

        # Normalize the audio
        audio = max_normalize(audio)

        # Get a random min_samples chunk from the audio
        start = np.random.randint(0, len(audio) - min_samples)
        end = start + min_samples
        chunk = audio[start:end]
        boundary = np.array([start, end])

        # Write the audio to disk as a .wav file
        es.MonoWriter(filename=output_path, 
                    format="wav", 
                    sampleRate=8000)(chunk)

        # Save the boundary
        np.save(os.path.join(output_dir, audio_name+".npy"), boundary)

        # Print progress
        if (i+1) % 10000 == 0 or (i+1) == len(paths) or i == 0:
            print(f"Processed [{i+1}/{len(paths)}]")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Extracts segments from audio files.')
    parser.add_argument('query_text', 
                        type=str,
                        help='Path to the text file containing test_query audio paths.')
    parser.add_argument('--output_dir', 
                        type=str,
                        default="../data/",
                        help='Path to the output directory. Segments will be written '
                         'here as in single .npz file. The directory structure will be '
                         'output_dir/test/<split>/audio_name[:2]/audio_name.npz')
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
                        default=8,
                        help="Number of workers to use for parallel processing.")
    args = parser.parse_args()

    # Read the text files containing the audio paths
    query_paths = []
    with open(args.query_text, "r") as f:
        query_paths = [line.strip() for line in f.readlines()]
    assert len(query_paths) > 0, "No query files found."
    print(f"Number of query files: {len(query_paths)}")

    # Calculate the minimum number of samples required for each audio file
    if len(query_paths) > 0:
        assert args.query_chunk_duration > 0, "noise_chunk_duration should be positive."
        min_query_samples = int(args.query_chunk_duration * args.sample_rate)

    # Cut segments from each audio file for each set and write them to disk

    # Determine the output directory for this split
    split_dir = os.path.join(args.output_dir, "test", "query_clean")
    main(query_paths, split_dir, args.sample_rate, min_query_samples)

    print("Done!")