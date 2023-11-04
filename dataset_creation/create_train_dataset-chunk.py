""" Creates a development dataset only for training by sampling a consecutive 
audio chunk from the audio files in the corresponding text files. The chunks 
are saved as .wav files and their start and end indices in the original audio 
file are saved as a .npy file. We do not use multiprocessing here to preserve 
reproducibility."""

import os
import sys
import argparse

import numpy as np

import essentia.standard as es

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.utils.audio_utils import max_normalize, get_random_chunk

SEED = 27
np.random.seed(SEED)

def main(paths, split_dir, sample_rate, min_duration):

    print(f"Writing chunks to {split_dir}...")
    os.makedirs(split_dir, exist_ok=True)

    # log file for recording failed audio files
    log_path = os.path.join(split_dir, "log.txt")
    # Clear the log file if it already exists
    if os.path.exists(log_path):
        open(log_path, "w").close()

    # Calculate the minimum number of samples required for each audio file
    min_samples = int(min_duration * sample_rate)

    # Sample chunks from each audio file
    for i,audio_path in enumerate(paths):

        # Create a directory for the audio segments following the same
        # directory structure as the original discotube dataset
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.join(split_dir, audio_name[:2])
        os.makedirs(output_dir, exist_ok=True)

        # Create the path to save the chunk
        output_path = os.path.join(output_dir, audio_name+".wav")

        # Skip if the chunk already exists
        if os.path.isfile(output_path):
            print(f"Chunk already exists. Skipping {audio_path}.")
            continue

        try:
            # Load the audio and downsample to sample_rate, 
            # use the lowest quality for high speed downsampling
            audio = es.MonoLoader(filename=audio_path, 
                                sampleRate=sample_rate, 
                                resampleQuality=4)()
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print(f"Could not load the audio file. Skipping {audio_path}.")
            with open(log_path, "a") as o_f:
                o_f.write(f"Error on: {audio_path}\n{str(e)}\n\n")
            continue

        if len(audio) < min_samples:
            print(f"Audio duration is too short. Skipping {audio_path}.")
            continue

        # We peak normalized the audio with sox before
        # audio = max_normalize(audio)

        # Get a random min_samples chunk from the audio
        chunk, boundary = get_random_chunk(audio, min_samples)

        # Write the audio to disk as a .wav file
        es.MonoWriter(filename=output_path, 
                    format="wav", 
                    sampleRate=sample_rate)(chunk)

        # Save the chunk boundary next to the audio file
        np.save(os.path.join(output_dir, audio_name+".npy"), boundary)

        # Print progress
        if (i+1) % 10000 == 0 or (i+1) == len(paths) or i == 0:
            print(f"Processed [{i+1}/{len(paths)}]")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Extracts chunks from audio files.')
    parser.add_argument('train_text', 
                        type=str, 
                        help='Path to the text file containing train audio paths.')
    parser.add_argument('output_dir', 
                        type=str,
                        help='Path to the output directory. Sampled chunks will '
                        'be written here inside the corresponding partition. '
                        'The directory structure will be:\n'
                        'output_dir/dev/<split>/audio_name[:2]/audio_name.wav')
    parser.add_argument('--chunk_duration', 
                        type=float, 
                        default=30.0,
                        help='Duration of a chunk in seconds.')
    parser.add_argument("--sample_rate",
                        type=int,
                        default=8000,
                        help="Sample rate to use for audio files.")
    args = parser.parse_args()

    # Read the text files
    with open(args.train_text, "r") as f:
        train_paths = [line.strip() for line in f.readlines()]
    assert len(train_paths) > 0, "No files found for training."
    print(f"Number of training files: {len(train_paths)}")

    # Create the split directory
    split_dir = os.path.join(args.output_dir, "train")

    # Process the paths and write the chunks to disk
    main(train_paths, split_dir, args.sample_rate, args.chunk_duration)

    print("Done!")