""" Creates a dummy test dataset. It is used for converting mp3 files to wav files adnd
normalizing the audio. We use multiprocessing to speed up the process."""

import os
import sys
import argparse
import multiprocessing

import essentia.standard as es

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.utils.audio_utils import max_normalize

def main(paths, split_dir, sample_rate, partition_idx):

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

        # Normalize the audio
        audio = max_normalize(audio)

        # Write the audio to disk as a .wav file
        es.MonoWriter(filename=output_path, 
                    format="wav", 
                    sampleRate=8000)(audio)

        # Print progress
        if (i+1) % 10000 == 0 or (i+1) == len(paths) or i == 0:
            print(f"Partition {partition_idx}: [{i+1}/{len(paths)}]")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Extracts segments from audio files.')
    parser.add_argument('noise_text', 
                        type=str,
                        help='Path to the text file containing test_noise audio paths.')
    parser.add_argument('--output_dir', 
                        type=str,
                        default="../data/",
                        help='Path to the output directory. Segments will be written '
                         'here as in single .npz file. The directory structure will be '
                         'output_dir/test/<split>/audio_name[:2]/audio_name.npz')
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
    noise_paths = []
    with open(args.noise_text, "r") as f:
        noise_paths = [line.strip() for line in f.readlines()]
    assert len(noise_paths) > 0, "No noise files found."
    print(f"Number of noise files: {len(noise_paths)}")

    # Determine the output directory for this split
    split_dir = os.path.join(args.output_dir, "test", "noise")
    print(f"Writing test noise segments to {split_dir}...")

    # Split the paths for parallel processing
    split_size = len(noise_paths) // args.num_workers
    processes = []
    for i in range(args.num_workers):
        start = i * split_size
        end = (i+1) * split_size if i < args.num_workers - 1 else len(noise_paths)
        process = multiprocessing.Process(target=main, 
                                          args=(noise_paths[start:end], 
                                          split_dir, 
                                          args.sample_rate, 
                                          i))
        process.start()
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("Done!")