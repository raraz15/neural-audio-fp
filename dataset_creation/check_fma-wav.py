import os
import sys
import json
import argparse
import numpy as np
import glob

import essentia.standard as es

FMA_DIR = "/mnt/mtgdb-audio/incoming/fma"
OUTPUT_DIR = "/home/oaraz/nextcore/fingerprinting/datasets/fma_wav_8k"

if __name__ == "__main__":

    audio_paths = sorted(glob.glob(os.path.join(FMA_DIR, "**", "*.mp3"), 
                        recursive=True))
    print(len(audio_paths))

    # Audio file metadata
    jsonl_path = os.path.join(OUTPUT_DIR, "analysis-mp3_Audioloader.json")
    # log file for recording failed audio files
    log_path = os.path.join(OUTPUT_DIR, "log-mp3_Audioloader.txt")

    for i,audio_path in enumerate(audio_paths):

        try:
            _, sampleRate, numberChannels, _, bit_rate, codec = es.AudioLoader(filename=audio_path)()
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print(f"Could not load the audio file. Skipping {audio_path}.")
            with open(log_path, "a") as o_f:
                o_f.write(f"Error on: {audio_path}\n{str(e)}\n\n")
            continue

        with open(jsonl_path, "a") as out_f:
            json.dump({"wav_name": os.path.basename(audio_path),
                        "sampleRate": sampleRate,
                        "numberChannels": numberChannels,
                        "bit_rate": bit_rate,
                        "codec": codec}, out_f)
            out_f.write("\n")

        # Print progress
        if (i+1) % 10000 == 0 or (i+1) == len(audio_paths) or i == 0:
            print(f"Processed [{i+1}/{len(audio_paths)}]")
