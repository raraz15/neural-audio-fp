import os
import json
import argparse

DURATION_THRESHOLD = 420

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", 
                        type=str, 
                        help="input .txt file. Should contain lines of .mp4 file paths.")
    args = parser.parse_args()

    output = args.input+".filtered"
    print(output)
    if os.path.exists(output):
        print(f"{output} already exists. Exiting...")
        exit()

    print(f"Filtering {args.input}...")
    with open(args.input, "r") as in_f, open(output, "w") as out_f:
        for i,mp4_path in enumerate(in_f.readlines()):
            if (i+1) % 10000 == 0:
                print(f"Processed {i+1} files.")
            meta_path = mp4_path[:-4]+"meta"
            with open(meta_path, "r") as meta_f:
                metadata = json.load(meta_f)
            if metadata["duration"] < DURATION_THRESHOLD:
                out_f.write(mp4_path)
        print(f"Processed {i+1} files.")
    print("Done!")