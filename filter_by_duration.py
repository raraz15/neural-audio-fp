import os
import json
import argparse

DURATION_THRESHOLD = 420

def process(mp4_path, out_f):

    meta_path = mp4_path[:-4]+"meta"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as meta_f:
            metadata = json.load(meta_f)
        if metadata["duration"] < DURATION_THRESHOLD:
            out_f.write(mp4_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", 
                        type=str, 
                        help="input .txt file. Should contain lines of .mp4 file paths.")
    parser.add_argument("--continue-read",
                        action="store_true",
                        help="If specified, will continue from the last processed file "
                            "pointing to input.filtered.")
    args = parser.parse_args()

    output = args.input+".filtered"
    print(f"The filtered file will be saved to {output}")

    if args.continue_read:

        print("Continuing from the last processed file...")

        # Find the last processed file
        with open(output, "r") as f:
            last_path = f.readlines()[-1]
        print(last_path)

        # Find the index of the last processed file in args.input
        found = False
        with open(args.input, "r") as f:
            for i,line in enumerate(f.readlines()):
                if line == last_path:
                    print(f"Continuing from {i+1}.")
                    found = True
                    break
        if not found:
            raise ValueError(f"Cannot find {last_path} in {args.input}")

        # Process the rest of the files
        with open(args.input, "r") as in_f, open(output, "a") as out_f:
            lines = in_f.readlines()
            for j,mp4_path in enumerate(lines[i+1:]):
                if (j+1) % 10000 == 0:
                    print(f"Processed {j} files [{j+i+1}/{len(lines)}].")
                try:
                    process(mp4_path, out_f)
                except:
                    print(f"Error processing {mp4_path}")
            print(f"Processed {j} files [{j+i+1}/{len(lines)}].")

    else:
        print(f"Filtering {args.input}...")
        with open(args.input, "r") as in_f, open(output, "w") as out_f:
            for i,mp4_path in enumerate(in_f.readlines()):
                if (i+1) % 10000 == 0:
                    print(f"Processed {i+1} files.")
                process(mp4_path, out_f)
            print(f"Processed {i+1} files.")

    print("Done!")