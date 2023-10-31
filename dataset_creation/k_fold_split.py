"""This script creates the k-fold split for the FMA dataset. The split is created
by first detecting the duration of each wav file in the dataset. Then, the wav files
are split into two sets: short and long. The long wav files are then split into k folds
and each fold is further split into train, val, and test_query sets. The short wav files
are put into the test_dummy_db."""

import os
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import glob
import json
import wave

import numpy as np

N_TRAIN = 10000
N_VAL = 1000
N_TEST_QUERY = 5000

T_MIN = 30

EXCLUDE = {"080601.wav"} # This file is corrupted

""" We fix the seed for reproducibility. However, due to the size of the dataset,
we cannot guarantee that the splits will be the same as the ones used in the paper. See
https://docs.python.org/3/library/random.html random.shuffle() for more details."""

SEED = 27
random.seed(SEED, version=2)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

if __name__ == "__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_dir", type=str, 
                        help="Path to the dataset directory containing wav files.")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Path to the output directory of the split information files."
                        "Defaults to the same directory as the wav_dir.")
    parser.add_argument("--num_folds", "-n", type=int, default=5,
                        help="Number of folds to create.")
    args = parser.parse_args()

    # Get the list of wav files
    wav_paths = glob.glob(os.path.join(args.wav_dir, "**", "*.wav"), recursive=True)
    print(f"Found {len(wav_paths):,} wav files.")
    # Make sure that the paths are absolute
    wav_paths = [os.path.abspath(path) for path in wav_paths]

    # Remove the excluded files convert to set
    wav_paths = set([path for path in wav_paths if os.path.basename(path) not in EXCLUDE])

    # Determine the output directory
    if args.output_dir is None:
        args.output_dir = args.wav_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Read each wav file and record their duration
    duration_file_path = os.path.join(args.output_dir, 'wav_durations.json')
    print(f"Writing the duration of each file to {duration_file_path}.")
    short_duration_wavs = set()
    with open(duration_file_path, 'w') as out_f:
        for wav_path in wav_paths:

            # Open the file without loading the data
            pt_wav = wave.open(wav_path, 'r')
            fs = pt_wav.getframerate()
            n_samples = pt_wav.getnframes()
            pt_wav.close()

            # Get the duration and fname of the wav file
            duration = n_samples/fs

            # Record the duration and sampling rate in a JSONL file
            out_f.write(json.dumps({
                                'wav_path': wav_path,
                                'duration(s)': duration,
                                'fs': fs}
                                )+"\n")

            # Record the wav file name if it is too short
            if duration < T_MIN:
                short_duration_wavs.update({wav_path})
    print(f"Found {len(short_duration_wavs):,} wav files shorter than {T_MIN} seconds.")

    # Get the list of long wav file names
    long_duration_wavs = list(wav_paths.difference(short_duration_wavs))
    print(f"{len(long_duration_wavs):,} wav files are longer.")

    # Make sure that we have enough data to create the folds
    n_per_fold = N_TRAIN + N_VAL + N_TEST_QUERY
    n_required = args.num_folds * (n_per_fold)
    assert n_required <= len(long_duration_wavs), \
        "The number of folds and the number of files per fold is too " \
        "large when short wav files are excluded."

    """Here we do a custom split of the data into folds. Since the total amount of 
    data is limited, we want keep a large amount of tracks for the test_dummy_db. 
    Therefore we fix the train, val, and test_query set sizes and make sure that 
    between all splits, each set is disjoint from itself."""

    print(f"Creating {args.num_folds:,} folds with {N_TRAIN:,} train, {N_VAL:,} val, "
          f"and {N_TEST_QUERY:,} test_query tracks.")

    # Shuffle the list of long wav files
    np.random.shuffle(long_duration_wavs)

    # Sample unique files for each fold (5 folds, 3 sets = 15 disjoint sets)
    mix_of_folds_and_sets = np.random.choice(long_duration_wavs, n_required, replace=False)

    for i in range(args.num_folds):

        # Get the data for the current fold
        fold = mix_of_folds_and_sets[i*n_per_fold:(i+1)*n_per_fold]

        # Split the fold data into train, val, and test_query
        train = fold[:N_TRAIN]
        val = fold[N_TRAIN:N_TRAIN+N_VAL]
        test_query = fold[N_TRAIN+N_VAL:]

        assert len(train) == N_TRAIN
        assert len(val) == N_VAL
        assert len(test_query) == N_TEST_QUERY

        # Remaining tracks are put into the test_dummy_db
        test_dummy = list(wav_paths.difference(fold))

        # Make sure that the test_dummy_db is disjoint from the other sets
        assert len(set(train).intersection(test_dummy)) == 0
        assert len(set(val).intersection(test_dummy)) == 0
        assert len(set(test_query).intersection(test_dummy)) == 0

        # Create the fold dir
        fold_dir = os.path.join(args.output_dir, str(i))
        os.makedirs(fold_dir, exist_ok=True)
        print(f"Writing the fold {i} data to {fold_dir}")
        
        # Write the data to text file
        train_file_path = os.path.join(fold_dir, 'train.txt')
        with open(train_file_path, 'w') as out_f:
            for line in train:
                out_f.write(f"{line}\n")

        val_file_path = os.path.join(fold_dir, 'val.txt')
        with open(val_file_path, 'w') as out_f:
            for line in val:
                out_f.write(f"{line}\n")

        test_query_file_path = os.path.join(fold_dir, 'test_query.txt')
        with open(test_query_file_path, 'w') as out_f:
            for line in test_query:
                out_f.write(f"{line}\n")

        test_dummy_file_path = os.path.join(fold_dir, 'test_dummy.txt')
        with open(test_dummy_file_path, 'w') as out_f:
            for line in test_dummy:
                out_f.write(f"{line}\n")

print("Done!")