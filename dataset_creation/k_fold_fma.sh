#!/bin/bash

source /usr/local/conda/etc/profile.d/conda.sh
conda activate afp

#############################################################################

if [ $# == 0 ]; then
    echo "Description: Creates 5-fold dataset splits for the wav paths in the WAV_DIR 
    with dataset_creation/k_fold_split.py. Then for each fold, creates a train
    and test_query dataset."
    echo "Usage: $0 param1 param2"
    echo "param1: wav_dir"
    echo "param2: fold_dir"
    exit 0
fi

#############################################################################

# Create 5-fold dataset splits
echo "Creating 5-fold dataset splits"
python dataset_creation/k_fold_split.py $1 -o $2

# For each fold create train, val, test_query datasets
for i in {0..4}
do
    fold_dir="$2/$i"
    echo $fold_dir

    echo "Creating the training dataset for fold $i"
    python dataset_creation/create_train_dataset-chunk.py "$fold_dir/train.txt" "$fold_dir/"

    echo "Creating test_query dataset for fold $i"
    python dataset_creation/create_test_query_dataset.py "$fold_dir/test_query.txt" "$fold_dir/"

done

