#!/bin/bash

# This script splits the data into train, val, eval and noise sets.
# Usage: ./train_val_eval_split.sh <input_file>
# Example: ./train_val_eval_split.sh data.txt

if [ $# -lt 1 ]; then
  echo 1>&2 "$0: provide a .txt file to split"
  exit 2
elif [ $# -gt 1 ]; then
  echo 1>&2 "$0: too many arguments"
  exit 2
fi

#######################################################################
# Specify here the sizes of the train, val, and eval and noise sets

TRAIN=20000
VAL=4000
EVAL=4000
NOISE=160000

TOTAL=$(($TRAIN + $VAL + $EVAL + $NOISE))
echo "Total number of samples to take: $TOTAL"

# Check if the total number of samples to take is greater than the
# number of samples in the file
N=$(wc -l < $1)
if (( $TOTAL > $N )); then
    echo "Total number of samples to take is greater "\
            "than the number of samples in the file: $N"
    exit 2
fi

#######################################################################

# First shuffle all the data
output="$1.shuf"
echo "Shuffling the data..."
shuf $1 > $output
echo $output
echo

# Get the required number of samples from the shuffled data
input=$output
output="$1.shuf.$TOTAL"
echo "Getting $TOTAL samples..."
head -n $TOTAL $input > $output
echo $output
echo

#######################################################################

# Split the file into train, val, eval and noise sets
echo "Splitting the data into train, val, eval and noise sets..."
input=$output

l=0
while IFS= read -r line
do
    if (( $l < $TRAIN )); then
        echo "$line" >> "$input.train"
    elif (( $l < $TRAIN + $VAL )); then
        echo "$line" >> "$input.val"
    elif (( $l < $TRAIN + $VAL + $EVAL )); then
        echo "$line" >> "$input.eval"
    elif (( $l < $TRAIN + $VAL + $EVAL + $NOISE )); then
        echo "$line" >> "$input.noise"
    else
        :
    fi

    l=$(($l+1))

done < "$input"
