source /usr/local/conda/etc/profile.d/conda.sh

conda activate afp

#############################################################################

if [ $# == 0 ]; then
    echo "Description: Runs the full pipeline for a particular commit. 
        CUDA_VISIBLE_DEVICES=ID run.py train checkpoint_name
        CUDA_VISIBLE_DEVICES=ID run.py generate checkpoint_name,
        CUDA_VISIBLE_DEVICES=ID run.py evaluate logs/emb/checkpoint_name/100/"
    echo "Usage: $0 param1 param2 param3 param4 param5"
    echo "param1: GPU ID"
    echo "param2: checkpoint_name"
    echo "param3: config_name"
    echo "param4: commit SHA"
    exit 0
fi

#############################################################################

git checkout dev

#############################################################################
# Train

git checkout $4

CUDA_VISIBLE_DEVICES=$1 python run.py train $2 --config=$3

#############################################################################
# Generate Fingerprints

git checkout $4

CUDA_VISIBLE_DEVICES=$1 python run.py generate $2 --config=$3

#############################################################################
# Evaluate 

git checkout $4

CUDA_VISIBLE_DEVICES=$1 python run.py evaluate "logs/emb/$2/100/"

#############################################################################

git checkout dev

#############################################################################

echo "Done!"
