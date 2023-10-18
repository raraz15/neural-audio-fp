source /usr/local/conda/etc/profile.d/conda.sh

conda activate afp

#############################################################################

if [ $# == 0 ]; then
    echo "Description: Runs the full pipeline for a particular commit. 
        CUDA_VISIBLE_DEVICES=ID run.py train CONFIG_PATH
        CUDA_VISIBLE_DEVICES=ID run.py generate CONFIG_PATH
        CUDA_VISIBLE_DEVICES=ID run.py evaluate logs/emb/CHECKPOINT_NAME/EPOCH/"
    echo "Usage: $0 param1 param2 param3 param4 param5"
    echo "param1: GPU ID"
    echo "param2: config path"
    echo "param3: model name"
    echo "param4: epoch"
    echo "param5: commit SHA"
    exit 0
fi

#############################################################################
# Train

git checkout $5

CUDA_VISIBLE_DEVICES=$1 python run.py train $2

#############################################################################
# Generate Fingerprints

git checkout $5

CUDA_VISIBLE_DEVICES=$1 python run.py generate $2

#############################################################################
# Evaluate

git checkout $5

CUDA_VISIBLE_DEVICES=$1 python run.py evaluate "logs/emb/$3/$4/"

#############################################################################

echo "Done!"
