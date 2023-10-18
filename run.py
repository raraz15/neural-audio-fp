# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import click
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def load_config(config_filepath: str):
    if os.path.exists(config_filepath):
        print(f'cli: Configuration from {config_filepath}')
        with open(config_filepath, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg
    else:
        sys.exit(f'cli: ERROR! Configuration file {config_filepath} is missing!!')

def print_config(cfg):
    os.system("")
    print('\033[36m' + yaml.dump(cfg, indent=4, width=120, sort_keys=False) +
          '\033[0m')
    return

@click.group()
def cli():
    """
    train-> generate-> evaluate.

    How to use each command: \b\n
        python run.py COMMAND --help

    """

    return

""" Train """
@cli.command()
@click.argument('config_path', required=True)
@click.option('--max_epoch', default=None, type=click.INT, help='Max epoch.')
@click.option('--deterministic', default=False, is_flag=True,
              help='Set the CUDA operaitions to be deterministic.')
def train(config_path, max_epoch, deterministic):
    """ Train a neural audio fingerprinter.

    ex) python run.py train CHECKPOINT_NAME --max_epoch=100

        # with custom config file
        python run.py train CHECKPOINT_NAME --max_epoch=100 -c CONFIG_NAME

    NOTE: If './LOG_ROOT_DIR/checkpoint/CHECKPOINT_NAME already exists, 
    the training will resume from the latest checkpoint in the directory.
    """

    from model.trainer import set_seed, set_global_determinism, trainer

    # Training settings
    set_seed()
    if deterministic:
        set_global_determinism()

    # Load the config file
    cfg = load_config(config_path)
    # Update the config file
    if max_epoch is not None:
        cfg['MODEL']['MAX_EPOCH'] = max_epoch
    print_config(cfg)

    # Train
    trainer(cfg)

""" Generate fingerprint (after training) """
@cli.command()
@click.argument('config_path', required=True)
@click.option('--checkpoint_dir', default='', type=click.STRING, 
              help="Directory containing the model checkpoint. If not specified, "
              "cfg['MODEL']['LOG_ROOT_DIR']/checkpoint/cfg['MODEL']['CHECKPOINT_NAME'] will be used.")
@click.option('--checkpoint_index', default=0, type=click.INT, 
            help="Checkpoint index. If not specified, the latest checkpoint " 
            "in the ./OUTPUT_ROOT_DIR/checkpoint/ will be loaded.")
@click.option('--source_root', '-s', default='', type=click.STRING, 
              help="Custom source root directory. The source must be 16-bit "
              "8 Khz mono WAV. This is only useful when constructing a database "
              "without synthesizing queries.")
@click.option('--output_root', '-o', default='', type=click.STRING, 
              help="Root directory where the generated embeddings (uncompressed) " 
              "will be stored. Default is OUTPUT_ROOT_DIR/CHECKPOINT_NAME defined in config.")
@click.option('--skip_dummy', default=False, is_flag=True, 
              help='Exclude dummy-DB from the default source.')
@click.option('--mixed_precision', default=False, is_flag=True,
              help='Use mixed precision during inference. The fingerprint '
              'will be saved in FP32 in both cases.')
def generate(config_path, checkpoint_dir, checkpoint_index, source_root, output_root, skip_dummy, mixed_precision):
    """ Generate fingerprints from a saved checkpoint.

    ex) python run.py generate CONFIG_PATH

    With custom source directory: \b\n
        python run.py generate CONFIG_PATH --source_root SOURCE_DIR

    • If CHECKPOINT_DIR is not specified, 
        cfg['MODEL']['LOG_ROOT_DIR']/checkpoint/cfg['MODEL']['CHECKPOINT_NAME'] will be used.
    • If CHECKPOINT_INDEX is not specified, the latest checkpoint in 
        cfg['MODEL']['LOG_ROOT_DIR']/checkpoint/cfg['MODEL']['CHECKPOINT_NAME'] will be used.
    • The default value for the fingerprinting source is [TEST_DUMMY_DB] and 
        [TEST_QUERY_DB] specified in config file. You can change the source
        by specifying the --source option.
    • The default value for the output root directory is
        cfg['MODEL']['LOG_ROOT_DIR']/emb/cfg['MODEL']['CHECKPOINT_NAME'].
        You can change the output root directory by specifying the --output_root option.

    """

    from model.utils.config_gpu_memory_lim import allow_gpu_memory_growth
    from model.generate import generate_fingerprint

    allow_gpu_memory_growth()

    # Load the config file
    cfg = load_config(config_path)
    # Generate fingerprints
    generate_fingerprint(cfg, 
                         checkpoint_dir=checkpoint_dir,
                         checkpoint_index=checkpoint_index,
                         source_root_dir=source_root,
                         output_root_dir=output_root,
                         skip_dummy=skip_dummy, 
                         mixed_precision=mixed_precision)

""" Search and evalutation """
@cli.command()
@click.argument('embedding_dir', required=True)
@click.option('--index_type', '-i', default='ivfpq', type=click.STRING,
              help="Index type must be one of {'L2', 'IVF', 'IVFPQ', "
              "'IVFPQ-RR', 'IVFPQ-ONDISK', HNSW'}")
@click.option('--test_seq_len', default='1 3 5 9 11 19', type=click.STRING,
              help="A set of different number of segments to test. "
              "Numbers are separated by spaces. Default is '1 3 5 9 11 19', "
              "which corresponds to '1s, 2s, 3s, 5s, 6s, 10s' with 1 sec "
              "segment duration and 0.5 sec hop duration.")
@click.option('--test_ids', '-t', default='./eval/test_ids_icassp2021.npy', type=click.STRING,
              help="One of {'all', 'equally_spaced', 'path/file.npy', (int)}. "
              "If 'all', test all IDs from the test. You can also specify a 1-D array "
              "file's location that contains the start indices to the the evaluation. "
              "Any numeric input N (int) > 0 will perform search test at random position "
              "(ID) N times. 'equally_spaced' will use boundary information to get an "
              "equal number of samples from each track. Default is 'path/file.npy'.")
@click.option('--nogpu', default=False, is_flag=True,
              help='Use this flag to use CPU only.')
def evaluate(embedding_dir, index_type, test_seq_len, test_ids, nogpu):
    """ Search and evalutation.

        ex) python run.py evaluate logs/emb/CHECKPOINT_NAME/CHECKPOINT_INDEX

    With options: \b\n

        ex) python run.py evaluate logs/emb/CHECKPOINT_NAME/CHECKPOINT_INDEX -i ivfpq -t 3000 --nogpu

    """

    from eval.eval_faiss import eval_faiss

    if nogpu:
        eval_faiss([embedding_dir, "--index_type", index_type, "--test_seq_len",
                    test_seq_len, "--test_ids", test_ids, "--nogpu"])
    else:
        eval_faiss([embedding_dir, "--index_type", index_type, "--test_seq_len",
                    test_seq_len, "--test_ids", test_ids])

if __name__ == '__main__':
    cli()
