# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import click
import yaml
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def load_config(config_fname):
    config_filepath = './config/' + config_fname + '.yaml'
    if os.path.exists(config_filepath):
        print(f'cli: Configuration from {config_filepath}')
    else:
        sys.exit(f'cli: ERROR! Configuration file {config_filepath} is missing!!')
    with open(config_filepath, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def update_config(cfg, key1: str, key2: str, val):
    cfg[key1][key2] = val
    return cfg

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
    pass

""" Train """
@cli.command()
@click.argument('checkpoint_name', required=True)
@click.option('--config', '-c', default='default', type=click.STRING,
              help="Name of model configuration located in './config/.'")
@click.option('--max_epoch', default=None, type=click.INT, help='Max epoch.')
@click.option('--deterministic', default=False, is_flag=True,
              help='Set the CUDA operaitions to be deterministic.')
def train(checkpoint_name, config, max_epoch, deterministic):
    """ Train a neural audio fingerprinter.

    ex) python run.py train CHECKPOINT_NAME --max_epoch=100

        # with custom config file
        python run.py train CHECKPOINT_NAME --max_epoch=100 -c CONFIG_NAME

    NOTE: If './LOG_ROOT_DIR/checkpoint/CHECKPOINT_NAME already exists, 
    the training will resume from the latest checkpoint in the directory.
    """

    from model.trainer import set_seed, set_global_determinism, trainer

    set_seed()
    if deterministic:
        set_global_determinism()
    cfg = load_config(config)
    if max_epoch:
        update_config(cfg, 'TRAIN', 'MAX_EPOCH', max_epoch)
    # Write the model name inside the config file
    update_config(cfg, 'MODEL', 'NAME', checkpoint_name)
    print_config(cfg)
    trainer(cfg, checkpoint_name)

""" Generate fingerprint (after training) """
@cli.command()
@click.argument('checkpoint_name', required=True)
@click.option('--checkpoint_type', default='custom', type=click.STRING,
              help="Checkpoint type must be one of {'best', 'custom'}. Default is 'custom'.")
@click.option('--checkpoint_index', default=None, type=click.INT,
            help="Checkpoint index. If not specified, the latest checkpoint " +
            "in the OUTPUT_ROOT_DIR will be loaded.")
@click.option('--config', '-c', default='default', type=click.STRING,
              help="Name of the model configuration file located in 'config/'." +
              " Default is 'default'")
@click.option('--source', '-s', default=None, type=click.STRING, required=False,
              help="Custom source root directory. The source must be 16-bit "
              "8 Khz mono WAV. This is only useful when constructing a database"
              " without synthesizing queries.")
@click.option('--output', '-o', default=None, type=click.STRING,
              help="Root directory where the generated embeddings (uncompressed)" +
              " will be stored. Default is OUTPUT_ROOT_DIR/CHECKPOINT_NAME " +
              "defined in config.")
@click.option('--skip_dummy', default=False, is_flag=True,
              help='Exclude dummy-DB from the default source.')
def generate(checkpoint_name, checkpoint_type, checkpoint_index, config, source, output, skip_dummy):
    """ Generate fingerprints from a saved checkpoint.

    ex) python run.py generate CHECKPOINT_NAME

    With custom config: \b\n
        python run.py generate CHECKPOINT_NAME -c CONFIG_NAME

    • If CHECKPOINT_INDEX is custom and CHECKPOINT_INDEX is not specified, 
        the latest checkpoint in the OUTPUT_ROOT_DIR will be loaded.
    • The default value for the fingerprinting source is [TEST_DUMMY_DB] and 
        [TEST_QUERY_DB] specified in config file.

    """
    from model.utils.config_gpu_memory_lim import allow_gpu_memory_growth
    from model.generate import generate_fingerprint

    cfg = load_config(config)
    allow_gpu_memory_growth()
    generate_fingerprint(cfg, checkpoint_type, checkpoint_name, checkpoint_index, source, output, skip_dummy)

""" Search and evalutation """
@cli.command()
@click.argument('checkpoint_name', required=True)
@click.argument('checkpoint_index', required=True)
@click.option('--config', '-c', default='default', required=False,
              type=click.STRING,
              help="Name of the model configuration file located in 'config/'."
              " Default is 'default'.")
@click.option('--index_type', '-i', default='ivfpq', type=click.STRING,
              help="Index type must be one of {'L2', 'IVF', 'IVFPQ', "
              "'IVFPQ-RR', 'IVFPQ-ONDISK', HNSW'}")
@click.option('--test_seq_len', default='1 3 5 9 11 19', type=click.STRING,
              help="A set of different number of segments to test. "
              "Numbers are separated by spaces. Default is '1 3 5 9 11 19', "
              "which corresponds to '1s, 2s, 3s, 5s, 6s, 10s' with 1 sec "
              "segment duration and 0.5 sec hop duration.")
@click.option('--test_ids', '-t', default='equally_spaced', type=click.STRING,
              help="One of {'all', 'equally_spaced', 'path/file.npy', (int)}. "
              "If 'all', test all IDs from the test. You can also specify a 1-D array "
              "file's location that contains the start indices to the the evaluation. "
              "Any numeric input N (int) > 0 will perform search test at random position "
              "(ID) N times. 'equally_spaced' will use boundary information to get an "
              "equal number of samples from each track. Default is 'equally_spaced'.")
@click.option('--nogpu', default=False, is_flag=True,
              help='Use this flag to use CPU only.')
def evaluate(checkpoint_name, checkpoint_index, config, index_type, test_seq_len, test_ids, nogpu):
    """ Search and evalutation.

    ex) python run.py evaluate CHECKPOINT_NAME CHECKPOINT_INDEX

    With options: \b\n

    ex) python run.py evaluate CHECKPOINT_NAME CHEKPOINT_INDEX -i ivfpq -t 3000 --nogpu

    • Currently, the 'evaluate' command does not reference any information other
        than the output log directory from the config file.
    """
    from eval.eval_faiss import eval_faiss

    cfg = load_config(config)
    emb_dir = cfg['MODEL']['LOG_ROOT_DIR'] + f"emb/{checkpoint_name}/{checkpoint_index}/"

    if nogpu:
        eval_faiss([emb_dir, "--index_type", index_type, "--test_seq_len",
                    test_seq_len, "--test_ids", test_ids, "--nogpu"])
    else:
        eval_faiss([emb_dir, "--index_type", index_type, "--test_seq_len",
                    test_seq_len, "--test_ids", test_ids])

if __name__ == '__main__':
    cli()
