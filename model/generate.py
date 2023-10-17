# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" generate.py """

import os
import csv
import sys

import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import Progbar

from model.dataset import Dataset
from model.fp.nnfp import FingerPrinter, get_fingerprinter

def get_checkpoint_index_and_restore_model(m_fp : FingerPrinter, checkpoint_dir: str, checkpoint_index=0):
    """ Load a trained fingerprinter.

    Args:
    -----
    m_fp: FingerPrinter
        Initialized Fingerprinter model.
    checkpoint_dir: str 
        Directory containing the checkpoints.
    checkpoint_index: int (default: 0) 
        0 means the latest checkpoint. Epoch index starts from 1.
    
    Returns:
    --------
    checkpoint_index: int
        Index of the checkpoint loaded.

    """

    # Create checkpoint
    checkpoint = tf.train.Checkpoint(model=m_fp)
    c_manager = tf.train.CheckpointManager(checkpoint,
                                           checkpoint_dir,
                                           max_to_keep=None)

    # Load a particular checkpoint
    if checkpoint_index==0:
        tf.print("\x1b[1;32mArgument 'checkpoint_index' was not specified.\x1b[0m")
        tf.print('\x1b[1;32mSearching for the latest checkpoint...\x1b[0m')
        latest_checkpoint = c_manager.latest_checkpoint
        if latest_checkpoint:
            checkpoint_index = int(latest_checkpoint.split(sep='ckpt-')[-1])
            status = checkpoint.restore(latest_checkpoint)
            status.expect_partial()
            tf.print(f'---Restored from {latest_checkpoint}---')
        else:
            raise FileNotFoundError(f'Cannot find checkpoint in {checkpoint_dir}')
    else:
        checkpoint_fpath = os.path.join(checkpoint_dir, 'ckpt-' + str(checkpoint_index))
        assert os.path.exists(checkpoint_fpath), \
            f'Cannot find checkpoint={checkpoint_index} in {checkpoint_dir}'
        status = checkpoint.restore(checkpoint_fpath) # Let TF to handle error cases.
        status.expect_partial()
        tf.print(f'---Restored from {checkpoint_fpath}---')

    return checkpoint_index

def prevent_overwrite(key, target_path):
    if (key == 'dummy_db') & os.path.exists(target_path):
        answer = input(f'{target_path} exists. Will you overwrite (y/N)?')
        if answer.lower() not in ['y', 'yes']:
            sys.exit()

def get_data_source(cfg: dict, source_root_dir="", bmat_source="", skip_dummy=False):

    assert not ((source_root_dir == "") and (bmat_source == "")), \
        "Only one of 'source_root_dir' and 'bmat_source' can be specified."

    # Create the dataset
    dataset = Dataset(cfg)
    ds = dict()

    # If source or bmat_source provided, only use the custom source
    if source_root_dir is not None:
        ds['custom_source'] = dataset.get_custom_db_ds(source_root_dir)
    elif bmat_source is not None:
        ds['bmat_custom_source'] = dataset.get_custom_bmat_db_ds(bmat_source)
    else: # Otherwise use the default sources
        # Create the clean and augmented query datasets
        ds['query'], ds['db'] = dataset.get_test_query_ds()
        # Create the dummy dataset if not skipped
        if skip_dummy:
            tf.print("Excluding \033[33m'dummy_db'\033[0m from source.")
        else:
            ds['dummy_db'] = dataset.get_test_noise_ds()
    # TODO: proper print
    tf.print(f'\x1b[1;32mData source: {list(ds.keys())}\x1b[0m')
    return ds

def generate_fingerprint(cfg: dict,
                         checkpoint_dir: str="",
                         checkpoint_index: int=0,
                         source_root_dir: str ="",
                         bmat_source: str ="",
                         output_root_dir: str="",
                         skip_dummy: bool=False,
                         ):
    """ Generate fingerprints from a trained model checkpoint.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    checkpoint_dir : str, optional
        Directory containing the checkpoints. (default: "")
        If not specified, load from the directory specified in the config file.
    checkpoint_index : int, optional
        Index of the checkpoint to load from. (default: 0)
        0 means the latest checkpoint. 97 means the 97th epoch.
    source : str, optional
        Path to the custom source. (default: "")
        If not specified, load from the default source specified in the config file.
    output_root_dir : str, optional
        Root directory for the output. (default: "")
        If not specified, load from the default directory specified in the config file.
    skip_dummy : bool, optional
        Whether generating the skip dummy_db. (default: False)

    After run, the output (generated fingerprints) directory will be:
      .
      └──output_root_dir (default=./logs/)
            └── emb
                └── checkpoint_name
                    └── checkpoint_index
                        ├── custom.mm
                        ├── custom_shape.npy
    """

    # Get information from the config file
    checkpoint_name = cfg['MODEL']['NAME']
    log_root_dir = cfg['MODEL']['LOG_ROOT_DIR'] # Can be overwritten by the arguments
    dim = cfg['MODEL']['ARCHITECTURE']['EMB_SZ']
    bsz = cfg['TEST']['BATCH_SZ']

    # Build the model checkpoint
    m_fp = get_fingerprinter(cfg, trainable=False)

    # If checkpoint directory is not specified, read it from the config file
    if checkpoint_dir == "":
        checkpoint_dir = os.path.join(log_root_dir, "checkpoint", f"{checkpoint_name}/")

    # Load checkpoint from checkpoint_dir using the epoch specified with checkpoint_index
    checkpoint_index = get_checkpoint_index_and_restore_model(m_fp, 
                                                            checkpoint_dir, 
                                                            checkpoint_index)

    # Determine the output_root_dir if not specified
    if output_root_dir == "":
        output_root_dir = os.path.join(log_root_dir, "emb/")
    # Create the output directory
    output_dir = os.path.join(output_root_dir, checkpoint_name, str(checkpoint_index))
    os.makedirs(output_dir, exist_ok=True)

    # Prevent overwriting the dummy_db as it is time-consuming
    if not skip_dummy:
        prevent_overwrite('dummy_db', os.path.join(output_dir, 'dummy_db.mm'))

    # Get data source
    """ ds = {'key1': <Dataset>, 'key2': <Dataset>, ...} """
    ds = get_data_source(cfg, source_root_dir=source_root_dir, bmat_source=bmat_source, skip_dummy=skip_dummy)

    dim = cfg['MODEL']['ARCHITECTURE']['EMB_SZ']
    bsz = cfg['TEST']['BATCH_SZ']

    # Generate
    sz_check = dict() # for warning message
    for key in ds.keys():

        n_items = ds[key].n_samples
        assert n_items > 0, f"Dataset '{key}' is empty."

        # Create a csv file containing the information of the segments if BMAT data is used
        if key == 'bmat_custom_source':

            if source_root_dir.split('/')[-1].lower() == 'queries':
                segments_csv = os.path.join(output_dir, 'queries_segments.csv')
            elif source_root_dir.split('/')[-1].lower() == 'references':
                segments_csv = os.path.join(output_dir, 'refs_segments.csv')
            else:
                raise NameError("Unknown type of audio. "
                                "It's not query nor reference")

            with open(segments_csv, 'w', newline='', encoding='utf-8') as csvfile:
                # from model/utils/dataloader_keras.py line 117
                fieldnames = ['segment_id', 'filename', 'intra_segment_id', 
                              'offset_min', 'offset_max']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for ii, seg in enumerate(ds[key].track_paths):
                    writer.writerow({'segment_id': ii,
                                    'filename': seg[0],
                                    'intra_segment_id': seg[1],
                                    'offset_min': seg[2],
                                    'offset_max': seg[3]})

        # Create memmap, and save shapes
        """
        Why use "memmap"?

        • First, we need to store a huge uncompressed embedding vectors until
          constructing a compressed DB with IVF-PQ (using FAISS). Handling a
          huge ndarray is not a memory-safe way: "memmap" consume 0 memory.

        • Second, Faiss-GPU does not support reconstruction of DB from
          compressed DB (index). In eval/eval_faiss.py, we need uncompressed
          vectors to calculate sequence-level matching score. The created
          "memmap" will be reused at that point.

        Reference:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        """

        arr_shape = (n_items, dim)
        arr = np.memmap(f'{output_dir}/{key}.mm',
                        dtype='float32',
                        mode='w+',
                        shape=arr_shape)
        # Save the shape of the memmap
        np.save(f'{output_dir}/{key}_shape.npy', arr_shape)

        # Fingerprinting loop
        tf.print(
            f"=== Generating fingerprint from \x1b[1;32m'{key}'\x1b[0m " +
            f"bsz={bsz}, {n_items} items, d={dim}"+ " ===")
        progbar = Progbar(len(ds[key]))

        """ Parallelism to speed up processing------------------------- """
        enq = tf.keras.utils.OrderedEnqueuer(ds[key],
                                              use_multiprocessing=True,
                                              shuffle=False)
        enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'],
                  max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])

        i = 0
        while i < len(enq.sequence):
            progbar.update(i)
            _, Xa = next(enq.get())
            emb = m_fp(Xa)
            # Write to disk. We must know the shape of the emb in advance
            arr[i*bsz : (i+1)*bsz, :] = emb.numpy()
            i += 1
        progbar.update(i, finalize=True)
        enq.stop()
        """ End of Parallelism----------------------------------------- """

        # Print summary
        tf.print(f'=== Succesfully stored {len(arr)} {key} fingerprints to {output_dir} ===')
        # Save the number of fingerprints for warning message
        sz_check[key] = len(arr)

        # Close memmap
        arr.flush()
        del(arr)

    # Summary of the model for reference
    m_fp.summary()

    # If query and dummy_db is created and the size of the dummy_db is not the same as the query, print a warning
    if ('db' in sz_check) and ('query' in sz_check) and sz_check['db'] != sz_check['query']:
        print("\033[93mWarning: 'db' and 'query' size does not match. "\
              "This can cause a problem in evaluation stage.\033[0m")

    print()
    print("\x1b[1;32m=== Fingerprinting completed ===\x1b[0m")
