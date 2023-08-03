# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" generate.py """

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from model.dataset import Dataset
from model.fp.nnfp import get_fingerprinter

def get_checkpoint_index_and_restore_model(m_fp, checkpoint_root_dir, checkpoint_name, checkpoint_index=None):
    """ Load a trained fingerprinter """

    # Create checkpoint
    checkpoint = tf.train.Checkpoint(model=m_fp)
    checkpoint_dir = checkpoint_root_dir + f'{checkpoint_name}/'
    c_manager = tf.train.CheckpointManager(checkpoint,
                                           checkpoint_dir,
                                           max_to_keep=None)

    # Load
    if checkpoint_index==None:
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
        checkpoint_fpath = checkpoint_dir + 'ckpt-' + str(checkpoint_index)
        status = checkpoint.restore(checkpoint_fpath) # Let TF to handle error cases.
        status.expect_partial()
        tf.print(f'---Restored from {checkpoint_fpath}---')
    return checkpoint_index

def prevent_overwrite(key, target_path):
    if (key == 'dummy_db') & os.path.exists(target_path):
        answer = input(f'{target_path} exists. Will you overwrite (y/N)?')
        if answer.lower() not in ['y', 'yes']: sys.exit()

def get_data_source(cfg, skip_dummy):
    dataset = Dataset(cfg)
    ds = dict()
    if skip_dummy:
        tf.print("Excluding \033[33m'dummy_db'\033[0m from source.")
    else:
        ds['dummy_db'] = dataset.get_test_noise_ds()
    # Create the clean and augmented query datasets
    ds['query'], ds['db'] = dataset.get_test_query_ds()
    tf.print(f'\x1b[1;32mData source: {list(ds.keys())}\x1b[0m',
             f'{dataset.ts_clean_query_dataset_dir}')
    return ds

def generate_fingerprint(cfg,
                         checkpoint_type,
                         checkpoint_name,
                         checkpoint_index,
                         output_root_dir,
                         skip_dummy):
    """
    After run, the output (generated fingerprints) directory will be:
      .
      └──logs
         └── emb
             └── CHECKPOINT_NAME
                 └── CHECKPOINT_INDEX
                     ├── db.mm
                     ├── db_shape.npy
                     ├── dummy_db.mm
                     ├── dummy_db_shape.npy
                     ├── query.mm
                     └── query_shape.npy
    """

    # Build the model checkpoint
    m_fp = get_fingerprinter(cfg, trainable=False)

    # Load from checkpoint
    log_root_dir = cfg['MODEL']['LOG_ROOT']
    if checkpoint_type.lower()=='best':
        checkpoint_dir = log_root_dir + "best_checkpoint/"
    elif checkpoint_type.lower()=='custom':
        checkpoint_dir = log_root_dir + "checkpoint/"
    else:
        raise ValueError(f'checkpoint_type must be one of {{"best", "custom"}}')
    checkpoint_index = get_checkpoint_index_and_restore_model(m_fp, 
                                                            checkpoint_dir, 
                                                            checkpoint_name, 
                                                            checkpoint_index)

    # Choose the output directory
    if not output_root_dir:
        output_root_dir = log_root_dir + "emb/"
    # Here the checkpoint_type does not matter because checkpoint_index is specified.
    output_dir = output_root_dir + f'{checkpoint_name}/{checkpoint_index}/'
    os.makedirs(output_dir, exist_ok=True)
    if not skip_dummy:
        prevent_overwrite('dummy_db', output_dir+'dummy_db.mm')

    # Get data source
    """ ds = {'key1': <Dataset>, 'key2': <Dataset>, ...} """
    ds = get_data_source(cfg, skip_dummy)

    bsz = int(cfg['TEST']['TS_BATCH_SZ'])
    dim = cfg['MODEL']['EMB_SZ']

    # Generate
    sz_check = dict() # for warning message
    for key in ds.keys():

        n_items = ds[key].n_samples
        assert n_items > 0, f"Dataset '{key}' is empty."

        # Create memmap, and save shapes
        """
        Why use "memmap"?

        • First, we need to store a huge uncompressed embedding vectors until
          constructing a compressed DB with IVF-PQ (using FAISS). Handling a
          huge ndarray is not a memory-safe way: "memmap" consume 0 memory.

        • Second, Faiss-GPU does not support reconstruction of DB from
          compressed DB (index). In eval/eval_faiss.py, we need uncompressed
          vectors to calaulate sequence-level matching score. The created
          "memmap" will be reused at that point.

        Reference:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        """

        arr_shape = (n_items, dim)
        arr = np.memmap(f'{output_dir}/{key}.mm',
                        dtype='float32',
                        mode='w+',
                        shape=arr_shape)
        np.save(f'{output_dir}/{key}_shape.npy', arr_shape)

        # Fingerprinting loop
        tf.print(
            f"=== Generating fingerprint from \x1b[1;32m'{key}'\x1b[0m " +
            f"bsz={bsz}, {n_items} items, d={dim}"+ " ===")
        progbar = Progbar(len(ds[key]))

        """ Parallelism to speed up preprocessing------------------------- """
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
            arr[i*bsz : (i+1)*bsz, :] = emb.numpy() # Writing on disk.
            i += 1
        progbar.update(i, finalize=True)
        enq.stop()
        """ End of Parallelism-------------------------------------------- """

        tf.print(f'=== Succesfully stored {arr_shape[0]} fingerprints to {output_dir} ===')
        sz_check[key] = len(arr)
        # Close memmap
        arr.flush()
        del(arr)

    # Summary of the model for reference
    m_fp.summary()

    if 'custom_source' in ds.keys():
        pass
    elif sz_check['db'] != sz_check['query']:
        print("\033[93mWarning: 'db' and 'qeury' size does not match. "\
              "This can cause a problem in evaluation stage.\033[0m")
    return
