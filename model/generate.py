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
from model.fp.nnfp import get_fingerprinter

def get_checkpoint_index_and_restore_model(m_fp, checkpoint_root_dir, checkpoint_name, checkpoint_index=None):
    """ Load a trained fingerprinter """

    # Create checkpoint
    checkpoint = tf.train.Checkpoint(model=m_fp)
    checkpoint_dir = os.path.join(checkpoint_root_dir, checkpoint_name)
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

def get_data_source(cfg, source, skip_dummy, isdir):
    dataset = Dataset(cfg)
    ds = dict()
    if source:
        ds['custom_source'] = dataset.get_custom_db_ds(source, isdir)
    else:
        # Create the clean and augmented query datasets
        ds['query'], ds['db'] = dataset.get_test_query_ds()
        # Create the dummy dataset
        if skip_dummy:
            tf.print("Excluding \033[33m'dummy_db'\033[0m from source.")
        else:
            ds['dummy_db'] = dataset.get_test_noise_ds()
        # TODO: proper print
        tf.print(f'\x1b[1;32mData source: {list(ds.keys())}\x1b[0m')
    return ds

def generate_fingerprint(cfg,
                         checkpoint_type,
                         checkpoint_name,
                         checkpoint_index,
                         source,
                         output_root_dir,
                         skip_dummy,
                         isdir):
    """
    After run, the output (generated fingerprints) directory will be:
      .
      └──logs
         └── emb
             └── CHECKPOINT_NAME
                 └── CHECKPOINT_INDEX
                     ├── custom.mm
                     ├── custom_shape.npy
    """

    # Build the model checkpoint
    m_fp = get_fingerprinter(cfg, trainable=False)

    # Load from checkpoint
    log_root_dir = cfg['MODEL']['LOG_ROOT_DIR']
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
    output_dir = os.path.join(output_root_dir, checkpoint_name, checkpoint_index)
    os.makedirs(output_root_dir, exist_ok=True)
    if not skip_dummy:
        prevent_overwrite('dummy_db', os.path.join(output_dir, 'dummy_db.mm'))

    # Get data source
    """ ds = {'key1': <Dataset>, 'key2': <Dataset>, ...} """
    ds = get_data_source(cfg, source, skip_dummy, isdir)

    dim = cfg['MODEL']['ARCHITECTURE']['EMB_SZ']
    bsz = cfg['TEST']['BATCH_SZ']

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

        if source.split('/')[-1].lower() == 'queries':
            segments_csv = os.path.join(output_root_dir, 'queries_segments.csv')
        elif source.split('/')[-1].lower() == 'references':
            segments_csv = os.path.join(output_root_dir, 'refs_segments.csv')
        else:
            raise NameError("Unknown type of audio. "
                            "It's not query nor reference")

        with open(segments_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['segment_id', 'filename', 'intra_segment_id', 'offset_min', 'offset_max'] # from model/utils/dataloader_keras.py line 117
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for ii, seg in enumerate(ds[key].track_paths):
                writer.writerow({'segment_id': ii,
                                 'filename': seg[0],
                                 'intra_segment_id': seg[1],
                                 'offset_min': seg[2],
                                 'offset_max': seg[3]})

        while i < len(enq.sequence):
            progbar.update(i)
            _, Xa = next(enq.get())
            emb = m_fp(Xa)
            # Write to disk. We must know the shape of the emb in advance
            arr[i*bsz : (i+1)*bsz, :] = emb.numpy()
            i += 1
        progbar.update(i, finalize=True)
        enq.stop()
        # End of Parallelism--------------------------------------------

        tf.print(f'=== Succesfully stored {len(arr)} fingerprints to {output_dir} ===')
        sz_check[key] = len(arr)
        # Close memmap
        arr.flush()
        del(arr)

    # Summary of the model for reference
    m_fp.summary()

    if sz_check['db'] != sz_check['query']:
        print("\033[93mWarning: 'db' and 'query' size does not match. "\
              "This can cause a problem in evaluation stage.\033[0m")

    print()
    print("\x1b[1;32m=== Fingerprinting completed ===\x1b[0m")
