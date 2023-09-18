# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.experimental import CosineDecay, CosineDecayRestarts
from tensorflow.keras.utils import Progbar

from model.dataset import Dataset
from model.fp.specaug_chain.specaug_chain import get_specaug_chain_layer
from model.fp.nnfp import get_fingerprinter
from model.fp.NTxent_loss_single_gpu import NTxentLoss
from model.fp.lamb_optimizer import LAMB
from model.utils.experiment_helper import ExperimentHelper
from model.utils.mini_search_subroutines import mini_search_eval

SEED = 27

def set_seed(seed: int = SEED):
    """Set seed for reproducibility."""

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seed set as {seed}")

def set_global_determinism():
    """ When running on the CuDNN backend, two further options must be set"""

    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print("Global determinism set")

def build_fp(cfg):
    """ Build fingerprinter """

    # m_specaug: spec-augmentation layer.
    m_specaug = get_specaug_chain_layer(cfg, trainable=False)
    assert(m_specaug.bypass==False) # Detachable by setting m_specaug.bypass.

    # m_fp: fingerprinter g(f(.)).
    m_fp = get_fingerprinter(cfg, trainable=False)

    return m_specaug, m_fp

@tf.function
def train_step(X, m_specaug, m_fp, loss_obj, helper):
    # X: (Xa, Xp)
    # Xa: anchors or originals, s.t. [xa_0, xa_1,...]
    # Xp: augmented replicas, s.t. [xp_0, xp_1] with xp_n = rand_aug(xa_n).
    # avg_loss: The cumulative average loss until current step within the current epoch.

    n_anchors = len(X[0])
    X = tf.concat(X, axis=0)
    feat = m_specaug(X) # (nA+nP, F, T, 1)
    m_fp.trainable = True
    with tf.GradientTape() as t:
        emb = m_fp(feat) # (BSZ, Dim)
        loss, sim_mtx, _ = loss_obj.compute_loss(emb[:n_anchors, :], 
                                                 emb[n_anchors:, :]) # {emb_org, emb_rep}
    g = t.gradient(loss, m_fp.trainable_variables)
    helper.optimizer.apply_gradients(zip(g, m_fp.trainable_variables))
    avg_loss = helper.update_tr_loss(loss) # To tensorboard
    return avg_loss, sim_mtx

@tf.function
def val_step(X, m_fp, loss_obj, helper):
    """ Validation step """

    n_anchors = len(X[0])
    X = tf.concat(X, axis=0)
    feat = X  # (nA+nP, F, T, 1)
    m_fp.trainable = False
    emb = m_fp(feat) # (BSZ, Dim)
    loss, sim_mtx, _ = loss_obj.compute_loss(emb[:n_anchors, :], 
                                             emb[n_anchors:, :]) # {emb_org, emb_rep}
    avg_loss = helper.update_val_loss(loss) # To tensorboard.
    return avg_loss, sim_mtx

@tf.function
def test_step(X, m_fp):
    """ Test step used for mini-search-validation """
    X = tf.concat(X, axis=0)
    feat = X  # (nA+nP, F, T, 1)
    m_fp.trainable = False
    emb_f = m_fp.front_conv(feat)  # (BSZ, Dim)
    emb_f_postL2 = tf.math.l2_normalize(emb_f, axis=1)
    emb_gf = m_fp.div_enc(emb_f)
    emb_gf = tf.math.l2_normalize(emb_gf, axis=1)
    return emb_f, emb_f_postL2, emb_gf # f(.), L2(f(.)), L2(g(f(.))

def mini_search_validation(ds, m_fp, mode='argmin',
                           scopes=[1, 3, 5, 9, 11, 19], max_n_samples=3000):
    """ Mini-search-validation """

    # Construct mini-DB
    key_strs = ['f', 'L2(f)', 'g(f)']
    m_fp.trainable = False
    (db, query, emb, dim) = (dict(), dict(), dict(), dict())
    dim['f'] = dim['L2(f)'] = m_fp.front_hidden_ch[-1]
    dim['g(f)'] = m_fp.emb_sz
    bsz = ds.bsz
    n_anchor = bsz // 2
    n_iter = min(len(ds), max_n_samples // bsz)
    for k in key_strs:
        (db[k], query[k]) = (tf.zeros((0, dim[k])), tf.zeros((0, dim[k])))
    for i in range(n_iter):
        _, _, Xa, Xp = ds.__getitem__(i)
        emb['f'], emb['L2(f)'], emb['g(f)'] = test_step((Xa, Xp), m_fp)
        for k in key_strs:
            db[k] = tf.concat((db[k], emb[k][:n_anchor, :]), axis=0)
            query[k] = tf.concat((query[k], emb[k][n_anchor:, :]), axis=0)

    # Search test
    accs_by_scope = dict()
    for k in key_strs:
        tf.print(f'======= mini-search-validation: \033[31m{mode} \033[33m{k} \033[0m=======' + '\033[0m')
        query[k] = tf.expand_dims(query[k], axis=1) # (nQ, d) --> (nQ, 1, d)
        accs_by_scope[k], _ = mini_search_eval(query[k], db[k], scopes, mode, display=True)
    return accs_by_scope, scopes, key_strs

def trainer(cfg, checkpoint_name):

    # Initialize the datasets
    tf.print('-----------Initializing the datasets-----------')
    dataset = Dataset(cfg)

    train_ds = dataset.get_train_ds(cfg['TRAIN']['REDUCE_ITEMS_P'])
    val_ds = dataset.get_val_ds()

    # Build models.
    m_specaug, m_fp = build_fp(cfg)

    # Learning schedule
    # TODO: warmup
    # TODO: shouldnt we update lr at every epoch?
    total_nsteps = cfg['TRAIN']['MAX_EPOCH'] * len(train_ds)
    if cfg['TRAIN']['LR']['SCHEDULE'].upper() == 'COS':
        lr_schedule = CosineDecay(
                    initial_learning_rate=float(cfg['TRAIN']['LR']['INITIAL_RATE']),
                    decay_steps=total_nsteps,
                    alpha=float(cfg['TRAIN']['LR']['ALPHA']))
    elif cfg['TRAIN']['LR']['SCHEDULE'].upper() == 'COS-RESTART':
        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=float(cfg['TRAIN']['LR']['INITIAL_RATE']),
            first_decay_steps=int(total_nsteps * 0.1), # TODO: configurable
            num_periods=0.5,
            alpha=float(cfg['TRAIN']['LR']['ALPHA'])) # Default 2e-6
    else:
        lr_schedule = float(cfg['TRAIN']['LR']['INITIAL_RATE'])

    # Optimizer
    if cfg['TRAIN']['OPTIMIZER'].upper() == 'LAMB':
        opt = LAMB(learning_rate=lr_schedule)
    elif cfg['TRAIN']['OPTIMIZER'].upper() == 'ADAM':
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        raise NotImplementedError(cfg['TRAIN']['OPTIMIZER'])

    # Experiment helper: see utils.experiment_helper.py for details.
    helper = ExperimentHelper(
        checkpoint_name=checkpoint_name,
        optimizer=opt,
        model_to_checkpoint=m_fp,
        cfg=cfg)

    # Loss objects
    if cfg['TRAIN']['LOSS']['LOSS_MODE'].upper() == 'NTXENT': # Default
        n_org = train_ds.n_anchor
        n_rep = train_ds.n_pos_bsz
        loss_obj_train = NTxentLoss(
            n_org=n_org,
            n_rep=n_rep,
            tau=cfg['TRAIN']['LOSS']['TAU'])
        loss_obj_val = NTxentLoss(
            n_org=n_org,
            n_rep=n_rep,
            tau=cfg['TRAIN']['LOSS']['TAU'])
    else:
        raise NotImplementedError(cfg['TRAIN']['LOSS']['LOSS_MODE'])

    # Training loop
    ep_start = helper.epoch
    ep_max = cfg['TRAIN']['MAX_EPOCH']
    if ep_start != 0:
        assert ep_start <= ep_max, f"When continuing training, MAX_EPOCH={ep_max} "\
        f"must be greater than or equal to where training was left off, which is {ep_start}"
    for ep in range(ep_start, ep_max + 1):
        tf.print(f'EPOCH: {ep}/{ep_max}')

        # Train
        progbar = Progbar(len(train_ds))
        """ Parallelism to speed up preprocessing.............. """
        enq = tf.keras.utils.OrderedEnqueuer(train_ds, 
                                            use_multiprocessing=True, 
                                            # We shuffle inside the dataset
                                            # OrderedEnqueuer calls train_ds.on_epoch_end()
                                            shuffle=False)
        enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'],
                  max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])
        i = 0
        while i < len(enq.sequence):
            _, _, Xa, Xp = next(enq.get())
            avg_loss, sim_mtx = train_step((Xa, Xp), m_specaug, m_fp,
                                            loss_obj_train, helper)
            progbar.add(1, values=[("tr loss", avg_loss)])
            i += 1
        enq.stop()
        """ End of Parallelism................................. """
        if cfg['TRAIN']['SAVE_IMG'] and (sim_mtx is not None):
            helper.write_image_tensorboard('tr_sim_mtx', sim_mtx.numpy())

        # Validate
        progbar = Progbar(len(val_ds))
        """ Parallelism to speed up preprocessing.............. """
        enq = tf.keras.utils.OrderedEnqueuer(val_ds, 
                                            use_multiprocessing=True, 
                                            shuffle=False)
        enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'],
                  max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])
        i = 0
        while i < len(enq.sequence):
            _, _, Xa, Xp = next(enq.get())
            avg_loss, sim_mtx = val_step((Xa, Xp), m_fp, loss_obj_val, helper)
            progbar.add(1, values=[("val loss", avg_loss)])
            i += 1
        enq.stop()
        """ End of Parallelism................................. """
        if cfg['TRAIN']['SAVE_IMG'] and (sim_mtx is not None):
            helper.write_image_tensorboard('val_sim_mtx', sim_mtx.numpy())

        # On epoch end
        tf.print('tr_loss:{:.4f}, val_loss:{:.4f}'.format(helper._tr_loss.result(), 
                                                          helper._val_loss.result()))
        helper.update_on_epoch_end(save_checkpoint_now=True)

        # Mini-search-validation (optional)
        if cfg['TRAIN']['MINI_TEST_IN_TRAIN']:
            accs_by_scope, scopes, key_strs = mini_search_validation(val_ds, m_fp)
            for k in key_strs:
                helper.update_minitest_acc(accs_by_scope[k], scopes, k)