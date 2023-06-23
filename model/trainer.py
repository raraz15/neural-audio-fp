# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" trainer.py """
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from model.dataset import Dataset
from model.fp.specaug_chain.specaug_chain import get_specaug_chain_layer
from model.fp.nnfp import get_fingerprinter
from model.fp.NTxent_loss_single_gpu import NTxentLoss
from model.fp.online_triplet_loss import OnlineTripletLoss
from model.fp.lamb_optimizer import LAMB
from model.utils.experiment_helper import ExperimentHelper
from model.utils.mini_search_subroutines import mini_search_eval


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
    avg_loss = helper.update_tr_loss(loss) # To tensorboard.
    return avg_loss, sim_mtx # avg_loss: average within the current epoch


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

# OGUZ: they set the n_anchor to 1/2 of the batch size!
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

    # Dataloader
    dataset = Dataset(cfg)

    # Build models.
    m_specaug, m_fp = build_fp(cfg)

    # Learning schedule
    total_nsteps = cfg['TRAIN']['MAX_EPOCH'] * len(dataset.get_train_ds())
    if cfg['TRAIN']['LR_SCHEDULE'].upper() == 'COS':
        lr_schedule = tf.keras.experimental.CosineDecay(
            initial_learning_rate=float(cfg['TRAIN']['LR']),
            decay_steps=total_nsteps,
            alpha=1e-06)
    elif cfg['TRAIN']['LR_SCHEDULE'].upper() == 'COS-RESTART':
        lr_schedule = tf.keras.experimental.CosineDecayRestarts(
            initial_learning_rate=float(cfg['TRAIN']['LR']),
            first_decay_steps=int(total_nsteps * 0.1),
            num_periods=0.5,
            alpha=2e-06)
    else:
        lr_schedule = float(cfg['TRAIN']['LR'])

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
    if cfg['LOSS']['LOSS_MODE'].upper() == 'NTXENT': # Default
        loss_obj_train = NTxentLoss(
            n_org=cfg['BSZ']['TR_N_ANCHOR'],
            n_rep=cfg['BSZ']['TR_BATCH_SZ'] - cfg['BSZ']['TR_N_ANCHOR'],
            tau=cfg['LOSS']['TAU'])
        loss_obj_val = NTxentLoss(
            n_org=cfg['BSZ']['VAL_N_ANCHOR'],
            n_rep=cfg['BSZ']['VAL_BATCH_SZ'] - cfg['BSZ']['VAL_N_ANCHOR'],
            tau=cfg['LOSS']['TAU'])
    elif cfg['LOSS']['LOSS_MODE'].upper() == 'ONLINE-TRIPLET': # Now-playing
        loss_obj_train = OnlineTripletLoss(
            bsz=cfg['BSZ']['TR_BATCH_SZ'],
            n_anchor=cfg['BSZ']['TR_N_ANCHOR'],
            mode = 'semi-hard',
            margin=cfg['LOSS']['MARGIN'])
        loss_obj_val = OnlineTripletLoss(
            bsz=cfg['BSZ']['VAL_BATCH_SZ'],
            n_anchor=cfg['BSZ']['VAL_N_ANCHOR'],
            mode = 'all', # use 'all' mode for validation
            margin=0.)
    else:
        raise NotImplementedError(cfg['LOSS']['LOSS_MODE'])

    # Training loop
    ep_start = helper.epoch
    ep_max = cfg['TRAIN']['MAX_EPOCH']
    for ep in range(ep_start, ep_max + 1):
        tf.print(f'EPOCH: {ep}/{ep_max}')

        # Train
        train_ds = dataset.get_train_ds(cfg['DATA_SEL']['REDUCE_ITEMS_P'])
        progbar = Progbar(len(train_ds))
        """ Parallelism to speed up preprocessing.............. """
        enq = tf.keras.utils.OrderedEnqueuer(train_ds, 
                                            use_multiprocessing=True, 
                                            shuffle=train_ds.shuffle)
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
        val_ds = dataset.get_val_ds() # max 500
        """ Parallelism to speed up preprocessing.............. """
        enq = tf.keras.utils.OrderedEnqueuer(val_ds, 
                                            use_multiprocessing=True, 
                                            shuffle=False)
        enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'],
                  max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])
        i = 0
        while i < len(enq.sequence):
            _, _, Xa, Xp = next(enq.get())
            _, sim_mtx = val_step((Xa, Xp), m_fp, loss_obj_val, helper)
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
            accs_by_scope, scopes, key_strs = mini_search_validation(
                val_ds, m_fp)
            for k in key_strs:
                helper.update_minitest_acc(accs_by_scope[k], scopes, k)
