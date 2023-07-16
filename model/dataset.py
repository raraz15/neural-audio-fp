# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" dataset.py """

import glob
from model.utils.dataloader_keras import genUnbalSequence

class Dataset:
    """
    Build dataset for train, validation and test.

    USAGE:
        dataset = Dataset(cfg)
        ds_train = dataset.get_train_ds()
        print(ds_train.__getitem__(0))

    ...

    Attributes
    ----------
    cfg : dict
        a dictionary containing configurations

    Public Methods
    --------------
    get_train_ds()
    get_val_ds()
    get_test_dummy_db_ds()
    get_test_query_db_ds()
    get_custom_db_ds(source_root_dir)

    """
    def __init__(self, cfg=dict()):

        # Data location
        self.source_root_dir = cfg['DIR']['SOURCE_ROOT_DIR']
        self.bg_root_dir = cfg['DIR']['BG_ROOT_DIR'] #!
        self.ir_root_dir = cfg['DIR']['IR_ROOT_DIR']

        # Data selection
        self.datasel_train = cfg['DATA_SEL']['TRAIN']
        self.datasel_test_dummy_db = cfg['DATA_SEL']['TEST_DUMMY_DB']
        self.datasel_test_query_db = cfg['DATA_SEL']['TEST_QUERY_DB']

        # BSZ
        self.tr_batch_sz = cfg['BSZ']['TR_BATCH_SZ']
        self.tr_n_anchor = cfg['BSZ']['TR_N_ANCHOR']
        self.val_batch_sz = cfg['BSZ']['VAL_BATCH_SZ']
        self.val_n_anchor = cfg['BSZ']['VAL_N_ANCHOR']
        self.ts_batch_sz = cfg['BSZ']['TS_BATCH_SZ']

        # Model parameters
        self.normalize_audio = cfg['MODEL']['NORMALIZE_AUDIO']
        self.dur = cfg['MODEL']['DUR']
        self.hop = cfg['MODEL']['HOP']
        self.fs = cfg['MODEL']['FS']
        self.scale = cfg['MODEL']['SCALE_INPUTS']
        self.stft_hop = cfg['MODEL']['STFT_HOP']
        self.n_fft = cfg['MODEL']['STFT_WIN']
        self.n_mels = cfg['MODEL']['N_MELS']
        self.fmin = cfg['MODEL']['F_MIN']
        self.fmax = cfg['MODEL']['F_MAX']

        # Time-domain augmentation parameter
        self.tr_snr = cfg['TD_AUG']['TR_SNR']
        self.ts_snr = cfg['TD_AUG']['TS_SNR']
        self.val_snr = cfg['TD_AUG']['VAL_SNR']
        self.tr_use_bg_aug = cfg['TD_AUG']['TR_BG_AUG']
        self.ts_use_bg_aug = cfg['TD_AUG']['TS_BG_AUG']
        self.val_use_bg_aug = cfg['TD_AUG']['VAL_BG_AUG']
        self.tr_use_ir_aug = cfg['TD_AUG']['TR_IR_AUG']
        self.ts_use_ir_aug = cfg['TD_AUG']['TS_IR_AUG']
        self.val_use_ir_aug = cfg['TD_AUG']['VAL_IR_AUG']

        # Pre-load file paths for augmentation
        self.tr_bg_fps = self.ts_bg_fps = self.val_bg_fps = None
        self.tr_ir_fps = self.ts_ir_fps = self.val_ir_fps = None
        self.__set_augmentation_fps()

        # Source (music) file paths
        self.tr_source_fps = self.val_source_fps = None
        self.ts_dummy_db_source_fps = None
        self.ts_query_icassp_fps = self.ts_db_icassp_fps = None
        self.ts_query_db_unseen_fps = None

    def __set_augmentation_fps(self):
        """
        Set file path lists:

            If validation set was not available, we replace it with subset of
            the trainset.
        """

        # TODO: Why do they use the same augmentation for train and val?
        # File lists for Augmentations
        if self.tr_use_bg_aug:
            self.tr_bg_fps = sorted(glob.glob(self.bg_root_dir +
                                              'tr/**/*.wav', recursive=True))
        if self.ts_use_bg_aug:
            self.ts_bg_fps = sorted(glob.glob(self.bg_root_dir +
                                              'ts/**/*.wav', recursive=True))
        if self.val_use_bg_aug:
            self.val_bg_fps = sorted(glob.glob(self.bg_root_dir +
                                               'tr/**/*.wav', recursive=True))

        if self.tr_use_ir_aug:
            self.tr_ir_fps = sorted(
                glob.glob(self.ir_root_dir + 'tr/**/*.wav', recursive=True))
        if self.ts_use_ir_aug:
            self.ts_ir_fps = sorted(
                glob.glob(self.ir_root_dir + 'ts/**/*.wav', recursive=True))
        if self.val_use_ir_aug:
            self.val_ir_fps = sorted(
                glob.glob(self.ir_root_dir + 'tr/**/*.wav', recursive=True))

    def get_train_ds(self, reduce_items_p=0):
        # Source (music) file paths for train set
        if self.datasel_train == '10k_icassp':
            _prefix = 'train-10k-30s/'
        else:
            raise NotImplementedError(self.datasel_train)
        self.tr_source_fps = sorted(
            glob.glob(self.source_root_dir + _prefix + '**/*.wav',
                      recursive=True))

        return genUnbalSequence(
            fns_event_list=self.tr_source_fps,
            bsz=self.tr_batch_sz,
            n_anchor=self.tr_n_anchor, #ex) bsz=40, n_anchor=8: 4 positive samples per anchor
            duration=self.dur,  # duration in seconds
            hop=self.hop,
            fs=self.fs,
            normalize_audio=self.normalize_audio,
            scale=self.scale,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            shuffle=True,
            random_offset_anchor=True,
            bg_mix_parameter=[self.tr_use_bg_aug, self.tr_bg_fps, self.tr_snr],
            ir_mix_parameter=[self.tr_use_ir_aug, self.tr_ir_fps],
            reduce_items_p=reduce_items_p
            )

    def get_val_ds(self, max_song=500):
        """
        Source (music) file paths for validation set.
        max_song: (int) <= 500.
        """

        self.val_source_fps = sorted(
            glob.glob(self.source_root_dir + 'val-query-db-500-30s/' +
                      '**/*.wav', recursive=True))[:max_song]

        return genUnbalSequence(
            self.val_source_fps,
            self.val_batch_sz,
            self.val_n_anchor,
            self.dur,
            self.hop,
            self.fs,
            normalize_audio=self.normalize_audio,
            scale=self.scale,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            shuffle=False,
            random_offset_anchor=False,
            bg_mix_parameter=[self.val_use_bg_aug, self.val_bg_fps, self.val_snr],
            ir_mix_parameter=[self.val_use_ir_aug, self.val_ir_fps],
            )

    # TODO: why does ts_n_anchor=ts_batch_sz makes it faster?
    def get_test_dummy_db_ds(self):
        """
        Test-dummy-DB without augmentation:
            In this case, high-speed fingerprinting is possible without
            augmentation by setting ts_n_anchor=ts_batch_sz.
        """

        # Source (music) file paths for test-dummy-DB set
        self.ts_dummy_db_source_fps = sorted(
            glob.glob(self.source_root_dir + 'test-dummy-db-100k-full/' +
                      '**/*.wav', recursive=True))
        if self.datasel_test_dummy_db in ['10k_full', '10k_30s']:
            self.ts_dummy_db_source_fps = self.ts_dummy_db_source_fps[:10000]
        elif self.datasel_test_dummy_db == '100k_full_icassp':
            self.ts_dummy_db_source_fps = self.ts_dummy_db_source_fps
        elif self.datasel_test_dummy_db.isnumeric():
            self.ts_dummy_db_source_fps = self.ts_dummy_db_source_fps[:int(
                self.datasel_test_db)]
        else:
            raise NotImplementedError(self.datasel_test_dummy_db)

        _ts_n_anchor = self.ts_batch_sz
        return genUnbalSequence(
            self.ts_dummy_db_source_fps,
            self.ts_batch_sz,
            _ts_n_anchor,
            self.dur,
            self.hop,
            self.fs,
            normalize_audio=self.normalize_audio,
            scale=self.scale,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            shuffle=False,
            random_offset_anchor=False,
            drop_the_last_non_full_batch=False) # No augmentations...

    def get_test_query_db_ds(self, datasel=None):
        """
        To select test dataset, you can use config file or datasel parameter.

        cfg['DATASEL']['TEST_QUERY_DB']:

            'unseen_icassp' will use pre-defined queries and DB.
                DB has selected clips without augmentations and queries are their
                augmented versions.
            'unseen_syn' will synthesize queries from DB in real-time.

        Returns
        -------
        (ds_query, ds_db)

        """
        if datasel:
            self.datasel

        if self.datasel_test_query_db=='unseen_icassp':
            self.ts_query_icassp_fps = sorted(
                glob.glob(self.source_root_dir + 'test-query-db-500-30s/' +
                      'query/**/*.wav', recursive=True))
            self.ts_db_icassp_fps = sorted(
                glob.glob(self.source_root_dir + 'test-query-db-500-30s/' +
                      'db/**/*.wav', recursive=True))

            _ts_n_anchor = self.ts_batch_sz
            ds_query = genUnbalSequence(
                self.ts_query_icassp_fps,
                self.ts_batch_sz,
                _ts_n_anchor,
                self.dur,
                self.hop,
                self.fs,
                normalize_audio=self.normalize_audio,
                scale=self.scale,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                shuffle=False,
                random_offset_anchor=False,
                drop_the_last_non_full_batch=False) # No augmentations...
            ds_db = genUnbalSequence(
                self.ts_db_icassp_fps,
                self.ts_batch_sz,
                _ts_n_anchor,
                self.dur,
                self.hop,
                self.fs,
                normalize_audio=self.normalize_audio,
                scale=self.scale,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                shuffle=False,
                random_offset_anchor=False,
                drop_the_last_non_full_batch=False) # No augmentations...
            return ds_query, ds_db
        elif self.datasel_test_query_db=='unseen_syn':
            self.ts_query_db_unseen_fps = sorted(
            glob.glob(self.source_root_dir + 'val-query-db-500-30s/' +
                      'db/**/*.wav', recursive=True))

            _query_ts_batch_sz = self.ts_batch_sz * 2
            _query_ts_n_anchor = self.ts_batch_sz

            ds_query = genUnbalSequence(
                self.ts_query_db_unseen_fps,
                _query_ts_batch_sz,
                _query_ts_n_anchor,
                self.dur,
                self.hop,
                self.fs,
                normalize_audio=self.normalize_audio,
                shuffle=False,
                random_offset_anchor=False,
                bg_mix_parameter=[self.ts_use_bg_aug, self.ts_bg_fps, self.ts_snr],
                ir_mix_parameter=[self.ts_use_ir_aug, self.ts_ir_fps],
                reduce_batch_first_half=True,
                drop_the_last_non_full_batch=False)

            _db_ts_n_anchor = self.ts_batch_sz
            ds_db = genUnbalSequence(
                self.ts_query_db_unseen_fps,
                self.ts_batch_sz,
                _db_ts_n_anchor,
                self.dur,
                self.hop,
                self.fs,
                normalize_audio=self.normalize_audio,
                shuffle=False,
                random_offset_anchor=False,
                drop_the_last_non_full_batch=False)
            return ds_query, ds_db
        else:
            raise NotImplementedError(self.datasel_test_query_db)

    def get_custom_db_ds(self, source_root_dir):
        """ Construct DB (or query) from custom source files. """
        fps = sorted(
            glob.glob(source_root_dir + '/**/*.wav', recursive=True))
        _ts_n_anchor = self.ts_batch_sz # Only anchors...
        return genUnbalSequence(
            fps,
            self.ts_batch_sz,
            _ts_n_anchor,
            self.dur,
            self.hop,
            self.fs,
            normalize_audio=self.normalize_audio,
            scale=self.scale,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            shuffle=False,
            random_offset_anchor=False,
            drop_the_last_non_full_batch=False) # No augmentations, No drop-samples.