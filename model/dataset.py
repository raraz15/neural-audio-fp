import os
import glob
from model.utils.dataloader_keras import genUnbalSequence
from model.utils.generation_dataloader_keras import genUnbalSequenceGeneration

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
    """

    def __init__(self, cfg=dict()):

        # Data locations
        self.dataset_dir_train = cfg['DIR']['DATA']['TRAIN_ROOT']
        self.dataset_dir_val = cfg['DIR']['DATA']['VAL_ROOT']
        self.dataset_dir_test_noise = cfg['DIR']['DATA']['TEST_NOISE_ROOT']
        self.dataset_dir_test_clean_query = cfg['DIR']['DATA']['TEST_CLEAN_QUERY_ROOT']
        self.dataset_dir_test_augmented_query = cfg['DIR']['DATA']['TEST_AUGMENTED_QUERY_ROOT']
        self.bg_root_dir = cfg['DIR']['DATA']['BG_ROOT']
        self.ir_root_dir = cfg['DIR']['DATA']['IR_ROOT']

        # BSZ
        self.tr_batch_sz = cfg['BSZ']['TR_BATCH_SZ']
        self.tr_n_anchor = cfg['BSZ']['TR_N_ANCHOR']
        self.val_batch_sz = cfg['BSZ']['VAL_BATCH_SZ']
        self.val_n_anchor = cfg['BSZ']['VAL_N_ANCHOR']
        self.ts_batch_sz = cfg['BSZ']['TS_BATCH_SZ']

        # Model parameters
        self.normalize_audio = cfg['MODEL']['NORMALIZE_AUDIO']
        # self.dur = cfg['MODEL']['DUR']
        # self.hop = cfg['MODEL']['HOP']
        self.fs = cfg['MODEL']['FS']
        self.scale = cfg['MODEL']['SCALE_INPUTS'] # TODO: change name
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

    def __set_augmentation_fps(self):
        """ Set file path lists for augmentations. Only accepts wav files.
        We use the same augmentations for the train and validation sets."""

        if self.tr_use_bg_aug:
            self.tr_bg_fps = sorted(glob.glob(self.bg_root_dir + 'tr/**/*.wav', 
                                              recursive=True))
            print(f"tr_bg_fps: {len(self.tr_bg_fps):,}")
        if self.val_use_bg_aug:
            self.val_bg_fps = self.tr_bg_fps
            print(f"val_bg_fps: {len(self.val_bg_fps):,}")
        if self.ts_use_bg_aug:
            self.ts_bg_fps = sorted(glob.glob(self.bg_root_dir + 'ts/**/*.wav', 
                                    recursive=True))
            print(f"ts_bg_fps: {len(self.ts_bg_fps):,}")

        if self.tr_use_ir_aug:
            self.tr_ir_fps = sorted(glob.glob(self.ir_root_dir + 'tr/**/*.wav', 
                                              recursive=True))
            print(f"tr_ir_fps: {len(self.tr_ir_fps):,}")
        if self.val_use_ir_aug:
            self.val_ir_fps = self.tr_ir_fps
            print(f"val_ir_fps: {len(self.val_ir_fps):,}")
        if self.ts_use_ir_aug:
            self.ts_ir_fps = sorted(glob.glob(self.ir_root_dir + 'ts/**/*.wav', 
                                              recursive=True))
            print(f"ts_ir_fps: {len(self.ts_ir_fps):,}")

    def get_train_ds(self, reduce_items_p=100):
        """ Source (music) file paths for training set. The folder structure
        should be as follows:
            self.dataset_dir_train/
                dir0/
                    track1/
                        clip1.npy
                        ...
                    track2/
                        clip1.npy
                        ...
                dir1/
                    track1/
                        clip1.npy
                        ...
                    track2/
                        clip1.npy
                        ...

            Parameters
            ----------
                reduce_items_p : int (default 100)
                    Reduce the number of items in each track to this percentage.
        """
        print(f"Creating the training dataset from {self.dataset_dir_train}...")
        assert reduce_items_p>0 and reduce_items_p<=100, \
            "reduce_items_p should be in (0, 100]"

        # Find the tracks and their segments
        self.tr_source_fps = {}
        main_dirs = os.listdir(self.dataset_dir_train)
        for main_dir in main_dirs:
            track_names = os.listdir(os.path.join(self.dataset_dir_train, main_dir))
            for track_name in track_names:
                track_dir = os.path.join(self.dataset_dir_train, main_dir, track_name)
                segment_paths = sorted(glob.glob(track_dir + '/*.npy', recursive=True))
                self.tr_source_fps[track_name] = segment_paths
        total_segments = sum([len(v) for v in self.tr_source_fps.values()])
        print(f"{len(self.tr_source_fps):,} tracks found.")
        print(f"{total_segments:,} segments found.")

        if reduce_items_p<100:
            print(f"Reducing the number of segments in each track to {reduce_items_p}%")
            self.tr_source_fps = {k: v[:int(len(v)*reduce_items_p/100)]
                                  for k,v in self.tr_source_fps.items()}
            total_segments = sum([len(v) for v in self.tr_source_fps.values()])
            print(f"Reduced to {total_segments:,} segments.")

        return genUnbalSequence(
            segment_dict=self.tr_source_fps,
            bsz=self.tr_batch_sz,
            n_anchor=self.tr_n_anchor, #ex) bsz=40, n_anchor=8: 4 positive samples per anchor
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
            ir_mix_parameter=[self.tr_use_ir_aug, self.tr_ir_fps])

    def get_val_ds(self):
        """ Source (music) file paths for validation set. The folder structure
        should be as follows:
            self.dataset_dir_val/
                track1/
                    clip1.npy
                    clip2.npy
                    ...
                    clipN.npy
                track2/
                    clip1.npy
                    clip2.npy
                    ...
                    clipN.npy
        """

        print(f"Creating the validation dataset from {self.dataset_dir_val}.")

        self.val_source_fps = {}
        main_dirs = os.listdir(self.dataset_dir_val)
        for main_dir in main_dirs:
            track_names = os.listdir(os.path.join(self.dataset_dir_val, main_dir))
            for track_name in track_names:
                track_dir = os.path.join(self.dataset_dir_val, main_dir, track_name)
                segment_paths = sorted(glob.glob(track_dir + '/*.npy', recursive=True))
                self.val_source_fps[track_name] = segment_paths
        total_segments = sum([len(v) for v in self.val_source_fps.values()])
        print(f"{len(self.val_source_fps):,} tracks found.")
        print(f"{total_segments:,} segments found.")

        return genUnbalSequence(
            segment_dict=self.val_source_fps,
            bsz=self.val_batch_sz,
            n_anchor=self.val_n_anchor,
            fs=self.fs,
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
    def get_test_noise_ds(self):
        """ Test-dummy-DB without augmentation. Adds noise tracks to the DB.:
            In this case, high-speed fingerprinting is possible without
            augmentation by setting ts_n_anchor=ts_batch_sz.

            Returns:
            --------
                ds_dummy_db : genUnbalSequenceGeneration
                    The dataset for test-dummy-DB. Noise tracks without augmentation.
        """

        print(f"Creating the test-dummy-DB dataset (noise tracks)...")
        self.ts_dummy_db_source_fps = sorted(
            glob.glob(self.dataset_dir_test_noise+ '/**/*.mp4', 
                      recursive=True))
        print(f"{len(self.ts_dummy_db_source_fps):,} tracks found at "
              f"{self.dataset_dir_test_noise}.")
        return genUnbalSequenceGeneration(
            track_paths=self.ts_dummy_db_source_fps,
            bsz=self.ts_batch_sz, # Only anchors
            duration=self.dur,
            hop=self.hop,
            fs=self.fs,
            normalize_audio=self.normalize_audio,
            scale=self.scale,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax)

    def get_test_query_ds(self):
        """ Create 2 databases for query segments. One of them is the augmented 
        version of the clean queries.

        Parameters:
        ----------
            augment : bool (default False)


        Returns
        -------
            (ds_query, ds_db)
                ds_query is the augmented version of the clean queries.
                ds_db is the clean queries without augmentation.

        """

        print(f"Creating the clean and augmented query datasets for testing...")

        self.ts_query_clean = sorted(
                glob.glob(self.dataset_dir_test_clean_query + '/**/*.mp4', 
                          recursive=True))
        print(f"{len(self.ts_query_clean):,} clean query tracks found at "
              f"{self.dataset_dir_test_clean_query}.")
        ds_db = genUnbalSequenceGeneration(
            track_paths=self.ts_query_clean,
            bsz=self.ts_batch_sz, # Only anchors
            duration=self.dur,
            hop=self.hop,
            fs=self.fs,
            normalize_audio=self.normalize_audio,
            scale=self.scale,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax)

        if not self.ts_use_bg_aug:
            self.ts_query_augmented = sorted(
                glob.glob(self.dataset_dir_test_augmented_query + '/**/*.mp4', 
                          recursive=True))
            print(f"{len(self.ts_query_augmented):,} augmented query tracks found at "
                  f"{self.dataset_dir_test_augmented_query}.")
            ds_query = genUnbalSequenceGeneration(
                track_paths=self.ts_query_augmented,
                bsz=self.ts_batch_sz, # Only anchors
                duration=self.dur,
                hop=self.hop,
                fs=self.fs,
                normalize_audio=self.normalize_audio,
                scale=self.scale,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax)
        else:
            print("Will augment the clean query tracks in real time. ")
            ds_query = genUnbalSequenceGeneration(
                track_paths=self.ts_query_clean,
                bsz=self.ts_batch_sz * 2, # Anchors and positives=augmentations
                n_anchor=self.ts_batch_sz,
                duration=self.dur,
                hop=self.hop,
                fs=self.fs,
                normalize_audio=self.normalize_audio,
                shuffle=False,
                random_offset_anchor=False,
                bg_mix_parameter=[self.ts_use_bg_aug, self.ts_bg_fps, self.ts_snr],
                ir_mix_parameter=[self.ts_use_ir_aug, self.ts_ir_fps],
                drop_the_last_non_full_batch=False)
        return ds_query, ds_db