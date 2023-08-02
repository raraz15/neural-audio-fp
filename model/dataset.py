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

        # Model parameters
        model_dict = cfg['MODEL']
        self.normalize_audio = model_dict['NORMALIZE_AUDIO']
        self.fs = model_dict['FS']
        self.scale = model_dict['SCALE_INPUTS'] # TODO: change name
        self.stft_hop = model_dict['STFT_HOP']
        self.n_fft = model_dict['STFT_WIN']
        self.n_mels = model_dict['N_MELS']
        self.fmin = model_dict['F_MIN']
        self.fmax = model_dict['F_MAX']
        # TODO: set full_segment_length

        # Train, Val Parameters
        train_dict = cfg['TRAIN']
        self.tr_dataset_dir = train_dict['DIR']['TRAIN_ROOT']
        self.val_dataset_dir = train_dict['DIR']['VAL_ROOT']
        self.tr_bg_root_dir = train_dict['DIR']['BG_ROOT']
        self.tr_ir_root_dir = train_dict['DIR']['IR_ROOT']
        # # We use the same augmentation for train and validation sets.

        self.tr_batch_sz = train_dict['BSZ']['TR_BATCH_SZ']
        self.tr_n_anchor = train_dict['BSZ']['TR_N_ANCHOR']
        self.val_batch_sz = train_dict['BSZ']['VAL_BATCH_SZ']
        self.val_n_anchor = train_dict['BSZ']['VAL_N_ANCHOR']
        self.tr_segments_per_track = train_dict['SEGMENTS_PER_TRACK']

        self.tr_use_bg_aug = train_dict['TD_AUG']['TR_BG_AUG']
        self.val_use_bg_aug = train_dict['TD_AUG']['VAL_BG_AUG']
        self.tr_use_ir_aug = train_dict['TD_AUG']['TR_IR_AUG']
        self.val_use_ir_aug = train_dict['TD_AUG']['VAL_IR_AUG']
        self.tr_snr = train_dict['TD_AUG']['TR_BG_SNR']
        self.val_snr = train_dict['TD_AUG']['VAL_BG_SNR']

        # Test Parameters
        test_dict = cfg['TEST']
        self.ts_noise_dataset_dir = test_dict['DIR']['NOISE_ROOT']
        self.ts_clean_query_dataset_dir = test_dict['DIR']['CLEAN_QUERY_ROOT']
        self.ts_augmented_query_dataset_dir = test_dict['DIR']['AUGMENTED_QUERY_ROOT']

        self.ts_use_bg_aug = test_dict['TD_AUG']['BG_AUG']
        self.ts_use_ir_aug = test_dict['TD_AUG']['IR_AUG']
        self.ts_snr = test_dict['TD_AUG']['BG_AUG_SNR']
        self.ts_segment_dur = test_dict['SEGMENT_DUR']
        self.ts_segment_hop = test_dict['SEGMENT_HOP']
        self.ts_batch_sz = test_dict['TS_BATCH_SZ']

        # Pre-load file paths for augmentation
        self.__set_augmentation_fps()

    def __set_augmentation_fps(self):
        """ Set file path lists for augmentations. Only accepts wav files.
        We use the same augmentations for the train and validation sets."""

        if self.tr_use_bg_aug:
            self.tr_bg_fps = sorted(glob.glob(self.tr_bg_root_dir + "**/*.wav", 
                                    recursive=True))
        if self.val_use_bg_aug:
            self.val_bg_fps = self.tr_bg_fps
        if self.ts_use_bg_aug:
            self.ts_bg_fps = sorted(glob.glob(self.tr_bg_root_dir + "**/*.wav", 
                                    recursive=True))

        if self.tr_use_ir_aug:
            self.tr_ir_fps = sorted(glob.glob(self.tr_ir_root_dir + "**/*.wav", 
                                    recursive=True))
        if self.val_use_ir_aug:
            self.val_ir_fps = self.tr_ir_fps
        if self.ts_use_ir_aug:
            self.ts_ir_fps = sorted(glob.glob(self.tr_ir_root_dir + "**/*.wav", 
                                    recursive=True))

    def get_train_ds(self, reduce_items_p=100):
        """ Source (music) file paths for training set. The folder structure
        should be as follows:
            self.tr_dataset_dir/
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
        print(f"Creating the training dataset from {self.tr_dataset_dir}...")
        assert reduce_items_p>0 and reduce_items_p<=100, \
            "reduce_items_p should be in (0, 100]"

        if self.tr_use_bg_aug:
            print(f"tr_bg_fps: {len(self.tr_bg_fps):,}")
        if self.tr_use_ir_aug:
            print(f"tr_ir_fps: {len(self.tr_ir_fps):,}")

        # Find the tracks and their segments
        self.tr_source_fps = {}
        main_dirs = os.listdir(self.tr_dataset_dir)
        for main_dir in main_dirs:
            track_names = os.listdir(os.path.join(self.tr_dataset_dir, main_dir))
            for track_name in track_names:
                track_dir = os.path.join(self.tr_dataset_dir, main_dir, track_name)
                segment_paths = sorted(glob.glob(track_dir + '/*.npy', recursive=True))
                self.tr_source_fps[track_name] = segment_paths
        total_segments = sum([len(v) for v in self.tr_source_fps.values()])
        print(f"{len(self.tr_source_fps):,} tracks found.")
        print(f"{total_segments:,} segments found.")

        if reduce_items_p<100:
            print(f"Reducing the number of tracks used to {reduce_items_p}%")
            self.tr_source_fps = {k: v
                                  for i,(k,v) in enumerate(self.tr_source_fps.items())
                                  if i < int(len(self.tr_source_fps)*reduce_items_p/100)}
            print(f"Reduced to {len(self.tr_source_fps):,} tracks.")
            total_segments = sum([len(v) for v in self.tr_source_fps.values()])
            print(f"Reduced to {total_segments:,} segments.")

        return genUnbalSequence(
            segment_dict=self.tr_source_fps,
            bsz=self.tr_batch_sz,
            n_anchor=self.tr_n_anchor, #ex) bsz=40, n_anchor=8: 4 positive samples per anchor
            fs=self.fs,
            normalize_audio=self.normalize_audio,
            segments_per_track=self.tr_segments_per_track,
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
            self.val_dataset_dir/
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

        print(f"Creating the validation dataset from {self.val_dataset_dir}...")
        if self.val_use_bg_aug:
            print(f"val_bg_fps: {len(self.val_bg_fps):,}")
        if self.val_use_ir_aug:
            print(f"val_ir_fps: {len(self.val_ir_fps):,}")

        self.val_source_fps = {}
        main_dirs = os.listdir(self.val_dataset_dir)
        for main_dir in main_dirs:
            track_names = os.listdir(os.path.join(self.val_dataset_dir, main_dir))
            for track_name in track_names:
                track_dir = os.path.join(self.val_dataset_dir, main_dir, track_name)
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
            segments_per_track=self.tr_segments_per_track,
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
            glob.glob(self.ts_noise_dataset_dir+ '/**/*.mp4', 
                      recursive=True))
        print(f"{len(self.ts_dummy_db_source_fps):,} tracks found at "
              f"{self.ts_noise_dataset_dir}.")
        return genUnbalSequenceGeneration(
            track_paths=self.ts_dummy_db_source_fps,
            bsz=self.ts_batch_sz, # Only anchors
            duration=self.ts_segment_dur,
            hop=self.ts_segment_hop,
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

        Returns
        -------
            (ds_query, ds_db)
                ds_query is the augmented version of the clean queries.
                ds_db is the clean queries without augmentation.

        """

        print(f"Creating the clean query dataset for testing...")
        self.ts_query_clean = sorted(
                glob.glob(self.ts_clean_query_dataset_dir + '/**/*.mp4', 
                          recursive=True))
        print(f"{len(self.ts_query_clean):,} clean query tracks found at "
              f"{self.ts_clean_query_dataset_dir}.")
        ds_db = genUnbalSequenceGeneration(
            track_paths=self.ts_query_clean,
            bsz=self.ts_batch_sz, # Only anchors
            duration=self.ts_segment_dur,
            hop=self.ts_segment_hop,
            fs=self.fs,
            normalize_audio=self.normalize_audio,
            scale=self.scale,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax)
        print(f"Creating the augmented query dataset...")
        if not (self.ts_use_bg_aug or self.ts_use_ir_aug):
            self.ts_query_augmented = sorted(
                glob.glob(self.ts_augmented_query_dataset_dir + '/**/*.mp4', 
                          recursive=True))
            print(f"{len(self.ts_query_augmented):,} augmented query tracks found at "
                  f"{self.ts_augmented_query_dataset_dir}.")
            ds_query = genUnbalSequenceGeneration(
                track_paths=self.ts_query_augmented,
                bsz=self.ts_batch_sz, # Only anchors
                duration=self.ts_segment_dur,
                hop=self.ts_segment_hop,
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
            if self.ts_use_bg_aug:
                print(f"ts_bg_fps: {len(self.ts_bg_fps):,}")
            if self.ts_use_ir_aug:
                print(f"ts_ir_fps: {len(self.ts_ir_fps):,}")
            ds_query = genUnbalSequenceGeneration(
                track_paths=self.ts_query_clean,
                bsz=self.ts_batch_sz * 2, # Anchors and positives=augmentations
                n_anchor=self.ts_batch_sz,
                duration=self.ts_segment_dur,
                hop=self.ts_segment_hop,
                fs=self.fs,
                normalize_audio=self.normalize_audio,
                shuffle=False,
                random_offset_anchor=False,
                bg_mix_parameter=[self.ts_use_bg_aug, self.ts_bg_fps, self.ts_snr],
                ir_mix_parameter=[self.ts_use_ir_aug, self.ts_ir_fps],
                drop_the_last_non_full_batch=False)
        return ds_query, ds_db