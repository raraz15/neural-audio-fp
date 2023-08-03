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
        self.segment_duration = model_dict['AUDIO']['SEGMENT_DUR']
        self.normalize_audio = model_dict['AUDIO']['NORMALIZE_SEGMENTS']
        self.fs = model_dict['AUDIO']['FS']
        self.stft_hop = model_dict['INPUT']['STFT_HOP']
        self.n_fft = model_dict['INPUT']['STFT_WIN']
        self.n_mels = model_dict['INPUT']['N_MELS']
        self.fmin = model_dict['INPUT']['F_MIN']
        self.fmax = model_dict['INPUT']['F_MAX']
        self.scale_inputs = model_dict['INPUT']['SCALE_INPUTS']

        # Train, Val Parameters
        train_dict = cfg['TRAIN']
        self.tr_dataset_dir = train_dict['DIR']['TRAIN_ROOT']
        self.val_dataset_dir = train_dict['DIR']['VAL_ROOT']
        self.dataset_audio_segment_duration = train_dict['INPUT_AUDIO_DUR']

        self.tr_batch_sz = train_dict['BSZ']['TR_BATCH_SZ']
        self.tr_n_anchor = train_dict['BSZ']['TR_N_ANCHOR']
        self.val_batch_sz = train_dict['BSZ']['VAL_BATCH_SZ']
        self.val_n_anchor = train_dict['BSZ']['VAL_N_ANCHOR']
        self.tr_segments_per_track = train_dict['SEGMENTS_PER_TRACK']

        self.tr_bg_root_dir = train_dict['DIR']['BG_ROOT']
        self.tr_use_bg_aug = train_dict['TD_AUG']['BG_AUG']
        self.tr_bg_snr = train_dict['TD_AUG']['BG_AUG_SNR']
        self.tr_ir_root_dir = train_dict['DIR']['IR_ROOT']
        self.tr_use_ir_aug = train_dict['TD_AUG']['IR_AUG']
        self.tr_max_ir_dur = train_dict['TD_AUG']['IR_AUG_MAX_DUR']

        # # We use the same augmentations for train and validation sets.
        self.val_bg_root_dir = self.tr_bg_root_dir
        self.val_use_bg_aug = self.tr_use_bg_aug
        self.val_bg_snr = self.tr_bg_snr
        self.val_ir_root_dir = self.tr_ir_root_dir
        self.val_use_ir_aug = self.tr_use_ir_aug
        self.val_max_ir_dur = self.tr_max_ir_dur

        # Test Parameters
        test_dict = cfg['TEST']
        self.ts_noise_dataset_dir = test_dict['DIR']['NOISE_ROOT']
        self.ts_clean_query_dataset_dir = test_dict['DIR']['CLEAN_QUERY_ROOT']
        self.ts_augmented_query_dataset_dir = test_dict['DIR']['AUGMENTED_QUERY_ROOT']

        self.ts_use_bg_aug = test_dict['TD_AUG']['BG_AUG']
        self.ts_bg_snr = test_dict['TD_AUG']['BG_AUG_SNR']
        self.ts_use_ir_aug = test_dict['TD_AUG']['IR_AUG']
        self.ts_max_ir_dur = test_dict['TD_AUG']['IR_AUG_MAX_DUR']

        self.ts_segment_dur = test_dict['SEGMENT_DUR']
        self.ts_segment_hop = test_dict['SEGMENT_HOP']
        self.ts_segments_per_track = test_dict['SEGMENTS_PER_TRACK']

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
            print(f"tr_bg_fps: {len(self.tr_bg_fps):>6,}")
        if self.tr_use_ir_aug:
            print(f"tr_ir_fps: {len(self.tr_ir_fps):>6,}")

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
            segment_duration=self.segment_duration,
            full_segment_duration=self.dataset_audio_segment_duration,
            bsz=self.tr_batch_sz,
            n_anchor=self.tr_n_anchor, #ex) bsz=40, n_anchor=8: 4 positive samples per anchor
            fs=self.fs,
            normalize_audio=self.normalize_audio,
            segments_per_track=self.tr_segments_per_track,
            scale_output=self.scale_inputs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            shuffle=True,
            random_offset_anchor=True,
            bg_mix_parameter=[self.tr_use_bg_aug, self.tr_bg_fps, self.tr_bg_snr],
            ir_mix_parameter=[self.tr_use_ir_aug, self.tr_ir_fps, self.tr_max_ir_dur])

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
            print(f"val_bg_fps: {len(self.val_bg_fps):>6,}")
        if self.val_use_ir_aug:
            print(f"val_ir_fps: {len(self.val_ir_fps):>6,}")

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
            scale_output=self.scale_inputs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            shuffle=False,
            random_offset_anchor=False,
            bg_mix_parameter=[self.val_use_bg_aug, self.val_bg_fps, self.val_bg_snr],
            ir_mix_parameter=[self.val_use_ir_aug, self.val_ir_fps, self.val_max_ir_dur],
            )

    def get_test_noise_ds(self):
        """ Test-dummy-DB without augmentation. Adds noise tracks to the DB.

            Returns:
            --------
                ds_dummy_db : genUnbalSequenceGeneration
                    The dataset for test-dummy-DB.
        """

        print(f"Creating the test-dummy-DB dataset (noise tracks)...")
        self.ts_noise_paths = sorted(
            glob.glob(self.ts_noise_dataset_dir+ '/**/*.npy', 
                    recursive=True))
        print(f"{len(self.ts_noise_paths):,} noise tracks found.")
        return genUnbalSequenceGeneration(
            track_paths=self.ts_noise_paths,
            duration=self.ts_segment_dur,
            hop=self.ts_segment_hop,
            fs=self.fs,
            normalize_audio=self.normalize_audio,
            scale=self.scale_inputs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            segments_per_track=self.ts_segments_per_track)

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
                glob.glob(self.ts_clean_query_dataset_dir + '/**/*.npy', 
                        recursive=True))
        print(f"{len(self.ts_query_clean):,} clean query tracks found.")
        ds_db = genUnbalSequenceGeneration(
            track_paths=self.ts_query_clean,
            duration=self.ts_segment_dur,
            hop=self.ts_segment_hop,
            fs=self.fs,
            normalize_audio=self.normalize_audio,
            scale=self.scale_inputs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            segments_per_track=self.ts_segments_per_track)
        print(f"Creating the augmented query dataset...")
        if not (self.ts_use_bg_aug and self.ts_use_ir_aug):
            self.ts_query_augmented = sorted(
                glob.glob(self.ts_augmented_query_dataset_dir + '/**/*.npy', 
                        recursive=True))
            print(f"{len(self.ts_query_augmented):,} augmented query tracks found")
            ds_query = genUnbalSequenceGeneration(
                track_paths=self.ts_query_augmented,
                duration=self.ts_segment_dur,
                hop=self.ts_segment_hop,
                fs=self.fs,
                normalize_audio=self.normalize_audio,
                scale=self.scale_inputs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                segments_per_track=self.ts_segments_per_track)
        else:
            print("Will augment the clean query tracks in real time. ")
            if self.ts_use_bg_aug:
                print(f"ts_bg_fps: {len(self.ts_bg_fps):>6,}")
            if self.ts_use_ir_aug:
                print(f"ts_ir_fps: {len(self.ts_ir_fps):>6,}")
            ds_query = genUnbalSequenceGeneration(
                self.ts_query_clean,
                segments_per_track=self.ts_segments_per_track,
                # bsz=self.ts_segments_per_track * 2, # Anchors + positives=augmentations
                duration=self.ts_segment_dur,
                hop=self.ts_segment_hop,
                normalize_audio=self.normalize_audio,
                fs=self.fs,
                scale=self.scale_inputs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                bg_mix_parameter=[self.ts_use_bg_aug, self.ts_bg_fps, self.ts_bg_snr],
                ir_mix_parameter=[self.ts_use_ir_aug, self.ts_ir_fps, self.ts_max_ir_dur],
                )
        return ds_query, ds_db