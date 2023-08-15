import os
import glob
from model.utils.dataloader_keras import genUnbalSequence
from model.utils.generation_dataloader_keras import genUnbalSequenceGeneration

class Dataset:
    """
    Build datasets for training, validation and test sets.

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
        self.segment_duration = cfg['MODEL']['AUDIO']['SEGMENT_DUR']
        self.normalize_segment = cfg['MODEL']['AUDIO']['NORMALIZE_SEGMENT']
        self.fs = cfg['MODEL']['AUDIO']['FS']
        self.stft_hop = cfg['MODEL']['INPUT']['STFT_HOP']
        self.n_fft = cfg['MODEL']['INPUT']['STFT_WIN']
        self.n_mels = cfg['MODEL']['INPUT']['N_MELS']
        self.fmin = cfg['MODEL']['INPUT']['F_MIN']
        self.fmax = cfg['MODEL']['INPUT']['F_MAX']
        self.scale_inputs = cfg['MODEL']['INPUT']['SCALE_INPUTS']

        # Train Parameters
        self.tr_tracks_dir = cfg['TRAIN']['TRACKS']['TRAIN_ROOT']

        self.dataset_audio_segment_duration = cfg['TRAIN']['INPUT_AUDIO_DUR']
        self.tr_batch_sz = cfg['TRAIN']['BSZ']['BATCH_SZ']
        self.tr_n_anchor = cfg['TRAIN']['BSZ']['N_ANCHOR']
        self.tr_segments_per_track = cfg['TRAIN']['SEGMENTS_PER_TRACK']

        self.tr_bg_root_dir = cfg['TRAIN']['AUG']['TD']['BG_ROOT']
        self.tr_use_bg_aug = cfg['TRAIN']['AUG']['TD']['BG']
        self.tr_bg_snr = cfg['TRAIN']['AUG']['TD']['BG_SNR']

        self.tr_ir_root_dir = cfg['TRAIN']['AUG']['TD']['IR_ROOT']
        self.tr_use_ir_aug = cfg['TRAIN']['AUG']['TD']['IR']
        self.tr_max_ir_dur = cfg['TRAIN']['AUG']['TD']['IR_MAX_DUR']
        self.tr_bg_fps = []
        self.tr_ir_fps = []

        # Validation Parameters
        self.val_tracks_dir = cfg['TRAIN']['TRACKS']['VAL_ROOT']
        # We use the same augmentations for train and validation sets

        # Test Parameters
        self.ts_noise_tracks_dir = cfg['TEST']['TRACKS']['NOISE_ROOT']
        self.ts_clean_query_tracks_dir = cfg['TEST']['TRACKS']['CLEAN_QUERY_ROOT']
        self.ts_augmented_query_tracks_dir = cfg['TEST']['TRACKS']['AUGMENTED_QUERY_ROOT']
        self.ts_bg_root_dir = cfg['TEST']['AUG']['TD']['BG_ROOT']
        self.ts_ir_root_dir = cfg['TEST']['AUG']['TD']['IR_ROOT']

        self.ts_segment_dur = cfg['TEST']['SEGMENT_DUR']
        self.ts_segment_hop = cfg['TEST']['SEGMENT_HOP']
        self.ts_segments_per_track = cfg['TEST']['SEGMENTS_PER_TRACK']

        self.ts_use_bg_aug = cfg['TEST']['AUG']['TD']['BG']
        self.ts_bg_snr = cfg['TEST']['AUG']['TD']['BG_SNR']
        self.ts_use_ir_aug = cfg['TEST']['AUG']['TD']['IR']
        self.ts_max_ir_dur = cfg['TEST']['AUG']['TD']['IR_MAX_DUR']
        self.ts_bg_fps = []
        self.ts_ir_fps = []

    def get_train_ds(self, reduce_items_p=100):
        """ Source (music) file paths for training set. The folder structure
        should be as follows:
            self.tr_tracks_dir/
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

        print("Creating the training dataset...")

        assert reduce_items_p>0 and reduce_items_p<=100, \
            "reduce_items_p should be in (0, 100]"

        # Find the augmentation files
        if self.tr_use_bg_aug:
            self.tr_bg_fps = sorted(glob.glob(self.tr_bg_root_dir + "**/*.wav", 
                                    recursive=True))
            print(f"tr_bg_fps: {len(self.tr_bg_fps):>6,}")
            assert len(self.tr_bg_fps)>0, "No background noise found."
        if self.tr_use_ir_aug:
            self.tr_ir_fps = sorted(glob.glob(self.tr_ir_root_dir + "**/*.wav", 
                                    recursive=True))
            print(f"tr_ir_fps: {len(self.tr_ir_fps):>6,}")
            assert len(self.tr_ir_fps)>0, "No impulse response found."

        # Find the tracks and their segments
        self.tr_source_fps = {}
        main_dirs = os.listdir(self.tr_tracks_dir)
        for main_dir in main_dirs:
            track_names = os.listdir(os.path.join(self.tr_tracks_dir, main_dir))
            for track_name in track_names:
                track_dir = os.path.join(self.tr_tracks_dir, main_dir, track_name)
                segment_paths = sorted(glob.glob(track_dir + '/*.npz', recursive=True))
                self.tr_source_fps[track_name] = segment_paths
        assert len(self.tr_source_fps)>0, "No training tracks found."
        total_segments = sum([len(v) for v in self.tr_source_fps.values()])
        assert total_segments>0, "No segments found."
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
            normalize_segment=self.normalize_segment,
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
            self.val_tracks_dir/
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

        print(f"Creating the validation dataset...")

        # Find the augmentation files
        if self.tr_use_bg_aug:
            print(f"val_bg_fps: {len(self.tr_bg_fps):>6,} (Same as the train set)")
        if self.tr_use_ir_aug:
            print(f"val_ir_fps: {len(self.tr_ir_fps):>6,} (Same as the train set)")

        # Find the tracks and their segments
        self.val_source_fps = {}
        main_dirs = os.listdir(self.val_tracks_dir)
        for main_dir in main_dirs:
            track_names = os.listdir(os.path.join(self.val_tracks_dir, main_dir))
            for track_name in track_names:
                track_dir = os.path.join(self.val_tracks_dir, main_dir, track_name)
                segment_paths = sorted(glob.glob(track_dir + '/*.npz', recursive=True))
                self.val_source_fps[track_name] = segment_paths
        assert len(self.val_source_fps)>0, "No validation tracks found."
        total_segments = sum([len(v) for v in self.val_source_fps.values()])
        assert total_segments>0, "No segments found."
        print(f"{len(self.val_source_fps):,} tracks found.")
        print(f"{total_segments:,} segments found.")

        return genUnbalSequence(
            segment_dict=self.val_source_fps,
            bsz=self.tr_batch_sz,
            n_anchor=self.tr_n_anchor,
            fs=self.fs,
            normalize_segment=self.normalize_segment,
            segments_per_track=self.tr_segments_per_track,
            scale_output=self.scale_inputs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            shuffle=False,
            random_offset_anchor=False,
            bg_mix_parameter=[self.tr_use_bg_aug, self.tr_bg_fps, self.tr_bg_snr],
            ir_mix_parameter=[self.tr_use_ir_aug, self.tr_ir_fps, self.tr_max_ir_dur],
            )

    def get_test_noise_ds(self):
        """ Test-dummy-DB without augmentation. Adds noise tracks to the DB.

            Returns:
            --------
                ds_dummy_db : genUnbalSequenceGeneration
                    The dataset for test-dummy-DB.
        """

        print(f"Creating the test-dummy-DB dataset (noise tracks)...")

        # Find the noise tracks and their segments
        self.ts_noise_paths = sorted(
            glob.glob(self.ts_noise_tracks_dir+ '/**/*.npz', 
                    recursive=True))
        assert len(self.ts_noise_paths)>0, "No noise tracks found."
        print(f"{len(self.ts_noise_paths):,} noise tracks found.")

        return genUnbalSequenceGeneration(
            track_paths=self.ts_noise_paths,
            segment_duration=self.ts_segment_dur,
            hop_duration=self.ts_segment_hop,
            normalize_segment=self.normalize_segment,
            fs=self.fs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            scale_output=self.scale_inputs,
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

        print("Creating the clean query dataset...")

        # Find the clean query tracks and their segments
        self.ts_query_clean = sorted(
                glob.glob(self.ts_clean_query_tracks_dir + '/**/*.npz', 
                        recursive=True))
        assert len(self.ts_query_clean)>0, "No clean query tracks found."
        print(f"{len(self.ts_query_clean):,} clean query tracks found.")

        # Create the clean query dataset
        ds_db = genUnbalSequenceGeneration(
            track_paths=self.ts_query_clean,
            segment_duration=self.ts_segment_dur,
            hop_duration=self.ts_segment_hop,
            normalize_segment=self.normalize_segment,
            fs=self.fs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            scale_output=self.scale_inputs,
            segments_per_track=self.ts_segments_per_track)

        print("Creating the augmented query dataset...")
        if self.ts_use_bg_aug and self.ts_use_ir_aug:

            print("Will augment the clean query tracks in real time. ")

            # Find the augmentation files
            self.ts_bg_fps = sorted(glob.glob(self.ts_bg_root_dir + "**/*.wav", 
                                    recursive=True))
            print(f"ts_bg_fps: {len(self.ts_bg_fps):>6,}")
            assert len(self.ts_bg_fps)>0, "No background noise found."
            self.ts_ir_fps = sorted(glob.glob(self.ts_ir_root_dir + "**/*.wav", 
                                    recursive=True))
            print(f"ts_ir_fps: {len(self.ts_ir_fps):>6,}")
            assert len(self.ts_ir_fps)>0, "No impulse response found."

            # Create the augmented query dataset
            ds_query = genUnbalSequenceGeneration(
                track_paths=self.ts_query_clean, # Augment the clean query tracks
                segment_duration=self.ts_segment_dur,
                hop_duration=self.ts_segment_hop,
                normalize_segment=self.normalize_segment,
                fs=self.fs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                scale_output=self.scale_inputs,
                segments_per_track=self.ts_segments_per_track,
                bg_mix_parameter=[self.ts_use_bg_aug, self.ts_bg_fps, self.ts_bg_snr],
                ir_mix_parameter=[self.ts_use_ir_aug, self.ts_ir_fps, self.ts_max_ir_dur],
                )

        elif (not self.ts_use_bg_aug) and (not self.ts_use_ir_aug):

            print("Using pre-augmented query tracks.")

            # Find the augmented query tracks and their segments
            self.ts_query_augmented = sorted(
                glob.glob(self.ts_augmented_query_tracks_dir + '/**/*.npz', 
                        recursive=True))
            assert len(self.ts_query_augmented)>0, "No augmented query tracks found."
            print(f"{len(self.ts_query_augmented):,} augmented query tracks found")

            # Create the augmented query dataset
            ds_query = genUnbalSequenceGeneration(
                track_paths=self.ts_query_augmented,
                segment_duration=self.ts_segment_dur,
                hop_duration=self.ts_segment_hop,
                normalize_segment=self.normalize_segment,
                fs=self.fs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                scale_output=self.scale_inputs,
                segments_per_track=self.ts_segments_per_track)

        else:

            # For now we do not support single augmentation
            raise ValueError("Invalid augmentation parameters.")

        return ds_query, ds_db