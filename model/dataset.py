import os
import glob

from model.utils.dev_dataloader_keras import SegmentDevLoader, TrackDevLoader
from model.utils.generation_dataloader_keras import GenerationLoader

class Dataset:
    """
    Build datasets for train, validation and test sets.

    USAGE
    -----
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

        self.cfg = cfg

        # Model parameters
        self.segment_duration = cfg['MODEL']['AUDIO']['SEGMENT_DUR']
        self.fs = cfg['MODEL']['AUDIO']['FS']
        self.stft_hop = cfg['MODEL']['INPUT']['STFT_HOP']
        self.n_fft = cfg['MODEL']['INPUT']['STFT_WIN']
        self.n_mels = cfg['MODEL']['INPUT']['N_MELS']
        self.fmin = cfg['MODEL']['INPUT']['F_MIN']
        self.fmax = cfg['MODEL']['INPUT']['F_MAX']
        self.scale_inputs = cfg['MODEL']['INPUT']['SCALE_INPUTS']

        # Train Parameters
        self.tr_audio_dir = cfg['TRAIN']['DIR']['TRAIN_ROOT']

        self.tr_segments_per_track = cfg['TRAIN']['AUDIO']['SEGMENTS_PER_TRACK']
        self.tr_offset_duration = cfg['TRAIN']['AUDIO']["MAX_OFFSET_DUR"]
        self.tr_batch_sz = cfg['TRAIN']['BSZ']['BATCH_SZ']
        self.tr_n_anchor = cfg['TRAIN']['BSZ']['N_ANCHOR']

        self.tr_bg_root_dir = cfg['TRAIN']['AUG']['TD']['BG_ROOT']
        self.tr_use_bg_aug = cfg['TRAIN']['AUG']['TD']['BG']
        self.tr_bg_snr = cfg['TRAIN']['AUG']['TD']['BG_SNR']
        self.tr_bg_fps = []

        self.tr_ir_root_dir = cfg['TRAIN']['AUG']['TD']['IR_ROOT']
        self.tr_use_ir_aug = cfg['TRAIN']['AUG']['TD']['IR']
        self.tr_max_ir_dur = cfg['TRAIN']['AUG']['TD']['IR_MAX_DUR']
        self.tr_ir_fps = []

        # Validation Parameters
        self.val_audio_dir = cfg['TRAIN']['DIR']['VAL_ROOT']
        # We use the same augmentations for train and validation sets

        # Test Parameters
        self.ts_noise_tracks_dir = cfg['TEST']['DIR']['NOISE_ROOT']
        self.ts_clean_query_tracks_dir = cfg['TEST']['DIR']['CLEAN_QUERY_ROOT']
        self.ts_augmented_query_tracks_dir = cfg['TEST']['DIR']['AUGMENTED_QUERY_ROOT']

        self.ts_segment_hop = cfg['TEST']['SEGMENT_HOP']
        self.ts_batch_sz = cfg['TEST']['BATCH_SZ']

        self.ts_bg_root_dir = cfg['TEST']['AUG']['TD']['BG_ROOT']
        self.ts_use_bg_aug = cfg['TEST']['AUG']['TD']['BG']
        self.ts_bg_snr = cfg['TEST']['AUG']['TD']['BG_SNR']
        self.ts_bg_fps = []

        self.ts_ir_root_dir = cfg['TEST']['AUG']['TD']['IR_ROOT']
        self.ts_use_ir_aug = cfg['TEST']['AUG']['TD']['IR']
        self.ts_max_ir_dur = cfg['TEST']['AUG']['TD']['IR_MAX_DUR']
        self.ts_ir_fps = []

    def get_train_ds(self, reduce_items_p=100):
        """ Source (music) file paths for training set. 

        When segmented tracks are used for training the folder structure
        should be as follows:
            self.tr_audio_dir/
                dir0/
                    track1/
                        segment1.wav
                        ...
                    track2/
                        segment1.wav
                        ...
                    ...
                dir1/
                    track1/
                        segment1.wav
                        ...
                    track2/
                        segment1.wav
                        ...
                    ...
                ...

        If full length tracks are used for training the folder structure
        should be as follows:
            self.tr_audio_dir/
                dir0/
                    track1.wav
                    ...
                dir1/
                    track1.wav
                    ...
                ...

        Parameters
        ----------
            reduce_items_p : int (default 100)
                Reduce the number of items in each track to this percentage.
        """

        print("Creating the training dataset...")

        assert reduce_items_p>0 and reduce_items_p<=100, "reduce_items_p should be in (0, 100]"

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

        # Determine the dataset
        if "discotube" in self.tr_audio_dir.lower():

            # Find the tracks and their segments
            self.tr_source_fps = {}
            main_dirs = os.listdir(self.tr_audio_dir)
            for main_dir in main_dirs:
                track_names = os.listdir(os.path.join(self.tr_audio_dir, main_dir))
                for track_name in track_names:
                    track_dir = os.path.join(self.tr_audio_dir, main_dir, track_name)
                    segment_paths = sorted(glob.glob(track_dir + '/*.wav', recursive=True))
                    self.tr_source_fps[track_name] = segment_paths
            assert len(self.tr_source_fps)>0, "No training tracks found."
            total_segments = sum([len(v) for v in self.tr_source_fps.values()])
            assert total_segments>0, "No segments found."
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

            return SegmentDevLoader(
                segment_dict=self.tr_source_fps,
                segment_duration=self.segment_duration,
                full_segment_duration=self.cfg['TRAIN']['AUDIO']['INPUT_AUDIO_DUR'],
                fs=self.fs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                scale_output=self.scale_inputs,
                segments_per_track=self.tr_segments_per_track,
                bsz=self.tr_batch_sz,
                n_anchor=self.tr_n_anchor, #ex) bsz=40, n_anchor=8: 4 positive samples per anchor
                shuffle=True,
                random_offset_anchor=True,
                offset_duration=self.tr_offset_duration,
                bg_mix_parameter=[self.tr_use_bg_aug, self.tr_bg_fps, self.tr_bg_snr],
                ir_mix_parameter=[self.tr_use_ir_aug, self.tr_ir_fps, self.tr_max_ir_dur])

        # If the training tracks are from the NAFP FMA dataset
        elif "music" in self.tr_audio_dir.lower():

            # Find the wav tracks 
            self.tr_source_fps = sorted(
                glob.glob(self.tr_audio_dir + "**/*.wav", recursive=True))
            assert len(self.tr_source_fps)>0, "No training tracks found."
            print(f"{len(self.tr_source_fps):,} tracks found.")

            if reduce_items_p<100:
                print(f"Reducing the number of tracks used to {reduce_items_p}%")
                self.tr_source_fps = {k: v
                                    for i,(k,v) in enumerate(self.tr_source_fps.items())
                                    if i < int(len(self.tr_source_fps)*reduce_items_p/100)}
                print(f"Reduced to {len(self.tr_source_fps):,} tracks.")

            return TrackDevLoader(
                track_paths=self.tr_source_fps,
                segment_duration=self.segment_duration,
                hop_duration=self.cfg['TRAIN']['AUDIO']['SEGMENT_HOP_DUR'],
                fs=self.fs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                scale_output=self.scale_inputs,
                segments_per_track=self.tr_segments_per_track,
                bsz=self.tr_batch_sz,
                n_anchor=self.tr_n_anchor, #ex) bsz=40, n_anchor=8: 4 positive samples per anchor
                shuffle=True,
                random_offset_anchor=True,
                offset_duration=self.tr_offset_duration,
                bg_mix_parameter=[self.tr_use_bg_aug, self.tr_bg_fps, self.tr_bg_snr],
                ir_mix_parameter=[self.tr_use_ir_aug, self.tr_ir_fps, self.tr_max_ir_dur])

        else:
            raise ValueError("Invalid training tracks directory.")

    def get_val_ds(self):
        """ Source (music) file paths for validation set. 

        When segmented tracks are used for training the folder structure
        should be as follows:
            self.val_audio_dir/
                dir0/
                    track1/
                        segment1.wav
                        ...
                    track2/
                        segment1.wav
                        ...
                    ...
                dir1/
                    track1/
                        segment1.wav
                        ...
                    track2/
                        segment1.wav
                        ...
                    ...
                ...

        If full length tracks are used for training the folder structure
        should be as follows:
            self.val_audio_dir/
                dir0/
                    track1.wav
                    ...
                dir1/
                    track1.wav
                    ...
                ...        """

        print(f"Creating the validation dataset...")

        # Find the augmentation files
        if self.tr_use_bg_aug:
            print(f"val_bg_fps: {len(self.tr_bg_fps):>6,} (Same as the training set)")
        if self.tr_use_ir_aug:
            print(f"val_ir_fps: {len(self.tr_ir_fps):>6,} (Same as the training set)")

        # Determine the dataset
        if "discotube" in self.val_audio_dir.lower():

            # Find the tracks and their segments
            self.val_source_fps = {}
            main_dirs = os.listdir(self.val_audio_dir)
            for main_dir in main_dirs:
                track_names = os.listdir(os.path.join(self.val_audio_dir, main_dir))
                for track_name in track_names:
                    track_dir = os.path.join(self.val_audio_dir, main_dir, track_name)
                    segment_paths = sorted(glob.glob(track_dir + '/*.wav', recursive=True))
                    self.val_source_fps[track_name] = segment_paths
            assert len(self.val_source_fps)>0, "No validation tracks found."
            total_segments = sum([len(v) for v in self.val_source_fps.values()])
            assert total_segments>0, "No segments found."
            print(f"{len(self.val_source_fps):,} tracks found.")
            print(f"{total_segments:,} segments found.")

            return SegmentDevLoader(
                segment_dict=self.val_source_fps,
                segment_duration=self.segment_duration,
                full_segment_duration=self.cfg['TRAIN']['AUDIO']['INPUT_AUDIO_DUR'],
                fs=self.fs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                segments_per_track=self.tr_segments_per_track,
                scale_output=self.scale_inputs,
                bsz=self.tr_batch_sz,
                n_anchor=self.tr_n_anchor,
                shuffle=False,
                random_offset_anchor=False,
                bg_mix_parameter=[self.tr_use_bg_aug, self.tr_bg_fps, self.tr_bg_snr],
                ir_mix_parameter=[self.tr_use_ir_aug, self.tr_ir_fps, self.tr_max_ir_dur])

        # If the validation tracks are from the NAFP FMA dataset
        elif "music" in self.val_audio_dir.lower():

            # Find the wav tracks
            self.val_source_fps = sorted(
                glob.glob(self.val_audio_dir + '**/*.wav', recursive=True))
            assert len(self.val_source_fps)>0, "No validation tracks found."
            print(f"{len(self.val_source_fps):,} tracks found.")

            return TrackDevLoader(
                track_paths=self.tr_source_fps,
                segment_duration=self.segment_duration,
                hop_duration=self.cfg['TRAIN']['AUDIO']['SEGMENT_HOP_DUR'],
                fs=self.fs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                scale_output=self.scale_inputs,
                segments_per_track=self.tr_segments_per_track,
                bsz=self.tr_batch_sz,
                n_anchor=self.tr_n_anchor, #ex) bsz=40, n_anchor=8: 4 positive samples per anchor
                shuffle=False,
                random_offset_anchor=False,
                bg_mix_parameter=[self.tr_use_bg_aug, self.tr_bg_fps, self.tr_bg_snr],
                ir_mix_parameter=[self.tr_use_ir_aug, self.tr_ir_fps, self.tr_max_ir_dur])

        else:
            raise ValueError("Invalid validation tracks directory.")

    def get_test_noise_ds(self):
        """ Test-dummy-DB without augmentation. Adds noise tracks to the DB.
        Supports both discotube and nafp datasets. The folder structure 
        should be as follows:
            self.ts_noise_tracks_dir/
                dir0/
                    track1.wav
                    track2.wav
                    ...
                dir1/
                    track1.wav
                    track2.wav
                    ...
                ...

        Returns:
        --------
            ds_dummy_db : genUnbalSequenceGeneration
                The dataset for test-dummy-DB.
        """

        print(f"Creating the test-dummy-DB dataset (noise tracks)...")

        # Find the noise tracks and their segments
        self.ts_noise_paths = sorted(
            glob.glob(self.ts_noise_tracks_dir+ '/**/*.wav', 
                    recursive=True))
        assert len(self.ts_noise_paths)>0, "No noise tracks found."
        print(f"{len(self.ts_noise_paths):,} noise tracks found.")

        return GenerationLoader(
            track_paths=self.ts_noise_paths,
            segment_duration=self.segment_duration,
            hop_duration=self.ts_segment_hop,
            fs=self.fs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            scale_output=self.scale_inputs,
            bsz=self.ts_batch_sz
            )

    def get_test_query_ds(self):
        """ Create 2 databases for query segments. One of them is the augmented 
        version of the clean queries. If the config does not specify a folder for
        augmented queries, then the clean queries are augmented in real time.
        Supports both discotube and nafp datasets. Query tracks folders structure 
        should be as follows:
            self.ts_clean_query_tracks_dir/
                dir0/
                    track1.wav
                    track2.wav
                    ...
                dir1/
                    track1.wav
                    track2.wav
                    ...
                ...

        Returns
        -------
            (ds_query, ds_db)
                ds_query is the augmented version of the clean queries.
                ds_db is the clean queries without augmentation.

        """

        print("Creating the clean query dataset...")

        # Find the clean query tracks and their segments
        self.ts_query_clean = sorted(
                glob.glob(self.ts_clean_query_tracks_dir + '/**/*.wav', 
                        recursive=True))
        assert len(self.ts_query_clean)>0, "No clean query tracks found."
        print(f"{len(self.ts_query_clean):,} clean query tracks found.")

        # Create the clean query dataset
        ds_db = GenerationLoader(
            track_paths=self.ts_query_clean,
            segment_duration=self.segment_duration,
            hop_duration=self.ts_segment_hop,
            fs=self.fs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            scale_output=self.scale_inputs,
            bsz=self.ts_batch_sz
            )

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
            ds_query = GenerationLoader(
                track_paths=self.ts_query_clean, # Augment the clean query tracks
                segment_duration=self.segment_duration,
                hop_duration=self.ts_segment_hop,
                fs=self.fs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                scale_output=self.scale_inputs,
                bg_mix_parameter=[self.ts_use_bg_aug, self.ts_bg_fps, self.ts_bg_snr],
                ir_mix_parameter=[self.ts_use_ir_aug, self.ts_ir_fps, self.ts_max_ir_dur],
                bsz=self.ts_batch_sz
                )

        elif (not self.ts_use_bg_aug) and (not self.ts_use_ir_aug):

            print("Using pre-augmented query tracks.")

            # Find the augmented query tracks and their segments
            self.ts_query_augmented = sorted(
                glob.glob(self.ts_augmented_query_tracks_dir + '/**/*.wav', 
                        recursive=True))
            assert len(self.ts_query_augmented)>0, "No augmented query tracks found."
            print(f"{len(self.ts_query_augmented):,} augmented query tracks found")

            # Create the augmented query dataset
            ds_query = GenerationLoader(
                track_paths=self.ts_query_augmented,
                segment_duration=self.segment_duration,
                hop_duration=self.ts_segment_hop,
                fs=self.fs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                scale_output=self.scale_inputs,
                bsz=self.ts_batch_sz
                )

        else:

                # For now we do not support single augmentation
                raise ValueError("Invalid augmentation parameters.")

        return ds_query, ds_db

    def get_custom_db_ds(self, source_root_dir):
        """ Construct DB (or query) from custom source files. """

        fps = sorted(
            glob.glob(source_root_dir + '/**/*.wav', recursive=True))

        return GenerationLoader(
            track_paths=fps,
            segment_duration=self.segment_duration,
            hop_duration=self.ts_segment_hop,
            fs=self.fs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            scale_output=self.scale_inputs,
            bsz=self.ts_batch_sz
            )# No augmentations, No drop-samples.