import glob
from pathlib import Path

from model.utils.dataloader_keras import SegmentLoader, ChunkLoader
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

    def __init__(self, cfg=dict(), is_training=True):
        """ Initialize the dataset object."""

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

        # Only read the related parameters.
        if is_training:

            # Train Parameters
            self.tr_audio_dir = cfg['TRAIN']['DIR']['TRAIN_ROOT']

            self.tr_audio_type = cfg['TRAIN']['AUDIO']['TYPE']
            self.tr_offset_duration = cfg['TRAIN']['AUDIO']["MAX_OFFSET_DUR"]
            self.tr_batch_sz = cfg['TRAIN']['TR_BATCH_SZ']

            self.tr_bg_root_dir = cfg['TRAIN']['AUG']['TD']['BG_ROOT']
            self.tr_use_bg_aug = cfg['TRAIN']['AUG']['TD']['BG']
            self.tr_bg_snr = cfg['TRAIN']['AUG']['TD']['BG_SNR']
            self.tr_bg_amp_range = cfg['TRAIN']['AUG']['TD']['BG_AMP_RANGE']
            self.tr_bg_fps = []

            self.tr_ir_root_dir = cfg['TRAIN']['AUG']['TD']['IR_ROOT']
            self.tr_use_ir_aug = cfg['TRAIN']['AUG']['TD']['IR']
            self.tr_max_ir_dur = cfg['TRAIN']['AUG']['TD']['IR_MAX_DUR']
            self.tr_ir_fps = []

            # Validation Parameters
            self.val_audio_dir = cfg['TRAIN']['DIR']['VAL_ROOT']
            self.val_batch_sz = cfg['TRAIN']['VAL_BATCH_SZ']
            # We use the same augmentations for train and validation sets

        else:

            # Test Parameters
            self.ts_noise_tracks_dir = cfg['TEST']['DIR']['NOISE_ROOT']
            self.ts_clean_query_tracks_dir = cfg['TEST']['DIR']['CLEAN_QUERY_ROOT']
            self.ts_augmented_query_tracks_dir = cfg['TEST']['DIR']['AUGMENTED_QUERY_ROOT']

            self.ts_segment_hop = cfg['TEST']['SEGMENT_HOP']
            self.ts_batch_sz = cfg['TEST']['BATCH_SZ']

            self.ts_bg_root_dir = cfg['TEST']['AUG']['TD']['BG_ROOT']
            self.ts_use_bg_aug = cfg['TEST']['AUG']['TD']['BG']
            self.ts_bg_snr = cfg['TEST']['AUG']['TD']['BG_SNR']
            self.ts_bg_amp_range = cfg['TEST']['AUG']['TD']['BG_AMP_RANGE']
            self.ts_bg_fps = []

            self.ts_ir_root_dir = cfg['TEST']['AUG']['TD']['IR_ROOT']
            self.ts_use_ir_aug = cfg['TEST']['AUG']['TD']['IR']
            self.ts_max_ir_dur = cfg['TEST']['AUG']['TD']['IR_MAX_DUR']
            self.ts_ir_fps = []

            # Check if the augmented query tracks are provided
            if self.ts_augmented_query_tracks_dir is not None:
                assert not (self.ts_use_bg_aug or self.ts_use_ir_aug), \
                    "Augmented query tracks are provided, so augmentation should not be enabled."

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

        # Determine the input audio type
        if self.tr_audio_type.lower() == "segment":

            # Find the tracks and their segments
            tr_audio_path = Path(self.tr_audio_dir)
            self.tr_source_fps = {
                track_name.name: sorted(track_name.glob('*.wav'))
                for main_dir in tr_audio_path.iterdir()
                for track_name in main_dir.iterdir()
            }

            assert self.tr_source_fps, "No validation tracks found."
            total_segments = sum(len(segments) for segments in self.tr_source_fps.values())
            assert total_segments > 0, "No segments found."

            print(f"{len(self.tr_source_fps):,} tracks found.")
            print(f"{total_segments:,} segments found.")

            # Reduce the number of items in each track if requested
            if reduce_items_p<100:
                print(f"Reducing the number of tracks used to {reduce_items_p}%")
                self.tr_source_fps = {
                    k: v
                    for i,(k,v) in enumerate(self.tr_source_fps.items())
                    if i < int(len(self.tr_source_fps)*reduce_items_p/100)
                }
                print(f"Reduced to {len(self.tr_source_fps):,} tracks.")
                total_segments = sum([len(v) for v in self.tr_source_fps.values()])
                print(f"Reduced to {total_segments:,} segments.")

            # Find the augmentation files after the music is loaded for nice print
            self._read_train_augmentations()

            return SegmentLoader(
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
                segments_per_track=self.cfg['TRAIN']['AUDIO']['SEGMENTS_PER_TRACK'],
                bsz=self.tr_batch_sz,
                shuffle=True,
                random_offset_anchor=True,
                offset_duration=self.tr_offset_duration,
                bg_mix_parameter=self.tr_bg_parameters,
                ir_mix_parameter=self.tr_ir_parameters)

        elif self.tr_audio_type.lower() == "chunk":

            # Find the wav tracks
            self.tr_source_fps = sorted(
                glob.glob(self.tr_audio_dir + "**/*.wav", recursive=True))
            assert len(self.tr_source_fps)>0, "No training tracks found."
            print(f"{len(self.tr_source_fps):,} tracks found.")

            # Reduce the total number of tracks if requested
            if reduce_items_p<100:
                print(f"Reducing the number of tracks used to {reduce_items_p}%")
                self.tr_source_fps = self.tr_source_fps[:int(len(self.tr_source_fps)*reduce_items_p/100)]
                print(f"Reduced to {len(self.tr_source_fps):,} tracks.")

            # Find the augmentation files after the music is loaded
            self._read_train_augmentations()

            return ChunkLoader(
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
                segments_per_track=self.cfg['TRAIN']['AUDIO']['SEGMENTS_PER_TRACK'],
                bsz=self.tr_batch_sz,
                shuffle=True,
                random_offset_anchor=True,
                offset_duration=self.tr_offset_duration,
                bg_mix_parameter=self.tr_bg_parameters,
                ir_mix_parameter=self.tr_ir_parameters)

        else:

            raise ValueError("Invalid audio type. We accept 'segment' or 'chunk'.")

    def get_val_ds(self, reduce_items_p=100):
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
                ...        

        Parameters
        ----------
            reduce_items_p : int (default 100)
                Reduce the number of items in each track to this percentage.

        """

        print(f"Creating the validation dataset...")

        # Determine the input audio type
        if self.tr_audio_type.lower() == "segment":

            # Find the tracks and their segments
            val_audio_path = Path(self.val_audio_dir)
            self.val_source_fps = {
                track_name.name: sorted(track_name.glob('*.wav'))
                for main_dir in val_audio_path.iterdir()
                for track_name in main_dir.iterdir()
            }

            assert self.val_source_fps, "No validation tracks found."
            total_segments = sum(len(segments) for segments in self.val_source_fps.values())
            assert total_segments > 0, "No segments found."

            print(f"{len(self.val_source_fps):,} tracks found.")
            print(f"{total_segments:,} segments found.")

            # Reduce the number of items in each track if requested
            if reduce_items_p<100:
                print(f"Reducing the number of tracks used to {reduce_items_p}%")
                self.val_source_fps = {k: v
                                    for i,(k,v) in enumerate(self.val_source_fps.items())
                                    if i < int(len(self.val_source_fps)*reduce_items_p/100)}
                print(f"Reduced to {len(self.val_source_fps):,} tracks.")
                total_segments = sum([len(v) for v in self.val_source_fps.values()])
                print(f"Reduced to {total_segments:,} segments.")

            self._get_val_augmentations()

             # Most of the parameters are same as the training set
            return SegmentLoader(
                segment_dict=self.val_source_fps,
                segment_duration=self.segment_duration,
                full_segment_duration=self.cfg['TRAIN']['AUDIO']['INPUT_AUDIO_DUR'],
                fs=self.fs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                segments_per_track=self.cfg['TRAIN']['AUDIO']['SEGMENTS_PER_TRACK'],
                scale_output=self.scale_inputs,
                bsz=self.tr_batch_sz,
                shuffle=False,
                random_offset_anchor=True,
                offset_duration=self.tr_offset_duration,
                bg_mix_parameter=self.tr_bg_parameters,
                ir_mix_parameter=self.tr_ir_parameters)

        elif self.tr_audio_type.lower() == "chunk":

            # Find the wav tracks
            self.val_source_fps = sorted(
                glob.glob(self.val_audio_dir + '**/*.wav', recursive=True))
            assert len(self.val_source_fps)>0, "No validation tracks found."
            print(f"{len(self.val_source_fps):,} tracks found.")

            # Reduce the total number of tracks if requested
            if reduce_items_p<100:
                print(f"Reducing the number of tracks used to {reduce_items_p}%")
                self.val_source_fps = self.val_source_fps[:int(len(self.val_source_fps)*reduce_items_p/100)]
                print(f"Reduced to {len(self.val_source_fps):,} tracks.")

            self._get_val_augmentations()

            return ChunkLoader(
                track_paths=self.val_source_fps,
                segment_duration=self.segment_duration,
                hop_duration=self.cfg['TRAIN']['AUDIO']['SEGMENT_HOP_DUR'],
                fs=self.fs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                scale_output=self.scale_inputs,
                segments_per_track=self.cfg['TRAIN']['AUDIO']['SEGMENTS_PER_TRACK'],
                bsz=self.tr_batch_sz,
                shuffle=False,
                random_offset_anchor=True,
                bg_mix_parameter=self.tr_bg_parameters,
                ir_mix_parameter=self.tr_ir_parameters)

        else:

            raise ValueError("Invalid audio type. We accept 'segment' or 'chunk'.")

    def get_test_dummy_db_ds(self):
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
            bsz=self.ts_batch_sz,
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
            bsz=self.ts_batch_sz,
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

            # Augmentation parameters
            self.ts_bg_parameters = [self.ts_use_bg_aug, 
                                    self.ts_bg_fps, 
                                    self.ts_bg_snr, 
                                    self.ts_bg_amp_range]
            self.ts_ir_parameters = [self.ts_use_ir_aug,
                                    self.ts_ir_fps,
                                    self.ts_max_ir_dur]

            # Create the augmented query dataset
            # Returns only the augmented segments, not the clean ones
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
                bg_mix_parameter=self.ts_bg_parameters,
                ir_mix_parameter=self.ts_ir_parameters,
                bsz=self.ts_batch_sz,
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
                bsz=self.ts_batch_sz,
                )

        else:

            # For now we do not support single augmentation
            raise ValueError("Invalid augmentation parameters.")

        return ds_query, ds_db

    def get_custom_db_ds(self, source_root_dir):
        """ Construct DB (or query) from custom source files. """

        fps = sorted(
            glob.glob(source_root_dir + '/**/*.wav', recursive=True))
        print(f"{len(fps):,} tracks found.")

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
            bsz=self.ts_batch_sz,
            )

    def _read_train_augmentations(self):

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

        # Collect the augmentation parameters
        self.tr_bg_parameters = [self.tr_use_bg_aug, 
                                self.tr_bg_fps, 
                                self.tr_bg_snr, 
                                self.tr_bg_amp_range]
        self.tr_ir_parameters = [self.tr_use_ir_aug,
                                self.tr_ir_fps,
                                self.tr_max_ir_dur]

    def _get_val_augmentations(self):

        # Find the augmentation files
        if self.tr_use_bg_aug:
            print(f"val_bg_fps: {len(self.tr_bg_fps):>6,} (Same as the training set)")
        if self.tr_use_ir_aug:
            print(f"val_ir_fps: {len(self.tr_ir_fps):>6,} (Same as the training set)")
