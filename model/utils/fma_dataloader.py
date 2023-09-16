import numpy as np
from tensorflow.keras.utils import Sequence

from model.fp.melspec.melspectrogram import Melspec_layer_essentia
from model.utils import audio_utils

class genUnbalSequence(Sequence):
    def __init__(
        self,
        track_paths,
        segment_duration=1,
        hop_duration=.5,
        fs=8000,
        n_fft=1024,
        stft_hop=256,
        n_mels=256,
        f_min=300,
        f_max=4000,
        scale_output=True,
        segments_per_track=59,
        bsz=120,
        n_anchor=60,
        shuffle=False,
        random_offset_anchor=False,
        offset_duration=0.2,
        bg_mix_parameter=[False],
        ir_mix_parameter=[False],
        ):
        """
        Parameters
        ----------
        track_paths : list(str), 
            Track file paths as a list. 
        segment_duration : (float), optional
            Duration of an audio segment in seconds. Default is 1.
        fs : (int), optional
            Sampling rate. Default is 8000.
        n_fft: (int), optional
            FFT size. Default is 1024.
        stft_hop : (int), optional
            STFT hop-size. Default is 256.
        n_mels : (int), optional
            Number of mel-bands. Default is 256.
        f_min : (int), optional
            Minimum frequency of the mel-bands. Default is 300.
        f_max : (int), optional
            Maximum frequency of the mel-bands. Default is 4000.
        scale_output : (bool), optional
            Scale the power mel-spectrogram to [-1, 1]. Default is True.
        bsz : (int), optional
            Batch size. Default is 120.
        n_anchor : (int), optional
            ex) bsz=40, n_anchor=8 --> 4 positive samples for each anchor. 
            Default is 60.
        shuffle : (bool), optional
            We read the tracks and Augmentation files with alphabetical order.
            If shuffle is True, we shuffle the order of the tracks and the
            augmentation files at the end of each epoch. Also the order of the
            segments of each track is shuffled at the end of each epoch.
            Default is False.
        random_offset_anchor : (bool), optional
            DESCRIPTION. Default is False.
        offset_duration : (float), optional
            How much a segment can be offsetted from its center in seconds. 
            Default is 0.2.
        bg_mix_parameter : list([(bool), list(str), (int, int)]), optional
            [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR)]. Default is [False].
        ir_mix_parameter : list([(bool), list(str), float], optional
            [True, IR_FILEPATHS, MAX_IR_DURATION]. Default is [False].
        """

        # Check parameters
        assert bsz >= n_anchor, "bsz should be >= n_anchor"
        assert n_anchor > 0, "n_anchor should be > 0"
        assert segments_per_track > 0, "segments_per_track should be > 0"
        assert segment_duration > 0, "segment_duration should be > 0"
        assert hop_duration > 0, "hop_duration should be > 0"
        assert segment_duration >= hop_duration, "segment_duration should be >= hop_duration"

        # Save the Input parameters
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.offset_duration = offset_duration
        self.max_offset_sample = int(self.offset_duration * fs)
        self.fs = fs

        # Melspec layer
        self.mel_spec = Melspec_layer_essentia(scale=scale_output,
                                            n_fft=n_fft, 
                                            stft_hop=stft_hop, 
                                            n_mels=n_mels, 
                                            fs=fs, 
                                            dur=segment_duration, 
                                            f_min=f_min, 
                                            f_max=f_max)

        # Training parameters
        self.segments_per_track = segments_per_track
        self.random_offset_anchor = random_offset_anchor
        self.bg_hop = segment_duration # For bg samples hop is equal to duration
        self.bsz = bsz
        self.n_anchor = n_anchor
        if bsz != n_anchor:
            self.n_pos_per_anchor = round((bsz - n_anchor) / n_anchor)
            self.n_pos_bsz = bsz - n_anchor
        else:
            self.n_pos_per_anchor = 0
            self.n_pos_bsz = 0
        self.shuffle = shuffle

        # self.reduce_items_p = reduce_items_p

        # Create segment information for each track
        self.track_seg_dict = audio_utils.get_fns_seg_dict(track_paths,
                                                        segment_mode='all',
                                                        fs=self.fs,
                                                        duration=self.segment_duration,
                                                        hop=self.hop_duration)
        # Filter out the tracks with less than segments_per_track
        self.track_seg_dict = {k: v 
                                for k, v in self.track_seg_dict.items() 
                                if len(v) >= segments_per_track}
        print(f"Number of tracks with at least {segments_per_track}"
              f" segments: {len(self.track_seg_dict):,}")

        # Filter out the tracks with less than segments_per_track
        self.track_seg_dict = {k: v 
                                for k, v in self.track_seg_dict.items() 
                                if len(v) >= segments_per_track}

        # Keep only segments_per_track segments for each track
        self.track_seg_dict = {k: v[:segments_per_track] 
                                for k, v in self.track_seg_dict.items()}

        # Determine the tracks to use at each epoch
        # Remove the tracks that do not fill the last batch. Each batch contains
        # a single segment from n_anchor tracks.
        self.n_tracks = int((len(self.track_seg_dict) // n_anchor) * n_anchor)
        self.track_seg_dict = {k: v 
                                for i, (k, v) in enumerate(self.track_seg_dict.items()) 
                                if i < self.n_tracks}
        self.n_samples = self.n_tracks * self.segments_per_track

        self.track_fnames = list(self.track_seg_dict.keys())

        # Save augmentation parameters, read the files, and store them in memory
        self.load_and_store_bg_samples(bg_mix_parameter)
        self.load_and_store_ir_samples(ir_mix_parameter)

    def __len__(self):
        """ Returns the number of batches per epoch. An epoch is defined as
        when all the segments of each track are seen once."""

        return int(self.n_samples//self.n_anchor)