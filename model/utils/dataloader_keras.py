# -*- coding: utf-8 -*-
""" dataloader_keras.py """
from tensorflow.keras.utils import Sequence
from model.utils.audio_utils import (bg_mix_batch, ir_aug_batch, load_audio,
                                     get_fns_seg_list, load_audio_multi_start)
from model.fp.melspec.melspectrogram import Melspec_layer_essentia
import numpy as np

# OGUZ: Is this a good idea?
MAX_IR_LENGTH = 600 # 50ms with fs=8000

class genUnbalSequence(Sequence):
    def __init__(
        self,
        fns_event_list,
        bsz=120,
        n_anchor=60,
        duration=1,
        hop=.5,
        fs=8000,
        segment_norm=True,
        n_fft=1024,
        stft_hop=256,
        n_mels=256,
        f_min=300,
        f_max=4000,
        shuffle=False,
        seg_mode="all",
        amp_mode='normal',
        random_offset_anchor=False,
        offset_margin_hop_rate=0.4,
        bg_mix_parameter=[False],
        ir_mix_parameter=[False],
        reduce_items_p=0,
        reduce_batch_first_half=False,
        experimental_mode=False,
        drop_the_last_non_full_batch=True
        ):
        """
        Parameters
        ----------
        fns_event_list : list(str), 
            Song file paths as a list. 
            [[filename, seg_idx, offset_min, offset_max], [ ... ] , ... [ ... ]]
        bsz : (int), optional
            In TPUs code, global batch size. The default is 120.
        n_anchor : TYPE, optional
            ex) bsz=40, n_anchor=8 --> 4 positive samples for each anchor
            (In TPUs code, global n_anchor). The default is 60.
        duration : (float), optional
            Duration in seconds. The default is 1.
        hop : (float), optional
            Hop-size in seconds. The default is .5.
        fs : (int), optional
            Sampling rate. The default is 8000.
        shuffle : (bool), optional
            Randomize samples from the original songs. BG/IRs will not be 
            affected by this parameter (BG/IRs are always shuffled). 
            The default is False.
        seg_mode : (str), optional
            DESCRIPTION. The default is "all".
        amp_mode : (str), optional
            DESCRIPTION. The default is 'normal'.
        random_offset_anchor : (bool), optional
            DESCRIPTION. The default is False.
        offset_margin_hop_rate : (float), optional
            For example, 0.4 means max 40 % overlaps. The default is 0.4.
        bg_mix_parameter : list([(bool), list(str), (int, int)]), optional
            [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR)]. The default is [False].
        ir_mix_parameter : list([(bool), list(str)], optional
            [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR)]. The default is [False].
        reduce_items_p : (int), optional
            Reduce dataset size to percent (%). Useful when debugging code 
            with small data. The default is 0.
        reduce_batch_first_half : (bool), optional
            Remove the first half of elements from each output batch. The
            resulting output batch will contain only replicas. This is useful
            when collecting synthesized queries only. Default is False.
        experimental_mode : (bool), optional
            In experimental mode, we use a set of pre-defined offsets for
            the multiple positive samples.. The default is False.
        drop_the_last_non_full_batch : (bool), optional
            Set as False in test. Default is True.
        """

        # Input parameters
        self.duration = duration
        self.hop = hop
        self.fs = fs
        self.shuffle = shuffle
        self.seg_mode = seg_mode
        self.amp_mode = amp_mode
        self.random_offset_anchor = random_offset_anchor
        self.offset_margin_hop_rate = offset_margin_hop_rate
        self.offset_margin_frame = int(hop * self.offset_margin_hop_rate * fs)

        # Melspec layer
        self.mel_spec = Melspec_layer_essentia(input_shape=(1, int(fs * duration)), 
                                            segment_norm=segment_norm,
                                            n_fft=n_fft, 
                                            stft_hop=stft_hop, 
                                            n_mels=n_mels, 
                                            fs=fs, 
                                            dur=duration, 
                                            f_min=f_min, 
                                            f_max=f_max)

        # Save bg_mix and ir_mix parameters
        self.bg_mix = bg_mix_parameter[0]
        self.ir_mix = ir_mix_parameter[0]
        if self.bg_mix == True:
            fns_bg_list = bg_mix_parameter[1]
            self.bg_snr_range = bg_mix_parameter[2]
        if self.ir_mix == True:
            fns_ir_list = ir_mix_parameter[1]

        # TODO: understand seg_mode
        if self.seg_mode in {'random_oneshot', 'all'}:
            self.fns_event_seg_list = get_fns_seg_list(fns_event_list,
                                                       self.seg_mode,
                                                       self.fs,
                                                       self.duration,
                                                       hop=self.hop)
        else:
            raise NotImplementedError("seg_mode={}".format(self.seg_mode))

        # Training parameters
        self.bsz = bsz
        self.n_anchor = n_anchor
        if bsz != n_anchor:
            self.n_pos_per_anchor = round((bsz - n_anchor) / n_anchor)
            self.n_pos_bsz = bsz - n_anchor
        else:
            self.n_pos_per_anchor = 0
            self.n_pos_bsz = 0

        self.drop_the_last_non_full_batch = drop_the_last_non_full_batch
        if self.drop_the_last_non_full_batch:
            self.n_samples = int((len(self.fns_event_seg_list) // n_anchor) * n_anchor)
        else:
            self.n_samples = len(self.fns_event_seg_list) # fp-generation

        if self.shuffle == True:
            self.index_event = np.random.permutation(self.n_samples)
        else:
            self.index_event = np.arange(self.n_samples)

        if self.bg_mix == True:
            self.fns_bg_seg_list = get_fns_seg_list(fns_bg_list, 
                                                    'all',
                                                    self.fs, 
                                                    self.duration)
            self.n_bg_samples = len(self.fns_bg_seg_list)
            if self.shuffle == True:
                self.index_bg = np.random.permutation(self.n_bg_samples)
            else:
                self.index_bg = np.arange(self.n_bg_samples)

        if self.ir_mix == True:
            self.fns_ir_seg_list = get_fns_seg_list(fns_ir_list, 
                                                    'first',
                                                    self.fs, 
                                                    self.duration)
            self.n_ir_samples = len(self.fns_ir_seg_list)
            if self.shuffle == True:
                self.index_ir = np.random.permutation(self.n_ir_samples)
            else:
                self.index_ir = np.arange(self.n_ir_samples)

        assert(reduce_items_p <= 100)
        self.reduce_items_p = reduce_items_p
        self.reduce_batch_first_half = reduce_batch_first_half
        self.experimental_mode = experimental_mode
        if experimental_mode:
            self.experimental_mode_offset_sec_list = (
                (np.arange(self.n_pos_per_anchor) -
                 (self.n_pos_per_anchor - 1) / 2) /
                self.n_pos_per_anchor) * self.hop

    def __len__(self):
        """ Returns the number of batches per epoch. """
        if self.reduce_items_p != 0:
            return int(
                np.ceil(self.n_samples / float(self.n_anchor)) *
                (self.reduce_items_p / 100))
        else:
            return int(np.ceil(self.n_samples / float(self.n_anchor)))

    def on_epoch_end(self):
        """ Re-shuffle """

        if self.shuffle == True:
            self.index_event = list(np.random.permutation(self.n_samples))

        if self.bg_mix == True and self.shuffle == True:
            self.index_bg = list(np.random.permutation(
                self.n_bg_samples))  # same number with event samples

        if self.ir_mix == True and self.shuffle == True:
            self.index_ir = list(np.random.permutation(
                self.n_ir_samples))  # same number with event samples

    def __getitem__(self, idx):
        """ Get anchor (original) and positive (replica) samples of audio and their power mel-spectrograms.

        Returns:
            Xa_batch: anchor samples (audio)
            Xp_batch: positive samples (audio)
            Xa_batch_mel: power mel-spectrogram of anchor samples
            Xp_batch_mel: power-mel spectrogram of positive samples"""

        global bg_sel_indices

        # Prepare indices for batch. n_anchor consecutive samples are taken at each iteration
        index_anchor_for_batch = self.index_event[idx*self.n_anchor:(idx + 1)*self.n_anchor]

        # Load anchor and positive audio samples
        Xa_batch, Xp_batch = self.__event_batch_load(index_anchor_for_batch)

        # If there are positive samples, check for augmentation
        if self.n_pos_bsz > 0:

            if self.bg_mix == True:
                # Prepare bg for positive samples
                bg_sel_indices = np.arange(idx*self.n_pos_bsz, (idx+1)*self.n_pos_bsz) % self.n_bg_samples
                index_bg_for_batch = self.index_bg[bg_sel_indices]
                # TODO: after determining the idx, just return from memory here
                Xp_bg_batch = self.__bg_batch_load(index_bg_for_batch)
                # mix
                Xp_batch = bg_mix_batch(Xp_batch,
                                        Xp_bg_batch,
                                        snr_range=self.bg_snr_range)

            if self.ir_mix == True:
                # Prepare ir for positive samples
                ir_sel_indices = np.arange(
                    idx * self.n_pos_bsz,
                    (idx + 1) * self.n_pos_bsz) % self.n_ir_samples
                index_ir_for_batch = self.index_ir[ir_sel_indices]
                Xp_ir_batch = self.__ir_batch_load(index_ir_for_batch)
                # ir aug
                Xp_batch = ir_aug_batch(Xp_batch, Xp_ir_batch)

        # Compute mel spectrograms
        Xa_batch_mel = self.mel_spec.compute_batch(Xa_batch)
        Xp_batch_mel = self.mel_spec.compute_batch(Xp_batch)

        # Fix the dimensions
        Xa_batch_mel = np.expand_dims(Xa_batch_mel, 3).astype(np.float32) # (n_anchor, n_mels, T, 1)
        if Xp_batch_mel.size>0:
            Xp_batch_mel = np.expand_dims(Xp_batch_mel, 3).astype(np.float32) # (n_pos, n_mels, T, 1)

        return Xa_batch, Xp_batch, Xa_batch_mel, Xp_batch_mel

    def __event_batch_load(self, anchor_idx_list):
        """ Get Xa_batch (anchor-original) and Xp_batch (positive replica) and samples of audio."""

        Xa_batch = None
        Xp_batch = None
        for idx in anchor_idx_list:  # idx: index for one sample

            # Determine anchor_start_sec by finding offset range for anchor
            _,_,offset_min, offset_max = self.fns_event_seg_list[idx]
            anchor_offset_min = np.max([offset_min, -self.offset_margin_frame])
            anchor_offset_max = np.min([offset_max, self.offset_margin_frame])
            if (self.random_offset_anchor == True) & (self.experimental_mode== False):
                # TODO: This is not a good way to generate random offset.
                # Usually, we can apply random offset to anchor only in training.
                np.random.seed(idx)
                # Calculate anchor_start_sec
                _anchor_offset_frame = np.random.randint(low=anchor_offset_min, high=anchor_offset_max)
                _anchor_offset_sec = _anchor_offset_frame / self.fs
                anchor_start_sec = self.fns_event_seg_list[idx][1] * self.hop + _anchor_offset_sec
            else:
                _anchor_offset_frame = 0
                anchor_start_sec = self.fns_event_seg_list[idx][1] * self.hop

            # Calculate multiple(=self.n_pos_per_anchor) pos_start_sec
            if self.n_pos_per_anchor > 0:
                pos_offset_min = np.max([(_anchor_offset_frame - self.offset_margin_frame),offset_min])
                pos_offset_max = np.min([(_anchor_offset_frame + self.offset_margin_frame),offset_max])
                if self.experimental_mode:
                    # In experimental_mode, we use a set of pre-defined offset for multiple positive replicas...
                    _pos_offset_sec_list = self.experimental_mode_offset_sec_list # [-0.2, -0.1,  0. ,  0.1,  0.2] for n_pos=5 with hop=0.5s
                    _pos_offset_sec_list[(
                        _pos_offset_sec_list <
                        pos_offset_min / self.fs)] = pos_offset_min / self.fs
                    _pos_offset_sec_list[(
                        _pos_offset_sec_list >
                        pos_offset_max / self.fs)] = pos_offset_max / self.fs
                    pos_start_sec_list = self.fns_event_seg_list[idx][1] * self.hop + _pos_offset_sec_list
                else:
                    if pos_offset_min==pos_offset_max==0:
                        # Only the case of running extras/dataset2wav.py as offset_margin_hot_rate=0
                        pos_start_sec_list = self.fns_event_seg_list[idx][1] * self.hop
                        pos_start_sec_list = [pos_start_sec_list]
                    else:
                        # Otherwise, we apply random offset to replicas 
                        _pos_offset_frame_list = np.random.randint(low=pos_offset_min,
                                                                    high=pos_offset_max,
                                                                    size=self.n_pos_per_anchor)
                        _pos_offset_sec_list = _pos_offset_frame_list / self.fs
                        pos_start_sec_list = self.fns_event_seg_list[idx][1] * self.hop + _pos_offset_sec_list
            else:
                pos_start_sec_list = []

            """ load audio returns: [anchor, pos1, pos2,..pos_n]: xs: ((1+n_pos)),T)"""
            start_sec_list = np.concatenate(([anchor_start_sec], pos_start_sec_list))
            xs = load_audio_multi_start(self.fns_event_seg_list[idx][0],
                                        start_sec_list, 
                                        self.duration, 
                                        self.fs,
                                        self.amp_mode)

            # Create a batch
            if Xa_batch is None:
                Xa_batch = xs[0, :].reshape((1, -1))
                Xp_batch = xs[1:, :]  # If self.n_pos_per_anchor==0: this produces an empty array with shape (0, T)
            else:
                Xa_batch = np.vstack((Xa_batch, xs[0, :].reshape((1, -1))))  # Xa_batch: (n_anchor, T)
                Xp_batch = np.vstack((Xp_batch, xs[1:, :]))  # Xp_batch: (n_pos, T)
                # OGUZ it should be (n_anchor*n_pos_per_anchor, T)

        return Xa_batch, Xp_batch

    def __bg_batch_load(self, idx_list):
        X_bg_batch = None  # (n_batch+n_batch//n_class, fs*k)
        # Random offset for each sample
        random_offset_sec = np.random.randint(0, self.duration * self.fs / 2, size=len(idx_list)) / self.fs
        for i, idx in enumerate(idx_list):
            idx = idx % self.n_bg_samples
            offset_sec = np.min([random_offset_sec[i], self.fns_bg_seg_list[idx][3] / self.fs])

            # Load audio with random offset
            X = load_audio(filename=self.fns_bg_seg_list[idx][0],
                           seg_start_sec=self.fns_bg_seg_list[idx][1] *
                           self.duration,
                           offset_sec=offset_sec,
                           seg_length_sec=self.duration,
                           seg_pad_offset_sec=0.,
                           fs=self.fs,
                           amp_mode='normal')
            X = X.reshape(1, -1)

            # Create a batch
            if X_bg_batch is None:
                X_bg_batch = X
            else:
                X_bg_batch = np.concatenate((X_bg_batch, X), axis=0)

        return X_bg_batch

    def __ir_batch_load(self, idx_list):
        X_ir_batch = None  # (n_batch+n_batch//n_class, fs*k)

        for idx in idx_list:
            idx = idx % self.n_ir_samples

            X = load_audio(filename=self.fns_ir_seg_list[idx][0],
                           seg_start_sec=self.fns_ir_seg_list[idx][1] *
                           self.duration,
                           offset_sec=0.0,
                           seg_length_sec=self.duration,
                           seg_pad_offset_sec=0.0,
                           fs=self.fs,
                           amp_mode='normal')
            if len(X) > MAX_IR_LENGTH:
                X = X[:MAX_IR_LENGTH]

            X = X.reshape(1, -1)

            if X_ir_batch is None:
                X_ir_batch = X
            else:
                X_ir_batch = np.concatenate((X_ir_batch, X), axis=0)

        return X_ir_batch
