import numpy as np
from tensorflow.keras.utils import Sequence

from model.utils.audio_utils import (bg_mix_batch, ir_aug_batch, load_audio,
                                     get_fns_seg_dict, load_audio_multi_start)
from model.fp.melspec.melspectrogram import Melspec_layer_essentia

MAX_IR_LENGTH = 600 # 50ms with fs=8000
#MAX_IR_LENGTH = 8000 # 1s with fs=8000

#TODO: order arguments
# TODO: calculate segments per track in Dataset class
class genUnbalSequence(Sequence):
    def __init__(
        self,
        fns_event_list,
        bsz=120,
        n_anchor=60,
        duration=1,
        hop=.5,
        fs=8000,
        scale=True,
        n_fft=1024,
        stft_hop=256,
        n_mels=256,
        f_min=300,
        f_max=4000,
        segments_per_track=58,
        shuffle=False,
        seg_mode="all",
        normalize_audio=True,
        random_offset_anchor=False,
        offset_margin_hop_rate=0.4,
        bg_mix_parameter=[False],
        ir_mix_parameter=[False],
        reduce_items_p=0,
        reduce_batch_first_half=False,
        drop_the_last_non_full_batch=True,
        ):
        """
        Parameters
        ----------
        fns_event_list : list(str), 
            Song file paths as a list. 
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
        scale : (bool), optional
            Scale the output. The default is True.
        shuffle : (bool), optional
            Randomize samples from the original songs. BG/IRs will not be 
            affected by this parameter (BG/IRs are always shuffled). 
            The default is False.
        seg_mode : (str), optional
            DESCRIPTION. The default is "all".
        normalize_audio : (str), optional
            DESCRIPTION. The default is True.
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
        drop_the_last_non_full_batch : (bool), optional
            Set as False in test. Default is True.
        """

        # Check parameters
        assert bsz >= n_anchor, "bsz should be >= n_anchor"
        assert n_anchor > 0, "n_anchor should be > 0"
        assert seg_mode in {'random_oneshot', 'all'}, "seg_mode should be 'random_oneshot' or 'all'"

        # Save the Input parameters
        self.duration = duration
        self.hop = hop
        self.fs = fs
        self.seg_mode = seg_mode
        self.normalize_audio = normalize_audio
        self.random_offset_anchor = random_offset_anchor
        self.offset_margin_hop_rate = offset_margin_hop_rate
        self.offset_margin_frame = int(hop * self.offset_margin_hop_rate * fs)
        self.bg_hop = duration # For bg samples hop is equal to duration
        self.segments_per_track = segments_per_track

        # Melspec layer
        self.mel_spec = Melspec_layer_essentia(scale=scale,
                                            n_fft=n_fft, 
                                            stft_hop=stft_hop, 
                                            n_mels=n_mels, 
                                            fs=fs, 
                                            dur=duration, 
                                            f_min=f_min, 
                                            f_max=f_max)

        # Training parameters
        self.bsz = bsz
        self.n_anchor = n_anchor
        if bsz != n_anchor:
            self.n_pos_per_anchor = round((bsz - n_anchor) / n_anchor)
            self.n_pos_bsz = bsz - n_anchor
        else:
            self.n_pos_per_anchor = 0
            self.n_pos_bsz = 0
        self.shuffle = shuffle
        self.drop_the_last_non_full_batch = drop_the_last_non_full_batch
        self.reduce_items_p = reduce_items_p
        self.reduce_batch_first_half = reduce_batch_first_half

        # Create segment information for each track
        self.fns_event_seg_dict = get_fns_seg_dict(fns_event_list,
                                                    self.seg_mode,
                                                    self.fs,
                                                    self.duration,
                                                    hop=self.hop)
        # Filter out the tracks with less than segments_per_track
        self.fns_event_seg_dict = {k: v for k, v in self.fns_event_seg_dict.items() if len(v) >= segments_per_track}
        # Keep only segments_per_track segments for each track
        self.fns_event_seg_dict = {k: v[:segments_per_track] for k, v in self.fns_event_seg_dict.items()}

        # TODO: can you salvage more tracks?
        # Determine the tracks to use for each epoch
        if self.drop_the_last_non_full_batch:
            self.n_tracks = int((len(self.fns_event_seg_dict) // n_anchor) * n_anchor)
            self.fns_event_seg_dict = {k: v for i, (k, v) in enumerate(self.fns_event_seg_dict.items()) if i < self.n_tracks}
        else:
            self.n_tracks = len(self.fns_event_seg_dict) # fp-generation
        self.event_fnames = list(self.fns_event_seg_dict.keys())

        # Save augmentation parameters, read the files, and store them in memory
        self.load_bg_samples(bg_mix_parameter)
        self.load_ir_samples(ir_mix_parameter)

        # Shuffle all index events if specified
        if self.shuffle:
            self.shuffle_events()
            self.shuffle_segments()

    def __len__(self):
        """ Returns the number of batches per epoch. """

        if self.reduce_items_p != 0:
            return int(np.ceil(self.n_tracks * self.segments_per_track / self.n_anchor) * (self.reduce_items_p / 100))
        else:
            return int(np.ceil(self.n_tracks * self.segments_per_track / self.n_anchor))

    def __getitem__(self, idx):
        """ Get anchor (original) and positive (replica) samples of audio and compute 
        their power mel-spectrograms for the current iteration.

        Parameters:
        ----------
            idx (int): iteration index

        Returns:
        --------
            Xa_batch: anchor audio samples (n_anchor, T)
            Xp_batch: positive audio samples (self.n_pos_bsz, T)
            Xa_batch_mel: power mel-spectrogram of anchor samples (n_anchor, n_mels, T, 1)
            Xp_batch_mel: power-mel spectrogram of positive samples (n_pos_bsz, n_mels, T, 1)

        """

        # Get self.n_anchor anchor filenames
        i0, i1 = idx*self.n_anchor, (idx+1)*self.n_anchor
        fnames = [self.event_fnames[i%self.n_tracks] for i in range(i0, i1)]
        # Load anchor and positive audio samples
        Xa_batch, Xp_batch = self.batch_load_track_segments(fnames, idx)

        # If positive samples is specified, check for each augmentation
        if self.n_pos_bsz > 0:

            # Indices of both augmentations. We use modulus arithmetic to
            i0, i1 = idx*self.n_pos_bsz, (idx+1)*self.n_pos_bsz

            if self.bg_mix == True:
                # Prepare BG for positive samples
                bg_fnames = [self.bg_fnames[i%self.n_bg_samples] for i in range(i0, i1)]
                bg_batch = self.batch_read_bg(bg_fnames, idx)
                # Mix
                Xp_batch = bg_mix_batch(Xp_batch,
                                        bg_batch,
                                        snr_range=self.bg_snr_range)

            if self.ir_mix == True:
                # Prepare IR for positive samples
                ir_fnames = [self.ir_fnames[i%self.n_ir_samples] for i in range(i0, i1)]
                ir_batch = self.batch_read_ir(ir_fnames)
                # Ir aug
                Xp_batch = ir_aug_batch(Xp_batch, ir_batch)

        # Compute mel spectrograms
        Xa_batch_mel = self.mel_spec.compute_batch(Xa_batch)
        Xp_batch_mel = self.mel_spec.compute_batch(Xp_batch)

        # Fix the dimensions
        Xa_batch_mel = np.expand_dims(Xa_batch_mel, 3).astype(np.float32)
        if Xp_batch_mel.size>0: # if there are positive samples
            Xp_batch_mel = np.expand_dims(Xp_batch_mel, 3).astype(np.float32)

        return Xa_batch, Xp_batch, Xa_batch_mel, Xp_batch_mel

    def batch_load_track_segments(self, fnames, idx):
        """ Load a single segment conditioned on idx from tracks with fnames. 
        Since we shuffle the segments of each track, we can use idx to get a
        different segment from each track. If self.n_pos_per_anchor > 0, we
        also load self.n_pos_per_anchor replicas for each anchor.

        Parameters:
        ----------
            fnames list(int): list of track fnames in the dataset.

        Returns:
        --------
            Xa_batch: (n_anchor, T)
            Xp_batch: (n_anchor*n_pos_per_anchor, T)

        """

        Xa_batch, Xp_batch = [], []
        for fname in fnames:

            # Load the segment information of the anchor track
            anchor_segments = self.fns_event_seg_dict[fname]
            # Get a random segment from the track
            seg_idx, offset_min, offset_max = anchor_segments[idx%self.segments_per_track]

            # Determine the anchor start time
            anchor_start_sec = seg_idx * self.hop

            # If random offset is specified, apply it to the anchor start time
            if self.random_offset_anchor:
                # Sample a random offset frame
                anchor_offset_min = np.max([offset_min, -self.offset_margin_frame])
                anchor_offset_max = np.min([offset_max, self.offset_margin_frame])
                _anchor_offset_frame = np.random.randint(low=anchor_offset_min, high=anchor_offset_max)
                anchor_start_sec += _anchor_offset_frame / self.fs
            else:
                _anchor_offset_frame = 0

            # Calculate multiple(=self.n_pos_per_anchor) pos_start_sec
            if self.n_pos_per_anchor > 0:
                pos_offset_min = np.max([(_anchor_offset_frame - self.offset_margin_frame), offset_min])
                pos_offset_max = np.min([(_anchor_offset_frame + self.offset_margin_frame), offset_max])
                pos_start_sec_list = seg_idx * self.hop
                if pos_offset_min==pos_offset_max==0:
                    # Only the case of running extras/dataset2wav.py as offset_margin_hot_rate=0
                    pos_start_sec_list = [pos_start_sec_list]
                else:
                    # Otherwise, we apply random offset to replicas 
                    _pos_offset_frame_list = np.random.randint(low=pos_offset_min,
                                                                high=pos_offset_max,
                                                                size=self.n_pos_per_anchor)
                    _pos_offset_sec_list = _pos_offset_frame_list / self.fs
                    pos_start_sec_list += _pos_offset_sec_list
            else:
                pos_start_sec_list = []

            # Load the anchor and positive samples
            start_sec_list = np.concatenate(([anchor_start_sec], pos_start_sec_list))
            xs = load_audio_multi_start(fname,
                                        start_sec_list, 
                                        self.duration, 
                                        self.fs,
                                        normalize=self.normalize_audio)
            Xa_batch.append(xs[0, :].reshape((1, -1)))
            Xp_batch.append(xs[1:, :]) # If self.n_pos_per_anchor==0: this produces an empty array with shape (0, T)

        # Create a batch
        Xa_batch = np.concatenate(Xa_batch, axis=0)
        Xp_batch = np.concatenate(Xp_batch, axis=0)

        return Xa_batch, Xp_batch

    def batch_read_bg(self, fnames, index):
        """ Read len(fnames) background samples from the memory. Each sample is
        randomly offsetted between its offset_min and self.bg_hop/2. We randomly
        coose a different segment at every iteration.

        Parameters:
        -----------
            fnames list(str): list of background fnames in the dataset.

        Returns:
        --------
            X_bg_batch (self.n_pos_bsz, T)

        """

        X_bg_batch = []
        for fname in fnames:

            # Read the complete background noise sample from memory
            X = self.bg_clips[fname]

            # Choose a different random segment every iteration
            bg_segments = self.fns_bg_seg_dict[fname]
            seg_idx, offset_min, offset_max = bg_segments[index%len(bg_segments)]

            # Randomly offset the segment
            random_offset_sec = np.random.randint(offset_min, int((self.bg_hop/2)*self.fs)) / self.fs
            offset_sec = np.min([random_offset_sec, offset_max / self.fs])
            # Calculate the start frame index
            start_frame_idx = np.floor((seg_idx*self.bg_hop + offset_sec)*self.fs).astype(int)
            seg_length_frame = np.floor(self.duration*self.fs).astype(int)
            assert start_frame_idx+seg_length_frame <= X.shape[0], \
                    f"start_frame_idx+seg_length_frame={start_frame_idx+seg_length_frame}" \
                        f"is larger than X.shape[0]={X.shape[0]}"

            # Load the offsetted segment
            X_bg_batch.append(X[start_frame_idx:start_frame_idx+seg_length_frame].reshape(1, -1))

        # Concatenate the samples
        X_bg_batch = np.concatenate(X_bg_batch, axis=0)

        return X_bg_batch

    def batch_read_ir(self, fnames):
        """ Read a segment of Impulse Response samples for given fnames from the memory.

        Parameters:
        ----------
            fnames list(str): list of IR fnames in the dataset.

        Returns:
        --------
            X_ir_batch (self.n_pos_bsz, T)

        """

        X_ir_batch = []
        for fname in fnames:
            # Read the IR sample from memory
            X_ir_batch.append(self.ir_clips[fname].reshape(1, -1))
        # Concatenate the samples to a batch
        X_ir_batch = np.concatenate(X_ir_batch, axis=0)

        return X_ir_batch

    # TODO: is shuffling different segments with same frequency good?
    # TODO: the current definition of epoch may lead to catastrophic forgetting?
    # TODO: can we shuffle the bg segments in a way that more of it is seen during training?
    def on_epoch_end(self):
        """ Routines to apply at the end of each epoch."""

        # Shuffle all  events if specified
        if self.shuffle:
            self.shuffle_events()
            self.shuffle_segments()

    def shuffle_events(self):
        """ Shuffle all events."""

        # Shuffle the order tracks
        np.random.shuffle(self.event_fnames)

        # Shuffle the order of augmentation types
        if self.bg_mix == True:
            np.random.shuffle(self.bg_fnames)
        if self.ir_mix == True:
            np.random.shuffle(self.ir_fnames)

    def shuffle_segments(self):
        """ Shuffle the order of segments of each track and background noise.
        We do not shuffle the order of IRs because we only use the first segment of
        each IR."""

        for fname in self.event_fnames:
            np.random.shuffle(self.fns_event_seg_dict[fname])

        if self.bg_mix == True:
            for fname in self.bg_fnames:
                np.random.shuffle(self.fns_bg_seg_dict[fname])

    def load_bg_samples(self, bg_mix_parameter):
        """ Load background noise samples in memory and their segmentation
        information.

        Parameters:
        ----------
            bg_mix_parameter list([(bool), list(str), (int, int)]):
                [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR)].

        """

        self.bg_mix = bg_mix_parameter[0]
        if self.bg_mix:
            self.bg_snr_range = bg_mix_parameter[2]
            self.fns_bg_seg_dict = get_fns_seg_dict(bg_mix_parameter[1], 
                                                    segment_mode='all',
                                                    fs=self.fs, 
                                                    duration=self.duration,
                                                    hop=self.bg_hop)
            self.bg_fnames = list(self.fns_bg_seg_dict.keys())
            # Load all bg clips in full duration
            self.bg_clips = {fn: load_audio(fn, fs=self.fs, normalize=self.normalize_audio) for fn in self.bg_fnames}
            self.n_bg_samples = len(self.bg_clips)

    def load_ir_samples(self, ir_mix_parameter):
        """ Load Impulse Response samples in memory and their segmentation
        information. We only use the first segment of each IR clip. These segments
        are truncated to MAX_IR_LENGTH.

        Parameters:
        ----------
            ir_mix_parameter list([(bool), list(str)]):
                [True, IR_FILEPATHS].

        """

        self.ir_mix = ir_mix_parameter[0]
        if self.ir_mix:
            self.fns_ir_seg_dict = get_fns_seg_dict(ir_mix_parameter[1], 
                                                    'first',
                                                    self.fs, 
                                                    self.duration)
            self.ir_fnames = list(self.fns_ir_seg_dict.keys())
            # Load all ir clips in full duration
            self.ir_clips = {}
            for fn in self.ir_fnames:
                X = load_audio(fn, 
                            seg_length_sec=self.duration,
                            fs=self.fs,
                            normalize=self.normalize_audio)
                # Truncate IR to MAX_IR_LENGTH
                if len(X) > MAX_IR_LENGTH:
                    X = X[:MAX_IR_LENGTH]
                self.ir_clips[fn] = X
            self.n_ir_samples = len(self.ir_clips)