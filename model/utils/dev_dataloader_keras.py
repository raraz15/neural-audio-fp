import numpy as np
from tensorflow.keras.utils import Sequence

from model.fp.melspec.melspectrogram import Melspec_layer_essentia
from model.utils import audio_utils

class DevLoader(Sequence):

    # TODO: remove shuffle and stuff that we do not use
    # TODO: fix both augmentations
    def __init__(
        self, 
        segment_duration=1, 
        fs=8000, 
        n_fft=1024, 
        stft_hop=256, 
        n_mels=256, 
        f_min=300, 
        f_max=4000, 
        scale_output=True, 
        segments_per_track=59, 
        bsz=120, 
        shuffle=False, 
        random_offset_anchor=False, 
        offset_duration=0.2, 
        ):

        # Check parameters
        assert bsz % 2 == 0, "bsz should be even"
        assert segments_per_track > 0, "segments_per_track should be > 0"
        assert segment_duration > 0, "segment_duration should be > 0"
        assert offset_duration >= 0, "offset_duration should be >= 0"

        # Save the Input parameters
        self.segment_duration = segment_duration
        self.offset_duration = offset_duration
        self.max_offset_sample = int(self.offset_duration * fs) # TODO: is this correct?
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
        self.shuffle = shuffle
        self.bsz = bsz
        self.n_pos_per_anchor = 1
        # Half of the batch is positive samples
        self.n_pos_bsz = bsz // (1+self.n_pos_per_anchor)
        # The rest are anchors
        self.n_anchor = bsz - self.n_pos_bsz

        self.track_seg_dict = {}
        self.n_samples = 0
        self.track_fnames = []
        self.n_tracks = 0

    def __len__(self):
        """ Returns the number of batches per epoch. An epoch is defined as
        when all the segments of each track are seen once."""

        return int(self.n_samples//self.n_anchor)

    def __getitem__(self, idx):
        """ Get a batch of anchor (original) and positive (replica) samples of audio 
        with their power mel-spectrograms. During training we follow a strategy that 
        allows us to use all the pre-defined number of segments of each track.

        Parameters:
        ----------
            idx (int):
                Batch index

        Returns:
        --------
            Xa_batch (ndarray):
                anchor audio samples (self.n_anchor, T)
            Xp_batch (ndarray):
                positive audio samples (self.n_pos_bsz, T)
            Xa_batch_mel (ndarray):
                power mel-spectrogram of anchor samples (self.n_anchor, n_mels, T, 1)
            Xp_batch_mel (ndarray):
                power-mel spectrogram of positive samples (self.n_pos_bsz, n_mels, T, 1)

        """

        # Indices of tracks to use in this batch
        i0, i1 = idx*self.n_anchor, (idx+1)*self.n_anchor
        # Get their filenames, each file will be used self.segments_per_track 
        # times during an epoch
        fnames = [self.track_fnames[i%self.n_tracks] for i in range(i0, i1)]

        # Load anchor and positive audio samples for each filename
        Xa_batch, Xp_batch = self.batch_load_track_segments(fnames, idx)

        # If positive samples is specified, check for each augmentation
        if self.n_pos_bsz > 0:

            # Indices of both augmentations. We use modulus arithmetic to
            # try to cover the whole set of augmentations.
            i0, i1 = idx*self.n_pos_bsz, (idx+1)*self.n_pos_bsz

            # TODO: merge if statements of augmentations
            if self.bg_mix:
                # Prepare BG for positive samples
                bg_fnames = [self.bg_fnames[i%self.n_bg_files] for i in range(i0, i1)]
                bg_batch = self.batch_read_bg(bg_fnames, idx)
                # Mix
                Xp_batch = audio_utils.bg_mix_batch(Xp_batch,
                                                    bg_batch,
                                                    snr_range=self.bg_snr_range)

            if self.ir_mix:
                # Prepare IR for positive samples
                ir_fnames = [self.ir_fnames[i%self.n_ir_files] for i in range(i0, i1)]
                ir_batch = self.batch_read_ir(ir_fnames)
                # Apply Room IR, normalize the output
                Xp_batch = audio_utils.ir_aug_batch(Xp_batch, ir_batch, normalize=True)

        # Compute mel spectrograms
        Xa_batch_mel = self.mel_spec.compute_batch(Xa_batch).astype(np.float32)
        Xp_batch_mel = self.mel_spec.compute_batch(Xp_batch).astype(np.float32)

        # Fix the dimensions
        Xa_batch_mel = np.expand_dims(Xa_batch_mel, 3)
        Xp_batch_mel = np.expand_dims(Xp_batch_mel, 3)

        return Xa_batch, Xp_batch, Xa_batch_mel, Xp_batch_mel

    def batch_load_track_segments(self, fnames, idx):
        raise NotImplementedError

    def batch_read_bg(self, fnames, index):
        """ Read len(fnames) background samples from the memory. Each sample is
        randomly offsetted between its offset_min and self.bg_hop/2. We randomly
        coose a different segment at every iteration conditioned on index.

        Parameters:
        -----------
            fnames list(str):
                list of background fnames in the dataset.

        Returns:
        --------
            X_bg_batch (ndarray):
                (self.n_pos_bsz, T)

        """

        X_bg_batch = []
        for fname in fnames:

            # Read the complete background noise sample from memory
            X = self.bg_clips[fname]

            # Choose a different segment every iteration
            bg_segments = self.fns_bg_seg_dict[fname]
            seg_idx, offset_min, offset_max = bg_segments[index%len(bg_segments)]

            # Randomly offset the segment
            random_offset_sec = np.random.randint(offset_min, int((self.bg_hop/2)*self.fs)) / self.fs
            offset_sec = np.min([random_offset_sec, offset_max / self.fs])
            # Calculate the start frame index
            start_frame_idx = np.floor((seg_idx*self.bg_hop + offset_sec)*self.fs).astype(int)
            seg_length_frame = np.floor(self.segment_duration*self.fs).astype(int)
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
            fnames list(str):
                list of IR fnames in the dataset.

        Returns:
        --------
            X_ir_batch (ndarray):
                (self.n_pos_bsz, T)

        """

        X_ir_batch = []
        for fname in fnames:
            # Read the IR sample from memory
            X_ir_batch.append(self.ir_clips[fname].reshape(1, -1))
        # Concatenate the samples to a batch
        X_ir_batch = np.concatenate(X_ir_batch, axis=0)

        return X_ir_batch

    def on_epoch_end(self):
        """ Routines to apply at the end of each epoch."""

        # Shuffle all  events if specified
        if self.shuffle:
            self.shuffle_events()
            self.shuffle_segments()

    def shuffle_events(self):
        """ Shuffle all events."""

        # Shuffle the order of tracks
        np.random.shuffle(self.track_fnames)

        # Shuffle the order of augmentation types
        if self.bg_mix:
            np.random.shuffle(self.bg_fnames)
        if self.ir_mix:
            np.random.shuffle(self.ir_fnames)

    def shuffle_segments(self):
        """ Shuffle the order of segments of each track and background noise.
        We do not shuffle the order of IRs because we only use the first segment of
        each IR."""

        # Shuffle the order of segments of each track
        for fname in self.track_fnames:
            np.random.shuffle(self.track_seg_dict[fname])

        # Shuffle the order of bg segments
        if self.bg_mix == True:
            for fname in self.bg_fnames:
                np.random.shuffle(self.fns_bg_seg_dict[fname])

    def load_and_store_bg_samples(self, bg_mix_parameter):
        """ Load background noise samples in memory and their segmentation
        information.

        Parameters:
        ----------
            bg_mix_parameter list([(bool), list(str), (int, int)]):
                [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR)].

        """

        self.bg_mix = bg_mix_parameter[0]
        if self.bg_mix:
            print("Loading Background Noise samples in memory...")
            self.bg_snr_range = bg_mix_parameter[2]
            self.fns_bg_seg_dict = audio_utils.get_fns_seg_dict(bg_mix_parameter[1], 
                                                                segment_mode='all',
                                                                fs=self.fs, 
                                                                duration=self.segment_duration,
                                                                hop=self.bg_hop)
            self.bg_fnames = list(self.fns_bg_seg_dict.keys())
            # Load all bg clips in full duration
            # TODO: do not normalize bg clips?
            self.bg_clips = {fn: audio_utils.load_audio(fn, fs=self.fs, 
                                                        normalize=True) # normalize the bg clips
                             for fn in self.bg_fnames}
            self.n_bg_files = len(self.bg_clips)

            # Check if we have enough bg samples. Ideally, every segment of every bg track 
            # should be used at least once. If not, we warn the user.
            total_seen = len(self.track_fnames)*self.segments_per_track / len(self.bg_fnames)
            x = 0
            for fname in self.bg_fnames:
                n_segments = len(self.fns_bg_seg_dict[fname])
                if total_seen < n_segments:
                    x += n_segments - total_seen
            if x > 0:
                print(f"WARNING: {100*x/sum([len(l) for l in self.fns_bg_seg_dict]):.2f} percent of"
                    " Background Noise segments won't be used."
                    "\nIncrease tracks or decrease segments_per_track to avoid this warning.")

    def load_and_store_ir_samples(self, ir_mix_parameter):
        """ Load Impulse Response samples in memory and their segmentation
        information. We only use the first segment of each IR clip. These segments
        are truncated to MAX_IR_DURATION.

        Parameters:
        ----------
            ir_mix_parameter list([(bool), list(str), float]):
                [True, IR_FILEPATHS, MAX_IR_DURATION].

        """

        self.ir_mix = ir_mix_parameter[0]
        if self.ir_mix:
            print("Loading Impulse Response samples in memory...")
            self.max_ir_length = int(ir_mix_parameter[2] * self.fs)
            self.fns_ir_seg_dict = audio_utils.get_fns_seg_dict(ir_mix_parameter[1], 
                                                    segment_mode='first', # TODO: all?
                                                    fs=self.fs, 
                                                    duration=self.segment_duration)
            self.ir_fnames = list(self.fns_ir_seg_dict.keys())
            # Load all ir clips in full duration
            self.ir_clips = {}
            for fn in self.ir_fnames:
                # TODO: do not normalize IR clips?
                X = audio_utils.load_audio(fn, 
                                            seg_length_sec=self.segment_duration,
                                            fs=self.fs,
                                            normalize=True)
                # Truncate IR to max_ir_length
                if len(X) > self.max_ir_length:
                    X = X[:self.max_ir_length]
                self.ir_clips[fn] = X
            self.n_ir_files = len(self.ir_clips)

            # Check if we have enough ir samples. Ideally, every segment of every ir track
            # should be used at least once. If not, we warn the user.
            total_seen = len(self.track_fnames)*self.segments_per_track / len(self.ir_fnames)
            x = 0
            for fname in self.ir_fnames:
                n_segments = len(self.fns_ir_seg_dict[fname])
                if total_seen < n_segments:
                    x += n_segments - total_seen
            if x > 0:
                print(f"WARNING: {100*x/sum([len(l) for l in self.fns_ir_seg_dict]):.2f} percent of"
                    " Impulse Response segments won't be used."
                    "\nIncrease tracks or decrease segments_per_track to avoid this warning.")

class SegmentDevLoader(DevLoader):

    def __init__(
        self, 
        segment_dict:dict,
        segment_duration=1, 
        full_segment_duration=2,
        fs=8000, 
        n_fft=1024, 
        stft_hop=256, 
        n_mels=256, 
        f_min=300, 
        f_max=4000, 
        scale_output=True, 
        segments_per_track=59, 
        bsz=120, 
        shuffle=False, 
        random_offset_anchor=False, 
        offset_duration=0.2, 
        bg_mix_parameter=[False], 
        ir_mix_parameter=[False],
        ):

        super().__init__(
            segment_duration, 
            fs, 
            n_fft, 
            stft_hop, 
            n_mels, 
            f_min, 
            f_max, 
            scale_output, 
            segments_per_track, 
            bsz, 
            shuffle, 
            random_offset_anchor, 
            offset_duration, 
            )

        # Determine the length of the audio segments in the dataset 
        # and how much of it will be used for training
        assert segment_duration<=full_segment_duration, \
                "segment_duration should be <= segment_duration"
        # self.segment_duration = segment_duration # already defined in base class
        self.full_segment_duration = full_segment_duration
        # Convert duration to samples
        self.segment_length = int(self.segment_duration * fs)
        self.full_segment_length = int(self.full_segment_duration * fs)

        # Align the centers of the full segment and the sub-segment
        self.relative_segment_position = int((self.full_segment_length - self.segment_length) / 2)

        # Based on the segment duration, determine the maximum allowed offset
        # for the anchor and positive samples. We align the centers of the 
        # segment and the full_segment
        self.offset_duration = offset_duration
        self.max_possible_offset_duration = (full_segment_duration - segment_duration) / 2
        assert self.offset_duration <= self.max_possible_offset_duration, \
                "offset_duration should be <= (segment_duration - sub_segment_duration)/2"
        #self.max_offset_sample = int(self.offset_duration * fs) # Done in base class

        # TODO: give segment_paths here?
        # Filter out the tracks with less than segments_per_track
        self.track_seg_dict = {k: v 
                                for k, v in segment_dict.items() 
                                if len(v) >= segments_per_track}
        print(f"Number of tracks with at least {segments_per_track}"
              f" segments: {len(self.track_seg_dict):,}")
        # Keep only segments_per_track segments for each track
        self.track_seg_dict = {k: v[:segments_per_track] 
                                for k, v in self.track_seg_dict.items()}

        # Determine the tracks to use at each epoch
        # Remove the tracks that do not fill the last batch. Each batch contains
        # a single segment from n_anchor tracks.
        self.n_tracks = int((len(self.track_seg_dict) // self.n_anchor) * self.n_anchor)
        self.track_seg_dict = {k: v 
                                for i, (k, v) in enumerate(self.track_seg_dict.items()) 
                                if i < self.n_tracks}
        self.n_samples = self.n_tracks * self.segments_per_track
        self.track_fnames = list(self.track_seg_dict.keys())

        # Save augmentation parameters, read the files, and store them in memory
        self.load_and_store_bg_samples(bg_mix_parameter)
        self.load_and_store_ir_samples(ir_mix_parameter)

        # Shuffle all events if specified
        if self.shuffle:
            self.shuffle_events()
            self.shuffle_segments()

    def batch_load_track_segments(self, fnames, idx):
        """ Load a single segment conditioned on idx from the tracks with fnames. 
        Since we shuffle the segments of each track at epoch end, we can use idx 
        to get a different segment. If self.n_pos_per_anchor > 0, we also load 
        self.n_pos_per_anchor replicas for each anchor.

        Parameters:
        ----------
            fnames list(int):
                list of track fnames in the dataset.
            idx (int):
                Batch index, used to get different segments from each 
                track throughout the epoch.

        Returns:
        --------
            Xa_batch (ndarray):
                (n_anchor, T)
            Xp_batch (ndarray):
                (n_anchor*n_pos_per_anchor, T) # TODO: numpy coolness?

        """

        # If segments_per_track is even, each epoch half of the segments will be
        # seen twice and the other half zero. We want to see all segments once, 
        # so we make sure that the second half of the segments are seen too.
        if self.segments_per_track%2==0 and idx>=self.__len__()/2:
            idx = (idx+1) % self.segments_per_track
        else:
            # If segments_per_track is odd, all segments will be seen once each epoch
            idx = idx % self.segments_per_track

        Xa_batch, Xp_batch = [], []
        for fname in fnames:

            # Load the full segment
            full_segment = audio_utils.load_audio(self.track_seg_dict[fname][idx], 
                                                  seg_start_sec=0,
                                                  offset_sec=0,
                                                  seg_length_sec=self.full_segment_duration,
                                                  fs=self.fs, 
                                                  normalize=False)
            assert full_segment.shape[0] == self.full_segment_length, \
                    f"full_segment.shape[0]={full_segment.shape[0]} but " \
                    f"self.full_segment_length={self.full_segment_length}"

            # The anchor start position inside the full segment when the centers are aligned
            anchor_start = self.relative_segment_position
            # This is the maximum allowed offset
            max_possible_offset = self.relative_segment_position

            # If random offset is specified, determine a random offset based 
            # on the offset margin. This means that at each iteration we will 
            # have a different shifted version of the anchor
            if self.random_offset_anchor:
                # Randomly offset the anchor sample between -offset_margin and offset_margin
                anchor_offset = np.random.randint(low=-self.max_offset_sample, 
                                                high=self.max_offset_sample)
            else:
                anchor_offset = 0
            # Apply the offset to the anchor start sample
            anchor_start += anchor_offset
            assert anchor_start>=0, "Start point is out of bounds"
            anchor_end = anchor_start + self.segment_length
            assert anchor_end<=self.full_segment_length, "End point is out of bounds"

            # Get the anchor sample, create a copy
            anchor = full_segment[anchor_start:anchor_end].copy()
            # Append it to the batch
            Xa_batch.append(anchor.reshape((1, -1)))

            # Randomly offset each positive sample with respect to the anchor
            if self.n_pos_per_anchor > 0:
                # Determine the distance to the segment edges from the anchor's offset
                dist_l = - max_possible_offset - anchor_offset
                dist_r = max_possible_offset - anchor_offset
                # Make sure that the offset range is within the full-segment
                pos_offset_min = np.max([-self.max_offset_sample, dist_l])
                pos_offset_max = np.min([self.max_offset_sample, dist_r])
                assert anchor_start+pos_offset_min>=0, \
                    "Start point range is out of bounds for the positive samples"
                assert anchor_start+pos_offset_max+self.segment_length<=self.full_segment_length, \
                    "End point range can be out of bounds for the positive samples"

                # Apply random offset to replicas
                pos_start_list = self.relative_segment_position + np.random.randint(low=pos_offset_min,
                                                                                    high=pos_offset_max,
                                                                                    size=self.n_pos_per_anchor)

                # Load the positive samples and append them to the batch
                for s_idx in pos_start_list:
                    e_idx = s_idx + self.segment_length
                    positive = full_segment[s_idx:e_idx].copy()
                    Xp_batch.append(positive.reshape((1, -1)))

        # Create the batch
        Xa_batch = np.concatenate(Xa_batch, axis=0)
        # If there are positive samples, concatenate them to a batch
        if self.n_pos_per_anchor>0:
            Xp_batch = np.concatenate(Xp_batch, axis=0)

        return Xa_batch, Xp_batch

class TrackDevLoader(DevLoader):

    def __init__(
        self, 
        track_paths:list,
        segment_duration=1, 
        hop_duration=.5,
        fs=8000, 
        n_fft=1024, 
        stft_hop=256, 
        n_mels=256, 
        f_min=300, 
        f_max=4000, 
        scale_output=True, 
        segments_per_track=58, 
        bsz=120, 
        shuffle=False, 
        random_offset_anchor=False, 
        offset_duration=0.2,
        bg_mix_parameter=[False],
        ir_mix_parameter=[False],
        ):
        super().__init__(segment_duration, 
                        fs, 
                        n_fft, 
                        stft_hop, 
                        n_mels, 
                        f_min, 
                        f_max, 
                        scale_output, 
                        segments_per_track, 
                        bsz, 
                        shuffle, 
                        random_offset_anchor, 
                        offset_duration)

        assert hop_duration > 0, "hop_duration should be > 0"
        assert self.segment_duration >= hop_duration, "segment_duration should be >= hop_duration"

        # Save the Input parameters
        self.hop_duration = hop_duration
        # Convert offset duration to samples
        self.max_offset_sample = int(self.offset_duration * fs)

        # Create segment information for each track. Since the tracks are not segmented
        # we segment them here using the hop_duration
        self.track_seg_dict = audio_utils.get_fns_seg_dict(track_paths,
                                                        segment_mode='all',
                                                        fs=self.fs,
                                                        duration=self.segment_duration,
                                                        hop=self.hop_duration)

        # Filter out the tracks with less than segments_per_track and
        # Keep only segments_per_track segments for each track
        self.track_seg_dict = {k: v[:segments_per_track] 
                                for k, v in self.track_seg_dict.items() 
                                if len(v) >= segments_per_track}

        # Determine the tracks to use at each epoch
        # Remove the tracks that do not fill the last batch. Each batch contains
        # a single segment from n_anchor tracks.
        self.n_tracks = int((len(self.track_seg_dict) // self.n_anchor) * self.n_anchor)
        self.track_seg_dict = {k: v 
                                for i, (k, v) in enumerate(self.track_seg_dict.items()) 
                                if i < self.n_tracks}
        self.track_fnames = list(self.track_seg_dict.keys())
        self.n_samples = self.n_tracks * self.segments_per_track # TODO

        # Save augmentation parameters, read the files, and store them in memory
        self.load_and_store_bg_samples(bg_mix_parameter)
        self.load_and_store_ir_samples(ir_mix_parameter)

        # Shuffle all events if specified
        if self.shuffle:
            self.shuffle_events()
            self.shuffle_segments()

    def batch_load_track_segments(self, fnames, idx):
        """ Load a single segment conditioned on idx from tracks with fnames. 
        Since we shuffle the segments of each track, we can use idx to get a
        different segment from each track. If self.n_pos_per_anchor > 0, we
        also load self.n_pos_per_anchor replicas for each anchor.

        Parameters:
        ----------
            fnames list(int):
                list of track fnames in the dataset.

        Returns:
        --------
            Xa_batch (ndarray):
                (n_anchor, T)
            Xp_batch (ndarray):
                (n_anchor*n_pos_per_anchor, T)

        """

        # If segments_per_track is even, each epoch half of the segments will be
        # seen twice and the other half zero. We want to see all segments once, 
        # so we make sure that the second half of the segments are seen too.
        if self.segments_per_track%2==0 and idx>=self.__len__()/2:
            idx = (idx+1) % self.segments_per_track
        else:
            # If segments_per_track is odd, all segments will be seen once each epoch
            idx = idx % self.segments_per_track

        Xa_batch, Xp_batch = [], []
        for fname in fnames:

            # Get the segment information of the random_idx segment of the track
            seg_idx, offset_min, offset_max = self.track_seg_dict[fname][idx]

            # Determine the anchor start time
            anchor_start_sec = seg_idx * self.hop_duration

            # If random offset is specified, determine a random offset based on the offset margin
            # This means that at each iteration we will have a different shifted version of the anchor
            if self.random_offset_anchor:
                # Sample a random offset sample
                anchor_offset_min = np.max([offset_min, - self.max_offset_sample])
                anchor_offset_max = np.min([offset_max, self.max_offset_sample])
                _anchor_offset_sample = np.random.randint(low=anchor_offset_min, 
                                                          high=anchor_offset_max)
            else:
                _anchor_offset_sample = 0

            # Apply the offset to the anchor start time
            anchor_start_sec += _anchor_offset_sample / self.fs

            # Based on the anchor offset time, sample an offset for each self.n_pos_per_anchor
            if self.n_pos_per_anchor > 0:
                pos_offset_min = np.max([offset_min, _anchor_offset_sample - self.max_offset_sample])
                pos_offset_max = np.min([offset_max, _anchor_offset_sample + self.max_offset_sample])
                if pos_offset_min==pos_offset_max==0:
                    # Only the case of running extras/dataset2wav.py as offset_margin_hop_rate=0
                    # TODO: is this still functional?
                    pos_start_sec_list = [seg_idx * self.hop_duration]
                else:
                    # Otherwise, we apply random offset to replicas 
                    _pos_offset_frame_list = np.random.randint(low=pos_offset_min,
                                                                high=pos_offset_max,
                                                                size=self.n_pos_per_anchor)
                    _pos_offset_sec_list = _pos_offset_frame_list / self.fs
                    pos_start_sec_list = [seg_idx * self.hop_duration] + _pos_offset_sec_list
            else:
                pos_start_sec_list = []

            # Load the anchor and positive samples
            start_sec_list = np.concatenate(([anchor_start_sec], pos_start_sec_list))
            xs = audio_utils.load_audio_multi_start(fname,
                                                    start_sec_list, 
                                                    self.segment_duration, 
                                                    fs=self.fs,
                                                    normalize=False) # In real life a segment is not normalized
            Xa_batch.append(xs[0, :].reshape((1, -1)))
            Xp_batch.append(xs[1:, :]) # If self.n_pos_per_anchor==0: this produces an empty array with shape (0, T)

        # Create a batch
        Xa_batch = np.concatenate(Xa_batch, axis=0)
        Xp_batch = np.concatenate(Xp_batch, axis=0)

        return Xa_batch, Xp_batch
