import numpy as np
from tensorflow.keras.utils import Sequence

from model.utils.audio_utils import background_mix, ir_aug, load_audio, get_fns_seg_dict, max_normalize, OLA

from model.fp.melspec.melspectrogram import Melspec_layer_essentia

SEED = 27 # Only used for random augmentation
np.random.seed(SEED) # TODO: other seeds?

class genUnbalSequenceGeneration(Sequence):
    def __init__(
        self,
        track_paths,
        duration=1,
        hop=.5,
        normalize_audio=True,
        fs=8000,
        scale=True,
        n_fft=1024,
        stft_hop=256,
        n_mels=256,
        f_min=300,
        f_max=4000,
        segments_per_track=120,
        bg_mix_parameter=[False],
        ir_mix_parameter=[False],
        ):
        """
        Parameters
        ----------
        track_paths : list(str), 
            Track .npz paths as a list. 
        duration : (float), optional
            Duration in seconds. The default is 1.
        hop : (float), optional
            Hop-size in seconds. The default is .5.
        normalize_audio : (str), optional
            DESCRIPTION. The default is True.
        fs : (int), optional
            Sampling rate. The default is 8000.
        scale : (bool), optional
            Scale the power mel-spectrogram. The default is True.
        segments_per_track : (int), optional
            Number of segments per track. The default is 120.
        """

        # Check parameters
        assert segments_per_track > 0, "segments_per_track should be > 0"

        # Save the Input parameters
        self.duration = duration
        self.hop = hop
        self.bg_hop = duration
        self.fs = fs
        self.normalize_audio = normalize_audio
        self.segments_per_track = segments_per_track
        self.segment_length = int(self.fs*self.duration)
        self.hop_length = int(self.fs*self.hop)
        self.chunk_length = int(((segments_per_track-1)*self.hop + self.duration)*fs)

        self.track_paths = track_paths
        self.n_tracks = len(self.track_paths)
        # This is sketchy. We assume that all tracks have the same number of
        # segments. This is true for the current dataset.
        self.n_samples = self.n_tracks * self.segments_per_track

        # Melspec layer
        self.mel_spec = Melspec_layer_essentia(scale=scale,
                                            n_fft=n_fft, 
                                            stft_hop=stft_hop, 
                                            n_mels=n_mels, 
                                            fs=fs, 
                                            dur=duration, 
                                            f_min=f_min, 
                                            f_max=f_max)

        # Save augmentation parameters, read the files, and store them in memory
        self.load_and_store_bg_samples(bg_mix_parameter)
        self.load_and_store_ir_samples(ir_mix_parameter)

    def __len__(self):
        """ Returns the number of batches. We fit segments_per_track segments
        to every batch. """

        return self.n_tracks

    def __getitem__(self, idx):
        """ Loads all the segments of track with idx and returns a batch of
        segments_per_track segments. If bg_mix is True, we mix the segments
        with background noise. If ir_mix is True, we convolve the segments
        with an Impulse Response.

        Parameters:
        ----------
            idx (int):
                Batch index

        Returns:
        --------
            X_batch (ndarray):
                audio samples (n_anchor, T)
            X_batch_mel (ndarray):
                power mel-spectrogram of samples (n_anchor, n_mels, T, 1)

        """

        # Load the segments from .npy file
        track_path = self.track_paths[idx]
        X = np.load(track_path)
        assert X.shape[1] == self.segment_length, \
                        "Loaded segment has wrong duration."

        # Keep only the first segments_per_track segments
        X = X[:self.segments_per_track]

        # Normalize the segments if required
        if self.normalize_audio:
            X = max_normalize(X)

        # Apply augmentations if specified
        if self.bg_mix:
            # Apply OLA to the segments
            X = OLA(X, self.hop_length)
            # Get a random background noise sample
            bg_fname = self.bg_fnames[idx%self.n_bg_files]
            bg_noise = self.read_bg(bg_fname)
            # Randoomly sample an SNR
            snr = np.random.rand()
            snr = snr * (self.bg_snr_range[1] - self.bg_snr_range[0]) + self.bg_snr_range[0]
            # Mix the OLA'd track with the background noise
            X = background_mix(X.reshape(-1),
                                bg_noise.reshape(-1),
                                snr_db=snr)
            # Reshape the track to (n_segments, segment_length)
            X = X.reshape(self.segments_per_track, -1)
        if self.ir_mix:
            # Apply OLA to the segments
            X = OLA(X, self.hop_length)
            # Get a random IR sample
            ir_fname = self.ir_fnames[idx%self.n_ir_files]
            ir = self.read_ir(ir_fname)
            # Convolve with IR
            X = ir_aug(X, ir)
            # Reshape the track to (n_segments, segment_length)
            X = X.reshape(self.segments_per_track, -1)

        # Compute mel spectrograms
        X_mel = self.mel_spec.compute_batch(X)
        # Fix the dimensions and types
        X_mel = np.expand_dims(X_mel, 3).astype(np.float32)

        return X, X_mel

    def read_bg(self, fname):
        """ Read background noise samples for given fnames from the memory and 
        return a batch. To simulate real life conditions during testing, we apply the
        same background noise to a segmented track. We cut the file into overlapping 
        segments of duration self.duration. self.frames_per_file segments are read from
        this file.

        Parameters:
        -----------
            fname (str):
                Background file name.

        Returns:
        --------
            bg_segments (1D ndarray):
                (self.segments_per_track, self.segment_length)

        """

        # Read the complete background noise sample from memory
        bg = self.bg_clips[fname]

        # Repeat the bg sample if it is shorter than the chunk length
        if len(bg) < self.chunk_length:
            # If the background noise sample is shorter than the chunk length,
            # we repeat it until we have a sample of length chunk_length
            n_repeats = int(np.ceil(self.chunk_length / len(bg)))
            bg = np.tile(bg, n_repeats)
        else:
            # Otherwise we cut a random part
            start = np.random.randint(0, len(bg) - self.chunk_length)
            bg = bg[start:start+self.chunk_length]

        return bg

        # # Create Overlapping segments of duration self.duration and hop self.hop
        # bg_segments = np.zeros((self.segments_per_track, self.segment_length))
        # for i in range(self.segments_per_track):
        #     start = i*self.hop_length
        #     end = start + self.segment_length
        #     bg_segments[i] = bg[start:end]

        # return bg_segments

    def read_ir(self, fname):
        """ Read the Impulse Response with fname from the memory.
        For realistic testing, we apply the same IR to a segmented track.

        Parameters:
        ----------
            fnames (str):
                IR fname.

        Returns:
        --------
            ir_segment (1D ndarray):
                (self.max_ir_length)

        """

        ir_segment = self.ir_clips[fname]

        return ir_segment

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
            self.fns_bg_seg_dict = get_fns_seg_dict(bg_mix_parameter[1], 
                                                    segment_mode='all',
                                                    fs=self.fs, 
                                                    duration=self.duration,
                                                    hop=self.duration)
            self.bg_fnames = list(self.fns_bg_seg_dict.keys())
            # Shuffle the bg_fnames
            np.random.shuffle(self.bg_fnames)
    
            # Load all bg clips in full duration
            self.bg_clips = {fn: load_audio(fn, fs=self.fs, normalize=self.normalize_audio) 
                             for fn in self.bg_fnames}
            self.n_bg_files = len(self.bg_clips)

            # Shuffle the segments of each bg clip
            for fname in self.bg_fnames:
                np.random.shuffle(self.fns_bg_seg_dict[fname])

    def load_and_store_ir_samples(self, ir_mix_parameter):
        """ Load Impulse Response samples in memory and their segmentation 
        information. We only use the first segment of each IR clip. These 
        segments are truncated to .

        Parameters:
        ----------
            ir_mix_parameter list([(bool), list(str), float]):
                [True, IR_FILEPATHS, 0.5].

        """

        self.ir_mix = ir_mix_parameter[0]
        if self.ir_mix:
            print("Loading Impulse Response samples in memory...")
            self.max_ir_length = int(ir_mix_parameter[2] * self.fs)
            self.fns_ir_seg_dict = get_fns_seg_dict(ir_mix_parameter[1], 
                                                    segment_mode='first',
                                                    fs=self.fs, 
                                                    duration=self.duration)
            self.ir_fnames = list(self.fns_ir_seg_dict.keys())
            # Shuffle the ir_fnames
            np.random.shuffle(self.ir_fnames)
            # Load all ir clips in full duration
            self.ir_clips = {}
            for fn in self.ir_fnames:
                X = load_audio(fn, 
                            seg_length_sec=self.duration,
                            fs=self.fs,
                            normalize=self.normalize_audio)
                # Truncate IR to MAX_IR_DURATION
                if len(X) > self.max_ir_length:
                    X = X[:self.max_ir_length]
                self.ir_clips[fn] = X
            self.n_ir_files = len(self.ir_clips)
