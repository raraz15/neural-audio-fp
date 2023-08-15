import numpy as np
from tensorflow.keras.utils import Sequence

from model.utils import audio_utils
from model.fp.melspec.melspectrogram import Melspec_layer_essentia

SEED = 27 # Used during augmentation
np.random.seed(SEED)

class genUnbalSequenceGeneration(Sequence):
    def __init__(
        self,
        track_paths,
        segment_duration=1,
        hop=.5,
        normalize_segment=True,
        fs=8000,
        n_fft=1024,
        stft_hop=256,
        n_mels=256,
        f_min=300,
        f_max=4000,
        scale_output=True,
        segments_per_track=120,
        bg_mix_parameter=[False],
        ir_mix_parameter=[False],
        ):
        """
        Parameters
        ----------
        track_paths : list(str), 
            Track .npy paths as a list. 
        segment_duration : (float), optional
            Segment duration in seconds. The default is 1.
        hop : (float), optional
            Hop-size in seconds. The default is .5.
        normalize_segment : (str), optional
            Normalize each audio segment. Default is True.
        fs : (int), optional
            Sampling rate. The default is 8000.
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
            Scale the power mel-spectrogram. The default is True.
        segments_per_track : (int), optional
            Number of segments per track. The default is 120.
        bg_mix_parameter : list([(bool), list(str), (int, int)]), optional
            [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR)]. Default is [False].
        ir_mix_parameter : list([(bool), list(str), float], optional
            [True, IR_FILEPATHS, MAX_IR_DURATION]. Default is [False].
        """

        # Check parameters
        assert segments_per_track > 0, "segments_per_track should be > 0"

        # Save the Input parameters
        self.segment_duration = segment_duration
        self.hop = hop
        self.bg_hop = segment_duration
        self.fs = fs
        self.normalize_segment = normalize_segment
        self.segments_per_track = segments_per_track
        self.segment_length = int(self.fs*self.segment_duration)
        self.hop_length = int(self.fs*self.hop)
        self.overlap = (self.segment_length - self.hop_length) / self.segment_length
        self.chunk_length = int(((segments_per_track-1)*self.hop + self.segment_duration)*fs)

        self.track_paths = track_paths
        self.n_tracks = len(self.track_paths)
        # We assume that all tracks have the same number of
        # segments. This is true for the current dataset.
        self.n_samples = self.n_tracks * self.segments_per_track

        # Melspec layer
        self.mel_spec = Melspec_layer_essentia(scale=scale_output,
                                            n_fft=n_fft, 
                                            stft_hop=stft_hop, 
                                            n_mels=n_mels, 
                                            fs=fs, 
                                            dur=self.segment_duration, 
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
        segments_per_track segments. Each batch contains self.segments_per_track
        segments from a single track. If bg_mix is True, we mix the segments
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
        segments_path = self.track_paths[idx]
        X = np.load(segments_path)
        assert X.shape[1] == self.segment_length, \
                        "Loaded a segment with wrong duration."

        # Keep only the first segments_per_track segments
        X = X[:self.segments_per_track]

        # Apply BG augmentations if specified
        if self.bg_mix:

            # Reconstruct the audio chunk from the segments
            X = audio_utils.OLA(X, self.overlap)

            # Get a random background noise sample
            bg_fname = self.bg_fnames[idx%self.n_bg_files]
            bg_noise = self.read_bg(bg_fname)

            # Randomly sample an SNR for mixing the background noise
            snr = audio_utils.sample_SNR(1, self.bg_snr_range)

            # Mix the OLA'd track with the background noise
            X = audio_utils.background_mix(X.reshape(-1),
                                           bg_noise.reshape(-1),
                                           snr_db=snr)

            # Cut the track to segments (n_segments, segment_length)
            X, _ = audio_utils.cut_to_segments(X, 
                                               self.segment_length, 
                                               self.hop_length)

        # Apply IR augmentations if specified
        if self.ir_mix:

            # Get a random IR sample
            ir_fname = self.ir_fnames[idx%self.n_ir_files]
            ir = self.read_ir(ir_fname).reshape(1,-1)
    
            # Apply the same IR to all segments
            # This is not realistic, and its a little bit cheating but we
            # do it because the original authors get good results with this 
            # approach.
            ir = np.repeat(ir, X.shape[0], axis=0)

            # Convolve with IR
            X = audio_utils.ir_aug_batch(X, ir)

        # Normalize the segments independently if required
        if self.normalize_segment:
            X = audio_utils.max_normalize(X)

        # Compute mel spectrograms
        X_mel = self.mel_spec.compute_batch(X)
        # Fix the dimensions and types
        X_mel = np.expand_dims(X_mel, 3).astype(np.float32)

        return X, X_mel

    def read_bg(self, fname):
        """ Read background noise samples for given fnames from the memory and 
        return a batch. To simulate real life conditions during testing, we apply the
        same background noise to a segmented track. We cut the file into overlapping 
        segments of duration self.segment_duration. self.frames_per_file segments are read from
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

        # If the background noise sample is shorter than the chunk length,
        # we repeat it until we have a sample of length chunk_length
        if len(bg) < self.chunk_length:
            n_repeats = int(np.ceil(self.chunk_length / len(bg)))
            bg = np.tile(bg, n_repeats)
            bg = bg[:self.chunk_length]
        elif len(bg) > self.chunk_length:
            # Otherwise we cut a random part
            start = np.random.randint(0, len(bg) - self.chunk_length)
            bg = bg[start:start+self.chunk_length]

        return bg

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
        information. We shuffle order of the filenames and shuffle the order
        of the segments of each file.

        Parameters:
        ----------
            bg_mix_parameter list([(bool), list(str), (int, int)]):
                [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR)].

        """

        self.bg_mix = bg_mix_parameter[0]
        if self.bg_mix:

            # Save the SNR range
            self.bg_snr_range = bg_mix_parameter[2]

            # Check the fnames
            print("Loading Background Noise samples in memory...")
            self.fns_bg_seg_dict = audio_utils.get_fns_seg_dict(bg_mix_parameter[1], 
                                                    segment_mode='all',
                                                    fs=self.fs, 
                                                    duration=self.segment_duration,
                                                    hop=self.segment_duration)
            self.bg_fnames = list(self.fns_bg_seg_dict.keys())

            # Shuffle the bg_fnames
            np.random.shuffle(self.bg_fnames)
    
            # Load all bg clips in full duration
            self.bg_clips = {fn: audio_utils.load_audio(fn, fs=self.fs, 
                                                        normalize=self.normalize_segment) 
                             for fn in self.bg_fnames}
            self.n_bg_files = len(self.bg_clips)

            # Shuffle the segments of each bg clip
            for fname in self.bg_fnames:
                np.random.shuffle(self.fns_bg_seg_dict[fname])

    def load_and_store_ir_samples(self, ir_mix_parameter):
        """ Load Impulse Response samples in memory and their segmentation 
        information. We only use the first segment of each IR clip. These 
        segments are truncated to MAX_IR_DURATION. We shuffle order of the 
        filenames.

        Parameters:
        ----------
            ir_mix_parameter list([(bool), list(str), float]):
                [True, IR_FILEPATHS, MAX_IR_DURATION].

        """

        self.ir_mix = ir_mix_parameter[0]
        if self.ir_mix:

            # Save the max ir length
            self.max_ir_length = int(ir_mix_parameter[2] * self.fs)

            # Check the fnames
            print("Loading Impulse Response samples in memory...")
            self.fns_ir_seg_dict = audio_utils.get_fns_seg_dict(ir_mix_parameter[1], 
                                                    segment_mode='first',
                                                    fs=self.fs, 
                                                    duration=self.segment_duration)
            self.ir_fnames = list(self.fns_ir_seg_dict.keys())

            # Shuffle the ir_fnames
            np.random.shuffle(self.ir_fnames)

            # Load all IR segments
            self.ir_clips = {fn: audio_utils.load_audio(fn, 
                                            seg_length_sec=self.segment_duration, 
                                            fs=self.fs, normalize=self.normalize_segment) 
                                            for fn in self.ir_fnames}

            # Truncate IRs to MAX_IR_DURATION
            self.ir_clips = {fn: X[:self.max_ir_length] for fn, X in self.ir_clips.items()}

            # Save the number of IR files
            self.n_ir_files = len(self.ir_clips)
