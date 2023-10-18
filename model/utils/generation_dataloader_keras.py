import numpy as np
from tensorflow.keras.utils import Sequence

from model.utils import audio_utils
from model.fp.melspec.melspectrogram import Melspec_layer_essentia

SEED = 27 # Used during augmentation
np.random.seed(SEED)

class GenerationLoader(Sequence):
    def __init__(
        self,
        track_paths: list,
        segment_duration=1,
        hop_duration=0.5,
        fs=8000,
        n_fft=1024,
        stft_hop=256,
        n_mels=256,
        f_min=300,
        f_max=4000,
        scale_output=True,
        bg_mix_parameter=[False],
        ir_mix_parameter=[False],
        bsz=120,
        ):
        """
        Parameters
        ----------
        track_paths : list(str), 
            Track .wav paths as a list.
        segment_duration : (float), optional
            Segment duration in seconds. The default is 1.
        hop_duration : (float), optional
            Hop-size of segments in seconds. The default is .5.
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
        bg_mix_parameter : list([(bool), list(str), (int, int)]), optional
            [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR)]. Default is [False].
        ir_mix_parameter : list([(bool), list(str), float], optional
            [True, IR_FILEPATHS, MAX_IR_DURATION]. Default is [False].
        bsz : (int), optional
            Generation batch size. The default is 120.
        """

        assert segment_duration > 0, "segment_duration should be > 0"
        assert hop_duration > 0, "hop_duration should be > 0"
        assert hop_duration <= segment_duration, \
            "hop_duration should be <= segment_duration"

        # Save the Input parameters
        self.segment_duration = segment_duration
        self.segment_length = int(fs*self.segment_duration)
        self.hop_duration = hop_duration
        self.hop_length = int(fs*hop_duration)
        self.bg_hop = segment_duration # Background noise hop-size is equal to segment duration
        self.fs = fs
        self.bsz = bsz

        # Create segment information for each track
        track_seg_dict = audio_utils.get_fns_seg_dict(track_paths,
                                                    segment_mode='all',
                                                    fs=self.fs,
                                                    duration=self.segment_duration,
                                                    hop=self.hop_duration)
        # Create a list of track-segment pairs. We connvert it to a list so that
        # each segment can be used during fp-generation.
        self.track_seg_list = [[fname, *seg] 
                               for fname, segments in track_seg_dict.items() 
                               for seg in segments]
        self.n_samples = len(self.track_seg_list)
        self.indexes = np.arange(self.n_samples)

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
        """ Returns the number of batches."""

        # Ceil is used to make sure that all the samples are used
        return int(np.ceil(self.n_samples/self.bsz))

    def __getitem__(self, idx):
        """ Loads  a batch of segments. If bg_mix and ir_mix are False, we
        simply load the segments. If bg_mix and ir_mix are True, we mix the
        segments with background noise and convolve them with a room Impulse
        Response (IR) in cascade. We use a different background noise and IR
        for each segment in the batch.

        Parameters:
        ----------
            idx (int):
                Batch index

        Returns:
        --------
            X_batch (ndarray):
                audio samples (bsz, T)
            X_batch_mel (ndarray):
                power mel-spectrogram of samples (bsz, n_mels, T, 1)

        """

        # Get the segments for this batch
        X_batch = []
        for i in self.indexes[idx*self.bsz:(idx+1)*self.bsz]:

            # Get the track information
            fname, seg_idx, _, _ = self.track_seg_list[i]

            # Determine the anchor start time
            anchor_start_sec = seg_idx * self.hop_duration

            # Load the anchor segment and append it to the batch
            xs = audio_utils.load_audio(fname,
                                        seg_start_sec=anchor_start_sec, 
                                        seg_length_sec=self.segment_duration, 
                                        fs=self.fs,
                                        normalize=False)
            X_batch.append(xs.reshape((1, -1)))
        # Create the batch of audio
        X_batch = np.concatenate(X_batch, axis=0)

        # Apply augmentations if specified
        if self.bg_mix and self.ir_mix:

            # TODO: simplify?
            # Get a batch of random background noise samples
            bg_noise_batch = []
            for i in np.arange(idx*self.bsz, (idx+1)*self.bsz) % self.n_bg_files:
                bg_noise_sample = self.bg_clips[self.bg_fnames[i]]
                if len(bg_noise_sample) > self.segment_length:
                    # Get a random segment of the background noise
                    start_idx = np.random.randint(0, len(bg_noise_sample) - self.segment_length)
                    bg_noise_sample = bg_noise_sample[start_idx:start_idx+self.segment_length]
                else:
                    # Pad the background noise with zeros
                    bg_noise_sample = np.pad(bg_noise_sample, 
                                            (0, self.segment_length - len(bg_noise_sample)))
                bg_noise_batch.append(bg_noise_sample)
            bg_noise_batch = np.concatenate(bg_noise_batch, axis=0)

            # Mix the batch of segments with the batch of background noises
            X_batch = audio_utils.bg_mix_batch(X_batch,
                                        bg_noise_batch,
                                        snr_range=self.bg_snr_range)

            # Get a batch of random IR samples
            ir_batch = []
            for i in np.arange(idx*self.bsz, (idx+1)*self.bsz) % self.n_ir_files:
                # IR has all the segments of the same length
                ir_batch.append(self.ir_clips[self.ir_fnames[i]])
            ir_batch = np.concatenate(ir_batch, axis=0)

            # Convolve with IR
            X_batch = audio_utils.ir_aug_batch(X_batch, ir_batch, normalize=True)

        else:

            # TODO: is this important?
            # Normalize the batch of segments
            X_batch = audio_utils.max_normalize(X_batch)

        # Compute mel spectrograms
        X_batch_mel = self.mel_spec.compute_batch(X_batch)
        # Fix the dimensions and types
        X_batch_mel = np.expand_dims(X_batch_mel, 3).astype(np.float32)

        return X_batch, X_batch_mel

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
                                                        normalize=True) 
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
                                            fs=self.fs, normalize=True) 
                                            for fn in self.ir_fnames}

            # Truncate IRs to self.max_ir_length
            self.ir_clips = {fn: X[:self.max_ir_length] for fn, X in self.ir_clips.items()}

            # Save the number of IR files
            self.n_ir_files = len(self.ir_clips)
