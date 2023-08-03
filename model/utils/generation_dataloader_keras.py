import numpy as np
from tensorflow.keras.utils import Sequence

from model.utils.audio_utils import max_normalize
from model.fp.melspec.melspectrogram import Melspec_layer_essentia

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
        self.fs = fs
        self.normalize_audio = normalize_audio
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

        self.track_paths = track_paths
        self.n_tracks = len(self.track_paths)
        # This is sketchy. We assume that all tracks have the same number of
        # segments. This is true for the current dataset.
        self.n_samples = self.n_tracks * self.segments_per_track

    def __len__(self):
        """ Returns the number of batches. We fit segments_per_track segments
        to every batch. """

        return self.n_tracks

    def __getitem__(self, idx):
        """ 

        Parameters:
        ----------
            idx (int):
                Batch index

        Returns:
        --------
            Xa_batch (ndarray):
                anchor audio samples (n_anchor, T)
            Xa_batch_mel (ndarray):
                power mel-spectrogram of anchor samples (n_anchor, n_mels, T, 1)

        """

        # Load the segments from .npy file
        track_path = self.track_paths[idx]
        Xa_batch = np.load(track_path)
        # Keep only the first segments_per_track segments
        Xa_batch = Xa_batch[:self.segments_per_track]
        # Normalize the audio if required
        if self.normalize_audio:
            Xa_batch = max_normalize(Xa_batch)

        # Compute mel spectrograms
        Xa_batch_mel = self.mel_spec.compute_batch(Xa_batch)

        # Fix the dimensions
        Xa_batch_mel = np.expand_dims(Xa_batch_mel, 3).astype(np.float32)

        return Xa_batch, Xa_batch_mel
