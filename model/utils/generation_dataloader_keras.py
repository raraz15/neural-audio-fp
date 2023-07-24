import numpy as np
from tensorflow.keras.utils import Sequence

from model.utils.audio_utils import get_fns_seg_dict, load_audio
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
        bsz=120,
        ):
        """
        Parameters
        ----------
        track_paths : list(str), 
            Track file paths as a list. 
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
        bsz : (int), optional
            In TPUs code, global batch size. The default is 120.
        """

        # Check parameters
        assert bsz > 0, "bsz should be > 0"

        # Save the Input parameters
        self.duration = duration
        self.hop = hop
        self.fs = fs
        self.normalize_audio = normalize_audio
        self.bg_hop = duration # For bg samples hop is equal to duration
        self.bsz = bsz
        self.n_anchor = self.bsz # During fp-generation all samples are anchors

        # Melspec layer
        self.mel_spec = Melspec_layer_essentia(scale=scale,
                                            n_fft=n_fft, 
                                            stft_hop=stft_hop, 
                                            n_mels=n_mels, 
                                            fs=fs, 
                                            dur=duration, 
                                            f_min=f_min, 
                                            f_max=f_max)

        # Create segment information for each track
        track_seg_dict = get_fns_seg_dict(track_paths,
                                        segment_mode='all',
                                        fs=self.fs,
                                        duration=self.duration,
                                        hop=self.hop)
        # Create a list of track-segment pairs. We connvert it to a list so that
        # each segment can be used during fp-generation.
        self.track_seg_list = [[fname, *seg] 
                               for fname, segments in track_seg_dict.items() 
                               for seg in segments]
        self.n_samples = len(self.track_seg_list)
        self.indexes = np.arange(self.n_samples)

    def __len__(self):
        """ Returns the number of batches"""

        return int(np.ceil(self.n_samples/self.n_anchor))

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

        Xa_batch = []
        for i in self.indexes[idx*self.n_anchor:(idx+1)*self.n_anchor]:

            # Load the segment information of the anchor track
            fname, seg_idx, _, _ = self.track_seg_list[i]

            # Determine the anchor start time
            anchor_start_sec = seg_idx * self.hop

            # Load the anchor 
            xs = load_audio(fname,
                            seg_start_sec=anchor_start_sec, 
                            seg_length_sec=self.duration, 
                            fs=self.fs,
                            normalize=self.normalize_audio)
            Xa_batch.append(xs.reshape((1, -1)))

        # Create a batch
        Xa_batch = np.concatenate(Xa_batch, axis=0)

        # Compute mel spectrograms
        Xa_batch_mel = self.mel_spec.compute_batch(Xa_batch)

        # Fix the dimensions
        Xa_batch_mel = np.expand_dims(Xa_batch_mel, 3).astype(np.float32)

        return Xa_batch, Xa_batch_mel
