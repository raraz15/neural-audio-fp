# -*- coding: utf-8 -*-

import numpy as np

import essentia.standard as es

# TODO: let essentia pad start and end
class Melspec_layer_essentia():

    def __init__(
            self,
            scale=True,
            n_fft=1024,
            stft_hop=256,
            n_mels=256,
            fs=8000,
            dur=1.,
            f_min=300.,
            f_max=4000.,
            amin=1e-5, # minimum mel-spectrogram amp.
            dynamic_range=80.,
            ):
        super().__init__()

        

        self.mel_fb_kwargs = {
            'sample_rate': fs,
            'n_freq': n_fft // 2 + 1,
            'n_mels': n_mels,
            'f_min': f_min,
            'f_max': f_max,
            }
        self.n_fft = n_fft
        self.stft_hop = stft_hop
        self.n_mels = n_mels
        self.amin = amin
        self.dynamic_range = dynamic_range
        self.scale = scale

        self.input_shape = (1, int(fs * dur))
        self.pad_l = n_fft // 2
        self.pad_r = n_fft // 2
        self.padded_input_shape = (1, int(fs * dur) + self.pad_l + self.pad_r)

        # Create the frame generator
        self.frame_generator = lambda x: es.FrameGenerator(x, 
                                            frameSize=n_fft, 
                                            hopSize=stft_hop, 
                                            startFromZero=True, # Do not zero-center the window to the first frame
                                            lastFrameToEndOfFile=False,
                                            validFrameThresholdRatio=1.0, # Discard if small frames are left at the end
                                            )

        # Create the window
        self.window = es.Windowing(type="hann", 
                                    normalized=False, # Seems like tf is not normalized
                                    size=n_fft,
                                    symmetric=False, # Why for spectral analysis is not symmetric?
                                    zeroPhase=False, # Probably not important since we are using amplitude spectrogram?
                                    )

        # Define the FFT
        self.spec = es.Spectrum(size=n_fft)

        # Define the Mel bands
        self.mb = es.MelBands(
            highFrequencyBound=f_max,
            inputSize=n_fft // 2 + 1,
            log=False,
            lowFrequencyBound=f_min,
            normalize="unit_tri",
            numberBands=n_mels,
            sampleRate=fs,
            type="magnitude",
            warpingFormula="slaneyMel",
            weighting="linear",
        )

    def compute_mel_spectrogram(self, audio):
        """Compute the Mel-spectrogram of an audio segment. The audio segment is padded 
        from both sides to center the first window.

        Inputs:
            audio: (T,)

        Returns:
            mel_spec: (F, T)
        """

        assert audio.shape[0] == self.input_shape[1], f'Input shape is {audio.shape[0]} '\
                                                        f"but should be {self.input_shape[1]}"

        # Pad the segment from both sides
        audio = self.pad_audio(audio)

        # Calculate the Magnitude Mel-spectrogram
        mel_spec = [self.mb(self.spec(self.window(frame))) for frame in self.frame_generator(audio)]
        mel_spec = np.array(mel_spec) # (n_frames, n_mels)
        # Clip magnitude below amin. This is to avoid log(0) in the next step
        mel_spec = np.where(mel_spec>self.amin, mel_spec, self.amin)

        # Convert to Power Mel-spectrogram
        mel_spec = 20*np.log10(mel_spec/np.max(mel_spec))
        # Clip below from -dynamic_range dB
        mel_spec = np.where(mel_spec>-self.dynamic_range, mel_spec, -self.dynamic_range)

        # Scale x to be in [-1, 1] if scale is True
        if self.scale:
            mel_spec = (mel_spec + self.dynamic_range/2) / (self.dynamic_range/2)

        return mel_spec.T # (n_mels, n_frames)

    def compute_batch(self, batch):
        """Returns: (B, F, T)"""

        return np.array([self.compute_mel_spectrogram(audio) for audio in batch])

    def pad_audio(self, audio):
        # Combining with how the framing is defined, zero-center the window to the first frame

        return np.concatenate((np.zeros(self.pad_l), audio, np.zeros(self.pad_r))).astype(np.float32)

def get_Melspec_layer_essentia(cfg):
    fs = cfg['MODEL']['FS']
    dur = cfg['MODEL']['DUR']
    n_fft = cfg['MODEL']['STFT_WIN']
    stft_hop = cfg['MODEL']['STFT_HOP']
    n_mels = cfg['MODEL']['N_MELS']
    f_min = cfg['MODEL']['F_MIN']
    f_max = cfg['MODEL']['F_MAX']

    return Melspec_layer_essentia(
                        scale=cfg['MODEL']['SCALE_INPUTS'], 
                        n_fft=n_fft, 
                        stft_hop=stft_hop, 
                        n_mels=n_mels, 
                        fs=fs, 
                        dur=dur, 
                        f_min=f_min, 
                        f_max=f_max)
