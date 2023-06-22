# -*- coding: utf-8 -*-
"""melsprctrogram.py"""    
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda, Permute
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank
import math


class Melspec_layer(Model):
    """
    A wrapper class, based on the implementation:
        https://github.com/keunwoochoi/kapre
        
    Input:
        (B,1,T)
    Output:
        (B,C,T,1) with C=Number of mel-bins
    
    USAGE:
        
        See get_melspec_layer() in the below.
        
    """
    def __init__(
            self,
            input_shape=(1, 8000),
            segment_norm=False,
            n_fft=1024,
            stft_hop=256,
            n_mels=256,
            fs=8000,
            dur=1.,
            f_min=300.,
            f_max=4000.,
            amin=1e-5, # minimum amp.
            dynamic_range=80.,
            name='Mel-spectrogram',
            trainable=False,
            **kwargs
            ):
        super(Melspec_layer, self).__init__(name=name, trainable=False, **kwargs)
        
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
        self.segment_norm = segment_norm
        
        # 'SAME' Padding layer
        self.pad_l = n_fft // 2
        self.pad_r = n_fft // 2
        self.padded_input_shape = (1, int(fs * dur) + self.pad_l + self.pad_r)
        self.pad_layer = Lambda(
            lambda z: tf.pad(z, tf.constant([[0, 0], [0, 0],
                                             [self.pad_l, self.pad_r]]))
            )
        
        # Construct log-power Mel-spec layer
        self.m = self.construct_melspec_layer(input_shape, name)

        # Permute layer
        self.p = tf.keras.Sequential(name='Permute')
        self.p.add(Permute((3, 2, 1), input_shape=self.m.output_shape[1:]))
        
        super(Melspec_layer, self).build((None, input_shape[0], input_shape[1]))
        
        
    def construct_melspec_layer(self, input_shape, name):
        m = tf.keras.Sequential(name=name)
        m.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        m.add(self.pad_layer)
        m.add(
            STFT(
                n_fft=self.n_fft,
                hop_length=self.stft_hop,
                pad_begin=False, # We do not use Kapre's padding, due to the @tf.function compatiability
                pad_end=False, # We do not use Kapre's padding, due to the @tf.function compatiability
                input_data_format='channels_first',
                output_data_format='channels_first')
            )
        m.add(
            Magnitude()
            )
        m.add(
            ApplyFilterbank(type='mel',
                            filterbank_kwargs=self.mel_fb_kwargs,
                            data_format='channels_first'
                            )
            )
        return m


    @tf.function
    def call(self, x):

        # Amplitude Mel-Spectrogram
        x = self.m(x)
        # Clip x below from amin
        x = tf.maximum(x, self.amin)
        # log-power Mel-spectrogram
        x_ref = tf.reduce_max(x) # Reference is the maximum value of x
        x = 20*tf.math.log(x/x_ref)/ tf.math.log(tf.constant(10, dtype=x.dtype))
        # Clip x below from -dynamic_range dB
        x = tf.maximum(x, -1 * self.dynamic_range)
        # Normalize x to be in [-1, 1]
        if self.segment_norm:
            x = (x + self.dynamic_range/2) / (self.dynamic_range/2)

        return self.p(x) # Permute((3,2,1))

class Melspec_layer_essentia():

    def __init__(
            self,
            input_shape=(1, 8000),
            segment_norm=False,
            n_fft=1024,
            stft_hop=256,
            n_mels=256,
            fs=8000,
            dur=1.,
            f_min=300.,
            f_max=4000.,
            amin=1e-5, # minimum mel-spectrogram amp.
            dynamic_range=80.,
            # name='Mel-spectrogram',
            # trainable=False,
            #**kwargs
            ):
        super().__init__()

        import numpy as np
        import essentia.standard as es

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
        self.segment_norm = segment_norm

        self.input_shape = input_shape
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

    def call(self, audio):

        assert audio.shape[1] == self.input_shape[1], f'Input shape is {audio.shape[1]} '\
                                                        f"but should be {self.input_shape[1]}"

        # Pad the segment from both sides
        audio = self.pad_audio(audio)

        # Calculate the Magnitude Mel-spectrogram
        mel_spec = [self.mb(self.spec(self.window(frame))) for frame in self.frame_generator(audio)]
        mel_spec = np.array(mel_spec) # (n_frames, n_mels)
        mel_spec = np.where(mel_spec>self.amin, mel_spec, self.amin) # Clip magnitude below amin

        # Convert to Power Mel-spectrogram
        mel_spec = 20*np.log10(mel_spec/np.max(mel_spec))
        # Clip x below from -dynamic_range dB
        mel_spec = np.where(mel_spec>-self.dynamic_range, mel_spec, -self.dynamic_range)

        # Scale x to be in [-1, 1] if segment_norm is True
        if self.segment_norm:
            mel_spec = (mel_spec + self.dynamic_range/2) / (self.dynamic_range/2)

        return mel_spec

    def pad_audio(self, audio):
        # Combining with how the framing is defined, zero-center the window to the first frame

        return np.concatenate((np.zeros(self.pad_l), audio, np.zeros(self.pad_r))).astype(np.float32)


def get_melspec_layer(cfg, trainable=False):
    fs = cfg['MODEL']['FS']
    dur = cfg['MODEL']['DUR']
    n_fft = cfg['MODEL']['STFT_WIN']
    stft_hop = cfg['MODEL']['STFT_HOP']
    n_mels = cfg['MODEL']['N_MELS']
    f_min = cfg['MODEL']['F_MIN']
    f_max = cfg['MODEL']['F_MAX']
    input_shape = (1, int(fs * dur))

    l = Melspec_layer(input_shape=input_shape,
                      segment_norm=cfg['MODEL']['SEGMENT_NORM'],
                      n_fft=n_fft,
                      stft_hop=stft_hop,
                      n_mels=n_mels,
                      fs=fs,
                      dur=dur,
                      f_min=f_min,
                      f_max=f_max)
    l.trainable = trainable
    return l
