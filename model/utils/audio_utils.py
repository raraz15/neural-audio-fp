# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" audio_utils.py """

import os
import wave
import numpy as np
import essentia.standard as es
from scipy.signal import convolve

#### File Check ####

def get_fns_seg_dict(fns_list=[],
                     segment_mode='all',
                     fs=8000,
                     duration=1,
                     hop=None):
    """
    Opens an audio file, checks its format and sample rate, and creates a list 
    of segments and possible offset ranges.

    Parameters:
    -----------
        fns_list: list of filenames. Only support .wav

    Returns:
    --------
        fns_event_seg_dict: list of segments.
        {filename: [[seg_idx, offset_min, offset_max], [ ... ] , ... [ ... ]]}
            filename is a string
            seg_idx is an integer
            offset_min is 0 or negative integer
            offset_max is 0 or positive integer
    """

    if hop == None:
        hop = duration

    # Get audio info
    n_samples_in_seg = int(fs * duration)
    n_samples_in_hop = int(fs * hop) # 2019 09.05

    fns_event_seg_dict = {}
    for filename in fns_list:

        fns_event_seg_dict[filename] = []

        # Only support .wav
        file_ext = os.path.splitext(filename)[1]
        if file_ext != '.wav':
            raise NotImplementedError(file_ext)

        # Open wav file
        pt_wav = wave.open(filename, 'r')

        # Check sample rate
        _fs = pt_wav.getframerate()
        if fs != _fs:
            raise ValueError('Sample rate should be {} but got {} for {}'.format(str(fs), str(_fs)), filename)

        # Get the number of samples the file has
        n_total_samples = pt_wav.getnframes()
        pt_wav.close()

        # Calculate number of segments and residual frames
        if n_total_samples > n_samples_in_seg:
            n_segs = int((n_total_samples - n_samples_in_seg + n_samples_in_hop) // n_samples_in_hop)
            assert n_segs > 0
        else:
            n_segs = 1 # load_audio can pad the audio if it is shorter than n_samples_in_seg
        residual_samples = np.max([0, n_total_samples - ((n_segs - 1) * n_samples_in_hop + n_samples_in_seg)])

        # Create a list of segments from the file
        if segment_mode == 'all': # Load all segments
            # A segment can be randomly offsetted to the left or right without going out of bounds
            for seg_idx in range(n_segs):
                # A segment can be offsetted max by n_samples_in_hop to the left or right
                offset_min, offset_max = -1*n_samples_in_hop, n_samples_in_hop
                if seg_idx == 0: # first seg
                    offset_min = 0 # no offset to the left
                if seg_idx == n_segs - 1: # last seg
                    offset_max = residual_samples # Maximal offset to the right is the residual frames
                fns_event_seg_dict[filename].append([seg_idx, offset_min, offset_max])
        elif segment_mode == 'first':
            # Load only the first segment
            seg_idx = 0
            offset_min, offset_max = 0, 0
            fns_event_seg_dict[filename].append([seg_idx, offset_min, offset_max])
        elif segment_mode == 'random_oneshot':
            # Load only one random segment
            seg_idx = np.random.randint(0, n_segs)
            offset_min, offset_max = n_samples_in_hop, n_samples_in_hop
            if seg_idx == 0:  # first seg
                offset_min = 0
            if seg_idx == n_segs - 1:  # last seg
                offset_max = residual_samples
            fns_event_seg_dict[filename].append([seg_idx, offset_min, offset_max])
        else:
            raise NotImplementedError(segment_mode)

    return fns_event_seg_dict

#### Audio Processing ####

def max_normalize(x):
    """
    Parameters
    ----------
    x : (ndarray)

    Returns
    -------
    (float)
        Max-normalized audio signal.
    """

    if len(x.shape)==1:
        max_val = np.max(np.abs(x))
        if max_val==0:
            return x
        else:
            return x / np.max(np.abs(max_val))
    elif len(x.shape)==2:
        max_val = np.max(np.abs(x), axis=1, keepdims=True)
        max_val[max_val==0] = 1
        return x / max_val
    else:
        raise NotImplementedError

#### Audio IO ####

def load_wav(filename=str(), 
            seg_start_sec=0.0,
            offset_sec=0.0,
            seg_length_sec=None,
            seg_pad_offset_sec=0.0,
            fs=8000):
    """
    Opens a wav file, checks its format and sample rate, and returns a segment.

    Parameters:
    -----------
        filename: string
        seg_start_sec: Start of the segment in seconds.
        offset_sec: Offset from seg_start_sec in seconds.
        seg_length_sec: Length of the segment in seconds.
        seg_pad_offset_sec: If padding is required (seg_length_sec is longer than file duration),
            pad the segment from the rgiht and give an offset from the left 
            with this amount of seconds.
        fs: sample rate
        normalize: max-normalize the audio signal

    Returns:
    --------
        audio: numpy array of shape (n_samples,)

    Returns:
        x: Segment (T,)
    """

    assert seg_start_sec>=0.0, "The start time must be positive"
    if seg_length_sec is not None:
        assert seg_length_sec>0.0, "If you specify a duration, it must be positive."\
                                    "Use None to read the rest of the file."

    # Open file
    pt_wav = wave.open(filename, 'r')

    # Check sample rate
    _fs = pt_wav.getframerate()
    if fs != _fs:
        raise ValueError(f'Sample rate should be {fs} but got {_fs} for {filename}')

    # Calculate segment start index
    start_frame_idx = np.floor((seg_start_sec + offset_sec) * fs).astype(int)
    pt_wav.setpos(start_frame_idx)

    # Determine the segment length
    if seg_length_sec is None:
        seg_length_frame = pt_wav.getnframes() - start_frame_idx
    else:
        # if seg_length_sec is bigger than the file size, it loads everything
        # else read seg_length_sec starting from start_frame_idx
        seg_length_frame = np.floor(seg_length_sec * fs).astype(int)

    # Load audio and close file
    x = pt_wav.readframes(seg_length_frame)
    pt_wav.close()

    # Convert bytes to float
    x = np.frombuffer(x, dtype=np.int16)
    x = x / 2**15 # normalize to [-1, 1] float

    # Pad the segment if it is shorter than seg_length_sec
    if len(x) < seg_length_frame:
        audio_arr = np.zeros(int(seg_length_sec * fs))
        seg_pad_offset_idx = int(seg_pad_offset_sec * fs)
        assert seg_pad_offset_idx + len(x) <= seg_length_frame, \
            "The padded segment is longer than input duration and seg_pad_offset_sec."
        audio_arr[seg_pad_offset_idx:seg_pad_offset_idx + len(x)] = x
        return audio_arr
    else:
        return x

def read_cut_resample(filename, 
                    seg_start_sec=0.0, 
                    offset_sec=0.0, 
                    seg_length_sec=None, 
                    fs=8000, 
                    resample_quality=4):
    """
    Reads an audio file, cuts a segment, mixes to mono and resamples it. 
    Also pads or cuts the segment if necessary. It can read any audio 
    file format supported by FFmpeg.

    Parameters:
        filename: (string)
        seg_start_sec: (float)
            Start of the segment in seconds.
        offset_sec: (float)
            Offset from seg_start_sec in seconds.
        seg_length_sec: (float)
            Length of the segment in seconds. If None, read the rest of the file.
        fs: (int)
            Sample rate.
        resample_quality: (int)
            Quality of the resampling. 0 is the slowest, 4 is the fastest.

    Returns:
        x: Segment (T,)
    """

    if seg_length_sec is not None:
        assert seg_length_sec>0.0, \
            "If specified, duration must be positive"
    assert seg_start_sec>=0.0, \
        "The start time must be positive"
    assert resample_quality in {0,1,2,3,4}, \
        "resample_quality should be in {0,1,2,3,4}"

    # Load the audio
    audio, _fs, numberChannels, _, _, _ = es.AudioLoader(filename=filename)()

    # Calculate the start and end samples
    t0 = seg_start_sec + offset_sec
    n0 = int(t0*_fs)
    assert n0>=0, "The start time + offset is before the start of the audio"

    if seg_length_sec is None:
        n1 = len(audio)
    else:
        n1 = n0 + int(_fs*seg_length_sec)
        assert n1<=len(audio), \
        "The end time is after the end of the audio. Specify a shorter duration."

    # Cut the audio
    audio = audio[n0:n1]

    # Convert to mono if necessary
    if numberChannels>1:
        audio = np.mean(audio, axis=1)

    # Resample if necessary
    if _fs!=fs:
        resampler = es.Resample(inputSampleRate=_fs, 
                                outputSampleRate=fs, 
                                quality=resample_quality)
        audio = resampler(audio)

    # Pad the audio to the correct length if necessary
    N_desired = int(seg_length_sec*fs)
    if (seg_length_sec is not None) and len(audio)<N_desired:
        audio = np.pad(audio, 
                    (0, N_desired-len(audio)), 
                    mode='constant')

    return audio

def load_audio(filename=str(),
               seg_start_sec=0.0,
               offset_sec=0.0,
               seg_length_sec=None,
               fs=8000,
               normalize=True):
    """ Loads an audio file.

    Parameters:
        filename: string
        seg_start_sec: start reading from this time in seconds
        offset_sec: offset the seg_start_sec by this amount of seconds
        seg_length_sec: read this amount of seconds. If None, read the rest of the file.

    Returns:
        audio: numpy array of shape (n_total_samples,)
    """

    assert (seg_length_sec is None) or (seg_length_sec > 0.0), 'seg_length_sec should be positive'\
                                        'or None (read all the rest of the file from seg_start_sec).'

    # Only support .wav or .mp3 or .mp4
    file_ext = os.path.splitext(filename)[1]
    if file_ext == '.wav':
        x = load_wav(filename=filename,
                     seg_start_sec=seg_start_sec,
                     offset_sec=offset_sec,
                     seg_length_sec=seg_length_sec,
                     fs=fs)
    elif file_ext == '.mp3' or file_ext == '.mp4':
        x = read_cut_resample(filename=filename,
                              seg_start_sec=seg_start_sec,
                              offset_sec=offset_sec,
                              seg_length_sec=seg_length_sec,
                              fs=fs)
    else:
        raise NotImplementedError(file_ext)

    # Max Normalize
    if normalize:
        x = max_normalize(x)

    return x

def load_audio_multi_start(filename=str(),
                           seg_start_sec_list=[],
                           seg_length_sec=float(),
                           fs=8000,
                           normalize=True):
    """ Load_audio wrapper for loading audio with multiple start indices each 
    with same duration. 

    Returns
    -------
        out : 2D array (float)
            Batch of audio signals. (B, T)

    """

    out = []
    for seg_start_sec in seg_start_sec_list:
        x = load_audio(filename=filename,
                       seg_start_sec=seg_start_sec,
                       seg_length_sec=seg_length_sec,
                       fs=fs,
                       normalize=normalize)
        out.append(x.reshape((1, -1)))
    out = np.vstack(out)
    return out  # (B,T)

def npy_to_wav(root_dir=str(), source_fs=int(), target_fs=int()):
    import wavio, glob, scipy

    fns = glob.glob(root_dir + '**/*.npy', recursive=True)
    for fname in fns:
        audio = np.load(fname)
        resampled_length = int(len(audio) / source_fs * target_fs)
        audio = scipy.signal.resample(audio, resampled_length)
        audio = audio * 2**15
        audio = audio.astype(np.int16)  # 16-bit PCM
        wavio.write(fname[:-4] + '.wav', audio, target_fs, sampwidth=2)

#### Background Noise Augmentation ####

def background_mix(x, x_bg, snr_db):
    """
    Parameters
    ----------
    x : 1D array (float)
        Input audio signal.
    x_bg : 1D array (float)
        Background noise signal.
    snr_db : (float)
        signal-to-noise ratio in decibel.

    Returns
    -------
    1D array
        Max-normalized mix of x and x_bg with SNR
    """

    def _RMS_amplitude(x):
        return np.sqrt(np.mean(x**2))

    assert len(x) == len(x_bg), 'x and x_bg should have the same length.'

    # Get the RMS Amplitude for each signal
    rms_bg = _RMS_amplitude(x_bg)
    rms_x = _RMS_amplitude(x)
    if rms_bg != 0 and rms_x != 0:

        # Normalize each signal by its RMS Amplitude
        x_bg = x_bg / rms_bg
        x = x / rms_x

        # Mix with snr_db
        magnitude = np.power(10, snr_db / 20.)
        x_mix = magnitude * x + x_bg

    elif rms_bg == 0 and rms_x == 0:
        # Both signals are zero so just return zeros
        x_mix = np.zeros_like(x)
        print('Both signals are zero!')
    else:
        # One of the signal is zero so just add them
        x_mix = x + x_bg

    # Max normalize the mix signal to avoid clipping
    x_mix = max_normalize(x_mix)

    return x_mix

def bg_mix_batch(event_batch, bg_batch, snr_range=(6, 24)):
    """ Mix a batch of events with a batch of background noise with a 
    uniformly random SNR (dB) from snr_range. SNR is sampled for each 
    sample in the batch.

    Parameters
    ----------
        event_batch : 2D array (float)
            Batch of event signals. (B, T)
        bg_batch : 2D array (float)
            Batch of background noise signals. (B, T)
        snr_range : tuple (float)
            SNR range in dB. (min, max)
    """

    assert snr_range[0] < snr_range[1], 'snr_range should be (min, max)'
    assert event_batch.shape == bg_batch.shape, \
        'event_batch and bg_batch should have the same shape.'

    # Initialize
    X_bg_mix = np.zeros((event_batch.shape[0], event_batch.shape[1]))

    # Random SNR for each sample in the batch
    min_snr, max_snr = snr_range
    snrs = np.random.rand(len(event_batch))
    snrs = snrs * (max_snr - min_snr) + min_snr

    # Mix each element with random SNR
    for i in range(len(event_batch)):
        X_bg_mix[i] = background_mix(x=event_batch[i],
                                    x_bg=bg_batch[i],
                                    snr_db=snrs[i])

    return X_bg_mix

#### Room IR Augmentation ####

def ir_aug(x, x_ir):
    """ Augment input signal with impulse response. The returned signal
    has the same length as x and is max-normalized."""

    assert len(x) > 0, 'x should not be empty.'
    assert len(x_ir) > 0, 'x_ir should not be empty.'
    assert len(x)>=len(x_ir), 'x should be longer than x_ir.'

    # Convolve with impulse response
    x_aug = convolve(x, x_ir, mode='same', method='fft')
    x_aug = max_normalize(x_aug)

    return x_aug

def ir_aug_batch(event_batch, ir_batch):
    """ Augment a batch of events with a batch of impulse responses. """

    n_batch = len(event_batch)
    X_ir_aug = np.zeros((n_batch, event_batch.shape[1]))

    for i in range(n_batch):
        x = event_batch[i]
        x_ir = ir_batch[i]
        x_aug = ir_aug(x, x_ir)
        X_ir_aug[i] = x_aug

    return X_ir_aug