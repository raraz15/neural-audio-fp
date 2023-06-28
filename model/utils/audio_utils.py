# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" audio_utils.py """

import os
import wave
import numpy as np

def max_normalize(x):
    """
    Parameters
    ----------
    x : (float)

    Returns
    -------
    (float)
        Max-normalized audio signal.
    """

    max_val = np.max(np.abs(x))
    if max_val==0:
        return x
    else:
        return x / np.max(np.abs(max_val))

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
    # Check length
    if len(x) > len(x_bg):  # This will not happen though...
        _x_bg = np.zeros(len(x))
        bg_start = np.random.randint(len(x) - len(x_bg))
        bg_end = bg_start + len(x_bg)
        _x_bg[bg_start:bg_end] = x_bg
        x_bg = _x_bg
    elif len(x) < len(x_bg):  # This will not happen though...
        bg_start = np.random.randint(len(x_bg) - len(x))
        bg_end = bg_start + len(x)
        x_bg = x_bg[bg_start:bg_end]

    # Normalize with energy
    rmse_bg = np.sqrt(np.mean(x_bg**2))
    x_bg = x_bg / rmse_bg
    rmse_x = np.sqrt(np.mean(x**2))
    x = x / rmse_x

    # Mix
    magnitude = np.power(10, snr_db / 20.)
    x_mix = magnitude * x + x_bg
    return max_normalize(x_mix)

# TODO: what is this?
def log_scale_random_number_batch(bsz=int(), amp_range=(0.1, 1.)):
    range_log = np.log10(amp_range)
    random_number_log = np.random.rand(bsz) * (np.max(range_log) - np.min(range_log)) + np.min(range_log)
    return np.power(10, random_number_log)

# TODO: how does it work?
def bg_mix_batch(event_batch,
                 bg_batch,
                 snr_range=(6, 24),
                 unit='db',
                 mode='energy'):
    X_bg_mix = np.zeros((event_batch.shape[0], event_batch.shape[1]))

    # Random SNR
    min_snr = np.min(snr_range)
    max_snr = np.max(snr_range)
    snrs = np.random.rand(len(event_batch))
    snrs = snrs * (max_snr - min_snr) + min_snr

    # Random amp (batch-wise)
    event_amp_ratio_batch = log_scale_random_number_batch(bsz=len(event_batch),
                                                          amp_range=(0.1, 1))

    for i in range(len(event_batch)):
        event_max = np.max(np.abs(event_batch[i]))
        bg_max = np.max(np.abs(bg_batch[i]))
        #event_amp_ratio = log_scale_random_number(amp_range=(0.01,1))

        if event_max == 0 or bg_max == 0:
            X_bg_mix[i] = event_batch[i] + bg_batch[i]
            X_bg_mix[i] = max_normalize(X_bg_mix[i])
        else:
            X_bg_mix[i] = background_mix(x=event_batch[i],
                                         x_bg=bg_batch[i],
                                         snr_db=snrs[i])
        X_bg_mix[i] = event_amp_ratio_batch[i] * X_bg_mix[i]

    return X_bg_mix

# TODO: OGUZ: is .real correct? Just taking the real port of the signal?
def ir_aug_batch(event_batch, ir_batch):
    n_batch = len(event_batch)
    X_ir_aug = np.zeros((n_batch, event_batch.shape[1]))
    for i in range(n_batch):
        x = event_batch[i]
        x_ir = ir_batch[i]
        # FFT -> multiply -> IFFT
        fftLength = np.maximum(len(x), len(x_ir))
        X = np.fft.fft(x, n=fftLength)
        X_ir = np.fft.fft(x_ir, n=fftLength)
        x_aug = np.fft.ifft(np.multiply(X_ir, X))[0:len(x)].real
        x_aug = max_normalize(x_aug)
        X_ir_aug[i] = x_aug
    return X_ir_aug

def get_fns_seg_list(fns_list=[],
                     segment_mode='all',
                     fs=22050,
                     duration=1,
                     hop=None):
    """
    Opens a file, checks its format and sample rate, and returns a list of segments.

    Parameters:
        fns_list: list of filenames. Only support .wav

    Returns: 
        fns_event_seg_list: list of segments.
        [[filename, seg_idx, offset_min, offset_max], [ ... ] , ... [ ... ]]
            filename is a string
            seg_idx is an integer
            offset_min is 0 or negative integer
            offset_max is 0 or positive integer
    """

    if hop == None:
        hop = duration

    # Get audio info
    n_frames_in_seg = fs * duration
    n_frames_in_hop = fs * hop # 2019 09.05

    fns_event_seg_list = []
    for filename in fns_list:

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

        # Determine number of segments
        n_frames = pt_wav.getnframes()
        if n_frames > n_frames_in_seg:
            n_segs = int((n_frames - n_frames_in_seg + n_frames_in_hop) // n_frames_in_hop)
            assert n_segs > 0
        else:
            n_segs = 1 # load_audio can pad the audio if it is shorter than n_frames_in_seg
        residual_frames = np.max([0, n_frames - ((n_segs - 1) * n_frames_in_hop + n_frames_in_seg)])

        pt_wav.close()

        if segment_mode == 'all': # Load all segments
            # A segment can be offsetted max by n_frames_in_hop to the left or right
            for seg_idx in range(n_segs):
                offset_min, offset_max = int(-1 * n_frames_in_hop), n_frames_in_hop
                if seg_idx == 0:  # first seg
                    offset_min = 0 # no offset to the left
                if seg_idx == (n_segs - 1): # last seg
                    offset_max = residual_frames # Maximal offset to the right is the residual frames
                fns_event_seg_list.append([filename, seg_idx, offset_min, offset_max])
        elif segment_mode == 'first':
            # Load only the first segment
            seg_idx = 0
            offset_min, offset_max = 0, 0
            fns_event_seg_list.append([filename, seg_idx, offset_min, offset_max])
        elif segment_mode == 'random_oneshot':
            # Load only one random segment
            seg_idx = np.random.randint(0, n_segs)
            offset_min, offset_max = n_frames_in_hop, n_frames_in_hop
            if seg_idx == 0:  # first seg
                offset_min = 0
            if seg_idx == (n_segs - 1):  # last seg
                offset_max = residual_frames
            fns_event_seg_list.append([filename, seg_idx, offset_min, offset_max])
        else:
            raise NotImplementedError(segment_mode)

    return fns_event_seg_list

def load_audio(filename=str(),
               seg_start_sec=0.0,
               offset_sec=0.0,
               seg_length_sec=None,
               seg_pad_offset_sec=0.0,
               fs=8000,
               amp_mode='normal'):
    """
        Open file to get file info --> Calulate index range
        --> Load sample by index --> Padding --> Max-Normalize --> Out
    """

    assert (seg_length_sec is None) or (seg_length_sec > 0.0), 'seg_length_sec should be positive'\
                                                'or None (read all the rest of the file).'

    # Only support .wav
    file_ext = os.path.splitext(filename)[1]
    if file_ext != '.wav':
        raise NotImplementedError(file_ext)

    # Open file
    pt_wav = wave.open(filename, 'r')

    # Check sample rate
    _fs = pt_wav.getframerate()
    if fs != _fs:
        raise ValueError('Sample rate should be {} but got {} for {}'.format(str(fs), str(_fs)), filename)

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

    # Max Normalize, random amplitude
    if amp_mode == 'normal':
        pass
    elif amp_mode == 'max_normalize':
        _x_max = np.max(np.abs(x))
        if _x_max != 0:
            x = x / _x_max
    else:
        raise ValueError('amp_mode={}'.format(amp_mode))

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

def load_audio_multi_start(filename=str(),
                           seg_start_sec_list=[],
                           seg_length_sec=float(),
                           fs=8000,
                           amp_mode='normal'):
    """ Load_audio wrapper for loading audio with multiple start indices. """
    # assert(len(seg_start_sec_list)==len(seg_length_sec))
    out = None
    for seg_start_sec in seg_start_sec_list:
        x = load_audio(filename=filename,
                       seg_start_sec=seg_start_sec,
                       seg_length_sec=seg_length_sec,
                       fs=fs,
                       amp_mode=amp_mode)
        x = x.reshape((1, -1))
        if out is None:
            out = x
        else:
            out = np.vstack((out, x))
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
