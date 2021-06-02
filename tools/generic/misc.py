import os
import json
import time
import numpy as np
import soundfile as sf
import samplerate as libsamplerate
import subprocess as sp

from glob import glob


def find_files(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)


def basename(x: str):
    return os.path.splitext(os.path.basename(x))[0]



def read_audio_file(file_path, samplerate=16000):
    signal, sr = sf.read(file_path)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    ratio = samplerate / sr
    if sr != samplerate:
        converter = 'sinc_best'  # or 'sinc_fastest', ...
        reasample_signal = libsamplerate.resample(signal, ratio, converter)
    # elif sr < target_sr:
    #     assert False, "sample rate must be greater or equal then target_sr"
    else:
        reasample_signal = signal

    return reasample_signal.astype("float32"), sr * ratio


def read_audio_ffmpeg(file, samplerate=16000):
    command = ["ffmpeg",
               '-loglevel', 'error',
               '-i', file,
               '-f', 's16le',
               '-acodec', 'pcm_s16le',
               '-ac', '1',
               '-ar', '{}'.format(samplerate),
               '-'
               ]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)

    signal = np.frombuffer(pipe.stdout.read(), np.int16)

    return signal


class Timer(object):
    def __init__(self):
        self._tic = time.time()

    @property
    def tic(self):
        self._tic = time.time()
        return self._tic

    @property
    def toc(self):
        return time.time() - self._tic

    @property
    def tictoc(self):
        duration = time.time() - self._tic
        _ = self.tic
        return duration


def greedy_decoder(output, blank_label=0, collapse_repeated=True):
    arg_maxes = np.argmax(output, axis=2)
    decodes = []

    for i, args in enumerate(arg_maxes):
        decode = []
        
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(decode)
    return decodes