import os
import random
import warnings
import numpy as np
from scipy.interpolate import interp1d
from tools.generic.sound import Sound


class RawExtractor:
    '''

    '''

    def __init__(self,
                 name: str="raw",
                 augmenter=None,
                 normed=True,
                 dithering=False,
                 random_start=False,
                 length=None,
                 pad_mode="constant"):
        """
        :param type: string, can be dft - Discrete Furrier Transform;
                                    dct - Discrete Cosine Transform;
                                    mfcc - MelFriequency Cepstral Coefficient
        :param config:
        """
        assert pad_mode in ["wrap", "constant"]
        if random_start:
            assert length is not None, "length should be not None if random_start is set"

        self.name = name
        self.augmenter = augmenter
        self.normed = normed
        self.dithering = dithering
        self.pad_mode = pad_mode

        self.random_start = random_start
        self.length = length

    def __call__(self, x: (np.ndarray, Sound), length=None, **kwargs):
        _x = x
        if isinstance(x, Sound):
            x = _x.sound

        x = self.align_feature(x, length=length)
        if self.augmenter is not None:
            x = self.augmenter(x)

        if self.normed:
            x = self.normalize(x)

        if self.dithering:
            x = self.make_dithering(x)

        X = x[None, ...]

        return X

    def align_feature(self, feature, length=None):
        if length is None:
            length = self.length

        if self.random_start:
            start_point = np.random.randint(np.clip(feature.shape[0] - length, 1, None))
            _length = length
        else:
            start_point = 0

        if length is not None:
            _length = length
        else:
            _length = feature.shape[0]

        end_point = start_point + _length
        feature = feature[start_point:end_point]
        if feature.shape[0] < _length:
            left_pad = np.random.randint(_length - feature.shape[0])
            right_pad = (_length - feature.shape[0]) - left_pad
            pad_size = (left_pad, right_pad)
            
            if self.pad_mode == "wrap":
                feature = np.pad(feature, pad_size, mode="wrap")
            elif self.pad_mode == "constant":
                feature = np.pad(feature, pad_size, mode="constant", constant_values=0.0)
            else:
                NotImplementedError("{} not implemented".format(self.pad_mode))

        return feature

    def normalize(self, x):
        x = x - np.mean(x)
        x = x / np.std(x)
        return x

    def make_dithering(self, x):
        return x + 1e-6 * np.random.randn(x.shape)

    @property
    def shape(self):
        length = self.length
        return length, self.dim

    @property
    def dim(self):
        return 1

    def load_config(self, path):
        pass

    def save_config(self, path):
        pass

