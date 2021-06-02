import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class Generator(Dataset):
    def __init__(self,
                 extractor,
                 target_converter,
                 sounds: (list, np.ndarray),
                 extractor_special=None,
                 batch_size=64,
                 to_categorical: bool=False,
                 is_balance_sample: bool=True,
                 length=None,
                 mode="train"
                 ):
        assert mode in ["train", "test", "valid"]

        self._batch_size = batch_size
        self.extractor = extractor
        self.extractor_special = extractor_special
        self.target_converter = target_converter
        self.sounds = sounds
        self.mode = mode

        self.to_categorical = to_categorical
        self.is_balance_sample = is_balance_sample

    @property
    def shape(self):
        return self.extractor.shape

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def dim(self):
        return self.extractor.dim

    def __getitem__(self, indices):
        if self.mode == "train":
            return self.__train_call(item=indices)
        elif self.mode == "test":
            return self.__test_call(item=indices)
        elif self.mode == "valid":
            return self.__valid_call(item=indices)

    def __len__(self):
        if self.mode == "train":
            return len(self.sounds) // self._batch_size

        return len(self.sounds)

    def __iter__(self):
        return self

    def __next__(self):
        return self[0]

    def __train_call(self, item, seed=None, extractor_special=None, repeat=2):
        assert repeat > 0

        sounds_in_batch = random.choices(self.sounds, k=self.batch_size)
        random.seed(None)

        X = []
        Y = []
        y_length = []

        for sound_item in sounds_in_batch:
            X.append(self.extractor(sound_item))
            if sound_item.id:
                y = self.target_converter(sound_item.id)
            else:
                y = [0]
            Y.append(np.array(y) + 1)
            y_length.append(len(y))

        if self.extractor_special is not None:        
            for sound_item in sounds_in_batch:
                for _ in range(repeat):
                    X.append(self.extractor_special(sound_item))
                    if sound_item.id:
                        y = self.target_converter(sound_item.id)
                    else:
                        y = [0]
                    Y.append(np.array(y) + 1)
                    y_length.append(len(y))

        Y = np.vstack([np.pad(_y, (0, max(y_length) - len(_y)), mode="constant", constant_values=0) for _y in Y])
        # Y = np.vstack([np.pad(_y, (0, max(y_length) - len(_y)), mode="edge") for _y in Y])
        if self.to_categorical:
            Y = Y == np.arange(len(self.target_converter))[None, :]
            Y = torch.FloatTensor(Y.astype("float32"))
        else:
            Y = torch.LongTensor(Y)

        y_length = np.hstack(y_length)

        X = np.vstack(X)
        X = torch.FloatTensor(X.astype("float32"))

        return X, \
               Y, \
               torch.LongTensor(y_length.astype("int"))

    def __test_call(self, item):
        return self.extractor(self.sounds[item])

    def __valid_call(self, item):
        sound = self.sounds[item]
        return self.extractor(sound), sound.id
