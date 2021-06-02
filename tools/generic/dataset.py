import os
from tqdm import tqdm
from tools.generic import sound
import collections


class DatasetReader:
    def __init__(self, path, memory_allocated=False, samplerate=16000, verbose=False):
        self.path = path
        self.verbose = verbose
        self.memory_allocated = memory_allocated
        self.samplerate = samplerate
        self.meta = self.read_csv(os.path.join(self.path, "train.csv"))
        self._sounds = None
        self._unlabeled_sounds = None
        
    @property
    def sounds(self):
        if self._sounds is None:
            sounds = []
            for meta_item in tqdm(self.meta, desc="create labeled sounds", disable=not self.verbose):
                if not meta_item.number:
                    continue
                sounds.append(sound.Sound(
                    path = os.path.join(self.path, meta_item.path),
                    sound_id = int(float(meta_item.number)) if meta_item.number else None,
                    samplerate = self.samplerate,
                    memory_allocated = self.memory_allocated,
                    extra_meta = meta_item
                ))
            self._sounds = sounds

        return self._sounds
    
    @property
    def unlabeled_sounds(self):
        """
        surprise - there are no unlabeled files
        """
        if self._unlabeled_sounds is None:
            sounds = []
            for meta_item in tqdm(self.meta, desc="create labeled sounds", disable=not self.verbose):
                if meta_item.number:
                    continue
                sounds.append(sound.Sound(
                    path = os.path.join(self.path, meta_item.path),
                    sound_id = int(float(meta_item.number)) if meta_item.number else None,
                    samplerate = self.samplerate,
                    memory_allocated = self.memory_allocated,
                    extra_meta = meta_item
                ))

            self._unlabeled_sounds = sounds
            
        return self._unlabeled_sounds
        
    @staticmethod
    def read_csv(file):
        files_arr = []
        with open(file, 'r', encoding='utf-8-sig', newline='') as f:
            header = f.readline().strip().split(",")
            print(header)
            meta = collections.namedtuple("meta", header)

            for line in f:
                line_meta = line.strip().split(",")
                files_arr.append(meta(**{name: val.strip() for name, val in zip(header, line_meta)}))

        return files_arr 
