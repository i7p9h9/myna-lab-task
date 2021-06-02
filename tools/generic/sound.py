from tools.generic.misc import read_audio_file


class Sound(object):
    def __init__(self,
                 path: str,
                 dataset: str=None,
                 sound_id: str=None,
                 file_id: str=None,
                 memory_allocated=False,
                 samplerate=16000,
                 extra_meta=None
                 ):
        self.path = path
        self.id = sound_id
        self.file_id = file_id
        self.samplerate = samplerate
        self.memory_allocated = memory_allocated
        self.dataset = dataset
        self.extra_meta = extra_meta

        self.__sound = self.__sound_handler()

    def get_sound(self):
        if self.memory_allocated:
            s = self.__sound
        else:
            s = read_audio_file(self.path, samplerate=self.samplerate)[0].astype("float32")
        return s

    def __sound_handler(self):
        if self.memory_allocated:
            s = read_audio_file(self.path, samplerate=self.samplerate)[0].astype("float32")
        else:
            s = self.path
        return s

    @property
    def sound(self):
        return self.get_sound()
