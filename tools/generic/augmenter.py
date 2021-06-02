import numpy as np
from scipy.signal import fftconvolve


class AugmentedSignal:
    def __init__(self, signal, clean_signal=None):
        self.signal = signal
        self.clean_signal = clean_signal


class Mixer(object):
    def __init__(self,
                 snr: (float, tuple),
                 duration: int=160000,
                 ideal_vad: bool=False,
                 noise: bool=True,
                 reverberate_banks: list=None):
        self.snr = snr
        self.noise = noise
        self.duration = duration
        self.ideal_vad = ideal_vad
        self.noise_mixture_power = (-6, 6)
        self.reverberate_banks = reverberate_banks

    def __call__(self, signal: np.ndarray, noise: (np.ndarray, list), snr=None):
        if not self.noise:
            return signal

        signal_length = signal.shape[0]
        if self.ideal_vad:
            length = signal_length
        else:
            length = self.duration if signal_length <= self.duration else signal_length

        for n_noise in range(len(noise)):
            while noise[n_noise].shape[0] < length:
                noise[n_noise] = np.hstack((noise[n_noise], noise[n_noise]))

        if isinstance(noise, np.ndarray):
            noise = [noise]

        cutted_noise = []
        for noise_item in noise:
            part_noise = self.cut(signal=noise_item, length=length)
            cutted_noise.append(part_noise)

        padded_signal, start_point, end_point = self.pad_signal(signal=signal, length=length)
        summed_mixture = self.random_sum(cutted_noise)

        scale = self.__compute_noise_scaling_factor(signal=signal, noise=summed_mixture, snr=snr)
        scale = np.maximum(scale, 1e-3)

        noised_signal = padded_signal + summed_mixture / scale
        # if np.abs(noised_signal).max() > 1.0:
        #     noised_signal /= (0.9 * np.abs(noised_signal).max() + 1e-7)

        # return noised_signal, padded_signal, start_point, end_point
        return noised_signal

    def random_sum(self, x):
        if len(x) == 1:
            return x[0]

        summed = np.zeros_like(x[0])
        for x_item in x:
            power_db = np.random.rand() * np.abs(self.noise_mixture_power[1] - self.noise_mixture_power[0]) + \
                       np.min(self.noise_mixture_power)
            power_linear = self.db_2_pow(power_db)

            summed += np.sqrt(power_linear) * (x_item / (x_item.std() + 1e-7))
        summed /= np.max(np.abs(summed))

        return summed

    @staticmethod
    def pow_2_db(ratio):
        return 10 * np.log10(ratio)

    @staticmethod
    def db_2_pow(db):
        return 10 ** (db / 10)

    @staticmethod
    def db_2_mag(db):
        return 20 ** (db / 20)

    def __compute_noise_scaling_factor(self, signal, noise, snr=None):
        """
        Different lengths for signal and noise allowed: longer noise assumed to be stationary enough,
        and rmse is calculated over the whole signal
        """
        if snr is None:
            if isinstance(self.snr, float):
                snr = self.snr
            elif isinstance(self.snr, tuple):
                assert len(self.snr) == 2
                w = np.max(self.snr) - np.min(self.snr)
                start = np.min(self.snr)
                snr = np.random.rand() * w + start
            else:
                TypeError("snr should be int or tuple, but received: {}".format(type(self.snr)))

        original_sn_rmse_ratio = np.std(signal) / (np.std(noise) + 1e-7)
        target_sn_rmse_ratio = self.db_2_mag(snr)
        # signal_scaling_factor = np.sqrt(target_sn_rmse_ratio / original_sn_rmse_ratio)
        signal_scaling_factor = target_sn_rmse_ratio / (original_sn_rmse_ratio + 1e-7)

        return signal_scaling_factor

    def pad_signal(self, signal, length):
        signal_length = signal.shape[0]
        if length == signal_length:
            return signal, 0, signal.shape[0]
        else:
            assert length > signal_length

        start_length = np.random.randint(length - signal_length)
        end_length = length - signal_length - start_length
        padded_signal = np.pad(signal, (start_length, end_length), mode='constant', constant_values=0)

        return padded_signal, start_length, end_length

    def cut(self, signal, length):
        start_point = np.random.randint(signal.shape[0] - length + 1)
        end_point = start_point + length

        return signal[start_point:end_point]


class Reverberation():
    def __init__(self, reverberate_banks, p=0.5):
        self.reverberate_banks = reverberate_banks
        self.p = p

    def reverberate(self, signal):
        impulse_response = self.reverberate_banks[np.random.randint(len(self.reverberate_banks))]
        impulse_response /= np.std(impulse_response)
        reverb_signal = fftconvolve(signal, (0.8 + np.random.rand() * 0.7) * impulse_response)
        return reverb_signal[:signal.shape[0]]

    def __call__(self, signal, force=False):
        if force or np.random.rand() <= self.p:
            # return self.reverberate(signal) / (0.9 * np.max(np.abs(signal)) + 1e-7)
            return self.reverberate(signal)
        else:
            return signal


class NoiseAppend():
    def __init__(self, noises, p=0.5, snr=(0, 12), min_noises=1, max_noises=1, duration=None):
        assert 1 <= max_noises
        assert min_noises <= max_noises

        self.noises = noises
        self.num_noises = len(self.noises)
        self.max_noises = max_noises
        self.min_noises = min_noises

        self.snr = snr
        self.p = p
        self.mixer = Mixer(snr=snr,
                           ideal_vad=duration is None,
                           noise=True,
                           duration=duration,
                           )

    def __call__(self, signal, force=False):
        if force or np.random.rand() <= self.p:
            n = np.random.randint(self.min_noises, self.max_noises + 1)
            noise = [x.sound for x in [self.noises[_m] for _m in np.random.randint(0, self.num_noises, n)]]
            return self.mixer(signal, noise=noise)
        else:
            return signal, signal


class OneOf():
    def __init__(self, methods: list, p: float=1.0):
        self.methods = methods
        self.p = p

    def __call__(self, x):
        if self.p >= np.random.rand():
            method = np.random.choice(self.methods)
            x = method(x)

        return x


class Copmose():
    def __init__(self, methods: list, p: float=1.0):
        self.methods = methods
        self.p = p

    def __call__(self, x):
        if self.p > np.random.rand():
            for method in self.methods:
                x = method(x)

        return x


if __name__ == "__main__":
    import os
    from tqdm import tqdm
    from idrnd.sound import Sound
    from idrnd.utils.generic import find_files, read_audio_file

    import h5py
    import soundfile as sf


    def read_rir(path):
        with h5py.File(path, 'r') as f:
            data = f['OneArray/FullArray'][:]  # запись в переменную data полного массива
        return data

    samplerate = 16000
    musan_music = find_files("/mnt/ssd/voice/musan/music", pattern="**/*.wav")
    speech_music = find_files("/mnt/ssd/voice/musan/speech", pattern="**/*.wav")
    voices = find_files("/mnt/ssd/voice/vox1_dev/wav", pattern="**/*.wav")
    rirs = read_rir("/mnt/ssd/voice/but_rirs/rir.hdf5")

    dest_path = "/mnt/ssd/voice/vox1_aug/"

    music_sounds = [Sound(path=x, memory_allocated=False) for x in musan_music]
    speech_sounds = [Sound(path=x, memory_allocated=False) for x in speech_music]

    music_aug = NoiseAppend(noises=music_sounds,
                            snr=(0, 6),
                            duration=samplerate * 5,
                            p=1.0,)
    speech_aug = NoiseAppend(noises=speech_sounds,
                             min_noises=3,
                             max_noises=7,
                             snr=(3, 12),
                             duration=samplerate * 5,
                             p=1.0)
    reverb_aug = Reverberation(reverberate_banks=rirs, p=1.0)
    reverb_plus_noise_aug = Copmose([reverb_aug, OneOf([music_aug, speech_aug], p=1.0)], p=1.0)

    aug = OneOf([music_aug, speech_aug, reverb_aug, reverb_plus_noise_aug], p=1.0)

    for n in tqdm(range(30)):
        v = np.random.choice(voices)
        s = read_audio_file(v)[0]
        s_aug = aug(s)

        try:
            sf.write(os.path.join(dest_path, "{}_{}".format(n, os.path.basename(v))), s_aug, samplerate=16000)
        except:
            pass
