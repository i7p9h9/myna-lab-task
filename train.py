import os
import sys
import argparse
import numpy as np
import num2words as num
import torch
import torch.nn.functional as F
from tqdm import tqdm

from tools.trainer import Trainer
from tools.generic.sound import Sound
from tools.generic import misc
from tools.generic import dataset
from tools.generic import augmenter
from tools.generic.generators import Generator
from tools.generic.extractors import RawExtractor
from tools.generic.encoder import NumEncoder
from tools.generic.dataflow_wrapper import dataflow_wrapper
from tools.generic.timer import Timer
from tools.generic.misc import greedy_decoder
from tools.generic.metrics import cer


def get_aug_normal(noise_sounds, music_sounds, noise_speech, rirs, duration=None, p=0.75):
    reverb_aug_025 = augmenter.Reverberation(rirs, p=0.5)
    reverb_aug_100 = augmenter.Reverberation(rirs, p=1.0)
    noise_aug = augmenter.NoiseAppend(noises=noise_sounds,
                                      snr=(-3, 6),
                                      duration=duration,
                                      p=1.0, )
    music_aug = augmenter.NoiseAppend(noises=music_sounds,
                                      snr=(0, 9),
                                      duration=duration,
                                      p=1.0, )
    babble_noise_aug = augmenter.NoiseAppend(noises=noise_speech,
                                             snr=(3, 9),
                                             min_noises=3,
                                             max_noises=7,
                                             duration=duration,
                                             p=1.0, )

    join_noise = augmenter.OneOf([noise_aug, babble_noise_aug, music_aug], p=1.0)
    noise_and_reverb = augmenter.Copmose([reverb_aug_025, join_noise])
    aug = augmenter.OneOf([join_noise, noise_and_reverb], p=p)
    
    return aug


def get_aug_hard(noise_sounds, music_sounds, noise_speech, rirs, duration=None, p=1.0):
    reverb_aug_025 = augmenter.Reverberation(rirs, p=0.5)
    reverb_aug_100 = augmenter.Reverberation(rirs, p=1.0)
    noise_aug = augmenter.NoiseAppend(noises=noise_sounds,
                                      snr=(-6, 3),
                                      duration=duration,
                                      p=1.0, )
    music_aug = augmenter.NoiseAppend(noises=music_sounds,
                                      snr=(-3, 6),
                                      duration=duration,
                                      p=1.0, )
    babble_noise_aug = augmenter.NoiseAppend(noises=noise_speech,
                                             snr=(0, 6),
                                             min_noises=3,
                                             max_noises=7,
                                             duration=duration,
                                             p=1.0, )

    join_noise = augmenter.OneOf([noise_aug, babble_noise_aug, music_aug], p=1.0)
    noise_and_reverb = augmenter.Copmose([reverb_aug_100, join_noise])
    aug = augmenter.OneOf([join_noise, noise_and_reverb], p=1.0)
    
    return aug
    

def get_augmenter(musan_folder: (str), 
                  rirs_folder: (str, list, np.ndarray), 
                  duration=None, 
                  p=0.5,
                  memory_allocated=True,
                  samplerate=16000
                  ):
    """
    :param musan_folder: path to musan with subfolders: "music", "speech", "noise"
    :param stage:
    :param duration: result signal duration is seconds
    :param rirs_folder: rlist of rirs or path to .h5 file with rirs
    :return:
    """
    
    assert os.path.exists(musan_folder), "musan path is not exist"

    simulated_rirs = misc.find_files(os.path.join(rirs_folder, "mediumroom"), "**/*.wav")
    simulated_rirs += misc.find_files(os.path.join(rirs_folder, "smallroom"), "**/*.wav")
    
    rirs = []
    for rir_file in tqdm(simulated_rirs[:], desc="read simulated rir's"):
        rirs.append(misc.read_audio_file(rir_file, samplerate=16000)[0].astype("float32"))

    musan_music = misc.find_files(os.path.join(musan_folder, "music"), pattern='**/*.wav')
    musan_common_noise = misc.find_files(os.path.join(musan_folder, "noise"), pattern='**/*.wav')
    musan_speech_noise = misc.find_files(os.path.join(musan_folder, "speech"), pattern='**/*.wav')
    
    music_sounds = sorted([Sound(path=x, memory_allocated=memory_allocated) for x in tqdm(musan_music, desc="music noise creation")], key=lambda x: x.path)
    noise_sounds = sorted([Sound(path=x, memory_allocated=memory_allocated) for x in tqdm(musan_common_noise, desc="common noise creation")], key=lambda x: x.path)
    noise_speech = sorted([Sound(path=x, memory_allocated=memory_allocated) for x in tqdm(musan_speech_noise, desc="speech noise creation")], key=lambda x: x.path)
    
    normal_aug = get_aug_normal(noise_sounds, music_sounds, noise_speech, rirs)
    hard_aug = get_aug_normal(noise_sounds, music_sounds, noise_speech, rirs)
    
    return normal_aug, hard_aug


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train speech to num')

    parser.add_argument('--dataset_folder',
                        type=str,
                        default=False,
                        required=True,
                        help="path train dataset")
    parser.add_argument('--save_folder',
                        type=str,
                        default=False,
                        required=True,
                        help="path train dataset")
    parser.add_argument('--musan',
                        type=str,
                        default=False,
                        required=True,
                        help="path to musan")
    parser.add_argument('--rirs',
                        type=str,
                        default=False,
                        required=True,
                        help="path to rirs")
    parser.add_argument('--job',
                        type=int,
                        default=1,
                        required=False,
                        help="num jobs")
    parser.add_argument('--epoch_steps',
                        type=int,
                        default=10000,
                        required=False,
                        help="steps per epoch")
    parser.add_argument('--total_epochs',
                        type=int,
                        default=2,
                        required=False,
                        help="num epochs")

    parsed = parser.parse_args()

    dataset_folder = parsed.dataset_folder
    save_folder = parsed.save_folder
    musan_folder = parsed.musan
    rirs_folder = parsed.rirs
    njobs = parsed.job
    epoch_steps = parsed.epoch_steps
    total_epochs = parsed.total_epochs

    ds = dataset.DatasetReader(path=dataset_folder, verbose=True)
    sounds = ds.sounds
    sounds_train = sounds[500:]
    sounds_valid = sounds[:500]
    sounds_unlabeled = ds.unlabeled_sounds

    # create augmentor
    aug_normal, aug_hard = get_augmenter(
        musan_folder=musan_folder,
        rirs_folder=rirs_folder,
        memory_allocated=True,
        p=1
    )

    # create extractors
    etrain = RawExtractor(
        length=16000 * 7,
        augmenter = aug_normal,
        normed=True,
        dithering=False,
        pad_mode="constant"
    )

    etrain_hard = RawExtractor(
        length=16000 * 7,
        augmenter = aug_hard,
        normed=True,
        dithering=False,
        pad_mode="constant"
    )

    etest = RawExtractor(
        length=None,
        augmenter = None,
        normed=True,
        dithering=False,
        pad_mode="constant"
    )

    # create generators
    enc = NumEncoder()
    generator_labeled = Generator(
        extractor=etrain,
        target_converter=enc,
        sounds=sounds_train,
        mode="train",
        to_categorical=False
    )

    generator_unlabeled = Generator(
        extractor=etrain,
        extractor_special=etrain_hard,
        target_converter=enc,
        sounds=sounds_unlabeled,
        mode="train",
        to_categorical=False
    )

    generator_valid = Generator(
        extractor=etest,
        target_converter=enc,
        sounds=sounds_valid,
        mode="valid",
        to_categorical=False
    )
    generator_labeled = dataflow_wrapper(generator_labeled, workers=12, num_prefetch=4)
    generator_unlabeled = dataflow_wrapper(generator_unlabeled, workers=12, num_prefetch=4)

    trainer = Trainer(epoch_steps=epoch_steps,
                      validation_period=1000,
                      save_folder=save_folder)

    unsupervised_epoch_start = 1
    for n_epoch in range(total_epochs):
        if n_epoch < unsupervised_epoch_start:
            unsupervised_interval = np.inf
        else:
            unsupervised_interval = 1
            
        trainer.epoch_step(n_epoch, 
                            generator_labeled=generator_labeled,
                            generator_unlabeled=generator_unlabeled,
                            generator_valid=generator_valid,
                            unsupervised_interval=unsupervised_interval                       
                      )

    trainer.load_state_dict(torch.load(os.path.join(save_folder, "final.torch")))
    torch.save(trainer.half().state_dict(), os.path.join(save_folder, "final-half.torch"))
    