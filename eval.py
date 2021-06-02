import os
import sys
import csv
import argparse
import numpy as np
import torch
from tqdm import tqdm

from tools.trainer import Trainer
from tools.generic import misc
from tools.generic.encoder import NumEncoder
from tools.generic.misc import greedy_decoder


def read_file(wav_file: str):
    x = misc.read_audio_file(wav_file, samplerate=16000)[0]
    x -= np.mean(x)
    x /= np.std(x)

    return x[None, :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train speech to num')

    parser.add_argument('--dataset_folder',
                        type=str,
                        default=False,
                        required=True,
                        help="path train dataset")
    parser.add_argument('--save_file',
                        type=str,
                        default=False,
                        required=True,
                        help="file, where result will be saved")
    parser.add_argument('--model',
                        type=str,
                        default=False,
                        required=True,
                        help="path to folder with final model")

    parsed = parser.parse_args()

    dataset_folder = parsed.dataset_folder
    save_file = parsed.save_file
    model_file = parsed.model

    print("asdf: {}".format(os.listdir(dataset_folder)))
    print("1: {}".format(model_file))
    print("2: {}".format(save_file))
    print("3: {}".format(dataset_folder))

    encoder = NumEncoder()
    trainer = Trainer(epoch_steps=1,
                      validation_period=1000,
                      save_folder=None)
    trainer.load_state_dict(torch.load(model_file))

    files = misc.find_files(dataset_folder, pattern="**/*.wav")

    result = dict()
    for f in tqdm(files, desc="process files"):
        x = read_file(f)
        p = trainer.predict(x) 
        decoded_seq = greedy_decoder(p.cpu().numpy(), blank_label=0)
        seq2num = encoder((np.asarray(decoded_seq).squeeze() - 1).tolist())
        
        key = f[len(dataset_folder):]
        result[key] = seq2num
    
    with open(os.path.join(save_file), 'w') as f:
        w = csv.writer(f, result.keys())
        w.writerows(result.items())

    torch.save(trainer.half().state_dict(), "final-half.torch")
