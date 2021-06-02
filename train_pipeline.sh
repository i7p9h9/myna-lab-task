#!/bin/bash

full_path=$(dirname "$(readlink -f "$0")")
cur_path=`pwd`
cd $full_path

n_jobs=1

while getopts ":j:" opt; do
  case $opt in
    j) n_jobs="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

cd ./local || exit
./download_musan.sh /processed
./download_rir.sh /processed

python3 resample.py --source_folder /dataset --dest_folder /processed/train_dataset --samplerate 16000 --job $n_jobs

cd $full_path|| exit
python3 train.py --dataset_folder /processed/train_dataset \
                 --save_folder /result \
                 --musan /processed/musan \
                 --rirs /processed/simulated_rirs_16k \
                 --job $n_jobs \
                 --epoch_steps 10000 \
                 --total_epochs 2

cd $cur_path || exit
