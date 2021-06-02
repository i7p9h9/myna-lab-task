#!/bin/bash

full_path=$(dirname "$(readlink -f "$0")")
cur_path=`pwd`
cd $full_path


# for i in "$@"
# do
# case $i in
#     -s|--save-file=*)
#     save_file="${i#*=}"
#     shift
#     ;;
#     -m|--model=*)
#     model_file="${i#*=}"
#     shift
#     ;;
#     -d|--data=*)
#     data_folder="${i#*=}"
#     shift
#     ;;
# esac
# done

while getopts ":s:m:d:" opt; do
  case $opt in
    s) save_file="$OPTARG"
    ;;
    m) model_file="$OPTARG"
    ;;
    d) data_folder="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

python3 eval.py --dataset_folder $data_folder \
                --save_file $save_file \
                --model $model_file

cd $cur_path || exit
