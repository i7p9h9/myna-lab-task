#!/bin/bash

. ./path.sh

full_path=$(dirname "$(readlink -f "$0")")
cur_path=`pwd`


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

echo $model_file
echo $save_file
echo $save_dir
echo $data_dir

model_dir=$(dirname "$(readlink -f "$model_file")")
save_dir=$(dirname "$(readlink -f "$save_file")")
data_dir="$(cd "$(dirname "$data_folder")"; pwd)/$(basename "$data_folder")"

save_file_rel=/result/$(basename "$save_file")
model_file_rel=/model/$(basename "$model_file")
data_file_rel=$data_dir/$(basename "$data_dir")

echo $full_path
echo $model_dir
echo $save_dir
echo $data_dir

if [[ "$(docker images -q speech2num:latest 2> /dev/null)" == "" ]]; then
  DOCKER_BUILDKIT=1 docker build -f speech2num.dockerfile -t speech2num .
fi

# Run container
docker run --rm -itd \
    --runtime=nvidia \
    -v $full_path/:/speech2num \
    -v $model_dir/:/model \
    -v $save_dir/:/result \
    -v $data_dir/:/data \
    --name speech2num \
    speech2num

# Run train
docker exec -it speech2num /bin/bash -c "/speech2num/eval_pipeline.sh -s $save_file_rel -m $model_file_rel -d /data"

# Kill container
docker kill speech2num

# Go back
cd $cur_path || exit
