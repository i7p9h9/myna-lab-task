#!/bin/bash

. ./path.sh

full_path=$(dirname "$(readlink -f "$0")")
cur_path=`pwd`

n_jobs=1

while getopts ":j:" opt; do
  case $opt in
    j) n_jobs="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Build docker image
if [[ "$(docker images -q speech2num:latest 2> /dev/null)" == "" ]]; then
  DOCKER_BUILDKIT=1 docker build -f speech2num.dockerfile -t speech2num .
fi

# Run container
docker run --rm -itd \
    --runtime=nvidia \
    --shm-size 60G \
    -v $full_path/:/speech2num \
    -v $DATASETS_DIR/:/dataset \
    -v $PROCESSED_DIR/:/processed \
    -v $RESULT_DIR/:/result \
    --name speech2num \
    speech2num

# Run train
docker exec -it speech2num /bin/bash -c "/speech2num/train_pipeline.sh -j $n_jobs"

# Kill container
docker kill speech2num

# Go back
cd $cur_path || exit
