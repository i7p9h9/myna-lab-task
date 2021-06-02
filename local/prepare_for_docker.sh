#!/bin/bash
#
# Setup docker on user's machine
#

apt-get update

apt-get install --yes \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"

# Nvidia setup
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  apt-key add -
distribution="ubuntu18.04"
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install docker-ce

usermod -aG docker "$USER"

apt-get install -y nvidia-docker2
pkill -SIGHUP dockerd
