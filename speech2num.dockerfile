FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 as build

# Fix for keyboard-configuration
ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8

# Common tools
RUN apt-get update -qq && \
    apt-get install --yes --no-install-recommends \
        build-essential \
        wget \
        curl \
        unzip \
        software-properties-common \
        git \
        libopenblas-dev \
        libcap-dev \
        liblapack-dev \
        gfortran \
        libgfortran3 \
        zlib1g-dev \
        automake \
        autoconf \
        sox \
        libtool \
        subversion \
        software-properties-common \
        libasound-dev \
        portaudio19-dev \
        pv \
        tar \
        libsndfile1 \
        util-linux \
        ffmpeg \
        bsdmainutils \ 
        parallel



# Python
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -qq && \
    apt-get install --yes --no-install-recommends \
        python3.6 \
        python3.6-dev \
        python3-setuptools \ 
        python3-pip \
        python3-pyaudio 

# CMake
RUN pip3 install cmake

# dataflow
RUN pip3 install --upgrade git+https://github.com/tensorpack/dataflow.git

# Install PVAD requirements
COPY requirements.txt ./
RUN python3.6 -m pip install --upgrade pip
RUN python3.6 -m pip install setuptools --upgrade
RUN python3.6 -m pip install -r requirements.txt 
RUN pip3 install python-prctl
