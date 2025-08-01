FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    zip \
    unzip \
    wget \
    git \
    cmake \
    apt-transport-https \
    iputils-ping \
    ca-certificates \
    software-properties-common \
    python3-opencv \
    iproute2 \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
 && /bin/bash ~/miniconda.sh -b -p $CONDA_DIR \
 && rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda init bash

RUN conda update -n base conda -y \
 && conda clean --all --yes

RUN conda create -y -n lerobot python=3.10

RUN conda run -n lerobot conda install ffmpeg -c conda-forge

RUN git clone https://github.com/ihmcrobotics/lerobot.git
WORKDIR lerobot

RUN conda run -n lerobot pip install -e ".[aloha, pusht]"

RUN echo 'alias cal="conda activate lerobot"' >> ~/.bashrc

WORKDIR /datasets
