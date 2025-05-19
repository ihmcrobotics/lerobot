FROM ihmcrobotics/nvidia:0.6

USER root

RUN apt-get --quiet 2 --yes update \
 && apt-get --quiet 2 --yes install \
    cmake \
    build-essential \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev \
    > /dev/null \
 && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
 && /bin/bash ~/miniconda.sh -b -p $CONDA_DIR \
 && rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

ARG HOST_USER
ARG HOST_GID
ARG HOST_GROUP

# Setup a robotlab user as the development user, to avoid using root.
RUN adduser --home /home/$HOST_USER --gecos "Rosie Robot,1117,1234567,2345678" --ingroup $HOST_GROUP --disabled-password $HOST_USER \
 && chown -R $HOST_USER /home/$HOST_USER \
 && adduser $HOST_USER sudo \
 && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $HOST_USER
WORKDIR /home/$HOST_USER

# Make sure the .config folder exists and is owned by the host user.
# This is useful for later installed apps.
RUN mkdir -p /home/$HOST_USER/.config
RUN chown -R $HOST_USER:$HOST_GROUP /home/$HOST_USER/.config

USER $HOST_USER

RUN conda init bash

RUN conda update -n base conda -y \
 && conda clean --all --yes

RUN conda create -y -n lerobot python=3.10

RUN conda run -n lerobot conda install ffmpeg -c conda-forge

RUN git clone https://github.com/ihmcrobotics/lerobot.git
WORKDIR lerobot

RUN conda run -n lerobot pip install -e ".[aloha, pusht]"

RUN echo 'alias cal="conda activate lerobot"' >> ~/.bashrc
