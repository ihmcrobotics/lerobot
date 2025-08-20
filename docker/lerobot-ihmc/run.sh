#!/bin/bash
# Immediately exit on any errors.
set -e
# Print commands as they are run.
set -o xtrace

cd ../..

docker build --tag ihmcrobotics/lerobot-ihmc:0.2 --file docker/lerobot-ihmc/Dockerfile .

docker run \
    --tty \
    --interactive \
    --rm \
    --dns=1.1.1.1 \
    --env "TERM=xterm-256color" `# Enable color in the terminal` \
    --privileged \
    --gpus all \
    --shm-size=20g \
    --volume /home/$USER/lerobot:/lerobot \
    --volume /home/$USER/datasets:/datasets \
    ihmcrobotics/lerobot-ihmc:0.2 bash
