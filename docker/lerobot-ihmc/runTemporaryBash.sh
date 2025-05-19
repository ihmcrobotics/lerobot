#!/bin/bash
# Immediately exit on any errors.
set -e
# Print commands as they are run.
set -o xtrace

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
    ihmcrobotics/lerobot-ihmc:0.1 bash
