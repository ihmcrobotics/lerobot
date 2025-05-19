#!/bin/bash
# Immediately exit on any errors.
set -e
# Print commands as they are run.
set -o xtrace

docker build --build-arg HOST_USER=$(whoami) \
             --build-arg HOST_GID=$(id -g) \
             --build-arg HOST_GROUP=users \
             --tag ihmcrobotics/lerobot-ihmc:0.1 .
