# Notes for IHMC workflow

## Training

Running the training, use tmux to allow it to run overnight without needing to leave your computer on:
```
tmux new -s lerobot
// Run train.sh
// To detach, press Ctrl+B, then D

// After many hours or the next,
// to check on the status:
tmux attach -t lerobot
```

## ROS 2 Integration

1. Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main).

I had to make sure the installation had user permissions:
```
# chown -R $USER:$USER /opt/miniconda3/
```

1. Install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

I needed to make sure this was setup (and restart shell after):
```
$ mamba shell init --shell bash --root-prefix=~/.local/share/mamba
```

1. Install ROS 2 Humble via [Robostack](https://robostack.github.io/GettingStarted.html).

Make sure you can run `rviz2`.

1. Setup lerobot dependencies:

```
cd lerobot

mamba install -c conda-forge cmake h5py imageio numba omegaconf opencv \
packaging pymunk pyzmq termcolor pytorch torchvision zarr flask

pip install datasets deepdiff diffusers draccus==0.10.0 einops gdown gymnasium==0.29.1 \
"huggingface-hub[hf-transfer,cli]" jsonlines av pynput rerun-sdk wandb torchcodec==0.2.1

pip install -e . --no-deps
```


