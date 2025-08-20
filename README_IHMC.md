# Notes for IHMC workflow

See the below notes for running training and inference.
Since for inference we require ROS 2 and we don't know how to build all the dependencies for training
when we include ROS 2 (via Robostack), we have two separate conda environments for training and inference.

## Dataset creation

Datasets are generated from IHMC logs using a robot specific app such as `H1RDXSCS2LogVisualizer`.
The result will be a dataset folder in the log directory that contains `meta/info.json` and more.

## Training

Setup your SSH config (`~/.ssh/config`) to allow the simple ssh and rsync commands to work:
```
Host gpu2
    HostName gpu2.ihmc.us
    User <username>
    ForwardAgent yes
```

Make sure you can login:
```
$ ssh gpu2
```

### Uploading your dataset

Make sure there's a `datasets` folder in your user home on gpu2:
```
gpu2:~ $ mkdir -p datasets
```

Back on your machine, use rsync to upload your dataset to the server:
```
dataset $ rsync -avz --exclude='.git' "$PWD" gpu2:~/datasets/
```

### Running training

Copy your locally cloned lerobot repo on the IHMC `ros2rebase` branch to the gpu server:
```
lerobot $ rsync -avz --exclude='.git' "$PWD" gpu2:~
```

Use tmux in order to train overnight without needing to leave a terminal open.

```
$ tmux ls               // list sessions to attach to
$ tmux new -s lerobot   // create a new session
```
To detach, press Ctrl+B, then D.

Build and run the Docker container:
```
~/lerobot/docker/lerobot-ihmc $ ./run.sh
```

Activate the conda environment:
```
$ conda activate lerobot
```

`cd` into a dataset and run the training:
```
/datasets/dataset $ python /lerobot/src/lerobot/scripts/train.py \
--dataset.repo_id=robotlab"$(pwd)" \
--dataset.root="$(pwd)" \
--policy.push_to_hub=false \
--policy.type=diffusion \
--output_dir=outputs/train"$(pwd)"
```

Pass the `--resume=true` to continue from a previous run if needed.

After training has finished, on your computer, copy the trained model back into your dataset folder:
```
dataset $ scp -r gpu2:~/datasets/$(basename "$PWD")/outputs/train/datasets/$(basename "$PWD")/checkpoints/last .
```

## Inference

1. Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main).
```
$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ chmod +x Miniconda3-latest-*.sh
$ ./Miniconda3-latest-*.sh
```
```
$ conda init
```
Close and re-open terminal

2. Install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).
```
$ conda install mamba -c conda-forge
```

I needed to setup the shell and restart it:
```
$ mamba shell init --shell bash --root-prefix=~/.local/share/mamba
```

For some reason on Arch Linux I needed to attain more permissions:
```
# chown -R duncan:duncan /opt/miniconda3/
```

1. Install ROS 2 via [Robostack](https://robostack.github.io/GettingStarted.html).
```
$ mamba create -n lerobot_inference
$ mamba activate lerobot_inference

$ conda config --env --add channels conda-forge
$ conda config --env --add channels robostack-humble
$ mamba install ros-humble-desktop

$ mamba deactivate
```
Make sure you can run `rviz2`.
```
$ mamba activate lerobot_inference
$ rviz2
```

1. Setup lerobot dependencies:

```
$ cd lerobot

$ mamba install -c conda-forge cmake h5py imageio numba omegaconf opencv \
packaging pymunk pyzmq termcolor pytorch torchvision zarr flask

$ pip install datasets deepdiff diffusers draccus==0.10.0 einops gdown gymnasium==0.29.1 \
"huggingface-hub[hf-transfer,cli]" jsonlines av pynput rerun-sdk wandb torchcodec==0.2.1 pyserial

$ pip install "numpy<2.3"

$ pip install -e . --no-deps
```


Set the domain ID:
```
$ export ROS_DOMAIN_ID=#
```

Run the policy:
```
$ python src/lerobot/robots/ihmc_ros_robot/ihmc_ros_robot.py \
--trained_policy=/path/to/dataset/last/pretrained_model/
```

