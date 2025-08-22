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

### Install mamba

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

### Create mamba environment with ROS 2

```
$ mamba env create -f ros2env.yaml
$ mamba activate lerobot_ros2
```
Make sure you can run `rviz2`.
```
$ rviz2
```

Setup the lerobot repo, skipping the dependencies:
```
lerobot $ pip install -e . --no-deps
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

