# CAT-K Simulator: Closed-loop Planning with LatentDriver on Waymax

> A high-performance, closed-loop planning simulator that integrates LatentDriver-style policy learning with Waymax simulation and the CAT-K reactive agent stack for non-ego agents.

![LatentDriver Overview](docs/latentdriver.jpg)

This repository combines two powerful lines of work:

- LatentDriver (policy learning and latent world modeling): https://github.com/Sephirex-X/LatentDriver
- CAT-K (controllable reactive traffic agents for simulation): https://github.com/NVlabs/catk

We build on LatentDriver’s encoder–latent-world–decoder paradigm and embed it into a fast Waymax-based simulator that supports both non-reactive (expert) and reactive NPC policies, including CAT-K and IDM. The result is a complete stack to preprocess WOMD, train policies via imitation, and evaluate them in closed loop with rigorous metrics.


## Table of Contents

- Overview
- Installation
- Dataset Preparation (WOMD)
- Training Tutorial
- Simulation and Evaluation
- Model and System Architecture
- Configuration and Hydra Usage
- Repository Structure
- Troubleshooting and Tips
- Acknowledgements and Citation


## Overview

- Goal: Learn and evaluate a planning policy that outputs either waypoints (∆x, ∆y, ∆yaw) or bicycle-control actions, and test it closed-loop in dense multi-agent traffic.
- Simulator: Waymax with minor extensions for metrics and CAT-K integration (included locally in `waymax`).
- Policy backbone: LatentDriver-style encoder (BERT), latent world model (GPT-2), and a multi-path attention decoder that produces a multi-modal Gaussian Mixture Model (GMM) over actions.
- NPC policies: Expert (log replay), IDM, or CAT-K (reactive agents). CAT-K can run single- or multi-GPU.
- Metrics: Arrival rate (at multiple thresholds), collision/overlap rate, off-road rate, and comfort/speed deviations.


## Installation

The stack uses PyTorch, JAX, and TensorFlow (for GPU memory handling). We strongly recommend a fresh Python 3.10 environment. Below is a tested CUDA12 setup; adapt versions to your system.

1) Conda environment

```bash
conda create -n ltdriver python=3.10 -y
conda activate ltdriver
```

2) Core frameworks (match your CUDA/cuDNN versions)

```bash
# TensorFlow (for GPU setup and memory growth)
pip install tensorflow==2.15.0

# JAX (choose the wheel matching your CUDA/cudnn)
pip install jax==0.4.10 jaxlib==0.4.10+cuda12.cudnn88 \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# PyTorch (CUDA 12.1 wheels)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

3) Repository and Python requirements

```bash
git clone https://github.com/Sephirex-X/LatentDriver  # Optional: upstream reference
# This repository is already cloned; install dependencies:
pip install -r requirements.txt

# Build custom C++/CUDA ops and Cython modules
python setup.py install
```

4) Verify GPU availability for all frameworks

```bash
python tools/check_tf_jax_torch.py
# Expect: tf/jax/torch on GPU -> True
```

Notes

- This repo includes a vendored Waymax under `waymax` with modifications for metrics and CAT-K agents; do NOT `pip install waymax` on top. If you already have Waymax installed globally, uninstall it: `pip uninstall waymax`.
- If your CUDA is located in a nonstandard prefix, `setup.py` attempts to bootstrap `CUDA_HOME`. Otherwise, export `CUDA_HOME` prior to building.


## Containerized Setup (Docker)

This section details a fully containerized workflow for environments where system-level reproducibility is paramount. We follow the official Docker APT repository for installation, then run a prebuilt Waymax/CUDA-enabled image. All commands target Ubuntu 22.04+ and require administrative privileges for the initial setup.

### Install Docker (Ubuntu)

1) Base packages and repository prerequisites

```bash
sudo apt update
sudo apt ugrade
sudo apt install -y ca-certificates curl gnupg lsb-release
```

2) Add Docker’s official GPG key

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

3) Register the Docker APT source

```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

4) Install Docker Engine, CLI, and Compose plugin

```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
```

5) Enable and start the daemon; verify service status

```bash
sudo systemctl enable docker
sudo systemctl start docker
sudo systemctl status docker
```

6) Optional: manage Docker without sudo

```bash
sudo usermod -aG docker $USER
newgrp docker
```

7) Sanity check

```bash
docker run hello-world
```

Expected output includes the banner: “Hello from Docker!”.


### Pull and Run the Waymax Environment Image

1) Pull the image

```bash
docker pull hyeobihyeobi/waymax_env:latest
```

2) Create a container with GPU access and X11 forwarding (if visualization is needed)

```bash
docker run -it --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /<path-to-location>:/workspace:rw \ # Path must include both dataset and code
  -e DISPLAY=$DISPLAY \
  --ipc=host --network=host \
  --name waymax \
  hyeobihyeobi/waymax_env:hdd zsh
```

3) Start and enter the container

```bash
docker start waymax
docker exec -it waymax /bin/zsh
```

Important: unlike a typical container entrypoint, ensure the interactive shell is `zsh` (`/bin/zsh`).


### Initial Configuration Inside the Container

1) Navigate to your code directory

```bash
cd <path-to-code>
```

2) Configure project and dataset roots by editing `scripts/set_env.sh`

```bash
vim scripts/set_env.sh
```

Set at minimum the first two lines to match your host mount layout:

```bash
export WAYMO_DATASET_PATH="/workspace/<path-to-WOMD>"  # parent of waymo_open_dataset_motion_v_1_3_1
export ROOT_PATH="/workspace/<path-to-project-root>"   # path to this repository in the container
```

3) Activate your conda environment

```bash
conda activate catk-ltdriver
```

4) Build the customized CUDA/C++ extensions

```bash
python setup.py install
```

5) Verify TensorFlow, JAX, and PyTorch detect GPUs

```bash
python tools/check_tf_jax_torch.py
```


### Optional: CUDA/PyTorch Dependency Alignment

If GPU detection fails (most commonly due to CUDA version mismatches), enforce alignment with the following pinned CUDA 12.1 components for PyTorch:

```bash
python -m pip install --upgrade --force-reinstall \
  nvidia-cuda-cupti-cu12==12.1.105 \
  nvidia-cuda-nvrtc-cu12==12.1.105 \
  nvidia-cuda-runtime-cu12==12.1.105 \
  nvidia-cudnn-cu12==9.1.0.70 \
  nvidia-cufft-cu12==11.0.2.54 \
  nvidia-cusolver-cu12==11.4.5.107 \
  nvidia-cusparse-cu12==12.1.0.106 \
  nvidia-nccl-cu12==2.21.5 \
  nvidia-nvjitlink-cu12==12.1.105
```

Align cuBLAS explicitly if needed:

```bash
python -m pip install --upgrade --force-reinstall nvidia-cublas-cu12==12.1.3.1
```

Finally, confirm compatibility:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('cuda available:', torch.cuda.is_available())"
```


## Dataset Preparation (WOMD)

We use the Waymo Open Motion Dataset (WOMD) tf.Example protos v1.1.x/1.3.x. Organize the raw data as follows:

```
waymo_open_dataset_motion_v_1_1_0/
└── uncompressed
    └── tf_example
        ├── training
        ├── validation
        └── testing
```

1) Export environment paths (edit `scripts/set_env.sh` as needed) and source:

```bash
source scripts/set_env.sh
```

This sets key variables (examples):

- `WOMD_VAL_PATH`/`WOMD_TRAIN_PATH`: tfrecord shards
- `PRE_PROCESS_VAL_PATH`/`PRE_PROCESS_TRAIN_PATH`: where map+route and intention labels are dumped
- `INTENTION_VAL_PATH`/`INTENTION_TRAIN_PATH`: intention labels
- `TRAINING_DATA_PATH`: collected training trajectories

2) Preprocess map, route, and intention labels (validation and training)

```bash
sh scripts/preprocess_data.sh
```

3) Collect training trajectories (expert rollouts with action inversion)

```bash
sh scripts/collecting_training_data.sh
```

After completion, `${TRAINING_DATA_PATH}` should contain:

```
${TRAINING_DATA_PATH}
├── val_preprocessed_path
├── train_preprocessed_path
├── val_intention_label
├── train_intention_label
└── train_data
    ├── data/            # per-scenario pickles
    └── name.txt         # scenario id list
```


## Training Tutorial

We follow LatentDriver’s training objective with a BERT encoder and a GPT-2 latent world model. Training uses PyTorch Lightning.

Prerequisites

- Place a pretrained BERT checkpoint at `checkpoints/pretrained_bert.pth.tar` (we provide one via HuggingFace: https://huggingface.co/Sephirex-x/LatentDriver/tree/main), or set `method.pretrain_enc` to your own.

Command (LatentDriver)

```bash
python train.py method=latentdriver \
  ++data_path=${TRAINING_DATA_PATH} \
  ++exp_name=your_exp_name \
  ++version=latentdriver \
  ++method.max_epochs=10 \
  ++method.train_batch_size=32 \
  ++method.learning_rate=2.0e-4 \
  ++load_num_workers=32 \
  ++method.num_of_decoder=3 \
  ++method.max_len=2 \
  ++method.est_layer=0 \
  ++method.pretrain_enc=checkpoints/pretrained_bert.pth.tar
```

Hydra merges `configs/train.yaml` and `configs/method/latentdriver.yaml`. Important knobs:

- `method.max_len`: history length used by the encoder.
- `method.num_of_decoder`: number of MPA decoder layers (J in the paper/ablation).
- `method.est_layer`: which decoder layer supervises world rollout during training.
- `method.bert_chunk_size`: reduce if BERT causes OOM; the model adaptively halves this on OOM.

Resume training: pass `++ckpt_path=...` to `trainer.fit` via the same CLI override.


## Setup Training (Configuration-first)

Before launching training, ensure that Hydra configuration reflects your hardware and the intended method (LatentDriver).

1) Configure `configs/train.yaml`

- Devices: set `devices` to the GPU IDs available on your host. Example for 4 GPUs:

```yaml
devices: [0, 1, 2, 3]
```

- Method selection: ensure the LatentDriver method is selected in the `defaults` list. Depending on the repository version, either of the following conventions may appear; select the one matching the file in `configs/method/`:

```yaml
defaults:
  - method: latentdriver   # matches configs/method/latentdriver.yaml
```

or

```yaml
defaults:
  - method: LatentDriver   # repository variant using capitalized option
```

If alternative methods (e.g., `planT`, `CarPLAN`) are listed, comment them out when training LatentDriver.

2) Configure `configs/method/latentdriver.yaml`

- Checkpoint path for training vs. simulation:

```yaml
# For training (do not load a model checkpoint to resume weights unless intended):
ckpt_path: null

# For simulation, set a concrete path, e.g.:
# ckpt_path: checkpoints/lantentdriver_t2_J3.ckpt
```

- Pretrained encoder: set `pretrain_enc` to your BERT checkpoint. We recommend placing it under the project root:

```yaml
pretrain_enc: <path-to-project-root>/checkpoints/pretrained_bert.pth.tar
```

These two toggles ensure the encoder is initialized from a strong prior while the main policy starts training from scratch unless you explicitly resume via `ckpt_path`.


## Start Training (Command-line)

1) Conda environment

```bash
conda activate catk-ltdriver
```

2) Environment variables

```bash
source scripts/set_env.sh
```

3) Launch training

```bash
python train.py method=latentdriver \
  ++data_path=${TRAINING_DATA_PATH} \
  ++exp_name=ltdriver-test-training \  # optionally change experiment name
  ++version=latentdriver \
  ++method.max_epochs=10 \
  ++method.train_batch_size=512 \      # tune to your GPU memory budget
  ++method.learning_rate=2.0e-4 \
  ++load_num_workers=32 \
  ++method.num_of_decoder=3 \
  ++method.max_len=2 \
  ++method.est_layer=0 \
  ++method.pretrain_enc=checkpoints/pretrained_bert.pth.tar
```

Notes

- Adjust `++method.train_batch_size` to avoid GPU OOM; mixed precision is enabled by default.
- Ensure `ckpt_path` is `null` in the method config unless explicitly resuming.


## Simulation (Quick Start)

1) Conda environment and variables

```bash
conda activate catk-ltdriver
source scripts/set_env.sh
```

2) Run closed-loop evaluation (CAT-K NPCs)

```bash
python simulate.py method=latentdriver \
  "++waymax_conf.path='${WOMD_VAL_PATH}'" \
  "++data_conf.path_to_processed_map_route='${PRE_PROCESS_VAL_PATH}'" \
  "++metric_conf.intention_label_path='${INTENTION_VAL_PATH}'" \
  "++batch_dims=[1,32]" \
  "++ego_control_setting.npc_policy_type=catk"
```

NPC policy options

- Log replay: `expert`
- IDM: `idm`
- CAT-K reactive rollout: `catk`

Example override:

```bash
"++ego_control_setting.npc_policy_type=expert"  # or idm | catk
```


## Simulation and Evaluation

We evaluate in closed loop on Waymax with various NPC policy settings: {`expert`, `idm`, `catk`}.

Basic LatentDriver evaluation with IDM NPCs

```bash
python simulate.py method=latentdriver \
  "++waymax_conf.path='${WOMD_VAL_PATH}'" \
  "++data_conf.path_to_processed_map_route='${PRE_PROCESS_VAL_PATH}'" \
  "++metric_conf.intention_label_path='${INTENTION_VAL_PATH}'" \
  "++batch_dims=[7,125]" \
  "++method.num_of_decoder=3" \
  "++method.ckpt_path=checkpoints/lantentdriver_t2_J3.ckpt" \
  "++ego_control_setting.npc_policy_type=idm"
```

With CAT-K NPCs (single or multi-GPU)

```bash
python simulate.py method=latentdriver \
  "++waymax_conf.path='${WOMD_VAL_PATH}'" \
  "++data_conf.path_to_processed_map_route='${PRE_PROCESS_VAL_PATH}'" \
  "++metric_conf.intention_label_path='${INTENTION_VAL_PATH}'" \
  "++batch_dims=[N, M]" \
  "++ego_control_setting.npc_policy_type=catk" \
  "++method.ckpt_path=checkpoints/lantentdriver_t2_J3.ckpt"
```

Notes on `batch_dims=[N, M]`

- N = number of local JAX devices; M = per-device batch size. If you request more devices than available, `simulate.py` will automatically clip N to `jax.local_device_count()` and increase M to keep overall batch close, while printing adjustments.
- For IDM, memory usage is higher than expert/CAT-K; consider reducing `M`.

Visualization

- Set `vis` to `'image'` or `'video'` to export to `vis_results/<model_name>/<image|video>/...`. The renderer groups outputs by predicted intention label and overlays trajectories, collisions, and endpoints.

Metrics

- `metric_conf.arrival_thres`: multiple arrival thresholds (e.g., [0.75, 0.8, 0.85, 0.9, 0.95]).
- `metric_conf.intention_label_path`: must point to precomputed intention labels for evaluation.


## Model and System Architecture

High-level pipeline

1) Observation encoding (BERT). Inputs are object-centric tokens from Waymax preprocessed scenes: route, vehicles, roadgraph, and SDC tokens. A type embedding and linear projection produce token features. The CLS token produces a summary embedding; we also support returning the full sequence for token-wise attention.
2) Latent world model (GPT-2). Representation queries attend to encoder tokens to produce a temporally indexed latent state distribution (μ, σ) per timestep. An ordering embedding enforces autoregressive structure. Actions are embedded and interleaved with latent state tokens, forming a sentence for GPT-2.
3) Multi-Path Attention Decoder (MPAD). A stack of cross-attention decoder layers fuses latent tokens into K modal queries. A GMM head predicts per-mode probabilities, elliptical parameters, and yaw. The highest-probability mode yields the control action used by the simulator.

Objective

- Imitation via GMM negative log-likelihood and classification for the selected mode, plus a yaw loss, averaged over decoder layers.
- Optional Kullback–Leibler term between posterior and prior latents in the world model (scaled by 1e-3) when using `enc-LWM-MPP`.

Action spaces

- Waypoints: `(∆x, ∆y, ∆yaw)` with user-specified ranges (see `configs/method/latentdriver.yaml`).
- Bicycle: `(acceleration, steering)` ranges configurable in method configs.

NPC policies

- `expert`: log-replay for all non-ego agents.
- `idm`: classic car-following for reactivity.
- `catk`: learned reactive agents from CAT-K, integrated in `waymax/agents/catk_*`. Multi-GPU dispatch is supported.


## Configuration and Hydra Usage

Hydra composes `configs/train.yaml` or `configs/simulate.yaml` with a method-specific file from `configs/method/`. Examples:

- `configs/method/latentdriver.yaml`: action space, latent world, decoder layers, optimizer/scheduler, and pretrained encoder paths.
- `configs/method/encoder/bert.yaml`: BERT size, gradient checkpointing, and embedding mode.
- `configs/method/world/latent_world_model.yaml`: latent dimension, GPT-2 depth/heads, representation/reconstruction query counts.

Common overrides

```bash
# Change batch devices and size
"++batch_dims=[4,100]"

# Switch NPCs
"++ego_control_setting.npc_policy_type=expert|idm|catk"

# Toggle visualization
"++vis='image'"  # or 'video' or False

# Switch action head
"++method.action_space.dynamic_type=waypoint|bicycle"
```


## Repository Structure

```
configs/                   # Hydra configs for training/simulation/methods
docs/                      # Installation, dataset, and training notes; figures
scripts/                   # set_env.sh, preprocess_data.sh, collecting_training_data.sh
simulator/                 # WaymoEnv wrapper, metrics, rendering, engines
  engines/{base,ltd}_simulator.py
src/
  dataloader/             # WOMD dataset wrapper and normalization
  policy/                 # LatentDriver, baselines, EasyChauffeur
    latentdriver/         # GPT-2 world, MPA decoder, GMM loss
    commons/enc/bert.py   # BERT token encoder
  preprocess/             # map/route dump and training data collection
  utils/                  # I/O, viz, discretizer, JAX initialization
waymax/                   # Vendored Waymax with CAT-K agents and metric tweaks
train.py                  # Training entry point (Hydra)
simulate.py               # Closed-loop evaluation (Hydra)
requirements.txt
setup.py                  # Builds CUDA/C++ ops (sort_vertices) and Cython (crdp)
```


## Troubleshooting and Tips

- BERT OOM during training: reduce `++method.bert_chunk_size` (e.g., 128→64). The code auto-halves on OOM and logs the effective chunk size.
- JAX device mismatch: `simulate.py` adapts `batch_dims` to `jax.local_device_count()` and rebalances per-device batch size.
- IDM memory: IDM NPCs increase memory footprint; reduce per-device batch (`batch_dims[1]`).
- Environment variables: always `source scripts/set_env.sh` before preprocessing/simulation; check paths resolve to existing directories.
- Mixed precision: Lightning uses `precision=16-mixed`; ensure recent CUDA drivers.


## Acknowledgements and Citation

- LatentDriver: https://github.com/Sephirex-X/LatentDriver
- CAT-K: https://github.com/NVlabs/catk
- Waymax: https://github.com/waymo-research/waymax

If you build on this repository in academic work, please cite the corresponding LatentDriver and CAT-K papers and Waymax.

License: See LICENSE (Apache-2.0).
