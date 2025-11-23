# VisionMamba Installation Guide

## System Requirements
- Linux OS (tested on Compute Canada)
- NVIDIA GPU with CUDA 12.2
- Conda/Miniconda
- GCC 12.3
- CUDA 12.2 toolkit

## Module Loading (Compute Canada specific)
```bash
module purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
```

## Installation Steps

### 1. Create Conda Environment
```bash
# Option A: From exported environment file
conda env create -f environment.yml

# Option B: Manual creation
conda create -n conda_vimsegdet python=3.10.13 -y
conda activate conda_vimsegdet
```

### 2. Install Core Packages
```bash
# Install via conda (preferred for stability)
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install opencv via conda
conda install -c conda-forge opencv libstdcxx-ng libgcc-ng -y

# Install other dependencies
pip install -r pip_requirements.txt --no-user
```

### 3. Install MMCV
```bash
pip install --no-user \
    mmcv-full==1.7.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

### 4. Clone VisionMamba Repository
```bash
git clone https://github.com/f1ibrahim-tmu/VisionMamba
cd VisionMamba
```

### 5. Install Mamba Dependencies (ON COMPUTE NODE)
```bash
# Request compute node
salloc --time=1:00:00 --mem=16G --cpus-per-task=4

# Set environment variables
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PIP_NO_USER=1

# Install causal-conv1d
pip install --no-user --no-build-isolation causal-conv1d==1.1.0

# Install mamba
cd mamba-1p1p1
pip install --no-user --no-build-isolation -e .
cd ..
```

### 6. Install MMSegmentation and MMDetection
```bash
# MMSegmentation
cd seg/mmsegmentation
pip install --no-user --no-build-isolation -e .
cd ../..

# MMDetection
cd det/mmdetection
pip install --no-user --no-build-isolation -e .
cd ../..
```

### 7. Setup Activation Script
Copy `activate_vim.sh` to your home directory and source it for each session.

## Critical Environment Variables
See `environment_vars.txt` for the complete list.

Most important:
- `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`
- `PATH="$CONDA_PREFIX/bin:$PATH"`
- `PYTHONNOUSERSITE=1`
- `PIP_USER=0`
- `PIP_NO_USER=1`

## Verification
```bash
python << 'EOF'
import torch, mmcv, mmseg, mmdet, cv2
from mamba_ssm import Mamba
import causal_conv1d

print(f"PyTorch: {torch.__version__}")
print(f"MMCV: {mmcv.__version__}")
print(f"MMSeg: {mmseg.__version__}")
print(f"MMDet: {mmdet.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"causal_conv1d: {causal_conv1d.__version__}")
print("Mamba: OK")
