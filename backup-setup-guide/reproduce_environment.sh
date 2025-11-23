#!/bin/bash
# Automated VisionMamba Environment Reproduction Script

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}VisionMamba Environment Reproduction${NC}"
echo "======================================"
echo ""

# Check if on HPC cluster
if [[ ! -f "/etc/slurm/slurm.conf" ]] && [[ ! -f "/etc/pbs.conf" ]]; then
    echo -e "${YELLOW}Warning: This doesn't appear to be an HPC cluster${NC}"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 1. Load modules (ADJUST FOR YOUR CLUSTER)
echo -e "\n${YELLOW}Step 1: Loading modules...${NC}"
echo "Note: Adjust these module names for your cluster"
module purge
module load StdEnv/2023 || echo "Adjust module names for your cluster"
module load gcc/12.3 || module load gcc
module load cuda/12.2 || module load cuda

echo -e "${GREEN}✓ Modules loaded${NC}"

# 2. Create conda environment
echo -e "\n${YELLOW}Step 2: Creating conda environment...${NC}"
if conda env list | grep -q conda_vimsegdet; then
    echo "Environment already exists. Remove it? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        conda env remove -n conda_vimsegdet -y
    else
        echo "Using existing environment"
    fi
fi

conda env create -f environment.yml
echo -e "${GREEN}✓ Conda environment created${NC}"

# 3. Activate environment
echo -e "\n${YELLOW}Step 3: Activating environment...${NC}"
conda activate conda_vimsegdet
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PIP_NO_USER=1

echo -e "${GREEN}✓ Environment activated${NC}"

# 4. Verify core packages
echo -e "\n${YELLOW}Step 4: Verifying core packages...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo -e "${RED}PyTorch verification failed${NC}"
    exit 1
}

python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')" || {
    echo -e "${RED}MMCV verification failed${NC}"
    exit 1
}

echo -e "${GREEN}✓ Core packages verified${NC}"

# 5. Setup activation script
echo -e "\n${YELLOW}Step 5: Installing activation script...${NC}"
cp activate_vim.sh ~/activate_vim.sh
chmod +x ~/activate_vim.sh
echo -e "${GREEN}✓ Activation script installed at ~/activate_vim.sh${NC}"

# 6. Instructions for compute node compilation
echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}IMPORTANT: Next Steps${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "You must compile causal-conv1d and mamba on a COMPUTE NODE:"
echo ""
echo "1. Request compute node:"
echo "   salloc --time=1:00:00 --mem=16G --cpus-per-task=4"
echo ""
echo "2. Activate environment:"
echo "   source ~/activate_vim.sh"
echo ""
echo "3. Clone VisionMamba:"
echo "   git clone https://github.com/f1ibrahim-tmu/VisionMamba"
echo "   cd VisionMamba"
echo ""
echo "4. Install causal-conv1d:"
echo "   pip install --no-user --no-build-isolation causal-conv1d==1.1.0"
echo ""
echo "5. Install mamba:"
echo "   cd mamba-1p1p1"
echo "   pip install --no-user --no-build-isolation -e ."
echo ""
echo "6. Install MMSeg and MMDet:"
echo "   cd ../seg/mmsegmentation"
echo "   pip install --no-user --no-build-isolation -e ."
echo "   cd ../../det/mmdetection"
echo "   pip install --no-user --no-build-isolation -e ."
echo ""
echo -e "${GREEN}Setup script complete!${NC}"

