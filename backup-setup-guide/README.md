# VisionMamba Environment Export

This directory contains everything needed to reproduce the VisionMamba environment on another HPC cluster.

## Files Included

- `environment.yml` - Complete conda environment specification
- `pip_requirements.txt` - All pip packages with exact versions
- `activate_vim.sh` - Environment activation script
- `loaded_modules.txt` - Module versions used
- `environment_vars.txt` - All environment variables
- `INSTALLATION_GUIDE.md` - Step-by-step installation instructions
- `package_versions.txt` - Key package versions for reference
- `reproduce_environment.sh` - Automated reproduction script

## Quick Start

1. Copy this directory to your new HPC cluster
2. Edit `reproduce_environment.sh` to match your cluster's module system
3. Run: `bash reproduce_environment.sh`
4. Follow any prompts and manual steps

## Manual Installation

See `INSTALLATION_GUIDE.md` for detailed step-by-step instructions.

## Verification

After installation, run:
```bash
source activate_vim.sh
python -c "from mamba_ssm import Mamba; print('Success!')"
```

## Notes

- Some packages (causal-conv1d, mamba) MUST be compiled on a compute node
- Adjust module names/versions for your specific HPC cluster
- PyTorch version (2.1.0) is critical for compatibility
