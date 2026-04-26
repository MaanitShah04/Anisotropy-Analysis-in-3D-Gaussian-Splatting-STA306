#!/bin/bash
# Setup script for Gaussian Splatting (local)
# Run this once from the directory where you want to install everything.

set -e

# 1. Clone the repo
git clone --recursive https://github.com/camenduru/gaussian-splatting
cd gaussian-splatting

# 2. Install Python dependencies
pip install plyfile
pip install submodules/diff-gaussian-rasterization
pip install git+https://github.com/camenduru/simple-knn

# 3. Download and extract the sample dataset
wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip
unzip tandt_db.zip

echo ""
echo "Setup complete. To train, run:"
echo "  python train.py -s gaussian-splatting/tandt/train"
