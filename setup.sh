#!/usr/bin/env bash
# Setup script for RNA 3D Structure Prediction with RhoFold+ GPU integration
set -e

echo "=== Installing PyTorch with CUDA 12.1 support ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing core dependencies ==="
pip install numba cupy-cuda12x numpy pandas tqdm scikit-learn scipy ripser persim huggingface_hub

echo "=== Installing RhoFold+ ==="
pip install "rhofold @ git+https://github.com/ml4bio/RhoFold.git"

echo "=== Verifying installation ==="
python -c "from rhofold_runner import RhoFoldRunner; r = RhoFoldRunner(); print('RhoFold available:', r.available)"

echo "=== Setup complete ==="
