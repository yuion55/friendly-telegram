#!/usr/bin/env bash
# Setup script for RNA 3D Structure Prediction with RhoFold+ GPU integration
#
# Usage:
#   ./setup.sh              — full online install (default)
#   ./setup.sh download     — download wheels & weights for offline use
#   ./setup.sh offline      — install from pre-downloaded wheels (no internet)
set -e

MODE="${1:-online}"
WHEELS_DIR="${WHEELS_DIR:-/kaggle/working/wheels}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/kaggle/working/rhofold_weights}"

if [ "$MODE" = "download" ]; then
    echo "=== Downloading wheels for offline use ==="
    mkdir -p "$WHEELS_DIR"
    pip download torch torchvision --dest "$WHEELS_DIR" \
        --index-url https://download.pytorch.org/whl/cu121
    pip download numba numpy pandas tqdm scikit-learn scipy ripser persim \
        huggingface_hub --dest "$WHEELS_DIR"
    pip download "rhofold @ git+https://github.com/ml4bio/RhoFold.git" \
        --dest "$WHEELS_DIR" --no-deps 2>/dev/null || \
        pip install --target "$WHEELS_DIR/rhofold_src" \
            "rhofold @ git+https://github.com/ml4bio/RhoFold.git" --no-deps

    echo "=== Downloading RhoFold+ pretrained weights ==="
    mkdir -p "$WEIGHTS_DIR"
    python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='ml4bio/RhoFold',
    filename='rhofold_pretrained.pt',
    cache_dir='$WEIGHTS_DIR',
    local_dir='$WEIGHTS_DIR',
)
print(f'Weights saved to: {path}')
"

    echo "=== Download complete ==="
    echo "Wheels:  $WHEELS_DIR"
    echo "Weights: $WEIGHTS_DIR"
    echo "Upload both directories as Kaggle datasets for offline use."

elif [ "$MODE" = "offline" ]; then
    echo "=== Installing from local wheels (offline) ==="
    WHEEL_SEARCH="${KAGGLE_WHEELS:-$WHEELS_DIR}"
    pip install --no-index --find-links "$WHEEL_SEARCH" \
        torch torchvision numpy numba pandas tqdm scikit-learn scipy \
        ripser persim huggingface_hub 2>/dev/null || true

    # Install RhoFold from local source if available
    if [ -d "$WHEEL_SEARCH/rhofold_src" ]; then
        export PYTHONPATH="$WHEEL_SEARCH/rhofold_src:$PYTHONPATH"
        echo "RhoFold added to PYTHONPATH from $WHEEL_SEARCH/rhofold_src"
    fi

    echo "=== Offline install complete ==="

else
    echo "=== Installing PyTorch with CUDA 12.1 support ==="
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    echo "=== Installing core dependencies ==="
    pip install numba cupy-cuda12x numpy pandas tqdm scikit-learn scipy ripser persim huggingface_hub

    echo "=== Installing RhoFold+ ==="
    pip install "rhofold @ git+https://github.com/ml4bio/RhoFold.git"

    echo "=== Verifying installation ==="
    python -c "from rhofold_runner import RhoFoldRunner; r = RhoFoldRunner(); print('RhoFold available:', r.available)"

    echo "=== Setup complete ==="
fi
