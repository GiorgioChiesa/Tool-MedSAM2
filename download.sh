#!/usr/bin/env bash
# Download MedSAM2 model checkpoints without wget/curl using Python

mkdir -p checkpoints

download_python () {
python3 - << EOF
import urllib.request, os, sys

url = "$1"
dest = "$2"

print(f"Downloading {os.path.basename(dest)}...")
try:
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")
except Exception as e:
    print(f"Failed to download {url}: {e}")
    sys.exit(1)
EOF
}

HF_BASE_URL="https://huggingface.co/wanglab/MedSAM2/resolve/main"

MODEL_FILES=(
    "MedSAM2_2411.pt"
    "MedSAM2_US_Heart.pt"
    "MedSAM2_MRI_LiverLesion.pt"
    "MedSAM2_CTLesion.pt"
    "MedSAM2_latest.pt"
)

for model in "${MODEL_FILES[@]}"; do
    download_python "${HF_BASE_URL}/${model}" "checkpoints/${model}"
done

ETA_BASE_URL="https://huggingface.co/yunyangx/efficient-track-anything/resolve/main"
ETA_MODELS=(
    "efficienttam_s_512x512.pt"
    "efficienttam_ti_512x512.pt"
)

for eta in "${ETA_MODELS[@]}"; do
    download_python "${ETA_BASE_URL}/${eta}" "checkpoints/${eta}"
done

SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2_MODEL="sam2.1_hiera_tiny.pt"

download_python "${SAM2_BASE_URL}/${SAM2_MODEL}" "checkpoints/${SAM2_MODEL}"

echo "All checkpoints downloaded successfully."
