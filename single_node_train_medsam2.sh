#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
# make sure the checkpoint is under `MedSAM2/checkpoints/sam2.1_hiera_tiny.pt`
config=configs/sam2.1_hiera_tiny_finetune512.yaml

output_path=/scratch/SAMS/MedSAM2v2/MedSAM2/exp_log/Provolone

# Function to run the training script
CUDA_VISIBLE_DEVICES=0,1 python3 training/train.py \
        -c $config \
        --output-path $output_path \
        --use-cluster 0 \
        --num-gpus 1 \
        --num-nodes 1 
        # --master-addr $MASTER_ADDR \
        # --main-port $MASTER_PORT

echo "training done"


