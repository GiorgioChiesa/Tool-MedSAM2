#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
# make sure the checkpoint is under `MedSAM2/checkpoints/sam2.1_hiera_tiny.pt`
config=configs/sam2.1_hiera_tiny_finetune512.yaml
output_path=/scratch/SAMS/MedSAM2v2/MedSAM2/exp_log/MedSAM2_Sint

# Function to run the training script
CUDA_VISIBLE_DEVICES=0,1 python training/train.py \
        -c $config \
        --output-path $output_path \
        --use-cluster 0 \
        --num-gpus 2 \
        --num-nodes 1 
        # --master-addr $MASTER_ADDR \
        # --main-port $MASTER_PORT

echo "training done"


