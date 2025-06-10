#!/bin/bash
#SBATCH -p gpu                     # specify GPU partition
#SBATCH -N 1                       # number of nodes
#SBATCH --ntasks-per-node=1        # number of tasks per node
#SBATCH --gpus-per-task=1          # request 1 GPU
#SBATCH --cpus-per-task=8          # number of CPUs per task
#SBATCH --mem=64G                  # memory per node (increased for 8B model)
#SBATCH -t 24:00:00                # job time limit <hr:min:sec>
#SBATCH -A tb901117                # specify your account ID
#SBATCH -J qwen_finetune           # job name
#SBATCH -o %j.out                  # output file
#SBATCH -e %j.err                  # error file

# Load modules (following Lanta documentation)
ml Mamba/23.11.0-0

# Activate conda environment
conda activate 120_wangmak

# Set Hugging Face cache directories (following Lanta best practices)
export HF_HUB_CACHE="/disk/home/tb1033/tb901117/.cache/huggingface"
export HF_HOME="/disk/home/tb1033/tb901117/.cache/huggingface"
export HF_DATASETS_CACHE="/disk/home/tb1033/tb901117/.cache/huggingface"

# Enable offline mode to prevent additional downloads during training
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Enable fast transfer (if model already downloaded)
export HF_HUB_ENABLE_HF_TRANSFER=1

# Set CUDA environment for better performance
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Create cache directory if it doesn't exist
mkdir -p "/disk/home/tb1033/tb901117/.cache/huggingface"

# Run the fine-tuning script
srun python finetune.py 