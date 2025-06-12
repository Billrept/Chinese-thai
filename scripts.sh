#!/bin/bash
#SBATCH -p gpu                     # specify GPU partition
#SBATCH -N 1                       # number of nodes
#SBATCH --ntasks-per-node=4        # number of tasks per node
#SBATCH --gpus-per-task=4          # request 4 GPUs
#SBATCH -t 24:00:00                # job time limit <hr:min:sec>
#SBATCH -A tb901117                # specify your account ID
#SBATCH -J qwen_finetune           # job name
#SBATCH -o %j.out                  # output file
#SBATCH -e %j.err                  # error file

# Load modules (following Lanta documentation)
ml Mamba/23.11.0-0

conda activate test

export HF_HUB_CACHE="~/.cache/huggingface"
export HF_HOME="~/.cache/huggingface"
export HF_DATASETS_CACHE="~/.cache/huggingface"

# Enable offline mode to prevent additional downloads during training
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Enable fast transfer (if model already downloaded)
export HF_HUB_ENABLE_HF_TRANSFER=1

# Set CUDA environment for better performance
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the fine-tuning script
srun python3 finetune.py 