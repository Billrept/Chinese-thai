#!/bin/bash
#SBATCH -p gpu                     # specify GPU partition
#SBATCH -N 1                       # number of nodes
#SBATCH --gpus-per-task=1         # number of CPU cores per task
#SBATCH --ntasks-per-node=4
#SBATCH -t 24:00:00                # job time limit <hr:min:sec>
#SBATCH -A tb901117                # specify your account ID
#SBATCH -J qwen_finetune           # job name
#SBATCH -o %j.out                  # output file
#SBATCH -e %j.err                  # error file

# Load modules
module load Mamba/23.11.0-0

conda activate 120_wangmak

# Run the fine-tuning script
srun python finetune.py 