#!/bin/bash -l
#SBATCH -p compute                 #specify partition
#SBATCH -N 1 -c 32                      #specify number of nodes
#SBATCH -ntasks-per-node=1          #specify number of cpus
#SBATCH -t 1:00:00                 #job time limit <hr:min:sec>
#SBATCH -J my_first_job                #job name
#SBATCH -A tb901117                #specify your account ID

ml Mamba/23.11.0-0
conda activate 120_wangmak

python finetune.py \
  --output_dir ./results \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_strategy steps \
  --save_steps 1000 \
  --logging_steps 100 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --fp16 \
  --report_to none