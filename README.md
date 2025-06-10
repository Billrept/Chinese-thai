# Qwen Fine-tuning on Lanta Supercomputer

This project implements fine-tuning of Qwen/Qwen2.5-8B-Instruct for Chinese-to-Thai translation following Lanta supercomputer best practices.

## 🔧 Setup Instructions

### Step 1: Environment Setup (on transfer.lanta.nstda.or.th)

```bash
# 1. SSH to transfer node
ssh username@transfer.lanta.nstda.or.th

# 2. Upload your project
scp -r CT-AI username@transfer.lanta.nstda.or.th:~/

# 3. Setup environment
cd CT-AI
chmod +x setup_environment.sh
./setup_environment.sh
```

### Step 2: Download Model (on transfer node)

```bash
# Download Qwen model to project cache
chmod +x download_model.py
python download_model.py
```

### Step 3: Prepare Training Data

Ensure you have:
- `train_processed.jsonl` - Training data
- `dev_processed.jsonl` - Validation data

Format: `{"prompt": "Chinese text", "target": "Thai translation"}`

### Step 4: Submit Training Job

```bash
# Transfer to compute nodes and submit job
ssh username@lanta.nstda.or.th
cd CT-AI
sbatch scripts.sh
```

## 📁 File Structure

```
CT-AI/
├── finetune.py              # Main fine-tuning script
├── scripts.sh               # SLURM batch script
├── setup_environment.sh     # Environment setup script
├── download_model.py        # Model download script
├── train_processed.jsonl    # Training data
├── dev_processed.jsonl      # Validation data
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## ⚙️ Configuration

### SLURM Configuration (scripts.sh)
- **Partition**: `gpu` (GPU nodes)
- **Resources**: 1 GPU, 8 CPUs, 64GB RAM
- **Time Limit**: 24 hours
- **Account**: `tb901117`

### Training Parameters (finetune.py)
- **Model**: Qwen/Qwen2.5-8B-Instruct
- **Batch Size**: 1 per device (8 effective with gradient accumulation)
- **Learning Rate**: 5e-6
- **Epochs**: 3
- **Sequence Length**: 2048 tokens
- **Precision**: FP16

## 🚀 Key Features Following Lanta Best Practices

1. **Proper Cache Management**: Uses `/disk/home/tb1033/tb901117/.cache/huggingface`
2. **Offline Mode**: Prevents additional downloads during training
3. **GPU Optimization**: Proper SLURM GPU configuration
4. **Environment**: Uses recommended Mamba/23.11.0-0 and PyTorch with CUDA 11.8
5. **Fast Transfer**: Enabled `hf_transfer` for faster downloads

## 📊 Monitoring

```bash
# Check job status
squeue -u $USER

# Monitor training logs
tail -f [job_id].out

# Check errors
tail -f [job_id].err
```

## 🔍 Troubleshooting

### Common Issues:

1. **Module not found**: Ensure environment is properly set up
2. **CUDA errors**: Check GPU allocation in SLURM script
3. **Cache issues**: Verify cache directory permissions
4. **Model download fails**: Run download script on transfer node first

### Environment Variables Set:

```bash
HF_HUB_CACHE="/disk/home/tb1033/tb901117/.cache/huggingface"
HF_HOME="/disk/home/tb1033/tb901117/.cache/huggingface"
HF_DATASETS_CACHE="/disk/home/tb1033/tb901117/.cache/huggingface"
HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=1
HF_HUB_ENABLE_HF_TRANSFER=1
```

## 📈 Expected Output

The fine-tuned model will be saved in `./qwen_finetuned/` with:
- Model weights
- Tokenizer
- Training configuration
- Checkpoints every 1000 steps

## 💡 Tips

1. **Pre-download models** on transfer node to avoid timeouts
2. **Use smaller batch sizes** if you encounter OOM errors
3. **Monitor GPU utilization** with `nvidia-smi`
4. **Save checkpoints frequently** for long training runs
5. **Use wandb** for experiment tracking (optional)

## 📞 Support

For Lanta-specific issues, refer to:
- [Lanta Documentation](https://thaisc.atlassian.net/wiki/spaces/LANTA/)
- [Hugging Face Model Guide](https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/744423439/Model+Hugging+Face)
