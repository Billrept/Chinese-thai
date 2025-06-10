#!/bin/bash
# Setup script for Lanta environment following official documentation
# Run this on transfer.lanta.nstda.or.th

echo "Setting up environment for Qwen fine-tuning on Lanta..."

# Load Mamba module (as recommended in Lanta docs)
ml Mamba/23.11.0-0

# Check if environment already exists
if conda env list | grep -q "120_wangmak"; then
    echo "Environment 120_wangmak already exists. Activating..."
    conda activate 120_wangmak
else
    echo "Creating new conda environment..."
    conda create -n 120_wangmak python=3.9 -y
    conda activate 120_wangmak
fi

# Install required packages following Lanta documentation
echo "Installing PyTorch with CUDA 11.8 (recommended for Lanta)..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing Hugging Face packages..."
pip install huggingface_hub
pip install transformers
pip install hf_transfer

echo "Installing additional required packages..."
pip install datasets
pip install accelerate

echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "✅ Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: python download_model.py (to download Qwen model)"
echo "2. Transfer your data and scripts to compute nodes"
echo "3. Submit job: sbatch scripts.sh"
