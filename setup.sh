#!/bin/bash

# Chinese-to-Thai Medical Translation Pipeline Setup Script

set -e

echo "🚀 Setting up Chinese-to-Thai Medical Translation Pipeline..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
required_version="3.8"

if [[ $(echo "$python_version >= $required_version" | bc -l 2>/dev/null || echo "$python_version < $required_version") == 0 ]]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Install PyTorch with appropriate backend
echo "🔥 Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if [[ $(uname -m) == "arm64" ]]; then
        echo "🍎 Installing PyTorch for Apple Silicon (M1/M2)..."
        pip install torch torchvision torchaudio
    else
        echo "🍎 Installing PyTorch for Intel Mac..."
        pip install torch torchvision torchaudio
    fi
elif command -v nvidia-smi &> /dev/null; then
    echo "🎮 CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Check for GPU availability
echo "🖥️ Checking hardware..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')

if hasattr(torch.backends, 'mps'):
    print(f'MPS available: {torch.backends.mps.is_available()}')
"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p results
mkdir -p checkpoints

# Download a small test model to verify everything works
echo "🧪 Testing model loading..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Testing model loading...')
try:
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    print('✅ Tokenizer loaded successfully')
    model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
    print('✅ Model loaded successfully')
    print('🎉 Setup verification completed!')
except Exception as e:
    print(f'❌ Error during verification: {e}')
"

echo "
🎉 Setup completed successfully!

Next steps:
1. Activate the environment: source venv/bin/activate
2. Test the pipeline: python3 pipeline_agent.py --mode baseline
3. Run full training: python3 pipeline_agent.py --mode full

Quick commands:
- Test translation: python3 translation_pipeline.py --mode test
- Run evaluation: python3 evaluate.py --help
- Start fine-tuning: python3 fine_tune.py --help
- Full pipeline: python3 pipeline_agent.py --help

Happy translating! 🌟
"
