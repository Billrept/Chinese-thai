#!/usr/bin/env python3

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set Hugging Face cache directories (following Lanta documentation)
os.environ["HF_HUB_CACHE"] = "/disk/home/tb1033/.cache/huggingface"
os.environ["HF_HOME"] = "/disk/home/tb1033/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/disk/home/tb1033/.cache/huggingface"

# Enable fast transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def download_qwen_model():
    """Download Qwen model and tokenizer"""
    model_name = "Qwen/Qwen2.5-8B-Instruct"
    
    print(f"Downloading {model_name}...")
    print(f"Cache directory: {os.environ['HF_HUB_CACHE']}")
    
    # Create cache directory
    os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir=os.environ["HF_HUB_CACHE"]
        )
        print("✓ Tokenizer downloaded successfully")
        
        # Download model
        print("Downloading model (this will take a while for 8B model)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=os.environ["HF_HUB_CACHE"]
        )
        print("✓ Model downloaded successfully")
        
        print(f"\nModel and tokenizer saved to: {os.environ['HF_HUB_CACHE']}")
        print("You can now run your fine-tuning job on compute nodes.")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Qwen Model Download Script for Lanta")
    print("=" * 60)
    
    success = download_qwen_model()
    
    if success:
        print("\n✅ Download completed successfully!")
        print("\nNext steps:")
        print("1. Transfer your training data to the compute nodes")
        print("2. Submit your SLURM job: sbatch scripts.sh")
    else:
        print("\n❌ Download failed. Please check the error messages above.")
