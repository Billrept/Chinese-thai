"""
Configuration file for Chinese-to-Thai Medical Translation Pipeline
"""

import platform
import torch

# Model configurations for different environments
MODEL_CONFIGS = {
    # For Mac testing (8B models)
    "mac_test": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",  # Better model for translation
        "max_length": 1024,
        "batch_size": 1,
        "temperature": 0.1,
        "max_new_tokens": 100
    },
    
    "mac_production": {
        "model_name": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "max_length": 2048,
        "batch_size": 1,
        "temperature": 0.3,
        "max_new_tokens": 200
    },
    
    # For supercomputer (32B models)
    "supercomputer": {
        "model_name": "deepseek-ai/deepseek-llm-67b-chat",  # Large model
        "max_length": 4096,
        "batch_size": 4,
        "temperature": 0.2,
        "max_new_tokens": 250
    },
    
    # Alternative large models
    "qwen_large": {
        "model_name": "Qwen/Qwen2-72B-Instruct",
        "max_length": 4096,
        "batch_size": 2,
        "temperature": 0.25,
        "max_new_tokens": 200
    },
    
    "llama_large": {
        "model_name": "meta-llama/Llama-2-70b-chat-hf",
        "max_length": 4096,
        "batch_size": 2,
        "temperature": 0.3,
        "max_new_tokens": 200
    }
}

# Medical-specific prompts and templates
MEDICAL_PROMPTS = {
    "basic": """Translate this Chinese medical conversation to Thai. Maintain medical accuracy and natural flow.

Chinese: {source}
Thai:""",
    
    "contextual": """You are a professional medical translator. Translate the following Chinese medical conversation to Thai.

Context: {context}
Chinese: {source}
Thai Translation:""",
    
    "few_shot": """You are a medical translator. Here are examples:

Chinese: 你好，有什么症状吗？
Thai: สวัสดี มีอาการอะไรบ้างครับ

Chinese: 我头疼得厉害
Thai: ฉันปวดหัวมาก

Now translate:
Chinese: {source}
Thai:""",
    
    "detailed": """You are a professional medical translator specializing in Chinese-to-Thai translation of medical conversations between doctors and patients. 

Guidelines:
- Maintain medical accuracy and proper terminology
- Preserve conversational tone and politeness levels
- Use appropriate Thai medical vocabulary
- Keep the meaning precise and clear

Medical Context: {context}

Chinese Text: {source}

Thai Translation:"""
}

# Evaluation settings
EVALUATION_CONFIG = {
    "bleu_settings": {
        "lowercase": True,
        "tokenize": "char",  # Character-level for Thai
        "smooth_method": "exp",
        "smooth_value": 0.0
    },
    
    "metrics": ["bleu", "chrf", "ter"],  # Multiple metrics for comprehensive evaluation
    
    # Post-processing rules
    "post_processing": {
        "remove_artifacts": ["</s>", "<s>", "<|end|>", "<|start|>", "Human:", "Assistant:"],
        "clean_patterns": [
            r'[A-Za-z]{5,}',  # Remove long English words
            r'\s+',           # Multiple spaces
            r'^\s*[\[\(].*?[\]\)]\s*',  # Remove leading bracketed text
        ]
    }
}

# Training and fine-tuning configurations
TRAINING_CONFIG = {
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "save_steps": 1000,
    "eval_steps": 500,
    "logging_steps": 100,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 4,
    "dataloader_num_workers": 4,
    "fp16": True,  # Enable mixed precision
    "gradient_checkpointing": True,  # Save memory
}

# Hardware-specific optimizations
HARDWARE_CONFIG = {
    "mac_m1": {
        "device": "mps",
        "fp16": False,  # MPS doesn't fully support fp16
        "gradient_checkpointing": True,
        "max_memory_per_gpu": None
    },
    
    "gpu_server": {
        "device": "cuda",
        "fp16": True,
        "gradient_checkpointing": True,
        "max_memory_per_gpu": "80GB"  # For A100
    }
}

# Data processing settings
DATA_CONFIG = {
    "max_source_length": 512,
    "max_target_length": 512,
    "preprocessing": {
        "normalize_chinese": True,
        "normalize_thai": True,
        "remove_empty": True,
        "min_length": 5,
        "max_length": 1000
    }
}

def detect_environment():
    """
    Detect the current environment and return appropriate model configuration key
    """
    system = platform.system()
    machine = platform.machine()
    
    if system == "Darwin":  # macOS
        if machine == "arm64":  # M1/M2 Mac
            return "mac_production"
        else:  # Intel Mac
            return "mac_test"
    elif torch.cuda.is_available():
        return "supercomputer"
    else:
        return "mac_test"  # Default fallback

# Add generation config to model configs
for config_name in MODEL_CONFIGS:
    if "generation_config" not in MODEL_CONFIGS[config_name]:
        MODEL_CONFIGS[config_name]["generation_config"] = {
            "do_sample": True,
            "temperature": MODEL_CONFIGS[config_name].get("temperature", 0.3),
            "max_new_tokens": MODEL_CONFIGS[config_name].get("max_new_tokens", 200),
            "pad_token_id": None,  # Will be set during initialization
            "eos_token_id": None,  # Will be set during initialization
            "repetition_penalty": 1.1,
            "batch_size": MODEL_CONFIGS[config_name].get("batch_size", 1)
        }
