#!/usr/bin/env python3
"""
Fine-tuning script for Chinese-to-Thai Medical Translation
Supports LoRA/QLoRA for efficient training on limited hardware
"""

import os
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
import wandb
from config import MODEL_CONFIGS, TRAINING_CONFIG, DATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="deepseek-ai/deepseek-coder-6.7b-instruct")
    use_lora: bool = field(default=True)
    use_qlora: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    target_modules: Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj")

@dataclass
class DataArguments:
    train_file: str = field(default="2025-mt_public_train-jsonl.jsonl")
    eval_file: str = field(default="2025-mt_public_dev-jsonl.jsonl")
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=1000)

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default="./fine_tuned_model")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=500)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=1000)
    eval_steps: int = field(default=500)
    evaluation_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    report_to: str = field(default="wandb")
    fp16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)

class MedicalTranslationDataset:
    def __init__(self, tokenizer, data_args: DataArguments):
        self.tokenizer = tokenizer
        self.data_args = data_args
        
    def load_jsonl_data(self, file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load data from JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        if max_samples:
            data = data[:max_samples]
            
        return data
    
    def create_training_prompt(self, source: str, target: str, context: str = "") -> str:
        """Create training prompt in instruction format"""
        prompt = f"""Below is an instruction that describes a medical translation task. Write a response that appropriately completes the request.

### Instruction:
Translate the following Chinese medical conversation to Thai. Maintain medical accuracy and natural conversational flow.

Context: {context[:300] if context else "Medical conversation"}

### Input:
{source}

### Response:
{target}"""
        
        return prompt
    
    def prepare_dataset(self, file_path: str, max_samples: Optional[int] = None) -> Dataset:
        """Prepare dataset for training"""
        logger.info(f"Loading data from {file_path}")
        data = self.load_jsonl_data(file_path, max_samples)
        
        # Create training examples
        training_examples = []
        for item in data:
            source = item['source']
            target = item['translation']
            context = item.get('context', '')
            
            # Create instruction-formatted prompt
            prompt = self.create_training_prompt(source, target, context)
            training_examples.append({"text": prompt})
        
        logger.info(f"Created {len(training_examples)} training examples")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(training_examples)
        return dataset

def setup_model_and_tokenizer(model_args: ModelArguments):
    """Setup model and tokenizer with optional quantization and LoRA"""
    
    # Configure quantization if using QLoRA
    bnb_config = None
    if model_args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"  # Important for training
    )
    
    # Add pad token if doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if not model_args.use_qlora else None,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    # Prepare model for training
    if model_args.use_qlora:
        model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    if model_args.use_lora or model_args.use_qlora:
        target_modules = model_args.target_modules.split(",") if model_args.target_modules else None
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default="2025-mt_public_train-jsonl.jsonl")
    parser.add_argument("--eval_file", type=str, default="2025-mt_public_dev-jsonl.jsonl")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # Environment
    parser.add_argument("--wandb_project", type=str, default="chinese-thai-medical-translation")
    parser.add_argument("--run_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize wandb
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"medical-translation-{args.model_name_or_path.split('/')[-1]}"
        )
    
    # Create argument objects
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        use_lora=args.use_lora,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    data_args = DataArguments(
        train_file=args.train_file,
        eval_file=args.eval_file,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=100,
        save_steps=1000,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb_project else "none"
    )
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    dataset_processor = MedicalTranslationDataset(tokenizer, data_args)
    
    train_dataset = dataset_processor.prepare_dataset(
        data_args.train_file, 
        data_args.max_train_samples
    )
    
    eval_dataset = dataset_processor.prepare_dataset(
        data_args.eval_file,
        data_args.max_eval_samples
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=False  # Don't pack sequences for this task
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Save training metrics
    if trainer.state.log_history:
        with open(os.path.join(training_args.output_dir, "training_metrics.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
