import os
import json
import torch

if not os.environ.get("HF_HUB_CACHE"):
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.environ["HF_HUB_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from transformers.tokenization_utils_base import BatchEncoding

def load_jsonl_as_dataset(jsonl_path: str) -> Dataset:
    data = {"prompt": [], "target": []}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data["prompt"].append(obj["prompt"])
            data["target"].append(obj["target"])
    return Dataset.from_dict(data)

def preprocess_function(examples, tokenizer, max_length=1024):
    inputs = []
    for prompt, target in zip(examples["prompt"], examples["target"]):
        # Format as instruction-following format
        formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>"
        inputs.append(formatted_text)
    
    # Tokenize with padding to max_length
    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=max_length,
        padding="max_length",  # Add consistent padding to all examples
        return_attention_mask=True,
    )
    
    # Create labels with same length as input_ids
    labels = []
    for i, (prompt, target) in enumerate(zip(examples["prompt"], examples["target"])):
        # Tokenize prompt separately to know where to mask
        prompt_formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt_tokens = tokenizer(prompt_formatted, add_special_tokens=False)["input_ids"]
        
        # Full sequence length to match input_ids
        full_length = len(model_inputs["input_ids"][i])
        
        # Start with all -100s (ignore all)
        label = [-100] * full_length
        
        # Only include the target tokens in loss calculation
        target_start = len(prompt_tokens)
        target_end = len(model_inputs["input_ids"][i])
        
        # Copy target tokens
        for j in range(target_start, target_end):
            if j < full_length:  # Safety check
                label[j] = model_inputs["input_ids"][i][j]
        
        labels.append(label)
    
    model_inputs["labels"] = labels
    return model_inputs

def main():
    training_args = TrainingArguments(
        output_dir="./qwen_finetuned",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Reduced to save memory
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=1000,  # Reduced evaluation frequency to save memory
        save_strategy="steps",
        save_steps=2000,  # Reduced save frequency
        logging_steps=100,
        learning_rate=5e-6,
        num_train_epochs=3,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,  # Use bfloat16 instead of fp16 for better compatibility
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
        remove_unused_columns=False,
        report_to=None,  # Changed from "none" to None
        save_total_limit=2,  # Reduced to save disk space
        load_best_model_at_end=False,  # Disabled to save memory
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        optim="adafactor",  # Use Adafactor optimizer which is more memory efficient
        ddp_find_unused_parameters=False,  # Optimize for single GPU
        max_grad_norm=1.0,  # Add gradient clipping for stability
    )

    # 1. Load datasets
    train_dataset = load_jsonl_as_dataset("train_processed.jsonl")
    eval_dataset  = load_jsonl_as_dataset("dev_processed.jsonl")

    # 2. Load tokenizer & model
    model_name = "Qwen/Qwen3-8B"
    cache_dir = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface"))
    
    # Try direct path to cached model first
    local_model_path = os.path.join(cache_dir, "models--Qwen--Qwen3-8B")
    
    if os.path.exists(local_model_path):
        print(f"Loading model from local path: {local_model_path}")
        snapshots_dir = os.path.join(local_model_path, "snapshots")
        if os.path.exists(snapshots_dir):
            # Get the first (and likely only) snapshot directory
            snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshot_dirs:
                actual_model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                print(f"Using snapshot path: {actual_model_path}")
                
                tokenizer = AutoTokenizer.from_pretrained(
                    actual_model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
                
                # Add padding token if it doesn't exist
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    actual_model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,  # Match the training dtype
                    device_map={"": 0},  # Force all model to single GPU
                    local_files_only=True,
                    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
                )
            else:
                raise ValueError("No snapshot directories found in cached model")
        else:
            raise ValueError("No snapshots directory found in cached model")
    else:
        # Fallback to original method if local cache not found
        print(f"Local cache not found at {local_model_path}, trying original method...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},  # Force all model to single GPU
            cache_dir=cache_dir,
            local_files_only=False
        )

    # 3. Preprocess / tokenize
    model.resize_token_embeddings(len(tokenizer))  # just in case
    
    # Clear any cached memory
    torch.cuda.empty_cache()
    
    # Enable model parallelism for large models
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    preprocess_fn = lambda examples: preprocess_function(
        examples,
        tokenizer=tokenizer,
        max_length=1024,    # Reduced length to save memory
    )

    train_tok = train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=["prompt", "target"],
    )
    eval_tok = eval_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=["prompt", "target"],
    )

    # 4. Data collator: will pad to longest in batch and create attention masks/labels
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        padding=True
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 6. Start training
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()
