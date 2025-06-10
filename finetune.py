import os
import json
import torch

if not os.environ.get("HF_HUB_CACHE"):
    cache_dir = "~/disk/home/tb1033/.cache/huggingface"
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

def preprocess_function(examples, tokenizer, max_length=2048):
    inputs = []
    for prompt, target in zip(examples["prompt"], examples["target"]):
        # Format as instruction-following format
        formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>"
        inputs.append(formatted_text)
    
    # Tokenize the concatenated text
    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=max_length,
        padding=False,  # We'll pad in the data collator
        return_attention_mask=True,
    )
    
    # For causal LM, labels are the same as input_ids
    # We'll mask the prompt tokens in the loss calculation later
    labels = []
    for i, (prompt, target) in enumerate(zip(examples["prompt"], examples["target"])):
        # Tokenize prompt separately to know where to mask
        prompt_formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt_tokens = tokenizer(prompt_formatted, add_special_tokens=False)["input_ids"]
        
        # Create labels: -100 for prompt tokens (ignored in loss), actual tokens for target
        input_ids = model_inputs["input_ids"][i]
        label = [-100] * len(prompt_tokens) + input_ids[len(prompt_tokens):]
        
        # Pad or truncate to match input length
        if len(label) < len(input_ids):
            label.extend(input_ids[len(label):])
        elif len(label) > len(input_ids):
            label = label[:len(input_ids)]
            
        labels.append(label)
    
    model_inputs["labels"] = labels
    return model_inputs

def main():
    training_args = TrainingArguments(
        output_dir="./qwen_finetuned",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=50,
        learning_rate=5e-6,
        num_train_epochs=3,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=None,  # Changed from "none" to None
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # 1. Load datasets
    train_dataset = load_jsonl_as_dataset("train_processed.jsonl")
    eval_dataset  = load_jsonl_as_dataset("dev_processed.jsonl")

    # 2. Load tokenizer & model
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,      # use fp16 on GPU
        device_map="auto"               # automatically place model shards on available GPUs
    )

    # 3. Preprocess / tokenize
    model.resize_token_embeddings(len(tokenizer))  # just in case

    preprocess_fn = lambda examples: preprocess_function(
        examples,
        tokenizer=tokenizer,
        max_length=2048,    # Reasonable length for fine-tuning
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
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8  # helps with tensor core alignment
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
