import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
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

def preprocess_function(examples, tokenizer, max_input_length=1024, max_target_length=256):
    inputs = examples["prompt"]
    targets = examples["target"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=max_input_length,
        padding="max_length",
        return_attention_mask=True,
    )

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            truncation=True,
            max_length=max_target_length,
            padding="max_length",
            return_attention_mask=False,
        )

    # For causal LM, we shift labels internally. We just pass labels["input_ids"] directly.
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = HfArgumentParser(Seq2SeqTrainingArguments)
    # You can override these defaults from the command line.
    training_args, = parser.parse_args_into_dataclasses()

    # 1. Load datasets
    train_dataset = load_jsonl_as_dataset("train_processed.jsonl")
    eval_dataset  = load_jsonl_as_dataset("dev_processed.jsonl")

    # 2. Load tokenizer & model
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
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
        max_input_length=4096,    # Qwen2.5-32B-Instruct can handle up to 131K tokens, but we cap for memory
        max_target_length=512      # Thai sentences are shorter
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
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,  # so that padded tokens are ignored in loss
        pad_to_multiple_of=8       # helps with tensor core alignment
    )

    # 5. Initialize Trainer
    trainer = Seq2SeqTrainer(
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
