#!/usr/bin/env python3
"""
Chinese-to-Thai Medical Translation Pipeline
Using open-source models for accurate medical conversation translation
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
import pandas as pd
from sacrebleu import BLEU
from tqdm import tqdm
import argparse
import logging
from typing import List, Dict, Tuple
import re
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedicalTranslationPipeline:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct", 
                 device: str = "auto", max_length: int = 2048):
        """
        Initialize the translation pipeline
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cpu', 'cuda', 'mps', 'auto')
            max_length: Maximum token length for generation
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Device selection with fallback
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.load_model()
        
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                **model_kwargs
            )
            
            if self.device != "cuda":  # For MPS or CPU
                self.model = self.model.to(self.device)
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_medical_prompt(self, source_text: str, context: str = "") -> str:
        """
        Create a specialized prompt for medical translation
        
        Args:
            source_text: Chinese text to translate
            context: Medical conversation context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Translate Chinese to Thai medical text:

Chinese: {source_text}
Thai:"""
        
        return prompt
    
    def translate_single(self, source_text: str, context: str = "") -> str:
        """
        Translate a single Chinese text to Thai
        
        Args:
            source_text: Chinese text to translate
            context: Medical conversation context
            
        Returns:
            Thai translation
        """
        try:
            prompt = self.create_medical_prompt(source_text, context)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length - 200,  # Leave room for generation
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Shorter for cleaner output
                    temperature=0.1,   # Very low temperature for consistency
                    do_sample=True,
                    top_p=0.8,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode and extract translation
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the Thai translation part
            if "Thai Translation:" in generated_text:
                translation = generated_text.split("Thai Translation:")[-1].strip()
            else:
                translation = generated_text[len(prompt):].strip()
            
            # Clean up the translation
            translation = self.clean_translation(translation)
            
            return translation
            
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return ""
    
    def clean_translation(self, translation: str) -> str:
        """Clean and post-process the translation"""
        # Split by common delimiters to get the first clean Thai sentence
        lines = translation.split('\n')
        for line in lines:
            line = line.strip()
            # Look for Thai characters (Thai Unicode range: 0E00-0E7F)
            if line and any('\u0e00' <= c <= '\u0e7f' for c in line):
                # Remove common artifacts and English text
                line = re.sub(r'[A-Za-z]{3,}', '', line)  # Remove English words
                line = re.sub(r'[0-9]+\s*[.:]', '', line)  # Remove numbering
                line = re.sub(r'\s+', ' ', line).strip()
                
                # Remove generation artifacts
                artifacts = ['</s>', '<s>', '<|end|>', '<|start|>', 'Human:', 'Assistant:', 'Thai:', 'Chinese:']
                for artifact in artifacts:
                    line = line.replace(artifact, '')
                
                line = line.strip()
                if len(line) > 5:  # Minimum reasonable length
                    return line
        
        # Fallback: clean the entire translation
        translation = re.sub(r'[A-Za-z]{3,}', '', translation)
        translation = re.sub(r'\s+', ' ', translation).strip()
        
        # Remove artifacts
        artifacts = ['</s>', '<s>', '<|end|>', '<|start|>', 'Human:', 'Assistant:', 'Thai:', 'Chinese:']
        for artifact in artifacts:
            translation = translation.replace(artifact, '')
        
        return translation.strip()[:200]  # Limit length
    
    def translate_batch(self, data: List[Dict]) -> List[str]:
        """
        Translate a batch of data
        
        Args:
            data: List of dictionaries with 'source' and 'context' keys
            
        Returns:
            List of Thai translations
        """
        translations = []
        
        for item in tqdm(data, desc="Translating"):
            source = item['source']
            context = item.get('context', '')
            
            translation = self.translate_single(source, context)
            translations.append(translation)
            
        return translations
    
    def evaluate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate BLEU-4 score
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            BLEU-4 score
        """
        bleu = BLEU()
        
        # Prepare references in the format expected by sacrebleu
        refs = [[ref] for ref in references]
        
        score = bleu.corpus_score(predictions, refs)
        return score.score

def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_translations(translations: List[str], output_path: str):
    """Save translations to plain text file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for translation in translations:
            f.write(translation + '\n')

def main():
    parser = argparse.ArgumentParser(description='Chinese-to-Thai Medical Translation')
    parser.add_argument('--model', type=str, default='deepseek-ai/deepseek-coder-6.7b-instruct',
                        help='Model name from HuggingFace')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu/cuda/mps/auto)')
    parser.add_argument('--mode', type=str, choices=['test', 'evaluate', 'submission'], 
                        default='test', help='Operation mode')
    parser.add_argument('--input_file', type=str, default='2025-mt_public_dev-jsonl.jsonl',
                        help='Input JSONL file')
    parser.add_argument('--output_file', type=str, default='submission.txt',
                        help='Output file for translations')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    logger.info(f"Initializing translation pipeline with model: {args.model}")
    pipeline = MedicalTranslationPipeline(
        model_name=args.model,
        device=args.device
    )
    
    # Load data
    logger.info(f"Loading data from: {args.input_file}")
    data = load_jsonl_data(args.input_file)
    
    if args.max_samples:
        data = data[:args.max_samples]
        logger.info(f"Using {len(data)} samples")
    
    if args.mode == 'test':
        # Quick test with a few samples
        test_data = data[:5]
        logger.info("Running test translation...")
        translations = pipeline.translate_batch(test_data)
        
        for i, (item, translation) in enumerate(zip(test_data, translations)):
            print(f"\n--- Example {i+1} ---")
            print(f"Chinese: {item['source']}")
            print(f"Generated Thai: {translation}")
            if 'translation' in item:
                print(f"Reference Thai: {item['translation']}")
                
    elif args.mode == 'evaluate':
        # Evaluate on development set
        logger.info("Running evaluation...")
        translations = pipeline.translate_batch(data)
        
        if 'translation' in data[0]:  # If references available
            references = [item['translation'] for item in data]
            bleu_score = pipeline.evaluate_bleu(translations, references)
            logger.info(f"BLEU-4 Score: {bleu_score:.4f}")
        
        # Save results
        save_translations(translations, args.output_file)
        logger.info(f"Translations saved to: {args.output_file}")
        
    elif args.mode == 'submission':
        # Generate submission file
        logger.info("Generating submission...")
        
        # Ensure we have exactly 2000 lines for submission
        if len(data) < 2000:
            logger.warning(f"Dataset has only {len(data)} samples, need 2000 for submission")
            
        submission_data = data[:2000]  # Take first 2000 samples
        translations = pipeline.translate_batch(submission_data)
        
        # Ensure exactly 2000 lines
        while len(translations) < 2000:
            translations.append("")  # Add empty lines if needed
        translations = translations[:2000]  # Truncate if over
        
        save_translations(translations, args.output_file)
        logger.info(f"Submission file saved: {args.output_file} ({len(translations)} lines)")

if __name__ == "__main__":
    main()
