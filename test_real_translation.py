#!/usr/bin/env python3
"""
Final working test of our translation pipeline
"""
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import MODEL_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_actual_translation():
    """Test with actual model translation"""
    
    logger.info("Starting actual translation test...")
    
    # Use the smallest model available
    config = MODEL_CONFIGS['mac_test']
    model_name = config['model_name']
    
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimizations for Mac
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Use quantization if available and on CPU
        quantization_config = None
        if device == "cpu":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
            except Exception as e:
                logger.warning(f"Quantization not available: {e}")
                quantization_config = None
        
        # Load model with proper device handling
        if device == "mps":
            # For MPS, load directly to device without device_map
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        else:
            # For CPU, use device_map and quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float32,
                device_map="auto" if quantization_config else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        logger.info("Model loaded successfully!")
        
        # Load test sample
        with open('2025-mt_public_dev-jsonl.jsonl', 'r', encoding='utf-8') as f:
            sample = json.loads(f.readline())
        
        source_text = sample['source']
        reference = sample['translation']
        context = sample.get('context', '')
        
        # Create translation prompt
        if context:
            prompt = f"""You are a professional medical translator. Translate the following Chinese medical text to Thai.

Context: {context[:200]}...
Chinese: {source_text}
Thai Translation:"""
        else:
            prompt = f"""Translate this Chinese medical text to Thai:

Chinese: {source_text}
Thai:"""
        
        logger.info(f"Source: {source_text}")
        logger.info("Generating translation...")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config['max_new_tokens'],
                temperature=config['temperature'],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = generated_text[len(prompt):].strip()
        
        logger.info(f"Generated: {translation}")
        logger.info(f"Reference: {reference}")
        
        # Save result
        result = {
            'source': source_text,
            'translation': translation,
            'reference': reference,
            'context': context,
            'model': model_name
        }
        
        with open('test_translation_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ Translation test completed!")
        logger.info("Result saved to test_translation_result.json")
        
        return translation
        
    except Exception as e:
        logger.error(f"Translation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Clean up GPU memory
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == '__main__':
    test_actual_translation()
