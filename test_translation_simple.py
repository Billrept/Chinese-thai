#!/usr/bin/env python3
"""
Simple translation test with improved pipeline
"""
import json
import logging
from translation_pipeline import MedicalTranslationPipeline
from config import MODEL_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_translation():
    """Test translation with new model"""
    
    # Use the updated config
    config = MODEL_CONFIGS['mac_test']
    model_name = config['model_name']
    
    logger.info(f"Testing with model: {model_name}")
    
    # Load a sample from dev data
    with open('2025-mt_public_dev-jsonl.jsonl', 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline())
    
    source_text = sample['source']
    reference = sample['translation']
    context = sample.get('context', '')
    
    logger.info(f"Source: {source_text}")
    logger.info(f"Reference: {reference}")
    
    try:
        # Initialize pipeline
        pipeline = MedicalTranslationPipeline(
            model_name=model_name,
            device="auto"
        )
        
        # Translate
        translation = pipeline.translate_single(source_text, context)
        
        logger.info(f"Translation: {translation}")
        
        # Save result
        result = {
            'source': source_text,
            'translation': translation,
            'reference': reference,
            'model': model_name
        }
        
        with open('simple_test_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_translation()
