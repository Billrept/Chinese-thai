#!/usr/bin/env python3
"""
Generate submission file using existing pipeline with smaller batch sizes
"""
import json
import logging
import time
from datetime import datetime
from translation_pipeline import MedicalTranslationPipeline
from config import MODEL_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_submission():
    """Generate submission with optimized settings"""
    
    logger.info("Starting submission generation...")
    
    # Load dev data
    dev_data = []
    with open('2025-mt_public_dev-jsonl.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            dev_data.append(json.loads(line))
    
    logger.info(f"Loaded {len(dev_data)} dev samples")
    
    # Take exactly 2000 samples as required
    if len(dev_data) > 2000:
        dev_data = dev_data[:2000]
        logger.info(f"Using first {len(dev_data)} samples for submission")
    
    # Use mac_test config for smaller model
    config = MODEL_CONFIGS['mac_test'].copy()
    # Reduce batch size for memory efficiency
    config['generation_config']['batch_size'] = 1
    config['generation_config']['max_new_tokens'] = 150
    
    logger.info(f"Using model: {config['model_name']}")
    
    try:
        # Initialize pipeline
        pipeline = MedicalTranslationPipeline(config)
        logger.info("Pipeline initialized successfully")
        
        # Generate translations
        translations = []
        start_time = time.time()
        
        for i, sample in enumerate(dev_data):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / max(i, 1)
                eta = avg_time * (len(dev_data) - i)
                logger.info(f"Progress: {i}/{len(dev_data)} ({i/len(dev_data)*100:.1f}%) - ETA: {eta/60:.1f} min")
            
            try:
                translation = pipeline.translate(
                    text=sample['source'],
                    context=sample.get('context', '')
                )
                translations.append(translation)
            except Exception as e:
                logger.warning(f"Error translating sample {i}: {e}")
                # Use source text as fallback
                translations.append(sample['source'])
        
        # Save submission file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = f"submission_{timestamp}.txt"
        
        with open(submission_file, 'w', encoding='utf-8') as f:
            for translation in translations:
                f.write(translation + '\n')
        
        logger.info(f"Submission saved to {submission_file}")
        logger.info(f"Total translations: {len(translations)}")
        
        # Also save detailed results
        results_file = f"submission_details_{timestamp}.json"
        detailed_results = []
        
        for i, (sample, translation) in enumerate(zip(dev_data, translations)):
            detailed_results.append({
                'id': sample.get('id', f'sample_{i}'),
                'source': sample['source'],
                'context': sample.get('context', ''),
                'translation': translation,
                'reference': sample.get('translation', '')
            })
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Detailed results saved to {results_file}")
        
        total_time = time.time() - start_time
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Average time per translation: {total_time/len(translations):.2f} seconds")
        
        return submission_file, results_file
        
    except Exception as e:
        logger.error(f"Error during submission generation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == '__main__':
    generate_submission()
