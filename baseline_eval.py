#!/usr/bin/env python3
"""
Baseline evaluation using mock translations to test our evaluation framework
"""
import json
import logging
import time
from datetime import datetime
from evaluate import TranslationEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_baseline_evaluation():
    """Run baseline evaluation with simple rule-based translations"""
    
    logger.info("Starting baseline evaluation...")
    
    # Load dev data
    dev_data = []
    with open('2025-mt_public_dev-jsonl.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Use first 100 samples for quick test
                break
            dev_data.append(json.loads(line))
    
    logger.info(f"Loaded {len(dev_data)} dev samples for baseline")
    
    # Create simple baseline translations
    # For now, we'll use a rule-based approach as baseline
    baseline_translations = []
    references = []
    
    for sample in dev_data:
        source = sample['source']
        reference = sample['translation']
        
        # Simple baseline: copy source (worst case scenario)
        baseline_translation = source
        
        baseline_translations.append(baseline_translation)
        references.append(reference)
    
    logger.info(f"Generated {len(baseline_translations)} baseline translations")
    
    # Evaluate baseline
    logger.info("Evaluating baseline translations...")
    evaluator = TranslationEvaluator()
    
    start_time = time.time()
    results = evaluator.evaluate_translations(
        translations=baseline_translations,
        references=references
    )
    eval_time = time.time() - start_time
    
    # Log results
    logger.info("=== BASELINE EVALUATION RESULTS ===")
    logger.info(f"BLEU-4: {results['bleu4']:.4f}")
    logger.info(f"CHRF: {results['chrf']:.4f}")
    logger.info(f"TER: {results['ter']:.4f}")
    logger.info(f"Evaluation time: {eval_time:.2f} seconds")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"baseline_results_{timestamp}.json"
    
    detailed_results = {
        "evaluation_metrics": results,
        "evaluation_time": eval_time,
        "num_samples": len(dev_data),
        "baseline_method": "copy_source",
        "samples": []
    }
    
    # Add sample translations for inspection
    for i, (sample, translation) in enumerate(zip(dev_data[:10], baseline_translations[:10])):
        detailed_results["samples"].append({
            "id": i,
            "source": sample['source'],
            "baseline_translation": translation,
            "reference": sample['translation'],
            "context": sample.get('context', '')
        })
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Detailed results saved to {results_file}")
    
    # Show some examples
    logger.info("\n=== SAMPLE TRANSLATIONS ===")
    for i in range(min(3, len(dev_data))):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"Source: {dev_data[i]['source']}")
        logger.info(f"Baseline: {baseline_translations[i]}")
        logger.info(f"Reference: {references[i]}")
    
    return results, results_file

if __name__ == '__main__':
    run_baseline_evaluation()
