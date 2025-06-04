#!/usr/bin/env python3
"""
Evaluation script for Chinese-to-Thai Medical Translation
Implements BLEU-4 and other metrics for comprehensive evaluation
"""

import json
import os
import argparse
from typing import List, Dict, Tuple
import numpy as np
from sacrebleu import BLEU, CHRF, TER
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationEvaluator:
    def __init__(self):
        """Initialize evaluator with different metrics"""
        self.bleu = BLEU()
        self.chrf = CHRF()
        self.ter = TER()
        
    def load_data(self, file_path: str) -> List[Dict]:
        """Load JSONL data"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def load_predictions(self, file_path: str) -> List[str]:
        """Load predictions from text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    def calculate_bleu_4(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate BLEU-4 score with detailed breakdown"""
        # Prepare references in the format expected by sacrebleu
        refs = [[ref] for ref in references]
        
        # Calculate corpus-level BLEU
        bleu_score = self.bleu.corpus_score(predictions, refs)
        
        # Calculate sentence-level BLEU scores
        sentence_scores = []
        for pred, ref in zip(predictions, references):
            sent_score = self.bleu.sentence_score(pred, [ref])
            sentence_scores.append(sent_score.score)
        
        return {
            "corpus_bleu": bleu_score.score,
            "sentence_bleu_mean": np.mean(sentence_scores),
            "sentence_bleu_std": np.std(sentence_scores),
            "sentence_bleu_scores": sentence_scores,
            "bleu_details": {
                "precision_1": bleu_score.precisions[0],
                "precision_2": bleu_score.precisions[1] if len(bleu_score.precisions) > 1 else 0,
                "precision_3": bleu_score.precisions[2] if len(bleu_score.precisions) > 2 else 0,
                "precision_4": bleu_score.precisions[3] if len(bleu_score.precisions) > 3 else 0,
                "brevity_penalty": bleu_score.bp,
                "length_ratio": bleu_score.ratio,
                "translation_length": bleu_score.sys_len,
                "reference_length": bleu_score.ref_len
            }
        }
    
    def calculate_other_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate CHRF and TER scores"""
        refs = [[ref] for ref in references]
        
        chrf_score = self.chrf.corpus_score(predictions, refs)
        ter_score = self.ter.corpus_score(predictions, refs)
        
        return {
            "chrf": chrf_score.score,
            "ter": ter_score.score
        }
    
    def analyze_length_distribution(self, predictions: List[str], references: List[str]) -> Dict:
        """Analyze length distribution of predictions vs references"""
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        return {
            "pred_length_mean": np.mean(pred_lengths),
            "pred_length_std": np.std(pred_lengths),
            "ref_length_mean": np.mean(ref_lengths),
            "ref_length_std": np.std(ref_lengths),
            "length_correlation": np.corrcoef(pred_lengths, ref_lengths)[0, 1]
        }
    
    def analyze_by_source_length(self, predictions: List[str], references: List[str], 
                                sources: List[str]) -> Dict:
        """Analyze performance by source sentence length"""
        source_lengths = [len(src.split()) for src in sources]
        
        # Group by source length ranges
        length_groups = defaultdict(list)
        for i, length in enumerate(source_lengths):
            if length <= 10:
                group = "short (≤10)"
            elif length <= 20:
                group = "medium (11-20)"
            elif length <= 30:
                group = "long (21-30)"
            else:
                group = "very_long (>30)"
            
            length_groups[group].append(i)
        
        # Calculate BLEU for each group
        results = {}
        for group, indices in length_groups.items():
            group_preds = [predictions[i] for i in indices]
            group_refs = [references[i] for i in indices]
            
            if group_preds:  # Only if group is not empty
                refs = [[ref] for ref in group_refs]
                bleu_score = self.bleu.corpus_score(group_preds, refs)
                results[group] = {
                    "count": len(indices),
                    "bleu": bleu_score.score
                }
        
        return results
    
    def find_best_worst_examples(self, predictions: List[str], references: List[str], 
                                sources: List[str], n: int = 5) -> Dict:
        """Find best and worst translation examples"""
        sentence_scores = []
        for pred, ref in zip(predictions, references):
            sent_score = self.bleu.sentence_score(pred, [ref])
            sentence_scores.append(sent_score.score)
        
        # Get indices of best and worst examples
        indices = np.argsort(sentence_scores)
        worst_indices = indices[:n]
        best_indices = indices[-n:]
        
        best_examples = []
        for idx in best_indices:
            best_examples.append({
                "source": sources[idx],
                "prediction": predictions[idx],
                "reference": references[idx],
                "bleu_score": sentence_scores[idx]
            })
        
        worst_examples = []
        for idx in worst_indices:
            worst_examples.append({
                "source": sources[idx],
                "prediction": predictions[idx],
                "reference": references[idx],
                "bleu_score": sentence_scores[idx]
            })
        
        return {
            "best_examples": best_examples,
            "worst_examples": worst_examples
        }
    
    def create_evaluation_report(self, predictions: List[str], references: List[str], 
                               sources: List[str], output_dir: str = "./evaluation_results"):
        """Create comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Calculating BLEU-4 scores...")
        bleu_results = self.calculate_bleu_4(predictions, references)
        
        logger.info("Calculating other metrics...")
        other_metrics = self.calculate_other_metrics(predictions, references)
        
        logger.info("Analyzing length distributions...")
        length_analysis = self.analyze_length_distribution(predictions, references)
        
        logger.info("Analyzing by source length...")
        length_group_analysis = self.analyze_by_source_length(predictions, references, sources)
        
        logger.info("Finding best/worst examples...")
        examples_analysis = self.find_best_worst_examples(predictions, references, sources)
        
        # Combine all results
        full_results = {
            "summary": {
                "corpus_bleu_4": bleu_results["corpus_bleu"],
                "chrf": other_metrics["chrf"],
                "ter": other_metrics["ter"],
                "num_examples": len(predictions)
            },
            "bleu_analysis": bleu_results,
            "other_metrics": other_metrics,
            "length_analysis": length_analysis,
            "performance_by_source_length": length_group_analysis,
            "example_analysis": examples_analysis
        }
        
        # Save detailed results
        with open(os.path.join(output_dir, "detailed_results.json"), "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        # Create visualizations
        self.create_visualizations(full_results, output_dir)
        
        # Create summary report
        self.create_summary_report(full_results, output_dir)
        
        return full_results
    
    def create_visualizations(self, results: Dict, output_dir: str):
        """Create visualization plots"""
        plt.style.use('seaborn-v0_8')
        
        # BLEU score distribution
        plt.figure(figsize=(10, 6))
        sentence_scores = results["bleu_analysis"]["sentence_bleu_scores"]
        plt.hist(sentence_scores, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Sentence-level BLEU Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sentence-level BLEU Scores')
        plt.axvline(results["bleu_analysis"]["sentence_bleu_mean"], 
                   color='red', linestyle='--', label=f'Mean: {results["bleu_analysis"]["sentence_bleu_mean"]:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bleu_distribution.png"), dpi=300)
        plt.close()
        
        # Performance by source length
        length_data = results["performance_by_source_length"]
        if length_data:
            groups = list(length_data.keys())
            bleu_scores = [length_data[group]["bleu"] for group in groups]
            counts = [length_data[group]["count"] for group in groups]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # BLEU by source length
            ax1.bar(groups, bleu_scores, alpha=0.7)
            ax1.set_xlabel('Source Length Group')
            ax1.set_ylabel('BLEU Score')
            ax1.set_title('BLEU Score by Source Length')
            ax1.tick_params(axis='x', rotation=45)
            
            # Sample count by source length
            ax2.bar(groups, counts, alpha=0.7, color='orange')
            ax2.set_xlabel('Source Length Group')
            ax2.set_ylabel('Number of Examples')
            ax2.set_title('Sample Distribution by Source Length')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "performance_by_length.png"), dpi=300)
            plt.close()
    
    def create_summary_report(self, results: Dict, output_dir: str):
        """Create a human-readable summary report"""
        report = f"""
# Chinese-to-Thai Medical Translation Evaluation Report

## Summary
- **Corpus BLEU-4 Score**: {results['summary']['corpus_bleu_4']:.4f}
- **CHRF Score**: {results['summary']['chrf']:.4f}
- **TER Score**: {results['summary']['ter']:.4f}
- **Number of Examples**: {results['summary']['num_examples']}

## BLEU Analysis
- **Corpus BLEU-4**: {results['bleu_analysis']['corpus_bleu']:.4f}
- **Sentence BLEU Mean**: {results['bleu_analysis']['sentence_bleu_mean']:.4f} (±{results['bleu_analysis']['sentence_bleu_std']:.4f})
- **Brevity Penalty**: {results['bleu_analysis']['bleu_details']['brevity_penalty']:.4f}
- **Length Ratio**: {results['bleu_analysis']['bleu_details']['length_ratio']:.4f}

### N-gram Precisions
- **1-gram**: {results['bleu_analysis']['bleu_details']['precision_1']:.4f}
- **2-gram**: {results['bleu_analysis']['bleu_details']['precision_2']:.4f}
- **3-gram**: {results['bleu_analysis']['bleu_details']['precision_3']:.4f}
- **4-gram**: {results['bleu_analysis']['bleu_details']['precision_4']:.4f}

## Length Analysis
- **Prediction Length**: {results['length_analysis']['pred_length_mean']:.2f} (±{results['length_analysis']['pred_length_std']:.2f}) words
- **Reference Length**: {results['length_analysis']['ref_length_mean']:.2f} (±{results['length_analysis']['ref_length_std']:.2f}) words
- **Length Correlation**: {results['length_analysis']['length_correlation']:.4f}

## Performance by Source Length
"""
        
        for group, data in results['performance_by_source_length'].items():
            report += f"- **{group}**: {data['bleu']:.4f} BLEU ({data['count']} examples)\n"
        
        report += f"""
## Best Translation Examples
"""
        for i, example in enumerate(results['example_analysis']['best_examples'][:3]):
            report += f"""
### Example {i+1} (BLEU: {example['bleu_score']:.4f})
- **Source**: {example['source']}
- **Prediction**: {example['prediction']}
- **Reference**: {example['reference']}
"""
        
        report += f"""
## Worst Translation Examples
"""
        for i, example in enumerate(results['example_analysis']['worst_examples'][:3]):
            report += f"""
### Example {i+1} (BLEU: {example['bleu_score']:.4f})
- **Source**: {example['source']}
- **Prediction**: {example['prediction']}
- **Reference**: {example['reference']}
"""
        
        with open(os.path.join(output_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
            f.write(report)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Chinese-to-Thai Medical Translation')
    parser.add_argument('--reference_file', type=str, required=True,
                        help='JSONL file with reference translations')
    parser.add_argument('--prediction_file', type=str, required=True,
                        help='Text file with predictions (one per line)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TranslationEvaluator()
    
    # Load data
    logger.info("Loading reference data...")
    reference_data = evaluator.load_data(args.reference_file)
    
    logger.info("Loading predictions...")
    predictions = evaluator.load_predictions(args.prediction_file)
    
    # Extract references and sources
    references = [item['translation'] for item in reference_data]
    sources = [item['source'] for item in reference_data]
    
    # Ensure same length
    min_length = min(len(predictions), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]
    sources = sources[:min_length]
    
    logger.info(f"Evaluating {min_length} examples...")
    
    # Run evaluation
    results = evaluator.create_evaluation_report(
        predictions, references, sources, args.output_dir
    )
    
    # Print summary
    print(f"\n=== Evaluation Results ===")
    print(f"Corpus BLEU-4: {results['summary']['corpus_bleu_4']:.4f}")
    print(f"CHRF: {results['summary']['chrf']:.4f}")
    print(f"TER: {results['summary']['ter']:.4f}")
    print(f"Detailed results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
