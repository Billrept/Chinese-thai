#!/usr/bin/env python3
"""
Agentic Workflow Pipeline for Chinese-to-Thai Medical Translation
Orchestrates the entire pipeline from data processing to submission
"""

import os
import json
import argparse
import logging
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TranslationPipelineAgent:
    def __init__(self, config_path: str = "pipeline_config.yaml"):
        """Initialize the pipeline agent"""
        self.config = self.load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"pipeline_output_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                "models": {
                    "mac_test": "microsoft/DialoGPT-medium",
                    "mac_production": "deepseek-ai/deepseek-coder-6.7b-instruct", 
                    "supercomputer": "deepseek-ai/deepseek-llm-67b-chat"
                },
                "data": {
                    "train_file": "2025-mt_public_train-jsonl.jsonl",
                    "dev_file": "2025-mt_public_dev-jsonl.jsonl"
                },
                "training": {
                    "use_lora": True,
                    "epochs": 3,
                    "batch_size": 2,
                    "learning_rate": 5e-5
                },
                "evaluation": {
                    "metrics": ["bleu", "chrf", "ter"],
                    "output_submission": True
                }
            }
            self.save_config(config, config_path)
        return config
    
    def save_config(self, config: Dict, config_path: str):
        """Save configuration to file"""
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def detect_environment(self) -> str:
        """Detect the current environment"""
        try:
            # Check for CUDA
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_count >= 4 or memory > 40e9:  # Multiple GPUs or >40GB memory
                    return "supercomputer"
                else:
                    return "gpu_workstation"
            
            # Check for Apple Silicon
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mac_production"
            else:
                return "mac_test"
                
        except ImportError:
            return "mac_test"
    
    def prepare_data(self) -> bool:
        """Prepare and validate data"""
        logger.info("Preparing data...")
        
        train_file = self.config["data"]["train_file"]
        dev_file = self.config["data"]["dev_file"]
        
        # Check if files exist
        if not os.path.exists(train_file):
            logger.error(f"Training file not found: {train_file}")
            return False
        
        if not os.path.exists(dev_file):
            logger.error(f"Development file not found: {dev_file}")
            return False
        
        # Validate data format
        try:
            with open(train_file, 'r') as f:
                sample = json.loads(f.readline())
                required_keys = ['source', 'translation', 'context']
                for key in required_keys:
                    if key not in sample:
                        logger.warning(f"Missing key '{key}' in training data")
        
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False
        
        logger.info("Data preparation completed successfully")
        return True
    
    def run_baseline_evaluation(self, environment: str) -> Dict:
        """Run baseline evaluation without fine-tuning"""
        logger.info("Running baseline evaluation...")
        
        model_name = self.config["models"][environment]
        
        # Run translation pipeline
        cmd = [
            "python", "translation_pipeline.py",
            "--model", model_name,
            "--mode", "evaluate",
            "--input_file", self.config["data"]["dev_file"],
            "--output_file", os.path.join(self.output_dir, "baseline_predictions.txt"),
            "--max_samples", "100"  # Quick baseline
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Baseline translation completed")
            
            # Run evaluation
            eval_cmd = [
                "python", "evaluate.py",
                "--reference_file", self.config["data"]["dev_file"],
                "--prediction_file", os.path.join(self.output_dir, "baseline_predictions.txt"),
                "--output_dir", os.path.join(self.output_dir, "baseline_eval")
            ]
            
            subprocess.run(eval_cmd, check=True)
            
            # Load results
            results_file = os.path.join(self.output_dir, "baseline_eval", "detailed_results.json")
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            return results["summary"]
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Baseline evaluation failed: {e}")
            return {}
    
    def run_fine_tuning(self, environment: str) -> bool:
        """Run fine-tuning process"""
        logger.info("Starting fine-tuning...")
        
        model_name = self.config["models"][environment]
        
        cmd = [
            "python", "fine_tune.py",
            "--model_name_or_path", model_name,
            "--train_file", self.config["data"]["train_file"],
            "--eval_file", self.config["data"]["dev_file"],
            "--output_dir", os.path.join(self.output_dir, "fine_tuned_model"),
            "--num_train_epochs", str(self.config["training"]["epochs"]),
            "--per_device_train_batch_size", str(self.config["training"]["batch_size"]),
            "--learning_rate", str(self.config["training"]["learning_rate"]),
            "--wandb_project", "chinese-thai-medical-translation",
            "--run_name", f"pipeline_{self.timestamp}"
        ]
        
        if self.config["training"]["use_lora"]:
            cmd.append("--use_lora")
        
        if environment == "mac_test":
            cmd.extend(["--max_train_samples", "1000"])  # Limit for testing
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("Fine-tuning completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Fine-tuning failed: {e}")
            return False
    
    def run_final_evaluation(self, environment: str) -> Dict:
        """Run final evaluation with fine-tuned model"""
        logger.info("Running final evaluation...")
        
        model_path = os.path.join(self.output_dir, "fine_tuned_model")
        
        # Run translation with fine-tuned model
        cmd = [
            "python", "translation_pipeline.py",
            "--model", model_path,
            "--mode", "evaluate", 
            "--input_file", self.config["data"]["dev_file"],
            "--output_file", os.path.join(self.output_dir, "final_predictions.txt")
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Run evaluation
            eval_cmd = [
                "python", "evaluate.py",
                "--reference_file", self.config["data"]["dev_file"],
                "--prediction_file", os.path.join(self.output_dir, "final_predictions.txt"),
                "--output_dir", os.path.join(self.output_dir, "final_eval")
            ]
            
            subprocess.run(eval_cmd, check=True)
            
            # Load results
            results_file = os.path.join(self.output_dir, "final_eval", "detailed_results.json")
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            return results["summary"]
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Final evaluation failed: {e}")
            return {}
    
    def generate_submission(self, environment: str) -> bool:
        """Generate final submission file"""
        logger.info("Generating submission file...")
        
        model_path = os.path.join(self.output_dir, "fine_tuned_model")
        
        # Create test data for submission (first 2000 lines of dev set)
        cmd = [
            "python", "translation_pipeline.py",
            "--model", model_path,
            "--mode", "submission",
            "--input_file", self.config["data"]["dev_file"],
            "--output_file", "submission.txt"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("Submission file generated: submission.txt")
            
            # Verify submission format
            with open("submission.txt", 'r') as f:
                lines = f.readlines()
                if len(lines) != 2000:
                    logger.warning(f"Submission file has {len(lines)} lines, expected 2000")
                else:
                    logger.info("Submission file format verified")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Submission generation failed: {e}")
            return False
    
    def create_pipeline_report(self, baseline_results: Dict, final_results: Dict):
        """Create comprehensive pipeline report"""
        report = f"""
# Chinese-to-Thai Medical Translation Pipeline Report

**Pipeline Run**: {self.timestamp}
**Environment**: {self.detect_environment()}

## Results Summary

### Baseline Model Performance
- **BLEU-4**: {baseline_results.get('corpus_bleu_4', 'N/A'):.4f}
- **CHRF**: {baseline_results.get('chrf', 'N/A'):.4f}
- **TER**: {baseline_results.get('ter', 'N/A'):.4f}

### Fine-tuned Model Performance  
- **BLEU-4**: {final_results.get('corpus_bleu_4', 'N/A'):.4f}
- **CHRF**: {final_results.get('chrf', 'N/A'):.4f}
- **TER**: {final_results.get('ter', 'N/A'):.4f}

### Improvement
- **BLEU-4 Improvement**: {final_results.get('corpus_bleu_4', 0) - baseline_results.get('corpus_bleu_4', 0):.4f}

## Configuration Used
```yaml
{yaml.dump(self.config, default_flow_style=False)}
```

## Output Files
- `fine_tuned_model/`: Fine-tuned model directory
- `baseline_eval/`: Baseline evaluation results
- `final_eval/`: Final evaluation results
- `submission.txt`: Final submission file
- `pipeline.log`: Pipeline execution log

## Next Steps
1. Review evaluation results in detail
2. Consider hyperparameter tuning if performance is suboptimal
3. Test on larger model if resources allow
4. Submit `submission.txt` for final evaluation
"""
        
        with open(os.path.join(self.output_dir, "pipeline_report.md"), 'w') as f:
            f.write(report)
    
    def run_full_pipeline(self, mode: str = "auto"):
        """Run the complete pipeline"""
        logger.info(f"Starting full pipeline in {mode} mode...")
        
        # Detect environment
        environment = self.detect_environment()
        logger.info(f"Detected environment: {environment}")
        
        # Step 1: Prepare data
        if not self.prepare_data():
            logger.error("Data preparation failed. Aborting pipeline.")
            return False
        
        # Step 2: Baseline evaluation
        baseline_results = self.run_baseline_evaluation(environment)
        if not baseline_results:
            logger.warning("Baseline evaluation failed, continuing anyway...")
            baseline_results = {}
        
        # Step 3: Fine-tuning
        if mode in ["full", "auto"]:
            if not self.run_fine_tuning(environment):
                logger.error("Fine-tuning failed. Aborting pipeline.")
                return False
            
            # Step 4: Final evaluation
            final_results = self.run_final_evaluation(environment)
            
            # Step 5: Generate submission
            self.generate_submission(environment)
        else:
            final_results = baseline_results
        
        # Step 6: Create report
        self.create_pipeline_report(baseline_results, final_results)
        
        logger.info(f"Pipeline completed successfully! Results in: {self.output_dir}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Chinese-to-Thai Medical Translation Pipeline')
    parser.add_argument('--mode', choices=['baseline', 'full', 'auto'], default='auto',
                        help='Pipeline mode: baseline (eval only), full (train+eval), auto (detect)')
    parser.add_argument('--config', type=str, default='pipeline_config.yaml',
                        help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    agent = TranslationPipelineAgent(args.config)
    success = agent.run_full_pipeline(args.mode)
    
    if success:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Results saved to: {agent.output_dir}")
        print(f"📄 Submission file: submission.txt")
    else:
        print(f"\n❌ Pipeline failed. Check logs for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
