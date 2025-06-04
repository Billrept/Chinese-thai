#!/usr/bin/env python3
"""
Demo script for Chinese-to-Thai Medical Translation Pipeline
Quick demonstration without requiring model downloads
"""

import json
import os
from typing import List, Dict

class MockTranslationPipeline:
    """Mock pipeline for demonstration purposes"""
    
    def __init__(self):
        self.sample_translations = {
            "你好，有什么症状吗？": "สวัสดีครับ มีอาการอะไรบ้างครับ",
            "我头疼得厉害": "ฉันปวดหัวมาก",
            "需要做检查吗？": "ต้องทำการตรวจหรือไม่",
            "请问医生": "ขอถามหมอครับ",
            "谢谢医生": "ขอบคุณหมอครับ"
        }
    
    def translate_single(self, text: str, context: str = "") -> str:
        """Mock translation for demo"""
        # Use sample translations or create a simple mock
        if text in self.sample_translations:
            return self.sample_translations[text]
        else:
            # Simple mock: add Thai greeting
            return f"การแปล: {text}"
    
    def translate_batch(self, data: List[Dict]) -> List[str]:
        """Mock batch translation"""
        return [self.translate_single(item['source'], item.get('context', '')) 
                for item in data]

def load_sample_data(filename: str, max_samples: int = 5) -> List[Dict]:
    """Load sample data for demonstration"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data.append(json.loads(line.strip()))
    return data

def demo_pipeline():
    """Demonstrate the translation pipeline"""
    print("🚀 Chinese-to-Thai Medical Translation Pipeline Demo")
    print("=" * 60)
    
    # Load sample data
    print("\n📊 Loading sample data...")
    try:
        train_data = load_sample_data("2025-mt_public_train-jsonl.jsonl", 3)
        dev_data = load_sample_data("2025-mt_public_dev-jsonl.jsonl", 3)
        
        print(f"✅ Loaded {len(train_data)} training samples")
        print(f"✅ Loaded {len(dev_data)} development samples")
        
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        return
    
    # Initialize mock pipeline
    print("\n🤖 Initializing translation pipeline...")
    pipeline = MockTranslationPipeline()
    print("✅ Pipeline initialized (using mock translator for demo)")
    
    # Demo translations
    print("\n🔄 Demo Translations:")
    print("-" * 40)
    
    for i, item in enumerate(train_data):
        source = item['source']
        reference = item['translation']
        context = item.get('context', '')[:100] + "..." if item.get('context', '') else ""
        
        # Get mock translation
        prediction = pipeline.translate_single(source, context)
        
        print(f"\n📝 Example {i+1}:")
        print(f"Context: {context}")
        print(f"Chinese: {source}")
        print(f"Reference: {reference}")
        print(f"Predicted: {prediction}")
        print("-" * 40)
    
    # Demo batch processing
    print("\n📦 Batch Processing Demo:")
    batch_predictions = pipeline.translate_batch(dev_data)
    
    for i, (item, pred) in enumerate(zip(dev_data, batch_predictions)):
        print(f"Batch {i+1}: {item['source'][:30]}... → {pred[:30]}...")
    
    # Demo evaluation metrics (mock)
    print("\n📈 Evaluation Metrics (Mock):")
    print(f"BLEU-4 Score: 35.42 (demo value)")
    print(f"CHRF Score: 52.18 (demo value)")
    print(f"Translation Speed: {len(batch_predictions)} sentences in 0.1s")
    
    # Demo pipeline stages
    print("\n🏗️ Pipeline Stages:")
    stages = [
        "✅ 1. Data Loading & Validation",
        "✅ 2. Model Initialization", 
        "✅ 3. Context-Aware Translation",
        "✅ 4. Medical Terminology Processing",
        "✅ 5. Quality Evaluation",
        "✅ 6. Submission Generation"
    ]
    
    for stage in stages:
        print(stage)
    
    print("\n🎯 Next Steps:")
    next_steps = [
        "1. Install dependencies: pip install -r requirements.txt",
        "2. Run setup script: ./setup.sh",
        "3. Test with real model: python translation_pipeline.py --mode test",
        "4. Run baseline: python pipeline_agent.py --mode baseline", 
        "5. Full training: python pipeline_agent.py --mode full"
    ]
    
    for step in next_steps:
        print(step)
    
    print("\n🌟 Pipeline demo completed successfully!")
    print("Ready for real model training and evaluation.")

def demo_config():
    """Demonstrate configuration options"""
    print("\n⚙️ Configuration Demo:")
    print("-" * 30)
    
    from config import MODEL_CONFIGS, TRAINING_CONFIG
    
    print("Available Models:")
    for env, config in MODEL_CONFIGS.items():
        print(f"  {env}: {config['model_name']}")
    
    print(f"\nTraining Settings:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")

def demo_file_structure():
    """Show the project structure"""
    print("\n📁 Project Structure:")
    print("-" * 25)
    
    structure = """
CT-AI/
├── 📋 pipeline_agent.py          # Main orchestrator
├── 🔄 translation_pipeline.py    # Translation engine  
├── 🎯 fine_tune.py              # Model training
├── 📊 evaluate.py               # BLEU-4 evaluation
├── ⚙️ config.py                 # Configuration
├── 📦 requirements.txt          # Dependencies
├── 🚀 setup.sh                  # Auto setup
├── 📖 README.md                 # Documentation
├── 🧪 demo.py                   # This demo
├── 📊 Data files:
│   ├── 2025-mt_public_train-jsonl.jsonl (18.6k samples)
│   └── 2025-mt_public_dev-jsonl.jsonl   (3k samples)
└── 📁 Generated outputs:
    ├── fine_tuned_model/         # Trained model
    ├── evaluation_results/       # Metrics & analysis
    └── submission.txt           # Final output
    """
    print(structure)

if __name__ == "__main__":
    demo_pipeline()
    demo_config()
    demo_file_structure()
