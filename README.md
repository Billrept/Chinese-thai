# Chinese-to-Thai Medical Translation Pipeline

An AI-powered translation system for Chinese-to-Thai medical conversations using open-source language models. This pipeline provides accurate medical translation with BLEU-4 evaluation and supports both Mac and supercomputer environments.

## 🎯 Project Overview

- **Task**: Translate Chinese medical conversations to Thai
- **Domain**: Medical conversations between doctors and patients
- **Evaluation**: Corpus-level 4-gram BLEU score vs professional translations
- **Output**: `submission.txt` with exactly 2,000 translation lines
- **Architecture**: Agentic workflow pipeline with fine-tuning capabilities

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Prep     │───▶│   Fine-tuning    │───▶│   Evaluation    │
│                 │    │                  │    │                 │
│ • Load JSONL    │    │ • LoRA/QLoRA     │    │ • BLEU-4        │
│ • Validate      │    │ • Medical Prompts│    │ • CHRF, TER     │
│ • Preprocess    │    │ • Efficient Train│    │ • Length Analysis│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                ▼
        ┌─────────────────────────────────────────────────────┐
        │              Translation Pipeline                   │
        │                                                     │
        │ • Context-aware prompts                            │
        │ • Medical terminology preservation                  │
        │ • Multi-environment support (Mac/GPU)              │
        │ • Submission generation                            │
        └─────────────────────────────────────────────────────┘
```

## 📊 Dataset

- **Training**: 18,600 Chinese-Thai medical conversation pairs
- **Development**: 3,000 pairs for validation
- **Format**: JSONL with `context`, `source`, and `translation` fields
- **Domain**: Real medical conversations (症状描述, 诊断, 治疗建议, etc.)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <repository>
cd CT-AI

# Run automated setup
./setup.sh

# Or manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Test the Pipeline

```bash
# Quick test with small model
python3 translation_pipeline.py --mode test --max_samples 5

# Baseline evaluation
python3 pipeline_agent.py --mode baseline
```

### 3. Full Training Pipeline

```bash
# Run complete pipeline (auto-detects environment)
python3 pipeline_agent.py --mode full

# Or specify configuration
python3 pipeline_agent.py --mode full --config custom_config.yaml
```

## 🖥️ Environment Support

### Mac (8B Models for Testing)
- **CPU/MPS**: Optimized for Apple Silicon and Intel Macs
- **Models**: DeepSeek-Coder-6.7B, Qwen2-7B
- **Memory**: Efficient with gradient checkpointing and LoRA

### Supercomputer (32B Models for Production)
- **Multi-GPU**: Automatic device mapping
- **Models**: DeepSeek-LLM-67B, Qwen2-72B, Llama-2-70B
- **Memory**: Full precision training with large batch sizes

## 📋 Usage Examples

### Basic Translation
```python
from translation_pipeline import MedicalTranslationPipeline

# Initialize pipeline
pipeline = MedicalTranslationPipeline(
    model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
    device="auto"
)

# Translate single text
translation = pipeline.translate_single(
    source_text="患者说头痛得厉害",
    context="医生询问患者症状"
)
print(translation)  # "ผู้ป่วยบอกว่าปวดหัวมาก"
```

### Batch Translation
```python
# Load data
data = [
    {"source": "你好，有什么症状吗？", "context": "医生问诊"},
    {"source": "我头疼得厉害", "context": "患者回答"}
]

# Translate batch
translations = pipeline.translate_batch(data)
```

### Fine-tuning
```bash
# Basic fine-tuning
python3 fine_tune.py \
    --model_name_or_path deepseek-ai/deepseek-coder-6.7b-instruct \
    --train_file 2025-mt_public_train-jsonl.jsonl \
    --eval_file 2025-mt_public_dev-jsonl.jsonl \
    --output_dir ./fine_tuned_model \
    --use_lora

# Advanced fine-tuning with QLoRA
python3 fine_tune.py \
    --model_name_or_path meta-llama/Llama-2-70b-chat-hf \
    --use_qlora \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4
```

### Evaluation
```bash
# Evaluate predictions
python3 evaluate.py \
    --reference_file 2025-mt_public_dev-jsonl.jsonl \
    --prediction_file predictions.txt \
    --output_dir evaluation_results

# Results include:
# - BLEU-4 score (primary metric)
# - CHRF and TER scores
# - Length analysis
# - Best/worst examples
# - Performance by source length
```

## ⚙️ Configuration

Create `pipeline_config.yaml`:

```yaml
models:
  mac_test: "microsoft/DialoGPT-medium"
  mac_production: "deepseek-ai/deepseek-coder-6.7b-instruct"
  supercomputer: "deepseek-ai/deepseek-llm-67b-chat"

data:
  train_file: "2025-mt_public_train-jsonl.jsonl"
  dev_file: "2025-mt_public_dev-jsonl.jsonl"

training:
  use_lora: true
  epochs: 3
  batch_size: 2
  learning_rate: 5e-5

evaluation:
  metrics: ["bleu", "chrf", "ter"]
  output_submission: true
```

## 📈 Optimization Strategies

### Memory Optimization
- **LoRA**: Low-rank adaptation for efficient fine-tuning
- **QLoRA**: 4-bit quantization for 70B+ models
- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision**: FP16/BF16 training

### Translation Quality
- **Medical Prompts**: Domain-specific instruction templates
- **Context Awareness**: Conversation history integration
- **Temperature Tuning**: Balanced creativity vs consistency
- **Post-processing**: Artifact removal and cleaning

### Evaluation Metrics
- **BLEU-4**: Primary competition metric
- **CHRF**: Character-level evaluation for Thai
- **TER**: Translation error rate
- **Length Analysis**: Source-target length correlation

## 📁 Project Structure

```
CT-AI/
├── pipeline_agent.py          # Main pipeline orchestrator
├── translation_pipeline.py    # Core translation engine
├── fine_tune.py              # LoRA/QLoRA fine-tuning
├── evaluate.py               # BLEU-4 evaluation
├── config.py                 # Configuration templates
├── requirements.txt          # Dependencies
├── setup.sh                  # Automated setup
├── README.md                 # This file
├── 2025-mt_public_train-jsonl.jsonl  # Training data
├── 2025-mt_public_dev-jsonl.jsonl    # Development data
└── pipeline_output_*/        # Generated results
    ├── fine_tuned_model/     # Trained model
    ├── baseline_eval/        # Baseline metrics
    ├── final_eval/          # Final metrics
    ├── pipeline_report.md    # Summary report
    └── submission.txt        # Final submission
```

## 🎯 Competition Submission

The pipeline generates `submission.txt` with exactly 2,000 lines:

```bash
# Generate submission
python3 pipeline_agent.py --mode full

# Verify format
wc -l submission.txt  # Should output: 2000 submission.txt
```

Each line contains the Thai translation of the corresponding Chinese medical conversation.

## 🔧 Troubleshooting

### Memory Issues
```bash
# Reduce batch size
python3 fine_tune.py --per_device_train_batch_size 1 --gradient_accumulation_steps 8

# Use QLoRA for large models
python3 fine_tune.py --use_qlora --model_name_or_path meta-llama/Llama-2-70b-chat-hf
```

### CUDA/MPS Issues
```bash
# Force CPU
python3 translation_pipeline.py --device cpu

# Check PyTorch installation
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Evaluation Errors
```bash
# Check file formats
head -5 2025-mt_public_dev-jsonl.jsonl
head -5 predictions.txt

# Validate JSONL
python3 -c "import json; [json.loads(line) for line in open('2025-mt_public_dev-jsonl.jsonl')]"
```

## 📊 Expected Performance

| Model Size | Environment | BLEU-4 (Baseline) | BLEU-4 (Fine-tuned) | Training Time |
|------------|-------------|-------------------|-------------------|---------------|
| 6.7B       | Mac M2      | ~15-20           | ~25-30           | 4-6 hours     |
| 7B         | RTX 4090    | ~18-23           | ~28-35           | 2-3 hours     |
| 32B        | A100 x4     | ~25-30           | ~35-45           | 6-8 hours     |
| 70B        | A100 x8     | ~30-35           | ~40-50           | 12-16 hours   |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/improvement`
3. Test your changes: `python3 -m pytest tests/`
4. Submit pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Medical conversation datasets
- Hugging Face Transformers
- DeepSeek, Qwen, and Llama model teams
- sacrebleu evaluation library
- Chinese and Thai NLP communities

---

**Happy Translating! 🌟**

For questions or support, please open an issue or contact the maintainers.
