# VLM Investigation: Agricultural Image Classification

This repository contains scripts for evaluating and fine-tuning Vision-Language Models (VLMs) on agricultural image classification tasks using the AgML dataset library.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Zero-Shot Classification](#zero-shot-classification)
  - [Fine-Tuning with LoRA](#fine-tuning-with-lora)
  - [Batch Processing](#batch-processing)
  - [LLM Judge Evaluation](#llm-judge-evaluation) ✨ **NEW**
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Analysis](#analysis)

---

## Overview

**Supported Models:**

*HuggingFace Models (Local):*
- `siglip2` - Google SigLIP v2 (base-patch16-224)
- `llava_next` - LLaVA-Next (llama3-8b)
- `qwen_vl` - Qwen2.5-VL (7B-Instruct)
- `yolo` - YOLO11 classification

*API-Based Models:*
- `gpt-5` - OpenAI GPT-5
- `gpt-5-nano` - OpenAI GPT-5-nano
  
**Datasets:**
- 28 agricultural classification datasets from AgML
- Includes plant disease detection, weed classification, pest identification, etc.
- See `datasets.txt` for complete list

---

## Setup

### 1. Environment Setup

**For HuggingFace Models:**
```bash
# Create conda environment
conda create -n vlm python=3.10
conda activate vlm

# Install dependencies
pip install torch torchvision transformers datasets peft
pip install agml pillow pyyaml scikit-learn pandas tqdm
pip install ultralytics  # for YOLO models
```

**For API-Based Models:**
```bash
# Or install individually:
pip install openai>=1.0.0           # For OpenAI GPT-4o
pip install google-generativeai>=0.3.0  # For Google Gemini
pip install anthropic>=0.18.0       # For Anthropic Claude
```

### 2. API Keys Setup (For API Models Only)

For API-based models, you need to set up API keys:

**Option 1: Interactive Setup Script**
```bash
bash scripts/setup_api_keys.sh
```

**Option 2: Manual Setup**
```bash
# Add to ~/.bashrc or export in terminal
export OPENAI_API_KEY='your-openai-key-here'
export GOOGLE_API_KEY='your-google-key-here'
export ANTHROPIC_API_KEY='your-anthropic-key-here'

# Apply changes
source ~/.bashrc
```

**Get API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Google Gemini: https://aistudio.google.com/app/apikey
- Anthropic Claude: https://console.anthropic.com/settings/keys

---

## Quick Start

### HuggingFace Models (Zero-Shot)

```bash
python zero_shot_classification.py \
    --dataset arabica_coffee_leaf_disease_classification \
    --model-type siglip2 \
    --config configs.yaml \
    --output-dir outputs/
```

### API Models (Zero-Shot)

```bash
# OpenAI GPT-4o-mini
python zero_shot_classification.py \
    --dataset arabica_coffee_leaf_disease_classification \
    --model-type gpt-5 \
    --config configs.yaml \
    --output-dir outputs/
```

### Single Dataset Fine-Tuning

```bash
python fine_tune_classification.py \
    --dataset tomato_leaf_disease \
    --model-type siglip2 \
    --config configs.yaml \
    --output-dir outputs/
```

### Fine-Tuning with Train/Val Splits

```bash
python fine_tune_classification.py \
    --splits-file splits.yaml \
    --fold fold_1 \
    --model-type siglip2 \
    --config configs.yaml \
    --output-dir outputs/
```

---

## Usage

### Zero-Shot Classification

Test pre-trained models without fine-tuning:

```bash
python zero_shot_classification.py \
    --dataset <dataset_name> \
    --model-type <model_type> \
    --config configs.yaml \
    --output-dir outputs/
```

**Arguments:**
- `--dataset`: Name of AgML dataset (required)
- `--model-type`: Model to use: `siglip2`, `llava_next`, `qwen_vl`, `gemma_3`
- `--config`: Path to YAML config file (default: `configs.yaml`)
- `--output-dir`: Output directory (default: `outputs/`)

**Example:**
```bash
python zero_shot_classification.py \
    --dataset corn_maize_leaf_disease \
    --model-type siglip2 \
    --config configs.yaml \
    --output-dir results/
```

---

### Fine-Tuning with LoRA

Fine-tune models using LoRA (Low-Rank Adaptation) for efficient training:

#### Option 1: Single Dataset (80/20 train/val split)

```bash
python fine_tune_classification.py \
    --dataset tomato_leaf_disease \
    --model-type siglip2 \
    --config configs.yaml \
    --output-dir outputs/
```

#### Option 2: Cross-Dataset Evaluation (using splits.yaml)

```bash
python fine_tune_classification.py \
    --splits-file splits.yaml \
    --fold fold_1 \
    --model-type siglip2 \
    --config configs.yaml \
    --output-dir outputs/
```

**Arguments:**
- `--dataset`: Single dataset name (for Option 1)
- `--splits-file`: Path to YAML file with train/val splits (for Option 2)
- `--fold`: Fold name from splits.yaml (e.g., `fold_1`, `fold_2`, `test_fold`)
- `--model-type`: Model to fine-tune (currently supports `siglip2`)
- `--config`: Path to YAML config file
- `--output-dir`: Output directory for checkpoints and results

**Splits File Format (`splits.yaml`):**
```yaml
fold_1:
  train:
    - arabica_coffee_leaf_disease_classification
    - banana_leaf_disease_classification
    - corn_maize_leaf_disease
    # ... more training datasets
  val:
    - tomato_leaf_disease
    - rice_leaf_disease_classification
    # ... more validation datasets
```

---

### Batch Processing

#### Local (Sequential Processing)

Use `scripts/baseline.sh` for batch processing multiple datasets:

**Zero-shot on all datasets:**
```bash
bash scripts/baseline.sh zero_shot siglip2
```

**Fine-tune on all datasets:**
```bash
bash scripts/baseline.sh fine_tune siglip2
```

The script reads dataset names from `datasets.txt` and processes them sequentially.

---

### LLM Judge Evaluation ✨ **NEW**

Evaluate predictions using an LLM judge that performs semantic label matching, accounting for synonyms and different naming conventions.

#### Why Use LLM Judge?

Traditional exact string matching fails when predictions and ground truth use different terminology:
- Ground truth: `"tomato_early_blight"`
- Prediction: `"early blight disease"`
- Exact match: ❌ (wrong)
- LLM Judge: ✅ (semantically correct)

#### Quick Start

**Option 1: OpenAI API (Fast, Paid)**
```bash
python evaluate_with_judge.py outputs/qwen_vl/tomato_leaf_disease/predictions.csv
```

**Option 2: GPT-OSS-120B (Free, Local, Open-Source)**
```bash
python evaluate_with_judge.py outputs/qwen_vl/tomato_leaf_disease/predictions.csv \
    --model openai/gpt-oss-120b \
    --provider hf \
    --device auto
```

#### Batch Evaluation
```bash
python evaluate_with_judge.py "outputs/qwen_vl/*/predictions.csv" \
    --model openai/gpt-oss-120b \
    --provider hf \
    --threshold 1
```

#### Confidence Scores

| Score | Meaning |
|-------|---------|
| 0 | Very unsure / different things |
| 1 | Could possibly be the same |
| 2 | Very confident / clearly the same |

Set `--threshold 1` (recommended) or `--threshold 2` (stricter)

#### Output Files
- `predictions_with_judge.csv` - Original predictions + judge scores
- `judge_metrics.json` - Summary metrics and accuracy gains
- `judge_report.txt` - Human-readable report with examples

#### Documentation
- 📖 [Full LLM Judge Guide](docs/LLM_JUDGE.md)
- 🚀 [GPT-OSS Quick Start](docs/GPT_OSS_QUICK_START.md)
- 📋 [Quick Reference](docs/LLM_JUDGE_QUICK_REF.md)
- 📓 [Jupyter Demo](experiments/llm_judge_demo.ipynb)

---

## Configuration

### Model Configurations (`configs.yaml`)

Each model has its own configuration section:

```yaml
siglip2:
  dtype: float16
  attn_implementation: sdpa
  batch_size: 12
  prompt_template: "This photo contains: {}."
  
  lora_config:
    r: 8
    lora_alpha: 16
    lora_dropout: 0.1
    target_modules: ["q_proj", "v_proj"]
    modules_to_save: ["classifier"]
  
  trainer_config:
    num_train_epochs: 1
    per_device_train_batch_size: 12
    gradient_accumulation_steps: 4
    learning_rate: 1.0e-3
    weight_decay: 0.02
    warmup_steps: 50
    logging_steps: 10
    save_strategy: "epoch"
    fp16: True
```

**Key Parameters:**
- `lora_config`: LoRA adapter settings (rank, alpha, target modules)
- `trainer_config`: Training hyperparameters (epochs, batch size, learning rate)
- `prompt_template`: Template for zero-shot classification prompts

---

## Output Structure

Results are organized by task, model, and dataset:

```
outputs/
├── zero_shot_classification/
│   └── siglip2/
│       └── tomato_leaf_disease/
│           ├── predictions.csv
│           ├── metrics.csv
│           ├── per_class.csv
│           ├── dataset_metrics.csv
│           └── confusion_matrix.csv
│
└── fine_tune_classification/
    └── siglip2/
        └── fold_1_train17_val10/
            ├── adapter_config.json
            ├── adapter_model.safetensors
            ├── predictions.csv
            ├── metrics.csv
            ├── per_class.csv
            ├── dataset_metrics.csv
            └── confusion_matrix.csv
```

### Output Files

**predictions.csv**
- Per-sample predictions with image paths, true labels, predicted labels, confidence scores
- Includes full probability distribution as JSON

**metrics.csv**
- Overall accuracy, precision, recall, F1 scores (weighted and macro)
- Total number of classes and images

**per_class.csv**
- Per-class precision, recall, F1, support
- Used for detailed class-level analysis

**dataset_metrics.csv**
- Per-dataset aggregated metrics (for multi-dataset experiments)
- Includes accuracy, precision, recall, F1, number of classes, total samples

**confusion_matrix.csv**
- Full confusion matrix with true vs predicted labels

---

## Analysis

### Aggregate Metrics by Dataset

Use the Jupyter notebook to compute dataset-level metrics from per-class results:

```bash
jupyter notebook aggregate_metrics_by_dataset.ipynb
```

The notebook:
1. Loads `per_class.csv`
2. Parses dataset names from class prefixes (`{dataset_name}-{class_name}`)
3. Aggregates metrics by dataset
4. Saves to `dataset_metrics.csv`

**Key Metrics:**
- **Accuracy**: Weighted average of per-class recall
- **Precision**: Average precision across classes
- **Recall**: Average recall across classes
- **F1 Score**: Average F1 across classes
- **Support**: Total number of samples per dataset
- **n_classes**: Number of classes per dataset

---

## Dataset Naming Convention

When combining multiple datasets for training, classes are prefixed with their dataset name:

```
{dataset_name}-{class_name}
```

**Examples:**
- `tomato_leaf_disease-Bacterial_spot`
- `corn_maize_leaf_disease-Common_rust`
- `arabica_coffee-Cerscospora`

This allows tracking which dataset each class belongs to for cross-dataset evaluation.

---