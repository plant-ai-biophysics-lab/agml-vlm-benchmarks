# VLM Investigation: Agricultural Image Classification

This repository contains scripts for evaluating and fine-tuning Vision-Language Models (VLMs) on agricultural image classification tasks using the AgML dataset library.

## Table of Contents
- [VLM Investigation: Agricultural Image Classification](#vlm-investigation-agricultural-image-classification)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Setup](#setup)
    - [1. Environment Setup](#1-environment-setup)
    - [2. API Keys Setup (For API Models Only)](#2-api-keys-setup-for-api-models-only)
  - [Quick Start](#quick-start)
    - [HuggingFace Models (Zero-Shot)](#huggingface-models-zero-shot)
    - [API Models (Zero-Shot)](#api-models-zero-shot)
  - [Usage](#usage)
    - [Zero-Shot Classification](#zero-shot-classification)
    - [Batch Processing](#batch-processing)
      - [Local (Sequential Processing)](#local-sequential-processing)
    - [LLM Judge Evaluation](#llm-judge-evaluation)
      - [Why Use LLM Judge?](#why-use-llm-judge)
      - [Quick Start](#quick-start-1)
      - [Batch Evaluation](#batch-evaluation)
      - [Confidence Scores](#confidence-scores)
      - [Output Files](#output-files)
  - [Output Structure](#output-structure)
    - [Output Files](#output-files-1)

---

## Overview

**Supported Models:**

*HuggingFace Models (Local):*
- `siglip2` - Google SigLIP v2 (base-patch16-224)
- `llava_next` - LLaVA-Next (llama3-8b)
- `qwen_vl` - Qwen2.5-VL (7B-Instruct or 72B-Instruct)
- `gemma_3` - Gemma 3 (4B-it)
- `deepseek_vl` - Deepseek VL 7B Chat
- `yolo` - YOLO11 classification

*API-Based Models:*
- `gpt-5` - OpenAI GPT-5
- `gpt-5-nano` - OpenAI GPT-5-nano
- `gemini-3-pro` - Gemini 3 Pro
- `claude-haiku-4-5` - Claude Haiku 4.5
  
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

### Batch Processing

#### Local (Sequential Processing)

Use `scripts/baseline.sh` for batch processing multiple datasets:

**Zero-shot on all datasets:**
```bash
bash scripts/baseline.sh zero_shot siglip2
```

---

### LLM Judge Evaluation

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

**Option 2: GPT-OSS-20B (Free, Local, Open-Source)**
```bash
python evaluate_with_judge.py outputs/qwen_vl/tomato_leaf_disease/predictions.csv \
    --model openai/gpt-oss-20b \
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