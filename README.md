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
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Analysis](#analysis)

---

## Overview

**Supported Models:**
- `siglip2` - Google SigLIP v2 (base-patch16-224)
- `llava_next` - LLaVA-Next (llama3-8b)
- `qwen_vl` - Qwen2.5-VL (7B-Instruct)
- `yolo` - YOLO11 classification

**Datasets:**
- 28 agricultural classification datasets from AgML
- Includes plant disease detection, weed classification, pest identification, etc.
- See `datasets.txt` for complete list

---

## Setup

### 1. Environment Setup

```bash
# Create conda environment
conda create -n vlm python=3.10
conda activate vlm

# Install dependencies
pip install torch torchvision transformers datasets peft
pip install agml pillow pyyaml scikit-learn pandas tqdm
pip install ultralytics  # for YOLO models
```

### 2. Directory Structure

```
vlm-investigation/
├── configs.yaml              # Model configurations
├── splits.yaml               # Train/val fold definitions
├── datasets.txt              # List of datasets for batch processing
├── fine_tune_classification.py
├── zero_shot_classification.py
├── farm_finetune.sh          # SLURM job for fine-tuning
├── farm_zero.sh              # SLURM job for zero-shot
├── scripts/
│   ├── baseline.sh           # Local batch processing
│   └── lora_finetune.sh      # Local fine-tuning script
├── models/
│   ├── siglip2.py
│   ├── llava_next.py
│   ├── qwen_vl.py
│   └── yolo11.py
├── tasks/
│   └── classification.py    # Dataset loading utilities
└── utils/
    └── utils.py              # Metrics and evaluation
```

---

## Quick Start

### Single Dataset Zero-Shot Classification

```bash
python zero_shot_classification.py \
    --dataset arabica_coffee_leaf_disease_classification \
    --model-type siglip2 \
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

#### SLURM Cluster (Parallel Jobs)

**Submit zero-shot job:**
```bash
sbatch farm_zero.sh siglip2
```

**Submit fine-tuning job for single model:**
```bash
sbatch farm_finetune.sh siglip2 fold_1
```

**Submit fine-tuning job for multiple models:**
```bash
sbatch farm_finetune.sh "siglip2 llava_next qwen_vl" fold_1
```

This will train all three models sequentially in a single SLURM job.

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

## Tips and Best Practices

1. **GPU Memory**: Adjust `batch_size` and `gradient_accumulation_steps` based on available GPU memory

2. **Learning Rate**: Start with `1e-3` for LoRA fine-tuning, reduce if training is unstable

3. **LoRA Rank**: Higher rank (`r`) = more parameters but better adaptation. Default `r=8` works well.

4. **Validation Strategy**: 
   - Single dataset: Use built-in 80/20 split
   - Cross-dataset: Use `splits.yaml` to evaluate generalization across datasets

5. **Prompt Engineering**: Modify `prompt_template` in `configs.yaml` for different zero-shot performance

6. **Monitoring**: Check SLURM output files in `outputs/` for job logs and errors

---

## Troubleshooting

**CUDA Out of Memory:**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps` (maintains effective batch size)
- Use `fp16: True` in trainer config

**Missing Datasets:**
- Verify dataset name matches AgML exactly (check `datasets.txt`)
- AgML will auto-download datasets on first use

**Model Loading Warnings:**
- "Some weights were not initialized" - Expected when adding classification head
- "Missing adapter keys" - Check model type matches between training and testing

**Dimension Mismatch Errors:**
- Ensure validation set uses same class naming convention as training
- Check `label2id` and `id2label` mappings are consistent

---