# VLM Investigation: Agricultural Image Classification

Lightweight repo for evaluating vision-language models on AgML classification datasets.

This README is intentionally focused on current, runnable workflows.

## What to run

### Zero-shot entrypoint

Use:

```bash
python zero_shot_classification.py \
  --dataset arabica_coffee_leaf_disease_classification \
  --model-type qwen_vl \
  --config configs.yaml \
  --output-dir outputs/
```

### In-context entrypoint (with optional fold)

Use:

```bash
python in_context_classification.py \
  --dataset arabica_coffee_leaf_disease_classification \
  --model-type qwen_vl_3 \
  --config configs.yaml \
  --output-dir outputs/
```

Fold mode (task-specific folds in `splits.yaml`):

```bash
python in_context_classification.py \
  --fold disease_fold_1 \
  --model-type qwen_vl_3 \
  --config configs.yaml \
  --splits-path splits.yaml \
  --output-dir outputs/
```

## Supported zero-shot model keys

These are the keys dispatched in `zero_shot_classification.py`.

### Local/Hugging Face models

- `siglip2`
- `llava_next`
- `qwen_vl`
- `qwen_vl_72b`
- `qwen_vl_3`
- `gemma_3`
- `deepseek_vl`

### API models

- `gpt-5`
- `gpt-5-nano`
- `gemini-3-pro-preview`
- `claude-haiku-4-5`

Additional accepted dispatcher aliases:

- `gemini_25_flash`
- `claude-sonnet-4-5`
- `claude-opus-4-5`

## Setup (minimal)

### Python environment

```bash
conda create -n vlm python=3.10 -y
conda activate vlm

pip install torch torchvision transformers datasets peft trl
pip install agml pillow pyyaml scikit-learn pandas tqdm
pip install ultralytics

# for API runs
pip install openai google-generativeai anthropic
```

### API keys

```bash
bash scripts/setup_api_keys.sh
source ~/.bashrc
```

Or export manually:

```bash
export OPENAI_API_KEY='...'
export GEMINI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
```

(`GOOGLE_API_KEY` is also accepted for Gemini.)

## Batch helpers

### Run many datasets from `datasets.txt`

```bash
bash scripts/baseline.sh zero_shot qwen_vl
```

### Quick API smoke test on one dataset

```bash
bash scripts/test_api_models.sh \
  --dataset bean_disease_uganda \
  --model gpt-5-nano \
  --plant-type bean \
  --task disease
```

## LLM judge evaluation (updated)

Evaluate semantic correctness (not exact string matching):

```bash
python evaluate_with_judge.py outputs/qwen_vl/tomato_leaf_disease/predictions.csv
```

Batch pattern:

```bash
python evaluate_with_judge.py "outputs/qwen_vl/*/predictions.csv" --threshold 1
```

Wrapper script:

```bash
bash scripts/run_judge.sh "outputs/qwen_vl/*/predictions.csv" \
  --model openai/gpt-oss-20b \
  --provider hf \
  --threshold 1 \
  --reasoning-level medium
```

Notes:

- `--threshold`: `0`, `1`, or `2` (default `1`)
- `--provider`: `openai`, `anthropic`, or `hf`
- `scripts/run_judge.sh` accepts both `--reasoning-level` and `--reasoning`

## Metrics guidance

- `metrics.csv` includes `accuracy`, `f1_macro`, and `f1_weighted`.
- For fold-based cross-dataset evaluation, prioritize averaging per-dataset `f1_macro` across folds.

## Outputs

Typical structure:

```text
outputs/
  <model_type>/
    <dataset>/
      predictions.csv
      metrics.csv
      per_class.csv
      confusion_matrix.csv
```

Judge outputs (same directory as `predictions.csv`):

- `predictions_with_judge.csv`
- `judge_metrics.json`
- `judge_report.txt`
