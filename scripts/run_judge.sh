#!/bin/bash
# Local LLM Judge Evaluation Script
# Evaluates predictions using an LLM judge for semantic label matching
#
# Usage:
#   ./scripts/run_judge.sh <predictions_pattern> [options]
#
# Examples:
#   # Single dataset
#   ./scripts/run_judge.sh outputs/qwen_vl/tomato_leaf_disease/predictions.csv
#
#   # All datasets for a model (use quotes!)
#   ./scripts/run_judge.sh "outputs/qwen_vl/*/predictions.csv"
#
#   # Custom model and settings
#   ./scripts/run_judge.sh "outputs/*/bean_disease/predictions.csv" \
#     --model openai/gpt-oss-120b \
#     --threshold 1

set -e  # Exit on error

# Default configuration
MODEL="${MODEL:-openai/gpt-oss-20b}"
PROVIDER="${PROVIDER:-hf}"
THRESHOLD="${THRESHOLD:-1}"
MAX_WORKERS="${MAX_WORKERS:-10}"
DEVICE="${DEVICE:-auto}"
REASONING_LEVEL="${REASONING_LEVEL:-medium}"

# Function to show usage
usage() {
    cat << EOF
Usage: $0 <predictions_pattern> [options]

Evaluate model predictions using an LLM judge for semantic label matching.

Arguments:
  predictions_pattern    Glob pattern or path to predictions.csv file(s)
                        Use quotes for patterns: "outputs/model/*/predictions.csv"

Options:
  --model MODEL         LLM model (default: openai/gpt-oss-20b)
                        - openai/gpt-oss-20b (fast, open-source)
                        - openai/gpt-oss-120b (better quality)
                        - gpt-4o-mini (OpenAI API, requires key)
  --provider PROVIDER   API provider: hf, openai, anthropic (default: hf)
  --threshold N         Confidence threshold 0-2 (default: 1)
                        0=very unsure, 1=possibly same, 2=clearly same
  --max-workers N       Parallel workers (default: 10)
  --device DEVICE       Device for local models: auto, cuda, cpu (default: auto)
  --reasoning LEVEL     Reasoning effort: low, medium, high (default: medium)
  --help               Show this help message

Environment Variables:
  MODEL, PROVIDER, THRESHOLD, MAX_WORKERS, DEVICE, REASONING_LEVEL
  (Command-line options override environment variables)

Examples:
  # Single dataset with default settings
  $0 outputs/qwen_vl/tomato_leaf_disease/predictions.csv

  # All datasets for a model
  $0 "outputs/qwen_vl/*/predictions.csv"

  # Custom model and threshold
  $0 "outputs/*/predictions.csv" --model openai/gpt-oss-120b --threshold 2

  # Use OpenAI API (requires OPENAI_API_KEY)
  $0 outputs/model/predictions.csv --model gpt-4o-mini --provider openai

Output Files (saved in same directory as predictions.csv):
  - predictions_with_judge.csv  Predictions with judge scores
  - judge_metrics.json          Summary metrics and accuracy gains
  - judge_report.txt            Human-readable report

EOF
    exit 0
}

# Parse arguments
PREDICTIONS_PATTERN=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --reasoning)
            REASONING_LEVEL="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            if [ -z "$PREDICTIONS_PATTERN" ]; then
                PREDICTIONS_PATTERN="$1"
            else
                echo "Error: Unknown option: $1"
                echo "Run with --help for usage information"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check predictions pattern provided
if [ -z "$PREDICTIONS_PATTERN" ]; then
    echo "Error: Predictions pattern required"
    echo ""
    usage
fi

# Print configuration
echo "======================================"
echo "LLM Judge Evaluation"
echo "======================================"
echo ""
echo "Configuration:"
echo "  Predictions: $PREDICTIONS_PATTERN"
echo "  Model:       $MODEL"
echo "  Provider:    $PROVIDER"
echo "  Threshold:   $THRESHOLD"
echo "  Max Workers: $MAX_WORKERS"
echo "  Device:      $DEVICE"
echo "  Reasoning:   $REASONING_LEVEL"
echo ""

# Check for required API keys
if [ "$PROVIDER" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "❌ Error: OPENAI_API_KEY not set"
        echo "   Set with: export OPENAI_API_KEY='your-key-here'"
        exit 1
    fi
    echo "✓ OpenAI API key found"
elif [ "$PROVIDER" = "anthropic" ]; then
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "❌ Error: ANTHROPIC_API_KEY not set"
        echo "   Set with: export ANTHROPIC_API_KEY='your-key-here'"
        exit 1
    fi
    echo "✓ Anthropic API key found"
fi

# Check GPU for local models
if [ "$PROVIDER" = "hf" ]; then
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
    else
        echo "⚠️  Warning: nvidia-smi not found, using CPU"
        DEVICE="cpu"
    fi
fi

echo ""
echo "======================================"
echo "Running evaluation..."
echo "======================================"
echo ""

# Run the judge evaluation
python evaluate_with_judge.py "$PREDICTIONS_PATTERN" \
    --model "$MODEL" \
    --provider "$PROVIDER" \
    --threshold "$THRESHOLD" \
    --max-workers "$MAX_WORKERS" \
    --device "$DEVICE" \
    --reasoning-level "$REASONING_LEVEL"

EXIT_CODE=$?

echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation Complete"
    echo "======================================"
    echo ""
    echo "Output files saved (in each dataset directory):"
    echo "  - predictions_with_judge.csv"
    echo "  - judge_metrics.json"
    echo "  - judge_report.txt"
else
    echo "❌ Evaluation Failed (exit code: $EXIT_CODE)"
    echo "======================================"
fi

exit $EXIT_CODE
