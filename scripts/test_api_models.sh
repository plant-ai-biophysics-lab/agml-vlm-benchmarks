#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 --dataset DATASET --model MODEL [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --dataset DATASET    Dataset name (e.g., bean_disease_uganda)"
    echo "  --model MODEL        API model key from zero_shot_classification.py"
    echo "                     Recommended: gpt-5, gpt-5-nano, gemini-3-pro-preview, claude-haiku-4-5"
    echo "                     Also accepted by dispatcher: gemini_25_flash, claude-sonnet-4-5, claude-opus-4-5"
    echo ""
    echo "Optional arguments:"
    echo "  --config FILE        Path to config file (default: configs.yaml)"
    echo "  --output-dir DIR     Output directory (default: ./outputs)"
    echo "  --plant-type TYPE    Plant type for prompt templates (e.g., 'coffee', 'bean')"
    echo "  --task TASK          Task type for prompt templates (e.g., 'disease', 'pest/damage', 'crops/weeds')"
    echo ""
    echo "Examples:"
    echo "  $0 --dataset bean_disease_uganda --model gpt-5-nano"
    echo "  $0 --dataset arabica_coffee_leaf_disease_classification --model gemini-3-pro-preview --plant-type coffee --task disease"
    echo ""
    exit 1
}

# Parse command line arguments
DATASET=""
MODEL=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$REPO_ROOT/configs.yaml"
OUTPUT_DIR="$REPO_ROOT/outputs"
PLANT_TYPE=""
TASK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --plant-type)
            PLANT_TYPE="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$DATASET" ] || [ -z "$MODEL" ]; then
    echo "Error: --dataset and --model are required"
    echo ""
    usage
fi

echo "======================================"
echo "Testing API Model: $MODEL"
echo "======================================"
echo ""
echo "Configuration:"
echo "  Dataset:    $DATASET"
echo "  Model:      $MODEL"
echo "  Config:     $CONFIG"
echo "  Output:     $OUTPUT_DIR"
if [ -n "$PLANT_TYPE" ]; then
    echo "  Plant Type: $PLANT_TYPE"
fi
if [ -n "$TASK" ]; then
    echo "  Task:       $TASK"
fi
echo ""

# Check API keys based on model type
echo "Checking API keys..."
if [[ "$MODEL" == gpt-* ]]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "❌ OPENAI_API_KEY not set (required for $MODEL)"
        echo "   Set with: export OPENAI_API_KEY='your-key-here'"
        exit 1
    else
        echo "✓ OPENAI_API_KEY is set"
    fi
elif [[ "$MODEL" == gemini* ]]; then
    if [ -z "$GEMINI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
        echo "❌ GEMINI_API_KEY or GOOGLE_API_KEY not set (required for $MODEL)"
        echo "   Set with: export GEMINI_API_KEY='your-key-here'"
        exit 1
    else
        echo "✓ Gemini API key is set"
    fi
elif [[ "$MODEL" == claude* ]] || [[ "$MODEL" == anthropic* ]]; then
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "❌ ANTHROPIC_API_KEY not set (required for $MODEL)"
        echo "   Set with: export ANTHROPIC_API_KEY='your-key-here'"
        exit 1
    else
        echo "✓ ANTHROPIC_API_KEY is set"
    fi
else
    echo "⚠️  Unknown model type: $MODEL"
    echo "   Proceeding anyway..."
fi
echo ""

# Run the test
echo "Running zero-shot classification..."
echo ""

# Build command with optional parameters
CMD="python $REPO_ROOT/zero_shot_classification.py --dataset \"$DATASET\" --model-type \"$MODEL\" --config \"$CONFIG\" --output-dir \"$OUTPUT_DIR\""

if [ -n "$PLANT_TYPE" ]; then
    CMD="$CMD --plant-type \"$PLANT_TYPE\""
fi

if [ -n "$TASK" ]; then
    CMD="$CMD --task \"$TASK\""
fi

eval $CMD

EXIT_CODE=$?

echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ API Testing Complete"
    echo "======================================"
    echo "Results saved to: $OUTPUT_DIR/$MODEL/$DATASET"
else
    echo "❌ API Testing Failed (exit code: $EXIT_CODE)"
    echo "======================================"
fi
