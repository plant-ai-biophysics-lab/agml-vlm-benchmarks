#!/bin/bash
echo "======================================"
echo "Testing API Models"
echo "======================================"
echo ""

# Configuration
DATASET="arabica_coffee_leaf_disease_classification"  # Small test dataset
CONFIG="configs.yaml"
OUTPUT_DIR="/group/jmearlesgrp/intermediate_data/eranario/vlm-investigation"

# Check API keys
echo "Checking API keys..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY not set"
else
    echo "OPENAI_API_KEY is set"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    python3 zero_shot_classification.py \
        --dataset $DATASET \
        --model-type gpt-5-nano \
        --config $CONFIG \
        --output-dir $OUTPUT_DIR
    echo ""
fi

echo "======================================"
echo "API Testing Complete"
echo "======================================"
echo "Check results in: $OUTPUT_DIR"
