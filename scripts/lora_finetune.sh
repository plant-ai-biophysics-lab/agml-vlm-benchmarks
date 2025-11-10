# Parse arguments
# Usage: sbatch farm_finetune.sh "model1 model2 model3" fold_name
# Example: sbatch farm_finetune.sh "siglip2 llava_next qwen_vl" fold_1
MODEL_TYPES="${1:-siglip2}"
FOLD="${2:-fold_1}"
SPLITS_FILE="splits.yaml"
CONFIG_FILE="configs.yaml"
OUTPUT_DIR="/group/jmearlesgrp/intermediate_data/eranario/vlm-investigation"

echo "======================================"
echo "Fine-tuning with train/val split"
echo "Job ID: $SLURM_JOB_ID"
echo "Models: $MODEL_TYPES"
echo "Fold: $FOLD"
echo "======================================"
echo ""

# Convert space-separated string to array
IFS=' ' read -ra MODEL_ARRAY <<< "$MODEL_TYPES"

# Loop through each model type
for MODEL_TYPE in "${MODEL_ARRAY[@]}"; do
    echo ""
    echo "======================================"
    echo "Starting fine-tuning: $MODEL_TYPE"
    echo "Started at: $(date)"
    echo "======================================"
    echo ""
    
    # Run fine-tuning with splits file
    python3 fine_tune_classification.py \
        --splits-file $SPLITS_FILE \
        --fold $FOLD \
        --model-type $MODEL_TYPE \
        --config $CONFIG_FILE \
        --output-dir $OUTPUT_DIR
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "======================================"
        echo "✓ $MODEL_TYPE completed successfully"
        echo "Finished at: $(date)"
        echo "======================================"
    else
        echo ""
        echo "======================================"
        echo "✗ $MODEL_TYPE failed with exit code $EXIT_CODE"
        echo "Failed at: $(date)"
        echo "======================================"
        # Continue to next model instead of stopping
    fi
done

echo ""
echo "======================================"
echo "All Fine-tuning Jobs Complete"
echo "Finished at: $(date)"
echo "======================================"
