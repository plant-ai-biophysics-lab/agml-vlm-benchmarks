# Configuration
MODE="${1:-zero_shot}"
MODEL_TYPE="${2:-gemma_3}"
CONFIG_FILE="../configs.yaml"
OUTPUT_DIR="/group/jmearlesgrp/intermediate_data/eranario/vlm-investigation/zero_shot_classification/oeq_1"
DATASET_FILE="../datasets2.txt"

echo "======================================"
echo "Starting batch processing"
echo "Mode: $MODE"
echo "Model: $MODEL_TYPE"
echo "Using Python: $(which python)"
echo "======================================"
echo ""

# Determine which script to run
if [ "$MODE" = "fine_tune" ]; then
    SCRIPT="../fine_tune_classification.py"
    echo "Running fine-tune classification"
elif [ "$MODE" = "zero_shot" ]; then
    SCRIPT="../zero_shot_classification.py"
    echo "Running zero-shot classification"
else
    echo "ERROR: Invalid mode '$MODE'. Must be 'zero_shot' or 'fine_tune'"
    exit 1
fi
echo ""

# Track progress
total_datasets=$(grep -v "^#" "$DATASET_FILE" | grep -v "^$" | wc -l)
current=0
failed_datasets=()

# Loop through each dataset
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^#.* ]] && continue
    
    # Parse dataset name and plant type (format: "dataset_name, plant_type")
    dataset=$(echo "$line" | cut -d',' -f1 | xargs)
    plant_type=$(echo "$line" | cut -d',' -f2 | xargs)
    
    # Skip if no dataset name
    [[ -z "$dataset" ]] && continue
    
    current=$((current + 1))
    
    echo ""
    echo "======================================"
    echo "[$current/$total_datasets] Processing: $dataset"
    echo "Plant type: $plant_type"
    echo "Started at: $(date)"
    echo "======================================"
    
    # Run the appropriate script
    # if python3 "$SCRIPT" \
    if python "$SCRIPT" \
        --dataset "$dataset" \
        --plant-type "$plant_type" \
        --model-type "$MODEL_TYPE" \
        --config "$CONFIG_FILE" \
        --output-dir "$OUTPUT_DIR"; then
        echo "✓ Successfully completed: $dataset"
    else
        echo "✗ FAILED: $dataset"
        failed_datasets+=("$dataset")
    fi
    
    echo "Finished at: $(date)"
    echo ""
    
done < "$DATASET_FILE"

# Summary
echo ""
echo "======================================"
echo "Batch Processing Complete"
echo "======================================"
echo "Total datasets: $total_datasets"
echo "Successful: $((total_datasets - ${#failed_datasets[@]}))"
echo "Failed: ${#failed_datasets[@]}"

if [ ${#failed_datasets[@]} -gt 0 ]; then
    echo ""
    echo "Failed datasets:"
    printf '  - %s\n' "${failed_datasets[@]}"
fi

echo ""
echo "Finished at: $(date)"