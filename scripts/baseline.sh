# Configuration
MODE="${1:-zero_shot}"
MODEL_TYPE="${2:-gemma_3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG_FILE="$REPO_ROOT/configs.yaml"
OUTPUT_DIR="$REPO_ROOT/outputs"
DATASET_FILE="$REPO_ROOT/datasets.txt"

echo "======================================"
echo "Starting batch processing"
echo "Mode: $MODE"
echo "Model: $MODEL_TYPE"
echo "Using Python: $(which python)"
echo "======================================"
echo ""

# Determine which script to run
if [ "$MODE" = "fine_tune" ]; then
    SCRIPT="$REPO_ROOT/fine_tune_classification.py"
    echo "Running fine-tune classification"
elif [ "$MODE" = "zero_shot" ]; then
    SCRIPT="$REPO_ROOT/zero_shot_classification.py"
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
    
    # Parse dataset name, plant type, and task (format: "dataset_name, plant_type, task")
    dataset=$(echo "$line" | cut -d',' -f1 | xargs)
    plant_type=$(echo "$line" | cut -d',' -f2 | xargs)
    task=$(echo "$line" | cut -d',' -f3 | xargs)
    
    # Skip if no dataset name
    [[ -z "$dataset" ]] && continue
    
    current=$((current + 1))
    
    echo ""
    echo "======================================"
    echo "[$current/$total_datasets] Processing: $dataset"
    echo "Plant type: $plant_type"
    echo "Task: $task"
    echo "Started at: $(date)"
    echo "======================================"
    
    # Run the appropriate script
    if python3 "$SCRIPT" \
        --dataset "$dataset" \
        --plant-type "$plant_type" \
        --task "$task" \
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