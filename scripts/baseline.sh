#!/bin/bash

# Configuration
MODE="${1:-zero_shot}"
shift         # Remove first argument (MODE)
MODELS=("$@") # All remaining arguments are model names
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=("gemma_3") # Default if no models specified
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG_FILE="$REPO_ROOT/configs_4.yaml"
OUTPUT_DIR="$REPO_ROOT/outputs"
DATASET_FILE="$REPO_ROOT/datasets_4.txt"

echo "======================================"
echo "Starting local batch processing"
echo "Mode: $MODE"
echo "Models: ${MODELS[@]}"
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
elif [ "$MODE" = "in_context" ]; then
    SCRIPT="$REPO_ROOT/in_context_classification.py"
    echo "Running in-context classification"
else
    echo "ERROR: Invalid mode '$MODE'. Must be 'zero_shot', 'fine_tune', or 'in_context'"
    exit 1
fi
echo ""

# Track overall progress
total_datasets=$(grep -v "^#" "$DATASET_FILE" | grep -v "^$" | wc -l)
total_models=${#MODELS[@]}
overall_failed=()

# Loop through each model
for MODEL_TYPE in "${MODELS[@]}"; do

    echo ""
    echo "======================================"
    echo "PROCESSING MODEL: $MODEL_TYPE"
    echo "======================================"
    echo ""

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
        echo "Model: $MODEL_TYPE [$current/$total_datasets] Processing: $dataset"
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
            overall_failed+=("$MODEL_TYPE: $dataset")
        fi
        
        echo "Finished at: $(date)"
        echo ""
        
    done < "$DATASET_FILE"

    # Model summary
    echo ""
    echo "======================================"
    echo "Model $MODEL_TYPE Complete"
    echo "======================================"
    echo "Total datasets: $total_datasets"
    echo "Successful: $((total_datasets - ${#failed_datasets[@]}))"
    echo "Failed: ${#failed_datasets[@]}"

    if [ ${#failed_datasets[@]} -gt 0 ]; then
        echo ""
        echo "Failed datasets for $MODEL_TYPE:"
        printf '  - %%s\n' "${failed_datasets[@]}"
    fi
    echo ""

done

# Overall Summary
echo ""
echo "======================================"
echo "ALL MODELS COMPLETE"
echo "======================================"
echo "Total models processed: $total_models"
echo "Total datasets per model: $total_datasets"
echo "Total runs: $((total_models * total_datasets))"
echo "Total failures: ${#overall_failed[@]}"

if [ ${#overall_failed[@]} -gt 0 ]; then
    echo ""
    echo "All failed runs:"
    printf '  - %%s\n' "${overall_failed[@]}"
fi

echo ""
echo "Finished at: $(date)"
