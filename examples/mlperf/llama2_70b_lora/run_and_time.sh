#!/bin/bash

# MLPerf Training v4.0+ Llama2 70B LoRA - Run and Time Script
# Based on MLPerf reference implementation pattern

set -euo pipefail

# Default values
SEED=${SEED:-42}
DATASET_PATH=${DATASET_PATH:-"./dataset"}
MODEL_PATH=${MODEL_PATH:-"./models/llama-2-70b"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs"}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_LENGTH=${MAX_LENGTH:-8192}
TARGET_ROUGE=${TARGET_ROUGE:-0.270}
GPUS=${GPUS:-1}

# MLPerf specific environment variables
export LOGMLPERF=1
export SUBMISSION_PLATFORM=${SUBMISSION_PLATFORM:-"tinybox"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log system information
echo "=== System Information ===" | tee "$OUTPUT_DIR/system_info.log"
echo "Date: $(date)" | tee -a "$OUTPUT_DIR/system_info.log"
echo "Hostname: $(hostname)" | tee -a "$OUTPUT_DIR/system_info.log"
echo "Python: $(python3 --version)" | tee -a "$OUTPUT_DIR/system_info.log"
echo "GPUs: $GPUS" | tee -a "$OUTPUT_DIR/system_info.log"

# Check for required paths
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path $DATASET_PATH does not exist"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path $MODEL_PATH does not exist"
    exit 1
fi

echo "=== Configuration ===" | tee -a "$OUTPUT_DIR/config.log"
echo "SEED: $SEED" | tee -a "$OUTPUT_DIR/config.log"
echo "DATASET_PATH: $DATASET_PATH" | tee -a "$OUTPUT_DIR/config.log"
echo "MODEL_PATH: $MODEL_PATH" | tee -a "$OUTPUT_DIR/config.log"
echo "OUTPUT_DIR: $OUTPUT_DIR" | tee -a "$OUTPUT_DIR/config.log"
echo "BATCH_SIZE: $BATCH_SIZE" | tee -a "$OUTPUT_DIR/config.log"
echo "LEARNING_RATE: $LEARNING_RATE" | tee -a "$OUTPUT_DIR/config.log"
echo "NUM_EPOCHS: $NUM_EPOCHS" | tee -a "$OUTPUT_DIR/config.log"
echo "MAX_LENGTH: $MAX_LENGTH" | tee -a "$OUTPUT_DIR/config.log"
echo "TARGET_ROUGE: $TARGET_ROUGE" | tee -a "$OUTPUT_DIR/config.log"

# Record start time
START_TIME=$(date +%s.%N)
echo "=== Training Started at $(date) ===" | tee "$OUTPUT_DIR/timing.log"

# Run training
echo "Starting Llama2 70B LoRA training..." | tee -a "$OUTPUT_DIR/timing.log"

python3 train.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --max_length "$MAX_LENGTH" \
    --target_rouge "$TARGET_ROUGE" \
    --seed "$SEED" \
    --gpus "$GPUS" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Record end time and compute duration
END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc -l)

echo "=== Training Completed at $(date) ===" | tee -a "$OUTPUT_DIR/timing.log"
echo "Duration: ${DURATION} seconds" | tee -a "$OUTPUT_DIR/timing.log"

# Convert to minutes for readability
DURATION_MIN=$(echo "scale=2; $DURATION / 60" | bc -l)
echo "Duration: ${DURATION_MIN} minutes" | tee -a "$OUTPUT_DIR/timing.log"

# Check if target was achieved
if [ -f "$OUTPUT_DIR/training_config.json" ]; then
    TARGET_ACHIEVED=$(python3 -c "
import json
with open('$OUTPUT_DIR/training_config.json', 'r') as f:
    config = json.load(f)
print(config.get('target_achieved', False))
")
    
    FINAL_ROUGE=$(python3 -c "
import json
with open('$OUTPUT_DIR/training_config.json', 'r') as f:
    config = json.load(f)
print(config.get('final_rouge_l', 0.0))
")
    
    echo "=== Results ===" | tee -a "$OUTPUT_DIR/timing.log"
    echo "Target ROUGE-L: $TARGET_ROUGE" | tee -a "$OUTPUT_DIR/timing.log"
    echo "Final ROUGE-L: $FINAL_ROUGE" | tee -a "$OUTPUT_DIR/timing.log"
    echo "Target Achieved: $TARGET_ACHIEVED" | tee -a "$OUTPUT_DIR/timing.log"
    
    if [ "$TARGET_ACHIEVED" = "True" ]; then
        echo "SUCCESS: Target achieved!" | tee -a "$OUTPUT_DIR/timing.log"
        exit 0
    else
        echo "FAILURE: Target not achieved" | tee -a "$OUTPUT_DIR/timing.log"
        exit 1
    fi
else
    echo "ERROR: Training configuration not found" | tee -a "$OUTPUT_DIR/timing.log"
    exit 1
fi