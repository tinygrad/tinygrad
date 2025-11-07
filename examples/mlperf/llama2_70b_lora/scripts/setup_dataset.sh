#!/bin/bash

# Setup script for SCROLLS GovReport dataset
# MLPerf Llama2 70B LoRA Training

set -euo pipefail

DATASET_DIR=${DATASET_DIR:-"./dataset/govreport"}
BASE_URL="https://github.com/tau-nlp/scrolls/raw/main/scrolls/gov_report"

echo "Setting up SCROLLS GovReport dataset..."

# Create dataset directory
mkdir -p "$DATASET_DIR"

# Files to download
FILES=("train.json" "validation.json" "test.json")

for file in "${FILES[@]}"; do
    if [ ! -f "$DATASET_DIR/$file" ]; then
        echo "Downloading $file..."
        curl -L "$BASE_URL/$file" -o "$DATASET_DIR/$file"
        echo "Downloaded $file"
    else
        echo "$file already exists, skipping..."
    fi
done

# Verify downloads
echo "Verifying dataset files..."
for file in "${FILES[@]}"; do
    if [ -f "$DATASET_DIR/$file" ]; then
        size=$(wc -c < "$DATASET_DIR/$file")
        echo "$file: $size bytes"
        
        # Basic JSON validation
        if python3 -c "import json; json.load(open('$DATASET_DIR/$file'))" 2>/dev/null; then
            echo "$file: Valid JSON"
        else
            echo "ERROR: $file is not valid JSON"
            exit 1
        fi
    else
        echo "ERROR: $file not found after download"
        exit 1
    fi
done

# Display dataset statistics
echo "Dataset statistics:"
echo "==================="

python3 -c "
import json
from pathlib import Path

dataset_dir = Path('$DATASET_DIR')

for split in ['train', 'validation', 'test']:
    file_path = dataset_dir / f'{split}.json'
    if file_path.exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f'{split.capitalize()}: {len(data)} examples')
        
        # Sample lengths
        if len(data) > 0:
            input_lengths = [len(ex['input'].split()) for ex in data[:100]]  # Sample first 100
            output_lengths = [len(ex['output'].split()) for ex in data[:100]]
            
            print(f'  Avg input length: {sum(input_lengths) / len(input_lengths):.0f} words')
            print(f'  Avg output length: {sum(output_lengths) / len(output_lengths):.0f} words')
"

echo ""
echo "Dataset setup complete!"
echo "Dataset location: $DATASET_DIR"