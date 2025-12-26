#!/bin/bash
set -e

# Set PYTHONPATH to the current directory to ensure tinygrad is importable
export PYTHONPATH=.

echo "Installing testing dependencies..."
# Install dependencies required for testing.
# -e installs in editable mode
# [testing] includes extra dependencies like torch, numpy, etc.
# NOTE: You can comment this out after the first run if dependencies are already installed.
pip install -e '.[testing]'

echo "Running all tests..."
# Run pytest on the 'test' directory.
# -o "python_files=..." overrides the default discovery pattern to include:
#   1. test_*.py (standard)
#   2. *_test.py (standard)
#   3. external_test_*.py (found in test/external/, usually excluded by default)
python3 -m pytest -o "python_files=test_*.py *_test.py external_test_*.py" test
