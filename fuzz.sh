#!/usr/bin/env bash

# Run fuzz CI locally.
set -xeuo pipefail

readonly VENV_DIR=/tmp/tiny-env
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
pip install --upgrade pip setuptools wheel
pip install -e '.[testing]'

# Test docs.
python docs/abstractions.py
python docs/abstractions2.py

# Test quickstart.
awk '/```python/{flag=1;next}/```/{flag=0}flag' docs/quickstart.md > quickstart.py
PYTHONPATH='.' python quickstart.py

# Run fuzz tests.
PYTHONPATH='.' python test/external/fuzz_symbolic.py
PYTHONPATH='.' python test/external/fuzz_shapetracker.py
PYTHONPATH='.' python test/external/fuzz_shapetracker_math.py

# Test ShapeTracker to_movement_ops.
python extra/to_movement_ops.py

# Use as an external package.
python -c "from tinygrad.tensor import Tensor; print(Tensor([1,2,3,4,5]))"

# Clean up.
set +u
deactivate

echo "Fuzzing passed. Congrats!"
