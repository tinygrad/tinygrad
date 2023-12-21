#!/usr/bin/env bash

# Run linter CI locally.
set -xeuo pipefail

readonly VENV_DIR=/tmp/tiny-env
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
pip install --upgrade pip setuptools wheel
pip install -e '.[linting]'

# Lint with pylint.
PYLINT_ARGS="-efail -wfail -cfail -rfail"
# Lint modules and tests separately.
pylint --rcfile=.pylintrc --disable=all --enable=W0311,C0303 --jobs=0 --indent-string='  ' $(find tinygrad -name '*.py' | xargs) || pylint-exit $PYLINT_ARGS $?
pylint --rcfile=.pylintrc --disable=all --enable=W0311,C0303 --jobs=0 --indent-string='  ' $(find test -name '*.py' | xargs) || pylint-exit $PYLINT_ARGS $?

# Lint with ruff.
ruff . --preview

# Check types with mypy.
mypy

# Check line count.
MAX_LINE_COUNT=5000 python sz.py

# Clean up.
set +u
deactivate

echo "Linting passed. Congrats!"
