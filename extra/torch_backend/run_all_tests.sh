#!/bin/bash
# Comprehensive test runner for torch backend

set -e
export PYTHONPATH=.
cd "$(dirname "$0")/../.."

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter not found at $PYTHON_BIN. Please create the virtualenv first." >&2
  exit 1
fi

echo "======================================================================"
echo "TORCH BACKEND TEST SUITE"
echo "======================================================================"

echo -e "\n>>> Additional Backend Tests\n"
PYTHONPATH=. "$PYTHON_BIN" extra/torch_backend/test_compile.py

echo -e "\n>>> CI Tests\n"
PYTHONPATH=. FORWARD_ONLY=1 TINY_BACKEND=1 "$PYTHON_BIN" test/test_ops.py TestOps.test_add
PYTHONPATH=. "$PYTHON_BIN" extra/torch_backend/example.py
PYTHONPATH=. "$PYTHON_BIN" extra/torch_backend/test.py
PYTHONPATH=. "$PYTHON_BIN" extra/torch_backend/torch_tests.py TestTinyBackendPRIVATEUSE1.test_unary_log_tiny_float32
PYTHONPATH=. "$PYTHON_BIN" extra/torch_backend/test_inplace.py

echo -e "\n>>> Unit Tests\n"
PYTHONPATH=. CPU_LLVM=1 LLVMOPT=0 TINY_BACKEND=1 "$PYTHON_BIN" -m pytest -x test/test_ops.py 
PYTHONPATH=. CPU_LLVM=1 LLVMOPT=0 TINY_BACKEND=1 "$PYTHON_BIN" -m pytest -x test/unit/test_linalg.py -v


echo -e "\n======================================================================"
echo "ALL TESTS PASSED!"
echo "======================================================================"

# Additional tests from torchbackend CI
echo -e "\n>>> Multi-GPU Test\n"
PYTHONPATH=. CPU_LLVM=1 GPUS=4 TORCH_DEBUG=1 "$PYTHON_BIN" extra/torch_backend/test_multigpu.py

echo -e "\n>>> Beautiful MNIST Test\n"
PYTHONPATH=. STEPS=20 TARGET_EVAL_ACC_PCT=90.0 TINY_BACKEND=1 "$PYTHON_BIN" examples/other_mnist/beautiful_mnist_torch.py

echo -e "\n>>> Some Torch Tests (may fail)\n"
PYTHONPATH=. "$PYTHON_BIN" -m pytest extra/torch_backend/torch_tests.py -v --tb=no --maxfail=55

