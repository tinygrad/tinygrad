#!/bin/bash
mypyc tinygrad/llops/ops_gpu.py tinygrad/shape/__init__.py tinygrad/ops.py tinygrad/ast.py \
  tinygrad/helpers.py tinygrad/mlops.py tinygrad/nn/__init__.py tinygrad/graph.py tinygrad/lazy.py \
  tinygrad/tensor.py 

