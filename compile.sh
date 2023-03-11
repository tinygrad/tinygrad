#!/bin/bash
mypyc --check-untyped-defs --explicit-package-bases --warn-unreachable tinygrad/shape/__init__.py tinygrad/shape/symbolic.py \
  tinygrad/nn/__init__.py tinygrad/helpers.py tinygrad/mlops.py tinygrad/tensor.py tinygrad/graph.py \
  #tinygrad/codegen/ast.py tinygrad/codegen/gpu.py tinygrad/ops.py tinygrad/runtime/ops_metal.py
  #tinygrad/runtime/ops_metal.py tinygrad/shape/__init__.py tinygrad/ops.py tinygrad/codegen/ast.py \
  #tinygrad/helpers.py tinygrad/mlops.py tinygrad/nn/__init__.py tinygrad/graph.py tinygrad/lazy.py tinygrad/tensor.py

