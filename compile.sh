#!/bin/bash
mypyc --explicit-package-bases tinygrad/shape/__init__.py
  #tinygrad/runtime/ops_metal.py tinygrad/shape/__init__.py tinygrad/ops.py tinygrad/codegen/ast.py \
  #tinygrad/helpers.py tinygrad/mlops.py tinygrad/nn/__init__.py tinygrad/graph.py tinygrad/lazy.py tinygrad/tensor.py

