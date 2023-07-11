#!/bin/bash
# note: if we compile tinygrad/nn/__init__.py __dict__ no longer works, and optimizers will silently fail
mypyc --check-untyped-defs --explicit-package-bases --warn-unreachable tinygrad/shape/shapetracker.py \
  tinygrad/shape/symbolic.py tinygrad/helpers.py tinygrad/mlops.py tinygrad/tensor.py tinygrad/graph.py \
  tinygrad/DType.py tinygrad/dtypes.py tinygrad/ImageDType.py tinygrad/GlobalCounters.py tinygrad/GRAPH_MACROS.py tinygrad/LightWeakSet.py \
  tinygrad/LightWeakValueDictionary.py \
  #tinygrad/codegen/gpu.py tinygrad/runtime/ops_metal.py
  #tinygrad/codegen/ast.py
  #tinygrad/nn/__init__.py
  #tinygrad/ops.py tinygrad/runtime/ops_metal.py tinygrad/runtime/ops_gpu.py tinygrad/runtime/ops_cpu.py tinygrad/lazy.py
