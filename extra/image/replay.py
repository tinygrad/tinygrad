#!/usr/bin/env python3
import pickle, sys
from tinygrad.uop.ops import RewriteTrace
from tinygrad.codegen import get_program
from tinygrad.renderer.cstyle import QCOMRenderer

PKL = sys.argv[1] if len(sys.argv) > 1 else "rewrites.pkl"
KERNEL_NAME = sys.argv[2] if len(sys.argv) > 2 else "r_64_32_16_4_4_6_3_3_4"

with open(PKL, "rb") as f: trace: RewriteTrace = pickle.load(f)

# find the kernel by display_name
for key in trace.keys:
  if len(key.keys) >= 2 and key.keys[0] == KERNEL_NAME:
    ast = key.keys[1]  # (function_name, ast)
    print(f"Found kernel: {key.keys[0]}")
    prg = get_program(ast, QCOMRenderer())
    print(prg.src)
    break
else:
  print(f"Kernel '{KERNEL_NAME}' not found. Available kernels:")
  for key in trace.keys:
    print(f"  {key.keys[0] if key.keys else key.display_name}")
