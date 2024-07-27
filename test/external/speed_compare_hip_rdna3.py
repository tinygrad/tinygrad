
import time
import os
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.engine.realize import CompiledRunner
from tinygrad.engine.search import bufs_from_lin

# Add the parent directory of 'extra' to the Python path
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from tinygrad.codegen.linearizer import Linearizer
from extra.optimization.helpers import load_worlds, ast_str_to_lin

if __name__ == "__main__":
  print("Starting benchmark...")
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  dev = Device["GPU"]
  print(f"Loaded {len(ast_strs)} ASTs")

  single = getenv("NUM", -1)
  if single != -1: 
    ast_strs = ast_strs[single:single+1]
    print(f"Running single AST: {single}")

  total_time = 0
  for num, ast in enumerate(ast_strs):
    print(f"Processing AST {num}...")
    lin = ast_str_to_lin(ast)
    lin.hand_coded_optimizations()
    print("Creating CompiledRunner...")
    cpu_prg = CompiledRunner(lin.to_program())

    bufs = bufs_from_lin(lin)

    # warmup
    try:
      print("Warming up...")
      cpu_prg(bufs, {}, wait=True)
    except RuntimeError as e:
      print(f"CPU failed ast {num}: {str(e)}")
      continue

    print("Running benchmark...")
    times = []
    for _ in range(20):
      st = time.perf_counter()
      cpu_prg(bufs, {}, wait=True)
      times.append(time.perf_counter() - st)
    
    best_time = min(times)
    total_time += best_time
    print(f"{num:4d} {best_time*1e6:7.2f} us {lin.name}")

  print(f"CPU total: {total_time*1000:.2f} ms")
