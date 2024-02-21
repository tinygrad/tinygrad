import random, traceback, ctypes
from typing import List, Tuple
import numpy as np
from collections import defaultdict
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import get_linearizer_actions, bufs_from_lin
from tinygrad.tensor import Tensor
from tinygrad.features.graph import print_tree
from tinygrad.helpers import getenv, from_mv, Context
from tinygrad.device import Device, Compiled
from tinygrad.codegen.linearizer import UOp

def tuplize_uops(uops:List[UOp]) -> Tuple: return tuple([(x.uop, x.dtype, tuple(uops.index(x) for x in x.vin), x.arg) for x in uops])

device = Device[Device.DEFAULT]

def get_fuzz_rawbufs(lin):
  rawbufs = bufs_from_lin(lin)

  # Reallocate output buffer with additional area to detect out-of-bounds writes.
  RED_AREA_SIZE = 1024 if isinstance(device, Compiled) else 0
  rawbufs[0] = get_fuzz_rawbuf_like(rawbufs[0], zero=True, size=rawbufs[0].size+RED_AREA_SIZE)
  with Context(DEBUG=0):
    for rawbuf in rawbufs[1:]:
      t = Tensor.uniform((rawbuf.size,), dtype=rawbuf.dtype)
      rawbuf.copyin(t.realize().lazydata.realized.as_buffer())
  return rawbufs

def get_fuzz_rawbuf_like(rawbuf, zero=False, size=None):
  rawbuf = type(rawbuf)(Device.DEFAULT, rawbuf.size if size is None else size, rawbuf.dtype)
  if zero:
    with Context(DEBUG=0):
      mv = memoryview(bytearray(rawbuf.size * rawbuf.dtype.itemsize))
      ctypes.memset(from_mv(mv), 0, len(mv))
      rawbuf.copyin(mv)
  return rawbuf

def run_linearizer(lin: Linearizer, rawbufs=None, var_vals=None):
  if rawbufs is None: rawbufs = bufs_from_lin(lin)
  if var_vals is None: var_vals = {v: v.min for v in lin.ast.vars()}

  # TODO: images needs required_optimization
  try:
    if isinstance(device, Compiled):
      prg = device.to_program(lin)
    else:
      prg = device.get_runner(lin.ast)
  except Exception:
    print(lin.ast)
    print(lin.applied_opts)
    traceback.print_exc()
    print("COMPILE FAILED!!")
    return "COMPILE_ERROR"

  try:
    prg.exec(rawbufs, var_vals)
  except Exception:
    print(lin.ast)
    print(lin.applied_opts)
    traceback.print_exc()
    print("EXEC FAILED!!")
    return "EXEC_ERROR"

  return "PASS"


def fuzz_linearizer(lin: Linearizer):
  random.seed(42)
  np.random.seed(42)
  print_tree(lin.ast)
  print(lin.colored_shape())
  rawbufs = get_fuzz_rawbufs(lin)
  FUZZ_BEAM=getenv("FUZZ_BEAM", 0)
  seen_uops = {}
  last_lins = [lin]
  failures = defaultdict(list)

  # get baseline unoptimized output
  unoptimized = Linearizer(lin.ast)
  var_vals = {v: random.randint(v.min, v.max) for v in lin.ast.vars()}
  if run_linearizer(unoptimized, rawbufs, var_vals) != "PASS":
    failures["BASELINE_ERROR"].append((unoptimized.ast, unoptimized.applied_opts))
    return failures
  ground_truth = np.frombuffer(rawbufs[0].as_buffer(), rawbufs[0].dtype.np).copy()

  for depth in range(getenv("DEPTH", 1 if FUZZ_BEAM else 10)):
    next_lins = []
    for lin in last_lins:
      actions = get_linearizer_actions(lin, include_0=False)
      if FUZZ_BEAM: print(f"testing {lin.applied_opts=} with {len(actions)} actions")
      if not actions: continue

      test_lins = list(actions.values())
      if not FUZZ_BEAM: test_lins = [random.choice(test_lins)]

      for test_lin in test_lins:
        if not FUZZ_BEAM and test_lin.applied_opts: print(f"applied opts: {test_lin.applied_opts}")

        # stop if kernel uops repeat
        tuops = tuplize_uops(test_lin.linearize().uops)
        if tuops in seen_uops:
          continue
        seen_uops[tuops] = tuple(test_lin.applied_opts)

        if not FUZZ_BEAM: print(test_lin.colored_shape())
        # get a new output buffer
        rawbufs[0] = get_fuzz_rawbuf_like(rawbufs[0], zero=True)
        var_vals = {v: random.randint(v.min, v.max) for v in test_lin.ast.vars()}
        if (msg := run_linearizer(test_lin, rawbufs, var_vals)) != "PASS":
          failures[msg].append((test_lin.ast, test_lin.applied_opts))
          continue

        result = np.frombuffer(rawbufs[0].as_buffer(), rawbufs[0].dtype.np)
        try:
          # compare memoryviews directly
          np.testing.assert_allclose(result, ground_truth, rtol=1e-2, atol=1e-2)
        except AssertionError:
          print(test_lin.ast)
          print(test_lin.applied_opts)
          traceback.print_exc()
          print("COMPARE FAILED!!")
          failures["COMPARE_ERROR"].append((test_lin.ast, test_lin.applied_opts))
          continue
        next_lins.append(test_lin)

    last_lins = next_lins
    if FUZZ_BEAM: print(f"depth={depth} total_lins={len(last_lins)} {failures=}")
  return failures

if __name__ == "__main__":
  ast_strs = load_worlds()
  print(f"{len(ast_strs)=}")
  tested = 0
  failures = defaultdict(list)
  for i, ast in enumerate(ast_strs[:getenv("FUZZ_N", len(ast_strs))]):
    if "dtypes.image" in ast and Device.DEFAULT != "GPU": continue  # IMAGE is only for GPU
    print(f"testing ast {i}")
    tested += 1
    lin = ast_str_to_lin(ast)
    for k, v in fuzz_linearizer(lin).items():
      for f in v:
        failures[k].append(f)
  for msg, errors in failures.items():
    for i, (ast, opts) in enumerate(errors):
      print(f"{msg} {i} AST: {ast}")
      print(f"{msg} {i} OPTS: {opts}\n")
  print(f"{tested=}")
  for msg, errors in failures.items():
    print(f"{msg}: {len(errors)}")
