import random, traceback, ctypes
from typing import List, Tuple
import numpy as np
from collections import Counter
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import get_linearizer_actions, bufs_from_lin
from tinygrad.tensor import Tensor
from tinygrad.graph import print_tree
from tinygrad.helpers import getenv, from_mv, Context
from tinygrad.device import Device, Compiled, Interpreted
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
    traceback.print_exc()
    print("COMPILE FAILED!!")
    return "COMPILE_ERROR"

  try:
    prg.exec(rawbufs, var_vals)
  except Exception:
    print(lin.ast)
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

  seen_uops = {}
  ground_truth = None
  while 1:
    if len(seen_uops) >= 10: break  # enough for this kernel
    actions = get_linearizer_actions(lin, include_0=False)
    if not actions: break
    lin = random.choice(list(actions.values()))
    if lin.applied_opts: print(f"applied action: {lin.applied_opts[-1]}")

    # stop if kernel uops repeat
    tuops = tuplize_uops(lin.linearize().uops)
    if tuops in seen_uops: break
    seen_uops[tuops] = tuple(lin.applied_opts)

    print(lin.colored_shape())
    # get a new output buffer
    rawbufs[0] = get_fuzz_rawbuf_like(rawbufs[0], zero=True)
    var_vals = {v: random.randint(v.min, v.max) for v in lin.ast.vars()}
    if (msg := run_linearizer(lin, rawbufs, var_vals)) != "PASS":
      print(f"{lin.applied_opts=}")
      return msg

    result = np.frombuffer(rawbufs[0].as_buffer(), rawbufs[0].dtype.np)
    if ground_truth is None:
      ground_truth = result
    else:
      try:
        # compare memoryviews directly
        np.testing.assert_allclose(result, ground_truth, rtol=1e-2, atol=1e-2)
      except AssertionError:
        print(lin.ast)
        traceback.print_exc()
        print(f"{lin.applied_opts=}")
        return "NOT_ALLCLOSE"
  return "PASS"


if __name__ == "__main__":
  ast_strs = load_worlds()
  print(f"{len(ast_strs)=}")
  tested = 0
  c = Counter()
  failed = []
  for i, ast in enumerate(ast_strs[:getenv("FUZZ_N", len(ast_strs))]):
    if "Variable" in ast and isinstance(device, Interpreted): continue  # no symbolic shape for Interpreted
    if "dtypes.image" in ast and Device.DEFAULT != "GPU": continue  # IMAGE is only for GPU
    print(f"testing ast {i}")
    tested += 1
    lin = ast_str_to_lin(ast)
    fuzz = str(fuzz_linearizer(lin))
    c[fuzz] += 1
    if fuzz != "PASS":
      failed.append(i)
  print(f"{tested=}")
  print(c.most_common())
  print(f"{failed=}")