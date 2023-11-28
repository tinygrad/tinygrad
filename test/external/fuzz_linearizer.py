import random, traceback
import numpy as np
from collections import Counter
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import get_linearizer_actions, bufs_from_lin, tuplize_uops
from tinygrad.graph import print_tree
from tinygrad.helpers import getenv
from tinygrad.device import Device, Compiled, Interpreted
from tinygrad.lazy import vars_from_ast

device = Device[Device.DEFAULT]

def run_linearizer(lin: Linearizer, rawbufs=None, var_vals=None):
  if rawbufs is None: rawbufs = bufs_from_lin(lin)
  if var_vals is None: var_vals = {v: v.min for v in vars_from_ast(lin.ast)}

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
  rawbufs = bufs_from_lin(lin)

  seen_uops = {}
  ground_truth = None
  while 1:
    if len(seen_uops) >= 20: break  # enough for this kernel
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
    rawbufs[0] = type(rawbufs[0])(rawbufs[0].size, rawbufs[0].dtype)
    var_vals = {v: random.randint(v.min, v.max) for v in vars_from_ast(lin.ast)}
    if (msg := run_linearizer(lin, rawbufs, var_vals)) != "PASS":
      print(f"{lin.applied_opts=}")
      return msg

    result = rawbufs[0].toCPU()
    if ground_truth is None:
      ground_truth = result
    else:
      try:
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