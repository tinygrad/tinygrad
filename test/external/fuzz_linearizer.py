import random
import numpy as np
from collections import Counter
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import get_linearizer_actions
from tinygrad.graph import print_tree
from tinygrad.helpers import prod
from tinygrad.ops import Device

random.seed(42)
np.random.seed(42)


def fuzz_linearizer(lin: Linearizer):
  print_tree(lin.ast)
  print(lin.colored_shape())

  outputbuffer = device.buffer(size=prod(lin.membufs[0].st.shape), dtype=lin.membufs[0].dtype)
  rawbufs = [outputbuffer]
  # deduped? can there be a missed number?
  for buf in sorted(lin.membufs[1:], key=lambda x: x.idx):
    assert len(rawbufs) == buf.idx
    idx, valid = buf.st.expr_idxs()
    # TODO: image type and variable shape
    size = idx.max+1
    # TODO: different range for int type v.s. float type
    # TODO: assert based on L2 distance not elementwise
    rawbuf = device.buffer.fromCPU(np.random.uniform(low=-1.0, high=1.0, size=size).astype(buf.dtype.np))
    rawbufs.append(rawbuf)

  # NOTE: copied from beam_search
  def tuplize_uops(uops): return tuple([(x.uop, x.dtype, tuple(x.num for x in x.vin), x.arg) for x in uops])
  seen_uops = {}

  output = None
  while 1:
    if len(seen_uops) >= 20:
      # enough for this kernel
      break
    # TODO: if this is too slow, we can reject sample until first valid action, instead of getting all actions first
    actions = get_linearizer_actions(lin.copy(), include_0=False)
    if not actions: break
    lin = random.choice(list(actions.values()))
    if lin.applied_opts: print(f"last action: {lin.applied_opts[-1]}")

    # TODO: why is there a noop action? local a local can permute and have a loop
    tuops = tuplize_uops(lin.copy().linearize().uops)
    if tuops in seen_uops:
      break
    seen_uops[tuops] = tuple(lin.applied_opts)

    print(lin.colored_shape())
    # get a new output buffer
    rawbufs[0] = device.buffer(size=prod(lin.membufs[0].st.shape), dtype=lin.membufs[0].dtype)

    # TODO: Interpreted backend
    try:
      prg = device.to_program(lin.copy())
    except:
      import traceback; traceback.print_exc()
      print("COMPILE FAILED!!")
      return "COMPILE_ERROR"

    try:
      prg.exec(rawbufs, force_wait=True)
    except:
      print("EXEC FAILED!!")
      return "EXEC_ERROR"

    result = rawbufs[0].toCPU()
    if output is None:
      output = result
    elif not np.allclose(result, output, rtol=1e-4, atol=1e-4):
      return "NOT_ALLCLOSE"
  return "PASS"

if __name__ == "__main__":
  device = Device[Device.DEFAULT]
  ast_strs = load_worlds()
  print(f"{len(ast_strs)=}")
  tested = 0
  c = Counter()
  # TODO: ast_strs[0] output contains nan?
  for ast in ast_strs[:20]:
    tested += 1
    lin = ast_str_to_lin(ast)
    c[fuzz_linearizer(lin)] += 1
  print(f"{tested=}")
  print(c.most_common())