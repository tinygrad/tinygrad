import random
import numpy as np
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
    rawbuf = device.buffer.fromCPU(np.random.default_rng(seed=42).standard_normal(size=size, dtype=buf.dtype.np))
    rawbufs.append(rawbuf)

  prg = device.to_program(lin.copy())
  prg.exec(rawbufs, force_wait=True)
  ground_truth = rawbufs[0].toCPU()
  print(f"{ground_truth=}")

  # NOTE: copied from beam_search
  def tuplize_uops(uops): return tuple([(x.uop, x.dtype, tuple(x.num for x in x.vin), x.arg) for x in uops])
  seen_uops = {}

  while 1:
    # TODO: if this is too slow, we can reject sample until first valid action, instead of getting all actions first
    actions = get_linearizer_actions(lin.copy(), include_0=False)
    if not actions: break
    lin = random.choice(list(actions.values()))

    # TODO: why is there a noop action?
    tuops = tuplize_uops(lin.copy().linearize().uops)
    if tuops in seen_uops:
      break
    seen_uops[tuops] = tuple(lin.applied_opts)

    print(lin.colored_shape())
    # get a new output buffer
    outputbuffer = device.buffer(size=prod(lin.membufs[0].st.shape), dtype=lin.membufs[0].dtype)
    rawbufs[0] = outputbuffer
    prg = device.to_program(lin.copy())
    prg.exec(rawbufs, force_wait=True)
    result = rawbufs[0].toCPU()
    print(result)
    np.testing.assert_allclose(result, ground_truth, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  device = Device[Device.DEFAULT]
  ast_strs = load_worlds()
  print(f"{len(ast_strs)=}")
  # TODO: ast_strs[0] output contains nan?
  for i in range(5):
    lin = ast_str_to_lin(ast_strs[i])
    fuzz_linearizer(lin)
