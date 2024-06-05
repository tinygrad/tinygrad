import numpy as np
import random
from dataclasses import replace
from typing import Dict, List
from tinygrad.device import Buffer, Device
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import DEBUG, colored, getenv
from tinygrad.shape.symbolic import Variable

FUZZ_ITERS = getenv("FUZZ_ITERS", 10)
FUZZ_SEED = getenv("FUZZ_SEED", -1)

class UOpsFuzzerRunner(CompiledRunner):
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False):
    assert self.p.uops is not None
    init_rawbufs, init_name = {x:x.as_buffer() for x in rawbufs}, self.p.function_name
    init_globals = {i[0]:buf for i, buf in zip(self.p.globals, rawbufs)}
    seed = FUZZ_SEED if FUZZ_SEED != -1 else random.randint(0,2**32-1)
    if DEBUG >= 1: print(colored(f"fuzzing {FUZZ_ITERS} UOps permutations for {init_name} with seed {seed}", "yellow"))

    super().__call__(rawbufs, var_vals, wait)
    ground_truth = {x:np.frombuffer(x.as_buffer(), x.dtype.np) for x in rawbufs}

    for i in range(FUZZ_ITERS):
      random.seed(seed+i)
      self.p.uops.linearize(fuzz=True)
      if DEBUG >= 6: self.p.uops.print()
      self.p = replace(self.p, name=(name:=f"{init_name}fuzz{i}"), src=Device[self.p.dname].renderer.render(name, self.p.uops), uops=self.p.uops)
      if DEBUG >= 4: print(self.p.src)
      self.lib = Device[self.p.dname].compiler.compile_cached(self.p.src)
      self.clprg = Device[self.p.dname].runtime(name, self.lib)
      for x in (rawbufs:=[init_globals[i[0]] for i in self.p.globals]): x.copyin(init_rawbufs[x])
      # verify
      super().__call__(rawbufs, var_vals, wait)
      for i, x in enumerate(rawbufs):
        try:
          np.testing.assert_allclose(np.frombuffer(x.as_buffer(), x.dtype.np), ground_truth[x], atol=1e-6, rtol=1e-6)
          if DEBUG >= 2: print(colored(name, "green"))
        except AssertionError as e:
          print(colored(name, "red"))
          raise e
