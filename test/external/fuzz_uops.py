import numpy as np
from dataclasses import replace
from typing import DefaultDict, Dict, List
from test.external.fuzz_schedule import find_all_toposorts
from tinygrad.codegen.uops import UOp, UOpGraph, UOps
from tinygrad.device import Buffer, Device
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import DEBUG, colored
from tinygrad.shape.symbolic import Variable

def fuzz_uops(graph:DefaultDict[UOp, List[UOp]], in_degree:DefaultDict[UOp, int]):
  paths: List[List[UOp]] = []
  # TODO: express DEFINE_ACC and loop children conditions in the graph, builtin.
  for p in find_all_toposorts(graph, in_degree):
    globals: List[UOp] = []
    assert p[-1].uop is UOps.SINK, f"didn't end with SINK, ended with {p[-1]}"
    for u in (path:=list(p[:-1])).copy():
      # TODO: add UOps.END*
      if u.uop is UOps.IF: path.append(UOp(UOps.ENDIF, None, (u,)))
      # TODO: fix DEFINE_ACC
      # globals aren't fuzzed
      if u.uop is UOps.DEFINE_GLOBAL:
        path.remove(u)
        globals.append(u)
    path = list(sorted(globals, key=lambda x:x.arg[0])) + path
    paths.append(path)
  return paths

class UOpsFuzzerRunner(CompiledRunner):
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False):
    assert self.p.uops is not None and len(self.p.uops.fuzz_paths) >= 1
    init_rawbufs, init_name = {x:x.as_buffer() for x in rawbufs}, self.p.function_name
    if DEBUG >= 1: print(colored(f"fuzzing {len(self.p.uops.fuzz_paths)} UOps permutations for {init_name}", "yellow"))

    super().__call__(rawbufs, var_vals, wait)
    ground_truth = [np.frombuffer(x.as_buffer(), x.dtype.np) for x in rawbufs]

    for i, path in enumerate(self.p.uops.fuzz_paths):
      # setup prg
      uops = UOpGraph()
      uops._uops = list(path)
      if DEBUG >= 7: uops.print()
      self.p = replace(self.p, name=(name:=f"{init_name}fuzz{i}"), src=Device[self.p.dname].renderer.render(name, uops))
      self.lib = Device[self.p.dname].compiler.compile_cached(self.p.src)
      self.clprg = Device[self.p.dname].runtime(name, self.lib)
      for x in rawbufs: x.copyin(init_rawbufs[x])
      if DEBUG >= 4: print(self.p.src)
      # verify
      super().__call__(rawbufs, var_vals, wait)
      for i, x in enumerate(rawbufs):
        try:
          np.testing.assert_allclose(np.frombuffer(x.as_buffer(), x.dtype.np), ground_truth[i])
          if DEBUG >= 2: print(colored(name, "green"))
        except AssertionError as e:
          print(colored(name, "red"))
          raise e
