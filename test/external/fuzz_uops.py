from typing import DefaultDict, List
from test.external.fuzz_schedule import find_all_toposorts
from tinygrad.codegen.uops import UOp, UOps

def fuzz_uops(graph:DefaultDict[UOp, List[UOp]], in_degree:DefaultDict[UOp, int]):
  paths = find_all_toposorts(graph, in_degree)
  # TODO: fuzz
  uops = paths[5]
  assert uops[-1].uop is UOps.SINK, f"didn't end with SINK, ended with {uops[-1]}"
  uops = uops[:-1]
  for u in uops:
    if u.uop is UOps.IF: uops = tuple((*uops, UOp(UOps.ENDIF, None, (u,))))
  return list(uops)
