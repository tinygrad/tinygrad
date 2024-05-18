import heapq
from typing import DefaultDict, Dict, List, Set, Tuple
from tinygrad.codegen.uops import UOp, UOps

def fuzz_uops(graph:DefaultDict[UOp, List[UOp]], in_degree:DefaultDict[UOp, int], queue:List[Tuple[int, UOp]], push, loops_children:Dict[UOp, Set[UOp]]):
  uops = find_all_topsorts(graph, in_degree, queue, push, loops_children)[-1]
  return uops

MAX_PATHS = 2
def find_all_topsorts(graph:DefaultDict[UOp, List[UOp]], in_degree:DefaultDict[UOp, int], queue:List[Tuple[int, UOp]], push, loops_children:Dict[UOp, Set[UOp]]):
  visited: Set[UOp] = set()
  ret: List[Tuple[UOp, ...]] = []
  path: List[UOp] = []
  ifs: List[UOp] = []
  global_bufs: List[UOp] = []

  # find a path
  while queue:
    _,x = heapq.heappop(queue)
    if in_degree[x] != 0 or x in visited: continue
    # insert to path
    path.append(x)
    visited.add(x)
    for u in graph[x]:
      in_degree[u] -= 1
      push(u)

  # modify the path
  for x in path:
    if x.uop is UOps.DEFINE_ACC and len(x.vin):
      path.remove(x)
      path.insert(min(path.index(l) for l in x.vin), x)
    elif x.uop is UOps.IF: path.insert(len(path)-1, UOp(UOps.ENDIF, None, (x,)))
  for u, ss in loops_children.items():
    last_op = max(path.index(s) for s in ss)
    path.insert(last_op+1, UOp(UOps.ENDLOOP, None, (u,)))

  # add to paths
  assert path[-1].uop is UOps.SINK, f"didn't end with SINK, ended with {path[-1]}"
  ret.append(tuple(path[:-1]))
  return ret
