from collections import defaultdict, deque
from typing import DefaultDict, Dict, List, Set, Tuple
from test.external.fuzz_schedule import find_all_toposorts
from tinygrad.codegen.uops import UOp, UOps

def fuzz_uops(graph:DefaultDict[UOp, List[UOp]], in_degree:DefaultDict[UOp, int], queue:List[Tuple[int, UOp]], push, loops_children:Dict[UOp, Set[UOp]]):
  uops: List[UOp] = []
  visited: Set[UOp] = set()
  blocks: List[List[UOp]] = []
  for root, subgraph in loops_children.items():
    blocks.append(block:=[root, *subgraph])
    visited = visited.union(block)
  
  blocks.append([])
  for node in in_degree:
    if node in visited: continue
    blocks[-1].append(node)
    visited.add(node)

  for b in blocks: uops.extend(*get_paths(b))

  print("---------------")
  for u in uops: print(u)
  return uops

def get_paths(block:List[UOp]) -> List[List[UOp]]:
  graph: DefaultDict[UOp, List[UOp]] = defaultdict(list)
  in_degree: Dict[UOp, int] = {}
  for u in block:
    in_degree[u] = 0
    for x in u.vin:
      graph[x].append(u)
      in_degree[u] += 1
      if x not in in_degree: in_degree[x] = 0

  queue = deque(x for x, deg in in_degree.items() if deg == 0)
  path: List[UOp] = []
  while queue:
    n = queue.popleft()
    path.append(n)
    for x in graph[n]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  for x in path:
    if x.uop is UOps.SINK: path.remove(x)
  assert all(degree == 0 for u, degree in in_degree.items() if u.uop is not UOps.SINK)
  return [path]
