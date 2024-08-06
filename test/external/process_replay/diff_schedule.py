# create a diff of two schedule graphs
import difflib #, ocdiff
from collections import defaultdict
from typing import DefaultDict, List, Set, Tuple
from tinygrad.device import Device
from tinygrad.helpers import colored
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import LazyOp
from tinygrad.engine.realize import get_kernel

def render(ast:LazyOp) -> str:
  k = get_kernel(Device[Device.DEFAULT].renderer, ast)
  return k.to_program().src

def diff_schedule(s):
  ast_for_buf: DefaultDict[LazyBuffer, List[LazyOp]] = defaultdict(list)
  for _,prescheduled in s:
    for ps in prescheduled.values():
      for buf in ps[0]: ast_for_buf[buf].append(ps[1])
  changed = 0
  seen_diff: Set[Tuple[LazyOp, LazyOp]] = set()
  for buf, ast in ast_for_buf.items():
    if len(set(ast)) == 1: continue
    if (ast[0], ast[1]) in seen_diff: continue
    seen_diff.add((ast[0], ast[1]))
    changed += 1
    #print(ocdiff.console_diff(render(ast[0]), render(ast[1])))
    diff = list(difflib.unified_diff(render(ast[0]).splitlines(), render(ast[1]).splitlines()))
    unified_diff = "\n".join(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None) for line in diff)
    print(unified_diff)
  print(f"{changed} unique kernel{'s' if changed>1 else ''} changed")
