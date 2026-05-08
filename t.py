import json
from tinygrad.uop.ops import UOp, Ops
from tinygrad.uop.spec import eval_pyrender, glbls
from tinygrad.runtime.autogen.amd.cdna import ins
from tinygrad.helpers import ansistrip
glbls.update(vars(ins))

with open("/tmp/debug5.json", "r") as f: rows = [json.loads(s) for s in f.read().splitlines()]

pending:str|None = None
data:list[dict] = []
for r in rows:
  if pending:
    if (v:=r.get("value")):
      if v.startswith("UOp("): uop = eval_pyrender("ast = "+r["value"])
      elif "ast = " in v: uop = eval_pyrender(r["value"])
      else: continue
      data.append({"name":pending, "uop":uop})
    pending = None
  if r.get("device", "") == "TINY" and (r["name"].startswith("Schedule") or r["name"].startswith("do_to_program")): pending = r["name"]

query = "r_2_32_2_2_4_4_8"
ast = next(v["uop"] for v in data if ansistrip(v["name"]).endswith(query))
uop_names = {v["uop"]:ansistrip(v["name"]) for v in data}

def call_label(c:UOp, ids:dict[UOp, int]) -> str:
  body = c.src[0]
  name = getattr(c.arg, "name", None) if c.arg is not None else None
  prg_name = uop_names.get(body, f"{body.op}")
  return f"CALL#{ids.setdefault(c, len(ids))} {prg_name}" + (f" call_name={name!r}" if name is not None else "") + f" args={len(c.src)-1}"

def print_ast(ast:UOp, indent:int):
  pad = "  "*indent
  for line in ast.pyrender().splitlines(): print(pad + line)

def call_deps(c:UOp) -> list[UOp]:
  deps:list[UOp] = []
  for s in c.src[1:]:
    for u in s.toposort(enter_calls=False):
      if u.op is Ops.CALL and u not in deps: deps.append(u)
  return deps

def print_call_graph(root:UOp, indent:int=0, seen:set[UOp]|None=None, ids:dict[UOp, int]|None=None):
  if seen is None: seen = set()
  if ids is None: ids = {}
  print("  "*indent + call_label(root, ids))
  print_ast(root.src[0], indent+1)
  if root in seen:
    print("  "*(indent+1) + "...")
    return
  seen.add(root)
  for dep in call_deps(root): print_call_graph(dep, indent+1, seen, ids)

for v in data:
  sink = v["uop"]
  for s in sink.toposort():
    if s.op is Ops.CALL and s.src[0] is ast:
      print(ansistrip(v["name"]))
      print_call_graph(s)
      break
