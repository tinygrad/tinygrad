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

def print_ast(ast:UOp, indent:int):
  pad = "  "*indent
  for line in ast.pyrender().splitlines(): print(pad + line)

def call_deps(c:UOp) -> list[UOp]:
  deps:list[UOp] = []
  for s in c.src[1:]:
    for u in s.toposort(enter_calls=False):
      if u.op is Ops.CALL and u not in deps: deps.append(u)
  return deps

for v in data:
  sink = v["uop"]
  for s in sink.toposort():
    if s.op is Ops.CALL and s.src[0] is ast:
      print(ansistrip(v["name"]))
      print(s)
      break
