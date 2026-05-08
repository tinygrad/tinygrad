import json
from tinygrad.uop.ops import UOp, Ops
from tinygrad.uop.spec import eval_pyrender, glbls
from tinygrad.runtime.autogen.amd.cdna import ins
from tinygrad.helpers import ansistrip, prod
from tinygrad.viz.serve import uop_to_json, VizData
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

query = "r_5_12_2_16_3_128_4"
ast = next(v["uop"] for v in data if ansistrip(v["name"]).endswith(query))
# DO NOT CHANGE THIS TO ANSISTRIP
uop_names = {v["uop"]:v["name"] for v in data}
viz_data = VizData()

def print_ast(ast:UOp, indent:int):
  pad = "  "*indent
  for line in ast.pyrender().splitlines(): print(pad + line)

def call_deps(c:UOp) -> list[UOp]:
  deps:list[UOp] = []
  for s in c.src[1:]:
    for u in s.toposort(enter_calls=False):
      if u.op is Ops.CALL and u not in deps: deps.append(u)
  return deps

def param_buf(u:UOp) -> UOp:
  while u.op is Ops.AFTER: u = u.src[0]
  return u

def graph_text(c:UOp) -> str:
  bufs:list[UOp] = []
  calls:list[UOp] = []
  op_w, buf_w = 4, 3

  def buf_id(buf:UOp) -> str:
    if buf not in bufs:
      bufs.append(buf)
      print(f"{'BUF':<{op_w}} {f'b{len(bufs)-1}':<{buf_w}} size={prod(buf.size()):>6} dtype={buf.dtype.base.name}")
    return f"b{bufs.index(buf)}"

  def emit_call(call:UOp):
    for dep in call_deps(call): emit_call(dep)
    if call in calls: return
    calls.append(call)
    body = call.src[0]
    slot_to_buf = {i:param_buf(src) for i,src in enumerate(call.src[1:])}
    for i,src in enumerate(call.src[1:]): buf_id(param_buf(src))
    print(f"{'CALL':<{op_w}} {uop_names.get(body, 'unknown').removeprefix('do_to_program for ')}")

    def print_access(access:str, index_uop:UOp):
      index_str = ' '.join(uop_to_json(viz_data, index_uop)[id(index_uop)]["label"].split("\n")[4:])
      print(f"{access:<{op_w}} {buf_id(slot_to_buf[index_uop.src[0].arg]):<{buf_w}} index={index_str}")

    for u in body.toposort():
      if u.op is Ops.INDEX and u.dtype.base == u.dtype: print_access("R", u)
      if u.op is Ops.STORE: print_access("W", u.src[0])

  emit_call(c)

for v in data:
  sink = v["uop"]
  for s in sink.toposort():
    if s.op is Ops.CALL and s.src[0] is ast:
      print(ansistrip(v["name"]))
      graph_text(s)
      break
