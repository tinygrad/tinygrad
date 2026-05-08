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

def param_buf(u:UOp) -> UOp:
  while u.op is Ops.AFTER: u = u.src[0]
  return u

def graph_text(c:UOp) -> str:
  bufs:list[UOp] = []
  op_w, buf_w = 4, 6

  def buf_id(buf:UOp) -> str:
    if buf not in bufs:
      bufs.append(buf)
      print(f"{'BUF':<{op_w}} {f'b{len(bufs)-1}':<{buf_w}} dtype={str(buf.dtype):<8} size={prod(buf.size())}")
    return f"b{bufs.index(buf)}"

  for call in c.toposort(enter_calls=False):
    if call.op is not Ops.CALL: continue
    body = call.src[0]
    buffer_ids = {i:buf_id(param_buf(src)) for i,src in enumerate(call.src[1:])}
    print(f"{'CALL':<{op_w}} {uop_names.get(body, '<unknown>').removeprefix('do_to_program for ')}")

    for u in body.toposort():
      if u.op is Ops.INDEX:
        index_str = ' '.join(uop_to_json(viz_data, u)[id(u)]["label"].split("\n")[4:])
        param = u.src[0].arg
        print(f"{'I':<{op_w}} {buffer_ids[param]:<{buf_w}} {param:<2} {index_str}")

for v in data:
  sink = v["uop"]
  for s in sink.toposort():
    if s.op is Ops.CALL and s.src[0] is ast:
      print(ansistrip(v["name"]))
      graph_text(s)
      break
