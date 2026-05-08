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

query = "r_2_32_2_2_4_4_8"
ast = next(v["uop"] for v in data if ansistrip(v["name"]).endswith(query))
# DO NOT CHANGE THIS TO ANSISTRIP
uop_names = {v["uop"]:v["name"] for v in data}
viz_data = VizData()

def graph_text(root:UOp) -> str:
  op_w, buf_w = 4, 6
  bufs:list[UOp] = []
  for c in root.toposort(enter_calls=False):
    if c.op is not Ops.CALL: continue
    buf_ids:dict[int, str] = {}
    for i,u in enumerate(c.src[1:]):
      while u.op is Ops.AFTER: u = u.src[0]
      assert u.op in {Ops.BUFFER, Ops.PARAM}
      buf_ids[i] = st = f"{str(u.op).split(".")[1].lower()[0]}{u.arg}"
      print(f"{u.op:<{op_w}} {st:<{buf_w}} {str(u.dtype):<8} {prod(u.size())}")
    body = c.src[0]
    print(f"{'CALL':<{op_w}} {uop_names.get(body, '<unknown>').removeprefix('do_to_program for ')}")
    for u in body.toposort():
      if u.op is Ops.INDEX:
        print(f"{'I':<{op_w}} {buf_ids[(p:=u.src[0].arg)]:<{buf_w}} {p:<2} {' '.join(uop_to_json(viz_data, u)[id(u)]['label'].split('\n')[4:])}")

for v in data:
  sink = v["uop"]
  for s in sink.toposort():
    if s.op is Ops.CALL and s.src[0] is ast:
      print(ansistrip(v["name"]))
      graph_text(s)
      break
