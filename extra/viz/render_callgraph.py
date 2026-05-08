# usage: DEBUG=5 python -m tinygrad.viz.cli --json | python ./extra/viz/render_callgraph.py
import json, sys
from tinygrad.uop.ops import UOp, Ops
from tinygrad.uop.spec import eval_pyrender
from tinygrad.helpers import ansistrip, prod
from tinygrad.viz.serve import uop_to_json, VizData

# decode the viz.cli UOp stream
# TODO: using pickle is better here
pending:str|None = None
data:list[dict] = []
for line in sys.stdin:
  r = json.loads(line)
  if pending:
    if (v:=r.get("value")):
      if v.startswith("UOp("): uop = eval_pyrender("ast = "+r["value"])
      elif "ast = " in v: uop = eval_pyrender(r["value"])
      else: continue
      data.append({"name":pending, "uop":uop})
    pending = None
  if r.get("device", "") == "TINY" and (r["name"].startswith("Schedule") or r["name"].startswith("do_to_program")): pending = r["name"]

# map calls to programs / buffers
uop_names = {v["uop"]:v["name"] for v in data}
ast = next(v["uop"] for v in data if ansistrip(v["name"]).endswith(sys.argv[1])) if len(sys.argv) > 1 else None
if ast is None: ast = [v["uop"] for v in data if v["name"].startswith("do_to_program")][-4]
root = next(s for v in data for s in v["uop"].toposort() if s.op is Ops.CALL and s.src[0] is ast)
op_w, buf_w = 4, 16
viz_data = VizData() # TODO: it should not need this
bufs:list[UOp] = []
for c in root.toposort(enter_calls=False):
  if c.op is not Ops.CALL: continue
  buf_ids:dict[int, str] = {}
  for i,u in enumerate(c.src[1:]):
    while u.op is Ops.AFTER: u = u.src[0]
    # TODO: this can always be the BUFFER once it reconstructs the param -> buffer mapping
    assert u.op in {Ops.BUFFER, Ops.PARAM}
    op_name = str(u.op).split(".")[1]
    buf_ids[i] = st = f"{op_name[0].lower()}{u.arg}"
    print(f"{op_name[:3]:<{op_w}} {st:<{buf_w}} {str(u.dtype):<8} {prod(u.size())}")
  body = c.src[0]
  print(f"{'CALL':<{op_w}} {uop_names.get(body, '<unknown>').removeprefix('do_to_program for ')}")
  for u in body.toposort():
    if u.op is Ops.INDEX:
      print(f"{'I':<{op_w}} {buf_ids[(p:=u.src[0].arg)]:<{buf_w}} {p:<2} {' '.join(uop_to_json(viz_data, u)[id(u)]['label'].split('\n')[4:])}")
