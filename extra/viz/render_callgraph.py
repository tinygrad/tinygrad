import json, sys, argparse
from tinygrad.uop.ops import UOp, Ops
from tinygrad.uop.spec import eval_pyrender
from tinygrad.helpers import ansistrip, prod, temp, TracingKey, DEBUG
from tinygrad.viz import serve as viz
from tinygrad.renderer import Renderer
from tinygrad.viz.cli import fmt_colored

parser = argparse.ArgumentParser()
parser.add_argument("--rewrites-path", type=str, metavar="PATH", help="Optional path to rewrites.pkl (default: latest rewrites)",
                    default=temp("rewrites.pkl", append_user=True))
parser.add_argument("--query", type=str, metavar="NAME", help="kernel name")
args = parser.parse_args()
data = viz.VizData(viz.load_pickle(args.rewrites_path, default=None))
uop_names = {viz._reconstruct(data, data.trace.rewrites[i][0].sink):k.display_name for i,k in enumerate(data.trace.keys) if isinstance(k.ret, Renderer)}
if not args.query:
  for v in uop_names.values(): print(v)
  exit(1)
ast = next(viz._reconstruct(data, data.trace.rewrites[i][0].sink) for i,k in enumerate(data.trace.keys) if ansistrip(k.display_name) == args.query)

def find_root(call_graphs:list[int], ast:UOp) -> tuple[int, UOp]:
  for i,c in enumerate(call_graphs):
    for u in viz._reconstruct(data, c).toposort(enter_calls=False):
      if u.op is Ops.CALL and u.src[0] is ast: return i,u

seen_bufs:set[UOp] = set()
for i,k in enumerate(data.trace.keys):
  # find call in schedule
  if not k.display_name.startswith("Schedule"): continue
  if not (call_graphs:=[s.sink for s in data.trace.rewrites[i] if s.name == "View Kernel Graph"]): continue
  if (found:=find_root(call_graphs, ast)) is None: continue
  call_idx, root = found
  # replace PARAM with BUFFER / BUFFER_VIEW
  param_replace:dict[UOp, UOp] = {}
  for s in data.trace.rewrites[i]:
    if s.name not in {"params to buffers", "memory plan"}: continue
    for m in s.matches: param_replace[viz._reconstruct(data, m[0])] = viz._reconstruct(data, m[1])
  root = root.substitute(param_replace)
  # print call
  for c in root.toposort(enter_calls=False):
    if c.op is not Ops.CALL or c.src[0].op is not Ops.SINK: continue
    arg_str:list[str] = []
    op_w, buf_w = 4, 32
    for u in c.src[1:]:
      while u.op is Ops.AFTER: u = u.src[0]
      op_name = str(u.op).split(".")[1]
      if u.op is Ops.MSTACK:
        arg_str.append(st:=f"{op_name[0].lower()} {u.device}")
      elif u.op is Ops.BUFFER_VIEW:
        arg_str.append(st:=f"{op_name[0].lower()}{u.src[0].arg}[{u.arg[0]}:{u.arg[1]}]")
      else:
        assert u.op in {Ops.BUFFER}, f"{u.op}"
        arg_str.append(st:=f"{op_name[0].lower()}{u.src[0].arg}")
      if u not in seen_bufs:
        print(f"{op_name[:3]:<{op_w}} {st:<{buf_w}} {str(u.dtype):<8} {prod(u.size())}")
        seen_bufs.add(u)
    body = c.src[0]
    print(f"{'CALL':<{op_w}} {fmt_colored(uop_names.get(body, '<unknown>').removeprefix('do_to_program for '))}")
    if DEBUG >= 3: print(body.pyrender())
    for u in body.toposort():
      if u.op is Ops.INDEX:
        print(f"{'I':<{op_w}} {arg_str[(p:=u.src[0].base.arg)]:<{buf_w}} {p:<2} {' '.join(viz.uop_to_json(data, u)[id(u)]['label'].split('\n')[4:])}")
