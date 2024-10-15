#!/usr/bin/env python3
import multiprocessing, pickle, functools, difflib, os, threading, json, time, sys, webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Optional
from tinygrad.helpers import colored, getenv, to_function_name, tqdm, word_wrap
from tinygrad.ops import TrackedRewriteContext, UOp, UOps, lines
from tinygrad.codegen.kernel import Kernel

uops_colors = {UOps.ALU: "#ffffc0", UOps.LOAD: "#ffc0c0", UOps.STORE: "#c0ffc0", UOps.CONST: "#e0e0e0", UOps.VCONST: "#e0e0e0",
               UOps.DEFINE_GLOBAL: "#ffe0b0", UOps.DEFINE_LOCAL: "#ffe0d0", UOps.DEFINE_ACC: "#f0ffe0", UOps.REDUCE: "#C4A484",
               UOps.RANGE: "#c8a0e0", UOps.ASSIGN: "#e0ffc0", UOps.BARRIER: "#ff8080", UOps.IF: "#c8b0c0", UOps.SPECIAL: "#c0c0ff",
               UOps.WMMA: "#efefc0", UOps.VIEW: "#C8F9D4", UOps.REDUCE_AXIS: "#f58488"}

# ** API spec

@dataclass
class GraphRewriteMetadata:
  """Specifies metadata about a single call to graph_rewrite"""
  loc: Tuple[str, int]
  """File_path, Lineno"""
  code_line: str
  """The Python line calling graph_rewrite"""
  kernel_name: Optional[str]
  """The kernel calling graph_rewrite"""
  upats: List[Tuple[Tuple[str, int], str]]
  """List of all the applied UPats"""

@dataclass
class GraphRewriteDetails(GraphRewriteMetadata):
  """Full details about a single call to graph_rewrite"""
  graphs: List[UOp]
  """Sink at every step of graph_rewrite"""
  diffs: List[List[str]]
  """.diff style before and after of the rewritten UOp child"""
  changed_nodes: List[List[int]]
  """Nodes that changed at every step of graph_rewrite"""
  kernel_code: Optional[str]
  """The program after all rewrites"""

# ** API functions

def get_metadata(contexts:List[Tuple[Any, List[TrackedRewriteContext]]]) -> List[List[Tuple[Any, TrackedRewriteContext, GraphRewriteMetadata]]]:
  kernels: Dict[Optional[str], List[Tuple[Any, TrackedRewriteContext, GraphRewriteMetadata]]] = {}
  for k,ctxs in contexts:
    name = to_function_name(k.name) if isinstance(k, Kernel) else None
    for ctx in ctxs:
      if ctx.sink.op is UOps.CONST: continue
      upats = [(upat.location, upat.printable()) for _,_,upat in ctx.rewrites]
      if name not in kernels: kernels[name] = []
      kernels[name].append((k, ctx, GraphRewriteMetadata(ctx.loc, lines(ctx.loc[0])[ctx.loc[1]-1].strip(), name, upats)))
  return list(kernels.values())

def uop_to_json(x:UOp) -> Dict[int, Tuple[str, str, List[int], str, str]]:
  assert isinstance(x, UOp)
  graph: Dict[int, Tuple[str, str, List[int], str, str]] = {}
  for u in x.sparents:
    if u.op is UOps.CONST: continue
    label = f"{str(u.op)[5:]}{(' '+word_wrap(str(u.arg).replace(':', ''))) if u.arg is not None else ''}\n{str(u.dtype)}"
    for idx,x in enumerate(u.src):
      if x.op is UOps.CONST: label += f"\nCONST{idx} {x.arg:g}"
    graph[id(u)] = (label, str(u.dtype), [id(x) for x in u.src if x.op is not UOps.CONST], str(u.arg), uops_colors.get(u.op, "#ffffff"))
  return graph
def _replace_uop(base:UOp, replaces:Dict[UOp, UOp]) -> UOp:
  if (found:=replaces.get(base)) is not None: return found
  replaces[base] = ret = base.replace(src=tuple(_replace_uop(x, replaces) for x in base.src))
  return ret
@functools.lru_cache(None)
def _prg(k:Optional[Kernel]) -> Optional[str]: return k.to_program().src if isinstance(k, Kernel) else None
def get_details(k:Any, ctx:TrackedRewriteContext, metadata:GraphRewriteMetadata) -> GraphRewriteDetails:
  g = GraphRewriteDetails(**asdict(metadata), graphs=[ctx.sink], diffs=[], changed_nodes=[], kernel_code=_prg(k))
  replaces: Dict[UOp, UOp] = {}
  sink = ctx.sink
  for i,(u0,u1,upat) in enumerate(ctx.rewrites):
    # first, rewrite this UOp with the current rewrite + all the seen rewrites before this
    replaces[u0] = u1
    new_sink = _replace_uop(sink, {**replaces})
    # sanity check
    if new_sink is sink:
      raise AssertionError(f"rewritten sink wasn't rewritten! {i} {upat.location}")
    # update ret data
    g.changed_nodes.append([id(x) for x in u1.sparents if x.op is not UOps.CONST])
    g.diffs.append(list(difflib.unified_diff(str(u0).splitlines(), str(u1).splitlines())))
    g.graphs.append(sink:=new_sink)
  return g

# ** HTTP server

class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    if (url:=urlparse(self.path)).path == "/":
      self.send_response(200)
      self.send_header("Content-type", "text/html")
      self.end_headers()
      with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f:
        ret = f.read()
    elif url.path == "/kernels":
      self.send_response(200)
      self.send_header("Content-type", "application/json")
      self.end_headers()
      query = parse_qs(url.query)
      if (qkernel:=query.get("kernel")) is not None:
        g = get_details(*kernels[int(qkernel[0])][int(query["idx"][0])])
        ret = json.dumps({**asdict(g), "graphs": list(map(uop_to_json, g.graphs))}).encode()
      else: ret = json.dumps([list(map(lambda x:asdict(x[2]), v)) for v in kernels]).encode()
    else:
      self.send_response(404)
      ret = b""
    return self.wfile.write(ret)

# ** main loop

def reloader():
  mtime = os.stat(__file__).st_mtime
  while not stop_reloader.is_set():
    if mtime != os.stat(__file__).st_mtime:
      print("reloading server...")
      os.execv(sys.executable, [sys.executable] + sys.argv)
    time.sleep(0.1)

if __name__ == "__main__":
  stop_reloader = threading.Event()
  multiprocessing.current_process().name = "VizProcess"    # disallow opening of devices
  st = time.perf_counter()
  print("*** viz is starting")
  with open("/tmp/rewrites.pkl", "rb") as f: contexts: List[Tuple[Any, List[TrackedRewriteContext]]] = pickle.load(f)
  print("*** unpickled saved rewrites")
  kernels = get_metadata(contexts)
  if getenv("FUZZ_VIZ"):
    ret = [get_details(*args) for v in tqdm(kernels) for args in v]
    print(f"fuzzed {len(ret)} rewrite details")
  print("*** loaded kernels")
  server = HTTPServer(('', PORT:=getenv("PORT", 8000)), Handler)
  reloader_thread = threading.Thread(target=reloader)
  reloader_thread.start()
  print(f"*** started viz on http://127.0.0.1:{PORT}")
  print(colored(f"*** ready in {(time.perf_counter()-st)*1e3:4.2f}ms", "green"))
  if getenv("BROWSER", 0): webbrowser.open(f"http://127.0.0.1:{PORT}")
  try: server.serve_forever()
  except KeyboardInterrupt:
    print("*** viz is shutting down...")
    stop_reloader.set()
