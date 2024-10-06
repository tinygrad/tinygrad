#!/usr/bin/env python3
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple
import pickle, os, sys, time, threading, webbrowser, json, difflib, contextlib, multiprocessing, functools
from dataclasses import asdict
from urllib.parse import parse_qs, urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from tinygrad.helpers import getenv, to_function_name
from tinygrad.ops import TrackedRewriteContext, UOp, UOps, UPat, lines
from tinygrad.engine.graph import uops_colors, word_wrap
from viz.spec import GraphRewriteDetails, GraphRewriteMetadata

def reconstruct_graph(sink:UOp, rewrites:List[Tuple[UOp, UOp, UPat]]) -> Tuple[List[UOp], List[List[str]], List[List[int]]]:
  uops: List[UOp] = [sink]
  diffs: List[List[str]] = []
  changed_nodes: List[List[int]] = []
  seen_replaces: Dict[UOp, UOp] = {}
  for i, (first, rewritten, _) in enumerate(rewrites):
    # first, rewrite this UOp with the current rewrite + all the seen rewrites before this
    seen_replaces[first] = rewritten
    new_sink = replace_uop(uops[-1], {**seen_replaces})
    # sanity check
    assert new_sink is not uops[-1], f"rewritten sink wasn't rewritten! {i}\n{new_sink}\n{uops[-1]}"
    # update ret data
    changed_nodes.append([id(x) for x in rewritten.sparents if x.op is not UOps.CONST])
    diffs.append(list(difflib.unified_diff(str(first).splitlines(), str(rewritten).splitlines())))
    uops.append(new_sink)
  return uops, diffs, changed_nodes

def uop_to_json(x:UOp) -> Dict[int, Tuple[str, str, List[int], str, str]]:
  assert isinstance(x, UOp)
  graph: Dict[int, Tuple[str, str, List[int], str, str]] = {}
  for u in x.sparents:
    if u.op is UOps.CONST: continue
    label = f"{str(u.op)[5:]}{(' '+word_wrap(str(u.arg).replace(':', ''))) if u.arg is not None else ''}\n{str(u.dtype)}"
    for idx,x in enumerate(u.src):
      if x.op is UOps.CONST: label += f"\nCONST{idx} {x.arg:g}"
    if getenv("WITH_SHAPE"):
      with contextlib.suppress(Exception): # if the UOp is indexed already it's fine
        if u.st is not None: label += f"\n{u.st.shape}"
    graph[id(u)] = (label, str(u.dtype), [id(x) for x in u.src if x.op is not UOps.CONST], str(u.arg), uops_colors.get(u.op, "#ffffff"))
  return graph

def replace_uop(base:UOp, replaces:Dict[UOp, UOp]) -> UOp:
  if (found:=replaces.get(base)) is not None: return found
  replaces[base] = ret = UOp(base.op, base.dtype, tuple(replace_uop(x, replaces) for x in base.src), base.arg)
  return ret

def load_kernels(contexts) -> DefaultDict[str, List[Tuple[GraphRewriteMetadata, TrackedRewriteContext]]]:
  kernels = defaultdict(list)
  for ctx in contexts:
    if ctx.sink.op is UOps.CONST: continue
    name = to_function_name(ctx.kernel.name) if ctx.kernel is not None else None
    upats = [(upat.location, upat.printable()) for _,_,upat in ctx.rewrites]
    kernels[name].append((GraphRewriteMetadata(ctx.loc, lines(ctx.loc[0])[ctx.loc[1]-1].strip(), name, upats), ctx))
  return kernels

@functools.lru_cache(None)
def get_src(k) -> Optional[str]: return k.to_program().src if k else None

class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    if (url:=urlparse(self.path)).path == "/favicon.svg":
      self.send_response(200)
      self.send_header("Content-type", "image/svg+xml")
      self.end_headers()
      with open(os.path.join(os.path.dirname(__file__), "favicon.svg"), "rb") as f:
        ret = f.read()
    if url.path == "/":
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
        metadata, ctx = list(kernels.values())[int(qkernel[0])][int(query["idx"][0])]
        graphs, diffs, changed_nodes = reconstruct_graph(ctx.sink, ctx.rewrites)
        ret = json.dumps(asdict(GraphRewriteDetails(**asdict(metadata), graphs=list(map(uop_to_json, graphs)),
                                                    diffs=diffs, changed_nodes=changed_nodes, kernel_code=get_src(ctx.kernel)))).encode()
      else: ret = json.dumps([list(map(lambda x:asdict(x[0]), v)) for v in kernels.values()]).encode()
    else:
      self.send_response(404)
      ret = b""
    return self.wfile.write(ret)

BROWSER = getenv("BROWSER", 1)
stop_reloader = threading.Event()
def reloader():
  mtime = os.stat(__file__).st_mtime
  while not stop_reloader.is_set():
    if mtime != os.stat(__file__).st_mtime:
      print("reloading server...")
      os.execv(sys.executable, [sys.executable] + sys.argv)
    time.sleep(0.1)

if __name__ == "__main__":
  multiprocessing.current_process().name = "VizProcess"    # disallow opening of devices
  print("*** viz is starting")
  with open("/tmp/rewrites.pkl", "rb") as f: contexts: List[TrackedRewriteContext] = pickle.load(f)
  print("*** unpickled saved rewrites")
  kernels = load_kernels(contexts)
  print("*** loaded kernels")
  server = HTTPServer(('', 8000), Handler)
  st = time.perf_counter()
  reloader_thread = threading.Thread(target=reloader)
  reloader_thread.start()
  if BROWSER: webbrowser.open("http://localhost:8000")
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("*** viz is shutting down...")
    stop_reloader.set()
