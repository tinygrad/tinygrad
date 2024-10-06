#!/usr/bin/env python3
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple
import pickle, os, sys, time, threading, webbrowser, json, contextlib, multiprocessing, difflib
from dataclasses import asdict
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from tinygrad.helpers import getenv, tqdm
from tinygrad.ops import TrackedRewriteContext, UOp, UOps, lines
from tinygrad.engine.graph import uops_colors, word_wrap
from viz.spec import GraphRewriteMetadata

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

def replace_uop(u, replaces): return r if (r:=replaces.get(u)) is not None else u.replace(src=tuple(replace_uop(x, replaces) for x in u.src))

def load_kernels(contexts:List[TrackedRewriteContext]) -> List[List[Dict]]:
  kernels: DefaultDict[Optional[str], List[Dict]] = defaultdict(list)
  for ctx in tqdm(contexts):
    if len(sink_graph:=uop_to_json(ctx.sink)) == 0: continue
    if ctx.kernel is not None:
      code = (p:=ctx.kernel.to_program()).src
      name = p.function_name
    else: code, name = None, None
    g = GraphRewriteMetadata(ctx.loc, lines(ctx.loc[0])[ctx.loc[1]-1].strip(), name, code, [], [sink_graph], [], [])
    replaces: Dict[UOp, UOp] = {}
    curr_uop = ctx.sink
    for u1,u2,upat in ctx.rewrites:
      replaces[u1] = u2
      replaced_uop = replace_uop(curr_uop, replaces)
      assert curr_uop is not replaced_uop
      g.upats.append((upat.location, upat.printable()))
      g.diffs.append(list(difflib.unified_diff(str(u1).splitlines(), str(u2).splitlines())))
      g.changed_nodes.append([id(x) for x in u2.sparents if x.op is not UOps.CONST])
      g.graphs.append(uop_to_json(curr_uop:=replaced_uop))
    kernels[name].append(asdict(g))
  return list(kernels.values())

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
      ret = json.dumps(kernels).encode()
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
