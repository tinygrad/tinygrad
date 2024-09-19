#!/usr/bin/env python3
from typing import Dict, List, Tuple
import pickle, os, sys, time, threading, webbrowser, json, difflib, contextlib
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from tinygrad.helpers import getenv
from tinygrad.ops import TrackedRewriteContext, UOp
from tinygrad.engine.graph import uops_colors, word_wrap

def uop_to_json(x:UOp) -> Dict[int, Tuple[str, str, List[int], str, str]]:
  assert isinstance(x, UOp)
  graph: Dict[int, Tuple[str, str, List[int], str, str]] = {}
  for u in x.sparents:
    label = f"{str(u.op)[5:]}{(' '+word_wrap(str(u.arg).replace(':', ''))) if u.arg is not None else ''}\n{str(u.dtype)}"
    if getenv("WITH_SHAPE"):
      with contextlib.suppress(Exception): # if the UOp is indexed already it's fine
        if u.st is not None: label += f"\n{u.st.shape}"
    graph[id(u)] = (label, str(u.dtype), [id(x) for x in u.src], str(u.arg), uops_colors.get(u.op, "#ffffff"))
  return graph

@dataclass(frozen=True)
class UOpRet:
  loc: str
  graphs: List[Tuple[UOp, UOp, UOp, UOp]]                  # snapshot of the entire AST after each rewrite
  diffs: List[Tuple[str, Tuple[str, int], List[str]]]      # the diffs for each rewrite
  extra: List[List[str]]                                   # these become code blocks in the UI

def replace_uop(base:UOp, replaces:Dict[bytes, UOp]) -> UOp:
  if (found:=replaces.get(base.key)): return found
  new_srcs = tuple(replace_uop(x, replaces) for x in base.src)
  replaces[base.key] = ret = UOp(base.op, base.dtype, new_srcs, base.arg) if new_srcs != base.src else base
  return ret

def create_graph(ctx:TrackedRewriteContext) -> UOpRet:
  uops: List[UOp] = [ctx.sink]
  graphs: List[Tuple[UOp, UOp, UOp, UOp]] = [(ctx.sink, ctx.sink, ctx.sink, ctx.sink)]
  diffs: List[Tuple[str, Tuple[str, int], List[str]]] = []
  extra: List[List[str]] = [[str(ctx.sink)]]
  seen_replaces: Dict[bytes, UOp] = {}
  for i, (first, rewritten, pattern) in enumerate(ctx.rewrites):
    if pattern.location[0].split("/")[-1] == "ops.py": continue
    # first, rewrite this UOp with the current rewrite + all the seen rewrites before this
    seen_replaces[first.key] = rewritten
    new_sink = replace_uop(uops[-1], {**seen_replaces})
    # sanity check
    assert new_sink is not uops[-1], f"rewritten sink wasn't rewritten! {i}\n{new_sink}\n{uops[-1]}"
    # update ret data
    diffs.append((str(pattern), pattern.location, list(difflib.unified_diff(str(first).splitlines(), str(rewritten).splitlines()))))
    graphs.append((new_sink, uops[-1], rewritten, first))
    uops.append(new_sink)
    extra.append([str(new_sink)])
  return UOpRet(ctx.loc, graphs, diffs, extra)

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
    elif url.path == "/graph":
      query = parse_qs(url.query)
      self.send_response(200)
      self.send_header("Content-type", "application/json")
      self.end_headers()
      with open("/tmp/rewrites.pkl", "rb") as f: contexts: List[TrackedRewriteContext] = pickle.load(f)
      rest = [x.loc for x in contexts]
      g = create_graph(contexts[int(query["uop_idx"][0])])
      ret = json.dumps(({"loc": g.loc, "graphs": [[uop_to_json(x) for x in graph] for graph in g.graphs],
                         "diffs": g.diffs, "extra": g.extra}, rest)).encode()
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
def main():
  try:
    st = time.perf_counter()
    reloader_thread = threading.Thread(target=reloader)
    reloader_thread.start()
    print("serving at port 8000")
    server_thread = threading.Thread(target=HTTPServer(('', 8000), Handler).serve_forever, daemon=True)
    server_thread.start()
    if BROWSER: webbrowser.open("http://localhost:8000")
    print(f"{(time.perf_counter()-st):.2f}s startup time")
    server_thread.join()
    reloader_thread.join()
  except KeyboardInterrupt:
    print("viz is shutting down...")
    stop_reloader.set()

if __name__ == "__main__":
  main()
