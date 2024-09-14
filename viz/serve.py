#!/usr/bin/env python3
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
import pickle, re, os, sys, time, threading, webbrowser, json
from tinygrad.codegen.uopgraph import linearize_uop
from tinygrad.device import Device
from tinygrad.engine.realize import get_runner
from tinygrad.helpers import getenv
from tinygrad.ops import UOp, UOps
from tinygrad.engine.graph import uops_colors, word_wrap
from http.server import HTTPServer, BaseHTTPRequestHandler

stop_reloader = threading.Event()
def reloader():
  mtime = os.stat(__file__).st_mtime
  while not stop_reloader.is_set():
    if mtime != os.stat(__file__).st_mtime:
      print("reloading server...")
      os.execv(sys.executable, [sys.executable] + sys.argv)
    time.sleep(0.1)

def uop_to_json(x:UOp) -> Dict[int, Tuple[str, str, List[int], str, str]]:
  assert isinstance(x, UOp)
  graph: Dict[int, Tuple[str, str, List[int], str, str]] = {}
  for u in x.sparents:
    label = f"{str(u.op)[5:]}{(' '+word_wrap(str(u.arg).replace(':', ''))) if u.arg is not None else ''}\n{str(u.dtype)}"
    graph[id(u)] = (label, str(u.dtype), [id(x) for x in u.src], str(u.arg), uops_colors.get(u.op, "#ffffff"))
  return graph

def uop_to_prg(ast:UOp) -> str:
  if any(x.op is UOps.SHAPETRACKER for x in ast.parents): return get_runner(Device.DEFAULT, ast).p.src
  return Device[Device.DEFAULT].renderer.render("test", linearize_uop(ast))

@dataclass(frozen=True)
class UOpRet:
  loc: str
  graphs: List[Dict[int, Tuple[str, str, List[int], str, str]]]
  extra: List[str]

def create_graph(ctx:Tuple[Tuple[str, int], UOp, List[Tuple[UOp, UOp]]]) -> UOpRet:
  loc, start, matches = ctx
  graphs = [uop_to_json(start)]
  # TODO: add the rewritten graphs
  for _ in matches: pass
  return UOpRet(f"{loc[0].split('/')[-1]}:{loc[1]}", graphs, [str(start), uop_to_prg(start)] if start.op is UOps.SINK else [str(start)])

class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path == "/favicon.svg":
      self.send_response(200)
      self.send_header("Content-type", "image/svg+xml")
      self.end_headers()
      with open(os.path.join(os.path.dirname(__file__), "favicon.svg"), "rb") as f:
        ret = f.read()
    if self.path == "/":
      self.send_response(200)
      self.send_header("Content-type", "text/html")
      self.end_headers()
      with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f:
        ret = f.read()
    elif re.search(r'/\d+', self.path):
      self.send_response(200)
      self.send_header("Content-type", "application/json")
      self.end_headers()
      with open("/tmp/rewrites.pkl", "rb") as f: contexts = pickle.load(f)
      ret = json.dumps(tuple(asdict(create_graph(contexts[int(self.path.split("/")[-1])])).values())+(len(contexts),)).encode()
    else:
      self.send_response(404)
      ret = b""
    return self.wfile.write(ret)

BROWSER = getenv("BROWSER", 1)
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
