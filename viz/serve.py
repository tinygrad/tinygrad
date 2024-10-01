#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import pickle, os, sys, time, threading, webbrowser, json, difflib, contextlib, re, multiprocessing
from dataclasses import dataclass, asdict
from urllib.parse import parse_qs, urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from tinygrad.helpers import getenv, to_function_name
from tinygrad.ops import TrackedRewriteContext, UOp, UOps, lines
from tinygrad.engine.graph import uops_colors, word_wrap

# **** /graph - detailed UOp + rewrites

@dataclass(frozen=True)
class RewriteLocation:
  filename: str
  code: str
  matcher_name: Optional[str]
  match_count: int
  @staticmethod
  def from_ctx(ctx:TrackedRewriteContext) -> RewriteLocation:
    fp, lineno = ctx.loc
    p = r"graph_rewrite\([^,]+,\s*([^>]+)\)"
    match = re.search(p, code:=lines(fp)[lineno-1].strip())
    return RewriteLocation(f"{fp.split('/')[-1]}:{lineno}", code, match.group(1).split(",")[0] if match is not None else None,
                           len(ctx.rewrites))
  def to_json(self): return asdict(self)

@dataclass(frozen=True)
class UOpRet:
  loc: RewriteLocation
  graphs: List[UOp]                                        # snapshot of the entire AST after each rewrite
  diffs: List[Tuple[str, Tuple[str, int], List[str]]]      # the diffs for each rewrite
  extra: List[List[str]]                                   # these become code blocks in the UI
  additions: List[List[int]]
  @staticmethod
  def from_ctx(ctx:TrackedRewriteContext) -> UOpRet:
    uops: List[UOp] = [ctx.sink]
    diffs: List[Tuple[str, Tuple[str, int], List[str]]] = []
    extra: List[List[str]] = [[str(ctx.sink)]]
    additions: List[List[int]] = [[]]
    seen_replaces: Dict[bytes, UOp] = {}
    for i, (first, rewritten, pattern) in enumerate(ctx.rewrites):
      # first, rewrite this UOp with the current rewrite + all the seen rewrites before this
      seen_replaces[first.key] = rewritten
      new_sink = replace_uop(uops[-1], {**seen_replaces})
      # sanity check
      assert new_sink is not uops[-1], f"rewritten sink wasn't rewritten! {i}\n{new_sink}\n{uops[-1]}"
      # update ret data
      additions.append([id(x) for x in rewritten.sparents if x.op is not UOps.CONST])
      diffs.append((pattern.printable(), pattern.location, list(difflib.unified_diff(str(first).splitlines(), str(rewritten).splitlines()))))
      uops.append(new_sink)
      extra.append([str(new_sink)])
    return UOpRet(RewriteLocation.from_ctx(ctx), uops, diffs, extra, additions)
  def to_json(self) -> Dict:
    return {**asdict(self), "loc":self.loc.to_json(), "graphs": list(map(uop_to_json, self.graphs))}

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

def replace_uop(base:UOp, replaces:Dict[bytes, UOp]) -> UOp:
  if (found:=replaces.get(base.key)) is not None: return found
  new_srcs = tuple(replace_uop(x, replaces) for x in base.src)
  replaces[base.key] = ret = UOp(base.op, base.dtype, new_srcs, base.arg) if new_srcs != base.src else base
  return ret

# **** /kernels - Overview of the kernel

@dataclass(frozen=True)
class KernelRet:
  name: str
  code: str
  ctxs: List[TrackedRewriteContext]
  def to_json(self) -> Dict: return {"name":self.name, "code":self.code, "ctxs":[RewriteLocation.from_ctx(x).to_json() for x in self.ctxs]}

def load_kernels(contexts:List[TrackedRewriteContext]) -> List[KernelRet]:
  ret: Dict[str, KernelRet] = {}
  for ctx in contexts:
    name = ctx.kernel.name if ctx.kernel is not None else "UNPARENTED"
    if ret.get(k:=to_function_name(name)) is None:
      ret[k] = KernelRet(k, ctx.kernel.to_program().src if ctx.kernel is not None else "", [])
    ret[k].ctxs.append(ctx)
  return list(ret.values())

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
      ret = json.dumps([x.to_json() for x in kernels]).encode()
    elif url.path == "/graph":
      query = parse_qs(url.query)
      self.send_response(200)
      self.send_header("Content-type", "application/json")
      self.end_headers()
      k = kernels[int(query["kernel_idx"][0])]
      g = UOpRet.from_ctx(k.ctxs[int(query["uop_idx"][0])])
      ret = json.dumps((g.to_json(), [x.loc for x in k.ctxs])).encode()
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
