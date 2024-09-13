#!/usr/bin/env python3
import pickle, json, re, os, sys, time, threading, webbrowser
from tinygrad.helpers import getenv
from tinygrad.ops import UOp
from tinygrad.engine.graph import uops_colors
from http.server import HTTPServer, BaseHTTPRequestHandler

stop_reloader = threading.Event()
def reloader():
  mtime = os.stat(__file__).st_mtime
  while not stop_reloader.is_set():
    if mtime != os.stat(__file__).st_mtime:
      print("reloading server...")
      os.execv(sys.executable, [sys.executable] + sys.argv)
    time.sleep(0.1)

def uop_to_json(x:UOp):
  assert isinstance(x, UOp)
  ret = {}
  for u in x.sparents: ret[id(u)] = (str(u.op)[5:], str(u.dtype), [id(x) for x in u.src], str(u.arg), uops_colors.get(u.op, "#ffffff"))
  return json.dumps(ret).encode()

class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path == "/":
      self.send_response(200)
      self.send_header('Content-type', 'text/html')
      self.end_headers()
      with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f:
        ret = f.read()
    elif re.search(r'/\d+', self.path):
      self.send_response(200)
      self.send_header('Content-type', 'application/json')
      self.end_headers()
      with open("/tmp/rewrites.pkl", "rb") as f: uops = pickle.load(f)
      ret = uop_to_json(uops[int(self.path.split("/")[-1])][0])
    else:
      self.send_response(404)
      ret = b""
    return self.wfile.write(ret)

RELOADER = getenv("RELOADER", 1)
BROWSER = getenv("BROWSER", 1)
def main():
  try:
    st = time.perf_counter()
    if RELOADER:
      reloader_thread = threading.Thread(target=reloader)
      reloader_thread.start()
    print("serving at port 8000")
    server_thread = threading.Thread(target=HTTPServer(('', 8000), Handler).serve_forever, daemon=True)
    server_thread.start()
    if BROWSER: webbrowser.open("http://localhost:8000")
    print(f"{(time.perf_counter()-st):.2f}s startup time")
    server_thread.join()
  except KeyboardInterrupt:
    print("viz is shutting down...")
    if RELOADER:
      stop_reloader.set()
      reloader_thread.join()

if __name__ == "__main__":
  main()
