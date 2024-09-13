#!/usr/bin/env python3
import pickle, json, re, os, sys, time, threading
from tinygrad.ops import UOp
from tinygrad.engine.graph import uops_colors
from http.server import HTTPServer, BaseHTTPRequestHandler

def reloader():
  mtime = os.stat(__file__).st_mtime
  while 1:
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
      with open("index.html", "rb") as f:
        ret = f.read()
    elif re.search(r'/\d+', self.path):
      self.send_response(200)
      self.send_header('Content-type', 'application/json')
      self.end_headers()
      ret = uop_to_json(uops[int(self.path.split("/")[-1])][0])
    else:
      self.send_response(404)
      ret = b""
    return self.wfile.write(ret)

if __name__ == "__main__":
  threading.Thread(target=reloader).start()
  with open("/tmp/rewrites.pkl", "rb") as f:
    uops = pickle.load(f)
  #print(uop_to_json(uops[0][0]))
  print("serving at port 8000")
  HTTPServer(('', 8000), Handler).serve_forever()
