# the CLOUD=1 device is a process boundary between the frontend/runtime
# normally tinygrad is    frontend <-> middleware <-> runtime <-> hardware
# with CLOUD tinygrad is  frontend <-> middleware <-> CloudDevice ///HTTP/// cloud_server <-> runtime <-> hardware
# this client and server can be on the same machine, same network, or just same internet
# it should be a secure (example: no use of pickle) boundary. HTTP is used for RPC

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import multiprocessing, functools, urllib.request, hashlib, json, time
from tinygrad.helpers import getenv
from tinygrad.device import Compiled, Allocator
from http.server import HTTPServer, BaseHTTPRequestHandler

# ***** backend *****

# TODO: don't hardcode METAL in backend
from tinygrad.runtime.ops_metal import MetalDevice

class CloudHandler(BaseHTTPRequestHandler):
  # state
  dev = MetalDevice("")

  buffers: Dict[int, Any] = {}
  buffer_num = 0
  programs: Dict[Tuple[str,str], Any] = {}

  def get_data(self):
    content_len = self.headers.get('Content-Length')
    assert content_len is not None
    return self.rfile.read(int(content_len))
  def get_json(self): return json.loads(self.get_data())

  def do_POST(self):
    #print("post", self.path)
    ret = b""
    if self.path == "/alloc":
      CloudHandler.buffer_num += 1
      CloudHandler.buffers[CloudHandler.buffer_num] = CloudHandler.dev.allocator.alloc(self.get_json()['size'])
      ret = str(CloudHandler.buffer_num).encode()
    elif self.path.startswith("/buffer"):
      buf = CloudHandler.buffers[int(self.path.split("/")[-1])]
      self.dev.allocator.copyin(buf, self.get_data())
    elif self.path.startswith("/program"):
      name, hsh = self.path.split("/")[-2:]
      lib = self.get_data()
      assert hashlib.sha256(lib).hexdigest() == hsh
      CloudHandler.programs[(name, hsh)] = CloudHandler.dev.runtime(name, lib)
    elif self.path.startswith("/exec"):
      name, hsh = self.path.split("/")[-2:]
      j = self.get_json()
      bufs = [CloudHandler.buffers[x] for x in j['bufs']]
      r = CloudHandler.programs[(name, hsh)](*bufs, global_size=j['global_size'], local_size=j['local_size'], vals=j['vals'], wait=j['wait'])
      if r is not None: ret = str(r).encode()
    else:
      self.send_response(404)
      self.end_headers()
      return 0
    self.send_response(200)
    self.end_headers()
    return self.wfile.write(ret)

  def do_GET(self):
    #print("get", self.path)
    ret = b""
    if self.path.startswith("/buffer"):
      buf = CloudHandler.buffers[int(self.path.split("/")[-1])]
      # TODO: all allocators should support as_buffer, and at least a way to get the size
      ret = bytes(self.dev.allocator.as_buffer(buf)) # type: ignore
    elif self.path.startswith("/ping"):
      pass
    else:
      self.send_response(404)
      self.end_headers()
    self.send_response(200)
    self.end_headers()
    return self.wfile.write(ret)

def cloud_server():
  print("start cloud server")
  server = HTTPServer(('', 6667), CloudHandler)
  server.serve_forever()

# ***** frontend *****

class CloudAllocator(Allocator):
  def __init__(self, device:CloudDevice): self.device = device
  def _alloc(self, size:int, options) -> int: return int(self.device.send("alloc", data=json.dumps({"size": size}).encode()))
  def copyin(self, dest:int, src:memoryview): self.device.send(f"buffer/{dest}", data=bytes(src))
  def copyout(self, dest:memoryview, src:int):
    resp = self.device.send(f"buffer/{src}")
    assert len(resp) == len(dest), f"buffer length mismatch {len(resp)} != {len(dest)}"
    dest[:] = resp

class CloudProgram:
  def __init__(self, device:CloudDevice, name:str, lib:bytes):
    self.device = device
    self.prgid = f"{name}/{hashlib.sha256(lib).hexdigest()}"
    self.device.send("program/"+self.prgid, lib)

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    args = {"bufs": bufs, "global_size": global_size, "local_size": local_size, "vals": vals, "wait": wait}
    ret = self.device.send("exec/"+self.prgid, json.dumps(args).encode())
    if wait: return float(ret)

from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.runtime.ops_metal import MetalCompiler

# TODO: don't hardcode METAL in frontend
class CloudDevice(Compiled):
  def send(self, path, data:Optional[bytes]=None) -> bytes:
    # TODO: retry logic
    with urllib.request.urlopen(self.host+path, data=data if data is not None else None, timeout=1.0) as r:
      assert r.status == 200
      return r.read()

  def __init__(self, device:str):
    if (host:=getenv("HOST", "")) != "":
      self.host = f"http://{host}/"
    else:
      p = multiprocessing.Process(target=cloud_server)
      p.daemon = True
      p.start()
      self.host = "http://127.0.0.1:6667/"
      while 1:
        try:
          self.send("ping")
          break
        except Exception as e:
          print(e)
          time.sleep(1)
    print(f"cloud with host {self.host}")
    # run the Renderer/Compiler on the frontend
    super().__init__(device, CloudAllocator(self), MetalRenderer(), MetalCompiler(), functools.partial(CloudProgram, self))
