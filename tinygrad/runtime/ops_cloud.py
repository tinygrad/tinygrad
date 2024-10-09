# the CLOUD=1 device is a process boundary between the frontend/runtime
# normally tinygrad is    frontend <-> middleware <-> runtime <-> hardware
# with CLOUD tinygrad is  frontend <-> middleware <-> CloudDevice ///HTTP/// cloud_server <-> runtime <-> hardware
# this client and server can be on the same machine, same network, or just same internet
# it should be a secure (example: no use of pickle) boundary. HTTP is used for RPC

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import multiprocessing, functools, urllib.request, urllib.error, hashlib, json, time, contextlib
from tinygrad.helpers import getenv, DEBUG
from tinygrad.device import Compiled, Allocator, Compiler, Device
from http.server import HTTPServer, BaseHTTPRequestHandler

# ***** backend *****
class CloudHandler(BaseHTTPRequestHandler):
  dname: str
  buffers: Dict[int, Tuple[Any, int]] = {}
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
      size = self.get_json()['size']
      CloudHandler.buffers[CloudHandler.buffer_num] = (Device[CloudHandler.dname].allocator.alloc(size), size)
      ret = str(CloudHandler.buffer_num).encode()
    elif self.path.startswith("/free"):
      buf,sz = CloudHandler.buffers[int(self.path.split("/")[-1])]
      Device[CloudHandler.dname].allocator.free(buf,sz)
    elif self.path.startswith("/buffer"):
      buf,_ = CloudHandler.buffers[int(self.path.split("/")[-1])]
      # TODO: remove bytearray to make it writable, CLANG backend needs that
      Device[CloudHandler.dname].allocator.copyin(buf, memoryview(bytearray(self.get_data())))
    elif self.path.startswith("/program"):
      name, hsh = self.path.split("/")[-2:]
      lib = self.get_data()
      assert hashlib.sha256(lib).hexdigest() == hsh
      CloudHandler.programs[(name, hsh)] = Device[CloudHandler.dname].runtime(name, lib)
    elif self.path.startswith("/exec"):
      name, hsh = self.path.split("/")[-2:]
      j = self.get_json()
      bufs = [CloudHandler.buffers[x][0] for x in j['bufs']]
      del j['bufs']
      r = CloudHandler.programs[(name, hsh)](*bufs, **j)
      if r is not None: ret = str(r).encode()
    elif self.path.startswith("/compile"):
      ret = Device[CloudHandler.dname].compiler.compile_cached(self.get_data().decode())
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
      buf,sz = CloudHandler.buffers[int(self.path.split("/")[-1])]
      ret = bytearray(sz)
      Device[CloudHandler.dname].allocator.copyout(memoryview(ret), buf)
    elif self.path.startswith("/dname"):
      ret = CloudHandler.dname.encode()
    else:
      self.send_response(404)
      self.end_headers()
    self.send_response(200)
    self.end_headers()
    return self.wfile.write(ret)

def cloud_server(port:int):
  multiprocessing.current_process().name = "MainProcess"
  CloudHandler.dname = getenv("CLOUDDEV", "METAL") if Device.DEFAULT == "CLOUD" else Device.DEFAULT
  print(f"start cloud server on {port} with device {CloudHandler.dname}")
  server = HTTPServer(('', port), CloudHandler)
  server.serve_forever()

# ***** frontend *****

class CloudCompiler(Compiler):
  def __init__(self, device:CloudDevice):
    self.device = device
    super().__init__()
  def compile(self, src:str) -> bytes: return self.device.send("compile", src.encode())

class CloudAllocator(Allocator):
  def __init__(self, device:CloudDevice):
    self.device = device
    super().__init__()
  def _alloc(self, size:int, options) -> int: return int(self.device.send("alloc", data=json.dumps({"size": size}).encode()))
  def _free(self, opaque, options):
    with contextlib.suppress(urllib.error.URLError): self.device.send(f"free/{opaque}", data=b"")
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
    super().__init__()

  def __call__(self, *bufs, global_size=None, local_size=None, vals:Tuple[int, ...]=(), wait=False):
    args = {"bufs": bufs, "vals": vals, "wait": wait}
    if global_size is not None: args["global_size"] = global_size
    if local_size is not None: args["local_size"] = local_size
    ret = self.device.send("exec/"+self.prgid, json.dumps(args).encode())
    if wait: return float(ret)

# TODO: are these abstractions right? they are not!
# CLOUD will get graph support after the JIT refactor. there's too much boilerplate now. refactor starts in toonygrad
# __call__ signature should be the same as Program, rawbufs should be splat ._buf, var_vals should be List[int]
# the /exec method should work for Graphs as well as Programs
"""
from tinygrad.device import Buffer
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.shape.symbolic import Variable
class CloudGraph(GraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    for ji in jit_cache:
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      prgid = cast(CloudProgram, prg.clprg).prgid

  def __call__(self, rawbufs: List[Buffer], var_vals: Dict[Variable, int], wait=False):
    pass
"""

from tinygrad.renderer.cstyle import MetalRenderer, AMDRenderer, ClangRenderer

# TODO: don't hardcode METAL in frontend
class CloudDevice(Compiled):
  def __init__(self, device:str):
    if (host:=getenv("HOST", "")) != "":
      self.host = f"http://{host}/"
    else:
      p = multiprocessing.Process(target=cloud_server, args=(6667,))
      p.daemon = True
      p.start()
      self.host = "http://127.0.0.1:6667/"
    if DEBUG >= 1: print(f"cloud with host {self.host}")
    while 1:
      try:
        clouddev = self.send("dname", timeout=0.1).decode()
        break
      except Exception as e:
        print(e)
        time.sleep(0.1)
    if DEBUG >= 1: print(f"remote has device {clouddev}")
    # ugh, there needs to be a better way to do this
    # TODO: how to we have BEAM be cached on the backend? this should just send a specification of the compute. rethink what goes in Renderer
    renderer = {"METAL": MetalRenderer, "AMD": AMDRenderer, "CLANG": ClangRenderer}[clouddev]()
    super().__init__(device, CloudAllocator(self), renderer, CloudCompiler(self), functools.partial(CloudProgram, self))

  def send(self, path, data:Optional[bytes]=None, timeout=60.0) -> bytes:
    # TODO: retry logic
    with urllib.request.urlopen(self.host+path, data=data if data is not None else None, timeout=timeout) as r:
      assert r.status == 200
      return r.read()

if __name__ == "__main__": cloud_server(getenv("PORT", 6667))
