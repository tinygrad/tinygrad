# the CLOUD=1 device is a process boundary between the frontend/runtime
# normally tinygrad is    frontend <-> middleware <-> runtime <-> hardware
# with CLOUD tinygrad is  frontend <-> middleware <-> CloudDevice ///HTTP/// cloud_server <-> runtime <-> hardware
# this client and server can be on the same machine, same network, or just same internet
# it should be a secure (example: no use of pickle) boundary. HTTP is used for RPC

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any, DefaultDict
from collections import defaultdict
import multiprocessing, functools, http.client, hashlib, json, time, contextlib, os, binascii
from dataclasses import dataclass, field
from tinygrad.helpers import getenv, DEBUG, fromimport, unwrap
from tinygrad.device import Compiled, Allocator, Compiler, Device
from http.server import HTTPServer, BaseHTTPRequestHandler

# ***** backend *****

@dataclass
class CloudSession:
  programs: Dict[Tuple[str, str], Any] = field(default_factory=dict)
  buffers: Dict[int, Tuple[Any, int]] = field(default_factory=dict)
  buffer_num = 0

class CloudHandler(BaseHTTPRequestHandler):
  protocol_version = 'HTTP/1.1'
  dname: str
  sessions: DefaultDict[str, CloudSession] = defaultdict(CloudSession)

  def setup(self):
    super().setup()
    print(f"connection established with {self.client_address}, socket: {self.connection.fileno()}")

  def get_data(self):
    content_len = self.headers.get('Content-Length')
    assert content_len is not None
    return self.rfile.read(int(content_len))
  def get_json(self): return json.loads(self.get_data())

  def _fail(self):
    self.send_response(404)
    self.end_headers()
    return 0

  def _do(self, method):
    session = CloudHandler.sessions[unwrap(self.headers.get("Cookie")).split("session=")[1]]
    ret = b""
    if self.path == "/renderer" and method == "GET":
      cls, args = Device[CloudHandler.dname].renderer.__reduce__()
      ret = json.dumps((cls.__module__, cls.__name__, args)).encode()
    elif self.path.startswith("/alloc") and method == "POST":
      size = int(self.path.split("=")[-1])
      session.buffer_num += 1
      session.buffers[session.buffer_num] = (Device[CloudHandler.dname].allocator.alloc(size), size)
      ret = str(session.buffer_num).encode()
    elif self.path.startswith("/buffer"):
      key = int(self.path.split("/")[-1])
      buf,sz = session.buffers[key]
      if method == "GET": Device[CloudHandler.dname].allocator.copyout(memoryview(ret:=bytearray(sz)), buf)
      elif method == "PUT": Device[CloudHandler.dname].allocator.copyin(buf, memoryview(bytearray(self.get_data())))
      elif method == "DELETE":
        Device[CloudHandler.dname].allocator.free(buf,sz)
        del session.buffers[key]
      else: return self._fail()
    elif self.path.startswith("/program"):
      name, hsh = self.path.split("/")[-2:]
      if method == "PUT":
        src = self.get_data()
        assert hashlib.sha256(src).hexdigest() == hsh
        lib = Device[CloudHandler.dname].compiler.compile_cached(src.decode())
        session.programs[(name, hsh)] = Device[CloudHandler.dname].runtime(name, lib)
      elif method == "POST":
        j = self.get_json()
        bufs = [session.buffers[x][0] for x in j['bufs']]
        del j['bufs']
        r = session.programs[(name, hsh)](*bufs, **j)
        if r is not None: ret = str(r).encode()
      elif method == "DELETE": del session.programs[(name, hsh)]
      else: return self._fail()
    else: return self._fail()
    self.send_response(200)
    self.send_header('Content-Length', str(len(ret)))
    self.end_headers()
    return self.wfile.write(ret)

  def do_GET(self): return self._do("GET")
  def do_POST(self): return self._do("POST")
  def do_PUT(self): return self._do("PUT")
  def do_DELETE(self): return self._do("DELETE")

def cloud_server(port:int):
  multiprocessing.current_process().name = "MainProcess"
  CloudHandler.dname = getenv("CLOUDDEV", "METAL") if Device.DEFAULT == "CLOUD" else Device.DEFAULT
  print(f"start cloud server on {port} with device {CloudHandler.dname}")
  server = HTTPServer(('', port), CloudHandler)
  server.serve_forever()

# ***** frontend *****

class CloudAllocator(Allocator):
  def __init__(self, device:CloudDevice):
    self.device = device
    super().__init__()
  def _alloc(self, size:int, options) -> int: return int(self.device.send("POST", f"alloc?size={size}"))
  def _free(self, opaque, options):
    with contextlib.suppress(ConnectionRefusedError, http.client.CannotSendRequest, http.client.RemoteDisconnected):
      self.device.send("DELETE", f"buffer/{opaque}", data=b"")
  def copyin(self, dest:int, src:memoryview): self.device.send("PUT", f"buffer/{dest}", data=bytes(src))
  def copyout(self, dest:memoryview, src:int):
    resp = self.device.send("GET", f"buffer/{src}")
    assert len(resp) == len(dest), f"buffer length mismatch {len(resp)} != {len(dest)}"
    dest[:] = resp

class CloudProgram:
  def __init__(self, device:CloudDevice, name:str, lib:bytes):
    self.device = device
    self.prgid = f"{name}/{hashlib.sha256(lib).hexdigest()}"
    self.device.send("PUT", "program/"+self.prgid, lib)
    super().__init__()
  def __del__(self): self.device.send("DELETE", "program/"+self.prgid)

  def __call__(self, *bufs, global_size=None, local_size=None, vals:Tuple[int, ...]=(), wait=False):
    args = {"bufs": bufs, "vals": vals, "wait": wait}
    if global_size is not None: args["global_size"] = global_size
    if local_size is not None: args["local_size"] = local_size
    ret = self.device.send("POST", "program/"+self.prgid, json.dumps(args).encode())
    if wait: return float(ret)

class CloudDevice(Compiled):
  def __init__(self, device:str):
    if (host:=getenv("HOST", "")) != "":
      self.host = host
    else:
      p = multiprocessing.Process(target=cloud_server, args=(6667,))
      p.daemon = True
      p.start()
      self.host = "127.0.0.1:6667"
    self.cookie = binascii.hexlify(os.urandom(0x10)).decode()
    if DEBUG >= 1: print(f"cloud with host {self.host}")
    while 1:
      try:
        self.conn = http.client.HTTPConnection(self.host, timeout=60.0)
        clouddev = json.loads(self.send("GET", "renderer").decode())
        break
      except Exception as e:
        print(e)
        time.sleep(0.1)
    if DEBUG >= 1: print(f"remote has device {clouddev}")
    # TODO: how to we have BEAM be cached on the backend? this should just send a specification of the compute. rethink what goes in Renderer
    assert clouddev[0].startswith("tinygrad.renderer."), f"bad renderer {clouddev}"
    renderer = fromimport(clouddev[0], clouddev[1])(*clouddev[2])
    super().__init__(device, CloudAllocator(self), renderer, Compiler(), functools.partial(CloudProgram, self))

  def send(self, method, path, data:Optional[bytes]=None) -> bytes:
    # TODO: retry logic
    self.conn.request(method, "/"+path, data, headers={"Cookie": f"session={self.cookie}"})
    response = self.conn.getresponse()
    assert response.status == 200, f"failed on {method} {path}"
    return response.read()

if __name__ == "__main__": cloud_server(getenv("PORT", 6667))
