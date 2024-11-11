# the CLOUD=1 device is a process boundary between the frontend/runtime
# normally tinygrad is    frontend <-> middleware <-> runtime <-> hardware
# with CLOUD tinygrad is  frontend <-> middleware <-> CloudDevice ///HTTP/// cloud_server <-> runtime <-> hardware
# this client and server can be on the same machine, same network, or just same internet
# it should be a secure (example: no use of pickle) boundary. HTTP is used for RPC

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any, DefaultDict, List
from collections import defaultdict
import multiprocessing, functools, http.client, hashlib, json, time, os, binascii, struct, ast, contextlib
from tinygrad.dtype import dtypes
from dataclasses import dataclass, field
from tinygrad.helpers import getenv, DEBUG, fromimport, unwrap, Timing
from tinygrad.device import Compiled, Allocator, Compiler, Device, BufferOptions
from http.server import HTTPServer, BaseHTTPRequestHandler

# ***** API *****

@dataclass(frozen=True)
class BufferAlloc:
  buffer_num: int
  size: int
  options: BufferOptions

@dataclass(frozen=True)
class BufferFree: buffer_num: int

@dataclass(frozen=True)
class CopyIn:
  buffer_num: int
  datahash: str

@dataclass(frozen=True)
class CopyOut: buffer_num: int

@dataclass(frozen=True)
class ProgramAlloc:
  name: str
  datahash: str

@dataclass(frozen=True)
class ProgramFree: datahash: str

@dataclass(frozen=True)
class ProgramExec:
  datahash: str
  bufs: Tuple[int, ...]
  vals: Tuple[int, ...]
  global_size: Optional[Tuple[int, ...]]
  local_size: Optional[Tuple[int, ...]]
  wait: bool

# for safe deserialization
whitelist = {x.__name__:x for x in [BufferAlloc, BufferFree, CopyIn, CopyOut, ProgramAlloc, ProgramFree, ProgramExec, BufferOptions]}
eval_fxns = {ast.Constant: lambda x: x.value, ast.Tuple: lambda x: tuple(map(safe_eval, x.elts)), ast.List: lambda x: list(map(safe_eval, x.elts)),
  ast.Call: lambda x: safe_eval(x.func)(*[safe_eval(arg) for arg in x.args], **{kwarg.arg: safe_eval(kwarg.value) for kwarg in x.keywords}),
  ast.Name: lambda x: whitelist[x.id], ast.Attribute: lambda x: {"imagef": dtypes.imagef, "imageh": dtypes.imageh}[x.attr]}
def safe_eval(node): return eval_fxns[node.__class__](node)

# ***** backend *****

@dataclass
class CloudSession:
  programs: Dict[str, Any] = field(default_factory=dict)
  # TODO: the buffer should track this internally
  buffers: Dict[int, Tuple[Any, int, Optional[BufferOptions]]] = field(default_factory=dict)

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
    if self.path == "/drain" and method == "POST":
      # extract the hash data
      h:Dict[str, bytes] = {}
      ptr = 0
      dat = self.get_data()
      while ptr < len(dat):
        datahash, datalen = binascii.hexlify(dat[ptr:ptr+0x20]).decode(), struct.unpack("<Q", dat[ptr+0x20:ptr+0x28])[0]
        h[datahash] = dat[ptr+0x28:ptr+0x28+datalen]
        ptr += 0x28+datalen
      # the cmds are always last (currently in datahash)
      for c in safe_eval(ast.parse(h[datahash], mode="eval").body):
        #print(c)
        match c:
          case BufferAlloc():
            session.buffers[c.buffer_num] = (Device[CloudHandler.dname].allocator.alloc(c.size, c.options), c.size, c.options)
          case BufferFree():
            buf,sz,buffer_options = session.buffers[c.buffer_num]
            Device[CloudHandler.dname].allocator.free(buf,sz,buffer_options)
            del session.buffers[c.buffer_num]
          case CopyIn(): Device[CloudHandler.dname].allocator.copyin(session.buffers[c.buffer_num][0], memoryview(bytearray(h[c.datahash])))
          case CopyOut():
            buf,sz,_ = session.buffers[c.buffer_num]
            Device[CloudHandler.dname].allocator.copyout(memoryview(ret:=bytearray(sz)), buf)
          case ProgramAlloc():
            lib = Device[CloudHandler.dname].compiler.compile_cached(h[c.datahash].decode())
            session.programs[c.datahash] = Device[CloudHandler.dname].runtime(c.name, lib)
          case ProgramFree(): del session.programs[c.datahash]
          case ProgramExec():
            bufs = [session.buffers[x][0] for x in c.bufs]
            extra_args = {k:v for k,v in [("global_size", c.global_size), ("local_size", c.local_size)] if v is not None}
            r = session.programs[c.datahash](*bufs, vals=c.vals, wait=c.wait, **extra_args)
            if r is not None: ret = str(r).encode()
    elif self.path == "/renderer" and method == "GET":
      cls, args = Device[CloudHandler.dname].renderer.__reduce__()
      ret = json.dumps((cls.__module__, cls.__name__, args)).encode()
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
  # TODO: ideally we shouldn't have to deal with images here
  def _alloc(self, size:int, options:BufferOptions) -> int:
    self.device.buffer_num += 1
    self.device.q(BufferAlloc(self.device.buffer_num, size, options))
    return self.device.buffer_num
  # TODO: options should not be here in any Allocator
  def _free(self, opaque:int, options): self.device.q(BufferFree(opaque))
  def copyin(self, dest:int, src:memoryview): self.device.q(CopyIn(dest, self.device.h(bytes(src))))
  def copyout(self, dest:memoryview, src:int):
    self.device.q(CopyOut(src))
    resp = self.device.drain()
    assert len(resp) == len(dest), f"buffer length mismatch {len(resp)} != {len(dest)}"
    dest[:] = resp

class CloudProgram:
  def __init__(self, device:CloudDevice, name:str, lib:bytes):
    self.device = device
    self.datahash = self.device.h(lib)
    self.device.q(ProgramAlloc(name, self.datahash))
    super().__init__()
  def __del__(self): self.device.q(ProgramFree(self.datahash))

  def __call__(self, *bufs, global_size=None, local_size=None, vals:Tuple[int, ...]=(), wait=False):
    self.device.q(ProgramExec(self.datahash, bufs, vals, global_size, local_size, wait))
    if wait: return float(self.device.drain())

class CloudDevice(Compiled):
  def __init__(self, device:str):
    if (host:=getenv("HOST", "")) != "":
      self.host = host
    else:
      p = multiprocessing.Process(target=cloud_server, args=(6667,))
      p.daemon = True
      p.start()
      self.host = "127.0.0.1:6667"

    # state for the connection
    self.session = binascii.hexlify(os.urandom(0x10)).decode()
    self.buffer_num = 0

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
    self.reset()
    super().__init__(device, CloudAllocator(self), renderer, Compiler(), functools.partial(CloudProgram, self))

  def __del__(self):
    # TODO: this is never being called
    # TODO: should close the whole session
    with contextlib.suppress(ConnectionRefusedError, http.client.CannotSendRequest, http.client.RemoteDisconnected): self.drain()

  def reset(self):
    self._q: List[Any] = []
    self._h: Dict[str, bytes] = {}

  def h(self, d:bytes):
    # these will be deleted from self._h after they are uploaded
    binhash = hashlib.sha256(d).digest()
    self._h[datahash:=binascii.hexlify(binhash).decode()] = binhash+struct.pack("<Q", len(d))+d
    return datahash
  def q(self, x):
    if DEBUG >= 3: print(x)
    self._q.append(x)
  def drain(self):
    self.h(repr(self._q).encode())
    data = b''.join(self._h.values())
    with Timing(f"*** send {len(self._q):-3d} requests {len(self._h):-3d} hashes with len {len(data)/1024:.2f} kB in ", enabled=DEBUG>=1):
      ret = self.send("POST", "drain", data)
    self.reset()
    return ret

  def send(self, method, path, data:Optional[bytes]=None) -> bytes:
    # TODO: retry logic
    self.conn.request(method, "/"+path, data, headers={"Cookie": f"session={self.session}"})
    response = self.conn.getresponse()
    assert response.status == 200, f"failed on {method} {path}"
    return response.read()

if __name__ == "__main__": cloud_server(getenv("PORT", 6667))
