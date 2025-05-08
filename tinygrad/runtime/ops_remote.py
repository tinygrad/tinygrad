# the REMOTE=1 device is a process boundary between the frontend/runtime
# normally tinygrad is    frontend <-> middleware <-> runtime <-> hardware
# with REMOTE tinygrad is  frontend <-> middleware <-> RemoteDevice ///HTTP/// remote_server <-> runtime <-> hardware
# this client and server can be on the same machine, same network, or just same internet
# it should be a secure (example: no use of pickle) boundary. HTTP is used for RPC

from __future__ import annotations
from typing import Callable, Optional, Any, cast
from collections import defaultdict
from dataclasses import dataclass, field
import multiprocessing, functools, http.client, hashlib, json, time, os, binascii, struct, ast, contextlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from tinygrad.renderer import Renderer, ProgramSpec
from tinygrad.dtype import DTYPES_DICT, dtypes
from tinygrad.ops import UOp, Ops, Variable, sint
from tinygrad.helpers import getenv, DEBUG, fromimport, unwrap, Timing
from tinygrad.engine.jit import GraphRunner, ExecItem, graph_class
from tinygrad.engine.realize import CompiledRunner
from tinygrad.device import Compiled, Buffer, Allocator, Compiler, Device, BufferSpec
from tinygrad.runtime.graph.cpu import CPUGraph

# ***** API *****

class RemoteRequest: pass

@dataclass(frozen=True)
class BufferAlloc(RemoteRequest): buffer_num: int; size: int; options: BufferSpec # noqa: E702

@dataclass(frozen=True)
class BufferFree(RemoteRequest): buffer_num: int # noqa: E702

@dataclass(frozen=True)
class CopyIn(RemoteRequest): buffer_num: int; datahash: str # noqa: E702

@dataclass(frozen=True)
class CopyOut(RemoteRequest): buffer_num: int

@dataclass(frozen=True)
class ProgramAlloc(RemoteRequest): name: str; datahash: str # noqa: E702

@dataclass(frozen=True)
class ProgramFree(RemoteRequest): name: str; datahash: str # noqa: E702

@dataclass(frozen=True)
class ProgramExec(RemoteRequest):
  name: str; datahash: str; bufs: tuple[int, ...]; vals: tuple[int, ...] # noqa: E702
  global_size: Optional[tuple[int, ...]]; local_size: Optional[tuple[int, ...]]; wait: bool # noqa: E702

@dataclass(frozen=True)
class GraphComputeItem:
  name: str
  datahash: str
  bufs: tuple[int, ...]
  vars: tuple[Variable, ...]
  global_size: tuple[sint, ...]|None
  local_size: tuple[sint, ...]|None

@dataclass(frozen=True)
class GraphAlloc(RemoteRequest):
  graph_num: int
  jit_cache: tuple[GraphComputeItem, ...]
  bufs: tuple[int, ...]
  var_vals: dict[Variable, int]

@dataclass(frozen=True)
class GraphFree(RemoteRequest):
  graph_num: int

@dataclass(frozen=True)
class GraphExec(RemoteRequest):
  graph_num: int
  bufs: tuple[int, ...]
  var_vals: dict[Variable, int]
  wait: bool

# for safe deserialization
eval_globals = {x.__name__:x for x in [BufferAlloc, BufferFree, CopyIn, CopyOut, ProgramAlloc, ProgramFree, ProgramExec, GraphComputeItem,
                                       GraphAlloc, GraphFree, GraphExec, BufferSpec, UOp, Ops, dtypes]}
attribute_whitelist: dict[Any, set[str]] = {dtypes: {*DTYPES_DICT.keys(), 'imagef', 'imageh'}, Ops: {x.name for x in Ops}}
eval_fxns = {ast.Constant: lambda x: x.value, ast.Tuple: lambda x: tuple(map(safe_eval, x.elts)), ast.List: lambda x: list(map(safe_eval, x.elts)),
  ast.Dict: lambda x: {safe_eval(k):safe_eval(v) for k,v in zip(x.keys, x.values)},
  ast.Call: lambda x: safe_eval(x.func)(*[safe_eval(arg) for arg in x.args], **{kwarg.arg: safe_eval(kwarg.value) for kwarg in x.keywords}),
  ast.Name: lambda x: eval_globals[x.id], ast.Attribute: lambda x: safe_getattr(safe_eval(x.value), x.attr)}
def safe_getattr(value, attr):
  assert attr in attribute_whitelist.get(value, set()), f'getattr({value}, {repr(attr)}) is not whitelisted'
  return getattr(value, attr)
def safe_eval(node): return eval_fxns[node.__class__](node)

class BatchRequest:
  def __init__(self):
    self._q: list[RemoteRequest] = []
    self._h: dict[str, bytes] = {}
  def h(self, d:bytes) -> str:
    binhash = hashlib.sha256(d).digest()
    self._h[datahash:=binascii.hexlify(binhash).decode()] = binhash+struct.pack("<Q", len(d))+d
    return datahash
  def q(self, x:RemoteRequest): self._q.append(x)
  def serialize(self) -> bytes:
    self.h(repr(self._q).encode())
    return b''.join(self._h.values())
  def deserialize(self, dat:bytes) -> BatchRequest:
    ptr = 0
    while ptr < len(dat):
      datahash, datalen = binascii.hexlify(dat[ptr:ptr+0x20]).decode(), struct.unpack("<Q", dat[ptr+0x20:ptr+0x28])[0]
      self._h[datahash] = dat[ptr+0x28:ptr+0x28+datalen]
      ptr += 0x28+datalen
    self._q = safe_eval(ast.parse(self._h[datahash], mode="eval").body)
    return self

# ***** backend *****

@dataclass
class RemoteSession:
  programs: dict[tuple[str, str], Any] = field(default_factory=dict)
  graphs: dict[int, GraphRunner] = field(default_factory=dict)
  buffers: dict[int, Buffer] = field(default_factory=dict)

class RemoteHandler(BaseHTTPRequestHandler):
  protocol_version = 'HTTP/1.1'
  device: str
  sessions: defaultdict[str, RemoteSession] = defaultdict(RemoteSession)

  def setup(self):
    super().setup()
    print(f"connection established with {self.client_address}, socket: {self.connection.fileno()}")

  def _do(self, method):
    session = RemoteHandler.sessions[unwrap(self.headers.get("Cookie")).split("session=")[1]]
    ret, status_code = b"", 200
    if self.path == "/batch" and method == "POST":
      # TODO: streaming deserialize?
      req = BatchRequest().deserialize(self.rfile.read(int(unwrap(self.headers.get('Content-Length')))))
      # the cmds are always last (currently in datahash)
      for c in req._q:
        if DEBUG >= 1: print(c)
        match c:
          case BufferAlloc():
            assert c.buffer_num not in session.buffers, f"buffer {c.buffer_num} already allocated"
            session.buffers[c.buffer_num] = Buffer(RemoteHandler.device, c.size, dtypes.uint8, options=c.options, preallocate=True)
          case BufferFree(): del session.buffers[c.buffer_num]
          case CopyIn(): session.buffers[c.buffer_num].copyin(memoryview(bytearray(req._h[c.datahash])))
          case CopyOut(): session.buffers[c.buffer_num].copyout(memoryview(ret:=bytearray(session.buffers[c.buffer_num].nbytes)))
          case ProgramAlloc():
            lib = Device[RemoteHandler.device].compiler.compile_cached(req._h[c.datahash].decode())
            session.programs[(c.name, c.datahash)] = Device[RemoteHandler.device].runtime(c.name, lib)
          case ProgramFree(): del session.programs[(c.name, c.datahash)]
          case ProgramExec():
            bufs = [session.buffers[x]._buf for x in c.bufs]
            extra_args = {k:v for k,v in [("global_size", c.global_size), ("local_size", c.local_size)] if v is not None}
            r = session.programs[(c.name, c.datahash)](*bufs, vals=c.vals, wait=c.wait, **extra_args)
            if r is not None: ret = str(r).encode()
          case GraphAlloc():
            graph_fn: Callable = unwrap(Device[RemoteHandler.device].graph)
            def _parse_ji(gi: GraphComputeItem):
              prg = session.programs[(gi.name, gi.datahash)]
              ps = ProgramSpec(gi.name, '', RemoteHandler.device, UOp(Ops.NOOP), vars=list(gi.vars),
                               global_size=list(cast(tuple[int], gi.global_size)) if gi.global_size is not None else None,
                               local_size=list(cast(tuple[int], gi.local_size)) if gi.local_size is not None else None)
              return ExecItem(CompiledRunner(ps, precompiled=b'', prg=prg), [session.buffers[buf] for buf in gi.bufs])
            assert c.graph_num not in session.graphs, f"graph {c.graph_num} already allocated"
            session.graphs[c.graph_num] = graph_fn(list(map(_parse_ji, c.jit_cache)), [session.buffers[buf] for buf in c.bufs], c.var_vals)
          case GraphFree(): del session.graphs[c.graph_num]
          case GraphExec():
            r = session.graphs[c.graph_num]([session.buffers[buf] for buf in c.bufs], c.var_vals, wait=c.wait)
            if r is not None: ret = str(r).encode()
    elif self.path == "/properties" and method == "GET":
      cls, args = Device[RemoteHandler.device].renderer.__reduce__()
      # CPUGraph re-renders kernel from uops specified in CompiledRunner, this is not supported
      graph_cls = gt if (gt:=graph_class(Device[RemoteHandler.device])) is not CPUGraph else None
      ret = json.dumps({'remotedev': RemoteHandler.device, 'renderer': (cls.__module__, cls.__name__, args), 'graph': graph_cls is not None}).encode()
    else: status_code = 404
    self.send_response(status_code)
    self.send_header('Content-Length', str(len(ret)))
    self.end_headers()
    return self.wfile.write(ret)

  def do_GET(self): return self._do("GET")
  def do_POST(self): return self._do("POST")

def remote_server(port:int):
  RemoteHandler.device = getenv("REMOTEDEV", next(Device.get_available_devices()) if Device.DEFAULT == "REMOTE" else Device.DEFAULT)
  print(f"start remote server on {port} with device {RemoteHandler.device}")
  server = HTTPServer(('', port), RemoteHandler)
  server.serve_forever()

# ***** frontend *****

class RemoteAllocator(Allocator):
  def __init__(self, dev:RemoteDevice):
    self.device = dev
    super().__init__()
  # TODO: ideally we shouldn't have to deal with images here
  def _alloc(self, size:int, options:BufferSpec) -> int:
    self.device.buffer_num += 1
    self.device.req.q(BufferAlloc(self.device.buffer_num, size, options))
    return self.device.buffer_num
  # TODO: options should not be here in any Allocator
  def _free(self, opaque:int, options): self.device.req.q(BufferFree(opaque))
  def _copyin(self, dest:int, src:memoryview): self.device.req.q(CopyIn(dest, self.device.req.h(bytes(src))))
  def _copyout(self, dest:memoryview, src:int):
    self.device.req.q(CopyOut(src))
    resp = self.device.batch_submit()
    assert len(resp) == len(dest), f"buffer length mismatch {len(resp)} != {len(dest)}"
    dest[:] = resp

class RemoteProgram:
  def __init__(self, dev:RemoteDevice, name:str, lib:bytes):
    self.dev, self.name = dev, name
    self.datahash = self.dev.req.h(lib)
    self.dev.req.q(ProgramAlloc(self.name, self.datahash))
    super().__init__()
  def __del__(self): self.dev.req.q(ProgramFree(self.name, self.datahash))

  def __call__(self, *bufs, global_size=None, local_size=None, vals:tuple[int, ...]=(), wait=False):
    self.dev.req.q(ProgramExec(self.name, self.datahash, bufs, vals, global_size, local_size, wait))
    if wait: return float(self.dev.batch_submit())

class RemoteDevice(Compiled):
  def __init__(self, device:str):
    if (host:=getenv("HOST", "")) != "": self.host = host
    else:
      multiprocessing.Process(target=remote_server, args=(6667,), name="MainProcess", daemon=True).start()
      self.host = "127.0.0.1:6667"

    # state for the connection
    self.session = binascii.hexlify(os.urandom(0x10)).decode()
    self.buffer_num, self.graph_num = 0, 0
    self.req: BatchRequest = BatchRequest()

    if DEBUG >= 1: print(f"remote with host {self.host}")
    while 1:
      try:
        self.conn = http.client.HTTPConnection(self.host, timeout=60.0)
        self.properties = json.loads(self.send("GET", "properties").decode())
        break
      except Exception as e:
        print(e)
        time.sleep(0.1)
    if DEBUG >= 1: print(f"remote has device {self.properties['remotedev']}")
    # TODO: how to we have BEAM be cached on the backend? this should just send a specification of the compute. rethink what goes in Renderer
    renderer = self.properties['renderer']
    if not renderer[0].startswith("tinygrad.renderer.") or not renderer[1].endswith("Renderer"): raise RuntimeError(f"bad renderer {renderer}")
    renderer_class = fromimport(renderer[0], renderer[1])  # TODO: is this secure?
    if not issubclass(renderer_class, Renderer): raise RuntimeError(f"renderer isn't a Renderer {renderer}")
    graph = fromimport('tinygrad.runtime.graph.remote', 'RemoteGraph') if self.properties['graph'] else None
    super().__init__(device, RemoteAllocator(self), renderer_class(*renderer[2]), Compiler(), functools.partial(RemoteProgram, self), graph)

  def __del__(self):
    # TODO: this is never being called
    # TODO: should close the whole session
    with contextlib.suppress(ConnectionRefusedError, http.client.CannotSendRequest, http.client.RemoteDisconnected): self.batch_submit()

  def batch_submit(self):
    data = self.req.serialize()
    with Timing(f"*** send {len(self.req._q):-3d} requests {len(self.req._h):-3d} hashes with len {len(data)/1024:.2f} kB in ", enabled=DEBUG>=1):
      ret = self.send("POST", "batch", data)
    self.req = BatchRequest()
    return ret

  def send(self, method, path, data:Optional[bytes]=None) -> bytes:
    # TODO: retry logic
    self.conn.request(method, "/"+path, data, headers={"Cookie": f"session={self.session}"})
    response = self.conn.getresponse()
    assert response.status == 200, f"failed on {method} {path}"
    return response.read()

if __name__ == "__main__": remote_server(getenv("PORT", 6667))
