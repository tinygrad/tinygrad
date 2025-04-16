# the CLOUD=1 device is a process boundary between the frontend/runtime
# normally tinygrad is    frontend <-> middleware <-> runtime <-> hardware
# with CLOUD tinygrad is  frontend <-> middleware <-> CloudDevice ///HTTP/// cloud_server <-> runtime <-> hardware
# this client and server can be on the same machine, same network, or just same internet
# it should be a secure (example: no use of pickle) boundary. HTTP is used for RPC

from __future__ import annotations
from typing import Optional, Any, cast
from collections import defaultdict
from dataclasses import dataclass, field
import multiprocessing, functools, http.client, hashlib, json, time, os, binascii, struct, ast, contextlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from tinygrad.renderer import Renderer, ProgramSpec
from tinygrad.dtype import DTYPES_DICT, dtypes
from tinygrad.ops import UOp, Ops, Variable, sint
from tinygrad.helpers import getenv, DEBUG, fromimport, unwrap, Timing
from tinygrad.engine.jit import ExecItem, GraphRunner, graph_class
from tinygrad.engine.realize import CompiledRunner, BufferXfer
from tinygrad.device import Compiled, Buffer, LRUAllocator, Compiler, Device, BufferSpec

# ***** API *****

@dataclass(frozen=True)
class CloudRequest: idx: int

@dataclass(frozen=True)
class BufferAlloc(CloudRequest): buffer_num: int; size: int; options: BufferSpec # noqa: E702

@dataclass(frozen=True)
class BufferFree(CloudRequest): buffer_num: int # noqa: E702

@dataclass(frozen=True)
class CopyIn(CloudRequest): buffer_num: int; datahash: str # noqa: E702

@dataclass(frozen=True)
class CopyOut(CloudRequest): buffer_num: int

@dataclass(frozen=True)
class Transfer(CloudRequest): buffer_num: int; sidx: int; sbuffer_num: int # noqa: E702

@dataclass(frozen=True)
class ProgramAlloc(CloudRequest): name: str; datahash: str # noqa: E702

@dataclass(frozen=True)
class ProgramFree(CloudRequest): name: str; datahash: str # noqa: E702

@dataclass(frozen=True)
class ProgramExec(CloudRequest):
  name: str; datahash: str; bufs: tuple[int, ...]; vals: tuple[int, ...] # noqa: E702
  global_size: Optional[tuple[int, ...]]; local_size: Optional[tuple[int, ...]]; wait: bool # noqa: E702

@dataclass(frozen=True)
class GraphComputeItem:
  idx: int
  name: str
  datahash: str
  bufs: tuple[int, ...]
  vars: tuple[Variable, ...]
  outs: tuple[int, ...]
  ins: tuple[int, ...]
  global_size: tuple[sint, ...]|None
  local_size: tuple[sint, ...]|None

@dataclass(frozen=True)
class GraphAlloc(CloudRequest):
  graph_num: int
  jit_cache: tuple[GraphComputeItem|Transfer, ...]
  bufs: tuple[tuple[int, int], ...]
  var_vals: dict[Variable, int]

@dataclass(frozen=True)
class GraphFree(CloudRequest):
  graph_num: int

@dataclass(frozen=True)
class GraphExec(CloudRequest):
  graph_num: int
  bufs: tuple[tuple[int, int], ...]
  var_vals: dict[Variable, int]
  wait: bool

# for safe deserialization
eval_globals = {x.__name__:x for x in [BufferAlloc, BufferFree, CopyIn, CopyOut, Transfer, ProgramAlloc, ProgramFree, ProgramExec, GraphComputeItem,
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
    self._q: list[CloudRequest] = []
    self._h: dict[str, bytes] = {}
  def h(self, d:bytes) -> str:
    binhash = hashlib.sha256(d).digest()
    self._h[datahash:=binascii.hexlify(binhash).decode()] = binhash+struct.pack("<Q", len(d))+d
    return datahash
  def q(self, x:CloudRequest): self._q.append(x)
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
class CloudSession:
  programs: dict[tuple[str, str], Any] = field(default_factory=dict)
  graphs: dict[int, GraphRunner] = field(default_factory=dict)
  buffers: dict[int, Buffer] = field(default_factory=dict)

class CloudHandler(BaseHTTPRequestHandler):
  protocol_version = 'HTTP/1.1'
  device: str
  sessions: defaultdict[tuple[str, int], CloudSession] = defaultdict(CloudSession)

  def setup(self):
    super().setup()
    print(f"connection established with {self.client_address}, socket: {self.connection.fileno()}")

  def _do(self, method):
    ret, status_code = b"", 200
    if self.path == "/batch" and method == "POST":
      # TODO: streaming deserialize?
      req = BatchRequest().deserialize(self.rfile.read(int(unwrap(self.headers.get('Content-Length')))))
      # the cmds are always last (currently in datahash)
      for c in req._q:
        if DEBUG >= 1: print(c)
        session_token = unwrap(self.headers.get("Cookie")).split("session=")[1]
        device, session = f"{CloudHandler.device}:{c.idx}", CloudHandler.sessions[(session_token, c.idx)]
        match c:
          case BufferAlloc():
            assert c.buffer_num not in session.buffers, f"buffer {c.buffer_num} already allocated"
            session.buffers[c.buffer_num] = Buffer(device, c.size, dtypes.uint8, options=c.options, preallocate=True)
          case BufferFree(): del session.buffers[c.buffer_num]
          case CopyIn(): session.buffers[c.buffer_num].copyin(memoryview(bytearray(req._h[c.datahash])))
          case CopyOut(): session.buffers[c.buffer_num].copyout(memoryview(ret:=bytearray(session.buffers[c.buffer_num].nbytes)))
          case Transfer():
            dbuf, sbuf = session.buffers[c.buffer_num], CloudHandler.sessions[(session_token, c.sidx)].buffers[c.sbuffer_num]
            assert dbuf.nbytes == sbuf.nbytes, f"{dbuf.nbytes} != {sbuf.nbytes}"
            allocator = Device[device].allocator
            assert hasattr(allocator, '_transfer'), f"Device {device} doesn't support transfers"
            allocator._transfer(dbuf._buf, sbuf._buf, dbuf.nbytes, dest_dev=Device[device], src_dev=Device[f"{CloudHandler.device}:{c.sidx}"])
          case ProgramAlloc():
            lib = Device[device].compiler.compile_cached(req._h[c.datahash].decode())
            session.programs[(c.name, c.datahash)] = Device[device].runtime(c.name, lib)
          case ProgramFree(): del session.programs[(c.name, c.datahash)]
          case ProgramExec():
            bufs = [session.buffers[x]._buf for x in c.bufs]
            extra_args = {k:v for k,v in [("global_size", c.global_size), ("local_size", c.local_size)] if v is not None}
            r = session.programs[(c.name, c.datahash)](*bufs, vals=c.vals, wait=c.wait, **extra_args)
            if r is not None: ret = str(r).encode()
          case GraphAlloc():
            graph_cls: type[GraphRunner] = unwrap(Device[device].graph)
            def _parse_ji(gi: GraphComputeItem|Transfer):
              match gi:
                case GraphComputeItem():
                  gi_session = CloudHandler.sessions[(session_token, gi.idx)]
                  prg = gi_session.programs[(gi.name, gi.datahash)]
                  ps = ProgramSpec(gi.name, '', f"{CloudHandler.device}:{gi.idx}", UOp(Ops.NOOP),
                                   vars=list(gi.vars), outs=list(gi.outs), ins=list(gi.ins),
                                   global_size=list(cast(tuple[int], gi.global_size)) if gi.global_size is not None else None,
                                   local_size=list(cast(tuple[int], gi.local_size)) if gi.local_size is not None else None)
                  return ExecItem(CompiledRunner(ps, precompiled=b'', prg=prg), [gi_session.buffers[buf] for buf in gi.bufs])
                case Transfer():
                  dbuf = CloudHandler.sessions[(session_token, gi.idx)].buffers[gi.buffer_num]
                  sbuf = CloudHandler.sessions[(session_token, gi.sidx)].buffers[gi.sbuffer_num]
                  assert dbuf.nbytes == sbuf.nbytes, f"{dbuf.nbytes} != {sbuf.nbytes}"
                  return ExecItem(BufferXfer(dbuf.nbytes, dbuf.device, sbuf.device), [dbuf, sbuf])
            bufs = [CloudHandler.sessions[(session_token, idx)].buffers[buf] for idx,buf in c.bufs]
            assert c.graph_num not in session.graphs, f"graph {c.graph_num} already allocated"
            session.graphs[c.graph_num] = graph_cls([_parse_ji(ji) for ji in c.jit_cache], bufs, c.var_vals)
          case GraphFree(): del session.graphs[c.graph_num]
          case GraphExec():
            r = session.graphs[c.graph_num]([CloudHandler.sessions[(session_token, idx)].buffers[buf] for idx,buf in c.bufs], c.var_vals, wait=c.wait)
            if r is not None: ret = str(r).encode()
    elif self.path == "/properties" and method == "GET":
      dev = Device[CloudHandler.device]
      cls, args = dev.renderer.__reduce__()
      transfer = hasattr(dev.allocator, '_transfer')
      graph = dev.graph is not None
      graph_multi = graph and graph_class(dev).supports_multi
      ret = json.dumps({'renderer': (cls.__module__, cls.__name__, args), 'transfer': transfer, 'graph': graph, 'graph_multi': graph_multi}).encode()
    else: status_code = 404
    self.send_response(status_code)
    self.send_header('Content-Length', str(len(ret)))
    self.end_headers()
    return self.wfile.write(ret)

  def do_GET(self): return self._do("GET")
  def do_POST(self): return self._do("POST")

def cloud_server(port:int):
  CloudHandler.device = getenv("CLOUDDEV", next(Device.get_available_devices()) if Device.DEFAULT == "CLOUD" else Device.DEFAULT)
  print(f"start cloud server on {port} with device {CloudHandler.device}")
  server = HTTPServer(('', port), CloudHandler)
  server.serve_forever()

# ***** frontend *****

class CloudAllocator(LRUAllocator):
  def __init__(self, dev:CloudDevice):
    self.dev = dev
    if dev.conn.properties['transfer']: self._transfer = self._transfer_impl
    super().__init__()
  # TODO: ideally we shouldn't have to deal with images here
  def _alloc(self, size:int, options:BufferSpec) -> int:
    self.dev.buffer_num += 1
    self.dev.conn.req.q(BufferAlloc(self.dev.idx, self.dev.buffer_num, size, options))
    return self.dev.buffer_num
  # TODO: options should not be here in any Allocator
  def _free(self, opaque:int, options): self.dev.conn.req.q(BufferFree(self.dev.idx, opaque))
  def _copyin(self, dest:int, src:memoryview): self.dev.conn.req.q(CopyIn(self.dev.idx, dest, self.dev.conn.req.h(bytes(src))))
  def _copyout(self, dest:memoryview, src:int):
    self.dev.conn.req.q(CopyOut(self.dev.idx, src))
    resp = self.dev.conn.batch_submit()
    assert len(resp) == len(dest), f"buffer length mismatch {len(resp)} != {len(dest)}"
    dest[:] = resp
  def _transfer_impl(self, dest, src, sz, src_dev, dest_dev):
    if src_dev.conn == dest_dev.conn:
      dest_dev.conn.req.q(Transfer(dest_dev.idx, dest, src_dev.idx, src))
    else:
      src_dev.allocator._copyout(tmp:=memoryview(bytearray(sz)), src)
      dest_dev.allocator._copyin(dest, tmp)

class CloudProgram:
  def __init__(self, dev:CloudDevice, name:str, lib:bytes):
    self.dev, self.name = dev, name
    self.datahash = self.dev.conn.req.h(lib)
    self.dev.conn.req.q(ProgramAlloc(self.dev.idx, self.name, self.datahash))
    super().__init__()
  def __del__(self): self.dev.conn.req.q(ProgramFree(self.dev.idx, self.name, self.datahash))

  def __call__(self, *bufs, global_size=None, local_size=None, vals:tuple[int, ...]=(), wait=False):
    self.dev.conn.req.q(ProgramExec(self.dev.idx, self.name, self.datahash, bufs, vals, global_size, local_size, wait))
    if wait: return float(self.dev.conn.batch_submit())

@functools.cache
class CloudConnection:
  def __init__(self, host:str):
    self.session = binascii.hexlify(os.urandom(0x10)).decode()
    self.req = BatchRequest()
    while True:
      try:
        self.conn = http.client.HTTPConnection(host, timeout=getenv("CLOUD_TIMEOUT", 300.0))
        self.properties = json.loads(self.send("GET", "properties").decode())
        break
      except Exception as e:
        print(e)
        time.sleep(0.1)

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

class CloudDevice(Compiled):
  def __init__(self, device:str):
    # per-connection state
    parts = device.split(':')[1:]
    if len(parts) > 3: raise RuntimeError(f"Too many ':'s in {device}")
    self.host: str = ':'.join(parts[0:2]).strip() if len(parts) >= 2 else CloudDevice.start_local()
    self.idx: int = int(parts[-1]) if len(parts) % 2 == 1 else 0
    self.conn: CloudConnection = CloudConnection(self.host)
    # per-device state
    self.buffer_num: int = 0
    self.graph_num: int = 0

    if DEBUG >= 1: print(f"remote has properties {self.conn.properties}")
    cloudrenderer = self.conn.properties['renderer']
    # TODO: how to we have BEAM be cached on the backend? this should just send a specification of the compute. rethink what goes in Renderer
    if not cloudrenderer[0].startswith("tinygrad.renderer.") or not cloudrenderer[1].endswith("Renderer"):
      raise RuntimeError(f"bad renderer {cloudrenderer}")
    renderer_class = fromimport(cloudrenderer[0], cloudrenderer[1])  # TODO: is this secure?
    if not issubclass(renderer_class, Renderer): raise RuntimeError(f"renderer isn't a Renderer {cloudrenderer}")

    from tinygrad.runtime.graph.cloud import CloudGraph
    super().__init__(device, CloudAllocator(self), renderer_class(*cloudrenderer[2]), Compiler(), functools.partial(CloudProgram, self),
                     CloudGraph.construct(self.host, self.conn.properties['graph_multi']) if self.conn.properties['graph'] else None)

  def __del__(self):
    # TODO: this is never being called
    # TODO: should close the whole session
    with contextlib.suppress(ConnectionRefusedError, http.client.CannotSendRequest, http.client.RemoteDisconnected): self.conn.batch_submit()

  @staticmethod
  @functools.cache
  def start_local():
    multiprocessing.Process(target=cloud_server, args=(6667,), name='MainProcess', daemon=True).start()
    return "127.0.0.1:6667"

if __name__ == "__main__": cloud_server(getenv("PORT", 6667))
