# an engine's captured graphs kept between starts: pickled with every buffer of the process (the trunk's, the draft's, the engine's state, the
# pool and what it lends, the host's) named by where it's reached from, nothing big serialized. a file is a start's of the same code, model and
# config (its key), checked on load by the dry runs' outputs (each graph's) against the ones it was saved with
import hashlib, io, json, os, pathlib, pickle, types
import numpy as np
from typing import Any
from tinygrad import Tensor, nn
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.engine.jit import _TinyJit
from tinygrad.helpers import ContextVar, cache_dir
from tinygrad.uop.ops import Ops, UOp
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported, _mailboxes

Named = Buffer|MultiBuffer|UOp

def buffers(eng) -> dict[int, tuple[str, Named]]:
  """every buffer an engine holds (its model's, its draft's, its attributes'), and its UOp, by id: a name of where it's reached from"""
  out:dict[int, tuple[str, Named]] = {}
  def add(name:str, b:Named):
    if id(b) in out: return
    out[id(b)] = (name, b)
    for i, x in enumerate(b.bufs if isinstance(b, MultiBuffer) else [b._base] if isinstance(b, Buffer) and b._base is not None else []):
      add(f"{name}.{i}", x)
  def walk(name:str, x:Any):
    if isinstance(x, Tensor):
      for i, u in enumerate([u for u in x.uop.toposort() if u.op is Ops.BUFFER and u.arg.buffer is not None]):
        add(f"{name}#{i}", u.arg.buffer)
        add(f"{name}#{i}:uop", u)
    elif isinstance(x, (list, tuple)):
      for i, y in enumerate(x): walk(f"{name}[{i}]", y)
    elif isinstance(x, dict):
      for k, y in x.items(): walk(f"{name}[{k!r}]", y)
    elif isinstance(x, (Buffer, MultiBuffer)): add(name, x)
    elif isinstance(x, UOp): walk(name, Tensor(x))
    elif hasattr(x, "__dict__") and not isinstance(x, (_TinyJit, types.MethodType, types.FunctionType)):  # (an object of tensors: a Linear's)
      for k, t in nn.state.get_state_dict(x).items(): walk(f"{name}.{k}", t)
  for k, t in nn.state.get_state_dict(eng.model).items(): walk(f"model.{k}", t)
  for k, t in nn.state.get_state_dict(eng.draft).items(): walk(f"draft.{k}", t)
  for k, v in vars(eng).items():
    if k not in ("model", "draft"): walk(f"eng.{k}", v)
  if amd_custom_kernels_supported(eng.devs): walk("allreduce", _mailboxes(eng.devs))  # (the small allreduces' mailboxes: every graph's)
  return out

def files(*paths:str) -> bytes:  # (files' identities: path, size, mtime)
  return json.dumps([[f, os.stat(f).st_size, os.stat(f).st_mtime_ns] for f in paths if os.path.exists(f)]).encode()

def code() -> bytes:  # (tinygrad's and k3's code)
  root = pathlib.Path(__file__).parents[2]
  srcs = sorted([*(root/"tinygrad").rglob("*.py"), *(root/"extra"/"k3").glob("*.py")])
  return b"".join(str(f.relative_to(root)).encode() + f.read_bytes() for f in srcs)

def key(eng) -> str:
  """a start's key: the code, the model's gguf, the draft's shape, every knob, the engine's shape"""
  h = hashlib.sha256(code() + files(eng.source))
  draft = [(type(eng.draft).__name__, eng.width, eng.taps), *((k, t.shape, t.dtype) for k, t in nn.state.get_state_dict(eng.draft).items())]
  h.update(repr(draft).encode())
  h.update(json.dumps({k: str(v.value) for k, v in sorted(ContextVar._cache.items())}).encode())
  h.update(repr((eng.model.config, eng.ms, eng.devs, eng.tokens, sorted(eng.jits), eng.chunk, eng.W, eng.slots, eng.tail, eng.dump_taps,
                   eng.keep_logits, eng.part, eng.layer_spans, eng.forecast_counts is not None)).encode())
  return h.hexdigest()[:32]

def path(eng) -> pathlib.Path: return pathlib.Path(cache_dir) / "k3" / f"{eng.graph_key}.pkl"

class _Pickler(pickle.Pickler):
  def __init__(self, f:io.BytesIO, names:dict[int, tuple[str, Named]]):
    super().__init__(f, protocol=5)
    self.names = names
    self.copied = 0
  def persistent_id(self, obj:Any) -> str|None:
    if not isinstance(obj, (Buffer, MultiBuffer, UOp)): return None
    if id(obj) not in self.names:
      self.copied += obj.nbytes if isinstance(obj, Buffer) and obj._base is None else 0
      return None
    return self.names[id(obj)][0]

class _Unpickler(pickle.Unpickler):
  def __init__(self, f:io.BytesIO, by_name:dict[str, Named]):
    super().__init__(f)
    self.by_name = by_name
  def persistent_load(self, pid:str) -> Named: return self.by_name[pid]

def save(eng, names:dict[int, tuple[str, Named]], graphs:dict[str, Any], fps:dict[str, dict[str, str]]):
  """the captured graphs and their dry runs' fingerprints, to the start's file, the buffers by their names before the graphs were captured
  (names: buffers' then)"""
  if not eng.source: return
  f = io.BytesIO()
  (pk:=_Pickler(f, names)).dump({"graphs": {name: jit.captured for name, jit in graphs.items()}, "fps": fps})
  if pk.copied > 1 << 20: print(f"start: {pk.copied / 1e6:.1f} MB of buffers not the engine's in the graphs file", flush=True)
  (p:=path(eng)).parent.mkdir(parents=True, exist_ok=True)
  (tmp:=p.with_suffix(".tmp")).write_bytes(f.getvalue())
  tmp.replace(p)

def load(eng, names:dict[int, tuple[str, Named]]) -> dict[str, Any]:
  """the start's file ({"graphs": name -> CapturedJit, "fps": stage -> name -> fingerprint}), its buffers by names (the engine's buffers'
  before its graphs are captured), empty without one (or one that doesn't load, or an engine without a source: nothing kept)"""
  if not eng.source or not (p:=path(eng)).exists(): return {}
  try: return _Unpickler(io.BytesIO(p.read_bytes()), dict(names.values())).load()
  except Exception as e:
    print(f"start: graphs file {p.name} not loaded ({type(e).__name__}: {e}), capturing", flush=True)
    return {}

def discard(eng):
  if (p:=path(eng)).exists(): p.unlink()

# the scorers (the quantized routers and guesses): of the gguf and the guess maps only, their bytes kept (a gpu's of each)

def scorers_path(eng, guess:str) -> pathlib.Path:
  guesses = sorted(str(p) for p in pathlib.Path(guess).glob("guess-*.npy")) if guess else []
  h = hashlib.sha256(code() + files(eng.source, *guesses) + repr((eng.model.config, eng.m, eng.devs, len(eng.types))).encode())
  return pathlib.Path(cache_dir) / "k3" / f"scorers-{h.hexdigest()[:32]}.pkl"

def lanes(t:Tensor) -> list[Buffer]: return b.bufs if isinstance(b:=t.uop.base.buffer, MultiBuffer) else [b]

def save_scorers(eng, guess:str, scorers:list[Any]):
  if not eng.source: return
  out = [((s.in_features, s.out_features, s.ggml_type, s.shard_axis) if isinstance(s, Linear) else None, (t:=s.weight if isinstance(s, Linear) else s)
          .shape, t.dtype, t.uop.axis, [bytes(b.as_memoryview()) for b in lanes(t)]) for s in scorers]
  (p:=scorers_path(eng, guess)).parent.mkdir(parents=True, exist_ok=True)
  (tmp:=p.with_suffix(".tmp")).write_bytes(pickle.dumps(out, protocol=5))
  tmp.replace(p)

def load_scorers(eng, guess:str) -> list[Any]|None:
  if not eng.source or not (p:=scorers_path(eng, guess)).exists(): return None
  out:list[Any] = []
  for linear, shape, dtype, axis, datas in pickle.loads(p.read_bytes()):
    assert axis in (0, None), "a scorer's lanes split on its first axis"
    t = Tensor(np.frombuffer(datas[0] if axis is None else b"".join(datas), np.dtype(dtype.fmt)).reshape(shape)).shard(eng.devs, axis)
    if linear is None: out.append(t)
    else:
      (lin:=Linear(linear[0], linear[1], bias=False)).ggml_type, lin.shard_axis, lin.weight = linear[2], linear[3], t
      out.append(lin)
  return out

def fingerprint(*xs:Any) -> str: return hashlib.sha256(b"".join(memoryview(x).tobytes() for x in xs)).hexdigest()[:16]
