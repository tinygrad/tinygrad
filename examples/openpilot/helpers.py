"""Shared compilation and input allocation for model and warp artifacts."""
import io, pickle, shutil, struct, tempfile, time
import numpy as np
from typing import Callable
from tinygrad import Tensor, Device, Context
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, graph_rewrite
from tinygrad.nn.state import get_parameters

def allocate_inputs(input_specs, initialize=None):
  """Initialize inputs before copying to devices."""
  arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype, _) in input_specs.items()}
  if initialize is not None: initialize(arrays)
  return {name: Tensor(arrays[name], device=device).realize() for name, (_, _, device) in input_specs.items()}


def dump_pickle(obj, f, *, out_of_band=False):
  if not out_of_band: return pickle.dump(obj, f)
  with tempfile.TemporaryFile(dir=".") as tmp:
    def buffer_callback(pb:pickle.PickleBuffer):
      data = pb.raw()
      tmp.write(struct.pack('<q', data.nbytes))
      tmp.write(data)
      pb.release()
    stream = io.BytesIO()
    pickle.Pickler(stream, protocol=5, buffer_callback=buffer_callback).dump(obj)
    opcodes = stream.getvalue()
    f.write(struct.pack('<q', len(opcodes)))
    f.write(opcodes)
    tmp.seek(0)
    shutil.copyfileobj(tmp, f)


def load_pickle(f, *, out_of_band=False):
  if not out_of_band: return pickle.load(f)
  opcodes = f.read(struct.unpack('<q', f.read(8))[0])
  def buffers():
    while h := f.read(8):
      pb = pickle.PickleBuffer(bytearray(struct.unpack('<q', h)[0]))
      if f.readinto(pb) != pb.raw().nbytes: raise EOFError("incomplete model buffer")
      yield pb
  return pickle.load(io.BytesIO(opcodes), buffers=buffers())


@Context(OPENPILOT_HACKS=1, **{'AMD': {'TC_OPT': 2, 'TC_MIN_GLOBALS': 32}}.get(Device.DEFAULT, {}))
def benchmark(fxn:Callable, cb=None, **kwargs):
  Device.default.synchronize()
  start = time.perf_counter()
  if (output := fxn(**kwargs)) is not None: output.realize()
  Device.default.synchronize()
  end = time.perf_counter()
  if cb: cb(end-start)
  return [t.numpy().copy() for t in get_parameters(kwargs.get('output_buffers', output))]

pm_retargetable = PatternMatcher([
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(), UPat(), UPat()), name="p"), lambda p: p.replace(src=p.src[:-1]) if p.arg.target.device == "CPU" else None)
])

def make_retargetable(jit): jit.captured._linear = graph_rewrite(jit.captured._linear, pm_retargetable, walk=True, enter_calls=True)
