"""Capture TinyJit artifacts using the backend selected by DEV."""
import io, pickle, shutil, struct, tempfile, time
from collections.abc import Callable
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.nn.state import get_parameters


def allocate_inputs(input_specs, packed_specs, initialize=None):
  """Allocate inputs and NumPy views, initializing before copying to devices."""
  arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype, _) in input_specs.items()}
  views = arrays.copy()
  if packed_specs:
    packed = views.pop('packed_inputs')
    views.update({name: packed[start:start+int(np.prod(shape))*np.dtype(dtype).itemsize].view(dtype).reshape(shape)
                  for name, (start, shape, dtype) in packed_specs.items()})
  if initialize is not None: initialize(views)
  return {name: Tensor(arrays[name], device=device).realize() for name, (_, _, device) in input_specs.items()}, views


def dump_pickle(obj, f, *, out_of_band=False):
  if not out_of_band: return pickle.dump(obj, f)
  with tempfile.TemporaryFile() as tmp:
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


def compile_jit(function:Callable, make_inputs:Callable[[int], tuple[tuple, dict]], benchmark_runs=20, *, out_of_band=False):
  """The factory creates fresh inputs, including any mutable state, for each seed."""
  if benchmark_runs < 1: raise ValueError("benchmark_runs must be at least 1")
  jit = TinyJit(function, prune=True)

  def run(fn, seed, count):
    args, kwargs = make_inputs(seed)
    result = None
    for i in range(count):
      Device.default.synchronize()
      start = time.perf_counter()
      output = fn(*args, **kwargs)
      Device.default.synchronize()
      print(f"  [{i+1}/{count}] {(time.perf_counter()-start)*1e3:.2f} ms")
      if i == 0:
        result = [t.numpy().copy() for t in get_parameters(output)], [t.numpy().copy() for t in get_parameters((args, kwargs))]
    return result

  expected = run(jit, 42, 3)
  with tempfile.TemporaryFile() as f:
    dump_pickle(jit, f, out_of_band=out_of_band)
    f.seek(0)
    loaded = load_pickle(f, out_of_band=out_of_band)
  for seed in (42, 43):
    reference = expected if seed == 42 else run(function, seed, 1)
    actual = run(loaded, seed, benchmark_runs)
    for ref_group, actual_group in zip(reference, actual, strict=True):
      for ref, value in zip(ref_group, actual_group, strict=True): np.testing.assert_array_equal(ref, value)
  # Preserve shared weight buffers when several JITs are saved in one artifact.
  return jit
