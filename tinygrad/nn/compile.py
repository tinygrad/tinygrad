"""Capture TinyJit artifacts using the backend selected by DEV."""
import argparse, io, pickle, shutil, struct, tempfile, time
from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.nn.state import get_parameters


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


def onnx_metadata(path):
  from tinygrad.nn.onnx import OnnxPBParser
  class MetadataParser(OnnxPBParser):
    def _parse_ModelProto(self) -> dict:
      obj:dict[str, Any] = {"graph": {"input": [], "output": []}, "metadata_props": []}
      for fid, wire_type in self._parse_message(self.reader.len):
        if fid == 7: obj["graph"] = self._parse_GraphProto()
        elif fid == 14: obj["metadata_props"].append(self._parse_StringStringEntryProto())
        else: self.reader.skip_field(wire_type)
      return obj
  model = MetadataParser(path).parse()
  return {"metadata": {p["key"]: p["value"] for p in model["metadata_props"]}} | {
    f"{kind}_shapes": {v["name"]: tuple(d if isinstance(d, int) else 0 for d in v["parsed_type"].shape) for v in model["graph"][kind]}
    for kind in ("input", "output")}


def compile_onnx(path, *, device_inputs=(), float32=False, output_name=None, benchmark_runs=20, out_of_band=False):
  from tinygrad.nn.onnx import OnnxRunner
  runner = OnnxRunner(path)
  if unknown := set(device_inputs) - runner.graph_inputs.keys(): raise ValueError(f"Unknown inputs: {unknown}")
  if output_name is not None and output_name not in runner.graph_outputs: raise ValueError(f"Unknown output: {output_name}")

  def make_inputs(seed):
    rng = np.random.default_rng(seed)
    inputs = {}
    for name, spec in sorted(runner.graph_inputs.items()):
      shape = tuple(s if isinstance(s, int) else 1 for s in spec.shape)
      dtype = dtypes.float32 if float32 and spec.dtype == dtypes.float16 else spec.dtype
      data = rng.standard_normal(shape) if dtypes.is_float(dtype) else rng.integers(0, 2 if dtype == dtypes.bool else 16, shape)
      inputs[name] = Tensor(data, dtype=dtype, device=Device.DEFAULT if name in device_inputs else 'NPY').contiguous().realize()
    return (), inputs

  def run(**inputs):
    outputs = runner({k: v.to(Device.DEFAULT).cast(runner.graph_inputs[k].dtype) for k, v in inputs.items()})
    if float32: outputs = {k: v.cast(dtypes.float32) for k, v in outputs.items()}
    if output_name is not None: return outputs[output_name]
    return next(iter(outputs.values())) if len(outputs) == 1 else outputs

  return compile_jit(run, make_inputs, benchmark_runs, out_of_band=out_of_band)


if __name__ == '__main__':
  from tinygrad.helpers import fetch
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('onnx')
  parser.add_argument('output')
  parser.add_argument('--device-input', action='append', default=[], help='input placed on DEV instead of host NPY (repeatable)')
  parser.add_argument('--float32', action='store_true', help='expose float16 inputs and model outputs as float32')
  parser.add_argument('--output-name', help='select one model output')
  parser.add_argument('--benchmark-runs', type=int, default=20)
  parser.add_argument('--out-of-band', action='store_true', help='stream protocol-5 buffers for large models')
  parser.add_argument('--metadata-output')
  args = parser.parse_args()
  path = fetch(args.onnx) if '://' in args.onnx else Path(args.onnx)
  jit = compile_onnx(path, device_inputs=args.device_input, float32=args.float32, output_name=args.output_name,
                     benchmark_runs=args.benchmark_runs, out_of_band=args.out_of_band)
  with open(args.output, 'wb') as f: dump_pickle(jit, f, out_of_band=args.out_of_band)
  if args.metadata_output:
    with open(args.metadata_output, 'wb') as f: pickle.dump(onnx_metadata(path), f)
