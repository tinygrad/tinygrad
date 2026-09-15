"""Compile an ONNX model into a TinyJit artifact."""
import argparse
from pathlib import Path
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.dtype import _to_np_dtype
from tinygrad.helpers import fetch
from examples.openpilot.helpers import allocate_inputs, compile_jit, dump_pickle
from tinygrad.nn.onnx import OnnxPBParser, OnnxRunner


def onnx_metadata(path):
  parser = OnnxPBParser(path, load_external_data=False)
  metadata, output_shapes = {}, {}
  for field, wire_type in parser._parse_message(parser.reader.len):
    if field == 14:
      entry = parser._parse_StringStringEntryProto()
      metadata[entry['key']] = entry['value']
    elif field == 7:
      # Read output declarations without parsing graph nodes or weight tensors.
      for field, wire_type in parser._parse_message(parser._decode_end_pos()):
        if field == 12:
          value = parser._parse_ValueInfoProto()
          output_shapes[value['name']] = value['parsed_type'].shape if value['parsed_type'] is not None else ()
        else: parser.reader.skip_field(wire_type)
    else: parser.reader.skip_field(wire_type)
  return metadata, output_shapes


def compile_onnx(path, *, device_inputs=(), benchmark_runs=20, out_of_band=False):
  runner = OnnxRunner(path)
  properties, output_shapes = onnx_metadata(path)
  metadata = {'metadata': properties} | {
    f'{kind}_shapes': {name: tuple(d if isinstance(d, int) else 0 for d in shape) for name, shape in shapes.items()}
    for kind, shapes in [('input', {name: spec.shape for name, spec in runner.graph_inputs.items()}), ('output', output_shapes)]}
  if '*' in device_inputs: device_inputs = tuple(runner.graph_inputs)
  if unknown := set(device_inputs) - runner.graph_inputs.keys(): raise ValueError(f"Unknown inputs: {unknown}")
  specs = {name: (tuple(s if isinstance(s, int) else 1 for s in spec.shape), np.dtype(_to_np_dtype(spec.dtype)).str,
                  Device.DEFAULT if name in device_inputs else 'NPY') for name, spec in runner.graph_inputs.items()}

  def make_inputs(seed):
    rng = np.random.default_rng(seed)
    def initialize(arrays):
      for name, value in arrays.items():
        dtype = runner.graph_inputs[name].dtype
        value[...] = (rng.standard_normal(value.shape) if dtypes.is_float(dtype) else
                      rng.integers(0, 256, value.shape, dtype=np.uint8) if dtype == dtypes.uint8 else
                      rng.integers(0, 2 if dtype == dtypes.bool else 16, value.shape))
    return (), allocate_inputs(specs, initialize)

  def run(**inputs): return runner({name: value.to(Device.DEFAULT) for name, value in inputs.items()})

  jit = compile_jit(run, make_inputs, benchmark_runs, out_of_band=out_of_band)
  return {'metadata': metadata, 'run': jit, 'input_specs': specs}


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('onnx')
  parser.add_argument('output')
  parser.add_argument('--device-input', action='append', default=[], help='input placed on DEV instead of host NPY (repeatable; * selects all)')
  parser.add_argument('--benchmark-runs', type=int, default=20)
  parser.add_argument('--out-of-band', action='store_true', help='stream protocol-5 buffers for large models')
  args = parser.parse_args()
  path = fetch(args.onnx) if '://' in args.onnx else Path(args.onnx)
  artifact = compile_onnx(path, device_inputs=args.device_input, benchmark_runs=args.benchmark_runs, out_of_band=args.out_of_band)
  with open(args.output, 'wb') as f: dump_pickle(artifact, f, out_of_band=args.out_of_band)
