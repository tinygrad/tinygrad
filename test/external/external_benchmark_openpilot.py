import csv, pathlib, time, numpy as np
from tinygrad.device import CompileError
import onnx
from onnx.helper import tensor_dtype_to_np_dtype
import onnxruntime as ort
from extra.onnx import get_run_onnx
from tinygrad.helpers import OSX, DEBUG, fetch
from tinygrad import Tensor, Device

def benchmark(mnm, nm, fxn):
  tms = []
  for _ in range(3):
    st = time.perf_counter_ns()
    ret = fxn()
    tms.append(time.perf_counter_ns() - st)
  print(f"{mnm:15s} {nm:25s} {min(tms)*1e-6:7.2f} ms")
  return min(tms), ret

BASE = pathlib.Path("/tmp/onnx")
def benchmark_model(m, devices):
  fn = fetch(m)
  onnx_model = onnx.load(fn)
  output_names = [out.name for out in onnx_model.graph.output]
  excluded = {inp.name for inp in onnx_model.graph.initializer}
  input_shapes = {inp.name:tuple(x.dim_value if x.dim_value != 0 else 1 for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input if inp.name not in excluded}  # noqa: E501
  input_types = {inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input if inp.name not in excluded}
  np_inputs = {k:Tensor.randn(shp).numpy().astype(input_types[k]) for k,shp in input_shapes.items()}
  assert len(input_shapes) < 30, f"too many input shapes {len(input_shapes)}"

  # print input names
  if DEBUG >= 2: print([inp.name for inp in onnx_model.graph.input if inp.name not in excluded])
  for device in devices:
    try:
      Device.DEFAULT = device
      inputs = {k:Tensor(inp) for k,inp in np_inputs.items()}
      tinygrad_model = get_run_onnx(onnx_model)
      benchmark(m, f"tinygrad_{device.lower()}_jitless", lambda: {k:v.numpy() for k,v in tinygrad_model(inputs).items()})

      from tinygrad.engine.jit import TinyJit
      tinygrad_jitted_model = TinyJit(lambda **kwargs: {k:v.realize() for k,v in tinygrad_model(kwargs).items()})
      for _ in range(3): {k:v.numpy() for k,v in tinygrad_jitted_model(**inputs).items()}
      benchmark(m, f"tinygrad_{device.lower()}_jit", lambda: {k:v.numpy() for k,v in tinygrad_jitted_model(**inputs).items()}) # noqa: F821
      del inputs, tinygrad_model, tinygrad_jitted_model
    except CompileError as e:
      raise e

  # bench onnxruntime
  ort_options = ort.SessionOptions()
  ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  ort_options.log_severity_level = 3  # no warnings
  for backend in ["CPU", "CUDA" if not OSX else "CoreML"]:  # https://onnxruntime.ai/docs/execution-providers/
    provider = backend+"ExecutionProvider"
    if provider not in ort.get_available_providers(): continue
    ort_sess = ort.InferenceSession(str(fn), ort_options, [provider])
    try:
      benchmark(m, f"onnxruntime_{backend.lower()}", lambda: ort_sess.run(output_names, np_inputs))
    except Exception as e: print(f"{m:16s}onnxruntime_{backend.lower()} {type(e).__name__:>25}")
    del ort_sess

  # validate outputs
  for device in devices:
    rtol, atol = 2e-3, 2e-3  # tolerance for fp16 models
    Device.DEFAULT = device
    inputs = {k:Tensor(inp) for k,inp in np_inputs.items()}
    tinygrad_model = get_run_onnx(onnx_model)
    tinygrad_out = tinygrad_model(inputs)

    ort_sess = ort.InferenceSession(str(fn), ort_options, ["CPUExecutionProvider"])
    onnx_out = ort_sess.run(output_names, np_inputs)
    onnx_out = dict([*list(zip(output_names, onnx_out))])

    assert_allclose(tinygrad_out, onnx_out, rtol=rtol, atol=atol)
    print(f"{m:16s}outputs validated on {device=} with rtol={rtol:.1e}, atol={atol:.1e}")

def assert_allclose(tiny_out:dict, onnx_out:dict, rtol=1e-5, atol=1e-5):
  assert len(tiny_out) == len(onnx_out) and tiny_out.keys() == onnx_out.keys()
  for k in tiny_out.keys():
    tiny_v, onnx_v = tiny_out[k], onnx_out[k]
    if tiny_v is None: assert tiny_v == onnx_v
    else: np.testing.assert_allclose(tiny_v.numpy(), onnx_v, rtol=rtol, atol=atol, err_msg=f"For tensor '{k}' in {tiny_out.keys()}")

if __name__ == "__main__":
  devices = [Device.DEFAULT]
  benchmark_model("https://github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx", devices)
