#!/usr/bin/env python3
import os, time, io, pathlib, sys, traceback, re
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

if os.getenv("OPT", None) is None:
  os.environ['OPT'] = '99'
if os.getenv("GPU", None) is None:
  os.environ['GPU'] = '1'
if os.getenv("IMAGE", None) is None:
  os.environ['IMAGE'] = '2'

from tinygrad.helpers import getenv, dtypes
ALLOWED_KERNEL_COUNT = getenv("ALLOWED_KERNEL_COUNT", 0)
DEBUGCL = getenv("DEBUGCL", 0)

import onnx
import numpy as np

import tinygrad.graph as graph
from tinygrad.helpers import GlobalCounters
from tinygrad.jit import TinyJit, CacheCollector

import pyopencl as cl
from tinygrad.runtime.ops_gpu import CL
from extra.utils import fetch
from extra.onnx import get_run_onnx
from tinygrad.tensor import Tensor

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx"

np.random.seed(1337)
def get_random_input_tensors(input_shapes):
  # this 16 is a random scale factor
  inputs = {k:Tensor.randn(*shp, requires_grad=False)*8 for k,shp in input_shapes.items()}
  np_inputs = {k:v.realize().numpy() for k,v in inputs.items()}
  return inputs, np_inputs

@TinyJit
def model_exec(run_onnx, using_graph, **inputs):
  ret = next(iter(run_onnx(inputs).values())).cast(dtypes.float32)
  GlobalCounters.reset()
  CacheCollector.start() # don't cache pre-realize
  if using_graph: graph.GRAPH = True
  print("realizing")
  return ret.realize()

def compile(dat, output_fn):
  Tensor.manual_seed(1337)
  Tensor.no_grad = True
  using_graph = graph.GRAPH
  if getenv("GRAPH") < 3: graph.GRAPH = False

  onnx_model = onnx.load(io.BytesIO(dat))
  run_onnx = get_run_onnx(onnx_model)
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}

  inputs, np_inputs = get_random_input_tensors(input_shapes)
  # run twice to trigger the JIT
  for i in range(2): tinygrad_out = model_exec(run_onnx, i == 1 and using_graph, **inputs)
  graph.GRAPH = False
  print("kernel count:", len(model_exec.jit_cache))
  assert len(model_exec.jit_cache) <= ALLOWED_KERNEL_COUNT or ALLOWED_KERNEL_COUNT == 0, "too many kernels!"

  # pull out inputs and put them in the jit cache
  input_rawbuffers = {k:inputs[k].lazydata.realized for k in inputs.keys()}
  for (j,i),(idx,_,_) in model_exec.input_replace.items(): model_exec.jit_cache[j][1][i] = input_rawbuffers[idx]

  # transform to CL.CACHE
  used_ops = 0
  cl_cache = []
  for prg,args,_ in model_exec.jit_cache:
    # pass these to thneed
    setattr(prg.clprg, 'op_estimate', prg.op_estimate)
    setattr(prg.clprg, 'prg', prg.prg)

    if getenv("VALIDTEST") == 1:
      src = re.search(r"=.*\?.*?read_image", prg.prg)
      if src is not None: raise Exception("Openpilot has valid checks!")

    global_size = prg.global_size + [1]*(3-len(prg.global_size))
    local_size = prg.local_size + [1]*(3-len(prg.local_size))
    cl_cache.append((prg.clprg, [[int(g*l) for g,l in zip(global_size, local_size)], local_size, *[x._buf for x in args]]))
    used_ops += prg.op_estimate

  from extra.thneed import Thneed
  t = Thneed(cl_cache, {k:v._buf for k,v in input_rawbuffers.items()})

  # save thneed (before run)
  t.save(output_fn)

  print(f"buffers to save: {len(t.buffers_to_save)}, inputs: {list(t.inputs.keys())}, outputs: {t.outputs}")
  runtime = t.run()
  print(f"network using {used_ops/1e9:.2f} GOPS with runtime {runtime*1e3:.2f} ms that's {used_ops/runtime*1e-9:.2f} GFLOPS")

  # confirm thneed found the right output
  thneed_out = np.empty((t.outputs[0].size//4,), dtype=np.float32).reshape(tinygrad_out.shape)
  cl.enqueue_copy(CL.cl_queue[0], thneed_out, t.outputs[0], is_blocking=True)
  np.testing.assert_allclose(thneed_out, tinygrad_out.numpy())

  # testing is float32 only (fix this)
  FLOAT16 = getenv("FLOAT16", 0)
  if FLOAT16 == 0:
    try:
      from test.models.test_onnx import run_onnx_torch
      torch_out = run_onnx_torch(onnx_model, np_inputs).numpy()
      print(thneed_out, torch_out, "mse", np.sum((thneed_out-torch_out)**2), "max err", np.max(np.abs((thneed_out-torch_out))))
      np.testing.assert_allclose(torch_out, thneed_out, atol=1e-4, rtol=1e-2)

      # test loading/run thneed
      _, new_np_inputs = get_random_input_tensors(input_shapes)
      new_torch_out = run_onnx_torch(onnx_model, new_np_inputs).numpy()

      # try old thneed with a different input
      for k,v in t.inputs.items():
        cl.enqueue_copy(CL.cl_queue[0], v, new_np_inputs[k], is_blocking=True)

      t.run()
      old_thneed_out = np.empty((t.outputs[0].size//4,), dtype=np.float32).reshape(tinygrad_out.shape)
      cl.enqueue_copy(CL.cl_queue[0], old_thneed_out, t.outputs[0], is_blocking=True)

      # compare thneed (rerun) with torch
      np.testing.assert_allclose(new_torch_out, old_thneed_out, atol=1e-4, rtol=1e-2)

      # load thneed and try that
      _, new_np_inputs = get_random_input_tensors(input_shapes)
      new_torch_out = run_onnx_torch(onnx_model, new_np_inputs).numpy()
      nt = Thneed()
      nt.load(output_fn)

      # inputs
      for k,v in nt.inputs.items():
        cl.enqueue_copy(CL.cl_queue[0], v, new_np_inputs[k], is_blocking=True)

      nt.run()
      new_thneed_out = np.empty((nt.outputs[0].size//4,), dtype=np.float32).reshape(tinygrad_out.shape)
      cl.enqueue_copy(CL.cl_queue[0], new_thneed_out, nt.outputs[0], is_blocking=True)

      # compare torch to thneed
      np.testing.assert_allclose(new_torch_out, new_thneed_out, atol=1e-4, rtol=1e-2)
      print("thneed self-test passed!")
    except ModuleNotFoundError as e:
      print(f"TEST NOT HAPPENING {e}")


# UNSAFE_FLOAT4=1 DEBUGCL=1 FLOAT16=1 python3 openpilot/compile.py
# 22.59 ms
if __name__ == "__main__":
  dat = fetch(OPENPILOT_MODEL if len(sys.argv) == 1 else sys.argv[1])
  compile(dat, sys.argv[2] if len(sys.argv) >= 3 else "/tmp/output.thneed")
