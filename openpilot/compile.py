#!/usr/bin/env python3
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from collections import defaultdict

import os
import time
import io

os.environ['OPT'] = '99'
if os.getenv("GPU", None) is None:
  os.environ['OPENCL'] = '1'

DEBUGCL = int(os.getenv("DEBUGCL", 0))

import onnx
import numpy as np

import tinygrad.ops as ops

from tinygrad.llops.ops_gpu import CL
from extra.utils import fetch
from extra.onnx import get_run_onnx
from test.test_onnx import run_onnx_torch
from tinygrad.tensor import Tensor
from tinygrad.helpers import prod

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/ea449f1fe0bbff0eff5b12d64f0b5e75b7983998/selfdrive/modeld/models/supercombo.onnx"

np.random.seed(1337)
def get_random_input_tensors():
  np_inputs = {
    "input_imgs": np.random.randn(*(1, 12, 128, 256)),
    "big_input_imgs": np.random.randn(*(1, 12, 128, 256)),
    "desire": np.zeros((1, 8)),
    "traffic_convention": np.array([[1., 0.]]),
    "initial_state": np.zeros((1, 512))
    #"initial_state": np.zeros((1, 768))
  }
  np_inputs = {k:v.astype(np.float32) for k,v in np_inputs.items()}
  inputs = {k:Tensor(v.astype(np.float32), requires_grad=False) for k,v in np_inputs.items()}
  for _,v in inputs.items(): v.realize()
  return inputs, np_inputs

# OPTWG=1 UNSAFE_FLOAT4=1 DEBUGCL=1 FLOAT16=1 MATMUL=1 python3 openpilot/compile.py
# 22.59 ms
if __name__ == "__main__":
  Tensor.no_grad = True
  using_graph = ops.GRAPH
  ops.GRAPH = False

  dat = fetch(OPENPILOT_MODEL)
  onnx_model = onnx.load(io.BytesIO(dat))
  run_onnx = get_run_onnx(onnx_model)
  inputs, _ = get_random_input_tensors()

  # initial run(s) to load weights
  for _ in range(2):
    st = time.monotonic()
    tinygrad_out = run_onnx(inputs)['outputs']
    mt = time.monotonic()
    tinygrad_out.realize()
    mt2 = time.monotonic()
    tinygrad_out = tinygrad_out.numpy()
    et = time.monotonic()
    print(f"ran openpilot model in {(et-st)*1000.0:.2f} ms, waited {(mt2-mt)*1000.0:.2f} ms for realize, {(et-mt2)*1000.0:.2f} ms for GPU queue")

  # real run
  inputs, np_inputs = get_random_input_tensors()
  tinygrad_out = run_onnx(inputs)['outputs']

  CL.CACHE = []
  if using_graph: ops.GRAPH = True
  CL.kernel_count = -1
  tinygrad_out.realize()
  ops.GRAPH = False
  print("kernel count:", len(CL.CACHE))

  # optimize local workgroups
  OPTWG = int(os.getenv("OPTWG", 0))
  if OPTWG:
    MAX_WORKGROUP = CL.cl_ctx.devices[0].max_work_group_size
    local_cl_cache = []
    for i, (prg, args) in enumerate(CL.CACHE):
      args = list(args)
      if args[1] is None and len(args[0]) == 2:
        args[1] = [min(MAX_WORKGROUP, args[0][0]), 1]
        if OPTWG == 2 and args[0][0] % args[1][0] != 0:
          args[1] = None

      if args[1] is None and len(args[0]) == 3:
        """
        if args[0][1] == 1 and args[0][2] == 1:
          args[1] = [min(1024, args[0][0]), 1, 1]
        else:
          args[1] = [1,min(16,args[0][1]),min(args[0][2], 4)]
          args[1][0] = min(32, min(args[0][0], 1024 // (args[1][1] * args[1][2])))
        """
        runtimes = []
        for l2 in [16,args[0][1],MAX_WORKGROUP]:
          for l3 in [4,16,args[0][2],MAX_WORKGROUP]:
            for l1 in [max(1, MAX_WORKGROUP//(l2*l3)), args[0][0], 4, 16, MAX_WORKGROUP]:
              if l1 > args[0][0] or l2 > args[0][1] or l3 > args[0][2]: continue
              local_args = (l1, l2, l3)
              if prod(local_args) > MAX_WORKGROUP: continue
              args[1] = local_args
              if OPTWG == 2:
                bad = any(g%l != 0 for g,l in zip(args[0], args[1]))
                if bad: continue
              e = prg.clprg(CL().cl_queue, *args)
              CL().cl_queue.finish()
              runtime = e.profile.end - e.profile.start
              #print(runtime, args[0], args[1])
              runtimes.append((runtime, local_args))
        #print(sorted(runtimes)[0:5])
        if len(runtimes) > 0:
          args[1] = sorted(runtimes)[0][1]
        else:
          args[1] = None
          print("couldn't optimize", args[0])

      local_cl_cache.append((prg, args))
  else:
    local_cl_cache = CL.CACHE[:]
  CL.CACHE = None

  # real CL ish
  for j in range(1):
    events = []
    st = time.monotonic()
    for i, (prg, args) in enumerate(local_cl_cache):
      #print(args)
      events.append(prg.clprg(CL().cl_queue, *args))
    mt = time.monotonic()
    CL().cl_queue.finish()
    et = time.monotonic()
    print(f"submit in {(mt-st)*1000.0:.2f} ms, total runtime is {(et-st)*1000.0:.2f} ms")
    total_runtime = 0
    runtimes = defaultdict(float)
    for i, ((prg, args), e) in enumerate(zip(local_cl_cache, events)):
      # profile types https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clGetEventProfilingInfo.html
      runtime = e.profile.end - e.profile.start
      runtimes[prg.name.rsplit("_", 1)[0]] += runtime
      if DEBUGCL:
        print(f"{i:3d} time {total_runtime/1e6:5.2f} ms running {prg.name:20s} with {str(args[0]):15s} {str(args[1]):15s} count {len(args)-2:2d} runtime {runtime/1e3:7.2f} us  {prg.options}")
        #if prg.name == "matmul": print(f"   {args[3].shape} {args[4].shape} -> {args[5].shape}")
      total_runtime += runtime
    for k,v in runtimes.items():
      print(f"{k:20s} runtime: {v/1e6:.2f} ms")
    print(f"total runtime: {total_runtime/1e6:.2f} ms")

  tinygrad_out = tinygrad_out.numpy()

  # float32 only
  if int(os.getenv("FLOAT16", 0)) == 0:
    torch_out = run_onnx_torch(onnx_model, np_inputs).numpy()
    print(tinygrad_out, torch_out)
    np.testing.assert_allclose(torch_out, tinygrad_out, atol=1e-4, rtol=1e-2)

  # save local_cl_cache as thneed
  import struct, json
  jdat = {"binaries": [], "programs": {}, "kernels": [], "objects": []}
  weights = []
  binaries = []
  saved_objs = set()
  saved_binaries = set()

  gobj = 0
  import pyopencl as cl
  for self, args in local_cl_cache:
    #if self.name not in jdat['programs']:
    #  jdat['programs'][self.name] = {"src": self.prg, "options": ' '.join(self.options)}

    if self.name not in saved_binaries:
      binary = self.clprogram.get_info(cl.program_info.BINARIES)
      assert len(binary) == 1
      jdat['binaries'].append({"name":self.name, "length":len(binary[0])})
      binaries.append(binary[0])
      saved_binaries.add(self.name)

    targs, args_size = [], []
    argdtypes = self.argdtypes if self.argdtypes is not None else [None]*(len(args)-2)
    for a,d in zip(args[2:], argdtypes):
      if d == np.int16:
        targs.append(struct.pack("H", a).decode("latin_1"))
        args_size.append(2)
      elif d == np.int32:
        targs.append(struct.pack("I", a).decode("latin_1"))
        args_size.append(4)
      elif isinstance(a, cl.LocalMemory):
        targs.append("")
        args_size.append(a.size)
      elif d is None:
        if getattr(a, "global_id", None) is None:
          setattr(a, "global_id", gobj)
          gobj += 1
        ptr = struct.pack("Q", a.global_id).decode("latin_1")
        if ptr not in saved_objs:
          if isinstance(a, cl.Buffer):
            jdat['objects'].append({
              "id": ptr, "arg_type": "float*", "needs_load": False, "size": a.size,
            })
          elif isinstance(a, cl.Image):
            #print(a.size, a.shape, a.row_pitch)
            # TODO: modify thneed to respect no buffer_id here
            jdat['objects'].append({
              "id": ptr, "needs_load": False, "size": a.size, "arg_type": "image2d_t",
              "width": a.shape[0], "height": a.shape[1], "row_pitch": a.row_pitch,
            })
          else:
            raise Exception("unknown object", a)
          #print(jdat['objects'][-1])
          saved_objs.add(ptr)
        targs.append(ptr)
        args_size.append(8)
      else:
        raise Exception("idk this type")

    jdat['kernels'].append({
      "name": self.name,
      "work_dim": len(args[0]),
      "global_work_size": args[0],
      "local_work_size": [1 for x in args[0]] if args[1] is None else args[1],
      "num_args": len(args)-2,
      "args": targs,
      "args_size": args_size 
    })

  print("saving thneed")
  with open("/tmp/output.thneed", "wb") as f:
    j = json.dumps(jdat, ensure_ascii=False).encode('latin_1')
    f.write(struct.pack("I", len(j)))
    f.write(j)
    f.write(b''.join(weights))
    f.write(b''.join(binaries))
