# this can be constructed from a cl_cache or loaded from a thneed file 
import os
import time
import struct
import numpy as np
from tinygrad.llops.ops_gpu import CL, CLProgram, CLBuffer
import pyopencl as cl

DEBUGCL = int(os.getenv("DEBUGCL", 0))
FLOAT16 = int(os.getenv("FLOAT16", 0))

class Thneed:
  def __init__(self, cl_cache=[]):
    self.cl_cache = cl_cache[:]

  def load(self, fn):
    pass

  def save(self, fn):
    # this is the struct that will be saved
    jdat = {"binaries": [], "programs": {}, "kernels": [], "objects": []}

    # NOTE: kernels_to_save is named wrong, it's actually buffers
    weights = []
    saved_objs = set()
    saved_binaries = set()
    for prg, args in enumerate(self.cl_cache):
      # get binaries for saving
      if prg.name not in saved_binaries:
        binary = prg.clprogram.get_info(cl.program_info.BINARIES)
        assert len(binary) == 1
        jdat['binaries'].append({"name":prg.name, "length":len(binary[0])})
        saved_binaries.add(prg.name)
      
      # get the args from the kernel, some need the data saved
      targs, args_size = [], []
      argdtypes = prg.argdtypes if prg.argdtypes is not None else [None]*(len(args)-2)
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
              needs_load = a in kernels_to_save
              jdat['objects'].append({
                "id": ptr, "arg_type": "float*", "needs_load": needs_load, "size": a.size,
              })
              if needs_load:
                data = np.empty(a.size//4, dtype=np.float32)
                CL.enqueue_copy(data, a, is_blocking=True)
                weights.append(data.tobytes())
            elif isinstance(a, cl.Image):
              needs_load = a in kernels_to_save
              row_pitch = (a.shape[0]*4*(2 if FLOAT16 else 4) + 63)//64 * 64
              size = row_pitch * a.shape[1]
              # this is *2 if float16 and *4 if float32
              buf = CLBuffer(size * (2 if FLOAT16 else 1))

              # zero out the buffer
              zeros = np.zeros(size, dtype=np.uint8)
              CL.enqueue_copy(buf.cl, zeros, is_blocking=True)

              CLProgram("from_image_strided", """
                __kernel void from_image_strided(read_only image2d_t in, __global float4 *out, int row_pitch) {
                  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
                  int2 l;
                  l.y = get_global_id(1);
                  l.x = get_global_id(0);
                  out[l.y*row_pitch + l.x] = read_imagef(in, smp, l);
                }
              """, argdtypes=(None, None, np.int32))(a.shape, None, a, buf.cl, row_pitch//(4*(2 if FLOAT16 else 4)))

              # multiple of 32 isn't enough
              jdat['objects'].append({
                "id": ptr, "needs_load": needs_load, "size": size, "arg_type": "image2d_t",
                "width": a.shape[0], "height": a.shape[1], "row_pitch": row_pitch, "float32": not FLOAT16,
              })

              if needs_load:
                data = np.empty(size//(2 if FLOAT16 else 4), dtype=np.float32)
                CL.enqueue_copy(data, buf.cl, is_blocking=True)
                if FLOAT16: data = data.astype(np.float16)
                weights.append(data.tobytes())
            else:
              raise Exception("unknown object", a)
            #print(jdat['objects'][-1])
            saved_objs.add(ptr)
          targs.append(ptr)
          args_size.append(8)
        else:
          raise Exception("idk this type")

      # save the kernel itself
      jdat['kernels'].append({
        "name": self.name,
        "work_dim": len(args[0]),
        "global_work_size": args[0],
        # TODO: C++ thneed requires a local_work_size, so we fill it with ones
        "local_work_size": [1 for _ in args[0]] if args[1] is None else args[1],
        "num_args": len(args)-2,
        "args": targs,
        "args_size": args_size 
      })

  def run(self):
    events = []
    st = time.monotonic()
    for prg, args in self.cl_cache:
      events.append(prg.clprg(CL().cl_queue, *args))
    mt = time.monotonic()
    CL().cl_queue.finish()
    et = time.monotonic()
    print(f"submit in {(mt-st)*1000.0:.2f} ms, total runtime is {(et-st)*1000.0:.2f} ms")

    if DEBUGCL:
      total_runtime = 0
      for i, ((prg, args), e) in enumerate(zip(self.cl_cache, events)):
        runtime = (e.profile.end - e.profile.start)
        print(f"{i:3d} time {total_runtime/1e6:5.2f} ms running {prg.name:20s} with {str(args[0]):15s} {str(args[1]):15s} count {len(args)-2:2d} runtime {runtime/1e3:7.2f} us  {prg.options}")
        total_runtime += runtime
      print(f"total runtime: {total_runtime/1e6:.2f} ms")

  # TODO: does this belong here?
  def optimize_local_workgroup(self):
    pass


