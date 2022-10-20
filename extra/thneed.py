# this can be constructed from a cl_cache or loaded from a thneed file 
import os
import time
from collections import defaultdict
from tinygrad.llops.ops_gpu import CL
import pyopencl as cl

DEBUGCL = int(os.getenv("DEBUGCL", 0))

class Thneed:
  def __init__(self, cl_cache=[]):
    self.cl_cache = cl_cache[:]

  def load(self, fn):
    pass

  def save(self, fn):
    # this is the struct that will be saved
    jdat = {"binaries": [], "programs": {}, "kernels": [], "objects": []}

    # get binaries for saving
    saved_binaries = set()
    for prg, args in enumerate(self.cl_cache):
      if prg.name not in saved_binaries:
        binary = prg.clprogram.get_info(cl.program_info.BINARIES)
        assert len(binary) == 1
        jdat['binaries'].append({"name":prg.name, "length":len(binary[0])})
        saved_binaries.add(prg.name)

  def run(self):
    events = []
    st = time.monotonic()
    for i, (prg, args) in enumerate(self.cl_cache):
      #print(args)
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


