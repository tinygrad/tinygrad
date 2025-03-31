import pickle, sys
from dataclasses import replace
from tinygrad import Device, Context
from tinygrad.device import Buffer
from tinygrad.helpers import getenv, BEAM
from tinygrad.engine.jit import TinyJit
from tinygrad.engine.realize import CompiledRunner
from tinygrad.renderer import ProgramSpec
from tinygrad.codegen.kernel import Kernel, Opt, OptOps

if __name__ == "__main__":
  with Context(DEBUG=0):
    with open(sys.argv[1], "rb") as f:
      fxn: TinyJit = pickle.load(f)
      print(f"{f.tell()/1e6:.2f}M loaded")
    print(type(fxn))

  knum = 1
  for ei in fxn.captured.jit_cache:
    # skip the copy and the first kernel
    if isinstance(ei.prg, CompiledRunner) and all(x is not None for x in ei.bufs):
      if knum == (pknum:=getenv("KNUM", 0)) or pknum == 0:
        p: ProgramSpec = ei.prg.p
        k = Kernel(p.ast, Device["DSP"].renderer)
        dsp_bufs = [Buffer("DSP", 8192+b.size, b.dtype).view(b.size, b.dtype, 4096) for b in ei.bufs]
        k.hand_coded_optimizations()
        #if knum == 13: k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
        p2 = k.to_program()
        new_ei = replace(ei, prg=CompiledRunner(p2), bufs=dsp_bufs)
        if getenv("MULTICORE", 0) == 1:
          new_ei.run({p2.vars[0]: 0})
          new_ei.run({p2.vars[0]: 1})
        else:
          new_ei.run()
      knum += 1
