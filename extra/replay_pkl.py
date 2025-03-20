import pickle, sys
from dataclasses import replace
from tinygrad import Device, Context
from tinygrad.device import Buffer
from tinygrad.helpers import getenv
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
        if not getenv("NOOPT"):
          # only NCHW
          """
          if knum in [6,7,9,11]:
            k.apply_opt(Opt(OptOps.PADTO, 1, 128))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 128))
          elif knum in [5,8]:
            k.apply_opt(Opt(op=OptOps.UNROLL, axis=1, arg=0))
            k.apply_opt(Opt(op=OptOps.UNROLL, axis=0, arg=0))
            k.apply_opt(Opt(OptOps.PADTO, 2, 128))
            k.apply_opt(Opt(OptOps.UPCAST, 2, 128))
          elif knum == 2:
            k.apply_opt(Opt(op=OptOps.UNROLL, axis=1, arg=0))
            k.apply_opt(Opt(op=OptOps.UNROLL, axis=0, arg=0))
            k.apply_opt(Opt(OptOps.PADTO, 2, 128))
            k.apply_opt(Opt(OptOps.UPCAST, 2, 128))
            #k.apply_opt(Opt(op=OptOps.UPCAST, axis=1, arg=4))
          elif knum == 1:
            k.apply_opt(Opt(op=OptOps.UNROLL, axis=2, arg=0))
            k.apply_opt(Opt(op=OptOps.UNROLL, axis=1, arg=0))
            #k.apply_opt(Opt(op=OptOps.UNROLL, axis=0, arg=0))
            k.apply_opt(Opt(OptOps.PADTO, 2, 128))
            k.apply_opt(Opt(OptOps.UPCAST, 2, 128))
          elif knum == 3:
            k.apply_opt(Opt(op=OptOps.UNROLL, axis=0, arg=4))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 128))
          else:
            k.hand_coded_optimizations()
          """
          if knum == 3:
            k.apply_opt(Opt(OptOps.UNROLL, 0, 0))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 16))
            k.apply_opt(Opt(OptOps.UPCAST, 0, 128//16))
            #k.apply_opt(Opt(OptOps.UPCAST, 0, 8))
            pass
          else:
            k.hand_coded_optimizations()
          #if knum in [5]: k.apply_opt(Opt(OptOps.UPCAST, 1, 2))
        p2 = k.to_program()
        new_ei = replace(ei, prg=CompiledRunner(p2), bufs=[Buffer("DSP", 1024+b.size*2, b.dtype).view(b.size, b.dtype, 512) for b in ei.bufs])
        new_ei.run()
      knum += 1
