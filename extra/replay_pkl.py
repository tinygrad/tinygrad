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
        if BEAM:
          from tinygrad.engine.search import beam_search
          k = beam_search(k, dsp_bufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
        elif not getenv("NOOPT"):
          if knum == 1:
            k.apply_opt(Opt(OptOps.UPCAST, 2, 32))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 4))
          elif knum == 66:
            k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
            k.apply_opt(Opt(OptOps.UPCAST, 0, 8))
          elif k.full_shape[-3:] == (32,3,3):
            #if k.full_shape[-4]%4 != 0: k.apply_opt(Opt(OptOps.PADTO, len(k.full_shape)-4, 4))
            # 3x3 dwconv
            k.apply_opt(Opt(OptOps.UNROLL, 0, 0))
            k.apply_opt(Opt(OptOps.UNROLL, 0, 0))
            k.apply_opt(Opt(OptOps.UPCAST, len(k.full_shape)-3, 32))
            if k.full_shape[-4]%4 == 0: k.apply_opt(Opt(OptOps.UPCAST, len(k.full_shape)-4, 4))
          elif len(k.full_shape) == 3 and k.full_shape[1] == 32:
            #if k.full_shape[0]%4 != 0: k.apply_opt(Opt(OptOps.PADTO, 0, 4))
            # weight without more
            k.apply_opt(Opt(OptOps.UNROLL, 0, 8))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 32))
            if k.full_shape[0]%4 == 0: k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
          elif len(k.full_shape) == 4 and k.full_shape[2] == 32:
            #if k.full_shape[1]%4 != 0: k.apply_opt(Opt(OptOps.PADTO, 1, 4))
            # weight with more
            k.apply_opt(Opt(OptOps.UNROLL, 0, 8))
            k.apply_opt(Opt(OptOps.UPCAST, 2, 32))
            if k.full_shape[1]%4 == 0: k.apply_opt(Opt(OptOps.UPCAST, 1, 4))
          elif len(k.full_shape) == 1:
            for sz in [128,64,32]:
              if k.full_shape[0]%sz == 0:
                k.apply_opt(Opt(OptOps.UPCAST, 0, sz))
                break
        p2 = k.to_program()
        new_ei = replace(ei, prg=CompiledRunner(p2), bufs=dsp_bufs)
        new_ei.run()
      knum += 1
