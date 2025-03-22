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
        dsp_bufs = [Buffer("DSP", 1024+b.size*2, b.dtype).view(b.size, b.dtype, 512) for b in ei.bufs]
        if BEAM:
          from tinygrad.engine.search import beam_search
          k = beam_search(k, dsp_bufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
        elif not getenv("NOOPT"):
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
          elif knum == 29:
            #k.apply_opt(Opt(OptOps.UPCAST, 1, 2))
            k.apply_opt(Opt(OptOps.PADTO, 1, 128))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 256))
            #k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
          else:
            k.hand_coded_optimizations()
          """
          """
          if knum == 3:
            # 12544x32 * 32x16 -> 12544x16

            k.apply_opt(Opt(OptOps.UNROLL, 0, 0))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 16))
            k.apply_opt(Opt(OptOps.UPCAST, 0, 128//16))
            #k.apply_opt(Opt(OptOps.UPCAST, 0, 256//16))
            #k.apply_opt(Opt(OptOps.UPCAST, 0, 8))
            pass
          elif knum == 6:
            k.apply_opt(Opt(OptOps.UNROLL, 0, 8))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 0))
          elif knum == 4:
            # 12544x16 * 16x96 -> 12544x96
            # (with the biased add)
            #k.apply_opt(Opt(OptOps.UPCAST, 1, 96))
            #k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
            #k.apply_opt(Opt(OptOps.UNROLL, 0, 0))
            #k.apply_opt(Opt(OptOps.PADTO, 0, 3))
            pass
          elif knum == 13:
            # 784x144 * 144x32 -> 784x32
            #k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
            #k.apply_opt(Opt(OptOps.UNROLL, 0, 2))
            #k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
            #k.apply_opt(Opt(OptOps.UPCAST, 0, 2))
            #k.apply_opt(Opt(OptOps.UPCAST, 1, 32))
            pass
          elif knum == 20:
            # 784x192 * 192x32 -> 784x32
            k.apply_opt(Opt(OptOps.UNROLL, 0, 8))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 32))
            k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
          elif knum == 35:
            k.apply_opt(Opt(OptOps.UNROLL, 0, 128))
            k.apply_opt(Opt(OptOps.UPCAST, 0, 2))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 64))
          elif knum == 37:
            pass
          elif knum == 24:
            #k.apply_opt(Opt(OptOps.UNROLL, 0, 0))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 64))
            #k.apply_opt(Opt(OptOps.UPCAST, 0, 2))
          """
          #if knum in [7, 11, 14, 18]:
            # alignment issue?
            #pass
          if knum == 4:
            k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 96))
            k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
          elif knum == 6:
            k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 24))
            k.apply_opt(Opt(OptOps.UPCAST, 0, 16))
          elif knum == 11:
            k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 144))
            #k.apply_opt(Opt(OptOps.UPCAST, 0, 8))
          elif knum == 14:
            k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 192))
            k.apply_opt(Opt(OptOps.UPCAST, 0, 2))
          elif knum == 37:
            k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
            k.apply_opt(Opt(OptOps.UPCAST, 1, 384))
          else:
            full_shape = k.full_shape
            out_shape = k.sts[0].shape
            out_strides = k.sts[0].real_strides()
            if len(out_strides) == 3:
              if full_shape[1] < 128:
                if full_shape[2] <= 32: k.apply_opt(Opt(OptOps.UNROLL, 0, 0))
                else: k.apply_opt(Opt(OptOps.UNROLL, 0, 8))
                k.apply_opt(Opt(OptOps.UPCAST, 1, full_shape[1]))
                if out_strides[0] < 128:
                  upcast_0 = 128//out_strides[0]
                  if out_shape[0]%upcast_0 == 0 and upcast_0 != 1: k.apply_opt(Opt(OptOps.UPCAST, 0, upcast_0))
              elif full_shape[1] % 128 == 0:
                k.apply_opt(Opt(OptOps.UPCAST, 1, 128))
            elif len(out_strides) == 1:
              #if full_shape[0]%128 == 0: k.apply_opt(Opt(OptOps.UPCAST, 0, 128))
              pass
            #print("here", out_shape, out_strides, k.name)
            #k.hand_coded_optimizations()
          #if knum in [5]: k.apply_opt(Opt(OptOps.UPCAST, 1, 2))
        p2 = k.to_program()
        new_ei = replace(ei, prg=CompiledRunner(p2), bufs=dsp_bufs)
        new_ei.run()
      knum += 1
