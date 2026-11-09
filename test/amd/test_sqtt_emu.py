import unittest, pickle, functools, lzma, statistics
from pathlib import Path
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Compiled, ProfileProgramEvent
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import DEV
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.ops_amd import ProfileSQTTEvent
from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp
from tinygrad.renderer.amd.sqtt import map_insts, IMMEDIATE, IMMEDIATE_MASK
from tinygrad.viz.serve import amd_decode, get_arch, get_elf_section

SAMPLE = Path(__file__).parents[2] / "extra/sqtt/examples/gfx1100/sram/rand.pkl.xz"

def waves(blob:bytes, lib:bytes) -> list[tuple[list, list[int]]]:
  ret: dict[int, tuple[list, list[int]]] = {}
  for p, info in map_insts(blob, lib, "gfx1100"):
    if info is None: continue
    insts, times = ret.setdefault(info.wave, ([], []))
    # two waves' immediates can share an IMMEDIATE_MASK, that depends on timing
    insts.append((info.pc, IMMEDIATE if isinstance(p, IMMEDIATE_MASK) else type(p)))
    times.append(p._time)
  return [ret[k] for k in sorted(ret)]

def span(ws) -> int: return max(t[-1] for _, t in ws) - min(t[0] for _, t in ws)

def kernel(A:UOp, name:str, threads:int, lib:bytes) -> UOp:
  insts = [x for x in amd_decode(get_elf_section(lib, ".text").content, get_arch("gfx1100")).values()
           if getattr(x, "op", None) is not SOPPOp.S_CODE_END]
  lds = UOp.placeholder((8192,), dtypes.uint8, 0, AddrSpace.LOCAL)
  sink = UOp.sink(A.base, lds, UOp.special(threads, "lidx0"), UOp.special(1, "gidx0"), arg=KernelInfo(f"sram_{name}"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

def emulate(name:str, threads:int, lib:bytes) -> bytes:
  a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=functools.partial(kernel, name=name, threads=threads, lib=lib))[0].realize()
  Device[Device.DEFAULT].synchronize()
  tag = next(e.tag for e in Compiled.profile_events if isinstance(e, ProfileProgramEvent) and e.name == f"sram_{name}")
  return next(e.blob for e in Compiled.profile_events if isinstance(e, ProfileSQTTEvent) and e.kern == tag and e.se == 0)

@unittest.skipUnless(Device.DEFAULT == "AMD" and DEV.interface.startswith("MOCK") and Device["AMD"].sqtt_enabled, "needs the mock in SQTT mode")
class TestSQTTEmu(unittest.TestCase):
  def test_recordings(self):
    # LDS-only kernels recorded on hw: name -> lib, threads, runs
    sample: dict = pickle.loads(lzma.decompress(SAMPLE.read_bytes()))
    hits, off = [], []
    for name, k in sample.items():
      runs = [waves(blob, k["lib"]) for blob in k["runs"]]
      emu = waves(emulate(name, k["threads"], k["lib"]), k["lib"])
      for (insts, t), *hw in zip(emu, *runs, strict=True):
        self.assertEqual(insts, hw[0][0], name)
        # only the issue deltas every recorded run agrees on
        for i in range(1, len(t)):
          if len(d := {ht[i] - ht[i-1] for _, ht in hw}) == 1: hits.append(t[i] - t[i-1] in d)
      off.append(abs(span(emu) / statistics.median(span(r) for r in runs) - 1))
    self.assertGreater(sum(hits) / len(hits), 0.9)
    self.assertLess(statistics.median(off), 0.02)
    # TODO: rand4w_2600 is 16% short, see Wave.refill
    self.assertLess(max(off), 0.2)

if __name__ == "__main__":
  unittest.main()
