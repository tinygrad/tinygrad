# the emulator's SQTT against 7900 XTX recordings of LDS-only kernels:
#   DEV=MOCK+AMD SQTT_BUFFER_SIZE=16 VIZ=-2 python3 -m pytest test/amd/test_sqtt_emu.py
import unittest, pickle, functools, lzma, statistics
from pathlib import Path
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Compiled
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import DEV
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp
from tinygrad.renderer.amd.sqtt import map_insts, ALUEXEC, VMEMEXEC, AluSrc, MemSrc
from tinygrad.viz.serve import amd_decode, get_arch, get_elf_section

EXAMPLES = Path(__file__).parents[2] / "extra/sqtt/examples/gfx1100"

def exec_class(inst) -> str:
  kind = type(inst).__name__
  return "D" if kind.startswith("DS") else "V" if kind.startswith("VOP") else "S" if kind.startswith(("SOP1", "SOP2", "SOPC", "SOPK")) else ""

def traced(blob:bytes, lib:bytes) -> tuple[list[dict], dict]:
  waves: dict[int, dict] = {}
  execs: dict[str, list[int]] = {"V":[], "S":[], "D":[]}
  for p, info in map_insts(blob, lib, "gfx1100"):
    if isinstance(p, ALUEXEC):
      if p.src in (AluSrc.VALU, AluSrc.VALU_SALU): execs["V"].append(p._time)
      if p.src in (AluSrc.SALU, AluSrc.VALU_SALU): execs["S"].append(p._time)
    if isinstance(p, VMEMEXEC) and p.src is MemSrc.LDS: execs["D"].append(p._time)
    if info is None: continue
    w = waves.setdefault(info.wave, {"insts":[], "pkts":[], "times":[]})
    # whether two waves' immediates share an IMMEDIATE_MASK is timing, not the instruction stream
    for k, v in (("insts", (info.pc, info.inst)), ("pkts", type(p).__name__.removesuffix("_MASK")), ("times", p._time)): w[k].append(v)
  ws, lags = [waves[k] for k in sorted(waves)], {}
  for c, xs in execs.items():
    ops = sorted((w["times"][k], j, k) for j, w in enumerate(ws) for k, (_, inst) in enumerate(w["insts"])
                 if w["pkts"][k] != "IMMEDIATE" and exec_class(inst) == c)
    # the emulator gives a WMMA no exec packet and can exec two vector ops in one cycle, so a kind that doesn't pair up is left out
    if len(ops) == len(xs): lags.update({(c, j, k): x - t for (t, j, k), x in zip(ops, sorted(xs))})
  return ws, lags

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
  tag = next(e.tag for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent" and e.name == f"sram_{name}")
  return next(e.blob for e in Compiled.profile_events if type(e).__name__ == "ProfileSQTTEvent" and e.kern == tag and e.se == 0)

def span(waves:list[dict]) -> int: return max(w["times"][-1] for w in waves) - min(w["times"][0] for w in waves)

def compare(name:str, threads:int, lib:bytes, blobs:list[bytes]) -> dict:
  # a delta or exec lag counts only if all runs agree on it
  runs = [traced(b, lib) for b in blobs]
  emu, emu_lags = traced(emulate(name, threads, lib), lib)
  out: dict = dict(stream=True, pkts=True, lags=emu_lags.keys() == runs[0][1].keys(), deltas=0, delta_misses=0, execs=0, exec_misses=0)
  for w, hw in zip(emu, zip(*(waves for waves, _ in runs)), strict=True):
    out["stream"] &= [pc for pc, _ in w["insts"]] == [pc for pc, _ in hw[0]["insts"]]
    out["pkts"] &= w["pkts"] == hw[0]["pkts"]
    for k in range(1, min(len(w["times"]), len(hw[0]["times"]))):
      if len(d:={h["times"][k] - h["times"][k-1] for h in hw}) > 1: continue
      out["deltas"], out["delta_misses"] = out["deltas"] + 1, out["delta_misses"] + (w["times"][k] - w["times"][k-1] not in d)
  for key, lag in emu_lags.items():
    if key not in runs[0][1] or len(d:={lags[key] for _, lags in runs}) > 1: continue
    out["execs"], out["exec_misses"] = out["execs"] + 1, out["exec_misses"] + (lag not in d)
  hw_span = statistics.median(span(waves) for waves, _ in runs)
  out["duration_err"] = abs(span(emu) - hw_span) / hw_span
  return out

# not cherry-picked: the first kernels of two random batches and of one with atomics, DPP, VOPD and WMMA, with every recorded run
SAMPLE: dict = pickle.loads(lzma.decompress((EXAMPLES / "sram/rand.pkl.xz").read_bytes()))
EXACT = {"rand1w_1800", "rand1w_1802", "rand1w_1803", "rand1w_1804", "rand1w_1805", "rand1w_1806", "rand1w_1807", "rand1w_1808", "rand1w_1809",
         "rand1w_1811", "rand1w_1813", "rand1w_1814", "rand1w_1815", "rand4w_2601", "rand4w_2605", "rand4w_2607", "ext1w_2007", "ext4w_2000"}
# fraction of exact deltas and exec lags, median and worst duration error
BOUNDS = {"rand1w": (0.995, 0.995, 0.02, 0.2), "rand4w": (0.97, 0.965, 0.02, 0.2), "ext1w": (0.97, 0.88, 0.02, 0.2), "ext4w": (0.91, 0.81, 0.02, 0.2)}

@unittest.skipUnless(Device.DEFAULT == "AMD" and DEV.interface.startswith("MOCK") and Device["AMD"].sqtt_enabled, "needs the mock in SQTT mode")
class TestSQTTEmu(unittest.TestCase):
  res: dict[str, dict]
  @classmethod
  def setUpClass(cls): cls.res = {name: compare(name, k["threads"], k["lib"], k["runs"]) for name, k in SAMPLE.items()}

  def test_stream(self):
    for name, r in self.res.items():
      with self.subTest(name): self.assertTrue(r["stream"] and r["pkts"] and (r["lags"] or name.startswith("ext")))

  def test_accuracy(self):
    for batch, (deltas, lags, med, worst) in BOUNDS.items():
      r = [v for name, v in self.res.items() if name.startswith(batch + "_")]
      with self.subTest(batch):
        self.assertGreaterEqual(1 - sum(v["delta_misses"] for v in r) / sum(v["deltas"] for v in r), deltas)
        self.assertGreaterEqual(1 - sum(v["exec_misses"] for v in r) / sum(v["execs"] for v in r), lags)
        self.assertLess(statistics.median(v["duration_err"] for v in r), med)
        self.assertLess(max(v["duration_err"] for v in r), worst)

  def test_exact(self):
    for name in EXACT:
      with self.subTest(name): self.assertEqual((self.res[name]["delta_misses"], self.res[name]["exec_misses"]), (0, 0))

if __name__ == "__main__":
  unittest.main()
