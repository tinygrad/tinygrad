#!/usr/bin/env python3
"""Emulator SQTT timing vs real gfx1100 traces: listed per-wave instruction deltas must match hardware exactly.

Run with: DEV=MOCK+AMD python -m pytest test/amd/test_sqtt_timing.py
"""
import ctypes, pickle, unittest
from collections import defaultdict
from pathlib import Path
from tinygrad.helpers import Context
from tinygrad.renderer.amd.sqtt import decode, INST, VALUINST, IMMEDIATE, IMMEDIATE_MASK, WAVESTART, WAVEEND
from tinygrad.runtime.support.elf import elf_loader
from test.mockgpu.amd.emu import run_asm, sqtt_traces

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "extra/sqtt/examples/gfx1100"

# example -> (workgroup size, {(wave, instruction index)} whose delta matches hardware)
# the remaining deltas are DRAM bound or not modeled yet
MATCHING = {
  "sync": (128, {
    (0,0), (0,2), (0,3), (0,4), (0,5), (0,7), (0,8), (0,9), (0,10), (0,11), (0,12), (0,13), (1,2), (1,3), (1,6), (1,7), (1,8), (1,9), (1,10),
    (1,11), (1,12), (1,13)
  }),
  "plus": (1, {
    (0,1), (0,2), (0,3), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10)
  }),
  "gemm": (32, {
    (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,10), (0,11), (0,12), (0,13), (0,16), (0,17), (0,18), (0,20), (0,21), (0,24),
    (0,25), (0,26), (0,29), (0,30), (0,31), (0,32), (0,33), (0,34), (0,36), (0,37), (0,38), (0,39), (0,40), (0,41), (0,42), (0,43), (0,44),
    (0,45), (0,46), (0,47), (0,48), (0,49), (0,50), (0,51), (0,53), (0,54), (0,55), (0,56), (0,57)
  }),
}

def wave_deltas(blob:bytes) -> list[list[tuple[str, int]]]:
  """Per traced-SIMD wave: (kind, cycles since the previous packet of that wave), in wave order."""
  last:dict[int, int] = {}
  out:dict[int, list[tuple[str, int]]] = defaultdict(list)
  for p in decode(blob):
    if isinstance(p, WAVESTART): last[p.wave] = p._time
    elif isinstance(p, IMMEDIATE_MASK):
      for w in [w for w in last if p.mask & (1 << w)]:
        out[w].append(("IMMEDIATE", p._time - last[w]))
        last[w] = p._time
    elif isinstance(p, (INST, VALUINST, IMMEDIATE)) and p.wave in last:
      kind = p.op.name if isinstance(p, INST) else type(p).__name__
      if kind.startswith("OTHER_"): continue
      out[p.wave].append((kind.replace("SGMEM_RD_2", "SGMEM_RD_1"), p._time - last[p.wave]))
      last[p.wave] = p._time
    elif isinstance(p, WAVEEND) and p.wave in last: out[p.wave].append(("WAVEEND", p._time - last.pop(p.wave)))
  return [s for _, s in sorted(out.items()) if s]

class TestSQTTTiming(unittest.TestCase):
  def setUp(self): self.enterContext(Context(ASM_CALL=0))

  def test_examples(self):
    for name, (local, matching) in MATCHING.items():
      with self.subTest(example=name):
        data = pickle.load(open(EXAMPLES_DIR / f"profile_{name}_run_0.pkl", "rb"))
        real = next(w for e in data if type(e).__name__ == "ProfileSQTTEvent" and (w := wave_deltas(e.blob)))
        lib = next(e for e in data if type(e).__name__ == "ProfileProgramEvent" and e.lib).lib
        image, sections, _ = elf_loader(lib)
        text = next(s for s in sections if s.name == ".text")
        code = bytes(image[text.header.sh_addr:text.header.sh_addr + text.header.sh_size])
        bufs = [(ctypes.c_uint8 * (1 << 20))() for _ in range(4)]
        args = (ctypes.c_uint64 * 64)(*[ctypes.addressof(b) for b in bufs])
        cbuf = (ctypes.c_char * len(code)).from_buffer_copy(code)
        sqtt_traces.clear()
        with Context(PROFILE=1): run_asm(ctypes.addressof(cbuf), len(code), 1, 1, 1, local, 1, 1, ctypes.addressof(args))
        emu = next(w for b in sqtt_traces if (w := wave_deltas(b)))
        for w, i in sorted(matching):
          self.assertEqual(emu[w][i], real[w][i], f"{name} wave {w} instruction {i}")

if __name__ == "__main__":
  unittest.main()
