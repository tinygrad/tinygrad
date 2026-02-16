# test to compare every packet with the rocprof decoder
import unittest, pickle
from typing import Iterator
from pathlib import Path
from tinygrad.helpers import DEBUG
from tinygrad.renderer.amd.sqtt import print_packets, map_insts
from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm
from test.amd.disasm import disasm

import tinygrad
EXAMPLES_DIR = Path(tinygrad.__file__).parent.parent / "extra/sqtt/examples"

def rocprof_inst_traces_match(sqtt, prg, target):
  from tinygrad.viz.serve import amd_decode
  from extra.sqtt.roc import decode as roc_decode, InstExec
  addr_table = amd_decode(prg.lib, target)
  disasm_map = {addr+prg.base:(disasm(inst), inst.size()) for addr,inst in addr_table.items()}
  rctx = roc_decode([sqtt], {prg.tag:disasm_map})
  rwaves = rctx.inst_execs.get((sqtt.kern, sqtt.exec_tag), [])
  rwaves_iter:dict[int, list[Iterator[InstExec]]] = {} # wave unit (0-15) -> list of inst trace iterators for all executions on that unit
  for w in rwaves: rwaves_iter.setdefault(w.wave_id, []).append(w.unpack_insts())

  if not rwaves: return 0, 0, 0

  passed_insts = 0
  for pkt, info in map_insts(sqtt.blob, prg.lib, target):
    if DEBUG >= 2: print_packets([pkt])
    if info is None: continue
    if DEBUG >= 2: print(f"{' '*29}{disasm(info.inst)}")
    rocprof_inst = next(rwaves_iter[info.wave][0])
    ref_pc = rocprof_inst.pc-prg.base
    # always check pc matches
    assert ref_pc == info.pc, f"pc mismatch {ref_pc}:{disasm_map[rocprof_inst.pc][0]} != {info.pc}:{disasm(info.inst)}"
    # special handling for s_endpgm, it marks the wave completion.
    if info.inst == s_endpgm():
      completed_wave = list(rwaves_iter[info.wave].pop(0))
      assert len(completed_wave) == 0, f"incomplete instructions in wave {info.wave}"
    # otherwise the packet timestamp is time + "stall"
    else:
      assert pkt._time == rocprof_inst.time+rocprof_inst.stall
    passed_insts += 1

  for k,v in rwaves_iter.items():
    assert len(v) == 0, f"incomplete wave {k}"

  return passed_insts, len(rwaves), len(rwaves_iter)

class TestSQTTMapBase(unittest.TestCase):
  target: str
  examples: dict

  @classmethod
  def setUpClass(cls):
    if cls is TestSQTTMapBase: raise unittest.SkipTest("base class")
    cls.examples = {}
    for pkl_path in sorted((EXAMPLES_DIR/cls.target).glob("*.pkl")):
      with open(pkl_path, "rb") as f:
        data = pickle.load(f)
      sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
      kern_events = {e.tag:e for e in data if type(e).__name__ == "ProfileProgramEvent"}
      if sqtt_events and kern_events:
        cls.examples[pkl_path.stem] = (sqtt_events, kern_events, cls.target)

  def test_rocprof_inst_traces_match(self):
    for name, (events, kern_events, target) in self.examples.items():
      for event in events:
        if not event.itrace: continue
        if event.kern not in kern_events: continue
        with self.subTest(example=name, kern=event.kern):
          passed_insts, n_waves, n_units = rocprof_inst_traces_match(event, kern_events[event.kern], target)
          if n_waves: print(f"{name}: passed for {passed_insts} instructions across {n_waves} waves scheduled on {n_units} wave units")

class TestSQTTMapRDNA3(TestSQTTMapBase): target = "gfx1100"

@unittest.skip("this doesn't work")
class TestSQTTMapRDNA4(TestSQTTMapBase): target = "gfx1200"

if __name__ == "__main__":
  unittest.main()
