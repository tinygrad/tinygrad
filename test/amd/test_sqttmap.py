# test to compare every packet with the rocprof decoder
import unittest
import pickle, itertools
from typing import Iterator
from pathlib import Path
from tinygrad.helpers import DEBUG, getenv, temp
from tinygrad.renderer.amd.sqtt import print_packets, map_insts
from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm
from tinygrad.viz.serve import sqtt_timeline
from test.amd.disasm import disasm

import tinygrad
EXAMPLES_DIR = Path(tinygrad.__file__).parent.parent / "extra/sqtt/examples"

def rocprof_inst_traces_match(sqtt, prg, target):
  from tinygrad.viz.serve import amd_decode
  from extra.sqtt.roc import decode as roc_decode, InstExec
  addr_table = amd_decode(prg.lib, target)
  disasm_map = {addr+prg.base:inst for addr,inst in addr_table.items()}
  rctx = roc_decode([sqtt], {prg.tag:disasm_map})
  rwaves = rctx.inst_execs.get((sqtt.kern, sqtt.exec_tag), [])
  rwaves_iter:dict[int, list[Iterator[InstExec]]] = {} # wave unit (0-15) -> list of inst trace iterators for all executions on that unit
  for w in rwaves: rwaves_iter.setdefault(w.wave_id, []).append(w.unpack_insts())

  if not rwaves: return 0, 0, 0

  passed_insts = 0
  for pkt, info in map_insts(sqtt.blob, prg.lib, target):
    if DEBUG >= 2: print_packets([(pkt, info)])
    if info is None: continue
    if DEBUG >= 2: print(f"{' '*29}{disasm(info.inst)}")
    rocprof_inst = next(rwaves_iter[info.wave][0])
    ref_pc = rocprof_inst.pc-prg.base
    # always check pc matches
    assert ref_pc == info.pc, f"pc mismatch {ref_pc}:{disasm_map[rocprof_inst.pc]} != {info.pc}:{disasm(info.inst)}"
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
    for pkl_path in ([Path(temp("profile.pkl", append_user=True))] if getenv("LOAD_PROFILE") else sorted((EXAMPLES_DIR/cls.target).glob("*.pkl"))):
      with open(pkl_path, "rb") as f:
        data = pickle.load(f)
      sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
      kern_events = {e.tag:e for e in data if type(e).__name__ == "ProfileProgramEvent"}
      if sqtt_events and kern_events:
        cls.examples[pkl_path.stem] = (sqtt_events, kern_events, cls.target)

  def test_rocprof_inst_traces_match(self):
    for name, (events, kern_events, target) in self.examples.items():
      if "sync" in name and self.target.startswith("gfx12"):
        self.skipTest("our timestamps are off by a few cycles because rocprof patches timestamps for rdna4 barriers")
      for event in events:
        if not event.itrace: continue
        if event.kern not in kern_events: continue
        with self.subTest(example=name, kern=event.kern):
          passed_insts, n_waves, n_units = rocprof_inst_traces_match(event, kern_events[event.kern], target)
          if n_waves: print(f"{name}: passed for {passed_insts} instructions across {n_waves} waves scheduled on {n_units} wave units")

  def test_sqtt_timeline(self):
    for name, (events, kern_events, target) in self.examples.items():
      for event in events:
        if (p:=kern_events.get(event.kern)) is None: continue
        with self.subTest(example=name, kern=event.kern):
          # skip if there's no SQTT frequency data
          if not (timeline:=list(sqtt_timeline(event.blob, p.lib, target))): continue
          if not (frequency:=[e.key for e in timeline if type(e).__name__ == "ProfilePointEvent" and e.name == "freq_hz"]): continue
          mean = sum(frequency) / len(frequency)
          variance = sum((v - mean) ** 2 for v in frequency) / len(frequency)
          self.assertGreater(mean, 0)
          if DEBUG >= 2: print(f"{name:20s} SE:{event.se} {mean/1e9:.2f} GHz mean, {variance/1e18:.2f} GHz^2 variance")
          events = [e for e in timeline if type(e).__name__ == "ProfileRangeEvent"]
          insts, execs = 0, 0
          for e in events:
            if "EXEC" in e.device:
              if "ALT" not in e.name.display_name: execs += 1
            elif "WAVE" in e.device:
              # sopk/immediates don't get ALU/MEM EXEC
              if e.name.display_name not in {"IMMEDIATE", "IMMEDIATE_MASK", "JUMP", "JUMP_NO", "MESSAGE", "BARRIER", "BARRIER_SIGNAL",
                                             "WAVEEND", "WAVERDY"}: insts += 1
            # OTHER_ is its own stream, it's the INST from other SIMDs that share the same EXEC.
            elif e.device.startswith("OTHER"): continue
            else: raise Exception(f"timeline row must be INST or EXEC, got {e.device}")
          self.assertEqual(execs, insts)

  def test_wave_sync(self):
    for name, (events, kern_events, target) in self.examples.items():
      for event in events:
        wave_barriers = {}
        wmma_dispatch_ids, wmma_execs = [], []
        pkt_idxs:dict[str, itertools.count] = {}
        _pc_map = {}
        for e in sqtt_timeline(event.blob, kern_events[event.kern].lib, target):
          if type(e).__name__ == "ProfilePointEvent" and e.key == 'pcMap': _pc_map = e.arg
          if type(e).__name__ != "ProfileRangeEvent": continue
          idx = next(pkt_idxs.setdefault(e.device, itertools.count()))
          if e.name.display_name == "BARRIER": wave_barriers.setdefault(e.device, []).append(e)
          if (info:=e.name.ret) is None: continue
          # detect WMMA dispatches from ISA (works for both RDNA3 VALUINST and RDNA4 INST)
          if info.startswith("PC:") and "wmma" in _pc_map.get(int(info.replace("PC:", "")), ""): wmma_dispatch_ids.append(f"{e.device}-{idx}")
          if info.replace("LINK:", "") in wmma_dispatch_ids: wmma_execs.append(e)
        self.assertEqual(len(wmma_execs), len(wmma_dispatch_ids))
        for e in wmma_execs:
          assert e.en-e.st > 1, f"WMMA EXEC must show take more than one cycle, got {e}"
        if not wave_barriers: continue
        for row, events in wave_barriers.items():
          for e in events:
            assert e.en-e.st > 1, f"all barriers must have a duration greater than 1, got {e}"

class TestSQTTMapRDNA3(TestSQTTMapBase): target = "gfx1100"

class TestSQTTMapRDNA4(TestSQTTMapBase): target = "gfx1200"

class TestSQTTMapCDNA(TestSQTTMapBase):
  target = "gfx950"
  def test_rocprof_inst_traces_match(self): self.skipTest("requires timestamp patching to match rocprof, currently it's off by a few cycles")

if __name__ == "__main__":
  unittest.main()
