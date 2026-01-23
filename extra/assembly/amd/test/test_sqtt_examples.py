#!/usr/bin/env python3
"""Tests for SQTT packet decoding using real captured examples."""
import pickle, unittest
from pathlib import Path
from tinygrad.helpers import DEBUG
from extra.assembly.amd.sqtt import (decode, LAYOUT_HEADER, WAVESTART, WAVESTART_L4, WAVEEND, INST, INST_L4, VALUINST, IMMEDIATE, IMMEDIATE_MASK,
                                     ALUEXEC, VMEMEXEC, PACKET_TYPES_L3, PACKET_TYPES_L4, InstOp, InstOpL4, print_packets)

EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "sqtt/examples"
# INST ops for non-traced SIMDs (excluded from instruction count)
OTHER_SIMD_OPS = {InstOp.OTHER_LDS_LOAD, InstOp.OTHER_LDS_STORE, InstOp.OTHER_LDS_STORE_64, InstOp.OTHER_LDS_STORE_128,
                  InstOp.OTHER_FLAT_LOAD, InstOp.OTHER_FLAT_STORE, InstOp.OTHER_FLAT_STORE_64, InstOp.OTHER_FLAT_STORE_96,
                  InstOp.OTHER_FLAT_STORE_128, InstOp.OTHER_GLOBAL_LOAD, InstOp.OTHER_GLOBAL_LOAD_VADDR,
                  InstOp.OTHER_GLOBAL_STORE_64, InstOp.OTHER_GLOBAL_STORE_96, InstOp.OTHER_GLOBAL_STORE_128,
                  InstOp.OTHER_GLOBAL_STORE_VADDR_128}
OTHER_SIMD_OPS_L4 = {InstOpL4.OTHER_VMEM, InstOpL4.UNK_60}

# ═══════════════════════════════════════════════════════════════════════════════
# ROCPROF DECODER
# ═══════════════════════════════════════════════════════════════════════════════

def run_rocprof_decoder(events: list, lib: bytes, base: int, target: int):
  """Run rocprof decoder on SQTT blobs, returning raw occupancy and instruction records."""
  from tinygrad.viz.serve import llvm_disasm
  from extra.sqtt.roc import decode as roc_decode
  occupancy_records: list[tuple[int, int, int, int, bool]] = []  # (wave_id, simd, cu, time, is_start)
  wave_insts: list[list[tuple[int, int]]] = []  # per-wave list of (time, stall)
  disasm = {addr+base:inst_disasm for addr, inst_disasm in llvm_disasm(110000, lib).items()}
  rctx = roc_decode(events, {(e:=events[0]).kern:disasm})
  occ_events = rctx.occ_events[(e.kern, e.exec_tag)]
  wave_events = rctx.inst_execs.get((e.kern, e.exec_tag), [])
  for e in occ_events: occupancy_records.append((e.wave_id, e.simd, e.cu, e.time, e.start))
  for e in wave_events: wave_insts.append([(i.time, i.stall) for i in e.unpack_insts()])
  return occupancy_records, wave_insts

class SQTTExamplesTestBase(unittest.TestCase):
  target: str

  @classmethod
  def setUpClass(cls):
    if cls is SQTTExamplesTestBase: raise unittest.SkipTest("base class")
    cls.examples = {}
    for pkl_path in sorted((EXAMPLES_DIR/cls.target).glob("*.pkl")):
      with open(pkl_path, "rb") as f:
        data = pickle.load(f)
      sqtt_events:dict[str, list] = {}
      for e in data:
        if type(e).__name__ == "ProfileDeviceEvent" and e.device.startswith("AMD"): cls.gfx_num = e.props["gfx_target_version"]
        if type(e).__name__ == "ProfileSQTTEvent": sqtt_events.setdefault(e.kern, []).append(e)
      prg = {e.name:e for e in data if type(e).__name__ == "ProfileProgramEvent"}
      for name, events in sqtt_events.items():
        cls.examples[pkl_path.stem+"_"+name] = (events, prg[name].lib, prg[name].base)

  def test_examples_loaded(self):
    self.assertGreater(len(self.examples), 0, "no example files found")

  def test_decode_all_examples(self):
    for name, (events, *_) in self.examples.items():
      for i, event in enumerate(events):
        with self.subTest(example=name, event=i):
          packets = list(decode(event.blob))
          if DEBUG >= 2: print(f"\n=== {name} event {i} ==="); print_packets(packets)
          self.assertGreater(len(packets), 0, f"no packets decoded from {name} event {i}")
          self.assertIsInstance(packets[0], LAYOUT_HEADER, f"first packet should be LAYOUT_HEADER in {name}")

  def test_packet_types_valid(self):
    all_classes = set(PACKET_TYPES_L3.values()) | set(PACKET_TYPES_L4.values())
    for name, (events, *_) in self.examples.items():
      for i, event in enumerate(events):
        with self.subTest(example=name, event=i):
          for pkt in decode(event.blob):
            # Use isinstance to handle layout-specific subclasses (e.g., WAVESTART_L4)
            self.assertTrue(any(isinstance(pkt, cls) for cls in all_classes), f"unknown packet type {type(pkt)} in {name}")

  def test_wave_lifecycle(self):
    for name, (events, *_) in self.examples.items():
      if "empty" in name: continue
      with self.subTest(example=name):
        all_packets = [p for e in events for p in decode(e.blob)]
        self.assertGreater(len([p for p in all_packets if isinstance(p, (WAVESTART, WAVESTART_L4))]), 0, f"no WAVESTART in {name}")
        self.assertGreater(len([p for p in all_packets if isinstance(p, WAVEEND)]), 0, f"no WAVEEND in {name}")

  def test_time_monotonic(self):
    for name, (events, *_) in self.examples.items():
      for i, event in enumerate(events):
        with self.subTest(example=name, event=i):
          times = [p._time for p in decode(event.blob)]
          self.assertEqual(times, sorted(times), f"timestamps not monotonic in {name}")

  def test_gemm_has_instructions(self):
    for name, (events, *_) in self.examples.items():
      if "gemm" not in name: continue
      with self.subTest(example=name):
        all_packets = [p for e in events for p in decode(e.blob)]
        self.assertGreater(len([p for p in all_packets if isinstance(p, (INST, INST_L4))]), 0, f"no INST packets in {name}")

  expected: dict[str, list[int]] = {}  # override in subclasses
  def test_packet_counts(self):
    if not self.expected: self.skipTest("no expected packet counts for this target")
    for name, (events, *_) in self.examples.items():
      with self.subTest(example=name):
        counts = [len(list(decode(e.blob))) for e in events]
        self.assertEqual(counts, self.expected[name], f"packet count mismatch in {name}")

  def test_rocprof_wave_times_match(self):
    """Wave start/end times must match rocprof exactly."""
    for name, (events, lib, base) in self.examples.items():
      with self.subTest(example=name):
        occupancy, _ = run_rocprof_decoder(events, lib, base, self.gfx_num)
        # extract from rocprof occupancy records
        roc_starts: dict[tuple[int, int, int], int] = {}
        roc_waves: list[tuple[int, int]] = []
        for wave_id, simd, cu, time, is_start in occupancy:
          key = (wave_id, simd, cu)
          if is_start: roc_starts[key] = time
          elif key in roc_starts: roc_waves.append((roc_starts.pop(key), time))
        # extract from our decoder
        our_waves: list[tuple[int, int]] = []
        for event in events:
          wave_starts: dict[tuple[int, int, int], int] = {}
          for p in decode(event.blob):
            if isinstance(p, (WAVESTART, WAVESTART_L4)): wave_starts[(p.wave, p.simd, p.cu)] = p._time
            elif isinstance(p, WAVEEND) and (key := (p.wave, p.simd, p.cu)) in wave_starts:
              our_waves.append((wave_starts[key], p._time))
        self.assertEqual(sorted(our_waves), sorted(roc_waves), f"wave times mismatch in {name}")

  def test_rocprof_inst_times_match(self):
    """Instruction times must match rocprof exactly (excluding s_endpgm)."""
    for name, (events, lib, base) in self.examples.items():
      with self.subTest(example=name):
        _, wave_insts = run_rocprof_decoder(events, lib, base, self.gfx_num)
        # skip last inst per wave (s_endpgm) - it needs special handling (time + duration instead of time + stall)
        roc_insts = [time + stall for insts in wave_insts for time, stall in insts[:-1]]
        # extract from our decoder
        our_insts: list[int] = []
        for event in events:
          for p in decode(event.blob):
            if isinstance(p, INST) and p.op not in OTHER_SIMD_OPS: our_insts.append(p._time)
            elif isinstance(p, INST_L4) and p.op not in OTHER_SIMD_OPS_L4: our_insts.append(p._time)
            elif isinstance(p, VALUINST): our_insts.append(p._time)
            elif isinstance(p, IMMEDIATE): our_insts.append(p._time)
            elif isinstance(p, IMMEDIATE_MASK):
              for _ in range(bin(p.mask).count('1')): our_insts.append(p._time)
        self.assertEqual(sorted(our_insts), sorted(roc_insts), f"instruction times mismatch in {name}")

class TestSQTTExamplesRDNA3(SQTTExamplesTestBase):
  target = "gfx1100"
  expected = {
    "profile_empty_run_0": [1803, 1908, 1928, 1979, 2006, 1912],
    "profile_empty_run_1": [1803, 1908, 1928, 1979, 2006, 1912],
    "profile_gemm_run_0": [2531, 1844, 1864, 1915, 1942, 1848, 3074, 1919, 1939, 1990, 2017, 1923, 19026, 1919, 1939, 1990, 2017, 1929],
    "profile_gemm_run_1": [2554, 1844, 1864, 1915, 1942, 1848, 3084, 1919, 1939, 1990, 2017, 1923, 19010, 1919, 1939, 1990, 2017, 1923],
    "profile_plus_run_0": [1900, 1908, 1928, 1979, 2006, 1912],
    "profile_plus_run_1": [1856, 1908, 1928, 1979, 2006, 1912],
  }

class TestSQTTExamplesRDNA4(SQTTExamplesTestBase):
  target = "gfx1200"

#class TestSQTTExamplesCDNA(TestSQTTExamples): target = "gfx950"

if __name__ == "__main__":
  unittest.main()
