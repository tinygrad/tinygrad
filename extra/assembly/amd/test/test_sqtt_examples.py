#!/usr/bin/env python3
"""Tests for SQTT packet decoding using real captured examples."""
import pickle, unittest
from pathlib import Path
from tinygrad.helpers import DEBUG, colored
from extra.assembly.amd.sqtt import decode, LAYOUT_HEADER, WAVESTART, WAVEEND, INST, VALUINST, ALUEXEC, VMEMEXEC, PACKET_TYPES, InstOp, AluSrc, MemSrc

EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "sqtt/examples"

PACKET_COLORS = {
  "INST": "WHITE", "VALUINST": "BLACK", "VMEMEXEC": "yellow", "ALUEXEC": "yellow",
  "IMMEDIATE": "YELLOW", "IMMEDIATE_MASK": "YELLOW", "WAVERDY": "cyan", "WAVEALLOC": "cyan",
  "WAVEEND": "blue", "WAVESTART": "blue", "PERF": "magenta", "EVENT": "red", "EVENT_BIG": "red",
  "REG": "green", "LAYOUT_HEADER": "white", "SNAPSHOT": "white", "UTILCTR": "green",
}

def format_packet(p, time_offset: int = 0) -> str:
  name, cycle = type(p).__name__, p._time - time_offset
  if isinstance(p, INST):
    op_name = p.op.name if isinstance(p.op, InstOp) else f"0x{p.op:02x}"
    fields = f"wave={p.wave} op={op_name}" + (" flag1" if p.flag1 else "") + (" flag2" if p.flag2 else "")
  elif isinstance(p, VALUINST): fields = f"wave={p.wave}" + (" flag" if p.flag else "")
  elif isinstance(p, ALUEXEC): fields = f"src={p.src.name if isinstance(p.src, AluSrc) else p.src}"
  elif isinstance(p, VMEMEXEC): fields = f"src={p.src.name if isinstance(p.src, MemSrc) else p.src}"
  elif isinstance(p, (WAVESTART, WAVEEND)): fields = f"wave={p.wave} simd={p.simd} cu={p.cu}"
  elif hasattr(p, '_values'):
    fields = " ".join(f"{k}=0x{v:x}" if k in {'snap', 'val32'} else f"{k}={v}"
                      for k, v in p._values.items() if not k.startswith('_') and k != 'delta')
  else: fields = ""
  color = PACKET_COLORS.get(name, "white")
  return f"{cycle:8}: {colored(f'{name:18}', color)} {fields}"

def print_packets(packets: list) -> None:
  timing_skip = {"NOP", "TS_DELTA_SHORT", "TS_WAVE_STATE", "TS_DELTA_OR_MARK", "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_S8_W3"}
  extra_skip = {"REG", "EVENT"}
  time_offset = packets[0]._time if packets else 0
  for p in packets:
    if type(p).__name__ not in timing_skip.union(extra_skip): print(format_packet(p, time_offset))

class TestSQTTExamples(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.examples = {}
    for pkl_path in sorted(EXAMPLES_DIR.glob("*.pkl")):
      with open(pkl_path, "rb") as f:
        data = pickle.load(f)
      sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
      if sqtt_events: cls.examples[pkl_path.stem] = sqtt_events

  def test_examples_loaded(self):
    self.assertGreater(len(self.examples), 0, "no example files found")

  def test_decode_all_examples(self):
    for name, events in self.examples.items():
      for i, event in enumerate(events):
        with self.subTest(example=name, event=i):
          packets = decode(event.blob)
          if DEBUG >= 2: print(f"\n=== {name} event {i} ==="); print_packets(packets)
          self.assertGreater(len(packets), 0, f"no packets decoded from {name} event {i}")
          self.assertIsInstance(packets[0], LAYOUT_HEADER, f"first packet should be LAYOUT_HEADER in {name}")

  def test_packet_types_valid(self):
    for name, events in self.examples.items():
      for i, event in enumerate(events):
        with self.subTest(example=name, event=i):
          packets = decode(event.blob)
          for pkt in packets:
            self.assertIn(type(pkt), PACKET_TYPES, f"unknown packet type {type(pkt)} in {name}")

  def test_wave_lifecycle(self):
    for name, events in self.examples.items():
      if "empty" in name: continue
      with self.subTest(example=name):
        all_packets = [p for e in events for p in decode(e.blob)]
        starts = [p for p in all_packets if isinstance(p, WAVESTART)]
        ends = [p for p in all_packets if isinstance(p, WAVEEND)]
        self.assertGreater(len(starts), 0, f"no WAVESTART in {name}")
        self.assertGreater(len(ends), 0, f"no WAVEEND in {name}")

  def test_time_monotonic(self):
    for name, events in self.examples.items():
      for i, event in enumerate(events):
        with self.subTest(example=name, event=i):
          packets = decode(event.blob)
          times = [p._time for p in packets]
          self.assertEqual(times, sorted(times), f"timestamps not monotonic in {name}")

  def test_gemm_has_instructions(self):
    for name, events in self.examples.items():
      if "gemm" not in name: continue
      with self.subTest(example=name):
        all_packets = [p for e in events for p in decode(e.blob)]
        insts = [p for p in all_packets if isinstance(p, INST)]
        self.assertGreater(len(insts), 0, f"no INST packets in gemm example {name}")

  def test_rocprof_wave_times_match(self):
    """Compare wave start/end times against rocprof decoder."""
    import ctypes, threading
    from tinygrad.runtime.autogen import rocprof
    from tinygrad.viz.serve import llvm_disasm

    for name, events in self.examples.items():
      with self.subTest(example=name):
        # load full profile to get program binary and device props
        with open(EXAMPLES_DIR / f"{name}.pkl", "rb") as f:
          profile = pickle.load(f)
        prg = next((e for e in profile if type(e).__name__ == "ProfileProgramEvent"), None)
        dev = next((e for e in profile if type(e).__name__ == "ProfileDeviceEvent" and e.props is not None), None)
        assert prg is not None and dev is not None, f"missing program or device info in {name}"

        # decode with our decoder
        our_waves: dict[tuple[int, int, int], tuple[int, int]] = {}
        for event in events:
          packets = decode(event.blob)
          wave_starts: dict[tuple[int, int, int], int] = {}
          for p in packets:
            if isinstance(p, WAVESTART): wave_starts[(p.wave, p.simd, p.cu)] = p._time
            elif isinstance(p, WAVEEND) and (key := (p.wave, p.simd, p.cu)) in wave_starts:
              our_waves[key] = (wave_starts[key], p._time)

        # decode with rocprof
        base = prg.base
        disasm = {addr + base: inst_disasm for addr, inst_disasm in llvm_disasm(dev.props["gfx_target_version"], prg.lib).items()}
        roc_waves: dict[tuple[int, int, int], tuple[int, int]] = {}
        blobs = [e.blob for e in events]
        blob_iter = iter(blobs)
        current_blob = [None]
        wave_starts_roc: dict[tuple[int, int, int], int] = {}

        @rocprof.rocprof_trace_decoder_se_data_callback_t
        def copy_cb(buf, buf_size, _):
          blob = next(blob_iter, None)
          if blob is None: return 0
          current_blob[0] = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
          buf[0] = ctypes.cast(current_blob[0], ctypes.POINTER(ctypes.c_ubyte))
          buf_size[0] = len(current_blob[0])
          return len(current_blob[0])

        @rocprof.rocprof_trace_decoder_trace_callback_t
        def trace_cb(record_type, events_ptr, n, _):
          if record_type == rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY:
            for ev in (rocprof.rocprofiler_thread_trace_decoder_occupancy_t * n).from_address(events_ptr):
              key = (ev.wave_id, ev.simd, ev.cu)
              if ev.start: wave_starts_roc[key] = ev.time
              elif key in wave_starts_roc: roc_waves[key] = (wave_starts_roc.pop(key), ev.time)
          return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

        @rocprof.rocprof_trace_decoder_isa_callback_t
        def isa_cb(instr_ptr, mem_size_ptr, size_ptr, pc, _):
          if pc.address not in disasm:
            mem_size_ptr[0] = 0
            return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
          instr, mem_size_ptr[0] = disasm[pc.address]
          if instr == "s_endpgm": mem_size_ptr[0] = 0
          if (max_sz := size_ptr[0]) == 0: return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES
          instr_bytes = instr.encode()
          str_sz = min(len(instr_bytes), max_sz - 1)
          ctypes.memmove(instr_ptr, instr_bytes, str_sz)
          size_ptr[0] = str_sz
          return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

        t = threading.Thread(target=lambda: rocprof.rocprof_trace_decoder_parse_data(copy_cb, trace_cb, isa_cb, None), daemon=True)
        t.start()
        t.join(timeout=10)
        self.assertFalse(t.is_alive(), f"rocprof decoder hung on {name}")

        # compare times
        common_keys = set(our_waves.keys()) & set(roc_waves.keys())
        self.assertGreater(len(common_keys), 0, f"no common waves found in {name}")
        for key in common_keys:
          our_start, our_end = our_waves[key]
          roc_start, roc_end = roc_waves[key]
          self.assertEqual(our_start, roc_start, f"start time mismatch for wave {key} in {name}: ours={our_start} roc={roc_start}")
          self.assertEqual(our_end, roc_end, f"end time mismatch for wave {key} in {name}: ours={our_end} roc={roc_end}")

if __name__ == "__main__":
  unittest.main()
