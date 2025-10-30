import ctypes, pathlib, argparse, pickle, re, functools, dataclasses, itertools
from extra.sqtt.rocprof import rocprof
from extra.sqtt.disasm import comgr_get_address_table
from tinygrad.helpers import temp, DEBUG
from tinygrad.device import ProfileEvent, ProfileProgramEvent
from tinygrad.runtime.ops_amd import ProfileSQTTEvent, ProfilePMCEvent

@dataclasses.dataclass
class InstInfo:
  typ:str=""
  inst:str=""
  hit:int=0
  lat:int=0
  stall:int=0
  def __str__(self): return f"{self.inst:>20} hits:{self.typ:>6} hits:{self.hit:>6} latency:{self.lat:>6} stall:{self.stall:>6}"

  def on_ev(self, ev):
    self.hit, self.lat, self.stall = self.hit + 1, self.lat + ev.duration, self.stall + ev.stall

class _ROCParseCtx:
  def __init__(self, sqtt_evs:list[ProfileSQTTEvent], prog_evs:list[ProfileProgramEvent]):
    self.sqtt_evs, self.prog_evs = iter(sqtt_evs), prog_evs
    self.wave_events, self.disasms, self.addr2prg = {}, {}, {}

    for prog in prog_evs:
      for addr, info in comgr_get_address_table(prog.lib).items():
        self.disasms[prog.base + addr] = info
        self.addr2prg[prog.base + addr] = prog

  def next_sqtt(self):
    x = next(self.sqtt_evs, None)
    self.active_se = x.se if x is not None else None
    return x

  def find_program(self, addr): return self.addr2prg[addr]

  def on_occupancy_ev(self, ev):
    if DEBUG >= 4: print("OCC", ev.time, self.active_se, ev.cu, ev.simd, ev.wave_id, ev.start)

  def on_wave_ev(self, ev):
    if DEBUG >= 4: print("WAVE", ev.wave_id, self.active_se, ev.cu, ev.simd, ev.contexts, ev.begin_time, ev.end_time)

    asm = {}
    for j in range(ev.instructions_size):
      inst_ev = ev.instructions_array[j]
      inst_typ = rocprof.rocprofiler_thread_trace_decoder_inst_category_t__enumvalues[inst_ev.category]
      asm.setdefault(inst_ev.pc.address, InstInfo(typ=inst_typ, inst=self.disasms[inst_ev.pc.address][0]))
      asm[inst_ev.pc.address].on_ev(inst_ev)

    self.wave_events[(self.find_program(ev.instructions_array[0].pc.address).name, ev.wave_id, ev.cu, ev.simd)] = asm

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', type=pathlib.Path, help='Path to profile', default=pathlib.Path(temp("profile.pkl", append_user=True)))
  args = parser.parse_args()

  with args.profile.open("rb") as f: profile = pickle.load(f)
  sqtt_events:list[ProfileSQTTEvent] = []
  pmc_events:list[ProfilePMCEvent] = []
  prog_events:list[ProfileProgramEvent] = []
  for e in profile:
    if isinstance(e, ProfileSQTTEvent): sqtt_events.append(e)
    if isinstance(e, ProfilePMCEvent): pmc_events.append(e)
    if isinstance(e, ProfileProgramEvent) and e.device.startswith("AMD"): prog_events.append(e)

  ROCParseCtx = _ROCParseCtx(sqtt_events, prog_events)

  @rocprof.rocprof_trace_decoder_se_data_callback_t
  def copy_cb(buf, buf_size, data_ptr):
    if (prof:=ROCParseCtx.next_sqtt()) is None: return 0
    buf[0] = ctypes.cast((ctypes.c_ubyte * len(prof.blob)).from_buffer_copy(prof.blob), ctypes.POINTER(ctypes.c_ubyte))
    buf_size[0] = len(prof.blob)
    return len(prof.blob)

  @rocprof.rocprof_trace_decoder_trace_callback_t
  def trace_cb(record_type, events_ptr, n, data_ptr):
    match record_type:
      case rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY:
        for ev in (rocprof.rocprofiler_thread_trace_decoder_occupancy_t * n).from_address(events_ptr): ROCParseCtx.on_occupancy_ev(ev)
      case rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:
        for ev in (rocprof.rocprofiler_thread_trace_decoder_wave_t * n).from_address(events_ptr): ROCParseCtx.on_wave_ev(ev)
      case _:
        if DEBUG >= 2: print(rocprof.rocprofiler_thread_trace_decoder_record_type_t__enumvalues[record_type], events_ptr, n)
    return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

  @rocprof.rocprof_trace_decoder_isa_callback_t
  def isa_cb(instr_ptr, mem_size_ptr, size_ptr, pc, data_ptr):
    instr, mem_size_ptr[0] = ROCParseCtx.disasms[pc.address]

    # this is the number of bytes to next instruction, set to 0 for end_pgm
    if instr == "s_endpgm": mem_size_ptr[0] = 0
    if (max_sz:=size_ptr[0]) == 0: return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES

    # truncate the instr if it doesn't fit
    if (str_sz:=len(instr_bytes:=instr.encode()))+1 > max_sz: str_sz = max_sz
    ctypes.memmove(instr_ptr, instr_bytes, str_sz)
    size_ptr[0] = str_sz

    return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

  try:
    rocprof.rocprof_trace_decoder_parse_data(copy_cb, trace_cb, isa_cb, None)
    print('SQTT:', ROCParseCtx.wave_events.keys())
  except Exception as e: print("Error in sqtt decoder:", e)

  for ev in pmc_events:
    print(f"PMC Event: dev={ev.device} kern={ev.kern}")
    ptr = 0
    for s in ev.sched:
      view = memoryview(ev.blob).cast('Q')
      print(f"\t{s.name}")
      for inst, se_idx, sa_idx, wgp_idx in itertools.product(range(s.inst), range(s.se), range(s.sa), range(s.wgp)):
        print(f"\t\tInst {inst} SE {se_idx} SA {sa_idx} WGP {wgp_idx}: {view[ptr]:#x}")
        ptr += 1
