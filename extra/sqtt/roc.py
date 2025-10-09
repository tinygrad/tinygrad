import ctypes, pathlib, argparse, pickle, re
from extra.sqtt.rocprof import rocprof
from tinygrad.helpers import temp, DEBUG
from tinygrad.device import ProfileEvent, ProfileProgramEvent
from tinygrad.runtime.ops_amd import ProfileSQTTEvent

THE_CTX = None
class CTX:
  def __init__(self, sqtt_evs:list[ProfileSQTTEvent], prog_evs:list[ProfileProgramEvent]):
    self.sqtt_evs = sqtt_evs
    self.prog_evs = prog_evs
    self.sqtt_ev_idx = 0
    self.prog_ev_idx = 0
    self.sz = 0
    self.buf_ptr = None

  def next_sqtt(self) -> ProfileSQTTEvent|None:
    if self.sqtt_ev_idx >= len(self.sqtt_evs): return None
    ev = self.sqtt_evs[self.sqtt_ev_idx]
    self.sqtt_ev_idx += 1
    return ev

def process_pc(pc):
  global THE_CTX
  print(hex(pc.address), pc.code_object_id, THE_CTX.prog_evs[pc.code_object_id].name)

@rocprof.rocprof_trace_decoder_se_data_callback_t
def copy_cb(buf, buf_size, data_ptr):
  global THE_CTX

  prof = THE_CTX.next_sqtt()
  if prof is None: return 0
  array = (ctypes.c_ubyte * len(prof.blob)).from_buffer_copy(prof.blob)

  buf[0] = ctypes.cast(array, ctypes.POINTER(ctypes.c_ubyte))
  buf_size[0] = len(prof.blob)
  return len(prof.blob)

@rocprof.rocprof_trace_decoder_trace_callback_t
def trace_cb(record_type, events_ptr, n, data_ptr):
  global THE_CTX
  print('trace', rocprof.rocprofiler_thread_trace_decoder_record_type_t__enumvalues[record_type], events_ptr, n)
  if record_type == rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY:
    x = (rocprof.rocprofiler_thread_trace_decoder_occupancy_t * n).from_address(events_ptr)
    for i in range(n):
      process_pc(x[i].pc)
      print('  ', x[i].time, x[i].cu, x[i].simd, x[i].wave_id, x[i].start)
  return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

@rocprof.rocprof_trace_decoder_isa_callback_t
def isa_cb(instr_ptr, mem_size_ptr, size_ptr, pc, data_ptr):
  global THE_CTX
  print('isa', instr_ptr, mem_size_ptr, size_ptr, pc)
  return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

def load_pickle(path:pathlib.Path|None) -> list:
  if path is None or not path.exists(): return []
  with path.open("rb") as f: return pickle.load(f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', type=pathlib.Path, help='Path to profile', default=pathlib.Path(temp("profile.pkl", append_user=True)))
  args = parser.parse_args()

  profile = load_pickle(args.profile)

  sqtt_events:list[ProfileSQTTEvent] = []
  prog_events:list[ProfileProgramEvent] = []
  for e in profile:
    if isinstance(e, ProfileSQTTEvent): sqtt_events.append(e)
    if isinstance(e, ProfileProgramEvent) and e.device.startswith("AMD"): prog_events.append(e)

  THE_CTX = CTX(sqtt_events, prog_events)
  rocprof.rocprof_trace_decoder_parse_data(copy_cb, trace_cb, isa_cb, None)

  # from hexdump import hexdump
  # hexdump(sqtt_events[0].blob)
  # print("-----")
  # hexdump(sqtt_events[1].blob)
