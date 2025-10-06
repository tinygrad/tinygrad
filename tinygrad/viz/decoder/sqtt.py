import ctypes, multiprocessing, enum, itertools
from dataclasses import dataclass
from tinygrad.helpers import init_c_struct_t, unwrap, fetch, DEBUG, pluralize
from tinygrad.device import ProfileEvent, ProfileProgramEvent
from tinygrad.runtime.ops_amd import ProfileSQTTEvent
from tinygrad.runtime.support.disassembler_amd import comgr_get_address_table

# ** base structs

# taken mostly from thread-trace/[trace_decoder_types, trace_decoder_api].h
# all timestamps are in hw clock units

TraceData = init_c_struct_t((
  # raw sqtt input buffer
  ("buf_ptr", ctypes.POINTER(ctypes.c_uint8)), ("sz", ctypes.c_uint64),
  # internal PC address table
  ("address_table", ctypes.py_object),
))

PC = init_c_struct_t((("addr", ctypes.c_size_t), ("marker_id", ctypes.c_size_t)))

# Describes an instruction execution event.
# time+duration marks execution completion time
# time+stall marks the successful issue time
# duration-stall is the execution time
Inst = init_c_struct_t((("category", ctypes.c_uint32, 8), ("stall", ctypes.c_uint32, 24), ("duration", ctypes.c_int32), ("time", ctypes.c_int64),
                        ("pc", PC)))

# instructions_array contains a time-ordered list of all (traced) instructions by the wave.
Wave = init_c_struct_t((("cu", ctypes.c_uint8), ("simd", ctypes.c_uint8), ("wave_id", ctypes.c_uint8), ("contexts", ctypes.c_uint8),
                        ("_rsvd1", ctypes.c_uint32), ("_rsvd2", ctypes.c_uint32), ("_rsvd3", ctypes.c_uint32), ("begin_time", ctypes.c_int64),
                        ("end_time", ctypes.c_int64), ("timeline_size", ctypes.c_size_t), ("instructions_size", ctypes.c_size_t),
                        ("timeline_array", ctypes.c_void_p), ("instructions_array", ctypes.POINTER(Inst))))

# Describes an occupancy event (wave started or wave ended).
Occupancy = init_c_struct_t((
  ("pc", PC),                    # Wave start address (program base)
  ("time", ctypes.c_uint64),
  ("reserved", ctypes.c_uint8),
  ("cu", ctypes.c_uint8),        # WGP ID
  ("simd", ctypes.c_uint8),      # SIMD ID [0,3] within compute unit
  ("slot", ctypes.c_uint8),      # Wave slot ID within SIMD
  ("start", ctypes.c_uint32, 1), # 1 if wave_start, 0 if a wave_end
  ("_rsvd", ctypes.c_uint32, 31)
))

class RecordType(enum.IntEnum): GFXIP = 0; OCCUPANCY = 1; WAVE = 3; INFO = 4 # noqa: E702

class DecoderStatus(enum.IntEnum): SUCCESS = 0; OUT_OF_RESOURCES = 2; INVALID_ARG = 3 # noqa: E702

# ** metrics

def decode_occupancy(events, n:int, data) -> None:
  print(f"Recieved {pluralize('occupancy event', n)}")
  wave_map:dict[tuple[int, int, int], tuple[int, int]] = {}
  wave_cnt = itertools.count(1)
  wave_timing:list[tuple[int, int, int]] = []
  simd_used:set[tuple[int, int]] = set()
  for i in range(n):
    e = events[i]
    print(f"Wave {'start' if e.start else 'end  '} clk={e.time} [cu={e.cu} simd={e.simd} slot={e.slot}]")
    simd_used.add((e.cu, e.simd))
    pc_unit = (e.pc.addr, e.cu, e.simd)
    if e.start: wave_map[pc_unit] = (e.time, next(wave_cnt))
    else:
      st, wave_id = wave_map.pop(pc_unit)
      wave_timing.append((st, e.time, wave_id))

  print(f"Traced {next(wave_cnt)} waves on HW units {simd_used}")
  for st,et,wave_id in wave_timing:
    print(f"Wave {wave_id:3d} took {et-st} clks")

@dataclass
class TimingInfo:
  hitcount:int=0
  latency:int=0
  stall:int=0
  idle:int=0

def decode_waves(events, n:int, data) -> None:
  print(f"Recieved {pluralize('wave event', n)}")
  timing_info:dict[int, TimingInfo] = {}
  for i in range(n):
    w = events[i]
    clk = w.begin_time
    for j in range(w.instructions_size):
      e = w.instructions_array[j]
      tmi = timing_info.setdefault(e.pc.addr, TimingInfo())
      tmi.hitcount += 1
      tmi.latency += e.duration
      tmi.stall += e.stall
      tmi.idle += e.time-clk
      clk = e.time
  for pc,tmi in timing_info.items():
    code, _ = data.address_table[pc]
    print(f"{code:<50} {tmi}")

# ** callbacks

@ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(TraceData))
def copy_cb(buf, buf_size, data_ptr):
  data = data_ptr.contents
  if DEBUG >= 2: print(f"[copy_cb] size={buf_size[0]} remaining={data.sz}")
  buf[0] = data.buf_ptr
  buf_size[0] = copied_sz = data.sz
  data.data = ctypes.cast(ctypes.addressof(data.buf_ptr.contents), ctypes.POINTER(ctypes.c_uint8))
  data.sz = 0
  return copied_sz

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(TraceData))
def trace_cb(record_type, events_ptr, n, data_ptr):
  if DEBUG >= 3: print(f"[trace_cb] {RecordType(record_type).name} {n}")
  match record_type:
    case RecordType.GFXIP: print(f"Decoding SQTT trace for gfx{events_ptr}")
    case RecordType.OCCUPANCY: decode_occupancy(ctypes.cast(events_ptr, ctypes.POINTER(Occupancy)), n, data_ptr.contents)
    case RecordType.WAVE: decode_waves(ctypes.cast(events_ptr, ctypes.POINTER(Wave)), n, data_ptr.contents)
  return DecoderStatus.SUCCESS

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), PC,
                  ctypes.POINTER(TraceData))
def isa_cb(instr_ptr, mem_size_ptr, size_ptr, pc, data_ptr):
  if DEBUG >= 3: print(f"[isa_cb] {pc.addr} {pc.marker_id}")
  data = data_ptr.contents
  instr, mem_size = data.address_table[pc.addr]
  # this is the number of bytes to next instruction, set to 0 for end_pgm
  if instr == "s_endpgm": mem_size = 0
  # truncate the instr if it doesn't fit
  if (max_sz:=size_ptr[0]) < 1:
    print(f"no space for instruction {instr}")
    return DecoderStatus.OUT_OF_RESOURCES
  if (str_sz:=len(instr_bytes:=instr.encode()))+1 > max_sz: str_sz = max_sz-1
  if str_sz > 0: ctypes.memmove(instr_ptr, instr_bytes, str_sz)
  instr_ptr[str_sz] = 0
  size_ptr[0] = str_sz+1
  mem_size_ptr[0] = mem_size
  return DecoderStatus.SUCCESS

# ** main loop

def worker(profile:list[ProfileEvent]):
  # get traces
  sqtt_events:list[ProfileSQTTEvent] = []
  prog_events:list[ProfileProgramEvent] = []
  for e in profile:
    if isinstance(e, ProfileSQTTEvent): sqtt_events.append(e)
    if isinstance(e, ProfileProgramEvent) and e.device.startswith("AMD"): prog_events.append(e)

  address_table = {}
  for e in prog_events:
    pc_base = unwrap(e.base)
    for k,v in comgr_get_address_table(unwrap(e.lib)).items(): address_table[k+pc_base] = v

  # pass to the "decoder", this doesn't ship with any of the AMD stuff. It is a standalone blob, no rocm install required.
  sqtt_blobs = b"".join([s.blob for s in sqtt_events])
  trace_data = TraceData(ctypes.cast(ctypes.create_string_buffer(sqtt_blobs), ctypes.POINTER(ctypes.c_uint8)), len(sqtt_blobs), address_table)
  decoder_path = fetch("https://github.com/ROCm/rocprof-trace-decoder/raw/5420409ad0963b2d76450add067b9058493ccbd0/releases/linux_glibc_2_28_x86_64/librocprof-trace-decoder.so")
  decoder = ctypes.CDLL(str(decoder_path))
  decoder.rocprof_trace_decoder_parse_data(copy_cb, trace_cb, isa_cb, ctypes.pointer(trace_data))

def decode_sqtt_packets(profile:list[ProfileEvent]):
  p = multiprocessing.Process(target=worker, args=(profile,))
  p.start()
  try:
    p.join()
  except KeyboardInterrupt:
    print("decoder is shutting down...")
    p.terminate()
    p.join()
