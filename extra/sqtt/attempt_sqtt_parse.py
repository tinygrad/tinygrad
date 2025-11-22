import pickle, sys
from tinygrad.helpers import getenv, Timing, colored
from extra.sqtt.roc import decode, ProfileSQTTEvent

# Instruction packets (one per ISA op)
# NOTE: these are bad guesses and may be wrong! feel free to update if you know better
# some names were taken from SQ_TT_TOKEN_MASK_TOKEN_EXCLUDE_SHIFT

# we see 18 opcodes
# opcodes(18):  1  2  3  4  5  6  8  9  F 10 11 12 14 15 16 17 18 19
# if you exclude everything, you are left with 6
# opcodes( 6): 10 11 14 15 16 17
# sometimes we see a lot of B, but not repeatable

# not seen
# 7 A C

GOOD_OPCODE_NAMES = {
  # gated by SQ_TT_TOKEN_EXCLUDE_VALUINST_SHIFT (but others must be enabled for it to show)
  0x01: "VALUINST",
  # gated by SQ_TT_TOKEN_EXCLUDE_VMEMEXEC_SHIFT
  0x02: "VMEMEXEC",
  # gated by SQ_TT_TOKEN_EXCLUDE_ALUEXEC_SHIFT
  0x03: "ALUEXEC",
  # gated by SQ_TT_TOKEN_EXCLUDE_IMMEDIATE_SHIFT
  0x04: "IMMEDIATE_4",
  0x05: "IMMEDIATE_5",
  # gated by SQ_TT_TOKEN_EXCLUDE_WAVERDY_SHIFT
  0x06: "WAVERDY",
  # gated by SQ_TT_TOKEN_EXCLUDE_WAVESTARTEND_SHIFT
  0x08: "WAVEEND",
  0x09: "WAVESTART",
  # gated by NOT SQ_TT_TOKEN_EXCLUDE_PERF_SHIFT
  0x0D: "PERF",
  # pure time
  0x0F: "TS_DELTA_SHORT_PLUS4",     # short delta; ROCm adds +4 before accumulate
  0x10: "NOP",
  # gated by SQ_TT_TOKEN_EXCLUDE_EVENT_SHIFT
  0x12: "EVENT",
  # some gated by SQ_TT_TOKEN_EXCLUDE_REG_SHIFT, some always there
  0x14: "REG",
  # marker
  0x16: "TS_DELTA36_OR_MARK",       # 36-bit long delta or 36-bit marker
  # this is the first packet
  0x17: "LAYOUT_MODE_HEADER",       # layout/mode/group + selectors A/B
  # gated by SQ_TT_TOKEN_EXCLUDE_INST_SHIFT
  0x18: "INST",
  # gated by SQ_TT_TOKEN_EXCLUDE_UTILCTR_SHIFT
  0x19: "UTILCTR",
}

OPCODE_NAMES = {
  **GOOD_OPCODE_NAMES,

  # ------------------------------------------------------------------------
  # 0x07–0x0F: pure timestamp-ish deltas
  # ------------------------------------------------------------------------
  0x07: "TS_DELTA_S8_W3",           # shift=8,  width=3  (small delta)
  0x0A: "TS_DELTA_S5_W2_A",         # shift=5,  width=2
  0x0B: "TS_DELTA_S5_W3_A",         # shift=5,  width=3
  0x0C: "TS_DELTA_S5_W3_B",         # shift=5,  width=3 (different consumer)

  # ------------------------------------------------------------------------
  # 0x10–0x19: timestamps, layout headers, events, perf
  # ------------------------------------------------------------------------

  0x11: "TS_WAVE_STATE_SAMPLE",     # wave stall/termination sample (byte at +10)
  0x13: "EVT_SMALL_GENERIC",        # same structural family as 0x08/0x12/0x19
  0x15: "PERFCOUNTER_SNAPSHOT",     # small delta + 50-ish bits of snapshot
}

# these tables are from rocprof trace decoder
# rocprof_trace_decoder_parse_data-0x11c6a0
# parse_sqtt_180 = b *rocprof_trace_decoder_parse_data-0x11c6a0+0x110040

# ---------- 1. local_138: 256-byte state->opcode table ----------

STATE_TO_OPCODE: bytes = bytes([
  0x10, 0x16, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x17, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x07, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x19, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x00, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x11, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x12, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x15, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x16, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x17, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x07, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x19, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x00, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x11, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x13, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x15, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
])

# opcode mask (the bits used to determine the opcode, worked out by looking at the repeats in STATE_TO_OPCODE)

opcode_mask = {
  0x10: 0b1111,

  0x16: 0b1111111,
  0x17: 0b1111111,
  0x07: 0b1111111,
  0x19: 0b1111111,
  0x11: 0b1111111,
  0x12: 0b11111111,
  0x13: 0b11111111,
  0x15: 0b1111111,

  0x18: 0b111,
  0x1: 0b111,

  0x5: 0b11111,
  0x6: 0b11111,
  0xb: 0b11111,
  0x8: 0b11111,
  0xc: 0b11111,
  0xd: 0b11111,

  0xf: 0b1111,
  0x14: 0b1111,

  0x9: 0b11111,
  0xa: 0b11111,

  0x4: 0b1111,
  0x3: 0b1111,
  0x2: 0b1111,
}

# ---------- 2. DAT_0012e280: nibble budget per opcode&0x1F ----------

NIBBLE_BUDGET = [
  0x08, 0x0C, 0x08, 0x08, 0x0C, 0x18, 0x18, 0x40, 0x14, 0x20, 0x30, 0x14, 0x34, 0x1C, 0x30, 0x08,
  0x04, 0x18, 0x18, 0x20, 0x40, 0x40, 0x30, 0x40, 0x14, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
]

# ---------- 3. delta_map from your hash nodes ----------

# opcode -> (shift, width)
DELTA_MAP_DEFAULT = {
  0x01: (3,  3),   # shift=3,  end=6
  0x02: (4,  2),   # shift=4,  end=6
  0x03: (4,  2),   # shift=4,  end=6
  0x04: (4,  3),   # shift=4,  end=7
  0x05: (5,  3),   # shift=5,  end=8
  0x06: (5,  3),   # shift=5,  end=8
  0x07: (8,  3),   # shift=8,  end=11
  0x08: (5,  3),   # shift=5,  end=8
  0x09: (5,  2),   # shift=5,  end=7
  0x0A: (5,  2),   # shift=5,  end=7
  0x0B: (5,  3),   # shift=5,  end=8
  0x0C: (5,  3),   # shift=5,  end=8
  0x0D: (5,  3),   # shift=5,  end=8
  # NOTE: 0x0e can never be decoded, it's not in the STATE_TO_OPCODE table
  #0x0E: (7,  2),   # shift=7,  end=9
  0x0F: (4,  4),   # shift=4,  end=8
  0x10: (0,  0),   # shift=0,  end=0  (no delta)
  0x11: (7,  9),   # shift=7,  end=16
  0x12: (8,  3),   # shift=8,  end=11
  0x13: (8,  3),   # shift=8,  end=11
  0x14: (4,  3),   # shift=4,  end=7
  0x15: (7,  3),   # shift=7,  end=10
  0x16: (12, 36),  # shift=12, end=48 (36-bit field, matches the 0x16 special-case)
  0x17: (0,  0),   # shift=0,  end=0  (no delta)
  0x18: (4,  3),   # shift=4,  end=7
  0x19: (7,  2),   # shift=7,  end=9
}

# ---------- 4. One-line-per-packet parser ----------

def reg_mask(opcode):
  nb_bits = NIBBLE_BUDGET[opcode & 0x1F]
  shift, width = DELTA_MAP_DEFAULT[opcode]
  delta_mask = ((1 << width) - 1) << shift
  assert delta_mask & opcode_mask[opcode] == 0, "masks shouldn't overlap"
  return ((1 << nb_bits) - 1) & ~(delta_mask | opcode_mask[opcode])

def decode_packet_fields(opcode: int, reg: int) -> str:
  """
  Decode packet payloads conservatively, using:
    - NIBBLE_BUDGET[opcode & 0x1F] to mask reg down to true width.
    - DELTA_MAP_DEFAULT[opcode] to expose the "primary" field (often delta).
    - Per-opcode layouts derived from rocprof's decompiled consumers.
  """
  # --- 0. Restrict to real packet bits not used in delta ---------------------------------
  pkt = reg & reg_mask(opcode)
  fields: list[str] = []

  match opcode:
    case 0x01: # VALUINST
      # 6 bit field
      fields.append(f"type = {pkt>>6:X}")
    case 0x02: # VMEMEXEC
      fields.append(f"type = {pkt>>6:X}")
    case 0x03: # ALUEXEC
      fields.append(f"type = {pkt>>6:X}")
    case 0x04: # IMMEDIATE_4
      # 5 bit field
      fields.append(f"type = {pkt>>7:X}")
    case 0x05: # IMMEDIATE_5
      # 16 bit field
      fields.append(f"type = {pkt>>8:X}")
    case 0x0d:
      # 20 bit field
      fields.append(f"arg = {pkt>>8:X}")
    case 0x12:
      fields.append(f"event = {pkt>>11:X}")
    case 0x15:
      fields.append(f"snap = {pkt>>10:X}")
    case 0x19:
      # wave end
      fields.append(f"ctr = {pkt>>9:X}")
    case 0x11:
      # DELTA_MAP_DEFAULT: shift=7, width=9 -> small delta.
      coarse    = pkt >> 16
      fields.append(f"coarse=0x{coarse:02x}")
      # From decomp:
      #  - when layout<3 and coarse&1, it sets a "has interesting wave" flag
      #  - when coarse&8, it marks all live waves as "terminated"
      if coarse & 0x01:
        fields.append("flag_wave_interest=1")
      if coarse & 0x08:
        fields.append("flag_terminate_all=1")
    case 0x6:
      # wave ready
      fields.append(f"wave = {pkt>>8:X}")
    case 0x8:
      # wave end
      fields.append(f"wave = {pkt>>8:X}")
    case 0x9:
      # From case 9 (WAVESTART) in multiple consumers:
      #   flag7  = (w >> 7) & 1        (low bit of uVar41)
      #   cls2   = (w >> 8) & 3        (class / group)
      #   slot4  = (w >> 10) & 0xf     (slot / group index)
      #   idx_lo = (w >> 0xd) & 0x1f   (low index, layout<4 path)
      #   idx_hi = (w >> 0xf) & 0x1f   (high index, layout>=4 path)
      #   id7    = (w >> 0x19) & 0x7f  (7-bit id)
      flag7   = (pkt >> 7) & 0x1
      cls2    = (pkt >> 8) & 0x3
      slot4   = (pkt >> 10) & 0xF
      idx_lo  = (pkt >> 13) & 0x1F
      idx_hi  = (pkt >> 15) & 0x1F
      id7     = (pkt >> 25) & 0x7F
      fields.append(f"flag7={flag7}")
      fields.append(f"cls2={cls2}")
      fields.append(f"slot4=0x{slot4:x}")
      fields.append(f"idx_lo5=0x{idx_lo:x}")
      fields.append(f"idx_hi5=0x{idx_hi:x}")
      fields.append(f"id7=0x{id7:x}")
    case 0x18:
      # From case 0x18:
      #   low3   = w & 7
      #   grp3   = (w >> 3) or (w >> 4) & 7   (layout-dependent)
      #   flags  = bits 6 (B6) and 7 (B7)
      #   hi8    = (w >> 0xc) & 0xff   (layout 4 path)
      #   hi7    = (w >> 0xd) & 0x7f   (other layouts)
      #   idx5   = (w >> 7) or (w >> 8) & 0x1f, used as wave index
      low3     = pkt & 0x7
      grp3_a   = (pkt >> 3) & 0x7
      grp3_b   = (pkt >> 4) & 0x7
      flag_b6  = (pkt >> 6) & 0x1
      flag_b7  = (pkt >> 7) & 0x1
      idx5_a   = (pkt >> 7) & 0x1F
      idx5_b   = (pkt >> 8) & 0x1F
      hi8      = (pkt >> 12) & 0xFF
      hi7      = (pkt >> 13) & 0x7F

      fields.append(f"low3={low3:x}")
      fields.append(f"grp3_a={grp3_a:x}")
      fields.append(f"grp3_b={grp3_b:x}")
      fields.append(f"flag_b6={flag_b6}")
      fields.append(f"flag_b7={flag_b7}")
      fields.append(f"idx5_a={idx5_a:x}")
      fields.append(f"idx5_b={idx5_b:x}")
      fields.append(f"hi8={hi8:02x}")
      fields.append(f"hi7={hi7:02x}")
    case 0x14:
      subop   = (pkt >> 16) & 0xFFFF       # (short)(w >> 0x10)
      val32   = (pkt >> 32) & 0xFFFFFFFF   # (uint)(w >> 0x20)
      slot    = (pkt >> 7) & 0x7           # index in local_168[...] tables
      hi_byte = (pkt >> 8) & 0xFF          # determines config vs marker

      fields.append(f"subop=0x{subop:04x}")
      fields.append(f"slot={slot}")
      fields.append(f"val32=0x{val32:08x}")

      if hi_byte & 0x80:
        # Config flavour: writes config words into per-slot state arrays.
        fields.append("kind=config")
        if subop == 0x000C:
          fields.append("slot=lo")
        elif subop == 0x000D:
          fields.append("slot=hi")
      else:
        # COR marker: subop 0xC342, payload "COR\0" → start of a COR region.
        if subop == 0xC342:
          fields.append("kind=cor_stream")
          if val32 == 0x434F5200:
            fields.append("cor_magic='COR\\0'")
    case 0x16:
      # Bits:
      #   bit8  -> 0x100
      #   bit9  -> 0x200
      #   bits 12..47 -> 36-bit field used as delta or marker
      bit8 = bool(pkt & 0x100)
      bit9 = bool(pkt & 0x200)
      if not bit9:
        mode = "delta"
      elif not bit8:
        mode = "marker"
      else:
        mode = "other"
      # need to use reg here
      val36 = (reg >> 12) & ((1 << 36) - 1)
      fields.append(f"mode={mode}")
      if mode != "delta":
        fields.append(f"val36=0x{val36:x}")
    case 0x17:
      # From decomp (two sites with identical logic):
      #   layout = (w >> 7) & 0x3f
      #   mode   = (w >> 0xd) & 3
      #   group  = (w >> 0xf) & 7
      #   sel_a  = (w >> 0x1c) & 0xf
      #   sel_b  = (w >> 0x21) & 7
      #   flag4  = (w >> 0x3b) & 1  (only meaningful when layout == 4)
      layout = (pkt >> 7)  & 0x3F
      mode   = (pkt >> 13) & 0x3
      group  = (pkt >> 15) & 0x7
      sel_a  = (pkt >> 0x1C) & 0xF
      sel_b  = (pkt >> 0x21) & 0x7
      flag4  = (pkt >> 0x3B) & 0x1

      fields.append(f"layout={layout}")
      fields.append(f"group={group}")
      fields.append(f"mode={mode}")
      fields.append(f"sel_a={sel_a}")
      fields.append(f"sel_b={sel_b}")
      if layout == 4:
        fields.append(f"layout4_flag={flag4}")
    case _:
      fields.append(f"& {reg_mask(opcode):X}")
  return ",".join(fields)

FILTER_LEVEL = getenv("FILTER", 2)

DEFAULT_FILTER = tuple()
# NOP + pure time
if FILTER_LEVEL >= 0: DEFAULT_FILTER += (0x10, 0xf)
# reg + event + sample + marker
if FILTER_LEVEL >= 1: DEFAULT_FILTER += (0x11, 0x12, 0x14, 0x16)
# instructions and runs + waverdy
if FILTER_LEVEL >= 3: DEFAULT_FILTER += (0x01, 0x02, 0x03, 0x04, 0x05, 0x6, 0x18)
# waves
if FILTER_LEVEL >= 4: DEFAULT_FILTER += (0x8, 0x9,)

def parse_sqtt_print_packets(data: bytes, filter=DEFAULT_FILTER, verbose=True) -> None:
  """
  Minimal debug: print ONE LINE per decoded token (packet).

  Now prints only the actual nibbles that belong to each packet, instead of
  the full 64-bit shift register.
  """
  n = len(data)
  time = 0
  last_printed_time = 0
  reg = 0          # shift register
  offset = 0       # bit offset, in steps of 4 (one nibble)
  nib_budget = 0x40
  flags = 0
  token_index = 0
  opcodes_seen = set()

  while (offset >> 3) < n:
    # 1) Fill register with nibbles according to nib_budget
    if nib_budget != 0:
      target = offset + 4 + ((nib_budget - 1) & ~3)
      while offset != target and (offset >> 3) < n:
        byte = data[offset >> 3]
        nib = (byte >> (offset & 4)) & 0xF
        reg = ((reg >> 4) | (nib << 60)) & ((1 << 64) - 1)
        offset += 4

    # 2) Decode token from low 8 bits
    opcode = STATE_TO_OPCODE[reg & 0xFF]
    opcodes_seen.add(opcode)

    # 4) Set next nibble budget based on opcode
    nib_budget = NIBBLE_BUDGET[opcode & 0x1F]

    # 5) Update time and handle special opcodes 0xF/0x16
    if opcode == 0x16:
      two_bits = (reg >> 8) & 0x3
      if two_bits == 1:
        flags |= 0x01

      # Common 36-bit field at bits [12..47]

      if (reg & 0x200) == 0:
        # delta mode: add 36-bit delta to time
        delta = (reg >> 12) & ((1 << 36) - 1)
      else:
        # marker / other modes: no time advance
        if (reg & 0x100) == 0:
          # real marker: bit9=1, bit8=0, non-zero payload
          # "other" 0x16 variants, ignored for timing
          delta = 0
    else:
      # 6) Generic opcode (including 0x0F)
      shift, width = DELTA_MAP_DEFAULT[opcode]
      delta = (reg >> shift) & ((1 << width) - 1)

      # opcode 0x0F has an offset of 4 to the delta
      if opcode == 0x0F:
        delta = delta + 4

    # Append extra decoded fields into the note string
    note = decode_packet_fields(opcode, reg)

    if verbose and (filter is None or opcode not in filter):
      print(
        f"{token_index:4d}  "
        f"off={offset//4:5d}  "
        f"op=0x{opcode:02x} "
        f"{OPCODE_NAMES[opcode]:24s} "
        f" time={time:8d}+{time-last_printed_time:8d}  "
        f"{reg&reg_mask(opcode):16X} "
        f"{note}"
      )
      last_printed_time = time

    time += delta
    token_index += 1

  # Optional summary at the end
  print(f"# done: tokens={token_index:_}, final_time={time}, flags=0x{flags:02x}")
  if verbose:
    print(f"opcodes({len(opcodes_seen):2d}):", ' '.join([colored(f"{op:2X}", "white" if op in GOOD_OPCODE_NAMES else "red") for op in opcodes_seen]))


def parse(fn:str):
  with Timing(f"unpickle {fn}: "): dat = pickle.load(open(fn, "rb"))
  if getenv("ROCM", 0):
    with Timing(f"decode {fn}: "): ctx = decode(dat)
  dat_sqtt = [x for x in dat if isinstance(x, ProfileSQTTEvent)]
  print(f"got {len(dat_sqtt)} SQTT events in {fn}")
  return dat_sqtt

if __name__ == "__main__":
  fn = "extra/sqtt/examples/profile_gemm_run_0.pkl"
  dat_sqtt = parse(sys.argv[1] if len(sys.argv) > 1 else fn)
  for i,dat in enumerate(dat_sqtt):
    with Timing(f"decode pkt {i} with len {len(dat.blob):_}: "):
      parse_sqtt_print_packets(dat.blob, verbose=getenv("V", 1))
