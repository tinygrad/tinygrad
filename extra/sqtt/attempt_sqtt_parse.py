import pickle
from hexdump import hexdump
from extra.sqtt.roc import decode, ProfileSQTTEvent
from tinygrad.helpers import getenv

# Instruction packets (one per ISA op)
# NOTE: these are bad guesses and may be wrong! feel free to update if you know better

OPCODE_NAMES = {
  # Small metadata / structural packets (NOT ISA op kinds)
  0x01: "META_SMALL_ID",          # 12-bit identifier / slot tag
  0x02: "META_FLAG",              # 1-byte flag/mode (CF/AF/8F/DF...)
  0x03: "META_SUBEVENT_CODE",     # 1-byte sub-event/classification code
  0x04: "META_BASE_INDEX_TAG",    # 12-bit base index/tag (..D, 9D, 10D, 58D...)

  # Instruction / timing / timestamp packets
  0x0F: "TIME_SHORT_DELTA_PLUS4", # short ts, raw_delta+4
  0x11: "TIME_WAVE_STATE",        # compact wave timing/stall state record
  0x14: "INST_EXEC_RECORD",       # per-instruction execution record
  0x16: "TIME_LONG_OR_MARKER",    # long delta / marker with 6-byte payload

  # State / control / perf snapshots
  0x09: "CONTROL_CONFIG_32B",     # 32-bit control/config word (bursts of FE88..., C488...)
  0x15: "PERFCOUNTER_SNAPSHOT",   # perf / TT configuration snapshot (8-byte)

  # Extra descriptors / events / metrics
  0x06: "META_DESCRIPTOR_24B",    # 24-bit descriptor (seen in complex kernels like GEMM)
  0x08: "EVENT_SMALL",            # small in-stream event (5-nibble payload)
  0x12: "TIME_SECONDARY_METRIC",  # 3-byte secondary timing/latency/perf metric
  0x18: "EVENT_SMALL_PAYLOAD",    # generic small side-band payload (5 nibbles)
  0x19: "EVENT_SUMMARY_48B",      # rare 6-byte summary/aggregate metric

  # Pseudo / unknown / not yet observed
  0x07: "UNK_DELTA",              # unknown
  0x0A: "UNK_DELTA2",             # unknown
  0x0B: "UNK_DELTA3",             # unknown
  0x0C: "UNK_DELTA4",             # unknown
  0x0D: "UNK_DELTA5",             # unknown
  0x0E: "UNK_DELTA6",             # unknown
  0x10: "UNK_PSEUDO",             # not seen; pseudo/placeholder
  0x17: "UNK_NO_DELTA",           # unknown, likely non-timing event
}

# these tables are from rocprof trace decoder
# rocprof_trace_decoder_parse_data-0x11c6a0
# parse_sqtt_180 = b *rocprof_trace_decoder_parse_data-0x11c6a0+0x110040

# ---------- 1. local_138: 256-byte state->token table ----------

STATE_TO_TOKEN: bytes = bytes([
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


# ---------- 2. DAT_0012e280: nibble budget per opcode&0x1F ----------

NIBBLE_BUDGET = [
  0x08, 0x0C, 0x08, 0x08, 0x0C, 0x18, 0x18, 0x40,
  0x14, 0x20, 0x30, 0x14, 0x34, 0x1C, 0x30, 0x08,
  0x04, 0x18, 0x18, 0x20, 0x40, 0x40, 0x30, 0x40,
  0x14, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
]
assert len(NIBBLE_BUDGET) == 32


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
  0x0E: (7,  2),   # shift=7,  end=9
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

def decode_packet_fields(opcode: int, reg: int, delta: int) -> str:
  """
  Conservative decoding of a few packet types.

  Rules:
    - We first mask the 64-bit shift register down to the actual packet
      width using NIBBLE_BUDGET[opcode & 0x1F], so we never read bits
      that aren't really part of the packet.
    - Only layouts that are clearly visible from the decompiled C are
      decoded, and names are kept generic (cfg_*, idx_*, id_*, etc).
  """
  # --- 0. Restrict to the real packet bits for this opcode -------------
  nb_bits = NIBBLE_BUDGET[opcode & 0x1F]  # this table is in bits
  if nb_bits <= 0 or nb_bits >= 64:
    pkt = reg & ((1 << 64) - 1)
  else:
    pkt = reg & ((1 << nb_bits) - 1)

  fields: list[str] = []

  # --- 1. Timestamp-ish opcodes ----------------------------------------

  if opcode == 0x0F:  # TIME_SHORT_DELTA_PLUS4
    # By the time we get here, `delta` is already raw_delta+4.
    fields.append(f"ts_short_plus4={delta}")
    return ", ".join(fields)

  if opcode == 0x11:  # TIME_WAVE_STATE (medium/large delta)
    shift, width = DELTA_MAP_DEFAULT[opcode]
    raw_delta = (pkt >> shift) & ((1 << width) - 1)
    coarse    = (pkt >> (shift + width)) & 0xFF  # next byte above delta
    fields.append(f"raw_delta={raw_delta}")
    if coarse:
      fields.append(f"raw_coarse=0x{coarse:02x}")
    return ", ".join(fields)

  if opcode == 0x16:  # TIME_LONG_OR_MARKER
    bit8 = bool(pkt & 0x100)
    bit9 = bool(pkt & 0x200)
    if not bit9:
      mode = "delta"
    elif not bit8:
      mode = "marker"
    else:
      mode = "other"
    val36 = (pkt >> 12) & ((1 << 36) - 1)
    fields.append(f"mode={mode}")
    fields.append(f"val36=0x{val36:x}")
    return ", ".join(fields)

  # --- 2. Opcode 0x14: exec/config record ------------------------------

  if opcode == 0x14:
    subop   = (pkt >> 16) & 0xFFFF       # matches (short)(w >> 0x10)
    val32   = (pkt >> 32) & 0xFFFFFFFF   # matches (uint)(w >> 0x20)
    slot    = (pkt >> 7) & 0x7           # used as (idx & 4) + (idx & 3)
    hi_byte = (pkt >> 8) & 0xFF

    fields.append(f"subop=0x{subop:04x}")
    fields.append(f"slot={slot}")
    fields.append(f"val32=0x{val32:08x}")

    if hi_byte & 0x80:
      # "config" flavour, writes into local_168[...] etc.
      fields.append("kind=config")
      if subop == 0x000C:
        fields.append("cfg_target=local_168[slot].lo")
      elif subop == 0x000D:
        fields.append("cfg_target=local_168[slot].hi")
    else:
      # COR marker: subop 0xC342, val32==0x434F5200 ("COR\0")
      if subop == 0xC342:
        fields.append("kind=cor_stream")
        if val32 == 0x434F5200:
          fields.append("cor_magic='COR\\0'")

    return ", ".join(fields)

  # --- 3. Opcode 0x17: mode/layout header ------------------------------

  if opcode == 0x17:
    # From case 0x17:
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

    return ", ".join(fields)

  # --- 4. Opcode 0x09: state-ish / indirection record ------------------

  if opcode == 0x09:
    # From case 9 on puVar58[1] (here pkt):
    #
    #   uVar41 = (w & 0xffffffff) >> 7;  local_520 = uVar41 & 1
    #   local_4a0 = (w >> 8) & 3;
    #   local_4a8 = (w >> 10) & (7 or 0xf)   (depends on local_494)
    #   uVar69    = (w >> 0xd) or (w >> 0xf) (depends on local_494)
    #   local_518 = (w >> 0x19) & 0x7f;
    #
    # We *don’t* know local_494 here, so we just expose the raw slices.
    flag7   = (pkt >> 7) & 0x1          # low bit of uVar41
    cls2    = (pkt >> 8) & 0x3          # local_4a0
    slot4   = (pkt >> 10) & 0xF         # superset of 3-bit local_4a8
    idx_lo  = (pkt >> 13) & 0x1F        # matches uVar69&0x1F when layout<4
    idx_hi  = (pkt >> 15) & 0x1F        # matches uVar69&0x1F when layout>=4
    id7     = (pkt >> 0x19) & 0x7F      # local_518

    fields.append(f"flag7={flag7}")
    fields.append(f"cls2={cls2}")
    fields.append(f"slot4=0x{slot4:x}")
    fields.append(f"idx_lo5=0x{idx_lo:x}")
    fields.append(f"idx_hi5=0x{idx_hi:x}")
    fields.append(f"id7=0x{id7:x}")
    return ", ".join(fields)

  # --- 5. Opcode 0x18: perf/event trigger ------------------------------

  if opcode == 0x18:
    # From case 0x18:
    #  - low 3 bits:    (w & 7)
    #  - mid 3 bits:    (w >> 3) & 7   or (w >> 4) & 7  (layout–dependent)
    #  - hi id:         (w >> 0xc) & 0xff   OR (w >> 0xd) & 0x7f
    #  - flag bits at 6 / 7
    #
    # The *real* semantics depend on global local_494 and accumulated
    # local_500, so we keep this as a raw view that’s still useful for
    # debugging, but not layout-dependent.
    low3     = pkt & 0x7
    grp3_a   = (pkt >> 3) & 0x7
    grp3_b   = (pkt >> 4) & 0x7
    flag_b6  = (pkt >> 6) & 0x1
    flag_b7  = (pkt >> 7) & 0x1
    idx5_a   = (pkt >> 7) & 0x1F
    idx5_b   = (pkt >> 8) & 0x1F
    hi8      = (pkt >> 12) & 0xFF
    hi7      = (pkt >> 13) & 0x7F

    fields.append(f"low3=0x{low3:x}")
    fields.append(f"grp3_a=0x{grp3_a:x}")
    fields.append(f"grp3_b=0x{grp3_b:x}")
    fields.append(f"flag_b6={flag_b6}")
    fields.append(f"flag_b7={flag_b7}")
    fields.append(f"idx5_a=0x{idx5_a:x}")
    fields.append(f"idx5_b=0x{idx5_b:x}")
    fields.append(f"hi8=0x{hi8:02x}")
    fields.append(f"hi7=0x{hi7:02x}")
    return ", ".join(fields)

  # --- 6. Generic tiny event-ish packets -------------------------------

  if opcode in (0x08, 0x12, 0x19):
    # These are all "small event" style tokens. The exact layout depends
    # on global state (local_494 etc), so we just show:
    #   - low 8 bits as a kind/flag byte
    #   - the rest as an opaque payload.
    kind    = pkt & 0xFF
    payload = pkt >> 8
    fields.append(f"kind_byte=0x{kind:02x}")
    if payload:
      fields.append(f"payload=0x{payload:x}")
    return ", ".join(fields)

  # --- 7. Everything else: no extra decode -----------------------------
  return ""

def parse_sqtt_print_packets(data: bytes, max_tokens: int = 100000) -> None:
  """
  Minimal debug: print ONE LINE per decoded token (packet).

  Now prints only the actual nibbles that belong to each packet, instead of
  the full 64-bit shift register.
  """
  n = len(data)
  time = 0
  reg = 0          # shift register
  offset = 0       # bit offset, in steps of 4 (one nibble)
  nib_budget = 0x40
  flags = 0
  token_index = 0

  while (offset >> 3) < n and token_index < max_tokens:
    # Remember where we started refilling for this step (bit offset),
    # but the *logical* start of the current packet is last_real_offset.
    refill_start = offset

    # 1) Fill register with nibbles according to nib_budget
    if nib_budget != 0:
      target = refill_start + 4 + ((nib_budget - 1) & ~3)
      cur = refill_start
      while cur != target and (cur >> 3) < n:
        byte_index = cur >> 3
        byte = data[byte_index]
        shift = 4 if (cur & 4) else 0  # low then high nibble
        nib = (byte >> shift) & 0xF
        reg = ((reg >> 4) | (nib << 60)) & ((1 << 64) - 1)
        cur += 4
      offset = cur

    # 2) Decode token from low 8 bits
    state = reg & 0xFF
    opcode = STATE_TO_TOKEN[state]

    # 3) Handle pseudo-token 0x10: need more bits, don't print. Looks like a NOP.
    if opcode == 0x10:
      # "need more bits" pseudo-token: adjust nibble budget and continue
      nib_budget = 4
      if (offset >> 3) >= n:
        break
      # Do NOT count this as a real packet; do not update last_real_offset.
      continue

    # 4) Set next nibble budget
    nb_index = opcode & 0x1F
    nib_budget = NIBBLE_BUDGET[nb_index]
    time_before = time
    note = ""
    # 5) Special opcode 0x16 (timestamp / marker)
    if opcode == 0x16:
      two_bits = (reg >> 8) & 0x3
      if two_bits == 1:
        flags |= 0x01

      if (reg & 0x200) == 0:
        # delta mode: 36-bit delta at bits [12..47]
        delta = (reg >> 12) & ((1 << 36) - 1)
        time += delta
        note = "0x16-delta"
      else:
        # marker mode if bit9==1 and bit8==0
        if (reg & 0x100) == 0:
          val = (reg >> 12) & ((1 << 36) - 1)
          delta = 0
          note = f"0x16-marker val=0x{val:x}"
        else:
          delta = 0
          note = "0x16-other"
    else:
      # 6) Generic opcode (including 0x0F)
      shift, width = DELTA_MAP_DEFAULT[opcode]
      mask = (1 << width) - 1
      delta = (reg >> shift) & mask

      # TODO: add more opcode parsers here that add notes to other opcodes
      if opcode == 0x0F:
        delta_with_fix = delta + 4
        note = f"0x0f (+4) raw_delta={delta}"
        time += delta_with_fix
        delta = delta_with_fix
      else:
        time += delta

    # ONE-LINE PRINT PER PACKET
    #assert last_real_offset%8 == 0
    #assert (offset)%8 == 0, f"misalign offset {offset}"

    # Append extra decoded fields into the note string
    extra = decode_packet_fields(opcode, reg, delta)
    if extra: note = (note + " ; " + extra) if note else extra

    BORING_OPCODES = {0x11, 0x14}
    if opcode not in BORING_OPCODES or getenv("BORING", 1):
      my_reg = reg
      my_reg &= (1 << nib_budget) - 1
      print(
        f"{token_index:4d}  "
        f"off={offset//4:5d}  "
        f"op=0x{opcode:02x} "
        f"{OPCODE_NAMES[opcode]:24s} "
        f" time={time_before:8d}+{delta:8d}  "
        f"{my_reg:16X} "
        f"{note}"
      )

    token_index += 1

  # Optional summary at the end
  print(f"# done: tokens={token_index}, final_time={time}, flags=0x{flags:02x}")

def parse(fn:str):
  dat = pickle.load(open(fn, "rb"))
  ctx = decode(dat)
  dat_sqtt = [x for x in dat if isinstance(x, ProfileSQTTEvent)]
  print(f"got {len(dat_sqtt)} SQTT events in {fn}")
  return dat_sqtt

if __name__ == "__main__":
  #dat_sqtt = parse("extra/sqtt/examples/profile_empty_run_0.pkl")
  dat_sqtt = parse("extra/sqtt/examples/profile_plus_run_0.pkl")
  #dat_sqtt = parse("extra/sqtt/examples/profile_gemm_run_0.pkl")
  blob_0 = dat_sqtt[0].blob
  parse_sqtt_print_packets(blob_0[8:])
