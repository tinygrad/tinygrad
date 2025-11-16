import pickle
from tinygrad.helpers import getenv
from extra.sqtt.roc import decode, ProfileSQTTEvent

# Instruction packets (one per ISA op)
# NOTE: these are bad guesses and may be wrong! feel free to update if you know better
# some names were taken from SQ_TT_TOKEN_MASK_TOKEN_EXCLUDE_SHIFT

OPCODE_NAMES = {
  # gated by SQ_TT_TOKEN_EXCLUDE_VMEMEXEC_SHIFT
  0x02: "VMEMEXEC",
  # gated by SQ_TT_TOKEN_EXCLUDE_ALUEXEC_SHIFT
  0x03: "ALUEXEC",
  # gated by SQ_TT_TOKEN_EXCLUDE_VALUINST_SHIFT (but others must be enabled for it to show)
  0x01: "VALUINST",
  # gated by SQ_TT_TOKEN_EXCLUDE_WAVERDY_SHIFT
  0x06: "WAVERDY",
  # gated by SQ_TT_TOKEN_EXCLUDE_WAVESTARTEND_SHIFT
  0x08: "WAVEEND",
  0x09: "WAVESTART",
  # gated by SQ_TT_TOKEN_EXCLUDE_IMMEDIATE_SHIFT
  0x04: "IMMEDIATE_4",
  0x05: "IMMEDIATE_5",
  # some gated by SQ_TT_TOKEN_EXCLUDE_REG_SHIFT, some always there
  0x14: "REG",
  # gated by SQ_TT_TOKEN_EXCLUDE_EVENT_SHIFT
  0x12: "EVENT",
  # gated by SQ_TT_TOKEN_EXCLUDE_INST_SHIFT
  0x18: "INST",
  # gated by SQ_TT_TOKEN_EXCLUDE_UTILCTR_SHIFT
  0x19: "UTILCTR",

  # ------------------------------------------------------------------------
  # 0x07–0x0F: pure timestamp-ish deltas
  # ------------------------------------------------------------------------
  0x07: "TS_DELTA_S8_W3",           # shift=8,  width=3  (small delta)
  0x0A: "TS_DELTA_S5_W2_A",         # shift=5,  width=2
  0x0B: "TS_DELTA_S5_W3_A",         # shift=5,  width=3
  0x0C: "TS_DELTA_S5_W3_B",         # shift=5,  width=3 (different consumer)
  0x0D: "TS_DELTA_S5_W3_C",         # shift=5,  width=3
  0x0E: "TS_DELTA_S7_W2",           # shift=7,  width=2
  0x0F: "TS_DELTA_SHORT_PLUS4",     # short delta; ROCm adds +4 before accumulate

  # ------------------------------------------------------------------------
  # 0x10–0x19: timestamps, layout headers, events, perf
  # ------------------------------------------------------------------------
  0x10: "PSEUDO_NEED_MORE_BITS",    # not a real packet; decoder refill hint

  0x11: "TS_WAVE_STATE_SAMPLE",     # wave stall/termination sample (byte at +10)
  0x13: "EVT_SMALL_GENERIC",        # same structural family as 0x08/0x12/0x19

  0x15: "PERFCOUNTER_SNAPSHOT",     # small delta + 50-ish bits of snapshot
  0x16: "TS_DELTA36_OR_MARK",       # 36-bit long delta or 36-bit marker
  0x17: "LAYOUT_MODE_HEADER",       # layout/mode/group + selectors A/B
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
  Decode packet payloads conservatively, using:
    - NIBBLE_BUDGET[opcode & 0x1F] to mask reg down to true width.
    - DELTA_MAP_DEFAULT[opcode] to expose the "primary" field (often delta).
    - Per-opcode layouts derived from rocprof's decompiled consumers.
  """
  # --- 0. Restrict to real packet bits ---------------------------------
  nb_bits = NIBBLE_BUDGET[opcode & 0x1F]
  if nb_bits <= 0 or nb_bits >= 64:
    pkt = reg & ((1 << 64) - 1)
  else:
    pkt = reg & ((1 << nb_bits) - 1)

  fields: list[str] = []

  shift, width = DELTA_MAP_DEFAULT.get(opcode, (0, 0))
  if width:
    field_mask = (1 << width) - 1
    shaped_field = (pkt >> shift) & field_mask
  else:
    field_mask = 0
    shaped_field = 0

  # =====================================================================
  # 1. Timestamp-centric opcodes (actually drive 'time')
  # =====================================================================

  if opcode == 0x0F:  # TS_DELTA_SHORT_PLUS4
    # In the caller, delta already has +4 applied.
    raw_delta = shaped_field
    fields.append(f"raw_delta={raw_delta}")
    fields.append(f"ts_short_plus4={delta}")
    return ", ".join(fields)

  if opcode == 0x11:  # TS_WAVE_STATE_SAMPLE
    # DELTA_MAP_DEFAULT: shift=7, width=9 -> small delta.
    raw_delta = shaped_field
    coarse    = (pkt >> (shift + width)) & 0xFF  # matches byte at +10 in C
    fields.append(f"raw_delta={raw_delta}")
    if coarse:
      fields.append(f"coarse_state=0x{coarse:02x}")
    # From decomp:
    #  - when layout<3 and coarse&1, it sets a "has interesting wave" flag
    #  - when coarse&8, it marks all live waves as "terminated"
    if coarse & 0x01:
      fields.append("flag_wave_interest=1")
    if coarse & 0x08:
      fields.append("flag_terminate_all=1")
    return ", ".join(fields)

  if opcode == 0x16:  # TS_DELTA36_OR_MARK
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
    val36 = (pkt >> 12) & ((1 << 36) - 1)
    fields.append(f"mode={mode}")
    if mode != "delta":
      fields.append(f"val36=0x{val36:x}")
    return ", ".join(fields)

  # For 0x07, 0x0A–0x0E, we know they drive time (via DELTA_MAP_DEFAULT),
  # but we don't see any other fields used in the decomp.
  if opcode in (0x07, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E):
    if width:
      raw_delta = shaped_field
      leftover  = pkt & ~(field_mask << shift)
      fields.append(f"raw_delta={raw_delta}")
      if leftover:
        fields.append(f"payload=0x{leftover:x}")
    return ", ".join(fields)

  # =====================================================================
  # 2. Small "meta + tiny delta" packets (0x01–0x06)
  # =====================================================================

  if opcode == 0x01:  # META_ID12_TS_SMALL
    id12 = pkt & 0xFFF
    fields.append(f"id12=0x{id12:03x}")
    if width:
      fields.append(f"field_s{shift}_w{width}={shaped_field}")
    return ", ".join(fields)

  if opcode == 0x02:  # META_FLAG8_TS_SMALL
    flag8 = pkt & 0xFF
    fields.append(f"flag8=0x{flag8:02x}")
    if width:
      fields.append(f"field_s{shift}_w{width}={shaped_field}")
    return ", ".join(fields)

  if opcode == 0x03:  # META_SUBEVENT8_TS_SMALL
    sub8 = pkt & 0xFF
    fields.append(f"subevent8=0x{sub8:02x}")
    if width:
      fields.append(f"field_s{shift}_w{width}={shaped_field}")
    return ", ".join(fields)

  if opcode == 0x04:  # META_BASE_INDEX12_TS
    idx12 = pkt & 0xFFF
    fields.append(f"base_index12=0x{idx12:03x}")
    if width:
      fields.append(f"field_s{shift}_w{width}={shaped_field}")
    return ", ".join(fields)

  if opcode in (0x05, 0x06):  # META_DESC24_TS_A/B
    desc24 = pkt & 0xFFFFFF
    fields.append(f"desc24=0x{desc24:06x}")
    if width:
      fields.append(f"field_s{shift}_w{width}={shaped_field}")
    return ", ".join(fields)

  # =====================================================================
  # 3. Opcode 0x14: exec/config record (+ COR marker)
  # =====================================================================

  if opcode == 0x14:  # INST_EXEC_OR_CFG
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
        fields.append("cfg_target=local_168[slot].lo")
      elif subop == 0x000D:
        fields.append("cfg_target=local_168[slot].hi")
    else:
      # COR marker: subop 0xC342, payload "COR\0" → start of a COR region.
      if subop == 0xC342:
        fields.append("kind=cor_stream")
        if val32 == 0x434F5200:
          fields.append("cor_magic='COR\\0'")
    return ", ".join(fields)

  # =====================================================================
  # 4. Opcode 0x17: layout / mode header
  # =====================================================================

  if opcode == 0x17:  # LAYOUT_MODE_HEADER
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
    return ", ".join(fields)

  # =====================================================================
  # 5. Opcode 0x09: state / route config record
  # =====================================================================

  if opcode == 0x09:  # PERF_ROUTE_CONFIG
    # From case 9 in multiple consumers:
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
    id7     = (pkt >> 0x19) & 0x7F

    fields.append(f"flag7={flag7}")
    fields.append(f"cls2={cls2}")
    fields.append(f"slot4=0x{slot4:x}")
    fields.append(f"idx_lo5=0x{idx_lo:x}")
    fields.append(f"idx_hi5=0x{idx_hi:x}")
    fields.append(f"id7=0x{id7:x}")
    return ", ".join(fields)

  # =====================================================================
  # 6. Opcode 0x18: perf/event selector (FUN_0010aba0)
  # =====================================================================

  if opcode == 0x18:  # PERF_EVENT_SELECT
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

  # =====================================================================
  # 7. Opcode 0x15: perfcounter snapshot
  # =====================================================================

  if opcode == 0x15:  # PERFCOUNTER_SNAPSHOT
    # NIBBLE_BUDGET gives full 64 bits here.
    # DELTA_MAP_DEFAULT: shift=7, width=3 → tiny delta field.
    raw_delta = shaped_field if width else 0
    # low bits below the delta field
    snap_low  = pkt & ((1 << shift) - 1) if shift else 0
    # everything above delta field
    snap_hi   = pkt >> (shift + width) if width else (pkt >> shift)

    fields.append(f"raw_delta={raw_delta}")
    fields.append(f"snap_low_s{shift}=0x{snap_low:x}")
    fields.append(f"snap_hi=0x{snap_hi:x}")
    return ", ".join(fields)

  # =====================================================================
  # 8. Small event-ish packets (0x08 / 0x12 / 0x13 / 0x19)
  # =====================================================================

  if opcode in (0x08, 0x12, 0x13, 0x19):
    # These are all "small event / metric" style tokens. The exact semantics
    # depend on layout (0x17) and accumulated state (local_500 etc), so we
    # expose:
    #   - low 8 bits as kind byte
    #   - rest as opaque payload.
    kind    = pkt & 0xFF
    payload = pkt >> 8
    fields.append(f"kind_byte=0x{kind:02x}")
    if payload:
      fields.append(f"payload=0x{payload:x}")
    return ", ".join(fields)

  # =====================================================================
  # 9. Pseudo opcode 0x10: never a "real" packet
  # =====================================================================

  if opcode == 0x10:  # PSEUDO_NEED_MORE_BITS
    # The main loop never prints these; they're just a control token.
    return ""

  # =====================================================================
  # 10. Generic fallback: expose the DELTA_MAP_DEFAULT field + leftover
  # =====================================================================

  if width:
    fields.append(f"field_s{shift}_w{width}={shaped_field}")
    leftover = pkt & ~(field_mask << shift)
    if leftover:
      fields.append(f"payload=0x{leftover:x}")

  return ", ".join(fields)

# 0xb is time something
# 0xd is time something
# 0xf is small time advance
# 0x11 is time advance
# 0x16 is big time advance + markers
# 0x14 is REG
DEFAULT_FILTER = (0xb, 0xd, 0xf, 0x11, 0x16, 0x14) if getenv("FILTER", 1) else None

def parse_sqtt_print_packets(data: bytes, max_tokens: int = 100000, filter=DEFAULT_FILTER) -> None:
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

      # Common 36-bit field at bits [12..47]

      if (reg & 0x200) == 0:
        # delta mode: add 36-bit delta to time
        delta = (reg >> 12) & ((1 << 36) - 1)
        time += delta
      else:
        # marker / other modes: no time advance
        if (reg & 0x100) == 0:
          # real marker: bit9=1, bit8=0, non-zero payload
          # "other" 0x16 variants, ignored for timing
          delta = 0
    else:
      # 6) Generic opcode (including 0x0F)
      shift, width = DELTA_MAP_DEFAULT[opcode]
      mask = (1 << width) - 1
      delta = (reg >> shift) & mask

      # TODO: add more opcode parsers here that add notes to other opcodes
      if opcode == 0x0F:
        delta_with_fix = delta + 4
        time += delta_with_fix
        delta = delta_with_fix
      else:
        time += delta

    # Append extra decoded fields into the note string
    note = decode_packet_fields(opcode, reg, delta)

    if filter is None or opcode not in filter:
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
  #dat_sqtt = parse("extra/sqtt/examples/profile_plus_run_0.pkl")
  dat_sqtt = parse("extra/sqtt/examples/profile_gemm_run_0.pkl")
  blob_0 = dat_sqtt[0].blob
  parse_sqtt_print_packets(blob_0[8:])
