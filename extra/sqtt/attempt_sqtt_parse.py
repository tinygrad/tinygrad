import pickle
from hexdump import hexdump
from extra.sqtt.roc import decode, ProfileSQTTEvent
from tinygrad.helpers import getenv

# Instruction packets (one per ISA op)
# NOTE: these are bad guesses and may be wrong! feel free to update if you know better
OPCODE_NAMES = {
    0x01: "INST_VALU",          # vector ALU
    0x02: "INST_FLAT",          # global/flat mem
    0x03: "INST_SMEM_OR_MSG",   # scalar mem / msg
    0x04: "INST_SCALAR_CTRL",   # scalar ctrl / immed

    # Timing / timestamp packets
    0x0F: "TS_SHORT_PLUS4",     # short ts, delta+4
    0x11: "TS_MEDIUM_DELTA",    # medium/large delta
    0x14: "PC_TICK_TINY_DELTA", # tiny tick per PC
    0x16: "TS_LONG_OR_MARKER",  # long delta / marker

    # State / control / region markers
    0x09: "STATE_SNAPSHOT",     # Δ=0 state dump
    0x15: "CONTROL_REGION_MARK",# region / wave mark

    # Event / structured / PC-lane packets
    0x06: "FIELD_SMALL_ZERO",   # extra per-inst field (spec)
    0x08: "EVENT_TINY_SPECIAL", # tiny special-unit evt (spec)
    0x12: "EVENT_SHORT_STRUCT", # short structured evt (spec)
    0x18: "EVENT_PC_LANE",      # PC / lane micro-event
    0x19: "EVENT_TINY_SECONDARY",# secondary tiny lane (spec)

    # Pseudo / unknown framework bits
    0x10: "PSEUDO_NEED_MORE_BITS", # pseudo, not real pkt
    0x07: "UNK_DELTA",           # unknown delta
    0x0A: "UNK_DELTA2",          # unknown
    0x0B: "UNK_DELTA3",          # unknown
    0x0C: "UNK_DELTA4",          # unknown
    0x0D: "UNK_DELTA5",          # unknown
    0x0E: "UNK_DELTA6",          # unknown
    0x17: "UNK_NO_DELTA",        # unknown, no delta
}

# rocprof_trace_decoder_parse_data-0x11c6a0
# parse_sqtt_180 = b *rocprof_trace_decoder_parse_data-0x11c6a0+0x110040

def parse(fn:str):
    dat = pickle.load(open(fn, "rb"))
    ctx = decode(dat)
    dat_sqtt = [x for x in dat if isinstance(x, ProfileSQTTEvent)]
    print(f"got {len(dat_sqtt)} SQTT events in {fn}")
    return dat_sqtt

# ---------- 1. local_138: 256-byte state->token table ----------

_LOCAL_138_QWORDS = [
    0x000c0b0501181610, 0x020304090118140f,
    0x000d080601181710, 0x0203040a0118140f,
    0x000c0b0501180710, 0x020304090118140f,
    0x000d080601181910, 0x0203040a0118140f,
    0x000c0b0501180010, 0x020304090118140f,
    0x000d080601181110, 0x0203040a0118140f,
    0x000c0b0501181210, 0x020304090118140f,
    0x000d080601181510, 0x0203040a0118140f,
    0x000c0b0501181610, 0x020304090118140f,
    0x000d080601181710, 0x0203040a0118140f,
    0x000c0b0501180710, 0x020304090118140f,
    0x000d080601181910, 0x0203040a0118140f,
    0x000c0b0501180010, 0x020304090118140f,
    0x000d080601181110, 0x0203040a0118140f,
    0x000c0b0501181310, 0x020304090118140f,
    0x000d080601181510, 0x0203040a0118140f,
]

STATE_TO_TOKEN: bytes = b"".join(
    q.to_bytes(8, "little") for q in _LOCAL_138_QWORDS
)
assert len(STATE_TO_TOKEN) == 256

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


# ---------- 3. delta extraction hook (for generic opcodes) ----------

def extract_delta(opcode: int, reg64: int) -> int:
    """
    Extract time delta bits for an opcode from reg64.

    delta_map: opcode -> (shift, width).
    If opcode missing, returns 0.
    """
    info = DELTA_MAP_DEFAULT.get(opcode)
    if info is None:
        return 0
    shift, width = info
    if width <= 0:
        return 0
    mask = (1 << width) - 1
    return (reg64 >> shift) & mask


# ---------- 4. One-line-per-packet parser ----------

def decode_packet_fields(opcode: int, reg: int, delta: int) -> str:
    """
    Conservative decoding of useful low-level fields from the 64-bit register.

    - Always reports the state (low 8 bits).
    - Reports the exact bit slice used for delta.
    - For a few special opcodes, exposes raw fields AMD clearly uses,
      but keeps names generic (f0/f1/… or raw_*).
    """
    fields = []

    """
    # Common: low 8 bits are the FSM state index
    state = reg & 0xFF
    fields.append(f"state=0x{state:02x}")

    # Delta bit slice (from your DELTA_MAP_DEFAULT)
    info = DELTA_MAP_DEFAULT.get(opcode)
    if info is not None:
        shift, width = info
        if width > 0:
            raw_delta_bits = (reg >> shift) & ((1 << width) - 1)
            fields.append(f"delta_bits=[{shift}:{shift+width})=0x{raw_delta_bits:x}")
    """

    # --- Special timestamp-ish opcodes -------------------------------------

    if opcode == 0x0F:  # TS_SHORT_PLUS4
        # delta here is already "delta + 4"
        fields.append(f"ts_short_plus4={delta}")

    elif opcode == 0x11:  # TS_MEDIUM_DELTA
        shift, width = DELTA_MAP_DEFAULT[opcode]
        raw_delta = (reg >> shift) & ((1 << width) - 1)
        coarse = (reg >> (shift + width)) & 0xFF  # just above delta
        fields.append(f"raw_delta={raw_delta}")
        if coarse:
            fields.append(f"raw_coarse=0x{coarse:02x}")

    elif opcode == 0x16:  # TS_LONG_OR_MARKER
        two_bits = (reg >> 8) & 0x3
        bit9 = bool(reg & 0x200)
        bit8 = bool(reg & 0x100)
        val36 = (reg >> 12) & ((1 << 36) - 1)

        if not bit9:
            mode = "delta"
        elif not bit8:
            mode = "marker"
        else:
            mode = "other"

        fields.append(f"mode={mode}")
        fields.append(f"subbits=0b{two_bits:02b}")
        fields.append(f"val36=0x{val36:x}")

    # --- Snapshot / state-like ---------------------------------------------

    elif opcode == 0x09:  # STATE_SNAPSHOT (we don't know exact layout)
        # This is the one you saw at token 183:
        #   reg = 0x00487001C488000C
        # We can at least expose a couple of chunks without naming them too boldly.
        f0 = (reg >> 8) & 0xFFFF
        f1 = (reg >> 24) & 0xFFFF
        fields.append(f"f0=0x{f0:04x}")
        fields.append(f"f1=0x{f1:04x}")

    # --- Tiny event / lane-ish opcodes -------------------------------------

    elif opcode in (0x08, 0x12, 0x18, 0x19):
        # These look like "event with payload" style tokens.
        # We can take a small ID and a 16-bit payload; semantics TBD.
        event_id = (reg >> 8) & 0x3F
        payload16 = (reg >> 16) & 0xFFFF
        fields.append(f"event_id=0x{event_id:x}")
        if payload16:
            fields.append(f"payload16=0x{payload16:04x}")

    # You can add other opcode-specific slices here as you learn more,
    # but it's safer to keep naming generic until confirmed.

    return ", ".join(fields)


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

    # Bit offset at the end of the previous REAL packet (not counting 0x10).
    last_real_offset = 0

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

        # 3) Handle pseudo-token 0x10: need more bits, don't print
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
            delta = extract_delta(opcode, reg)
            # TODO: add more opcode parsers here that add notes to other opcodes
            if opcode == 0x0F:
                delta_with_fix = delta + 4
                note = f"0x0f (+4) raw_delta={delta}"
                time += delta_with_fix
                delta = delta_with_fix
            else:
                time += delta

        # ONE-LINE PRINT PER PACKET
        assert last_real_offset%8 == 0
        assert (offset-last_real_offset)%8 == 0

        # Append extra decoded fields into the note string
        extra = decode_packet_fields(opcode, reg, delta)
        if extra:
            note = (note + " ; " + extra) if note else extra

        BORING_OPCODES = {0xf, 0x11, 0x12, 0x14, 0x15, 0x16}
        if opcode not in BORING_OPCODES or getenv("BORING"):
            print(
                f"{token_index:4d}  "
                f"offB={last_real_offset//8:4d}+{(offset-last_real_offset)//8:<2d} "
                f"op=0x{opcode:02x} {OPCODE_NAMES[opcode]:20s}  "
                f"time={time_before:8d}->{time:8d}  "
                f"{reg:016X}  "
                f"{note}"
            )
            #f"delta={delta:8d}  "

        token_index += 1
        # This real packet ends here; next one starts at current offset.
        last_real_offset = offset

    # Optional summary at the end
    print(f"# done: tokens={token_index}, final_time={time}, flags=0x{flags:02x}")

if __name__ == "__main__":
    #dat_sqtt = parse("extra/sqtt/examples/profile_empty_run_0.pkl")
    dat_sqtt = parse("extra/sqtt/examples/profile_plus_run_0.pkl")
    #dat_sqtt = parse("extra/sqtt/examples/profile_gemm_run_0.pkl")
    blob_0 = dat_sqtt[0].blob
    hexdump(blob_0[8:0x108])
    parse_sqtt_print_packets(blob_0[8:])
