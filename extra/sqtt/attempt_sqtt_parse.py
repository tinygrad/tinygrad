import pickle
from hexdump import hexdump
from extra.sqtt.roc import decode, ProfileSQTTEvent

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

def packet_nibbles_hex(data: bytes, start_bit: int, end_bit: int) -> str:
    """
    Return the hex nibble string from data[start_bit..end_bit) in 4-bit steps.
    start_bit and end_bit are bit offsets, always multiples of 4 here.
    """
    cur = start_bit
    n = len(data)
    out = []

    while cur < end_bit and (cur >> 3) < n:
        byte_index = cur >> 3
        byte = data[byte_index]
        shift = 4 if (cur & 4) else 0  # same convention as the main loop
        nib = (byte >> shift) & 0xF
        out.append(f"{nib:x}")
        cur += 4

    return "".join(out)


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

        # For printing: the logical nibble range for THIS real packet is
        # [last_real_offset, offset).
        packet_start_bit = last_real_offset
        packet_end_bit = offset
        packet_nibbles = packet_nibbles_hex(data, packet_start_bit, packet_end_bit)

        off_bytes = packet_start_bit >> 3  # starting byte of this packet in the stream
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
            if opcode == 0x0F:
                delta_with_fix = delta + 4
                note = f"0x0f (+4) raw_delta={delta}"
                time += delta_with_fix
                delta = delta_with_fix
            else:
                time += delta

        # ONE-LINE PRINT PER PACKET
        print(
            f"{token_index:4d}  "
            f"offB={off_bytes:4d}  "
            f"op=0x{opcode:02x}  "
            f"time={time_before:8d}->{time:8d}  "
            f"delta={delta:8d}  "
            f"{note:30s}  "
            f"nibbles={packet_nibbles}"
        )

        token_index += 1
        # This real packet ends here; next one starts at current offset.
        last_real_offset = offset

    # Optional summary at the end
    print(f"# done: tokens={token_index}, final_time={time}, flags=0x{flags:02x}")

if __name__ == "__main__":
    dat_sqtt = parse("extra/sqtt/examples/profile_empty_run_0.pkl")
    #dat_sqtt = parse("extra/sqtt/examples/profile_plus_run_0.pkl")
    #dat_sqtt = parse("extra/sqtt/examples/profile_gemm_run_0.pkl")
    blob_0 = dat_sqtt[0].blob
    hexdump(blob_0[8:0x108])
    parse_sqtt_print_packets(blob_0[8:])
