import math, operator, struct
from collections.abc import Generator, Mapping
from test.mockgpu.qcom.errors import input_boundary
from test.mockgpu.qcom.state import buffer_at

DST = 255 << 32
REPEAT = 3 << 40
SRC1_R, SRC2_R, SY = 1 << 43, 1 << 51, 1 << 60
SS = 1 << 44
CAT2_FIELDS = 0xffffffff | DST | REPEAT | SRC1_R | SRC2_R | SY
COMPARE_FIELDS = CAT2_FIELDS | (1 << 46) | (7 << 48) | (1 << 50)
SFU_FIELDS = DST | REPEAT | SRC1_R | SS | SY | 0xffff
SFU = {0x8010000000000000: 'rcp', 0x8030000000000000: 'rsq', 0x8050000000000000: 'log2', 0x8070000000000000: 'exp2',
       0x8090000000000000: 'sin', 0x80b0000000000000: 'cos', 0x80d0000000000000: 'sqrt'}
MAD_FIELDS = DST | REPEAT | SRC1_R | SY | SS | (255 << 47) | 0xffffdfff
SELECT_FIELDS = MAD_FIELDS & ~((1 << 14) | (1 << 30) | (1 << 31))
SHIFT_FIELDS = DST | 255 | (31 << 16) | SRC1_R | SRC2_R | SY
SHIFTS = (0x46d0000020000000, 0x4710000020000000, 0x46f0400020000000)
SHIFT_BASES = {54: 0x46d0000000000000, 55: 0x46f0000000000000, 56: 0x4710000000000000}
SHIFT_OPS = {54: operator.lshift, 55: operator.rshift, 56: lambda value, count: signed(value) >> count}
CAT3_OPS = {0xa: operator.rshift, 0xb: operator.lshift, 0xc: operator.and_}
ALU = {0x4210000000000000: 'add.u', 0x4250000000000000: 'sub.u', 0x4650000000000000: 'mull.u', 0x4010000000000000: 'add.f',
       0x4030000000000000: 'min.f',
       0x4070000000000000: 'mul.f', 0x4050000000000000: 'max.f', 0x40d0000000000000: 'absneg.f',
       0x4090000000000000: 'sign.f', 0x4130000000000000: 'floor.f', 0x41b0000000000000: 'trunc.f',
       0x4310000000000000: 'max.u', 0x4330000000000000: 'max.s',
       0x4350000000000000: 'absneg.s',
       0x4690000000000000: 'clz.s', 0x46b0000000000000: 'clz.b',
       0x43b0000000000000: 'or.b', 0x43f0000000000000: 'xor.b'}
COMPARE = {0x4290000000000000: 'cmps.u', 0x42b0000000000000: 'cmps.s', 0x40b0000000000000: 'cmps.f'}
FLUT = {0x2800: 0, 0x2801: 0x3f000000, 0x2802: 0x3f800000, 0x2803: 0x40000000, 0x2808: 0x3fb8aa3b, 0x280b: 0x40800000}

def normal(bits: int) -> bool: return bits & 0x7fffffff == 0 or 0 < (bits >> 23) & 255 < 255
def signed(value: int) -> int: return value - (1 << 32) if value & 0x80000000 else value
def float_value(bits: int, ctx: str, *, special: bool = False) -> float:
  if not normal(bits) and not (special and bits & 0x7f800000 == 0x7f800000):
    raise ValueError(f'{ctx}: non-normal float input unsupported')
  return struct.unpack('<f', bits.to_bytes(4, 'little'))[0]

def float_result(value: float, ctx: str) -> int:
  if math.isnan(value): return 0x7fc00000
  try: bits = int.from_bytes(struct.pack('<f', value), 'little')
  except OverflowError: bits = 0xff800000 if value < 0 else 0x7f800000
  if bits & 0x7fffffff and not bits & 0x7f800000: raise ValueError(f'{ctx}: subnormal float result unsupported')
  return bits

def half_value(bits: int, ctx: str) -> float:
  if bits & 0x7fff and (bits >> 10) & 31 == 0: raise ValueError(f'{ctx}: subnormal half input unsupported')
  return struct.unpack('<e', (bits & 0xffff).to_bytes(2, 'little'))[0]

def half_result(value: float, ctx: str) -> int:
  if math.isnan(value): return 0x7e00
  try: bits = int.from_bytes(struct.pack('<e', value), 'little')
  except (OverflowError, struct.error): bits = 0xfc00 if value < 0 else 0x7c00
  if bits & 0x7fff and (bits >> 10) & 31 == 0: raise ValueError(f'{ctx}: subnormal half result unsupported')
  return bits

def sfu_value(op: str, bits: int, ctx: str) -> int:
  value = float_value(bits, ctx, special=True)
  if math.isnan(value): return 0x7fc00000
  if op in ('rsq', 'sqrt') and value < 0: return 0x7fc00000
  if op in ('rcp', 'rsq') and value == 0: return (bits & 0x80000000) | 0x7f800000
  if op == 'rcp': result = 1.0/value
  elif op == 'rsq': result = 1.0/math.sqrt(value)
  elif op == 'sqrt': result = math.sqrt(value)
  elif op == 'log2': result = math.log2(value) if value > 0 else float('-inf') if value == 0 else float('nan')
  elif op == 'sin': result = math.sin(value)
  elif op == 'cos': result = math.cos(value)
  else:
    try: result = math.exp2(value)
    except OverflowError: result = math.inf
  return float_result(result, ctx)

def _raw_convert(value: int, src: int, dst: int) -> int | None:
  if src == dst or (src in (3, 5) and dst in (3, 5)): return value
  return None

def _numeric_convert(value: int, src: int, dst: int, ctx: str) -> int:
  dst_half = dst in (0, 2, 4, 6)
  if src == 0: number = half_value(value, ctx)
  elif src == 1: number = float_value(value, ctx)
  elif src == 4: number = value - (1 << 16) if value & 0x8000 else value
  elif src == 5: number = signed(value)
  elif src == 6: number = (value & 0xff) - (1 << 8) if dst in (1, 4, 5) and value & 0x80 else value & 0xff
  else: number = value
  if dst == 0: return half_result(number, ctx)
  if dst == 1: return int.from_bytes(struct.pack('<f', number), 'little')
  if dst == 4: low, high = -(1 << 15), 1 << 15
  elif dst == 5: low, high = -(1 << 31), 1 << 31
  elif dst == 6: return int(number) & 0xff
  elif dst == 2: return int(number) & 0xffff
  else: low, high = 0, 1 << 32
  if src == 1 and not low <= int(number) < high:
    raise ValueError(f'{ctx}: float-to-integer range unsupported')
  result = int(number)
  if not low <= result < high: raise ValueError(f'{ctx}: integer conversion range unsupported')
  return result & (0xffff if dst_half else 0xffffffff)

def convert_value(value: int, src: int, dst: int, ctx: str) -> int:
  raw = _raw_convert(value, src, dst)
  return raw if raw is not None else _numeric_convert(value, src, dst, ctx)

def float_alu(op: str, a: int, b: int, ctx: str) -> int:
  left, right = (float_value(x, ctx, special=True) for x in (a, b))
  if op == 'max.f' and (math.isnan(left) or math.isnan(right)): return float_result(right if math.isnan(left) else left, ctx)
  if op == 'max.f': return a & b if left == right == 0 else a if left >= right else b
  if op == 'absneg.f': return a
  if op == 'floor.f': return float_result(math.floor(left), ctx)
  if op == 'trunc.f': return float_result(math.trunc(left), ctx)
  if op == 'sign.f': return float_result(-1.0 if left < 0 else 1.0 if left > 0 else left, ctx)
  if op == 'min.f':
    if math.isnan(left) or math.isnan(right): return float_result(right if math.isnan(left) else left, ctx)
    if left == right == 0: return a & b
    return a if left <= right else b
  return float_result(left + right if op == 'add.f' else left * right, ctx)

# Encoding and lowering: Mesa 461196a, src/freedreno/{isa,ir3}; scope in README.
@input_boundary
def execute(program: bytes, registers: Mapping[int, int], constants: Mapping[int, int] | None = None,
            memory: Mapping[int, bytearray] | None = None, *, padded: bool = False) -> dict[int, int]:
  buffers: dict[int, bytearray] = {}
  for base, data in (memory.items() if memory is not None else ()):
    if type(base) is not int or base < 0 or type(data) is not bytearray or not data or base+len(data) > 1 << 64:
      raise ValueError(f'IR3 pc=0x0: invalid memory region {base!r}')
    if any(base < other+len(blob) and other < base+len(data) for other, blob in buffers.items()):
      raise ValueError(f'IR3 pc=0x0: overlapping memory region {base:#x}')
    if memory is not None and any(data is memory[other] for other in buffers): raise ValueError('IR3 pc=0x0: aliased host buffers')
    buffers[base] = data.copy()
  regs = execute_lane(program, registers, constants or {}, buffers, padded=padded)
  if memory is not None:
    for base, data in buffers.items(): memory[base][:] = data
  return regs

@input_boundary
def execute_lane(program: bytes, registers: Mapping[int, int], consts: Mapping[int, int], buffers: Mapping[int, bytearray],
                 *, padded: bool = False) -> dict[int, int]:
  words = _decode_program(program)
  lane = lane_steps(program, words, registers, consts, buffers, {}, 0, padded=padded)
  while True:
    try: _, event = next(lane)
    except StopIteration as done: return done.value
    if not isinstance(event, bool): raise ValueError('IR3: barrier requires a workgroup')

@input_boundary
def execute_group(program: bytes, registers: list[dict[int, int]], consts: Mapping[int, int], buffers: Mapping[int, bytearray],
                  shared_size: int, *, padded: bool = False, _words: tuple[int, ...] | None = None) -> list[dict[int, int]]:
  if not registers or not 0 <= shared_size <= 32768: raise ValueError('IR3: invalid workgroup or shared size')
  words = _decode_program(program) if _words is None else _words
  shared: dict[int, int] = {}
  lanes = [lane_steps(program, words, initial, consts, buffers, shared, shared_size, padded=padded) for initial in registers]
  while True:
    barriers, results = [], []
    for lane in lanes:
      try: barriers.append(next(lane))
      except StopIteration as done: results.append(done.value)
    if not barriers: return results
    if results or len({pc for pc, _ in barriers}) != 1: raise ValueError('IR3: divergent workgroup barrier')
    if any(isinstance(event, bool) for _, event in barriers):
      if not all(isinstance(event, bool) and event == barriers[0][1] for _, event in barriers):
        raise ValueError('IR3: divergent workgroup branch')
      continue
    writes: dict[int, int] = {}
    for _, pending in barriers:
      assert isinstance(pending, dict)
      if writes.keys() & pending.keys(): raise ValueError('IR3: racing shared writes at barrier')
      writes.update(pending)
    shared.update(writes)

def _decode_program(program: bytes) -> tuple[int, ...]:
  if not program or len(program) % 8: raise ValueError(f'IR3 pc={len(program)//8*8:#x}: empty or truncated program ({len(program)} bytes)')
  if len(program) > 4096*8: raise ValueError('IR3 pc=0x0: program exceeds 4096 instruction limit')
  return struct.unpack(f'<{len(program)//8}Q', program)

def lane_steps(program: bytes, words: tuple[int, ...], registers: Mapping[int, int], consts: Mapping[int, int], buffers: Mapping[int, bytearray],
               shared: dict[int, int], shared_size: int, *, padded: bool = False
               ) -> Generator[tuple[int, dict[int, int] | bool], None, dict[int, int]]:
  # The caller owns the submission snapshot; lanes share its working memory.
  regs = dict(registers)
  pending: dict[int, int] = {}
  pending_half: dict[int, int] = {}
  pending_ss: dict[int, int] = {}
  shared_writes: dict[int, int] = {}
  predicate: bool | None = None
  active: bool | None = None
  halves: dict[int, int] = {}
  locked: set[int] = set()
  locked_ss: set[int] = set()
  for bank, limit, name in ((regs, 244, 'register'), (consts, 2048, 'constant')):
    for reg, value in bank.items():
      if type(reg) is not int or not 0 <= reg < limit: raise ValueError(f'IR3 pc=0x0: invalid initial {name} {reg!r}')
      if type(value) is not int or not 0 <= value <= 0xffffffff: raise ValueError(f'IR3 pc=0x0: invalid {name} {reg} value {value!r}')
  def read(reg: int, ctx: str, half: bool = False) -> int:
    if reg >= 244: raise ValueError(f'{ctx}: special/reserved register unsupported')
    if half:
      if reg in pending_half: raise ValueError(f'{ctx}: half register {reg} needs sy')
      if reg not in halves: raise ValueError(f'{ctx}: uninitialized half register {reg}')
      return halves[reg]
    if reg in pending or reg in pending_ss: raise ValueError(f'{ctx}: register {reg} needs sy/ss')
    if reg not in regs: raise ValueError(f'{ctx}: uninitialized register {reg}')
    return regs[reg]

  def writable(reg: int, ctx: str, half: bool = False):
    if reg >= 244: raise ValueError(f'{ctx}: special/reserved register unsupported')
    if half:
      if reg in pending_half: raise ValueError(f'{ctx}: half register {reg} needs sy before overwrite')
      return
    if reg in pending or reg in pending_ss or reg in locked or reg in locked_ss:
      raise ValueError(f'{ctx}: register {reg} needs sy/ss before overwrite')

  def sync_global():
    regs.update(pending)
    halves.update(pending_half)
    pending.clear()
    pending_half.clear()
    locked.clear()

  def publish_ss():
    regs.update(pending_ss)
    pending_ss.clear()
    locked.clear()
    locked_ss.clear()

  def operand(code: int, ctx: str, offset=0) -> int:
    if code < 256: return read(code+offset, ctx)
    if code & ~2047 == 0x1000:
      index = (code & 2047)+offset
      if index >= 2048 or index not in consts: raise ValueError(f'{ctx}: uninitialized constant {index}')
      return consts[index]
    if code & ~2047 == 0x2000: return ((code & 1023) - (code & 1024)) & 0xffffffff
    raise ValueError(f'{ctx}: unsupported source encoding {code:#x}')

  def float_operand(code: int, ctx: str, offset=0) -> int:
    source = code & 0x3fff
    if source in FLUT: value = FLUT[source]
    else: value = operand(source, ctx, offset)
    if code & 0x8000: value &= 0x7fffffff
    return value ^ (0x80000000 if code & 0x4000 else 0)

  def execute_conversion(word: int, ctx: str) -> bool:
    if word & ~(DST | 255 | REPEAT | SRC1_R | SY | SS | (7 << 46) | (7 << 50)) != 0x2000000000000000:
      return False
    dst, src = (word >> 32) & 255, word & 255
    src_type, dst_type = (word >> 50) & 7, (word >> 46) & 7
    if src_type not in range(7) or dst_type not in range(7) or (dst_type == 4 and src_type == 2):
      raise ValueError(f'{ctx}: unsupported conversion types')
    if word & SY: sync_global()
    if word & SS: publish_ss()
    for index in range(((word >> 40) & 3)+1):
      writable(dst+index, ctx, half=dst_type in (0, 2, 4, 6))
      reg = src+index if word & SRC1_R else src
      value = read(reg, ctx, half=src_type in (0, 2, 4, 6))
      converted = convert_value(value, src_type, dst_type, ctx)
      (halves if dst_type in (0, 2, 4, 6) else regs)[dst+index] = converted
    return True

  def execute_memory(word: int, ctx: str) -> bool:
    local_load = word & ~(DST | (255 << 14) | (255 << 24) | SY) == 0xc046000000800001
    local_store = word & ~((255 << 41) | (255 << 1) | (255 << 24)) == 0xc106010000800000
    load = word & ~(DST | (255 << 14) | (7 << 24) | SY) in (0xc004000000800001, 0xc006000000800001, 0xc00c000000800001)
    global_store = word & ~((255 << 41) | (255 << 1) | (7 << 24) | SY) in (0xc0c6010000800000, 0xc0cc010000800000)
    typed_store = ((word >> 61) & 7) == 6 and ((word >> 54) & 31) == 3 and ((word >> 49) & 7) in (0, 2, 6)
    if not (local_load or local_store or load or global_store or typed_store): return False
    if word & SY: sync_global()
    if local_load or local_store:
      addr_reg, reg = ((word >> 14) & 255, (word >> 32) & 255) if local_load else ((word >> 41) & 255, (word >> 1) & 255)
      addr, count = read(addr_reg, ctx), (word >> 24) & 255
      if count not in range(1, 5) or addr & 3 or addr+count*4 > shared_size: raise ValueError(f'{ctx}: invalid shared range {addr:#x}+{count*4}')
      for index in range(count):
        offset = addr+index*4
        if local_load:
          writable(reg+index, ctx)
          if offset not in shared_writes and offset not in shared: raise ValueError(f'{ctx}: uninitialized shared address {offset:#x}')
          pending_ss[reg+index] = shared_writes[offset] if offset in shared_writes else shared[offset]
        else: shared_writes[offset] = read(reg+index, ctx)
      if local_load: locked_ss.add(addr_reg)
      return True
    addr_reg, reg = ((word >> 14) & 255, (word >> 32) & 255) if load else ((word >> 41) & 255, (word >> 1) & 255)
    count = (word >> 24) & 7
    if load and count == 0: count = 1
    if count not in range(1, 5): raise ValueError(f'{ctx}: unsupported memory count {count}')
    mem_type = (word >> 49) & 7
    half, width = mem_type in (2, 6), 2 if mem_type == 2 else 1 if mem_type == 6 else 4
    size = count*width
    ctx += f' {"ldg" if load else "stg"}.u{width*8} addr_reg={addr_reg} reg={reg} count={count} offset=0'
    addr = read(addr_reg, ctx) | (read(addr_reg+1, ctx) << 32)
    if addr % width or addr+size > 1 << 64: raise ValueError(f'{ctx}: invalid address {addr:#x}+{size}')
    try: base = buffer_at(buffers, addr, size, ctx)
    except ValueError as error: raise ValueError(f'{ctx}: unmapped range {addr:#x}+{size}') from error
    data = buffers[base]
    if load:
      for index in range(count): writable(reg+index, ctx, half=half)
      for index in range(count):
        (pending_half if half else pending)[reg+index] = int.from_bytes(data[addr-base+index*width:addr-base+(index+1)*width], 'little')
      locked.update((addr_reg, addr_reg+1))
    else:
      values = (read(reg+index, ctx, half=half) & ((1 << (width*8))-1) for index in range(count))
      data[addr-base:addr-base+size] = b''.join(value.to_bytes(width, 'little') for value in values)
    return True

  def execute_shift(word: int, ctx: str) -> bool:
    src1, src2 = word & 0xffff, (word >> 16) & 0xffff
    multisrc = ((src1 >= 0x2000 and src2 < 256) or
                (0 < src1 < 256 and 0x2000 <= src2 < 0x2800) or
                (src1 < 256 and src2 < 256 and (src1 or src2)))
    shift = word & ~SHIFT_FIELDS
    single = shift in SHIFTS and (shift & (1 << 46) or not (multisrc and word & SY))
    special = shift == 0x46f0000020000000 and (word & (SRC1_R | SRC2_R) or
                                                ((word & 255) == (word >> 32) & 255 and word & 255))
    if single or special:
      if word & SY and not multisrc: raise ValueError(f'{ctx}: unsupported shift sync')
      if word & SY: sync_global()
      dst, src, count = (word >> 32) & 255, word & 255, (word >> 16) & 31
      half = shift in (0x46f0400020000000, 0x46f0000020000000)
      writable(dst, ctx, half=half)
      result = SHIFT_OPS[(word >> 53) & 63](read(src, ctx), count)
      (halves if half else regs)[dst] = result & (0xffff if half else 0xffffffff)
      return True
    opcode, base = (word >> 53) & 63, word & ~(DST | REPEAT | 0xffffffff | SY)
    if opcode not in SHIFT_BASES or not multisrc or word & (1 << 46) or base != SHIFT_BASES[opcode]: return False
    if word & SY: sync_global()
    dst, src_a, src_b = (word >> 32) & 255, src1, src2
    half = bool((word >> 52) & 1) == bool(word & (1 << 46))
    writable(dst, ctx, half=half)
    count, value = operand(src_a, ctx) & 31, operand(src_b, ctx)
    result = SHIFT_OPS[opcode](value, count)
    (halves if half else regs)[dst] = result & (0xffff if half else 0xffffffff)
    return True

  def execute_cat3(word: int, ctx: str) -> bool:
    src1, src3, opcode = word & 0x1fff, (word >> 16) & 0x1fff, (word >> 55) & 15
    forbidden = ((3 << 40) | (1 << 45) | (1 << 46) | (1 << 59) |
                 (1 << 14) | (1 << 29) | (1 << 30) | (1 << 31))
    if opcode != 0xc: forbidden |= 1 << 44
    if not ((word >> 61) == 3 and opcode in CAT3_OPS and word & (1 << 13) and
            (word & (1 << 42) or opcode == 0xc) and not word & forbidden and
            (src1 < 256 or src1 & 0x1000) and src3 < 256): return False
    full = bool(word & (1 << 42))
    def source(code: int) -> int:
      if code & 0x1000:
        if code & 0x800: raise ValueError(f'{ctx}: unsupported shlg immediate encoding {code:#x}')
        return code & 0x7ff
      if code & 0x800: raise ValueError(f'{ctx}: unsupported shlg source encoding {code:#x}')
      return read(code & 255, ctx, half=not full)
    if word & SY: sync_global()
    if word & SS: publish_ss()
    dst = (word >> 32) & 255
    dst_half = ((word >> 42) & 1) == ((word >> 46) & 1)
    writable(dst, ctx, half=dst_half)
    left = read((word >> 47) & 255, ctx, half=not full)
    value = source(src1)
    right = source(src3)
    result = CAT3_OPS[opcode](left, value if opcode == 0xc else value & 31) | right
    (halves if dst_half else regs)[dst] = result & (0xffff if dst_half else 0xffffffff)
    return True

  def execute_control(word: int, pc: int, ctx: str) -> tuple[bool, int, bool | None]:
    nonlocal active
    branch = word & ~0xffffffff
    if branch in (0x0080000000000000, 0x0100000000000000):
      if active is not None: raise ValueError(f'{ctx}: branch inside predication unsupported')
      if branch == 0x0080000000000000 and predicate is None: raise ValueError(f'{ctx}: uninitialized predicate')
      target = pc + signed(word & 0xffffffff) * 8
      if not 0 <= target < len(program): raise ValueError(f'{ctx}: invalid branch target {target:#x}')
      taken = branch == 0x0100000000000000 or bool(predicate)
      return True, target if taken else pc + 8, taken
    if word == 0x0682000000000000:
      if active is not None or predicate is None: raise ValueError(f'{ctx}: invalid predt state')
      active = predicate
      return True, pc + 8, None
    if word == 0x0782000000000000:
      if active is None: raise ValueError(f'{ctx}: prede without predt')
      active = None
      return True, pc + 8, None
    return False, pc, None

  def execute_alu(word: int, ctx: str) -> None:
    base = word & ~CAT2_FIELDS
    op = ALU.get(base)
    if op is None: raise ValueError(f'{ctx}: unsupported instruction or bits')
    if word & SY: sync_global()
    dst, src1, src2 = (word >> 32) & 255, word & 0xffff, (word >> 16) & 0xffff
    ctx += f' {op} dst={dst} src1={src1} src2={src2}'
    repeat = (word >> 40) & 3
    unary_ops = ('absneg.f', 'sign.f', 'floor.f', 'trunc.f', 'clz.s', 'clz.b')
    for index in range(repeat+1):
      writable(dst+index, ctx)
      if op == 'absneg.s':
        modifier = (src1 >> 14) & 3
        if modifier == 0: raise ValueError(f'{ctx}: absneg.s requires source modifier')
        value = signed(operand(src1 & 0x3fff, ctx, index if repeat and word & SRC1_R else 0))
        if modifier & 2: value = abs(value)
        if modifier & 1: value = -value
        regs[dst+index] = value & 0xffffffff
        continue
      source = float_operand if op.endswith('.f') else operand
      a = source(src1, ctx, index if repeat and word & SRC1_R else 0)
      if op == 'absneg.f' and src2: raise ValueError(f'{ctx}: unsupported unary source bits')
      b = source(src2, ctx, index if repeat and word & SRC2_R else 0) if op not in unary_ops else 0
      if op == 'add.u': result = (a + b) & 0xffffffff
      elif op == 'sub.u': result = (a - b) & 0xffffffff
      elif op == 'mull.u': result = (a & 0xffff) * (b & 0xffff)
      elif op == 'max.u': result = max(a, b)
      elif op == 'max.s': result = max(signed(a), signed(b)) & 0xffffffff
      elif op == 'xor.b': result = (a ^ b) & 0xffffffff
      elif op == 'or.b': result = (a | b) & 0xffffffff
      elif op in ('clz.s', 'clz.b'): result = 32 - (a & 0xffffffff).bit_length()
      else: result = float_alu(op, a, b, ctx)
      regs[dst+index] = result

  def execute_compare(word: int, ctx: str) -> bool:
    compare = COMPARE.get(word & ~COMPARE_FIELDS)
    if compare is None: return False
    condition = (word >> 48) & 7
    if condition > 5: raise ValueError(f'{ctx}: unsupported comparison condition')
    if word & SY: sync_global()
    dst, repeat = (word >> 32) & 255, (word >> 40) & 3
    for index in range(repeat+1):
      writable(dst+index, ctx, half=True)
      source = float_operand if compare == 'cmps.f' else operand
      a = source(word & 0xffff, ctx, index if repeat and word & SRC1_R else 0)
      b = source((word >> 16) & 0xffff, ctx, index if repeat and word & SRC2_R else 0)
      if compare == 'cmps.f': left, right = float_value(a, ctx, special=True), float_value(b, ctx, special=True)
      elif compare == 'cmps.s': left, right = signed(a), signed(b)
      else: left, right = a, b
      halves[dst+index] = int((left < right, left <= right, left > right, left >= right, left == right, left != right)[condition])
    return True

  def execute_mad(word: int, ctx: str) -> bool:
    if word & ~MAD_FIELDS != 0x6380000000000000: return False
    if word & SY: sync_global()
    repeat, dst = (word >> 40) & 3, (word >> 32) & 255
    for index in range(repeat+1):
      writable(dst+index, ctx)
      a = operand(word & 8191, ctx, index if repeat and word & SRC1_R else 0)
      b = read(((word >> 47) & 255) + (index if repeat and word & (1 << 15) else 0), ctx)
      c = operand((word >> 16) & 8191, ctx, index if repeat and word & (1 << 29) else 0)
      a, b, c = (value ^ (0x80000000 if word & (1 << bit) else 0) for value, bit in ((a, 14), (b, 30), (c, 31)))
      # A6xx mad is unfused (ir3_compiler_nir.c:729-732).
      regs[dst+index] = float_alu('add.f', float_alu('mul.f', a, b, ctx), c, ctx)
    return True

  def execute_sfu(word: int, ctx: str) -> bool:
    op = SFU.get(word & ~SFU_FIELDS)
    if op is None: return False
    if word & SY: sync_global()
    dst, src, repeat = (word >> 32) & 255, word & 0xffff, (word >> 40) & 3
    for index in range(repeat+1):
      writable(dst+index, ctx)
      offset = index if repeat and word & SRC1_R else 0
      pending_ss[dst+index] = sfu_value(op, float_operand(src, ctx, offset), ctx)
      if src & 0x3fff < 256: locked_ss.add((src & 255)+offset)
    return True

  def execute_select(word: int, ctx: str) -> bool:
    if word & ~SELECT_FIELDS not in (0x6480000000000000, 0x6580000000000000): return False
    if word & SY: sync_global()
    def select_condition(reg: int) -> int:
      # The model keeps full-first/half-fallback explicit; it is not a hardware bank alias.
      return read(reg, ctx) if reg in regs or reg in pending or reg in pending_ss else read(reg, ctx, half=True)
    dst, repeat = (word >> 32) & 255, (word >> 40) & 3
    for index in range(repeat+1):
      writable(dst+index, ctx)
      a = operand(word & 8191, ctx, index if repeat and word & SRC1_R else 0)
      condition_reg = ((word >> 47) & 255) + (index if repeat and word & (1 << 15) else 0)
      condition = select_condition(condition_reg)
      b = operand((word >> 16) & 8191, ctx, index if repeat and word & (1 << 29) else 0)
      regs[dst+index] = a if condition else b
    return True

  def publishes_ss(word: int, nop: bool) -> bool:
    if not word & SS: return False
    base = word & ~SS
    return nop or base & ~(DST | 0xffffffff) == 0x204cc00000000000 or \
      base & ~CAT2_FIELDS in ALU or base & ~COMPARE_FIELDS in COMPARE or \
      base & ~MAD_FIELDS == 0x6380000000000000 or base & ~SHIFT_FIELDS in SHIFTS or \
      word & ~SFU_FIELDS in SFU or base & ~(CAT2_FIELDS | (1 << 46) | (1 << 52)) == 0x4380000000000000 or \
      base & ~SELECT_FIELDS in (0x6480000000000000, 0x6580000000000000)

  pc, steps = 0, 0
  while pc < len(program):
    word = words[pc >> 3]
    ctx = f'IR3 pc={pc:#x} word={word:#018x} cat={word >> 61}'
    pc, steps = pc+8, steps+1
    if steps > 65536: raise ValueError(f'{ctx}: execution step limit exceeded')
    handled, target, taken = execute_control(word, pc-8, ctx)
    if handled:
      if taken is not None: yield pc-8, taken
      pc = target
      continue
    nop = word & ~((7 << 40) | SS) == 0 and (word >> 40) & 7 <= 5
    global_store = word & ~((255 << 41) | (255 << 1) | (7 << 24) | SY) in (0xc0c6010000800000, 0xc0cc010000800000)
    if active is not None:
      if not nop and not global_store: raise ValueError(f'{ctx}: unsupported predicated instruction')
      if not active:
        if global_store and (word >> 24) & 7 not in range(1, 5): raise ValueError(f'{ctx}: unsupported memory count')
        continue
    if word == 0x0300000000000000:
      tail = program[pc:]
      if tail and not (padded and len(program) % 128 == 0 and len(tail) < 256 and not any(tail)):
        raise ValueError(f'{ctx}: trailing instructions after end unsupported')
      if pending or pending_half: raise ValueError(f'{ctx}: pending loads need sy before end')
      if pending_ss: raise ValueError(f'{ctx}: pending results need ss before end')
      return regs
    if word == 0xe042000000000000:
      if pending or pending_half or pending_ss: raise ValueError(f'{ctx}: pending loads at barrier')
      yield pc-8, shared_writes.copy()
      shared_writes.clear()
      continue
    if execute_memory(word, ctx): continue
    if publishes_ss(word, nop):
      publish_ss()
      word &= ~SS
    # Logical nop delays do not publish loads; the ss flag above does.
    if nop: continue
    if execute_select(word, ctx):
      continue
    if word & ~(CAT2_FIELDS | (1 << 46) | (1 << 52)) == 0x4380000000000000:
      if word & SY: sync_global()
      dst, repeat = (word >> 32) & 255, (word >> 40) & 3
      dst_half = bool((word >> 52) & 1) == bool(word & (1 << 46))
      source_half = not bool(word & (1 << 52))
      for index in range(repeat+1):
        writable(dst+index, ctx, half=dst_half)
        def and_source(code: int, relative: bool) -> int:
          pattern = (code >> 11) & 7
          if pattern == 4: return code & 0x7ff
          if pattern == 0:
            reg = code & 255
            offset = index if repeat and relative else 0
            return read(reg+offset, ctx, half=source_half)
          raise ValueError(f'{ctx}: unsupported half AND source encoding {code:#x}')
        result = and_source(word & 0xffff, bool(word & SRC1_R)) & and_source((word >> 16) & 0xffff, bool(word & SRC2_R))
        (halves if dst_half else regs)[dst+index] = result
      continue
    if execute_sfu(word, ctx):
      continue
    if word & ~0xffffffff == 0x42b400f800000000:
      predicate = operand(word & 0xffff, ctx) == operand((word >> 16) & 0xffff, ctx)
      continue
    if word & ~(0xffffffff | SRC1_R | SRC2_R) == 0x42b300f800000000:
      predicate = signed(operand(word & 0xffff, ctx)) >= signed(operand((word >> 16) & 0xffff, ctx))
      continue
    if execute_conversion(word, ctx):
      continue
    if word & ~(DST | 0xffffffff) == 0x204cc00000000000:
      dst = (word >> 32) & 255
      writable(dst, ctx)
      regs[dst] = word & 0xffffffff
      continue
    if word & ~(DST | 2047) == 0x202cc00000000000:
      dst, src = (word >> 32) & 255, word & 2047
      ctx += f' mov.u32u32 dst={dst} const={src}'
      writable(dst, ctx)
      if src not in consts: raise ValueError(f'{ctx}: uninitialized constant {src}')
      regs[dst] = consts[src]
      continue
    if execute_shift(word, ctx): continue
    if execute_cat3(word, ctx): continue
    if execute_mad(word, ctx): continue
    if word & ~(DST | (255 << 47) | (8191 << 16) | 8191 | SRC1_R | (1 << 15)) == 0x6180000000000000:
      dst = (word >> 32) & 255
      writable(dst, ctx)
      a, b, c = operand(word & 8191, ctx), read((word >> 47) & 255, ctx), operand((word >> 16) & 8191, ctx)
      regs[dst] = ((a & 0xffff) * (b & 0xffff0000) + c) & 0xffffffff
      continue
    if execute_compare(word, ctx):
      continue
    # Repeat increments the destination and sources with (r); at repeat=0 those bits encode nop delay.
    execute_alu(word, ctx)
  raise ValueError(f'IR3 pc={len(program):#x}: missing end')
