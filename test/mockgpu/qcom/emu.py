"""A630 instruction execution for MockGPU, decoded by the existing Mesa library."""
from __future__ import annotations
import ctypes, dataclasses, functools, math, os, struct, tempfile
from collections.abc import Callable, Generator
from tinygrad.runtime.autogen import libc, mesa

def f32bits(value:float) -> int: return struct.unpack('<I', struct.pack('<f', ctypes.c_float(value).value))[0]
def bitsf32(value:int) -> float: return struct.unpack('<f', struct.pack('<I', value & 0xffffffff))[0]
def f16bits(value:float) -> int:
  try: return struct.unpack('<H', struct.pack('<e', value))[0]
  except OverflowError: return 0xfc00 if value < 0 else 0x7c00
def bitsf16(value:int) -> float: return struct.unpack('<e', struct.pack('<H', value & 0xffff))[0]
def rounded_float(value:int|float, half:bool, rounding:int) -> int:
  bits = (f16bits if half else f32bits)(value)
  rounded = (bitsf16 if half else bitsf32)(bits)
  if rounding == 1 or math.isnan(value) or rounded == value: return bits
  up = rounding == 2 or (rounding == 0 and value < 0)
  if (up and rounded < value) or (not up and rounded > value):
    bits += -1 if up == bool(bits & (0x8000 if half else 0x80000000)) else 1
  return bits
def special_float(name:str, value:float) -> float:
  if name in ('floor.f', 'ceil.f', 'trunc.f', 'rndne.f'):
    if not math.isfinite(value) or value == 0: return value
    rounders:dict[str, Callable[[float], int]] = {'floor.f': math.floor, 'ceil.f': math.ceil, 'trunc.f': math.trunc, 'rndne.f': round}
    rounded = float(rounders[name](value))
    return math.copysign(rounded, value) if rounded == 0 else rounded
  if name == 'rcp': return math.copysign(math.inf, value) if value == 0 else 1/value
  if name in ('sqrt', 'rsq'):
    root = math.nan if value < 0 else math.sqrt(value)
    return root if name == 'sqrt' else math.copysign(math.inf, root) if root == 0 else 1/root
  if name == 'log2': return -math.inf if value == 0 else math.nan if value < 0 else math.log2(value)
  try: return {'exp2': math.exp2, 'sin': math.sin, 'cos': math.cos}[name](value)
  except OverflowError: return math.inf
  except ValueError: return math.nan
def signed(value:int, bits:int=32) -> int:
  value &= (1 << bits)-1
  return value-(1 << bits) if value & (1 << (bits-1)) else value

def convert(value:int, source:int, destination:int, rounding:int) -> int:
  if source == destination: return value
  widths = (16, 32, 16, 32, 16, 32, 8, 8)
  if source == 0: number:int|float = bitsf16(value)
  elif source == 1: number = bitsf32(value)
  # Despite its U8 name, COV sign-extends bytes; Mesa uses AND 0xff for zero-extension (create_cov in ir3_compiler_nir.c).
  elif source in (4, 5, 6): number = signed(value, widths[source])
  else: number = value & ((1 << widths[source])-1)
  if destination in (0, 1): return rounded_float(number, destination == 0, rounding)
  if isinstance(number, float): number = (math.trunc, round, math.ceil, math.floor)[rounding](number)
  return number & ((1 << widths[destination])-1)

class Fields:
  fields:tuple[tuple[str, int | str], ...]
  @functools.cached_property
  def first(self) -> dict[str, int|str]: return dict(reversed(self.fields))
  @functools.cached_property
  def last(self) -> dict[str, int|str]: return dict(self.fields)
  def field(self, name:str, default:int=0) -> int: return int(self.first.get(name, default))
  def last_field(self, name:str, default:int=0) -> int: return int(self.last.get(name, default))
  def has(self, name:str) -> bool: return name in self.first

@dataclasses.dataclass(frozen=True)
class Operand(Fields):
  encoded:int
  fields:tuple[tuple[str, int | str], ...]
  category:int
  @functools.cached_property
  def half(self) -> bool:
    if self.has('HALF'): return bool(self.field('HALF'))
    return self.category in (2,4) and (self.encoded >> 11) & 7 == 5 and bool(self.encoded & (1 << 10))

@dataclasses.dataclass(frozen=True)
class Instruction(Fields):
  word:int
  fields:tuple[tuple[str, int | str], ...]
  @property
  def category(self) -> int: return self.word >> 61
  @functools.cached_property
  def name(self) -> str:
    if self.category == 7 and (opcode:=(self.word >> 55) & 15) in (0, 1): return ('bar', 'fence')[opcode]
    if self.category == 1:
      opcode = (self.word >> 57) & 3
      if opcode == 2: return ('swz', 'gat', 'sct', 'invalid')[(self.word >> 40) & 3]
      if opcode == 3: return 'movmsk'
      if opcode == 0: return 'movs' if (self.word >> 53) & 3 == 0 and self.word & (1 << 31) else 'mov'
    return next((str(value) for key,value in self.fields if key == 'NAME'), '')
  @functools.cached_property
  def operands(self) -> dict[str, Operand]: return {}
  @functools.cached_property
  def first(self) -> dict[str, int|str]:
    # MOVA's display asserts its types and destination, so Mesa does not emit callbacks for those fields.
    values = super().first
    if self.category == 1:
      values.update(SRC_TYPE=(self.word >> 50) & 7, DST_TYPE=(self.word >> 46) & 7, DST=(self.word >> 32) & 255,
                    DST_HALF=int(((self.word >> 46) & 7) in (0,2,4,6,7)), DST_REL=(self.word >> 49) & 1)
    return values
  def operand(self, name:str) -> Operand:
    if name in self.operands: return self.operands[name]
    index = next(i for i,(key,_) in enumerate(self.fields) if key == name)
    boundaries = {'DST', 'SRC'} if self.category == 1 else {'DST', 'SRC1', 'SRC2', 'SRC3', 'SRC4', 'SIZE', 'OFF'}
    end = next((i for i in range(index+1, len(self.fields)) if self.fields[i][0] in boundaries - {name}), len(self.fields))
    fields = self.fields[index+1:end]
    if self.category == 1:
      fields += (('HALF', int(self.field('SRC_TYPE') in (0,2,4,6,7))), ('REL_CONST', (self.word >> 10) & 1))
    elif self.category == 3:
      fields += (('SRC_R', self.field(name+'_R')), ('ABSNEG', self.field(name+'_NEG')), ('HALF', self.field('HALF')))
      # Qualcomm emits this form for low-16-bit multiplication plus a full 32-bit addend, all in full GPRs.
      if self.name == 'mad.u16' and not self.field('DST_HALF'): fields = (('HALF', 0),)+fields
    self.operands[name] = Operand(int(self.fields[index][1]), fields, self.category)
    return self.operands[name]

@functools.cache
def decode(program:bytes) -> tuple[Instruction, ...]:
  if len(program) % 8: raise ValueError('A630 instructions must contain complete 64-bit words')
  rows:list[tuple[int, list[tuple[str, int | str]]]] = []
  errors:list[Exception] = []
  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
  def pre(_data, _pc, instruction):
    rows.append((ctypes.cast(instruction, ctypes.POINTER(ctypes.c_uint64)).contents.value, []))
  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(mesa.struct_isa_decode_value))
  def field(_data, name, value):
    try:
      decoded = value.contents
      rows[-1][1].append((ctypes.string_at(name).decode(), ctypes.string_at(decoded.str).decode() if decoded.str else decoded.num))
    except Exception as error: errors.append(error)
  with tempfile.TemporaryFile('w+') as output:
    fd = os.dup(output.fileno())
    fp = libc.fdopen(fd, b'w')
    if not fp:
      os.close(fd)
      raise OSError('fdopen failed while decoding A630 instructions')
    try:
      options = mesa.struct_isa_decode_options(gpu_id=630, show_errors=True, branch_labels=False, pre_instr_cb=pre, field_cb=field)
      mesa.ir3_isa_disasm(program, len(program), ctypes.cast(fp, ctypes.POINTER(mesa.struct__IO_FILE)), options)
    finally: libc.fclose(fp)
  if errors: raise errors[0]
  return tuple(Instruction(word, tuple(fields)) for word,fields in rows)

class Memory:
  def __init__(self, ranges:tuple[tuple[int, int], ...]): self.ranges = ranges
  def check(self, address:int, size:int):
    if not any(start <= address and address+size <= start+length for start,length in self.ranges):
      raise ValueError(f'A630 access outside mapped memory: {address:#x} + {size}')
  def read(self, address:int, size:int) -> int:
    self.check(address, size)
    return int.from_bytes(ctypes.string_at(address, size), 'little')
  def write(self, address:int, size:int, value:int):
    self.check(address, size)
    ctypes.memmove(address, (value & ((1 << (size*8))-1)).to_bytes(size, 'little'), size)

@dataclasses.dataclass(frozen=True)
class Image:
  pointer:int
  width:int
  height:int
  pitch:int
  half:bool
  swizzle:tuple[int, ...] = (0, 1, 2, 3)
  def load(self, memory:Memory, x:int, y:int) -> list[float]:
    values = [0.0]*4
    if 0 <= x < self.width and 0 <= y < self.height:
      size = 2 if self.half else 4
      offset = self.pointer+y*self.pitch+x*4*size
      values = [(bitsf16 if self.half else bitsf32)(memory.read(offset+i*size, size)) for i in range(4)]
    components = values+[0.0, 1.0]
    return [components[index] for index in self.swizzle]
  def store(self, memory:Memory, x:int, y:int, values:list[float]):
    if not (0 <= x < self.width and 0 <= y < self.height): return
    size = 2 if self.half else 4
    offset = self.pointer+y*self.pitch+x*4*size
    for i,value in enumerate(values): memory.write(offset+i*size, size, (f16bits if self.half else f32bits)(value))

# Mesa's IR3 immediate floating-point table (src/freedreno/isa/ir3-common.xml).
FLOAT_IMMEDIATES = (0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1/math.pi, 1/math.log2(math.e), math.log2(math.e), 1/math.log2(10), math.log2(10), 4.0)

class Thread:
  def __init__(self, constants:tuple[int, ...], constant_demotion:bool=False):
    self.constants, self.regs, self.half_regs = constants, [0]*256, [0]*256
    self.constant_demotion = constant_demotion
  def source(self, operand:Operand, repeat:int=0, floating:bool=False) -> int:
    increment = repeat if operand.field('SRC_R') else 0
    if operand.has('IMMED'):
      immediate = operand.field('IMMED')
      if operand.category in (2, 4) and (operand.encoded >> 11) & 7 == 5:
        if immediate >= len(FLOAT_IMMEDIATES): raise ValueError(f'Unsupported floating immediate {immediate}')
        return (f16bits if operand.half else f32bits)(FLOAT_IMMEDIATES[immediate])
      if operand.category in (2, 4): immediate = signed(immediate, 11)
      return immediate & (0xffff if operand.half else 0xffffffff)
    if operand.has('OFFSET'):
      index = signed(self.half_regs[244], 16)+signed(operand.field('OFFSET'), 10)+increment
      constant = bool(operand.field('REL_CONST')) if operand.category == 1 else bool(operand.encoded & (1 << 10))
    else:
      constant = operand.has('CONST')
      index = (operand.last_field('CONST')*4 + operand.last_field('SWIZ') if constant else operand.field('SRC', operand.encoded & 255))+increment
    if index < 0: raise ValueError('Negative A630 register index')
    if constant:
      if not operand.half: return self.constants[index]
      if not self.constant_demotion: return (self.constants[index//2] >> (16*(index%2))) & 0xffff
      return f16bits(bitsf32(self.constants[index])) if floating else self.constants[index] & 0xffff
    registers = self.half_regs if operand.half else self.regs
    return registers[index]
  def float_bits(self, operand:Operand, repeat:int=0) -> int:
    bits = self.source(operand, repeat, floating=True)
    sign = 0x8000 if operand.half else 0x80000000
    if operand.field('ABSNEG') & 2: bits &= sign-1
    if operand.field('ABSNEG') & 1: bits ^= sign
    return bits
  def float_source(self, operand:Operand, repeat:int=0) -> float:
    return (bitsf16 if operand.half else bitsf32)(self.float_bits(operand, repeat))
  def signed_source(self, operand:Operand, repeat:int=0) -> int:
    value = signed(self.source(operand, repeat), 16 if operand.half else 32)
    if operand.field('ABSNEG') & 2: value = abs(value)
    return -value if operand.field('ABSNEG') & 1 else value
  def bit_source(self, operand:Operand, repeat:int=0) -> int:
    value = self.source(operand, repeat)
    return (value ^ (0xffff if operand.half else 0xffffffff)) if operand.field('ABSNEG') & 1 else value
  def destination(self, instruction:Instruction, value:int, repeat:int=0, half:bool|None=None):
    if half is None: half = bool(instruction.field('DST_HALF'))
    relative = signed(self.half_regs[244], 16) if instruction.field('DST_REL') else 0
    self.write(instruction.field('DST')+relative+repeat, value, half)
  def write(self, index:int, value:int, half:bool=False):
    registers, mask = (self.half_regs, 0xffff) if half else (self.regs, 0xffffffff)
    if not 0 <= index < len(registers): raise ValueError(f'Invalid A630 register index {index}')
    registers[index] = value & mask
  def float_destination(self, instruction:Instruction, value:float, repeat:int=0):
    if instruction.field('SAT'): value = min(1.0, max(0.0, value))
    self.destination(instruction, (f16bits if instruction.field('DST_HALF') else f32bits)(value), repeat)

def run_thread(program:bytes, constants:tuple[int, ...], memory:Memory, initial_registers:dict[int, int]|None=None,
               shared:int=0, private:int=0, textures:tuple[Image, ...]=(), images:tuple[Image, ...]=(), start:int=0,
               constant_demotion:bool=False) -> Generator[int, None, Thread]:
  thread = Thread(constants, constant_demotion)
  for index,value in (initial_registers or {}).items(): thread.regs[index] = value & 0xffffffff
  instructions, pc = decode(program), start
  returns:list[int] = []
  predication:bool|None = None
  predicate = False
  while 0 <= pc < len(instructions):
    current_pc, instruction = pc, instructions[pc]
    pc += 1
    name = instruction.name
    if instruction.field('JP'): predicate = bool(thread.regs[248])
    if name in ('predt', 'predf', 'prede'):
      predication = None if name == 'prede' else name == 'predt'
      predicate = bool(thread.regs[248])
      continue
    if predication is not None and predicate != predication: continue
    if name == 'end': return thread
    if name == 'call':
      returns.append(pc)
      pc = current_pc+signed(instruction.field('IMMED'))
      continue
    if name == 'ret':
      if not returns: raise ValueError('A630 return without a call')
      pc = returns.pop()
      continue
    if name in ('nop', 'fence'): continue
    if name == 'bar':
      yield current_pc
      continue
    if name in ('br', 'brao', 'braa', 'jump'):
      condition = bool(thread.regs[248+instruction.field('COMP1')]) != bool(instruction.field('INV1'))
      if name in ('brao', 'braa'):
        second = bool(thread.regs[248+instruction.field('COMP2')]) != bool(instruction.field('INV2'))
        condition = condition or second if name == 'brao' else condition and second
      if name == 'jump' or condition: pc = current_pc+signed(instruction.field('IMMED'))
      continue
    for repeat in range(instruction.field('REPEAT')+1):
      if name == 'mov':
        value = convert(thread.source(instruction.operand('SRC'), repeat, floating=instruction.field('SRC_TYPE') == 0), instruction.field('SRC_TYPE'),
                        instruction.field('DST_TYPE'), instruction.field('ROUND'))
        thread.destination(instruction, value, repeat)
      elif name in ('swz', 'gat', 'sct'):
        src = ([instruction.field('SRC0')+i for i in range(4)] if name == 'sct'
               else [instruction.field('SRC'+str(i)) for i in range(2 if name == 'swz' else 4)])
        dst = ([instruction.field('DST0')+i for i in range(4)] if name == 'gat'
               else [instruction.field('DST'+str(i)) for i in range(2 if name == 'swz' else 4)])
        registers = thread.half_regs if instruction.field('HALF') else thread.regs
        # Swaps and overlapping gathers/scatters must read all old values before the first write.
        values = [convert(registers[index], instruction.field('SRC_TYPE'), instruction.field('DST_TYPE'), instruction.field('ROUND'))
                  for index in src]
        for index,value in zip(dst, values): thread.write(index, value, bool(instruction.field('DST_HALF')))
      elif name in ('add.f', 'mul.f', 'min.f', 'max.f'):
        left, right = [thread.float_source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2')]
        float_value = {'add.f': lambda: left+right, 'mul.f': lambda: left*right, 'min.f': lambda: min(left, right),
                       'max.f': lambda: max(left, right)}[name]()
        thread.float_destination(instruction, float_value, repeat)
      elif name == 'absneg.f':
        operand = instruction.operand('SRC1')
        thread.float_destination(instruction, thread.float_source(operand, repeat), repeat)
      elif name in ('rcp', 'rsq', 'sqrt', 'exp2', 'log2', 'sin', 'cos', 'hrsq', 'hexp2', 'hlog2'):
        thread.float_destination(instruction, special_float(name.removeprefix('h'), thread.float_source(instruction.operand('SRC'), repeat)), repeat)
      elif name in ('floor.f', 'ceil.f', 'trunc.f', 'rndne.f'):
        thread.float_destination(instruction, special_float(name, thread.float_source(instruction.operand('SRC1'), repeat)), repeat)
      elif name == 'absneg.s': thread.destination(instruction, thread.signed_source(instruction.operand('SRC1'), repeat), repeat)
      elif name == 'not.b': thread.destination(instruction, ~thread.bit_source(instruction.operand('SRC1'), repeat), repeat)
      elif name in ('clz.b', 'clz.s'):
        operand = instruction.operand('SRC1')
        bits = thread.signed_source(operand, repeat) if name == 'clz.s' else thread.bit_source(operand, repeat)
        if bits < 0: bits = ~bits
        # Unlike a CPU CLZ, Adreno returns -1 when no qualifying bit exists (Mesa's find_msb/find_lsb lowering).
        thread.destination(instruction, (16 if operand.half else 32)-bits.bit_length() if bits else -1, repeat)
      elif name == 'sign.f':
        number = thread.float_source(instruction.operand('SRC1'), repeat)
        thread.float_destination(instruction, number if number == 0 else float((number > 0)-(number < 0)), repeat)
      elif name in ('min.s', 'max.s'):
        lhs, rhs = [thread.signed_source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2')]
        thread.destination(instruction, min(lhs, rhs) if name == 'min.s' else max(lhs, rhs), repeat)
      elif name in ('add.u', 'add.s', 'sub.u', 'sub.s', 'min.u', 'max.u', 'and.b', 'or.b', 'xor.b', 'shl.b', 'shr.b', 'ashr.b',
                    'mull.u', 'mul.u24', 'mul.s24', 'getbit.b'):
        reader = thread.signed_source if name.endswith('.s') else thread.bit_source if name.endswith('.b') else thread.source
        left, right = [reader(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2')]
        value = {'add.u': lambda: left+right, 'add.s': lambda: left+right, 'sub.u': lambda: left-right, 'sub.s': lambda: left-right,
                 'min.u': lambda: min(left, right), 'max.u': lambda: max(left, right),
                 'and.b': lambda: left & right, 'or.b': lambda: left | right, 'xor.b': lambda: left ^ right,
                 'shl.b': lambda: left << (right & 31), 'shr.b': lambda: left >> (right & 31),
                 'ashr.b': lambda: signed(left, 16 if instruction.operand('SRC1').half else 32) >> (right & 31),
                 'getbit.b': lambda: (left >> (right & 31)) & 1,
                 'mull.u': lambda: (left & 0xffff)*(right & 0xffff),
                 'mul.u24': lambda: (left & 0xffffff)*(right & 0xffffff), 'mul.s24': lambda: signed(left, 24)*signed(right, 24)}[name]()
        thread.destination(instruction, value, repeat)
      elif name in ('madsh.m16', 'madsh.u16'):
        a, b, addend = [thread.source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2', 'SRC3')]
        thread.destination(instruction, (a & 0xffff)*(b & 0xffff0000)+addend, repeat)
      elif name in ('mad.u16', 'mad.s16', 'mad.u24', 'mad.s24'):
        width = int(name[-2:])
        operands = [instruction.operand(key) for key in ('SRC1', 'SRC2', 'SRC3')]
        numbers = [thread.signed_source(operand, repeat) if '.s' in name else thread.source(operand, repeat) for operand in operands]
        a, b = [signed(value, width) if '.s' in name else value & ((1 << width)-1) for value in numbers[:2]]
        thread.destination(instruction, a*b+numbers[2], repeat)
      elif name in ('sad.s16', 'sad.s32'):
        # Mesa lowers iadd3 to SAD: these instructions add three signed operands.
        thread.destination(instruction, sum(thread.signed_source(instruction.operand(key), repeat) for key in ('SRC1','SRC2','SRC3')), repeat)
      elif name in ('mad.f32', 'mad.f16'):
        fa, fb, fc = [thread.float_source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2', 'SRC3')]
        # A6xx MAD is unfused: Mesa's ir3_compiler_nir.c permits lowering it to MUL followed by ADD.
        product = bitsf16(f16bits(fa*fb)) if name == 'mad.f16' else bitsf32(f32bits(fa*fb))
        thread.float_destination(instruction, product+fc, repeat)
      elif name in ('sel.b32', 'sel.b16'):
        yes, selector, no = [thread.bit_source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2', 'SRC3')]
        thread.destination(instruction, yes if selector else no, repeat)
      elif name in ('sel.f32', 'sel.f16'):
        # The math library uses an ordered floating predicate, including NaN as the false case.
        operand = instruction.operand('SRC1' if thread.float_source(instruction.operand('SRC2'), repeat) >= 0 else 'SRC3')
        thread.float_destination(instruction, thread.float_source(operand, repeat), repeat)
      elif name in ('sel.s32', 'sel.s16'):
        operand = instruction.operand('SRC1' if thread.signed_source(instruction.operand('SRC2'), repeat) >= 0 else 'SRC3')
        thread.destination(instruction, thread.signed_source(operand, repeat), repeat)
      elif name in ('cmps.u', 'cmps.s', 'cmps.f', 'cmpv.u', 'cmpv.s', 'cmpv.f'):
        operands = [instruction.operand(key) for key in ('SRC1', 'SRC2')]
        if name.endswith('.f'): left, right = [thread.float_source(operand, repeat) for operand in operands]
        elif name.endswith('.s'): left, right = [thread.signed_source(operand, repeat) for operand in operands]
        else: left, right = [thread.source(operand, repeat) for operand in operands]
        result = (left < right, left <= right, left > right, left >= right, left == right, left != right)[instruction.field('COND')]
        # Bit 42 inverts comparisons, although this Mesa decoder labels it SAT like the arithmetic instructions.
        if instruction.field('SAT'): result = not result
        thread.destination(instruction, -int(result) if name.startswith('cmpv') else int(result), repeat)
      elif name in ('shrg', 'shlg', 'shrm', 'shlm'):
        shift, value, extra = [thread.bit_source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2', 'SRC3')]
        value = (value >> (shift & 31)) if name in ('shrg', 'shrm') else (value << (shift & 31))
        value = value & extra if name in ('shrm', 'shlm') else value | extra
        thread.destination(instruction, value, repeat)
      elif name == 'andg':
        numbers = [thread.bit_source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2', 'SRC3')]
        thread.destination(instruction, (numbers[1] & numbers[0]) | numbers[2], repeat)
      elif name == 'isam':
        if any(instruction.field(flag) for flag in ('3D','A','O','P','S2EN_BINDLESS')): raise ValueError('Unsupported A630 texture addressing mode')
        coordinate = instruction.field('SRC1')
        texels = textures[instruction.field('TEX')].load(memory, signed(thread.regs[coordinate]), signed(thread.regs[coordinate+1]))
        output = 0
        for component,texel in enumerate(texels):
          if instruction.field('WRMASK') & (1 << component):
            thread.float_destination(instruction, texel, output)
            output += 1
      elif name in ('stib.b', 'ldib.b'):
        if instruction.field('MODE') or instruction.field('D') != 2: raise ValueError('Unsupported A630 image addressing mode')
        resource = images[instruction.field('SSBO')]
        coordinate, registers = instruction.field('SRC2'), thread.half_regs if instruction.field('TYPE_HALF') else thread.regs
        x, y = signed(thread.regs[coordinate]), signed(thread.regs[coordinate+1])
        count, register = instruction.field('TYPE_SIZE'), instruction.field('SRC1')
        if name == 'stib.b':
          texels = [(bitsf16 if instruction.field('TYPE_HALF') else bitsf32)(registers[register+i]) for i in range(count)]
          resource.store(memory, x, y, texels)
        else:
          for i,texel in enumerate(resource.load(memory, x, y)[:count]):
            half = bool(instruction.field('TYPE_HALF'))
            thread.write(register+i, (f16bits if half else f32bits)(texel), half)
      elif name in ('ldg', 'stg', 'ldg.a', 'stg.a', 'ldl', 'stl', 'ldp', 'stp'):
        kind = instruction.field('TYPE')
        width = (2, 4, 2, 4, 2, 4, 1, 1)[kind]
        load = name.startswith('ld')
        if name.startswith(('ldg', 'stg')):
          pointer_reg = instruction.field('SRC1')
          address = thread.regs[pointer_reg] | (thread.regs[pointer_reg+1] << 32)
          source = instruction.field('SRC3')
        else:
          address = (shared if name.endswith('l') else private) + thread.regs[instruction.field('SRC' if load else 'DST')]
          source = instruction.field('SRC')
        offset = instruction.field('OFF') if name.endswith('.a') else signed(instruction.field('OFF'), 13)
        if name.endswith('.a'): offset = ((thread.regs[instruction.field('SRC2')] << instruction.field('SRC2_SHIFT'))+offset)*width
        address += offset
        for component in range(instruction.field('SIZE')):
          if load:
            thread.destination(instruction, memory.read(address+component*width, width), component, half=kind in (0,2,4,6,7))
          else:
            registers = thread.half_regs if kind in (0,2,4,6,7) else thread.regs
            memory.write(address+component*width, width, registers[source+component])
      else: raise ValueError(f'A630 instruction is not implemented: {name} ({instruction.word:#x})')
  raise ValueError('A630 instruction stream ended without an end instruction')

def run_scalar(program:bytes, constants:tuple[int, ...], memory:Memory, initial_registers:dict[int, int]|None=None) -> Thread:
  worker = run_thread(program, constants, memory, initial_registers)
  try: next(worker)
  except StopIteration as completed: return completed.value
  finally: worker.close()
  raise ValueError('A barrier requires workgroup execution')

def run_workgroup(program:bytes, constants:tuple[int, ...], memory:Memory, registers:list[dict[int, int]], shared_size:int, private_size:int,
                  textures:tuple[Image, ...]=(), images:tuple[Image, ...]=(), start:int=0, constant_demotion:bool=False):
  shared = ctypes.create_string_buffer(shared_size)
  private = [ctypes.create_string_buffer(private_size) for _ in registers]
  shared_address = ctypes.addressof(shared)
  workers = [run_thread(program, constants,
                        Memory(memory.ranges+((shared_address, shared_size), (ctypes.addressof(scratch), private_size))),
                        initial, shared_address, ctypes.addressof(scratch), textures, images, start, constant_demotion)
             for initial,scratch in zip(registers, private)]
  try:
    while workers:
      waiting = []
      for worker in workers:
        try: waiting.append((worker, next(worker)))
        except StopIteration: pass
      if len({pc for _,pc in waiting}) > 1: raise RuntimeError('A630 workgroup reached different barrier instructions')
      workers = [worker for worker,_ in waiting]
  finally:
    for worker in workers: worker.close()
