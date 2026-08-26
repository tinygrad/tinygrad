import ctypes, functools, math, struct
from tinygrad.runtime.autogen import mesa

def _field(fields, name):
  try: return next(value for field, value in fields if field == name)
  except StopIteration as exc:
    raise ValueError(f'missing IR3 field {name}') from exc

def _field_or(fields, name, default):
  return next((value for field, value in fields if field == name), default)

def _reject_unsupported_modifiers(fields, name):
  unsupported = {'UL', 'EI'}
  if bad := next((field for field, value in fields if field in unsupported and value), None):
    raise NotImplementedError(f'unsupported IR3 modifier {bad}')

def _reg(value, half=False):
  constant = bool(value & 0x1000)
  value &= 0xfff
  if not constant and value // 4 == 61: kind = 'a'
  elif not constant and value // 4 == 62: kind = 'p'
  else: kind = ('hc' if constant else 'hr') if half else ('c' if constant else 'r')
  return kind, value // 4, value % 4

def _src(value, half=False):
  mod = (value >> 14) & 3
  if value & 0x3800 == 0x2800:
    flut = (0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1 / math.pi, 1 / math.log2(math.e), math.log2(math.e), 1 / math.log2(10), math.log2(10), 4.0)
    index = value & 0x3ff
    if index >= len(flut): raise ValueError(f'invalid IR3 float immediate {index}')
    return _float_bits(flut[index], half), mod
  if value & 0x2000:
    immediate = value & 0x7ff
    return immediate - 0x800 if immediate & 0x400 else immediate, mod
  return _reg(value, half), mod

def _cat1_src(fields, half=False):
  if any(field == 'IMMED' for field, _ in fields): return _field(fields, 'IMMED')
  if any(field == 'CONST' for field, _ in fields): return _reg(0x1000 | _field(fields, 'CONST'), half)
  if any(field == 'OFFSET' for field, _ in fields): return ('rel', _field(fields, 'OFFSET'), int(any(field == 'CONST' for field, _ in fields)))
  return _reg(_field(fields, 'SRC'), half)

def _as_f32(bits): return ctypes.c_float.from_buffer_copy(ctypes.c_uint32(bits)).value

def _f32_bits(value): return ctypes.c_uint32.from_buffer_copy(ctypes.c_float(value)).value

@functools.cache
def local_id_regs(local_size, register, order=(0, 1, 2)):
  x_size, y_size, z_size = local_size
  lanes = [(x, y, z) for z in range(z_size) for y in range(y_size) for x in range(x_size)]
  return {('r', register, component): [lane[axis] for lane in lanes] for component, axis in enumerate(order)}

def workgroup_id_regs(workgroup_id, lane_count, register):
  return {('r', register, component): [workgroup_id[component]] * lane_count for component in range(3)}

class IR3ConstantBank:
  """One launch's immutable constant words, shared as broadcast rows across every wave."""
  def __init__(self, words):
    self.words, self._rows = tuple(words), {}

  def row(self, kind, index, lanes):
    # Rows are immutable by construction: the emulator only mutates r/hr rows in
    # place (relr/relhr destinations) and always replaces c/hc rows wholesale.
    by_lanes = self._rows.setdefault((kind, index), {})
    if (row := by_lanes.get(lanes)) is None:
      word = self.words[index]
      row = by_lanes[lanes] = [word if kind == 'c' else word & 0xffff] * lanes
    return row

class IR3RegisterFile(dict):
  """Wave registers whose lane count stays authoritative even with no materialized rows.

  Constant rows install lazily from the shared bank on first read; a local write
  shadows the bank row.  Untouched bank rows stay absent from iteration, so only
  constants a kernel actually reads are ever allocated."""
  def __init__(self, lanes, initial=None, constants=None):
    super().__init__(initial if initial is not None else {})
    self.lanes, self.constants = lanes, constants

  def __missing__(self, key):
    kind, number, component = key
    if self.constants is not None and kind in ('c', 'hc'):
      index = number * 4 + component
      if 0 <= index < len(self.constants.words):
        self[key] = row = self.constants.row(kind, index, self.lanes)
        return row
    raise KeyError(key)

  def get(self, key, default=None):
    try: return self[key]
    except KeyError: return default

  def materialize_constants(self):
    if self.constants is None: return
    for index in range(len(self.constants.words)):
      for kind in ('c', 'hc'):
        key = (kind, index // 4, index % 4)
        if key not in self: self[key] = self.constants.row(kind, index, self.lanes)

def _lane_count(regs):
  if isinstance(regs, IR3RegisterFile): return regs.lanes
  return len(next(iter(regs.values()))) if regs else 0

def _next_reg(reg):
  kind, number, component = reg
  return kind, number + (component == 3), (component + 1) % 4

def _reg_offset(reg, offset):
  kind, number, component = reg
  return kind, number + (component + offset) // 4, (component + offset) % 4

def _values(regs, src, lanes):
  if isinstance(src, int): return [src] * lanes
  values = regs.get(src)
  return [0] * lanes if values is None else values

def _float_values(regs, src, lanes, half):
  values = _values(regs, src, lanes)
  # A half-precision constant source can name a full-float constant slot.  The
  # A6xx constant path converts those values to f16 for floating-point ALU
  # instructions, while integer/bit operations still consume the raw bits.
  # Mesa emits both forms in the fp8 conversion sequence.
  if half and isinstance(src, tuple) and src[0] == 'hc' and (full := regs.get(('c', src[1], src[2]))) is not None:
    return [_float_bits(_float(value), True) if value >> 16 else value & 0xffff for value in full]
  return values
def _write(regs, dst, values, mask):
  if all(mask):
    # Every caller passes a freshly materialized list, so an all-active write
    # can install it without merging a previous row.
    regs[dst] = values
    return
  previous = regs.get(dst)
  regs[dst] = [value if active else old for value, old, active in
               zip(values, [0] * len(values) if previous is None else previous, mask, strict=True)]
def _s32(value): return ctypes.c_int32(value).value
def _signed(value, half=False): return ctypes.c_int16(value).value if half else _s32(value)
def _compare(lhs, rhs, condition):
  return (lhs < rhs, lhs <= rhs, lhs > rhs, lhs >= rhs, lhs == rhs, lhs != rhs)[condition]

def _float(value, half=False):
  mask, size, fmt = (0xffff, 2, '<e') if half else (0xffffffff, 4, '<f')
  return struct.unpack(fmt, (value & mask).to_bytes(size, 'little'))[0]
def _float_bits(value, half=False):
  try: return int.from_bytes(struct.pack('<e' if half else '<f', value), 'little')
  except OverflowError: return 0x7c00 if half and value > 0 else 0xfc00 if half else 0x7f800000 if value > 0 else 0xff800000

def _mod_float(value, modifier, half):
  value = _float(value, half)
  return -abs(value) if modifier == 3 else abs(value) if modifier == 2 else -value if modifier == 1 else value

def _mod_int(value, modifier, half=False):
  value = _signed(value, half)
  return -abs(value) if modifier == 3 else abs(value) if modifier == 2 else -value if modifier == 1 else value

def _typed_value(value, typ):
  if typ == mesa.TYPE_F16: return _float(value, True)
  if typ == mesa.TYPE_F32: return _float(value)
  if typ == mesa.TYPE_U16: return value & 0xffff
  if typ == mesa.TYPE_S16: return ctypes.c_int16(value).value
  if typ == mesa.TYPE_S32: return _s32(value)
  if typ in (mesa.TYPE_U8, mesa.TYPE_U8_32): return value & 0xff
  return value & 0xffffffff

def _convert(value, src_type, dst_type, rounding):
  if src_type == dst_type:
    return value & (0xffff if dst_type in (mesa.TYPE_F16, mesa.TYPE_U16, mesa.TYPE_S16, mesa.TYPE_U8) else 0xffffffff)
  # IR3 has no distinct S8 source type. A U8-to-signed widening conversion is
  # the encoding Mesa uses to sign-extend an 8-bit integer.
  value = ctypes.c_int8(value).value if src_type == mesa.TYPE_U8 and dst_type in (mesa.TYPE_S16, mesa.TYPE_S32) else \
    _typed_value(value, src_type)
  if dst_type == mesa.TYPE_F16: return _float_bits(float(value), True)
  if dst_type == mesa.TYPE_F32: return _float_bits(float(value))
  if isinstance(value, float):
    if not math.isfinite(value): value = 0
    elif rounding == 1: value = round(value)
    elif rounding == 2: value = math.ceil(value)
    elif rounding == 3: value = math.floor(value)
    else: value = math.trunc(value)
  width = 16 if dst_type in (mesa.TYPE_U16, mesa.TYPE_S16) else 8 if dst_type in (mesa.TYPE_U8, mesa.TYPE_U8_32) else 32
  return int(value) & ((1 << width) - 1)

def _itemsize(typ): return 2 if typ in (mesa.TYPE_F16, mesa.TYPE_U16, mesa.TYPE_S16) else 1 if typ in (mesa.TYPE_U8, mesa.TYPE_U8_32) else 4

@functools.cache
def _alu_runner(name, condition, modifiers, source_half, dest_half):
  float_op = name.endswith('.f') or name in {'rcp', 'rsq', 'log2', 'exp2', 'sin', 'cos', 'sqrt', 'hrsq', 'hlog2', 'hexp2'}
  def run(values):
    v = tuple(_mod_float(x, m, source_half) if float_op else _mod_int(x, m, source_half) if m else x
              for x, m in zip(values, modifiers, strict=True))
    if name in {'add.f', 'mul.f', 'min.f', 'max.f'}:
      if name == 'add.f': out = v[0] + v[1]
      elif name == 'mul.f': out = v[0] * v[1]
      elif math.isnan(v[0]): out = v[1]
      elif math.isnan(v[1]): out = v[0]
      else: out = min(v) if name == 'min.f' else max(v)
      return _float_bits(out, dest_half)
    if name in {'cmps.f', 'cmpv.f'}: return int(_compare(v[0], v[1], condition))
    if name == 'sign.f': return _float_bits(-1.0 if v[0] < 0 else 1.0 if v[0] > 0 else v[0], dest_half)
    if name == 'absneg.f': return _float_bits(v[0], dest_half)
    if name in {'floor.f', 'ceil.f', 'rndne.f', 'rndaz.f', 'trunc.f'}:
      if not math.isfinite(v[0]): return _float_bits(v[0], dest_half)
      if name == 'floor.f': rounded = math.floor(v[0])
      elif name == 'ceil.f': rounded = math.ceil(v[0])
      elif name == 'rndne.f': rounded = round(v[0])
      elif name == 'rndaz.f': rounded = math.copysign(math.ceil(abs(v[0])), v[0])
      else: rounded = math.trunc(v[0])
      return _float_bits(float(rounded), dest_half)
    if name in {'rcp', 'rsq', 'log2', 'exp2', 'sin', 'cos', 'sqrt', 'hrsq', 'hlog2', 'hexp2'}:
      try:
        if name == 'rcp': out = 1.0 / v[0]
        elif name in {'rsq', 'hrsq'}: out = 1.0 / math.sqrt(v[0])
        elif name in {'log2', 'hlog2'}: out = math.log2(v[0])
        elif name in {'exp2', 'hexp2'}: out = 2.0 ** v[0]
        elif name == 'sin': out = math.sin(v[0])
        elif name == 'cos': out = math.cos(v[0])
        else: out = math.sqrt(v[0])
      except (OverflowError, ValueError, ZeroDivisionError):
        if name in {'log2', 'hlog2'}: out = -math.inf if v[0] == 0 else math.nan
        elif name == 'rcp': out = math.copysign(math.inf, v[0])
        elif name in {'rsq', 'hrsq'}: out = math.inf if v[0] == 0 else math.nan
        elif name in {'exp2', 'hexp2'}: out = math.inf if v[0] > 0 else 0.0
        else: out = math.nan
      return _float_bits(out, dest_half)
    a, b = (v + (0,))[:2]
    if name in {'add.u', 'add.s'}: out = a + b
    elif name in {'sub.u', 'sub.s'}: out = a - b
    elif name in {'cmps.u', 'cmpv.u'}: return int(_compare(a & 0xffffffff, b & 0xffffffff, condition))
    elif name in {'cmps.s', 'cmpv.s'}: return int(_compare(_signed(a, source_half), _signed(b, source_half), condition))
    elif name == 'min.u': out = min(a & 0xffffffff, b & 0xffffffff)
    elif name == 'max.u': out = max(a & 0xffffffff, b & 0xffffffff)
    elif name == 'min.s': out = min(_signed(a, source_half), _signed(b, source_half))
    elif name == 'max.s': out = max(_signed(a, source_half), _signed(b, source_half))
    elif name == 'absneg.s': out = a
    elif name == 'and.b': out = a & b
    elif name == 'or.b': out = a | b
    elif name == 'xor.b': out = a ^ b
    elif name == 'not.b': out = ~a
    elif name == 'mul.u24':
      bits = 16 if source_half else 24
      out = (a & ((1 << bits) - 1)) * (b & ((1 << bits) - 1))
    elif name == 'mul.s24':
      bits, sign = (16, 1 << 15) if source_half else (24, 1 << 23)
      a, b = a & ((1 << bits) - 1), b & ((1 << bits) - 1)
      out = (a - (1 << bits) if a & sign else a) * (b - (1 << bits) if b & sign else b)
    elif name == 'mull.u': out = (a & 0xffff) * (b & 0xffff)
    elif name == 'bfrev.b': out = int(f'{a & 0xffffffff:032b}'[::-1], 2)
    elif name == 'clz.s': out = 32 if a == 0 else 31 - int(math.log2(abs(_s32(a))))
    elif name == 'clz.b': out = 0xffffffff if not a & 0xffffffff else 32 - (a & 0xffffffff).bit_length()
    elif name == 'cbits.b': out = (a & 0xffffffff).bit_count()
    elif name == 'shl.b': out = a << (b & 31)
    elif name == 'shr.b': out = (a & 0xffffffff) >> (b & 31)
    elif name == 'ashr.b': out = _signed(a, source_half) >> (b & 31)
    elif name == 'getbit.b': out = (a >> (b & 31)) & 1
    else: raise NotImplementedError(f'unsupported IR3 ALU {name}')
    return out & (0xffff if dest_half else 0xffffffff)
  return run
