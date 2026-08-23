from __future__ import annotations
import ctypes, math, re, struct
from dataclasses import dataclass, field

TYBITS = {'pred':1,'b8':8,'s8':8,'u8':8,'b16':16,'s16':16,'u16':16,'f16':16,
          'b32':32,'s32':32,'u32':32,'f32':32,'b64':64,'s64':64,'u64':64,'f64':64}
FLOATS, SIGNS = frozenset({'f16','f32','f64'}), frozenset({'s8','s16','s32','s64'})
ROUNDS = frozenset({'rn','rz','rm','rp','rzi','rni','rmi','rpi','approx'})
CMP = {'eq':lambda a,b:a==b,'ne':lambda a,b:a!=b,'neu':lambda a,b:a!=b,
       'lt':lambda a,b:a<b,'gt':lambda a,b:a>b,'le':lambda a,b:a<=b,'ge':lambda a,b:a>=b}
PARAM_RE = re.compile(r'\.param\s+\.(\w+)\s+(\w+)')
SHARED_RE = re.compile(r'\.shared(?:\s+\.align\s+\d+)?\s+\.b8\s+(\w+)\[(\d+)\]')
ENTRY_RE = re.compile(r'\.entry\s+(\S+)')

def _mask(bits:int) -> int: return (1<<bits)-1 if bits < 64 else 0xFFFFFFFFFFFFFFFF
def _sext(x:int, bits:int) -> int:
  x &= _mask(bits)
  return x - (1<<bits) if x & (1<<(bits-1)) else x
def _pack_f(v:float, bits:int) -> int:
  if math.isnan(v): return {16:0x7e00, 32:0x7fc00000, 64:0x7ff8000000000000}[bits]
  sign_inf = {16:(0xfc00,0x7c00), 32:(0xff800000,0x7f800000), 64:(0xfff0000000000000,0x7ff0000000000000)}[bits]
  if not math.isfinite(v): return sign_inf[0] if math.copysign(1.0, v) < 0 else sign_inf[1]
  fmt = {16:'<eH', 32:'<fI', 64:'<dQ'}[bits]
  try: return struct.unpack(fmt[2], struct.pack(fmt[:2], float(v)))[0]
  except OverflowError: return sign_inf[0] if math.copysign(1.0, v) < 0 else sign_inf[1]
def _unpack_f(x:int, bits:int) -> float:
  if bits == 16: return struct.unpack('<e', struct.pack('<H', x & 0xffff))[0]
  if bits == 32: return struct.unpack('<f', struct.pack('<I', x & 0xffffffff))[0]
  return struct.unpack('<d', struct.pack('<Q', x & 0xffffffffffffffff))[0]
def _parse_imm(tok:str) -> int:
  t = tok.rstrip('UuLl')
  if t[:2] in {'0f','0F','0d','0D'}: return int(t[2:], 16)
  if t[:2] in {'0x','0X'}: return int(t, 16)
  return int(t, 10)

def _split_ops(s:str) -> list[str]:
  parts, buf, d1, d2 = [], [], 0, 0
  for ch in s:
    if ch == '{': d1 += 1
    elif ch == '}': d1 -= 1
    elif ch == '[': d2 += 1
    elif ch == ']': d2 -= 1
    if ch == ',' and d1 == 0 and d2 == 0:
      parts.append(''.join(buf).strip())
      buf = []
    else: buf.append(ch)
  if buf: parts.append(''.join(buf).strip())
  return parts

@dataclass
class Inst:
  pred: str|None
  pred_neg: bool
  op: str
  parts: list[str]
  ops: list[str]
  is_bar: bool = False

@dataclass
class Kernel:
  name: str
  params: list[tuple[str,str]] = field(default_factory=list)
  shared: dict[str,tuple[int,int]] = field(default_factory=dict)  # name -> (off, size)
  shared_size: int = 0
  labels: dict[str,int] = field(default_factory=dict)
  insts: list[Inst] = field(default_factory=list)

def parse_ptx(src:str) -> Kernel:
  k = Kernel("")
  in_entry = False
  for raw in src.splitlines():
    line = ' '.join(re.sub(r'//.*', '', raw).split()).rstrip(',')
    if not line: continue
    if not in_entry:
      if (m:=ENTRY_RE.search(line)):
        k.name, in_entry = m.group(1).rstrip('('), True
      continue
    if line == '}': break
    if line in '{)': continue
    if (m:=PARAM_RE.search(line)):
      k.params.append((m.group(2), m.group(1)))
      continue
    if (m:=SHARED_RE.search(line)):
      k.shared[m.group(1)] = (k.shared_size, int(m.group(2)))
      k.shared_size += int(m.group(2))
      continue
    if line.startswith('.'): continue
    if line.endswith(':'):
      k.labels[line[:-1]] = len(k.insts)
      continue
    pred, pred_neg, rest = None, False, line
    if rest.startswith('@'):
      neg = rest.startswith('@!')
      tok, rest = rest.split(None, 1)
      pred, pred_neg = tok[2 if neg else 1:], neg
    if not rest.endswith(';'): rest += ';'
    body = rest[:-1].strip()
    op, _, argstr = body.partition(' ')
    parts = op.split('.')
    inst = Inst(pred, pred_neg, parts[0], parts[1:], _split_ops(argstr) if argstr else [])
    inst.is_bar = parts[0] == 'bar'
    k.insts.append(inst)
  if not k.name: raise RuntimeError("no PTX .entry found")
  return k

class Thread:
  def __init__(self, tid, ntid, ctaid, nctaid):
    self.tid, self.ntid, self.ctaid, self.nctaid = tid, ntid, ctaid, nctaid
    self.regs: dict[str,int] = {}
    self.pc = 0
    self.halted = False
    self.wait_bar = False
  def special(self, name:str) -> int:
    axis = {'x':0,'y':1,'z':2}[name[-1]]
    if name.startswith('%tid'): return self.tid[axis]
    if name.startswith('%ntid'): return self.ntid[axis]
    if name.startswith('%ctaid'): return self.ctaid[axis]
    if name.startswith('%nctaid'): return self.nctaid[axis]
    raise RuntimeError(f"unknown special {name}")

class Emu:
  def __init__(self, k:Kernel, args:list[int], shared:bytearray):
    self.k, self.shared = k, shared
    self.params = {n: args[i] if i < len(args) else 0 for i,(n,_) in enumerate(k.params)}
  def read_reg(self, th:Thread, tok:str) -> int:
    tok = tok.strip()
    if tok.startswith('%') and '.' in tok: return th.special(tok)
    if tok.startswith('%'): return th.regs.get(tok, 0)
    if tok in self.k.shared: return self.k.shared[tok][0]
    if (m:=re.fullmatch(r'([A-Za-z_]\w*)\[(\d+)\]', tok)): return self.k.shared[m.group(1)][0] + int(m.group(2))
    return _parse_imm(tok)
  def write_reg(self, th:Thread, tok:str, val:int): th.regs[tok.strip()] = val
  def ty(self, parts:list[str]) -> str:
    for p in reversed(parts):
      if p in TYBITS: return p
    return 'b32'
  def vec(self, parts:list[str]) -> int:
    for p in parts:
      if p.startswith('v') and p[1:].isdigit(): return int(p[1:])
    return 1
  def space(self, parts:list[str]) -> str:
    for p in ('global','shared','param','local','const'):
      if p in parts: return p
    return 'reg'

  def _host_ok(self, addr:int, n:int) -> bool:
    if n <= 0 or addr <= 0: return False
    mem = getattr(self, '_mem', None)
    if mem is None:
      try:
        from test.mockgpu.cuda import cuda_state
        mem = cuda_state.memory
      except Exception: mem = {}
      self._mem = mem
    if not mem: return True
    for base, mv in mem.items():
      if base <= addr and addr + n <= base + mv.nbytes: return True
    return False
  def load_bytes(self, addr:int, n:int, space:str) -> bytes:
    if space == 'shared':
      end = min(addr+n, len(self.shared))
      if addr < 0 or addr >= len(self.shared): return b'\x00'*n
      return bytes(self.shared[addr:end]).ljust(n, b'\x00')
    if not self._host_ok(addr, n): return b'\x00'*n
    return ctypes.string_at(addr, n)
  def store_bytes(self, addr:int, data:bytes, space:str):
    if space == 'shared':
      if 0 <= addr < len(self.shared): self.shared[addr:addr+len(data)] = data[:max(0, len(self.shared)-addr)]
      return
    if self._host_ok(addr, len(data)): ctypes.memmove(addr, data, len(data))

  def parse_mem(self, th:Thread, tok:str) -> tuple[int,str|None]:
    tok = tok.strip()
    if tok.startswith('{'): return 0, None
    if (m:=re.fullmatch(r'\[([^+\]]+)(?:\+(\d+))?\]', tok)):
      base, off = m.group(1).strip(), int(m.group(2) or 0)
      if base in self.params: return self.params[base] + off, 'param'
      if base in self.k.shared: return self.k.shared[base][0] + off, 'shared'
      return self.read_reg(th, base) + off, None
    return self.read_reg(th, tok), None

  def vec_regs(self, tok:str) -> list[str]:
    tok = tok.strip()
    if tok.startswith('{'): return [x.strip() for x in tok[1:-1].split(',')]
    return [tok]

  def exec_ldst(self, th:Thread, inst:Inst, is_st:bool):
    ty, n, space = self.ty(inst.parts), self.vec(inst.parts), self.space(inst.parts)
    bits, nbytes = TYBITS[ty], TYBITS[ty]//8
    if is_st: mem, regs = inst.ops[0], self.vec_regs(inst.ops[1])
    else: regs, mem = self.vec_regs(inst.ops[0]), inst.ops[1]
    addr, memspace = self.parse_mem(th, mem)
    if memspace: space = memspace
    if space == 'param':
      m = re.match(r'\[([^+\]]+)(?:\+(\d+))?\]', mem.strip())
      val = self.params[m.group(1).strip()] if m else 0
      off = int(m.group(2) or 0) if m else 0
      if not is_st: self.write_reg(th, regs[0], (val >> (8*off)) & _mask(bits))
      return
    total = nbytes * n
    if is_st:
      chunk = b''.join(struct.pack('<'+{1:'B',2:'H',4:'I',8:'Q'}[nbytes], self.read_reg(th, r) & _mask(bits)) for r in regs)
      self.store_bytes(addr, chunk, space)
    else:
      data = self.load_bytes(addr, total, space)
      fmt = '<'+{1:'B',2:'H',4:'I',8:'Q'}[nbytes]*n
      for r,v in zip(regs, struct.unpack(fmt, data)):
        iv = int(v)
        if ty in SIGNS: iv = _sext(iv, bits)
        self.write_reg(th, r, iv)

  def exec_mov(self, th:Thread, inst:Inst):
    bits = TYBITS[self.ty(inst.parts)]
    self.write_reg(th, inst.ops[0], self.read_reg(th, inst.ops[1]) & _mask(bits))

  def exec_cvt(self, th:Thread, inst:Inst):
    types = [p for p in inst.parts if p in TYBITS]
    dst_ty, src_ty = (types[0], types[1] if len(types)>1 else types[0])
    rnd = next((p for p in inst.parts if p in ROUNDS), '')
    src = self.read_reg(th, inst.ops[1])
    if src_ty in FLOATS: sv = _unpack_f(src, TYBITS[src_ty])
    elif src_ty in SIGNS: sv = float(_sext(src, TYBITS[src_ty]))
    else: sv = float(src & _mask(TYBITS[src_ty]))
    if dst_ty in FLOATS:
      if rnd in {'rzi','rz'} and src_ty in FLOATS and math.isfinite(sv): sv = math.trunc(sv)
      self.write_reg(th, inst.ops[0], _pack_f(sv, TYBITS[dst_ty]))
    else:
      bits = TYBITS[dst_ty]
      if not math.isfinite(sv):
        sat = (1<<(bits-1))-1 if dst_ty in SIGNS else _mask(bits)
        iv = 0 if math.isnan(sv) else sat * (-1 if sv < 0 and dst_ty in SIGNS else 1)
      else:
        try: iv = int(math.trunc(sv) if rnd in {'rzi','rz'} else (round(sv) if rnd in {'rni','rn'} else sv))
        except (ValueError, OverflowError): iv = 0
      self.write_reg(th, inst.ops[0], iv & _mask(bits))

  def exec_setp(self, th:Thread, inst:Inst):
    cmpop = next(p for p in inst.parts if p in CMP)
    ty, bits = self.ty(inst.parts), TYBITS[self.ty(inst.parts)]
    a, b = self.read_reg(th, inst.ops[1]), self.read_reg(th, inst.ops[2])
    if ty in FLOATS: av, bv = _unpack_f(a, bits), _unpack_f(b, bits)
    elif ty in SIGNS: av, bv = _sext(a, bits), _sext(b, bits)
    else: av, bv = a & _mask(bits), b & _mask(bits)
    self.write_reg(th, inst.ops[0], int(CMP[cmpop](av, bv)))

  def exec_selp(self, th:Thread, inst:Inst):
    bits = TYBITS[self.ty(inst.parts)]
    a, b, p = self.read_reg(th, inst.ops[1]), self.read_reg(th, inst.ops[2]), self.read_reg(th, inst.ops[3])
    self.write_reg(th, inst.ops[0], (a if p else b) & _mask(bits))

  def exec_alu(self, th:Thread, inst:Inst):
    ty, bits = self.ty(inst.parts), TYBITS[self.ty(inst.parts)]
    xs = [self.read_reg(th, o) for o in inst.ops[1:]]
    op = inst.op
    if ty in FLOATS:
      fs = [_unpack_f(x, bits) for x in xs]
      def f1(fn, x):
        try: return fn(x)
        except (ValueError, OverflowError, ZeroDivisionError): return float('nan')
      if op == 'add': y = fs[0]+fs[1]
      elif op == 'mul': y = fs[0]*fs[1]
      elif op in {'fma','mad'}: y = fs[0]*fs[1]+fs[2]
      elif op == 'max': y = fs[1] if math.isnan(fs[0]) else (fs[0] if math.isnan(fs[1]) else max(fs[0], fs[1]))
      elif op == 'min': y = fs[1] if math.isnan(fs[0]) else (fs[0] if math.isnan(fs[1]) else min(fs[0], fs[1]))
      elif op == 'div':
        try: y = fs[0]/fs[1]
        except ZeroDivisionError: y = float('nan') if fs[0] == 0 else math.copysign(float('inf'), fs[0])
      elif op == 'rcp':
        try: y = 1.0/fs[0]
        except ZeroDivisionError: y = math.copysign(float('inf'), fs[0])
      elif op == 'ex2':
        try: y = math.pow(2.0, fs[0])
        except OverflowError: y = float('inf') if fs[0] > 0 else 0.0
        except (ValueError, ZeroDivisionError): y = float('nan')
      elif op == 'lg2': y = f1(math.log2, fs[0]) if fs[0] > 0 else (float('-inf') if fs[0]==0 else float('nan'))
      elif op == 'sin': y = f1(math.sin, fs[0])
      elif op == 'cos': y = f1(math.cos, fs[0])
      elif op == 'tan': y = f1(math.tan, fs[0])
      elif op == 'sqrt': y = f1(math.sqrt, fs[0]) if fs[0] >= 0 else float('nan')
      elif op == 'abs': y = abs(fs[0]) if not math.isnan(fs[0]) else fs[0]
      elif op == 'neg': y = -fs[0]
      elif op == 'trunc': y = math.trunc(fs[0]) if math.isfinite(fs[0]) else fs[0]
      elif op == 'copysign': y = math.copysign(abs(fs[0]) if not math.isnan(fs[0]) else fs[0], fs[1])
      else: raise RuntimeError(f"unimpl float op {op}.{'.'.join(inst.parts)}")
      self.write_reg(th, inst.ops[0], _pack_f(y, bits))
      return
    signed = ty in SIGNS
    ivs = [_sext(x, bits) if signed else x & _mask(bits) for x in xs]
    if op == 'add': y = ivs[0]+ivs[1]
    elif op == 'mul': y = ivs[0]*ivs[1]
    elif op in {'mad','fma'}: y = ivs[0]*ivs[1]+ivs[2]
    elif op == 'max': y = max(ivs[0], ivs[1])
    elif op == 'min': y = min(ivs[0], ivs[1])
    elif op == 'div': y = int(ivs[0]/ivs[1]) if ivs[1] else 0  # toward zero
    elif op == 'rem': y = ivs[0] - int(ivs[0]/ivs[1])*ivs[1] if ivs[1] else ivs[0]
    elif op == 'and': y = ivs[0]&ivs[1]
    elif op == 'or': y = ivs[0]|ivs[1]
    elif op == 'xor': y = ivs[0]^ivs[1]
    elif op == 'shl':
      amt = ivs[1] & _mask(32)
      y = 0 if amt >= bits else (ivs[0] << amt)
    elif op == 'shr':
      amt = ivs[1] & _mask(32)
      y = (ivs[0] >> amt) if amt < bits else ( -1 if signed and ivs[0] < 0 else 0)
    elif op == 'neg': y = -ivs[0]
    elif op == 'abs': y = abs(ivs[0])
    elif op == 'not': y = ~ivs[0]
    else: raise RuntimeError(f"unimpl int op {op}.{'.'.join(inst.parts)}")
    if ty == 'pred': y = int(bool(y))
    self.write_reg(th, inst.ops[0], y & _mask(bits))

  def step(self, th:Thread) -> str|None:
    inst = self.k.insts[th.pc]
    if inst.pred is not None:
      pv = self.read_reg(th, inst.pred if inst.pred.startswith('%') else '%'+inst.pred)
      if bool(pv) == inst.pred_neg:
        th.pc += 1
        return None
    if inst.op == 'ret':
      th.halted = True
      return None
    if inst.op == 'bra':
      th.pc = self.k.labels[inst.ops[0].lstrip('$')]
      return None
    if inst.is_bar:
      th.wait_bar = True
      return 'bar'
    if inst.op == 'ld': self.exec_ldst(th, inst, False)
    elif inst.op == 'st': self.exec_ldst(th, inst, True)
    elif inst.op == 'mov': self.exec_mov(th, inst)
    elif inst.op == 'cvt': self.exec_cvt(th, inst)
    elif inst.op == 'setp': self.exec_setp(th, inst)
    elif inst.op == 'selp': self.exec_selp(th, inst)
    else: self.exec_alu(th, inst)
    th.pc += 1
    return None

def run_kernel(k:Kernel, args:list[int], grid:tuple[int,int,int], block:tuple[int,int,int], smem:int):
  nctaid, ntid = grid, block
  shsz = max(k.shared_size, smem)
  for gz in range(nctaid[2]):
    for gy in range(nctaid[1]):
      for gx in range(nctaid[0]):
        shared = bytearray(shsz)
        emu = Emu(k, args, shared)
        threads = [Thread((lx,ly,lz), ntid, (gx,gy,gz), nctaid)
                   for lz in range(ntid[2]) for ly in range(ntid[1]) for lx in range(ntid[0])]
        while not all(t.halted for t in threads):
          runnable = [t for t in threads if not t.halted and not t.wait_bar]
          if not runnable:
            for t in threads:
              if t.wait_bar:
                t.wait_bar = False
                t.pc += 1
            continue
          for t in runnable:
            if t.pc >= len(k.insts):
              t.halted = True
              continue
            emu.step(t)

def _src_str(source) -> str:
  if isinstance(source, str): return source
  if isinstance(source, bytes): return source.split(b'\x00', 1)[0].decode()
  if isinstance(source, ctypes.c_char_p): return ctypes.string_at(source).decode()
  addr = source if isinstance(source, int) else ctypes.cast(source, ctypes.c_void_p).value
  return ctypes.string_at(addr).decode()

def _arg_int(v) -> int:
  if v is None: return 0
  if isinstance(v, int): return v
  return ctypes.cast(v, ctypes.c_void_p).value or 0

def ptx_run(source, n_args:int, args, blck_x:int, blck_y:int, blck_z:int,
            grid_x:int, grid_y:int, grid_z:int, shared_mem_size:int):
  src = _src_str(source)
  k = parse_ptx(src)
  argv = [_arg_int(args[i]) for i in range(n_args)]
  run_kernel(k, argv, (grid_x, grid_y, grid_z), (blck_x, blck_y, blck_z), shared_mem_size)
