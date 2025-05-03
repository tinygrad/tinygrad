from typing import Dict, List, Union, cast, Callable, Tuple
import subprocess
from functools import reduce
from operator import add, mul
import yaml

from tinygrad.device import Compiler
from tinygrad.helpers import getenv, temp
from tinygrad.renderer import Renderer, TensorCore
from tinygrad.ops import Ops, UOp, PatternMatcher, UPat, DType, GroupOp
from tinygrad.dtype import dtypes

DEBUG = getenv("DEBUG")

def asm_emulated_mod(ctx, d, a, b, dt, name) -> List[str]:
  t0, t1, t2 = ctx.tmpv[:3]
  if dtypes.is_float(dt):
    return [
      f"v_rcp_{name} {t0}, {b}", # t0 = 1 / b
      f"v_mul_{name} {t1}, {a}, {t0}", # t1 = a / b
      f"v_floor_{name} {t1}, {t1}", # t1 = floor(a / b)
      f"v_mul_{name} {t2}, {t1}, {b}", # t2 = floor(a / b) * b
      f"v_sub_{name} {d}, {a}, {t2}" # d = a - floor(a / b) * b
    ]
  else:
    raise NotImplementedError

def render_where(ctx, d, a, b, c, dt, name) -> List[str]:
  if a == "vcc":
    return [f"v_cndmask_b32 {d}, {c}, {b}"]
  return [f"v_cmp_neq_b32 {a}, 0", f"v_cndmask_b32 {d}, {c}, {b}"]

asm_for_op: Dict[Tuple[Ops, ...], Callable] = {
  (Ops.RECIP,): lambda ctx, d, a, dt, name: f"v_rcp_{name} {d}, {a}",
  (Ops.EXP2,): lambda ctx, d, a, dt, name: f"v_exp_{name} {d}, {a}", # exp is exp2
  (Ops.LOG2,): lambda ctx, d, a, dt, name: f"v_log_{name} {d}, {a}",
  (Ops.SIN,):  lambda ctx, d, a, dt, name: f"v_sin_{name} {d}, {a}",
  (Ops.SQRT,): lambda ctx, d, a, dt, name: f"v_sqrt_{name} {d}, {a}",
  (Ops.SHL,): lambda ctx, d, a, b, dt, name: f"v_lshlrev_b32 {d}, {b}, {a}",
  (Ops.SHR,): lambda ctx, d, a, b, dt, name: f"v_lshrrev_b32 {d}, {b}, {a}",
  (Ops.ADD,): lambda ctx, d, a, b, dt, name: f"v_add_{name} {d}, {a}, {b}",
  (Ops.MUL,): lambda ctx, d, a, b, dt, name: f"v_mul_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_mul_lo_{name} {d}, {a}, {b}",
  (Ops.XOR,): lambda ctx, d, a, b, dt, name: f"v_xor_b32 {d}, {a}, {b}",
  (Ops.AND,): lambda ctx, d, a, b, dt, name: f"v_and_b32 {d}, {a}, {b}",
  (Ops.OR,):  lambda ctx, d, a, b, dt, name: f"v_or_b32 {d}, {a}, {b}",
  (Ops.IDIV,): lambda ctx, d, a, b, dt, name: f"v_div_fixup_{name} {d}, {a}, {b}, {a}",
  (Ops.MAX,): lambda ctx, d, a, b, dt, name: f"v_max_{name} {d}, {a}, {b}",
  (Ops.MOD,): lambda ctx, d, a, b, dt, name: asm_emulated_mod(ctx, d, a, b, dt, name),
  (Ops.CMPLT,): lambda ctx, d, a, b, dt, name: f"v_cmp_lt_{name} {d}, {a}, {b}",
  (Ops.CMPNE,): lambda ctx, d, a, b, dt, name: f"v_cmp_ne_{name} {d}, {a}, {b}",
  (Ops.MULACC,): lambda ctx, a, b, c, dt, name: (
    f"v_fmac_{name} {c}, {a}, {b}" if dtypes.is_float(dt) else f"v_mad_{name} {c}, {a}, {b}, {c}"),
  (Ops.WHERE,): lambda ctx, d, a, b, c, dt, name: render_where(ctx, d, a, b, c, dt, name),
}

def get_reg_range(reg: str) -> List[int]:
  if "[" in reg:
    a, *b = reg[reg.index("[")+1 : reg.index("]")].split(":")
    return [int(a)] if not b else list(range(int(a), int(b[0]) + 1))
  i = next(i for i, c in enumerate(reg) if c.isdigit())
  return [int(reg[i:])]

def get_regs_contained(reg: str) -> List[str]:
  if "[" in reg:
    a, *b = reg[reg.index("[")+1 : reg.index("]")].split(":")
    return [f"{reg[0]}{a}"] if not b else [f"{reg[0]}{i}" for i in range(int(a), int(b[0]) + 1)]
  return [reg]

def render_reg_range(regs: List[str]) -> str:
  p = regs[0][0]
  r = list(sorted(reduce(add, map(get_reg_range, regs))))
  return f"{p}{r[0]}" if len(r) == 1 else f"{p}[{r[0]}:{r[-1]}]"

def render_addr_calc(ctx, base_ptr, offset_reg, dt) -> Tuple[List[str], str, str]:
  instructions = []

  base_lo, base_hi = get_regs_contained(ctx.r[base_ptr])
  addr_lo, addr_hi, tmp = ctx.tmpv[:3]
  tmp_off = ctx.tmps[:1]

  # Ensure offset is in a VGPR for src1 of VOP2
  if offset_reg[0] == "s":
    instructions.append(f"s_mul_lo_u32 {tmp_off}, {offset_reg}, {dt.itemsize * dt.vcount}")
    instructions.append(f"v_mov_b32 {tmp}, {tmp_off}")
  else:
    instructions.append(f"v_mul_lo_u32 {tmp}, {offset_reg}, {dt.itemsize * dt.vcount}")
  off = tmp

  # Reset overflow
  instructions.append("s_mov_b64 vcc, 0")

  # Add low 32-bits: addr_lo = base_lo + offset
  instructions.append(f"v_add_co_ci_u32 {addr_lo}, {base_lo}, {off}")

  # Add high 32-bits: addr_hi = base_hi + carry
  if base_hi[0] == "s":
    instructions.append(f"v_mov_b32 {tmp}, {base_hi}")
    base_hi = tmp

  instructions.append(f"v_add_co_ci_u32 {addr_hi}, 0, {base_hi}")

  return instructions, addr_lo, addr_hi

def is_smem_load(x):
  return any(_x.op is Ops.DEFINE_LOCAL for _x in x.src[0].toposort)

def render_add_64(ctx, dst, val: int) -> List[str]:
  instructions = []

  base_lo, base_hi = get_regs_contained(dst)
  tmp = ctx.tmpv[0]

  # Reset overflow
  instructions.append("s_mov_b64 vcc, 0")

  instructions.append(f"v_mov_b32 {tmp}, {val}")
  # Add low 32-bits: addr_lo = base_lo + offset
  instructions.append(f"v_add_co_ci_u32 {base_lo}, {base_lo}, {tmp}")

  # Add high 32-bits: addr_hi = base_hi + carry
  if base_hi[0] == "s":
    instructions.append(f"v_mov_b32 {tmp}, {base_hi}")
    base_hi = tmp

  instructions.append(f"v_add_co_ci_u32 {base_hi}, 0, {base_hi}")

  return instructions


def render_load(ctx, d, ptr, offset, dt, smem) -> List[str]:
  def load_instruction(load_type, reg, addr):
    return f"{load_type} {reg}, {addr} off"

  load_bits = max(int(ctx.types[dt.scalar()][1:]) * dt.count, 32)
  load_type = "ds_load_b" if smem else "global_load_b"

  addr_setup_ins = []
  tmp_addr_base = ctx.tmpv[0]
  addr_reg_base: str

  if len(get_reg_range(ctx.r[ptr])) == 2:
    base_addr_ins, addr_lo_v, addr_hi_v = render_addr_calc(ctx, ptr, ctx.r[offset], offset.dtype)
    addr_setup_ins += base_addr_ins
    addr_reg_base = render_reg_range([addr_lo_v, addr_hi_v])
  else:
    addr_setup_ins.append(f"v_add_u32 {tmp_addr_base}, {ctx.r[ptr]}, {ctx.r[offset]}")
    addr_reg_base = tmp_addr_base

  if dt.count > 1:
    load_ins = []
    full_chunks = dt.itemsize // 16
    remaining_bytes = dt.itemsize % 16

    for i in range(full_chunks):
      reg_slice = slice_reg(d, i * 4, (i + 1) * 4 - 1)
      offset_val = 16
      load_ins.extend(render_add_64(ctx, addr_reg_base, offset_val))
      load_ins.append(load_instruction(f"{load_type}128", reg_slice, addr_reg_base))

    if remaining_bytes > 0:
      regs_used = full_chunks * 4
      rem_regs = remaining_bytes // dt.scalar().itemsize
      reg_slice = slice_reg(d, regs_used, regs_used + rem_regs)
      offset_val = 16
      if len(get_regs_contained(addr_reg_base)) == 2:
        load_ins.extend(render_add_64(ctx, addr_reg_base, offset_val))
      else:
        load_ins.append(f"v_add_u32 {addr_reg_base}, {offset_val}")
      load_ins.append(load_instruction(f"{load_type}{remaining_bytes // 4 * 8}", reg_slice, addr_reg_base))

    return addr_setup_ins + load_ins

  load_ins = [load_instruction(f"{load_type}{load_bits}", d, addr_reg_base)]
  return addr_setup_ins + load_ins

def render_store(ctx, x, ptr, offset, val) -> List[str]:
  store_bits = max(int(ctx.types[val.dtype.scalar()][1:]) * x.dtype.count, 32)

  addr_setup_ins = []
  if len(get_reg_range(ctx.r[ptr])) == 2:
    base_addr_ins, addr_lo_v, addr_hi_v = render_addr_calc(ctx, ptr, ctx.r[offset], offset.dtype)
    addr_setup_ins += base_addr_ins
    addr_reg_base = render_reg_range([addr_lo_v, addr_hi_v])
  else:
    addr_setup_ins.append(f"v_add_u32 {ctx.tmpv[0]}, {ctx.r[ptr]}, {ctx.r[offset]}")
    addr_reg_base = ctx.tmpv[0]

  store_ins = []
  if val.dtype.count > 1:
    full_chunks = val.dtype.itemsize // 8
    remaining_bytes = val.dtype.itemsize % 8

    for i in range(full_chunks):
      reg_slice = slice_reg(ctx.r[val], i * 2, (i + 1) * 2 - 1)
      addr_tmp = render_reg_range(ctx.tmpv[:2])
      offset_val = 8
      store_ins.extend(render_add_64(ctx, addr_reg_base, offset_val))
      store_ins.append(f"global_store_b64 {addr_tmp}, {reg_slice} off")

    if remaining_bytes > 0:
      regs_used = full_chunks * 2
      rem_regs = remaining_bytes // val.dtype.scalar().itemsize
      reg_slice = slice_reg(x, regs_used, regs_used + rem_regs)
      addr_tmp = ctx.tmpv[2]
      offset_val = 8
      store_ins.extend(render_add_64(ctx, addr_reg_base, offset_val))
      store_ins.append(f"global_store_b{remaining_bytes // 4 * 8} {addr_tmp}, {reg_slice} off")
  else:
    store_ins.append(f"global_store_b{store_bits} {addr_reg_base}, {ctx.r[val]} off")

  return addr_setup_ins + store_ins

def render_const_mod(ctx, d, val, modulus) -> List[str]:
  if not dtypes.is_unsigned(val.dtype):
    raise NotImplementedError

  divisor = modulus.arg
  assert divisor >= 1 and divisor <= 0xFFFFFFFF

  nbits = 32
  two_to_n = 1 << nbits  # 2^32

  m, r = two_to_n // divisor, two_to_n % divisor
  if r >= divisor // 2:
    m += 1

  t0, t1, t2 = ctx.tmpv[:3]

  return [
    f"v_mul_hi_u32 {t0}, {ctx.r[val]}, {m}",
    f"v_lshrrev_b32 {t1}, 32, {t0}",
    f"v_mul_lo_u32 {t2}, {t1}, {divisor}",
    f"v_sub_u32 {d}, {ctx.r[val]}, {t2}",
  ]

def slice_reg(reg: str, start: int, end: int) -> str:
  s = int(reg[2:].split(":")[0])
  return f"{reg[0]}[{s + start}:{s + end}]"

def render_wmma(ctx, uop):
  shape, dtype_in, dtype_out = uop.arg[1], uop.arg[2], uop.arg[3]
  assert shape == (16, 16, 16)

  a_base = ctx.r[uop.src[0]]
  b_base = ctx.r[uop.src[1]]
  c_base = ctx.r[uop.src[2]]
  d_base = ctx.r[uop]

  a = slice_reg(a_base, 0, 7)
  b = slice_reg(b_base, 0, 7)
  c = slice_reg(c_base, 0, 7)
  d = slice_reg(d_base, 0, 7)

  return [f"v_wmma_f16_16x16x16_f16 {d}, {a}, {b}, {c}"]

supports_half: List[Ops] = [Ops.EXP2, Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPLT, Ops.WHERE]
doesnt_support_half: Tuple[Ops, ...] = tuple(op for op in asm_for_op.keys() if op not in supports_half)
rdna3_rewrite = PatternMatcher([
  (UPat(Ops.CONST, name="x", dtype=dtypes.bool), lambda ctx, x: f"v_cmp_ne_u32 {ctx.r[x]}, {str(x.arg)}, 0"),
  (UPat(Ops.CONST, name="x"), lambda ctx, x: f"v_mov_b32 {ctx.r[x]}, {str(x.arg)}"),

  *[
    (UPat(op, name="x"), lambda ctx, x, op=op: asm_for_op[(op,)](
      ctx, ctx.r[x], *[ctx.r[v] for v in x.src], x.dtype, ctx.types[x.src[0].dtype]))
    for op in [
      Ops.RECIP, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT,
      Ops.ADD, Ops.MUL, Ops.XOR, Ops.AND, Ops.OR,
      Ops.SHL, Ops.SHR, Ops.IDIV, Ops.MAX, Ops.CMPLT, Ops.CMPNE
    ]
  ],

  (UPat(Ops.MULACC, name="x"),
    lambda ctx, x: asm_for_op[(Ops.MULACC,)](ctx, ctx.r[x.src[0]], ctx.r[x.src[1]], ctx.r[x.src[2]], x.dtype, ctx.types[x.dtype])),

  # only local idx
  (UPat(Ops.SPECIAL, name="x"),
    lambda ctx, x: [
     f"v_lshrrev_b32 {ctx.r[x]}, {10 * x.arg[0]}, v0", # the isa document disagrees with the emulator
     f"v_and_b32 {ctx.r[x]}, {ctx.r[x]}, 0x3FF",
   ]),

  (UPat(Ops.WHERE, name="x"),
    lambda ctx, x: asm_for_op[(Ops.WHERE,)](ctx, ctx.r[x], ctx.r[x.src[0]], ctx.r[x.src[1]], ctx.r[x.src[2]], x.dtype, ctx.types[x.dtype])),

  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx, x: [f"s_load_b64 {ctx.r[x]}, s[0:1], {ctx.args[x][".offset"]}", "s_waitcnt lgkmcnt(0)"]),
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx, x: [f"s_load_b32 {ctx.r[x]}, s[0:1], {ctx.args[x][".offset"]}", "s_waitcnt lgkmcnt(0)"]),

  (UPat(Ops.MOD, name="x", src=(UPat.var("val"), UPat.cvar("modulus"))),
    lambda ctx, x, val, modulus: render_const_mod(ctx, ctx.r[x], val, modulus)),

  (UPat(Ops.MOD, name="x"),
    lambda ctx, x: asm_for_op[(Ops.MOD,)](ctx, ctx.r[x], ctx.r[x.src[0]], ctx.r[x.src[1]], x.dtype, ctx.types[x.dtype])),

  (UPat(Ops.LOAD, name="x", src=(UPat.var("ptr"), UPat.var("offset"))), lambda ctx, x, ptr, offset:
    render_load(ctx, ctx.r[x], ptr, offset, x.dtype, is_smem_load(x))
  ),

  (UPat(Ops.STORE, name="x", src=(UPat.var("ptr"), UPat.var("offset"), UPat.var("val"))),
   lambda ctx, x, ptr, offset, val: render_store(ctx, x, ptr, offset, val)),

  (UPat(Ops.ASSIGN, name="x"),
   lambda ctx, x: f"v_mov_b{ctx.types[x.dtype][1:] * x.dtype.count} {ctx.r[x.src[0]]}, {ctx.r[x.src[1]]}"),

  (UPat(Ops.RANGE, name="x"),
   lambda ctx, x: [f"v_mov_b32 {ctx.r[x]}, {ctx.r[x.src[0]]}", f"LABEL_LOOP_{ctx.r[x][1:]}:"]),

  (UPat(Ops.WMMA, name="x"),
   lambda ctx, x: list(render_wmma(ctx, x))),

  (UPat(Ops.IF, name="x"),
   lambda ctx, x: [
    f"s_and_saveexec_b64 {ctx.r[x]}, vcc",
    f"LABEL_IF_{ctx.r[x.src[0]][1:]}:",
   ] if ctx.r[x.src[0]][0] == 'v' else [
    f"s_cmp_eq_u32 {ctx.r[x.src[0]]}, 0", # may be unnecessary
    f"s_cbranch_scc0 LABEL_ELSE_{ctx.r[x.src[0]][1:]}",
   ]),

  (UPat(Ops.ENDIF, name="x"),
   lambda ctx, x: [
    f"s_xor_b64 {render_reg_range(ctx.tmps[:2])}, exec, {ctx.r[x.src[0]]}",
    f"s_and_b64 exec, exec, {render_reg_range(ctx.tmps[:2])}",
    f"LABEL_ELSE_{ctx.r[x.src[0].src[0]][1:]}:",
   ] if ctx.r[x.src[0].src[0]][0] == 'v' else [
    f"LABEL_ELSE_{ctx.r[x.src[0].src[0]][1:]}:",
   ]),

  (UPat(Ops.ENDRANGE, name="x", src=(UPat.var("src0"),)),
   lambda ctx, x, src0: [
       asm_for_op[(Ops.ADD,)](ctx, ctx.r[src0], ctx.r[src0], "1", dtypes.int, ctx.types[dtypes.int]),
       asm_for_op[(Ops.CMPLT,)](ctx, ctx.r[src0], ctx.r[src0], ctx.l[src0], dtypes.int, ctx.types[dtypes.int]),
       f"s_cbranch_vccnz LABEL_LOOP_{ctx.r[src0][1:]}"
   ]),

  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: []),

  (UPat(Ops.DEFINE_ACC, name="x", src=(UPat.cvar("pred", dtype=dtypes.bool),), allow_any_len=True),
   lambda ctx, x, pred: [
       f"v_cmp_neq_{ctx.types[x.src[0].dtype]} {str(pred.arg)}, 0",
       f"v_cndmask_b32 {ctx.r[x]}, 0.0, 0.0"
   ])
])

# vop2 only for now (how can this be expanded easily?)
dual_combs: List[Tuple[Ops, Ops]] = [
  (Ops.ADD, Ops.MUL),
  (Ops.MUL, Ops.ADD),
]


# fused_ops = PatternMatcher([
#   (UPat(Ops.ADD, name="x",
#         dtype=dtypes.float32,
#         src=(UPat(Ops.MUL, name="a1"), UPat.var("a2"))), lambda: ctx, x, y: ),
# ])

class RDNA3Renderer(Renderer):
  device = "ROCm"
  extra_matcher = None
  tmpv = [f"v{i}" for i in range(3, 6)]
  tmps = [f"s{i}" for i in range(60, 63)]


  # tc_8168_f16 = [TensorCore(dims=(8,16,8), threads=32, elements_per_thread=(4,2,4), dtype_in=di, dtype_out=do, opts=(),
  #   swizzle=(((6,7,2,3,4),(0,1,8,5,9)), ((6,7,8,0,1),(2,3,4,9,5)))) for di,do in [(dtypes.half,dtypes.float), (dtypes.half,dtypes.half)]]

  def __init__(self, arch:str="gfx1100", device="ROCm"):
    self.device, self.tensor_cores, self.arch = device, [], arch
  def __reduce__(self): return self.__class__, (self.arch, self.device)

  # language options
  kernel_prefix = """
.rodata
.global {function_name}.kd
.type {function_name}.kd,STT_OBJECT
.align 0x10
.amdhsa_kernel {function_name}"""
  # barrier = "bar.sync\t0;"
  supports_half = supports_half
  # HACK: Use s16 and u16 for int8 and uint8 buffers. This can be wrong in cast.
  types: Dict[DType, str] = { dtypes.int8: "i16", dtypes.int16: "i16", dtypes.int32: "i32", dtypes.int64: "i64",
                              dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
                              dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "u32" }

  mem_types: Dict[DType, str] =  types.copy()
  mem_types.update({dtypes.int8: "i8", dtypes.uint8: "u8", dtypes.bool: "u8", dtypes.float16: "b16"})

  def render(self, uops:List[UOp]) -> str:
    self.v_cnt = 6  # v[0:2] is local_xyz, v[3:5] are used for temporaries
    self.s_cnt = 2  # s[0:1] is the global address, s[2:4] is global_xyz

    self.r: Dict[UOp, str] = {}
    ins: List[str] = []
    self.args: Dict[UOp, dict] = {}
    self.l: Dict[UOp, str] = {}
    allocate = 0
    args_offset = 0

    def alloc_vregs(elems: int):
      if elems == 1:
        reg = f"v{self.v_cnt}"
      else:
        reg = f"v[{self.v_cnt}:{self.v_cnt + elems - 1}]"
      self.v_cnt += elems
      return reg

    for u in uops:
      if u.op == Ops.DEFINE_GLOBAL:
        self.args[u] = {".address_space": "global", ".name": f"buf_{u.arg}", ".offset": args_offset, ".size": 8,
                     ".type_name": u.dtype.name+"*", ".value_kind": "global_buffer"}
        args_offset += 8
        self.s_cnt += self.s_cnt%2
        self.r[u] = f"s[{self.s_cnt}:{self.s_cnt+1}]"
        self.s_cnt += 2
      elif u.op == Ops.DEFINE_LOCAL:
        self.args[u] = {".name": f"buf_{u.arg}", ".offset": args_offset, ".size": 4,
                     ".type_name": u.dtype.name+"*", ".value_kind": "by_value"}
        args_offset += 4
        allocate += u.dtype.itemsize * u.dtype.vcount
        self.r[u] = f"s{self.s_cnt}"
        self.s_cnt += 1

    # reserve for global_xyz
    self.s_cnt = 16

    for u in uops:
      print(uops.index(u), u.op)
      if any(src.op is Ops.LOAD for src in u.src):
        ins.append("  s_waitcnt vmcnt(0)")

      if u.op in {Ops.CMPLT, Ops.CMPNE}:
        self.r[u] = f"s{self.s_cnt}"
        self.s_cnt+= 1
      elif u.op == Ops.MULACC:
        self.r[u] = self.r[u.src[0]]
      elif u.op in GroupOp.ALU | {Ops.CONST, Ops.LOAD, Ops.RANGE}:
        self.r[u] = alloc_vregs(u.dtype.itemsize // 4)
      elif u.op == Ops.IF:
        self.r[u] = f"s[{self.s_cnt}:{self.s_cnt + 1}]"
        self.s_cnt += 2
      elif u.op == Ops.SPECIAL:
        if u.arg[1].startswith("lidx"):
          self.r[u] = f"v{self.v_cnt}"
          self.v_cnt += 1
        elif u.arg[1].startswith("gidx"):
          self.r[u] = f"s{13 + u.arg[0]}"# the isa document disagrees with the emulator
          continue
        else:
          raise NotImplementedError
      elif u.op == Ops.WMMA:
        def alloc_wmma_tile(dtype: DType, num_tiles: int):
          elems = 16 * 16
          # the whole wavefront is cooporating (assume wave32, 32bit lanes)
          eff_size = 32 * 4
          return alloc_vregs((elems * dtype.itemsize) // eff_size * num_tiles)

        (N, M, K), dtype_in, dtype_out = u.arg[1], u.arg[2], u.arg[3]
        MT, NT, KT = 16, 16, 16

        # num_a_tiles = (M // MT) * (K // KT)
        # num_b_tiles = (K // KT) * (N // NT)
        num_c_tiles = (M // MT) * (N // NT)

        self.r[u] = alloc_wmma_tile(dtype_out, num_c_tiles)
      if u.op in {Ops.BITCAST}:
        # noops
        self.r[u] = self.r[u.src[0]]
      if u.op == Ops.RANGE:
        # TODO: what is the arg?
        self.l[u] = self.r[u.src[1]]

      if (l:=cast(Union[str, List[str]], rdna3_rewrite.rewrite(u, ctx=self))) is None:
        raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      l = [l] if isinstance(l, str) else l
      l = [f"  {u}" if u[-1] != ":" else u for u in l]
      ins.extend(l)

    args = list(self.args.values())
    ins.append("  s_waitcnt vmcnt(0)")
    metadata = {
      "amdhsa.kernels": [{".args": args,
        ".group_segment_fixed_size": allocate, ".kernarg_segment_align": 8, ".kernarg_segment_size": args[-1][".offset"] + args[-1][".size"],
        ".language": "OpenCL C", ".language_version": [1, 2], ".max_flat_workgroup_size": 256,
        ".name": "kernel", ".private_segment_fixed_size": 0, ".sgpr_count": self.s_cnt, ".sgpr_spill_count": 0,
        ".symbol": f"kernel.kd", ".uses_dynamic_stack": False, ".vgpr_count": self.v_cnt, ".vgpr_spill_count": 0,
        ".wavefront_size": 32}],
      "amdhsa.target": "amdgcn-amd-amdhsa--gfx1100", "amdhsa.version": [1, 2], "COMPUTE_PGM_RSRC2.tgid_x_en": 1, "COMPUTE_PGM_RSRC2.tgid_y_en": 1, "COMPUTE_PGM_RSRC2.tgid_z_en": 1}

    boilerplate_start = f"""
.rodata
.global kernel.kd
.type kernel.kd,STT_OBJECT
.align 0x10
.amdhsa_kernel kernel"""

    kernel_desc = {
      ".amdhsa_group_segment_fixed_size": allocate, ".amdhsa_private_segment_fixed_size": 0, ".amdhsa_kernarg_size": 0,
      ".amdhsa_next_free_vgpr": self.v_cnt,
      ".amdhsa_reserve_vcc": 0, ".amdhsa_reserve_xnack_mask": 0,
      ".amdhsa_next_free_sgpr": self.s_cnt,
      ".amdhsa_float_round_mode_32": 0, ".amdhsa_float_round_mode_16_64": 0, ".amdhsa_float_denorm_mode_32": 3, ".amdhsa_float_denorm_mode_16_64": 3,
      ".amdhsa_dx10_clamp": 1, ".amdhsa_ieee_mode": 1, ".amdhsa_fp16_overflow": 0,
      ".amdhsa_workgroup_processor_mode": 1, ".amdhsa_memory_ordered": 1, ".amdhsa_forward_progress": 0, ".amdhsa_enable_private_segment": 0,
      ".amdhsa_system_sgpr_workgroup_id_x": 1, ".amdhsa_system_sgpr_workgroup_id_y": 1, ".amdhsa_system_sgpr_workgroup_id_z": 1,
      ".amdhsa_system_sgpr_workgroup_info": 0, ".amdhsa_system_vgpr_workitem_id": 2,
      ".amdhsa_exception_fp_ieee_invalid_op": 0, ".amdhsa_exception_fp_denorm_src": 0,
      ".amdhsa_exception_fp_ieee_div_zero": 0, ".amdhsa_exception_fp_ieee_overflow": 0, ".amdhsa_exception_fp_ieee_underflow": 0,
      ".amdhsa_exception_fp_ieee_inexact": 0, ".amdhsa_exception_int_div_zero": 0,
      ".amdhsa_user_sgpr_dispatch_ptr": 0, ".amdhsa_user_sgpr_queue_ptr": 0, ".amdhsa_user_sgpr_kernarg_segment_ptr": 1,
      ".amdhsa_user_sgpr_dispatch_id": 0, ".amdhsa_user_sgpr_private_segment_size": 0, ".amdhsa_wavefront_size32": 1, ".amdhsa_uses_dynamic_stack": 0}

    code_start = f""".end_amdhsa_kernel
.text
.global kernel
.type kernel,@function
.p2align 8
kernel:
"""

    ins += ["  s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)", "  s_endpgm", "  s_code_end"]
    return ".amdgpu_metadata\n" + yaml.dump(metadata) + ".end_amdgpu_metadata" + \
           boilerplate_start + "\n" + "\n".join("%s %d" % x for x in kernel_desc.items()) + "\n" + code_start + \
           "\n".join(ins) + f"\n.size kernel, .-kernel"

class RDNACompiler(Compiler):
  def __init__(self, arch:str="gfx1100", cache_key="rdna"):
    self.arch = arch
    super().__init__(f"compile_{cache_key}_{self.arch}")

  def compile(self, src:str) -> bytes:
    if DEBUG >= 4:
      print(src)
    code, obj = temp("code"), temp("obj")
    with open(code, "w") as f: f.write(src)
    subprocess.run(["llvm-mc", "--arch=amdgcn", f"--mcpu={self.arch}", "--triple=amdgcn-amd-amdhsa", "-filetype=obj", "-o", obj, code])
    return open(obj, "rb").read()

from tinygrad.runtime.ops_amd import AMDAllocator, AMDDevice, AMDProgram
from tinygrad.helpers import flat_mv
import numpy as np

def test_simple():
  device = AMDDevice()
  allocator = AMDAllocator(device)
  compiler = RDNACompiler("gfx1100")
  renderer = RDNA3Renderer("gfx1100")

  uops = []

  gC = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 0)

  cK = UOp(Ops.CONST, dtypes.uint32, (), 4)
  c0 = UOp(Ops.CONST, dtypes.uint32, (), 0)

  store = UOp(Ops.STORE, dtypes.float32, (gC, c0, cK))
  uops += [gC, cK, c0, store]

  exe = compiler.compile(renderer.render(uops))
  prog = AMDProgram(device, "test", exe)
  a = allocator.alloc(1*8)
  prog(a, wait=True)
  na = np.empty(1, np.uint64)
  allocator._copyout(flat_mv(na.data), a)
  assert na == [4]

def test_scalar_add():
  device = AMDDevice()
  allocator = AMDAllocator(device)
  compiler = RDNACompiler("gfx1100")
  renderer = RDNA3Renderer("gfx1100")

  uops = []
  gA = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 0)
  gB = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 1)
  gC = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 2)
  c0 = UOp(Ops.CONST, dtypes.uint32, (), 0)
  a_val = UOp(Ops.LOAD, dtypes.float32, (gA, c0))
  b_val = UOp(Ops.LOAD, dtypes.float32, (gB, c0))
  sum_val = UOp(Ops.ADD, dtypes.float32, (a_val, b_val))
  store = UOp(Ops.STORE, dtypes.float32, (gC, c0, sum_val))
  uops += [gA, gB, gC, c0, a_val, b_val, sum_val, store]

  exe = compiler.compile(renderer.render(uops))
  prog = AMDProgram(device, "scalar_add", exe)
  a = allocator.alloc(4)
  b = allocator.alloc(4)
  c = allocator.alloc(4)
  allocator._copyin(a, memoryview(np.array([3.5], dtype=np.float32)))
  allocator._copyin(b, memoryview(np.array([2.0], dtype=np.float32)))
  prog(a, b, c, wait=True)
  out = np.empty(1, dtype=np.float32)
  allocator._copyout(flat_mv(out.data), c)
  assert np.allclose(out, [5.5])

def test_matrix_transpose():
  device = AMDDevice()
  allocator = AMDAllocator(device)
  compiler = RDNACompiler("gfx1100")
  renderer = RDNA3Renderer("gfx1100")

  M, N = 2, 3
  uops = []
  gA = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 0)
  gT = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 1)
  uops += [gA, gT]

  cM = UOp(Ops.CONST, dtypes.uint32, (), M)
  cN = UOp(Ops.CONST, dtypes.uint32, (), N)
  i = UOp(Ops.SPECIAL, dtypes.uint32, (), (0, "lidx0"))
  j = UOp(Ops.SPECIAL, dtypes.uint32, (), (1, "lidx1"))

  a_offset = UOp(Ops.MUL, dtypes.uint32, (i, cN))
  a_idx = UOp(Ops.ADD, dtypes.uint32, (a_offset, j))
  val = UOp(Ops.LOAD, dtypes.float32, (gA, a_idx))

  t_offset = UOp(Ops.MUL, dtypes.uint32, (j, cM))
  t_idx = UOp(Ops.ADD, dtypes.uint32, (t_offset, i))
  store = UOp(Ops.STORE, dtypes.float32, (gT, t_idx, val))
  uops += [cM, cN, i, j, a_offset, a_idx, val, t_offset, t_idx, store]

  exe = compiler.compile(renderer.render(uops))
  prog = AMDProgram(device, "transpose", exe)

  a = allocator.alloc(M * N * 4)
  t = allocator.alloc(M * N * 4)

  mat = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32).flatten()
  allocator._copyin(a, memoryview(mat))
  prog(a, t, global_size=(1, 1, 1), local_size=(M, N, 1), wait=True)

  out = np.empty(M * N, dtype=np.float32)
  allocator._copyout(memoryview(out), t)

  expected = np.transpose(mat.reshape(M, N)).flatten()
  assert np.allclose(out, expected)

def test_uniform_conditional():
  device = AMDDevice()
  allocator = AMDAllocator(device)
  compiler = RDNACompiler("gfx1100")
  renderer = RDNA3Renderer("gfx1100")


  for val in [0, 1]:
    uops = []
    gC = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 0)
    c0 = UOp(Ops.CONST, dtypes.uint32, (), 0)

    lidx0 = UOp(Ops.SPECIAL, dtypes.uint32, (), (0, "lidx0"))

    v = UOp(Ops.CONST, dtypes.uint32, (), val)
    pred = UOp(Ops.CMPNE, dtypes.bool, (lidx0, v))

    if_op = UOp(Ops.IF, dtypes.void, (pred,))
    val_then = UOp(Ops.CONST, dtypes.float32, (), 1.0)
    store_then = UOp(Ops.STORE, dtypes.float32, (gC, c0, val_then))
    endif = UOp(Ops.ENDIF, dtypes.void, (if_op,))

    uops += [lidx0, gC, c0, v, pred, if_op, val_then, store_then, endif]

    exe = compiler.compile(renderer.render(uops))
    prog = AMDProgram(device, "test_uniform_conditional", exe)

    c = allocator.alloc(4)
    prog(c, wait=True)

    out = np.empty(1, dtype=np.float32)
    allocator._copyout(flat_mv(out.data), c)

    assert np.allclose(out, [float(0 == val)])

def test_wmma_matmul():
  device = AMDDevice()
  allocator = AMDAllocator(device)
  compiler = RDNACompiler("gfx1100")
  renderer = RDNA3Renderer("gfx1100")

  uops = []
  shape = (16, 16, 16)

  gA = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), (), 0)
  gB = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), (), 1)
  gC = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 2)
  gD = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 3)
  uops += [gA, gB, gC, gD]

  i = UOp(Ops.SPECIAL, dtypes.uint32, (), (0, "lidx0"))  # thread id in wave32
  uops += [i]

  const_16 = UOp(Ops.CONST, dtypes.uint32, (), 16)
  const_32 = UOp(Ops.CONST, dtypes.uint32, (), 32)
  uops += [const_16, const_32]

  const_8 = UOp(Ops.CONST, dtypes.uint32, (), 8)
  a_offset = UOp(Ops.MUL, dtypes.uint32, (i, const_16))  # 16 FP16 elems per thread
  b_offset = UOp(Ops.MUL, dtypes.uint32, (i, const_16))
  c_offset = UOp(Ops.MUL, dtypes.uint32, (i, const_8))  # 8 FP32 elems
  uops += [a_offset, b_offset, const_8, c_offset]

  a_vec = UOp(Ops.LOAD, dtypes.half.vec(16), (gA, a_offset))
  b_vec = UOp(Ops.LOAD, dtypes.half.vec(16), (gB, b_offset))
  c_vec = UOp(Ops.LOAD, dtypes.float32.vec(8), (gC, c_offset))
  uops += [a_vec, b_vec, c_vec]

  wmma = UOp(Ops.WMMA, dtypes.float32.vec(8), (a_vec, b_vec, c_vec), (None, shape, dtypes.half, dtypes.float32))
  uops += [wmma]

  store = UOp(Ops.STORE, dtypes.void, (gD, c_offset, wmma))
  uops += [store]

  # Compile and run
  exe = compiler.compile(renderer.render(uops))
  prog = AMDProgram(device, "test_wmma_matmul", exe)

  # Allocate exact amount: 32 threads × per-thread tile
  a = allocator.alloc(32 * 16 * 2)   # 16 FP16 = 32B
  b = allocator.alloc(32 * 16 * 2)
  c = allocator.alloc(32 * 8 * 4)    # 8 FP32 = 32B
  d = allocator.alloc(32 * 8 * 4)

  # Fill tiles
  A = np.arange(32 * 16, dtype=np.float16)
  B = np.arange(32 * 16, dtype=np.float16)
  C = np.zeros(32 * 8, dtype=np.float32)

  allocator._copyin(a, memoryview(A))
  allocator._copyin(b, memoryview(B))
  allocator._copyin(c, memoryview(C))

  prog(a, b, c, d, global_size=(1, 1, 1), local_size=(32, 1, 1), wait=True)

  out = np.empty(32 * 8, dtype=np.float32)
  allocator._copyout(memoryview(out), d)

  # Check: matmul + acc per thread
  expected = np.zeros(32 * 8, dtype=np.float32)
  for t in range(32):
    tile_A = A[t * 16:t * 16 + 16].astype(np.float32)
    tile_B = B[t * 16:t * 16 + 16].astype(np.float32)
    acc = C[t * 8:t * 8 + 8]
    dot = np.dot(tile_A.reshape(1, 16), tile_B.reshape(16, 1)).flatten()[:8]  # Just approximate to 8 FP32 outs
    expected[t * 8:t * 8 + 8] = acc + dot

  assert np.allclose(out, expected, rtol=1e-2, atol=1e-2), f"Got {out}, expected {expected}"

if __name__ == "__main__":
  test_simple()
  test_scalar_add()
  test_matrix_transpose()
  test_uniform_conditional()
  test_wmma_matmul()
