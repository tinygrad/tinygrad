from typing import DefaultDict, Dict, List, Union, Optional, cast, Callable, Tuple
from functools import reduce
from operator import add,mul
import struct
import yaml

from tinygrad.renderer import Renderer
from tinygrad.ops import Ops, UOp, PatternMatcher, UPat, DType, GroupOp
from tinygrad.dtype import dtypes

def asm_emulated_mod(ctx, d, a, b, dt, name) -> List[str]:
  t0, t1, t2 = ctx.tmp_vregs()
  if dtypes.is_float(dt):
    return [
      f"v_rcp_{name} {t0}, {b}", # t0 = 1 / b
      f"v_mul_{name} {t1}, {a}, {t0}", # t1 = a / b
      f"v_floor_{name} {t1}, {t1}", # t1 = floor(a / b)
      f"v_mul_{name} {t2}, {t1}, {b}", # t2 = floor(a / b) * b
      f"v_sub_{name} {d}, {a}, {t2}" # d = a - floor(a / b) * b
    ]
  else:
    return [
      f"v_div_u32 {t0}, {a}, {b}", # t0 = a / b
      f"v_mul_lo_u32 {t1}, {t0}, {b}", # t1 = (a / b) * b
      f"v_sub_u32 {d}, {a}, {t1}" # a - (a / b) * b
    ]

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
  (Ops.CMPLT,): lambda ctx, d, a, b, dt, name: f"v_cmp_lt_{name} {a}, {b}",
  (Ops.CMPNE,): lambda ctx, d, a, b, dt, name: f"v_cmp_neq_{name} {a}, {b}",
  (Ops.MULACC,): lambda ctx, a, b, c, dt, name: (
    f"v_fmac_{name} {c}, {a}, {b}" if dtypes.is_float(dt) else f"v_mad_{name} {c}, {a}, {b}, {c}"),
  (Ops.WHERE,): lambda ctx, d, a, b, c, dt, name: render_where(ctx, d, a, b, c, dt, name),
}

def get_reg_range(reg: str) -> List[int]:
  if '[' in reg:
    a, *b = reg[reg.index('[')+1 : reg.index(']')].split(':')
    return [int(a)] if not b else list(range(int(a), int(b[0]) + 1))
  i = next(i for i, c in enumerate(reg) if c.isdigit())
  return [int(reg[i:])]

def get_regs_contained(reg: str) -> List[int]:
  if reg == 'vcc':
    return reg
  if '[' in reg:
    a, *b = reg[reg.index('[')+1 : reg.index(']')].split(':')
    return [f"{reg[0]}{a}"] if not b else [f"{reg[0]}{i}" for i in range(int(a), int(b[0]) + 1)]
  return [reg]

def render_reg_range(p: str, regs: List[str]) -> str:
  r = list(sorted(reduce(add, map(get_reg_range, regs))))
  return f"{p}{r[0]}" if len(r) == 1 else f"{p}[{r[0]}:{r[-1]}]"

def render_addr_calc(ctx, ptr, offset) -> Tuple[List[str], str, str]:
  instructions = []
  ptr = ctx.r[ptr]
  offset = ctx.r[offset]

  ptr_lo, ptr_hi = get_regs_contained(ptr)
  addr_lo_v, addr_hi_v, temp_v = ctx.tmp_vregs()

  instructions.append("s_mov_b64 vcc, 0")

  if offset[0] == 's':
    if ptr_lo[0] == 's': # S+S -> need offset in VGPR for S1
      instructions.append(f"v_mov_b32 {temp_v}, {offset}")
      instructions.append(f"v_add_co_ci_u32 {addr_lo_v}, {ptr_lo}, {temp_v}") # S0=SGPR, S1=VGPR
    else: # V+S -> swap operands
      instructions.append(f"v_add_co_ci_u32 {addr_lo_v}, {offset}, {ptr_lo}")   # S0=SGPR, S1=VGPR
  else: # offset is VGPR
    # S+V or V+V -> both OK as is, since offset (S1) is VGPR
    instructions.append(f"v_add_co_ci_u32 {addr_lo_v}, {ptr_lo}, {offset}") # S0=any, S1=VGPR

  # High 32-bit add: addr_hi_v = ptr_hi + 0 + carry

  # S1 must be VGPR. If ptr_hi is SGPR, it needs to be moved.
  if ptr_hi[0] == 's':
    instructions.append(f"v_mov_b32 {temp_v}, {ptr_hi}")
    instructions.append(f"v_add_co_ci_u32 {addr_hi_v}, 0, {temp_v}") # S0=imm, S1=VGPR
  else: # ptr_hi is VGPR
    instructions.append(f"v_add_co_ci_u32 {addr_hi_v}, 0, {ptr_hi}") # S0=imm, S1=VGPR

  return instructions, addr_lo_v, addr_hi_v

def render_load(ctx, x, ptr, offset) -> List[str]:
  load_bits = ctx.types[x.dtype][1:] * x.dtype.count
  addr_setup_ins, addr_lo_v, addr_hi_v = render_addr_calc(ctx, ptr, offset)
  if any(_x.op is Ops.DEFINE_LOCAL for _x in x.src[0].toposort):
    # shared mem
    load_ins = f"ds_load_b{load_bits} {ctx.r[x]}, {render_reg_range('v', [addr_lo_v, addr_hi_v])}"
  else:
    load_ins = f"flat_load_b{load_bits} {ctx.r[x]}, {render_reg_range('v', [addr_lo_v, addr_hi_v])} offset:0"
  return addr_setup_ins + [load_ins]

def render_store(ctx, x, ptr, offset, val) -> List[str]:
  store_bits = ctx.types[x.dtype][1:] * x.dtype.count
  addr_setup_ins, addr_lo_v, addr_hi_v = render_addr_calc(ctx, ptr, offset)
  if any(_x.op is Ops.DEFINE_LOCAL for _x in x.src[0].toposort):
    # shared mem
    store_ins = f"ds_store_b{store_bits} {render_reg_range('v', [addr_lo_v, addr_hi_v])}, {ctx.r[val]}"
  else:
    store_ins = f"flat_store_b{store_bits} {render_reg_range('v', [addr_lo_v, addr_hi_v])}, {ctx.r[val]} offset:0"
  return addr_setup_ins + [store_ins]

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

  (UPat(Ops.WHERE, name="x"),
    lambda ctx, x: asm_for_op[(Ops.WHERE,)](ctx, ctx.r[x], ctx.r[x.src[0]], ctx.r[x.src[1]], ctx.r[x.src[2]], x.dtype, ctx.types[x.dtype])),

  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx, x: [f"s_load_b64 {ctx.r[x]}, s[0:1], {x.arg * 8}", "s_waitcnt lgkmcnt(0)"]),

  (UPat(Ops.MOD, name="x"),
    lambda ctx, x: asm_for_op[(Ops.MOD,)](ctx, ctx.r[x], ctx.r[x.src[0]], ctx.r[x.src[1]], x.dtype, ctx.types[x.dtype])),

  (UPat(Ops.LOAD, name="x", src=(UPat.var("ptr"), UPat.var("offset"))), lambda ctx, x, ptr, offset:
    render_load(ctx, x, ptr, offset)
  ),

  (UPat(Ops.STORE, name="x", src=(UPat.var("ptr"), UPat.var("offset"), UPat.var("val"))),
   lambda ctx, x, ptr, offset, val: render_store(ctx, x, ptr, offset, val)),

  (UPat(Ops.ASSIGN, name="x"), 
   lambda ctx, x: f"v_mov_b{ctx.types[x.dtype][1:] * x.dtype.count} {ctx.r[x.src[0]]}, {ctx.r[x.src[1]]}"),

  (UPat(Ops.RANGE, name="x"),
   lambda ctx, x: [f"v_mov_b32 {ctx.r[x]}, {ctx.r[x.src[0]]}", f"LABEL_LOOP_{ctx.r[x][1:]}:"]),

  (UPat(Ops.ENDRANGE, name="x", src=(UPat.var("src0"),)), 
   lambda ctx, x, src0: [
       asm_for_op[(Ops.ADD,)](ctx, ctx.r[src0], ctx.r[src0], "1", dtypes.int, ctx.types[dtypes.int]),
       asm_for_op[(Ops.CMPLT,)](ctx, ctx.r[src0], ctx.r[src0], ctx.l[src0], dtypes.int, ctx.types[dtypes.int]),
       f"s_cbranch_vccnz LABEL_LOOP_{ctx.r[src0][1:]}"
   ]),

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
#         src=(UPat(Ops.MUL, name='a1'), UPat.var("a2"))), lambda: ctx, x, y: ),
# ])

class RDNA3Renderer(Renderer):
  device = "ROCm"
  # tensor_cores = [tc for tc in CUDARenderer.tensor_cores if tc.dtype_in == dtypes.half]
  extra_matcher = None

  def __init__(self, arch:str, device="ROCm"):
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

  def tmp_vregs(self):
    return self.tmpv
  
  def render(self, name:str, uops:List[UOp]) -> str:
    self.tmpv = [f"v{i}" for i in range(3, 6)]
    self.v_cnt = 6  # v[0:2] is local_xyz, v[3:5] are used for temporaries
    self.s_cnt = 5  # s[0:1] is the address, s[2:4] is global_xyz

    self.r: Dict[UOp, str] = {}
    ins: List[str] = []
    args: List[dict] = []
    self.l: Dict[UOp, int] = {}

    for u in uops:
      if any(src.op is Ops.LOAD for src in u.src):
        ins.append("  s_waitcnt vmcnt(0)")

      if u.op in {Ops.CMPLT, Ops.CMPNE}:
        # HACK: this will not be valid if we do another comparison in the way.
        # saving vcc state isn't fun. this might be invalid because we also do
        # operations on these predicate registers so I'm not sure what to do.
        self.r[u] = "vcc"
      elif u.op == Ops.MULACC:
        self.r[u] = self.r[u.src[0]]
      elif u.op in GroupOp.ALU or u.op == Ops.CONST or u.op == Ops.LOAD or u.op == Ops.RANGE:
        self.r[u] = f"v{self.v_cnt}"
        self.v_cnt += 1
      elif u.op == Ops.SPECIAL:
        if u.arg[1].startswith("lidx"):
          self.r[u] = f'v{u.arg[0]}'
        elif u.arg[1].startswith("gidx"):
          self.r[u] = f's{2+u.arg[0]}'
        else:
          raise NotImplementedError
        continue
      elif u.op == Ops.DEFINE_GLOBAL:
        size = u.dtype.count * u.dtype.itemsize
        i = u.arg
        args.append({'.address_space': 'global', '.name': f'buf_{u.arg}', '.offset': i * 8, '.size': size,
                     '.type_name': u.dtype.name+"*", '.value_kind': 'by_value'})
        self.s_cnt += self.s_cnt%2
        self.r[u] = f"s[{self.s_cnt}:{self.s_cnt+1}]"
        self.s_cnt += 2
      if u.op == Ops.RANGE:
        # TODO: what is the arg?
        self.l[u] = self.r[u.src[1]]

      if (l:=cast(Union[str, List[str]], rdna3_rewrite.rewrite(u, ctx=self))) is None:
        raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      l = [l] if isinstance(l, str) else l
      l = [f"  {u}" if u[-1] != ':' else u for u in l]
      ins.extend(l)

    ins.append("  s_waitcnt vmcnt(0)")
    metadata = {
      'amdhsa.kernels': [{'.args': args,
        '.group_segment_fixed_size': 0, '.kernarg_segment_align': 8, '.kernarg_segment_size': args[-1][".offset"] + args[-1][".size"],
        '.language': 'OpenCL C', '.language_version': [1, 2], '.max_flat_workgroup_size': 256,
        '.name': name, '.private_segment_fixed_size': 0, '.sgpr_count': self.s_cnt, '.sgpr_spill_count': 0,
        '.symbol': f'{name}.kd', '.uses_dynamic_stack': False, '.vgpr_count': self.v_cnt, '.vgpr_spill_count': 0,
        '.wavefront_size': 32}],
      'amdhsa.target': 'amdgcn-amd-amdhsa--gfx1100', 'amdhsa.version': [1, 2]}

    boilerplate_start = f"""
.rodata
.global {name}.kd
.type {name}.kd,STT_OBJECT
.align 0x10
.amdhsa_kernel {name}"""

    kernel_desc = {
      '.amdhsa_group_segment_fixed_size': 0, '.amdhsa_private_segment_fixed_size': 0, '.amdhsa_kernarg_size': 0,
      '.amdhsa_next_free_vgpr': self.v_cnt,
      '.amdhsa_reserve_vcc': 0, '.amdhsa_reserve_xnack_mask': 0,
      '.amdhsa_next_free_sgpr': self.s_cnt,
      '.amdhsa_float_round_mode_32': 0, '.amdhsa_float_round_mode_16_64': 0, '.amdhsa_float_denorm_mode_32': 3, '.amdhsa_float_denorm_mode_16_64': 3,
      '.amdhsa_dx10_clamp': 1, '.amdhsa_ieee_mode': 1, '.amdhsa_fp16_overflow': 0,
      '.amdhsa_workgroup_processor_mode': 1, '.amdhsa_memory_ordered': 1, '.amdhsa_forward_progress': 0, '.amdhsa_enable_private_segment': 0,
      '.amdhsa_system_sgpr_workgroup_id_x': 1, '.amdhsa_system_sgpr_workgroup_id_y': 1, '.amdhsa_system_sgpr_workgroup_id_z': 1,
      '.amdhsa_system_sgpr_workgroup_info': 0, '.amdhsa_system_vgpr_workitem_id': 2,
      '.amdhsa_exception_fp_ieee_invalid_op': 0, '.amdhsa_exception_fp_denorm_src': 0,
      '.amdhsa_exception_fp_ieee_div_zero': 0, '.amdhsa_exception_fp_ieee_overflow': 0, '.amdhsa_exception_fp_ieee_underflow': 0,
      '.amdhsa_exception_fp_ieee_inexact': 0, '.amdhsa_exception_int_div_zero': 0,
      '.amdhsa_user_sgpr_dispatch_ptr': 0, '.amdhsa_user_sgpr_queue_ptr': 0, '.amdhsa_user_sgpr_kernarg_segment_ptr': 1,
      '.amdhsa_user_sgpr_dispatch_id': 0, '.amdhsa_user_sgpr_private_segment_size': 0, '.amdhsa_wavefront_size32': 1, '.amdhsa_uses_dynamic_stack': 0}

    code_start = f""".end_amdhsa_kernel
.text
.global {name}
.type {name},@function
.p2align 8
{name}:
"""

    ins += ['  s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)', '  s_endpgm', '  s_code_end']
    return ".amdgpu_metadata\n" + yaml.dump(metadata) + ".end_amdgpu_metadata" + \
           boilerplate_start + "\n" + '\n'.join("%s %d" % x for x in kernel_desc.items()) + "\n" + code_start + \
           '\n'.join(ins) + f"\n.size {name}, .-{name}"

# def create_rdna3_matmul_uops(K=4, N=128):
#     uops = []
#     
#     ## Global Buffers
#     gA = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 0)  # Input A (MxK)
#     gB = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 1)  # Input B (KxN)
#     gC = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 2)  # Output C (MxN)
#     uops += [gA, gB, gC]
#
#     ## Constants
#     cK = UOp(Ops.CONST, dtypes.int32, (), K)
#     cN = UOp(Ops.CONST, dtypes.int32, (), N)
#     c0_f = UOp(Ops.CONST, dtypes.float32, (), 0.0)
#     uops += [cK, cN, c0_f]
#
#     ## Thread Coordinates
#     i = UOp(Ops.SPECIAL, dtypes.int32, (), (0, "lidx0"))  # Row index
#     j = UOp(Ops.SPECIAL, dtypes.int32, (), (1, "lidx1"))  # Column index
#     uops += [i, j]
#
#     acc = c0_f
#     uops.append(acc)
#
#     for k in range(K):
#         a_offset = UOp(Ops.MUL, dtypes.int32, (i, cK))
#         c = UOp(Ops.CONST, dtypes.int32, (), k)
#         a_idx = UOp(Ops.ADD, dtypes.int32, (a_offset, c))
#         a_val = UOp(Ops.LOAD, dtypes.float32, (gA, a_idx))
#
#         b_offset = UOp(Ops.MUL, dtypes.int32, (c, cN))
#         b_idx = UOp(Ops.ADD, dtypes.int32, (b_offset, j))
#         b_val = UOp(Ops.LOAD, dtypes.float32, (gB, b_idx))
#         
#         ac = UOp(Ops.MULACC, dtypes.float32, (a_val, b_val, acc))
#         
#         uops += [c, a_offset, a_idx, a_val, b_offset, b_idx, b_val, ac]
#
#     m = UOp(Ops.MUL, dtypes.int32, (i, cN))
#     output_idx = UOp(Ops.ADD, dtypes.int32, (m, j))
#     store = UOp(Ops.STORE, dtypes.float32, (gC, output_idx, acc))
#     uops += [m, output_idx, store]
#
#     return uops
def create_rdna3_matmul_uops(K=4, N=128):
    uops = []

    ## Global Buffers
    gA = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 0)  # Input A (MxK)
    gB = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 1)  # Input B (KxN)
    gC = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), (), 2)  # Output C (MxN)
    uops += [gA, gB, gC]

    ## Constants
    cK = UOp(Ops.CONST, dtypes.uint32, (), K)  # Loop bound K
    cN = UOp(Ops.CONST, dtypes.uint32, (), N)  # Matrix width N
    acc = UOp(Ops.CONST, dtypes.float32, (), 0.0)  # Zero for accumulator
    c0 = UOp(Ops.CONST, dtypes.uint32, (), 0)  # Loop start
    uops += [cK, cN, acc, c0]

    ## Thread Coordinates
    i = UOp(Ops.SPECIAL, dtypes.uint32, (), (0, "lidx0"))  # Row index
    j = UOp(Ops.SPECIAL, dtypes.uint32, (), (1, "lidx1"))  # Column index
    uops += [i, j]

    ## Loop Variable
    k = UOp(Ops.RANGE, dtypes.uint32, (c0, cK))  # k ranges from 0 to K
    uops.append(k)

    ## Loop Body
    # A[i][k]
    a_offset = UOp(Ops.MUL, dtypes.uint32, (i, cK))  # i * K
    a_idx = UOp(Ops.ADD, dtypes.uint32, (a_offset, k))  # i * K + k
    a_val = UOp(Ops.LOAD, dtypes.float32, (gA, a_idx))  # A[i][k]

    # B[k][j]
    b_offset = UOp(Ops.MUL, dtypes.uint32, (k, cN))  # k * N
    b_idx = UOp(Ops.ADD, dtypes.uint32, (b_offset, j))  # k * N + j
    b_val = UOp(Ops.LOAD, dtypes.float32, (gB, b_idx))  # B[k][j]

    # Accumulate: acc += A[i][k] * B[k][j]
    ac = UOp(Ops.MULACC, dtypes.float32, (a_val, b_val, acc))
    uops += [a_offset, a_idx, a_val, b_offset, b_idx, b_val, ac]

    ## End Loop
    end_k = UOp(Ops.ENDRANGE, dtypes.bool, (k,))  # Loop k until k < K
    uops.append(end_k)

    ## Store Result
    m = UOp(Ops.MUL, dtypes.uint32, (i, cN))  # i * N
    output_idx = UOp(Ops.ADD, dtypes.uint32, (m, j))  # i * N + j
    store = UOp(Ops.STORE, dtypes.float32, (gC, output_idx, acc))
    uops += [m, output_idx, store]

    return uops

if __name__ == '__main__':
  renderer = RDNA3Renderer("motherfucker")
  print(renderer.render("bitch", create_rdna3_matmul_uops()))
