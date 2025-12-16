from typing import Callable, cast
import struct, yaml
from collections import defaultdict
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace
from tinygrad.renderer import Renderer
from tinygrad.helpers import prod, get_single_element
from tinygrad.codegen.late.devectorizer import no_vectorized_alu

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    if dtype == dtypes.double: return "0x%016X" % struct.unpack("Q", struct.pack("d", x))[0]
    if dtype == dtypes.half: return "0x%04X" % struct.unpack("H", struct.pack("e", x))[0]
    return "0x%08X" % struct.unpack("I", struct.pack("f", x))[0]
  return str(int(x))

# RDNA3 uses different instruction names and formats than PTX
asm_for_op: dict[Ops, Callable] = {
  Ops.RECIPROCAL: lambda d,a,dt,name: f"v_rcp_{name} {d}, {a}",
  Ops.EXP2: lambda d,a,dt,name: f"v_exp_{name} {d}, {a}", Ops.LOG2: lambda d,a,dt,name: f"v_log_{name} {d}, {a}",
  Ops.SIN: lambda d,a,dt,name: f"v_sin_{name} {d}, {a}", Ops.SQRT: lambda d,a,dt,name: f"v_sqrt_{name} {d}, {a}",
  Ops.TRUNC: lambda d,a,dt,name: f"v_trunc_{name} {d}, {a}",
  Ops.SHR: lambda d,a,b,dt,name: f"v_lshrrev_b32 {d}, {b}, {a}",  # Note: operand order is reversed
  Ops.SHL: lambda d,a,b,dt,name: f"v_lshlrev_b32 {d}, {b}, {a}",  # Note: operand order is reversed
  Ops.ADD: lambda d,a,b,dt,name: f"v_add_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_add_nc_u32 {d}, {a}, {b}",
  Ops.SUB: lambda d,a,b,dt,name: f"v_sub_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_sub_nc_u32 {d}, {a}, {b}",
  Ops.MUL: lambda d,a,b,dt,name: f"v_mul_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_mul_lo_u32 {d}, {a}, {b}",
  Ops.XOR: lambda d,a,b,dt,name: f"v_xor_b32 {d}, {a}, {b}",
  Ops.AND: lambda d,a,b,dt,name: f"v_and_b32 {d}, {a}, {b}",
  Ops.OR: lambda d,a,b,dt,name: f"v_or_b32 {d}, {a}, {b}",
  Ops.MAX: lambda d,a,b,dt,name: f"v_max_{name} {d}, {a}, {b}",
  Ops.CMPEQ: lambda d,a,b,dt,name: f"v_cmp_eq_{name} {d}, {a}, {b}",
  Ops.CMPLT: lambda d,a,b,dt,name: f"v_cmp_lt_{name} {d}, {a}, {b}",
  # RDNA uses v_cmp_ne for integers, v_cmp_neq for floats
  Ops.CMPNE: lambda d,a,b,dt,name: f"v_cmp_neq_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_cmp_ne_{name} {d}, {a}, {b}",
  # v_mad_u32_u24 only handles 24-bit unsigned operands; for signed ints, use v_mad_i32_i24
  Ops.MULACC: lambda d,a,b,c,dt,name: f"v_fma_{name} {d}, {a}, {b}, {c}" if dtypes.is_float(dt) else f"v_mad_i32_i24 {d}, {a}, {b}, {c}",
  Ops.WHERE: lambda d,a,b,c,dt,name: f"v_cndmask_b32 {d}, {c}, {b}, {a}",
}

# Pattern matcher for RDNA3-specific rewrites
rdna_matcher = PatternMatcher([
  # cast void does nothing
  (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) or x.src[0].dtype == dtypes.void else None),
  # devectorize ALU operations - RDNA doesn't have vector float ALU
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name="alu"), no_vectorized_alu),
  # compute byte offset for INDEX operations at UOp level (like PTX)
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx")), name="op", allow_any_len=True), lambda buf, idx, op:
    UOp(Ops.INDEX, dtype=dtypes.int32, src=(buf, idx.cast(dtypes.int32)*buf.dtype.itemsize)+op.src[2:])
      if op.dtype != dtypes.int32 and isinstance(buf.dtype, PtrDType) and buf.dtype.addrspace != AddrSpace.REG else None),
])

def mem_type(x:UOp) -> str:
  match x.op:
    case Ops.AFTER: return mem_type(x.src[0])
    case Ops.DEFINE_LOCAL: return 'ds'
    case Ops.DEFINE_GLOBAL: return 'global'
    case _: raise RuntimeError(f"{x.op} needs to be memory")

def mem_size_suffix(dt:DType) -> str:
  """Get memory instruction suffix based on dtype size"""
  if dt.itemsize == 1: return "byte"
  if dt.itemsize == 2: return "short"
  if dt.itemsize == 4: return "dword"
  if dt.itemsize == 8: return "dwordx2"
  raise RuntimeError(f"Unsupported dtype size: {dt.itemsize}")

def global_store(addr:str, data:str, base:str, dt:DType) -> str:
  suffix = mem_size_suffix(dt)
  return f"global_store_{suffix} {addr}, {data}, {base}"

def global_load(dest:str, addr:str, base:str, dt:DType) -> str:
  suffix = mem_size_suffix(dt)
  if dt.itemsize == 1:
    return f"global_load_ubyte {dest}, {addr}, {base}"  # unsigned byte load
  return f"global_load_{suffix} {dest}, {addr}, {base}"

string_rewrite = PatternMatcher([
  # const rendering
  (UPat.cvar("x", dtypes.bool), lambda ctx, x: f"v_mov_b32 {ctx.r[x]}, {1 if x.arg else 0}"),
  (UPat.cvar("x"), lambda ctx, x: f"v_mov_b32 {ctx.r[x]}, {render_val(x.arg, x.dtype)}"),
  # special registers
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: ctx.render_special(x)),
  # define global
  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx, x: f"s_load_b64 {ctx.r[x]}, s[0:1], {x.arg*8}"),
  # comparison ops
  (UPat((Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ), name="x", allow_any_len=True, src=(UPat.var("src0"),)),
    lambda ctx, x, src0: ctx.code_for_op[x.op](ctx.r[x], *[ctx.r[v] for v in x.src], src0.dtype, ctx.types[src0.dtype])),
  # WHERE: if condition is SGPR, use directly; if VGPR, compare to 0 first to get VCC
  (UPat(Ops.WHERE, name="x", src=(UPat.var("cond"), UPat.var("true_val"), UPat.var("false_val"))),
   lambda ctx, x, cond, true_val, false_val: f"v_cndmask_b32 {ctx.r[x]}, {ctx.r[false_val]}, {ctx.r[true_val]}, {ctx.r[cond]}"
     if ctx.r[cond].startswith('s') else [
       f"v_cmp_ne_u32 vcc_lo, {ctx.r[cond]}, 0",
       f"v_cndmask_b32 {ctx.r[x]}, {ctx.r[false_val]}, {ctx.r[true_val]}, vcc_lo"]),
  # alu ops
  (UPat(GroupOp.ALU, name="x"), lambda ctx, x: ctx.code_for_op[x.op](ctx.r[x], *[ctx.r[v] for v in x.src], x.dtype, ctx.types[x.dtype])),
  # bitcast/cast
  (UPat(Ops.BITCAST, name="x", src=(UPat.var("a"),), allow_any_len=True), lambda ctx, x, a: f"v_mov_b32 {ctx.r[x]}, {ctx.r[a]}"),
  # cast from bool: if bool is in SGPR (comparison result), use v_cndmask_b32; if in VGPR (loaded from memory), convert directly
  (UPat(Ops.CAST, name="x", src=(UPat(dtype=dtypes.bool, name="a"),)),
   lambda ctx, x, a: f"v_cndmask_b32 {ctx.r[x]}, {render_val(0, x.dtype)}, {render_val(1, x.dtype)}, {ctx.r[a]}"
     if ctx.r[a].startswith('s') else f"v_cvt_f32_u32 {ctx.r[x]}, {ctx.r[a]}" if dtypes.is_float(x.dtype) else f"v_mov_b32 {ctx.r[x]}, {ctx.r[a]}"),
  # cast TO bool: compare to 0, result to vcc_lo, then cndmask to VGPR
  (UPat(Ops.CAST, name="x", dtype=dtypes.bool, src=(UPat.var("a"),)),
   lambda ctx, x, a: [
     f"v_cmp_{'neq' if dtypes.is_float(a.dtype) else 'ne'}_{ctx.types[a.dtype]} vcc_lo, {ctx.r[a]}, 0",
     f"v_cndmask_b32 {ctx.r[x]}, 0, 1, vcc_lo"]),
  (UPat(Ops.CAST, name="x", src=(UPat.var("a"),)), lambda ctx, x, a: ctx.render_cast(x, a)),
  # store / load for global memory
  # store boolean value - if SGPR (comparison result), convert via cndmask; if VGPR, store directly
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var", dtype=dtypes.bool))),
   lambda ctx, idx, var, buf, index_op: [
     f"v_cndmask_b32 v{ctx.get_scratch_vgpr()}, 0, 1, {ctx.r[var]}",
     f"global_store_byte {ctx.r[index_op]}, v{ctx.get_scratch_vgpr()}, {ctx.r[buf]}"]
       if ctx.r[var].startswith('s') else f"global_store_byte {ctx.r[index_op]}, {ctx.r[var]}, {ctx.r[buf]}"),
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var"))),
   lambda ctx, idx, var, buf, index_op: global_store(ctx.r[index_op], ctx.r[var], ctx.r[buf], buf.dtype.base)),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx"), UPat.var("gate")), name="index_op"), UPat.var("alt")), allow_any_len=True),
    lambda ctx, x, idx, alt, gate, buf, index_op: [
    f"v_mov_b32 {ctx.r[x]}, {ctx.r[alt]}",
    f"s_and_b32 vcc_lo, exec_lo, {ctx.r[gate]}",
    f"s_and_saveexec_b32 s{ctx.scratch_sgpr}, vcc_lo",
    global_load(ctx.r[x], ctx.r[index_op], ctx.r[buf], buf.dtype.base),
    f"s_mov_b32 exec_lo, s{ctx.scratch_sgpr}"]),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx")), name="index_op"),), allow_any_len=True),
    lambda ctx, x, idx, buf, index_op: global_load(ctx.r[x], ctx.r[index_op], ctx.r[buf], buf.dtype.base)),
  # store / load for local memory (LDS) - DEFINE_LOCAL directly
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_LOCAL), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var"))),
   lambda ctx, idx, var, index_op: f"ds_write_b32 {ctx.r[index_op]}, {ctx.r[var]}"),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_LOCAL), UPat.var("idx")), name="index_op"),), allow_any_len=True),
    lambda ctx, x, idx, index_op: f"ds_read_b32 {ctx.r[x]}, {ctx.r[index_op]}"),
  # store / load for local memory (LDS) - DEFINE_LOCAL wrapped in AFTER
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.AFTER, src=(UPat(Ops.DEFINE_LOCAL),), allow_any_len=True), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var"))),
   lambda ctx, idx, var, index_op: f"ds_write_b32 {ctx.r[index_op]}, {ctx.r[var]}"),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.AFTER, src=(UPat(Ops.DEFINE_LOCAL),), allow_any_len=True), UPat.var("idx")), name="index_op"),), allow_any_len=True),
    lambda ctx, x, idx, index_op: f"ds_read_b32 {ctx.r[x]}, {ctx.r[index_op]}"),
  # simple
  (UPat(Ops.DEFINE_REG, src=()), lambda ctx: []),
  (UPat(Ops.RANGE, name="r"), lambda ctx, r: [
    f"v_mov_b32 {ctx.r[r]}, -1",  # Start at -1, incremented to 0 on first iteration
    f"s_branch LOOP_END_{ctx.r[r][1:]}",
    f"LOOP_{ctx.r[r][1:]}:"]),
  (UPat(Ops.END, name="x", src=(UPat(), UPat(Ops.RANGE, name="r"))), lambda ctx, x, r: [
    f"LOOP_END_{ctx.r[r][1:]}:",
    f"v_add_nc_u32 {ctx.r[r]}, {ctx.r[r]}, 1",
    f"v_cmp_lt_i32 vcc_lo, {ctx.r[r]}, {ctx.r[r.src[0]]}",
    f"s_cbranch_vccnz LOOP_{ctx.r[r][1:]}"]),
  (UPat(Ops.DEFINE_LOCAL, name="x"),
   lambda ctx, x: []),  # local memory is handled differently in RDNA
  (UPat(Ops.IF, name="x"), lambda ctx, x: [
    f"s_and_b32 vcc_lo, exec_lo, {ctx.r[x.src[0]]}",
    f"s_and_saveexec_b32 s{ctx.scratch_sgpr}, vcc_lo",
    f"s_cbranch_execz IF_END_{ctx.uops.index(x)}"]),
  (UPat(Ops.ENDIF, name="x"), lambda ctx, x: [
    f"IF_END_{ctx.uops.index(x.src[0])}:",
    f"s_mov_b32 exec_lo, s{ctx.scratch_sgpr}"]),
  (UPat(Ops.BARRIER, name="x"), lambda ctx, x: "s_barrier"),
])

class RDNARenderer(Renderer):
  device = "AMD"
  suffix = "RDNA"
  supports_float4 = False  # RDNA assembly doesn't support vector ALU ops
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 1024)
  shared_max = 65536
  code_for_op = asm_for_op
  extra_matcher = rdna_matcher

  def __init__(self, arch:str="gfx1100"):
    self.arch = arch
    # gfx1100 = RDNA3, gfx1201 = RDNA4
    # wavefront size is 32 for RDNA3
    self.wave32 = True

  def __reduce__(self): return self.__class__, (self.arch,)

  types: dict[DType, str] = {
    dtypes.int8: "i32", dtypes.int16: "i32", dtypes.int32: "i32", dtypes.int64: "i64",
    dtypes.uint8: "u32", dtypes.uint16: "u32", dtypes.uint32: "u32", dtypes.uint64: "u64",
    dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "i32"
  }

  def is_local(self, index_op: UOp) -> bool:
    """Check if an INDEX operation is for local memory"""
    return isinstance(index_op.dtype, PtrDType) and index_op.dtype.addrspace == AddrSpace.LOCAL

  def render_special(self, x: UOp) -> str:
    # SPECIAL arg is like 'g0', 'g1', 'g2' for global dims or 'l0', 'l1', 'l2' for local dims
    dim = int(x.arg[-1])
    if x.arg[0] == 'g':
      # group id is in s2, s3, s4
      return f"v_mov_b32 {self.r[x]}, s{2+dim}"
    elif x.arg[0] == 'l':
      # local id: v0 contains packed xyz (v0.x = bits 0-9, v0.y = bits 10-19, v0.z = bits 20-29)
      if dim == 0:
        return f"v_and_b32 {self.r[x]}, 0x3ff, v0"
      elif dim == 1:
        return f"v_bfe_u32 {self.r[x]}, v0, 10, 10"
      elif dim == 2:
        return f"v_bfe_u32 {self.r[x]}, v0, 20, 10"
    else:
      # 'i' for global index = gid * local_size + lid - this is computed later
      return f"v_mov_b32 {self.r[x]}, 0"  # placeholder
    raise RuntimeError(f"Unknown special: {x.arg}")

  def render_cast(self, x: UOp, a: UOp) -> str:
    # RDNA3 cast instructions
    if x.dtype == dtypes.float32 and a.dtype == dtypes.int32:
      return f"v_cvt_f32_i32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.float32 and a.dtype == dtypes.uint32:
      return f"v_cvt_f32_u32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.int32 and a.dtype == dtypes.float32:
      return f"v_cvt_i32_f32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.uint32 and a.dtype == dtypes.float32:
      return f"v_cvt_u32_f32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.float16 and a.dtype == dtypes.float32:
      return f"v_cvt_f16_f32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.float32 and a.dtype == dtypes.float16:
      return f"v_cvt_f32_f16 {self.r[x]}, {self.r[a]}"
    # fallback: just move (same size types)
    return f"v_mov_b32 {self.r[x]}, {self.r[a]}"

  def render_kernel(self, kernel, function_name, bufs, v_cnt, s_cnt, uops) -> str:
    # Build metadata for kernel
    args = []
    for name, dtype in bufs:
      if name.startswith("data"):
        i = int(name[4:])
        args.append({'.address_space': 'global', '.name': f'buf_{i}', '.offset': i*8, '.size': 8,
                     '.type_name': 'void*', '.value_kind': 'global_buffer'})
      else:
        # variable
        args.append({'.name': name, '.offset': len(args)*8, '.size': 8, '.value_kind': 'by_value'})

    kernarg_size = (args[-1][".offset"] + args[-1][".size"]) if args else 0

    # Kernel descriptor
    kernel_desc = {
      '.amdhsa_group_segment_fixed_size': self.lds_size, '.amdhsa_private_segment_fixed_size': 0, '.amdhsa_kernarg_size': kernarg_size,
      '.amdhsa_next_free_vgpr': v_cnt,
      '.amdhsa_reserve_vcc': 0, '.amdhsa_reserve_xnack_mask': 0,
      '.amdhsa_next_free_sgpr': s_cnt,
      '.amdhsa_float_round_mode_32': 0, '.amdhsa_float_round_mode_16_64': 0,
      '.amdhsa_float_denorm_mode_32': 3, '.amdhsa_float_denorm_mode_16_64': 3,
      '.amdhsa_dx10_clamp': 1, '.amdhsa_ieee_mode': 1, '.amdhsa_fp16_overflow': 0,
      '.amdhsa_workgroup_processor_mode': 1, '.amdhsa_memory_ordered': 1, '.amdhsa_forward_progress': 0,
      '.amdhsa_enable_private_segment': 0,
      '.amdhsa_system_sgpr_workgroup_id_x': 1, '.amdhsa_system_sgpr_workgroup_id_y': 1, '.amdhsa_system_sgpr_workgroup_id_z': 1,
      '.amdhsa_system_sgpr_workgroup_info': 0, '.amdhsa_system_vgpr_workitem_id': 2,
      '.amdhsa_exception_fp_ieee_invalid_op': 0, '.amdhsa_exception_fp_denorm_src': 0,
      '.amdhsa_exception_fp_ieee_div_zero': 0, '.amdhsa_exception_fp_ieee_overflow': 0,
      '.amdhsa_exception_fp_ieee_underflow': 0, '.amdhsa_exception_fp_ieee_inexact': 0,
      '.amdhsa_exception_int_div_zero': 0, '.amdhsa_user_sgpr_dispatch_ptr': 0, '.amdhsa_user_sgpr_queue_ptr': 0,
      '.amdhsa_user_sgpr_kernarg_segment_ptr': 1, '.amdhsa_user_sgpr_dispatch_id': 0,
      '.amdhsa_user_sgpr_private_segment_size': 0, '.amdhsa_wavefront_size32': 1, '.amdhsa_uses_dynamic_stack': 0
    }

    metadata = {
      'amdhsa.kernels': [{
        '.args': args,
        '.group_segment_fixed_size': self.lds_size, '.kernarg_segment_align': 8, '.kernarg_segment_size': kernarg_size,
        '.language': 'OpenCL C', '.language_version': [1, 2], '.max_flat_workgroup_size': 256,
        '.name': function_name, '.private_segment_fixed_size': 0, '.sgpr_count': s_cnt, '.sgpr_spill_count': 0,
        '.symbol': f'{function_name}.kd', '.uses_dynamic_stack': False, '.vgpr_count': v_cnt, '.vgpr_spill_count': 0,
        '.wavefront_size': 32
      }],
      'amdhsa.target': f'amdgcn-amd-amdhsa--{self.arch}', 'amdhsa.version': [1, 2]
    }

    boilerplate_start = f"""
.rodata
.global {function_name}.kd
.type {function_name}.kd,STT_OBJECT
.align 0x10
.amdhsa_kernel {function_name}"""

    code_start = f""".end_amdhsa_kernel
.text
.global {function_name}
.type {function_name},@function
.p2align 8
{function_name}:
"""

    kernel_str = '\n'.join(kernel)
    # NOTE: .text must be first line for HIPCompiler to detect as assembly
    return ".text\n" + \
           f".global {function_name}\n" + \
           f".type {function_name},@function\n" + \
           ".p2align 8\n" + \
           f"{function_name}:\n" + \
           kernel_str + f"\n.size {function_name}, .-{function_name}\n" + \
           "\n.rodata\n" + \
           f".global {function_name}.kd\n" + \
           f".type {function_name}.kd,STT_OBJECT\n" + \
           ".align 0x10\n" + \
           f".amdhsa_kernel {function_name}\n" + \
           '\n'.join("%s %d" % x for x in kernel_desc.items()) + "\n" + \
           ".end_amdhsa_kernel\n" + \
           ".amdgpu_metadata\n" + yaml.dump(metadata) + ".end_amdgpu_metadata"

  def render(self, uops:list[UOp]) -> str:
    kernel:list[str] = []
    bufs = []

    r: dict[UOp, str|list[str]] = {}
    self.r = r
    self.uops = uops
    self.scratch_sgpr = 100  # scratch register for exec manipulation
    self.lds_size = 0  # track local memory (LDS) usage
    # Scratch VGPR will be allocated after we know how many VGPRs the kernel uses
    self.scratch_vgpr = -1  # will be set after register allocation

    # === LIVENESS ANALYSIS ===
    # Compute last use position for each UOp
    last_use: dict[UOp, int] = {}
    # Track which UOps alias to other UOps (share registers)
    aliases: dict[UOp, UOp] = {}

    # First pass: find all RANGE/END pairs to identify loop ranges
    # Map from RANGE position to END position
    loop_ranges: dict[int, int] = {}
    range_positions: dict[UOp, int] = {}
    for i, u in enumerate(uops):
      if u.op is Ops.RANGE:
        range_positions[u] = i
      if u.op is Ops.END and len(u.src) >= 2 and u.src[1].op is Ops.RANGE:
        range_op = u.src[1]
        if range_op in range_positions:
          loop_ranges[range_positions[range_op]] = i

    for i, u in enumerate(uops):
      for src in u.src:
        last_use[src] = i
      # For INDEX inside LOAD/STORE, track the INDEX sources too
      if u.op in {Ops.LOAD, Ops.STORE} and u.src[0].op is Ops.INDEX:
        for src in u.src[0].src:
          last_use[src] = i
      # For END, track RANGE.src[0] (loop bound) which is used in rendering
      if u.op is Ops.END and len(u.src) >= 2 and u.src[1].op is Ops.RANGE:
        range_op = u.src[1]
        if len(range_op.src) > 0:
          last_use[range_op.src[0]] = i
      # Track aliases - UOps that share registers
      if u.op is Ops.AFTER:
        aliases[u] = u.src[0]
      if u.op in {Ops.CAST, Ops.BITCAST} and (u.src[0].dtype == u.dtype or isinstance(u.src[0].dtype, PtrDType)):
        aliases[u] = u.src[0]
      # LOAD/STORE/INDEX on REG addrspace alias to the DEFINE_REG
      if u.op in {Ops.INDEX, Ops.LOAD, Ops.STORE} and len(u.src) > 0:
        if isinstance(u.src[0].dtype, PtrDType) and u.src[0].dtype.addrspace == AddrSpace.REG:
          if u.op is Ops.INDEX:
            aliases[u] = u.src[0]  # INDEX on REG aliases to DEFINE_REG
          elif u.op is Ops.LOAD:
            aliases[u] = u.src[0]  # LOAD from REG aliases to INDEX (which aliases to DEFINE_REG)

    # Second pass: extend last_use for values used inside loops
    # Values used inside a loop must stay alive until the END of the loop
    uop_positions = {u: i for i, u in enumerate(uops)}
    for uop, use_pos in list(last_use.items()):
      # Check if this use position is inside any loop
      for range_pos, end_pos in loop_ranges.items():
        if range_pos < use_pos <= end_pos:
          # This value is used inside a loop, extend its lifetime to loop END
          last_use[uop] = max(last_use[uop], end_pos)
      # Also check if the uop itself is defined inside a loop
      if uop in uop_positions:
        def_pos = uop_positions[uop]
        for range_pos, end_pos in loop_ranges.items():
          if range_pos < def_pos <= end_pos:
            # This value is defined inside a loop
            # If it's used inside the loop, extend to END
            if use_pos <= end_pos:
              last_use[uop] = max(last_use[uop], end_pos)

    # === REGISTER ALLOCATOR ===
    # Track free registers (available for reuse)
    free_vgprs: list[int] = []
    free_sgprs: list[int] = []
    # Track SGPR pairs to prevent individual registers from being freed
    sgpr_pairs: set[int] = set()
    # Track which UOp uses which register (for freeing)
    vgpr_owner: dict[int, UOp] = {}
    sgpr_owner: dict[int, UOp] = {}
    # v[0:2] is local_xyz, we start allocating from v3
    next_vgpr = 3
    # s[0:1] is kernarg ptr, s[2:4] is group id xyz, we start from s5
    next_sgpr = 5
    max_vgpr = 3
    max_sgpr = 5

    # Build reverse alias map: owner -> all UOps that alias to it (directly or transitively)
    def get_root_owner(u: UOp) -> UOp:
      while u in aliases:
        u = aliases[u]
      return u

    alias_groups: dict[UOp, list[UOp]] = defaultdict(list)
    for u in aliases:
      root = get_root_owner(u)
      alias_groups[root].append(u)

    def free_dead_regs(pos: int):
      """Free registers whose owners (and all aliases) are no longer live after position pos"""
      nonlocal free_vgprs, free_sgprs
      dead_vgprs = []
      for reg, owner in vgpr_owner.items():
        # Check if owner and all its aliases are dead
        owner_last_use = last_use.get(owner, -1)
        # Also check any UOps that alias to this owner
        for alias_uop in alias_groups.get(owner, []):
          owner_last_use = max(owner_last_use, last_use.get(alias_uop, -1))
        if owner_last_use < pos:
          dead_vgprs.append(reg)
      dead_sgprs = []
      for reg, owner in sgpr_owner.items():
        owner_last_use = last_use.get(owner, -1)
        for alias_uop in alias_groups.get(owner, []):
          owner_last_use = max(owner_last_use, last_use.get(alias_uop, -1))
        if owner_last_use < pos:
          dead_sgprs.append(reg)
      for reg in dead_vgprs:
        del vgpr_owner[reg]
        free_vgprs.append(reg)
      for reg in dead_sgprs:
        # Don't free SGPRs that are part of a pair (used for 64-bit values like buffer addresses)
        if reg in sgpr_pairs:
          continue
        del sgpr_owner[reg]
        free_sgprs.append(reg)

    def alloc_vgpr(owner: UOp) -> str:
      nonlocal next_vgpr, max_vgpr
      if free_vgprs:
        reg = free_vgprs.pop()
      else:
        reg = next_vgpr
        next_vgpr += 1
        max_vgpr = max(max_vgpr, next_vgpr)
      vgpr_owner[reg] = owner
      return f"v{reg}"

    def alloc_sgpr(owner: UOp) -> str:
      nonlocal next_sgpr, max_sgpr
      if free_sgprs:
        reg = free_sgprs.pop()
      else:
        reg = next_sgpr
        next_sgpr += 1
        max_sgpr = max(max_sgpr, next_sgpr)
      sgpr_owner[reg] = owner
      return f"s{reg}"

    def alloc_sgpr_pair(owner: UOp) -> str:
      """Allocate an aligned pair of SGPRs for 64-bit values"""
      nonlocal next_sgpr, max_sgpr
      # Align to even
      if next_sgpr % 2 != 0:
        next_sgpr += 1
      reg = next_sgpr
      next_sgpr += 2
      max_sgpr = max(max_sgpr, next_sgpr)
      sgpr_owner[reg] = owner
      sgpr_owner[reg+1] = owner
      # Mark these as part of a pair - never free individually
      sgpr_pairs.add(reg)
      sgpr_pairs.add(reg+1)
      return f"s[{reg}:{reg+1}]"

    def get_scratch_vgpr() -> int:
      """Get or allocate a scratch VGPR for temporary operations"""
      nonlocal next_vgpr, max_vgpr
      if self.scratch_vgpr < 0:
        # Allocate a new scratch VGPR (not tracked in owner map, stays allocated)
        self.scratch_vgpr = next_vgpr
        next_vgpr += 1
        max_vgpr = max(max_vgpr, next_vgpr)
      return self.scratch_vgpr

    # Make get_scratch_vgpr available to pattern matcher via self
    self.get_scratch_vgpr = get_scratch_vgpr

    name = "test"
    pending_waits = set()  # track which ops need waits

    for i, u in enumerate(uops):
      # Free registers that are no longer needed
      free_dead_regs(i)
      if u.op in {Ops.NOOP, Ops.GROUP}: continue
      if u.op is Ops.AFTER:
        r[u] = r[u.src[0]]
        continue
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
        continue
      if u.op is Ops.VECTORIZE:
        r[u] = [cast(str,r[x]) for x in u.src]
        continue
      if u.op is Ops.GEP:
        r[u] = r[u.src[0]][get_single_element(u.arg)]
        continue
      if u.op in {Ops.CAST, Ops.BITCAST} and (u.src[0].dtype == u.dtype or isinstance(u.src[0].dtype, PtrDType)):
        r[u] = r[u.src[0]]
        continue

      # Register allocation with liveness-based reuse
      if u.op is Ops.DEFINE_GLOBAL:
        r[u] = alloc_sgpr_pair(u)
        bufs.append((f"data{u.arg}", u.dtype))
      elif u.op is Ops.DEFINE_LOCAL:
        # Local memory - DEFINE_LOCAL.dtype contains the LDS size in the ptr size
        lds_size = u.dtype.size * u.dtype.itemsize if isinstance(u.dtype, PtrDType) else 0
        self.lds_size = max(getattr(self, 'lds_size', 0), lds_size)
        r[u] = u.arg  # Store the offset (arg is the offset in bytes)
        continue
      elif u.op is Ops.DEFINE_VAR:
        r[u] = alloc_sgpr(u)
        bufs.append((u.arg[0], u.dtype))
      elif u.op is Ops.SPECIAL:
        r[u] = alloc_vgpr(u)
      elif u.op is Ops.INDEX:
        r[u] = alloc_vgpr(u)
      elif u.op is Ops.LOAD:
        r[u] = alloc_vgpr(u)
        pending_waits.add(u)
      elif u.op is Ops.CONST:
        r[u] = alloc_vgpr(u)
      elif u.op is Ops.RANGE:
        r[u] = alloc_vgpr(u)
      elif u.op is Ops.END:
        r[u] = "vcc_lo"  # comparison result
      elif u.op in GroupOp.ALU or u.op in {Ops.CAST, Ops.BITCAST}:
        # Check if any source needs a wait
        for src in u.src:
          if src in pending_waits:
            kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
            pending_waits.clear()
            break
        # Only direct comparison ops go to SGPR; other bool ops stay in VGPR
        # because their inputs might be in VGPR (from memory loads)
        if u.dtype == dtypes.bool and u.op in {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}:
          r[u] = alloc_sgpr(u)  # comparison results go in SGPR
        else:
          r[u] = alloc_vgpr(u)
      elif u.op is Ops.IF:
        r[u] = alloc_sgpr(u)
      elif u.op is Ops.DEFINE_REG:
        r[u] = [alloc_vgpr(u) for _ in range(u.ptrdtype.size)]
        continue

      # Handle register-based INDEX/LOAD/STORE (accumulator spills)
      if u.op in {Ops.INDEX, Ops.LOAD, Ops.STORE} and isinstance(u.src[0].dtype, PtrDType) and u.src[0].dtype.addrspace == AddrSpace.REG:
        if u.op is Ops.INDEX:
          assert u.src[1].op == Ops.CONST, f"index on REG in rdna only supported on CONST, not {u.src[1].op}"
          r[u] = r[u.src[0]][u.src[1].arg]
        else:
          r[u] = r[u.src[0]]
          if u.op is Ops.STORE:
            kernel.append(f"v_mov_b32 {r[u.src[0]]}, {r[u.src[1]]}")
        continue

      # Skip INDEX as it's handled inline
      if u.op is Ops.INDEX:
        # The byte offset is already computed at UOp level by rdna_matcher
        # We just need to move it to the allocated register (and optionally add base offset for local)
        buf, idx = u.src[0], u.src[1]
        if len(u.src) > 2:  # gated
          pass  # handled in LOAD
        is_local = isinstance(u.dtype, PtrDType) and u.dtype.addrspace == AddrSpace.LOCAL
        base_offset = r[buf] if is_local else 0  # local memory has base offset from DEFINE_LOCAL

        if idx.op is Ops.CONST and idx.arg == 0:
          if is_local and base_offset:
            kernel.append(f"v_mov_b32 {r[u]}, {base_offset}")
          else:
            kernel.append(f"v_mov_b32 {r[u]}, 0")
        else:
          # Wait for idx if needed
          if idx in pending_waits:
            kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
            pending_waits.clear()
          # idx is already the byte offset (computed at UOp level by rdna_matcher)
          if is_local and base_offset:
            kernel.append(f"v_add_nc_u32 {r[u]}, {base_offset}, {r[idx]}")
          else:
            kernel.append(f"v_mov_b32 {r[u]}, {r[idx]}")
        continue

      # Check if STORE needs a wait for any pending loads
      if u.op is Ops.STORE:
        for src in u.src:
          if src in pending_waits or (src.op is Ops.INDEX and any(s in pending_waits for s in src.src)):
            kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
            pending_waits.clear()
            break

      # Render the instruction
      if (l:=cast(str|list[str], string_rewrite.rewrite(u, ctx=self))) is None:
        raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      if isinstance(l, str):
        kernel.append(l)
      else:
        kernel.extend(l)

      # Add wait after loads
      if u.op is Ops.DEFINE_GLOBAL:
        kernel.append("s_waitcnt lgkmcnt(0)")

    # Final waitcnt and end program - always wait to ensure stores complete
    kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
    kernel.extend(['s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)', 's_endpgm', 's_code_end'])

    return self.render_kernel(kernel, name, bufs, max_vgpr, max_sgpr, uops)
