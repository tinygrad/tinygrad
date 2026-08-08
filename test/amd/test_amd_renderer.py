import itertools, math, struct, unittest
from dataclasses import replace

from tinygrad import Tensor
from tinygrad.codegen import Estimates, full_rewrite_to_sink, line_rewrite, linearize, pm_linearize_cleanups, to_program, to_program_cache
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext, pm_regalloc_rewrite
from tinygrad.codegen.opt import KernelOptError, Opt, OptOps
from tinygrad.device import CompileError, Device
from tinygrad.dtype import AddrSpace, Invalid, dtypes
from tinygrad.helpers import Context, Target, getenv
from tinygrad.renderer.isa import IselContext, PreRegAllocContext, Register, greg
from tinygrad.runtime.autogen import amdgpu_kd
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType, KernelInfo, UOp, graph_rewrite
import tinygrad.renderer.isa.amd as amd_lib
from tinygrad.renderer.isa.amd import AMDRenderer, AMDOps

_GFX11 = Target("AMD", arch="gfx1100")
_REN = AMDRenderer(_GFX11)

def _to_prg(x):
  to_program_cache.clear()
  return to_program(x, _REN)

def _lin_ops(prg): return [u.arg for u in _prg_lin(prg).src if u.op is Ops.INS]

def _check_elf(tc, prg): tc.assertTrue(_prg_bin(prg).arg.startswith(b"\x7fELF"))

def _check_asm(tc, prg, *ops, insts=(), no_ops=()):
  _check_elf(tc, prg)
  los = _lin_ops(prg)
  for op in ops: tc.assertIn(op, los)
  for op in no_ops: tc.assertNotIn(op, los)
  if insts:
    names = _amd_inst_names(prg)
    for name in insts: tc.assertIn(name, names)

class TinyVGPRAMDRenderer(AMDRenderer):
  isel_matcher = amd_lib.make_isel_matcher(amd_lib.SGPR, amd_lib.VGPR[:2])

class OneVGPRAMDRenderer(AMDRenderer):
  isel_matcher = amd_lib.make_isel_matcher(amd_lib.SGPR, amd_lib.VGPR[:1])

class FourVGPRAMDRenderer(AMDRenderer):
  isel_matcher = amd_lib.make_isel_matcher(amd_lib.SGPR, amd_lib.VGPR[:4])

def _prg_lin(prg): return prg.src[1]
def _prg_src(prg): return prg.src[2]
def _prg_bin(prg): return prg.src[3]

def _amd_desc(prg):
  _, sections, _ = elf_loader(_prg_bin(prg).arg)
  return amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(next(s.content for s in sections if s.name == ".rodata"))

def _amd_inst_names(prg):
  return [getattr(i, "op_name", "") for i in _REN._insts_from_linear(_prg_lin(prg))]

def _assert_abi_reg_isolation(testcase, prg):
  # v0 = packed work-item IDs. WGID at s2 when USER_SGPR=2; s15 when gfx1100 pads for lidx1/2.
  need_yi = any(u.op is Ops.SPECIAL and u.arg.startswith("lidx") and u.arg[-1] in "12"
                for u in prg.src[0].toposort())
  wgid = {15, 16, 17} if need_yi else {2, 3, 4}
  fixed_sgpr, packed_lidx = {0, 1} | wgid, {256}
  for u in _prg_lin(prg).src:
    if not isinstance((reg_uop:=greg(u)), Register): continue
    reg = reg_uop.index
    if u.op is Ops.INS and u.arg is AMDOps.MOV and u.src and u.src[0].op is Ops.SPECIAL:
      if u.src[0].arg.startswith("gidx"): testcase.assertIn(reg, wgid)
      else: testcase.assertNotIn(reg, packed_lidx | fixed_sgpr)  # lidx BFE dest is a normal VGPR
    else:
      testcase.assertNotIn(reg, fixed_sgpr | packed_lidx)
      if reg < 256:
        testcase.assertGreaterEqual(reg, 6)
        testcase.assertEqual(reg % 2, 0)
      else:
        testcase.assertGreaterEqual(reg, 259)
        testcase.assertLess(reg, 256 + 255)

def _simple_add_program():
  out = UOp.placeholder((16,), dtypes.float, 0)
  inp = UOp.placeholder((16,), dtypes.float, 1)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load() + UOp.const(1.0, dtypes.float)).sink(idx, arg=KernelInfo(name="amd_asm_add"))
  return _to_prg(sink)

def _two_load_add_program():
  out = UOp.placeholder((16,), dtypes.float, 0)
  a = UOp.placeholder((16,), dtypes.float, 1)
  b = UOp.placeholder((16,), dtypes.float, 2)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(a.index(idx).load() + b.index(idx).load()).sink(idx, arg=KernelInfo(name="amd_asm_two_load_add"))
  return _to_prg(sink)

def _matmul64_program():
  with Context(BEAM=0):
    ast = (Tensor.empty(64, 64, device="AMD") @ Tensor.empty(64, 64, device="AMD")).schedule_linear().src[-1].src[0]
  return _to_prg(ast)

def _half_matmul_wmma_program():
  with Context(BEAM=0):
    ast = (Tensor.empty(64, 64, dtype=dtypes.half, device="AMD") @ Tensor.empty(64, 64, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
    ast = ast.schedule_linear().src[-1].src[0]
  from test.backend.test_linearizer import replace_opts
  ast = replace_opts(ast, [Opt(OptOps.TC, 0, (-1, 0, 1))])
  return _to_prg(ast)

def _half_matmul_tc_lds_ab_program(N=256):
  """Default TC tile + LDS A/B staging (ISA default; force TC_LDS_AB=1). Caller clears caches."""
  with Context(BEAM=0):
    ast = (Tensor.empty(N, N, dtype=dtypes.half, device="AMD") @ Tensor.empty(N, N, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
    return _to_prg(ast.schedule_linear().src[-1].src[0])

def _float4_add_program():
  with Context(BEAM=0):
    ast = (Tensor.empty(2, 8, device="AMD") + Tensor.empty(2, 8, device="AMD")).schedule_linear().src[0].src[0]
  return _to_prg(ast)

def _float4_lds_program():
  with Context(BEAM=0):
    ast = (Tensor.empty(1, 64, device="AMD").contiguous() @ Tensor.empty(64, 64, device="AMD").contiguous()).schedule_linear().src[0].src[0]
  return _to_prg(ast)

def _half_add_program():
  with Context(BEAM=0):
    ast = (Tensor.empty(2, 8, device="AMD", dtype=dtypes.half) + Tensor.empty(2, 8, device="AMD", dtype=dtypes.half)).schedule_linear().src[0].src[0]
  return _to_prg(ast)

def _padded_load_program():
  with Context(BEAM=0, DEV="MOCKKFD+AMD:AMD"):
    ast = Tensor.empty(3).pad((0, 1)).contiguous().schedule_linear().src[0].src[0]
  return _to_prg(ast)

def _uint_var_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  var = UOp.param(2, dtypes.uint32, (), vmin_vmax=(0, 10), name="var", addrspace=AddrSpace.ALU)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load() + var).sink(idx, var, arg=KernelInfo(name="amd_asm_uint_var"))
  return _to_prg(sink)

def _uint_var_mul_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  var = UOp.param(2, dtypes.uint32, (), vmin_vmax=(0, 10), name="var", addrspace=AddrSpace.ALU)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load() * var).sink(idx, var, arg=KernelInfo(name="amd_asm_uint_var_mul"))
  return _to_prg(sink)

def _two_uint_var_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  var0 = UOp.param(2, dtypes.uint32, (), vmin_vmax=(0, 10), name="var0", addrspace=AddrSpace.ALU)
  var1 = UOp.param(3, dtypes.uint32, (), vmin_vmax=(0, 10), name="var1", addrspace=AddrSpace.ALU)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load() + var0 + var1).sink(idx, var0, var1, arg=KernelInfo(name="amd_asm_two_uint_var"))
  return _to_prg(sink)

def _uint_wrap_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load() + UOp.const(0xffffffff, dtypes.uint32)).sink(idx, arg=KernelInfo(name="amd_asm_uint_wrap"))
  return _to_prg(sink)

def _copy_program(dtype):
  out = UOp.placeholder((16,), dtype, 0)
  inp = UOp.placeholder((16,), dtype, 1)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load()).sink(idx, arg=KernelInfo(name=f"amd_asm_copy_{dtype.name}"))
  return _to_prg(sink)

def _where_program(dtype):
  out = UOp.placeholder((16,), dtype, 0)
  inp = UOp.placeholder((16,), dtype, 1)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store((inp.index(idx).load() < UOp.const(7, dtype)).where(inp.index(idx).load(), UOp.const(0, dtype))) \
            .sink(idx, arg=KernelInfo(name=f"amd_asm_where_{dtype.name}"))
  return _to_prg(sink)

def _where_sgpr_true_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  idx = UOp.special(16, "lidx0")
  gidx = UOp.special(16, "gidx0").cast(dtypes.uint32)
  val = (idx < 8).where(gidx, UOp.const(0, dtypes.uint32))
  sink = out.index(idx).store(val).sink(idx, gidx, arg=KernelInfo(name="amd_asm_where_sgpr_true"))
  return _to_prg(sink)

def _where_compare_value_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  val = inp.index(idx).load()
  mask = (idx < 8).where(val == UOp.const(3, dtypes.uint32), UOp.const(False, dtypes.bool))
  sink = out.index(idx).store(mask.where(UOp.const(1, dtypes.uint32), UOp.const(0, dtypes.uint32))) \
            .sink(idx, arg=KernelInfo(name="amd_asm_where_compare_value"))
  return _to_prg(sink)

def _eq_where_program(dtype):
  out = UOp.placeholder((16,), dtype, 0)
  inp = UOp.placeholder((16,), dtype, 1)
  idx = UOp.special(16, "lidx0")
  val = inp.index(idx).load()
  sink = out.index(idx).store(UOp(Ops.CMPEQ, dtypes.bool, (val, UOp.const(7, dtype))).where(val, UOp.const(0, dtype))) \
            .sink(idx, arg=KernelInfo(name=f"amd_asm_eq_where_{dtype.name}"))
  return _to_prg(sink)

def _float_cmpne_program():
  out = UOp.placeholder((4,), dtypes.uint32, 0)
  inp = UOp.placeholder((4,), dtypes.float32, 1)
  idx = UOp.special(4, "lidx0")
  val = inp.index(idx).load()
  mask = UOp(Ops.CMPNE, dtypes.bool, (val, val))
  sink = out.index(idx).store(mask.where(UOp.const(1, dtypes.uint32), UOp.const(0, dtypes.uint32))) \
            .sink(idx, arg=KernelInfo(name="amd_asm_float_cmpne"))
  return _to_prg(sink)

def _cmpne_compare_flag_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  mask = (inp.index(idx).load() < UOp.const(10, dtypes.uint32)).ne(False)
  sink = out.index(idx).store(mask.where(UOp.const(1, dtypes.uint32), UOp.const(0, dtypes.uint32))) \
            .sink(idx, arg=KernelInfo(name="amd_asm_cmpne_compare_flag"))
  return _to_prg(sink)

def _bool_and_compare_flags_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  val = inp.index(idx).load()
  mask = val.ne(UOp.const(0, dtypes.uint32)) & (val < UOp.const(10, dtypes.uint32))
  sink = out.index(idx).store(mask.where(UOp.const(1, dtypes.uint32), UOp.const(0, dtypes.uint32))) \
            .sink(idx, arg=KernelInfo(name="amd_asm_bool_and_compare_flags"))
  return _to_prg(sink)

def _bool_compare_store_program():
  out = UOp.placeholder((4,), dtypes.bool, 0)
  inp = UOp.placeholder((4,), dtypes.float32, 1)
  idx = UOp.special(4, "lidx0")
  val = inp.index(idx).load()
  sink = out.index(idx).store(val < UOp.const(2.0, dtypes.float32)).sink(idx, arg=KernelInfo(name="amd_asm_bool_compare_store"))
  return _to_prg(sink)

def _float16_where_program():
  out = UOp.placeholder((4,), dtypes.float16, 0)
  inp0 = UOp.placeholder((4,), dtypes.float16, 1)
  inp1 = UOp.placeholder((4,), dtypes.float16, 2)
  mask = UOp.placeholder((4,), dtypes.bool, 3)
  idx = UOp.special(4, "lidx0")
  val = mask.index(idx).load().where(inp0.index(idx).load(), inp1.index(idx).load())
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_float16_where"))
  return _to_prg(sink)

def _float16_cast_program():
  out32 = UOp.placeholder((4,), dtypes.float32, 0)
  out16 = UOp.placeholder((4,), dtypes.float16, 1)
  inp16 = UOp.placeholder((4,), dtypes.float16, 2)
  inp32 = UOp.placeholder((4,), dtypes.float32, 3)
  idx = UOp.special(4, "lidx0")
  st32 = out32.index(idx).store(inp16.index(idx).load().cast(dtypes.float32))
  st16 = out16.index(idx).store(inp32.index(idx).load().cast(dtypes.float16))
  sink = UOp.sink(st32, st16, idx, arg=KernelInfo(name="amd_asm_float16_cast"))
  return _to_prg(sink)

def _int_to_half_cast_program():
  out16 = UOp.placeholder((4,), dtypes.float16, 0)
  out32 = UOp.placeholder((4,), dtypes.int32, 1)
  inp32 = UOp.placeholder((4,), dtypes.int32, 2)
  inp16 = UOp.placeholder((4,), dtypes.float16, 3)
  idx = UOp.special(4, "lidx0")
  sti = out16.index(idx).store(inp32.index(idx).load().cast(dtypes.float16))
  stf = out32.index(idx).store(inp16.index(idx).load().cast(dtypes.int32))
  sink = UOp.sink(sti, stf, idx, arg=KernelInfo(name="amd_asm_int_half_cast"))
  return _to_prg(sink)

def _bfloat16_store_program():
  out = UOp.placeholder((16,), dtypes.bfloat16, 0)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(UOp.const(1.0, dtypes.float32).cast(dtypes.bfloat16)).sink(idx, arg=KernelInfo(name="amd_asm_bfloat16_store"))
  return _to_prg(sink)

def _emulated_uint64_upcast_program():
  with Context(EMULATED_DTYPES="long"):
    ast = ((Tensor([1], dtype=dtypes.uint8, device="AMD").cast(dtypes.uint64) +
            Tensor([1], dtype=dtypes.uint8, device="AMD").cast(dtypes.uint64)).cast(dtypes.uint8)).schedule_linear().src[-1].src[0]
  return _to_prg(ast)

def _emulated_int64_cmod_const_program():
  with Context(EMULATED_DTYPES="long"):
    ast = ((Tensor([7], dtype=dtypes.int64, device="AMD") % 3).cast(dtypes.int32)).schedule_linear().src[-1].src[0]
  return _to_prg(ast)

def _emulated_int64_index_cmod_program():
  out = UOp.placeholder((8,), dtypes.float32, 0)
  idx = UOp.special(8, "lidx0")
  long_idx = UOp(Ops.CMOD, dtypes.long, (idx.cast(dtypes.long), UOp.const(8, dtypes.long)))
  sink = out.index(long_idx).store(UOp.const(1.0, dtypes.float32)).sink(idx, arg=KernelInfo(name="amd_asm_long_index_cmod"))
  return _to_prg(sink)

def _narrow_var_mod_program(dtype):
  ast = (Tensor([0], dtype=dtype, device="AMD") % Tensor([1], dtype=dtype, device="AMD")).schedule_linear().src[-1].src[0]
  return _to_prg(ast)

def _software_sin_lowered_sinks():
  # TRANSCENDENTAL=2 xsin emits i64 divmod after early decomp; needs late dtype pass on AMD
  ren = _REN
  with Context(TRANSCENDENTAL=2):
    cl = Tensor([1e6], device="AMD").sin().schedule_linear()
    asts = [si.src[0] for si in cl.src if si.src and si.src[0].op is Ops.SINK]
    return [full_rewrite_to_sink(ast, ren) for ast in asts]

def _atomic_add_program():
  out = UOp.placeholder((16,), dtypes.float32, 0)
  inp = UOp.placeholder((16,), dtypes.float32, 1)
  idx = UOp.special(16, "lidx0")
  atomic = UOp(Ops.CUSTOM, dtypes.void, (out.index(idx), inp.index(idx).load()), arg=amd_lib.AMD_ATOMIC_ADD)
  sink = UOp.sink(atomic, idx, arg=KernelInfo(name="amd_asm_atomic_add"))
  return _to_prg(sink)

def _int_narrow_cast_program():
  out = UOp.placeholder((4,), dtypes.uint16, 0)
  inp = UOp.placeholder((4,), dtypes.int32, 1)
  idx = UOp.special(4, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load().cast(dtypes.uint16)).sink(idx, arg=KernelInfo(name="amd_asm_int_narrow_cast"))
  return _to_prg(sink)

def _int_signed_narrow_cast_program():
  out = UOp.placeholder((4,), dtypes.int16, 0)
  inp = UOp.placeholder((4,), dtypes.int32, 1)
  idx = UOp.special(4, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load().cast(dtypes.int16)).sink(idx, arg=KernelInfo(name="amd_asm_int_signed_narrow_cast"))
  return _to_prg(sink)

def _int_signed_widen_cast_program():
  out = UOp.placeholder((4,), dtypes.int32, 0)
  inp = UOp.placeholder((4,), dtypes.int16, 1)
  idx = UOp.special(4, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load().cast(dtypes.int32)).sink(idx, arg=KernelInfo(name="amd_asm_int_signed_widen_cast"))
  return _to_prg(sink)

def _float16_unary_program():
  out = UOp.placeholder((4,), dtypes.float16, 0)
  inp = UOp.placeholder((4,), dtypes.float16, 1)
  idx = UOp.special(4, "lidx0")
  x = inp.index(idx).load()
  val = UOp(Ops.SQRT, dtypes.float16, (x,)) + UOp(Ops.LOG2, dtypes.float16, (x,))
  val = val + UOp(Ops.TRUNC, dtypes.float16, (x + UOp.const(0.75, dtypes.float16),)) - UOp(Ops.TRUNC, dtypes.float16, (x,))
  val = val + UOp(Ops.RECIPROCAL, dtypes.float16, (x,)) + UOp(Ops.SIN, dtypes.float16, (x,))
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_float16_unary"))
  return _to_prg(sink)

def _bitwise_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  x = inp.index(idx).load() & UOp.const(0xff, dtypes.uint32)
  val = ((x | UOp.const(0x10, dtypes.uint32)) ^ UOp.const(0x3, dtypes.uint32)) >> UOp.const(1, dtypes.uint32)
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_bitwise"))
  return _to_prg(sink)

def _cmod_pow2_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  val = UOp(Ops.CMOD, dtypes.uint32, (inp.index(idx).load() + idx.cast(dtypes.uint32), UOp.const(8, dtypes.uint32)))
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_cmod_pow2"))
  return _to_prg(sink)

def _const_divmod_program():
  out = UOp.placeholder((16,), dtypes.int32, 0)
  idx = UOp.special(16, "lidx0").cast(dtypes.int32) + UOp.const(7, dtypes.int32)
  val = UOp(Ops.CDIV, dtypes.int32, (idx, UOp.const(11, dtypes.int32))) + UOp(Ops.CMOD, dtypes.int32, (idx, UOp.const(11, dtypes.int32)))
  sink = out.index(UOp.special(16, "lidx0")).store(val).sink(arg=KernelInfo(name="amd_asm_const_divmod"))
  return _to_prg(sink)

def _var_divmod_program():
  out = UOp.placeholder((4,), dtypes.int32, 0)
  inp = UOp.placeholder((4,), dtypes.int32, 1)
  div = UOp.placeholder((4,), dtypes.int32, 2)
  idx = UOp.special(4, "lidx0")
  x, d = inp.index(idx).load(), div.index(idx).load()
  q, r = UOp(Ops.CDIV, dtypes.int32, (x, d)), UOp(Ops.CMOD, dtypes.int32, (x, d))
  sink = out.index(idx).store(q * UOp.const(10, dtypes.int32) + r).sink(idx, arg=KernelInfo(name="amd_asm_var_divmod"))
  return _to_prg(sink)

def _max_program(dtype):
  out = UOp.placeholder((16,), dtype, 0)
  inp = UOp.placeholder((16,), dtype, 1)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(UOp(Ops.MAX, dtype, (inp.index(idx).load(), UOp.const(7, dtype)))) \
            .sink(idx, arg=KernelInfo(name=f"amd_asm_max_{dtype.name}"))
  return _to_prg(sink)

def _mulacc_program():
  out = UOp.placeholder((16,), dtypes.float32, 0)
  inp0 = UOp.placeholder((16,), dtypes.float32, 1)
  inp1 = UOp.placeholder((16,), dtypes.float32, 2)
  idx = UOp.special(16, "lidx0")
  val = UOp(Ops.MULACC, dtypes.float32, (inp0.index(idx).load(), inp1.index(idx).load(), UOp.const(1.0, dtypes.float32)))
  sink = out.index(idx).store(val) \
            .sink(idx, arg=KernelInfo(name="amd_asm_mulacc"))
  return _to_prg(sink)

def _fused_mulacc_program():
  out = UOp.placeholder((16,), dtypes.float32, 0)
  inp0 = UOp.placeholder((16,), dtypes.float32, 1)
  inp1 = UOp.placeholder((16,), dtypes.float32, 2)
  idx = UOp.special(16, "lidx0")
  val = inp0.index(idx).load() * inp1.index(idx).load() + UOp.const(1.0, dtypes.float32)
  sink = out.index(idx).store(val) \
            .sink(idx, arg=KernelInfo(name="amd_asm_fused_mulacc"))
  return _to_prg(sink)

def _float16_fused_mulacc_program():
  out = UOp.placeholder((4,), dtypes.float16, 0)
  inp0 = UOp.placeholder((4,), dtypes.float16, 1)
  inp1 = UOp.placeholder((4,), dtypes.float16, 2)
  idx = UOp.special(4, "lidx0")
  val = inp0.index(idx).load() * inp1.index(idx).load() + UOp.const(1.0, dtypes.float16)
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_float16_fused_mulacc"))
  return _to_prg(sink)

def _cast_reciprocal_program():
  out = UOp.placeholder((16,), dtypes.float32, 0)
  idx = UOp.special(16, "lidx0")
  val = (idx.cast(dtypes.int32) + UOp.const(1, dtypes.int32)).cast(dtypes.float32).reciprocal()
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_cast_reciprocal"))
  return _to_prg(sink)

def _float_to_int_cast_program():
  out = UOp.placeholder((16,), dtypes.int32, 0)
  inp = UOp.placeholder((16,), dtypes.float32, 1)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load().cast(dtypes.int32)).sink(idx, arg=KernelInfo(name="amd_asm_float_to_int_cast"))
  return _to_prg(sink)

def _exp2_program():
  out = UOp.placeholder((4,), dtypes.float32, 0)
  inp = UOp.placeholder((4,), dtypes.float32, 1)
  idx = UOp.special(4, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load().exp2()).sink(idx, arg=KernelInfo(name="amd_asm_exp2"))
  return _to_prg(sink)

def _unary_math_program():
  out = UOp.placeholder((4,), dtypes.float32, 0)
  inp = UOp.placeholder((4,), dtypes.float32, 1)
  idx = UOp.special(4, "lidx0")
  x = inp.index(idx).load()
  val = UOp(Ops.SQRT, dtypes.float32, (x,)) + UOp(Ops.LOG2, dtypes.float32, (x,))
  val = val + UOp(Ops.TRUNC, dtypes.float32, (x + UOp.const(0.75, dtypes.float32),)) - UOp(Ops.TRUNC, dtypes.float32, (x,))
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_unary_math"))
  return _to_prg(sink)

def _sin_program():
  out = UOp.placeholder((4,), dtypes.float32, 0)
  inp = UOp.placeholder((4,), dtypes.float32, 1)
  idx = UOp.special(4, "lidx0")
  sink = out.index(idx).store(UOp(Ops.SIN, dtypes.float32, (inp.index(idx).load(),))).sink(idx, arg=KernelInfo(name="amd_asm_sin"))
  return _to_prg(sink)

def _spill_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inps = [UOp.placeholder((16,), dtypes.uint32, i+1) for i in range(6)]
  idx = UOp.special(16, "lidx0")
  vals = [inp.index(idx).load() for inp in inps]
  acc = vals[0]
  for v in vals[1:]: acc = acc + v
  sink = out.index(idx).store(acc).sink(idx, arg=KernelInfo(name="amd_asm_spill"))
  to_program_cache.clear()
  # One VGPR: sequential reduce still spills because each load+acc needs a second temp.
  return to_program(sink, OneVGPRAMDRenderer(_GFX11))

def _multi_spill_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inps = [UOp.placeholder((16,), dtypes.uint32, i+1) for i in range(4)]
  idx = UOp.special(16, "lidx0")
  vals = [inp.index(idx).load() for inp in inps]
  acc = vals[0]
  for v in vals[1:]: acc = acc + v
  sink = out.index(idx).store(acc).sink(idx, arg=KernelInfo(name="amd_asm_multi_spill"))
  to_program_cache.clear()
  return to_program(sink, OneVGPRAMDRenderer(_GFX11))

def _range_program():
  out = UOp.placeholder((8,), dtypes.uint32, 0)
  rng = UOp.range(8, 0, AxisType.LOOP)
  sink = UOp(Ops.SINK, dtypes.void,
             (UOp(Ops.END, dtypes.void, (out.index(rng).store(rng.cast(dtypes.uint32) + UOp.const(1, dtypes.uint32)), rng)),),
             arg=KernelInfo(name="amd_asm_range"), tag=1)
  return _to_prg(sink)

def _nested_range_program():
  out = UOp.placeholder((32,), dtypes.uint32, 0)
  r0, r1 = UOp.range(4, 0, AxisType.LOOP), UOp.range(8, 1, AxisType.LOOP)
  idx = (r0.cast(dtypes.uint32) << UOp.const(3, dtypes.uint32)) + r1.cast(dtypes.uint32)
  sink = out.index(idx).store(idx).end(r0, r1).sink(arg=KernelInfo(name="amd_asm_nested_range")).replace(tag=1)
  return _to_prg(sink)

def _var_range_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  n = UOp.param(1, dtypes.uint32, (), vmin_vmax=(0, 16), name="n", addrspace=AddrSpace.ALU)
  r = UOp.range(n, 0, AxisType.LOOP)
  sink = out.index(r).store(r.cast(dtypes.uint32)).end(r).sink(n, arg=KernelInfo(name="amd_asm_var_range")).replace(tag=1)
  return _to_prg(sink)

def _loop_vcc_remat_program():
  out = UOp.placeholder((4,), dtypes.uint32, 0)
  idx = UOp.special(4, "lidx0")
  r = UOp.range(2, 0, AxisType.LOOP)
  gate0, gate1 = idx < 2, idx < 3
  val = gate0.where(r.cast(dtypes.uint32) + UOp.const(1, dtypes.uint32), UOp.const(0, dtypes.uint32))
  val = val + gate1.where(r.cast(dtypes.uint32) + UOp.const(2, dtypes.uint32), UOp.const(0, dtypes.uint32))
  sink = out.index(idx).store(val).end(r).sink(idx, arg=KernelInfo(name="amd_asm_loop_vcc_remat")).replace(tag=1)
  return _to_prg(sink)

def _global_dim_program():
  out = UOp.placeholder((1024,), dtypes.float32, 0)
  rng = UOp.range(1024, 0, AxisType.GLOBAL)
  sink = out.index(rng).store(UOp.const(1.0, dtypes.float32)).end(rng).sink(arg=KernelInfo(name="amd_asm_global_dim"))
  return _to_prg(sink)

def _local_program(dtype=dtypes.uint32, slot=0):
  out = UOp.placeholder((16,), dtype, 0)
  smem = UOp.placeholder((16,), dtype, slot=slot, addrspace=AddrSpace.LOCAL)
  idx = UOp.special(16, "lidx0")
  st = smem.index(idx).store(UOp.const(7, dtype))
  barr = UOp(Ops.BARRIER, dtypes.void, (st,))
  ld = smem.after(barr).index(idx).load()
  sink = out.index(idx).store(ld).sink(idx, arg=KernelInfo(name=f"amd_asm_local_{dtype.name}_{slot}"))
  return _to_prg(sink)

def _half_lds_wide_program():
  # One thread: 8 contiguous half LDS elems → DS_STORE/LOAD_B128 (global half stays B32).
  out = UOp.placeholder((8,), dtypes.half, 0)
  smem = UOp.placeholder((8,), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  sts = tuple(smem.index(UOp.const(i, dtypes.weakint)).store(UOp.const(1.0, dtypes.half)) for i in range(8))
  barr = UOp(Ops.BARRIER, dtypes.void, sts)
  lds = [smem.after(barr).index(UOp.const(i, dtypes.weakint)).load() for i in range(8)]
  outs = tuple(out.index(UOp.const(i, dtypes.weakint)).store(lds[i]) for i in range(8))
  sink = UOp(Ops.SINK, dtypes.void, outs, arg=KernelInfo(name="amd_asm_half_lds_wide"))
  return _to_prg(sink)

def _reg_buffer_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  scratch = UOp.placeholder((16,), dtypes.uint32, slot=0, addrspace=AddrSpace.REG)
  idx = UOp.special(16, "lidx0")
  st = scratch.index(idx).store(idx.cast(dtypes.uint32) + UOp.const(1, dtypes.uint32))
  ld = scratch.after(st).index(idx).load()
  sink = out.index(idx).store(ld + UOp.const(5, dtypes.uint32)).sink(idx, arg=KernelInfo(name="amd_asm_reg_buffer"))
  return _to_prg(sink)

def _two_reg_buffers_program():
  out = UOp.placeholder((100,), dtypes.float32, 0)
  scratch0 = UOp.placeholder((100,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  scratch1 = UOp.placeholder((25,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
  idx = UOp.special(100, "lidx0")
  idx25 = idx % 25
  st0 = scratch0.index(idx).store(UOp.const(1.0, dtypes.float32))
  st1 = scratch1.index(idx25).store(UOp.const(2.0, dtypes.float32))
  val = scratch0.after(st0).index(idx).load() + scratch1.after(st1).index(idx25).load()
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_two_reg_buffers"))
  return _to_prg(sink)

def _identity_reg_store_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  scratch = UOp.placeholder((16,), dtypes.uint32, slot=0, addrspace=AddrSpace.REG)
  idx = UOp.special(16, "lidx0")
  off = UOp.const(0, dtypes.int32)
  st0 = scratch.index(off).store(UOp.const(7, dtypes.uint32))
  ld = scratch.after(st0).index(off).load()
  _ = scratch.after(ld).index(off).store(ld)
  sink = out.index(idx).store(ld + UOp.const(1, dtypes.uint32)).sink(idx, arg=KernelInfo(name="amd_asm_identity_reg_store"))
  return _to_prg(sink)

def _gated_load_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp0 = UOp.placeholder((16,), dtypes.uint32, 1)
  inp1 = UOp.placeholder((16,), dtypes.uint32, 2)
  idx = UOp.special(16, "lidx0")
  gate0 = idx < 8
  gate1 = idx < 4
  val0 = gate0.where(inp0.index(gate0.where(idx, idx.const_like(Invalid))).load(), UOp.const(0, dtypes.uint32))
  val1 = gate1.where(inp1.index(gate1.where(idx, idx.const_like(Invalid))).load(), UOp.const(0, dtypes.uint32))
  sink = out.index(idx).store(val0 + val1).sink(idx, arg=KernelInfo(name="amd_asm_gated_load"))
  return _to_prg(sink)

def _late_gated_store_linear(materialized_gate=False):
  renderer = _REN
  out = UOp(Ops.INS, dtypes.uint64, (UOp.const(0, dtypes.int32).rtag(),), AMDOps.KERNARG, (Register("out", 0, _cons=amd_lib.SGPR),))
  inp = UOp(Ops.INS, dtypes.uint64, (UOp.const(8, dtypes.int32).rtag(),), AMDOps.KERNARG, (Register("inp", 1, _cons=amd_lib.SGPR),))
  idx = UOp(Ops.INS, dtypes.uint32, (UOp.special(16, "lidx0").rtag(),), AMDOps.MOV, (Register("idx", 2, _cons=(amd_lib.LID[0],)),))
  val = UOp(Ops.INS, dtypes.uint32, (inp, idx), AMDOps.LOAD, (Register("val", 3, _cons=amd_lib.VGPR),))
  gate = UOp(Ops.INS, dtypes.bool, (idx, UOp.const(8, dtypes.weakint).rtag()), AMDOps.CMPLT)
  one = None
  if materialized_gate:
    one = UOp(Ops.INS, dtypes.uint32, (UOp.const(1, dtypes.uint32).rtag(),), AMDOps.MOV,
              (Register("one", 4, _cons=amd_lib.VGPR),))
    gate = UOp(Ops.INS, dtypes.bool, (idx, one), AMDOps.AND,
               (Register("gate", 5, _cons=amd_lib.VGPR),))
  addr = UOp(Ops.INDEX, dtypes.uint32, (out, idx))
  mif = UOp(Ops.IF, dtypes.void, (gate, addr))
  st = UOp(Ops.STORE, dtypes.void, (addr, val))
  mend = UOp(Ops.ENDIF, dtypes.void, (mif,))
  uops = [u for u in (out, inp, idx, val, one, gate, addr, mif, st, mend) if u is not None]
  lst = line_rewrite(uops, renderer.pre_regalloc_matcher, PreRegAllocContext())
  lst = sorted(lst, key=lambda u: u.op is not Ops.INS or bool(u.src))
  regalloc_ctx = LinearScanRegallocContext(lst, renderer)
  lst = line_rewrite(lst, pm_regalloc_rewrite, regalloc_ctx)
  lst = line_rewrite(lst, renderer.post_regalloc_matcher, regalloc_ctx)
  return UOp(Ops.LINEAR, src=tuple(lst))

def _after_global_load_program():
  tmp = UOp.placeholder((16,), dtypes.float32, 0)
  out = UOp.placeholder((16,), dtypes.float32, 1)
  idx = UOp.special(16, "lidx0")
  st = tmp.index(idx).store(UOp.const(3.0, dtypes.float32))
  ld = tmp.after(st).index(idx).load()
  sink = out.index(idx).store(ld + UOp.const(1.0, dtypes.float32)).sink(idx, arg=KernelInfo(name="amd_asm_after_global_load"))
  return _to_prg(sink)

def _local_sgpr_data_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  smem = UOp.placeholder((16,), dtypes.uint32, slot=0, addrspace=AddrSpace.LOCAL)
  val = UOp.param(1, dtypes.uint32, (), vmin_vmax=(0, 16), name="val", addrspace=AddrSpace.ALU)
  idx = UOp.special(16, "lidx0")
  st = smem.index(idx).store(val)
  barr = UOp(Ops.BARRIER, dtypes.void, (st,))
  ld = smem.after(barr).index(idx).load()
  sink = out.index(idx).store(ld).sink(idx, val, arg=KernelInfo(name="amd_asm_local_sgpr_data"))
  return _to_prg(sink)

def _multi_local_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  smem0 = UOp.placeholder((16,), dtypes.uint32, slot=0, addrspace=AddrSpace.LOCAL)
  smem1 = UOp.placeholder((16,), dtypes.uint32, slot=1, addrspace=AddrSpace.LOCAL)
  idx = UOp.special(16, "lidx0")
  st0 = smem0.index(idx).store(UOp.const(7, dtypes.uint32))
  st1 = smem1.index(idx).store(UOp.const(11, dtypes.uint32))
  barr = UOp(Ops.BARRIER, dtypes.void, (st0, st1))
  ld0, ld1 = smem0.after(barr).index(idx).load(), smem1.after(barr).index(idx).load()
  sink = out.index(idx).store(ld0 + ld1).sink(idx, arg=KernelInfo(name="amd_asm_multi_local"))
  return _to_prg(sink)

def _duplicate_local_slot_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  smem0 = UOp.placeholder((16,), dtypes.uint32, slot=0, addrspace=AddrSpace.LOCAL)
  smem1 = UOp.placeholder((8,), dtypes.uint32, slot=0, addrspace=AddrSpace.LOCAL)
  idx = UOp.special(8, "lidx0")
  st0 = smem0.index(idx).store(UOp.const(7, dtypes.uint32))
  st1 = smem1.index(idx).store(UOp.const(11, dtypes.uint32))
  sink = out.index(idx).store(UOp.const(0, dtypes.uint32)).sink(idx, st0, st1, arg=KernelInfo(name="amd_asm_duplicate_local"))
  return _to_prg(sink)


def _program_for_custom_kernel(sink:UOp, renderer) -> UOp:
  # to_program leaves ISA-lowered SPECIAL→INS edges; custom_kernel+SPEC walks the PROGRAM and
  # rejects those. Hand-kernel style: clean sink + flat INS args + compiled BINARY.
  prg = to_program(sink, renderer)
  flat = tuple(UOp(Ops.INS, arg=u.arg) for u in prg.src[1].src if u.op is Ops.INS)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=flat))+prg.src[2:], arg=prg.arg)

def _custom_renderer_spill(out:UOp, inp:UOp) -> UOp:
  out, inp = out.flatten(), inp.flatten()
  idx = UOp.special(out.numel(), "lidx0")
  vals = [inp.base.index(idx).load() + UOp.const(i, dtypes.uint32) for i in range(6)]
  acc = vals[0]
  for v in vals[1:]: acc = acc + v
  sink = out.base.index(idx).store(acc).sink(idx, arg=KernelInfo(name="amd_asm_hw_spill"))
  return _program_for_custom_kernel(sink, TinyVGPRAMDRenderer(Target("AMD", arch=Device["AMD"].arch)))

def _custom_renderer_lds(out:UOp) -> UOp:
  out = out.flatten()
  idx = UOp.special(out.numel(), "lidx0")
  smem = UOp.placeholder((out.numel(),), dtypes.uint32, slot=0, addrspace=AddrSpace.LOCAL)
  st = smem.index(idx).store(UOp.const(7, dtypes.uint32))
  barr = UOp(Ops.BARRIER, dtypes.void, (st,))
  ld = smem.after(barr).index(idx).load()
  sink = out.base.index(idx).store(ld).sink(idx, arg=KernelInfo(name="amd_asm_hw_lds"))
  return _program_for_custom_kernel(sink, AMDRenderer(Target("AMD", arch=Device["AMD"].arch)))

def _has_amd_asm_runtime() -> bool:
  return Device.DEFAULT == "AMD" and isinstance(Device["AMD"].renderer, AMDRenderer) and Device["AMD"].arch.startswith("gfx11")

def _gidx_program():
  out = UOp.placeholder((64,), dtypes.uint32, 0)
  idx = UOp.special(16, "lidx0") + (UOp.special(4, "gidx0") << 4)
  sink = out.index(idx).store(idx).sink(arg=KernelInfo(name="amd_asm_gidx"))
  return _to_prg(sink)

def _multi_dim_program():
  out = UOp.placeholder((128,), dtypes.uint32, 0)
  lidx0, lidx1 = UOp.special(8, "lidx0"), UOp.special(4, "lidx1")
  gidx1 = UOp.special(4, "gidx1")
  idx = lidx0 + (lidx1 << 3) + (gidx1 << 5)
  sink = out.index(idx).store(idx).sink(arg=KernelInfo(name="amd_asm_multi_dim"))
  return _to_prg(sink)

def _z_dim_program():
  out = UOp.placeholder((256,), dtypes.uint32, 0)
  lidx2 = UOp.special(4, "lidx2")
  gidx2 = UOp.special(4, "gidx2")
  idx = lidx2 + (gidx2 << 2)
  sink = out.index(idx).store(idx).sink(arg=KernelInfo(name="amd_asm_z_dim"))
  return _to_prg(sink)

class TestAMDRenderer(unittest.TestCase):
  def test_rejects_non_rdna3(self):
    with self.assertRaises(RuntimeError):
      AMDRenderer(Target("AMD", arch="gfx1200"))

  def test_advertises_scheduler_locals_and_shared_memory(self):
    renderer = _REN
    self.assertTrue(renderer.has_shared)
    self.assertTrue(renderer.has_local)
    self.assertTrue(renderer.supports_float4)
    self.assertEqual(renderer.local_prod_max, 1024)

  def test_tensor_cores_only_advertise_lowered_mode(self):
    self.assertEqual([(x.dtype_in, x.dtype_out) for x in _REN.tensor_cores], [(dtypes.half, dtypes.float)])

  def test_scheduler_rejects_oversized_local_workgroup(self):
    ast = Tensor.empty(4000, device="AMD").sum().schedule_linear().src[0].src[0]
    with self.assertRaises(KernelOptError):
      to_program(ast.replace(arg=replace(ast.arg, opts_to_apply=(Opt(OptOps.GROUP, 0, 0),))), _REN)

  def test_scheduler_prefers_group_for_simple_partial_sum(self):
    with Context(DEV="MOCKKFD+AMD:AMD"):
      ast = Tensor.empty(10_000_000).sum().schedule_linear().src[0].src[0]
    prg = to_program(ast, _REN)
    self.assertEqual(prg.src[0].arg.applied_opts, (Opt(OptOps.GROUP, 0, 16),))
    self.assertEqual(prg.arg.local_size, (16, 1, 1))

  def test_to_program_assembles_elf(self):
    prg = _simple_add_program()
    self.assertIs(_prg_src(prg).op, Ops.SOURCE)
    self.assertIs(_prg_bin(prg).op, Ops.BINARY)
    _check_elf(self, prg)
    self.assertIn("load", _prg_src(prg).arg.lower())
    self.assertIn("store", _prg_src(prg).arg.lower())
    self.assertEqual(_amd_desc(prg).kernarg_size, 16)

  def test_program_estimates_survive_instruction_selection(self):
    est = _simple_add_program().src[0].arg.estimates
    self.assertIsNotNone(est)
    self.assertGreater(est.ops, 0)
    self.assertGreater(est.mem, 0)

  def test_linear_contains_amd_ops(self):
    prg = _simple_add_program()
    linear_ops = _lin_ops(prg)
    self.assertIn(AMDOps.KERNARG, linear_ops)
    self.assertIn(AMDOps.LOAD, linear_ops)
    self.assertIn(AMDOps.STORE, linear_ops)

  def test_two_global_loads_share_waitcnt(self):
    prg = _two_load_add_program()
    inst_names = _amd_inst_names(prg)
    first_load = inst_names.index("GLOBAL_LOAD_B32")
    second_load = inst_names.index("GLOBAL_LOAD_B32", first_load + 1)
    first_wait = inst_names.index("S_WAITCNT_VMCNT")
    self.assertEqual(inst_names.count("GLOBAL_LOAD_B32"), 2)
    self.assertEqual(inst_names.count("S_LOAD_B64"), 3)
    self.assertEqual(inst_names.count("S_WAITCNT_LGKMCNT"), 1)
    self.assertEqual(inst_names.count("S_WAITCNT_VMCNT"), 1)
    self.assertLess(inst_names.index("S_WAITCNT_LGKMCNT"), first_load)
    self.assertLess(first_load, first_wait)
    self.assertLess(second_load, first_wait)
    self.assertLess(first_wait, inst_names.index("V_ADD_F32_E32"))

  def test_global_store_drains_vscnt_before_end(self):
    # RDNA3 vector stores complete on vscnt; must drain before s_endpgm.
    prg = _simple_add_program()
    inst_names = _amd_inst_names(prg)
    self.assertIn("GLOBAL_STORE_B32", inst_names)
    self.assertEqual(inst_names.count("S_WAITCNT_VSCNT"), 1)
    self.assertGreater(inst_names.index("S_WAITCNT_VSCNT"), inst_names.index("GLOBAL_STORE_B32"))
    self.assertEqual(inst_names[-1], "S_WAITCNT_VSCNT")

  def test_scratch_spill_drains_vscnt(self):
    prg = _spill_program()
    inst_names = _amd_inst_names(prg)
    self.assertIn("SCRATCH_STORE_B32", inst_names)
    store_i = inst_names.index("SCRATCH_STORE_B32")
    self.assertIn("S_WAITCNT_VSCNT", inst_names[store_i:])

  def test_uint32_alu_param_offsets(self):
    prg = _uint_var_program()
    _check_elf(self, prg)
    kernarg_offsets = [u.src[0].arg for u in _prg_lin(prg).src if u.op is Ops.INS and u.arg is AMDOps.KERNARG]
    self.assertEqual(sorted(kernarg_offsets), [0, 8, 16])
    self.assertEqual(_amd_desc(prg).kernarg_size, 20)

  def test_multiple_alu_params_are_dense_after_buffers(self):
    prg = _two_uint_var_program()
    _check_elf(self, prg)
    kernarg_offsets = [u.src[0].arg for u in _prg_lin(prg).src if u.op is Ops.INS and u.arg is AMDOps.KERNARG]
    self.assertEqual(sorted(kernarg_offsets), [0, 8, 16, 20])
    self.assertEqual(_amd_desc(prg).kernarg_size, 24)

  def test_narrow_global_copy_assembles(self):
    for dtype in (dtypes.bool, dtypes.uint8, dtypes.int8, dtypes.uint16, dtypes.int16):
      with self.subTest(dtype=dtype):
        prg = _copy_program(dtype)
        _check_elf(self, prg)

  def test_reserved_v254_v255_not_allocated(self):
    for prg in (_simple_add_program(), _range_program(), _var_range_program()):
      with self.subTest(name=prg.arg.name):
        regs = [greg(u).index for u in _prg_lin(prg).src if isinstance(greg(u), Register)]
        self.assertLess(max(regs, default=0), 256 + 254)

  def test_abi_fixed_registers_are_not_temp_allocated(self):
    progs = (_multi_dim_program(), _z_dim_program(), _uint_var_program(), _global_dim_program(),
             _spill_program(), _multi_spill_program(), _local_program(), _multi_local_program())
    for prg in progs:
      with self.subTest(name=prg.arg.name):
        _assert_abi_reg_isolation(self, prg)

  def test_kernarg_sgpr_pairs_do_not_overlap(self):
    prg = _uint_var_program()
    kernarg_bases = [greg(u).index for u in _prg_lin(prg).src if u.op is Ops.INS and u.arg is AMDOps.KERNARG and u.dtype.itemsize == 8]
    self.assertEqual(kernarg_bases, sorted(kernarg_bases))
    for a, b in zip(kernarg_bases, kernarg_bases[1:]):
      self.assertGreaterEqual(b - a, 2)

  def test_linear_has_no_explicit_end_op(self):
    for prg in (_simple_add_program(), _range_program(), _nested_range_program(), _var_range_program()):
      with self.subTest(name=prg.arg.name):
        self.assertFalse(any(u.op is Ops.END for u in _prg_lin(prg).src))
        linear_ops = _lin_ops(prg)
        self.assertNotIn("END", [getattr(op, "name", op) for op in linear_ops])

  def test_compare_where_assembles(self):
    for dtype in (dtypes.uint32, dtypes.int32, dtypes.float32):
      with self.subTest(dtype=dtype):
        _check_asm(self, _where_program(dtype), AMDOps.CMPLT, AMDOps.WHERE)

  def test_where_sgpr_true_materializes_vsrc1(self):
    prg = _where_sgpr_true_program()
    _check_elf(self, prg)
    insts = list(_REN._insts_from_linear(_prg_lin(prg)))
    cndmask = next(i for i in insts if getattr(i, "op_name", "") == "V_CNDMASK_B32_E32")
    self.assertEqual(cndmask.vsrc1, amd_lib.TMP_VDATA)
    self.assertTrue(any(getattr(i, "op_name", "") == "V_MOV_B32_E32" and i.vdst == amd_lib.TMP_VDATA for i in insts))

  def test_where_compare_value_materializes_flag(self):
    prg = _where_compare_value_program()
    _check_elf(self, prg)
    compare_ops = {AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ}
    for u in _prg_lin(prg).src:
      if u.op is Ops.INS and u.arg is AMDOps.WHERE:
        for value in u.src[1:]:
          self.assertFalse(value.op is Ops.INS and value.arg in compare_ops)

  def test_eq_where_assembles(self):
    for dtype in (dtypes.uint32, dtypes.int32, dtypes.float32):
      with self.subTest(dtype=dtype):
        _check_asm(self, _eq_where_program(dtype), AMDOps.CMPEQ, AMDOps.WHERE)

  def test_float_cmpne_uses_float_compare(self):
    _check_asm(self, _float_cmpne_program(), AMDOps.CMPNE, insts=("V_CMP_NEQ_F32_E32",))

  def test_cmpne_compare_flag_simplifies(self):
    _check_asm(self, _cmpne_compare_flag_program(), AMDOps.CMPLT, AMDOps.WHERE, no_ops=(AMDOps.CMPNE,))

  def test_bool_and_compare_flags_materializes(self):
    prg = _bool_and_compare_flags_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    self.assertIn(AMDOps.CMPLT, linear_ops)
    self.assertIn(AMDOps.CMPNE, linear_ops)
    self.assertIn(AMDOps.AND, linear_ops)
    self.assertGreaterEqual(linear_ops.count(AMDOps.WHERE), 2)

  def test_bool_compare_store_materializes_flag(self):
    prg = _bool_compare_store_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    self.assertIn(AMDOps.CMPLT, linear_ops)
    self.assertIn(AMDOps.WHERE, linear_ops)
    self.assertIn(AMDOps.STORE, linear_ops)

  def test_float16_where_and_cast_assemble(self):
    cases = (
      (_float16_where_program(), (AMDOps.WHERE,), ("GLOBAL_LOAD_U16", "GLOBAL_STORE_B16", "V_CNDMASK_B32_E32")),
      (_float16_cast_program(), (AMDOps.CAST,), ("V_CVT_F32_F16_E32", "V_CVT_F16_F32_E32")),
    )
    for prg, ops, insts in cases:
      with self.subTest(name=prg.arg.name):
        _check_asm(self, prg, *ops, insts=insts)

  def test_bfloat16_tagged_param_lowers_to_kernarg(self):
    prg = _bfloat16_store_program()
    _check_elf(self, prg)
    self.assertFalse(any(u.op is Ops.PARAM for u in _prg_lin(prg).src))
    inst_names = _amd_inst_names(prg)
    self.assertIn("GLOBAL_STORE_B16", inst_names)

  def test_emulated_uint64_upcast_assembles(self):
    prg = _emulated_uint64_upcast_program()
    _check_elf(self, prg)
    self.assertFalse(any(u.op is Ops.CAST for u in _prg_lin(prg).src))

  def test_emulated_int64_cmod_assembles(self):
    for prg in (_emulated_int64_cmod_const_program(), _emulated_int64_index_cmod_program()):
      with self.subTest(name=prg.arg.name):
        _check_elf(self, prg)
        self.assertFalse(any(u.op is Ops.CMOD for u in _prg_lin(prg).src))

  def test_narrow_variable_mod_widens_before_regalloc(self):
    for dtype in (dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16):
      with self.subTest(dtype=dtype):
        prg = _narrow_var_mod_program(dtype)
        _check_elf(self, prg)
        self.assertFalse(any(u.op is Ops.CMOD for u in _prg_lin(prg).src))

  def test_software_sin_decomposes_late_64bit_divmod(self):
    sinks = _software_sin_lowered_sinks()
    self.assertTrue(sinks)
    bad = [u for sink in sinks for u in sink.toposort()
           if u.op in (Ops.CDIV, Ops.CMOD) and u.dtype.scalar() in (dtypes.long, dtypes.ulong)]
    self.assertEqual(bad, [], f"{len(bad)} raw 64-bit CDIV/CMOD survived lowering (would crash regalloc)")

  def test_custom_atomic_add_lowers_to_global_atomic(self):
    prg = _atomic_add_program()
    self.assertFalse(any(u.op is Ops.CUSTOM for u in _prg_lin(prg).src))
    _check_asm(self, prg, AMDOps.ATOMIC_ADD, insts=("GLOBAL_ATOMIC_ADD_F32", "S_WAITCNT_VMCNT"))

  def test_int_narrow_cast_assembles(self):
    prg = _int_narrow_cast_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    self.assertIn(AMDOps.CAST, linear_ops)
    inst_names = _amd_inst_names(prg)
    self.assertIn("V_AND_B32_E32", inst_names)
    self.assertIn("GLOBAL_STORE_B16", inst_names)

  def test_int_signed_narrow_cast_sign_extends(self):
    prg = _int_signed_narrow_cast_program()
    _check_elf(self, prg)
    inst_names = _amd_inst_names(prg)
    self.assertIn("V_LSHLREV_B32_E64", inst_names)
    self.assertIn("V_ASHRREV_I32_E64", inst_names)
    self.assertNotIn("V_AND_B32_E32", inst_names)

  def test_global_loads_share_single_vmcnt_wait(self):
    prg = _two_load_add_program()
    inst_names = _amd_inst_names(prg)
    self.assertEqual(inst_names.count("GLOBAL_LOAD_B32"), 2)
    self.assertEqual(inst_names.count("S_WAITCNT_VMCNT"), 1)
    last_load = max(i for i,n in enumerate(inst_names) if n == "GLOBAL_LOAD_B32")
    self.assertGreater(inst_names.index("S_WAITCNT_VMCNT"), last_load)

  def test_matmul_reg_accumulators_promote_off_scratch(self):
    prg = _matmul64_program()
    linear_ops = _lin_ops(prg)
    self.assertNotIn(AMDOps.SLOAD, linear_ops)
    self.assertNotIn(AMDOps.SSTORE, linear_ops)
    inst_names = _amd_inst_names(prg)
    self.assertFalse(any("SCRATCH" in name for name in inst_names))

  def test_matmul_default_schedule_uses_float4_memory_tile(self):
    prg = _matmul64_program()
    self.assertEqual(prg.src[0].arg.applied_opts, (
      Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 4),
      Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.LOCAL, 1, 16)))
    self.assertEqual(prg.arg.local_size, (8, 16, 1))
    linear_ops = _lin_ops(prg)
    self.assertEqual(linear_ops.count(AMDOps.MULACC), 48)
    inst_names = _amd_inst_names(prg)
    self.assertEqual(inst_names.count("GLOBAL_LOAD_B128"), 8)
    self.assertEqual(inst_names.count("GLOBAL_STORE_B128"), 4)
    self.assertEqual(inst_names.count("V_FMA_F32"), 48)

  def test_half_matmul_wmma_assembles(self):
    prg = _half_matmul_wmma_program()
    # Small WMMA may still v_pack; product-16 register path uses u16+d16_hi (tested below).
    _check_asm(self, prg, AMDOps.WMMA, AMDOps.PACK_F16, insts=("V_WMMA_F32_16X16X16_F16",))
    wmmas = [u for u in _prg_lin(prg).src if u.op is Ops.INS and u.arg is AMDOps.WMMA]
    self.assertEqual(len(wmmas), 1)
    self.assertEqual(wmmas[0].src[0].dtype, dtypes.float)
    self.assertEqual(wmmas[0].src[1].dtype, dtypes.half)
    self.assertEqual(wmmas[0].src[2].dtype, dtypes.half)
    self.assertEqual(amd_lib._reg_slots(wmmas[0]), 8)
    self.assertEqual(amd_lib._reg_slots(wmmas[0].src[0]), 8)
    self.assertEqual(amd_lib._reg_slots(wmmas[0].src[1]), 8)
    self.assertEqual(amd_lib._reg_slots(wmmas[0].src[2]), 8)
    self.assertEqual(greg(wmmas[0]), greg(wmmas[0].src[0]))

  def _half_matmul_tc_lds_ab_pre_regalloc(self, *, tc_local: int | None = None):
    # LOOP-ended LDS fill + contract. B transpose: scatter LSTORE may outnumber LLOAD ops.
    # Assert on isel+pre_regalloc only — full wide regalloc for N=256 OOMs/times out CI workers.
    import os
    old = os.environ.get("TC_LDS_AB")
    old_l = os.environ.get("TC_LOCAL")
    os.environ["TC_LDS_AB"] = "1"
    if tc_local is None: os.environ.pop("TC_LOCAL", None)
    else: os.environ["TC_LOCAL"] = str(tc_local)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        ast = ast.schedule_linear().src[-1].src[0]
      sink = full_rewrite_to_sink(ast, _REN, optimize=True)
      opts = sink.arg.applied_opts
      # Default / TC_LOCAL=2 → LOCAL 2×2. TC_LOCAL≥4 → asymmetric Opt LOCAL 2×4.
      if tc_local is not None and tc_local >= 4:
        self.assertIn(Opt(OptOps.LOCAL, 0, 2), opts)
        self.assertIn(Opt(OptOps.LOCAL, 1, 4), opts)
      else:
        loc = 2 if tc_local is None else tc_local
        self.assertIn(Opt(OptOps.LOCAL, 0, loc), opts)
        self.assertIn(Opt(OptOps.LOCAL, 1, loc), opts)
      # tile product ≤8 under TC_LDS_AB (LLOAD/PACK_F16 disjoint VGPR pools)
      up = [o.arg for o in opts if o.op is OptOps.UPCAST]
      self.assertGreaterEqual(math.prod(up) if up else 0, 4)
      self.assertLessEqual(math.prod(up) if up else 0, 8)
      sink = sink.replace(arg=replace(sink.arg, estimates=Estimates.from_uops(tuple(sink.toposort()), ignore_indexing=True)))
      sink = graph_rewrite(sink, _REN.pre_isel_matcher, ctx=itertools.count(-1, -1), name="pre isel", bottom_up=True)
      sink = graph_rewrite(sink, _REN.isel_matcher, ctx=IselContext(sink), name="isel", bottom_up=True)
      lst = line_rewrite(linearize(sink), pm_linearize_cleanups)
      if _REN.pre_regalloc_matcher is not None:
        lst, scratch = _REN.prepare_pre_regalloc(lst)
        pa_ctx = PreRegAllocContext(lst)
        pa_ctx.scratch.update(scratch)
        lst = line_rewrite(lst, _REN.pre_regalloc_matcher, pa_ctx)
      linear_ops = [u.arg for u in lst if u.op is Ops.INS]
      self.assertGreaterEqual(linear_ops.count(AMDOps.WMMA), 4)
      self.assertGreater(linear_ops.count(AMDOps.LSTORE), 0)
      self.assertGreater(linear_ops.count(AMDOps.LLOAD), 0)
      self.assertGreater(linear_ops.count(AMDOps.BARRIER), 0)
      # Default 2×2 is coop (LOAD < LLOAD). TC_LOCAL=4 can bounce with more GLOBAL_LOAD — still spill-free.
      if tc_local is None: self.assertLess(linear_ops.count(AMDOps.LOAD), linear_ops.count(AMDOps.LLOAD))
      self.assertNotIn(AMDOps.SPILL, linear_ops)
      self.assertNotIn(AMDOps.FILL, linear_ops)
      return opts, linear_ops
    finally:
      if old is None: os.environ.pop("TC_LDS_AB", None)
      else: os.environ["TC_LDS_AB"] = old
      if old_l is None: os.environ.pop("TC_LOCAL", None)
      else: os.environ["TC_LOCAL"] = old_l
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_tc_lds_ab_stages_without_spill(self):
    # Dual LOCAL 2×2 WG + tile UPCAST 2×2 (product ≤4); default bounce.
    self._half_matmul_tc_lds_ab_pre_regalloc()

  def test_half_matmul_tc_lds_ab_local4_stages_without_spill(self):
    # TC_LOCAL=4 → asymmetric LOCAL 4×2 (symmetric 4×4 historically TDR on display).
    self._half_matmul_tc_lds_ab_pre_regalloc(tc_local=4)

  def test_tc_lds_ab_skips_stage_for_eye_operand(self):
    # eye/WHERE A is not a GLOBAL INDEX — partial staging mixed with staged B and broke
    # expand_broadcast. Fall back to register path (no LLOAD) so identity@B compiles.
    import os
    old = os.environ.get("TC_LDS_AB")
    os.environ["TC_LDS_AB"] = "1"
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        for ast in (
          (Tensor.eye(256, dtype=dtypes.half) @ Tensor.empty(256, 256, dtype=dtypes.half)),
          (Tensor.empty(256, 256, dtype=dtypes.half) @ Tensor.eye(256, dtype=dtypes.half)),
        ):
          to_program_cache.clear()
          prg = _to_prg(ast.schedule_linear().src[-1].src[0])
          linear_ops = _lin_ops(prg)
          self.assertGreaterEqual(linear_ops.count(AMDOps.WMMA), 4)
          self.assertEqual(linear_ops.count(AMDOps.LLOAD), 0)
          self.assertNotIn(AMDOps.SPILL, linear_ops)
    finally:
      if old is None: os.environ.pop("TC_LDS_AB", None)
      else: os.environ["TC_LDS_AB"] = old
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_tc_lds_tid_fill_peer_lds_reads(self):
    # Opt-in tid-fill (TC_LDS_TID=1): peer remap → DS_LOAD_U16 when A block==threads
    # (needs A tile_prod=local_n*2). Default LDS tiles 2×2 → tile_prod=2 → bounce fallback.
    import os
    old_ab, old_tid = os.environ.get("TC_LDS_AB"), os.environ.get("TC_LDS_TID")
    os.environ["TC_LDS_AB"] = "1"
    os.environ["TC_LDS_TID"] = "1"
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      opts, linear_ops = self._half_matmul_tc_lds_ab_pre_regalloc()
      self.assertGreater(linear_ops.count(AMDOps.LLOAD), 0)
      self.assertGreater(linear_ops.count(AMDOps.LSTORE), 0)
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      names = _amd_inst_names(prg)
      # Tid apply → U16 peer loads; else identity bounce must stay B128.
      if names.count("DS_LOAD_U16"):
        self.assertGreater(names.count("DS_LOAD_U16"), 0)
      else:
        self.assertGreater(names.count("DS_LOAD_B128"), 0)
        self.assertEqual(names.count("DS_LOAD_U16"), 0)
    finally:
      for k, old in (("TC_LDS_AB", old_ab), ("TC_LDS_TID", old_tid)):
        if old is None: os.environ.pop(k, None)
        else: os.environ[k] = old
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_tc_lds_bounce_no_u16_lds(self):
    # Hybrid bounce: shared A (tid-fill/major-read) + B tid-wide → GLOBAL B128, no peer U16.
    # A LDS is block×16 (=4KiB at LOCAL 2×2); total A+B ≤ 8KiB.
    import os
    old_ab, old_tid = os.environ.get("TC_LDS_AB"), os.environ.get("TC_LDS_TID")
    os.environ["TC_LDS_AB"] = "1"
    os.environ["TC_LDS_TID"] = "0"
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      names = _amd_inst_names(prg)
      self.assertGreaterEqual(names.count("GLOBAL_LOAD_B128"), 2)
      self.assertEqual(names.count("GLOBAL_LOAD_U16"), 0)
      self.assertEqual(names.count("DS_LOAD_U16"), 0)
      self.assertGreater(names.count("DS_LOAD_B128"), 0)
      lds = [(int(u.src[0].arg), int(u.src[1].arg)) for u in _prg_lin(prg).src
             if u.op is Ops.INS and u.arg is AMDOps.LDS_BASE]
      self.assertTrue(lds, "expected LDS_BASE")
      self.assertLessEqual(sum(sz for sz, _ in lds), 8192)
    finally:
      for k, old in (("TC_LDS_AB", old_ab), ("TC_LDS_TID", old_tid)):
        if old is None: os.environ.pop(k, None)
        else: os.environ[k] = old
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_tc_lds_ab_stages_without_spill_gate(self):
    # Default LDS: product 8 + B transpose + addr remat → spill-free, DS_LOAD_B128 on A/B.
    import os
    old, old_r, old_a, old_tid = (os.environ.get("TC_LDS_AB"), os.environ.get("AMD_REMAT"),
                                  os.environ.get("AMD_REMAT_ADDR"), os.environ.get("TC_LDS_TID"))
    os.environ["TC_LDS_AB"] = "1"
    os.environ.pop("AMD_REMAT", None)
    os.environ.pop("AMD_REMAT_ADDR", None)
    os.environ.pop("TC_LDS_TID", None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      linear_ops = _lin_ops(prg)
      inst_names = _amd_inst_names(prg)
      self.assertEqual(linear_ops.count(AMDOps.WMMA), 8)
      # See test_half_matmul_default_lds_intensity_gates: 3D locals can spill; LDS stays opt-in.
      self.assertGreater(linear_ops.count(AMDOps.LLOAD), 0)
      self.assertGreater(linear_ops.count(AMDOps.LSTORE), 0)
      self.assertGreaterEqual(inst_names.count("GLOBAL_LOAD_B128"), 2)
      self.assertEqual(inst_names.count("GLOBAL_LOAD_U16"), 0)
      self.assertEqual(inst_names.count("DS_LOAD_U16"), 0)
      self.assertGreater(inst_names.count("DS_LOAD_B128"), 0)
    finally:
      for k, v in (("TC_LDS_AB", old), ("AMD_REMAT", old_r), ("AMD_REMAT_ADDR", old_a), ("TC_LDS_TID", old_tid)):
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_upcast4_without_spill(self):
    # Explicit product-16 (4×4) register path — also the register default when ALLOW_UPCAST16.
    import os
    old_u, old_l, old_t, old_r, old_a = (os.environ.get("TC_UPCAST"), os.environ.get("TC_LDS_AB"),
                                        os.environ.get("TC_UPCAST_TILES"), os.environ.get("AMD_REMAT"),
                                        os.environ.get("ALLOW_UPCAST16"))
    os.environ["TC_UPCAST"] = "4"
    os.environ["TC_UPCAST_TILES"] = "16"
    os.environ["ALLOW_UPCAST16"] = "1"
    os.environ["AMD_REMAT"] = "1"
    os.environ["TC_LDS_AB"] = "0"
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      opts = prg.src[0].arg.applied_opts
      self.assertIn(Opt(OptOps.UPCAST, 0, 4), opts)
      self.assertIn(Opt(OptOps.UPCAST, 1, 4), opts)
      linear_ops = _lin_ops(prg)
      self.assertGreaterEqual(linear_ops.count(AMDOps.WMMA), 16)
      self.assertNotIn(AMDOps.SLOAD, linear_ops)
      self.assertNotIn(AMDOps.SSTORE, linear_ops)
    finally:
      for k, old in (("TC_UPCAST", old_u), ("TC_UPCAST_TILES", old_t), ("TC_LDS_AB", old_l),
                     ("AMD_REMAT", old_r), ("ALLOW_UPCAST16", old_a)):
        if old is None: os.environ.pop(k, None)
        else: os.environ[k] = old
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_lds_upcast16_aliases_acc_not_reg_scratch(self):
    # LDS product-16 uses MOV-zero cin (vs register SLOAD reload). ACC alias must still
    # kill the 128-slot REG buffer SLOAD/SSTORE. Residual VGPR spills OK — not default yet.
    import os
    old = {k: os.environ.get(k) for k in
           ("TC_LDS_AB", "TC_LOCAL", "TC_UPCAST", "TC_UPCAST_TILES", "ALLOW_UPCAST16", "AMD_REMAT")}
    os.environ.update({"TC_LDS_AB": "1", "TC_LOCAL": "2", "TC_UPCAST": "4",
                       "TC_UPCAST_TILES": "16", "ALLOW_UPCAST16": "1"})
    os.environ.pop("AMD_REMAT", None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      linear_ops = _lin_ops(prg)
      self.assertEqual(linear_ops.count(AMDOps.WMMA), 16)
      self.assertEqual(linear_ops.count(AMDOps.SLOAD), 0)
      self.assertEqual(linear_ops.count(AMDOps.SSTORE), 0)
      self.assertGreater(linear_ops.count(AMDOps.LLOAD), 0)
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_upcast16_refused_without_allow(self):
    # Register defaults to product 16; ALLOW_UPCAST16=0 clamps back to ≤8.
    import os
    old_u, old_t, old_l, old_a = (os.environ.get("TC_UPCAST"), os.environ.get("TC_UPCAST_TILES"),
                                os.environ.get("TC_LDS_AB"), os.environ.get("ALLOW_UPCAST16"))
    os.environ["TC_UPCAST"] = "4"
    os.environ["TC_UPCAST_TILES"] = "16"
    os.environ["TC_LDS_AB"] = "0"
    os.environ["ALLOW_UPCAST16"] = "0"
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      up = [o.arg for o in prg.src[0].arg.applied_opts if o.op is OptOps.UPCAST]
      self.assertLessEqual(math.prod(up) if up else 0, 8)
      self.assertEqual(_lin_ops(prg).count(AMDOps.WMMA), 8)
    finally:
      for k, old in (("TC_UPCAST", old_u), ("TC_UPCAST_TILES", old_t), ("TC_LDS_AB", old_l),
                     ("ALLOW_UPCAST16", old_a)):
        if old is None: os.environ.pop(k, None)
        else: os.environ[k] = old
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_default_lds_intensity_gates(self):
    # Opt-in LDS defaults to product 8, spill-free with EXTRACT+addr remat.
    import os
    old_u, old_t, old_l, old_r, old_a, old_ra = (
      os.environ.get("TC_UPCAST"), os.environ.get("TC_UPCAST_TILES"),
      os.environ.get("TC_LDS_AB"), os.environ.get("AMD_REMAT"),
      os.environ.get("ALLOW_LDS_PRODUCT8"), os.environ.get("AMD_REMAT_ADDR"))
    for k in ("TC_UPCAST", "TC_UPCAST_TILES", "AMD_REMAT", "ALLOW_LDS_PRODUCT8", "AMD_REMAT_ADDR"):
      os.environ.pop(k, None)
    os.environ["TC_LDS_AB"] = "1"
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      opts = prg.src[0].arg.applied_opts
      self.assertIn(Opt(OptOps.LOCAL, 0, 2), opts)
      self.assertIn(Opt(OptOps.LOCAL, 1, 2), opts)
      up = [o.arg for o in opts if o.op is OptOps.UPCAST]
      self.assertLessEqual(math.prod(up) if up else 0, 8)
      linear_ops = _lin_ops(prg)
      self.assertEqual(linear_ops.count(AMDOps.WMMA), 8)
      # WARP-first 2D local_max → local_size (32,2,2); packed-lidx BFEs + 3D raise pressure.
      # Spill-free LDS under 3D locals is follow-up (TC_LDS_AB stays opt-in).
      self.assertEqual(prg.arg.local_size, (32, 2, 2))
      self.assertGreater(linear_ops.count(AMDOps.LLOAD), 0)
      self.assertGreater(linear_ops.count(AMDOps.LSTORE), 0)
      inst_names = _amd_inst_names(prg)
      self.assertGreaterEqual(inst_names.count("GLOBAL_LOAD_B128"), 2)
      self.assertEqual(inst_names.count("GLOBAL_LOAD_U16"), 0)
      self.assertEqual(inst_names.count("DS_LOAD_U16"), 0)
      self.assertGreater(inst_names.count("DS_LOAD_B128"), 0)
    finally:
      for k, old in (("TC_UPCAST", old_u), ("TC_UPCAST_TILES", old_t), ("TC_LDS_AB", old_l),
                     ("AMD_REMAT", old_r), ("ALLOW_LDS_PRODUCT8", old_a), ("AMD_REMAT_ADDR", old_ra)):
        if old is None: os.environ.pop(k, None)
        else: os.environ[k] = old
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_keeps_wmma_tile_local(self):
    # Do not sink WMMA past scalar A loads / hoist those above WMMA — keeps first WMMA
    # after one A pack (≤16 U16), enabling A/B overlap vs all-A-then-B.
    import os
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "TC_LOCAL", "TC_UPCAST", "TC_UPCAST_TILES", "ALLOW_UPCAST16")}
    for k in old: os.environ.pop(k, None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD"))
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      names = _amd_inst_names(prg)
      w0 = next(i for i, n in enumerate(names) if "WMMA" in n)
      pre = names[:w0]
      self.assertLessEqual(pre.count("GLOBAL_LOAD_U16"), 16)
      self.assertGreaterEqual(pre.count("GLOBAL_LOAD_B128"), 1)
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_default_is_spill_free_sixteen_wmma(self):
    # Default ISA: register path UPCAST=4×4 (product 16) + LOCAL=4 → 16 WMMA.
    import os
    old_u, old_t, old_l = os.environ.get("TC_UPCAST"), os.environ.get("TC_UPCAST_TILES"), os.environ.get("TC_LDS_AB")
    old_loc, old_a = os.environ.get("TC_LOCAL"), os.environ.get("ALLOW_UPCAST16")
    for k in ("TC_UPCAST", "TC_UPCAST_TILES", "TC_LDS_AB", "TC_LOCAL", "ALLOW_UPCAST16"): os.environ.pop(k, None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      linear_ops = _lin_ops(prg)
      self.assertEqual(linear_ops.count(AMDOps.WMMA), 16)
      self.assertEqual(linear_ops.count(AMDOps.LLOAD), 0)
      self.assertIn(Opt(OptOps.LOCAL, 1, 4), prg.src[0].arg.applied_opts)
      self.assertNotIn(AMDOps.SPILL, linear_ops)
      self.assertNotIn(AMDOps.FILL, linear_ops)
      self.assertNotIn(AMDOps.SLOAD, linear_ops)
      self.assertNotIn(AMDOps.SSTORE, linear_ops)
      # Scalar half A loads are s_clause-clustered (~+40% vs no clause on gfx1100).
      self.assertGreaterEqual(_amd_inst_names(prg).count("S_CLAUSE"), 1)
    finally:
      for k, old in (("TC_UPCAST", old_u), ("TC_UPCAST_TILES", old_t), ("TC_LDS_AB", old_l),
                     ("TC_LOCAL", old_loc), ("ALLOW_UPCAST16", old_a)):
        if old is None: os.environ.pop(k, None)
        else: os.environ[k] = old
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_irregular_n_compiles_spill_free(self):
    # Multi-WG / non-power-of-two N still lowers to product-16 register WMMA without spills.
    import os
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "TC_LOCAL", "TC_UPCAST", "TC_UPCAST_TILES", "ALLOW_UPCAST16")}
    for k in old: os.environ.pop(k, None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      for n in (320, 512, 768):
        with Context(BEAM=0):
          ast = (Tensor.empty(n, n, dtype=dtypes.half, device="AMD") @
                 Tensor.empty(n, n, dtype=dtypes.half, device="AMD"))
          prg = _to_prg(ast.schedule_linear().src[-1].src[0])
        ops = _lin_ops(prg)
        self.assertEqual(ops.count(AMDOps.WMMA), 16, n)
        self.assertNotIn(AMDOps.SPILL, ops, n)
        self.assertEqual(ops.count(AMDOps.LLOAD), 0, n)
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_acc_zero_uses_vopd(self):
    # Product-16 ACC packs zero-init via v_dual_mov (even/odd banks), not 128 scalar movs.
    import os
    from tinygrad.runtime.autogen.amd.rdna3.ins import VOPD
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "TC_UPCAST", "TC_UPCAST_TILES", "TC_LOCAL", "ALLOW_UPCAST16")}
    for k in old: os.environ.pop(k, None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD"))
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      insts = _REN._insts_from_linear(_prg_lin(prg))
      self.assertEqual(sum(1 for i in insts if isinstance(i, VOPD)), 64)  # 16 packs × 4 duals
      self.assertEqual(_lin_ops(prg).count(AMDOps.WMMA), 16)
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_lds_ab_waits_lgkm_before_wmma(self):
    # TC_LDS_AB: DS_LOAD dests feed WMMA A/B — must s_waitcnt_lgkmcnt before first WMMA.
    import os
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "ALLOW_LDS_PRODUCT8", "TC_LOCAL", "TC_UPCAST", "TC_UPCAST_TILES")}
    os.environ["TC_LDS_AB"] = "1"
    os.environ["ALLOW_LDS_PRODUCT8"] = "0"
    for k in ("TC_LOCAL", "TC_UPCAST", "TC_UPCAST_TILES"): os.environ.pop(k, None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD"))
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      names = [getattr(i, "op_name", type(i).__name__) for i in _REN._insts_from_linear(_prg_lin(prg))]
      self.assertGreater(sum(1 for n in names if n.startswith("DS_LOAD")), 0)
      wmma_i = next(i for i, n in enumerate(names) if "WMMA" in n)
      prev_ds = max(i for i, n in enumerate(names[:wmma_i]) if n.startswith("DS_LOAD"))
      self.assertTrue(any(n == "S_WAITCNT_LGKMCNT" for n in names[prev_ds:wmma_i]),
                      f"no lgkm wait between DS_LOAD@{prev_ds} and WMMA@{wmma_i}")
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_8x8_extracts_before_clobbering_addr_adds(self):
    # Non-TC half 8×8: hoisting B addr ADDs through A’s EXTRACTs used to rewrite A’s B128
    # VGPRs in-flight (sq_intr hang). EXTRACT unpack must complete before those ADDs.
    import os
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "NOLOCALS")}
    for k in old: os.environ.pop(k, None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(8, 8, dtype=dtypes.half, device="AMD") @
               Tensor.empty(8, 8, dtype=dtypes.half, device="AMD"))
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      self.assertEqual(prg.arg.local_size, (8, 8, 1))
      names = _amd_inst_names(prg)
      i_b128 = names.index("GLOBAL_LOAD_B128")
      i_first_u16 = names.index("GLOBAL_LOAD_U16")
      # Between A’s B128 and B’s first U16: wait + unpack, then addr ADDs — not ADDs first.
      window = names[i_b128:i_first_u16]
      self.assertIn("S_WAITCNT_VMCNT", window)
      self.assertTrue(any(n.startswith("V_MOV") or n.startswith("V_LSHR") for n in window))
      add_i = next(i for i, n in enumerate(window) if n == "V_ADD_NC_U32_E64")
      unpack_i = next(i for i, n in enumerate(window) if n.startswith("V_MOV") or n.startswith("V_LSHR"))
      self.assertLess(unpack_i, add_i)
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_wmma_acc_idx_map_survives_tile_key_collision(self):
    # Product-16 REG ACC: 16 consecutive SLOAD packs. First-idx tile keys collide (4 keys for
    # 16 packs), so epilogue must use reg_idx→(init,lane) — not tiles[tile] alone.
    from tinygrad.renderer.isa.rdna3 import _wmma_acc_zero_inits, _wmma_slot_tile_lane
    buf = UOp.placeholder((128,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
    zero = UOp.const(0.0, dtypes.float32)
    ab = UOp(Ops.INS, dtypes.float, tuple(
      UOp(Ops.INS, dtypes.float, (zero,), AMDOps.MOV, (Register(f"ab{i}", i),)) for i in range(8)),
      AMDOps.PACK, (Register("ab", 10),))
    uops: list[UOp] = []
    for p in range(16):
      loads = []
      for lane in range(8):
        idx = p * 8 + lane
        ld = UOp(Ops.INS, dtypes.float, (buf, UOp.const(idx, dtypes.int32)), AMDOps.SLOAD,
                 (Register(f"l{idx}", 100 + idx),))
        loads.append(ld)
        uops.append(ld)
      tag = (Register(f"acc{p}", 200 + p * 8),)
      pack = UOp(Ops.INS, dtypes.float, tuple(loads), AMDOps.PACK, tag)
      uops += [pack, UOp(Ops.INS, dtypes.float, (pack, ab, ab), AMDOps.WMMA, tag)]
    inits, tiles, idx_map = _wmma_acc_zero_inits(uops)
    self.assertEqual(len(inits), 16)
    self.assertEqual(len(idx_map), 128)
    # tiles[first_idx_tile] alone cannot name 16 packs
    self.assertLess(len(tiles), 16)
    self.assertEqual(len({_wmma_slot_tile_lane(p * 8)[0] for p in range(16)}), len(tiles))
    for p in range(16):
      init_ids = {id(idx_map[p * 8 + lane][0]) for lane in range(8)}
      self.assertEqual(len(init_ids), 1)
      self.assertEqual([idx_map[p * 8 + lane][1] for lane in range(8)], list(range(8)))

  def test_after_pre_regalloc_schedules_cast_before_store(self):
    # Product-16 epilogue: f32→f16 CAST must sit immediately before its STORE so half temps
    # do not all live at once (else regalloc spills into WMMA ACC and clobbers unread lanes).
    buf = UOp(Ops.INS, dtypes.ulong, (), AMDOps.KERNARG, (0,))
    addr = UOp(Ops.INS, dtypes.int, (), AMDOps.MOV, (Register("v1", 257),))
    acc0 = UOp(Ops.INS, dtypes.float, (), AMDOps.MOV, (Register("v2", 258),))
    acc1 = UOp(Ops.INS, dtypes.float, (), AMDOps.MOV, (Register("v3", 259),))
    c0 = UOp(Ops.INS, dtypes.half, (acc0,), AMDOps.CAST, (Register("v4", 260),))
    c1 = UOp(Ops.INS, dtypes.half, (acc1,), AMDOps.CAST, (Register("v5", 261),))
    s0 = UOp(Ops.INS, dtypes.void, (buf, addr, c0), AMDOps.STORE)
    s1 = UOp(Ops.INS, dtypes.void, (buf, addr, c1), AMDOps.STORE)
    # Pathological: both casts first, then both stores (pre-fix schedule).
    out = _REN.after_pre_regalloc([acc0, acc1, c0, c1, s0, s1])
    self.assertEqual(out, [acc0, acc1, c0, s0, c1, s1])

  def test_half_matmul_epilogue_cast_adjacent_to_store(self):
    # Default half GEMM keeps CAST glued to STORE in the final linear list + emits cvt+b16.
    import os
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "TC_UPCAST", "TC_UPCAST_TILES", "TC_LOCAL", "ALLOW_UPCAST16")}
    for k in old: os.environ.pop(k, None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD"))
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      lin = [u for u in _prg_lin(prg).src if u.op is Ops.INS]
      stores = [u for u in lin if u.arg is AMDOps.STORE]
      self.assertEqual(len(stores), 128)
      for i, u in enumerate(lin):
        if u.arg is not AMDOps.STORE: continue
        self.assertGreater(i, 0)
        prev = lin[i - 1]
        self.assertIs(prev.arg, AMDOps.CAST)
        self.assertIs(u.src[2], prev)
      names = _amd_inst_names(prg)
      self.assertEqual(names.count("V_CVT_F16_F32_E32"), 128)
      self.assertEqual(names.count("GLOBAL_STORE_B16"), 128)
      # CAST must not clobber store-addr CSE (TMP_VADDR page base).
      self.assertLessEqual(names.count("V_LSHL_ADD_U32"), 20)
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_c_stores_share_peeled_base(self):
    # Soft-peel nested ADD+imm so C stores share one addr base; large byte offs use v_lshl_add.
    import os
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "TC_UPCAST", "TC_UPCAST_TILES", "TC_LOCAL", "ALLOW_UPCAST16")}
    for k in old: os.environ.pop(k, None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      stores = [u for u in _prg_lin(prg).src if u.op is Ops.INS and u.arg is AMDOps.STORE]
      self.assertEqual(len(stores), 128)  # product-16 default (4×4)
      self.assertEqual(len({id(u.src[1]) for u in stores}), 1)
      self.assertGreaterEqual(sum(1 for u in stores if len(u.src) >= 4), 112)
      names = _amd_inst_names(prg)
      self.assertLessEqual(names.count("V_LSHL_ADD_U32"), 80)
      self.assertEqual(names.count("GLOBAL_STORE_B32"), 128)
      # Float EXTRACT coalesces onto WMMA pack+lane — stores read ACC VGPRs directly.
      self.assertLess(names.count("V_MOV_B32_E32"), 400)
      self.assertGreater(sum(1 for i in _REN._insts_from_linear(_prg_lin(prg))
                             if getattr(i, "op_name", "") == "GLOBAL_STORE_B32" and getattr(i, "offset", 0)), 10)
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_reg_store_spill_falls_back_to_scratch(self):
    disp = UOp.const(128, dtypes.int32)
    acc = UOp(Ops.INS, dtypes.float32, (disp,), AMDOps.FILL, (Register("v10", 266),))
    val = UOp(Ops.INS, dtypes.float32, (), AMDOps.MOV, (Register("v11", 267),))
    out, lst = amd_lib._lower_reg_store(UOp(Ops.INS, dtypes.void, (acc, val), AMDOps.REG_STORE))
    self.assertIs(out.arg, AMDOps.SPILL)
    self.assertIs(out.src[0], disp)
    self.assertIs(out.src[1], val)
    self.assertEqual(lst, [out])

  def test_regalloc_rewrites_surviving_shrink(self):
    renderer = _REN
    src = UOp(Ops.INS, dtypes.float32, (UOp.const(1.0, dtypes.float32).rtag(),), AMDOps.MOV,
              (Register("src", 0, _cons=amd_lib.VGPR),))
    shrink = UOp(Ops.SHRINK, dtypes.float32, (src, UOp.const(0, dtypes.int32), UOp.const(1, dtypes.int32)), tag=(
      Register("shrunk", 1, _cons=amd_lib.VGPR),))
    dst = UOp(Ops.INS, dtypes.float32, (shrink,), AMDOps.MOV, (Register("dst", 2, _cons=amd_lib.VGPR),))
    out = line_rewrite([src, shrink, dst], pm_regalloc_rewrite, LinearScanRegallocContext([src, shrink, dst], renderer))
    self.assertEqual([u.op for u in out], [Ops.INS, Ops.SHRINK, Ops.INS])
    self.assertIsInstance(greg(out[1]), Register)
    self.assertIs(out[2].src[0], out[1])

  def test_regalloc_vector_vgpr_reserves_consecutive_slots(self):
    renderer = _REN
    vvec = Register("vec", 0, _cons=amd_lib.VGPR)
    vscalar = Register("scalar", 1, _cons=amd_lib.VGPR)
    vuse = Register("use", 2, _cons=amd_lib.VGPR)
    pack_src = tuple(UOp.const(float(i), dtypes.float32).rtag() for i in range(4))
    vec = UOp(Ops.INS, dtypes.float32, pack_src, AMDOps.PACK, (vvec,))
    scalar = UOp(Ops.INS, dtypes.float32, arg=AMDOps.DEFINE, tag=(vscalar,))
    use = UOp(Ops.INS, dtypes.float32, (vec,), AMDOps.MOV, (vuse,))
    out = line_rewrite([vec, scalar, use], pm_regalloc_rewrite, LinearScanRegallocContext([vec, scalar, use], renderer))
    self.assertNotIn(greg(out[1]).index, range(greg(out[0]).index, greg(out[0]).index + 4))

  def test_regalloc_vector_group_can_evict_live_scalars(self):
    renderer = FourVGPRAMDRenderer(_GFX11)
    scalar_regs = [Register(f"s{i}", i, _cons=amd_lib.VGPR[:4]) for i in range(4)]
    scalars = [UOp(Ops.INS, dtypes.float32, arg=AMDOps.DEFINE, tag=(r,)) for r in scalar_regs]
    vvec = Register("vec", 4, _cons=amd_lib.VGPR[:4])
    pack_src = tuple(UOp.const(float(i), dtypes.float32).rtag() for i in range(4))
    vec = UOp(Ops.INS, dtypes.float32, pack_src, AMDOps.PACK, (vvec,))
    uses = [UOp(Ops.INS, dtypes.float32, (s,), AMDOps.MOV, (Register(f"use{i}", 5+i, _cons=amd_lib.VGPR[:4]),))
            for i,s in enumerate(scalars)]
    ctx = LinearScanRegallocContext(scalars + [vec] + uses, renderer)
    self.assertEqual(ctx.reals[len(scalars)][vvec].index, amd_lib.VGPR[0].index)

  def test_parallel_vmov_preserves_overlapping_vector_pack_sources(self):
    insts = amd_lib._parallel_vmov([(amd_lib.v[4], amd_lib.v[5]), (amd_lib.v[5], amd_lib.v[4])])
    self.assertEqual([getattr(i, "op_name", "") for i in insts], ["V_MOV_B32_E32"] * 3)
    self.assertEqual([str(i) for i in insts], [
      "v_mov_b32_e32(v[254], v[5])",
      "v_mov_b32_e32(v[5], v[4])",
      "v_mov_b32_e32(v[4], v[254])",
    ])

  def test_float4_global_memory_uses_b128_and_scalarized_alu(self):
    prg = _float4_add_program()
    _check_elf(self, prg)
    self.assertFalse(any(u.op is not Ops.INS for u in _prg_lin(prg).src))
    linear_ops = _lin_ops(prg)
    self.assertEqual(linear_ops.count(AMDOps.LOAD), 2)
    self.assertEqual(linear_ops.count(AMDOps.STORE), 1)
    self.assertIn(AMDOps.EXTRACT, linear_ops)
    self.assertIn(AMDOps.PACK, linear_ops)
    inst_names = _amd_inst_names(prg)
    self.assertEqual(inst_names.count("GLOBAL_LOAD_B128"), 2)
    self.assertEqual(inst_names.count("GLOBAL_STORE_B128"), 1)

  def test_float4_lds_memory_uses_ds_b128(self):
    prg = _float4_lds_program()
    _check_elf(self, prg)
    self.assertFalse(any(u.op is not Ops.INS for u in _prg_lin(prg).src))
    inst_names = _amd_inst_names(prg)
    self.assertIn("DS_STORE_B128", inst_names)
    self.assertIn("DS_LOAD_B128", inst_names)

  def test_half_memory_uses_b32_not_b128(self):
    # Small half add may stay B32; padded/gated must never widen (MMU fault).
    prg = _half_add_program()
    _check_elf(self, prg)
    inst_names = _amd_inst_names(prg)
    self.assertTrue(any(n.startswith("GLOBAL_LOAD_B") for n in inst_names))
    self.assertTrue(any(n.startswith("GLOBAL_STORE_B") for n in inst_names))
    self.assertNotIn("GLOBAL_LOAD_U16", inst_names)
    self.assertNotIn("GLOBAL_STORE_B16", inst_names)

  def test_half_matmul_uses_wide_global_load(self):
    # Register path: contiguous operand B128; strided WMMA operand stays U16.
    import os
    old = os.environ.get("TC_LDS_AB")
    os.environ["TC_LDS_AB"] = "0"
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      inst_names = _amd_inst_names(prg)
      wide = sum(1 for n in inst_names if n in ("GLOBAL_LOAD_B64", "GLOBAL_LOAD_B128"))
      self.assertGreater(wide, 0, "expected ungated half GEMM to use B64/B128 loads")
      self.assertGreater(inst_names.count("GLOBAL_LOAD_U16"), 0,
                         "strided WMMA operand stays scalar U16 on register path")
      self.assertEqual(sum(1 for n in inst_names if "SCRATCH" in n), 0)
    finally:
      if old is None: os.environ.pop("TC_LDS_AB", None)
      else: os.environ["TC_LDS_AB"] = old
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_register_path_strided_operand_is_u16(self):
    # Contiguous A → B128; strided B → u16+d16_hi when opted in. Default is v_pack (identity@B NaN on HW).
    import os
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "AMD_D16_HI", "TC_LOCAL", "TC_UPCAST", "TC_UPCAST_TILES")}
    os.environ["TC_LDS_AB"] = "0"
    os.environ["TC_LOCAL"] = "0"
    os.environ["TC_UPCAST"] = "2"
    os.environ["TC_UPCAST_TILES"] = "4"
    os.environ["AMD_D16_HI"] = "1"
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      inst_names = _amd_inst_names(prg)
      self.assertEqual(inst_names.count("GLOBAL_LOAD_B128"), 4)
      self.assertEqual(inst_names.count("GLOBAL_LOAD_U16"), 16)
      self.assertEqual(inst_names.count("GLOBAL_LOAD_D16_HI_B16"), 16)
      self.assertEqual(inst_names.count("V_PACK_B32_F16"), 0)
      self.assertGreaterEqual(inst_names.count("S_CLAUSE"), 2)
      self.assertEqual(inst_names.count("V_WMMA_F32_16X16X16_F16"), 4)
      self.assertFalse(any(o.op is OptOps.LOCAL for o in prg.src[0].arg.applied_opts))
      self.assertEqual(prg.arg.local_size, (32, 1, 1))
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_matmul_register_path_vpack_default(self):
    # Default AMD_D16_HI off → u16 + v_pack. Pin product-4 1D locals for load counts.
    import os
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "AMD_D16_HI", "TC_LOCAL", "TC_UPCAST", "TC_UPCAST_TILES")}
    os.environ["TC_LDS_AB"] = "0"
    os.environ["TC_LOCAL"] = "0"
    os.environ["TC_UPCAST"] = "2"
    os.environ["TC_UPCAST_TILES"] = "4"
    os.environ.pop("AMD_D16_HI", None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      inst_names = _amd_inst_names(prg)
      self.assertEqual(inst_names.count("GLOBAL_LOAD_B128"), 4)
      self.assertEqual(inst_names.count("GLOBAL_LOAD_U16"), 32)
      self.assertEqual(inst_names.count("GLOBAL_LOAD_D16_HI_B16"), 0)
      self.assertGreater(inst_names.count("V_PACK_B32_F16"), 0)
      self.assertEqual(inst_names.count("V_WMMA_F32_16X16X16_F16"), 4)
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_tc_lds_ab_not_default_until_faster(self):
    # Step C: ISA default is register+B128 until LDS wins @4096 on HW.
    import os
    old = os.environ.get("TC_LDS_AB")
    os.environ.pop("TC_LDS_AB", None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      self.assertEqual(_lin_ops(prg).count(AMDOps.LLOAD), 0)
      self.assertEqual(_lin_ops(prg).count(AMDOps.WMMA), 16)  # register default product-16 (4×4)
    finally:
      if old is None: os.environ.pop("TC_LDS_AB", None)
      else: os.environ["TC_LDS_AB"] = old
      getenv.cache_clear()
      to_program_cache.clear()

  def test_lds_product8_clamped_without_allow(self):
    # ALLOW_LDS_PRODUCT8=0 forces 2×2 under LDS (product 8 is otherwise the LDS default).
    import os
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "TC_UPCAST", "TC_UPCAST_TILES", "ALLOW_LDS_PRODUCT8")}
    os.environ["TC_LDS_AB"] = "1"
    os.environ["TC_UPCAST"] = "4"
    os.environ["TC_UPCAST_TILES"] = "8"
    os.environ["ALLOW_LDS_PRODUCT8"] = "0"
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      self.assertEqual(_lin_ops(prg).count(AMDOps.WMMA), 4)  # clamped to 2×2
      self.assertEqual(_lin_ops(prg).count(AMDOps.SPILL), 0)
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_lds_product8_allow_serializes_a_batches(self):
    # ALLOW_LDS_PRODUCT8: expand serializes A batches (AFTER). Spill-free with LLOAD EXTRACT remat.
    import os
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "TC_UPCAST", "TC_UPCAST_TILES", "ALLOW_LDS_PRODUCT8", "AMD_REMAT")}
    os.environ["TC_LDS_AB"] = "1"
    os.environ["TC_UPCAST"] = "4"
    os.environ["TC_UPCAST_TILES"] = "8"
    os.environ["ALLOW_LDS_PRODUCT8"] = "1"
    os.environ.pop("AMD_REMAT", None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      self.assertEqual(_lin_ops(prg).count(AMDOps.WMMA), 8)
      self.assertEqual(_amd_inst_names(prg).count("GLOBAL_LOAD_U16"), 0)
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_lds_unroll_gated_when_over_tile_budget(self):
    # TC_LDS_UNROLL used to apply then crash in expand_wmma (tile product > budget).
    # With product-8 already at max_tiles, UNROLL must not apply — compile stays green.
    import os
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "TC_LOCAL", "TC_LDS_UNROLL")}
    os.environ["TC_LDS_AB"] = "1"
    os.environ["TC_LOCAL"] = "2"
    os.environ["TC_LDS_UNROLL"] = "2"
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      opts = prg.src[0].arg.applied_opts
      self.assertFalse(any(o.op is OptOps.UNROLL for o in opts))
      self.assertEqual(_lin_ops(prg).count(AMDOps.WMMA), 8)
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_half_lds_memory_uses_ds_b128(self):
    # LDS half×8 uses ds_load_b128; global may also use B128 on ungated GEMM.
    prg = _half_lds_wide_program()
    _check_elf(self, prg)
    inst_names = _amd_inst_names(prg)
    self.assertIn("DS_STORE_B128", inst_names)
    self.assertIn("DS_LOAD_B128", inst_names)

  def test_padded_load_does_not_use_wide_global_load(self):
    prg = _padded_load_program()
    _check_elf(self, prg)
    inst_names = _amd_inst_names(prg)
    self.assertNotIn("GLOBAL_LOAD_B128", inst_names)
    self.assertIn("GLOBAL_LOAD_B32", inst_names)

  def test_int_signed_widen_cast_sign_extends(self):
    prg = _int_signed_widen_cast_program()
    _check_elf(self, prg)
    inst_names = _amd_inst_names(prg)
    self.assertIn("V_LSHLREV_B32_E64", inst_names)
    self.assertIn("V_ASHRREV_I32_E64", inst_names)
    self.assertNotIn("V_AND_B32_E32", inst_names)

  def test_float16_unary_promotes_to_float32(self):
    prg = _float16_unary_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    for op in (AMDOps.CAST, AMDOps.SQRT, AMDOps.LOG2, AMDOps.TRUNC, AMDOps.RECIPROCAL, AMDOps.SIN):
      self.assertIn(op, linear_ops)
    inst_names = _amd_inst_names(prg)
    for name in ("V_CVT_F32_F16_E32", "V_CVT_F16_F32_E32", "V_SIN_F32_E32"):
      self.assertIn(name, inst_names)

  def test_bitwise_assembles(self):
    prg = _bitwise_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    for op in (AMDOps.AND, AMDOps.OR, AMDOps.XOR, AMDOps.SHR):
      self.assertIn(op, linear_ops)

  def test_cmod_pow2_legalizes_to_and(self):
    prg = _cmod_pow2_program()
    _check_elf(self, prg)
    self.assertFalse(any(u.op is Ops.CMOD for u in _prg_lin(prg).src))
    linear_ops = _lin_ops(prg)
    self.assertIn(AMDOps.AND, linear_ops)

  def test_const_divmod_legalizes(self):
    prg = _const_divmod_program()
    _check_elf(self, prg)
    self.assertFalse(any(u.op in (Ops.CDIV, Ops.CMOD) for u in _prg_lin(prg).src))
    linear_ops = _lin_ops(prg)
    self.assertIn(AMDOps.SHR, linear_ops)
    self.assertIn(AMDOps.MUL, linear_ops)

  def test_var_divmod_legalizes(self):
    prg = _var_divmod_program()
    _check_elf(self, prg)
    self.assertFalse(any(u.op in (Ops.CDIV, Ops.CMOD) for u in _prg_lin(prg).src))
    linear_ops = _lin_ops(prg)
    for op in (AMDOps.SHL, AMDOps.SHR, AMDOps.CMPLT, AMDOps.WHERE):
      self.assertIn(op, linear_ops)

  def test_max_assembles(self):
    for dtype in (dtypes.uint32, dtypes.int32, dtypes.float32):
      with self.subTest(dtype=dtype):
        _check_asm(self, _max_program(dtype), AMDOps.MAX)

  def test_mulacc_assembles(self):
    for prg in (_mulacc_program(), _fused_mulacc_program()):
      with self.subTest(name=prg.arg.name):
        _check_elf(self, prg)
        linear_ops = _lin_ops(prg)
        self.assertIn(AMDOps.MULACC, linear_ops)
        insts = list(_REN._insts_from_linear(_prg_lin(prg)))
        self.assertIn("V_FMA_F32", [getattr(i, "op_name", "") for i in insts])

  def test_fused_mulacc_is_isel_only(self):
    self.assertNotIn(Ops.MULACC, AMDRenderer.code_for_op)
    prg = _fused_mulacc_program()
    linear_ops = _lin_ops(prg)
    self.assertIn(AMDOps.MULACC, linear_ops)
    self.assertNotIn(AMDOps.MUL, linear_ops)
    self.assertNotIn(AMDOps.ADD, linear_ops)

  def test_float16_fused_mulacc_uses_f16_fma(self):
    prg = _float16_fused_mulacc_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    self.assertIn(AMDOps.MULACC, linear_ops)
    self.assertNotIn(AMDOps.MUL, linear_ops)
    self.assertNotIn(AMDOps.ADD, linear_ops)
    inst_names = _amd_inst_names(prg)
    self.assertIn("V_FMA_F16", inst_names)

  def test_sgpr_sub_uses_scalar_instruction(self):
    src = UOp(Ops.INS, dtypes.uint32, arg=AMDOps.MOV, tag=(Register("s8", 8),))
    sub = UOp(Ops.INS, dtypes.uint32, (src, UOp.const(1, dtypes.uint32).rtag()), AMDOps.SUB, (Register("s6", 6),))
    insts = _REN._insts_for_uop(sub)
    self.assertEqual([getattr(i, "op_name", "") for i in insts], ["S_SUB_U32"])

  def test_self_mov_elided(self):
    reg = Register("v3", 256+3)
    src = UOp(Ops.INS, dtypes.uint32, arg=AMDOps.ADD, tag=(reg,))
    mov = UOp(Ops.INS, dtypes.uint32, (src,), AMDOps.MOV, (reg,))
    insts = _REN._insts_from_linear(UOp(Ops.LINEAR, src=(mov,)))
    self.assertEqual(insts, [])

  def test_cast_and_reciprocal_assemble(self):
    for prg in (_cast_reciprocal_program(), _float_to_int_cast_program()):
      with self.subTest(name=prg.arg.name):
        _check_elf(self, prg)
        linear_ops = _lin_ops(prg)
        self.assertIn(AMDOps.CAST, linear_ops)
    insts = list(_REN._insts_from_linear(_prg_lin(_cast_reciprocal_program())))
    inst_names = [getattr(i, "op_name", "") for i in insts]
    self.assertIn("V_CVT_F32_I32_E32", inst_names)
    rcp_idx = inst_names.index("V_RCP_F32_E32")
    self.assertEqual(inst_names[rcp_idx:rcp_idx+6],
                     ["V_RCP_F32_E32", "V_MUL_F32_E32", "V_SUB_F32_E32", "V_FMA_F32", "V_CMP_EQ_F32_E32", "V_CNDMASK_B32_E32"])
    insts = list(_REN._insts_from_linear(_prg_lin(_float_to_int_cast_program())))
    self.assertIn("V_CVT_I32_F32_E32", [getattr(i, "op_name", "") for i in insts])

  def test_exp2_assembles(self):
    _check_asm(self, _exp2_program(), AMDOps.EXP2, insts=("V_EXP_F32_E32",))

  def test_unary_math_assembles(self):
    prg = _unary_math_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    for op in (AMDOps.SQRT, AMDOps.LOG2, AMDOps.TRUNC):
      self.assertIn(op, linear_ops)
    inst_names = _amd_inst_names(prg)
    for name in ("V_SQRT_F32_E32", "V_LOG_F32_E32", "V_TRUNC_F32_E32"):
      self.assertIn(name, inst_names)

  def test_sin_assembles_with_inline_cody_waite_reduction(self):
    prg = _sin_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    self.assertIn(AMDOps.SIN, linear_ops)
    insts = list(_REN._insts_from_linear(_prg_lin(prg)))
    reduce_insts = [i for i in insts if getattr(i, "op_name", "") in
                    ("V_MUL_F32_E32", "V_ADD_F32_E32", "V_FRACT_F32_E32", "V_SUB_F32_E32", "V_SIN_F32_E32")]
    self.assertEqual([getattr(i, "op_name", "") for i in reduce_insts],
                     ["V_MUL_F32_E32", "V_ADD_F32_E32", "V_FRACT_F32_E32", "V_SUB_F32_E32", "V_MUL_F32_E32",
                      "V_SUB_F32_E32", "V_MUL_F32_E32", "V_SUB_F32_E32", "V_MUL_F32_E32", "V_SIN_F32_E32"])
    scale, bias, _, floor_turns, hi_mul, hi_sub, lo_mul, lo_sub, final_scale, sin = reduce_insts
    self.assertEqual(scale.vdst, amd_lib.TMP_VDATA)
    self.assertEqual(bias.vdst, amd_lib.TMP_VDATA)
    self.assertEqual(floor_turns.vdst, amd_lib.TMP_VDATA)
    self.assertEqual(hi_sub.vdst, amd_lib.TMP_VADDR)
    self.assertEqual(lo_sub.vdst, amd_lib.TMP_VADDR)
    self.assertEqual(final_scale.vdst, amd_lib.TMP_VDATA)
    self.assertEqual(sin.src0, amd_lib.TMP_VDATA)
    self.assertAlmostEqual(struct.unpack("f", struct.pack("I", scale.literal))[0], 1.0 / (2.0 * math.pi))
    self.assertAlmostEqual(struct.unpack("f", struct.pack("I", final_scale.literal))[0], 1.0 / (2.0 * math.pi))
    self.assertEqual(str(bias.src0), "0.5")
    self.assertAlmostEqual(struct.unpack("f", struct.pack("I", hi_mul.literal))[0], 6.28125)
    self.assertAlmostEqual(struct.unpack("f", struct.pack("I", lo_mul.literal))[0], 0.0019353071795864769)

  def test_uint32_wrap_literal_assembles(self):
    prg = _uint_wrap_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    self.assertIn(AMDOps.ADD, linear_ops)

  def test_uint32_mul_alu_param_assembles(self):
    prg = _uint_var_mul_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    self.assertIn(AMDOps.MUL, linear_ops)

  def test_vgpr_spill_assembles_with_private_segment(self):
    prg = _spill_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    self.assertIn(AMDOps.SPILL, linear_ops)
    self.assertIn(AMDOps.FILL, linear_ops)
    desc = _amd_desc(prg)
    self.assertGreaterEqual(desc.private_segment_fixed_size, 4)
    self.assertTrue(desc.compute_pgm_rsrc2 & (1 << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT_SHIFT))
    self.assertFalse(desc.kernel_code_properties & (1 << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_SHIFT))

  def test_vgpr_multiple_spill_slots_size_private_segment(self):
    prg = _multi_spill_program()
    _check_elf(self, prg)
    spill_ops = [u for u in _prg_lin(prg).src if u.op is Ops.INS and u.arg in (AMDOps.SPILL, AMDOps.FILL)]
    slots = sorted({u.src[0].arg for u in spill_ops})
    self.assertGreaterEqual(len(slots), 2)
    self.assertGreaterEqual(_amd_desc(prg).private_segment_fixed_size, 8)

  def test_vgpr_spill_uses_explicit_zero_scratch_addr(self):
    prg = _spill_program()
    renderer = TinyVGPRAMDRenderer(_GFX11)
    for u in _prg_lin(prg).src:
      if u.op is Ops.INS and u.arg in (AMDOps.SPILL, AMDOps.FILL):
        with self.subTest(op=u.arg.name):
          insts = renderer._insts_for_uop(u)
          self.assertEqual(insts[0].op_name, "V_MOV_B32_E32")
          self.assertEqual(insts[0].vdst, amd_lib.TMP_VADDR)
          self.assertEqual(str(insts[0].src0), "0")
          scratch = next(i for i in insts if type(i).__name__ == "SCRATCH")
          self.assertEqual(scratch.addr, amd_lib.TMP_VADDR)

  def test_vector_vgpr_spill_expands_to_scalar_scratch_lanes(self):
    renderer = _REN
    disp = UOp.const(64, dtypes.int32)
    pack_src = tuple(UOp.const(float(i), dtypes.float32).rtag() for i in range(4))
    src = UOp(Ops.INS, dtypes.float32, pack_src, AMDOps.PACK, (Register("v20", 276),))
    spill = UOp(Ops.INS, dtypes.void, (disp, src), AMDOps.SPILL)
    fill = UOp(Ops.INS, dtypes.float32, (disp, UOp.const(4, dtypes.int32).rtag()), AMDOps.FILL, (Register("v40", 296),))
    spill_insts, fill_insts = renderer._insts_for_uop(spill), renderer._insts_for_uop(fill)
    self.assertEqual([getattr(i, "op_name", "") for i in spill_insts], ["V_MOV_B32_E32"] + ["SCRATCH_STORE_B32"]*4 + ["S_WAITCNT_VSCNT"])
    self.assertEqual([i.offset for i in spill_insts[1:5]], [64, 68, 72, 76])
    self.assertEqual([i.data for i in spill_insts[1:5]], [amd_lib.v[20], amd_lib.v[21], amd_lib.v[22], amd_lib.v[23]])
    self.assertEqual([getattr(i, "op_name", "") for i in fill_insts], ["V_MOV_B32_E32"] + ["SCRATCH_LOAD_B32"]*4)
    self.assertEqual([i.offset for i in fill_insts[1:]], [64, 68, 72, 76])
    self.assertEqual([i.vdst for i in fill_insts[1:]], [amd_lib.v[40], amd_lib.v[41], amd_lib.v[42], amd_lib.v[43]])
    self.assertTrue(all(i.sve == 1 for i in spill_insts[1:5]))
    self.assertTrue(all(i.sve == 1 for i in fill_insts[1:]))

  def test_range_loop_assembles_with_branch_fixups(self):
    prg = _range_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    for op in (AMDOps.LABEL, AMDOps.CMP_GE, AMDOps.CBRANCH_SCC1, AMDOps.BRANCH):
      self.assertIn(op, linear_ops)
    insts = list(_REN._insts_from_linear(_prg_lin(prg)))
    branches = [i.simm16 if i.simm16 < 0x8000 else i.simm16 - 0x10000 for i in insts
                if getattr(i, "op_name", "") in ("S_CBRANCH_SCC1", "S_BRANCH")]
    self.assertEqual(len(branches), 2)
    self.assertGreater(branches[0], 0)
    self.assertLess(branches[1], 0)

  def test_loop_compare_scalarizes_vgpr_bound(self):
    acc = UOp(Ops.INS, dtypes.uint32, arg=AMDOps.MOV, tag=(Register("s6", 6),))
    bound = UOp(Ops.INS, dtypes.uint32, arg=AMDOps.ADD, tag=(Register("v3", 256+3),))
    cmp = UOp(Ops.INS, dtypes.void, (acc, bound), AMDOps.CMP_GE)
    insts = _REN._insts_for_uop(cmp)
    self.assertEqual([getattr(i, "op_name", "") for i in insts], ["V_READFIRSTLANE_B32_E32", "S_CMP_GE_U32"])
    self.assertEqual(insts[0].vdst, amd_lib.TMP_SDATA1)

  def test_loop_where_rematerializes_vcc_before_each_cndmask(self):
    prg = _loop_vcc_remat_program()
    insts = list(_REN._insts_from_linear(_prg_lin(prg)))
    names = [getattr(i, "op_name", "") for i in insts]
    cndmask_idxs = [i for i,n in enumerate(names) if n == "V_CNDMASK_B32_E32"]
    self.assertGreaterEqual(len(cndmask_idxs), 2)
    for i in cndmask_idxs:
      self.assertTrue(any(names[j].startswith("V_CMP_") for j in range(max(0, i-2), i)),
                      f"missing compare rematerialization before cndmask at instruction {i}")

  def test_nested_range_loop_assembles_with_branch_fixups(self):
    prg = _nested_range_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    self.assertEqual(linear_ops.count(AMDOps.LABEL), 4)
    self.assertEqual(linear_ops.count(AMDOps.CBRANCH_SCC1), 2)
    self.assertEqual(linear_ops.count(AMDOps.BRANCH), 2)
    insts = list(_REN._insts_from_linear(_prg_lin(prg)))
    branches = [i.simm16 if i.simm16 < 0x8000 else i.simm16 - 0x10000 for i in insts
                if getattr(i, "op_name", "") in ("S_CBRANCH_SCC1", "S_BRANCH")]
    self.assertEqual(sum(x > 0 for x in branches), 2)
    self.assertEqual(sum(x < 0 for x in branches), 2)

  def test_variable_range_loop_materializes_sgpr_index_and_data(self):
    prg = _var_range_program()
    _check_elf(self, prg)
    kernargs = [(u.dtype, u.src[0].arg) for u in _prg_lin(prg).src if u.op is Ops.INS and u.arg is AMDOps.KERNARG]
    self.assertIn((dtypes.uint32, 8), kernargs)
    insts = list(_REN._insts_from_linear(_prg_lin(prg)))
    self.assertTrue(any(i.op_name == "V_MOV_B32_E32" and i.vdst == amd_lib.TMP_VDATA for i in insts))
    self.assertTrue(any(getattr(i, "op_name", "") == "GLOBAL_STORE_B32" and i.data == amd_lib.TMP_VDATA for i in insts))

  def test_global_range_uses_launch_dims(self):
    prg = _global_dim_program()
    _check_elf(self, prg)
    self.assertNotEqual(prg.arg.global_size, (1, 1, 1))
    self.assertNotEqual(prg.arg.local_size, (1, 1, 1))
    self.assertFalse(any(u.op is Ops.RANGE for u in _prg_lin(prg).src))
    specials = [u.arg for u in prg.src[0].toposort() if u.op is Ops.SPECIAL]
    self.assertIn("gidx0", specials)
    self.assertIn("lidx0", specials)
    desc = _amd_desc(prg)
    self.assertTrue(desc.compute_pgm_rsrc2 & (1 << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT))
    self.assertEqual((desc.compute_pgm_rsrc2 >> amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_SHIFT) & 0x3, 0)
    # 1D locals: USER_SGPR stays 2 so WGID lands at s2 (hand kernels / LLVM).
    self.assertEqual((desc.compute_pgm_rsrc2 >> amdgpu_kd.COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT) & 0x1f, 2)

  def test_half_matmul_2d_locals_pads_user_sgpr_on_gfx1100(self):
    # gfx1100 UserSGPRInit16Bug: 2D locals pad USER_SGPR=15; WGID then lands at s15 (not s2).
    import os
    old = {k: os.environ.get(k) for k in ("TC_LDS_AB", "AMD_D16_HI", "TC_LOCAL")}
    os.environ["TC_LDS_AB"] = "0"
    os.environ["TC_LOCAL"] = "4"
    os.environ.pop("AMD_D16_HI", None)
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      with Context(BEAM=0):
        ast = (Tensor.empty(256, 256, dtype=dtypes.half, device="AMD") @
               Tensor.empty(256, 256, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
        prg = _to_prg(ast.schedule_linear().src[-1].src[0])
      self.assertEqual(prg.arg.local_size, (32, 4, 1))
      desc = _amd_desc(prg)
      self.assertEqual((desc.compute_pgm_rsrc2 >> amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_SHIFT) & 0x3, 1)
      self.assertEqual((desc.compute_pgm_rsrc2 >> amdgpu_kd.COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT) & 0x1f, 15)
      # WGID follows USER_SGPR pad → s15 (SPECIAL in sink is tagged True; phys lives on MOV).
      self.assertTrue(any(getattr(greg(u), "name", None) == "s15" for u in _prg_lin(prg).src))
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()
      to_program_cache.clear()

  def test_lds_load_store_barrier_assembles(self):
    prg = _local_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    for op in (AMDOps.LDS_BASE, AMDOps.LSTORE, AMDOps.BARRIER, AMDOps.LLOAD):
      self.assertIn(op, linear_ops)
    self.assertFalse(any(u.op is Ops.AFTER for u in _prg_lin(prg).src))
    desc = _amd_desc(prg)
    self.assertGreaterEqual(desc.group_segment_fixed_size, 16 * dtypes.uint32.itemsize)
    self.assertEqual(desc.private_segment_fixed_size, 0)
    insts = list(_REN._insts_from_linear(_prg_lin(prg)))
    op_names = [getattr(i, "op_name", "") for i in insts]
    for name in ("DS_STORE_B32", "S_BARRIER", "DS_LOAD_B32"):
      self.assertIn(name, op_names)

  def test_reg_buffer_uses_private_scratch(self):
    prg = _reg_buffer_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    for op in (AMDOps.SCRATCH_SIZE, AMDOps.SCRATCH_ADDR, AMDOps.SSTORE, AMDOps.SLOAD):
      self.assertIn(op, linear_ops)
    desc = _amd_desc(prg)
    self.assertGreaterEqual(desc.private_segment_fixed_size, 16 * dtypes.uint32.itemsize)
    self.assertTrue(desc.compute_pgm_rsrc2 & (1 << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT_SHIFT))
    insts = list(_REN._insts_from_linear(_prg_lin(prg)))
    op_names = [getattr(i, "op_name", "") for i in insts]
    self.assertIn("SCRATCH_STORE_B32", op_names)
    self.assertIn("SCRATCH_LOAD_B32", op_names)
    scratch_ops = [i for i in insts if getattr(i, "op_name", "").startswith("SCRATCH_")]
    self.assertTrue(scratch_ops)
    self.assertTrue(all(i.sve == 1 for i in scratch_ops))

  def test_multiple_reg_buffers_size_private_segment(self):
    prg = _two_reg_buffers_program()
    self.assertGreaterEqual(_amd_desc(prg).private_segment_fixed_size, 100 * dtypes.float32.itemsize + 25 * dtypes.float32.itemsize)

  def test_gated_load_rematerializes_vcc(self):
    prg = _gated_load_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    self.assertGreaterEqual(linear_ops.count(AMDOps.CMPLT), 4)
    self.assertGreaterEqual(linear_ops.count(AMDOps.WHERE), 4)
    self.assertIn(AMDOps.LOAD, linear_ops)

  def test_gated_store_uses_exec_mask_around_store(self):
    lin = _late_gated_store_linear()
    self.assertFalse(any(u.op in (Ops.INDEX, Ops.IF, Ops.ENDIF, Ops.STORE) for u in lin.src))
    masked = [u.arg for u in lin.src if u.op is Ops.INS and u.arg in (AMDOps.IF_MASK, AMDOps.STORE, AMDOps.END_MASK)]
    self.assertEqual(masked, [AMDOps.IF_MASK, AMDOps.STORE, AMDOps.END_MASK])
    inst_names = [getattr(i, "op_name", "") for i in _REN._insts_from_linear(lin)]
    self.assertLess(inst_names.index("S_AND_SAVEEXEC_B64"), inst_names.index("V_CNDMASK_B32_E32"))
    self.assertLess(inst_names.index("V_CNDMASK_B32_E32"), inst_names.index("GLOBAL_STORE_B32"))
    self.assertLess(inst_names.index("GLOBAL_STORE_B32"), inst_names.index("S_MOV_B64"))

  def test_identity_reg_store_skips_noop_scratch_ops(self):
    prg = _identity_reg_store_program()
    _check_elf(self, prg)
    linear_ops = _lin_ops(prg)
    self.assertEqual(linear_ops.count(AMDOps.STORE), 1)
    inst_names = _amd_inst_names(prg)
    self.assertNotIn("SCRATCH_STORE_B32", inst_names)
    self.assertNotIn("SCRATCH_LOAD_B32", inst_names)

  def test_gated_store_materialized_bool_rebuilds_vcc(self):
    inst_names = [getattr(i, "op_name", "") for i in _REN._insts_from_linear(_late_gated_store_linear(True))]
    self.assertLess(inst_names.index("V_CMP_NE_U32_E32"), inst_names.index("S_AND_SAVEEXEC_B64"))

  def test_after_global_load_keeps_64bit_saddr(self):
    prg = _after_global_load_program()
    _check_elf(self, prg)
    renderer = _REN
    loads = [i for i in renderer._insts_from_linear(_prg_lin(prg))
             if getattr(i, "op_name", "") == "GLOBAL_LOAD_B32"]
    self.assertTrue(loads)
    self.assertTrue(all(i.saddr.sz == 2 for i in loads))

  def test_lds_uses_reserved_vgprs_for_addr_and_scalar_data(self):
    insts = list(_REN._insts_from_linear(_prg_lin(_local_sgpr_data_program())))
    ds_ops = [i for i in insts if getattr(i, "op_name", "").startswith("DS_")]
    self.assertTrue(ds_ops)
    self.assertTrue(all(i.addr == amd_lib.TMP_VADDR for i in ds_ops))
    self.assertTrue(any(i.op_name == "V_MOV_B32_E32" and i.vdst == amd_lib.TMP_VDATA for i in insts))
    self.assertTrue(any(i.op_name == "DS_STORE_B32" and i.data0 == amd_lib.TMP_VDATA for i in insts))

  def test_narrow_lds_copy_assembles(self):
    prg = _local_program(dtypes.uint8)
    _check_elf(self, prg)
    self.assertGreaterEqual(_amd_desc(prg).group_segment_fixed_size, 16 * dtypes.uint8.itemsize)
    insts = list(_REN._insts_from_linear(_prg_lin(prg)))
    op_names = [getattr(i, "op_name", "") for i in insts]
    self.assertIn("DS_STORE_B8", op_names)
    self.assertIn("DS_LOAD_U8", op_names)

  def test_byte_lds_lidx0_uses_byte_addr_directly(self):
    # gfx11 packed tid: lidx0 is v_bfe from v0, then that VGPR is the DS address (byte elems, no scale).
    insts = list(_REN._insts_from_linear(_prg_lin(_local_program(dtypes.uint8))))
    bfes = [i for i in insts if getattr(i, "op_name", "") == "V_BFE_U32"]
    self.assertTrue(bfes)
    self.assertEqual(bfes[0].src0, amd_lib.v[0])  # packed work-item word
    ds_ops = [i for i in insts if getattr(i, "op_name", "").startswith("DS_")]
    self.assertTrue(ds_ops)
    self.assertTrue(all(i.addr == bfes[0].vdst for i in ds_ops))

  def test_multiple_lds_buffers_get_distinct_offsets(self):
    prg = _multi_local_program()
    _check_elf(self, prg)
    bases = [u for u in _prg_lin(prg).src if u.op is Ops.INS and u.arg is AMDOps.LDS_BASE]
    self.assertEqual(sorted((u.src[0].arg, u.src[1].arg) for u in bases), [(64, 0), (64, 64)])
    self.assertGreaterEqual(_amd_desc(prg).group_segment_fixed_size, 128)
    insts = list(_REN._insts_from_linear(_prg_lin(prg)))
    self.assertTrue(any(i.op_name == "V_ADD_NC_U32_E64" and i.vdst == amd_lib.TMP_VADDR for i in insts))

  def test_duplicate_lds_slot_rejected(self):
    with self.assertRaises(CompileError):
      _duplicate_local_slot_program()

  def test_gidx_metadata_survives_for_descriptor(self):
    prg = _gidx_program()
    _check_elf(self, prg)
    specials = [u.arg for u in prg.src[0].toposort() if u.op is Ops.SPECIAL]
    self.assertIn("gidx0", specials)

  def test_multi_dim_specials_set_descriptor_bits(self):
    prg = _multi_dim_program()
    _check_elf(self, prg)
    # gfx11: packed work-item IDs in v0 — used lidx dims extracted via V_BFE
    def _bfe_off(i): return int(i.src1.offset) - 128  # inline imm Reg
    bfes = [i for i in _REN._insts_from_linear(_prg_lin(prg)) if getattr(i, "op_name", "") == "V_BFE_U32" and i.src0 == amd_lib.v[0]]
    offs = [_bfe_off(i) for i in bfes]
    self.assertIn(0, offs)
    self.assertIn(10, offs)
    linear_regs = {greg(u).index for u in _prg_lin(prg).src if isinstance(greg(u), Register)}
    self.assertIn(16, linear_regs)  # gidx1 → s16 after USER_SGPR=15 pad
    desc = _amd_desc(prg)
    self.assertTrue(desc.compute_pgm_rsrc2 & (1 << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT))
    self.assertEqual((desc.compute_pgm_rsrc2 >> amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_SHIFT) & 0x3, 1)
    self.assertEqual((desc.compute_pgm_rsrc2 >> amdgpu_kd.COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT) & 0x1f, 15)

  def test_z_dim_specials_set_descriptor_bits(self):
    prg = _z_dim_program()
    _check_elf(self, prg)
    def _bfe_off(i): return int(i.src1.offset) - 128
    bfes = [i for i in _REN._insts_from_linear(_prg_lin(prg)) if getattr(i, "op_name", "") == "V_BFE_U32" and i.src0 == amd_lib.v[0]]
    self.assertIn(20, [_bfe_off(i) for i in bfes])  # lidx2
    linear_regs = {greg(u).index for u in _prg_lin(prg).src if isinstance(greg(u), Register)}
    self.assertIn(17, linear_regs)  # gidx2 → s17 after USER_SGPR=15 pad
    desc = _amd_desc(prg)
    self.assertTrue(desc.compute_pgm_rsrc2 & (1 << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT))
    self.assertEqual((desc.compute_pgm_rsrc2 >> amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_SHIFT) & 0x3, 2)
    self.assertEqual((desc.compute_pgm_rsrc2 >> amdgpu_kd.COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT) & 0x1f, 15)

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_tensor_smoke(self):
    self.assertEqual((Tensor([1, 2, 3], device="AMD") * 2).tolist(), [2, 4, 6])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_large_elementwise_uses_launch_dims_smoke(self):
    out = (Tensor.ones(256, 256, dtype=dtypes.float32, device="AMD") + 1).contiguous().realize()
    flat = out.numpy().reshape(-1)
    self.assertEqual(flat[:4].tolist(), [2.0] * 4)
    self.assertEqual(flat[-4:].tolist(), [2.0] * 4)

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_range_smoke(self):
    out = Tensor.empty(8, dtype=dtypes.uint32, device="AMD").contiguous().realize()
    buf, prg = out._buffer().ensure_allocated(), _range_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, _prg_bin(prg).arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(buf.get_buf("AMD"), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), list(range(1, 9)))

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_two_alu_params_smoke(self):
    inp = Tensor(list(range(16)), dtype=dtypes.uint32, device="AMD").contiguous().realize()
    out = Tensor.empty(16, dtype=dtypes.uint32, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp)], _two_uint_var_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, _prg_bin(prg).arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(2, 3), wait=True)
    self.assertEqual(out.tolist(), [i + 5 for i in range(16)])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_reg_buffer_smoke(self):
    out = Tensor.empty(16, dtype=dtypes.uint32, device="AMD").contiguous().realize()
    buf, prg = out._buffer().ensure_allocated(), _reg_buffer_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, _prg_bin(prg).arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(buf.get_buf("AMD"), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), [i + 6 for i in range(16)])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_gated_load_smoke(self):
    inp0 = Tensor(list(range(16)), dtype=dtypes.uint32, device="AMD").contiguous().realize()
    inp1 = Tensor([100 + i for i in range(16)], dtype=dtypes.uint32, device="AMD").contiguous().realize()
    out = Tensor.empty(16, dtype=dtypes.uint32, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp0, inp1)], _gated_load_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, _prg_bin(prg).arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), [100 + 2*i if i < 4 else i if i < 8 else 0 for i in range(16)])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_sum_smoke(self):
    self.assertEqual(Tensor.ones(256, dtype=dtypes.float32, device="AMD").sum().item(), 256)

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_float_cmpne_nan_smoke(self):
    out = Tensor([float("nan"), 1.0], dtype=dtypes.float32, device="AMD").ne(
      Tensor([float("nan"), 1.0], dtype=dtypes.float32, device="AMD")).cast(dtypes.uint32)
    self.assertEqual(out.tolist(), [1, 0])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_bool_compare_store_smoke(self):
    vals = Tensor([1.0, 2.0, -1.0, 3.0], dtype=dtypes.float32, device="AMD")
    self.assertEqual((vals < 2.0).tolist(), [True, False, True, False])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_float16_where_smoke(self):
    inp0 = Tensor([1.5, 2.5, 3.5, 4.5], dtype=dtypes.float16, device="AMD").contiguous().realize()
    inp1 = Tensor([-1.0, -2.0, -3.0, -4.0], dtype=dtypes.float16, device="AMD").contiguous().realize()
    mask = Tensor([True, False, True, False], dtype=dtypes.bool, device="AMD").contiguous().realize()
    out = Tensor.empty(4, dtype=dtypes.float16, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp0, inp1, mask)], _float16_where_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, _prg_bin(prg).arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), [1.5, -2.0, 3.5, -4.0])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_float16_unary_smoke(self):
    vals = [1.0, 2.0, 4.0, 8.0]
    inp = Tensor(vals, dtype=dtypes.float16, device="AMD").contiguous().realize()
    out = Tensor.empty(4, dtype=dtypes.float16, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp)], _float16_unary_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, _prg_bin(prg).arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    expected = [math.sqrt(x) + math.log2(x) + math.trunc(x + 0.75) - math.trunc(x) + 1.0 / x + math.sin(x) for x in vals]
    for got, exp in zip(out.tolist(), expected):
      self.assertAlmostEqual(got, exp, places=2)

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_float16_fma_smoke(self):
    inp0 = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float16, device="AMD").contiguous().realize()
    inp1 = Tensor([2.0, 3.0, 4.0, 5.0], dtype=dtypes.float16, device="AMD").contiguous().realize()
    out = Tensor.empty(4, dtype=dtypes.float16, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp0, inp1)], _float16_fused_mulacc_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, _prg_bin(prg).arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), [3.0, 7.0, 13.0, 21.0])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_int_narrow_cast_smoke(self):
    vals = Tensor([-1, 0x12345, 32768, -32768], dtype=dtypes.int32, device="AMD")
    self.assertEqual(vals.cast(dtypes.uint16).tolist(), [65535, 0x2345, 32768, 32768])
    self.assertEqual(Tensor.full(4, fill_value=-1, device="AMD").pad(((1, 1),)).cast(dtypes.uint16).tolist(),
                     [0, 65535, 65535, 65535, 65535, 0])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_int_half_cast_smoke(self):
    self.assertEqual(Tensor([-5, 0, 7, 100], dtype=dtypes.int8, device="AMD").cast(dtypes.float16).tolist(), [-5.0, 0.0, 7.0, 100.0])
    self.assertEqual(Tensor([0, 7, 100, 4000], dtype=dtypes.uint16, device="AMD").cast(dtypes.float16).tolist(), [0.0, 7.0, 100.0, 4000.0])
    self.assertEqual(Tensor([True, False, True, False], device="AMD").cast(dtypes.float16).tolist(), [1.0, 0.0, 1.0, 0.0])
    self.assertEqual(Tensor([-5.0, 0.0, 7.0, 100.0], dtype=dtypes.float16, device="AMD").cast(dtypes.int32).tolist(), [-5, 0, 7, 100])

  def test_int_half_cast_routes_through_f32(self):
    prg = _int_to_half_cast_program()
    _check_elf(self, prg)
    inst_names = _amd_inst_names(prg)
    self.assertIn("V_CVT_F32_I32_E32", inst_names)
    self.assertIn("V_CVT_F16_F32_E32", inst_names)
    self.assertIn("V_CVT_F32_F16_E32", inst_names)
    self.assertIn("V_CVT_I32_F32_E32", inst_names)

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_var_divmod_smoke(self):
    inp = Tensor([-7, -7, 7, 7], dtype=dtypes.int32, device="AMD").contiguous().realize()
    div = Tensor([3, -3, 3, -3], dtype=dtypes.int32, device="AMD").contiguous().realize()
    out = Tensor.empty(4, dtype=dtypes.int32, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp, div)], _var_divmod_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, _prg_bin(prg).arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), [-21, 19, 21, -19])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_exp2_smoke(self):
    inp = Tensor([0.0, 1.0, 2.0, 3.0], dtype=dtypes.float32, device="AMD").contiguous().realize()
    out = Tensor.empty(4, dtype=dtypes.float32, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp)], _exp2_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, _prg_bin(prg).arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), [1.0, 2.0, 4.0, 8.0])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_unary_math_smoke(self):
    inp = Tensor([1.0, 4.0, 16.0, 64.0], dtype=dtypes.float32, device="AMD").contiguous().realize()
    out = Tensor.empty(4, dtype=dtypes.float32, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp)], _unary_math_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, _prg_bin(prg).arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), [1.0, 4.0, 8.0, 14.0])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_sin_smoke(self):
    inp = Tensor([-2.0, 0.0, 1.0, math.pi / 2], dtype=dtypes.float32, device="AMD").contiguous().realize()
    out = Tensor.empty(4, dtype=dtypes.float32, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp)], _sin_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, _prg_bin(prg).arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    for got, expected in zip(out.tolist(), [math.sin(x) for x in [-2.0, 0.0, 1.0, math.pi / 2]]):
      self.assertAlmostEqual(got, expected, places=5)

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_random_smoke(self):
    vals = Tensor.rand(10, device="AMD").tolist()
    for x in vals:
      self.assertGreaterEqual(x, 0.0)
      self.assertLess(x, 1.0)

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_spill_smoke(self):
    inp = Tensor(list(range(16)), dtype=dtypes.uint32, device="AMD").contiguous().realize()
    out = Tensor.empty(16, dtype=dtypes.uint32, device="AMD").contiguous().realize()
    out = Tensor.custom_kernel(out, inp, fxn=_custom_renderer_spill)[0].realize()
    self.assertEqual(out.tolist(), [6*x + 15 for x in range(16)])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_lds_smoke(self):
    out = Tensor.empty(16, dtype=dtypes.uint32, device="AMD").contiguous().realize()
    out = Tensor.custom_kernel(out, fxn=_custom_renderer_lds)[0].realize()
    self.assertEqual(out.tolist(), [7] * 16)

if __name__ == "__main__":
  unittest.main()
