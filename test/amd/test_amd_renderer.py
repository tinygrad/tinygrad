import math, struct, unittest
from dataclasses import replace

from tinygrad import Tensor
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, to_program, to_program_cache
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext, pm_regalloc_rewrite
from tinygrad.codegen.opt import KernelOptError, Opt, OptOps
from tinygrad.device import CompileError, Device
from tinygrad.dtype import AddrSpace, Invalid, dtypes
from tinygrad.helpers import Context, Target
from tinygrad.renderer.isa import PreRegAllocContext
from tinygrad.runtime.autogen import amdgpu_kd
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
import tinygrad.renderer.isa.amd as amd_isa
from tinygrad.renderer.isa.amd import AMDRenderer, AMDOps

class TinyVGPRAMDRenderer(AMDRenderer):
  isel_matcher = amd_isa.make_isel_matcher(amd_isa.SGPR, amd_isa.VGPR[:2])

class OneVGPRAMDRenderer(AMDRenderer):
  isel_matcher = amd_isa.make_isel_matcher(amd_isa.SGPR, amd_isa.VGPR[:1])

def _amd_desc(prg):
  _, sections, _ = elf_loader(prg.src[4].arg)
  return amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(next(s.content for s in sections if s.name == ".rodata"))

def _assert_abi_reg_isolation(testcase, prg):
  fixed_sgpr, fixed_vgpr = {0, 1, 2, 3, 4}, {256, 257, 258}
  for u in prg.src[2].src:
    if not isinstance((reg_uop:=getattr(u, "reg", None)), amd_isa.Register): continue
    reg = reg_uop.index
    # Fixed ABI registers should only enter linear IR through hardware SPECIAL definitions.
    if u.op is Ops.INS and u.arg is AMDOps.MOV and u.src and u.src[0].op is Ops.SPECIAL:
      testcase.assertIn(reg, fixed_sgpr | fixed_vgpr)
    else:
      testcase.assertNotIn(reg, fixed_sgpr | fixed_vgpr)
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
  sink = out.index(idx).store(inp.index(idx).load() + UOp.const(dtypes.float, 1.0)).sink(idx, arg=KernelInfo(name="amd_asm_add"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _two_load_add_program():
  out = UOp.placeholder((16,), dtypes.float, 0)
  a = UOp.placeholder((16,), dtypes.float, 1)
  b = UOp.placeholder((16,), dtypes.float, 2)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(a.index(idx).load() + b.index(idx).load()).sink(idx, arg=KernelInfo(name="amd_asm_two_load_add"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _matmul64_program():
  with Context(BEAM=0):
    ast = (Tensor.empty(64, 64, device="AMD") @ Tensor.empty(64, 64, device="AMD")).schedule_linear().src[-1].src[0]
  to_program_cache.clear()
  return to_program(ast, AMDRenderer(Target("AMD", arch="gfx1100")))

def _uint_var_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  var = UOp.param(2, dtypes.uint32, (), vmin_vmax=(0, 10), name="var", addrspace=AddrSpace.ALU)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load() + var).sink(idx, var, arg=KernelInfo(name="amd_asm_uint_var"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _uint_var_mul_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  var = UOp.param(2, dtypes.uint32, (), vmin_vmax=(0, 10), name="var", addrspace=AddrSpace.ALU)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load() * var).sink(idx, var, arg=KernelInfo(name="amd_asm_uint_var_mul"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _two_uint_var_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  var0 = UOp.param(2, dtypes.uint32, (), vmin_vmax=(0, 10), name="var0", addrspace=AddrSpace.ALU)
  var1 = UOp.param(3, dtypes.uint32, (), vmin_vmax=(0, 10), name="var1", addrspace=AddrSpace.ALU)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load() + var0 + var1).sink(idx, var0, var1, arg=KernelInfo(name="amd_asm_two_uint_var"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _uint_wrap_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load() + UOp.const(dtypes.uint32, 0xffffffff)).sink(idx, arg=KernelInfo(name="amd_asm_uint_wrap"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _copy_program(dtype):
  out = UOp.placeholder((16,), dtype, 0)
  inp = UOp.placeholder((16,), dtype, 1)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load()).sink(idx, arg=KernelInfo(name=f"amd_asm_copy_{dtype.name}"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _where_program(dtype):
  out = UOp.placeholder((16,), dtype, 0)
  inp = UOp.placeholder((16,), dtype, 1)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store((inp.index(idx).load() < UOp.const(dtype, 7)).where(inp.index(idx).load(), UOp.const(dtype, 0))) \
            .sink(idx, arg=KernelInfo(name=f"amd_asm_where_{dtype.name}"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _where_sgpr_true_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  idx = UOp.special(16, "lidx0")
  gidx = UOp.special(16, "gidx0").cast(dtypes.uint32)
  val = (idx < UOp.const(dtypes.uint32, 8)).where(gidx, UOp.const(dtypes.uint32, 0))
  sink = out.index(idx).store(val).sink(idx, gidx, arg=KernelInfo(name="amd_asm_where_sgpr_true"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _where_compare_value_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  val = inp.index(idx).load()
  mask = (idx < UOp.const(dtypes.uint32, 8)).where(val == UOp.const(dtypes.uint32, 3), UOp.const(dtypes.bool, False))
  sink = out.index(idx).store(mask.where(UOp.const(dtypes.uint32, 1), UOp.const(dtypes.uint32, 0))) \
            .sink(idx, arg=KernelInfo(name="amd_asm_where_compare_value"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _eq_where_program(dtype):
  out = UOp.placeholder((16,), dtype, 0)
  inp = UOp.placeholder((16,), dtype, 1)
  idx = UOp.special(16, "lidx0")
  val = inp.index(idx).load()
  sink = out.index(idx).store(UOp(Ops.CMPEQ, dtypes.bool, (val, UOp.const(dtype, 7))).where(val, UOp.const(dtype, 0))) \
            .sink(idx, arg=KernelInfo(name=f"amd_asm_eq_where_{dtype.name}"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _float_cmpne_program():
  out = UOp.placeholder((4,), dtypes.uint32, 0)
  inp = UOp.placeholder((4,), dtypes.float32, 1)
  idx = UOp.special(4, "lidx0")
  val = inp.index(idx).load()
  mask = UOp(Ops.CMPNE, dtypes.bool, (val, val))
  sink = out.index(idx).store(mask.where(UOp.const(dtypes.uint32, 1), UOp.const(dtypes.uint32, 0))) \
            .sink(idx, arg=KernelInfo(name="amd_asm_float_cmpne"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _cmpne_compare_flag_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  mask = (inp.index(idx).load() < UOp.const(dtypes.uint32, 10)).ne(False)
  sink = out.index(idx).store(mask.where(UOp.const(dtypes.uint32, 1), UOp.const(dtypes.uint32, 0))) \
            .sink(idx, arg=KernelInfo(name="amd_asm_cmpne_compare_flag"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _bool_and_compare_flags_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  val = inp.index(idx).load()
  mask = val.ne(UOp.const(dtypes.uint32, 0)) & (val < UOp.const(dtypes.uint32, 10))
  sink = out.index(idx).store(mask.where(UOp.const(dtypes.uint32, 1), UOp.const(dtypes.uint32, 0))) \
            .sink(idx, arg=KernelInfo(name="amd_asm_bool_and_compare_flags"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _bool_compare_store_program():
  out = UOp.placeholder((4,), dtypes.bool, 0)
  inp = UOp.placeholder((4,), dtypes.float32, 1)
  idx = UOp.special(4, "lidx0")
  val = inp.index(idx).load()
  sink = out.index(idx).store(val < UOp.const(dtypes.float32, 2.0)).sink(idx, arg=KernelInfo(name="amd_asm_bool_compare_store"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _float16_where_program():
  out = UOp.placeholder((4,), dtypes.float16, 0)
  inp0 = UOp.placeholder((4,), dtypes.float16, 1)
  inp1 = UOp.placeholder((4,), dtypes.float16, 2)
  mask = UOp.placeholder((4,), dtypes.bool, 3)
  idx = UOp.special(4, "lidx0")
  val = mask.index(idx).load().where(inp0.index(idx).load(), inp1.index(idx).load())
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_float16_where"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _float16_cast_program():
  out32 = UOp.placeholder((4,), dtypes.float32, 0)
  out16 = UOp.placeholder((4,), dtypes.float16, 1)
  inp16 = UOp.placeholder((4,), dtypes.float16, 2)
  inp32 = UOp.placeholder((4,), dtypes.float32, 3)
  idx = UOp.special(4, "lidx0")
  st32 = out32.index(idx).store(inp16.index(idx).load().cast(dtypes.float32))
  st16 = out16.index(idx).store(inp32.index(idx).load().cast(dtypes.float16))
  sink = UOp.sink(st32, st16, idx, arg=KernelInfo(name="amd_asm_float16_cast"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _int_to_half_cast_program():
  out16 = UOp.placeholder((4,), dtypes.float16, 0)
  out32 = UOp.placeholder((4,), dtypes.int32, 1)
  inp32 = UOp.placeholder((4,), dtypes.int32, 2)
  inp16 = UOp.placeholder((4,), dtypes.float16, 3)
  idx = UOp.special(4, "lidx0")
  sti = out16.index(idx).store(inp32.index(idx).load().cast(dtypes.float16))
  stf = out32.index(idx).store(inp16.index(idx).load().cast(dtypes.int32))
  sink = UOp.sink(sti, stf, idx, arg=KernelInfo(name="amd_asm_int_half_cast"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _bfloat16_store_program():
  out = UOp.placeholder((16,), dtypes.bfloat16, 0)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(UOp.const(dtypes.float32, 1.0).cast(dtypes.bfloat16)).sink(idx, arg=KernelInfo(name="amd_asm_bfloat16_store"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _emulated_uint64_upcast_program():
  with Context(EMULATED_DTYPES="long"):
    ast = ((Tensor([1], dtype=dtypes.uint8, device="AMD").cast(dtypes.uint64) +
            Tensor([1], dtype=dtypes.uint8, device="AMD").cast(dtypes.uint64)).cast(dtypes.uint8)).schedule_linear().src[-1].src[0]
  to_program_cache.clear()
  return to_program(ast, AMDRenderer(Target("AMD", arch="gfx1100")))

def _emulated_int64_cmod_const_program():
  with Context(EMULATED_DTYPES="long"):
    ast = ((Tensor([7], dtype=dtypes.int64, device="AMD") % 3).cast(dtypes.int32)).schedule_linear().src[-1].src[0]
  to_program_cache.clear()
  return to_program(ast, AMDRenderer(Target("AMD", arch="gfx1100")))

def _emulated_int64_index_cmod_program():
  out = UOp.placeholder((8,), dtypes.float32, 0)
  idx = UOp.special(8, "lidx0")
  long_idx = UOp(Ops.CMOD, dtypes.long, (idx.cast(dtypes.long), UOp.const(dtypes.long, 8)))
  sink = out.index(long_idx).store(UOp.const(dtypes.float32, 1.0)).sink(idx, arg=KernelInfo(name="amd_asm_long_index_cmod"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _narrow_var_mod_program(dtype):
  ast = (Tensor([0], dtype=dtype, device="AMD") % Tensor([1], dtype=dtype, device="AMD")).schedule_linear().src[-1].src[0]
  to_program_cache.clear()
  return to_program(ast, AMDRenderer(Target("AMD", arch="gfx1100")))

def _software_sin_lowered_sinks():
  # TRANSCENDENTAL=2 forces tinygrad's software xsin. Its Payne-Hanek range reduction emits ulong // and %
  # (see codegen/decomp/transcendental.py) AFTER the early dtype-decomp pass. AMD emulates 64-bit ints, so
  # codegen must run dtype decomposition again, otherwise raw 64-bit CDIV/CMOD reach regalloc and crash.
  # Lowering only (pre-regalloc) keeps this fast; the slow part of software sin is regalloc over the big graph.
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  with Context(TRANSCENDENTAL=2):
    cl = Tensor([1e6], device="AMD").sin().schedule_linear()
    asts = [si.src[0] for si in cl.src if si.src and si.src[0].op is Ops.SINK]
    return [full_rewrite_to_sink(ast, ren) for ast in asts]

def _atomic_add_program():
  out = UOp.placeholder((16,), dtypes.float32, 0)
  inp = UOp.placeholder((16,), dtypes.float32, 1)
  idx = UOp.special(16, "lidx0")
  atomic = UOp(Ops.CUSTOM, dtypes.void, (out.index(idx, ptr=True), inp.index(idx).load()), arg=amd_isa.AMD_ATOMIC_ADD)
  sink = UOp.sink(atomic, idx, arg=KernelInfo(name="amd_asm_atomic_add"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _int_narrow_cast_program():
  out = UOp.placeholder((4,), dtypes.uint16, 0)
  inp = UOp.placeholder((4,), dtypes.int32, 1)
  idx = UOp.special(4, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load().cast(dtypes.uint16)).sink(idx, arg=KernelInfo(name="amd_asm_int_narrow_cast"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _int_signed_narrow_cast_program():
  out = UOp.placeholder((4,), dtypes.int16, 0)
  inp = UOp.placeholder((4,), dtypes.int32, 1)
  idx = UOp.special(4, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load().cast(dtypes.int16)).sink(idx, arg=KernelInfo(name="amd_asm_int_signed_narrow_cast"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _int_signed_widen_cast_program():
  out = UOp.placeholder((4,), dtypes.int32, 0)
  inp = UOp.placeholder((4,), dtypes.int16, 1)
  idx = UOp.special(4, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load().cast(dtypes.int32)).sink(idx, arg=KernelInfo(name="amd_asm_int_signed_widen_cast"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _float16_unary_program():
  out = UOp.placeholder((4,), dtypes.float16, 0)
  inp = UOp.placeholder((4,), dtypes.float16, 1)
  idx = UOp.special(4, "lidx0")
  x = inp.index(idx).load()
  val = UOp(Ops.SQRT, dtypes.float16, (x,)) + UOp(Ops.LOG2, dtypes.float16, (x,))
  val = val + UOp(Ops.TRUNC, dtypes.float16, (x + UOp.const(dtypes.float16, 0.75),)) - UOp(Ops.TRUNC, dtypes.float16, (x,))
  val = val + UOp(Ops.RECIPROCAL, dtypes.float16, (x,)) + UOp(Ops.SIN, dtypes.float16, (x,))
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_float16_unary"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _bitwise_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  val = (((inp.index(idx).load() & UOp.const(dtypes.uint32, 0xff)) | UOp.const(dtypes.uint32, 0x10)) ^ UOp.const(dtypes.uint32, 0x3)) >> UOp.const(dtypes.uint32, 1)
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_bitwise"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _cmod_pow2_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  val = UOp(Ops.CMOD, dtypes.uint32, (inp.index(idx).load() + idx.cast(dtypes.uint32), UOp.const(dtypes.uint32, 8)))
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_cmod_pow2"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _const_divmod_program():
  out = UOp.placeholder((16,), dtypes.int32, 0)
  idx = UOp.special(16, "lidx0").cast(dtypes.int32) + UOp.const(dtypes.int32, 7)
  val = UOp(Ops.CDIV, dtypes.int32, (idx, UOp.const(dtypes.int32, 11))) + UOp(Ops.CMOD, dtypes.int32, (idx, UOp.const(dtypes.int32, 11)))
  sink = out.index(UOp.special(16, "lidx0")).store(val).sink(arg=KernelInfo(name="amd_asm_const_divmod"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _var_divmod_program():
  out = UOp.placeholder((4,), dtypes.int32, 0)
  inp = UOp.placeholder((4,), dtypes.int32, 1)
  div = UOp.placeholder((4,), dtypes.int32, 2)
  idx = UOp.special(4, "lidx0")
  x, d = inp.index(idx).load(), div.index(idx).load()
  q, r = UOp(Ops.CDIV, dtypes.int32, (x, d)), UOp(Ops.CMOD, dtypes.int32, (x, d))
  sink = out.index(idx).store(q * UOp.const(dtypes.int32, 10) + r).sink(idx, arg=KernelInfo(name="amd_asm_var_divmod"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _max_program(dtype):
  out = UOp.placeholder((16,), dtype, 0)
  inp = UOp.placeholder((16,), dtype, 1)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(UOp(Ops.MAX, dtype, (inp.index(idx).load(), UOp.const(dtype, 7)))) \
            .sink(idx, arg=KernelInfo(name=f"amd_asm_max_{dtype.name}"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _mulacc_program():
  out = UOp.placeholder((16,), dtypes.float32, 0)
  inp0 = UOp.placeholder((16,), dtypes.float32, 1)
  inp1 = UOp.placeholder((16,), dtypes.float32, 2)
  idx = UOp.special(16, "lidx0")
  val = UOp(Ops.MULACC, dtypes.float32, (inp0.index(idx).load(), inp1.index(idx).load(), UOp.const(dtypes.float32, 1.0)))
  sink = out.index(idx).store(val) \
            .sink(idx, arg=KernelInfo(name="amd_asm_mulacc"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _fused_mulacc_program():
  out = UOp.placeholder((16,), dtypes.float32, 0)
  inp0 = UOp.placeholder((16,), dtypes.float32, 1)
  inp1 = UOp.placeholder((16,), dtypes.float32, 2)
  idx = UOp.special(16, "lidx0")
  val = inp0.index(idx).load() * inp1.index(idx).load() + UOp.const(dtypes.float32, 1.0)
  sink = out.index(idx).store(val) \
            .sink(idx, arg=KernelInfo(name="amd_asm_fused_mulacc"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _float16_fused_mulacc_program():
  out = UOp.placeholder((4,), dtypes.float16, 0)
  inp0 = UOp.placeholder((4,), dtypes.float16, 1)
  inp1 = UOp.placeholder((4,), dtypes.float16, 2)
  idx = UOp.special(4, "lidx0")
  val = inp0.index(idx).load() * inp1.index(idx).load() + UOp.const(dtypes.float16, 1.0)
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_float16_fused_mulacc"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _cast_reciprocal_program():
  out = UOp.placeholder((16,), dtypes.float32, 0)
  idx = UOp.special(16, "lidx0")
  val = (idx.cast(dtypes.int32) + UOp.const(dtypes.int32, 1)).cast(dtypes.float32).reciprocal()
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_cast_reciprocal"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _float_to_int_cast_program():
  out = UOp.placeholder((16,), dtypes.int32, 0)
  inp = UOp.placeholder((16,), dtypes.float32, 1)
  idx = UOp.special(16, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load().cast(dtypes.int32)).sink(idx, arg=KernelInfo(name="amd_asm_float_to_int_cast"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _exp2_program():
  out = UOp.placeholder((4,), dtypes.float32, 0)
  inp = UOp.placeholder((4,), dtypes.float32, 1)
  idx = UOp.special(4, "lidx0")
  sink = out.index(idx).store(inp.index(idx).load().exp2()).sink(idx, arg=KernelInfo(name="amd_asm_exp2"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _unary_math_program():
  out = UOp.placeholder((4,), dtypes.float32, 0)
  inp = UOp.placeholder((4,), dtypes.float32, 1)
  idx = UOp.special(4, "lidx0")
  x = inp.index(idx).load()
  val = UOp(Ops.SQRT, dtypes.float32, (x,)) + UOp(Ops.LOG2, dtypes.float32, (x,))
  val = val + UOp(Ops.TRUNC, dtypes.float32, (x + UOp.const(dtypes.float32, 0.75),)) - UOp(Ops.TRUNC, dtypes.float32, (x,))
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_unary_math"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _sin_program():
  out = UOp.placeholder((4,), dtypes.float32, 0)
  inp = UOp.placeholder((4,), dtypes.float32, 1)
  idx = UOp.special(4, "lidx0")
  sink = out.index(idx).store(UOp(Ops.SIN, dtypes.float32, (inp.index(idx).load(),))).sink(idx, arg=KernelInfo(name="amd_asm_sin"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _spill_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  vals = [inp.index(idx).load() + UOp.const(dtypes.uint32, i) for i in range(6)]
  acc = vals[0]
  for v in vals[1:]: acc = acc + v
  sink = out.index(idx).store(acc).sink(idx, arg=KernelInfo(name="amd_asm_spill"))
  to_program_cache.clear()
  return to_program(sink, TinyVGPRAMDRenderer(Target("AMD", arch="gfx1100")))

def _multi_spill_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp = UOp.placeholder((16,), dtypes.uint32, 1)
  idx = UOp.special(16, "lidx0")
  vals = [inp.index(idx).load() + UOp.const(dtypes.uint32, i) for i in range(4)]
  acc = vals[0]
  for v in vals[1:]: acc = acc + v
  sink = out.index(idx).store(acc).sink(idx, arg=KernelInfo(name="amd_asm_multi_spill"))
  to_program_cache.clear()
  return to_program(sink, OneVGPRAMDRenderer(Target("AMD", arch="gfx1100")))

def _range_program():
  out = UOp.placeholder((8,), dtypes.uint32, 0)
  rng = UOp.range(8, 0, AxisType.LOOP)
  sink = UOp(Ops.SINK, dtypes.void,
             (UOp(Ops.END, dtypes.void, (out.index(rng).store(rng.cast(dtypes.uint32) + UOp.const(dtypes.uint32, 1)), rng)),),
             arg=KernelInfo(name="amd_asm_range"), tag=1)
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _nested_range_program():
  out = UOp.placeholder((32,), dtypes.uint32, 0)
  r0, r1 = UOp.range(4, 0, AxisType.LOOP), UOp.range(8, 1, AxisType.LOOP)
  idx = (r0.cast(dtypes.uint32) << UOp.const(dtypes.uint32, 3)) + r1.cast(dtypes.uint32)
  sink = out.index(idx).store(idx).end(r0, r1).sink(arg=KernelInfo(name="amd_asm_nested_range")).replace(tag=1)
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _var_range_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  n = UOp.param(1, dtypes.uint32, (), vmin_vmax=(0, 16), name="n", addrspace=AddrSpace.ALU)
  r = UOp.range(n, 0, AxisType.LOOP)
  sink = out.index(r).store(r.cast(dtypes.uint32)).end(r).sink(n, arg=KernelInfo(name="amd_asm_var_range")).replace(tag=1)
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _loop_vcc_remat_program():
  out = UOp.placeholder((4,), dtypes.uint32, 0)
  idx = UOp.special(4, "lidx0")
  r = UOp.range(2, 0, AxisType.LOOP)
  gate0, gate1 = idx < UOp.const(dtypes.uint32, 2), idx < UOp.const(dtypes.uint32, 3)
  val = gate0.where(r.cast(dtypes.uint32) + UOp.const(dtypes.uint32, 1), UOp.const(dtypes.uint32, 0))
  val = val + gate1.where(r.cast(dtypes.uint32) + UOp.const(dtypes.uint32, 2), UOp.const(dtypes.uint32, 0))
  sink = out.index(idx).store(val).end(r).sink(idx, arg=KernelInfo(name="amd_asm_loop_vcc_remat")).replace(tag=1)
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _global_dim_program():
  out = UOp.placeholder((1024,), dtypes.float32, 0)
  rng = UOp.range(1024, 0, AxisType.GLOBAL)
  sink = out.index(rng).store(UOp.const(dtypes.float32, 1.0)).end(rng).sink(arg=KernelInfo(name="amd_asm_global_dim"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _local_program(dtype=dtypes.uint32, slot=0):
  out = UOp.placeholder((16,), dtype, 0)
  smem = UOp.placeholder((16,), dtype, slot=slot, addrspace=AddrSpace.LOCAL)
  idx = UOp.special(16, "lidx0")
  st = smem.index(idx).store(UOp.const(dtype, 7))
  barr = UOp(Ops.BARRIER, dtypes.void, (st,))
  ld = smem.after(barr).index(idx).load()
  sink = out.index(idx).store(ld).sink(idx, arg=KernelInfo(name=f"amd_asm_local_{dtype.name}_{slot}"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _reg_buffer_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  scratch = UOp.placeholder((16,), dtypes.uint32, slot=0, addrspace=AddrSpace.REG)
  idx = UOp.special(16, "lidx0")
  st = scratch.index(idx).store(idx.cast(dtypes.uint32) + UOp.const(dtypes.uint32, 1))
  ld = scratch.after(st).index(idx).load()
  sink = out.index(idx).store(ld + UOp.const(dtypes.uint32, 5)).sink(idx, arg=KernelInfo(name="amd_asm_reg_buffer"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _two_reg_buffers_program():
  out = UOp.placeholder((100,), dtypes.float32, 0)
  scratch0 = UOp.placeholder((100,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  scratch1 = UOp.placeholder((25,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
  idx = UOp.special(100, "lidx0")
  idx25 = idx % UOp.const(dtypes.uint32, 25)
  st0 = scratch0.index(idx).store(UOp.const(dtypes.float32, 1.0))
  st1 = scratch1.index(idx25).store(UOp.const(dtypes.float32, 2.0))
  val = scratch0.after(st0).index(idx).load() + scratch1.after(st1).index(idx25).load()
  sink = out.index(idx).store(val).sink(idx, arg=KernelInfo(name="amd_asm_two_reg_buffers"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _gated_load_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  inp0 = UOp.placeholder((16,), dtypes.uint32, 1)
  inp1 = UOp.placeholder((16,), dtypes.uint32, 2)
  idx = UOp.special(16, "lidx0")
  gate0 = idx < UOp.const(dtypes.uint32, 8)
  gate1 = idx < UOp.const(dtypes.uint32, 4)
  val0 = gate0.where(inp0.index(gate0.where(idx, idx.const_like(Invalid))).load(), UOp.const(dtypes.uint32, 0))
  val1 = gate1.where(inp1.index(gate1.where(idx, idx.const_like(Invalid))).load(), UOp.const(dtypes.uint32, 0))
  sink = out.index(idx).store(val0 + val1).sink(idx, arg=KernelInfo(name="amd_asm_gated_load"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _late_gated_store_linear(materialized_gate=False):
  renderer = AMDRenderer(Target("AMD", arch="gfx1100"))
  out = UOp(Ops.INS, dtypes.uint64, (UOp.const(dtypes.int32, 0).rtag(),), AMDOps.KERNARG, (amd_isa.Register("out", 0, _cons=amd_isa.SGPR),))
  inp = UOp(Ops.INS, dtypes.uint64, (UOp.const(dtypes.int32, 8).rtag(),), AMDOps.KERNARG, (amd_isa.Register("inp", 1, _cons=amd_isa.SGPR),))
  idx = UOp(Ops.INS, dtypes.uint32, (UOp.special(16, "lidx0").rtag(),), AMDOps.MOV, (amd_isa.Register("idx", 2, _cons=(amd_isa.LID[0],)),))
  val = UOp(Ops.INS, dtypes.uint32, (inp, idx), AMDOps.LOAD, (amd_isa.Register("val", 3, _cons=amd_isa.VGPR),))
  gate = UOp(Ops.INS, dtypes.bool, (idx, UOp.const(dtypes.uint32, 8).rtag()), AMDOps.CMPLT)
  one = None
  if materialized_gate:
    one = UOp(Ops.INS, dtypes.uint32, (UOp.const(dtypes.uint32, 1).rtag(),), AMDOps.MOV,
              (amd_isa.Register("one", 4, _cons=amd_isa.VGPR),))
    gate = UOp(Ops.INS, dtypes.bool, (idx, one), AMDOps.AND,
               (amd_isa.Register("gate", 5, _cons=amd_isa.VGPR),))
  addr = UOp(Ops.INDEX, dtypes.uint32, (out, idx))
  mif = UOp(Ops.IF, dtypes.void, (gate, addr))
  st = UOp(Ops.STORE, dtypes.void, (addr, val))
  mend = UOp(Ops.ENDIF, dtypes.void, (mif,))
  lst = line_rewrite([u for u in (out, inp, idx, val, one, gate, addr, mif, st, mend) if u is not None], renderer.pre_regalloc_matcher, PreRegAllocContext())
  lst = sorted(lst, key=lambda u: u.op is not Ops.INS or bool(u.src))
  regalloc_ctx = LinearScanRegallocContext(lst, renderer)
  lst = line_rewrite(lst, pm_regalloc_rewrite, regalloc_ctx)
  lst = line_rewrite(lst, renderer.post_regalloc_matcher, regalloc_ctx)
  return UOp(Ops.LINEAR, src=tuple(lst))

def _after_global_load_program():
  tmp = UOp.placeholder((16,), dtypes.float32, 0)
  out = UOp.placeholder((16,), dtypes.float32, 1)
  idx = UOp.special(16, "lidx0")
  st = tmp.index(idx).store(UOp.const(dtypes.float32, 3.0))
  ld = tmp.after(st).index(idx).load()
  sink = out.index(idx).store(ld + UOp.const(dtypes.float32, 1.0)).sink(idx, arg=KernelInfo(name="amd_asm_after_global_load"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _local_sgpr_data_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  smem = UOp.placeholder((16,), dtypes.uint32, slot=0, addrspace=AddrSpace.LOCAL)
  val = UOp.param(1, dtypes.uint32, (), vmin_vmax=(0, 16), name="val", addrspace=AddrSpace.ALU)
  idx = UOp.special(16, "lidx0")
  st = smem.index(idx).store(val)
  barr = UOp(Ops.BARRIER, dtypes.void, (st,))
  ld = smem.after(barr).index(idx).load()
  sink = out.index(idx).store(ld).sink(idx, val, arg=KernelInfo(name="amd_asm_local_sgpr_data"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _multi_local_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  smem0 = UOp.placeholder((16,), dtypes.uint32, slot=0, addrspace=AddrSpace.LOCAL)
  smem1 = UOp.placeholder((16,), dtypes.uint32, slot=1, addrspace=AddrSpace.LOCAL)
  idx = UOp.special(16, "lidx0")
  st0 = smem0.index(idx).store(UOp.const(dtypes.uint32, 7))
  st1 = smem1.index(idx).store(UOp.const(dtypes.uint32, 11))
  barr = UOp(Ops.BARRIER, dtypes.void, (st0, st1))
  ld0, ld1 = smem0.after(barr).index(idx).load(), smem1.after(barr).index(idx).load()
  sink = out.index(idx).store(ld0 + ld1).sink(idx, arg=KernelInfo(name="amd_asm_multi_local"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _duplicate_local_slot_program():
  out = UOp.placeholder((16,), dtypes.uint32, 0)
  smem0 = UOp.placeholder((16,), dtypes.uint32, slot=0, addrspace=AddrSpace.LOCAL)
  smem1 = UOp.placeholder((8,), dtypes.uint32, slot=0, addrspace=AddrSpace.LOCAL)
  idx = UOp.special(8, "lidx0")
  st0 = smem0.index(idx).store(UOp.const(dtypes.uint32, 7))
  st1 = smem1.index(idx).store(UOp.const(dtypes.uint32, 11))
  sink = out.index(idx).store(UOp.const(dtypes.uint32, 0)).sink(idx, st0, st1, arg=KernelInfo(name="amd_asm_duplicate_local"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))


def _custom_renderer_spill(out:UOp, inp:UOp) -> UOp:
  out, inp = out.flatten(), inp.flatten()
  idx = UOp.special(out.numel(), "lidx0")
  vals = [inp.base.index(idx).load() + UOp.const(dtypes.uint32, i) for i in range(6)]
  acc = vals[0]
  for v in vals[1:]: acc = acc + v
  sink = out.base.index(idx).store(acc).sink(idx, arg=KernelInfo(name="amd_asm_hw_spill"))
  return to_program(sink, TinyVGPRAMDRenderer(Target("AMD", arch=Device["AMD"].arch)))

def _custom_renderer_lds(out:UOp) -> UOp:
  out = out.flatten()
  idx = UOp.special(out.numel(), "lidx0")
  smem = UOp.placeholder((out.numel(),), dtypes.uint32, slot=0, addrspace=AddrSpace.LOCAL)
  st = smem.index(idx).store(UOp.const(dtypes.uint32, 7))
  barr = UOp(Ops.BARRIER, dtypes.void, (st,))
  ld = smem.after(barr).index(idx).load()
  sink = out.base.index(idx).store(ld).sink(idx, arg=KernelInfo(name="amd_asm_hw_lds"))
  return to_program(sink, AMDRenderer(Target("AMD", arch=Device["AMD"].arch)))

def _has_amd_asm_runtime() -> bool:
  return Device.DEFAULT == "AMD" and isinstance(Device["AMD"].renderer, AMDRenderer) and Device["AMD"].arch.startswith("gfx11")

def _gidx_program():
  out = UOp.placeholder((64,), dtypes.uint32, 0)
  idx = UOp.special(16, "lidx0") + (UOp.special(4, "gidx0") << UOp.const(dtypes.uint32, 4))
  sink = out.index(idx).store(idx).sink(arg=KernelInfo(name="amd_asm_gidx"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _multi_dim_program():
  out = UOp.placeholder((128,), dtypes.uint32, 0)
  lidx0, lidx1 = UOp.special(8, "lidx0"), UOp.special(4, "lidx1")
  gidx1 = UOp.special(4, "gidx1")
  idx = lidx0 + (lidx1 << UOp.const(dtypes.uint32, 3)) + (gidx1 << UOp.const(dtypes.uint32, 5))
  sink = out.index(idx).store(idx).sink(arg=KernelInfo(name="amd_asm_multi_dim"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

def _z_dim_program():
  out = UOp.placeholder((256,), dtypes.uint32, 0)
  lidx2 = UOp.special(4, "lidx2")
  gidx2 = UOp.special(4, "gidx2")
  idx = lidx2 + (gidx2 << UOp.const(dtypes.uint32, 2))
  sink = out.index(idx).store(idx).sink(arg=KernelInfo(name="amd_asm_z_dim"))
  to_program_cache.clear()
  return to_program(sink, AMDRenderer(Target("AMD", arch="gfx1100")))

class TestAMDRenderer(unittest.TestCase):
  def test_rejects_non_rdna3(self):
    with self.assertRaises(RuntimeError):
      AMDRenderer(Target("AMD", arch="gfx1200"))

  def test_advertises_scheduler_locals_and_shared_memory(self):
    renderer = AMDRenderer(Target("AMD", arch="gfx1100"))
    self.assertTrue(renderer.has_shared)
    self.assertTrue(renderer.has_local)
    self.assertFalse(renderer.supports_float4)
    self.assertEqual(renderer.local_prod_max, 1024)

  def test_scheduler_rejects_oversized_local_workgroup(self):
    ast = Tensor.empty(4000, device="AMD").sum().schedule_linear().src[0].src[0]
    with self.assertRaises(KernelOptError):
      to_program(ast.replace(arg=replace(ast.arg, opts_to_apply=(Opt(OptOps.GROUP, 0, 0),))), AMDRenderer(Target("AMD", arch="gfx1100")))

  def test_to_program_assembles_elf(self):
    prg = _simple_add_program()
    self.assertIs(prg.src[3].op, Ops.SOURCE)
    self.assertIs(prg.src[4].op, Ops.BINARY)
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    self.assertIn("load", prg.src[3].arg.lower())
    self.assertIn("store", prg.src[3].arg.lower())
    self.assertEqual(_amd_desc(prg).kernarg_size, 16)

  def test_program_estimates_survive_instruction_selection(self):
    est = _simple_add_program().src[0].arg.estimates
    self.assertIsNotNone(est)
    self.assertGreater(est.ops, 0)
    self.assertGreater(est.mem, 0)

  def test_linear_contains_amd_ops(self):
    prg = _simple_add_program()
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.KERNARG, linear_ops)
    self.assertIn(AMDOps.LOAD, linear_ops)
    self.assertIn(AMDOps.STORE, linear_ops)

  def test_two_global_loads_share_waitcnt(self):
    prg = _two_load_add_program()
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
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

  def test_uint32_alu_param_offsets(self):
    prg = _uint_var_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    kernarg_offsets = [u.src[0].arg for u in prg.src[2].src if u.op is Ops.INS and u.arg is AMDOps.KERNARG]
    self.assertEqual(sorted(kernarg_offsets), [0, 8, 16])
    self.assertEqual(_amd_desc(prg).kernarg_size, 20)

  def test_multiple_alu_params_are_dense_after_buffers(self):
    prg = _two_uint_var_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    kernarg_offsets = [u.src[0].arg for u in prg.src[2].src if u.op is Ops.INS and u.arg is AMDOps.KERNARG]
    self.assertEqual(sorted(kernarg_offsets), [0, 8, 16, 20])
    self.assertEqual(_amd_desc(prg).kernarg_size, 24)

  def test_narrow_global_copy_assembles(self):
    for dtype in (dtypes.bool, dtypes.uint8, dtypes.int8, dtypes.uint16, dtypes.int16):
      with self.subTest(dtype=dtype):
        prg = _copy_program(dtype)
        self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))

  def test_reserved_v254_v255_not_allocated(self):
    for prg in (_simple_add_program(), _range_program(), _var_range_program()):
      with self.subTest(name=prg.arg.name):
        regs = [u.reg.index for u in prg.src[2].src if isinstance(getattr(u, "reg", None), amd_isa.Register)]
        self.assertLess(max(regs, default=0), 256 + 254)

  def test_abi_fixed_registers_are_not_temp_allocated(self):
    for prg in (_multi_dim_program(), _z_dim_program(), _uint_var_program(), _global_dim_program(), _spill_program(), _multi_spill_program(), _local_program(), _multi_local_program()):
      with self.subTest(name=prg.arg.name):
        _assert_abi_reg_isolation(self, prg)

  def test_kernarg_sgpr_pairs_do_not_overlap(self):
    prg = _uint_var_program()
    kernarg_bases = [u.reg.index for u in prg.src[2].src if u.op is Ops.INS and u.arg is AMDOps.KERNARG and u.dtype.itemsize == 8]
    self.assertEqual(kernarg_bases, sorted(kernarg_bases))
    for a, b in zip(kernarg_bases, kernarg_bases[1:]):
      self.assertGreaterEqual(b - a, 2)

  def test_linear_has_no_explicit_end_op(self):
    for prg in (_simple_add_program(), _range_program(), _nested_range_program(), _var_range_program()):
      with self.subTest(name=prg.arg.name):
        self.assertFalse(any(u.op is Ops.END for u in prg.src[2].src))
        linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
        self.assertNotIn("END", [getattr(op, "name", op) for op in linear_ops])

  def test_compare_where_assembles(self):
    for dtype in (dtypes.uint32, dtypes.int32, dtypes.float32):
      with self.subTest(dtype=dtype):
        prg = _where_program(dtype)
        self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
        linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
        self.assertIn(AMDOps.CMPLT, linear_ops)
        self.assertIn(AMDOps.WHERE, linear_ops)

  def test_where_sgpr_true_materializes_vsrc1(self):
    prg = _where_sgpr_true_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
    cndmask = next(i for i in insts if getattr(i, "op_name", "") == "V_CNDMASK_B32_E32")
    self.assertEqual(cndmask.vsrc1, amd_isa.TMP_VDATA)
    self.assertTrue(any(getattr(i, "op_name", "") == "V_MOV_B32_E32" and i.vdst == amd_isa.TMP_VDATA for i in insts))

  def test_where_compare_value_materializes_flag(self):
    prg = _where_compare_value_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    compare_ops = {AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ}
    for u in prg.src[2].src:
      if u.op is Ops.INS and u.arg is AMDOps.WHERE:
        for value in u.src[1:]:
          self.assertFalse(value.op is Ops.INS and value.arg in compare_ops)

  def test_eq_where_assembles(self):
    for dtype in (dtypes.uint32, dtypes.int32, dtypes.float32):
      with self.subTest(dtype=dtype):
        prg = _eq_where_program(dtype)
        self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
        linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
        self.assertIn(AMDOps.CMPEQ, linear_ops)
        self.assertIn(AMDOps.WHERE, linear_ops)

  def test_float_cmpne_uses_float_compare(self):
    prg = _float_cmpne_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.CMPNE, linear_ops)
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
    self.assertIn("V_CMP_NEQ_F32_E32", [getattr(i, "op_name", "") for i in insts])

  def test_cmpne_compare_flag_simplifies(self):
    prg = _cmpne_compare_flag_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.CMPLT, linear_ops)
    self.assertIn(AMDOps.WHERE, linear_ops)
    self.assertNotIn(AMDOps.CMPNE, linear_ops)

  def test_bool_and_compare_flags_materializes(self):
    prg = _bool_and_compare_flags_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.CMPLT, linear_ops)
    self.assertIn(AMDOps.CMPNE, linear_ops)
    self.assertIn(AMDOps.AND, linear_ops)
    self.assertGreaterEqual(linear_ops.count(AMDOps.WHERE), 2)

  def test_bool_compare_store_materializes_flag(self):
    prg = _bool_compare_store_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.CMPLT, linear_ops)
    self.assertIn(AMDOps.WHERE, linear_ops)
    self.assertIn(AMDOps.STORE, linear_ops)

  def test_float16_where_assembles(self):
    prg = _float16_where_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.WHERE, linear_ops)
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
    for name in ("GLOBAL_LOAD_U16", "GLOBAL_STORE_B16", "V_CNDMASK_B32_E32"):
      self.assertIn(name, inst_names)

  def test_float16_cast_assembles(self):
    prg = _float16_cast_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.CAST, linear_ops)
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
    for name in ("V_CVT_F32_F16_E32", "V_CVT_F16_F32_E32"):
      self.assertIn(name, inst_names)

  def test_bfloat16_tagged_param_lowers_to_kernarg(self):
    prg = _bfloat16_store_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    self.assertFalse(any(u.op is Ops.PARAM for u in prg.src[2].src))
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
    self.assertIn("GLOBAL_STORE_B16", inst_names)

  def test_emulated_uint64_upcast_assembles(self):
    prg = _emulated_uint64_upcast_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    self.assertFalse(any(u.op is Ops.CAST for u in prg.src[2].src))

  def test_emulated_int64_cmod_const_assembles(self):
    prg = _emulated_int64_cmod_const_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))

  def test_emulated_int64_index_cmod_assembles(self):
    prg = _emulated_int64_index_cmod_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    self.assertFalse(any(u.op is Ops.CMOD for u in prg.src[2].src))

  def test_narrow_variable_mod_widens_before_regalloc(self):
    for dtype in (dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16):
      with self.subTest(dtype=dtype):
        prg = _narrow_var_mod_program(dtype)
        self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
        self.assertFalse(any(u.op is Ops.CMOD for u in prg.src[2].src))

  def test_software_sin_decomposes_late_64bit_divmod(self):
    sinks = _software_sin_lowered_sinks()
    self.assertTrue(sinks)
    bad = [u for sink in sinks for u in sink.toposort()
           if u.op in (Ops.CDIV, Ops.CMOD) and u.dtype.scalar() in (dtypes.long, dtypes.ulong)]
    self.assertEqual(bad, [], f"{len(bad)} raw 64-bit CDIV/CMOD survived lowering (would crash regalloc)")

  def test_custom_atomic_add_lowers_to_global_atomic(self):
    prg = _atomic_add_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    self.assertFalse(any(u.op is Ops.CUSTOM for u in prg.src[2].src))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.ATOMIC_ADD, linear_ops)
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
    self.assertIn("GLOBAL_ATOMIC_ADD_F32", inst_names)
    self.assertIn("S_WAITCNT_VMCNT", inst_names)

  def test_int_narrow_cast_assembles(self):
    prg = _int_narrow_cast_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.CAST, linear_ops)
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
    self.assertIn("V_AND_B32_E32", inst_names)
    self.assertIn("GLOBAL_STORE_B16", inst_names)

  def test_int_signed_narrow_cast_sign_extends(self):
    prg = _int_signed_narrow_cast_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
    # signed narrowing must sign-extend (shift left then arithmetic shift right), not mask
    self.assertIn("V_LSHLREV_B32_E64", inst_names)
    self.assertIn("V_ASHRREV_I32_E64", inst_names)
    self.assertNotIn("V_AND_B32_E32", inst_names)

  def test_global_loads_share_single_vmcnt_wait(self):
    # two independent global loads feeding one add should batch into a single vmcnt wait,
    # not one wait per load (s_waitcnt vmcnt stalls the wave, so batching overlaps latency)
    prg = _two_load_add_program()
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
    self.assertEqual(inst_names.count("GLOBAL_LOAD_B32"), 2)
    self.assertEqual(inst_names.count("S_WAITCNT_VMCNT"), 1)
    # the single wait must come after both loads (before the consuming add)
    last_load = max(i for i,n in enumerate(inst_names) if n == "GLOBAL_LOAD_B32")
    self.assertGreater(inst_names.index("S_WAITCNT_VMCNT"), last_load)

  def test_matmul_reg_accumulators_promote_off_scratch(self):
    prg = _matmul64_program()
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertNotIn(AMDOps.SLOAD, linear_ops)
    self.assertNotIn(AMDOps.SSTORE, linear_ops)
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
    self.assertFalse(any("SCRATCH" in name for name in inst_names))

  def test_reg_store_spill_falls_back_to_scratch(self):
    # a promoted accumulator that regalloc spilled appears as a FILL; the store must write back to that scratch
    # slot instead of crashing (regressed test_ops conv2d, which spills its accumulators under register pressure).
    disp = UOp.const(dtypes.int32, 128)
    acc = UOp(Ops.INS, dtypes.float32, (disp,), AMDOps.FILL, (amd_isa.Register("v10", 266),))
    val = UOp(Ops.INS, dtypes.float32, (), AMDOps.MOV, (amd_isa.Register("v11", 267),))
    out, lst = amd_isa._lower_reg_store(UOp(Ops.INS, dtypes.void, (acc, val), AMDOps.REG_STORE))
    self.assertIs(out.arg, AMDOps.SPILL)
    self.assertIs(out.src[0], disp)
    self.assertIs(out.src[1], val)
    self.assertEqual(lst, [out])

  def test_int_signed_widen_cast_sign_extends(self):
    prg = _int_signed_widen_cast_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
    # signed widening sign-extends from the source width via the same shift pair, not a mask
    self.assertIn("V_LSHLREV_B32_E64", inst_names)
    self.assertIn("V_ASHRREV_I32_E64", inst_names)
    self.assertNotIn("V_AND_B32_E32", inst_names)

  def test_float16_unary_promotes_to_float32(self):
    prg = _float16_unary_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    for op in (AMDOps.CAST, AMDOps.SQRT, AMDOps.LOG2, AMDOps.TRUNC, AMDOps.RECIPROCAL, AMDOps.SIN):
      self.assertIn(op, linear_ops)
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
    for name in ("V_CVT_F32_F16_E32", "V_CVT_F16_F32_E32", "V_SIN_F32_E32"):
      self.assertIn(name, inst_names)

  def test_bitwise_assembles(self):
    prg = _bitwise_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    for op in (AMDOps.AND, AMDOps.OR, AMDOps.XOR, AMDOps.SHR):
      self.assertIn(op, linear_ops)

  def test_cmod_pow2_legalizes_to_and(self):
    prg = _cmod_pow2_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    self.assertFalse(any(u.op is Ops.CMOD for u in prg.src[2].src))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.AND, linear_ops)

  def test_const_divmod_legalizes(self):
    prg = _const_divmod_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    self.assertFalse(any(u.op in (Ops.CDIV, Ops.CMOD) for u in prg.src[2].src))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.SHR, linear_ops)
    self.assertIn(AMDOps.MUL, linear_ops)

  def test_var_divmod_legalizes(self):
    prg = _var_divmod_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    self.assertFalse(any(u.op in (Ops.CDIV, Ops.CMOD) for u in prg.src[2].src))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    for op in (AMDOps.SHL, AMDOps.SHR, AMDOps.CMPLT, AMDOps.WHERE):
      self.assertIn(op, linear_ops)

  def test_max_assembles(self):
    for dtype in (dtypes.uint32, dtypes.int32, dtypes.float32):
      with self.subTest(dtype=dtype):
        prg = _max_program(dtype)
        self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
        linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
        self.assertIn(AMDOps.MAX, linear_ops)

  def test_mulacc_assembles(self):
    for prg in (_mulacc_program(), _fused_mulacc_program()):
      with self.subTest(name=prg.arg.name):
        self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
        linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
        self.assertIn(AMDOps.MULACC, linear_ops)
        insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
        self.assertIn("V_FMA_F32", [getattr(i, "op_name", "") for i in insts])

  def test_fused_mulacc_is_isel_only(self):
    self.assertNotIn(Ops.MULACC, AMDRenderer.code_for_op)
    prg = _fused_mulacc_program()
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.MULACC, linear_ops)
    self.assertNotIn(AMDOps.MUL, linear_ops)
    self.assertNotIn(AMDOps.ADD, linear_ops)

  def test_float16_fused_mulacc_uses_f16_fma(self):
    prg = _float16_fused_mulacc_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.MULACC, linear_ops)
    self.assertNotIn(AMDOps.MUL, linear_ops)
    self.assertNotIn(AMDOps.ADD, linear_ops)
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
    self.assertIn("V_FMA_F16", inst_names)

  def test_sgpr_sub_uses_scalar_instruction(self):
    src = UOp(Ops.INS, dtypes.uint32, arg=AMDOps.MOV, tag=(amd_isa.Register("s8", 8),))
    sub = UOp(Ops.INS, dtypes.uint32, (src, UOp.const(dtypes.uint32, 1).rtag()), AMDOps.SUB, (amd_isa.Register("s6", 6),))
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_for_uop(sub)
    self.assertEqual([getattr(i, "op_name", "") for i in insts], ["S_SUB_U32"])

  def test_self_mov_elided(self):
    reg = amd_isa.Register("v3", 256+3)
    src = UOp(Ops.INS, dtypes.uint32, arg=AMDOps.ADD, tag=(reg,))
    mov = UOp(Ops.INS, dtypes.uint32, (src,), AMDOps.MOV, (reg,))
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(UOp(Ops.LINEAR, src=(mov,)))
    self.assertEqual(insts, [])

  def test_cast_and_reciprocal_assemble(self):
    for prg in (_cast_reciprocal_program(), _float_to_int_cast_program()):
      with self.subTest(name=prg.arg.name):
        self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
        linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
        self.assertIn(AMDOps.CAST, linear_ops)
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(_cast_reciprocal_program().src[2])
    inst_names = [getattr(i, "op_name", "") for i in insts]
    self.assertIn("V_CVT_F32_I32_E32", inst_names)
    rcp_idx = inst_names.index("V_RCP_F32_E32")
    self.assertEqual(inst_names[rcp_idx:rcp_idx+4], ["V_RCP_F32_E32", "V_MUL_F32_E32", "V_SUB_F32_E32", "V_FMA_F32"])
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(_float_to_int_cast_program().src[2])
    self.assertIn("V_CVT_I32_F32_E32", [getattr(i, "op_name", "") for i in insts])

  def test_exp2_assembles(self):
    prg = _exp2_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.EXP2, linear_ops)
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
    self.assertIn("V_EXP_F32_E32", [getattr(i, "op_name", "") for i in insts])

  def test_unary_math_assembles(self):
    prg = _unary_math_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    for op in (AMDOps.SQRT, AMDOps.LOG2, AMDOps.TRUNC):
      self.assertIn(op, linear_ops)
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
    for name in ("V_SQRT_F32_E32", "V_LOG_F32_E32", "V_TRUNC_F32_E32"):
      self.assertIn(name, inst_names)

  def test_sin_assembles_with_inline_cody_waite_reduction(self):
    prg = _sin_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.SIN, linear_ops)
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
    reduce_insts = [i for i in insts if getattr(i, "op_name", "") in
                    ("V_MUL_F32_E32", "V_ADD_F32_E32", "V_FRACT_F32_E32", "V_SUB_F32_E32", "V_SIN_F32_E32")]
    self.assertEqual([getattr(i, "op_name", "") for i in reduce_insts],
                     ["V_MUL_F32_E32", "V_ADD_F32_E32", "V_FRACT_F32_E32", "V_SUB_F32_E32", "V_MUL_F32_E32",
                      "V_SUB_F32_E32", "V_MUL_F32_E32", "V_SUB_F32_E32", "V_MUL_F32_E32", "V_SIN_F32_E32"])
    scale, bias, _, floor_turns, hi_mul, hi_sub, lo_mul, lo_sub, final_scale, sin = reduce_insts
    self.assertEqual(scale.vdst, amd_isa.TMP_VDATA)
    self.assertEqual(bias.vdst, amd_isa.TMP_VDATA)
    self.assertEqual(floor_turns.vdst, amd_isa.TMP_VDATA)
    self.assertEqual(hi_sub.vdst, amd_isa.TMP_VADDR)
    self.assertEqual(lo_sub.vdst, amd_isa.TMP_VADDR)
    self.assertEqual(final_scale.vdst, amd_isa.TMP_VDATA)
    self.assertEqual(sin.src0, amd_isa.TMP_VDATA)
    self.assertAlmostEqual(struct.unpack("f", struct.pack("I", scale.literal))[0], 1.0 / (2.0 * math.pi))
    self.assertAlmostEqual(struct.unpack("f", struct.pack("I", final_scale.literal))[0], 1.0 / (2.0 * math.pi))
    self.assertEqual(str(bias.src0), "0.5")
    self.assertAlmostEqual(struct.unpack("f", struct.pack("I", hi_mul.literal))[0], 6.28125)
    self.assertAlmostEqual(struct.unpack("f", struct.pack("I", lo_mul.literal))[0], 0.0019353071795864769)

  def test_uint32_wrap_literal_assembles(self):
    prg = _uint_wrap_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.ADD, linear_ops)

  def test_uint32_mul_alu_param_assembles(self):
    prg = _uint_var_mul_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.MUL, linear_ops)

  def test_vgpr_spill_assembles_with_private_segment(self):
    prg = _spill_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertIn(AMDOps.SPILL, linear_ops)
    self.assertIn(AMDOps.FILL, linear_ops)
    desc = _amd_desc(prg)
    self.assertGreaterEqual(desc.private_segment_fixed_size, 4)
    self.assertTrue(desc.compute_pgm_rsrc2 & (1 << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT_SHIFT))
    self.assertFalse(desc.kernel_code_properties & (1 << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_SHIFT))

  def test_vgpr_multiple_spill_slots_size_private_segment(self):
    prg = _multi_spill_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    spill_ops = [u for u in prg.src[2].src if u.op is Ops.INS and u.arg in (AMDOps.SPILL, AMDOps.FILL)]
    self.assertEqual(sorted({u.src[0].arg for u in spill_ops}), [0, 4])
    self.assertGreaterEqual(_amd_desc(prg).private_segment_fixed_size, 8)

  def test_vgpr_spill_uses_explicit_zero_scratch_addr(self):
    prg = _spill_program()
    renderer = TinyVGPRAMDRenderer(Target("AMD", arch="gfx1100"))
    for u in prg.src[2].src:
      if u.op is Ops.INS and u.arg in (AMDOps.SPILL, AMDOps.FILL):
        with self.subTest(op=u.arg.name):
          insts = renderer._insts_for_uop(u)
          self.assertEqual(insts[0].op_name, "V_MOV_B32_E32")
          self.assertEqual(insts[0].vdst, amd_isa.TMP_VADDR)
          self.assertEqual(str(insts[0].src0), "0")
          scratch = next(i for i in insts if type(i).__name__ == "SCRATCH")
          self.assertEqual(scratch.addr, amd_isa.TMP_VADDR)

  def test_range_loop_assembles_with_branch_fixups(self):
    prg = _range_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    for op in (AMDOps.LABEL, AMDOps.CMP_GE, AMDOps.CBRANCH_SCC1, AMDOps.BRANCH):
      self.assertIn(op, linear_ops)
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
    branches = [i.simm16 if i.simm16 < 0x8000 else i.simm16 - 0x10000 for i in insts
                if getattr(i, "op_name", "") in ("S_CBRANCH_SCC1", "S_BRANCH")]
    self.assertEqual(len(branches), 2)
    self.assertGreater(branches[0], 0)
    self.assertLess(branches[1], 0)

  def test_loop_compare_scalarizes_vgpr_bound(self):
    acc = UOp(Ops.INS, dtypes.uint32, arg=AMDOps.MOV, tag=(amd_isa.Register("s6", 6),))
    bound = UOp(Ops.INS, dtypes.uint32, arg=AMDOps.ADD, tag=(amd_isa.Register("v3", 256+3),))
    cmp = UOp(Ops.INS, dtypes.void, (acc, bound), AMDOps.CMP_GE)
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_for_uop(cmp)
    self.assertEqual([getattr(i, "op_name", "") for i in insts], ["V_READFIRSTLANE_B32_E32", "S_CMP_GE_U32"])
    self.assertEqual(insts[0].vdst, amd_isa.TMP_SDATA1)

  def test_loop_where_rematerializes_vcc_before_each_cndmask(self):
    prg = _loop_vcc_remat_program()
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
    names = [getattr(i, "op_name", "") for i in insts]
    cndmask_idxs = [i for i,n in enumerate(names) if n == "V_CNDMASK_B32_E32"]
    self.assertGreaterEqual(len(cndmask_idxs), 2)
    for i in cndmask_idxs:
      self.assertTrue(any(names[j].startswith("V_CMP_") for j in range(max(0, i-2), i)),
                      f"missing compare rematerialization before cndmask at instruction {i}")

  def test_nested_range_loop_assembles_with_branch_fixups(self):
    prg = _nested_range_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertEqual(linear_ops.count(AMDOps.LABEL), 4)
    self.assertEqual(linear_ops.count(AMDOps.CBRANCH_SCC1), 2)
    self.assertEqual(linear_ops.count(AMDOps.BRANCH), 2)
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
    branches = [i.simm16 if i.simm16 < 0x8000 else i.simm16 - 0x10000 for i in insts
                if getattr(i, "op_name", "") in ("S_CBRANCH_SCC1", "S_BRANCH")]
    self.assertEqual(sum(x > 0 for x in branches), 2)
    self.assertEqual(sum(x < 0 for x in branches), 2)

  def test_variable_range_loop_materializes_sgpr_index_and_data(self):
    prg = _var_range_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    kernargs = [(u.dtype, u.src[0].arg) for u in prg.src[2].src if u.op is Ops.INS and u.arg is AMDOps.KERNARG]
    self.assertIn((dtypes.uint32, 8), kernargs)
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
    self.assertTrue(any(i.op_name == "V_MOV_B32_E32" and i.vdst == amd_isa.TMP_VDATA for i in insts))
    self.assertTrue(any(getattr(i, "op_name", "") == "GLOBAL_STORE_B32" and i.data == amd_isa.TMP_VDATA for i in insts))

  def test_global_range_uses_launch_dims(self):
    prg = _global_dim_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    self.assertNotEqual(prg.arg.global_size, (1, 1, 1))
    self.assertNotEqual(prg.arg.local_size, (1, 1, 1))
    self.assertFalse(any(u.op is Ops.RANGE for u in prg.src[2].src))
    specials = [u.arg for u in prg.src[0].toposort() if u.op is Ops.SPECIAL]
    self.assertIn("gidx0", specials)
    self.assertIn("lidx0", specials)
    desc = _amd_desc(prg)
    self.assertTrue(desc.compute_pgm_rsrc2 & (1 << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT))
    self.assertEqual((desc.compute_pgm_rsrc2 >> amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_SHIFT) & 0x3, 0)

  def test_lds_load_store_barrier_assembles(self):
    prg = _local_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    for op in (AMDOps.LDS_BASE, AMDOps.LSTORE, AMDOps.BARRIER, AMDOps.LLOAD):
      self.assertIn(op, linear_ops)
    self.assertFalse(any(u.op is Ops.AFTER for u in prg.src[2].src))
    desc = _amd_desc(prg)
    self.assertGreaterEqual(desc.group_segment_fixed_size, 16 * dtypes.uint32.itemsize)
    self.assertEqual(desc.private_segment_fixed_size, 0)
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
    op_names = [getattr(i, "op_name", "") for i in insts]
    for name in ("DS_STORE_B32", "S_BARRIER", "DS_LOAD_B32"):
      self.assertIn(name, op_names)

  def test_reg_buffer_uses_private_scratch(self):
    prg = _reg_buffer_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    for op in (AMDOps.SCRATCH_SIZE, AMDOps.SCRATCH_ADDR, AMDOps.SSTORE, AMDOps.SLOAD):
      self.assertIn(op, linear_ops)
    desc = _amd_desc(prg)
    self.assertGreaterEqual(desc.private_segment_fixed_size, 16 * dtypes.uint32.itemsize)
    self.assertTrue(desc.compute_pgm_rsrc2 & (1 << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT_SHIFT))
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
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
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_ops = [u.arg for u in prg.src[2].src if u.op is Ops.INS]
    self.assertGreaterEqual(linear_ops.count(AMDOps.CMPLT), 4)
    self.assertGreaterEqual(linear_ops.count(AMDOps.WHERE), 4)
    self.assertIn(AMDOps.LOAD, linear_ops)

  def test_gated_store_uses_exec_mask_around_store(self):
    lin = _late_gated_store_linear()
    self.assertFalse(any(u.op in (Ops.INDEX, Ops.IF, Ops.ENDIF, Ops.STORE) for u in lin.src))
    masked = [u.arg for u in lin.src if u.op is Ops.INS and u.arg in (AMDOps.IF_MASK, AMDOps.STORE, AMDOps.END_MASK)]
    self.assertEqual(masked, [AMDOps.IF_MASK, AMDOps.STORE, AMDOps.END_MASK])
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(lin)]
    self.assertLess(inst_names.index("S_AND_SAVEEXEC_B64"), inst_names.index("GLOBAL_STORE_B32"))
    self.assertLess(inst_names.index("GLOBAL_STORE_B32"), inst_names.index("S_MOV_B64"))

  def test_gated_store_materialized_bool_rebuilds_vcc(self):
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(_late_gated_store_linear(True))]
    self.assertLess(inst_names.index("V_CMP_NE_U32_E32"), inst_names.index("S_AND_SAVEEXEC_B64"))

  def test_after_global_load_keeps_64bit_saddr(self):
    prg = _after_global_load_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    loads = [i for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2]) if getattr(i, "op_name", "") == "GLOBAL_LOAD_B32"]
    self.assertTrue(loads)
    self.assertTrue(all(i.saddr.sz == 2 for i in loads))

  def test_lds_uses_reserved_vgprs_for_addr_and_scalar_data(self):
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(_local_sgpr_data_program().src[2])
    ds_ops = [i for i in insts if getattr(i, "op_name", "").startswith("DS_")]
    self.assertTrue(ds_ops)
    self.assertTrue(all(i.addr == amd_isa.TMP_VADDR for i in ds_ops))
    self.assertTrue(any(i.op_name == "V_MOV_B32_E32" and i.vdst == amd_isa.TMP_VDATA for i in insts))
    self.assertTrue(any(i.op_name == "DS_STORE_B32" and i.data0 == amd_isa.TMP_VDATA for i in insts))

  def test_narrow_lds_copy_assembles(self):
    prg = _local_program(dtypes.uint8)
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    self.assertGreaterEqual(_amd_desc(prg).group_segment_fixed_size, 16 * dtypes.uint8.itemsize)
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
    op_names = [getattr(i, "op_name", "") for i in insts]
    self.assertIn("DS_STORE_B8", op_names)
    self.assertIn("DS_LOAD_U8", op_names)

  def test_byte_lds_lidx0_uses_byte_addr_directly(self):
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(_local_program(dtypes.uint8).src[2])
    ds_ops = [i for i in insts if getattr(i, "op_name", "").startswith("DS_")]
    self.assertTrue(ds_ops)
    self.assertTrue(all(i.addr == amd_isa.v[0] for i in ds_ops))

  def test_multiple_lds_buffers_get_distinct_offsets(self):
    prg = _multi_local_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    bases = [u for u in prg.src[2].src if u.op is Ops.INS and u.arg is AMDOps.LDS_BASE]
    self.assertEqual(sorted((u.src[0].arg, u.src[1].arg) for u in bases), [(64, 0), (64, 64)])
    self.assertGreaterEqual(_amd_desc(prg).group_segment_fixed_size, 128)
    insts = AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])
    self.assertTrue(any(i.op_name == "V_ADD_NC_U32_E64" and i.vdst == amd_isa.TMP_VADDR for i in insts))

  def test_duplicate_lds_slot_rejected(self):
    with self.assertRaises(CompileError):
      _duplicate_local_slot_program()

  def test_gidx_metadata_survives_for_descriptor(self):
    prg = _gidx_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    specials = [u.arg for u in prg.src[0].toposort() if u.op is Ops.SPECIAL]
    self.assertIn("gidx0", specials)

  def test_multi_dim_specials_set_descriptor_bits(self):
    prg = _multi_dim_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_regs = {u.reg.index for u in prg.src[2].src if getattr(u, "reg", None) is not None}
    self.assertIn(256, linear_regs)
    self.assertIn(257, linear_regs)
    self.assertIn(3, linear_regs)
    desc = _amd_desc(prg)
    self.assertTrue(desc.compute_pgm_rsrc2 & (1 << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT))
    self.assertEqual((desc.compute_pgm_rsrc2 >> amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_SHIFT) & 0x3, 1)

  def test_z_dim_specials_set_descriptor_bits(self):
    prg = _z_dim_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    linear_regs = {u.reg.index for u in prg.src[2].src if getattr(u, "reg", None) is not None}
    self.assertIn(258, linear_regs)
    self.assertIn(4, linear_regs)
    desc = _amd_desc(prg)
    self.assertTrue(desc.compute_pgm_rsrc2 & (1 << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT))
    self.assertEqual((desc.compute_pgm_rsrc2 >> amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_SHIFT) & 0x3, 2)

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
    rt = Device["AMD"].runtime(prg.arg.function_name, prg.src[4].arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(buf.get_buf("AMD"), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), list(range(1, 9)))

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_two_alu_params_smoke(self):
    inp = Tensor(list(range(16)), dtype=dtypes.uint32, device="AMD").contiguous().realize()
    out = Tensor.empty(16, dtype=dtypes.uint32, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp)], _two_uint_var_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, prg.src[4].arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(2, 3), wait=True)
    self.assertEqual(out.tolist(), [i + 5 for i in range(16)])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_reg_buffer_smoke(self):
    out = Tensor.empty(16, dtype=dtypes.uint32, device="AMD").contiguous().realize()
    buf, prg = out._buffer().ensure_allocated(), _reg_buffer_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, prg.src[4].arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(buf.get_buf("AMD"), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), [i + 6 for i in range(16)])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_gated_load_smoke(self):
    inp0 = Tensor(list(range(16)), dtype=dtypes.uint32, device="AMD").contiguous().realize()
    inp1 = Tensor([100 + i for i in range(16)], dtype=dtypes.uint32, device="AMD").contiguous().realize()
    out = Tensor.empty(16, dtype=dtypes.uint32, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp0, inp1)], _gated_load_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, prg.src[4].arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
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
    rt = Device["AMD"].runtime(prg.arg.function_name, prg.src[4].arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), [1.5, -2.0, 3.5, -4.0])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_float16_unary_smoke(self):
    vals = [1.0, 2.0, 4.0, 8.0]
    inp = Tensor(vals, dtype=dtypes.float16, device="AMD").contiguous().realize()
    out = Tensor.empty(4, dtype=dtypes.float16, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp)], _float16_unary_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, prg.src[4].arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
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
    rt = Device["AMD"].runtime(prg.arg.function_name, prg.src[4].arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
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
    # RDNA has no direct int<->f16 convert; these route through f32. Cover signed widening, unsigned, and bool sources.
    self.assertEqual(Tensor([-5, 0, 7, 100], dtype=dtypes.int8, device="AMD").cast(dtypes.float16).tolist(), [-5.0, 0.0, 7.0, 100.0])
    self.assertEqual(Tensor([0, 7, 100, 4000], dtype=dtypes.uint16, device="AMD").cast(dtypes.float16).tolist(), [0.0, 7.0, 100.0, 4000.0])
    self.assertEqual(Tensor([True, False, True, False], device="AMD").cast(dtypes.float16).tolist(), [1.0, 0.0, 1.0, 0.0])
    self.assertEqual(Tensor([-5.0, 0.0, 7.0, 100.0], dtype=dtypes.float16, device="AMD").cast(dtypes.int32).tolist(), [-5, 0, 7, 100])

  def test_int_half_cast_routes_through_f32(self):
    # the int->f16 leg should select two converts (int->f32, f32->f16), never a raw CAST left for regalloc
    prg = _int_to_half_cast_program()
    self.assertTrue(prg.src[4].arg.startswith(b"\x7fELF"))
    inst_names = [getattr(i, "op_name", "") for i in AMDRenderer(Target("AMD", arch="gfx1100"))._insts_from_linear(prg.src[2])]
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
    rt = Device["AMD"].runtime(prg.arg.function_name, prg.src[4].arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), [-21, 19, 21, -19])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_exp2_smoke(self):
    inp = Tensor([0.0, 1.0, 2.0, 3.0], dtype=dtypes.float32, device="AMD").contiguous().realize()
    out = Tensor.empty(4, dtype=dtypes.float32, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp)], _exp2_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, prg.src[4].arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), [1.0, 2.0, 4.0, 8.0])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_unary_math_smoke(self):
    inp = Tensor([1.0, 4.0, 16.0, 64.0], dtype=dtypes.float32, device="AMD").contiguous().realize()
    out = Tensor.empty(4, dtype=dtypes.float32, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp)], _unary_math_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, prg.src[4].arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
    rt(*(b.get_buf("AMD") for b in bufs), global_size=prg.arg.global_size, local_size=prg.arg.local_size, vals=(), wait=True)
    self.assertEqual(out.tolist(), [1.0, 4.0, 8.0, 14.0])

  @unittest.skipUnless(_has_amd_asm_runtime(), "requires DEV=AMD:AMD or DEV=MOCKKFD+AMD:AMD on gfx11")
  def test_hardware_sin_smoke(self):
    inp = Tensor([-2.0, 0.0, 1.0, math.pi / 2], dtype=dtypes.float32, device="AMD").contiguous().realize()
    out = Tensor.empty(4, dtype=dtypes.float32, device="AMD").contiguous().realize()
    bufs, prg = [x._buffer().ensure_allocated() for x in (out, inp)], _sin_program()
    rt = Device["AMD"].runtime(prg.arg.function_name, prg.src[4].arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars, prg=prg)
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
