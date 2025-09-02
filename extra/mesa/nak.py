from __future__ import annotations
from tinygrad import Tensor, dtypes
import tinygrad.runtime.autogen.nak as nak
import tinygrad.runtime.autogen.nir as nir
import tinygrad.runtime.autogen.libc as libc
from tinygrad.runtime.ops_nv import NVDevice, NVProgram
import ctypes, os

stdout = ctypes.POINTER(nir.struct__IO_FILE).in_dll(libc._libraries['libc'], "stdout")
nir_intrinsic_infos = nir.nir_intrinsic_infos.in_dll(nir._libraries['FIXME_STUB'], "nir_intrinsic_infos")
glsl_type_builtin_uint64_t = nir.struct_glsl_type.in_dll(nir._libraries['FIXME_STUB'], "glsl_type_builtin_uint64_t")

def BITFIELD_BIT(b): return 1 << b
def BITFIELD_MASK(b): return 0xFFFFFFFF if b == 32 else BITFIELD_BIT(b & 31) - 1

"""
struct nak_shader_bin *
nak_compile_shader(nir_shader *nir, bool dump_asm,
                   const struct nak_compiler *nak,
                   nir_variable_mode robust2_modes,
                   const struct nak_fs_key *fs_key);

nir_builder nir_builder_init_simple_shader(
  mesa_shader_stage stage,                    // MESA_SHADER_COMPUTE?
  const nir_shader_compiler_options *options, // {}
  const char *name, ...);

https://elixir.bootlin.com/mesa/mesa-25.2.0/source/src/nouveau/vulkan/nvk_shader.c#L479
"""

# https://elixir.bootlin.com/mesa/mesa-25.2.0/source/src/compiler/glsl_types.h#L172
def glsl_base_type_get_bit_size(base_type: nir.glsl_base_type) -> int:
  return {
      nir.GLSL_TYPE_BOOL : 1,
      nir.GLSL_TYPE_INT : 32, nir.GLSL_TYPE_UINT : 32, nir.GLSL_TYPE_FLOAT : 32, nir.GLSL_TYPE_SUBROUTINE : 32, nir.GLSL_TYPE_COOPERATIVE_MATRIX : 32,
      nir.GLSL_TYPE_FLOAT16 : 16, nir.GLSL_TYPE_BFLOAT16 : 16, nir.GLSL_TYPE_UINT16: 16, nir.GLSL_TYPE_INT16 : 16,
      nir.GLSL_TYPE_UINT8 : 8, nir.GLSL_TYPE_INT8 : 8, nir.GLSL_TYPE_FLOAT_E4M3FN : 8, nir.GLSL_TYPE_FLOAT_E5M2 : 8,
      nir.GLSL_TYPE_DOUBLE : 64, nir.GLSL_TYPE_INT64 : 64, nir.GLSL_TYPE_UINT64 : 64, nir.GLSL_TYPE_IMAGE : 64, nir.GLSL_TYPE_SAMPLER : 64, nir.GLSL_TYPE_TEXTURE : 64,
    }[int(base_type)]

def show_layout(struct):
  print(struct)
  for k, _ in struct._fields_: print(f"  {k}: 0x{getattr(struct, k).offset:X}")

def nir_src_for_ssa(d):
  src = nir.nir_src()
  src.ssa = d
  print(d, src.ssa.contents)
  return src

def nir_intrinsic_set(typ, instr, val):
  info = nir_intrinsic_infos[instr.contents.intrinsic]
  assert info.index_map[typ] > 0
  instr.contents.const_index[info.index_map[typ] - 1] = val

def nir_intrinsic_set_access(instr, val): nir_intrinsic_set(nir.NIR_INTRINSIC_ACCESS, instr, val)
def nir_intrinsic_set_param_idx(instr, val): nir_intrinsic_set(nir.NIR_INTRINSIC_PARAM_IDX, instr, val)
def nir_intrinsic_set_write_mask(instr, val): nir_intrinsic_set(nir.NIR_INTRINSIC_WRITE_MASK, instr, val)
def nir_intrinsic_set_align_mul(instr, val): nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_MUL, instr, val)
def nir_intrinsic_set_align_offset(instr, val): nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_OFFSET, instr, val)
def nir_intrinsic_set_align(instr, mul, off):
  assert off < mul
  nir_intrinsic_set_align_mul(instr, mul)
  nir_intrinsic_set_align_offset(instr, off)
def nir_intrinsic_set_range_base(instr, val): nir_intrinsic_set(nir.NIR_INTRINSIC_RANGE_BASE, instr, val)
def nir_intrinsic_set_range(instr, val): nir_intrinsic_set(nir.NIR_INTRINSIC_RANGE, instr, val)

def nir_build_load_param(b:nir.nir_builder, num:int, bit_sz:int, idx:int) -> ctypes._Pointer[nir.nir_def]:
  intrin = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_load_param)
  intrin.contents.num_components = num
  nir.nir_def_init(intrin.contents.instr, getattr(intrin.contents, "def"), intrin.contents.num_components, bit_sz)
  nir_intrinsic_set_param_idx(intrin, idx)
  nir.nir_builder_instr_insert(b, intrin.contents.instr)
  return ctypes.pointer(getattr(intrin.contents, "def"))

def nir_load_param(b:nir.nir_builder, idx:int) -> ctypes._Pointer[nir.nir_def]:
  assert idx < b.impl.contents.function.contents.num_params
  param = b.impl.contents.function.contents.params[idx]
  return nir_build_load_param(b, param.num_components, param.bit_size, idx)

def nir_build_deref_var(b:nir.nir_builder, var:ctypes._Pointer[nir.nir_variable]) -> ctypes._Pointer[nir.nir_deref_instr]:
  deref = nir.nir_deref_instr_create(b.shader, nir.nir_deref_type_var)
  deref.contents.modes, deref.contents.type, deref.contents.var = var.contents.data.mode, var.contents.type, var
  nir.nir_def_init(deref.contents.instr, getattr(deref.contents, "def"), 1, b.shader.contents.info.cs.ptr_size)
  nir.nir_builder_instr_insert(b, deref.contents.instr)
  return deref

def nir_build_load_deref(b:nir.nir_builder, num:int, bit_sz:int, src0:ctypes._Pointer[nir.nir_def], access:int) -> ctypes._Pointer[nir.nir_def]:
  intrin = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_load_deref)
  intrin.contents.num_components = num
  nir.nir_def_init(intrin.contents.instr, getattr(intrin.contents, "def"), intrin.contents.num_components, bit_sz)
  ctypes.cast(intrin.contents.src, ctypes.POINTER(nir.nir_src))[0] = nir_src_for_ssa(src0)
  nir_intrinsic_set_access(intrin, access)
  nir.nir_builder_instr_insert(b, intrin.contents.instr)
  return ctypes.pointer(getattr(intrin.contents, "def"))

def nir_load_deref(b:nir.nir_builder, d:ctypes._Pointer[nir.nir_deref_instr]) -> ctypes._Pointer[nir.nir_def]:
  return nir_build_load_deref(b, d.contents.type.contents.vector_elements, glsl_base_type_get_bit_size(d.contents.type.contents.base_type), ctypes.pointer(getattr(d.contents, "def")), 0)

def nir_load_ubo(b:nir.nir_builder, num:int, bit_sz:int, src0:ctypes._Pointer[nir.nir_def], src1:ctypes._Pointer[nir.nir_def],
                 access:int=0, align_mul:int=0, align_offset:int=0, range_base:int=0, range_:int=0) -> ctypes._Pointer[nir.nir_def]:
  intrin = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_load_ubo)
  intrin.contents.num_components = num
  nir.nir_def_init(intrin.contents.instr, getattr(intrin.contents, "def"), intrin.contents.num_components, bit_sz)
  arr = ctypes.cast(intrin.contents.src, ctypes.POINTER(nir.nir_src))
  arr[0], arr[1] = nir_src_for_ssa(src0), nir_src_for_ssa(src1)
  nir_intrinsic_set_access(intrin, access)
  nir_intrinsic_set_align(intrin, align_mul if align_mul else getattr(intrin.contents, "def").bit_size // 8, align_offset)
  nir_intrinsic_set_range_base(intrin, range_base)
  nir_intrinsic_set_range(intrin, range_)
  nir.nir_builder_instr_insert(b, intrin.contents.instr)
  return ctypes.pointer(getattr(intrin.contents, "def"))

def nir_ldc_nv(b:nir.nir_builder, num:int, bit_sz:int, src0:ctypes._Pointer[nir.nir_def], src1:ctypes._Pointer[nir.nir_def],
               access:int=0, align_mul:int=0, align_offset:int=0) -> ctypes._Pointer[nir.nir_def]:
  intrin = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_ldc_nv)
  intrin.contents.num_components = num
  nir.nir_def_init(intrin.contents.instr, getattr(intrin.contents, "def"), intrin.contents.num_components, bit_sz)
  arr = ctypes.cast(intrin.contents.src, ctypes.POINTER(nir.nir_src))
  arr[0], arr[1] = nir_src_for_ssa(src0), nir_src_for_ssa(src1)
  nir_intrinsic_set_access(intrin, access)
  nir_intrinsic_set_align(intrin, align_mul if align_mul else getattr(intrin.contents, "def").bit_size // 8, align_offset)
  nir.nir_builder_instr_insert(b, intrin.contents.instr)
  return ctypes.pointer(getattr(intrin.contents, "def"))

def nir_iadd(b:nir.nir_builder, src0:nir.nir_def, src1:nir.nir_def) -> nir.nir_def: return nir.nir_build_alu2(b, nir.nir_op_iadd, src0, src1)

def nir_imm_int(b:nir.nir_builder, x:int) -> ctypes._Pointer[nir.nir_def]:
  load_const = nir.nir_load_const_instr_create(b.shader, 1, 32)
  ctypes.cast(load_const.contents.value, ctypes.POINTER(ctypes.c_int))[0] = x
  nir.nir_builder_instr_insert(b, load_const.contents.instr)
  return ctypes.pointer(getattr(load_const.contents, 'def'))

def nir_store_global(b:nir.nir_builder, addr:ctypes._Pointer[nir.nir_def], align:int, value:ctypes._Pointer[nir.nir_def], write_mask:int):
  store = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_store_global)
  store.contents.num_components = value.contents.num_components
  arr = ctypes.cast(store.contents.src, ctypes.POINTER(nir.nir_src))
  arr[0], arr[1] = nir_src_for_ssa(value), nir_src_for_ssa(addr)
  nir_intrinsic_set_write_mask(store, write_mask & BITFIELD_MASK(value.contents.num_components))
  nir_intrinsic_set_align(store, align, 0)
  nir.nir_builder_instr_insert(b, store.contents.instr)

input(f"pid: {os.getpid()}. press enter to continue...")

dev = NVDevice()
b = nir.nir_builder_init_simple_shader(nir.MESA_SHADER_COMPUTE, nir.nir_shader_compiler_options(), None)
data0 = nir_ldc_nv(b, 1, 64, nir_imm_int(b, 0), nir_imm_int(b, 0x160))
nir_store_global(b, data0, 4, nir_imm_int(b, 1337), ~0)
"""
param = nir.nir_parameter(1, 64, False, type=ctypes.pointer(glsl_type_builtin_uint64_t), name=ctypes.create_string_buffer(b"hi"))
b.impl.contents.function.contents.num_params = 1
b.impl.contents.function.contents.params = ctypes.pointer(param)
data0 = nir_load_param(b, 0)
nir_store_global(b, data0, 4, nir_imm_int(b, 1), ~0)
"""
nir.nir_print_shader(b.shader, stdout)
cc = nak.nak_compiler_create(nak.struct_nv_device_info(sm=int(dev.arch[3:]), max_warps_per_mp=dev.max_warps_per_sm))
nak.nak_preprocess_nir(ctypes.cast(b.shader, ctypes.POINTER(nak.struct_nir_shader)), cc)
nir.nir_print_shader(b.shader, stdout)
out = nak.nak_compile_shader(ctypes.cast(b.shader, ctypes.POINTER(nak.struct_nir_shader)), True, cc, 0, None)

print(ctypes.string_at(out.contents.asm_str).decode())
if input("write to file? (y/n) ") == "y":
  with open("out.cubin", "wb") as f: f.write(ctypes.string_at(out.contents.code, out.contents.code_size))

print(f"""info:
  gprs: 0x{out.contents.info.num_gprs:X}""")
if b.shader.contents.constant_data_size > 0: print("constant data!")
prog = NVProgram(dev, "fxn", bytearray(ctypes.string_at(out.contents.code, out.contents.code_size)), raw=True, regs_usage=out.contents.info.num_gprs)

a = dev.allocator.alloc(4)
prog(a, wait=True)
a_out = bytearray(4)
dev.allocator._copyout(memoryview(a_out), a)
import struct
print(struct.unpack("I", a_out))

