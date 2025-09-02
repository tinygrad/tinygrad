from typing import cast
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat
import tinygrad.runtime.autogen.nir as nir
import tinygrad.runtime.autogen.libc as libc
from tinygrad.runtime.support.nak import nak_nir_options
import ctypes, struct

# FIXME: this is because clang2py produces bad output for hidden symbols
nir_intrinsic_infos = nir.nir_intrinsic_infos.in_dll(nir._libraries['FIXME_STUB'], "nir_intrinsic_infos")
stdout = ctypes.POINTER(nir.struct__IO_FILE).in_dll(libc._libraries['libc'], "stdout")

def BITFIELD_BIT(b): return 1 << b
def BITFIELD_MASK(b): return 0xFFFFFFFF if b == 32 else BITFIELD_BIT(b & 31) - 1

# TODO: @functools.cache
def nir_imm(b:nir.nir_builder, x, dtype:DType) -> nir.nir_def:
  assert dtype.fmt
  instr = nir.nir_load_const_instr_create(b.shader, 1, dtype.itemsize * 8)
  struct.pack_into(dtype.fmt, (ctypes.c_ubyte * dtype.itemsize).from_address(ctypes.addressof(instr.contents.value)), 0, x)
  nir.nir_builder_instr_insert(b, instr.contents.instr)
  return getattr(instr.contents, "def")

def nir_src_for_ssa(d:nir.nir_def) -> nir.nir_src: return nir.nir_src(ssa=ctypes.pointer(d))
def nir_intrinsic_set(typ, instr:nir.nir_intrinsic_instr, val:int):
  info = nir_intrinsic_infos[instr.contents.intrinsic]
  assert info.index_map[typ] > 0
  instr.contents.const_index[info.index_map[typ] - 1] = val

def nir_build_alu(b:nir.nir_builder, op, *srcs:list[nir.nir_def]) -> nir.nir_def:
  if len(srcs) == 1: return nir.nir_build_alu1(b, op, srcs[0]).contents
  if len(srcs) == 2: return nir.nir_build_alu2(b, op, srcs[0], srcs[1]).contents
  if len(srcs) == 3: return nir.nir_build_alu3(b, op, srcs[0], srcs[1], srcs[2]).contents
  return nir.nir_build_alu4(b, op, srcs[0], srcs[1], srcs[2], srcs[3]).contents

def nir_store_global(b:nir.nir_builder, addr:nir.nir_def, value:nir.nir_def, write_mask:int):
  store = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_store_global)
  store.contents.num_components = value.num_components # is this right?
  arr = ctypes.cast(store.contents.src, ctypes.POINTER(nir.nir_src))
  arr[0], arr[1] = nir_src_for_ssa(value), nir_src_for_ssa(addr)
  # TODO: think about what these should be set to
  nir_intrinsic_set(nir.NIR_INTRINSIC_WRITE_MASK, store, write_mask & BITFIELD_MASK(value.num_components))
  nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_MUL, store, 4)
  nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_OFFSET, store, 0) # is setting to zero the default case?
  nir.nir_builder_instr_insert(b, store.contents.instr)
  return nir.nir_def() # FIXME!

class NIRRenderer(Renderer):
  device = "NV"
  suffix = "NAK"
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max

  extra_matcher = PatternMatcher([
    (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat(Ops.CONST, dtype=dtypes.int, name="idx"))),
      lambda buf,idx: UOp(Ops.INDEX, src=(buf, UOp(Ops.CONST, dtype=dtypes.long, arg=idx.arg))))
  ])

  def_rewrite = PatternMatcher([
    (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx')), allow_any_len=True),
      lambda ctx,buf,idx: nir_build_alu(ctx[0], nir.nir_op_iadd, ctx[1][buf], ctx[1][idx])),
    # TODO: local store (a la ptx's mem_types)
    (UPat(Ops.STORE, src=(UPat.var("addr"), UPat.var("val"))),
      lambda ctx,addr,val: nir_store_global(ctx[0], ctx[1][addr], ctx[1][val], ~0))
  ])

  def __init__(self, arch:str, device="NV"): self.device, self.arch = device, arch

  def param(self, b:nir.nir_builder, dtype:DType, idx:int) -> nir.nir_def:
    intrin = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_ldc_nv)
    intrin.contents.num_components = 1
    nir.nir_def_init(intrin.contents.instr, getattr(intrin.contents, "def"), 1, 64 if isinstance(dtype, PtrDType) else dtype.itemsize * 8)
    arr = ctypes.cast(intrin.contents.src, ctypes.POINTER(nir.nir_src))
    # is this the right offset?
    arr[0], arr[1] = nir_src_for_ssa(nir_imm(b, 0, dtypes.int)), nir_src_for_ssa(nir_imm(b, 0x160 + idx * 8, dtypes.int))
    # TODO: are these values correct?
    nir_intrinsic_set(nir.NIR_INTRINSIC_ACCESS, intrin, 0)
    nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_MUL, intrin, getattr(intrin.contents, "def").bit_size // 8)
    nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_OFFSET, intrin, 0)
    nir.nir_builder_instr_insert(b, intrin.contents.instr)
    return getattr(intrin.contents, "def")

  def render(self, uops:list[UOp]) -> str:
    b = nir.nir_builder_init_simple_shader(nir.MESA_SHADER_COMPUTE, nak_nir_options, None)
    r: dict[UOp, nir.nir_def] = {}
    args: list[tuple[nir.nir_def, DType]] = []

    for u in uops:
      nir.nir_print_shader(b.shader, stdout)
      if u.op is Ops.NOOP: continue
      if u.op is Ops.SINK:
        # TODO: https://elixir.bootlin.com/mesa/mesa-25.2.1/source/src/compiler/nir/nir_builder.c#L33
        if u.arg is not None: pass
        continue
      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        assert u.op is Ops.DEFINE_GLOBAL
        r[u] = self.param(b, u.dtype, u.arg)
        args.append((r[u], u.dtype))
      elif u.op in (Ops.DEFINE_LOCAL, Ops.DEFINE_REG): raise NotImplementedError("DEFINE_LOCAL/REG")
      elif u.op is Ops.CONST: r[u] = nir_imm(b, u.arg, u.dtype)
      else:
        print(u)
        if (d:=self.def_rewrite.rewrite(u, ctx=(b,r))) is None:
          nir.nir_print_shader(b.shader, stdout)
          raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        r[u] = cast(nir.nir_def, d)
    print(b.shader.contents)
    import tinygrad.runtime.autogen.nak as nak
    import os
    input(f"pid: {os.getpid()}")
    cc = nak.nak_compiler_create(nak.struct_nv_device_info(sm=86, max_warps_per_mp=48))
    nak.nak_preprocess_nir(nak.struct_nir_shader.from_buffer(b.shader.contents), cc)
    out = nak.nak_compile_shader(ctypes.cast(b.shader, ctypes.POINTER(nak.struct_nir_shader)), True, cc, 0, None)
    print(ctypes.string_at(out.contents.asm_str).decode())
    return b.shader.contents
