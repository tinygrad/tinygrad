from typing import cast
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.uop.ops import GroupOp, Ops, UOp, PatternMatcher, UPat
import tinygrad.runtime.autogen.nir as nir
import tinygrad.runtime.autogen.libc as libc
import tinygrad.runtime.support.nak as nak
import ctypes, struct

# FIXME: this is because clang2py produces bad output for hidden symbols
nir_intrinsic_infos = nir.nir_intrinsic_infos.in_dll(nir._libraries['FIXME_STUB'], "nir_intrinsic_infos")
stdout = ctypes.POINTER(nir.struct__IO_FILE).in_dll(libc._libraries['libc'], "stdout")

def BITFIELD_BIT(b): return 1 << b
def BITFIELD_MASK(b): return 0xFFFFFFFF if b == 32 else BITFIELD_BIT(b & 31) - 1

def nir_mov_alu(b:nir.nir_builder, src:nir.nir_alu_src, num_components:int) -> nir.nir_def:
  mov = nir.nir_alu_instr_create(b.shader, nir.nir_op_mov)
  nir.nir_def_init(mov.contents.instr, getattr(mov.contents, "def"), num_components, src.src.ssa.contents.bit_size)
  mov.contents.exact, mov.contents.fp_fast_math = b.exact, b.fp_fast_math
  ctypes.cast(mov.contents.src, ctypes.POINTER(nir.nir_alu_src))[0] = src
  nir.nir_builder_instr_insert(b, mov.contents.instr)
  return getattr(mov.contents, "def")

def nir_swizzle(b:nir.nir_builder, src:nir.nir_def, swiz:list[int]) -> nir.nir_def:
  alu_src, is_id = nir.nir_alu_src(), True
  alu_src.src = nir_src_for_ssa(src)
  for i, s in enumerate(swiz):
    if i != s: is_id = False
    alu_src.swizzle[i] = s
  if len(swiz) == src.num_components and is_id: return src
  return nir_mov_alu(b, alu_src, len(swiz))

def nir_channel(b:nir.nir_builder, src:nir.nir_def, c:int) -> nir.nir_def: return nir_swizzle(b, src, [c])

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

def nir_build_alu(b:nir.nir_builder, op, *srcs:nir.nir_def) -> nir.nir_def:
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

def nir_load_global(b:nir.nir_builder, addr:nir.nir_def, dtype:DType):
  load = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_load_global)
  load.contents.num_components = dtype.count
  ctypes.cast(load.contents.src, ctypes.POINTER(nir.nir_src))[0] = nir_src_for_ssa(addr)
  nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_MUL, load, 4)
  nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_OFFSET, load, 0) # TODO
  nir.nir_def_init(load.contents.instr, getattr(load.contents, "def"), dtype.count, 8 * dtype.itemsize)
  nir.nir_builder_instr_insert(b, load.contents.instr)
  return getattr(load.contents, "def")

def nir_gid(b:nir.nir_builder) -> nir.nir_def:
  intrin = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_load_workgroup_id)
  nir.nir_def_init(intrin.contents.instr, getattr(intrin.contents, "def"), 3, 32)
  nir.nir_builder_instr_insert(b, intrin.contents.instr)
  return getattr(intrin.contents, "def")

def nir_lid(b:nir.nir_builder) -> nir.nir_def:
  intrin = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_load_local_invocation_id)
  nir.nir_def_init(intrin.contents.instr, getattr(intrin.contents, "def"), 3, 32)
  nir.nir_builder_instr_insert(b, intrin.contents.instr)
  return getattr(intrin.contents, "def")

def nv_param(b:nir.nir_builder, dtype:DType, idx:int) -> nir.nir_def:
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

# alu ops, aop[<dtype>][<op>]
# TODO: test imod?
u_aop = { Ops.ADD: nir.nir_op_uadd_sat, Ops.MUL: nir.nir_op_imul, Ops.IDIV: nir.nir_op_udiv, Ops.MOD: nir.nir_op_umod, Ops.CMPLT: nir.nir_op_ult,
          Ops.CMPNE: nir.nir_op_ine, Ops.CMPEQ: nir.nir_op_ieq, Ops.OR: nir.nir_op_ior, Ops.AND: nir.nir_op_iand, Ops.XOR: nir.nir_op_ixor}
s_aop = {**u_aop, Ops.ADD: nir.nir_op_iadd, Ops.CMPLT: nir.nir_op_ilt, Ops.IDIV: nir.nir_op_idiv, Ops.MOD: nir.nir_op_irem}
f_aop = { Ops.ADD: nir.nir_op_fadd, Ops.MUL: nir.nir_op_fmul, Ops.CMPLT: nir.nir_op_flt, Ops.CMPNE: nir.nir_op_fneu, Ops.CMPEQ: nir.nir_op_fequ,
              Ops.FDIV: nir.nir_op_fdiv}
aop = {**{x:u_aop for x in (dtypes.bool,)+dtypes.uints}, **{x:s_aop for x in dtypes.sints}, **{x:f_aop for x in dtypes.floats}}

def ncast(b:nir.nir_builder, src:nir.nir_def, input_type:DType, output_type:DType) -> nir.nir_def:
  match input_type, output_type:
    case dtypes.int, dtypes.long: return nir_build_alu(b, nir.nir_op_i2i64, src)
    case dtypes.long, PtrDType(): return src
    case _: raise NotImplementedError(f"cast from {input_type} -> {output_type} not implemented")

class NIRRenderer(Renderer):
  device = "NV"
  suffix = "NAK"
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max

  extra_matcher = PatternMatcher([
    (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat(Ops.CONST, dtype=dtypes.int, name="idx"))),
      lambda buf,idx: UOp(Ops.INDEX, dtype=dtypes.long, src=(buf, UOp(Ops.CONST, dtype=dtypes.long, arg=idx.arg)))),
    (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat(dtype=dtypes.int, name="idx"))),
     lambda buf,idx: UOp(Ops.INDEX, dtype=dtypes.long, src=(buf, UOp(Ops.CAST, dtype=dtypes.long, src=(idx,)))))
  ])

  def_rewrite = PatternMatcher([
    (UPat(Ops.CONST, name="x"), lambda ctx,x: nir_imm(ctx[0], x.arg, x.dtype)),
    (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx,x: nv_param(ctx[0], x.dtype, x.arg)),
    (UPat(Ops.SPECIAL, name="x"),
      lambda ctx,x: nir_channel(ctx[0], nir_gid(ctx[0]) if x.arg[0][0] == 'g' else nir_lid(ctx[0]), int(x.arg[0][-1]))),
    (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx')), allow_any_len=True),
      lambda ctx,buf,idx: nir_build_alu(ctx[0], nir.nir_op_iadd, ctx[1][buf],
                                        nir_build_alu(ctx[0], nir.nir_op_imul, ctx[1][idx], nir_imm(ctx[0], buf.dtype.itemsize, dtypes.long)))),
    # TODO: local (a la ptx's mem_types)
    (UPat(Ops.STORE, src=(UPat.var("addr"), UPat.var("val")), allow_any_len=True),
      lambda ctx,addr,val: nir_store_global(ctx[0], ctx[1][addr], ctx[1][val], ~0)),
    (UPat(Ops.LOAD, src=(UPat.var("addr")), name="x"), lambda ctx,x,addr: nir_load_global(ctx[0], ctx[1][addr], x.dtype)),
    (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: nir_build_alu(ctx[0], nir.nir_op_vec4, *[ctx[1][src] for src in x.src])),
    (UPat(GroupOp.Binary, name="x"), lambda ctx,x: nir_build_alu(ctx[0], aop[x.src[0].dtype.scalar()][x.op], ctx[1][x.src[0]], ctx[1][x.src[1]])),
    (UPat(Ops.CAST, name="x"), lambda ctx,x: ncast(ctx[0], ctx[1][x.src[0]], x.src[0].dtype, x.dtype)),
  ])

  def __init__(self, arch:str, device="NV"): self.device, self.arch = device, arch

  def render(self, uops:list[UOp]) -> str:
    b = nir.nir_builder_init_simple_shader(nir.MESA_SHADER_COMPUTE, nak.nir_options, None)
    for u in [u for u in uops if u.op is Ops.SPECIAL and u.arg[0][0] == "l"]: b.shader.contents.info.workgroup_size[int(u.arg[0][-1])] = u.arg[1]
    r: dict[UOp, nir.nir_def] = {}

    for u in uops:
      match u:
        case UOp(Ops.NOOP): pass
        # TODO: https://elixir.bootlin.com/mesa/mesa-25.2.1/source/src/compiler/nir/nir_builder.c#L33
        case UOp(Ops.SINK):
          if u.arg is not None: b.shader.contents.info.name = nir.char_pointer_cast(u.arg.function_name)
        case UOp(Ops.DEFINE_LOCAL) | UOp(Ops.DEFINE_REG): raise NotImplementedError("DEFINE_LOCAL/REG")
        case _:
          if (d:=self.def_rewrite.rewrite(u, ctx=(b,r))) is None:
            nir.nir_print_shader(b.shader, stdout)
            raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
          r[u] = cast(nir.nir_def, d)
    return b.shader.contents
