from typing import Callable, cast, Tuple, Type, TypeVar
from tinygrad.dtype import AddrSpace, DType, PtrDType, dtypes
from tinygrad.helpers import all_same, get_single_element
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.uop.ops import GroupOp, Ops, UOp, PatternMatcher, UPat
import tinygrad.runtime.autogen.nir as nir
import tinygrad.runtime.autogen.libc as libc
import tinygrad.runtime.support.nak as nak
import ctypes, struct

# FIXME: this is because clang2py produces bad output?
nir_intrinsic_infos = nir.nir_intrinsic_infos.in_dll(nir._libraries['FIXME_STUB'], "nir_intrinsic_infos")
assert libc._libraries['libc']
stdout = ctypes.POINTER(nir.struct__IO_FILE).in_dll(libc._libraries['libc'], "stdout")
s = nir.char_pointer_cast

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
  instr = nir.nir_load_const_instr_create(b.shader, 1, 1 if dtype == dtypes.bool else dtype.itemsize * 8)
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

def nir_build_phi(b:nir.nir_builder, srcs:list[Tuple[nir.nir_block, nir.nir_def]]) -> Tuple[nir.nir_phi_instr, nir.nir_def]:
  assert all_same([src[1].num_components for src in srcs]) and all_same([src[1].bit_size for src in srcs])
  phi = nir.nir_phi_instr_create(b.shader)
  nir.nir_def_init(phi.contents.instr, getattr(phi.contents, "def"), srcs[0][1].num_components, srcs[0][1].bit_size)
  for (pred,src) in srcs: nir.nir_phi_instr_add_src(phi, pred, src)
  return (phi.contents, getattr(phi.contents, "def"))

T = TypeVar('T', bound=ctypes.Structure)
def nir_cf_node_prev(n:nir.nir_cf_node, t:Type[T]) -> T: return ctypes.cast(n.node.prev, ctypes.POINTER(t)).contents
def nir_cf_node_next(n:nir.nir_cf_node, t:Type[T]) -> T: return ctypes.cast(n.node.next, ctypes.POINTER(t)).contents
def nir_before_cf_node(n:nir.nir_cf_node) -> nir.nir_cursor:
  if n.contents.type == nir.nir_cf_node_block: return nir.nir_cursor(nir.nir_cursor_after_block, block=ctypes.cast(n, ctypes.POINTER(nir.nir_block)))
  return nir.nir_cursor(nir.nir_cursor_before_block, block=nir_cf_node_next(n, nir.nir_block))
def nir_before_cf_list(l:nir.struct_exec_list) -> nir.nir_cursor:
  if (fn:=ctypes.cast(l.head_sentinel.next, ctypes.POINTER(nir.nir_cf_node))).contents.type == nir.nir_cf_node_block:
    return nir.nir_cursor(nir.nir_cursor_before_block, block=ctypes.cast(fn, ctypes.POINTER(nir.nir_block)))
  return nir.nir_cursor(nir.nir_cursor_after_block, block=nir_cf_node_next(fn.contents, nir.nir_block))
def nir_cursor_current_block(c:nir.nir_cursor) -> nir.nir_block:
  return c.instr.contents.block.contents if c.option == nir.nir_cursor_before_instr or c.option == nir.nir_cursor_after_instr else c.block.contents

def nir_store(b:nir.nir_builder, space:AddrSpace, addr:nir.nir_def, value:nir.nir_def, dtype:DType):
  intrin = getattr(nir, f"nir_intrinsic_store_{'global' if space == AddrSpace.GLOBAL else ('shared' if space == AddrSpace.LOCAL else 'deref')}")
  store = nir.nir_intrinsic_instr_create(b.shader, intrin)
  store.contents.num_components = value.num_components
  arr = ctypes.cast(store.contents.src, ctypes.POINTER(nir.nir_src))
  if space == AddrSpace.REG: arr[1], arr[0] = nir_src_for_ssa(value), nir_src_for_ssa(addr)
  else: arr[0], arr[1] = nir_src_for_ssa(value), nir_src_for_ssa(addr)
  nir_intrinsic_set(nir.NIR_INTRINSIC_WRITE_MASK, store, BITFIELD_MASK(value.num_components))
  if space != AddrSpace.REG:
    nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_MUL, store, dtype.itemsize)
    nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_OFFSET, store, 0) # is setting to zero the default case?
  nir.nir_builder_instr_insert(b, store.contents.instr)
  return addr

def nir_load(b:nir.nir_builder, space:AddrSpace, addr:nir.nir_def, dtype:DType) -> nir.nir_def:
  intrin = getattr(nir, f"nir_intrinsic_load_{'global' if space == AddrSpace.GLOBAL else ('shared' if space == AddrSpace.LOCAL else 'deref')}")
  load = nir.nir_intrinsic_instr_create(b.shader, intrin)
  load.contents.num_components = dtype.count
  ctypes.cast(load.contents.src, ctypes.POINTER(nir.nir_src))[0] = nir_src_for_ssa(addr)
  if space != AddrSpace.REG:
    nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_MUL, load, dtype.itemsize)
    nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_OFFSET, load, 0)
  nir.nir_def_init(load.contents.instr, getattr(load.contents, "def"), dtype.count, dtype.itemsize * 8 // dtype.count)
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

def nir_reg_idx(b:nir.nir_builder, reg:nir.nir_variable, idx:nir.nir_def) -> nir.nir_def:
  parent = nir.nir_deref_instr_create(b.shader, nir.nir_deref_type_var)
  parent.contents.modes, parent.contents.type, parent.contents.var = reg.data.mode, reg.type, ctypes.pointer(reg)
  nir.nir_def_init(parent.contents.instr, getattr(parent.contents, "def"), 1, 64)
  nir.nir_builder_instr_insert(b, parent.contents.instr)
  deref = nir.nir_deref_instr_create(b.shader, nir.nir_deref_type_array)
  deref.contents.modes, deref.contents.type = reg.data.mode, nir.glsl_get_array_element(reg.type)
  deref.contents.parent, deref.contents.arr.index = nir_src_for_ssa(getattr(parent.contents, "def")), nir_src_for_ssa(idx)
  nir.nir_def_init(deref.contents.instr, getattr(deref.contents, "def"), 1, 64)
  nir.nir_builder_instr_insert(b, deref.contents.instr)
  return getattr(deref.contents, "def")

def nir_barrier(b:nir.nir_builder):
  barrier = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_barrier)
  nir_intrinsic_set(nir.NIR_INTRINSIC_EXECUTION_SCOPE, barrier, nir.SCOPE_WORKGROUP)
  nir_intrinsic_set(nir.NIR_INTRINSIC_MEMORY_SCOPE, barrier, 0)
  nir_intrinsic_set(nir.NIR_INTRINSIC_MEMORY_SEMANTICS, barrier, 0)
  nir_intrinsic_set(nir.NIR_INTRINSIC_MEMORY_MODES, barrier, 0) # TODO
  nir.nir_builder_instr_insert(b, barrier.contents.instr)
  return nir.nir_def()

# alu ops, aop[<dtype>][<op>]
u_aop = { Ops.ADD: nir.nir_op_uadd_sat, Ops.MUL: nir.nir_op_imul, Ops.IDIV: nir.nir_op_udiv, Ops.MOD: nir.nir_op_umod, Ops.CMPLT: nir.nir_op_ult,
          Ops.CMPNE: nir.nir_op_ine, Ops.CMPEQ: nir.nir_op_ieq, Ops.OR: nir.nir_op_ior, Ops.AND: nir.nir_op_iand, Ops.XOR: nir.nir_op_ixor,
          Ops.WHERE: nir.nir_op_bcsel, Ops.MAX: nir.nir_op_umax}
s_aop = {**u_aop, Ops.ADD: nir.nir_op_iadd, Ops.CMPLT: nir.nir_op_ilt, Ops.IDIV: nir.nir_op_idiv, Ops.MOD: nir.nir_op_irem, Ops.MAX: nir.nir_op_imax}
f_aop = { Ops.ADD: nir.nir_op_fadd, Ops.MUL: nir.nir_op_fmul, Ops.CMPLT: nir.nir_op_flt, Ops.CMPNE: nir.nir_op_fneu, Ops.CMPEQ: nir.nir_op_feq,
         Ops.FDIV: nir.nir_op_fdiv, Ops.RECIP: nir.nir_op_frcp, Ops.MAX: nir.nir_op_fmax, Ops.TRUNC: nir.nir_op_ftrunc, Ops.SIN: nir.nir_op_fsin,
         Ops.EXP2: nir.nir_op_fexp2, Ops.LOG2: nir.nir_op_flog2}
aop = {**{x:u_aop for x in (dtypes.bool,)+dtypes.uints}, **{x:s_aop for x in dtypes.sints}, **{x:f_aop for x in dtypes.floats}}

def code(t:DType) -> str: return "i" if t in dtypes.ints else ("f" if t in dtypes.floats else "b")
def ncast(b:nir.nir_builder, src:nir.nir_def, it:DType, ot:DType) -> nir.nir_def:
  if isinstance(it, PtrDType) and ot == dtypes.long: return src
  if ot == dtypes.bool: return nir_build_alu(b, getattr(nir, f"nir_op_{code(it)}ne{'u' if code(it) == 'f' else ''}"), src, nir_imm(b, 0, it))
  return nir_build_alu(b, getattr(nir, f"nir_op_{code(it)}2{code(ot)}{ot.itemsize * 8}"), src)

def nif(b:nir.nir_builder, cond:nir.nir_def, go:Callable):
  nif = nir.nir_push_if(b, cond)
  go()
  nir.nir_pop_if(b, nif)

def nir_jump(b:nir.nir_builder, t:nir.nir_jump_type): nir.nir_builder_instr_insert(b, nir.nir_jump_instr_create(b.shader, t).contents.instr)

def if_phi(b:nir.nir_builder, cond:nir.nir_def, then_def:nir.nir_def, else_def:nir.nir_def) -> nir.nir_def:
  nir.nir_pop_if(b, nir.nir_push_if(b, cond))
  return nir.nir_if_phi(b, then_def, else_def).contents

# this is a ridiculous hack, but I can't find a better way to grab the glsl_type objects
glsl_base = {**{d:getattr(nir, f"GLSL_TYPE_{'U' if d in dtypes.uints else ''}INT{d.itemsize*8 if d.itemsize != 4 else ''}") for d in dtypes.ints},
             **{getattr(dtypes,d):getattr(nir, f"GLSL_TYPE_{d.upper()}") for d in ['bool', 'double', 'float', 'float16', 'bfloat16']},
             dtypes.fp8e4m3: nir.GLSL_TYPE_FLOAT_E4M3FN, dtypes.fp8e5m2: nir.GLSL_TYPE_FLOAT_E5M2}
def glsl_type(t:DType) -> nir.struct_glsl_type:
  if isinstance(t, PtrDType): return nir.glsl_array_type(glsl_type(t.base), t.size, 0).contents
  return nir.glsl_get_base_glsl_type(nir.glsl_type(base_type=glsl_base[t])).contents

def ensure(_) -> nir.nir_def: return nir.nir_def()

class NIRRenderer(Renderer):
  device = "NV"
  suffix = "NAK"
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max
  code_for_op = {**{k:lambda:None for k in u_aop.keys()}, **{k:lambda:None for k in s_aop.keys()}, **{k:lambda:None for k in f_aop.keys()}}

  extra_matcher = PatternMatcher([
    # move addrspace to load/store (.value because arg needs to be comparable)
    (UPat((Ops.LOAD,Ops.STORE), src=(UPat(Ops.INDEX, name="idx"),), allow_any_len=True, name="x"),
     lambda x,idx: UOp(x.op, x.dtype, x.src, idx.src[0].ptrdtype.addrspace.value)),
    # handle negative unsigned CONST
    (UPat.cvar("x", dtypes.uints), lambda x: UOp(Ops.CONST, dtype=x.dtype, arg=x.dtype.max+x.arg+1) if x.arg < 0 else None),
    # from ptx
    (UPat.var('x', dtype=dtypes.bool)<UPat.var('y'), lambda x,y: (x^True)&y),
    # load/store bool -> uint8
    (UPat(Ops.LOAD, dtypes.bool, name="x"),
     lambda x: x.replace(dtype=dtypes.uint8, src=x.src[0:1]+((x.src[1].cast(dtypes.uint8),) if len(x.src)>=2 else ())+x.src[2:]).cast(dtypes.bool)),
    (UPat(Ops.STORE, src=(UPat(), UPat(dtype=dtypes.bool)), name="x", allow_any_len=True),
     lambda x: x.replace(src=x.src[0:1] + (x.src[1].cast(dtypes.uint8),) + x.src[2:])),
    # load/store use pointer arithmetic, and the cast does nothing
    (UPat((Ops.LOAD,Ops.STORE), src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"))),), allow_any_len=True, name="x"),
     lambda x,buf,idx:
       x.replace(src=(buf.cast(dtypes.long)+idx.cast(dtypes.long)*buf.dtype.itemsize,)+x.src[1:]) if buf.dtype.addrspace != AddrSpace.REG else None),
    (UPat(Ops.CAST, name="x"),
     lambda x: x.src[0] if isinstance(x.dtype, PtrDType) or x.src[0].dtype == dtypes.void else None),
    # move mask from INDEX to the load/store to enable pointer arithmetic
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"), UPat.var("gate"))), UPat.var("alt")), name="x"),
     lambda x,buf,idx,gate,alt: UOp(Ops.LOAD, alt.dtype, (buf.index(idx), alt, gate), x.arg)),
    (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"), UPat())), UPat.var("val"), UPat.var("gate")), allow_any_len=True),
     lambda buf,idx,val,gate: UOp.store(buf.index(idx), val, gate)),
  ])

  def_rewrite = PatternMatcher([
    (UPat(Ops.CONST, name="x"), lambda ctx,x: nir_imm(ctx[0], x.arg, x.dtype)),
    (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx,x: nv_param(ctx[0], x.dtype, x.arg)),
    (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: nir_channel(ctx[0], nir_gid(ctx[0]) if x.arg[0][0] == 'g' else nir_lid(ctx[0]), int(x.arg[0][-1]))),
    (UPat(Ops.STORE, src=(UPat.var("addr"), UPat.var("val")), allow_any_len=True, name="x"),
     lambda ctx,x,addr,val: nir_store(ctx[0], AddrSpace(x.arg), ctx[1][addr], ctx[1][val], val.dtype)),
    (UPat(Ops.LOAD, src=(UPat.var("addr"),), name="x"), lambda ctx,x,addr: nir_load(ctx[0], AddrSpace(x.arg), ctx[1][addr], x.dtype)),
    (UPat(Ops.LOAD, name="x", src=(UPat.var('addr'), UPat(name='alt'), UPat(name="gate", op=GroupOp.ALU))),
     lambda ctx,x,addr,alt,gate: if_phi(ctx[0], ctx[1][gate], nir_load(ctx[0], AddrSpace(x.arg), ctx[1][addr], x.dtype), ctx[1][alt])),
    (UPat(Ops.LOAD, src=(UPat.var("addr"),), allow_any_len=True, name="x"),
     lambda ctx,x,addr: nir_load(ctx[0], AddrSpace(x.arg), ctx[1][addr], x.dtype)),
    (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: nir_build_alu(ctx[0], getattr(nir, f"nir_op_vec{x.dtype.count}"), *[ctx[1][src] for src in x.src])),
    (UPat(GroupOp.ALU, name="x"), lambda ctx,x: nir_build_alu(ctx[0], aop[x.src[0].dtype.scalar()][x.op], *[ctx[1][src] for src in x.src])),
    (UPat(Ops.CAST, name="x"), lambda ctx,x: ncast(ctx[0], ctx[1][x.src[0]], x.src[0].dtype, x.dtype)),
    (UPat(Ops.BITCAST, src=(UPat.var("a"),), allow_any_len=True), lambda ctx,a: ctx[1][a]),
    (UPat(Ops.GEP, src=(UPat.var("a"),), name="x"), lambda ctx,x,a: nir_channel(ctx[0], ctx[1][a], get_single_element(x.arg))),
    (UPat(Ops.DEFINE_REG, name="x"), lambda ctx,x:
     nir.nir_local_variable_create(ctx[0].impl, glsl_type(dtypes.uint8.ptr(x.dtype.size) if x.dtype.base == dtypes.bool else x.dtype),
                                   s(f"acc{x.arg[0]}")).contents),
    (UPat(Ops.INDEX, src=(UPat.var("reg"), UPat.var("idx"))), lambda ctx,reg,idx: nir_reg_idx(ctx[0], ctx[1][reg], ctx[1][idx])),
    (UPat(Ops.BARRIER), lambda ctx: nir_barrier(ctx[0])),
    (UPat(Ops.IF, name="x"), lambda ctx,x: nir.nir_push_if(ctx[0], ctx[1][x.src[0]])),
    (UPat(Ops.ENDIF, name="x"), lambda ctx,x: ensure(nir.nir_pop_if(ctx[0], ctx[1][x.src[0]])))
  ])

  def __init__(self, dev, device="NV"): self.device, self.dev = device, dev

  def render(self, uops:list[UOp]) -> str:
    b = nir.nir_builder_init_simple_shader(nir.MESA_SHADER_COMPUTE, self.dev.compiler.nir_options, None)
    # FIXME: this is wrong? wg_sz should be global size?
    for u in [u for u in uops if u.op is Ops.SPECIAL and u.arg[0][0] == "l"]: b.shader.contents.info.workgroup_size[int(u.arg[0][-1])] = u.arg[1]
    r: dict[UOp,nir.nir_def] = {}
    ranges: list[Tuple[nir.nir_loop, nir.nir_phi_instr]] = []

    # import os
    # input(f"pid: {os.getpid()}")
    nir.glsl_type_singleton_init_or_ref() # TODO: call glsl_type_singleton_decref somewhere
    for u in uops:
      # print(u)
      # nir.nir_print_shader(b.shader, stdout)
      if u.op == Ops.NOOP: pass
      elif u.op == Ops.SINK:
        # why do we care about setting this?
        if u.arg is not None: b.shader.contents.info.name = s(u.arg.function_name)
      elif u.op == Ops.DEFINE_LOCAL:
        r[u] = nir_imm(b, b.shader.contents.info.shared_size, dtypes.long)
        b.shader.contents.info.shared_size += u.dtype.nbytes()
      elif u.op == Ops.RANGE:
        zero = nir_imm(b, 0, u.dtype)
        phi, r[u] = nir_build_phi(b, [(nir_cf_node_prev((loop:=nir.nir_push_loop(b)).contents.cf_node, nir.nir_block), zero)])
        nif(b, nir_build_alu(b, nir.nir_op_inot, nir_build_alu(b, aop[u.dtype][Ops.CMPLT], r[u], r[u.src[0]])),
            lambda: nir_jump(b, cast(ctypes.c_uint32, nir.nir_jump_break)))
        ranges.append((loop, phi))
      elif u.op == Ops.ENDRANGE:
        loop, phi = ranges.pop()
        nir.nir_phi_instr_add_src(phi, nir_cursor_current_block(b.cursor),
                                  nir_build_alu(b, aop[u.src[0].dtype][Ops.ADD], r[u.src[0]], nir_imm(b, 1, u.src[0].dtype)))
        nir.nir_instr_insert(nir_before_cf_list(loop.contents.body), phi.instr)
        nir.nir_pop_loop(b, loop)
      else:
        if (d:=self.def_rewrite.rewrite(u, ctx=(b,r))) is None:
          nir.nir_print_shader(b.shader, stdout)
          raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        r[u] = cast(nir.nir_def, d)
    nir.nir_print_shader(b.shader, stdout)
    return b.shader.contents
