from typing import Callable, cast, Tuple, Type, TypeVar
from tinygrad.dtype import AddrSpace, DType, PtrDType, dtypes
from tinygrad.helpers import all_same, get_single_element
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.uop.ops import GroupOp, Ops, UOp, PatternMatcher, UPat
import tinygrad.runtime.autogen.nir as nir
import tinygrad.runtime.autogen.libc as libc
import ctypes, struct

# FIXME: this is because clang2py produces bad output?
nir_intrinsic_infos = nir.nir_intrinsic_infos.in_dll(nir._libraries['FIXME_STUB'], "nir_intrinsic_infos")
assert libc._libraries['libc']
stdout = ctypes.POINTER(nir.struct__IO_FILE).in_dll(libc._libraries['libc'], "stdout")
s = nir.char_pointer_cast
def g(s:str): return getattr(nir, s)
def d(i) -> nir.nir_def: return getattr(i.contents, "def")

def BITFIELD_BIT(b): return 1 << b
def BITFIELD_MASK(b): return 0xFFFFFFFF if b == 32 else BITFIELD_BIT(b & 31) - 1

def nir_mov_alu(b:nir.nir_builder, src:nir.nir_alu_src, num_components:int) -> nir.nir_def:
  mov = nir.nir_alu_instr_create(b.shader, nir.nir_op_mov)
  nir.nir_def_init(mov.contents.instr, d(mov), num_components, src.src.ssa.contents.bit_size)
  mov.contents.exact, mov.contents.fp_fast_math = b.exact, b.fp_fast_math
  ctypes.cast(mov.contents.src, ctypes.POINTER(nir.nir_alu_src))[0] = src
  nir.nir_builder_instr_insert(b, mov.contents.instr)
  return d(mov)

def nir_swizzle(b:nir.nir_builder, src:nir.nir_def, swiz:list[int]) -> nir.nir_def:
  alu_src, is_id = nir.nir_alu_src(), True
  alu_src.src = nir_src_for_ssa(src)
  for i, s in enumerate(swiz):
    if i != s: is_id = False
    alu_src.swizzle[i] = s
  if len(swiz) == src.num_components and is_id: return src
  return nir_mov_alu(b, alu_src, len(swiz))

def nchannel(b:nir.nir_builder, src:nir.nir_def, c:int) -> nir.nir_def: return nir_swizzle(b, src, [c])

def nimm(b:nir.nir_builder, x, dtype:DType) -> nir.nir_def:
  assert dtype.fmt
  instr = nir.nir_load_const_instr_create(b.shader, 1, 1 if dtype == dtypes.bool else dtype.itemsize * 8)
  struct.pack_into(dtype.fmt, (ctypes.c_ubyte * dtype.itemsize).from_address(ctypes.addressof(instr.contents.value)), 0, x)
  nir.nir_builder_instr_insert(b, instr.contents.instr)
  return d(instr)

def nir_src_for_ssa(d:nir.nir_def) -> nir.nir_src: return nir.nir_src(ssa=ctypes.pointer(d))
def nir_intrinsic_set(typ, instr:nir.nir_intrinsic_instr, val:int):
  info = nir_intrinsic_infos[instr.contents.intrinsic]
  assert info.index_map[typ] > 0
  instr.contents.const_index[info.index_map[typ] - 1] = val

def nalu(b:nir.nir_builder, op, *srcs:nir.nir_def) -> nir.nir_def:
  if len(srcs) == 1: return nir.nir_build_alu1(b, op, srcs[0]).contents
  if len(srcs) == 2: return nir.nir_build_alu2(b, op, srcs[0], srcs[1]).contents
  if len(srcs) == 3: return nir.nir_build_alu3(b, op, srcs[0], srcs[1], srcs[2]).contents
  return nir.nir_build_alu4(b, op, srcs[0], srcs[1], srcs[2], srcs[3]).contents

def nphi(b:nir.nir_builder, srcs:list[Tuple[nir.nir_block, nir.nir_def]]) -> Tuple[nir.nir_phi_instr, nir.nir_def]:
  assert all_same([src[1].num_components for src in srcs]) and all_same([src[1].bit_size for src in srcs])
  phi = nir.nir_phi_instr_create(b.shader)
  nir.nir_def_init(phi.contents.instr, d(phi), srcs[0][1].num_components, srcs[0][1].bit_size)
  for (pred,src) in srcs: nir.nir_phi_instr_add_src(phi, pred, src)
  return (phi.contents, d(phi))

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
def current_block(c:nir.nir_cursor) -> nir.nir_block:
  return c.instr.contents.block.contents if c.option == nir.nir_cursor_before_instr or c.option == nir.nir_cursor_after_instr else c.block.contents

def nstore(b:nir.nir_builder, space:AddrSpace, addr:nir.nir_def, value:nir.nir_def, dtype:DType):
  intrin = g(f"nir_intrinsic_store_{'global' if space == AddrSpace.GLOBAL else ('shared' if space == AddrSpace.LOCAL else 'deref')}")
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

def nload(b:nir.nir_builder, space:AddrSpace, addr:nir.nir_def, dtype:DType) -> nir.nir_def:
  intrin = g(f"nir_intrinsic_load_{'global' if space == AddrSpace.GLOBAL else ('shared' if space == AddrSpace.LOCAL else 'deref')}")
  load = nir.nir_intrinsic_instr_create(b.shader, intrin)
  load.contents.num_components = dtype.count
  ctypes.cast(load.contents.src, ctypes.POINTER(nir.nir_src))[0] = nir_src_for_ssa(addr)
  if space != AddrSpace.REG:
    nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_MUL, load, dtype.itemsize)
    nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_OFFSET, load, 0)
  nir.nir_def_init(load.contents.instr, d(load), dtype.count, dtype.itemsize * 8 // dtype.count)
  nir.nir_builder_instr_insert(b, load.contents.instr)
  return d(load)

def ngid(b:nir.nir_builder) -> nir.nir_def:
  intrin = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_load_workgroup_id)
  nir.nir_def_init(intrin.contents.instr, d(intrin), 3, 32)
  nir.nir_builder_instr_insert(b, intrin.contents.instr)
  return d(intrin)

def nlid(b:nir.nir_builder) -> nir.nir_def:
  intrin = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_load_local_invocation_id)
  nir.nir_def_init(intrin.contents.instr, d(intrin), 3, 32)
  nir.nir_builder_instr_insert(b, intrin.contents.instr)
  return d(intrin)


def nreg_idx(b:nir.nir_builder, reg:nir.nir_variable, idx:nir.nir_def) -> nir.nir_def:
  parent = nir.nir_deref_instr_create(b.shader, nir.nir_deref_type_var)
  parent.contents.modes, parent.contents.type, parent.contents.var = reg.data.mode, reg.type, ctypes.pointer(reg)
  nir.nir_def_init(parent.contents.instr, d(parent), 1, 64)
  nir.nir_builder_instr_insert(b, parent.contents.instr)
  deref = nir.nir_deref_instr_create(b.shader, nir.nir_deref_type_array)
  deref.contents.modes, deref.contents.type = reg.data.mode, nir.glsl_get_array_element(reg.type)
  deref.contents.parent, deref.contents.arr.index = nir_src_for_ssa(d(parent)), nir_src_for_ssa(idx)
  nir.nir_def_init(deref.contents.instr, d(deref), 1, 64)
  nir.nir_builder_instr_insert(b, deref.contents.instr)
  return d(deref)

def nbarrier(b:nir.nir_builder):
  barrier = nir.nir_intrinsic_instr_create(b.shader, nir.nir_intrinsic_barrier)
  # TODO: what are the right values here?
  nir_intrinsic_set(nir.NIR_INTRINSIC_EXECUTION_SCOPE, barrier, nir.SCOPE_WORKGROUP)
  nir_intrinsic_set(nir.NIR_INTRINSIC_MEMORY_SCOPE, barrier, 0)
  nir_intrinsic_set(nir.NIR_INTRINSIC_MEMORY_SEMANTICS, barrier, 0)
  nir_intrinsic_set(nir.NIR_INTRINSIC_MEMORY_MODES, barrier, 0)
  nir.nir_builder_instr_insert(b, barrier.contents.instr)

# alu ops, aop[<dtype>][<op>]
u_aop = { Ops.ADD: nir.nir_op_iadd, Ops.MUL: nir.nir_op_imul, Ops.IDIV: nir.nir_op_udiv, Ops.MOD: nir.nir_op_umod, Ops.CMPLT: nir.nir_op_ult,
          Ops.CMPNE: nir.nir_op_ine, Ops.CMPEQ: nir.nir_op_ieq, Ops.OR: nir.nir_op_ior, Ops.AND: nir.nir_op_iand, Ops.XOR: nir.nir_op_ixor,
          Ops.WHERE: nir.nir_op_bcsel, Ops.MAX: nir.nir_op_umax}
s_aop = {**u_aop, Ops.CMPLT: nir.nir_op_ilt, Ops.IDIV: nir.nir_op_idiv, Ops.MOD: nir.nir_op_irem, Ops.MAX: nir.nir_op_imax}
f_aop = { Ops.ADD: nir.nir_op_fadd, Ops.MUL: nir.nir_op_fmul, Ops.CMPLT: nir.nir_op_flt, Ops.CMPNE: nir.nir_op_fneu, Ops.CMPEQ: nir.nir_op_feq,
          Ops.FDIV: nir.nir_op_fdiv, Ops.RECIP: nir.nir_op_frcp, Ops.MAX: nir.nir_op_fmax, Ops.TRUNC: nir.nir_op_ftrunc, Ops.SIN: nir.nir_op_fsin,
          Ops.EXP2: nir.nir_op_fexp2, Ops.LOG2: nir.nir_op_flog2}
aop = {**{x:u_aop for x in (dtypes.bool,)+dtypes.uints}, **{x:s_aop for x in dtypes.sints}, **{x:f_aop for x in dtypes.floats}}

def c(t:DType, u:bool=True) -> str: return "u" if t in dtypes.uints and u else ("i" if t in dtypes.ints else ("f" if t in dtypes.floats else "b"))
def ncast(b:nir.nir_builder, src:nir.nir_def, it:DType, ot:DType) -> nir.nir_def:
  if isinstance(it, PtrDType) and ot == dtypes.long: return src
  if ot == dtypes.bool: return nalu(b, g(f"nir_op_{c(it, False)}ne{'u' if c(it) == 'f' else ''}"), src, nimm(b, 0, it))
  return nalu(b, g(f"nir_op_{c(it)}2{c(it) if it in dtypes.ints and ot in dtypes.ints else c(ot, ot == dtypes.bool)}{ot.itemsize*8}"), src)

def nif(b:nir.nir_builder, cond:nir.nir_def, go:Callable):
  nif = nir.nir_push_if(b, cond)
  go()
  nir.nir_pop_if(b, nif)

def njump(b:nir.nir_builder, t): nir.nir_builder_instr_insert(b, nir.nir_jump_instr_create(b.shader, t).contents.instr)

def if_phi(b:nir.nir_builder, cond:nir.nir_def, then_fn:Callable[[],nir.nir_def], else_def:nir.nir_def) -> nir.nir_def:
  nif = nir.nir_push_if(b, cond)
  then_def = then_fn()
  nir.nir_pop_if(b, nif)
  return nir.nir_if_phi(b, then_def, else_def).contents

# this is a ridiculous hack, but I can't find a better way to grab the glsl_type objects
glsl_base = {**{d:g(f"GLSL_TYPE_{'U' if d in dtypes.uints else ''}INT{d.itemsize*8 if d.itemsize != 4 else ''}") for d in dtypes.ints},
             **{getattr(dtypes,d):g(f"GLSL_TYPE_{d.upper()}") for d in ['bool', 'double', 'float', 'float16', 'bfloat16']},
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

  def nv_param(self, dtype:DType) -> nir.nir_def:
    intrin = nir.nir_intrinsic_instr_create(self.b.shader, nir.nir_intrinsic_ldc_nv)
    intrin.contents.num_components = 1
    nir.nir_def_init(intrin.contents.instr, d(intrin), 1, 64 if isinstance(dtype, PtrDType) else dtype.itemsize * 8)
    arr = ctypes.cast(intrin.contents.src, ctypes.POINTER(nir.nir_src))
    arr[0], arr[1] = nir_src_for_ssa(nimm(self.b, 0, dtypes.int)), nir_src_for_ssa(nimm(self.b, 0x160 + self.param_idx * 8, dtypes.int))
    nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_MUL, intrin, d(intrin).bit_size // 8)
    nir.nir_builder_instr_insert(self.b, intrin.contents.instr)
    self.param_idx += 1
    return d(intrin)

  def_rewrite = PatternMatcher([
    (UPat(Ops.CONST, name="x"), lambda ctx,x: nimm(ctx.b, x.arg, x.dtype)),
    (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx,x: ctx.nv_param(x.dtype)),
    (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: nchannel(ctx.b, ngid(ctx.b) if x.arg[0] == 'g' else nlid(ctx.b), int(x.arg[-1]))),
    (UPat(Ops.STORE, src=(UPat.var("loc"), UPat.var("val")), allow_any_len=True, name="x"),
     lambda ctx,x,loc,val: nstore(ctx.b, AddrSpace(x.arg), ctx.r[loc], ctx.r[val], val.dtype)),
    (UPat(Ops.LOAD, src=(UPat.var("loc"),), name="x"), lambda ctx,x,loc: nload(ctx.b, AddrSpace(x.arg), ctx.r[loc], x.dtype)),
    (UPat(Ops.LOAD, name="x", src=(UPat.var('loc'), UPat(name='alt'), UPat(name="gate", op=GroupOp.ALU))),
     lambda ctx,x,loc,alt,gate: if_phi(ctx.b, ctx.r[gate], lambda: nload(ctx.b, AddrSpace(x.arg), ctx.r[loc], x.dtype), ctx.r[alt])),
    (UPat(Ops.LOAD, src=(UPat.var("loc"),), allow_any_len=True, name="x"), lambda ctx,x,loc: nload(ctx.b, AddrSpace(x.arg), ctx.r[loc], x.dtype)),
    (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: nalu(ctx.b, g(f"nir_op_vec{x.dtype.count}"), *[ctx.r[src] for src in x.src])),
    (UPat(GroupOp.ALU, name="x"), lambda ctx,x: nalu(ctx.b, aop[x.src[0].dtype.scalar()][x.op], *[ctx.r[src] for src in x.src])),
    (UPat(Ops.CAST, name="x"), lambda ctx,x: ncast(ctx.b, ctx.r[x.src[0]], x.src[0].dtype, x.dtype)),
    (UPat(Ops.BITCAST, src=(UPat.var("a"),), allow_any_len=True), lambda ctx,a: ctx.r[a]),
    (UPat(Ops.GEP, src=(UPat.var("a"),), name="x"), lambda ctx,x,a: nchannel(ctx.b, ctx.r[a], get_single_element(x.arg))),
    (UPat(Ops.DEFINE_REG, name="x"), lambda ctx,x:
     nir.nir_local_variable_create(ctx.b.impl, glsl_type(dtypes.uint8.ptr(x.dtype.size) if x.dtype.base == dtypes.bool else x.dtype),
                                   s(f"acc{x.arg[0]}")).contents),
    (UPat(Ops.INDEX, src=(UPat.var("reg"), UPat.var("idx"))), lambda ctx,reg,idx: nreg_idx(ctx.b, ctx.r[reg], ctx.r[idx])),
    (UPat(Ops.BARRIER), lambda ctx: ensure(nbarrier(ctx.b))),
    (UPat(Ops.IF, name="x"), lambda ctx,x: nir.nir_push_if(ctx.b, ctx.r[x.src[0]])),
    (UPat(Ops.ENDIF, name="x"), lambda ctx,x: ensure(nir.nir_pop_if(ctx.b, ctx.r[x.src[0]])))
  ])

  def __init__(self, dev, device="NV"): self.device, self.dev = device, dev

  def render(self, uops:list[UOp]) -> str:
    self.b = nir.nir_builder_init_simple_shader(nir.MESA_SHADER_COMPUTE, self.dev.compiler.nir_options, None)
    for u in [u for u in uops if u.op is Ops.SPECIAL and u.arg[0] == "l"]: self.b.shader.contents.info.workgroup_size[int(u.arg[-1])] = u.src[0].arg
    self.r, self.param_idx = {}, 0
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
        if u.arg is not None: self.b.shader.contents.info.name = s(u.arg.function_name)
      elif u.op == Ops.DEFINE_LOCAL:
        self.r[u] = nimm(self.b, self.b.shader.contents.info.shared_size, dtypes.long)
        self.b.shader.contents.info.shared_size += u.dtype.nbytes()
      elif u.op == Ops.RANGE:
        zero = nimm(self.b, 0, u.dtype)
        phi, self.r[u] = nphi(self.b, [(nir_cf_node_prev((loop:=nir.nir_push_loop(self.b)).contents.cf_node, nir.nir_block), zero)])
        nif(self.b, nalu(self.b, nir.nir_op_inot, nalu(self.b, aop[u.dtype][Ops.CMPLT], self.r[u], self.r[u.src[0]])),
            lambda: njump(self.b, nir.nir_jump_break))
        ranges.append((loop, phi))
      elif u.op == Ops.ENDRANGE:
        loop, phi = ranges.pop()
        nir.nir_phi_instr_add_src(phi, current_block(self.b.cursor), nalu(self.b, aop[u.src[0].dtype][Ops.ADD], self.r[u.src[0]],
                                                                          nimm(self.b, 1, u.src[0].dtype)))
        nir.nir_instr_insert(nir_before_cf_list(loop.contents.body), phi.instr)
        nir.nir_pop_loop(self.b, loop)
      else:
        if (d:=self.def_rewrite.rewrite(u, ctx=self)) is None:
          nir.nir_print_shader(self.b.shader, stdout)
          raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        self.r[u] = cast(nir.nir_def, d)
    nir.nir_print_shader(self.b.shader, stdout)
    return self.b.shader.contents
