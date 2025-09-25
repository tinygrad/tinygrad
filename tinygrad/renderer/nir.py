from typing import Callable, cast, Tuple
from functools import partial
from tinygrad.dtype import AddrSpace, DType, PtrDType, dtypes
from tinygrad.helpers import all_same
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

def nalu(b:nir.nir_builder, op:str, *srcs:nir.nir_def) -> nir.nir_def:
  if len(srcs) == 1: return nir.nir_build_alu1(b, g(f"nir_op_{op}"), srcs[0]).contents
  if len(srcs) == 2: return nir.nir_build_alu2(b, g(f"nir_op_{op}"), srcs[0], srcs[1]).contents
  if len(srcs) == 3: return nir.nir_build_alu3(b, g(f"nir_op_{op}"), srcs[0], srcs[1], srcs[2]).contents
  return nir.nir_build_alu4(b, g(f"nir_op_{op}"), srcs[0], srcs[1], srcs[2], srcs[3]).contents

def nphi(b:nir.nir_builder, srcs:list[Tuple[nir.nir_block, nir.nir_def]]) -> Tuple[nir.nir_phi_instr, nir.nir_def]:
  assert all_same([src[1].num_components for src in srcs]) and all_same([src[1].bit_size for src in srcs])
  phi = nir.nir_phi_instr_create(b.shader)
  nir.nir_def_init(phi.contents.instr, d(phi), srcs[0][1].num_components, srcs[0][1].bit_size)
  for (pred,src) in srcs: nir.nir_phi_instr_add_src(phi, pred, src)
  return (phi.contents, d(phi))

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

def deref_var(b:nir.nir_builder, var:nir.nir_variable) -> nir.nir_def:
  deref = nir.nir_deref_instr_create(b.shader, nir.nir_deref_type_var)
  deref.contents.modes, deref.contents.type, deref.contents.var = var.data.mode, var.type, ctypes.pointer(var)
  nir.nir_def_init(deref.contents.instr, d(deref), 1, 32)
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
u_aop = { Ops.ADD: "iadd", Ops.MUL: "imul", Ops.IDIV: "udiv", Ops.MOD: "umod", Ops.CMPLT: "ult", Ops.CMPNE: "ine", Ops.CMPEQ: "ieq", Ops.OR: "ior",
          Ops.AND: "iand", Ops.XOR: "ixor", Ops.WHERE: "bcsel", Ops.MAX: "umax"}
s_aop = {**u_aop, Ops.CMPLT: "ilt", Ops.IDIV: "idiv", Ops.MOD: "irem", Ops.MAX: "imax"}
f_aop = { Ops.ADD: "fadd", Ops.MUL: "fmul", Ops.CMPLT: "flt", Ops.CMPNE: "fneu", Ops.CMPEQ: "feq", Ops.FDIV: "fdiv", Ops.RECIP: "frcp",
          Ops.MAX: "fmax", Ops.TRUNC: "ftrunc", Ops.SIN: "fsin", Ops.EXP2: "fexp2", Ops.LOG2: "flog2"}
aop = {**{x:u_aop for x in (dtypes.bool,)+dtypes.uints}, **{x:s_aop for x in dtypes.sints}, **{x:f_aop for x in dtypes.floats}}

def c(t:DType, u:bool=True) -> str: return "u" if t in dtypes.uints and u else ("i" if t in dtypes.ints else ("f" if t in dtypes.floats else "b"))
def ncast(b:nir.nir_builder, src:nir.nir_def, it:DType, ot:DType) -> nir.nir_def:
  if isinstance(it, PtrDType) and ot == dtypes.long: return src
  if ot == dtypes.bool: return nalu(b, c(it, False)+'ne'+('u' if c(it) == 'f' else ''), src, nimm(b, 0, it))
  return nalu(b, f"{c(it)}2{c(it) if it in dtypes.ints and ot in dtypes.ints else c(ot, ot == dtypes.bool)}{ot.itemsize*8}", src)

def nif(b:nir.nir_builder, cond:nir.nir_def, then_fn:Callable, else_fn:Callable):
  nif = nir.nir_push_if(b, cond)
  t = then_fn()
  nir.nir_push_else(b, nif)
  e = else_fn()
  nir.nir_pop_if(b, nif)
  return t, e

def njump(b:nir.nir_builder, typ, tgt=None, cond=None, else_tgt=None):
  jmp = nir.nir_jump_instr_create(b.shader, typ)
  if tgt is not None: jmp.contents.target = ctypes.pointer(tgt)
  if cond is not None: jmp.contents.condition, jmp.contents.else_target = nir_src_for_ssa(cond), ctypes.pointer(else_tgt)
  nir.nir_builder_instr_insert(b, jmp.contents.instr)

def if_phi(b:nir.nir_builder, cond, then_fn, else_fn): return nir.nir_if_phi(b, *nif(b, cond, then_fn, else_fn)).contents

# this is a ridiculous hack, but I can't find a better way to grab the glsl_type objects
glsl_base = {**{d:g(f"GLSL_TYPE_{'U' if d in dtypes.uints else ''}INT{d.itemsize*8 if d.itemsize != 4 else ''}") for d in dtypes.ints},
             **{getattr(dtypes,d):g(f"GLSL_TYPE_{d.upper()}") for d in ['bool', 'double', 'float', 'float16', 'bfloat16']},
             dtypes.fp8e4m3: nir.GLSL_TYPE_FLOAT_E4M3FN, dtypes.fp8e5m2: nir.GLSL_TYPE_FLOAT_E5M2}
def glsl_type(t:DType) -> nir.struct_glsl_type:
  if isinstance(t, PtrDType): return nir.glsl_array_type(glsl_type(t.base), t.size, 0).contents
  return nir.glsl_get_base_glsl_type(nir.glsl_type(base_type=glsl_base[t])).contents

def ensure(_) -> nir.nir_def: return nir.nir_def()

def nidx(b:nir.nir_builder, buf, off, dtype, gate=None) -> nir.nir_def:
  def reg():
    deref = nir.nir_deref_instr_create(b.shader, nir.nir_deref_type_array)
    deref.contents.modes, deref.contents.type = buf.data.mode, nir.glsl_get_array_element(buf.type)
    deref.contents.parent, deref.contents.arr.index = nir_src_for_ssa(deref_var(b, buf)), nir_src_for_ssa(off)
    nir.nir_def_init(deref.contents.instr, d(deref), 1, 32)
    nir.nir_builder_instr_insert(b, deref.contents.instr)
    return d(deref)
  f = reg if dtype.addrspace == AddrSpace.REG else lambda: nalu(b, "iadd", buf, nalu(b, "imul", off, nimm(b, dtype.itemsize, dtypes.long)))
  return if_phi(b, gate, f, lambda: buf) if gate is not None else f()

class NIRRenderer(Renderer):
  device = "NV"
  suffix = "NAK"
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max
  code_for_op = {**{k:lambda:None for k in u_aop.keys()}, **{k:lambda:None for k in s_aop.keys()}, **{k:lambda:None for k in f_aop.keys()}}

  extra_matcher = PatternMatcher([
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
    (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("off")), allow_any_len=True, name="x"),
     lambda x,buf,off: x.replace(src=(buf,off.cast(dtypes.long))+x.src[2:]) if buf.dtype.addrspace != AddrSpace.REG and off.op != Ops.CAST else None),
    (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) or x.src[0].dtype == dtypes.void else None),
  ])

  def param(self, dtype:DType, sz:int) -> nir.nir_def: raise NotImplementedError("needs param")

  def_rewrite = PatternMatcher([
    (UPat(Ops.CONST, name="x"), lambda ctx,x: nimm(ctx.b, x.arg, x.dtype)),
    (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx,x: ctx.param(x.dtype, 8)),
    (UPat(Ops.DEFINE_VAR, name="x"), lambda ctx,x: ctx.param(x.dtype, 4)),
    (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: nchannel(ctx.b, ngid(ctx.b) if x.arg[0] == 'g' else nlid(ctx.b), int(x.arg[-1]))),
    (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("off"))), UPat.var("val")), allow_any_len=True, name="x"),
     lambda ctx,x,buf,off,val: nstore(ctx.b, buf.ptrdtype.addrspace, nidx(ctx.b, ctx.r[buf], ctx.r[off], buf.dtype), ctx.r[val], val.dtype)),
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("off"), UPat.var("gate"))), UPat.var("alt")), allow_any_len=True, name="x"),
     lambda ctx,x,buf,off,alt,gate: if_phi(ctx.b, ctx.r[gate],
      lambda: nload(ctx.b, buf.ptrdtype.addrspace, nidx(ctx.b, ctx.r[buf], ctx.r[off], buf.dtype, ctx.r[gate]), x.dtype), lambda: ctx.r[alt])),
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("off"))),), allow_any_len=True, name="x"),
     lambda ctx,x,buf,off: nload(ctx.b, buf.ptrdtype.addrspace, nidx(ctx.b, ctx.r[buf], ctx.r[off], buf.dtype), x.dtype)),
    (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: nalu(ctx.b, f"vec{x.dtype.count}", *[ctx.r[src] for src in x.src])),
    (UPat(GroupOp.ALU, name="x"), lambda ctx,x: nalu(ctx.b, aop[x.src[0].dtype.scalar()][x.op], *[ctx.r[src] for src in x.src])),
    (UPat(Ops.CAST, name="x"), lambda ctx,x: ncast(ctx.b, ctx.r[x.src[0]], x.src[0].dtype, x.dtype)),
    (UPat(Ops.BITCAST, src=(UPat.var("a"),), allow_any_len=True), lambda ctx,a: ctx.r[a]),
    (UPat(Ops.GEP, src=(UPat.var("a"),), name="x"), lambda ctx,x,a: nchannel(ctx.b, ctx.r[a], x.arg[0])),
    (UPat(Ops.DEFINE_REG, name="x"), lambda ctx,x:
     nir.nir_local_variable_create(ctx.b.impl, glsl_type(dtypes.uint8.ptr(x.dtype.size) if x.dtype.base == dtypes.bool else x.dtype),
                                   f"acc{x.arg[0]}".encode()).contents),
    (UPat(Ops.BARRIER), lambda ctx: ensure(nbarrier(ctx.b))),
    (UPat(Ops.IF, name="x"), lambda ctx,x: nir.nir_push_if(ctx.b, ctx.r[x.src[0]])),
    (UPat(Ops.ENDIF, name="x"), lambda ctx,x: ensure(nir.nir_pop_if(ctx.b, ctx.r[x.src[0]])))
  ])

  def __init__(self, dev, device): self.device, self.dev = device, dev

  def prerender(self, uops:list[UOp]):
    nir.glsl_type_singleton_init_or_ref() # TODO: call glsl_type_singleton_decref somewhere
    self.b = nir.nir_builder_init_simple_shader(nir.MESA_SHADER_COMPUTE, self.dev.compiler.nir_options, None)

  def render(self, uops:list[UOp]):
    self.prerender(uops)
    for u in [u for u in uops if u.op is Ops.SPECIAL and u.arg[0] == "l"]: self.b.shader.contents.info.workgroup_size[int(u.arg[-1])] = u.src[0].arg
    self.r, self.param_idx, ranges = {}, 0, []

    # import os
    # input(f"pid: {os.getpid()}")
    for u in uops:
      # print(u)
      # nir.nir_print_shader(self.b.shader, stdout)
      if u.op == Ops.NOOP: pass
      elif u.op == Ops.SINK:
        # why do we care about setting this?
        if u.arg is not None: self.b.shader.contents.info.name = s(u.arg.function_name)
      elif u.op == Ops.DEFINE_LOCAL:
        self.r[u] = nimm(self.b, self.b.shader.contents.info.shared_size, dtypes.long)
        self.b.shader.contents.info.shared_size += u.dtype.nbytes()
      elif u.op == Ops.RANGE:
        ranges.append(i:=deref_var(self.b, nir.nir_local_variable_create(self.b.impl, glsl_type(u.dtype), f"idx{u.arg[0]}".encode()).contents))
        nstore(self.b, AddrSpace.REG, i, nimm(self.b, 0, u.dtype), u.dtype)
        nir.nir_push_loop(self.b)
        self.r[u] = nload(self.b, AddrSpace.REG, i, u.dtype)
      elif u.op == Ops.ENDRANGE:
        nif(self.b, nalu(self.b, "ilt", x:=nalu(self.b, "iadd", self.r[u.src[0]], nimm(self.b, 1, u.src[0].dtype)), self.r[u.src[0].src[0]]),
            partial(nstore, self.b, AddrSpace.REG, ranges.pop(), x, u.src[0].dtype), lambda: njump(self.b, nir.nir_jump_break))
        nir.nir_pop_loop(self.b, None)
      elif u.op == Ops.INDEX: pass
      else:
        if (d:=self.def_rewrite.rewrite(u, ctx=self)) is None:
          nir.nir_print_shader(self.b.shader, stdout)
          raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        self.r[u] = cast(nir.nir_def, d)
    nir.nir_print_shader(self.b.shader, stdout)
    nir.nir_validate_shader(self.b.shader, b"after render")
    blob = nir.struct_blob()
    nir.nir_serialize(blob, self.b.shader, False)
    return ctypes.string_at(blob.data, blob.size)

class NAKRenderer(NIRRenderer):
  def __init__(self, dev, device="NV"): super().__init__(dev, device)

  def param(self, dtype:DType, sz:int) -> nir.nir_def:
    intrin = nir.nir_intrinsic_instr_create(self.b.shader, nir.nir_intrinsic_ldc_nv)
    intrin.contents.num_components = 1
    nir.nir_def_init(intrin.contents.instr, d(intrin), 1, sz * 8)
    arr = ctypes.cast(intrin.contents.src, ctypes.POINTER(nir.nir_src))
    arr[0], arr[1] = nir_src_for_ssa(nimm(self.b, 0, dtypes.int)), nir_src_for_ssa(nimm(self.b, self.param_idx, dtypes.int))
    nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_MUL, intrin, sz)
    nir.nir_builder_instr_insert(self.b, intrin.contents.instr)
    self.param_idx += sz
    return d(intrin)

class LVPRenderer(NIRRenderer):
  has_local = False
  global_max = (1, 0, 0)

  def __init__(self, dev, device="CPU"): super().__init__(dev, device)

  def param(self, dtype:DType, sz:int) -> nir.nir_def:
    intrin = nir.nir_intrinsic_instr_create(self.b.shader, nir.nir_intrinsic_load_ubo)
    intrin.contents.num_components = 1
    nir.nir_def_init(intrin.contents.instr, d(intrin), 1, sz * 8)
    arr = ctypes.cast(intrin.contents.src, ctypes.POINTER(nir.nir_src))
    arr[0], arr[1] = nir_src_for_ssa(nimm(self.b, 0, dtypes.int)), nir_src_for_ssa(nimm(self.b, self.param_idx, dtypes.int))
    nir_intrinsic_set(nir.NIR_INTRINSIC_ALIGN_MUL, intrin, sz)
    nir_intrinsic_set(nir.NIR_INTRINSIC_RANGE, intrin, self.paramsz)
    nir.nir_builder_instr_insert(self.b, intrin.contents.instr)
    self.param_idx += sz
    return d(intrin)

  def prerender(self, uops:list[UOp]):
    super().prerender(uops)
    # TODO: does this do anything?
    self.paramsz = sum([8 if u.op == Ops.DEFINE_GLOBAL else u.dtype.itemsize for u in uops if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR)])

