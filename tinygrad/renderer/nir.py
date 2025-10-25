from typing import Callable, cast, Any
from tinygrad.dtype import AddrSpace, DType, PtrDType, dtypes
from tinygrad.helpers import DEBUG, OSX, unwrap
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.uop.ops import GroupOp, Ops, UOp, PatternMatcher, UPat
import tinygrad.runtime.autogen.mesa as mesa
import base64, ctypes, ctypes.util, struct, functools, inspect

def g(s:str): return getattr(mesa, s)
def nsrc(d:mesa.nir_def) -> mesa.nir_src: return mesa.nir_src(ssa=ctypes.pointer(d))

# this is a ridiculous hack, but I can't find a better way to grab the glsl_type objects
glsl_base = {**{d:g(f"GLSL_TYPE_{'U' if d in dtypes.uints else ''}INT{d.itemsize*8 if d.itemsize != 4 else ''}") for d in dtypes.ints},
             **{getattr(dtypes,d):g(f"GLSL_TYPE_{d.upper()}") for d in ['double', 'float', 'float16']}, dtypes.bool: mesa.GLSL_TYPE_UINT8}
def glsl_type(t:DType) -> mesa.struct_glsl_type:
  if isinstance(t, PtrDType): return mesa.glsl_array_type(glsl_type(t.base), t.size, 0).contents
  return mesa.glsl_get_base_glsl_type(mesa.glsl_type(base_type=glsl_base[t])).contents

# alu ops, aop[<dtype>][<op>]
u_aop = { Ops.ADD: "iadd", Ops.MUL: "imul", Ops.IDIV: "udiv", Ops.MOD: "umod", Ops.CMPLT: "ult", Ops.CMPNE: "ine", Ops.CMPEQ: "ieq", Ops.OR: "ior",
          Ops.AND: "iand", Ops.XOR: "ixor", Ops.WHERE: "bcsel", Ops.MAX: "umax"}
s_aop = {**u_aop, Ops.CMPLT: "ilt", Ops.IDIV: "idiv", Ops.MOD: "irem", Ops.MAX: "imax"}
f_aop = { Ops.ADD: "fadd", Ops.MUL: "fmul", Ops.CMPLT: "flt", Ops.CMPNE: "fneu", Ops.CMPEQ: "feq", Ops.FDIV: "fdiv", Ops.RECIP: "frcp",
          Ops.MAX: "fmax", Ops.TRUNC: "ftrunc", Ops.SIN: "fsin", Ops.EXP2: "fexp2", Ops.LOG2: "flog2"}
aop = {**{x:u_aop for x in (dtypes.bool,)+dtypes.uints}, **{x:s_aop for x in dtypes.sints}, **{x:f_aop for x in dtypes.floats}}

def c(t:DType, u:bool=True) -> str: return "u" if t in dtypes.uints and u else ("i" if t in dtypes.ints else ("f" if t in dtypes.floats else "b"))
def ncast(b:mesa.nir_builder, src:mesa.nir_def, it:DType, ot:DType) -> mesa.nir_def:
  if isinstance(it, PtrDType) and ot == dtypes.long: return src
  if ot == dtypes.bool: return nalu(b, c(it, False)+'ne'+('u' if c(it) == 'f' else ''), src, nimm(b, 0, it))
  return nalu(b, f"{c(it)}2{c(it) if it in dtypes.ints and ot in dtypes.ints else c(ot, ot == dtypes.bool)}{ot.itemsize*8}", src)

def nif(b:mesa.nir_builder, cond:mesa.nir_def, then_fn:Callable, else_fn:Callable):
  nif = mesa.nir_push_if(b, cond)
  t = then_fn()
  mesa.nir_push_else(b, nif)
  e = else_fn()
  mesa.nir_pop_if(b, nif)
  return t, e

def nalu(b:mesa.nir_builder, op:str, *srcs:mesa.nir_def) -> mesa.nir_def: return g(f"nir_build_alu{len(srcs)}")(b, g(f"nir_op_{op}"), *srcs).contents

def nir_instr(nc=1, bs=lambda: None, intrins=None, srcs=None, has_def=True, df=None, also=lambda: None, **contents):
  def dec(f:Callable):
    @functools.wraps(f)
    def wrapper(*args, **kwargs) -> mesa.nir_def:
      (ba:=inspect.signature(f).bind(*args, **kwargs)).apply_defaults()
      def go(g): return g(**{nm: ba.arguments[nm] for nm in inspect.signature(g).parameters}) if callable(g) else g

      instr = f(*args, **kwargs)
      if has_def: mesa.nir_def_init(instr.contents.instr, getattr(instr.contents, "def"), go(nc), go(bs))
      for k, v in go(intrins or {}).items():
        idx = mesa.nir_intrinsic_infos[instr.contents.intrinsic].index_map[g(f"NIR_INTRINSIC_{k}")]
        assert idx > 0
        instr.contents.const_index[idx - 1] = go(v)
      for i, src in enumerate(go(srcs or [])): ctypes.cast(instr.contents.src, ctypes.POINTER(mesa.nir_src))[i] = go(src)
      for k,v in {k:vcomp for k,v in contents.items() if (vcomp:=go(v)) is not None}.items(): setattr(instr.contents, k, go(v))
      mesa.nir_builder_instr_insert(ba.arguments['b'], instr.contents.instr)
      go(also)
      return getattr(instr.contents, "def") if has_def else (mesa.nir_def() if df is None else go(df))
    return wrapper
  return dec

@nir_instr(nc=1, bs=lambda src: src.bit_size, exact=lambda b:b.exact, fp_fast_math=lambda b:b.fp_fast_math)
def nchannel(b:mesa.nir_builder, src:mesa.nir_def, c:int):
  alu_src = mesa.nir_alu_src(src=nsrc(src))
  alu_src.swizzle[0] = c
  mov = mesa.nir_alu_instr_create(b.shader, mesa.nir_op_mov)
  ctypes.cast(mov.contents.src, ctypes.POINTER(mesa.nir_alu_src))[0] = alu_src
  return mov

@nir_instr(nc=1, bs=lambda dtype: 1 if dtype == dtypes.bool else dtype.itemsize * 8)
def nimm(b:mesa.nir_builder, x, dtype:DType) -> mesa.nir_def:
  instr = mesa.nir_load_const_instr_create(b.shader, 1, 1 if dtype == dtypes.bool else dtype.itemsize * 8)
  struct.pack_into(unwrap(dtype.fmt), (ctypes.c_ubyte * dtype.itemsize).from_address(ctypes.addressof(instr.contents.value)), 0, x)
  return instr

deref_var = nir_instr(nc=1, bs=32, modes=lambda var:var.data.mode, type=lambda var:var.type, var=lambda var:ctypes.pointer(var))( # pylint: disable=W0108
  lambda b, var: mesa.nir_deref_instr_create(b.shader, mesa.nir_deref_type_var))

def iointr(space): return {"ALIGN_MUL":lambda dtype:dtype.itemsize} if space != AddrSpace.REG else {}
def scope(space): return 'global' if space == AddrSpace.GLOBAL else ('shared' if space == AddrSpace.LOCAL else 'deref')
nstore = nir_instr(has_def=False, df=lambda addr:addr, intrins=lambda space,val: {"WRITE_MASK":(1<<val.num_components)-1, **iointr(space)},
  num_components=lambda val:val.num_components, srcs=lambda space, addr, val: [nsrc(val), nsrc(addr)][::1 if space != AddrSpace.REG else -1])(
    lambda b, space, addr, val, dtype: mesa.nir_intrinsic_instr_create(b.shader, g(f"nir_intrinsic_store_{scope(space)}")))
nload = nir_instr(nc=lambda dtype:dtype.count, bs=lambda dtype:dtype.itemsize*8//dtype.count, num_components=lambda dtype:dtype.count,
  intrins=lambda space:{**({"ACCESS":mesa.ACCESS_CAN_REORDER} if space==AddrSpace.GLOBAL else {}), **iointr(space)}, srcs=lambda addr: [nsrc(addr)])(
    lambda b, space, addr, dtype: mesa.nir_intrinsic_instr_create(b.shader, g(f"nir_intrinsic_load_{scope(space)}")))

ngid = nir_instr(nc=3, bs=32)(lambda b: mesa.nir_intrinsic_instr_create(b.shader, mesa.nir_intrinsic_load_workgroup_id))
nlid = nir_instr(nc=3, bs=32)(lambda b: mesa.nir_intrinsic_instr_create(b.shader, mesa.nir_intrinsic_load_local_invocation_id))

nbarrier = nir_instr(has_def=False, intrins={"EXECUTION_SCOPE":mesa.SCOPE_WORKGROUP})(
  lambda b: mesa.nir_intrinsic_instr_create(b.shader, mesa.nir_intrinsic_barrier))

@nir_instr(has_def=False, target=lambda tgt:tgt and ctypes.pointer(tgt), condition=lambda cond:cond and nsrc(cond),
           else_target=lambda else_tgt: else_tgt and ctypes.pointer(else_tgt))
def njump(b:mesa.nir_builder, typ, tgt=None, cond=None, else_tgt=None): return mesa.nir_jump_instr_create(b.shader, typ)

def if_phi(b:mesa.nir_builder, cond, then_fn, else_fn): return mesa.nir_if_phi(b, *nif(b, cond, then_fn, else_fn)).contents

def nidx(b:mesa.nir_builder, buf, off, dtype, gate=None) -> mesa.nir_def:
  @nir_instr(nc=1, bs=32, modes=lambda buf: buf.data.mode, type=lambda buf: mesa.glsl_get_array_element(buf.type))
  def reg(b, buf):
    deref = mesa.nir_deref_instr_create(b.shader, mesa.nir_deref_type_array)
    deref.contents.parent, deref.contents.arr.index = nsrc(deref_var(b, buf)), nsrc(off)
    return deref
  f = (functools.partial(reg, b, buf) if dtype.addrspace == AddrSpace.REG else
       lambda: nalu(b, "iadd", buf, nalu(b, "imul", off, nimm(b, dtype.itemsize, dtypes.long))))
  return if_phi(b, gate, f, lambda: buf) if gate is not None else f()

class NIRRenderer(Renderer):
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

  def_rewrite = PatternMatcher([
    (UPat(Ops.CONST, name="x"), lambda ctx,x: nimm(ctx.b, x.arg, x.dtype)),
    (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx,x: ctx.param(ctx.b, x.dtype, 8)),
    (UPat(Ops.DEFINE_VAR, name="x"), lambda ctx,x: ctx.param(ctx.b, x.dtype, 4)),
    (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: nchannel(ctx.b, ngid(ctx.b) if x.arg[0] == 'g' else nlid(ctx.b), int(x.arg[-1]))),
    (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat.var("buf"),UPat.var("off")), allow_any_len=True), UPat.var("val")), allow_any_len=True, name="x"),
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
    (UPat(Ops.DEFINE_REG, name="x"), lambda ctx,x:mesa.nir_local_variable_create(ctx.b.impl, glsl_type(x.dtype), f"acc{x.arg[0]}".encode()).contents),
    (UPat(Ops.BARRIER), lambda ctx: nbarrier(ctx.b)),
    (UPat(Ops.IF, name="x"), lambda ctx,x: mesa.nir_push_if(ctx.b, ctx.r[x.src[0]])),
    (UPat(Ops.ENDIF, name="x"), lambda ctx,x: (lambda _: mesa.nir_def())(mesa.nir_pop_if(ctx.b, ctx.r[x.src[0]])))
  ])

  def __init__(self): mesa.glsl_type_singleton_init_or_ref()

  def __del__(self):
    try: mesa.glsl_type_singleton_decref()
    except FileNotFoundError: pass

  @property
  def nir_options(self): raise NotImplementedError("needs nir_options")
  def param(self, b:mesa.nir_builder, dtype:DType, sz:int) -> mesa.nir_def: raise NotImplementedError("needs param")
  def prerender(self, uops:list[UOp]):
    self.b = mesa.nir_builder_init_simple_shader(mesa.MESA_SHADER_COMPUTE, mesa.nir_shader_compiler_options.from_buffer_copy(self.nir_options), None)

  def render(self, uops:list[UOp]):
    self.prerender(uops)
    for u in [u for u in uops if u.op is Ops.SPECIAL and u.arg[0] == "l"]: self.b.shader.contents.info.workgroup_size[int(u.arg[-1])] = u.src[0].arg
    self.r: dict[UOp, Any] = {}
    self.param_idx, ranges = 0, []

    for u in uops:
      if u.op in {Ops.NOOP, Ops.GROUP, Ops.INDEX}: pass
      elif u.op is Ops.AFTER:
        self.r[u] = self.r[u.src[0]]
      elif u.op == Ops.SINK:
        if u.arg is not None: self.b.shader.contents.info.name = mesa.char_pointer_cast(u.arg.function_name)
      elif u.op == Ops.DEFINE_LOCAL:
        self.r[u] = nimm(self.b, self.b.shader.contents.info.shared_size, dtypes.long)
        self.b.shader.contents.info.shared_size += u.dtype.nbytes()
      elif u.op == Ops.RANGE:
        ranges.append(i:=deref_var(self.b, mesa.nir_local_variable_create(self.b.impl, glsl_type(u.dtype), f"idx{u.arg[0]}".encode()).contents))
        nstore(self.b, AddrSpace.REG, i, nimm(self.b, 0, u.dtype), u.dtype)
        mesa.nir_push_loop(self.b)
        self.r[u] = nload(self.b, AddrSpace.REG, i, u.dtype)
      elif u.op == Ops.END:
        r = u.src[1]
        nif(self.b, nalu(self.b, "ilt", x:=nalu(self.b, "iadd", self.r[r], nimm(self.b, 1, r.dtype)), self.r[r.src[0]]),
            functools.partial(nstore, self.b, AddrSpace.REG, ranges.pop(), x, r.dtype), lambda: njump(self.b, mesa.nir_jump_break))
        mesa.nir_pop_loop(self.b, None)
      else:
        if (d:=self.def_rewrite.rewrite(u, ctx=self)) is None: raise RuntimeError(f"failed to render {u.op} srcs {[x.dtype for x in u.src]}")
        self.r[u] = cast(mesa.nir_def, d)

    mesa.nir_validate_shader(self.b.shader, b"after render")
    if DEBUG >= 4: mesa.nir_print_shader(self.b.shader, ctypes.POINTER(mesa.struct__IO_FILE).in_dll(ctypes.CDLL(ctypes.util.find_library('c')),
                                                                                                    "__stdoutp" if OSX else "stdout"))
    mesa.nir_serialize(blob:=mesa.struct_blob(), self.b.shader, False)
    ret = base64.b64encode(ctypes.string_at(blob.data, blob.size)).decode()

    mesa.ralloc_free(self.b.shader)
    ctypes.CDLL(None).free(blob.data)
    del self.b, self.r

    return ret

class NAKRenderer(NIRRenderer):
  device = "NV"
  def __init__(self, dev=None, nir_options=None):
    self.dev, self._nir_options = dev, nir_options
    super().__init__()

  def __reduce__(self): return NAKRenderer, (None, self.nir_options,)

  @property
  def nir_options(self):
    if self._nir_options is None: self._nir_options = self.dev.compiler.nir_options
    return self._nir_options

  param = nir_instr(nc=1, num_components=1, bs=lambda sz:sz*8, also=lambda self,sz: setattr(self, "param_idx", self.param_idx + sz),
    intrins={"ALIGN_MUL":lambda sz:sz}, srcs=lambda self,b: [nsrc(nimm(b, 0, dtypes.int)), nsrc(nimm(b, self.param_idx, dtypes.int))])(
       lambda self, b, dtype, sz: mesa.nir_intrinsic_instr_create(b.shader, mesa.nir_intrinsic_ldc_nv))

class LVPRenderer(NIRRenderer):
  device = "CPU"
  has_local = False
  has_shared = False
  global_max = (1, 0, 0)
  nir_options = mesa.lvp_nir_options

  param = nir_instr(nc=1, bs=lambda sz: sz * 8, num_components=1, intrins={"ALIGN_MUL":lambda sz: sz, "RANGE":lambda self: self.param_sz},
    srcs=lambda b, self: [nsrc(nimm(b, 0, dtypes.int)), nsrc(nimm(b, self.param_idx, dtypes.int))], also=lambda self, sz:
    setattr(self, "param_idx", self.param_idx+sz))(lambda self, b, dtype, sz: mesa.nir_intrinsic_instr_create(b.shader, mesa.nir_intrinsic_load_ubo))

  def prerender(self, uops:list[UOp]):
    super().prerender(uops)
    self.param_sz = sum([8 if u.op == Ops.DEFINE_GLOBAL else u.dtype.itemsize for u in uops if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR)])

