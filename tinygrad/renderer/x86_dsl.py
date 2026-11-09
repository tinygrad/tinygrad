from tinygrad.dtype import DType, dtypes, truncate, AddrSpace
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Ops
from tinygrad.uop.render import render_uir
from tinygrad.renderer.x86 import ISARenderer, X86Ops, IselContext, _xmm_sz, stack_pointer, args_pointer

def imm(dt:DType, v:int) -> UOp: return UOp.cconst(truncate[dt](v), dt)

# a memory address operand is (base, index, displacement, scale)
def fold_address(x:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  def _disp(v:int) -> UOp: return imm(dtypes.int32 if abs(v) > dtypes.int8.max else dtypes.int8, v)
  def _cast(v:UOp) -> UOp: return v.cast(dtypes.int64) if v.vmin < 0 else v.cast(dtypes.uint32) if v.dtype.itemsize < 4 else v
  if x.op not in {Ops.INDEX, Ops.SHRINK}: return (x, UOp(Ops.NOOP), _disp(0), imm(dtypes.uint32, x.dtype.itemsize))
  base, idx = x.src[0], x.src[1]
  scale = 1 if base in (stack_pointer(), args_pointer()) else base.dtype.itemsize
  sz = imm(dtypes.uint32, base.dtype.itemsize)
  if idx.op is Ops.ADD and (c:=idx.src[1]).op is Ops.CAST and c.src[0].op is Ops.CONST:
    return (base, _cast(idx.src[0]), _disp(c.src[0].val * scale), sz)
  if idx.op is Ops.CAST and idx.src[0].op is Ops.CONST: return (base, UOp(Ops.NOOP), _disp(idx.src[0].val * scale), sz)
  return (base, _cast(idx), _disp(0), sz)

def store_op(op:X86Ops, base:UOp, index:UOp, displacement:UOp, scale:UOp, value:UOp, *extra:UOp) -> UOp:
  dst = base.param_like(0)
  val = UOp.param(4, value.dtype, (1,), addrspace=AddrSpace.REG).load()
  body = [dst.index(0).store(val)]
  if index.op is not Ops.NOOP: body.append(UOp.param(1, index.dtype, (1,), addrspace=AddrSpace.REG).load())
  return UOp(Ops.ASM, src=tuple(body), arg=op).call(base, index, displacement, scale, value, *extra)

def load_op(op:X86Ops, dt:DType, count:int, base:UOp, index:UOp, displacement:UOp, scale:UOp, *extra:UOp) -> UOp:
  dst = UOp.alloc((count,), dt, addrspace=AddrSpace.REG)
  mem = UOp.param(1, dt, (count,))
  body = [UOp.param(0, dt, (count,), addrspace=AddrSpace.REG).index(0).store(mem.index(0).load())]
  if index.op is not Ops.NOOP: body.append(UOp.param(2, index.dtype, (1,), addrspace=AddrSpace.REG).load())
  ins = UOp(Ops.ASM, src=tuple(body), arg=op).call(dst, base, index, displacement, scale, *extra)
  return dst.after(ins)

def reg_op(op:X86Ops, dt:DType, count:int, *src:UOp) -> UOp:
  dst = UOp.alloc((count,), dt, addrspace=AddrSpace.REG)
  reads = [UOp.param(i+1, s.dtype, (s.max_numel(),), addrspace=AddrSpace.REG).load() for i,s in enumerate(src)]
  body = [UOp.param(0, dt, (count,), addrspace=AddrSpace.REG).index(0).store(reads[0]), *reads[1:]]
  ins = UOp(Ops.ASM, src=tuple(body), arg=op).call(dst, *src)
  return dst.after(ins)

def float_binary(x:UOp) -> UOp:
  op = X86Ops.VADDSS if x.dtype is dtypes.float32 else X86Ops.VADDSD
  return reg_op(op, x.dtype, x.max_numel(), *x.src)

def abi(ctx:IselContext, x:UOp) -> UOp|None:
  if x not in ctx.arg_offsets: return None
  dt = x.dtype if x.addrspace is AddrSpace.ALU else dtypes.uint64
  dst = UOp.alloc((1,), dt, addrspace=AddrSpace.REG)
  args = UOp.param(1, dtypes.uint64, (1,), addrspace=AddrSpace.REG, name="rdi")
  body = UOp.param(0, dt, (1,), addrspace=AddrSpace.REG).index(0).store(args.index(0).load())
  ins = UOp(Ops.ASM, src=(body,), arg=X86Ops.MOV).call(dst, args, UOp(Ops.NOOP), imm(dtypes.int8, ctx.arg_offsets[x]), imm(dtypes.uint32, 8))
  return dst.after(ins)

def float_load(a:UOp, x:UOp):
  if a.addrspace is AddrSpace.REG: return None
  if x.max_numel() * x.dtype.itemsize == 2: return load_op(X86Ops.VPINSRW, x.dtype, x.max_numel(), *fold_address(a), imm(dtypes.uint8, 0))
  return load_op(_xmm_sz(x), x.dtype, x.max_numel(), *fold_address(a))

def float_store(a:UOp, b:UOp, x:UOp):
  if a.addrspace is AddrSpace.REG: return None
  if b.max_numel() * b.dtype.itemsize == 2: return store_op(X86Ops.VPEXTRW, *fold_address(a), b, imm(dtypes.uint8, 0))
  return store_op(X86Ops[_xmm_sz(b).name+"m"], *fold_address(a), b)

class X86DSLRenderer(ISARenderer):
  device = "CPU"
  has_local = False
  global_max = (1, 0, 0)
  pre_isel_matcher = PatternMatcher([])
  isel_matcher = PatternMatcher([
    (UPat(Ops.PARAM, name="x"), abi),
    (UPat(Ops.LOAD, dtypes.floats, src=(UPat(name="a"),), name="x"), float_load),
    (UPat(Ops.ADD, (dtypes.float32, dtypes.float64), name="x"), float_binary),
    (UPat.var("a").store(UPat.var("b", dtypes.floats), name="x"), float_store),
    #lambda a,b,x: None if a.addrspace is AddrSpace.REG else
    # asm(X86Ops.VPEXTRW, *fold_address(a), b, imm(dtypes.uint8, 0)) if b.max_numel() * b.dtype.itemsize == 2 else
    # asm(X86Ops[_xmm_sz(b).name+"m"], *fold_address(a), b)),
  ])
  pre_regalloc_matcher = PatternMatcher([])
  post_regalloc_matcher = PatternMatcher([])

  def render(self, uops:list[UOp]) -> str: return render_uir(uops)
  def asm_str(self, uops:list[UOp], function_name:str) -> str: return self.render(uops)
