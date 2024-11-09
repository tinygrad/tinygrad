from typing import List, Dict
from tinygrad.renderer import Renderer
from tinygrad.ops import UOp, PatternMatcher, UPat, Ops, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType

# this will unify ptx and llvm
type_map = {
  dtypes.int8: "i8", dtypes.int16: "i16", dtypes.int32: "i32", dtypes.int64: "i64",
  dtypes.uint8: "i8", dtypes.uint16: "i16", dtypes.uint32: "i32", dtypes.uint64: "i64",
  dtypes.float16: "half", dtypes.float32: "float", dtypes.float64: "double", dtypes.bool: "i1", dtypes.void: "void"}
def ldt(dt:DType):
  if isinstance(dt, PtrDType): return ldt(dt.base) + "*"
  return type_map[dt]

def lcast(input_type:DType, output_type:DType):
  if dtypes.is_unsigned(input_type) and dtypes.is_unsigned(output_type): return 'trunc' if output_type.itemsize < input_type.itemsize else 'zext'
  if dtypes.is_unsigned(input_type) and dtypes.is_float(output_type): return 'uitofp'
  if dtypes.is_signed(input_type) and dtypes.is_float(output_type): return 'sitofp'
  if dtypes.is_unsigned(input_type) and dtypes.is_signed(output_type): return 'sext'
  if dtypes.is_float(input_type) and dtypes.is_unsigned(output_type): return 'fptoui'
  if dtypes.is_float(input_type) and dtypes.is_signed(output_type): return 'fptosi'
  raise RuntimeError(f"cast from {input_type} to {output_type} not supported")

unsigned_lop = {
  Ops.ADD: "add", Ops.MUL: "mul", Ops.CMPLT: "icmp ult", Ops.CMPNE: "icmp ne", Ops.OR: "or", Ops.AND: "and", Ops.XOR: "xor", Ops.IDIV: "udiv",
}
signed_lop = {**unsigned_lop, Ops.CMPLT: "icmp slt", Ops.IDIV: "sdiv"}
float_lop = {Ops.ADD: "fadd", Ops.MUL: "fmul", Ops.CMPLT: "fcmp ult", Ops.CMPNE: "fcmp une",}

# total lop
lop = {**{x:unsigned_lop for x in (dtypes.bool,)+dtypes.uints}, **{x:signed_lop for x in dtypes.sints}, **{x:float_lop for x in dtypes.floats}}

llvm_rewrite = PatternMatcher([
  (UPat(Ops.INDEX, name="x"), lambda ctx,x:
   f"  {ctx[x]} = getelementptr inbounds {ldt(x.dtype.base)}, {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, i32 {ctx[x.src[1]]}"),
  (UPat(Ops.LOAD, name="x"), lambda ctx,x: f"  {ctx[x]} = load {ldt(x.dtype)}, {ldt(x.src[0].dtype)} {ctx[x.src[0]]}"),
  (UPat((Ops.STORE, Ops.ASSIGN), name="x"), lambda ctx,x: f"  store {ldt(x.src[1].dtype)} {ctx[x.src[1]]}, {ldt(x.src[0].dtype)} {ctx[x.src[0]]}"),
  (UPat(Ops.DEFINE_ACC, name="x"), lambda ctx, x: f"  {ctx[x]} = alloca {ldt(x.dtype.base)}"),
  (UPat(Ops.RANGE, name="x"), lambda ctx,x:
   f"  br label %loop_entry_{x.arg[0]}\nloop_entry_{x.arg[0]}:\n"
   f"  br label %loop_body_{x.arg[0]}\nloop_body_{x.arg[0]}:\n"
   f"  {ctx[x]} = phi {ldt(x.dtype)} [{ctx[x.src[0]]}, %loop_entry_{x.arg[0]}], [{ctx[x]}phi, %loop_latch_{x.arg[0]}]"),
  (UPat(Ops.WHERE, name="x"), lambda ctx,x:
   f"  {ctx[x]} = select {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, {ldt(x.src[1].dtype)} {ctx[x.src[1]]}, {ldt(x.src[2].dtype)} {ctx[x.src[2]]}"),
  (UPat(GroupOp.Binary, name="x"), lambda ctx,x: f"  {ctx[x]} = {lop[x.src[0].dtype][x.op]} {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, {ctx[x.src[1]]}"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"  {ctx[x]} = bitcast {ldt(x.src[0].dtype)} {ctx[x.src[0]]} to {ldt(x.dtype)}"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"  {ctx[x]} = {lcast(x.src[0].dtype, x.dtype)} {ldt(x.src[0].dtype)} {ctx[x.src[0]]} to {ldt(x.dtype)}"),
  (UPat(Ops.ENDRANGE, name="x"), lambda ctx,x:
   f"  br label %loop_latch_{x.src[0].arg[0]}\nloop_latch_{x.src[0].arg[0]}:\n"
   f"  {ctx[x.src[0]]}phi = add i32 {ctx[x.src[0]]}, 1\n  {ctx[x]} = icmp ult i32 {ctx[x.src[0]]}phi, {ctx[x.src[0].src[1]]}\n"
   f"  br i1 {ctx[x]}, label %loop_body_{x.src[0].arg[0]}, label %loop_exit_{x.src[0].arg[0]}\nloop_exit_{x.src[0].arg[0]}:")
])

# DEFINE_ACC is an alloca ptr in LLVM, do a fixup for that
extra_pm = PatternMatcher([
  (UPat(Ops.DEFINE_ACC, name="x"), lambda x:
    UOp(Ops.DEFINE_ACC, x.dtype.ptr(), x.src, x.arg).load(dtype=x.dtype) if not isinstance(x.dtype, PtrDType) else None),
  (UPat(Ops.ASSIGN, src=(UPat(Ops.LOAD), ), allow_any_len=True, name="x"), lambda x: UOp(Ops.ASSIGN, x.dtype, (x.src[0].src[0],)+x.src[1:])),
])

class LLVM2Renderer(Renderer):
  device = "LLVM"
  supports_vectorized = False
  supports_float4 = False
  has_local = False
  has_shared = False
  global_max = None
  extra_matcher = extra_pm

  def __getitem__(self, key): return self.r[key]  # hacky helper
  def render(self, name: str, uops: List[UOp]) -> str:
    r: Dict[UOp, str] = {}
    bufs: Dict[str, DType] = {}
    self.r = r
    kernel = []
    self.var_counter = 0

    for u in uops:
      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        r[u] = f"%data{u.arg}" if u.op is Ops.DEFINE_GLOBAL else f"%{u.arg[0]}"
        bufs[r[u]] = u.dtype
      elif u.op is Ops.CONST:
        r[u] = f"{u.arg}" if u.dtype is not dtypes.bool else f"{int(u.arg)}"
      else:
        if u.op is Ops.ASSIGN: r[u] = r[u.src[1]]
        else: r[u] = f"%v{self.var_counter}"
        self.var_counter += 1
        l = llvm_rewrite.rewrite(u, ctx=self)
        if l is None: raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        kernel.append(l)

    args = ', '.join([f"{ldt(dtype)} {name}" for name, dtype in bufs.items()])
    code = f"define void @{name}({args}) {{\n" + '\n'.join(kernel) + "\n  ret void\n}"
    return code
