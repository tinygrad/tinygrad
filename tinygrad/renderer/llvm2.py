from typing import List, Dict, Tuple
from tinygrad.renderer import Renderer
from tinygrad.ops import UOp, PatternMatcher, UPat, Ops, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType

# this will unify ptx and llvm
type_map = {
  dtypes.int8: "i8", dtypes.int16: "i16", dtypes.int32: "i32", dtypes.int64: "i64",
  dtypes.uint8: "i8", dtypes.uint16: "i16", dtypes.uint32: "i32", dtypes.uint64: "i64",
  dtypes.float16: "half", dtypes.float32: "float", dtypes.float64: "double", dtypes.bool: "i1", dtypes.void: "void"}
def llvm_dtype(dt:DType):
  if isinstance(dt, PtrDType): return llvm_dtype(dt.base) + "*"
  return type_map[dt]

code_for_op = {
  Ops.ADD: "add", Ops.MUL: "mul",
}

float_code_for_op = {
  Ops.ADD: "fadd", Ops.MUL: "fmul",
}

llvm_rewrite = PatternMatcher([
  #(UPat(Ops.CONST, name="x"), lambda ctx,x: f"constant {ctx.render_dtype(x.dtype)} {x.arg}"),
  (UPat(Ops.INDEX, name="x"), lambda ctx,x: f"  {ctx[x]} = getelementptr inbounds {llvm_dtype(x.dtype.base)}, {llvm_dtype(x.src[0].dtype)} {ctx[x.src[0]]}, i32 {ctx[x.src[1]]}"),
  (UPat(Ops.LOAD, name="x"), lambda ctx,x: f"  {ctx[x]} = load {llvm_dtype(x.dtype)}, {llvm_dtype(x.src[0].dtype)} {ctx[x.src[0]]}"),
  (UPat(Ops.STORE, name="x"), lambda ctx,x: f"  store {llvm_dtype(x.src[1].dtype)} {ctx[x.src[1]]}, {llvm_dtype(x.src[0].dtype)} {ctx[x.src[0]]}"),
  (UPat(Ops.RANGE, name="x"), lambda ctx,x:
   f"  br label %loop_entry_{x.arg[0]}\nloop_entry_{x.arg[0]}:\n"
   f"  br label %loop_body_{x.arg[0]}\nloop_body_{x.arg[0]}:\n"
   f"  {ctx[x]} = phi {llvm_dtype(x.dtype)} [0, %loop_entry_{x.arg[0]}], [{ctx[x]}phi, %loop_latch_{x.arg[0]}]"),
  (UPat(GroupOp.ALU, dtype=dtypes.floats, name="x"), lambda ctx,x: f"  {ctx[x]} = {float_code_for_op[x.op]} {llvm_dtype(x.dtype)} {ctx[x.src[0]]}, {ctx[x.src[1]]}"),
  (UPat(GroupOp.ALU, name="x"), lambda ctx,x: f"  {ctx[x]} = {code_for_op[x.op]} {llvm_dtype(x.dtype)} {ctx[x.src[0]]}, {ctx[x.src[1]]}"),
  (UPat(Ops.ENDRANGE, name="x"), lambda ctx,x:
   f"  br label %loop_latch_{x.src[0].arg[0]}\nloop_latch_{x.src[0].arg[0]}:\n"
   f"  {ctx[x.src[0]]}phi = add i32 {ctx[x.src[0]]}, 1\n  {ctx[x]} = icmp ult i32 {ctx[x.src[0]]}phi, 3\n"
   f"  br i1 {ctx[x]}, label %loop_body_{x.src[0].arg[0]}, label %loop_exit_{x.src[0].arg[0]}\nloop_exit_{x.src[0].arg[0]}:")
])

class LLVM2Renderer(Renderer):
  device = "LLVM"
  supports_vectorized = False
  supports_float4 = False
  has_local = False
  has_shared = False
  global_max = None

  def __getitem__(self, key): return self.r[key]  # hacky helper
  def render(self, name: str, uops: List[UOp]) -> str:
    r: Dict[UOp, str] = {}
    bufs: Dict[str, DType] = {}
    self.r = r
    kernel = []
    self.var_counter = 0

    for u in uops:
      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        r[u] = f"%data{u.arg}" if u.op is Ops.DEFINE_GLOBAL else u.arg[0]
        bufs[r[u]] = u.dtype
      elif u.op is Ops.CONST:
        r[u] = f"{u.arg}"
      else:
        r[u] = f"%v{self.var_counter}"
        self.var_counter += 1
        kernel.append(str(llvm_rewrite.rewrite(u, ctx=self)))

    args = ', '.join([f"{llvm_dtype(dtype)} {name}" for name, dtype in bufs.items()])
    code = f"define void @{name}({args}) {{\n" + '\n'.join(kernel) + "\n  ret void\n}"
    return code
