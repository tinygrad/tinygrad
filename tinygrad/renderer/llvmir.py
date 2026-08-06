import math, struct, sys
from tinygrad.codegen.opt import tc
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import HIPRenderer, create_non_native_float_pats, pm_manual_bf16_cast
from tinygrad.codegen.decomp.transcendental import xexp2, xlog2
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, GroupOp, range_str
from tinygrad.dtype import dtypes, float_to_fp8, DType, truncate, AddrSpace
from tinygrad.helpers import prod, Target, NUM_CPU_THREADS, getenv, OSX

def is_volatile(u:UOp) -> bool: return (buf:=u.buf_uop).op is Ops.PARAM and buf.arg.volatile

def ldt(dt:DType, count=1, ptr=False):
  if ptr: return ldt(dt, count) + "*"
  if count > 1: return f"<{count} x {ldt(dt, 1, ptr)}>"
  return {dtypes.void: "void", dtypes.bool: "i1", dtypes.int8: "i8", dtypes.int16: "i16", dtypes.int32: "i32", dtypes.int64: "i64",
          dtypes.uint8: "i8", dtypes.uint16: "i16", dtypes.uint32: "i32", dtypes.uint64: "i64", dtypes.fp8e4m3: "i8", dtypes.fp8e5m2: "i8",
          dtypes.float16: "half", dtypes.bfloat16: "bfloat", dtypes.float32: "float", dtypes.float64: "double"}[dt]

def lconst(x, dtype:DType):
  if dtype in dtypes.floats:
    if dtype in dtypes.fp8s: return float_to_fp8(x, dtype)
    if math.isinf(x) or math.isnan(x): return "0x%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    return truncate[dtype](x)
  return int(x)

def lcast(input_type:DType, output_type:DType):
  if dtypes.is_float(input_type):
    if dtypes.is_float(output_type): return 'fpext' if output_type.itemsize > input_type.itemsize else 'fptrunc'
    if dtypes.is_int(output_type): return 'fptoui' if dtypes.is_unsigned(output_type) else 'fptosi'
  if dtypes.is_unsigned(input_type) or dtypes.is_bool(input_type):
    if dtypes.is_float(output_type): return 'uitofp'
    if dtypes.is_int(output_type): return 'trunc' if output_type.itemsize < input_type.itemsize else 'zext'
  if dtypes.is_int(input_type):
    if dtypes.is_float(output_type): return 'sitofp'
    if dtypes.is_int(output_type): return 'trunc' if output_type.itemsize < input_type.itemsize else 'sext'
  raise NotImplementedError(f"cast from {input_type} -> {output_type} not implemented")

def render_wmma_amd(ctx, wmma: UOp, cdna=False, rdna4=False) -> str:
  dt_map = {dtypes.half: "f16", dtypes.float: "f32", dtypes.ushort: "bf16.1k" if cdna else "bf16", dtypes.bfloat16: "bf16.1k" if cdna else "bf16",
            dtypes.fp8e4m3: ".fp8.fp8", dtypes.fp8e5m2: ".bf8.bf8", dtypes.int8: "iu8", dtypes.int32: "i32"}
  # https://github.com/llvm/llvm-project/blob/main/clang/test/CodeGenOpenCL/builtins-amdgcn-mfma.cl
  N,M,K = wmma.arg[0]
  if cdna:
    if K == 32: dt_map.update({dtypes.half: ".f16", dtypes.bfloat16: ".bf16"})
    scaled = K == 128
    args = [f"{ldt(w.dtype, w.max_numel())} {ctx[w]}" for w in wmma.src]
    # scaled mfma call require E8M0 scale args, byte = 0x7F = 127, scale = 2^(127 - 127) = 1.0
    if scaled:
      _fmt = { dtypes.fp8e5m2:1, dtypes.fp8e4m3:0 }
      # (a_fp8_fmt, b_fp8_fmt, opsel, scale_a, opsel, scale_b)
      args.extend([f"i32 {_fmt[wmma.arg[1]]}", f"i32 {_fmt[wmma.arg[1]]}", "i32 0", "i32 127", "i32 0", "i32 127"])
    else: args.extend(["i32 0", "i32 0", "i32 0"]) # (cbsz, blgp, ?)

    scale = "scale." if scaled else ""
    dt_in = dt_map[wmma.arg[1]] if not scaled else ".f8f6f4"
    return f"{ctx[wmma]} = call {ldt(wmma.dtype, wmma.max_numel())} @llvm.amdgcn.mfma.{scale}{dt_map[wmma.src[-1].dtype]}" + \
           f".{N}x{M}x{K}{dt_in}(" + ", ".join(args) + ")"
  # https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/AMDGPU/GlobalISel/llvm.amdgcn.wmma_32.ll
  # example: %wmma0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16(<16 x half> %v99,<16 x half> %v100,<8 x float> %v101)
  args = [f"{ldt(w.dtype, w.max_numel())} {ctx[w]}" for w in wmma.src]
  if wmma.arg[1] == dtypes.int8: args = ["i1 true", args[0], "i1 true", args[1], args[2]]  # iu8 flags A/B signed
  if wmma.dtype != dtypes.float: args.append("i1 false") # opsel
  def _bf16(dt:DType): return dtypes.ushort if dt is dtypes.bfloat16 else dt
  suffix = f".v{wmma.max_numel()}{dt_map[_bf16(wmma.dtype)]}.v{wmma.src[0].max_numel()}{dt_map[_bf16(wmma.arg[1])]}" if rdna4 else ""
  # bfloat treated as i16 in LLVM call
  return f"{ctx[wmma]} = call {ldt(_bf16(wmma.dtype), wmma.max_numel())} @llvm.amdgcn.wmma.{dt_map[wmma.src[-1].dtype]}.16x16x16." + \
    f"{dt_map[wmma.arg[1]]}{suffix}(" + ", ".join(args) + ")"

# llvm ops, lop[<dtype>][<op>]
unsigned_lop = { Ops.ADD: "add", Ops.MUL: "mul", Ops.CDIV: "udiv", Ops.CMOD: "urem",
                 Ops.CMPLT: "icmp ult", Ops.CMPNE: "icmp ne", Ops.CMPEQ: "icmp eq", Ops.OR: "or", Ops.AND: "and", Ops.XOR: "xor",
                 Ops.SHL: "shl", Ops.SHR: "lshr",}
signed_lop = {**unsigned_lop, Ops.ADD: "add nsw", Ops.CMPLT: "icmp slt", Ops.CDIV: "sdiv", Ops.CMOD: "srem", Ops.SHR: "ashr"}
flags = " nsz arcp contract afn"
float_lop = {Ops.ADD: "fadd"+flags, Ops.MUL: "fmul"+flags, Ops.CMPLT: f"fcmp{flags} olt",
    Ops.CMPNE: f"fcmp{flags} une", Ops.CMPEQ: f"fcmp{flags} oeq", Ops.FDIV: "fdiv"+flags}
lop = {**{x:unsigned_lop for x in (dtypes.bool,)+dtypes.uints}, **{x:signed_lop for x in dtypes.sints}, **{x:float_lop for x in dtypes.floats}}

base_rewrite = PatternMatcher([
  # register index
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.cvar("idx")), name="x"), lambda ctx,buf,idx,x:
   f"  {ctx[x]} = extractelement {ldt(buf.dtype, buf.max_numel())} {ctx[buf]}, i32 {idx.val}" if buf.addrspace == AddrSpace.ALU else None),

  # load/store
  (UPat.var('idx').load(name="x"), lambda ctx,x,idx:
   f"  {ctx[x]} = load {'volatile ' if is_volatile(idx) else ''}{ldt(idx.dtype, idx.max_numel())}, "
   f"{ldt(idx.dtype, idx.max_numel(), True)} {ctx[idx]}"),
  (UPat.var('idx').store(UPat.var("var")), lambda ctx,idx,var:
   f"  store {'volatile ' if is_volatile(idx) else ''}{ldt(var.dtype, idx.max_numel())} {ctx[var]}, "
   f"{ldt(idx.dtype, idx.max_numel(), True)} {ctx[idx]}"),

  # GEP/VECTORIZE/CAST for float4 support
  (UPat(Ops.STACK, name="x"), lambda ctx,x:
   [(f"{ctx[x]}.{i}" if i+1 != len(x.src) else f"{ctx[x]}")+
     f" = insertelement {ldt(x.dtype, x.max_numel())} "+(f"{ctx[x]}.{i-1}" if i != 0 else "poison")+
     f", {ldt(u.dtype)} {ctx[u]}, i32 {i}" for i,u in enumerate(x.src)]),
  # unary/binary/ternary ops
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x:
   f"{ctx[x]} = bitcast {ldt(x.src[0].dtype, x.src[0].max_numel())} {ctx[x.src[0]]} to {ldt(x.dtype, x.max_numel())}"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"{ctx[x]} = {lcast(x.src[0].dtype, x.dtype)} {ldt(x.src[0].dtype)} {ctx[x.src[0]]} to {ldt(x.dtype)}"),
  (UPat(Ops.TRUNC, name="x"),
   lambda ctx,x: f"{ctx[x]} = call {ldt(x.dtype)} @llvm.trunc.{ldt(x.dtype)}({ldt(x.src[0].dtype)} {ctx[x.src[0]]})"),
  (UPat(GroupOp.Binary, name="x"), lambda ctx,x:
   f"{ctx[x]} = {lop[x.src[0].dtype][x.op]} {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, {ctx[x.src[1]]}"),
  (UPat(Ops.WHERE, name="x"), lambda ctx,x:
   f"{ctx[x]} = select {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, {ldt(x.src[1].dtype)} {ctx[x.src[1]]}, {ldt(x.src[2].dtype)} {ctx[x.src[2]]}"),

  # control flow
  (UPat(Ops.START, name="x"), lambda ctx,x: ctx._render_fn(x)),
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: f"br label %{ctx[x]}\n\n{ctx[x]}:"),
  (UPat(Ops.END, name="x"), lambda ctx,x: f"br i1 {ctx[x.src[2]]}, label %{ctx[x.src[1]]}, label %{ctx[x]}\n\n{ctx[x]}:"),
  (UPat(Ops.IF, name="x"), lambda ctx,x: f"br i1 {ctx[x.src[1]]}, label %if_then_{ctx[x][1:]}, label %if_else_{ctx[x][1:]}"),
  (UPat((Ops.THEN, Ops.ELSE), name="x"), lambda ctx,x: f"br label %{ctx[x]}\n\n{ctx[x]}:"),
  (UPat(Ops.ENDIF, name="x"), lambda ctx,x: f"br label %{ctx[x]}\n\n{ctx[x]}:"),
  (UPat(Ops.SINK), lambda: "ret void\n}"),
  (UPat(GroupOp.Block, name="x"), lambda ctx,x: f"{ctx[x]}:"),
  # phi
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.ENDIF, src=(UPat.var("a"), UPat.var("b"))),), name="x"), lambda ctx,a,b,x:
   f"{ctx[x]} = phi {ldt(x.dtype, x.max_numel())} [ {ctx[a.get_arg(x.arg)]}, %{ctx[a]} ], [ {ctx[b.get_arg(x.arg)]}, %{ctx[b]} ]"),
  # loop phi, here the second input comes from the loop backedge
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.RANGE, name="a"),), name="x"), lambda ctx,a,x:
   f"{ctx[x]} = phi {ldt(x.dtype, x.max_numel())} [ {ctx[a.get_arg(x.arg)]}, %{ctx[a.src[0]]} ], [ {ctx.phi_backedge(ctx.backedge[a], x)}, %{ctx[ctx.backedge[a].src[0]]} ]"),

  (UPat(Ops.BARRIER), lambda ctx: "")
])

class LLVMRenderer(Renderer):
  abi: str | None
  string_rewrite: PatternMatcher
  code_for_op = {k:lambda:None for v in lop.values() for k in v.keys()}

  extra_matcher = create_non_native_float_pats((dtypes.bfloat16,)) + pm_manual_bf16_cast
  def __getitem__(self, key): return self.r[key]  # hacky helper
  def phi_backedge(self, end:UOp, phi:UOp):
    backedge = end.get_arg(phi.arg)
    if backedge not in self.r: self.r[backedge] = f"{self.r[phi]}.next"
    return self.r[backedge]
  def _render_fn(self, x:UOp):
    # NOTE: CPUAllocator promises 0x20 alignment
    args = ", ".join([f"{ldt(s.dtype, ptr=s.addrspace == AddrSpace.GLOBAL)}{' noalias align 32' if s.addrspace == AddrSpace.GLOBAL else ''} {self.r[s]}" for s in x.src])
    return "\n".join((self.prefix or []) + [f"define{' ' + self.abi if self.abi else ''} void @{x.arg}({args}) #0", "{"])

  # here we attach the label and tail control to each block
  def _render_block(self, block:UOp, sched:list[str]) -> str:
    tail = self.tail_control[block]
    if tail.op is Ops.IF: sched.append(f"br i1 {self.r[tail.src[1]]}, label %{self.r[self.if_targets[tail][Ops.THEN]]}, label %{self.r[self.if_targets[tail][Ops.ELSE]]}")
    elif tail.op is Ops.END: sched.append(f"br i1 {self.r[tail.src[2]]}, label %{self.r[tail.src[1]]}, label %{self.r[tail]}")
    elif tail.op is Ops.SINK: sched.append("ret void\n}")
    else: sched.append(f"br label %{self.r[tail]}")
    header = (self._render_fn(block) if block.op is Ops.START else "") + f"\n{self.r[block]}:"
    return "\n".join([header] + [f"  {s}" for s in sched])

  def _render_kernel(self, uops:list[UOp], prefix:list[str]|None=None) -> tuple[tuple[str, ...], str]:
    self.prefix = prefix
    self.r: dict[UOp, str] = {}
    r = self.r
    vc = 0
    b_id = 0

    # get the loop backedges and labels
    self.backedge: dict[UOp, UOp] = {}
    self.tail_control: dict[UOp, UOp] = {}
    self.if_targets: dict[UOp, dict[Ops, UOp]] = {}
    for u in uops:
      if u.op not in GroupOp.Control: continue
      # record the tail control (jump at end of block)
      if u.op is Ops.ENDIF:
        for s in u.src: self.tail_control[s] = u
      else:
        if u.op in (Ops.THEN, Ops.ELSE): self.if_targets.setdefault(u.src[0], {})[u.op] = u
        if u.src[0].op in GroupOp.Block: self.tail_control[u.src[0]] = u
      if u.op not in GroupOp.Block: continue
      if u.op is Ops.END:
        self.backedge[u.src[1]] = u
        r[u] = f"loop.{range_str(u.src[1])}.exit"
      elif u.op is Ops.RANGE: r[u] = f"loop.{range_str(u)}.body"
      else:
        if u.op is Ops.START: r[u] = f"bb{b_id}.entry"
        elif u.op is Ops.THEN: r[u] = f"bb{b_id}.if.then"
        elif u.op is Ops.ELSE: r[u] = f"bb{b_id}.if.else"
        elif u.op is Ops.ENDIF: r[u] = f"bb{b_id}.if.end"
        b_id += 1

    local_args: list[str] = []
    blocks: dict[UOp, list[str]] = {}
    block: list[str] = []
    for u in uops:
      if u.op in GroupOp.Block: blocks[u] = block = []
      if u.op in GroupOp.Control: continue
      if u.op in {Ops.NOOP, Ops.GROUP}: continue
      if u.op is Ops.AFTER: r[u] = r[u.src[0]]
      # this a block arg access, no phi
      elif u.op is Ops.GETTUPLE and u.src[0].op not in (Ops.RANGE, Ops.ENDIF): r[u] = r[u.src[0].get_arg(u.arg)]
      elif u.op is Ops.PARAM: r[u] = f"%data{u.arg.slot}"
      elif u.op is Ops.BUFFER:
        r[u] = f"%local_{str(u.arg.slot)}"
        size = u.max_numel()
        if self.has_local:
          local_args.append(f"@{r[u][1:]} = internal unnamed_addr addrspace(3) global [{size} x {ldt(u.dtype)}] undef, align 16")
          block.append(f"{r[u]} = addrspacecast [{size} x {ldt(u.dtype)}] addrspace(3)* @{r[u][1:]} to [{size} x {ldt(u.dtype)}]*")
        else:
          block.append(f"{r[u]} = alloca [{size} x {ldt(u.dtype)}], align 16")
      elif u.op is Ops.CONST: r[u] = lconst(u.val, u.dtype)
      elif u.op is Ops.CAST and ldt(u.dtype) == ldt(u.src[0].dtype):
        r[u] = r[u.src[0]] # cast from signed to unsigned of the same size is a noop, or pointer cast
      else:
        # if it's an assign target, it's already preallocated
        if u not in r:
          r[u] = f"%v{vc}"
          vc += 1

        # do the rendering of the llvm ir code
        l: list[str]|str|None = self.string_rewrite.rewrite(u, ctx=self)
        if l is None: raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        else: block.extend([l] if isinstance(l, str) else l)
    
    return tuple(local_args), "\n".join(self._render_block(b, sched) for b,sched in blocks.items())

class CPULLVMRenderer(LLVMRenderer):
  has_local = False
  has_threads = bool(getenv("THREADS", 1))
  global_max = (NUM_CPU_THREADS.value, 0, 0)
  abi = 'win64cc' if sys.platform == 'win32' else None
  string_rewrite = base_rewrite
  def render(self, uops: list[UOp]) -> str: return "\n".join((k:=self._render_kernel(uops))[0] + (k[1], self._render_footer(uops)))
  def _render_footer(self, uops: list[UOp]) -> str: return 'attributes #0 = { alwaysinline nounwind "no-builtins" "no-trapping-math"="true" }'
  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_llvm import CPULLVMCompiler
    self.compiler = CPULLVMCompiler(target.arch.split(","))

  # FIXME: fp16 works on non-osx, but only if the cpu supports it
  def supported_dtypes(self):
    return {d for d in super().supported_dtypes() if
            (d != dtypes.bfloat16 or self.target.arch.startswith(("x86", "arm"))) and (d != dtypes.half or OSX) and d not in dtypes.fp8s}

barrier = 'fence syncscope("workgroup") release\ntail call void @llvm.amdgcn.s.barrier()\nfence syncscope("workgroup") acquire\n'
code_for_workitem = {"g": lambda x: f"tail call i32 @llvm.amdgcn.workgroup.id.{chr(120+int(x))}()",
                     "l": lambda x: f"tail call i32 @llvm.amdgcn.workitem.id.{chr(120+int(x))}()"}
# https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html#llvm-ir-intrinsics
llvm_intrinsics = {Ops.SQRT: "sqrt", Ops.LOG2: "log2", Ops.EXP2: "exp2"}
class AMDLLVMRenderer(LLVMRenderer):
  shared_max = HIPRenderer.shared_max
  global_max = HIPRenderer.global_max
  global_prod_max = HIPRenderer.global_prod_max
  abi = "amdgpu_kernel"
  code_for_op = {**LLVMRenderer.code_for_op, **{op: lambda: None for op in llvm_intrinsics}}
  string_rewrite = PatternMatcher([
    (UPat(Ops.SPECIAL, name="x"), lambda ctx, x: f"{ctx[x]} = " + f"{ code_for_workitem[x.arg[0]](x.arg[-1])}; "),
    (UPat(tuple(llvm_intrinsics), name="x"),
    lambda ctx, x: f"{ctx[x]} = call {ldt(x.dtype)} @llvm.{llvm_intrinsics[x.op]}.{ldt(x.dtype)}({ldt(x.src[0].dtype)} {ctx[x.src[0]]})"),
    (UPat(Ops.BARRIER), lambda ctx: barrier),
    (UPat(Ops.CAST, dtypes.fp8s, (UPat(dtype=dtypes.float),), name="x",), lambda ctx,x:
      f"{ctx[x]} = call i8 @f32_to_fp8({ldt(x.src[0].dtype)}  {ctx[x.src[0]]}, i1 {'1' if x.dtype == dtypes.fp8e5m2 else '0'})"),
    (UPat(Ops.CAST, dtypes.float, (UPat.var("y", dtypes.fp8s),), name="x",), lambda ctx,x,y:
      f"{ctx[x.src[0]]}_i32 = zext i8 {ctx[x.src[0]]} to i32\n"
      f"{ctx[x]} = call float @llvm.amdgcn.cvt.f32.{'bf8' if y.dtype == dtypes.fp8e5m2 else 'fp8'}(i32 {ctx[x.src[0]]}_i32, i32 0)"),
  ]) + base_rewrite
  extra_matcher = LLVMRenderer.extra_matcher + create_non_native_float_pats(dtypes.fp8s) + PatternMatcher([
    # amd llvm intrinsics llvm.log2/llvm.exp2 don't support double
    (UPat(Ops.LOG2, dtype=dtypes.double, src=(UPat.var("d"),)), xlog2),
    (UPat(Ops.EXP2, dtype=dtypes.double, src=(UPat.var("d"),)), xexp2),
  ])
  def asm(self, prg: UOp, lin: UOp) -> bytes:
    from tinygrad.renderer.amd.elf import assemble_linear
    return assemble_linear(prg, lin, self.target.arch)
  def render(self, uops: list[UOp]) -> str:
    prefix = ["""define i8 @f32_to_fp8(float %val, i1 %is_bf8) {
entry: %ival = bitcast float %val to i32\n  %exp = and i32 %ival, 2139095040\n  %is_special = icmp eq i32 %exp, 2139095040
br i1 %is_special, label %select_clip, label %clip
clip: br i1 %is_bf8, label %bf8_clip, label %fp8_clip
bf8_clip: %clamped_bf8 = call float @llvm.amdgcn.fmed3.f32(float %val, float 57344.0, float -57344.0)\n  br label %select_clip
fp8_clip: %clamped_fp8 = call float @llvm.amdgcn.fmed3.f32(float %val, float 448.0, float -448.0)    \n  br label %select_clip
select_clip: %phi_val = phi float [%val, %entry], [%clamped_bf8, %bf8_clip], [%clamped_fp8, %fp8_clip]\n  br i1 %is_bf8, label %do_bf8, label %do_fp8
do_bf8: %packed_bf8 = call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %phi_val, float %phi_val, i32 0, i1 false)\n  br label %exit
do_fp8: %packed_fp8 = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %phi_val, float %phi_val, i32 0, i1 false)\n  br label %exit
exit: %packed = phi i32 [%packed_bf8, %do_bf8], [%packed_fp8, %do_fp8]\n  %trunc = trunc i32 %packed to i8\n  ret i8 %trunc
}""".replace(": ", ":\n  ")] if any(u.dtype in dtypes.fp8s for u in uops) else []
    return "\n".join((k:=self._render_kernel(uops, prefix))[0] + (k[1], self._render_footer(uops)))
  def _render_footer(self, uops: list[UOp]) -> str:
    # TODO: this is copied from cstyle
    local_dims = [u.src[0] for u in uops if u.op is Ops.SPECIAL and u.arg[0] == "l"]
    requiredMaxThreadsPerBlock = prod([d.vmax for d in local_dims])
    attributes = ["alwaysinline", "nounwind", '"no-builtins"',
                  f'"amdgpu-flat-work-group-size"="1,{requiredMaxThreadsPerBlock}"', '"no-trapping-math"="true"']
    return 'attributes #0 = { ' + ' '.join(attributes) + ' }'
  @staticmethod
  def is_rdna4(arch): return arch.split(':')[0] in {'gfx1200', 'gfx1201'}
  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_llvm import AMDLLVMCompiler
    self.compiler, self.tensor_cores, self.is_cdna = AMDLLVMCompiler(target.arch), tc.get_amd(target.arch), HIPRenderer.is_cdna(target.arch)
    self.string_rewrite += PatternMatcher([
      (UPat(Ops.WMMA, name="wmma"), lambda ctx, wmma, rdna4=AMDLLVMRenderer.is_rdna4(target.arch), cdna=self.is_cdna:
        render_wmma_amd(ctx, wmma, cdna, rdna4))
    ])
    if self.is_cdna:
      self.extra_matcher += PatternMatcher([
        (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
          lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint32), x.src[1].bitcast(dtypes.uint32), x.src[2]))
          if x.arg[0][2] == 128 and x.src[0].dtype.itemsize <= 8 else None),
        (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
          lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint16), x.src[1].bitcast(dtypes.uint16), x.src[2]))
          if x.max_numel() == 4 and x.src[0].dtype == dtypes.bfloat16 and x.src[0].max_numel() == 4 else None),
        (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
          lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint64), x.src[1].bitcast(dtypes.uint64), x.src[2]))
          if x.max_numel() == 4 and x.src[0].dtype in dtypes.fp8_ocp and x.src[0].max_numel() == 8 else None),
      ])
    if target.arch in {"gfx1100", "gfx1151"}:
      self.extra_matcher += PatternMatcher([
        (UPat(Ops.WMMA, name="x", dtype=dtypes.int32), lambda x: x.replace(
          src=(x.src[0].bitcast(dtypes.uint32), x.src[1].bitcast(dtypes.uint32), x.src[2]))
          if x.src[0].dtype == dtypes.int8 and x.src[0].max_numel() == 16 else None),
        (UPat(Ops.WMMA, name="x", dtype=dtypes.half), lambda x: UOp(Ops.STACK, src=tuple(x.replace(
          src=(x.src[0], x.src[1], UOp(Ops.STACK, src=tuple(x.src[2].index(UOp.const(j//2, dtypes.int16))
            if j%2 == 0 else UOp.const(0.0, x.src[2].dtype)
            for j in range(x.max_numel()*2)))),
          arg=(*x.arg[:4], None)).index(UOp.const(i*2, dtypes.int16))
          for i in range(x.max_numel()))) if x.max_numel() == 8 else None),
        (UPat(Ops.WMMA, name="x"), lambda x: x.replace(
          src=(x.src[0].bitcast(dtypes.uint16), x.src[1].bitcast(dtypes.uint16), x.src[2]))
          if x.src[0].dtype == dtypes.bfloat16 and x.src[0].max_numel() == 16 else None),
      ])
    if target.arch in {"gfx1200", "gfx1201"}:
      self.extra_matcher += PatternMatcher([
        (UPat(Ops.WMMA, name="x", dtype=dtypes.bfloat16), lambda x: x.replace(
          dtype=dtypes.uint16,
          src=(x.src[0].bitcast(dtypes.uint16), x.src[1].bitcast(dtypes.uint16), x.src[2].bitcast(dtypes.uint16)))
            .bitcast(dtypes.bfloat16) if x.max_numel() == 8 and x.src[0].dtype == dtypes.bfloat16 and x.src[0].max_numel() == 8 else None),
        (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
          lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint16), x.src[1].bitcast(dtypes.uint16), x.src[2]))
          if x.max_numel() == 8 and x.src[0].dtype == dtypes.bfloat16 and x.src[0].max_numel() == 8 else None)
      ])

  def supported_dtypes(self): return {d for d in super().supported_dtypes()
                                      if (d not in dtypes.fp8_ocp or self.target.arch == "gfx950") and d not in dtypes.fp8_fnuz}
