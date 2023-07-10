from tinygrad.lazy import LazyBuffer
from tinygrad.shape.symbolic import NumNode, Variable
from tinygrad.codegen.cstyle import render_cl
from tinygrad.helpers import dtypes
from tinygrad.codegen.linearizer import Linearizer, LocalBuffer, UOps
from tinygrad.codegen.cstyle import CStyleLanguage
from typing import Dict, Callable, List, Union
from tinygrad.runtime.lib import RawConst
from tinygrad.ops import UnaryOps, Op, BinaryOps, ASTRunner, FusedOps
import math

type_map = {dtypes.float: "f32", dtypes.half: "f16", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool"}
code_for_op: Dict[Op, Callable] = {
    UnaryOps.EXP2: lambda x: f"exp2({x})", UnaryOps.LOG2: lambda x: f"log2({x})", UnaryOps.SIN: lambda x: f"sin({x})", UnaryOps.SQRT: lambda x: f"sqrt({x})",
    BinaryOps.ADD: lambda x,y: f"({x}+{y})", BinaryOps.SUB: lambda x,y: f"({x}-{y})", BinaryOps.MUL: lambda x,y: f"({x}*{y})", BinaryOps.DIV: lambda x,y: f"({x}/{y})",
    BinaryOps.MAX: lambda x,y: f"max({x},{y})", BinaryOps.CMPEQ: lambda x,y: f"f32({x}=={y})",
    FusedOps.MULACC: lambda x,y,z: f"fma({x},{y},{z})",
  }
class WGSLLanguage(CStyleLanguage):
  gid = [f"i32(gindex.{'xyz'[x]})" for x in range(3)]
  lid = [f"i32(lindex.{'xyz'[x]})" for x in range(3)]
  size_prefix = "let"
  code_for_op = code_for_op
  barrier="workgroupBarrier();"
  generic_var_prefix = "var "
  external_local_bufs = True

  def render_local(self, name: str, size: int):
    return f"var<workgroup> {name}: array<f32,{size}>;"
  
  def render_kernel(self, kernel: List[str], bufs: List[LocalBuffer | LazyBuffer], bufnames: List[str], local_size: List[int], prekernel: List[str]) -> str:
    local_size = local_size if len(local_size) else [1]
    bind_it = iter(range(len(bufs)))
    prg = "\n".join(prekernel+[f"@group(0) @binding({next(bind_it)}) var<storage,read_write> data{i}: array<{type_map[x.dtype]}>;" for i,x in enumerate(bufs) if not isinstance(x, LocalBuffer) and not isinstance(x.realized, RawConst)])
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn KERNEL_NAME_PLACEHOLDER(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{\n" + "\n".join(kernel) + "\n}"
    return prg
  def render_for(self, expr: str, min: int, max: int) -> str:
    return f"for(var {expr} = {min}; {expr} <= {max}; {expr}++) {{"
  def render_conditional(self, cond: str, x: str, y: str) -> str:
    return f"select({x}, f32({y}), bool({cond}))"
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    return f"f32({super().render_load(output_dtype, buf_name, buf_dtype, idx, local)})"

class WGSLCodegen(Linearizer):
  supports_float4 = False
  supports_constant_folding = True
  def float_const(self, x: float) -> str: return f"{x}f" if not math.isinf(x) else ("-" if x < 0 else "") + "0x1.fffffep+127f" # TODO: Replace with inf when its added to the spec

  def codegen(self):
    self.process()
    self.hand_coded_optimizations()
    self.limit_global_dims(3)
    self.linearize()

    kernel,shared,global_size,local_size = [],[],[],[]
    depth = 0
    def kk(s): kernel.append(" "*depth+s)
    bufnames = [b.name if isinstance(b, LocalBuffer) else f"data{i}" for i,b in enumerate(self.bufs)]
    pend_close = None
    for uop,newvar,vin,args in self.uops:
      if uop == UOps.LOOP:
        for i,var in enumerate(args[0]):
          if isinstance(var, NumNode): 
            if args[1] == "global": global_size.append(1)
            if args[1] == "local": local_size.append(1)
            kk("{")
          else:
            if args[1] == "global" or args[1] == "local":
              kk(f"{{ let {var.expr} = i32({'g' if args[1] == 'global' else 'l'}index.{'xyz'[len(args[0])-1-i]}); // {var.max+1}")
              (global_size if args[1] == "global" else local_size).append(var.max+1)
            else:
              kk(f"for(var {var.expr} = 0; {var.expr} <= {var.max}; {var.expr}++) {{")
        depth += 1
      elif uop == UOps.ENDLOOP:
        if args[1] == "local":
          kk(f"if ({Variable.sum(args[0]).render(render_cl)} == 0) {{")
          pend_close = "}"*(len(args[0])+1) + f" // {args[1]}"
        else:
          if args[1] == "global" and pend_close:
            depth -= 1
            kk(pend_close) 
            pend_close = None
          depth -= 1
          kk("}"*len(args[0])  + f" /* {args[1]} */")
      elif uop == UOps.LOAD and newvar is not None:
        if self.bufs[args.i] is not None and isinstance(self.bufs[args.i].realized, RawConst):
          assert newvar.dtype == dtypes.float, "only floats"
          assert not math.isnan(self.bufs[args.i].realized._buf), "nans are not supported in webgpu"
          val = self.float_const(self.bufs[args.i].realized._buf)
        else:
          val = f"{bufnames[args.i]}[{args.idx.render(render_cl)}]"
        if args.valid.min == 1: kk(f"var {newvar.render()} = f32({val});")
        elif args.valid.render(render_cl) == "0": kk(f"var {newvar.render()} = 0.0f;")
        else: kk(f"var {newvar.render()} = select(0.0f, f32({val}), bool({args.valid.render(render_cl)}));")
      elif uop == UOps.ALU:
        assert newvar is not None
        if newvar in vin: kk(f"{newvar.render()} = {code_for_op[args](*[x.render() for x in vin])};")
        else: kk(f"var {newvar.render()} = {code_for_op[args](*[x.render() for x in vin])};")
      elif uop == UOps.STORE:
        val = vin[0].render()
        if vin[0].dtype != self.bufs[args.i].dtype:
          val = f"{type_map[self.bufs[args.i].dtype]}({val})"
        kk(f"{bufnames[args.i]}[{args.idx.render(render_cl)}] = {val};")
      elif uop == UOps.CONST:
        assert newvar is not None
        kk(f"var {newvar.render()} = {self.float_const(args)};")
      elif uop == UOps.DEFINE_LOCAL: shared.append(f"var<workgroup> {args[0]}: array<f32,{args[1]}>;")
      elif uop == UOps.BARRIER: kk("workgroupBarrier();")
      else: raise RuntimeError(f"failed to render {uop}")
    assert all(x <= 65535 for x in global_size), "WEBGPU max global size is 65535 in any dimension"
    assert len([x for x in self.bufs if not isinstance(x, LocalBuffer) and not isinstance(x.realized, RawConst)]) <= 31, "WEBGPU max number of buffers is 31" 
    function_name = f"{self.function_name}_{id(self.key)}" # Function name itself isn't unique
    bind_it = iter(range(len(self.bufs)))
    global_size, local_size = global_size[::-1] if len(global_size) else [1], local_size[::-1] if len(local_size) else [1]
    prg = "\n".join(shared+[f"@group(0) @binding({next(bind_it)}) var<storage,read_write> data{i}: array<{type_map[x.dtype]}>;" for i,x in enumerate(self.bufs) if not isinstance(x, LocalBuffer) and not isinstance(x.realized, RawConst)])
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{\n" + "\n".join(kernel) + "\n}"
    return ASTRunner(function_name, prg, global_size, local_size)
