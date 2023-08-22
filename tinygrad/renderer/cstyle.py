from typing import Dict, List, Optional, NamedTuple, Tuple, Union
import math
from tinygrad.codegen.linearizer import UOps, UOp, MemOp, ConstOp
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.helpers import ImageDType, dtypes, getenv, prod, DType
from tinygrad.shape.symbolic import DivNode, AndNode, render_python, NumNode, Variable, sym_render

# div is different in cl than python
render_cl = render_python.copy()
render_cl[DivNode] = lambda self,ops,ctx: f"({self.a.render(ops, ctx)}/{self.b})"
render_cl[AndNode] = lambda self,ops,ctx: f"({'&&'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})"

class CStyleLanguage(NamedTuple):
  size_prefix: str = "int"
  generic_var_prefix: str = ""
  kernel_prefix: str = ""
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_prefix: str = ""
  arg_int_prefix: str = ""
  barrier: str = ""
  gid: List[str] = []
  lid: List[str] = []
  global_max: List[int] = []
  local_max: List[int] = []
  extra_args: List[str] = []
  float4: Optional[str] = None
  half_prekernel: Optional[str] = None
  uses_vload: bool = False
  external_local_bufs: bool = False
  uses_ptr_arithmetic: bool = False
  launch_bounds: bool = False
  code_for_op: Dict = {
    UnaryOps.EXP2: lambda x: f"exp2({x})",
    UnaryOps.LOG2: lambda x: f"log2({x})",
    UnaryOps.SIN: lambda x: f"sin({x})",
    UnaryOps.SQRT: lambda x: f"sqrt({x})",
    BinaryOps.ADD: lambda a,b: f"({a}+{b})", BinaryOps.SUB: lambda a,b: f"({a}-{b})",
    BinaryOps.MUL: lambda a,b: f"({a}*{b})", BinaryOps.DIV: lambda a,b: f"({a}/{b})",
    BinaryOps.MAX: lambda a,b: f"max({a},{b})", BinaryOps.MOD: lambda a,b: f"({a}%{b})",
    BinaryOps.CMPLT: lambda a,b: f"({a}<{b})", TernaryOps.MULACC: lambda a,b,c: f"(({a}*{b})+{c})",
    TernaryOps.WHERE: lambda a,b,c: f"({a}!=0?{b}:{c})"
  }

  # returns a str expression of the casted xs with the given type
  def render_cast(self, x:List[str], var_dtype:DType) -> str:
    assert len(x) == var_dtype.sz, f"cast is wrong size {len(x)} != {var_dtype.sz}"
    assert self.float4 is not None, "cast is not supported on this platform"
    if var_dtype == dtypes._float4: return f"{self.float4}({','.join(x)})"
    if var_dtype == dtypes._float2: return f"{self.float4.replace('float4', 'float2')}({','.join(x)})"
    raise NotImplementedError(f"no cast for {var_dtype}")

  # returns a str expression of the const with the given type
  def render_const(self, x:Union[float,int], var_dtype) -> str:
    if math.isnan(x): val = "NAN"
    elif math.isinf(x): val = ("-" if x < 0 else "") + "INFINITY"
    else: val = f"{x}f" if dtypes.is_float(var_dtype) and isinstance(x, float) else f"{int(x)}"
    return self.render_cast([val]*var_dtype.sz, var_dtype) if var_dtype.sz > 1 else val

  # returns a str expression of the loaded value with the output type
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert output_dtype == dtypes._float4, "images must be float4"
      return f"read_imagef({buf_name}, smp, (int2)({idx[0].render(render_cl)}, {idx[1].render(render_cl)}))"
    if self.uses_vload and buf_dtype == dtypes.float16:
      return f"vload_half{'' if output_dtype.sz == 1 else str(output_dtype.sz)}(0, {buf_name}+{idx.render(render_cl, strip_parens=True)})"
    if output_dtype.sz > 1:
      return f"({output_dtype.name})(*(({self.smem_prefix if local else self.buffer_prefix}{buf_dtype.name}{output_dtype.sz}*)({buf_name}+{idx.render(render_cl, strip_parens=True)})))"
    return f"*({buf_name}+{idx.render(render_cl, strip_parens=True)})" if self.uses_ptr_arithmetic else f"{buf_name}[{idx.render(render_cl)}]"

  def render_local(self, name:str, size:int):
    return self.smem_prefix + f"float {name}[{size}];"

  def render_for(self, expr: str, _min:int, _max:Union[int,str]) -> str:
    return f"for (int {expr} = {_min}; {expr} <= {_max}; ++{expr}) {{"

  def render_conditional(self, cond: str, x:str, y:str) -> str:
    return f"({cond})?({x}):{y}"

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,DType]], global_size:List[int], local_size:List[int], prekernel:List[str]) -> Tuple[str,List[int],List[int]]:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,dtype in bufs) else ""
    buftypes = [(name,f"{'read_only' if i > 0 else 'write_only'} image2d_t" if dtype.name.startswith('image') else
                self.arg_int_prefix if dtype == dtypes._arg_int32 else
                ("const " if i > 0 else "")+self.buffer_prefix+dtype.name+"*"+self.buffer_suffix) for i,(name,dtype) in enumerate(bufs)]
    prg = ''.join([f"{self.kernel_prefix} void {f'__launch_bounds__ ({prod(local_size)}, 1) ' if self.launch_bounds else ''}{function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    if self.half_prekernel and any(dtype == dtypes.float16 for _,dtype in bufs): prg = ''.join([f"{self.half_prekernel}", "\n", prg])

    return prg, global_size[::-1], local_size[::-1]

  # returns a str statement that does the store
  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert var_dtype == dtypes._float4, "images must be float4"
      return f"write_imagef({buf_name}, (int2)({idx[0].render(render_cl)}, {idx[1].render(render_cl)}), {var_name});"
    if self.uses_vload and buf_dtype == dtypes.float16:
      return f"vstore_half{'' if var_dtype.sz == 1 else str(var_dtype.sz)}({var_name}, 0, {buf_name}+{idx.render(render_cl, strip_parens=True)});"
    if var_dtype.sz > 1:
      return f"*(({self.smem_prefix if local else self.buffer_prefix}{buf_dtype.name}{var_dtype.sz}*)({buf_name}+{idx.render(render_cl, strip_parens=True)})) = ({buf_dtype.name}{var_dtype.sz}){var_name};"
    return f"*({buf_name}+{idx.render(render_cl, strip_parens=True)}) = {var_name};" if self.uses_ptr_arithmetic else f"{buf_name}[{idx.render(render_cl)}] = {var_name};"

def add_gl_dimension(prefix: str, args, i:int, var, local_size:List[int], xid:List[str]):
  # for M1 tensor core stuff, support > 3 dims
  if i >= 2 and len(args[0]) > len(xid):
    # do this on the x dim for warps
    if len(local_size) == 2: local_size.append(1)
    local_size[-1] *= var.max+1
    lidx = Variable(xid[0], 0, prod(x.max+1 for x in args[0][2:])-1)
    lidx = (lidx//((lidx.max+1)//local_size[-1]))%(var.max+1)
    assert lidx.max == var.max and lidx.min == var.min
    return "{" if isinstance(var, NumNode) else f"{{ {prefix} {var.expr} = {lidx.render(render_cl)};  /* {var.max+1} */"
  local_size.append(var.max+1)
  return "{" if isinstance(var, NumNode) else f"{{ {prefix} {var.expr} = {xid[min(len(xid), len(args[0]))-1-i]};  /* {var.max+1} */"

def uops_to_cstyle(lang:CStyleLanguage, function_name:str, uops:List[UOp])  -> Tuple[str, List[int], List[int]]:
  global_size: List[int] = []
  local_size: List[int] = []
  kernel,prekernel = [],[]
  pend_close = None
  bufs = []
  depth = 0
  def kk(s): kernel.append("  "*depth+s)

  for uop,newvar,vin,args in uops:
    if uop == UOps.LOOP:
      for i,var in enumerate(args[0]):
        if args[1] == "global" and lang.gid:
          kk(add_gl_dimension(lang.size_prefix, args, i, var, global_size, lang.gid))
        elif args[1] == "local" and lang.lid:
          kk(add_gl_dimension(lang.size_prefix, args, i, var, local_size, lang.lid))
        else:
          if getenv("NOUNROLL") and not isinstance(var, NumNode): kk("#pragma unroll(1)")   # prevent loop unrolling
          kk("{" if isinstance(var, NumNode) else lang.render_for(var.expr, var.min, sym_render(var.max)))
      depth += 1
    elif uop == UOps.BARRIER:
      kk(lang.barrier)
    elif uop == UOps.ENDLOOP:
      if args[1] == "local" and lang.lid:
        # TODO: this is a bit of a hack. the local loop isn't real on the GPU
        kk(f"if ({Variable.sum(args[0]).render(render_cl)} == 0) {{")
        pend_close = "}"*(len(args[0])+1) + f" /* {args[1]} */"
      else:
        if args[1] == "global" and pend_close:
          depth -= 1
          kk(pend_close)
          pend_close = None
        depth -= 1
        kk("}"*len(args[0]) + f" /* {args[1]} */")
    elif uop == UOps.WMMA:
      if args == "METAL":
        # ((lidx2*32)+(lidx3*4)+(lidx4*16)+(lidx5*8)+(lidx6*2))
        kk("{ simdgroup_float8x8 a,b,c;")
        kk(f"a.thread_elements()[0] = {vin[0].render()}; a.thread_elements()[1] = {vin[1].render()};")
        kk(f"b.thread_elements()[0] = {vin[2].render()}; b.thread_elements()[1] = {vin[3].render()};")
        kk(f"c.thread_elements()[0] = {vin[4].render()}; c.thread_elements()[1] = {vin[5].render()};")
        kk("simdgroup_multiply_accumulate(c, a, b, c);")
        kk(f"{vin[4].render()} = c.thread_elements()[0]; {vin[5].render()} = c.thread_elements()[1]; }}")
      elif args == "HIP":
        kk("{")
        kk(f"half16 a_frag = {{ {','.join(['(half)'+x.render() for x in vin[8:8+16]])} }};")
        kk(f"half16 b_frag = {{ {','.join(['(half)'+x.render() for x in vin[8+16:8+32]])} }};")
        kk(f"float8 c_frag = {{ {','.join([x.render() for x in vin[:8]])} }};")
        kk("c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);")
        for i in range(8): kk(f"{vin[i].render()} = c_frag[{i}];")
        kk("}")
      else:
        raise NotImplementedError(f"WMMA not implemented for {args}")
    elif uop == UOps.ALU:
      assert newvar is not None
      kk(f"{lang.generic_var_prefix if newvar not in vin else ''}{newvar.render(newvar not in vin and lang.generic_var_prefix == '')} = {lang.code_for_op[args](*[x.render() for x in vin])};")
    elif uop == UOps.LOAD:
      assert newvar is not None and isinstance(args, (MemOp, ConstOp))
      # valids are handled here
      if isinstance(args, ConstOp):
        val = lang.render_const(args.value, newvar.dtype)
      else:
        val = lang.render_load(newvar.dtype, args.name, args.memory_dtype, args.idx, args.local)
      if args.valid.min == 0 and args.valid.max == 1: val = lang.render_conditional(args.valid.render(render_cl), val, lang.render_const(args.invalid_value, newvar.dtype))
      kk(f"{lang.generic_var_prefix}{newvar.render(lang.generic_var_prefix == '')} = {val};")
    elif uop == UOps.STORE:
      assert args.valid.min == 1 and isinstance(args, MemOp), "store must be valid and to memory"
      # TODO: instead of dtypes.float, a base type
      kk(lang.render_store(args.name, args.memory_dtype, vin[0].render(), vin[0].dtype if vin[0].offset is None else dtypes.float, args.idx, args.local))
    elif uop == UOps.CAST and newvar is not None and newvar.dtype.sz > 1:
      kk(f"{newvar.render(True)} = {lang.render_cast([x.render() for x in vin], newvar.dtype)};")
    elif uop == UOps.DEFINE_LOCAL:
      if lang.external_local_bufs:
        prekernel.append(lang.render_local(args[0], args[1]))
      else:
        kk(lang.render_local(args[0], args[1]))
    elif uop == UOps.DEFINE_GLOBAL:
      bufs.append(args)
    else:
      raise RuntimeError(f"failed to render {uop}")

  return lang.render_kernel(function_name, kernel, bufs, global_size, local_size, prekernel)
