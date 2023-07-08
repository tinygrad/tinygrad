from typing import Final, Dict, ClassVar, List, Optional, NamedTuple, DefaultDict, Tuple, Union
import math, collections
from tinygrad.codegen.linearizer import Linearizer, UOps, UOp, LocalBuffer
from tinygrad.ops import ASTRunner, UnaryOps, BinaryOps, TernaryOps
from tinygrad.helpers import ImageDType, dtypes, colored, getenv, prod, DType
from tinygrad.runtime.lib import RawConst
from tinygrad.shape.symbolic import DivNode, AndNode, render_python, NumNode, Variable
from tinygrad.lazy import LazyBuffer

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
  barrier: str = ""
  gid: List[str] = []
  lid: List[str] = []
  extra_args: List[str] = []
  float4: Optional[str] = None
  half_prekernel: Optional[str] = None
  uses_vload: bool = False
  external_local_bufs: bool = False
  code_for_op: Dict = {
    UnaryOps.EXP2: lambda x: f"exp2({x})",
    UnaryOps.LOG2: lambda x: f"log2({x})",
    UnaryOps.SIN: lambda x: f"sin({x})",
    UnaryOps.SQRT: lambda x: f"sqrt({x})",
    BinaryOps.ADD: lambda a,b: f"({a}+{b})", BinaryOps.SUB: lambda a,b: f"({a}-{b})",
    BinaryOps.MUL: lambda a,b: f"({a}*{b})", BinaryOps.DIV: lambda a,b: f"({a}/{b})",
    BinaryOps.MAX: lambda a,b: f"max({a},{b})",
    BinaryOps.CMPEQ: lambda a,b: f"({a}=={b})", TernaryOps.MULACC: lambda a,b,c: f"(({a}*{b})+{c})",
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
    else: val = f"{x}" + ("" if dtypes.is_int(var_dtype) else "f")
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
    return f"{buf_name}[{idx.render(render_cl)}]"
  
  def render_local(self, name:str, size:int): 
    return self.smem_prefix + f"float {name}[{size}];"
  
  def render_for(self, expr: str, _min:int, _max:int) -> str:
    return f"for (int {expr} = {_min}; {expr} <= {_max}; ++{expr}) {{"
  
  def render_conditional(self, cond: str, x:str, y:str) -> str:
    return f"({cond})?({x}):{y}"

  def render_kernel(self, kernel:List[str], bufs:List[Union[LocalBuffer,LazyBuffer]], bufnames:List[str], global_size:List[int], local_size:List[int], prekernel:List[str]) -> Tuple[str,List[int],List[int]]:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(x.dtype, ImageDType) for x in bufs) else ""
    buftypes = [(i,f"{'read_only' if i > 0 else 'write_only'} image2d_t" if x.dtype.name.startswith('image') else
                ("const " if i > 0 else "")+self.buffer_prefix+x.dtype.name+"*"+self.buffer_suffix) for i,x in enumerate(bufs)
                if not isinstance(x, LocalBuffer) and not isinstance(x.realized, RawConst)]
    prg = ''.join([f"{self.kernel_prefix} void KERNEL_NAME_PLACEHOLDER(",] +
    [', '.join([f'{t} {bufnames[i]}' for i,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    if self.half_prekernel and any(x.dtype == dtypes.float16 for x in bufs): prg = ''.join([f"{self.half_prekernel}", "\n", prg])

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
    return f"{buf_name}[{idx.render(render_cl)}] = {var_name};"

def add_gl_dimension(prefix: str, args, i:int, var, local_size:List[int], xid:List[str]):
  # for M1 tensor core stuff, support > 3 dims
  if i >= 2 and len(args[0]) > len(xid):
    # do this on the x dim for warps
    if len(local_size) == 2: local_size.append(1)
    local_size[-1] *= var.max+1
    lidx = Variable(xid[0], 0, prod(x.max+1 for x in args[0][2:])-1)
    lidx = (lidx//((lidx.max+1)//local_size[-1]))%(var.max+1)
    assert lidx.max == var.max and lidx.min == var.min
    return f"{{ {prefix} {var.expr} = {lidx.render(render_cl)};  /* {var.max+1} */"
  local_size.append(var.max+1)
  return f"{{ {prefix} {var.expr} = {xid[min(len(xid), len(args[0]))-1-i]};  /* {var.max+1} */"

def uops_to_cstyle(uops:List[UOp], bufs:List[Union[LocalBuffer,LazyBuffer]], lang:CStyleLanguage) -> Tuple[str, List[int], List[int]]:
  kernel,global_size,local_size,prekernel = [],[],[],[]
  pend_close = None
  bufnames = [b.name if isinstance(b, LocalBuffer) else f"data{i}" for i,b in enumerate(bufs)]
  depth = 0
  def kk(s): kernel.append("  "*depth+s)

  for uop,newvar,vin,args in uops:
    if uop == UOps.LOOP:
      for i,var in enumerate(args[0]):
        if isinstance(var, NumNode):
          if args[1] == "global" and lang.gid: global_size.append(1)
          if args[1] == "local" and lang.lid: local_size.append(1)
          # one number, not an index
          kk("{")
        else:
          if args[1] == "global" and lang.gid:
            kk(add_gl_dimension(lang.size_prefix, args, i, var, global_size, lang.gid))
          elif args[1] == "local" and lang.lid:
            kk(add_gl_dimension(lang.size_prefix, args, i, var, local_size, lang.lid))
          else:
            if getenv("NOUNROLL"): kk("#pragma unroll(1)")   # prevent loop unrolling
            kk(lang.render_for(var.expr, var.min, var.max))
      depth += 1
    elif uop == UOps.BARRIER:
      kk(lang.barrier)
    elif uop == UOps.ENDLOOP:
      if args[1] == "local" and len(lang.lid):
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
      # ((lidx2*32)+(lidx3*4)+(lidx4*16)+(lidx5*8)+(lidx6*2))
      kk("{ simdgroup_float8x8 a,b,c;")
      kk(f"a.thread_elements()[0] = {vin[0].render()}; a.thread_elements()[1] = {vin[1].render()};")
      kk(f"b.thread_elements()[0] = {vin[2].render()}; b.thread_elements()[1] = {vin[3].render()};")
      kk(f"c.thread_elements()[0] = {vin[4].render()}; c.thread_elements()[1] = {vin[5].render()};")
      kk("simdgroup_multiply_accumulate(c, a, b, c);")
      kk(f"{vin[4].render()} = c.thread_elements()[0]; {vin[5].render()} = c.thread_elements()[1]; }}")
    elif uop == UOps.CONST:
      assert newvar is not None
      kk(f"{lang.generic_var_prefix}{newvar.render(lang.generic_var_prefix == '')} = {lang.render_const(args, newvar.dtype)};")
    elif uop == UOps.ALU:
      assert newvar is not None
      kk(f"{lang.generic_var_prefix if newvar not in vin else ''}{newvar.render(newvar not in vin and lang.generic_var_prefix == '')} = {lang.code_for_op[args](*[x.render() for x in vin])};")
    elif uop == UOps.LOAD:
      assert newvar is not None
      # valids are handled here
      if args.valid.max == 0:
        val = lang.render_const(0.0, newvar.dtype)
      elif isinstance(bufs[args.i].realized, RawConst):
        val = lang.render_const(bufs[args.i].realized._buf, newvar.dtype)
      else:
        val = lang.render_load(newvar.dtype, bufnames[args.i], bufs[args.i].dtype, args.idx, isinstance(bufs[args.i], LocalBuffer))
      if args.valid.min == 0 and args.valid.max == 1: val = lang.render_conditional(args.valid.render(render_cl), val, lang.render_const(0.0, newvar.dtype))
      kk(f"{lang.generic_var_prefix}{newvar.render(lang.generic_var_prefix == '')} = {val};")
    elif uop == UOps.STORE:
      assert args.valid.min == 1, "store must be valid"
      # TODO: instead of dtypes.float, a base type
      kk(lang.render_store(bufnames[args.i], bufs[args.i].dtype, vin[0].render(), vin[0].dtype if vin[0].offset is None else dtypes.float, args.idx, isinstance(bufs[args.i], LocalBuffer)))
    elif uop == UOps.CAST and newvar is not None and newvar.dtype.sz > 1:
      kk(f"{newvar.render(True)} = {lang.render_cast([x.render() for x in vin], newvar.dtype)};")
    elif uop == UOps.DEFINE_LOCAL:
      if lang.external_local_bufs:
        prekernel.append(lang.render_local(args[0], args[1]))
      else:
        kk(lang.render_local(args[0], args[1]))
    else:
      raise RuntimeError(f"failed to render {uop}")

  return lang.render_kernel(kernel, bufs, bufnames, global_size, local_size, prekernel)

class CStyleCodegen(Linearizer):
  lang: ClassVar[CStyleLanguage] = CStyleLanguage()
  supports_constant_folding: bool = True
  supports_float4: bool = True
  supports_float4_alu: bool = True

  # for renaming
  kernel_cnt: Final[DefaultDict[str, int]] = collections.defaultdict(int)
  kernel_name_cache: Final[Dict[str, Tuple[str, str]]] = {}

  def codegen(self):
    self.process()
    self.hand_coded_optimizations()
    self.limit_global_dims(len(self.lang.gid))  # NOTE: this is optional now
    self.linearize()

    prg, global_size, local_size = uops_to_cstyle(self.uops, self.bufs, self.lang)

    # painfully name the function something unique
    if prg in CStyleCodegen.kernel_name_cache: function_name, display_name = CStyleCodegen.kernel_name_cache[prg]
    else:
      CStyleCodegen.kernel_cnt[self.function_name] += 1
      suffix = f"{'n'+str(CStyleCodegen.kernel_cnt[self.function_name]-1)}" if CStyleCodegen.kernel_cnt[self.function_name] > 1 else ""
      CStyleCodegen.kernel_name_cache[prg] = function_name, display_name = self.function_name+suffix, self.display_name+colored(suffix, 'BLACK')

    return ASTRunner(function_name, prg.replace("KERNEL_NAME_PLACEHOLDER", function_name),
      global_size, local_size,
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=display_name)
