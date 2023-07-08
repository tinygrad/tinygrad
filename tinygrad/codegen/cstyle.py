from typing import Final, Dict, Callable, ClassVar, List, Optional, NamedTuple, DefaultDict, Tuple, Set, Union
import math, collections
from tinygrad.codegen.linearizer import Linearizer, UOps, UOp, LocalBuffer
from tinygrad.ops import ASTRunner, Op, UnaryOps, BinaryOps, FusedOps
from tinygrad.helpers import ImageDType, dtypes, colored
from tinygrad.runtime.lib import RawConst
from tinygrad.shape.symbolic import DivNode, AndNode, render_python, NumNode, Variable
from tinygrad.lazy import LazyBuffer

# div is different in cl than python
render_cl = render_python.copy()
render_cl[DivNode] = lambda self,ops,ctx: f"({self.a.render(ops, ctx)}/{self.b})"
render_cl[AndNode] = lambda self,ops,ctx: f"({'&&'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})"

class CStyleLanguage(NamedTuple):
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

  # returns a str expression of the const with the given type
  def render_const(self, x:Union[float,int], var_dtype=dtypes.float) -> str:
    if math.isnan(x): val = "NAN"
    elif math.isinf(x): val = ("-" if x < 0 else "") + "INFINITY"
    else: val = f"{x}" + ("f" if isinstance(x, float) else "")
    return f"{self.float4}({val}, {val}, {val}, {val})" if var_dtype == dtypes._float4 else val

  # returns a str expression of the loaded value with the output type
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert output_dtype == dtypes._float4, "images must be float4"
      return f"read_imagef({buf_name}, smp, (int2)({idx[0].render(render_cl)}, {idx[1].render(render_cl)}))"
    elif self.uses_vload and buf_dtype == dtypes.float16:
      return f"vload_half{'' if output_dtype.sz == 1 else str(output_dtype.sz)}(0, {buf_name}+{idx.render(render_cl)})"
    elif output_dtype == dtypes._float4:
      return f"({output_dtype.name})(*(({self.smem_prefix if local else self.buffer_prefix}{buf_dtype.name}{output_dtype.sz}*)({buf_name}+{idx.render(render_cl)})))"
    else:
      return f"{buf_name}[{idx.render(render_cl)}]"

  # returns a str statement that does the store
  def render_store(self, buf_name, buf_dtype, var_name, var_dtype, idx, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert var_dtype == dtypes._float4, "images must be float4"
      return f"write_imagef({buf_name}, (int2)({idx[0].render(render_cl)}, {idx[1].render(render_cl)}), {var_name});"
    if self.uses_vload and buf_dtype == dtypes.float16:
      return f"vstore_half{'' if var_dtype.sz == 1 else str(var_dtype.sz)}(0, {buf_name}+{idx.render(render_cl)})"
    elif var_dtype.sz > 1:
      return f"*(({self.smem_prefix if local else self.buffer_prefix}{buf_dtype.name}{var_dtype.sz}*)({buf_name}+{idx.render(render_cl)})) = ({buf_dtype.name}{var_dtype.sz}){var_name};"
    else:
      return f"{buf_name}[{idx.render(render_cl)}] = {var_name};"

code_for_op: Final[Dict[Op, Callable]] = {
  UnaryOps.EXP2: lambda x: f"exp2({x})",
  UnaryOps.LOG2: lambda x: f"log2({x})",
  UnaryOps.SIN: lambda x: f"sin({x})",
  UnaryOps.SQRT: lambda x: f"sqrt({x})",
  BinaryOps.ADD: lambda a,b: f"({a}+{b})", BinaryOps.SUB: lambda a,b: f"({a}-{b})",
  BinaryOps.MUL: lambda a,b: f"({a}*{b})", BinaryOps.DIV: lambda a,b: f"({a}/{b})",
  BinaryOps.MAX: lambda a,b: f"max({a},{b})",
  BinaryOps.CMPEQ: lambda a,b: f"({a}=={b})", FusedOps.MULACC: lambda a,b,c: f"(({a}*{b})+{c})"
}

def uops_to_cstyle(uops:List[UOp], bufs:List[Union[LocalBuffer,LazyBuffer]], lang:CStyleLanguage) -> Tuple[str, List[int], List[int]]:
  prekernel: Set[str] = set()
  kernel = []
  global_size = []
  local_size = []
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
            assert len(args[0]) <= len(lang.gid), f"too many global dimensions, has {len(args[0])} and {len(lang.gid)} are supported"
            kk(f"{{ int {var.expr} = {lang.gid[len(args[0])-1-i]};  /* {var.max+1} */")
            global_size.append(var.max+1)
          elif args[1] == "local" and lang.lid:
            assert len(args[0]) <= len(lang.lid)
            kk(f"{{ int {var.expr} = {lang.lid[len(args[0])-1-i]};  /* {var.max+1} */")
            local_size.append(var.max+1)
          else:
            kk(f"for (int {var.expr} = {var.min}; {var.expr} <= {var.max}; ++{var.expr}) {{")
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
    elif uop == UOps.CONST:
      assert newvar is not None
      kk(f"{newvar.render(True)} = {lang.render_const(args)};")
    elif uop == UOps.ALU:
      assert newvar is not None
      kk(f"{newvar.render(newvar not in vin)} = {code_for_op[args](*[x.render() for x in vin])};")
    elif uop == UOps.LOAD:
      assert newvar is not None
      # valids are handled here
      if args.valid.max == 0:
        val = lang.render_const(0.0, newvar.dtype)
      elif isinstance(bufs[args.i].realized, RawConst):
        val = lang.render_const(bufs[args.i].realized._buf, newvar.dtype)
      else:
        val = lang.render_load(newvar.dtype, bufnames[args.i], bufs[args.i].dtype, args.idx, isinstance(bufs[args.i], LocalBuffer))
      if args.valid.min == 0 and args.valid.max == 1: val = f"({args.valid.render(render_cl)}) ? ({val}) : ({lang.render_const(0.0, newvar.dtype)});"
      kk(f"{newvar.render(True)} = {val};")
    elif uop == UOps.STORE:
      assert args.valid.min == 1, "store must be valid"
      kk(lang.render_store(bufnames[args.i], bufs[args.i].dtype, vin[0].render(), vin[0].dtype, args.idx, isinstance(bufs[args.i], LocalBuffer)))
    elif uop == UOps.CAST and newvar is not None and newvar.dtype == dtypes._float4:
      kk(f"{newvar.render(True)} = {lang.float4}({','.join([x.render() for x in vin])});")
    elif uop == UOps.DEFINE_LOCAL:
      kk(lang.smem_prefix + f"float {args[0]}[{args[1]}];")
    else:
      raise RuntimeError(f"failed to render {uop}")

  if any(isinstance(x.dtype, ImageDType) for x in bufs): prekernel.add("const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n")
  buftypes = [(i,f"{'read_only' if i > 0 else 'write_only'} image2d_t" if x.dtype.name.startswith('image') else
               ("const " if i > 0 else "")+lang.buffer_prefix+x.dtype.name+"*"+lang.buffer_suffix) for i,x in enumerate(bufs)
               if not isinstance(x, LocalBuffer) and not isinstance(x.realized, RawConst)]
  prg = ''.join([f"{lang.kernel_prefix} void KERNEL_NAME_PLACEHOLDER(",] +
    [', '.join([f'{t} {bufnames[i]}' for i,t in buftypes] + lang.extra_args)] +
    [") {\n"] + list(prekernel) + ['\n'.join(kernel), "\n}"])

  if lang.half_prekernel and any(x.dtype == dtypes.float16 for x in bufs): prg = ''.join([f"{lang.half_prekernel}", "\n", prg])
  return prg, global_size, local_size

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
    self.limit_global_dims(len(self.lang.gid))
    self.linearize()

    prg, global_size, local_size = uops_to_cstyle(self.uops, self.bufs, self.lang)

    # painfully name the function something unique
    if prg in CStyleCodegen.kernel_name_cache: function_name, display_name = CStyleCodegen.kernel_name_cache[prg]
    else:
      CStyleCodegen.kernel_cnt[self.function_name] += 1
      suffix = f"{'n'+str(CStyleCodegen.kernel_cnt[self.function_name]-1)}" if CStyleCodegen.kernel_cnt[self.function_name] > 1 else ""
      CStyleCodegen.kernel_name_cache[prg] = function_name, display_name = self.function_name+suffix, self.display_name+colored(suffix, 'BLACK')

    return ASTRunner(function_name, prg.replace("KERNEL_NAME_PLACEHOLDER", function_name),
      global_size[::-1], local_size[::-1],
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=display_name)
