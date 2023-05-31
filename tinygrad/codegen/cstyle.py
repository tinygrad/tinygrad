from typing import Final, Dict, Callable, ClassVar, List, Optional, NamedTuple, DefaultDict, Tuple, Set, Union
import math, collections
from tinygrad.codegen.linearizer import Linearizer, UOps, UOp, LocalBuffer, LocalTypes
from tinygrad.ops import ASTRunner, Op, UnaryOps, BinaryOps, FusedOps
from tinygrad.helpers import getenv, partition, ImageDType, DEBUG, dtypes, colored
from tinygrad.runtime.lib import RawConst
from tinygrad.shape.symbolic import DivNode, AndNode, render_python, NumNode, Variable, Node, SumNode, MulNode
from tinygrad.lazy import LazyBuffer

# div is different in cl than python
render_cl = render_python.copy()
render_cl[DivNode] = lambda self,ops,ctx: f"({self.a.render(ops, ctx)}/{self.b})"
render_cl[AndNode] = lambda self,ops,ctx: f"({'&&'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})"

NATIVE_EXPLOG = getenv("NATIVE_EXPLOG", 0)  # this is needed as a switch for the tests to pass

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

def to_image_idx(base_shape:Tuple[int, ...], idxy:Node, valid:Node, validhacks=False) -> Tuple[Node, Node]:
  idy = (idxy//(4*base_shape[1]))
  if validhacks and valid.min == 0:
    idx = (idxy//4) + (idy*-base_shape[1])
    # find the ones in idx that didn't factorize and remove them (TODO: this is not universal)
    if isinstance(idx, SumNode):
      unfactored, idx_nodes = partition(idx.nodes, lambda x: isinstance(x, MulNode) and x.b == -base_shape[1])
      assert len(unfactored) <= 1
      idx = Variable.sum(idx_nodes)
      unfactored = (Variable.sum(unfactored) // base_shape[1])
      idy += unfactored
      # ugh really...handtuned garbage
      if idx.min >= (base_shape[1]*3)//4:
        idx -= base_shape[1]
        idy += 1
  else:
    idx = (idxy//4)%base_shape[1]
  if DEBUG >= 5: print("to_image_idx", base_shape, idx.min, idx.max, idy.min, idy.max, idx, idy)
  return idx, idy

code_for_op: Final[Dict[Op, Callable]] = {
  UnaryOps.EXP: lambda x: f"native_exp({x})" if NATIVE_EXPLOG else f"exp({x})",
  UnaryOps.LOG: lambda x: f"native_log({x})" if NATIVE_EXPLOG else f"log({x})",
  UnaryOps.SIN: lambda x: f"sin({x})",
  BinaryOps.ADD: lambda a,b: f"({a}+{b})", BinaryOps.SUB: lambda a,b: f"({a}-{b})",
  BinaryOps.MUL: lambda a,b: f"({a}*{b})", BinaryOps.DIV: lambda a,b: f"({a}/{b})",
  BinaryOps.POW: lambda a,b: f"pow({a},{b})", BinaryOps.MAX: lambda a,b: f"max({a},{b})",
  BinaryOps.CMPEQ: lambda a,b: f"({a}=={b})", FusedOps.MULACC: lambda a,b,c: f"(({b}*{c})+{a})"
}

def uops_to_cstyle(uops:List[UOp], bufs:List[Union[LocalBuffer,LazyBuffer]], lang:CStyleLanguage) -> Tuple[str, List[int], List[int]]:
  prekernel: Set[str] = set()
  kernel = []
  global_size = []
  local_size = []
  pend_close = None

  bufnames = ["temp" if isinstance(b, LocalBuffer) else f"data{i}" for i,b in enumerate(bufs)]

  depth = 0
  def kk(s): kernel.append("  "*depth+s)

  for uop,newvar,vin,args in uops:
    if uop == UOps.LOOP:
      root = None
      for i,var in enumerate(args[0]):
        if isinstance(var, NumNode):
          if args[1] == "global" and lang.gid: global_size.append(1)
          if args[1] == "local" and lang.lid: local_size.append(1)
          # one number, not an index
          kk("{")
        else:
          if args[1] == "global" and lang.gid:
            if len(args[0]) >= 4 and len(args[0])-i > 2:
              # sometimes, there's more dimensions. compact all the dimensions into the last CL dimension
              # TODO: these compactions should be searchable (they sort of are with reshapes and permutes)
              if i == 0:
                kk(f"{{ int {var.expr} = {lang.gid[-1]};  /* {var.max+1} */")
                root = var.expr
                global_size.append(var.max+1)
              else:
                kk(f"{{ int {var.expr} = {root} % {var.max+1}; {root} /= {var.max+1};")
                global_size[-1] *= var.max+1
            else:
              kk(f"{{ int {var.expr} = {lang.gid[len(args[0])-1-i]};  /* {var.max+1} */")
              global_size.append(var.max+1)
          elif args[1] == "local" and lang.lid:
            assert len(args[0]) <= len(lang.lid)
            kk(f"{{ int {var.expr} = {lang.lid[len(args[0])-1-i]};  /* {var.max+1} */")
            local_size.append(var.max+1)
          else:
            kk(f"for (int {var.expr} = {var.min}; {var.expr} <= {var.max}; ++{var.expr}) {{")
      depth += 1
    elif uop == UOps.ENDLOOP:
      if args[1] == "local" and len(lang.lid):
        # TODO: this is a bit of a hack. the local loop isn't real on the GPU
        kk(lang.barrier)
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
      if args == -math.inf:
        kk(f"{newvar.render(True)} = -INFINITY;")
      elif newvar.ltype == LocalTypes.float4:
        kk(f"{newvar.render(True)} = {{ {args}f, {args}f, {args}f, {args}f }};")
      else:
        kk(f"{newvar.render(True)} = {args}f;")
    elif uop == UOps.ALU:
      assert newvar is not None
      if newvar in vin:
        kk(f"{newvar.render()} = {code_for_op[args](*[x.render() for x in vin])};")
      else:
        kk(f"{newvar.render(True)} = {code_for_op[args](*[x.render() for x in vin])};")
    elif uop == UOps.LOAD and newvar is not None:
      # TODO: merge with CONST?
      if bufs[args.i] is not None and isinstance(bufs[args.i].realized, RawConst):
        assert newvar.ltype == LocalTypes.float, "const can't be float4"
        # nan? inf?
        val = f"{bufs[args.i].realized._buf}" + ("f" if not dtypes.is_int(bufs[args.i].dtype) else "")
      elif isinstance(bufs[args.i].dtype, ImageDType):
        assert newvar.ltype == LocalTypes.float4, "image must be float4"
        prekernel.add("const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n")
        idx, idy = to_image_idx(bufs[args.i].dtype.shape, args.idx, args.valid)
        val = f"read_imagef({bufnames[args.i]}, smp, (int2)({idx.render(render_cl)}, {idy.render(render_cl)}))"
      else:
        if lang.uses_vload and bufs[args.i].dtype == dtypes.float16:
          if newvar.ltype == LocalTypes.float4:
            val = f"vload_half4({(args.idx//4).render(render_cl)}, {bufnames[args.i]})"
          else:
            val = f"vload_half({args.idx.render(render_cl)}, {bufnames[args.i]})"
        else:
          if newvar.ltype == LocalTypes.float4:
            val = f"({newvar.ltype.name})((({lang.smem_prefix if isinstance(bufs[args.i], LocalBuffer) else lang.buffer_prefix}{bufs[args.i].dtype.name}4*){bufnames[args.i]})[{(args.idx//4).render(render_cl)}])"
          else:
            val = f"{bufnames[args.i]}[{args.idx.render(render_cl)}]"
      # NOTE: if min and max are both 0, it should be a CONST in the Linearizer
      if args.valid.min == 1: kk(f"{newvar.render(True)} = {val};")
      else:
        zero = f"{lang.float4}(0.0f, 0.0f, 0.0f, 0.0f);" if newvar.ltype == LocalTypes.float4 else "0.0f"
        kk(f"{newvar.render(True)} = ({args.valid.render(render_cl)}) ? ({val}) : {zero};")
    elif uop == UOps.STORE and (vin[0].ltype == LocalTypes.float or (vin[0].ltype == LocalTypes.float4 and vin[0].offset is not None)):
      assert not isinstance(bufs[args.i].dtype, ImageDType), "image store must be float4"
      assert args.valid.min == 1, "store must be valid"
      if lang.uses_vload and bufs[args.i].dtype == dtypes.float16:
        kk(f"vstore_half({vin[0].render()}, {args.idx.render(render_cl)}, {bufnames[args.i]});")
      else:
        kk(f"{bufnames[args.i]}[{args.idx.render(render_cl)}] = {vin[0].render()};")
    elif uop == UOps.CAST and newvar is not None and newvar.ltype == LocalTypes.float4:
      kk(f"{newvar.render(True)} = {lang.float4}({','.join([x.render() for x in vin])});")
    elif uop == UOps.STORE and len(vin) != 0 and vin[0].ltype == LocalTypes.float4 and vin[0].offset is None:
      assert args.valid.min == 1, "store must be valid"
      if isinstance(bufs[args[0]].dtype, ImageDType):
        idx, idy = to_image_idx(bufs[args.i].dtype.shape, args[1], args[2])
        kk(f"write_imagef({bufnames[args.i]}, (int2)({idx.render(render_cl)}, {idy.render(render_cl)}), {vin[0].render()});")
      else:
        kk(f"(({lang.smem_prefix if isinstance(bufs[args.i], LocalBuffer) else lang.buffer_prefix}float4*){bufnames[args.i]})[{(args.idx//4).render(render_cl)}] = {vin[0].render()};")
    elif uop == UOps.DEFINE_LOCAL:
      kk(lang.smem_prefix + f"float {args[0]}[{args[1]}];")
    else:
      raise RuntimeError(f"failed to render {uop}")

  buftypes = [(i,f"{'read_only' if i > 0 else 'write_only'} image2d_t" if x.dtype.name.startswith('image') else
               ("const " if i > 0 else "")+lang.buffer_prefix+x.dtype.name+"*"+lang.buffer_suffix) for i,x in enumerate(bufs)
               if not isinstance(x, LocalBuffer) and not isinstance(x.realized, RawConst)]
  prg = ''.join([f"{lang.kernel_prefix} void KERNEL_NAME_PLACEHOLDER(",] +
    [', '.join([f'{t} {bufnames[i]}' for i,t in buftypes] + lang.extra_args)] +
    [") {\n"] + list(prekernel) + ['\n'.join(kernel), "\n}"])

  if lang.half_prekernel:
    prg =''.join([f"{lang.half_prekernel}", "\n", prg])
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

    # sometimes, there's more dimensions than len(self.lang.gid).
    # compact all the dimensions into the first
    # NOTE: this might make multiview shapetrackers
    # TODO: this exposes bugs in the optimizers assuming the strides are on a single view
    """
    if len(self.lang.gid) and self.first_reduce > len(self.lang.gid):
      num_to_merge = (self.first_reduce - len(self.lang.gid))+1
      self.reshape_and_permute(lambda x: (prod(x[0:num_to_merge]),)+x[num_to_merge:], None)
      if DEBUG >= 4: print("reshaped to", self.full_shape, "due to too many global dimensions")
    """

    self.linearize()

    prg, global_size, local_size = uops_to_cstyle(self.uops, self.bufs, self.lang)

    # if we have local_sizes, we have to correct the global_size
    for i,s in enumerate(local_size): global_size[i] *= s

    # painfully name the function something unique
    if prg in CStyleCodegen.kernel_name_cache: function_name, display_name = CStyleCodegen.kernel_name_cache[prg]
    else:
      CStyleCodegen.kernel_cnt[self.function_name] += 1
      suffix = f"{'n'+str(CStyleCodegen.kernel_cnt[self.function_name]-1)}" if CStyleCodegen.kernel_cnt[self.function_name] > 1 else ""
      CStyleCodegen.kernel_name_cache[prg] = function_name, display_name = self.function_name+suffix, self.display_name+colored(suffix, 'black', bright=True)

    return ASTRunner(function_name, prg.replace("KERNEL_NAME_PLACEHOLDER", function_name),
      global_size[::-1] if len(global_size) else [1], local_size[::-1] if len(local_size) else None,
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=display_name)
