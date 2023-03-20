from typing import Final, Dict, Callable, ClassVar, List, Optional, NamedTuple, DefaultDict, Tuple, Set
import math, collections
from tinygrad.codegen.linearizer import Linearizer, UOps
from tinygrad.ops import ASTRunner, Op, UnaryOps, BinaryOps, FusedOps
from tinygrad.helpers import getenv, all_same, partition, ImageDType, DEBUG, dtypes
from tinygrad.runtime.lib import RawConst
from tinygrad.shape.symbolic import DivNode, AndNode, render_python, NumNode, Variable, Node, SumNode, MulNode

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

class CStyleCodegen(Linearizer):
  lang: ClassVar[CStyleLanguage] = CStyleLanguage()
  supports_constant_folding: bool = True
  supports_float4: bool = True

  # for renaming
  kernel_cnt: Final[DefaultDict[str, int]] = collections.defaultdict(int)
  kernel_name_cache: Final[Dict[str, str]] = {}

  code_for_op: Final[Dict[Op, Callable]] = {
    UnaryOps.EXP: lambda x: f"native_exp({x})" if NATIVE_EXPLOG else f"exp({x})",
    UnaryOps.LOG: lambda x: f"native_log({x})" if NATIVE_EXPLOG else f"log({x})",
    BinaryOps.ADD: lambda a,b: f"({a}+{b})", BinaryOps.SUB: lambda a,b: f"({a}-{b})",
    BinaryOps.MUL: lambda a,b: f"({a}*{b})", BinaryOps.DIV: lambda a,b: f"({a}/{b})",
    BinaryOps.POW: lambda a,b: f"pow({a},{b})", BinaryOps.MAX: lambda a,b: f"max({a},{b})",
    BinaryOps.CMPEQ: lambda a,b: f"({a}=={b})", FusedOps.MULACC: lambda a,b,c: f"(({b}*{c})+{a})"
  }

  def group_float4(self, grp:List[str]) -> str:
    if all(g.endswith(e) for g,e in zip(grp, [".x", ".y", ".z", ".w"])) and all_same([g.split(".")[0] for g in grp]): return grp[0].split(".")[0]
    else: return f"{self.lang.float4}({','.join(g for g in grp)})"

  def codegen(self):
    self.process()

    # sometimes, there's more dimensions than len(self.lang.gid).
    # compact all the dimensions into the first
    # NOTE: this might make multiview shapetrackers
    # NOTE: you ABSOLUTELY must do this before upcasting. the strides on the upcast are wrong if you don't
    # TODO: this exposes bugs in the optimizers assuming the strides are on a single view
    """
    if len(self.lang.gid) and self.first_reduce > len(self.lang.gid):
      num_to_merge = (self.first_reduce - len(self.lang.gid))+1
      self.reshape_and_permute(lambda x: (prod(x[0:num_to_merge]),)+x[num_to_merge:], None)
      if DEBUG >= 4: print("reshaped to", self.full_shape, "due to too many global dimensions")
    """

    self.hand_coded_optimizations()
    self.linearize()

    prekernel: Set[str] = set()
    kernel = []
    global_size = []
    local_size = []
    pend_close = None

    depth = 0
    def kk(s): kernel.append("  "*depth+s)

    for uop,newvar,args in self.uops:
      if uop == UOps.LOOP:
        root = None
        for i,var in enumerate(args[0]):
          if isinstance(var, NumNode):
            if args[1] == "global" and self.lang.gid: global_size.append(1)
            if args[1] == "local" and self.lang.lid: local_size.append(1)
            # one number, not an index
            kk("{")
          else:
            if args[1] == "global" and self.lang.gid:
              if len(args[0]) >= 4 and len(args[0])-i > 2:
                # sometimes, there's more dimensions. compact all the dimensions into the last CL dimension
                # TODO: these compactions should be searchable (they sort of are with reshapes and permutes)
                if i == 0:
                  kk(f"{{ int {var.expr} = {self.lang.gid[-1]};  /* {var.max+1} */")
                  root = var.expr
                  global_size.append(var.max+1)
                else:
                  kk(f"{{ int {var.expr} = {root} % {var.max+1}; {root} /= {var.max+1};")
                  global_size[-1] *= var.max+1
              else:
                kk(f"{{ int {var.expr} = {self.lang.gid[len(args[0])-1-i]};  /* {var.max+1} */")
                global_size.append(var.max+1)
            elif args[1] == "local" and self.lang.lid:
              assert len(args[0]) <= len(self.lang.lid)
              kk(f"{{ int {var.expr} = {self.lang.lid[len(args[0])-1-i]};  /* {var.max+1} */")
              local_size.append(var.max+1)
            else:
              kk(f"for (int {var.expr} = {var.min}; {var.expr} <= {var.max}; ++{var.expr}) {{")
        depth += 1
      if uop == UOps.ENDLOOP:
        if args[1] == "local" and len(self.lang.lid):
          # TODO: this is a bit of a hack. the local loop isn't real on the GPU
          kk(self.lang.barrier)
          kk(f"if ({Variable.sum(args[0]).render(render_cl)} == 0) {{")
          pend_close = "}"*(len(args[0])+1) + f" /* {args[1]} */"
        else:
          if args[1] == "global" and pend_close:
            depth -= 1
            kk(pend_close)
            pend_close = None
          depth -= 1
          kk("}"*len(args[0]) + f" /* {args[1]} */")
      if uop == UOps.CONST:
        if args[0] == -math.inf:
          kk(f"float {newvar} = -INFINITY;")
        else:
          kk(f"float {newvar} = {args[0]}f;")
      if uop == UOps.ALU:
        if newvar is None:
          kk(f"{args[2]} = {self.code_for_op[args[0]](*args[1])};")
        else:
          kk(f"float {newvar} = {self.code_for_op[args[0]](*args[1])};")
      # TODO: refactor the next 14 lines
      if uop == UOps.LOAD:
        # TODO: merge with CONST?
        if self.bufs[args[0]] is not None and isinstance(self.bufs[args[0]].realized, RawConst):
          # nan? inf?
          val = f"{self.bufs[args[0]].realized._buf}f"
        else:
          if self.lang.uses_vload and self.bufs[args[0]].dtype == dtypes.float16:
            val = f"vload_half({args[1].render(render_cl)}, {self.registers[args[0]].name})"
          else:
            val = f"{self.registers[args[0]].name}[{args[1].render(render_cl)}]"
        # NOTE: if min and max are both 0, it should be a CONST in the Linearizer
        if args[2].min == 1: kk(f"float {newvar} = {val};")
        else: kk(f"float {newvar} = ({args[2].render(render_cl)}) ? ({val}) : 0.0f;")
      if uop == UOps.LOAD4:
        if self.bufs[args[0]] is not None and isinstance(self.bufs[args[0]].dtype, ImageDType):
          prekernel.add("const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n")
          idx, idy = to_image_idx(self.bufs[args[0]].dtype.shape, args[1], args[2])
          val = f"read_imagef({self.registers[args[0]].name}, smp, (int2)({idx.render(render_cl)}, {idy.render(render_cl)}))"
        else:
          val = f"(({self.lang.buffer_prefix if self.bufs[args[0]] is not None else self.lang.smem_prefix}float4*){self.registers[args[0]].name})[{(args[1]//4).render(render_cl)}]"
        # NOTE: if min and max are both 0, it should be a CONST in the Linearizer
        if args[2].min == 1: kk(f"float4 {newvar} = {val};")
        else: kk(f"float4 {newvar} = ({args[2].render(render_cl)}) ? ({val}) : {self.group_float4(['0.0f']*4)};")
      if uop == UOps.STORE:
        assert args[2].min == 1, "store must be valid"
        if self.lang.uses_vload and self.bufs[args[0]].dtype == dtypes.float16:
          kk(f"vstore_half({args[3]}, {args[1].render(render_cl)}, {self.registers[args[0]].name});")
        else:
          kk(f"{self.registers[args[0]].name}[{args[1].render(render_cl)}] = {args[3]};")
      if uop == UOps.STORE4:
        assert args[2].min == 1, "store must be valid"
        if self.bufs[args[0]] is not None and isinstance(self.bufs[args[0]].dtype, ImageDType):
          idx, idy = to_image_idx(self.bufs[args[0]].dtype.shape, args[1], args[2])
          kk(f"write_imagef({self.registers[args[0]].name}, (int2)({idx.render(render_cl)}, {idy.render(render_cl)}), {self.group_float4(args[3])});")
        else:
          kk(f"(({self.lang.buffer_prefix if self.bufs[args[0]] is not None else self.lang.smem_prefix}float4*){self.registers[args[0]].name})[{(args[1]//4).render(render_cl)}] = {self.group_float4(args[3])};")
      if uop == UOps.DEFINE_LOCAL:
        kk(self.lang.smem_prefix + f"float {args[0]}[{args[1]}];")

    buftypes = [(i,f"{'read_only' if i > 0 else 'write_only'} image2d_t" if x.dtype.name.startswith('image') else self.lang.buffer_prefix+x.dtype.name+"*"+self.lang.buffer_suffix) for i,x in enumerate(self.bufs) if x is not None and not isinstance(x.realized, RawConst)]
    prg = ''.join([f"{self.lang.kernel_prefix} void KERNEL_NAME_PLACEHOLDER(",] +
      [', '.join([f'{"const" if i > 0 else ""} {t} data{i}' for i,t in buftypes] + self.lang.extra_args)] +
      [") {\n"] + list(prekernel) + ['\n'.join(kernel), "\n}"])

    # if we have local_sizes, we have to correct the global_size
    for i,s in enumerate(local_size): global_size[i] *= s

    # painfully name the function something unique
    function_name = self.function_name
    if prg in CStyleCodegen.kernel_name_cache: function_name = CStyleCodegen.kernel_name_cache[prg]
    else:
      CStyleCodegen.kernel_cnt[function_name] += 1
      if CStyleCodegen.kernel_cnt[function_name] > 1: function_name = f"{function_name}{'n'+str(CStyleCodegen.kernel_cnt[function_name]-1)}"
      CStyleCodegen.kernel_name_cache[prg] = function_name

    return ASTRunner(function_name, prg.replace("KERNEL_NAME_PLACEHOLDER", function_name),
      global_size[::-1] if len(global_size) else [1], local_size[::-1] if len(local_size) else None,
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate)
