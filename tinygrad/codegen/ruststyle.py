from typing import Final, Dict, Callable, List, DefaultDict, Tuple, Set, Union
import math, collections
from tinygrad.codegen.linearizer import Linearizer, UOps, UOp, LocalBuffer, LocalTypes
from tinygrad.ops import ASTRunner, Op, UnaryOps, BinaryOps, FusedOps
from tinygrad.helpers import ImageDType, colored
from tinygrad.runtime.lib import RawConst
from tinygrad.shape.symbolic import DivNode, AndNode, render_python, NumNode
from tinygrad.lazy import LazyBuffer

# div is different in cl than python
render_cl = render_python.copy()
render_cl[DivNode] = lambda self,ops,ctx: f"({self.a.render(ops, ctx)}/{self.b})"
render_cl[AndNode] = lambda self,ops,ctx: f"({'&&'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})"

code_for_op: Final[Dict[Op, Callable]] = {
  UnaryOps.EXP: lambda x:  f"exp({x})",
  UnaryOps.LOG: lambda x:  f"log({x})",
  BinaryOps.ADD: lambda a,b: f"({a}+{b})", BinaryOps.SUB: lambda a,b: f"({a}-{b})",
  BinaryOps.MUL: lambda a,b: f"({a}*{b})", BinaryOps.DIV: lambda a,b: f"({a}/{b})",
  BinaryOps.POW: lambda a,b: f"pow({a},{b})", BinaryOps.MAX: lambda a,b: f"max({a},{b})",
  BinaryOps.CMPEQ: lambda a,b: f"({a}=={b})", FusedOps.MULACC: lambda a,b,c: f"(({b}*{c})+{a})"
}

def uops_to_rust(uops:List[UOp], bufs:List[Union[LocalBuffer,LazyBuffer]]) -> str:
  kernel = []

  bufnames = ["temp" if isinstance(b, LocalBuffer) else f"data{i}" for i,b in enumerate(bufs)]

  depth = 0
  def kk(s): kernel.append("  "*depth+s)

  mutations = []

  for uop,newvar,vin,args in uops:
    if uop == UOps.LOOP:
      for var in args[0]:
        if isinstance(var, NumNode):
          kk("{")
        else:
          kk(f"for {var.expr} in {var.min}..={var.max}isize {{")
      depth += 1
    elif uop == UOps.ENDLOOP:
      if args[1] == "global":
        depth -= 1
      depth -= 1
      kk("}"*len(args[0]) + f" // {args[1]}")
    elif uop == UOps.CONST:
      assert newvar is not None
      if args == -math.inf:
        kk(f"{newvar.render(True)} = f32::NEG_INFINITY;")
      elif newvar.ltype == LocalTypes.float4:
        assert(False)
      else:
        kk(f"{newvar.render(True)} = {args};")
    elif uop == UOps.ALU:
      assert newvar is not None
      if newvar in vin:
        kk(f"{newvar.render()} = {code_for_op[args](*[x.render() for x in vin])};")
      else:
        kk(f"{newvar.render(True)} = {code_for_op[args](*[x.render() for x in vin])};")
    elif uop == UOps.LOAD and newvar is not None and newvar.ltype == LocalTypes.float:
      assert not isinstance(bufs[args.i].dtype, ImageDType), "image load must be float4"
      # TODO: merge with CONST?
      if bufs[args.i] is not None and isinstance(bufs[args.i].realized, RawConst):
        # nan? inf?
        val = f"{bufs[args.i].realized._buf}"
      else:
        val = f"unsafe {{ {bufnames[args.i]}[{args.idx.render(render_cl)} as usize] }}"
      # NOTE: if min and max are both 0, it should be a CONST in the Linearizer
      if args.valid.min == 1: kk(f"let {newvar.name} = {val};")
      else: kk(f"let {newvar.name} = if {args.valid.render(render_cl)} {{{val}}} else {{0.0}};")
    elif uop == UOps.STORE and vin[0].ltype == LocalTypes.float:
      assert not isinstance(bufs[args.i].dtype, ImageDType), "image store must be float4"
      assert args.valid.min == 1, "store must be valid"
      kk(f"unsafe {{ {bufnames[args.i]}[{args.idx.render(render_cl)} as usize] = {vin[0].render()}; }}")
      mutations.append(bufnames[args.i])
    elif uop == UOps.DEFINE_LOCAL:
      kk(f"static mut {args[0]} : &'static mut [f32] = &mut[0.0; {args[1]}];")
    else:
      raise RuntimeError(f"failed to render {uop}")

  buftypes = [(i,f"{x.dtype.name}", x.realized.size) for i,x in enumerate(bufs) if not isinstance(x, LocalBuffer) and not isinstance(x.realized, RawConst)]
  prg = ''.join(["#[inline(always)]\npub fn KERNEL_NAME_PLACEHOLDER(",] +
    [', '.join([f'{bufnames[i]} : &mut [{t}; {s}]' if bufnames[i] in mutations else f'{bufnames[i]} : &[{t}; {s}]' for i,t,s in buftypes])] +
    [") {\n"] + ['\n'.join(kernel), "\n}"])


  args = ', '.join([f'std::convert::TryInto::try_into(std::slice::from_raw_parts_mut({bufnames[i]} as *mut f32, {s})).unwrap()' if bufnames[i] in mutations else f'std::convert::TryInto::try_into(std::slice::from_raw_parts({bufnames[i]} as *const f32, {s})).unwrap()' for i,t,s in buftypes])

  prg += ''.join([f"\n\n#[no_mangle]\npub extern \"C\" fn KERNEL_NAME_PLACEHOLDER_c(",] +
    [', '.join([f'{bufnames[i]} : *mut f32' if bufnames[i] in mutations else f'{bufnames[i]} : *const f32' for i,t,s in buftypes])] +
    [") {\n"] + [f"unsafe {{ KERNEL_NAME_PLACEHOLDER({args}) }};\n" + "}\n\n"])

  return prg

class RustCodegen(Linearizer):
  supports_constant_folding: bool = True

  # for renaming
  kernel_cnt: Final[DefaultDict[str, int]] = collections.defaultdict(int)
  kernel_name_cache: Final[Dict[str, Tuple[str, str]]] = {}

  def codegen(self):
    self.process()
    self.hand_coded_optimizations()

    # sometimes, there's more dimensions than len(self.lang.gid).
    # compact all the dimensions into the first
    # NOTE: this might make multiview shapetrackers
    # TODO: this exposes bugs in the optimizers assuming the strides are on a single vie
    """
    if len(self.lang.gid) and self.first_reduce > len(self.lang.gid):
      num_to_merge = (self.first_reduce - len(self.lang.gid))+1
      self.reshape_and_permute(lambda x: (prod(x[0:num_to_merge]),)+x[num_to_merge:], None)
      if DEBUG >= 4: print("reshaped to", self.full_shape, "due to too many global dimensions")
    """

    self.linearize()

    prg = uops_to_rust(self.uops, self.bufs)


    # painfully name the function something unique
    if prg in RustCodegen.kernel_name_cache: function_name, display_name = RustCodegen.kernel_name_cache[prg]
    else:
      RustCodegen.kernel_cnt[self.function_name] += 1
      suffix = f"{'n'+str(RustCodegen.kernel_cnt[self.function_name]-1)}" if RustCodegen.kernel_cnt[self.function_name] > 1 else ""
      RustCodegen.kernel_name_cache[prg] = function_name, display_name = self.function_name+suffix, self.display_name+colored(suffix, 'black', bright=True)

    return ASTRunner(function_name, prg.replace("KERNEL_NAME_PLACEHOLDER", function_name), None,
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=display_name)
