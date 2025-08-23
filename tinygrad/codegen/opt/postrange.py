from dataclasses import replace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, AxisType
from tinygrad.helpers import colored
from tinygrad.codegen.opt.kernel import Opt, OptOps, axis_colors
from tinygrad.dtype import dtypes

def add_name(s:UOp):
  rng = sorted([u for u in s.toposort() if u.op is Ops.RANGE], key=lambda x: x.arg)
  maxarg = max([x.arg[0] for x in rng])
  name = "k"+colored('_', 'BLACK').join(['']+[colored(s.src[0].render(), axis_colors[s.arg[1]]) for s in rng])
  opts_to_apply = list(s.arg.opts_to_apply)

  if len(opts_to_apply):
    opt:Opt = opts_to_apply.pop(0)
    if opt.op == OptOps.UPCAST:
      re_rng = rng[opt.axis]
      assert re_rng.src[0].op is Ops.CONST
      assert re_rng.src[0].arg % opt.arg == 0
      srng = re_rng.replace(src=(UOp.const(dtypes.int, re_rng.src[0].arg // opt.arg),)) * opt.arg + \
        UOp.range(dtypes.int, opt.arg, maxarg+1, AxisType.UPCAST)
      s = s.substitute({re_rng: srng})
    else:
      raise RuntimeError(f"op not supported {opt.op}")

  return s.replace(arg=replace(s.arg, name=name, opts_to_apply=tuple(opts_to_apply)))

pm_postrange_opt = PatternMatcher([
  (UPat(Ops.SINK, name="s"), add_name),
])