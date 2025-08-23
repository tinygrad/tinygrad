from dataclasses import replace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp
from tinygrad.helpers import colored
from tinygrad.codegen.opt.kernel import Opt, OptOps, axis_colors

def add_name(s:UOp):
  rng = sorted([u for u in s.toposort() if u.op is Ops.RANGE], key=lambda x: x.arg)
  name = "k"+colored('_', 'BLACK').join(['']+[colored(s.src[0].render(), axis_colors[s.arg[1]]) for s in rng])
  opts_to_apply = list(s.arg.opts_to_apply)
  if len(opts_to_apply):
    opt:Opt = opts_to_apply.pop(0)
    if opt.op == OptOps.UPCAST:
      #rng[opt.axis]
      #opt.arg
      pass
  return s.replace(arg=replace(s.arg, name=name, opts_to_apply=tuple(opts_to_apply)))

pm_postrange_opt = PatternMatcher([
  (UPat(Ops.SINK, name="s"), add_name),
])