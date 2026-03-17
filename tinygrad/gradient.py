from typing import cast
import math, dataclasses
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, all_metadata
from tinygrad.helpers import argsort

def reduce_gradient(ctx:UOp, ret:UOp, op:Ops):
  def broadcast_to_input(x): return x.reshape(x.shape+(1,)*(len(ret.src[0].shape)-len(x.shape))).expand(ret.src[0].shape)
  if op == Ops.ADD: return (broadcast_to_input(ctx),)
  if op == Ops.MAX:
    assert ret.op is Ops.REDUCE_AXIS, "only works on REDUCE_AXIS"
    mask = ret.src[0].eq(broadcast_to_input(ret)).cast(ctx.dtype)
    count = mask.r(Ops.ADD, ret.arg[1])
    return ((mask/broadcast_to_input(count)) * broadcast_to_input(ctx),)
  if op == Ops.MUL: return (broadcast_to_input(ctx * ret) / ret.src[0],)

def call_gradient(ctx:UOp, k:UOp, needed:set[int]) -> tuple[UOp|None, ...]:
  fxn, args = k.src[0], k.src[1:]
  if k.arg.grad_fxn is not None:
    if ctx.op is Ops.TUPLE:
      real = [g for g in ctx.src if g.op is not Ops.NOOP]
      return (None,) + (k.arg.grad_fxn(*real, call=k) if len(real) > 1 else k.arg.grad_fxn(real[0], k))
    return (None,) + k.arg.grad_fxn(ctx, k)
  assert fxn.op is Ops.TUPLE, f"expected TUPLE body for gradient, got {fxn.op}"
  params = {x.arg:x for x in fxn.toposort(enter_calls=False) if x.op == Ops.PARAM}

  # only pass real grad outputs as external args, embed const 0 in the body for NOOP slots
  ext_grad_args: list[UOp] = []
  root_grad_srcs: list[UOp] = []
  for i, g in enumerate(ctx.src):
    if g.op is Ops.NOOP:
      root_grad_srcs.append(fxn.src[i].const_like(0))
    else:
      root_grad_srcs.append(g.param_like(len(args) + len(ext_grad_args)))
      ext_grad_args.append(g)
  root_grad = UOp(Ops.TUPLE, src=tuple(root_grad_srcs))
  grads = compute_gradient(fxn, root_grad, set(params.values()))

  # for precompiled calls, pass forward outputs to backward so intermediates aren't recomputed
  fwd_outs: tuple[UOp, ...] = ()
  fwd_subs: dict[UOp, UOp] = {}
  if k.arg.precompile:
    fwd_slot_base = len(args) + len(ext_grad_args)
    for i, src in enumerate(fxn.src):
      fwd_subs[src] = src.param_like(fwd_slot_base + i)
      fwd_outs += (k.gettuple(i),)

  # collect gradient bodies per param, only for needed params
  grad_bodies: list[tuple[int, UOp]] = []
  for i in range(len(args)):
    if needed is not None and i not in needed: continue
    if (p:=params.get(i, None)) is not None and p in grads:
      grad_body = grads[p].substitute(fwd_subs) if fwd_subs else grads[p]
      assert not grad_body.op_in_backward_slice_with_self(Ops.BUFFER), "BUG: BUFFER in backward slice of grad"
      grad_bodies.append((i, grad_body))

  # create a single backward CALL with all grads as a TUPLE
  bwd_call = UOp.maketuple(*(gb for _, gb in grad_bodies)).call(
    *args, *ext_grad_args, *fwd_outs, name=(k.arg.name or "")+"_backward", precompile=k.arg.precompile_backward)

  ret: list[UOp|None] = [None]
  bwd_idx = 0
  for i in range(len(args)):
    if bwd_idx < len(grad_bodies) and grad_bodies[bwd_idx][0] == i:
      ret.append(bwd_call.gettuple(bwd_idx))
      bwd_idx += 1
    else:
      ret.append(None)
  return tuple(ret)

# ctx is grad_output
pm_gradient = PatternMatcher([
  (UPat(Ops.CAST, name="ret"), lambda ctx, ret: (ctx.cast(ret.src[0].dtype),)),
  (UPat(Ops.RECIPROCAL, name="ret"), lambda ctx, ret: (-ctx * ret * ret,)),
  (UPat(Ops.SIN, name="ret"), lambda ctx, ret: ((math.pi/2 - ret.src[0]).sin() * ctx,)),
  (UPat(Ops.LOG2, name="ret"), lambda ctx, ret: (ctx / (ret.src[0] * math.log(2)),)),
  (UPat(Ops.EXP2, name="ret"), lambda ctx, ret: (ret * ctx * math.log(2),)),
  (UPat(Ops.SQRT, name="ret"), lambda ctx, ret: (ctx / (ret*2),)),
  (UPat((Ops.CMPLT, Ops.CMPNE)), lambda: (None, None)),
  (UPat(Ops.ADD), lambda ctx: (ctx, ctx)),
  (UPat(Ops.POW, name="ret", src=(UPat.var("b"), UPat.var("e"))), lambda ctx, ret, b, e:
    (ctx * (b.eq(0)&e.eq(0)).where(e, e*b.pow(e-1)), ctx * b.eq(0).where((e<0).where(ret.const_like(-math.inf), 0), ret*b.log2()*math.log(2.0)))),
  (UPat(Ops.MAX, src=(UPat.var("x"), UPat.var("y"))), lambda ctx, x, y:
    ((x>y).where(ctx, (x.eq(y)).where(ctx * 0.5, 0)), (x<y).where(ctx, (x.eq(y)).where(ctx * 0.5, 0)))),
  (UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx)),
  (UPat(Ops.WHERE, name="ret"), lambda ctx, ret: (None, ret.src[0].where(ctx, ctx.const_like(0)), ret.src[0].where(ctx.const_like(0), ctx))),
  (UPat(Ops.REDUCE_AXIS, name="ret"), lambda ctx, ret: reduce_gradient(ctx, ret, ret.arg[0])),
  (UPat(Ops.CONTIGUOUS), lambda ctx: (ctx,)),
  (UPat(Ops.CONTIGUOUS_BACKWARD), lambda ctx: (ctx.contiguous(),)),
  (UPat(Ops.RESHAPE, name="ret"), lambda ctx, ret: (ctx.reshape(ret.src[0].shape), None)),
  (UPat(Ops.EXPAND, name="ret"), lambda ctx, ret: (ctx.r(Ops.ADD,tuple(i for i,(s,n) in enumerate(zip(ret.src[0].shape, ret.shape)) if s!=n)), None)),
  (UPat(Ops.PAD, name="ret"), lambda ctx, ret: (ctx.shrink(tuple([(p[0], s+p[0]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None)),
  (UPat(Ops.SHRINK, name="ret"), lambda ctx, ret: (ctx.pad(tuple([(p[0], s-p[1]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None)),
  (UPat(Ops.PERMUTE, name="ret"), lambda ctx, ret: (ctx.permute(argsort(ret.marg)),)),
  (UPat(Ops.FLIP, name="ret"), lambda ctx, ret: (ctx.flip([i for i,x in enumerate(ret.marg) if x]),)),
  (UPat(Ops.COPY, name="ret"), lambda ctx, ret: (ctx.copy_to_device(ret.src[0].device), None)),
  (UPat(Ops.MULTI, name="ret"), lambda ctx, ret: ctx.shard(ret.device, ret.axis).src),
  (UPat(Ops.TUPLE), lambda ctx: ctx.src),
  # NOTE: this is only correct when the KERNEL has a single output
  (UPat(Ops.AFTER), lambda ctx: (ctx, ctx)),
  # there's no gradient for bitcast
  (UPat(Ops.BITCAST), lambda: (None,)),
])

def _deepwalk(root:UOp, targets:set[UOp]) -> tuple[list[UOp], dict[UOp, bool]]:
  # compute the target path (top down)
  in_target_path: dict[UOp, bool] = {}
  for u in root.toposort(): in_target_path[u] = any(x in targets or in_target_path[x] for x in u.src)
  # don't flow through DETACH or anything not in target path
  return list(root.toposort(lambda node: node.op is not Ops.DETACH and in_target_path[node])), in_target_path

def compute_gradient(root:UOp, root_grad:UOp, targets:set[UOp]) -> dict[UOp, UOp]:
  walk, in_target_path = _deepwalk(root, targets)
  grads: dict[UOp, UOp] = {root: root_grad}
  for t0 in reversed(walk):
    if t0 not in grads: continue
    # GETTUPLE: accumulate gradient into a TUPLE UOp on the CALL, process when we hit the CALL
    if t0.op is Ops.GETTUPLE:
      k = t0.src[0]  # the CALL
      assert k.op is Ops.CALL and k.src[0].op is Ops.TUPLE
      n_outputs = len(k.src[0].src)
      prev: tuple[UOp, ...] = grads[k].src if k in grads else tuple(UOp(Ops.NOOP) for _ in range(n_outputs))
      grads[k] = UOp.maketuple(*(prev[i] + grads[t0] if i == t0.arg and prev[i].op is not Ops.NOOP else
                                 grads[t0] if i == t0.arg else prev[i] for i in range(n_outputs)))
      continue
    # CALL: pass needed param set so backward only computes required gradients
    if t0.op is Ops.CALL:
      needed = {i for i, arg in enumerate(t0.src[1:]) if arg in targets or in_target_path.get(arg, False)}
      lgrads:tuple[UOp|None, ...]|None = call_gradient(grads[t0], t0, needed)
    else:
      lgrads = cast(tuple[UOp|None, ...]|None, pm_gradient.rewrite(t0, ctx=grads[t0]))
    if lgrads is None: raise RuntimeError(f"failed to compute gradient for {t0.op}\n\nin {str(t0)[0:1000]}...")
    assert len(lgrads) == len(t0.src), f"got {len(lgrads)} gradient, expected {len(t0.src)}"
    for k,v in zip(t0.src, lgrads):
      if v is None: continue
      if k in grads: grads[k] = grads[k] + v
      else: grads[k] = v
      if len(forward_metadata:=all_metadata.get(t0, ())):
        backward_metadata = tuple(dataclasses.replace(x, backward=True) for x in forward_metadata)
        # we add the backward metadata to everything new in the graph
        for bw_uop in v.toposort(lambda x: x not in (t0, *t0.src, grads[t0])):
          all_metadata[bw_uop] = all_metadata.get(bw_uop, ())+backward_metadata
  return grads
