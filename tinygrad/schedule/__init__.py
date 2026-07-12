import time, inspect
from collections import Counter, deque
from dataclasses import replace
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, UOpMetaClass, track_rewrites, graph_rewrite, gate_kernel_sink, KernelInfo, ssimplify
from tinygrad.uop.spec import type_verify, spec_tensor
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, pluralize, SCACHE, BASEDIR, partition

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in {Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.MSELECT, Ops.MSTACK, Ops.BIND}: s = s.src[0]
  return s

# a buffer state is AFTER | BUFFER | PARAM. MSELECT/MSTACK join per-device states, BIND is not a buffer dependency
def _states(s: UOp) -> list[UOp]:
  s = _unwrap_src(s)
  if s.op in {Ops.MSELECT, Ops.MSTACK}: return [st for ss in s.src for st in _states(ss)]
  if s.op is Ops.BIND: return []
  assert s.op in {Ops.AFTER, Ops.BUFFER, Ops.PARAM}, f"input to kernel must resolve to a buffer state, not {s.op}"
  return [s]

def _split_after(after: UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
  kernels, remaining = partition(after.src[1:], lambda s: s.op in {Ops.CALL, Ops.END})
  deps, remaining = partition(remaining, lambda s: s.op is Ops.AFTER)
  if invalid := [s for s in remaining if s.op is not Ops.STORE]:
    raise AssertionError(f"AFTER source should be CALL, END, STORE, or AFTER, not {invalid[0].op}")
  return tuple(kernels), tuple(deps)

def _kernel_io(call:UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]|None:
  if call.op is not Ops.CALL or call.src[0].op is not Ops.SINK: return None
  out_slots = {x.src[0].src[0].arg.slot for x in call.src[0].toposort()
               if x.op is Ops.STORE and x.src[0].op is Ops.INDEX and x.src[0].src[0].op is Ops.PARAM}
  if len(out_slots) != 1 or min(out_slots) < 0 or max(out_slots) >= len(call.src)-1: return None
  return tuple(x for i,x in enumerate(call.src[1:]) if i in out_slots), tuple(x for i,x in enumerate(call.src[1:]) if i not in out_slots)

def _input_indices(call:UOp, inputs:tuple[UOp, ...]) -> set[tuple[UOp, tuple[UOp, ...]]]:
  return {(call.src[x.src[0].arg.slot+1], x.src[1:]) for x in call.src[0].toposort()
          if x.op is Ops.INDEX and x.src[0].op is Ops.PARAM and call.src[x.src[0].arg.slot+1] in inputs}

def _is_sum_sumsq(a:UOp, b:UOp) -> bool:
  av, bv = a.src[0], b.src[0]
  return (bv.op is Ops.MUL and bv.src == (av, av)) or (av.op is Ops.MUL and av.src == (bv, bv))

def _substitute_uops(ast:UOp, mapping:dict[UOp, UOp]) -> UOp:
  memo: dict[UOp, UOp] = {}
  def rec(x:UOp) -> UOp:
    if x in mapping: return mapping[x]
    if x not in memo:
      src = tuple(rec(s) for s in x.src)
      memo[x] = x if src == x.src else x.replace(src=src)
    return memo[x]
  return rec(ast)

def _remap_params(ast:UOp, remap:dict[int, int]) -> UOp:
  return _substitute_uops(ast, {x:x.replace(arg=replace(x.arg, slot=remap[x.arg.slot])) for x in ast.toposort()
                                if x.op is Ops.PARAM and x.arg.slot >= 0})

def _fuse_adjacent_reductions(linearized:list[UOp]) -> list[UOp]:
  ret: list[UOp] = []
  i = 0
  while i < len(linearized):
    if i+1 == len(linearized):
      ret.append(linearized[i])
      break
    a, b = linearized[i:i+2]
    aio, bio = _kernel_io(a), _kernel_io(b)
    ar = [x for x in a.src[0].toposort() if x.op is Ops.REDUCE] if aio is not None else []
    br = [x for x in b.src[0].toposort() if x.op is Ops.REDUCE] if bio is not None else []
    compatible = aio is not None and bio is not None and len(ar) == len(br) == 1 and ar[0].arg == br[0].arg and \
      ar[0].src[1:] == br[0].src[1:] and _is_sum_sumsq(ar[0], br[0]) and aio[1] == bio[1] and \
      _input_indices(a, aio[1]) == _input_indices(b, bio[1]) and aio[0] != bio[0] and \
      len(a.src[0].src) == len(b.src[0].src) == 1 and a.src[0].src[0].op is Ops.END and b.src[0].src[0].op is Ops.END and \
      a.src[0].src[0].src[1:] == b.src[0].src[0].src[1:]
    if not compatible:
      ret.append(a)
      i += 1
      continue
    assert aio is not None and bio is not None
    args = list(a.src[1:])
    if len(set(args)) != len(args) or any(x in args for x in bio[0]):
      ret.append(a)
      i += 1
      continue
    args.extend(bio[0])
    remap = {slot:args.index(x) for slot,x in enumerate(b.src[1:])}
    bast = _remap_params(b.src[0], remap)
    ret.append(a.src[0].replace(src=a.src[0].src+bast.src).call(*args))
    i += 2
  return ret

def _fuse_dependent_reductions(linearized:list[UOp]) -> list[UOp]:
  ret = list(linearized)
  i = 0
  while i < len(ret):
    a, aio = ret[i], _kernel_io(ret[i])
    if aio is None or len(a.src[0].src) != 1 or a.src[0].src[0].op is not Ops.END:
      i += 1
      continue
    ar = [x for x in a.src[0].toposort() if x.op is Ops.REDUCE]
    if len(ar) != 1 or ar[0].arg != (Ops.MAX, 0) or len(ar[0].src[1:]) != 2 or any(int(x.vmax)+1 != 2 for x in ar[0].src[1:]):
      i += 1
      continue
    aend, astore = a.src[0].src[0], a.src[0].src[0].src[0]
    if astore.op is not Ops.STORE:
      i += 1
      continue
    spatial = tuple(int(x.vmax)+1 for x in aend.src[-2:])
    if len(aend.src[1:]) != 4 or spatial[0] != spatial[1] or spatial[0] & (spatial[0]-1):
      i += 1
      continue
    for j in range(i+1, min(i+6, len(ret))):
      b, bio = ret[j], _kernel_io(ret[j])
      if bio is None or len(b.src[0].src) != 1 or b.src[0].src[0].op is not Ops.END: continue
      br = [x for x in b.src[0].toposort() if x.op is Ops.REDUCE]
      if len(br) != 1 or br[0].arg != (Ops.ADD, 0) or ar[0].src[1:] != br[0].src[1:]: continue
      bend = b.src[0].src[0]
      if aend.src[1:] != bend.src[1:] or aio[0][0] not in bio[1] or set(bio[1]) != set(aio[1]+aio[0]): continue
      args = list(a.src[1:])
      if len(set(args)) != len(args) or bio[0][0] in args: continue
      args.extend(bio[0])
      pout_slot = tuple(b.src[1:]).index(aio[0][0])
      pout_idxs = [x for x in b.src[0].toposort() if x.op is Ops.INDEX and x.src[0].op is Ops.PARAM and x.src[0].arg.slot == pout_slot]
      if not pout_idxs or any(x.src[1:] != astore.src[0].src[1:] for x in pout_idxs): continue
      mapping = {x:astore.src[1] for x in pout_idxs}
      for x in b.src[0].toposort():
        if x.op is Ops.PARAM and x.arg.slot >= 0 and x.arg.slot != pout_slot:
          mapping[x] = x.replace(arg=replace(x.arg, slot=args.index(b.src[x.arg.slot+1])))
      bend = _substitute_uops(bend, mapping)
      ret[i] = a.src[0].replace(src=(aend, bend)).call(*args)
      ret.pop(j)
      break
    i += 1
  return ret

def _fuse_direct_conv_bwd_activation(linearized:list[UOp]) -> list[UOp]:
  from tinygrad.codegen.opt.gemm import DirectConvBwdActivationMatch, _match_gemm
  cases = {
    (288,64): (32, False,
      {Ops.CONST:22, Ops.MUL:18, Ops.ADD:18, Ops.AND:8, Ops.RANGE:6, Ops.CMPLT:4, Ops.PARAM:3, Ops.INDEX:3,
       Ops.CAST:3, Ops.FLOORMOD:2, Ops.FLOORDIV:2, Ops.WHERE:2, Ops.EXP2:1, Ops.RECIPROCAL:1, Ops.REDUCE:1,
       Ops.STORE:1, Ops.END:1, Ops.SINK:1}),
    (576,64): (16, True,
      {Ops.CONST:23, Ops.ADD:19, Ops.MUL:18, Ops.AND:8, Ops.RANGE:6, Ops.PARAM:4, Ops.INDEX:4, Ops.CAST:4,
       Ops.CMPLT:4, Ops.FLOORMOD:2, Ops.FLOORDIV:2, Ops.WHERE:2, Ops.EXP2:1, Ops.RECIPROCAL:1, Ops.REDUCE:1,
       Ops.STORE:1, Ops.END:1, Ops.SINK:1}),
    (2304,256): (8, True,
      {Ops.CONST:23, Ops.ADD:19, Ops.MUL:18, Ops.AND:8, Ops.RANGE:6, Ops.PARAM:4, Ops.INDEX:4, Ops.CAST:4,
       Ops.CMPLT:4, Ops.FLOORMOD:2, Ops.FLOORDIV:2, Ops.WHERE:2, Ops.EXP2:1, Ops.RECIPROCAL:1, Ops.REDUCE:1,
       Ops.STORE:1, Ops.END:1, Ops.SINK:1}),
  }
  ret = list(linearized)
  i = 0
  while i < len(ret):
    call = ret[i]
    if call.op is not Ops.CALL or call.src[0].op is not Ops.SINK or \
       (g:=_match_gemm(call.src[0], "AMD", "gfx11")) is None or \
       (g.n, g.k) not in cases or g.old is not None or g.a_kxm or not g.b_kxn:
      i += 1
      continue
    patch_grad = call.src[g.c.arg.slot+1]
    consumers = [j for j,x in enumerate(ret) if j > i and patch_grad in x.src[1:]]
    if len(consumers) != 1 or consumers[0] > i+3:
      i += 1
      continue
    j, consumer = consumers[0], ret[consumers[0]]
    io = _kernel_io(consumer)
    params = sorted((x.arg.slot, x.dtype, x.max_numel()) for x in consumer.src[0].toposort() if x.op is Ops.PARAM)
    op_counts = Counter(x.op for x in consumer.src[0].toposort())
    spatial, residual, expected_ops = cases[(g.n,g.k)]
    if g.m % (spatial*spatial):
      i += 1
      continue
    batch, output_size = g.m//(spatial*spatial), g.m*(g.n//9)
    if batch not in (1024,1280,1536):
      i += 1
      continue
    if batch == 1024 and (g.n,g.k) == (288,64): expected_ops = expected_ops | {Ops.CONST:expected_ops[Ops.CONST]-1}
    expected_params = ([(0,dtypes.float,output_size),(1,dtypes.float,output_size),(2,dtypes.half,output_size),
                        (3,dtypes.half,g.m*g.n)] if residual else
                       [(0,dtypes.float,output_size),(1,dtypes.half,output_size),(2,dtypes.half,g.m*g.n)])
    expected_srcs = 5 if residual else 4
    if io is None or len(io[0]) != 1 or len(io[1]) != expected_srcs-2 or len(consumer.src) != expected_srcs or \
       consumer.src[-1] is not patch_grad or params != expected_params or op_counts != Counter(expected_ops):
      i += 1
      continue
    reduces = [x for x in consumer.src[0].toposort() if x.op is Ops.REDUCE]
    if len(reduces) != 1 or reduces[0].arg != (Ops.ADD, 0) or tuple(int(x.vmax)+1 for x in reduces[0].src[1:]) != (4,4):
      i += 1
      continue
    info = DirectConvBwdActivationMatch(g.m, g.n//9, g.k, spatial, residual)
    ast = call.src[0].replace(tag=info, arg=replace(call.src[0].arg, name="direct_conv_bwd_activation"))
    extras = (consumer.src[2], consumer.src[3]) if residual else (consumer.src[2],)
    ret[j] = ast.call(io[0][0], *extras, call.src[g.a.arg.slot+1], call.src[g.b.arg.slot+1])
    ret.pop(i)
  return ret

def _fuse_gemm_output_transpose(linearized:list[UOp]) -> list[UOp]:
  from tinygrad.codegen.opt.gemm import GemmOutputNCHW, _match_gemm
  from tinygrad.device import Device
  cases = {
    (64,288): (32, 7),
    (256,576): (16, 6),
    (512,2304): (8, 7),
  }
  ret = list(linearized)
  i = 0
  while i < len(ret):
    call = ret[i]
    if call.op is not Ops.CALL or call.src[0].op is not Ops.SINK or not isinstance(call.device, str):
      i += 1
      continue
    target = Device[call.device].renderer.target
    if target.device != "AMD" or not target.arch.startswith("gfx11") or \
       (g:=_match_gemm(call.src[0], target.device, target.arch)) is None or (g.n,g.k) not in cases or \
       g.old is not None or g.a_kxm or g.b_kxn:
      i += 1
      continue
    gemm_out = call.src[g.c.arg.slot+1]
    consumers = [j for j,x in enumerate(ret) if j > i and gemm_out in x.src[1:]]
    if consumers != [i+1]:
      i += 1
      continue
    j, consumer = consumers[0], ret[consumers[0]]
    io, spatial_const = _kernel_io(consumer), cases[(g.n,g.k)]
    spatial, const_count = spatial_const
    op_counts = Counter(x.op for x in consumer.src[0].toposort())
    expected_ops = {Ops.CONST:const_count, Ops.ADD:6, Ops.MUL:5, Ops.RANGE:4, Ops.PARAM:2, Ops.INDEX:2,
                    Ops.STORE:1, Ops.END:1, Ops.SINK:1}
    if io is None or len(io[0]) != 1 or len(io[1]) != 1 or len(consumer.src) != 3 or io[1][0] is not gemm_out or \
       len(consumer.src[0].src) != 1 or op_counts != Counter(expected_ops):
      i += 1
      continue
    end = consumer.src[0].src[0]
    end_shape = tuple(int(x.vmax)+1 for x in end.src[1:])
    if end.op is not Ops.END or len(end_shape) != 4 or end_shape[0] not in (1024,1280,1536) or end_shape != (end_shape[0],g.n,spatial,spatial):
      i += 1
      continue
    store, (batch, channel, y, x) = end.src[0], end.src[1:]
    if store.op is not Ops.STORE or store.src[0].op is not Ops.INDEX or store.src[1].op is not Ops.INDEX:
      i += 1
      continue
    out_idx, in_idx = (ssimplify(u.src[1].get_idx()) for u in (store.src[0], store.src[1]))
    spatial_size = spatial**2
    if out_idx is not ssimplify(batch*(g.n*spatial_size)+channel*spatial_size+y*spatial+x) or \
       in_idx is not ssimplify(batch*(spatial_size*g.n)+y*(spatial*g.n)+x*g.n+channel):
      i += 1
      continue
    args = list(call.src[1:])
    args[g.c.arg.slot] = io[0][0]
    ret[i] = call.src[0].replace(tag=GemmOutputNCHW(spatial)).call(*args)
    ret.pop(j)
    i += 1
  return ret

def _fuse_channel_reductions(linearized:list[UOp]) -> list[UOp]:
  from tinygrad.codegen.opt.reduce import ActivationVarGradElementwiseSumDMeanMatch, DualActivationReduceElementwiseMatch
  from tinygrad.codegen.opt.reduce import DualBNGradDMean512Match, DualChannelReduceMatch, DualMoments512Match
  from tinygrad.codegen.opt.reduce import _match_activation_elementwise, _match_activation_elementwise_512, _match_activation_sum
  from tinygrad.codegen.opt.reduce import _match_activation_sum_512
  from tinygrad.codegen.opt.reduce import _match_activation_dmean_512, _match_activation_var_grad, _match_channel_reduce
  from tinygrad.codegen.opt.reduce import _match_moment_mean_512, _match_moment_var_512
  from tinygrad.codegen.opt.reduce import _match_bn_dmean_512, _match_bn_sum_512, _match_bn_var_512
  from tinygrad.device import Device
  ret = list(linearized)
  i = 0
  while i+1 < len(ret):
    a, b = ret[i:i+2]
    if a.op is not Ops.CALL or b.op is not Ops.CALL or a.src[0].op is not Ops.SINK or b.src[0].op is not Ops.SINK or \
       not isinstance(a.device,str) or a.device != b.device:
      i += 1
      continue
    target = Device[a.device].renderer.target
    mean_batch, var_batch = _match_moment_mean_512(a.src[0],target.device,target.arch), _match_moment_var_512(b.src[0],target.device,target.arch)
    if mean_batch is None or mean_batch != var_batch or \
       a.src[1] is not b.src[3] or a.src[2] is not b.src[2]:
      i += 1
      continue
    ret[i] = a.src[0].replace(tag=DualMoments512Match(mean_batch)).call(a.src[1],b.src[1],a.src[2])
    ret.pop(i+1)
    i += 1
  i = 0
  while i+1 < len(ret):
    a, b = ret[i:i+2]
    if a.op is not Ops.CALL or b.op is not Ops.CALL or a.src[0].op is not Ops.SINK or b.src[0].op is not Ops.SINK or \
       not isinstance(a.device,str) or a.device != b.device:
      i += 1
      continue
    target = Device[a.device].renderer.target
    var_batch, sum_batch = _match_bn_var_512(a.src[0],target.device,target.arch), _match_bn_sum_512(b.src[0],target.device,target.arch)
    if var_batch is None or var_batch != sum_batch or \
       a.src[6] is not b.src[2]:
      i += 1
      continue
    dmean_idx = next((j for j in range(i+2,min(i+6,len(ret))) if ret[j].op is Ops.CALL and ret[j].src[0].op is Ops.SINK and
      _match_bn_dmean_512(ret[j].src[0],target.device,target.arch) == var_batch and ret[j].src[2] is a.src[1] and
      ret[j].src[3] is a.src[4] and ret[j].src[4] is a.src[5] and ret[j].src[5] is a.src[2] and ret[j].src[6] is a.src[6]),None)
    if dmean_idx is None:
      i += 1
      continue
    d = ret[dmean_idx]
    ret[i] = a.src[0].replace(tag=DualBNGradDMean512Match(var_batch)).call(
      a.src[1],b.src[1],d.src[1],a.src[2],a.src[3],a.src[4],a.src[5],a.src[6],b.src[3])
    ret.pop(dmean_idx)
    ret.pop(i+1)
    i += 1
  i = 0
  while i+1 < len(ret):
    a, b = ret[i:i+2]
    if a.op is not Ops.CALL or b.op is not Ops.CALL or a.src[0].op is not Ops.SINK or b.src[0].op is not Ops.SINK or \
       not isinstance(a.device,str) or a.device != b.device:
      i += 1
      continue
    target = Device[a.device].renderer.target
    var_batch, elem_batch = _match_activation_var_grad(a.src[0],target.device,target.arch), \
                            _match_activation_elementwise_512(b.src[0],target.device,target.arch)
    if var_batch is None or var_batch != elem_batch or \
       a.src[5] is not b.src[2] or a.src[2] is not b.src[3] or a.src[6] is not b.src[4] or a.src[7] is not b.src[5]:
      i += 1
      continue
    c = ret[i+2] if i+2 < len(ret) else None
    fuse_sum = c is not None and c.op is Ops.CALL and c.src[0].op is Ops.SINK and \
      _match_activation_sum_512(c.src[0],target.device,target.arch) == var_batch and a.src[6] is c.src[2] and a.src[7] is c.src[3]
    if not fuse_sum:
      i += 1
      continue
    assert c is not None
    dmean_idx = next((j for j in range(i+3,min(i+7,len(ret))) if ret[j].op is Ops.CALL and ret[j].src[0].op is Ops.SINK and
      _match_activation_dmean_512(ret[j].src[0],target.device,target.arch) == var_batch and ret[j].src[2] is a.src[1] and
      ret[j].src[3] is a.src[4] and ret[j].src[4] is b.src[1]),None)
    if dmean_idx is None:
      i += 1
      continue
    d = ret[dmean_idx]
    ret[i] = a.src[0].replace(tag=ActivationVarGradElementwiseSumDMeanMatch(var_batch)).call(
      a.src[1],b.src[1],c.src[1],d.src[1],a.src[2],a.src[3],a.src[4],a.src[5],a.src[6],a.src[7],c.src[4])
    ret.pop(dmean_idx)
    ret.pop(i+2)
    ret.pop(i+1)
    i += 1
  i = 0
  while i < len(ret):
    a = ret[i]
    if a.op is not Ops.CALL or a.src[0].op is not Ops.SINK or not isinstance(a.device,str):
      i += 1
      continue
    target = Device[a.device].renderer.target
    am = _match_channel_reduce(a.src[0],target.device,target.arch)
    if am is None or am.kind != "activation":
      i += 1
      continue
    for j in range(i+1,min(i+4,len(ret))):
      b = ret[j]
      sm = _match_activation_sum(b.src[0],target.device,target.arch) if b.op is Ops.CALL and b.src[0].op is Ops.SINK else None
      if sm != (am.channels,am.spatial,am.groups) or a.src[5] is not b.src[2] or a.src[6] is not b.src[3]: continue
      bout = b.src[1]
      if any(bout in x.src[1:] for x in ret[i+1:j]): continue
      elem_idx = next((k for k in range(i+1,min(i+5,len(ret))) if k != j and ret[k].op is Ops.CALL and ret[k].src[0].op is Ops.SINK and
        _match_activation_elementwise(ret[k].src[0],target.device,target.arch) == (am.channels,am.spatial,am.groups) and
        a.src[4] is ret[k].src[2] and a.src[5] is ret[k].src[4] and a.src[6] is ret[k].src[5]),None)
      if elem_idx is None: continue
      elem = ret[elem_idx]
      ret[i] = a.src[0].replace(tag=DualActivationReduceElementwiseMatch(am.groups,am.channels,am.spatial)).call(
        a.src[1],bout,elem.src[1],a.src[2],a.src[3],a.src[4],elem.src[3],a.src[5],a.src[6])
      for k in sorted((j,elem_idx),reverse=True): ret.pop(k)
      break
    i += 1
  i = 0
  while i+1 < len(ret):
    a, b = ret[i:i+2]
    if a.op is not Ops.CALL or b.op is not Ops.CALL or a.src[0].op is not Ops.SINK or b.src[0].op is not Ops.SINK or \
       not isinstance(a.device, str) or a.device != b.device:
      i += 1
      continue
    target = Device[a.device].renderer.target
    am, bm = _match_channel_reduce(a.src[0], target.device, target.arch), _match_channel_reduce(b.src[0], target.device, target.arch)
    if am is None or bm is None or am.kind != "centered" or bm.kind != "scale" or \
       (am.groups,am.channels,am.spatial) != (bm.groups,bm.channels,bm.spatial) or a.src[5] is not b.src[4]:
      i += 1
      continue
    ret[i] = a.src[0].replace(tag=DualChannelReduceMatch(am.groups, am.channels, am.spatial)).call(
      a.src[1], b.src[1], a.src[2], a.src[3], a.src[4], b.src[2], b.src[3], a.src[5])
    ret.pop(i+1)
    i += 1
  return ret

def create_schedule(sched_sink:UOp) -> UOp:
  with cpu_profile(TracingKey("toposort sched_sink")):
    # build kernel dependency graph: edges from producer kernel to consumer kernels
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}
    writes: dict[UOp, list[tuple[UOp, UOp, tuple[UOp, ...]]]] = {}  # buffer -> (AFTER, prior state, new kernels)
    reads: list[tuple[UOp, UOp, UOp]] = []  # (reader AFTER, reader kernel, buffer state read)
    for u in sched_sink.toposort(gate_kernel_sink):
      if u.op is not Ops.AFTER: continue
      kernels, after_deps = _split_after(u)
      prev_state = _unwrap_src(u.src[0])
      prev_kernels = set(_split_after(prev_state)[0]) if prev_state.op is Ops.AFTER else set()
      writes.setdefault(u.buf_uop, []).append((u, prev_state, tuple(k for k in kernels if k not in prev_kernels)))
      for k in kernels:
        in_degree.setdefault(k, 0)
        if k.op is Ops.END: assert k.src[0].op is Ops.CALL, f"END src[0] should be KERNEL, not {k.src[0].op}"
        kernel_deps = k.src[0].src[1:] if k.op is Ops.END else k.src[1:]
        read_states = [st for s in kernel_deps for st in _states(s)]
        reads += [(u, k, st) for st in read_states]
        # RAW deps: a kernel runs after the kernels that produced the states it reads or joins
        for st in read_states + [st for s in after_deps for st in _states(s)]:
          if st.op is Ops.AFTER:
            for t in _split_after(st)[0]:
              children.setdefault(t, []).append(k)
              in_degree[k] += 1
    # WAR deps: a kernel reading buffer state S must run before another write that supersedes S. an AFTER only
    # supersedes its immediate prior state; join members already present in that prior state are ordering deps, not writes
    for u, k, s in reads:
      for a, prev_state, write_kernels in writes.get(s.buf_uop, []):
        if a is u or prev_state is not s: continue
        for t in write_kernels:
          if t is not k and t not in k.backward_slice:
            children.setdefault(k, []).append(t)
            in_degree[t] += 1

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque(k for k,v in in_degree.items() if v == 0)
    linearized: list[UOp] = []
    while len(queue):
      rk = queue.popleft()
      if rk.op is Ops.LINEAR:
        linearized.extend(rk.src)
      else:
        k = rk.src[0] if rk.op is Ops.END else rk
        assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
        buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:] if s.op is not Ops.BIND)
        linearized.append(k.src[0].call(*buf_uops))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)
    if any(in_degree.values()): raise RuntimeError("cycle detected in assign graph")
  return UOp(Ops.LINEAR, src=tuple(_fuse_gemm_output_transpose(_fuse_channel_reductions(
    _fuse_direct_conv_bwd_activation(_fuse_dependent_reductions(_fuse_adjacent_reductions(linearized)))))))

from tinygrad.schedule.memory import memory_plan_rewrite
from tinygrad.engine.realize import capturing, pm_flatten_linear
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.helpers import CAPTURING
from tinygrad.uop.ops import PatternMatcher, UPat, ParamArg
from tinygrad.dtype import AddrSpace

def create_new_buffer(ctx:tuple[dict[UOp, UOp], tuple[UOp, ...]], b:UOp):
  if (ret:=ctx[0].get(b, None)) is None: ctx[0][b] = ret = UOp.new_buffer(b.device, b.max_numel(), b.dtype)
  return ret

pm_post_sched_cache = PatternMatcher([
  # only resolve buffer PARAMs (slot>=0); ALU/shape vars use slot=-1 and must not be swapped for call args
  (UPat(Ops.PARAM, name="x"), lambda ctx,x: ctx[1][x.arg.slot] if x.arg.slot >= 0 else None),
  # create new BUFFERs
  (UPat(Ops.BUFFER, src=(UPat(),), name="b"), lambda ctx,b:
   create_new_buffer(ctx, b) if isinstance(b.arg, ParamArg) and b.addrspace is AddrSpace.GLOBAL else None),
])

pm_resolve_linear_call = PatternMatcher([
  # call LINEAR is resolved here
  (UPat(Ops.CALL, src=(UPat(Ops.LINEAR),), name="linear_call", allow_any_len=True), lambda linear_call:
   graph_rewrite(linear_call.src[0], pm_post_sched_cache, ctx=({}, linear_call.src[1:]), walk=True, name="params to buffers")),
])+pm_flatten_linear

schedule_cache: dict[bytes, UOp] = {}
# ctx is just for DEBUG on inner
def lower_sink_to_linear(function:UOp) -> UOp|None:
  st = time.perf_counter()
  if isinstance(function.arg, KernelInfo): return None
  cache_key = function.key if SCACHE else b""
  if not SCACHE or (sc_ret:=schedule_cache.get(cache_key, None)) is None:
    if SPEC: type_verify(function, spec_tensor)
    # support recursive CALLs
    linear = create_schedule(get_kernel_graph(function))
    if SCACHE: schedule_cache[cache_key] = linear
  else:
    # schedule cache hit
    linear = sc_ret
  if (DEBUG >= 1 and len(linear.src) > 1) or DEBUG >= 3:
    for frm in inspect.stack():
      if frm.filename == "<string>": continue
      if frm.filename.startswith(str(BASEDIR / "apps")): break
      if not frm.filename.startswith(str(BASEDIR)) and not frm.filename.endswith("/contextlib.py"): break
    else:
      frm = None
    print(f"scheduled {len(linear.src):5d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {' cache hit' if SCACHE and sc_ret is not None else 'CACHE MISS'} {cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache):7d} uops in cache"+("" if frm is None else f" | {frm.filename}:{frm.lineno}"))
  return linear

pm_schedule = PatternMatcher([
  (UPat(Ops.SINK, name="function"), lower_sink_to_linear),
])

@track_rewrites(lambda _,ret: f"Schedule {pluralize('Kernel', len(ret[0].src))}")
def create_linear_with_vars(big_sink:UOp) -> tuple[UOp, dict[str, int]]:
  # big_sink srcs are all the Tensors
  linear_call = graph_rewrite(big_sink, pm_schedule, name="schedule to linear", enter_calls=True)

  # this recursively resolves the linear_call and allocates buffers
  linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call")

  # vars used in the schedule
  used_vars = set().union(*[{v.expr for v in si.src[0].variables()} for si in linear.src])
  # get var_vals
  var_vals: dict[str, int] = {}
  for b in big_sink.src[1:]:
    if b.op is Ops.BIND:
      nm = b.src[0].expr
      if nm not in used_vars: continue
      val = b.src[1].arg
      if var_vals.get(nm, val) != val: raise RuntimeError(f"bind mismatch on {nm}, {var_vals[nm]} != {val}")
      var_vals[nm] = val

  # jit captures this schedule, no need to execute.
  if len(capturing) and CAPTURING:
    capturing[0].add_linear(linear, var_vals)
    return UOp(Ops.LINEAR, src=()), var_vals

  held_bufs = ({b for b in linear_call.src[1:] if b.op is Ops.BUFFER} if linear_call.op is Ops.CALL else set())
  return memory_plan_rewrite(linear, held_bufs), var_vals
