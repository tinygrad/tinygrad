import itertools
from tinygrad.dtype import dtypes, to_dtype
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp
from tinygrad.uop.ops import graph_rewrite, rewrite_group, shape_to_shape_arg, ParamArg, identity_element
from tinygrad.uop.movement import mop_cleanup
from tinygrad.helpers import prod, getenv, all_int, DEBUG, SPLIT_REDUCEOP, OPENPILOT_HACKS, FLOAT16, argsort
from tinygrad.schedule.indexing import apply_movement_op
from tinygrad.schedule.allreduce import create_allreduce_function, is_allreduce_linear_output, _allreduce_view
from tinygrad.schedule.multi import multi_pm

def walk_mop(u:UOp):
  if u.op is Ops.SHRINK and u.tag == ("allreduce",): return u
  if u.op in GroupOp.Movement or u.op in {Ops.INDEX, Ops.UNSHARD}: return walk_mop(u.src[0])
  return u

def found_after(ctx:dict[UOp, UOp], after:UOp, src:UOp):
  if (x:=src).op is Ops.CAST and x.dtype == dtypes.half and FLOAT16: x, after = x.src[0], after.cast(dtypes.float)
  while True:
    if x.op is Ops.PERMUTE: x, after = x.src[0], after.permute(argsort(x.marg))
    elif x.op is Ops.RESHAPE: x, after = x.src[0], after.reshape(x.src[0].shape)
    elif x.op is Ops.WHERE and x.src[2].base.is_invalid and x.src[1].op is Ops.PAD:
      x, after = x.src[1].src[0], after.shrink(tuple((o, s+o) for (o,_),s in zip(x.src[1].marg, x.src[1].src[0].shape)))
    else: break
  ctx[x] = after

# *** fold moved AFTERs (hack for openpilot) ***
pm_fold_moved_after = PatternMatcher([
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE, src=(UPat(), UPat((*GroupOp.Movement,Ops.CAST,Ops.WHERE), name="src")))), name="after"), found_after),
  # replace ALU sources with AFTER versions found above
  (UPat(GroupOp.ALU, name="alu"), lambda ctx,alu: alu.replace(src=new_src) if (new_src:=tuple(ctx.get(s, s) for s in alu.src)) != alu.src else None),
])

# movement op on INDEX as a PatternMatcher
def _mop_index(r:UOp, idx:UOp):
  # Tagged all-reduce SHRINKs are physical runtime views, not logical movement ops.
  if r.op is Ops.SHRINK and r.tag == ("allreduce",): return None
  idxs = idx.src[1:]
  if len(idxs) == len(r.shape):
    return r.src[0].index(*apply_movement_op(r.op, r.src[0].shape, r.marg, idxs), arg=idx.arg)
  if r.op is Ops.RESHAPE:
    src_prefix = len(r.src[0].shape) - len(r.shape[len(idxs):])
    if src_prefix >= 0 and r.src[0].shape[src_prefix:] == r.shape[len(idxs):]:
      if src_prefix == 0: return r.src[0] if r.src[0].dtype == idx.dtype else None
      ret = r.src[0].index(*apply_movement_op(r.op, r.src[0].shape[:src_prefix], r.shape[:len(idxs)], idxs), arg=idx.arg)
      return ret if ret.shape == idx.shape else None

def move_mop_after(r:UOp, a:UOp) -> UOp|None:
  if r.op is Ops.SHRINK and r.tag == ("allreduce",): return None
  return UOp(r.op, src=(a.replace(src=(r.src[0],)+a.src[1:]),)+r.src[1:], arg=r.arg)

def move_mop_before_end(r:UOp, a:UOp) -> UOp|None:
  if r.op is Ops.SHRINK and r.tag == ("allreduce",): return None
  return a.replace(src=(r.src[0],)+a.src[1:])

pm_mops = PatternMatcher([
  # handle movement ops on INDEX
  (UPat(GroupOp.Movement, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"), _mop_index),
  # move movement ops and INDEX after AFTER
  (UPat(GroupOp.Movement|{Ops.INDEX}, name="r").after(name="a", allow_any_len=True), move_mop_after),
  (UPat(GroupOp.Movement, name="r").end(name="a", allow_any_len=True), move_mop_before_end),
])

# *****************
# 0. do some cleanup rewrites, mostly copied from the old stuff

def fix_store_hazard(target:UOp, src:UOp):
  if (base:=target.base) not in src.toposort(enter_calls=False): return None
  # PERMUTE and FLIP reorder indices, SHRINK can have overlapping regions when dest is also shrunk
  unsafe = {Ops.PERMUTE, Ops.FLIP} | ({Ops.SHRINK} if target.op_in_backward_slice_with_self(Ops.SHRINK) else set())
  reaches_base: dict[UOp, bool] = {}
  for s in src.toposort(gate=lambda s: s.op is not Ops.CONTIGUOUS):
    reaches_base[s] = s is base or any(reaches_base.get(c) for c in s.src)
    if reaches_base[s] and s.op in unsafe and not (s is target and s.op is Ops.SHRINK): return target.store(src.contiguous())

def split_reduceop(reduce:UOp, x:UOp):
  if prod(reduce.shape) == 0: return None
  if not SPLIT_REDUCEOP or not all_int(x.shape) or (prod(x.shape)//prod(reduce.shape))<getenv("REDUCEOP_SPLIT_THRESHOLD", 32768): return None
  # if there are few globals, make some reduces into globals by splitting into two kernels
  # cap output buffer to 2**22: heuristic number of global outputs to achieve max occupancy with enough locals+upcasts for gemm
  #   ~2**10 should be enough if GROUP is used
  # 256 split maximum should be "negligible reduce" for low prod(reduce.shape), 8 split minimum.
  # split is moved to the end to provide maximum locality for the second phase reduce.

  # get expanded by rangeifying the UOp x
  indexed = x.index(*[UOp.range(s, i) if resolve(s>1) else 0 for i,s in enumerate(x.shape)])
  range_nums = [y.arg[0] for y in indexed.substitute({x.base:UOp(Ops.NOOP)}, extra_pm=pm_mops).ranges]
  is_expanded = [i not in range_nums for i in range(len(x.shape))]

  if not (split_candidates:=[(i,d) for i in range(reduce.arg[1])
                             for d in range(min(256,2**getenv("REDUCEOP_SPLIT_SIZE",22)//prod(reduce.shape)),8-1,-1)
                             if x.shape[i]%d==0 and not is_expanded[i]]): return None
  dim_to_split, divisor = split_candidates[0]
  splitted_shape = x.shape[:dim_to_split]+(divisor,)+(x.shape[dim_to_split]//divisor,)+x.shape[dim_to_split+1:]
  splitted = x.reshape(splitted_shape).permute(tuple([d for d in range(len(splitted_shape)) if d!=dim_to_split]+[dim_to_split]))
  if DEBUG >= 3: print(f"split {divisor}: {x.shape} -> {splitted.shape} -> {reduce.shape}")
  # reduce original axes, then split
  return splitted._rop(reduce.arg[0], tuple(range(reduce.arg[1]))).contiguous()._rop(reduce.arg[0], (len(reduce.shape),)).reshape(reduce.shape)

pm_gather_params = PatternMatcher([ (UPat(Ops.PARAM, name="p"), lambda ctx, p: ctx.append(p) if p.arg.slot >= 0 else None), ])
def resolve_function(c:UOp, allow_param_mismatch=True) -> UOp|None:
  if c.arg.precompile: return None
  params: list[UOp] = []
  graph_rewrite(c.src[0], pm_gather_params, bottom_up=True, ctx=params, name="gather params")
  params = sorted(params, key=lambda x: x.arg.slot)
  args = c.src[1:]

  # NOTE: this isn't really needed. it's okay if there's unused args in the function
  if not allow_param_mismatch:
    if [x.arg.slot for x in params] != list(range(len(params))): raise RuntimeError(f"params not in order: {[x.arg.slot for x in params]}")
    if len(params) != len(args): raise TypeError(f"expected {len(params)} args, got {len(args)}")

  dict_map = {x:args[x.arg.slot] for x in params}
  for i, (p, a) in enumerate(dict_map.items()):
    if p.axis != a.axis: raise TypeError(f"arg {i} axis mismatch: expected {p.axis}, got {a.axis}")
    if p.max_shape != a.max_shape: raise TypeError(f"arg {i} shape mismatch: expected {p.shape}, got {a.shape}")
    if p.dtype != a.dtype: raise TypeError(f"arg {i} dtype mismatch: expected {p.dtype}, got {a.dtype}")
  return c.src[0].substitute(dict_map, walk=True)

# shape-changing bitcast
def expand_bitcast(bc:UOp) -> UOp|None:
  x = bc.src[0]
  if (ns:=bc.dtype.itemsize) == (os:=x.dtype.itemsize) or (isinstance(x.device, str) and x.device.startswith(("DISK", "TINYFS"))): return None
  new_uint, tmp = to_dtype(f"uint{8*ns}"), x.bitcast(to_dtype(f"uint{8*os}"))
  if ns > os:
    tmp = tmp.reshape(x.shape[:-1] + (x.shape[-1]//(rate := ns//os), rate))
    parts = [tmp.shrink((None,)*(len(tmp.shape)-1) + ((i, i+1),)).cast(new_uint)<<8*i*os for i in range(rate)]
    return parts[0].usum(*parts[1:]).squeeze(-1).bitcast(bc.dtype)
  parts = [tmp>>8*i*ns for i in range(os//ns)]
  return parts[0].stack(*parts[1:], dim=-1).flatten(-2).cast(new_uint).bitcast(bc.dtype)

def forward_assembled_store(output:UOp, target:UOp, src:UOp) -> UOp|None:
  """Retarget a complete set of disjoint slice writes to an already allocated output buffer."""
  while target.op is Ops.RESHAPE: target = target.src[0]
  while src.op in {Ops.RESHAPE, Ops.CONTIGUOUS, Ops.CAST}: src = src.src[0]
  if target.dtype != src.dtype or target.numel() != src.numel() or target.device != src.device: return None
  # The destination may be a contiguous view into a larger allocation. This is the form produced by slice-wise
  # overwrite of packed gradients. Preserve `output` as the dependency state, but retarget the assembled producer
  # to the view itself. Non-contiguous or unrelated targets still take the ordinary materialize-and-copy path.
  if target is not output:
    if target.base is not output.base or target.contiguous_view_offset() is None: return None
    destination = target
  else: destination = output

  # A replicated MSTACK often consists of COPYs of one freshly produced buffer. Produce directly into shard zero,
  # then transfer from that stable shard into the other output buffers. This removes both the producer temporary and
  # the final identity assembly kernels without duplicating the producer computation.
  if src.op is Ops.MSTACK and isinstance(output.device, tuple) and len(src.src) == len(output.device) \
     and all(s.op is Ops.COPY and s.shape == output.shape and s.dtype == output.dtype for s in src.src):
    origins = [s.src[0] for s in src.src]
    # Host-backed replication is an input transfer, not an assembled device-side producer to redirect.
    if all(s is origins[0] for s in origins) and origins[0].op is Ops.AFTER and \
       not any(x.device == "PYTHON" for x in origins[0].toposort() if isinstance(x.device, str)):
      origin, base = origins[0], origins[0].src[0].base
      if all(d.op is Ops.STORE and d.src[0].base is base and base not in d.src[1].toposort(enter_calls=False) for d in origin.src[1:]):
        targets = [_allreduce_view(destination.mselect(i).buf_uop, 0, destination.numel()) for i in range(len(src.src))]
        produced = targets[0].after(*(d.substitute({base:targets[0]}) for d in origin.src[1:]))
        states = [produced] + [t.after(t.store(produced.copy_to_device(s.device))) for t,s in zip(targets[1:], src.src[1:])]
        return output.after(*states)
  if src.op is not Ops.AFTER or src.src[0].base.op not in {Ops.BUFFER, Ops.PARAM}: return None
  if not any(s.op is Ops.AFTER and s.src[0].op is Ops.SHRINK and s.src[0].tag == ("allreduce",) for s in src.src[1:]): return None
  return output.after(*(s.substitute({src.src[0].base:destination}) for s in src.src[1:]))

def forward_assembled_accumulate(output:UOp, target:UOp, old:UOp, src:UOp) -> UOp|None:
  """Accumulate an assembled allreduce in its owner slices before the existing allgather."""
  while target.op is Ops.RESHAPE: target = target.src[0]
  while old.op is Ops.RESHAPE: old = old.src[0]
  accum_debug = getenv("PERSISTENT_ACCUM_DEBUG")
  if old is not target: return None
  while src.op in {Ops.RESHAPE, Ops.CONTIGUOUS, Ops.CAST}: src = src.src[0]
  if target.dtype != src.dtype or target.numel() != src.numel() or target.device != src.device: return None
  if target is output: destination = output
  elif target.base is output.base and target.contiguous_view_offset() is not None: destination = target
  else: return None

  # A direct all-to-all result is a complete matrix of physical slice states: one state per (rank, chunk).
  # Exactly one state per chunk is an owner reduction; all other states copy from that owner through AFTER.
  if src.op is not Ops.AFTER or not isinstance(src.device, tuple): return None
  states, ndev, assembled_output = src.src[1:], len(src.device), src.src[0].buf_uop
  if len(states) != ndev*ndev: return None
  stores: list[UOp] = []
  coordinates: list[tuple[int, int, int]] = []
  for state in states:
    if (state.op is not Ops.AFTER or len(state.src) != 2 or state.src[0].op is not Ops.SHRINK or state.src[0].tag != ("allreduce",)
        or state.src[1].op is not Ops.STORE or state.src[1].src[0] is not state.src[0]): return None
    view = state.src[0]
    if (view.src[0].op is not Ops.MSELECT or view.src[0].src[0].buf_uop is not assembled_output
        or view.src[1].op is not Ops.CONST or view.src[2].op is not Ops.CONST): return None
    coordinates.append((view.src[0].arg, view.src[1].val, view.src[2].val))
    stores.append(state.src[1])
  owners = [state for state,store in zip(states, stores) if store.src[1].op is not Ops.COPY]
  gathers = [store for store in stores if store.src[1].op is Ops.COPY]
  if len(owners) != ndev or len(gathers) != ndev*(ndev-1): return None
  chunks = {(start, size) for _,start,size in coordinates}
  if (len(chunks) != ndev or set(rank for rank,_,_ in coordinates) != set(range(ndev))
      or any(sum((start, size) == chunk for _,start,size in coordinates) != ndev for chunk in chunks)): return None
  owner_chunks = {(owner.src[0].src[1].val, owner.src[0].src[2].val) for owner in owners}
  if owner_chunks != chunks: return None
  if any(assembled_output in owner.src[1].src[1].toposort(enter_calls=False) for owner in owners): return None
  if any(len(gather.src[1].src) != 1 or gather.src[1].src[0] not in owners for gather in gathers): return None

  # Preserve the reduced BF16 value as an input to the add. Only the owner store is changed; gather stores remain
  # overwrites and therefore broadcast the already accumulated owner value exactly once.
  owner_map: dict[UOp, UOp] = {}
  for owner in owners:
    store = owner.src[1]
    accumulated = store.src[0].alu(Ops.ADD, store.src[1]).rtag(("allreduce_accumulate",))
    rewritten_owner = owner.replace(src=(owner.src[0], store.replace(src=(store.src[0], accumulated))))
    owner_map[owner] = rewritten_owner.substitute({assembled_output:destination})
  rewritten = [owner_map[state] if state in owner_map else state.substitute({assembled_output:destination, **owner_map}) for state in states]
  if accum_debug: print(f"persistent accumulate forwarded: {destination.shape} at {destination.contiguous_view_offset()}")
  return output.after(*rewritten)

def forward_linear_store(ctx:dict[UOp, UOp], output:UOp, target:UOp, src:UOp) -> UOp|None:
  """Collect caller-provided allreduce outputs so each shared LINEAR invocation is redirected exactly once."""
  while target.op is Ops.RESHAPE: target = target.src[0]
  while src.op in {Ops.RESHAPE, Ops.CONTIGUOUS, Ops.CAST}: src = src.src[0]
  if target.dtype != src.dtype or target.numel() != src.numel() or target.device != src.device: return None
  if target is output: destination = output
  elif target.base is output.base and target.contiguous_view_offset() is not None: destination = target
  else: return None
  linear_calls = [x for x in src.src[1:] if x.op is Ops.CALL and x.src[0].op is Ops.LINEAR] if src.op is Ops.AFTER else []
  if len(linear_calls) != 1 or len(src.src) != 2 or (offset:=destination.contiguous_view_offset()) is None: return None
  call, old_output = linear_calls[0], src.src[0].buf_uop
  arg_idxs = [i for i,x in enumerate(call.src[1:], start=1) if x.buf_uop is old_output]
  if len(arg_idxs) != 1 or not is_allreduce_linear_output(call.src[0], arg_idxs[0]-1): return None
  arg_idx, prior = arg_idxs[0], ctx.get(call, call)
  physical = _allreduce_view(destination.buf_uop, offset, offset+destination.numel()).reshape(call.src[arg_idx].shape)
  ctx[call] = prior.replace(src=prior.src[:arg_idx]+(physical,)+prior.src[arg_idx+1:])
  # Keep the original call as a token until every store has been visited. get_kernel_graph substitutes the single,
  # fully redirected call afterward, avoiding one producer clone per returned gradient.
  return output.after(call)

def _linear_allreduce_view(x:UOp, output:UOp) -> tuple[int, int, int]|None:
  """Return (rank, offset, size) when x is a physical slice of a LINEAR allreduce output."""
  x = x.buf_uop
  if (x.op is not Ops.SHRINK or x.tag != ("allreduce",) or x.src[0].op is not Ops.MSELECT
      or x.src[0].src[0].buf_uop is not output or x.src[1].op is not Ops.CONST or x.src[2].op is not Ops.CONST): return None
  return x.src[0].arg, x.src[1].val, x.src[2].val

def _accumulate_linear_allreduce(linear:UOp, slot:int) -> UOp|None:
  params = [x for x in linear.toposort() if x.op is Ops.PARAM and isinstance(x.arg, ParamArg) and x.arg.slot == slot]
  if len(params) != 1 or not isinstance(params[0].device, tuple): return None
  output, ndev = params[0], len(params[0].device)
  rewritten, owners = list(linear.src), {}
  for call_idx,call in enumerate(linear.src):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.SINK: continue
    output_args = [(i,key) for i,arg in enumerate(call.src[1:]) if (key:=_linear_allreduce_view(arg, output)) is not None]
    if len(output_args) != 1: continue
    arg_idx, key = output_args[0]
    body_params = [x for x in call.src[0].toposort() if x.op is Ops.PARAM and isinstance(x.arg, ParamArg) and x.arg.slot == arg_idx]
    if len(body_params) != 1: return None
    param = body_params[0]
    stores = [x for x in call.src[0].toposort() if x.op is Ops.STORE and param in x.src[0].toposort()]
    loads = [x for x in call.src[0].toposort() if x.op is Ops.LOAD and param in x.src[0].toposort()]
    if len(stores) != 1 or loads: continue
    store = stores[0]
    accumulated = store.src[0].alu(Ops.ADD, store.src[1]).rtag(("allreduce_accumulate",))
    rewritten[call_idx] = call.replace(src=(call.src[0].substitute({store:store.replace(src=(store.src[0], accumulated))}),)+call.src[1:])
    if key in owners: return None
    owners[key] = call_idx
  if len(owners) != ndev: return None

  # Every other use of an assembled output slice must be an allgather COPY. Its destination is write-only and
  # every source is an owner slice whose reducer call precedes the copy in the LINEAR ordering.
  owner_calls = set(owners.values())
  for call_idx,call in enumerate(linear.src):
    if call.op is not Ops.CALL: continue
    output_args = [(i,key) for i,arg in enumerate(call.src[1:]) if (key:=_linear_allreduce_view(arg, output)) is not None]
    if not output_args or call_idx in owner_calls: continue
    if call.src[0].op is not Ops.COPY or len(output_args) != 2 or output_args[0][0] != 0 or output_args[1][0] == 0: return None
    source_key = output_args[1][1]
    if source_key not in owners or owners[source_key] >= call_idx: return None
  return linear.replace(src=tuple(rewritten))

def _accumulate_linear_replicated(linear:UOp, slot:int) -> UOp|None:
  """Add into a replicated LINEAR output with exactly one independent write-only producer per rank."""
  nodes = linear.toposort()
  params = [x for x in nodes if x.op is Ops.PARAM and isinstance(x.arg, ParamArg) and x.arg.slot == slot]
  if len(params) != 1 or not isinstance(params[0].device, tuple): return None
  output, ndev = params[0], len(params[0].device)
  consumers: dict[UOp, list[UOp]] = {}
  for x in nodes:
    for s in x.src: consumers.setdefault(s, []).append(x)
  selects = consumers.get(output, [])
  if len(selects) == 1 and selects[0].op is Ops.CALL:
    call = selects[0]
    if call not in linear.src or call.src[0].op is not Ops.SINK: return None
    direct_arg_idxs = [i for i,arg in enumerate(call.src[1:]) if arg.buf_uop is output]
    if len(direct_arg_idxs) != 1: return None
    arg_idx = direct_arg_idxs[0]
    body_params = [x for x in call.src[0].toposort() if x.op is Ops.PARAM and isinstance(x.arg, ParamArg) and x.arg.slot == arg_idx]
    if len(body_params) != 1: return None
    param = body_params[0]
    stores = [x for x in call.src[0].toposort() if x.op is Ops.STORE and param in x.src[0].toposort()]
    loads = [x for x in call.src[0].toposort() if x.op is Ops.LOAD and param in x.src[0].toposort()]
    if len(stores) != 1 or loads: return None
    store = stores[0]
    accumulated = store.src[0].alu(Ops.ADD, store.src[1]).rtag(("allreduce_accumulate",))
    new_call = call.replace(src=(call.src[0].substitute({store:store.replace(src=(store.src[0], accumulated))}),)+call.src[1:])
    return linear.replace(src=tuple(new_call if x is call else x for x in linear.src))
  if len(selects) != ndev or any(x.op is not Ops.MSELECT or consumers.get(x) is None or
                                 any(y.op is not Ops.CALL for y in consumers[x]) for x in selects): return None

  rewritten, ranks = list(linear.src), set()
  for call_idx,call in enumerate(linear.src):
    ranked_output_args:list[tuple[int, int]] = [(i,arg.buf_uop.arg) for i,arg in enumerate(call.src[1:])
      if arg.buf_uop.op is Ops.MSELECT and isinstance(arg.buf_uop.arg, int) and arg.buf_uop.src[0].buf_uop is output]
    if not ranked_output_args: continue
    if call.op is not Ops.CALL or call.src[0].op is not Ops.SINK or len(ranked_output_args) != 1: return None
    arg_idx, rank = ranked_output_args[0]
    body_params = [x for x in call.src[0].toposort() if x.op is Ops.PARAM and isinstance(x.arg, ParamArg) and x.arg.slot == arg_idx]
    if len(body_params) != 1: return None
    param = body_params[0]
    stores = [x for x in call.src[0].toposort() if x.op is Ops.STORE and param in x.src[0].toposort()]
    loads = [x for x in call.src[0].toposort() if x.op is Ops.LOAD and param in x.src[0].toposort()]
    if len(stores) != 1 or loads or rank in ranks: return None
    store = stores[0]
    accumulated = store.src[0].alu(Ops.ADD, store.src[1]).rtag(("allreduce_accumulate",))
    rewritten[call_idx] = call.replace(src=(call.src[0].substitute({store:store.replace(src=(store.src[0], accumulated))}),)+call.src[1:])
    ranks.add(rank)
  return linear.replace(src=tuple(rewritten)) if ranks == set(range(ndev)) else None

def forward_linear_accumulate(ctx:dict[UOp, UOp], output:UOp, target:UOp, old:UOp, src:UOp) -> UOp|None:
  """Retarget a precompiled allreduce and add the old destination only in its owner reducer calls."""
  while target.op is Ops.RESHAPE: target = target.src[0]
  while old.op is Ops.RESHAPE: old = old.src[0]
  if old is not target: return None
  while src.op in {Ops.RESHAPE, Ops.CONTIGUOUS, Ops.CAST}: src = src.src[0]
  if target.dtype != src.dtype or target.numel() != src.numel() or target.device != src.device: return None
  if target is output: destination = output
  elif target.base is output.base and target.contiguous_view_offset() is not None: destination = target
  else: return None
  linear_calls = [x for x in src.src[1:] if x.op is Ops.CALL and x.src[0].op is Ops.LINEAR] if src.op is Ops.AFTER else []
  if len(linear_calls) != 1 or len(src.src) != 2 or (offset:=destination.contiguous_view_offset()) is None: return None
  call, old_output = linear_calls[0], src.src[0].buf_uop
  arg_idxs = [i for i,x in enumerate(call.src[1:], start=1) if x.buf_uop is old_output]
  if len(arg_idxs) != 1: return None
  arg_idx, prior = arg_idxs[0], ctx.get(call, call)
  linear = (_accumulate_linear_allreduce(prior.src[0], arg_idx-1) if is_allreduce_linear_output(call.src[0], arg_idx-1)
            else _accumulate_linear_replicated(prior.src[0], arg_idx-1))
  if linear is None: return None
  physical = _allreduce_view(destination.buf_uop, offset, offset+destination.numel()).reshape(call.src[arg_idx].shape)
  ctx[call] = prior.replace(src=(linear,)+prior.src[1:arg_idx]+(physical,)+prior.src[arg_idx+1:])
  if getenv("PERSISTENT_ACCUM_DEBUG"): print(f"persistent accumulate forwarded linear: {destination.shape} at {offset}")
  return output.after(call)

pm_forward_linear_store = PatternMatcher([
  (UPat(Ops.AFTER, src=(UPat.var("output"), UPat(Ops.STORE, src=(UPat.var("target"),
    UPat(Ops.ADD, src=(UPat.var("old"), UPat.var("src"))))))), forward_linear_accumulate),
  (UPat(Ops.AFTER, src=(UPat.var("output"), UPat(Ops.STORE, src=(UPat.var("target"), UPat.var("src"))))), forward_linear_store),
])

earliest_rewrites = mop_cleanup+PatternMatcher([
  # ALLREDUCE lowering can introduce these after the multi pass has already visited the parent.
  (UPat(Ops.MSELECT, src=(UPat(Ops.MSTACK, name="mstack"),), name="ms"), lambda mstack,ms: mstack.src[ms.arg]),
  # resolve FUNCTION calls (inline the body)
  (UPat(Ops.FUNCTION, name="c"), resolve_function),

  # resolve TUPLE+GETTUPLE
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g,t: t.src[g.arg]),

  # resolve allreduce (must be bottom up)
  (UPat(Ops.AFTER, src=(UPat.var("output"), UPat(Ops.STORE, src=(UPat.var("target"),
    UPat(Ops.ADD, src=(UPat.var("old"), UPat.var("src"))))))), forward_assembled_accumulate),
  (UPat(Ops.AFTER, src=(UPat.var("output"), UPat(Ops.STORE, src=(UPat.var("target"), UPat.var("src"))))), forward_assembled_store),
  (UPat(Ops.ALLREDUCE, src=(UPat.var("buf"),), name="red"), create_allreduce_function),

  # split_reduceop
  (UPat(Ops.REDUCE, name="reduce", src=(UPat.var("x"),)), split_reduceop),

  # remove DETACH/CONTIGUOUS_BACKWARD (TODO: this is copied in allocations)
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),

  # SINK only ever references the base
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=tuple(y.unsharded_base for y in x.src))),

  # ** copy rules **

  # COPY transfers a contiguous range, so materialize a source that's resized (shrink/pad/expand) or reordered (permute/flip)
  (UPat(Ops.COPY, src=(UPat(GroupOp.Movement, name="r"),), name="c"),
   lambda c,r: c.replace(src=(r.contiguous(),)) if r.tag != ("allreduce",) and
   (resolve(r.numel() != r.base.numel(), False) or r.contiguous_view_offset() is None) else None),

  # copy to same device is a no-op
  (UPat(Ops.COPY, src=(UPat.var("x"),), name="copy"), lambda x,copy: x if x.device == copy.device else None),

  # copy on reshape is reshape on copy
  (UPat(Ops.COPY, src=(UPat(Ops.RESHAPE, name="shp"),), name="cpy"), lambda shp,cpy: shp.src[0].copy_to_device(cpy.device).reshape(shp.shape)),

  # reshaping on STORE can be a NOOP
  (UPat(Ops.STORE, src=(UPat(Ops.RESHAPE, src=(UPat.var("dst",),), allow_any_len=True),
                        UPat(Ops.RESHAPE, src=(UPat.var("src",),), allow_any_len=True))),
   lambda dst,src: dst.store(src) if dst.shape == src.shape else None),

  # ** store rules **

  # fix store hazard (dest is in used in src) by adding contiguous: TestAssign.test_post_flipped_assignment
  (UPat(Ops.STORE, src=(UPat(name="target"), UPat(name="src"))), fix_store_hazard),

  # remove two STOREs that store the same thing to the same place: TestSchedule.test_dedup_Assign
  (UPat.var("buf").after(UPat.var("buf").store(UPat.var("src")), name="a1").after(UPat.var("a1").store(UPat.var("src"))), lambda buf,src,a1:a1),

  # store a buffer's own current contents back into itself: TestAssign.test_nested_after_contiguous_store_no_init
  (UPat.var("buf").after(UPat.var("buf").store(UPat.var("buf").after(UPat.var("buf").store(UPat.var("src")), name="a1"))), lambda buf,src,a1:a1),

  # move bitcast from store dest to source: TestAssign.test_assign_bitcast
  (UPat(Ops.STORE, src=(UPat(Ops.BITCAST, src=(UPat(name="target"),)), UPat(name="src"))),
   lambda target, src: target.store(src.bitcast(target.dtype))),

  (UPat(Ops.BITCAST, name="bc"), expand_bitcast),

  # ** size 0 **

  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if 0 in x.shape and 0 not in reduce.shape else None),
  # handle size 0
  (UPat(GroupOp.All-{Ops.SINK}, name="x"), lambda x: x.const_like(0).rtag(x.tag) if x._shape is not None and 0 in x.shape else None),

  # remove movement ops from SINK/AFTER. TODO: should be generic
  (UPat(Ops.SINK, name="s"), lambda s: s.replace(src=tuple(walk_mop(u) for u in s.src if u.op is not Ops.NOOP))),
  (UPat(Ops.AFTER, name="s"), lambda s: s.replace(src=(s.src[0],)+tuple(walk_mop(u) for u in s.src[1:] if u.op is not Ops.NOOP))),
])

def convert_copy_to_store(ctx, copy:UOp, existing_buf:UOp|None=None):
  # Tagged copies are the payload of the physical-view STORE synthesized below. Leave them intact for
  # split_copy_slice instead of recursively materializing another destination.
  if copy.tag == ("allreduce",): return None
  input_src = copy.src[0]
  # A hardware-slice COPY already under STORE is ready for split_copy_slice. A standalone COPY still needs its
  # destination buffer; its generated STORE will then take the same direct SDMA lowering path.
  is_slice_copy = ((input_src.op is Ops.SHRINK and input_src.tag == ("allreduce",)) or
                   (input_src.op is Ops.AFTER and input_src.src[0].op is Ops.SHRINK and input_src.src[0].tag == ("allreduce",)))
  # An AFTER(view, STORE) is an already reduced allgather source. Preserve its COPY even while visiting the child
  # bottom-up, so the parent STORE can lower source and destination slices together instead of inserting a staging copy.
  source_is_stored = input_src.op is Ops.AFTER and any(s.op is Ops.STORE for s in input_src.src[1:])
  if is_slice_copy and (existing_buf is not None or source_is_stored): return None
  if is_slice_copy:
    # Preserve the old SLICE lowering shape: standalone transfers first acquire their destination, then the
    # resulting STORE is split into a direct runtime copy whose source and destination retain their offsets.
    buf = UOp(Ops.BUFFER, src=(shape_to_shape_arg(input_src.max_shape),), arg=ParamArg(next(ctx), copy.dtype, device=copy.device))
    return buf.after(buf.store(copy.rtag(("allreduce",)))).reshape(copy.shape)
  # if it's a COPY, we need to give the input buffer identity
  if not input_src.has_buffer_identity(after_ok=True) and copy.op is Ops.COPY: input_src = input_src.contiguous()
  input_src = input_src.flatten()
  if existing_buf is not None:
    # if the existing buffer is not a full buffer, we can't use it
    if not existing_buf.has_buffer_identity(after_ok=True): return None
    # if there's already a buffer, we just use it
    return existing_buf.flatten().store(input_src)
  # create the output buffer
  buf = UOp(Ops.BUFFER, src=(shape_to_shape_arg(input_src.max_shape),), arg=ParamArg(next(ctx), copy.dtype, device=copy.device))
  # reshape back to input
  return buf.after(buf.store(input_src)).reshape(copy.shape)

pm_copy_to_store = PatternMatcher([
  (UPat(name="existing_buf").store(UPat(Ops.COPY, name="copy")), convert_copy_to_store),
  (UPat(Ops.COPY, name="copy"), convert_copy_to_store),
])

@rewrite_group(new_ctx=False)
def prepare_rangeify(sink:UOp) -> UOp:
  # prepare for rangeify
  tsink = graph_rewrite(sink, multi_pm, name="multi_pm")
  if OPENPILOT_HACKS: tsink = graph_rewrite(tsink, pm_fold_moved_after, ctx={}, name="fold moved afters")
  linear_outputs:dict[UOp, UOp] = {}
  tsink = graph_rewrite(tsink, pm_forward_linear_store, ctx=linear_outputs, bottom_up=True, name="forward linear outputs")
  tsink = tsink.substitute(linear_outputs, name="merge forwarded linear outputs")
  tsink = graph_rewrite(tsink, pm_mops+earliest_rewrites, bottom_up=True, name="earliest rewrites")
  tsink = graph_rewrite(tsink, pm_copy_to_store, ctx=itertools.count(0), bottom_up=True, name="convert copy to store")
  return tsink
