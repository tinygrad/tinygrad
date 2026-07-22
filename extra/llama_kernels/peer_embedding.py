from tinygrad import Tensor, dtypes
from tinygrad.helpers import prod
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

NDEV = 8

def peer_embedding_bwd(grad_emb:UOp, call:UOp) -> tuple[UOp|None, ...]:
  from tinygrad.nn import _embedding_bwd
  weight, idx = call.src[1:]
  if not (isinstance(weight.device, tuple) and len(weight.device) == NDEV and isinstance(grad_emb.device, tuple) and
          grad_emb.dtype == dtypes.bfloat16 and idx.dtype == dtypes.int32): return _embedding_bwd(grad_emb, call)

  local_vocab, embed = weight.shape[0] // NDEV, weight.shape[1]
  tokens = idx.numel() // NDEV
  assert idx.numel() == tokens * NDEV, f"idx {idx.shape} does not match {NDEV} gradient shards of {tokens} tokens"
  grads = tuple(grad_emb.mselect(i) for i in range(NDEV))
  idxs = tuple(idx.mselect(i) for i in range(NDEV))
  device = weight.device[0].split(":")[0]

  def zero_kernel(out:UOp) -> UOp:
    i = UOp.range(out.numel(), 0)
    return out.flatten()[i].store(0).end(i).sink(arg=KernelInfo(name="zero"))

  def embedding_bwd_kernel(device_num:int, out:UOp, *srcs:UOp) -> UOp:
    grad_srcs, idx_srcs = srcs[:NDEV], srcs[NDEV:]
    i = UOp.range(tokens * NDEV, 0)
    j_inner = UOp.range(min(256, embed), 2, AxisType.LOOP if device in ("CPU", "NULL") else AxisType.LOCAL)
    j_outer = UOp.range((embed + 255) // 256, 1)
    j = j_outer * 256 + j_inner
    shard, local_i = i // tokens, i % tokens
    token = sum((x.flatten().index(local_i.valid(shard.eq(s))).load().cast(dtypes.index)
                 for s,x in enumerate(idx_srcs)), start=UOp.const(dtypes.index, 0))
    in_range = (token >= device_num * local_vocab) & (token < (device_num + 1) * local_vocab) & (j < embed)
    output_token = (token - device_num * local_vocab).clip(0, local_vocab-1)
    local_j = j.clip(0, embed-1)
    grad_val = sum((g.flatten().index((local_i * embed + local_j).valid(in_range & shard.eq(s))).load().cast(dtypes.float)
                    for s,g in enumerate(grad_srcs)), start=UOp.const(dtypes.float, 0))
    if device in ("CPU", "NULL"): atomic_arg = "__atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED);"
    elif device == "AMD": atomic_arg = "__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);"
    else: raise NotImplementedError(f"no atomics for device {device}")
    return UOp(Ops.CUSTOM, src=(out.index(output_token, local_j), grad_val), arg=atomic_arg).end(i, j_outer, j_inner).sink(
      arg=KernelInfo(name="peer_embedding_bwd", opts_to_apply=()))

  local_outs = []
  for i,d in enumerate(weight.device):
    out = Tensor.empty(local_vocab, embed, dtype=dtypes.float32, device=d).uop.custom_kernel(fxn=zero_kernel)[0]
    srcs = (out, *grads, *idxs)
    placeholders = tuple(UOp.placeholder_like(x, slot=j) for j,x in enumerate(srcs))
    local_outs.append(out.after(embedding_bwd_kernel(i, *placeholders).call(*srcs)))
  local_outs = [x.cast(weight.dtype).contiguous() for x in local_outs]
  return (local_outs[0].mstack(*local_outs[1:]).multi(0), None)

def peer_assemble_add(dest:UOp, src:UOp) -> UOp:
  assert isinstance(dest.device, tuple) and dest.axis is None and src.axis is not None and dest.device == src.device
  ndev, axis = len(src.device), src.axis
  shard_shape = tuple(s//ndev if i == axis else s for i,s in enumerate(src.shape))
  shard_numel = prod(shard_shape)
  assert src.op is Ops.MULTI
  shards = tuple(src.src[0].mselect(i) for i in range(ndev))
  device = dest.device[0].split(":")[0]

  def assemble_kernel(out:UOp, *parts:UOp) -> UOp:
    if device in ("CPU", "NULL"):
      ranges = (i:=UOp.range(shard_numel, 0),)
    else:
      threads = min(256, shard_numel)
      assert shard_numel % threads == 0
      wg, lid = UOp.range(shard_numel // threads, 0, AxisType.GLOBAL), UOp.range(threads, 1, AxisType.LOCAL)
      ranges, i = (wg, lid), wg * threads + lid
    stores = []
    for s,part in enumerate(parts):
      ptr = out.flatten().index(s * shard_numel + i)
      stores.append(ptr.store(ptr.load() + part.flatten().index(i).load().cast(out.dtype)))
    return UOp.group(*stores).end(*ranges).sink(arg=KernelInfo(name="peer_assemble_add", opts_to_apply=()))

  calls = []
  for i in range(ndev):
    out = dest.mselect(i)
    args = (out, *shards)
    placeholders = tuple(UOp.placeholder_like(x, slot=j) for j,x in enumerate(args))
    calls.append(assemble_kernel(*placeholders).call(*args))
  return dest.after(*calls)
