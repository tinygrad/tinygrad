import functools, itertools
from tinygrad.helpers import all_int, prod, DEBUG, RING, ALL2ALL, getenv
from tinygrad.uop.ops import UOp
from tinygrad.dtype import Invalid

# *** allreduce implementation ***
def handle_allreduce(buf:UOp, red:UOp) -> UOp|None:
  if not isinstance(buf.device, tuple): return None
  assert all_int(buf.shape), f"does not support symbolic shape {buf.shape}"
  ndev, shape, numel = len(buf.device), buf.shape, prod(buf.shape)
  op, device = red.arg

  # ring allreduce doesn't provide a benefit with only 2 nodes or where number of elements is less than 256k (empirically)
  # fallback to naive allreduce to save on kernel dispatch, chunking and reassembling chunks.
  use_all2all = (ALL2ALL >= 2 or (ndev > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and ALL2ALL >= 1))
  use_ring = not use_all2all and (RING >= 2 or (ndev > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and RING >= 1))
  if DEBUG >= 2: print(f"{'ALL2ALL' if use_all2all else 'RING' if use_ring else 'NAIVE'} ALLREDUCE {ndev}x{numel} | {buf.dtype}")

  # contiguous before we copy it
  buf = buf.contiguous()

  # naive: copy to all devices. if you shrink later, that'll be handled
  if not use_ring and not use_all2all:
    if isinstance(device, str):
      return functools.reduce(lambda x,y: x.alu(op, y), [buf.mselect(i).copy_to_device(device) for i in range(ndev)])
    # copies are gathered into per-source scratch buffers with MSELECT stores, a device's own shard is read directly
    dnum = UOp.variable("_device_num", 0, ndev-1)
    terms:list[UOp] = []
    for i in range(ndev):
      scratch = UOp.new_buffer(device, numel, buf.dtype)
      state = scratch.after(*[scratch.mselect(j).store(buf.mselect(i).copy_to_device(device[j])) for j in range(ndev) if j != i])
      terms.append(dnum.eq(i).where(buf, state.reshape(shape)))
    return functools.reduce(lambda x,y: x.alu(op, y), terms)

  # chunk data into ndev pieces
  factor = next((f for f in [32, 16, 8, 4, 2] if numel % f == 0), 1)
  base, left = divmod(numel // factor,  ndev)
  chunks = list(itertools.pairwise(itertools.accumulate([(base + 1) * factor] * left + [base * factor] * (ndev - left), initial=0)))

  # reduce-scatter: with all2all chunk i is reduced on buf.device[i], with ring it ends up on buf.device[(i-1)%ndev]
  reduced_chunks:list[UOp] = []
  starts:list[int] = []
  for i,(s,e) in enumerate(chunks):
    if use_all2all:
      chunks_on_i = [buf.mselect(j).reshape((numel,)).shrink(((s,e),)).copy_to_device(buf.device[i]) for j in range(ndev)]
      reduced_chunks.append(functools.reduce(lambda x,y: x.alu(op, y), chunks_on_i))
      starts.append(i)
    else:
      chunk, reduced = buf.reshape((numel,)).shrink(((s,e),)), buf.reshape((numel,)).shrink(((s,e),))
      for step in range(ndev-1):
        src, dest = (i+step)%ndev, (i+step+1)%ndev
        cp = reduced.copy_to_device(buf.device[dest], src if isinstance(reduced.device, tuple) else None)
        reduced = cp.alu(op, chunk.copy_to_device(buf.device[dest], dest))
      reduced_chunks.append(reduced)
      starts.append((i+ndev-1)%ndev)

  # single device output: copy the reduced chunks straight to the output device
  if isinstance(device, str):
    return UOp.usum(*[rc.copy_to_device(device).pad(((s,numel-e),)) for (s,e),rc in zip(chunks, reduced_chunks)]).reshape(shape)

  # allgather: store each chunk into an MSELECT of a scratch buffer, then copy it to the other devices
  gathered:list[UOp] = []
  for (s,e),rc,start in zip(chunks, reduced_chunks, starts):
    scratch = UOp.new_buffer(device, e-s, buf.dtype)
    state = scratch.after(scratch.mselect(start).store(rc))
    if use_all2all:
      state = state.after(*[scratch.mselect(j).store(state.mselect(start).copy_to_device(device[j])) for j in range(ndev) if j != start])
    else:
      # forward the chunk around the ring
      for step in range(1, ndev):
        prev, dest = (start+step-1)%ndev, (start+step)%ndev
        state = state.after(scratch.mselect(dest).store(state.mselect(prev).copy_to_device(device[dest])))
    gathered.append(state)

  # reassemble
  return UOp.usum(*[g.pad(((s,numel-e),)) for (s,e),g in zip(chunks, gathered)]).reshape(shape)

def create_allreduce_function(buf:UOp, red:UOp, output:UOp|None=None) -> UOp|None:
  if output is None: output = UOp.const(red.dtype, Invalid, shape=red.shape).clone(device=red.device)
  to = red.param_like(0)
  src = buf.param_like(1)
  red = src.allreduce(*red.arg)
  return output.after(to.after(to.store(handle_allreduce(src, red))).sink().call(output, buf.contiguous(), name="allreduce", precompile=True))
