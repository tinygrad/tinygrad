import functools, itertools
from tinygrad.helpers import all_int, all_same, prod, DEBUG, RING, ALL2ALL, getenv
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import Invalid, dtypes

# *** allreduce implementation ***
def handle_allreduce(buf:UOp, red:UOp, output:UOp|None=None) -> UOp|None:
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
    return functools.reduce(lambda x,y: x.alu(op, y), [buf.mselect(i).copy_to_device(device) for i in range(ndev)])

  # chunk data into ndev pieces
  factor = next((f for f in [32, 16, 8, 4, 2] if numel % f == 0), 1)
  base, left = divmod(numel // factor,  ndev)
  chunks = list(itertools.pairwise(itertools.accumulate([(base + 1) * factor] * left + [base * factor] * (ndev - left), initial=0)))
  direct_stack = all_same([e-s for s,e in chunks])

  # reduce-scatter
  reduced_chunks:list[UOp] = []
  reduced_deps:list[UOp|None] = []
  for i,(s,e) in enumerate(chunks):
    if use_all2all:
      chunks_on_i = [buf.mselect(j).reshape((numel,)).shrink(((s,e),)).copy_to_device(buf.device[i]) for j in range(ndev)]
      reduced = functools.reduce(lambda x,y: x.alu(op, y), chunks_on_i)
      dep = None
      if not isinstance(device, str) and not direct_stack:
        tmp = reduced.empty_like()
        dep = tmp.after(tmp.store(reduced))
        reduced = dep if output is None else tmp
      reduced_chunks.append(reduced)
      reduced_deps.append(dep)
    else:
      chunk, reduced = buf.reshape((numel,)).shrink(((s,e),)), buf.reshape((numel,)).shrink(((s,e),))
      for step in range(ndev-1):
        src, dest = (i+step)%ndev, (i+step+1)%ndev
        cp = reduced.copy_to_device(buf.device[dest], src if isinstance(reduced.device, tuple) else None)
        reduced = cp.alu(op, chunk.copy_to_device(buf.device[dest], dest))
      reduced_chunks.append(reduced)
      reduced_deps.append(None)

  # Write each reduced STACK chunk into its final slice so all gather copies lower directly to SDMA.
  if direct_stack:
    stack = UOp(Ops.STACK, src=tuple(reduced_chunks))
    if output is None: output = UOp.empty(*shape, dtype=stack.dtype, device=device)
    states = [[UOp(Ops.SLICE, output.dtype, (output.mselect(i), UOp.const(dtypes.index, s)), e-s) for s,e in chunks] for i in range(ndev)]
    for i,rc in enumerate(stack.src):
      owner = i if use_all2all else (i-1)%ndev
      target_slice = states[owner][i]
      states[owner][i] = target_slice.after(target_slice.store(rc.cast(output.dtype)))
      source = states[owner][i]
      for step in range(1, ndev):
        dest_idx = (owner+step)%ndev
        cp = source.copy_to_device(buf.device[dest_idx])
        target_slice = states[dest_idx][i]
        states[dest_idx][i] = target_slice.after(target_slice.store(cp))
        if use_ring: source = states[dest_idx][i]
    return output.after(*itertools.chain.from_iterable(states))

  # allgather
  copied_chunks:list[UOp] = []
  for i,rc in enumerate(reduced_chunks):
    if isinstance(device, str): copied_chunks.append(rc.copy_to_device(device))
    elif use_all2all:
      dep = reduced_deps[i]
      copy_src = dep if dep is not None else rc
      copied_chunks.append(UOp.mstack(*(rc if j == i else copy_src.copy_to_device(buf.device[j]) for j in range(ndev))))
    else:
      chain:list[UOp] = [rc]
      for step in range(ndev-1):
        chain.append(rc := rc.copy_to_device(buf.device[(i+step)%ndev]))
      copied_chunks.append(UOp.mstack(*(chain[(j-i+1)%ndev] for j in range(ndev))))

  # reassemble
  if output is not None and use_all2all:
    flat_out = output.reshape((numel,))
    deps = [d for d in reduced_deps if d is not None]
    return output.after(*deps, *[flat_out.shrink(((s,e),)).store(c) for (s,e),c in zip(chunks, copied_chunks)])
  return UOp.usum(*[c.pad(((s,numel-e),)) for (s,e),c in zip(chunks, copied_chunks)]).reshape(shape)

def create_allreduce_function(buf:UOp, red:UOp, output:UOp|None=None) -> UOp|None:
  if output is None: output = UOp.const(red.dtype, Invalid, shape=red.shape).clone(device=red.device)
  to = output.param_like(0)
  src = buf.param_like(1)
  red = src.allreduce(*red.arg)
  ret = handle_allreduce(src, red, to)
  assert ret is not None
  body = ret if ret.op is Ops.AFTER and ret.src[0] is to else to.after(to.store(ret.cast(to.dtype)))
  return output.after(body.sink().call(output, buf.contiguous(), name="allreduce", precompile=True))
