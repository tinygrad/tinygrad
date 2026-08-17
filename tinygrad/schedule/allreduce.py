import functools, itertools
from tinygrad.helpers import all_int, all_same, prod, DEBUG, RING, ALL2ALL, getenv
from tinygrad.uop.ops import Ops, UOp

# *** allreduce implementation ***
def allreduce_modes(ndev:int, numel:int) -> tuple[bool, bool]:
  use_all2all = ALL2ALL >= 2 or (ndev > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and ALL2ALL >= 1)
  use_ring = not use_all2all and (RING >= 2 or (ndev > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and RING >= 1))
  return use_all2all, use_ring

def _allreduce_view(buf:UOp, start:int, end:int) -> UOp:
  base = buf.buf_uop.rtag(("allreduce_base",)).flatten()
  return base.shrink(((start, end),)).rtag(("allreduce",))

def _allreduce_input_view(buf:UOp, start:int, end:int) -> UOp:
  # Tag the flattened view, not the buffer identity: this protects its physical offset through rewrites while
  # keeping the COPY argument attached to the realized staging allocation.
  return buf.buf_uop.flatten().rtag(("allreduce_base",)).shrink(((start, end),)).rtag(("allreduce",))

def _allreduce_chunk(buf:UOp, start:int, end:int, input_staged:bool, physical:bool=False) -> UOp:
  # All-to-all selects concrete children and can transfer physical views directly. Ring carries the multi-device
  # value through successive hops, so retain semantic views for its existing materialization path.
  chunks = [(_allreduce_input_view(buf.mselect(i), start, end) if physical else
             buf.mselect(i).buf_uop.flatten().shrink(((start, end),))) for i in range(len(buf.device))]
  if not input_staged: chunks = [chunk.after(buf) for chunk in chunks]
  return UOp.mstack(*chunks)

def handle_allreduce(buf:UOp, red:UOp, output:UOp|None=None, input_staged:bool=False) -> UOp|None:
  if not isinstance(buf.device, tuple): return None
  ndev, shape, numel = len(buf.device), buf.shape, prod(buf.shape)
  op, device = red.arg

  # ring allreduce doesn't provide a benefit with only 2 nodes or where number of elements is less than 256k (empirically)
  # fallback to naive allreduce to save on kernel dispatch, chunking and reassembling chunks.
  concrete = all_int(shape)
  if concrete:
    assert isinstance(numel, int)
    use_all2all, use_ring = allreduce_modes(ndev, numel)
  else: use_all2all, use_ring = False, False
  if DEBUG >= 2: print(f"{'ALL2ALL' if use_all2all else 'RING' if use_ring else 'NAIVE'} ALLREDUCE {ndev}x{numel} | {buf.dtype}")

  buf = buf.pad_to(buf.max_shape)
  # SDMA reads can outlive the compute allocation that produced them, so reduce-scatter needs stable storage.
  # A precompiled allreduce's PARAM is already backed by the contiguous CALL argument below.
  if not input_staged:
    staged = buf.empty_like()
    buf = staged.after(staged.store(buf))

  # naive: copy to all devices. if you shrink later, that'll be handled
  if not use_ring and not use_all2all:
    return functools.reduce(lambda x,y: x.alu(op, y), [buf.mselect(i).copy_to_device(device) for i in range(ndev)]).shrink_to(shape)

  # chunk data into ndev pieces
  assert isinstance(numel, int)
  factor = next((f for f in [32, 16, 8, 4, 2] if numel % f == 0), 1)
  base, left = divmod(numel // factor,  ndev)
  chunks = list(itertools.pairwise(itertools.accumulate([(base + 1) * factor] * left + [base * factor] * (ndev - left), initial=0)))
  direct_stack = all_same([e-s for s,e in chunks]) and not isinstance(device, str)

  # reduce-scatter
  reduced_chunks:list[UOp] = []
  for i,(s,e) in enumerate(chunks):
    chunk = _allreduce_chunk(buf, s, e, input_staged, physical=use_all2all)
    if use_all2all:
      # _allreduce_chunk is an MSTACK of concrete physical views. Select its children directly so the view
      # remains the COPY argument instead of depending on a later MSELECT(MSTACK) simplification.
      chunks_on_i = [chunk.src[j].copy_to_device(buf.device[i]) for j in range(ndev)]
      reduced_chunks.append(functools.reduce(lambda x,y: x.alu(op, y), chunks_on_i))
    else:
      reduced = chunk
      for step in range(ndev-1):
        src, dest = (i+step)%ndev, (i+step+1)%ndev
        cp = reduced.copy_to_device(buf.device[dest], src if isinstance(reduced.device, tuple) else None)
        reduced = cp.alu(op, chunk.copy_to_device(buf.device[dest], dest))
      reduced_chunks.append(reduced)

  # Equal chunks can be reduced and gathered directly into their final storage. This avoids materializing
  # a padded MSTACK and then running a full-size reassembly kernel on every device.
  if direct_stack:
    if output is None: output = UOp.empty(*shape, dtype=reduced_chunks[0].dtype, device=device)
    states = [[_allreduce_view(output.mselect(j), s, e) for s,e in chunks] for j in range(ndev)]
    for i,rc in enumerate(reduced_chunks):
      owner = i if use_all2all else (i-1) % ndev
      target = states[owner][i]
      states[owner][i] = target.after(target.store(rc.cast(output.dtype)))
      source = states[owner][i]
      for step in range(1, ndev):
        dest = (owner+step) % ndev
        target = states[dest][i]
        states[dest][i] = target.after(target.store(source.copy_to_device(buf.device[dest])))
        if use_ring: source = states[dest][i]
    return output.after(*itertools.chain.from_iterable(states))

  # allgather
  copied_chunks:list[UOp] = []
  for i,rc in enumerate(reduced_chunks):
    if isinstance(device, str): copied_chunks.append(rc.copy_to_device(device))
    elif use_all2all: copied_chunks.append(UOp.mstack(*(rc.copy_to_device(buf.device[j]) for j in range(ndev))))
    else:
      chain:list[UOp] = [rc]
      for step in range(ndev-1):
        chain.append(rc := rc.copy_to_device(buf.device[(i+step)%ndev]))
      copied_chunks.append(UOp.mstack(*(chain[(j-i+1)%ndev] for j in range(ndev))))

  # reassemble
  return UOp.usum(*[c.pad(((s,numel-e),)) for (s,e),c in zip(chunks, copied_chunks)]).reshape(shape)

def create_allreduce_function(buf:UOp, red:UOp, output:UOp|None=None) -> UOp|None:
  if output is None: output = UOp.invalids(red.shape, dtype=red.dtype, device=red.device)
  if isinstance(buf.device, tuple) and all_int(buf.shape) and allreduce_modes(len(buf.device), prod(buf.shape))[0]:
    ret = handle_allreduce(buf, red, output)
    assert ret is not None
    return ret if ret.op is Ops.AFTER and ret.src[0] is output else output.after(output.store(ret))
  to = red.param_like(0)
  src = buf.param_like(1)
  red = src.allreduce(*red.arg)
  ret = handle_allreduce(src, red, to, input_staged=True)
  assert ret is not None
  body = ret if ret.op is Ops.AFTER and ret.src[0] is to else to.after(to.store(ret))
  return output.after(body.sink().call(output, buf.contiguous(), name="allreduce", precompile=True))
