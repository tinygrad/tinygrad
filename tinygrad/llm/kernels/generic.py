from __future__ import annotations
import functools
from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

def _topk_256_sort(values:UOp, indices:UOp, next_values:UOp, next_indices:UOp, ready:UOp, lane:UOp) -> tuple[UOp, UOp, UOp]:
  for size in (2, 4, 8, 16, 32, 64, 128, 256):
    for stride in (128, 64, 32, 16, 8, 4, 2, 1)[9-size.bit_length():]:
      partner = lane ^ stride
      a_value, b_value = values.after(ready)[lane], values.after(ready)[partner]
      a_index, b_index = indices.after(ready)[lane], indices.after(ready)[partner]
      a_first = (a_value < b_value) | (a_value.eq(b_value) & (a_index > b_index))
      want_first = (lane & stride).eq(0).eq((lane & size).eq(0))
      take_a = want_first.eq(a_first)
      ready = UOp.group(next_values.after(ready)[lane].store(take_a.where(a_value, b_value)),
                        next_indices.after(ready)[lane].store(take_a.where(a_index, b_index))).barrier()
      values, next_values, indices, next_indices = next_values, values, next_indices, indices
  return values, indices, ready

@functools.cache
def _topk_256_kernel(out:UOp, sel:UOp, x:UOp, k:int, softmax:bool=False) -> UOp:
  outer, lane = UOp.range(out.shape[0], 0), UOp.range(256, 1, axis_type=AxisType.LOCAL)
  values = UOp.placeholder((256,), x.dtype, 0, addrspace=AddrSpace.LOCAL)
  indices = UOp.placeholder((256,), dtypes.int32, 1, addrspace=AddrSpace.LOCAL)
  next_values = UOp.placeholder((256,), x.dtype, 2, addrspace=AddrSpace.LOCAL)
  next_indices = UOp.placeholder((256,), dtypes.int32, 3, addrspace=AddrSpace.LOCAL)
  ready = UOp.group(values.after(outer)[lane].store(x[outer, lane]),
                    indices.after(outer)[lane].store(lane.int())).barrier()
  values, indices, ready = _topk_256_sort(values, indices, next_values, next_indices, ready, lane)
  valid = lane < k
  src = 256 - k + lane
  value = values.after(ready)[src]
  if softmax:
    max_value = values.after(ready)[255]
    value = (value - max_value).exp() / sum(((values.after(ready)[256-k+i] - max_value).exp() for i in range(k)),
                                            UOp.const(x.dtype, 0))
  stores = (out[outer, lane.valid(valid)].store(value),
            sel[outer, lane.valid(valid)].store(indices.after(ready)[src]))
  return UOp.group(*stores).end(outer, lane).sink(arg=KernelInfo(name=f"topk_256{'_softmax' if softmax else ''}", opts_to_apply=()))

def topk_256(x:Tensor, k:int, softmax:bool=False) -> tuple[Tensor, Tensor]:
  outer = int(x.numel()) // 256
  values = Tensor.empty(outer, k, dtype=x.dtype, device=x.device)
  indices = Tensor.empty(outer, k, dtype=dtypes.int32, device=x.device)
  values, indices = Tensor.custom_kernel(values, indices, x.reshape(outer, 256),
    fxn=lambda out,sel,x:_topk_256_kernel(out, sel, x, k, softmax))[:2]
  return values.reshape(*x.shape[:-1], k), indices.reshape(*x.shape[:-1], k)

@functools.cache
def _inverse_unit_lower_kernel(out:UOp, x:UOp, n:int) -> UOp:
  outer_count = 1
  for dim in out.shape[:-2]:
    assert isinstance(dim, int)
    outer_count *= dim
  outer, lane = UOp.range(outer_count, 0), UOp.range(n, 1, axis_type=AxisType.LOCAL)
  raw = UOp.placeholder((n*n,), x.dtype, 0, addrspace=AddrSpace.LOCAL)
  solved = UOp.placeholder((n*n,), x.dtype, 1, addrspace=AddrSpace.LOCAL)
  ready = UOp.group(*(raw[row*n+lane].store(x.flatten()[outer*n*n+row*n+lane]) for row in range(n))).barrier()
  for row in range(n):
    base, previous = raw.after(ready), solved.after(ready)
    value = base[row*n+lane] + sum((base[row*n+i] * previous[i*n+lane] for i in range(row)), UOp.const(x.dtype, 0))
    ready = solved.after(ready)[row*n+lane].store((lane < row).where(value, UOp.const(x.dtype, 0))).barrier()
  result = solved.after(ready)
  stores = [out.flatten()[outer*n*n+row*n+lane].store(lane.eq(row).where(UOp.const(x.dtype, 1), result[row*n+lane]))
            for row in range(n)]
  return UOp.group(*stores).end(outer, lane).sink(arg=KernelInfo(name="inverse_unit_lower", opts_to_apply=()))

def inverse_unit_lower(x:Tensor) -> Tensor:
  """Reference-ordered inverse of I-x for a strictly lower-triangular x."""
  n = x.shape[-1]
  assert isinstance(n, int)
  if n == 64 and str(x.device).startswith("AMD"):
    out = Tensor.empty(*x.shape, dtype=x.dtype, device=x.device)
    return Tensor.custom_kernel(out, x, fxn=lambda out,x:_inverse_unit_lower_kernel(out, x, n))[0]
  rows = [x[..., 0, :].const_like(0)]
  for i in range(1, n):
    prefix = x[..., i, :i]
    previous = Tensor.stack(*rows, dim=-2)[..., :, :i]
    rows.append((prefix + (prefix.unsqueeze(-1) * previous).sum(-2)).pad((0, n-i)))
  return Tensor.stack(*rows, dim=-2) + Tensor.eye(n, dtype=x.dtype).to(x.device)
