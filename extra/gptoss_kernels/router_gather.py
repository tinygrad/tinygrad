"""Fuse the exact GPT-OSS inverse-dispatch gather into the router input gradient."""
from dataclasses import dataclass,replace
import functools
import hashlib
from pathlib import Path

from tinygrad import Tensor,dtypes
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from tinygrad.uop.ops import UOp,Ops,KernelInfo,ProgramInfo,CallInfo
from extra.hipcc import hip_program

ROOT = Path(__file__).resolve().parents[2]
KERNEL_FILE = ROOT/'extra/thunder/amd/moe_router_dgrad_gather.cpp'
GATHER_SOURCE_SHA = '4360f226c16d5481ce9d0e06d319345f0dbff4d1a36732d79ddd93ce21935987'
GATHER_BINARY_SHA = '1ea9f86bb336035b92468cf401882073a50ef9cb406bd09fe578f75aeea8f9fb'


def sha(data): return hashlib.sha256(data).hexdigest()


MARKER = ('gptoss_inverse_dispatch_gather', 1)


def fresh_invalid_output(output: UOp) -> bool:
  if output.base.is_invalid: return True
  if output.op in (Ops.RESHAPE, Ops.UNSHARD): return fresh_invalid_output(output.src[0])
  if output.op is not Ops.AFTER or len(output.src) != 2: return False
  destination, initialization = output.src
  base = destination
  while base.op in (Ops.RESHAPE, Ops.UNSHARD): base = base.src[0]
  return (base.op is Ops.ALLOC and initialization.op is Ops.STORE and len(initialization.src) == 2
          and initialization.src[0] is destination and fresh_invalid_output(initialization.src[1]))


@dataclass(frozen=True)
class GatherInputs:
  table: UOp
  indices: UOp
  output: UOp
  call: UOp


def mark_gather(output: UOp) -> UOp:
  """Only the existing grouped_gather_sum_rows factory may attach this marker, before returning its result."""
  assert output.op is Ops.AFTER and len(output.src) == 2
  call = output.src[1]
  assert call.op is Ops.CALL and len(call.src) == 4 and call.src[1] is output.src[0]
  assert call.arg.aux is None and call.arg.grad_fxn is None
  assert fresh_invalid_output(output.src[0])
  # Keep the original body and every original argument, including all input ordering dependencies.
  return output.replace(src=(output.src[0], call.replace(arg=replace(call.arg, aux=(*MARKER, call.body.key)))))


def gather_inputs(gradient: UOp, *, tokens: int = 16384, rows: int = 73728, width: int = 3072) -> GatherInputs | None:
  """Recognize only a full element-preserving reshape of the marked producer's sole output.

  No AFTER dependency, arithmetic, cast, slice, permute, or unshard operation is stripped.
  Shapes are parameterized solely for tiny CPU graph tests; the proposed production caller uses the defaults.
  """
  node = gradient
  while node.op is Ops.RESHAPE:
    if node.numel() != node.src[0].numel() or node.dtype != node.src[0].dtype: return None
    node = node.src[0]
  if node.op is not Ops.AFTER or len(node.src) != 2: return None
  call = node.src[1]
  if call.op is not Ops.CALL or call.arg.aux != (*MARKER, call.body.key) or len(call.src) != 4: return None
  output, table, indices = call.src[1:]
  if node.src[0] is not output or call.arg.grad_fxn is not None: return None
  if not fresh_invalid_output(output): return None
  if not (gradient.dtype == output.dtype == table.dtype == dtypes.bfloat16 and indices.dtype == dtypes.int32): return None
  if not (gradient.device == output.device == table.device == indices.device): return None
  groups = len(table.device) if isinstance(table.device, tuple) else 1
  if (table.shape, indices.shape, output.shape) != ((groups, rows, width), (groups, tokens * 4), (groups, tokens, width)): return None
  if isinstance(table.device, tuple) and not (table.axis == indices.axis == output.axis == gradient.axis == 0): return None
  if (table.shard_shape, indices.shard_shape, output.shard_shape) != ((1, rows, width), (1, tokens * 4), (1, tokens, width)): return None
  if gradient.shard_shape != (tokens, width): return None
  if output.storage_base in (table.storage_base, indices.storage_base): return None
  return GatherInputs(table, indices, output, call)


@functools.cache
def router_gather_kernel(*args: UOp) -> UOp:
  """Use the six actual supplied custom-kernel parameters; never manufacture replacement argument identities."""
  assert len(args) == 6
  shapes = [((16384,2880),dtypes.bfloat16),((32,2880),dtypes.bfloat16),((16384,32),dtypes.float32),
            ((1,73728,3072),dtypes.bfloat16),((1,65536),dtypes.int32),((16384,96),dtypes.uint8)]
  assert [(a.shape,a.dtype) for a in args] == shapes, 'local argument shape/dtype mismatch'
  assert all(a.base.op is Ops.PARAM and a.base.arg.slot == i for i, a in enumerate(args)), 'argument slot mismatch'
  out, weight, gradient, table, indices, e8 = args
  zero = UOp.const(0, dtypes.int32)
  accesses = (out.base.index(zero).store(UOp.const(0, out.dtype)),
              *(u.base.index(zero).load() for u in (weight, gradient, table, indices, e8)))
  elems = 16384 * 2880
  sink = UOp.sink(*accesses, UOp.special(16, 'lidx0'), UOp.special(4, 'lidx1'),
                  UOp.special(9, 'gidx0'), UOp.special(2048, 'gidx1'),
                  arg=KernelInfo('moe_router_dgrad_gather', estimates=Estimates(ops=2*elems*32 + 5*elems,
                    mem=5*elems*2 + 16384*32*4 + 32*2880*2 + 16384*90 + 16384*4*4)))
  source = KERNEL_FILE.read_text()
  binary = HIPCompiler('gfx950').compile_cached(source)
  return hip_program(sink, source, binary)


def native_gather(body: UOp) -> bool:
  """Exact saved native gather only. In particular, NULL/CPU generated reductions are not this ordered HIP kernel."""
  if body.op is not Ops.PROGRAM or len(body.src) != 4: return False
  sink, linear, source, binary = body.src
  if source.op is not Ops.SOURCE or binary.op is not Ops.BINARY or not isinstance(source.arg, str) or not isinstance(binary.arg, bytes):
    return False
  if sha(source.arg.encode()) != GATHER_SOURCE_SHA or sha(binary.arg) != GATHER_BINARY_SHA: return False
  if sink.op is not Ops.SINK or linear.op is not Ops.LINEAR or linear.src != (*sink.src, sink): return False
  params = tuple(UOp.placeholder(shape, dtype, slot).base for slot, (shape, dtype) in enumerate([
    ((1, 16384, 3072), dtypes.bfloat16), ((1, 73728, 3072), dtypes.bfloat16), ((1, 65536), dtypes.int32)]))
  if len(sink.src) != 5 or sink.src[:3] != params: return False
  if sink.src[3:] != (UOp.special(64, 'lidx0'), UOp.special(16384, 'gidx0')): return False
  info = ProgramInfo.from_sink(sink)
  return (info.global_size == (16384, 1, 1) and info.local_size == (64, 1, 1) and not info.vars
          and info.globals == info.ins == info.outs == (0, 1, 2))


def mark_native(result: Tensor) -> Tensor:
  groups = len(result.device) if isinstance(result.device, tuple) else 1
  if result.shape != (groups, 16384, 3072): return result
  if result.uop.op is not Ops.AFTER or len(result.uop.src) != 2: return result
  call = result.uop.src[1]
  if result.uop.src[0].storage_base.op is not Ops.ALLOC: return result
  if call.op is not Ops.CALL or not native_gather(call.body) or call.arg != CallInfo(): return result
  # Marking validates fresh scratch and absence of extra CALL effects. An unknown producer is unchanged.
  try: marked = mark_gather(result.uop)
  except AssertionError: return result
  if gather_inputs(marked.reshape(-1, 3072)) is None: return result
  return Tensor(marked)


def compatible(weight: Tensor, gradient: Tensor, residual: Tensor, e8: Tensor, shape: tuple[int, ...]) -> bool:
  dev = gradient.device
  groups = len(dev) if isinstance(dev, tuple) else 1
  if any(t.device != dev for t in (weight, residual, e8)): return False
  if tuple(shape) != (groups*16384, 2880): return False
  if [(t.uop.shard_shape, t.dtype) for t in (weight, gradient, residual, e8)] != [
      ((32, 2880), dtypes.bfloat16), ((16384, 32), dtypes.float32), ((16384, 3072), dtypes.bfloat16),
      ((16384, 96), dtypes.uint8)]: return False
  return not isinstance(dev, tuple) or (weight.uop.axis is None and gradient.uop.axis == residual.uop.axis == e8.uop.axis == 0)


def gathered_router(out:Tensor,weight:Tensor,gradient:Tensor,residual:Tensor,e8:Tensor,shape:tuple[int,...]) -> Tensor|None:
  hit = gather_inputs(residual.uop)
  if hit is None or hit.output.storage_base.op is not Ops.ALLOC or hit.call.arg != CallInfo(aux=hit.call.arg.aux) \
     or not native_gather(hit.call.body) or not compatible(weight,gradient,residual,e8,shape): return None
  return Tensor.custom_kernel(out,weight,gradient,Tensor(hit.table),Tensor(hit.indices),e8,fxn=router_gather_kernel)[0]
