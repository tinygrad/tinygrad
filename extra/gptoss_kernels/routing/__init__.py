"""Stable forward/inverse routing maps for GPT-OSS's 16,384-token AMD shards."""
import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.renderer import Estimates
from extra.hipcc import HIPCCCompiler, hip_program
from extra.llama_kernels import alloc_like, owned_empty

_SHAPES = ((65536,), (256, 32), (32,), (33,), (73728,))

@functools.cache
def _routing_maps_kernel(*args:UOp, phase:int) -> UOp:
  shapes, writes, workgroups, name = (
    ((_SHAPES[0], _SHAPES[1], _SHAPES[0]), 2, 256, "histogram"),
    ((_SHAPES[1], _SHAPES[2], _SHAPES[3], _SHAPES[1]), 3, 1, "scan"),
    ((_SHAPES[0], _SHAPES[4], _SHAPES[0], _SHAPES[0], _SHAPES[1], _SHAPES[2], _SHAPES[3]), 2, 288, "scatter"),
  )[phase]
  assert len(args) == len(shapes) and all(a.shard_shape == (1, *s) and a.dtype == dtypes.int32 for a, s in zip(args, shapes))
  zero = UOp.const(0, dtypes.int32)
  access = tuple(a.flatten().index(zero).store(UOp.const(0, a.dtype)) if i < writes else a.flatten().index(zero).load()
                 for i, a in enumerate(args))
  sink = UOp.sink(*(a.base for a in args), *access, UOp.special(1024 if phase == 1 else 256, "lidx0"), UOp.special(workgroups, "gidx0"),
    arg=KernelInfo("gptoss_route_"+name, estimates=Estimates(mem=sum(math.prod(a.shard_shape)*a.dtype.itemsize for a in args))))
  source = (pathlib.Path(__file__).parent/"route.cpp").read_text()
  # Keep the toolchain whose three images passed the exact integer/replay checks.
  binary = HIPCCCompiler("gfx950", ["-std=c++20", f"-DROUTE_PHASE={phase}", "-O3"],
    hipcc_path="/opt/rocm-7.2.1/bin/hipcc", rocm_path="/opt/rocm-7.2.1").compile_cached(source)
  return hip_program(sink, source, binary)

def routing_maps(topi:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  """Return destination rows, padded inverse rows, expert counts and aligned offsets."""
  assert topi.dtype == dtypes.int32 and topi.uop.shard_shape == (1, 16384, 4)
  groups = len(topi.device) if isinstance(topi.device, tuple) else 1
  assert topi.shape == (groups, 16384, 4) and topi.uop.axis == (0 if groups > 1 else None)
  def output(shape): return owned_empty(alloc_like((groups, *shape), dtypes.int32, topi.device, 0 if groups > 1 else None))
  flattened = topi.reshape(groups, 65536)
  ranks, histogram, *_ = Tensor.custom_kernel(output(_SHAPES[0]), output(_SHAPES[1]), flattened,
    fxn=functools.partial(_routing_maps_kernel, phase=0))
  prefixes, counts, off, *_ = Tensor.custom_kernel(output(_SHAPES[1]), output(_SHAPES[2]), output(_SHAPES[3]), histogram,
    fxn=functools.partial(_routing_maps_kernel, phase=1))
  dest, source, *_ = Tensor.custom_kernel(output(_SHAPES[0]), output(_SHAPES[4]), flattened, ranks, prefixes, counts, off,
    fxn=functools.partial(_routing_maps_kernel, phase=2))
  return dest, source, counts, off
