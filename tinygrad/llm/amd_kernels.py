import functools

from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import KernelInfo, Ops, UOp

_DIM, _HIDDEN = 1024, 3584
_Q8_BLOCK = 32
_Q8_BLOCK_BYTES = 34


def _q8_bytes(numel:int) -> int:
  assert numel % _Q8_BLOCK == 0
  return (numel // _Q8_BLOCK) * _Q8_BLOCK_BYTES


def _find_raw_q8_blocks(weight:Tensor) -> Tensor|None:
  seen:set[int] = set()
  stack = [weight.uop]
  while stack:
    u = stack.pop()
    if id(u) in seen: continue
    seen.add(id(u))
    if u.op is Ops.CONTIGUOUS and u.dtype.scalar() == dtypes.uchar:
      return Tensor(u)
    stack.extend(u.src)
  return None


_GATE_UP_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int DIM = 1024;
constexpr int HIDDEN = 3584;
constexpr int THREADS = 32;
constexpr int ROWS_PER_GROUP = 1;
constexpr float RMS_EPS = 1.0e-6f;

extern "C" __global__ __launch_bounds__(THREADS) void fused_gate_up_q8(
    float* __restrict__ z,
    const float* __restrict__ x_norm,
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w) {
  int tid = threadIdx.x;
  int row = blockIdx.x;

  int block = tid;
  int base = (row * (DIM / 32) + block) * 34;
  const unsigned char* gb = gate_w + base;
  const unsigned char* ub = up_w + base;
  float gs = float(*reinterpret_cast<const _Float16*>(gb));
  float us = float(*reinterpret_cast<const _Float16*>(ub));
  float gacc = 0.0f, uacc = 0.0f;
  #pragma unroll
  for (int offset = 0; offset < 32; offset++) {
    int i = block * 32 + offset;
    float x = x_norm[i];
    gacc += float(*reinterpret_cast<const int8_t*>(gb + 2 + offset)) * gs * x;
    uacc += float(*reinterpret_cast<const int8_t*>(ub + 2 + offset)) * us * x;
  }
  #pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    gacc += __shfl_down(gacc, delta, 32);
    uacc += __shfl_down(uacc, delta, 32);
  }
  if (tid == 0) {
    z[row] = (gacc / (1.0f + exp2f(-1.4426950408889634f * gacc))) * uacc;
  }
}
"""


@functools.cache
def _compiled_gate_up() -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(getenv("AMD_ARCH", "gfx1100"), []).compile_cached(_GATE_UP_HIP_SRC)


def _gate_up_kernel(z:UOp, x_norm:UOp, gate_w:UOp, up_w:UOp) -> UOp:
  assert z.numel() == _HIDDEN, f"fused_gate_up_q8 expects hidden={_HIDDEN}, got {z.numel()}"
  ops = 4 * _HIDDEN * _DIM + 2 * _HIDDEN
  mem = (_DIM + _HIDDEN) * 4 + 2 * _q8_bytes(_HIDDEN * _DIM)
  sink = UOp.sink(
    UOp.special(_HIDDEN, "gidx0"), UOp.special(32, "lidx0"),
    z, x_norm, gate_w, up_w,
    arg=KernelInfo(name="fused_gate_up_q8", estimates=Estimates(ops=ops, mem=mem)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_GATE_UP_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_gate_up())))


def fused_gate_up(x_norm:Tensor, gate_w:Tensor, up_w:Tensor) -> Tensor:
  assert x_norm.numel() == _DIM, f"fused_gate_up only supports dim={_DIM}, got {x_norm.numel()}"
  gate_raw = _find_raw_q8_blocks(gate_w)
  up_raw = _find_raw_q8_blocks(up_w)
  if gate_raw is None or up_raw is None:
    raise NotImplementedError("fused_gate_up: all weights must trace to a raw uchar buffer")
  z = Tensor.empty(_HIDDEN, dtype=dtypes.float, device=x_norm.device)
  z, *_ = Tensor.custom_kernel(z, x_norm.reshape(-1), gate_raw, up_raw, fxn=_gate_up_kernel)
  return z.reshape(*x_norm.shape[:-1], _HIDDEN)
