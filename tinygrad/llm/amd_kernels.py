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
constexpr int THREADS = 256;
constexpr int ROWS_PER_GROUP = 8;
constexpr float RMS_EPS = 1.0e-6f;

__device__ inline float half_bits_to_float(unsigned short h) {
  unsigned int sign = h >> 15;
  unsigned int exp = (h >> 10) & 31u;
  unsigned int mant = h & 1023u;
  float v = exp == 0 ? float(mant) * 5.960464477539063e-8f : (1.0f + float(mant) * 0.0009765625f) * exp2f(float(int(exp) - 15));
  return sign ? -v : v;
}

extern "C" __global__ __launch_bounds__(THREADS) void fused_gate_up_q8(
    float* __restrict__ z,
    const float* __restrict__ x_norm,
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w) {
  __shared__ float red_gate[THREADS];
  __shared__ float red_up[THREADS];

  int tid = threadIdx.x;
  int wave = tid >> 6;
  int lane = tid & 63;
  int half = lane >> 5;
  int lane32 = lane & 31;
  int row = blockIdx.x * ROWS_PER_GROUP + wave * 2 + half;

  int block = lane32;
  int base = (row * (DIM / 32) + block) * 34;
  const unsigned char* gb = gate_w + base;
  const unsigned char* ub = up_w + base;
  unsigned short gh = (unsigned short)(gb[0]) | ((unsigned short)(gb[1]) << 8);
  unsigned short uh = (unsigned short)(ub[0]) | ((unsigned short)(ub[1]) << 8);
  float gs = half_bits_to_float(gh);
  float us = half_bits_to_float(uh);
  float gacc = 0.0f, uacc = 0.0f;
  #pragma unroll
  for (int offset = 0; offset < 32; offset++) {
    int i = block * 32 + offset;
    const unsigned char* gb = gate_w + base;
    const unsigned char* ub = up_w + base;
    float x = x_norm[i];
    gacc += float(*reinterpret_cast<const int8_t*>(gb + 2 + offset)) * gs * x;
    uacc += float(*reinterpret_cast<const int8_t*>(ub + 2 + offset)) * us * x;
  }
  red_gate[tid] = gacc;
  red_up[tid] = uacc;
  __syncthreads();

  if (lane32 == 0) {
    float g = 0.0f, u = 0.0f;
    int base_tid = (tid >> 5) << 5;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
      g += red_gate[base_tid + i];
      u += red_up[base_tid + i];
    }
    z[row] = (g / (1.0f + expf(-g))) * u;
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
    UOp.special(_HIDDEN//8, "gidx0"), UOp.special(256, "lidx0"),
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
