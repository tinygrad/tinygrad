import functools

from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import KernelInfo, Ops, UOp

_DIM, _HIDDEN = 1024, 3584
_GDN_HV, _GDN_V, _GDN_K = 16, 128, 128
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
  # RDNA3 supports wave32 and wave64. This kernel is one 32-lane reduction per
  # output row, so forcing wave32 avoids issuing each VALU instruction twice.
  return HIPCCCompiler(getenv("AMD_ARCH", "gfx1100"), ["-mno-wavefrontsize64"]).compile_cached(_GATE_UP_HIP_SRC)


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


_GDN_RECURRENT_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int HV = 16;
constexpr int V = 128;
constexpr int K = 128;
constexpr int LANES_PER_ROW = 16;
constexpr int ELEMS_PER_LANE = 8;
constexpr int THREADS = 128;
constexpr int GROUPS = THREADS / LANES_PER_ROW;
constexpr int ILP_ROWS = 4;
constexpr int ROWS_PER_BLOCK = GROUPS * ILP_ROWS;

extern "C" __global__ __launch_bounds__(THREADS) void gdn_recurrent_update(
    float* __restrict__ core_out,
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha,
    const _Float16* __restrict__ beta) {
  int tid = threadIdx.x;
  int lane = tid & (LANES_PER_ROW - 1);
  int group = tid >> 4;
  int hv = blockIdx.x;
  int row_base = blockIdx.y * ROWS_PER_BLOCK + group * ILP_ROWS;
  int k_base = lane * ELEMS_PER_LANE;

  float qv[ELEMS_PER_LANE], kv[ELEMS_PER_LANE], h[ILP_ROWS][ELEMS_PER_LANE];
  #pragma unroll
  for (int i = 0; i < ELEMS_PER_LANE; i++) {
    int kk = k_base + i;
    qv[i] = q[hv * K + kk];
    kv[i] = k[hv * K + kk];
  }

  float a = alpha[hv];
  float b = float(beta[hv]);
  float dot_hk[ILP_ROWS];
  #pragma unroll
  for (int r = 0; r < ILP_ROWS; r++) dot_hk[r] = 0.0f;

  #pragma unroll
  for (int r = 0; r < ILP_ROWS; r++) {
    int row = row_base + r;
    int base = (hv * V + row) * K + k_base;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; i++) {
      float hvv = state[base + i] * a;
      h[r][i] = hvv;
      dot_hk[r] += hvv * kv[i];
    }
  }

  #pragma unroll
  for (int delta = 8; delta > 0; delta >>= 1) {
    #pragma unroll
    for (int r = 0; r < ILP_ROWS; r++) dot_hk[r] += __shfl_xor(dot_hk[r], delta, LANES_PER_ROW);
  }

  float dot_hq[ILP_ROWS];
  #pragma unroll
  for (int r = 0; r < ILP_ROWS; r++) {
    int row = row_base + r;
    float vn = (v[hv * V + row] - dot_hk[r]) * b;
    dot_hq[r] = 0.0f;
    int base = (hv * V + row) * K + k_base;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; i++) {
      float updated = h[r][i] + kv[i] * vn;
      state[base + i] = updated;
      dot_hq[r] += updated * qv[i];
    }
  }

  #pragma unroll
  for (int delta = 8; delta > 0; delta >>= 1) {
    #pragma unroll
    for (int r = 0; r < ILP_ROWS; r++) dot_hq[r] += __shfl_xor(dot_hq[r], delta, LANES_PER_ROW);
  }

  if (lane == 0) {
    #pragma unroll
    for (int r = 0; r < ILP_ROWS; r++) core_out[hv * V + row_base + r] = dot_hq[r];
  }
}
"""


@functools.cache
def _compiled_gdn_recurrent() -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(getenv("AMD_ARCH", "gfx1100"), ["-mno-wavefrontsize64"]).compile_cached(_GDN_RECURRENT_HIP_SRC)


def _gdn_recurrent_kernel(core:UOp, state:UOp, q:UOp, k:UOp, v:UOp, alpha:UOp, beta:UOp) -> UOp:
  assert core.numel() == _GDN_HV * _GDN_V, f"gdn_recurrent_update expects core size 2048, got {core.numel()}"
  assert state.numel() == _GDN_HV * _GDN_V * _GDN_K, f"gdn_recurrent_update expects state size 262144, got {state.numel()}"
  ops = _GDN_HV * _GDN_V * _GDN_K * 4
  mem = (2 * _GDN_HV * _GDN_V * _GDN_K + 3 * _GDN_HV * _GDN_K + 2 * _GDN_HV * _GDN_V) * 4
  sink = UOp.sink(
    UOp.special(_GDN_HV, "gidx0"), UOp.special(_GDN_V // 32, "gidx1"), UOp.special(128, "lidx0"),
    core, state, q, k, v, alpha, beta,
    arg=KernelInfo(name="gdn_recurrent_update", estimates=Estimates(ops=ops, mem=mem)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_GDN_RECURRENT_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_gdn_recurrent())))


def gdn_recurrent_update(state:Tensor, q:Tensor, k:Tensor, v:Tensor, alpha:Tensor, beta:Tensor) -> Tensor:
  assert state.numel() == _GDN_HV * _GDN_V * _GDN_K, f"gdn_recurrent_update only supports state 16x128x128, got {state.shape}"
  assert q.numel() == _GDN_HV * _GDN_K and k.numel() == _GDN_HV * _GDN_K, f"gdn_recurrent_update q/k shape mismatch: {q.shape} {k.shape}"
  assert v.numel() == _GDN_HV * _GDN_V, f"gdn_recurrent_update v shape mismatch: {v.shape}"
  core = Tensor.empty(_GDN_HV, _GDN_V, dtype=dtypes.float, device=state.device)
  core, *_ = Tensor.custom_kernel(core, state, q.reshape(-1), k.reshape(-1), v.reshape(-1), alpha.reshape(-1), beta.reshape(-1), fxn=_gdn_recurrent_kernel)
  return core.reshape(1, 1, _GDN_HV, _GDN_V)


_GDN_RECURRENT_CONV_HIP_SRC = _GDN_RECURRENT_HIP_SRC.replace(
"""extern "C" __global__ __launch_bounds__(THREADS) void gdn_recurrent_update(
    float* __restrict__ core_out,
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha,
    const _Float16* __restrict__ beta) {
  int tid = threadIdx.x;
  int lane = tid & (LANES_PER_ROW - 1);
  int group = tid >> 4;
  int hv = blockIdx.x;
  int row_base = blockIdx.y * ROWS_PER_BLOCK + group * ILP_ROWS;
  int k_base = lane * ELEMS_PER_LANE;

  float qv[ELEMS_PER_LANE], kv[ELEMS_PER_LANE], h[ILP_ROWS][ELEMS_PER_LANE];
  #pragma unroll
  for (int i = 0; i < ELEMS_PER_LANE; i++) {
    int kk = k_base + i;
    qv[i] = q[hv * K + kk];
    kv[i] = k[hv * K + kk];
  }
""",
"""extern "C" __global__ __launch_bounds__(THREADS) void gdn_recurrent_update_conv(
    float* __restrict__ core_out,
    float* __restrict__ state,
    const float* __restrict__ conv_out,
    const float* __restrict__ alpha,
    const _Float16* __restrict__ beta) {
  int tid = threadIdx.x;
  int lane = tid & (LANES_PER_ROW - 1);
  int group = tid >> 4;
  int hv = blockIdx.x;
  int row_base = blockIdx.y * ROWS_PER_BLOCK + group * ILP_ROWS;
  int k_base = lane * ELEMS_PER_LANE;

  float qv[ELEMS_PER_LANE], kv[ELEMS_PER_LANE], h[ILP_ROWS][ELEMS_PER_LANE];
  float qsum = 0.0f, ksum = 0.0f;
  #pragma unroll
  for (int i = 0; i < ELEMS_PER_LANE; i++) {
    int kk = k_base + i;
    float qraw = conv_out[hv * K + kk];
    float kraw = conv_out[HV * K + hv * K + kk];
    qv[i] = qraw;
    kv[i] = kraw;
    qsum += qraw * qraw;
    ksum += kraw * kraw;
  }
  #pragma unroll
  for (int delta = 8; delta > 0; delta >>= 1) {
    qsum += __shfl_xor(qsum, delta, LANES_PER_ROW);
    ksum += __shfl_xor(ksum, delta, LANES_PER_ROW);
  }
  float qscale = rsqrtf(qsum) * 0.08838834764831845f;
  float kscale = rsqrtf(ksum);
  #pragma unroll
  for (int i = 0; i < ELEMS_PER_LANE; i++) {
    qv[i] *= qscale;
    kv[i] *= kscale;
  }
""").replace("v[hv * V + row]", "conv_out[2 * HV * K + hv * V + row]")


@functools.cache
def _compiled_gdn_recurrent_conv() -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(getenv("AMD_ARCH", "gfx1100"), ["-mno-wavefrontsize64"]).compile_cached(_GDN_RECURRENT_CONV_HIP_SRC)


def _gdn_recurrent_conv_kernel(core:UOp, state:UOp, conv_out:UOp, alpha:UOp, beta:UOp) -> UOp:
  assert core.numel() == _GDN_HV * _GDN_V and state.numel() == _GDN_HV * _GDN_V * _GDN_K and conv_out.numel() == 3 * _GDN_HV * _GDN_K
  ops = _GDN_HV * _GDN_V * _GDN_K * 4
  mem = (2 * _GDN_HV * _GDN_V * _GDN_K + 3 * _GDN_HV * _GDN_K + _GDN_HV * _GDN_V) * 4
  sink = UOp.sink(
    UOp.special(_GDN_HV, "gidx0"), UOp.special(_GDN_V // 32, "gidx1"), UOp.special(128, "lidx0"),
    core, state, conv_out, alpha, beta,
    arg=KernelInfo(name="gdn_recurrent_update_conv", estimates=Estimates(ops=ops, mem=mem)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_GDN_RECURRENT_CONV_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_gdn_recurrent_conv())))


def gdn_recurrent_update_conv(state:Tensor, conv_out:Tensor, alpha:Tensor, beta:Tensor) -> Tensor:
  assert state.numel() == _GDN_HV * _GDN_V * _GDN_K and conv_out.numel() == 3 * _GDN_HV * _GDN_K
  core = Tensor.empty(_GDN_HV, _GDN_V, dtype=dtypes.float, device=state.device)
  core, *_ = Tensor.custom_kernel(core, state, conv_out.reshape(-1), alpha.reshape(-1), beta.reshape(-1), fxn=_gdn_recurrent_conv_kernel)
  return core.reshape(1, 1, _GDN_HV, _GDN_V)
