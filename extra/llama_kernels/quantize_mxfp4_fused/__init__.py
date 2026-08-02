import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.helpers import ceildiv
from extra.llama_kernels import alloc_like, compile_hip

BLK = 32
LOG2E = 1.4426950408889634

def _amd_cast_transpose_src() -> str:
  # Keep AMD's submitted kernel body verbatim, replacing only its templated launch signature with tinygrad's fixed entry point.
  src = (pathlib.Path(__file__).parent/"cast_transpose_mxfp4_shuffled.hip").read_text()
  start = src.index("template<")
  body = src.index(") {", start) + 3
  end = src.index("\n}\n\n}  // namespace te_mxfp4", body)
  entry = '''extern "C" __global__ __launch_bounds__(256, 8)
void quantize_mxfp4_dual(uint8_t* __restrict__ rowwise_fp4, uint8_t* __restrict__ rowwise_scale,
                         uint8_t* __restrict__ colwise_fp4, uint8_t* __restrict__ colwise_scale,
                         const uint16_t* __restrict__ input) {
    constexpr bool USE_ROWWISE = true;
    constexpr bool USE_COLWISE = true;
    constexpr bool SHUFFLE_SCALES = SHUFFLE_SCALES_VALUE;
    constexpr bool USE_HADAMARD = USE_HADAMARD_VALUE;
    constexpr bool SHUFFLE_ROWWISE_FP4 = SHUFFLE_ROWWISE_FP4_VALUE;
    constexpr bool SHUFFLE_COLWISE_FP4 = SHUFFLE_COLWISE_FP4_VALUE;
    constexpr int M = M_DIM, N = N_DIM;
    constexpr int rowwise_scale_stride = N_DIM / 32;
    constexpr int colwise_scale_stride = M_DIM / 32;
    constexpr int rowwise_scale_N = N_DIM / 32;
    constexpr int rowwise_scale_M_pad = M_DIM;
    constexpr int rowwise_scale_N_pad = N_DIM / 32;
    constexpr int colwise_scale_M = N_DIM;
    constexpr int colwise_scale_N = M_DIM / 32;
    constexpr int colwise_scale_M_pad = N_DIM;
    constexpr int colwise_scale_N_pad = M_DIM / 32;
'''
  return src[:start] + entry + src[body:end] + "\n}\n\n}  // namespace te_mxfp4\n#endif\n"

@functools.cache
def _custom_quantize_mxfp4_dual(row_q:UOp, row_s:UOp, col_q:UOp, col_s:UOp, x:UOp,
                                use_hadamard:bool, shuffle_row:bool, shuffle_col:bool, shuffle_scales:bool) -> UOp:
  M, N = x.shape
  assert M % 256 == 0 and N % 256 == 0, f"AMD MXFP4 cast-transpose requires multiples of 256, got {x.shape}"
  threads = UOp.special(256, "lidx0")
  groups_m, groups_n = UOp.special(ceildiv(M, 128), "gidx0"), UOp.special(ceildiv(N, 64), "gidx1")
  mem = M*N*2 + M*N + M*N//16
  sink = UOp.sink(row_q.base, row_s.base, col_q.base, col_s.base, x.base, threads, groups_m, groups_n,
                  arg=KernelInfo(f"quantize_mxfp4_dual_{M}_{N}", estimates=Estimates(ops=M*N, mem=mem)))
  src = _amd_cast_transpose_src()
  defines = [f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DUSE_HADAMARD_VALUE={'true' if use_hadamard else 'false'}",
             f"-DSHUFFLE_ROWWISE_FP4_VALUE={'true' if shuffle_row else 'false'}",
             f"-DSHUFFLE_COLWISE_FP4_VALUE={'true' if shuffle_col else 'false'}",
             f"-DSHUFFLE_SCALES_VALUE={'true' if shuffle_scales else 'false'}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def quantize_mxfp4_dual(x:Tensor, *, use_hadamard:bool=True, shuffle_row:bool=False, shuffle_col:bool=False,
                        shuffle_scales:bool=True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  assert x.dtype == dtypes.bfloat16 and x.ndim == 2, f"expected BF16 matrix, got {x.dtype} {x.shape}"
  M, N = x.shape
  assert M % 256 == 0 and N % 256 == 0, f"AMD MXFP4 cast-transpose requires multiples of 256, got {x.shape}"
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  col_axis = None if axis is None else 1-axis
  row_q = alloc_like((M, N//2), dtypes.uint8, x.device, axis)
  row_s = alloc_like((M, N//BLK), dtypes.uint8, x.device, axis)
  col_q = alloc_like((N, M//2), dtypes.uint8, x.device, col_axis)
  col_s = alloc_like((N, M//BLK), dtypes.uint8, x.device, col_axis)
  fxn = functools.partial(_custom_quantize_mxfp4_dual, use_hadamard=use_hadamard, shuffle_row=shuffle_row,
                          shuffle_col=shuffle_col, shuffle_scales=shuffle_scales)
  row_q, row_s, col_q, col_s, *_ = Tensor.custom_kernel(row_q, row_s, col_q, col_s, x.contiguous(), fxn=fxn)
  return row_q, row_s, col_q, col_s

def _e2m1_code(x:UOp) -> UOp:
  mag = x.abs()
  code = ((mag > .25).cast(dtypes.uint8) + (mag >= .75).cast(dtypes.uint8) +
          (mag > 1.25).cast(dtypes.uint8) + (mag >= 1.75).cast(dtypes.uint8) +
          (mag > 2.5).cast(dtypes.uint8) + (mag >= 3.5).cast(dtypes.uint8) +
          (mag > 5.0).cast(dtypes.uint8))
  return code | ((x < 0).cast(dtypes.uint8) << 3)

@functools.cache
def _custom_quantize_mxfp4(packed_out:UOp, e8_out:UOp, x:UOp) -> UOp:
  n_elems = math.prod(x.shape)
  x, packed_out, e8_out = x.reshape(n_elems), packed_out.reshape(n_elems//2), e8_out.reshape(n_elems//BLK)
  block = UOp.range(n_elems//BLK, 0)
  vals = [x[block*BLK+i].cast(dtypes.float) for i in range(BLK)]
  amax = functools.reduce(lambda a,b: a.maximum(b), (v.abs() for v in vals))
  amax_rounded = ((amax.bitcast(dtypes.uint32) + 0x200000) & 0xFF800000).bitcast(dtypes.float)
  scale_exp = amax_rounded.maximum(2**-126).log2().floor().sub(2).maximum(-127).minimum(127)
  qscale = (-scale_exp).exp2()

  packed_store = None
  for i in range(BLK//2):
    packed = _e2m1_code(vals[i*2] * qscale) | (_e2m1_code(vals[i*2+1] * qscale) << 4)
    packed_store = (packed_out if packed_store is None else packed_out.after(packed_store))[block*(BLK//2)+i].store(packed)
  assert packed_store is not None
  e8_store = e8_out.after(packed_store)[block].store((scale_exp + 127).cast(dtypes.uint8))
  return e8_store.end(block).sink(arg=KernelInfo(f"quantize_mxfp4_{n_elems}"))

def quantize_mxfp4_fused(x:Tensor, packed_shape:tuple[int, ...]|None=None, packed_axis:int|None=None) -> tuple[Tensor, Tensor]:
  assert x.dtype == dtypes.bfloat16 and x.ndim >= 2, f"expected bf16 with ndim >= 2, got {x.dtype} {x.shape}"
  *batch, K = x.shape
  rows = math.prod(batch)
  assert K % 256 == 0 and rows % 32 == 0, f"mxfp4 quantization needs rows%32 and K%256, got {x.shape}"
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  packed = alloc_like(packed_shape or (*batch, K//2), dtypes.uint8, x.device, packed_axis if packed_shape is not None else axis)
  e8 = alloc_like((*batch, K//BLK), dtypes.uint8, x.device, axis)
  packed, e8, *_ = Tensor.custom_kernel(packed, e8, x, fxn=_custom_quantize_mxfp4)
  return packed, e8

@functools.cache
def _custom_silu_mul_quantize_mxfp4(act_out:UOp, packed_out:UOp, scale_out:UOp, x_w13:UOp) -> UOp:
  *prefix, two_k = x_w13.shape
  rows, K = math.prod(prefix), two_k//2
  n_elems = rows*K

  x_w13, act_out = x_w13.reshape(rows*two_k), act_out.reshape(n_elems)
  packed_out, scale_out = packed_out.reshape(n_elems//2), scale_out.reshape(n_elems//BLK)
  block = UOp.range(n_elems//BLK, 0)
  out_idx, row = block*BLK, block*BLK//K
  col = out_idx%K
  acts = [x_w13[row*two_k+col+i] * (1.0 + (x_w13[row*two_k+col+i]*-LOG2E).exp2()).reciprocal() *
          x_w13[row*two_k+K+col+i] for i in range(BLK)]
  acts_f = [x.cast(dtypes.float) for x in acts]
  amax = functools.reduce(lambda a,b: a.maximum(b), (x.abs() for x in acts_f))
  amax_rounded = ((amax.bitcast(dtypes.uint32) + 0x200000) & 0xFF800000).bitcast(dtypes.float)
  scale_exp = amax_rounded.maximum(2**-126).log2().floor().sub(2).maximum(-127).minimum(127)
  qscale = (-scale_exp).exp2()

  store = None
  for i,act in enumerate(acts): store = (act_out if store is None else act_out.after(store))[out_idx+i].store(act)
  assert store is not None
  for i in range(BLK//2):
    packed = _e2m1_code(acts_f[i*2]*qscale) | (_e2m1_code(acts_f[i*2+1]*qscale) << 4)
    store = packed_out.after(store)[block*(BLK//2)+i].store(packed)
  scale_k = K//BLK
  scale_row, scale_col = block//scale_k, block%scale_k
  row_group, row_half, row_lane = scale_row//32, scale_row%32//16, scale_row%16
  col_group, col_half, col_lane = scale_col//8, scale_col%8//4, scale_col%4
  scale_idx = (((((row_group*(scale_k//8)+col_group)*4+col_lane)*16+row_lane)*2+col_half)*2+row_half)
  scale_store = scale_out.after(store)[scale_idx].store((scale_exp+127).cast(dtypes.uint8))
  return scale_store.end(block).sink(arg=KernelInfo(f"silu_mul_quantize_mxfp4_{n_elems}"))

@functools.cache
def _custom_silu_mul_bwd_mxfp4(grad_out:UOp, x_w13:UOp, grad_act:UOp) -> UOp:
  *prefix, two_k = x_w13.shape
  rows, K = math.prod(prefix), two_k//2
  n_elems = rows*K
  grad_out, x_w13, grad_act = grad_out.reshape(rows*two_k), x_w13.reshape(rows*two_k), grad_act.reshape(n_elems)
  idx = UOp.range(n_elems, 0)
  row, col = idx//K, idx%K

  g = grad_act[idx].cast(dtypes.float)
  w1 = x_w13[row*two_k+col].cast(dtypes.float)
  w3 = x_w13[row*two_k+K+col].cast(dtypes.float)
  sig = (1.0+(w1*-LOG2E).exp2()).reciprocal()
  grad_w1 = grad_out[row*two_k+col].store((g*sig*(1.0+w1*(1.0-sig))*w3).cast(dtypes.bfloat16))
  grad_w3 = grad_out.after(grad_w1)[row*two_k+K+col].store((g*w1*sig).cast(dtypes.bfloat16))
  return grad_w3.end(idx).sink(arg=KernelInfo(f"silu_mul_bwd_mxfp4_{n_elems}"))

def _silu_mul_quantize_mxfp4_bwd(gradient:UOp, kernel:UOp):
  _, _, _, x_w13 = kernel.src[1:]
  axis = x_w13.axis if isinstance(x_w13.device, tuple) else None
  grad_out = alloc_like(x_w13.shape, dtypes.bfloat16, x_w13.device, axis)
  grad_out, *_ = Tensor.custom_kernel(grad_out, Tensor(x_w13, device=x_w13.device), Tensor(gradient, device=x_w13.device),
                                      fxn=_custom_silu_mul_bwd_mxfp4)
  return (None, None, None, grad_out.uop)

def silu_mul_quantize_mxfp4(x_w13:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  assert x_w13.dtype == dtypes.bfloat16 and x_w13.ndim >= 2 and x_w13.shape[-1] % 512 == 0
  *prefix, two_k = x_w13.shape
  rows, K = math.prod(prefix), two_k//2
  axis = x_w13.uop.axis if isinstance(x_w13.device, tuple) else None
  act = alloc_like((*prefix, K), dtypes.bfloat16, x_w13.device, axis)
  packed = alloc_like((*prefix, K//2), dtypes.uint8, x_w13.device, axis)
  scale = alloc_like((rows, K//BLK), dtypes.uint8, x_w13.device, 0 if axis is not None else None)
  act, packed, scale, *_ = Tensor.custom_kernel(act, packed, scale, x_w13, fxn=_custom_silu_mul_quantize_mxfp4,
                                                 grad_fxn=_silu_mul_quantize_mxfp4_bwd)
  return act, packed, scale
