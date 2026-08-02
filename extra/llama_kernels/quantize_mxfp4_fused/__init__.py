import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.helpers import ceildiv
from extra.llama_kernels import NUM_WG, alloc_like, compile_hip

BLK = 32

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
  M, N = math.prod(x.shape[:-1]), x.shape[-1]
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
  assert x.dtype == dtypes.bfloat16 and x.ndim >= 2, f"expected BF16 matrix, got {x.dtype} {x.shape}"
  M, N = math.prod(x.shape[:-1]), x.shape[-1]
  assert M % 256 == 0 and N % 256 == 0, f"AMD MXFP4 cast-transpose requires multiples of 256, got {x.shape}"
  axis = 0 if isinstance(x.device, tuple) and x.uop.axis is not None else None
  col_axis = None if axis is None else 1-axis
  row_q = alloc_like((M, N//2), dtypes.uint8, x.device, axis)
  row_s = alloc_like((M, N//BLK), dtypes.uint8, x.device, axis)
  col_q = alloc_like((N, M//2), dtypes.uint8, x.device, col_axis)
  col_s = alloc_like((N, M//BLK), dtypes.uint8, x.device, col_axis)
  fxn = functools.partial(_custom_quantize_mxfp4_dual, use_hadamard=use_hadamard, shuffle_row=shuffle_row,
                          shuffle_col=shuffle_col, shuffle_scales=shuffle_scales)
  row_q, row_s, col_q, col_s, *_ = Tensor.custom_kernel(row_q, row_s, col_q, col_s, x, fxn=fxn)
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
def _custom_swiglu(out:UOp, x_w13:UOp) -> UOp:
  hidden, n_elems = x_w13.shape[-1]//2, math.prod(x_w13.shape[:-1]) * x_w13.shape[-1]//2
  threads, workgroups = UOp.special(512, "lidx0"), UOp.special(NUM_WG, "gidx0")
  sink = UOp.sink(out.base, x_w13.base, threads, workgroups,
                  arg=KernelInfo(f"swiglu_fwd_{n_elems}", estimates=Estimates(ops=5*n_elems, mem=6*n_elems)))
  src = (pathlib.Path(__file__).parent/"swiglu.hip").read_text()
  lib = compile_hip(src, [f"-DN_ELEMS={n_elems}", f"-DHIDDEN={hidden}", f"-DNUM_WG={NUM_WG}", "-DTHREADS_PER_WG=512"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def _custom_swiglu_bwd(grad_out:UOp, x_w13:UOp, grad_act:UOp) -> UOp:
  hidden, n_elems = x_w13.shape[-1]//2, math.prod(x_w13.shape[:-1]) * x_w13.shape[-1]//2
  threads, workgroups = UOp.special(512, "lidx0"), UOp.special(NUM_WG, "gidx0")
  sink = UOp.sink(grad_out.base, x_w13.base, grad_act.base, threads, workgroups,
                  arg=KernelInfo(f"swiglu_bwd_{n_elems}", estimates=Estimates(ops=10*n_elems, mem=10*n_elems)))
  src = (pathlib.Path(__file__).parent/"swiglu.hip").read_text()
  lib = compile_hip(src, [f"-DN_ELEMS={n_elems}", f"-DHIDDEN={hidden}", f"-DNUM_WG={NUM_WG}", "-DTHREADS_PER_WG=512", "-DSWIGLU_BACKWARD"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _swiglu_bwd(gradient:UOp, kernel:UOp):
  _, x_w13 = kernel.src[1:]
  axis = x_w13.axis if isinstance(x_w13.device, tuple) else None
  grad_out = alloc_like(x_w13.shape, dtypes.bfloat16, x_w13.device, axis)
  grad_out, *_ = Tensor.custom_kernel(grad_out, Tensor(x_w13, device=x_w13.device), Tensor(gradient, device=x_w13.device),
                                      fxn=_custom_swiglu_bwd)
  return (None, grad_out.uop)

def swiglu(x_w13:Tensor) -> Tensor:
  assert x_w13.dtype == dtypes.bfloat16 and x_w13.ndim >= 2 and x_w13.shape[-1] % 32 == 0
  *prefix, two_k = x_w13.shape
  K = two_k//2
  axis = x_w13.uop.axis if isinstance(x_w13.device, tuple) else None
  out = alloc_like((*prefix, K), dtypes.bfloat16, x_w13.device, axis)
  return Tensor.custom_kernel(out, x_w13, fxn=_custom_swiglu, grad_fxn=_swiglu_bwd)[0]
