import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.mixin.gradient import gradient_auxiliary_mailbox
from tinygrad.renderer import Estimates
from extra.llama_kernels import alloc_like, compile_hip

# A backward producer may create the exact row/column representations consumed by
# MXFP4 GEMM backward. The BF16 gradient remains the autograd value and is the key.
_grad_mxfp4_mailbox = gradient_auxiliary_mailbox("mxfp4")

def alloc_mxfp4_row_outputs(x:Tensor, *, flatten_row:bool=False) -> tuple[Tensor, Tensor]:
  M, N = math.prod(x.shape[:-1]), x.shape[-1]
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  row_axis = 0 if flatten_row and axis is not None else axis
  shape = (M, N//2) if flatten_row else (*x.shape[:-1], N//2)
  scale_shape = (M, N//32) if flatten_row else (*x.shape[:-1], N//32)
  return alloc_like(shape, dtypes.uint8, x.device, row_axis), alloc_like(scale_shape, dtypes.uint8, x.device, row_axis)

def alloc_mxfp4_outputs(x:Tensor, *, flatten_row:bool=False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  M, N = math.prod(x.shape[:-1]), x.shape[-1]
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  col_axis = None if axis is None else (0 if axis == x.ndim-1 else 1)
  row_fp4, row_scale = alloc_mxfp4_row_outputs(x, flatten_row=flatten_row)
  return (row_fp4, row_scale, alloc_like((N, M//2), dtypes.uint8, x.device, col_axis),
          alloc_like((N, M//32), dtypes.uint8, x.device, col_axis))

@functools.cache
def _custom_quantize_mxfp4(row_fp4:UOp, row_scale:UOp, col_fp4:UOp, col_scale:UOp, x:UOp, *,
                           shuffle_row:bool, shuffle_col:bool, write_row:bool, write_col:bool) -> UOp:
  M, N = math.prod(x.shape[:-1]), x.shape[-1]
  assert M % 256 == 0 and N % 256 == 0, f"MXFP4 quantization requires multiples of 256, got {x.shape}"
  direction = "dual" if write_row and write_col else "row" if write_row else "col"
  name = f"quantize_mxfp4_{direction}_{M}_{N}"
  mem = M*N*2 + (M*N//2 + M*N//32) * (write_row + write_col)
  outputs = (row_fp4, row_scale, col_fp4, col_scale)
  sink = UOp.sink(*(o.base for o in outputs), x.base,
                  *(UOp(Ops.CUSTOM, dtypes.void, (o.base.index(0),), arg="") for o in outputs),
                  UOp.special(256, "lidx0"), UOp.special(M//128, "gidx0"), UOp.special(N//64, "gidx1"),
                  arg=KernelInfo(name, estimates=Estimates(ops=12*M*N, mem=mem)))
  src = (pathlib.Path(__file__).parent/"quantize_mxfp4.cpp").read_text()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
    UOp(Ops.BINARY, arg=compile_hip(src, [f"-I{pathlib.Path(__file__).parent}", f"-DKERNEL_NAME={name}", f"-DM_DIM={M}", f"-DN_DIM={N}",
                                             f"-DWRITE_ROWWISE_VALUE={int(write_row)}", f"-DWRITE_COLWISE_VALUE={int(write_col)}",
                                             f"-DSHUFFLE_ROWWISE_FP4_VALUE={int(shuffle_row)}",
                                             f"-DSHUFFLE_COLWISE_FP4_VALUE={int(shuffle_col)}"]))))

def quantize_mxfp4(x:Tensor, *, shuffle_row:bool=False, shuffle_col:bool=False, flatten_row:bool=False,
                   out:tuple[Tensor, Tensor, Tensor, Tensor]|None=None, row:bool=True, col:bool=True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  assert x.dtype == dtypes.bfloat16 and x.ndim >= 2, f"expected BF16 matrix, got {x.dtype} {x.shape}"
  assert row or col, "at least one MXFP4 direction must be requested"
  assert out is None or (row and col), "cached output refresh requires both MXFP4 directions"
  M, N = math.prod(x.shape[:-1]), x.shape[-1]
  assert M % 256 == 0 and N % 256 == 0, f"MXFP4 quantization requires multiples of 256, got {x.shape}"
  # Updated flat-model weights are contiguous slices of one packed allocation. Pass the physical view (including its
  # byte offset and producer state) to the opaque quantizer instead of materializing one temporary per layer.
  if (offset:=x.uop.contiguous_view_offset()) is not None and x.uop.numel() != x.uop.base.numel():
    physical = UOp(Ops.SLICE, x.dtype, (x.uop.buf_uop, UOp.const(offset)), x.uop.numel(), tag=("allreduce",))
    if x.uop.base.op is Ops.AFTER: physical = physical.after(x.uop.base)
    x = Tensor(physical.reshape(x.shape))
  if out is not None: outputs = out
  elif row and col: outputs = alloc_mxfp4_outputs(x, flatten_row=flatten_row)
  else:
    axis = x.uop.axis if isinstance(x.device, tuple) else None
    row_axis = 0 if flatten_row and axis is not None else axis
    col_axis = None if axis is None else (0 if axis == x.ndim-1 else 1)
    outputs = (alloc_like((M, N//2) if flatten_row else (*x.shape[:-1], N//2), dtypes.uint8, x.device, row_axis) if row else
                 alloc_like((1,), dtypes.uint8, x.device, None),
               alloc_like((M, N//32) if flatten_row else (*x.shape[:-1], N//32), dtypes.uint8, x.device, row_axis) if row else
                 alloc_like((1,), dtypes.uint8, x.device, None),
               alloc_like((N, M//2), dtypes.uint8, x.device, col_axis) if col else alloc_like((1,), dtypes.uint8, x.device, None),
               alloc_like((N, M//32), dtypes.uint8, x.device, col_axis) if col else alloc_like((1,), dtypes.uint8, x.device, None))
  fxn = functools.partial(_custom_quantize_mxfp4, shuffle_row=shuffle_row, shuffle_col=shuffle_col, write_row=row, write_col=col)
  ret = Tensor.custom_kernel(*outputs, x, fxn=fxn)
  return ret[0], ret[1], ret[2], ret[3]
