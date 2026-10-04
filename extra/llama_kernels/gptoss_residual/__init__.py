from __future__ import annotations
import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.renderer import Estimates
from tinygrad.helpers import ALLREDUCE_CAST, getenv
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from extra.llama_kernels import alloc_like, compile_hip, dname_of, owned_empty

REAL_D, PAD_D, THREADS = 2880, 3072, 256

@functools.cache
def _custom_wo_bias_sum(out:UOp, inp:UOp, *, stage:str) -> UOp:
  assert not getenv("AMD_COMGR_FAST_MATH", 0), "ordered GPTOSS bias reduction requires precise FP32 additions"
  assert stage in ("partial", "final")
  partial = stage == "partial"
  assert out.shape == ((16, REAL_D) if partial else (REAL_D,))
  assert inp.shape == ((16384, REAL_D) if partial else (16, REAL_D))
  assert out.dtype == (dtypes.float32 if partial else dtypes.bfloat16)
  assert inp.dtype == (dtypes.bfloat16 if partial else dtypes.float32)
  zero = UOp.const(0, dtypes.int32)
  accesses = (out.index(zero).store(UOp.const(0, out.dtype)), inp.index(zero).load())
  launch = (UOp.special(64, "lidx0"), UOp.special(45, "gidx0")) + ((UOp.special(16, "gidx1"),) if partial else ())
  sink = UOp.sink(out.base, inp.base, *accesses, *launch, arg=KernelInfo(f"gptoss_wo_bias_{stage}"))
  src = (pathlib.Path(__file__).parent/f"wo_bias_{stage}.cpp").read_text()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=HIPCompiler("gfx950").compile_cached(src))))

def _wo_bias_sum(g:Tensor) -> Tensor:
  dp = len(g.device) if isinstance(g.device, tuple) else 1
  assert g.dtype == dtypes.bfloat16 and g.shape[-1] == REAL_D
  assert dp == 1 or (g.uop.axis == 0 and ALLREDUCE_CAST)
  x = g.reshape(-1, REAL_D)
  assert x.shape == (dp*16384, REAL_D)
  axis = 0 if dp > 1 else None
  partial = owned_empty(alloc_like((dp*16, REAL_D), dtypes.float32, g.device, axis))
  local = owned_empty(alloc_like((REAL_D,), dtypes.bfloat16, g.device))
  partial = Tensor.custom_kernel(partial, x, fxn=functools.partial(_custom_wo_bias_sum, stage="partial"))[0]
  local = Tensor.custom_kernel(local, partial, fxn=functools.partial(_custom_wo_bias_sum, stage="final"))[0]
  # This is precisely the original reduction's BF16 ALLREDUCE after multi
  # lowering. Emit the local vector directly: a singleton row introduces a copy
  # when it is reshaped to the existing aggregation's 1D input.
  return Tensor(local.uop.allreduce(Ops.ADD, g.device)) if dp > 1 else local

@functools.cache
def _custom_residual_join(out:UOp, x:UOp, proj:UOp, bias:UOp, moe:UOp, *, dname:str) -> UOp:
  assert out.shape == x.shape == moe.shape and x.dtype == out.dtype == moe.dtype == dtypes.bfloat16
  assert x.shape[-1] == REAL_D and proj.shape == (math.prod(x.shape[:-1]), PAD_D) and proj.dtype == dtypes.bfloat16
  assert bias.shape == (REAL_D,) and bias.dtype == dtypes.bfloat16
  axis = x.axis if isinstance(x.device, tuple) else None
  local_shape = x.shard_shape if axis is not None else x.shape
  rows = math.prod(local_shape[:-1])
  zero = UOp.const(0, dtypes.int32)
  accesses = (out.index(zero).store(UOp.const(0, out.dtype)), x.index(zero).load(), proj.index(zero).load(),
              bias.index(zero).load(), moe.index(zero).load())
  sink = UOp.sink(out.base, x.base, proj.base, bias.base, moe.base, *accesses,
                  UOp.special(THREADS, "lidx0"), UOp.special(rows, "gidx0"),
                  arg=KernelInfo(f"gptoss_residual_join_vec_{rows}_{REAL_D}_{PAD_D}",
                                 estimates=Estimates(ops=3*rows*REAL_D, mem=8*rows*REAL_D+2*REAL_D)))
  src = (pathlib.Path(__file__).parent/"residual.cpp").read_text()
  # Fast math reassociates the BF16 additions across their required rounding boundaries.
  defines = [f"-DROWS={rows}", f"-DREAL_D={REAL_D}", f"-DPAD_D={PAD_D}", "-fno-fast-math"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def _residual_join_bwd(gradient:UOp, kernel:UOp) -> tuple:
  _, x, proj, bias, _ = kernel.src[1:]
  g = Tensor(gradient, device=x.device)
  dbias = g.float().sum(axis=tuple(range(g.ndim-1))).cast(bias.dtype)
  local_shape = g.uop.shard_shape if isinstance(g.device, tuple) else g.shape
  if getenv("GPTOSS_WO_BIAS_SUM", 0) and not getenv("AMD_COMGR_FAST_MATH", 0) and dname_of(g.device) == "AMD" \
      and g.dtype == dtypes.bfloat16 and local_shape[-1] == REAL_D and math.prod(local_shape[:-1]) == 16384 \
      and (not isinstance(g.device, tuple) or (g.uop.axis == 0 and ALLREDUCE_CAST)):
    dbias = _wo_bias_sum(g)
  dproj = g.reshape(-1, REAL_D).pad(((0, 0), (0, PAD_D-REAL_D)))
  assert dproj.shape == proj.shape
  return None, gradient, dproj.uop, dbias.uop, gradient

def gptoss_residual_join(x:Tensor, attn:Tensor, moe:Tensor) -> Tensor:
  assert x.shape == attn.shape == moe.shape and x.dtype == attn.dtype == moe.dtype == dtypes.bfloat16
  from extra.gptoss_kernels.rmsnorm import _gptoss_residual_inputs
  # Pass the actual padded parent, not its logical SHRINK: custom-kernel inputs are materialized by the scheduler,
  # so a SHRINK would be compacted to REAL_D while the HIP source still reads PAD_D. Reuse the exact view matcher
  # used by residual RMSNorm backward, and retain the ordinary expression for any other projection layout.
  if (inputs := _gptoss_residual_inputs((x + attn).uop)) is None: return (x + attn) + moe
  _, proj, bias = inputs
  assert proj.device == bias.device == x.device
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  # Mark fresh destination storage before the opaque CALL so precompiled returns can bind it directly.
  out = owned_empty(alloc_like(x.shape, dtypes.bfloat16, x.device, axis))
  fxn = functools.partial(_custom_residual_join, dname=dname_of(x.device))
  return Tensor.custom_kernel(out, x, proj, bias, moe, fxn=fxn, grad_fxn=_residual_join_bwd)[0]
