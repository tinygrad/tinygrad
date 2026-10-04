from __future__ import annotations
import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from extra.gemm.cdna_asm_gemm import FP8_DTYPE
from extra.llama_kernels import alloc_like, compile_hip, dname_of, owned_empty

_SRC = pathlib.Path(__file__).parent/"fused_adam_mxfp8.cpp"
_SRC_BF16 = pathlib.Path(__file__).parent/"fused_adam_bf16.cpp"

@functools.cache
def _custom_gather_compact_q(out:UOp, *parts:UOp) -> UOp:
  # Exact DP8 GPT-OSS FC1/FC2 ABI. Transport words are raw FP8 bytes, never converted to floating point.
  _, rows, cols = out.shape
  live_rows = 5760 if rows == 5888 else 2880
  assert out.shape in ((32,5888,3072), (32,3072,3072)) and out.dtype in dtypes.fp8s
  assert len(parts) == 8 and all(p.shape == (4,live_rows,720) and p.dtype == dtypes.uint32 for p in parts)
  zero = UOp.const(0, dtypes.int32)
  accesses = (out.index(zero).store(UOp.const(0, out.dtype)),) + tuple(p.index(zero).load() for p in parts)
  mem = out.numel() + sum(p.numel()*4 for p in parts)
  sink = UOp.sink(out.base, *(p.base for p in parts), *accesses,
                  UOp.special(256, "lidx0"), UOp.special(4*rows*cols//16//256, "gidx0"), UOp.special(8, "gidx1"),
                  arg=KernelInfo(f"gptoss_compact_weight_gather_{rows}", estimates=Estimates(ops=0, lds=mem, mem=mem)))
  src = (_SRC.parent/"gather_compact_q.cpp").read_text()
  lib = compile_hip(src, [f"-DPHYSICAL_ROWS={rows}", f"-DLIVE_ROWS={live_rows}"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def gather_compact_q(parts:tuple[Tensor, ...], like:Tensor) -> Tensor:
  # The destination is the existing replicated parameter, not a temporary followed by a full-weight copy.
  return Tensor.custom_kernel(like, *parts, fxn=_custom_gather_compact_q)[0]

def _float_define(x:float) -> str: return f"{x:.9g}f"

@functools.cache
def _custom_fused_adam_bf16_vocab(m:UOp, v:UOp, master:UOp, grad:UOp, *args:UOp, dname:str,
                                  b1:float, b2:float, eps:float, raw_clip:bool=False) -> UOp:
  clip_scale = args[0] if raw_clip else None
  lr, b1_t, b2_t = args[1:] if raw_clip else args
  n = math.prod(master.shape)
  assert m.shape == v.shape == master.shape == grad.shape and n % 1024 == 0
  assert m.dtype == v.dtype == grad.dtype == dtypes.bfloat16 and master.dtype == dtypes.float32
  assert lr.shape == b1_t.shape == b2_t.shape == (1,)
  if clip_scale is not None: assert clip_scale.shape in ((), (1,)) and clip_scale.dtype == dtypes.float32
  threads, groups = UOp.special(256, "lidx0"), UOp.special(n // 1024, "gidx0")
  zero = UOp.const(0, dtypes.int32)
  outputs = (m, v, master)
  inputs = (m, v, master, grad, *((clip_scale,) if clip_scale is not None else ()), lr, b1_t, b2_t)
  accesses = tuple(x.index(zero).store(UOp.const(0, x.dtype)) for x in outputs) + tuple(x.index(zero).load() for x in inputs)
  sink = UOp.sink(*(x.base for x in inputs), *accesses, threads, groups,
                  arg=KernelInfo(f"fused_adam_bf16_vocab_{n}"+("_rawclip1" if raw_clip else ""),
                                 estimates=Estimates(ops=18*n, mem=18*n)))
  src = _SRC_BF16.read_text()
  # Use the same COMGR path as tinygrad's generated reference kernels. Prefixing defines keeps the source and
  # compiler expression contraction behavior aligned with the byte-exact baseline.
  defines = "\n".join((f"#define N_ELEMS {n}", f"#define B1 {_float_define(b1)}", f"#define B2 {_float_define(b2)}",
                        f"#define OMB1 {_float_define(1.0-b1)}", f"#define OMB2 {_float_define(1.0-b2)}",
                        f"#define EPS {_float_define(eps)}", f"#define GPTOSS_ADAM_BF16_RAW_CLIP {int(raw_clip)}"))
  lib = HIPCompiler("gfx950").compile_cached(defines+"\n"+src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def fused_adam_bf16_vocab(m:Tensor, v:Tensor, master:Tensor, grad:Tensor,
                          lr:Tensor, b1_t:Tensor, b2_t:Tensor, *, b1:float, b2:float, eps:float,
                          clip_scale:Tensor|None=None) -> tuple[Tensor, Tensor, Tensor]:
  """Update GPT-OSS' BF16 vocabulary optimizer state and FP32 master in one pass."""
  assert m.shape == v.shape == master.shape == grad.shape
  assert m.device == v.device == master.device == grad.device
  if clip_scale is not None:
    assert clip_scale.shape in ((), (1,)) and clip_scale.dtype == dtypes.float32 and clip_scale.device == master.device
  m, v, master = m.contiguous(), v.contiguous(), master.contiguous()
  outs = Tensor.custom_kernel(m, v, master, grad, *((clip_scale,) if clip_scale is not None else ()), lr, b1_t, b2_t,
    fxn=functools.partial(_custom_fused_adam_bf16_vocab, dname=dname_of(master.device), b1=b1, b2=b2, eps=eps,
                          raw_clip=clip_scale is not None))
  return outs[0], outs[1], outs[2]

@functools.cache
def _custom_fused_adam_mxfp8(m:UOp, v:UOp, master:UOp, q:UOp, e8:UOp, *args:UOp,
                             dname:str, b1:float, b2:float, eps:float, weight_decay:float,
                             grouped_si:bool=False, raw_clip:bool=False, fc1_split_part:str|None=None, compact_q:bool=False) -> UOp:
  si = args[0] if grouped_si else None
  if grouped_si: args = args[1:]
  clip_scale = args[-4] if raw_clip else None
  grads, (lr, b1_t, b2_t) = args[:-(4 if raw_clip else 3)], args[-3:]
  n = math.prod(master.shape)
  assert len(grads) == 1 and m.shape == v.shape == master.shape and all(g.shape == master.shape for g in grads)
  assert q.shape == ((master.shape[0], 5760 if grouped_si else 2880, 720) if compact_q else master.shape)
  assert not raw_clip or len(grads) == 1
  assert m.dtype == v.dtype and m.dtype in (dtypes.bfloat16, dtypes.float32)
  assert master.dtype == dtypes.float32 and all(g.dtype == dtypes.bfloat16 for g in grads)
  if clip_scale is not None: assert clip_scale.shape in ((), (1,)) and clip_scale.dtype == dtypes.float32
  assert q.dtype == (dtypes.uint32 if compact_q else FP8_DTYPE) and e8.shape == (*master.shape[:-1], master.shape[-1] // 32)
  assert n % 32 == 0
  if si is not None:
    assert len(master.shape) == 3 and master.shape[-1] % 256 == 0 and n % 256 == 0
    assert si.dtype == dtypes.uint32 and si.shape == (master.shape[0], master.shape[-1] // 128, master.shape[1])
  # The GPT-OSS down weight is physically (E,3072,3072), but only its leading 2880x2880 rectangle is model state.
  # Restrict this specialization to the single-gradient BF16-state production path; generic and FC1 Adam are unchanged.
  gptoss_down_shape = len(master.shape) == 3 and master.shape[-2:] == (3072, 3072)
  gptoss_down_logical = bool(getenv("GPTOSS_ADAM_DOWN_LOGICAL", 1)) and gptoss_down_shape and si is None and \
    len(grads) == 1 and m.dtype == dtypes.bfloat16
  gptoss_down_pipeline = bool(getenv("GPTOSS_ADAM_DOWN_PIPELINE", 0)) and gptoss_down_logical
  gptoss_fc1_pipeline = bool(getenv("GPTOSS_ADAM_FC1_PIPELINE", 1)) and si is not None
  gptoss_fc1_row_skip = bool(getenv("GPTOSS_ADAM_FC1_ROW_SKIP", 1)) and gptoss_fc1_pipeline and \
    master.shape[-2:] == (5888, 3072) and len(grads) == 1 and m.dtype == dtypes.bfloat16
  if compact_q: assert gptoss_down_logical or (gptoss_fc1_row_skip and fc1_split_part is not None)
  assert fc1_split_part in (None, "main", "tail")
  if fc1_split_part is not None:
    assert gptoss_fc1_row_skip and grouped_si and raw_clip and len(master.shape) == 3, \
      "split FC1 Adam is restricted to the exact BF16-state/single-raw-gradient grouped production ABI"
  live_n = master.shape[0] * 2880 * 2880 if gptoss_down_logical else n
  # Split launches own disjoint columns; padding initialization is not an Adam update. The unsplit row-skip
  # kernel still updates all 3072 columns of each live row, unlike the 64-column split tail.
  update_n = master.shape[0]*5760*({"main":2816, "tail":64}.get(fc1_split_part, 3072)) if gptoss_fc1_row_skip else live_n
  threads = UOp.special(64 if grouped_si or gptoss_down_pipeline else 256, "lidx0")
  if fc1_split_part == "main": groups = (UOp.special(11, "gidx0"), UOp.special(master.shape[0] * 5760, "gidx1"))
  elif fc1_split_part == "tail": groups = (UOp.special(master.shape[0] * 5888, "gidx0"),)
  else: groups = (UOp.special((n + 255) // 256, "gidx0"),)
  # This precompiled kernel updates optimizer state in place and its quantized outputs are immediately gathered by
  # ZeRO on SDMA queues. Describe the writes so graph replay waits for them instead of copying partial q/e8/SI data.
  zero = UOp.const(0, dtypes.int32)
  outputs = (m, v, master, q, e8) + ((si,) if si is not None else ())
  inputs = (m, v, master, *grads, *((clip_scale,) if clip_scale is not None else ()), lr, b1_t, b2_t)
  accesses = tuple(x.index(zero).store(UOp.const(0, x.dtype)) for x in outputs) + tuple(x.index(zero).load() for x in inputs)
  sink = UOp.sink(m.base, v.base, master.base, q.base, e8.base, *((si.base,) if si is not None else ()),
                  *(g.base for g in grads), *((clip_scale.base,) if clip_scale is not None else ()),
                  lr.base, b1_t.base, b2_t.base, *accesses, threads, *groups,
                  arg=KernelInfo(f"fused_adam_mxfp8_{n}"+(f"_downlog{int(gptoss_down_logical)}" if gptoss_down_shape else "")+
                                 (f"_downpipe{int(gptoss_down_pipeline)}" if gptoss_down_shape else "")+
                                 (f"_fc1pipe{int(gptoss_fc1_pipeline)}" if si is not None else "")+
                                 (f"_rowskip{int(gptoss_fc1_row_skip)}" if si is not None else "")+
                                 (f"_split_{fc1_split_part}" if fc1_split_part is not None else "")+
                                 ("_rawclip1" if raw_clip else "")+("_compactq1" if compact_q else ""),
                                 estimates=Estimates(ops=24*update_n,
                                 mem=(19 if m.dtype == dtypes.bfloat16 else 27)*live_n+(n-live_n))))
  src = _SRC.read_text()
  lib = compile_hip(src, [f"-DN_ELEMS={n}", f"-DSTATE_BF16={int(m.dtype == dtypes.bfloat16)}",
                          f"-DB1={_float_define(b1)}", f"-DB2={_float_define(b2)}",
                          f"-DOMB1={_float_define(1.0-b1)}", f"-DOMB2={_float_define(1.0-b2)}",
                          f"-DEPS={_float_define(eps)}", f"-DWEIGHT_DECAY={_float_define(weight_decay)}",
                          f"-DEMIT_GROUPED_SI={int(si is not None)}", f"-DLAST_DIM={master.shape[-1]}",
                          f"-DEXPERT_ROWS={master.shape[1] if len(master.shape) == 3 else 1}",
                          f"-DGPTOSS_ADAM_FC1_PIPELINE={int(gptoss_fc1_pipeline)}",
                          f"-DGPTOSS_ADAM_FC1_ROW_SKIP={int(gptoss_fc1_row_skip)}",
                          f"-DGPTOSS_ADAM_FC1_SPLIT_MAIN={int(fc1_split_part == 'main')}",
                          f"-DGPTOSS_ADAM_FC1_SPLIT_TAIL={int(fc1_split_part == 'tail')}",
                          f"-DGPTOSS_ADAM_DOWN_LOGICAL={int(gptoss_down_logical)}",
                          f"-DGPTOSS_ADAM_DOWN_PIPELINE={int(gptoss_down_pipeline)}",
                          f"-DGPTOSS_ADAM_RAW_CLIP={int(raw_clip)}", f"-DGPTOSS_ADAM_COMPACT_Q={int(compact_q)}"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def fused_adam_mxfp8(m:Tensor, v:Tensor, master:Tensor, grad:Tensor, lr:Tensor, b1_t:Tensor, b2_t:Tensor,
                      *, b1:float, b2:float, eps:float, weight_decay:float,
                      experts:int=1, out_si:Tensor|None=None,
                      clip_scale:Tensor|None=None, compact_q:bool=False) -> tuple[Tensor, ...]:
  """Update AdamW state/master in place; compact q is raw uint32 transport storage, otherwise FP8."""
  assert m.shape == v.shape == master.shape == grad.shape and master.shape[-1] % 32 == 0
  assert m.device == v.device == master.device == grad.device
  # Replicated optimizer initialization may still be a lazy COPY. Native in-place outputs need storage,
  # not the COPY expression itself; this is a no-op once the persistent state is realized.
  m, v, master = m.contiguous(), v.contiguous(), master.contiguous()
  axis = master.uop.axis if isinstance(master.device, tuple) else None
  if compact_q: assert master.ndim == 3 and master.shape[-2:] in ((5888, 3072), (3072, 3072))
  q_shape = (master.shape[0], 5760 if experts > 1 else 2880, 720) if compact_q else master.shape
  q = alloc_like(q_shape, dtypes.uint32 if compact_q else FP8_DTYPE, master.device, axis)
  e8 = alloc_like((*master.shape[:-1], master.shape[-1] // 32), dtypes.uint8, master.device, axis)
  fresh_si = out_si is None
  if experts > 1:
    assert master.ndim == 3 and master.shape[0] == experts and axis in (None, 0)
    si_shape, si_axis = (experts, master.shape[-1] // 128, master.shape[1]), axis
    if out_si is None: out_si = alloc_like(si_shape, dtypes.uint32, master.device, si_axis)
    else: assert out_si.shape == si_shape and out_si.dtype == dtypes.uint32 and out_si.uop.axis == si_axis
  else: assert out_si is None, "out_si requires grouped experts"
  grads = (grad,)
  if clip_scale is not None:
    assert clip_scale.shape in ((), (1,)) and clip_scale.dtype == dtypes.float32 and clip_scale.device == master.device
  kernel_args = (m, v, master, q, e8) + ((out_si,) if out_si is not None else ()) + \
    (*grads, *((clip_scale,) if clip_scale is not None else ()), lr, b1_t, b2_t)
  # GPT-OSS FC1 has eleven complete 256-column blocks and one 64-value tail per live row. Splitting the rare tail
  # keeps the hot binary branch-free and also omits all persistent-state work for the physical padding rectangle.
  fc1_split = bool(getenv("GPTOSS_ADAM_FC1_SPLIT", 0)) and out_si is not None and \
    clip_scale is not None and m.dtype == dtypes.bfloat16 and master.ndim == 3 and master.shape[-2:] == (5888, 3072)
  if fc1_split:
    # Only newly allocated quantized outputs need return-buffer binding. State and caller-owned SI stay in place.
    q, e8 = owned_empty(q), owned_empty(e8)
    assert out_si is not None
    if fresh_si: out_si = owned_empty(out_si)
    kernel_args = (*kernel_args[:3], q, e8, out_si, *kernel_args[6:])
    fxn_args = dict(dname=dname_of(master.device), b1=b1, b2=b2, eps=eps, weight_decay=weight_decay,
                    grouped_si=True, raw_clip=True, compact_q=compact_q)
    mid = Tensor.custom_kernel(*kernel_args,
      fxn=functools.partial(_custom_fused_adam_mxfp8, **fxn_args, fc1_split_part="main"))
    outs = Tensor.custom_kernel(*mid[:6], *kernel_args[6:],
      fxn=functools.partial(_custom_fused_adam_mxfp8, **fxn_args, fc1_split_part="tail"))
    return tuple(outs[:6])
  outs = Tensor.custom_kernel(*kernel_args,
    fxn=functools.partial(_custom_fused_adam_mxfp8, dname=dname_of(master.device), b1=b1, b2=b2, eps=eps,
                          weight_decay=weight_decay, grouped_si=out_si is not None, raw_clip=clip_scale is not None, compact_q=compact_q))
  return tuple(outs[:6] if out_si is not None else outs[:5])
