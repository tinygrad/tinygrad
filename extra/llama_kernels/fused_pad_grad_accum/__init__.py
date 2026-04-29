from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import THREADS_PER_WG, dname_of, compile_hip

ELEMS_PER_THREAD = 8  # vectorized 16-byte load (uint4 = 8 bf16)

def _build_src(n_chunks:int) -> str:
  template = (pathlib.Path(__file__).parent/"fused_pad_grad_accum.cpp").read_text()
  params = "".join(f",\n    const __hip_bfloat16* __restrict__ chunk{i}" for i in range(n_chunks))
  dispatch = "\n    ".join(f"case {i}: chunk_ptr = chunk{i}; break;" for i in range(n_chunks))
  return (template.replace("__FUSED_PAD_GRAD_ACCUM_PARAMS", params)
                  .replace("__FUSED_PAD_GRAD_ACCUM_DISPATCH", dispatch))

@functools.cache
def _custom_fused_pad_grad_accum(grad_buf:UOp, *chunk_uops, dname:str, n_chunks:int, chunk_size:int) -> UOp:
  total = n_chunks * chunk_size
  elems_per_block = THREADS_PER_WG * ELEMS_PER_THREAD
  assert chunk_size % elems_per_block == 0, f"chunk_size {chunk_size} must be multiple of {elems_per_block}"
  num_wg = total // elems_per_block
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(num_wg, "gidx0")
  mem = total * 2 * 3
  sink = UOp.sink(grad_buf.base, *(c.base for c in chunk_uops), threads, workgroups,
                  arg=KernelInfo(f"fused_pad_grad_accum_n{n_chunks}_c{chunk_size}",
                                 estimates=Estimates(ops=2*total, mem=mem)))
  src = _build_src(n_chunks)
  defines = [f"-DCHUNK_SIZE={chunk_size}", f"-DTHREADS_PER_WG={THREADS_PER_WG}", f"-DELEMS_PER_THREAD={ELEMS_PER_THREAD}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def can_fused_pad_grad_accum(grad_buf:Tensor, chunks:list[Tensor]) -> bool:
  if not chunks or grad_buf.dtype != dtypes.bfloat16: return False
  if any(c.dtype != dtypes.bfloat16 for c in chunks): return False
  chunk_shape = chunks[0].shape
  if any(c.shape != chunk_shape for c in chunks): return False
  chunk_size, total = 1, 1
  for d in chunk_shape: chunk_size *= d
  for d in grad_buf.shape: total *= d
  return total == len(chunks) * chunk_size and chunk_size % (THREADS_PER_WG * ELEMS_PER_THREAD) == 0

def fused_pad_grad_accum(grad_buf:Tensor, chunks:list[Tensor]) -> Tensor:
  # NOTE: grad_buf += cat(*chunks, dim=0) in one HBM pass (in-place add). Returns new grad_buf Tensor.
  # Requires uniform chunk shapes and chunk_size % (THREADS_PER_WG*ELEMS_PER_THREAD) == 0.
  assert chunks and grad_buf.dtype == dtypes.bfloat16
  for c in chunks: assert c.dtype == dtypes.bfloat16, f"chunk dtype must be bf16, got {c.dtype}"
  chunk_size, total = 1, 1
  for d in chunks[0].shape: chunk_size *= d
  for d in grad_buf.shape: total *= d
  assert total == len(chunks) * chunk_size, f"grad_buf size {total} != n_chunks {len(chunks)} * chunk_size {chunk_size}"
  fxn = functools.partial(_custom_fused_pad_grad_accum, dname=dname_of(grad_buf.device),
                          n_chunks=len(chunks), chunk_size=chunk_size)
  out, *_ = Tensor.custom_kernel(grad_buf, *chunks, fxn=fxn)
  return out
