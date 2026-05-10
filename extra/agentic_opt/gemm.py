from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp


def _uop_gemm(c:UOp, a:UOp, b:UOp) -> UOp:
  batch, M, N = c.shape
  K = a.shape[-1]
  assert b.shape[-2] == K, f"{a.shape=} {b.shape=}"
  bi = UOp.range(batch, 0)
  mi = UOp.range(M, 1)
  ni = UOp.range(N, 2)
  ki = UOp.range(K, 3, axis_type=AxisType.REDUCE)
  ai = 0 if a.shape[0] == 1 else bi
  aval = a[ai, mi, ki].cast(dtypes.float32)
  bval = b[ki, ni].cast(dtypes.float32) if b.ndim == 2 else b[0 if b.shape[0] == 1 else bi, ki, ni].cast(dtypes.float32)
  val = (aval * bval).reduce(ki, arg=Ops.ADD, dtype=dtypes.float32).cast(c.dtype.base)
  return c[bi, mi, ni].store(val).end(ni, mi, bi).sink(arg=KernelInfo(name=f"agentic_gemm_{batch}_{M}_{N}_{K}", opts_to_apply=()))


def _as_batched_a(a:Tensor) -> tuple[Tensor, bool]:
  if a.ndim == 2: return a.unsqueeze(0), True
  if a.ndim == 3: return a, False
  raise RuntimeError(f"expected A shape (M, K) or (B, M, K), got {a.shape}")


def _check_b(a:Tensor, b:Tensor):
  if b.ndim not in (2, 3): raise RuntimeError(f"expected B shape (K, N) or (B, K, N), got {b.shape}")
  if a.shape[-1] != b.shape[-2]: raise RuntimeError(f"contracting dimension mismatch: {a.shape=} {b.shape=}")
  if b.ndim == 3 and b.shape[0] != a.shape[0]:
    raise RuntimeError(f"B batch must match A batch, got {a.shape=} {b.shape=}")


def _custom_gemm_bw(grad:UOp, kernel:UOp):
  _, a_u, b_u = kernel.src[1:]
  a, b, g = Tensor(a_u, device=a_u.device), Tensor(b_u, device=b_u.device), Tensor(grad, device=grad.device)
  if b.ndim == 2:
    grad_a = gemm(g, b.T)
    grad_b = gemm(a.permute(2, 0, 1).reshape(a.shape[2], -1), g.reshape(-1, g.shape[-1]))
  else:
    grad_a = gemm(g, b.transpose(1, 2))
    grad_b = gemm(a.transpose(1, 2), g)
  return None, grad_a.uop, grad_b.uop


def _gemm_out_dtype(a:Tensor, b:Tensor):
  supported = (dtypes.float32, dtypes.float16, dtypes.bfloat16)
  if a.dtype not in supported or b.dtype not in supported: raise RuntimeError(f"agentic GEMM ignores unsupported dtypes {a.dtype} and {b.dtype}")
  if a.dtype == b.dtype: return a.dtype
  if dtypes.float32 in (a.dtype, b.dtype): return dtypes.float32
  raise RuntimeError(f"agentic GEMM requires matching low-precision dtypes, got {a.dtype} and {b.dtype}")


def gemm(a:Tensor, b:Tensor) -> Tensor:
  out_dtype = _gemm_out_dtype(a, b)
  a3, squeeze = _as_batched_a(a)
  _check_b(a3, b)
  batch = a3.shape[0] if b.ndim == 2 else max(a3.shape[0], b.shape[0])
  out = Tensor.empty(batch, a3.shape[1], b.shape[-1], dtype=out_dtype, device=a3.device)
  ret = Tensor.custom_kernel(out, a3, b, fxn=_uop_gemm, grad_fxn=_custom_gemm_bw)[0]
  return ret.squeeze(0) if squeeze and b.ndim == 2 else ret


def _assert_close(name:str, got:Tensor, ref:Tensor, atol:float=1e-5, rtol:float=1e-5):
  import numpy as np
  np.testing.assert_allclose(got.numpy(), ref.numpy(), atol=atol, rtol=rtol)
  print(f"ok {name}")


def _kernel_count(t:Tensor) -> int:
  return len(t.schedule_linear().src)


def _forward_kernel_count(a_shape:tuple[int, ...], b_shape:tuple[int, ...], dtype) -> int:
  a = Tensor.empty(*a_shape, dtype=dtype)
  b = Tensor.empty(*b_shape, dtype=dtype)
  return _kernel_count(gemm(a, b))


def _backward_kernel_count(a_shape:tuple[int, ...], b_shape:tuple[int, ...], dtype) -> int:
  a = Tensor.empty(*a_shape, dtype=dtype, requires_grad=True)
  b = Tensor.empty(*b_shape, dtype=dtype, requires_grad=True)
  out = gemm(a, b)
  grad = Tensor.empty(*out.shape, dtype=dtypes.float32)
  ga, gb = out.gradient(a, b, gradient=grad)
  linear, _ = Tensor.linear_with_vars(ga, gb)
  return len(linear.src)


def _run_case(name:str, a_shape:tuple[int, ...], b_shape:tuple[int, ...], dtype=dtypes.float32):
  import numpy as np
  Tensor.manual_seed(3000 + len(name))
  a0 = Tensor.randn(*a_shape, dtype=dtype).realize()
  b0 = Tensor.randn(*b_shape, dtype=dtype).realize()
  got = gemm(a0, b0)
  ref = a0 @ b0
  kcount = _forward_kernel_count(a_shape, b_shape, dtype)
  if kcount != 1: raise AssertionError(f"forward {name} expected 1 kernel, got {kcount}")
  atol = rtol = 2e-2 if dtype == dtypes.bfloat16 else 1e-5
  _assert_close(f"forward {name}", got.float(), ref.float(), atol=atol, rtol=rtol)

  a, b = Tensor(a0.numpy(), requires_grad=True), Tensor(b0.numpy(), requires_grad=True)
  gout = Tensor.randn(*got.shape, dtype=dtypes.float32).realize()
  gemm(a, b).backward(gout)
  Tensor.realize(a.grad, b.grad)

  ar, br = Tensor(a0.numpy(), requires_grad=True), Tensor(b0.numpy(), requires_grad=True)
  (ar @ br).backward(gout)
  Tensor.realize(ar.grad, br.grad)

  np.testing.assert_allclose(a.grad.numpy(), ar.grad.numpy(), atol=atol, rtol=rtol)
  np.testing.assert_allclose(b.grad.numpy(), br.grad.numpy(), atol=atol, rtol=rtol)
  print(f"ok backward {name} kernels={_backward_kernel_count(a_shape, b_shape, dtype)}")


def _main():
  print("agentic_opt.gemm diagnostics")
  _run_case("2d_fp32", (5, 7), (7, 3))
  _run_case("batched_a_2d_b_fp32", (2, 5, 7), (7, 3))
  _run_case("batched_a_batched_b_fp32", (2, 5, 7), (2, 7, 3))
  _run_case("2d_bf16", (5, 8), (8, 4), dtype=dtypes.bfloat16)


if __name__ == "__main__":
  _main()
