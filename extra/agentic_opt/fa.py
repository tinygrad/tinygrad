import math

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp


def _score_expr(q:UOp, k:UOp, b:UOp, qpos:UOp, h:UOp, h_kv:UOp, kpos:UOp, dsum:UOp) -> UOp:
  D = q.shape[3]
  dot = (q[b, qpos, h, dsum].cast(dtypes.float32) * k[b, kpos, h_kv, dsum].cast(dtypes.float32)).reduce(dsum, arg=Ops.ADD, dtype=dtypes.float32)
  score = dot * (1.0 / math.sqrt(D))
  return (kpos <= qpos).where(score, -math.inf)


def _softmax_parts(q:UOp, k:UOp, b:UOp, qpos:UOp, h:UOp, h_kv:UOp, base:int) -> tuple[UOp, UOp]:
  N = q.shape[1]
  km, dm = UOp.range(N, base, axis_type=AxisType.REDUCE), UOp.range(q.shape[3], base+1, axis_type=AxisType.REDUCE)
  m_score = _score_expr(q, k, b, qpos, h, h_kv, km, dm)
  m = m_score.reduce(km, arg=Ops.MAX, dtype=dtypes.float32)
  kl, dl = UOp.range(N, base+2, axis_type=AxisType.REDUCE), UOp.range(q.shape[3], base+3, axis_type=AxisType.REDUCE)
  l_score = _score_expr(q, k, b, qpos, h, h_kv, kl, dl)
  l_term = l_score.eq(-math.inf).where(0.0, (l_score - m).exp())
  l = l_term.reduce(kl, arg=Ops.ADD, dtype=dtypes.float32)
  return m, l


def _logsumexp_from_parts(m:UOp, l:UOp) -> UOp:
  return l.eq(0.0).where(-math.inf, m + l.log2() * math.log(2))


def _prob_from_lse(score:UOp, lse:UOp) -> UOp:
  return score.eq(-math.inf).where(0.0, (score - lse).exp())


def _uop_flash_forward(out:UOp, l_vec:UOp, q:UOp, k:UOp, v:UOp) -> UOp:
  B, N, H, D = q.shape
  H_KV = k.shape[2]
  group_size = H // H_KV

  b = UOp.range(B, 0)
  qpos = UOp.range(N, 1)
  h = UOp.range(H, 2)
  dout = UOp.range(D, 3)
  h_kv = h if group_size == 1 else h // group_size

  m, l = _softmax_parts(q, k, b, qpos, h, h_kv, 10)
  lse = _logsumexp_from_parts(m, l)
  kpos = UOp.range(N, 20, axis_type=AxisType.REDUCE)
  dscore = UOp.range(D, 21, axis_type=AxisType.REDUCE)
  score = _score_expr(q, k, b, qpos, h, h_kv, kpos, dscore)
  val = (_prob_from_lse(score, lse) * v[b, kpos, h_kv, dout].cast(dtypes.float32)).reduce(kpos, arg=Ops.ADD, dtype=dtypes.float32)
  store_out = out[b, qpos, h, dout].store(val.cast(out.dtype.base)).end(dout, h, qpos, b)

  bl = UOp.range(B, 30)
  hl = UOp.range(H, 31)
  ql = UOp.range(N, 32)
  h_kv_l = hl if group_size == 1 else hl // group_size
  ml, ll = _softmax_parts(q, k, bl, ql, hl, h_kv_l, 40)
  store_l = l_vec[bl, hl, 0, ql].store(_logsumexp_from_parts(ml, ll)).end(ql, hl, bl)
  return UOp.group(store_out, store_l).sink(arg=KernelInfo(name=f"agentic_fa_fwd_{B}_{N}_{H}_{H_KV}_{D}", opts_to_apply=()))


def _uop_fa_backward_pre(delta:UOp, out:UOp, dout:UOp) -> UOp:
  B, N, H, D = out.shape
  b, h, qpos = UOp.range(B, 0), UOp.range(H, 1), UOp.range(N, 2)
  d = UOp.range(D, 3, axis_type=AxisType.REDUCE)
  val = (out[b, qpos, h, d].cast(dtypes.float32) * dout[b, qpos, h, d].cast(dtypes.float32)).reduce(d, arg=Ops.ADD, dtype=dtypes.float32)
  return delta[b, h, 0, qpos].store(val).end(qpos, h, b).sink(arg=KernelInfo(name=f"agentic_fa_bwd_pre_{B}_{N}_{H}_{D}", opts_to_apply=()))


def _uop_fa_backward(dq:UOp, dkp:UOp, dvp:UOp, dout:UOp, q:UOp, k:UOp, v:UOp, l_vec:UOp, delta:UOp) -> UOp:
  B, N, H, D = q.shape
  H_KV, group_size = k.shape[2], H // k.shape[2]

  b, qpos, h, d = UOp.range(B, 0), UOp.range(N, 1), UOp.range(H, 2), UOp.range(D, 3)
  h_kv = h if group_size == 1 else h // group_size
  kr = UOp.range(N, 10, axis_type=AxisType.REDUCE)
  dscore_sum = UOp.range(D, 11, axis_type=AxisType.REDUCE)
  score_d = UOp.range(D, 12, axis_type=AxisType.REDUCE)
  score = _score_expr(q, k, b, qpos, h, h_kv, kr, score_d)
  p = _prob_from_lse(score, l_vec[b, h, 0, qpos])
  do_dot_v = (dout[b, qpos, h, dscore_sum].cast(dtypes.float32) * v[b, kr, h_kv, dscore_sum].cast(dtypes.float32)).reduce(dscore_sum, arg=Ops.ADD, dtype=dtypes.float32)
  dscore = p * (do_dot_v - delta[b, h, 0, qpos])
  dq_val = (dscore * k[b, kr, h_kv, d].cast(dtypes.float32)).reduce(kr, arg=Ops.ADD, dtype=dtypes.float32) * (1.0 / math.sqrt(D))
  store_dq = dq[b, qpos, h, d].store(dq_val.cast(dq.dtype.base)).end(d, h, qpos, b)

  bgk, kposk, hkvk, dk = UOp.range(B*group_size, 30), UOp.range(N, 31), UOp.range(H_KV, 32), UOp.range(D, 33)
  bk, gk = bgk // group_size, bgk % group_size
  hk = hkvk * group_size + gk
  qrk = UOp.range(N, 34, axis_type=AxisType.REDUCE)
  dk_sum = UOp.range(D, 35, axis_type=AxisType.REDUCE)
  dk_score_d = UOp.range(D, 36, axis_type=AxisType.REDUCE)
  dk_score = _score_expr(q, k, bk, qrk, hk, hkvk, kposk, dk_score_d)
  dk_p = _prob_from_lse(dk_score, l_vec[bk, hk, 0, qrk])
  dk_do_dot_v = (dout[bk, qrk, hk, dk_sum].cast(dtypes.float32) * v[bk, kposk, hkvk, dk_sum].cast(dtypes.float32)).reduce(dk_sum, arg=Ops.ADD, dtype=dtypes.float32)
  dk_dscore = dk_p * (dk_do_dot_v - delta[bk, hk, 0, qrk])
  dk_val = (dk_dscore * q[bk, qrk, hk, dk].cast(dtypes.float32)).reduce(qrk, arg=Ops.ADD, dtype=dtypes.float32) * (1.0 / math.sqrt(D))
  store_dk = dkp[bgk, kposk, hkvk, dk].store(dk_val.cast(dkp.dtype.base)).end(dk, hkvk, kposk, bgk)

  bgv, kposv, hkvv, dv = UOp.range(B*group_size, 50), UOp.range(N, 51), UOp.range(H_KV, 52), UOp.range(D, 53)
  bv, gv = bgv // group_size, bgv % group_size
  hv = hkvv * group_size + gv
  qrv = UOp.range(N, 54, axis_type=AxisType.REDUCE)
  dv_score_d = UOp.range(D, 55, axis_type=AxisType.REDUCE)
  dv_score = _score_expr(q, k, bv, qrv, hv, hkvv, kposv, dv_score_d)
  dv_p = _prob_from_lse(dv_score, l_vec[bv, hv, 0, qrv])
  dv_val = (dv_p * dout[bv, qrv, hv, dv].cast(dtypes.float32)).reduce(qrv, arg=Ops.ADD, dtype=dtypes.float32)
  store_dv = dvp[bgv, kposv, hkvv, dv].store(dv_val.cast(dvp.dtype.base)).end(dv, hkvv, kposv, bgv)

  return UOp.group(store_dq, store_dk, store_dv).sink(arg=KernelInfo(name=f"agentic_fa_bwd_{B}_{N}_{H}_{H_KV}_{D}", opts_to_apply=()))


def _uop_fa_backward_post(dk:UOp, dv:UOp, dkp:UOp, dvp:UOp) -> UOp:
  B, N, H_KV, D = dk.shape
  group_size = dkp.shape[0] // B
  b, kpos, h_kv, d = UOp.range(B, 0), UOp.range(N, 1), UOp.range(H_KV, 2), UOp.range(D, 3)
  gk, gv = UOp.range(group_size, 4, axis_type=AxisType.REDUCE), UOp.range(group_size, 5, axis_type=AxisType.REDUCE)
  dk_val = dkp[b*group_size + gk, kpos, h_kv, d].cast(dtypes.float32).reduce(gk, arg=Ops.ADD, dtype=dtypes.float32)
  dv_val = dvp[b*group_size + gv, kpos, h_kv, d].cast(dtypes.float32).reduce(gv, arg=Ops.ADD, dtype=dtypes.float32)
  store_dk = dk[b, kpos, h_kv, d].store(dk_val.cast(dk.dtype.base))
  store_dv = dv[b, kpos, h_kv, d].store(dv_val.cast(dv.dtype.base))
  return UOp.group(store_dk, store_dv).end(d, h_kv, kpos, b).sink(arg=KernelInfo(name=f"agentic_fa_bwd_post_{B}_{N}_{H_KV}_{D}_{group_size}", opts_to_apply=()))


def flash_attention(xq, xk, xv, attn_mask:Tensor|None=None, is_causal:bool=False):
  # Inputs match flat_llama: xq: (B, N, H, D), xk/xv: (B, N, H_KV, D).
  # The public return mirrors the AMD path: attn plus saved attn/l_vec values.
  if len(xq.shape) == 3: xq, xk, xv = xq.unsqueeze(0), xk.unsqueeze(0), xv.unsqueeze(0)
  if attn_mask is not None: raise RuntimeError("attn_mask not supported")
  if not is_causal: raise RuntimeError("only causal attention supported")
  if len(xq.shape) != 4 or len(xk.shape) != 4 or len(xv.shape) != 4: raise RuntimeError("expected q/k/v shapes (B, N, H, D)")
  if xq.shape[0] != xk.shape[0] or xq.shape[0] != xv.shape[0]: raise RuntimeError("batch mismatch")
  if xq.shape[1] != xk.shape[1] or xq.shape[1] != xv.shape[1]: raise RuntimeError("sequence length mismatch")
  if xq.shape[3] != xk.shape[3] or xq.shape[3] != xv.shape[3]: raise RuntimeError("head dimension mismatch")
  if xq.shape[2] % xk.shape[2] != 0: raise RuntimeError(f"query heads must be a multiple of key/value heads, got {xq.shape[2]=} {xk.shape[2]=}")

  def grad_fxn(grad:UOp, *args):
    kernel = args[-1]
    out_u, l_vec_u, q_u, k_u, v_u = kernel.src[1:]
    out_t = Tensor(out_u.after(kernel), device=out_u.device)
    l_vec_t = Tensor(l_vec_u.after(kernel), device=l_vec_u.device)
    dout_t = Tensor(grad, device=grad.device)
    q_t, k_t, v_t = Tensor(q_u, device=q_u.device), Tensor(k_u, device=k_u.device), Tensor(v_u, device=v_u.device)
    B, N, H, D = q_t.shape
    H_KV, group_size = k_t.shape[2], q_t.shape[2] // k_t.shape[2]

    delta = Tensor.empty(B, H, 1, N, dtype=dtypes.float32, device=q_t.device)
    delta = Tensor.custom_kernel(delta, out_t, dout_t, fxn=_uop_fa_backward_pre)[0]

    dq = Tensor.empty_like(q_t)
    dkp = Tensor.empty(B*group_size, N, H_KV, D, dtype=k_t.dtype, device=k_t.device)
    dvp = Tensor.empty(B*group_size, N, H_KV, D, dtype=v_t.dtype, device=v_t.device)
    dk = Tensor.empty_like(k_t)
    dv = Tensor.empty_like(v_t)

    dq, dkp, dvp = Tensor.custom_kernel(dq, dkp, dvp, dout_t, q_t, k_t, v_t, l_vec_t, delta, fxn=_uop_fa_backward)[:3]

    if group_size == 1:
      dk, dv = dkp, dvp
    else:
      dk, dv = Tensor.custom_kernel(dk, dv, dkp, dvp, fxn=_uop_fa_backward_post)[:2]
    return None, None, dq.uop, dk.uop, dv.uop

  out = xq.empty_like()
  l_vec = Tensor.empty(xq.shape[0], xq.shape[2], 1, xq.shape[1], dtype=dtypes.float32, device=xq.device)
  attn, l_vec = Tensor.custom_kernel(out, l_vec, xq, xk, xv, fxn=_uop_flash_forward, grad_fxn=grad_fxn)[:2]
  return attn, attn, l_vec


def _reference_attention(xq:Tensor, xk:Tensor, xv:Tensor, attn_mask:Tensor|None=None, is_causal:bool=False) -> Tensor:
  return xq.transpose(1, 2).scaled_dot_product_attention(
    xk.transpose(1, 2), xv.transpose(1, 2), attn_mask=attn_mask, is_causal=is_causal,
    enable_gqa=(xq.shape[2] != xk.shape[2])).transpose(1, 2)


def _reference_lse(xq:Tensor, xk:Tensor, attn_mask:Tensor|None=None, is_causal:bool=False) -> Tensor:
  q, k = xq.transpose(1, 2), xk.transpose(1, 2)
  if q.shape[1] != k.shape[1]: k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
  scores = (q.unsqueeze(3).float() * k.unsqueeze(2).float()).sum(-1) * (1.0 / math.sqrt(xq.shape[3]))
  if is_causal: scores = scores + Tensor.full((1, 1, xq.shape[1], xq.shape[1]), -math.inf, dtype=dtypes.float32).triu(1)
  if attn_mask is not None:
    scores = attn_mask.where(scores, -math.inf) if attn_mask.dtype == dtypes.bool else scores + attn_mask.float()
  return scores.logsumexp(axis=-1).unsqueeze(2)


def _assert_close(name:str, got:Tensor, ref:Tensor, atol:float=1e-5, rtol:float=1e-5):
  import numpy as np
  np.testing.assert_allclose(got.numpy(), ref.numpy(), atol=atol, rtol=rtol)
  print(f"ok {name}")


def _kernel_count(t:Tensor) -> int:
  return len(t.schedule_linear().src)


def _forward_kernel_count(*, B:int, N:int, H:int, H_KV:int, D:int, dtype) -> int:
  xq = Tensor.empty(B, N, H, D, dtype=dtype)
  xk = Tensor.empty(B, N, H_KV, D, dtype=dtype)
  xv = Tensor.empty(B, N, H_KV, D, dtype=dtype)
  return _kernel_count(flash_attention(xq, xk, xv, is_causal=True)[0])


def _backward_kernel_count(*, B:int, N:int, H:int, H_KV:int, D:int) -> int:
  xq = Tensor.empty(B, N, H, D, dtype=dtypes.float32, requires_grad=True)
  xk = Tensor.empty(B, N, H_KV, D, dtype=dtypes.float32, requires_grad=True)
  xv = Tensor.empty(B, N, H_KV, D, dtype=dtypes.float32, requires_grad=True)
  grad = Tensor.empty(B, N, H, D, dtype=dtypes.float32)
  out = flash_attention(xq, xk, xv, is_causal=True)[0]
  gq, gk, gv = out.gradient(xq, xk, xv, gradient=grad)
  linear, _ = Tensor.linear_with_vars(gq, gk, gv)
  return len(linear.src)


def _run_forward_case(name:str, *, B:int, N:int, H:int, H_KV:int, D:int, dtype):
  Tensor.manual_seed(1000 + len(name))
  xq = Tensor.randn(B, N, H, D, dtype=dtype).realize()
  xk = Tensor.randn(B, N, H_KV, D, dtype=dtype).realize()
  xv = Tensor.randn(B, N, H_KV, D, dtype=dtype).realize()
  got, _, l_vec = flash_attention(xq, xk, xv, is_causal=True)
  ref = _reference_attention(xq, xk, xv, is_causal=True)
  ref_l = _reference_lse(xq, xk, is_causal=True)
  kcount = _forward_kernel_count(B=B, N=N, H=H, H_KV=H_KV, D=D, dtype=dtype)
  if kcount != 1: raise AssertionError(f"forward {name} expected 1 kernel, got {kcount}")
  print(f"ok forward {name} kernels={kcount}")
  atol = rtol = 2e-2 if dtype == dtypes.bfloat16 else 1e-5
  _assert_close(f"forward {name}", got.float(), ref.float(), atol=atol, rtol=rtol)
  _assert_close(f"forward_lse {name}", l_vec.float(), ref_l.float(), atol=atol, rtol=rtol)


def run_forward_tests():
  cases = [
    ("causal_gqa", dict(B=1, N=8, H=2, H_KV=1, D=4, dtype=dtypes.float32)),
    ("causal_no_gqa", dict(B=2, N=7, H=2, H_KV=2, D=4, dtype=dtypes.float32)),
    ("mixed_gqa_causal", dict(B=2, N=7, H=4, H_KV=2, D=8, dtype=dtypes.float32)),
    ("bf16_causal_gqa", dict(B=1, N=8, H=2, H_KV=1, D=4, dtype=dtypes.bfloat16)),
  ]
  for name, kwargs in cases: _run_forward_case(name, **kwargs)


def _run_backward_case(name:str, *, B:int, N:int, H:int, H_KV:int, D:int):
  import numpy as np
  print(f"info backward {name} kernels={_backward_kernel_count(B=B, N=N, H=H, H_KV=H_KV, D=D)} (custom UOp)")
  Tensor.manual_seed(2000 + len(name))
  xq0 = Tensor.randn(B, N, H, D, dtype=dtypes.float32).realize()
  xk0 = Tensor.randn(B, N, H_KV, D, dtype=dtypes.float32).realize()
  xv0 = Tensor.randn(B, N, H_KV, D, dtype=dtypes.float32).realize()
  grad = Tensor.randn(B, N, H, D, dtype=dtypes.float32).realize()

  xq, xk, xv = Tensor(xq0.numpy(), requires_grad=True), Tensor(xk0.numpy(), requires_grad=True), Tensor(xv0.numpy(), requires_grad=True)
  got = flash_attention(xq, xk, xv, is_causal=True)[0]
  got.backward(grad)
  Tensor.realize(xq.grad, xk.grad, xv.grad)

  rq, rk, rv = Tensor(xq0.numpy(), requires_grad=True), Tensor(xk0.numpy(), requires_grad=True), Tensor(xv0.numpy(), requires_grad=True)
  ref = _reference_attention(rq, rk, rv, is_causal=True)
  ref.backward(grad)
  Tensor.realize(rq.grad, rk.grad, rv.grad)

  for label, a, b in (("q", xq.grad, rq.grad), ("k", xk.grad, rk.grad), ("v", xv.grad, rv.grad)):
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-5, rtol=1e-5)
    print(f"ok backward {name} grad_{label}")


def _main():
  print("agentic_opt.fa diagnostics")
  run_forward_tests()

  _run_backward_case("causal", B=1, N=6, H=2, H_KV=1, D=4)
  _run_backward_case("causal_no_gqa", B=2, N=5, H=2, H_KV=2, D=4)
  _run_backward_case("mixed_gqa_causal", B=2, N=5, H=4, H_KV=2, D=8)

  try:
    xq = Tensor.empty(1, 4, 2, 4)
    xk = Tensor.empty(1, 4, 1, 4)
    xv = Tensor.empty(1, 4, 1, 4)
    attn_mask = Tensor.empty(1, 1, 4, 4, dtype=dtypes.bool)
    flash_attention(xq, xk, xv, attn_mask=attn_mask, is_causal=True)
  except RuntimeError as e:
    print(f"ok mask rejected: {e}")
  else:
    raise AssertionError("expected attn_mask to be rejected")

  try:
    flash_attention(xq, xk, xv, is_causal=False)
  except RuntimeError as e:
    print(f"ok non-causal rejected: {e}")
  else:
    raise AssertionError("expected non-causal attention to be rejected")


if __name__ == "__main__":
  _main()
