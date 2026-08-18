import functools, unittest

from tinygrad import Context, Device, Tensor, dtypes
import extra.thunder.amd.fa as fa


B, N, H, H_KV, D = 2, 8192, 32, 8, 128


def asm_forward(o, lse, q, k, v):
  return fa.custom_asm_fa_forward(o, lse, q, k, v, B=B, N=N, H=H, H_KV=H_KV, D=D)


def run_forward(forward, q:Tensor, k:Tensor, v:Tensor) -> tuple[Tensor, Tensor]:
  out = Tensor.invalids(B, N, H, D, dtype=dtypes.bfloat16)
  lse = Tensor.invalids(B, H, 1, N, dtype=dtypes.float32)
  out, lse = Tensor.custom_kernel(out, lse, q, k, v, fxn=forward)[:2]
  Tensor.realize(out, lse)
  return out, lse


def run_backward(out:Tensor, lse:Tensor, q:Tensor, k:Tensor, v:Tensor, dout:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  arch = Device[Device.DEFAULT].renderer.target.arch
  group_size, heads_per_wg = H // H_KV, 2

  dq = Tensor.invalids(B, H, N, D, dtype=dtypes.bfloat16)
  delta = Tensor.invalids(B, H, 1, N, dtype=dtypes.float32)
  pre = functools.partial(fa.custom_fa_backward_pre, device=Device.DEFAULT, arch=arch, B=B, N=N, H=H, H_KV=H_KV, D=D)
  delta, dq = Tensor.custom_kernel(delta, dq, out, dout, fxn=pre)[:2]

  dk_partial = Tensor.invalids(B * group_size // heads_per_wg, N, H_KV, D, dtype=dtypes.bfloat16)
  dv_partial = Tensor.invalids(B * group_size // heads_per_wg, N, H_KV, D, dtype=dtypes.bfloat16)
  bwd = functools.partial(fa.custom_fa_backward, device=Device.DEFAULT, arch=arch, B=B, N=N, H=H, H_KV=H_KV, D=D)
  dq, dk_partial, dv_partial = Tensor.custom_kernel(dq, dk_partial, dv_partial, dout, q, k, v, lse, delta, fxn=bwd)[:3]

  dq = dq.reshape(B, H, N//16, 4, 2, 2, D//32, 4, 4, 2).permute(0, 1, 2, 7, 8, 3, 4, 6, 5, 9).reshape(B, H, N, D).transpose(1, 2)
  dk = dk_partial.reshape(B, group_size // heads_per_wg, N, H_KV, D).sum(1)
  dv = dv_partial.reshape(B, group_size // heads_per_wg, N, H_KV, D).sum(1)
  Tensor.realize(dq, dk, dv)
  return dq, dk, dv


def relative_l2(a:Tensor, b:Tensor) -> float:
  diff = a.float() - b.float()
  return float((diff.square().sum() / b.float().square().sum()).sqrt().item())


class TestAsmFANumerics(unittest.TestCase):
  def test_localizes_forward_mismatch(self):
    if Device[Device.DEFAULT].renderer.target.arch != "gfx950": self.skipTest("translated FA requires gfx950")

    Tensor.manual_seed(42)
    with Context(DEBUG=0):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      dout = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16).contiguous()
      Tensor.realize(q, k, v, dout)

    hk_forward = functools.partial(fa.custom_hk_fa_forward, device=Device.DEFAULT,
      arch=Device[Device.DEFAULT].renderer.target.arch, B=B, N=N, H=H, H_KV=H_KV, D=D, has_sink=False)
    asm_out, asm_lse = run_forward(asm_forward, q, k, v)
    hk_out, hk_lse = run_forward(hk_forward, q, k, v)

    hk_grads = run_backward(hk_out, hk_lse, q, k, v, dout)
    lse_only_grads = run_backward(hk_out, asm_lse, q, k, v, dout)
    out_only_grads = run_backward(asm_out, hk_lse, q, k, v, dout)
    asm_grads = run_backward(asm_out, asm_lse, q, k, v, dout)

    errors = {"out": relative_l2(asm_out, hk_out), "lse": relative_l2(asm_lse, hk_lse)}
    for prefix, grads in (("lse_only", lse_only_grads), ("out_only", out_only_grads), ("combined", asm_grads)):
      errors.update({f"{prefix}_{name}": relative_l2(got, ref) for name, got, ref in zip(("dq", "dk", "dv"), grads, hk_grads)})

    # The translated Aiter forward is close elementwise, but its output and LSE independently move the shared backward.
    self.assertGreater(errors["out"], 2e-3)
    self.assertGreater(errors["lse_only_dq"], 2e-3)
    self.assertGreater(errors["out_only_dq"], 2e-3)
    self.assertGreater(errors["combined_dq"], 2e-3)


if __name__ == "__main__": unittest.main()
