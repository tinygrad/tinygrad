import os
os.environ["AITER_ASM_DIR"] = "/home/qazal/code/aiter/hsa/"
import torch
import torch.nn.functional as F
import aiter

B, H, S, D = 8, 8, 8192, 128
device = "cuda:5"

torch.manual_seed(0)
# Create inputs with requires_grad for backward
q = (torch.randn(B, H, S, D, device=device, dtype=torch.float32) - 0.5).to(torch.bfloat16).contiguous().requires_grad_(True)
k = (torch.randn(B, H, S, D, device=device, dtype=torch.float32) - 0.5).to(torch.bfloat16).contiguous().requires_grad_(True)
v = (torch.randn(B, H, S, D, device=device, dtype=torch.float32) - 0.5).to(torch.bfloat16).contiguous().requires_grad_(True)

# Reference: torch SDPA forward + backward
out_torch = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
dout = torch.randn_like(out_torch)
dq_torch, dk_torch, dv_torch = torch.autograd.grad(out_torch, (q, k, v), dout, retain_graph=True)

print("torch forward:", out_torch.flatten()[-5:].detach().float().cpu().numpy())
print("torch dq:", dq_torch.flatten()[-5:].detach().float().cpu().numpy())
print("torch dk:", dk_torch.flatten()[-5:].detach().float().cpu().numpy())
print("torch dv:", dv_torch.flatten()[-5:].detach().float().cpu().numpy())

# aiter: needs BSHD layout
q_aiter = q.permute(0, 2, 1, 3).contiguous().requires_grad_(True)
k_aiter = k.permute(0, 2, 1, 3).contiguous().requires_grad_(True)
v_aiter = v.permute(0, 2, 1, 3).contiguous().requires_grad_(True)

out_aiter, softmax_lse = aiter.flash_attn_func(
  q_aiter, k_aiter, v_aiter,
  dropout_p=0.0,
  causal=True,
  window_size=(-1, -1),
  deterministic=False,  # v3 bwd needs non-deterministic
  return_lse=True,      # backward needs lse
)
out_aiter_bhsd = out_aiter.permute(0, 2, 1, 3).contiguous()

# backward through aiter
dout_aiter = dout.permute(0, 2, 1, 3).contiguous()
dq_aiter, dk_aiter, dv_aiter = torch.autograd.grad(out_aiter, (q_aiter, k_aiter, v_aiter), dout_aiter)

# permute gradients back to BHSD
dq_aiter_bhsd = dq_aiter.permute(0, 2, 1, 3).contiguous()
dk_aiter_bhsd = dk_aiter.permute(0, 2, 1, 3).contiguous()
dv_aiter_bhsd = dv_aiter.permute(0, 2, 1, 3).contiguous()

print("\naiter forward:", out_aiter_bhsd.flatten()[-5:].detach().float().cpu().numpy())
print("aiter dq:", dq_aiter_bhsd.flatten()[-5:].detach().float().cpu().numpy())
print("aiter dk:", dk_aiter_bhsd.flatten()[-5:].detach().float().cpu().numpy())
print("aiter dv:", dv_aiter_bhsd.flatten()[-5:].detach().float().cpu().numpy())

# Compare
print("\n=== Comparison ===")
torch.testing.assert_close(out_aiter_bhsd, out_torch, rtol=2e-2, atol=2e-2)
print("forward: PASS")
torch.testing.assert_close(dq_aiter_bhsd, dq_torch, rtol=2e-2, atol=2e-2)
print("dq: PASS")
torch.testing.assert_close(dk_aiter_bhsd, dk_torch, rtol=2e-2, atol=2e-2)
print("dk: PASS")
torch.testing.assert_close(dv_aiter_bhsd, dv_torch, rtol=2e-2, atol=2e-2)
print("dv: PASS")
