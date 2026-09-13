#!/usr/bin/env python3
import functools

from tinygrad import Device, Tensor, dtypes

from extra.llama_kernels.quantize_mxfp4 import alloc_mxfp4_outputs, quantize_mxfp4
from extra.thunder.amd.fa import custom_fused_qkv_rope_backward, custom_fused_qkv_rope_backward_mxfp4


B, N, H, H_KV, D = 2, 8192, 32, 8, 128
GROUP = H // H_KV
PACKED_N = H_KV * (GROUP + 2) * D


def inverse_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  pairs = x.reshape(*x.shape[:-1], D // 2, 2).float()
  freqs = freqs_cis[:, :N].float()
  return Tensor.stack(pairs[..., 0] * freqs[..., 0] + pairs[..., 1] * freqs[..., 1],
                      -pairs[..., 0] * freqs[..., 1] + pairs[..., 1] * freqs[..., 0], dim=-1) \
    .flatten(-2).cast(dtypes.bfloat16)


def check(name:str, actual:Tensor, expected:Tensor, atol:float) -> None:
  max_abs = (actual.float() - expected.float()).abs().max().item()
  correct = actual.allclose(expected, atol=atol, rtol=0).item()
  print(f"{name}: max_abs={max_abs:.8f} atol={atol:g} correct={correct}")
  assert correct, f"{name} mismatch"


def main() -> None:
  assert Device.DEFAULT == "AMD", f"this test requires DEV=AMD, got {Device.DEFAULT}"
  Tensor.manual_seed(0)

  dq = (Tensor.randn(B, N, H, D) * 0.1).cast(dtypes.bfloat16).contiguous().realize()
  dk = (Tensor.randn(B, N, H, D) * 0.1).cast(dtypes.bfloat16).contiguous().realize()
  dv = (Tensor.randn(B, N, H, D) * 0.1).cast(dtypes.bfloat16).contiguous().realize()
  freqs_cis = (Tensor.randn(1, N * 2, 1, D // 2, 2) * 0.1).cast(dtypes.bfloat16).contiguous().realize()

  dxqkv = Tensor.empty(B, N, PACKED_N, dtype=dtypes.bfloat16)
  quant = alloc_mxfp4_outputs(dxqkv, flatten_row=True)
  arch = Device[Device.DEFAULT].renderer.target.arch
  fxn = functools.partial(custom_fused_qkv_rope_backward_mxfp4, device=Device.DEFAULT, arch=arch,
                          B=B, N=N, H=H, H_KV=H_KV, D=D, expanded_fa_grads=True)
  dxqkv, row_fp4, row_scale, col_fp4, col_scale, *_ = \
    Tensor.custom_kernel(dxqkv, *quant, dq, dk, dv, freqs_cis, fxn=fxn)
  # The BF16 dxqkv is a logical autograd/mailbox key only. The MXFP4 kernel deliberately leaves its storage unwritten.
  Tensor.realize(row_fp4, row_scale, col_fp4, col_scale)

  dq_ref = inverse_rope(dq, freqs_cis).reshape(B, N, H_KV, GROUP, D)
  dk_sum = dk.float().reshape(B, N, H_KV, GROUP, D).sum(3).cast(dtypes.bfloat16)
  dv_sum = dv.float().reshape(B, N, H_KV, GROUP, D).sum(3).cast(dtypes.bfloat16)
  dxqkv_ref = Tensor.cat(dq_ref, inverse_rope(dk_sum, freqs_cis).unsqueeze(3), dv_sum.unsqueeze(3), dim=3) \
    .reshape(B, N, PACKED_N)
  quant_ref = quantize_mxfp4(dxqkv_ref, flatten_row=True)
  Tensor.realize(dxqkv_ref, *quant_ref)

  print(f"dq={dq.shape} dk={dk.shape} dv={dv.shape} freqs_cis={freqs_cis.shape}")
  print(f"dxqkv={dxqkv.shape} row_fp4={row_fp4.shape} row_scale={row_scale.shape} "
        f"col_fp4={col_fp4.shape} col_scale={col_scale.shape}")
  for name, actual, expected in zip(("row_fp4", "row_scale", "col_fp4", "col_scale"),
                                    (row_fp4, row_scale, col_fp4, col_scale), quant_ref):
    check(name, actual, expected, 0.0)

  # Exercise the packed dQ layout written by the fast FP8 kernel, including the
  # native-gradient routing which previously discarded its unpacking view.
  from extra.thunder.amd.fa import _fa_native_grads
  from extra.thunder.amd.fa_fp8_bwd import unpack_dq, cast_gradients
  packed = dq.reshape(B,N//16,4,2,2,H,8,16).permute(0,5,1,6,3,2,7,4).contiguous().reshape(B,N,H,D)
  raw = Tensor.custom_kernel(*(Tensor.empty_like(x) for x in (dq,dk,dv)),packed,dk,dv,fxn=cast_gradients)[:3]
  logical_dq = unpack_dq(raw[0])
  logical_dk,logical_dv = [x.reshape(B,N,H_KV,GROUP,D).sum(3) for x in raw[1:]]
  native = _fa_native_grads(logical_dq.uop,logical_dk.uop,logical_dv.uop)
  assert native is not None and native[3:] == (True,True)
  contiguous = _fa_native_grads(raw[0].uop,logical_dk.uop,logical_dv.uop)
  assert contiguous is not None and contiguous[3:] == (True,False)
  output = Tensor.empty(B,N,PACKED_N,dtype=dtypes.bfloat16)
  packed_quant = alloc_mxfp4_outputs(output,flatten_row=True)
  ret = Tensor.custom_kernel(output,*packed_quant,*(Tensor(x) for x in native[:3]),freqs_cis,
    fxn=functools.partial(custom_fused_qkv_rope_backward_mxfp4,device=Device.DEFAULT,arch=arch,
                          B=B,N=N,H=H,H_KV=H_KV,D=D,expanded_fa_grads=native[3],packed_fp8_dq=native[4]))
  for name,actual,expected in zip(("row_fp4","row_scale","col_fp4","col_scale"),ret[1:5],quant_ref):
    check(f"packed dQ {name}",actual,expected,0.0)
  plain = Tensor.custom_kernel(Tensor.empty_like(output),*(Tensor(x) for x in native[:3]),freqs_cis,
    fxn=functools.partial(custom_fused_qkv_rope_backward,device=Device.DEFAULT,arch=arch,
                          B=B,N=N,H=H,H_KV=H_KV,D=D,expanded_fa_grads=native[3],packed_fp8_dq=native[4]))[0]
  check("packed dQ BF16 RoPE",plain,dxqkv_ref,0.0)


if __name__ == "__main__":
  main()
