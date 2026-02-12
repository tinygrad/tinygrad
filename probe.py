import pathlib, ctypes, struct, math
import numpy as np
from tinygrad import Tensor, dtypes, Device
from aiter_args import FwdArgs, BwdOdoArgs, BwdMainArgs, BwdDqConvertArgs

def float_to_u32(f: float) -> int:
  return struct.unpack('I', struct.pack('f', f))[0]

Device.DEFAULT = "HIP"
dev = Device[Device.DEFAULT]

CO_DIR = pathlib.Path("/home/qazal/code/tinygrad/extra/aiter/hsa/gfx950")
binary = (CO_DIR / "fmha_v3_fwd/fwd_hd128_bf16_causal.co").read_bytes()

B, H, S, D = 8, 8, 8192, 128

Tensor.manual_seed(0)
q, k, v = [Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16) for _ in range(3)]
out_aiter = Tensor.empty(B, S, H, D, dtype=dtypes.bfloat16)

q_aiter = q.permute(0, 2, 1, 3).contiguous()
k_aiter = k.permute(0, 2, 1, 3).contiguous()
v_aiter = v.permute(0, 2, 1, 3).contiguous()
lse = Tensor.empty(B, H, S, dtype=dtypes.float32)  # logsumexp for backward
inputs = [out_aiter, q_aiter, k_aiter, v_aiter, lse]
Tensor.realize(*inputs)

want = Tensor.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

bufs = [t.uop.buffer.ensure_allocated()._buf for t in inputs]

args = FwdArgs()

# Pointers
args.R = bufs[0].value
args.Q = bufs[1].value
args.K = bufs[2].value
args.V = bufs[3].value
args.LSE = bufs[4].value
args.ptr_qseq = 0
args.ptr_kseq = 0
args.ptr_qseq_padding = 0
args.ptr_kseq_padding = 0

# Simple scalars (computed from B, H, S, D)
args.scalar = float_to_u32(1.0 / math.sqrt(D))  # softmax_scale
args.seq_len = S
args.kv_seq_len = S
args.qk_head_dim = D
args.v_head_dim = D
args.q_head_num = H
args.gqa = 1          # nhead_q // nhead_k (no GQA)
args.msk_opt = 5      # causal mask
args.lse = 1          # return LSE (needed for backward)
args.lse_Hs = S * 4   # LSE head stride in bytes (fp32)

# Byte strides for BSHD layout tensors (bf16 = 2 bytes)
elem_size = 2
Seqs_stride = H * D * elem_size       # seqlen stride
Hs_stride = D * elem_size             # head stride
Bs_stride = S * H * D * elem_size     # batch stride

args.Seqs = Seqs_stride               # 2048
args.Ts = S * D // 2                  # tile stride: S * D / 2 = 524288
args.Hs = Hs_stride                   # 256
args.Bs = Bs_stride                   # 16777216
args.k_Seqs = Seqs_stride
args.k_Hs = Hs_stride
args.k_Bs = Bs_stride
args.v_Seqs = Seqs_stride
args.v_Hs = Hs_stride
args.v_Bs = Bs_stride
args.r_Seqs = Seqs_stride
args.r_Hs = Hs_stride
args.r_Bs = Bs_stride

arg_size = ctypes.c_size_t(ctypes.sizeof(args))
blob = (ctypes.c_ubyte * ctypes.sizeof(args)).from_buffer_copy(ctypes.string_at(ctypes.addressof(args), ctypes.sizeof(args)))
extra = (ctypes.c_void_p * 5)(1, ctypes.cast(ctypes.byref(blob), ctypes.c_void_p), 2,
                                 ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p), 3)

prg = dev.runtime("_ZN5aiter26fmha_fwd_hd128_bf16_causalE", binary)
prg.vargs = extra

# Grid: (seqlen / tile_size, nhead, batch) where tile_size = 512 for this kernel
grid = (S // 512, H, B)
et = prg(global_size=grid, local_size=(512,1,1), wait=True)
print(f"Forward: {et*1e3:.3f} ms")

out_aiter_bhsd = out_aiter.permute(0, 2, 1, 3).contiguous()
np.testing.assert_allclose(out_aiter_bhsd.numpy(), want.numpy(), rtol=2e-2, atol=2e-2)
print("Forward: PASSED")


# ============================================================================
# BACKWARD
# ============================================================================

# Create gradient of output (dO) and D buffer for rowsum(O * dO)
dout = Tensor.randn(B, H, S, D, dtype=dtypes.bfloat16)
dout_aiter = dout.permute(0, 2, 1, 3).contiguous()  # BSHD layout
delta = Tensor.empty(B, H, S, dtype=dtypes.float32)  # D = rowsum(O * dO)
Tensor.realize(dout_aiter, delta)

# --- Backward ODO kernel: computes D = rowsum(O * dO) ---
odo_binary = (CO_DIR / "fmha_v3_bwd/bwd_hd128_odo_bf16.co").read_bytes()

odo_args = BwdOdoArgs()
odo_args.O = bufs[0].value       # forward output (BSHD)
odo_args.dO = dout_aiter.uop.buffer.ensure_allocated()._buf.value
odo_args.D = delta.uop.buffer.ensure_allocated()._buf.value

# Strides for O/dO in BSHD layout (bf16)
odo_args.Hs_odo = D * elem_size           # head stride = 256
odo_args.BAs_odo = S * H * D * elem_size  # batch stride = 16777216
odo_args.Seqs_odo = H * D * elem_size     # seq stride = 2048

# Strides for D output [B, H, S] (fp32)
odo_args.Hs_d = S * 4             # head stride = 32768
odo_args.BAs_d = H * S * 4        # batch stride = 262144
odo_args.Seqs_d = 4               # seq stride = 4 (one fp32 element)

odo_args.seqlen_q = S
odo_args.head_dim = D
odo_args.ptr_seqstart_q = 0
odo_args.ptr_seqstart_q_padded = 0

odo_arg_size = ctypes.c_size_t(ctypes.sizeof(odo_args))
odo_blob = (ctypes.c_ubyte * ctypes.sizeof(odo_args)).from_buffer_copy(
  ctypes.string_at(ctypes.addressof(odo_args), ctypes.sizeof(odo_args)))
odo_extra = (ctypes.c_void_p * 5)(1, ctypes.cast(ctypes.byref(odo_blob), ctypes.c_void_p), 2,
                                   ctypes.cast(ctypes.pointer(odo_arg_size), ctypes.c_void_p), 3)

odo_prg = dev.runtime("_ZN5aiter23fmha_bwd_hd128_odo_bf16E", odo_binary)
odo_prg.vargs = odo_extra

# Grid for ODO: (seqlen / 128, nhead, batch), block=(256,1,1)
odo_grid = (S // 128, H, B)
odo_et = odo_prg(global_size=odo_grid, local_size=(256, 1, 1), wait=True)
print(f"Backward ODO: {odo_et*1e3:.3f} ms")


# --- Backward Main kernel: computes dQ (fp32), dK, dV ---
main_binary = (CO_DIR / "fmha_v3_bwd/bwd_hd128_bf16_causal_br_a32_psskddv.co").read_bytes()

# Output buffers - note: dQ acc has shape [1, B, H, S, D] for atomic accumulation
dq_acc = Tensor.zeros(1, B, H, S, D, dtype=dtypes.float32).contiguous()  # dQ accumulator
dk_aiter = Tensor.empty(B, S, H, D, dtype=dtypes.bfloat16)  # dK (BSHD bf16)
dv_aiter = Tensor.empty(B, S, H, D, dtype=dtypes.bfloat16)  # dV (BSHD bf16)
Tensor.realize(dq_acc, dk_aiter, dv_aiter)

main_args = BwdMainArgs()

# Output pointers
main_args.dQ = dq_acc.uop.buffer.ensure_allocated()._buf.value
main_args.dK = dk_aiter.uop.buffer.ensure_allocated()._buf.value
main_args.dV = dv_aiter.uop.buffer.ensure_allocated()._buf.value

# Input pointers
main_args.Q = bufs[1].value   # Q (BSHD)
main_args.K = bufs[2].value   # K (BSHD)
main_args.V = bufs[3].value   # V (BSHD)
main_args.dO = dout_aiter.uop.buffer.ensure_allocated()._buf.value
main_args.Lse = bufs[4].value  # LSE from forward
main_args.D = delta.uop.buffer.ensure_allocated()._buf.value

# Scalars
main_args.scalar = float_to_u32(1.0 / math.sqrt(D))
main_args.log2e = float_to_u32(math.log2(math.e))  # 1.4426950408889634
main_args.seqlen_q = S
main_args.seqlen_k = S
main_args.head_dim_q = D
main_args.head_dim_k = D
main_args.nhead_q = H
main_args.ratio = 1  # GQA ratio
main_args.Ts = S * D // 2

# Q/K/V input strides (BSHD bf16)
main_args.Seqs_q = H * D * elem_size
main_args.Hs_q = D * elem_size
main_args.BAs_q = S * H * D * elem_size
main_args.Seqs_k = H * D * elem_size
main_args.Hs_k = D * elem_size
main_args.BAs_k = S * H * D * elem_size
main_args.Seqs_v = H * D * elem_size
main_args.Hs_v = D * elem_size
main_args.BAs_v = S * H * D * elem_size

# dO input strides (BSHD bf16)
main_args.Seqs_do = H * D * elem_size
main_args.Hs_do = D * elem_size
main_args.BAs_do = S * H * D * elem_size

# dK/dV output strides (BSHD bf16)
main_args.Seqs_dk = H * D * elem_size
main_args.Hs_dk = D * elem_size
main_args.BAs_dk = S * H * D * elem_size
main_args.Seqs_dv = H * D * elem_size
main_args.Hs_dv = D * elem_size
main_args.BAs_dv = S * H * D * elem_size

# LSE/D strides [B, H, S] fp32
main_args.Hs_lsed = S * 4

# Optional pointers (not used)
main_args.ptr_seqstart_q = 0
main_args.ptr_seqstart_k = 0
main_args.ptr_seqstart_q_padded = 0
main_args.ptr_seqstart_k_padded = 0
main_args.max_seq_len_dq = S
main_args.mask_x = 0
main_args.mask_y = 0

main_arg_size = ctypes.c_size_t(ctypes.sizeof(main_args))
main_blob = (ctypes.c_ubyte * ctypes.sizeof(main_args)).from_buffer_copy(
  ctypes.string_at(ctypes.addressof(main_args), ctypes.sizeof(main_args)))
main_extra = (ctypes.c_void_p * 5)(1, ctypes.cast(ctypes.byref(main_blob), ctypes.c_void_p), 2,
                                    ctypes.cast(ctypes.pointer(main_arg_size), ctypes.c_void_p), 3)

main_prg = dev.runtime("_ZN5aiter41fmha_bwd_hd128_bf16_causal_br_a32_psskddvE", main_binary)
main_prg.vargs = main_extra

# Grid for main bwd: (seqlen / 512, nhead, batch), block=(256,1,1)
main_grid = (S // 512, H, B)
main_et = main_prg(global_size=main_grid, local_size=(256, 1, 1), wait=True)
print(f"Backward Main: {main_et*1e3:.3f} ms")

# --- dQ Convert kernel: converts fp32 dQ accumulator to bf16 ---
dq_convert_binary = (CO_DIR / "fmha_v3_bwd/bwd_hd128_dq_convert_bf16.co").read_bytes()

dq_aiter = Tensor.empty(B, S, H, D, dtype=dtypes.bfloat16)  # final dQ (BSHD bf16)
Tensor.realize(dq_aiter)

dq_conv_args = BwdDqConvertArgs()
dq_conv_args.dQ_acc = dq_acc.uop.buffer.ensure_allocated()._buf.value  # input (BHSD fp32)
dq_conv_args.dQ = dq_aiter.uop.buffer.ensure_allocated()._buf.value     # output (BSHD bf16)

# dQ_acc strides (BHSD fp32)
dq_conv_args.Seqs_dQ_acc = D * 4          # seq stride = 512
dq_conv_args.Hs_dQ_acc = S * D * 4        # head stride = 4194304
dq_conv_args.BAs_dQ_acc = H * S * D * 4   # batch stride = 33554432

# dQ output strides (BSHD bf16)
dq_conv_args.Seqs_dQ = H * D * elem_size  # 2048
dq_conv_args.Hs_dQ = D * elem_size        # 256
dq_conv_args.BAs_dQ = S * H * D * elem_size  # 16777216

dq_conv_args.seqlen_q = S
dq_conv_args.head_dim = D
dq_conv_args.ptr_seqstart_q = 0
dq_conv_args.ptr_seqstart_q_padded = 0
dq_conv_args.max_seqlen_dq = S

dq_conv_arg_size = ctypes.c_size_t(ctypes.sizeof(dq_conv_args))
dq_conv_blob = (ctypes.c_ubyte * ctypes.sizeof(dq_conv_args)).from_buffer_copy(
  ctypes.string_at(ctypes.addressof(dq_conv_args), ctypes.sizeof(dq_conv_args)))
dq_conv_extra = (ctypes.c_void_p * 5)(1, ctypes.cast(ctypes.byref(dq_conv_blob), ctypes.c_void_p), 2,
                                       ctypes.cast(ctypes.pointer(dq_conv_arg_size), ctypes.c_void_p), 3)

dq_conv_prg = dev.runtime("_ZN5aiter30fmha_bwd_hd128_dq_convert_bf16E", dq_convert_binary)
dq_conv_prg.vargs = dq_conv_extra

# Grid for dQ convert: (seqlen / 64, nhead, batch), block=(256,1,1)
dq_conv_grid = (S // 64, H, B)
dq_conv_et = dq_conv_prg(global_size=dq_conv_grid, local_size=(256, 1, 1), wait=True)
print(f"Backward dQ Convert: {dq_conv_et*1e3:.3f} ms")

# ============================================================================
# COMPARE WITH TINYGRAD
# ============================================================================

# Convert AITER outputs from BSHD to BHSD for comparison
dq_aiter_bhsd = dq_aiter.permute(0, 2, 1, 3).contiguous()
dk_aiter_bhsd = dk_aiter.permute(0, 2, 1, 3).contiguous()
dv_aiter_bhsd = dv_aiter.permute(0, 2, 1, 3).contiguous()

dq_np = dq_aiter_bhsd.numpy()
dk_np = dk_aiter_bhsd.numpy()
dv_np = dv_aiter_bhsd.numpy()

# Get tinygrad's backward gradients - use fresh tensors from numpy to avoid graph issues
q_np, k_np, v_np = q.numpy(), k.numpy(), v.numpy()
dout_np = dout.numpy()

q2 = Tensor(q_np, dtype=dtypes.bfloat16, requires_grad=True)
k2 = Tensor(k_np, dtype=dtypes.bfloat16, requires_grad=True)
v2 = Tensor(v_np, dtype=dtypes.bfloat16, requires_grad=True)
want2 = Tensor.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0, is_causal=True)
dq_want, dk_want, dv_want = Tensor.gradient(want2, q2, k2, v2, gradient=Tensor(dout_np, dtype=dtypes.bfloat16))

print("\n=== Gradient Comparison ===")
dq_want_np = dq_want.numpy()
dk_want_np = dk_want.numpy()
dv_want_np = dv_want.numpy()

# Check dK and dV first (these should be close)
np.testing.assert_allclose(dk_np, dk_want_np, rtol=2e-2, atol=2e-2)
print("dK: PASSED")
np.testing.assert_allclose(dv_np, dv_want_np, rtol=2e-2, atol=2e-2)
print("dV: PASSED")

np.testing.assert_allclose(dq_np, dq_want_np, rtol=2e-2, atol=2e-2)
print("dQ: PASSED")
