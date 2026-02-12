"""ASM SDPA (Scaled Dot-Product Attention) using aiter FMHA kernels."""
import pathlib, ctypes, struct, math, functools, atexit
from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates

def float_to_u32(f: float) -> int:
  return struct.unpack('I', struct.pack('f', f))[0]

CO_DIR = pathlib.Path(__file__).parent / "atn" / "co"

# ============================================================================
# KERNEL ARGUMENT STRUCTURES
# ============================================================================

class FwdArgs(ctypes.Structure):
  _pack_ = 1
  _fields_ = [
    ("R", ctypes.c_uint64), ("R_pad", ctypes.c_uint64),
    ("Q", ctypes.c_uint64), ("Q_pad", ctypes.c_uint64),
    ("K", ctypes.c_uint64), ("K_pad", ctypes.c_uint64),
    ("V", ctypes.c_uint64), ("V_pad", ctypes.c_uint64),
    ("LSE", ctypes.c_uint64), ("LSE_pad", ctypes.c_uint64),
    ("scalar", ctypes.c_uint32), ("scalar_pad", ctypes.c_uint32 * 3),
    ("seq_len", ctypes.c_uint32), ("seq_len_pad", ctypes.c_uint32 * 3),
    ("Seqs", ctypes.c_uint32), ("Seqs_pad", ctypes.c_uint32 * 3),
    ("Ts", ctypes.c_uint32), ("Ts_pad", ctypes.c_uint32 * 3),
    ("Hs", ctypes.c_uint32), ("Hs_pad", ctypes.c_uint32 * 3),
    ("Bs", ctypes.c_uint32), ("Bs_pad", ctypes.c_uint32 * 3),
    ("gqa", ctypes.c_uint32), ("gqa_pad", ctypes.c_uint32 * 3),
    ("k_Seqs", ctypes.c_uint32), ("k_Seqs_pad", ctypes.c_uint32 * 3),
    ("k_Hs", ctypes.c_uint32), ("k_Hs_pad", ctypes.c_uint32 * 3),
    ("k_Bs", ctypes.c_uint32), ("k_Bs_pad", ctypes.c_uint32 * 3),
    ("msk_opt", ctypes.c_uint32), ("msk_opt_pad", ctypes.c_uint32 * 3),
    ("lse", ctypes.c_uint32), ("lse_pad", ctypes.c_uint32 * 3),
    ("kv_seq_len", ctypes.c_uint32), ("kv_seq_len_pad", ctypes.c_uint32 * 3),
    ("qk_head_dim", ctypes.c_uint32), ("qk_head_dim_pad", ctypes.c_uint32 * 3),
    ("v_head_dim", ctypes.c_uint32), ("v_head_dim_pad", ctypes.c_uint32 * 3),
    ("q_head_num", ctypes.c_uint32), ("q_head_num_pad", ctypes.c_uint32 * 3),
    ("v_Seqs", ctypes.c_uint32), ("v_Seqs_pad", ctypes.c_uint32 * 3),
    ("v_Hs", ctypes.c_uint32), ("v_Hs_pad", ctypes.c_uint32 * 3),
    ("v_Bs", ctypes.c_uint32), ("v_Bs_pad", ctypes.c_uint32 * 3),
    ("r_Seqs", ctypes.c_uint32), ("r_Seqs_pad", ctypes.c_uint32 * 3),
    ("r_Hs", ctypes.c_uint32), ("r_Hs_pad", ctypes.c_uint32 * 3),
    ("r_Bs", ctypes.c_uint32), ("r_Bs_pad", ctypes.c_uint32 * 3),
    ("ptr_qseq", ctypes.c_uint64), ("ptr_qseq_pad", ctypes.c_uint64),
    ("ptr_kseq", ctypes.c_uint64), ("ptr_kseq_pad", ctypes.c_uint64),
    ("lse_Hs", ctypes.c_uint32), ("lse_Hs_pad", ctypes.c_uint32 * 3),
    ("ptr_qseq_padding", ctypes.c_uint64), ("ptr_qseq_padding_pad", ctypes.c_uint64),
    ("ptr_kseq_padding", ctypes.c_uint64), ("ptr_kseq_padding_pad", ctypes.c_uint64),
  ]
assert ctypes.sizeof(FwdArgs) == 512

class BwdOdoArgs(ctypes.Structure):
  _pack_ = 1
  _fields_ = [
    ("O", ctypes.c_uint64), ("O_pad", ctypes.c_uint64),
    ("dO", ctypes.c_uint64), ("dO_pad", ctypes.c_uint64),
    ("D", ctypes.c_uint64), ("D_pad", ctypes.c_uint64),
    ("Hs_odo", ctypes.c_uint32), ("Hs_odo_pad", ctypes.c_uint32 * 3),
    ("BAs_odo", ctypes.c_uint32), ("BAs_odo_pad", ctypes.c_uint32 * 3),
    ("Seqs_odo", ctypes.c_uint32), ("Seqs_odo_pad", ctypes.c_uint32 * 3),
    ("Hs_d", ctypes.c_uint32), ("Hs_d_pad", ctypes.c_uint32 * 3),
    ("BAs_d", ctypes.c_uint32), ("BAs_d_pad", ctypes.c_uint32 * 3),
    ("Seqs_d", ctypes.c_uint32), ("Seqs_d_pad", ctypes.c_uint32 * 3),
    ("seqlen_q", ctypes.c_uint32), ("seqlen_q_pad", ctypes.c_uint32 * 3),
    ("head_dim", ctypes.c_uint32), ("head_dim_pad", ctypes.c_uint32 * 3),
    ("ptr_seqstart_q", ctypes.c_uint32), ("ptr_seqstart_q_pad", ctypes.c_uint32 * 3),
    ("ptr_seqstart_q_padded", ctypes.c_uint32), ("ptr_seqstart_q_padded_pad", ctypes.c_uint32 * 3),
  ]
assert ctypes.sizeof(BwdOdoArgs) == 208

class BwdMainArgs(ctypes.Structure):
  _pack_ = 1
  _fields_ = [
    ("dQ", ctypes.c_uint64), ("dQ_pad", ctypes.c_uint64),
    ("dK", ctypes.c_uint64), ("dK_pad", ctypes.c_uint64),
    ("dV", ctypes.c_uint64), ("dV_pad", ctypes.c_uint64),
    ("Q", ctypes.c_uint64), ("Q_pad", ctypes.c_uint64),
    ("K", ctypes.c_uint64), ("K_pad", ctypes.c_uint64),
    ("V", ctypes.c_uint64), ("V_pad", ctypes.c_uint64),
    ("dO", ctypes.c_uint64), ("dO_pad", ctypes.c_uint64),
    ("Lse", ctypes.c_uint64), ("Lse_pad", ctypes.c_uint64),
    ("D", ctypes.c_uint64), ("D_pad", ctypes.c_uint64),
    ("scalar", ctypes.c_uint32), ("scalar_pad", ctypes.c_uint32 * 3),
    ("log2e", ctypes.c_uint32), ("log2e_pad", ctypes.c_uint32 * 3),
    ("seqlen_q", ctypes.c_uint32), ("seqlen_q_pad", ctypes.c_uint32 * 3),
    ("Ts", ctypes.c_uint32), ("Ts_pad", ctypes.c_uint32 * 3),
    ("Hs_q", ctypes.c_uint32), ("Hs_q_pad", ctypes.c_uint32 * 3),
    ("BAs_q", ctypes.c_uint32), ("BAs_q_pad", ctypes.c_uint32 * 3),
    ("Seqs_q", ctypes.c_uint32), ("Seqs_q_pad", ctypes.c_uint32 * 3),
    ("ratio", ctypes.c_uint32), ("ratio_pad", ctypes.c_uint32 * 3),
    ("Hs_k", ctypes.c_uint32), ("Hs_k_pad", ctypes.c_uint32 * 3),
    ("BAs_k", ctypes.c_uint32), ("BAs_k_pad", ctypes.c_uint32 * 3),
    ("Seqs_k", ctypes.c_uint32), ("Seqs_k_pad", ctypes.c_uint32 * 3),
    ("Seqs_dk", ctypes.c_uint32), ("Seqs_dk_pad", ctypes.c_uint32 * 3),
    ("seqlen_k", ctypes.c_uint32), ("seqlen_k_pad", ctypes.c_uint32 * 3),
    ("head_dim_q", ctypes.c_uint32), ("head_dim_q_pad", ctypes.c_uint32 * 3),
    ("head_dim_k", ctypes.c_uint32), ("head_dim_k_pad", ctypes.c_uint32 * 3),
    ("nhead_q", ctypes.c_uint32), ("nhead_q_pad", ctypes.c_uint32 * 3),
    ("Hs_v", ctypes.c_uint32), ("Hs_v_pad", ctypes.c_uint32 * 3),
    ("BAs_v", ctypes.c_uint32), ("BAs_v_pad", ctypes.c_uint32 * 3),
    ("Seqs_v", ctypes.c_uint32), ("Seqs_v_pad", ctypes.c_uint32 * 3),
    ("Hs_do", ctypes.c_uint32), ("Hs_do_pad", ctypes.c_uint32 * 3),
    ("BAs_do", ctypes.c_uint32), ("BAs_do_pad", ctypes.c_uint32 * 3),
    ("Seqs_do", ctypes.c_uint32), ("Seqs_do_pad", ctypes.c_uint32 * 3),
    ("Hs_dk", ctypes.c_uint32), ("Hs_dk_pad", ctypes.c_uint32 * 3),
    ("BAs_dk", ctypes.c_uint32), ("BAs_dk_pad", ctypes.c_uint32 * 3),
    ("Hs_dv", ctypes.c_uint32), ("Hs_dv_pad", ctypes.c_uint32 * 3),
    ("BAs_dv", ctypes.c_uint32), ("BAs_dv_pad", ctypes.c_uint32 * 3),
    ("Seqs_dv", ctypes.c_uint32), ("Seqs_dv_pad", ctypes.c_uint32 * 3),
    ("Hs_lsed", ctypes.c_uint32), ("Hs_lsed_pad", ctypes.c_uint32 * 3),
    ("ptr_seqstart_q", ctypes.c_uint32), ("ptr_seqstart_q_pad", ctypes.c_uint32 * 3),
    ("ptr_seqstart_k", ctypes.c_uint32), ("ptr_seqstart_k_pad", ctypes.c_uint32 * 3),
    ("ptr_seqstart_q_padded", ctypes.c_uint32), ("ptr_seqstart_q_padded_pad", ctypes.c_uint32 * 3),
    ("ptr_seqstart_k_padded", ctypes.c_uint32), ("ptr_seqstart_k_padded_pad", ctypes.c_uint32 * 3),
    ("max_seq_len_dq", ctypes.c_uint32), ("max_seq_len_dq_pad", ctypes.c_uint32 * 3),
    ("mask_x", ctypes.c_uint32), ("mask_x_pad", ctypes.c_uint32 * 3),
    ("mask_y", ctypes.c_uint32), ("mask_y_pad", ctypes.c_uint32 * 3),
  ]
assert ctypes.sizeof(BwdMainArgs) == 704

class BwdDqConvertArgs(ctypes.Structure):
  _pack_ = 1
  _fields_ = [
    ("dQ_acc", ctypes.c_uint64), ("dQ_acc_pad", ctypes.c_uint64),
    ("dQ", ctypes.c_uint64), ("dQ_pad", ctypes.c_uint64),
    ("Hs_dQ_acc", ctypes.c_uint32), ("Hs_dQ_acc_pad", ctypes.c_uint32 * 3),
    ("BAs_dQ_acc", ctypes.c_uint32), ("BAs_dQ_acc_pad", ctypes.c_uint32 * 3),
    ("Seqs_dQ_acc", ctypes.c_uint32), ("Seqs_dQ_acc_pad", ctypes.c_uint32 * 3),
    ("Hs_dQ", ctypes.c_uint32), ("Hs_dQ_pad", ctypes.c_uint32 * 3),
    ("BAs_dQ", ctypes.c_uint32), ("BAs_dQ_pad", ctypes.c_uint32 * 3),
    ("Seqs_dQ", ctypes.c_uint32), ("Seqs_dQ_pad", ctypes.c_uint32 * 3),
    ("seqlen_q", ctypes.c_uint32), ("seqlen_q_pad", ctypes.c_uint32 * 3),
    ("head_dim", ctypes.c_uint32), ("head_dim_pad", ctypes.c_uint32 * 3),
    ("ptr_seqstart_q", ctypes.c_uint32), ("ptr_seqstart_q_pad", ctypes.c_uint32 * 3),
    ("ptr_seqstart_q_padded", ctypes.c_uint32), ("ptr_seqstart_q_padded_pad", ctypes.c_uint32 * 3),
    ("max_seqlen_dq", ctypes.c_uint32), ("max_seqlen_dq_pad", ctypes.c_uint32 * 3),
  ]
assert ctypes.sizeof(BwdDqConvertArgs) == 208

# ============================================================================
# FORWARD KERNEL
# ============================================================================

def build_fwd_kernargs(B, H, S, D, bufs, var_vals) -> bytes:
  """Build raw kernargs for the forward kernel from HCQ buffers."""
  args = FwdArgs()
  args.R, args.Q, args.K, args.V, args.LSE = bufs[0].va_addr, bufs[1].va_addr, bufs[2].va_addr, bufs[3].va_addr, bufs[4].va_addr
  args.ptr_qseq, args.ptr_kseq, args.ptr_qseq_padding, args.ptr_kseq_padding = 0, 0, 0, 0
  args.scalar, args.seq_len, args.kv_seq_len = float_to_u32(1.0 / math.sqrt(D)), S, S
  args.qk_head_dim, args.v_head_dim, args.q_head_num, args.gqa = D, D, H, 1
  args.msk_opt, args.lse, args.lse_Hs = 5, 1, S * 4  # causal mask
  elem_size = 2
  Seqs_stride, Hs_stride, Bs_stride = H * D * elem_size, D * elem_size, S * H * D * elem_size
  args.Seqs, args.Ts, args.Hs, args.Bs = Seqs_stride, S * D // 2, Hs_stride, Bs_stride
  args.k_Seqs, args.k_Hs, args.k_Bs = Seqs_stride, Hs_stride, Bs_stride
  args.v_Seqs, args.v_Hs, args.v_Bs = Seqs_stride, Hs_stride, Bs_stride
  args.r_Seqs, args.r_Hs, args.r_Bs = Seqs_stride, Hs_stride, Bs_stride
  return bytes(ctypes.string_at(ctypes.addressof(args), ctypes.sizeof(args)))

def aiter_fmha_fwd(out:UOp, q:UOp, k:UOp, v:UOp, lse:UOp, dname:str) -> UOp:
  """Create the PROGRAM UOp for aiter FMHA forward kernel."""
  B, S, H, D = out.shape
  binary = (CO_DIR / "fwd_hd128_bf16_causal.co").read_bytes()
  gidx0, gidx1, gidx2 = UOp.special(S // 512, "gidx0"), UOp.special(H, "gidx1"), UOp.special(B, "gidx2")
  lidx0 = UOp.special(512, "lidx0")
  kernargs_builder = functools.partial(build_fwd_kernargs, B, H, S, D)
  name = "aiter_fmha_fwd_hd128_bf16_causal"
  ops, mem = B * H * S * S * D * 2, (out.size + q.size + k.size + v.size + lse.size) * 2
  sink = UOp.sink(out.base, q.base, k.base, v.base, lse.base, gidx0, gidx1, gidx2, lidx0,
                  arg=KernelInfo(name=name, estimates=Estimates(ops=ops, mem=mem), kernargs_builder=kernargs_builder))
  src = f"; prebuilt aiter kernel: {name}"
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)))

# ============================================================================
# BACKWARD KERNELS
# ============================================================================

def build_odo_kernargs(B, H, S, D, bufs, var_vals) -> bytes:
  """Build kernargs for the ODO (rowsum(O * dO)) backward kernel."""
  args = BwdOdoArgs()
  elem_size = 2
  args.D, args.O, args.dO = bufs[0].va_addr, bufs[1].va_addr, bufs[2].va_addr
  args.Hs_odo, args.BAs_odo, args.Seqs_odo = D * elem_size, S * H * D * elem_size, H * D * elem_size
  args.Hs_d, args.BAs_d, args.Seqs_d = S * 4, H * S * 4, 4
  args.seqlen_q, args.head_dim = S, D
  args.ptr_seqstart_q, args.ptr_seqstart_q_padded = 0, 0
  return bytes(ctypes.string_at(ctypes.addressof(args), ctypes.sizeof(args)))

def aiter_fmha_bwd_odo(delta:UOp, out:UOp, dout:UOp, dname:str) -> UOp:
  """Create PROGRAM UOp for backward ODO kernel."""
  B, S, H, D = out.shape
  binary = (CO_DIR / "bwd_hd128_odo_bf16.co").read_bytes()
  gidx0, gidx1, gidx2 = UOp.special(S // 128, "gidx0"), UOp.special(H, "gidx1"), UOp.special(B, "gidx2")
  lidx0 = UOp.special(256, "lidx0")
  kernargs_builder = functools.partial(build_odo_kernargs, B, H, S, D)
  name = "aiter_fmha_bwd_hd128_odo_bf16"
  ops, mem = B * H * S * D * 2, (delta.size * 4 + out.size * 2 + dout.size * 2)
  sink = UOp.sink(delta.base, out.base, dout.base, gidx0, gidx1, gidx2, lidx0,
                  arg=KernelInfo(name=name, estimates=Estimates(ops=ops, mem=mem), kernargs_builder=kernargs_builder))
  src = f"; prebuilt aiter kernel: {name}"
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)))

def build_main_kernargs(B, H, S, D, bufs, var_vals) -> bytes:
  """Build kernargs for the main backward kernel (computes dQ_acc, dK, dV)."""
  args = BwdMainArgs()
  elem_size = 2
  args.dQ, args.dK, args.dV = bufs[0].va_addr, bufs[1].va_addr, bufs[2].va_addr
  args.Q, args.K, args.V = bufs[3].va_addr, bufs[4].va_addr, bufs[5].va_addr
  args.dO, args.Lse, args.D = bufs[6].va_addr, bufs[7].va_addr, bufs[8].va_addr
  args.scalar, args.log2e = float_to_u32(1.0 / math.sqrt(D)), float_to_u32(math.log2(math.e))
  args.seqlen_q, args.seqlen_k, args.head_dim_q, args.head_dim_k = S, S, D, D
  args.nhead_q, args.ratio, args.Ts = H, 1, S * D // 2
  Seqs_stride, Hs_stride, Bs_stride = H * D * elem_size, D * elem_size, S * H * D * elem_size
  args.Seqs_q, args.Hs_q, args.BAs_q = Seqs_stride, Hs_stride, Bs_stride
  args.Seqs_k, args.Hs_k, args.BAs_k = Seqs_stride, Hs_stride, Bs_stride
  args.Seqs_v, args.Hs_v, args.BAs_v = Seqs_stride, Hs_stride, Bs_stride
  args.Seqs_do, args.Hs_do, args.BAs_do = Seqs_stride, Hs_stride, Bs_stride
  args.Seqs_dk, args.Hs_dk, args.BAs_dk = Seqs_stride, Hs_stride, Bs_stride
  args.Seqs_dv, args.Hs_dv, args.BAs_dv = Seqs_stride, Hs_stride, Bs_stride
  args.Hs_lsed = S * 4
  args.ptr_seqstart_q, args.ptr_seqstart_k, args.ptr_seqstart_q_padded, args.ptr_seqstart_k_padded = 0, 0, 0, 0
  args.max_seq_len_dq, args.mask_x, args.mask_y = S, 0, 0
  return bytes(ctypes.string_at(ctypes.addressof(args), ctypes.sizeof(args)))

def aiter_fmha_bwd_main(dq_acc:UOp, dk:UOp, dv:UOp, q:UOp, k:UOp, v:UOp, dout:UOp, lse:UOp, delta:UOp, dname:str) -> UOp:
  """Create PROGRAM UOp for main backward kernel."""
  B, S, H, D = q.shape
  binary = (CO_DIR / "bwd_hd128_bf16_causal_br_a32_psskddv.co").read_bytes()
  gidx0, gidx1, gidx2 = UOp.special(S // 512, "gidx0"), UOp.special(H, "gidx1"), UOp.special(B, "gidx2")
  lidx0 = UOp.special(256, "lidx0")
  kernargs_builder = functools.partial(build_main_kernargs, B, H, S, D)
  name = "aiter_fmha_bwd_hd128_bf16_causal_br_a32_psskddv"
  ops = B * H * S * S * D * 6
  mem = sum(u.size * (4 if u.dtype == dtypes.float32 else 2) for u in [dq_acc, dk, dv, q, k, v, dout, lse, delta])
  sink = UOp.sink(dq_acc.base, dk.base, dv.base, q.base, k.base, v.base, dout.base, lse.base, delta.base,
                  gidx0, gidx1, gidx2, lidx0,
                  arg=KernelInfo(name=name, estimates=Estimates(ops=ops, mem=mem), kernargs_builder=kernargs_builder))
  src = f"; prebuilt aiter kernel: {name}"
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)))

def build_dq_convert_kernargs(B, H, S, D, bufs, var_vals) -> bytes:
  """Build kernargs for the dQ convert kernel (fp32 -> bf16)."""
  args = BwdDqConvertArgs()
  elem_size = 2
  args.dQ, args.dQ_acc = bufs[0].va_addr, bufs[1].va_addr
  args.Seqs_dQ_acc, args.Hs_dQ_acc, args.BAs_dQ_acc = D * 4, S * D * 4, H * S * D * 4
  args.Seqs_dQ, args.Hs_dQ, args.BAs_dQ = H * D * elem_size, D * elem_size, S * H * D * elem_size
  args.seqlen_q, args.head_dim = S, D
  args.ptr_seqstart_q, args.ptr_seqstart_q_padded, args.max_seqlen_dq = 0, 0, S
  return bytes(ctypes.string_at(ctypes.addressof(args), ctypes.sizeof(args)))

def aiter_fmha_bwd_dq_convert(dq:UOp, dq_acc:UOp, dname:str) -> UOp:
  """Create PROGRAM UOp for dQ convert kernel."""
  B, S, H, D = dq.shape
  binary = (CO_DIR / "bwd_hd128_dq_convert_bf16.co").read_bytes()
  gidx0, gidx1, gidx2 = UOp.special(S // 64, "gidx0"), UOp.special(H, "gidx1"), UOp.special(B, "gidx2")
  lidx0 = UOp.special(256, "lidx0")
  kernargs_builder = functools.partial(build_dq_convert_kernargs, B, H, S, D)
  name = "aiter_fmha_bwd_hd128_dq_convert_bf16"
  ops, mem = B * H * S * D, (dq.size * 2 + dq_acc.size * 4)
  sink = UOp.sink(dq.base, dq_acc.base, gidx0, gidx1, gidx2, lidx0,
                  arg=KernelInfo(name=name, estimates=Estimates(ops=ops, mem=mem), kernargs_builder=kernargs_builder))
  src = f"; prebuilt aiter kernel: {name}"
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)))

# ============================================================================
# BACKWARD GRADIENT FUNCTION
# ============================================================================

def _zero_kernel(out:UOp) -> UOp:
  """Kernel to zero out a buffer."""
  i = UOp.range(out.size, 0)
  return out.flatten()[i].store(0).end(i).sink(arg=KernelInfo(name="zero"))

def _sharded_empty_like(ref: Tensor, axis: int|None=None) -> Tensor:
  return _sharded_empty(ref.shape, ref, axis)

# ============================================================================
# MAIN ASM SDPA FUNCTION
# ============================================================================

counters = {"used":0, "todos":[]}
def todo(msg:str) -> bool: counters["todos"].append(msg); return False
atexit.register(lambda: print(f'asm_atn: {counters["used"]} used, {len(counters["todos"])} not used'))

def can_use_asm_atn(q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> bool:
  if q.dtype not in {dtypes.bfloat16}: return todo("dtype!=bf16")
  if not is_causal: return todo("not_causal")
  if q.ndim != 4: return todo("ndim!=4")
  B, H, S, D = q.shape
  if D != 128: return todo("D!=128")
  if S % 512 != 0: return todo("S%512")
  # multi-device: must all be sharded on batch axis (axis=0 in BHSD layout)
  if isinstance(q.device, tuple):
    if not (q.uop.axis == 0 and k.uop.axis == 0 and v.uop.axis == 0): return todo("shard_axis!=0")
  dname = q.device[0] if isinstance(q.device, tuple) else q.device
  arch = getattr(Device[dname].renderer, "arch", "")
  if not arch.startswith("gfx950"): return todo("arch!=gfx950")
  return True

def _sharded_empty(shape, ref: Tensor, axis: int) -> Tensor:
  """Create an empty tensor with proper sharding for multi-device."""
  if not isinstance(ref.device, tuple): return Tensor.empty(*shape, dtype=ref.dtype, device=ref.device)
  local_shape = tuple(s // len(ref.device) if i == axis else s for i, s in enumerate(shape))
  return Tensor(Tensor.empty(*local_shape, dtype=ref.dtype, device=ref.device).uop.multi(axis), dtype=ref.dtype, device=ref.device)

def _sharded_empty_f32(shape, ref: Tensor, axis: int) -> Tensor:
  """Create an empty f32 tensor with proper sharding for multi-device."""
  if not isinstance(ref.device, tuple): return Tensor.empty(*shape, dtype=dtypes.float32, device=ref.device)
  local_shape = tuple(s // len(ref.device) if i == axis else s for i, s in enumerate(shape))
  return Tensor(Tensor.empty(*local_shape, dtype=dtypes.float32, device=ref.device).uop.multi(axis), dtype=dtypes.float32, device=ref.device)

def asm_sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
  """
  Scaled dot-product attention using optimized aiter ASM kernels.

  Args:
    q, k, v: Input tensors in BHSD layout [B, H, S, D]

  Returns:
    Output tensor in BHSD layout [B, H, S, D]
  """
  counters["used"] += 1
  B, H_q, S, D = q.shape
  print("[asm_atn]", q.shape, k.shape, v.shape)
  H_kv = k.shape[1]  # K/V may have fewer heads (GQA)
  dname = q.device[0] if isinstance(q.device, tuple) else q.device

  # Permute to BSHD layout for internal computation
  q_perm, k_perm, v_perm = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
  out = _sharded_empty((B, S, H_q, D), q, axis=0)
  lse = _sharded_empty_f32((B, H_q, S), q, axis=0)

  def grad_fxn(gradient: UOp, _) -> tuple[None, UOp, UOp, UOp, None]:
    """Backward pass - uses closure to access original tensors."""
    dout = Tensor(gradient, device=gradient.device)

    # Create intermediate tensors using q as reference for sharding (q has proper MULTI structure)
    delta = _sharded_empty_f32((B, H_q, S), q, axis=0)
    delta, *_ = Tensor.custom_kernel(delta, out, dout, fxn=functools.partial(aiter_fmha_bwd_odo, dname=dname))

    # dq_acc has extra leading dim of 1, B axis is at index 1
    dq_acc = _sharded_empty_f32((1, B, H_q, S, D), q, axis=1)
    dq_acc = Tensor.custom_kernel(dq_acc, fxn=_zero_kernel)[0]

    # dk/dv use K/V head count (H_kv), not Q head count (H_q)
    dk = _sharded_empty((B, S, H_kv, D), k, axis=0)
    dv = _sharded_empty((B, S, H_kv, D), v, axis=0)
    dq_acc, dk, dv, *_ = Tensor.custom_kernel(dq_acc, dk, dv, q_perm, k_perm, v_perm, dout, lse, delta,
                                               fxn=functools.partial(aiter_fmha_bwd_main, dname=dname))

    dq = _sharded_empty((B, S, H_q, D), q, axis=0)
    dq, *_ = Tensor.custom_kernel(dq, dq_acc, fxn=functools.partial(aiter_fmha_bwd_dq_convert, dname=dname))
    return (None, dq.uop, dk.uop, dv.uop, None)

  out, *_ = Tensor.custom_kernel(out, q_perm, k_perm, v_perm, lse,
                                  fxn=functools.partial(aiter_fmha_fwd, dname=dname),
                                  grad_fxn=grad_fxn)
  return out.permute(0, 2, 1, 3)
