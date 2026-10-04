import math, pathlib, functools, struct

from tinygrad import Device, Tensor
from tinygrad.dtype import DTypeLike, dtypes
from tinygrad.helpers import DEBUG, TRAINING, getenv
from tinygrad.renderer import Estimates
from extra.hipcc import HIPCCCompiler
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.uop.ops import UOp, Ops, KernelInfo

GPTOSS_QKV_ROPE_PACKED = getenv("GPTOSS_QKV_ROPE_PACKED", 0)
GPTOSS_QKV_ROPE_BWD_QGROUP = getenv("GPTOSS_QKV_ROPE_BWD_QGROUP", 0)

def _sharded_empty(shape:Tensor, ref:Tensor, axis:int|None, dtype:DTypeLike|None=None) -> Tensor:
  dtype = dtype or ref.dtype
  if not isinstance(ref.device, tuple): return Tensor.invalids(*shape, dtype=dtype, device=ref.device)
  shard_axis = ref.uop.axis if axis is None else axis
  shape = tuple(s // len(ref.device) if i == shard_axis else s for i, s in enumerate(shape))
  axis = ref.uop.axis if axis is None else axis
  return Tensor(Tensor.invalids(*shape, dtype=dtype, device=ref.device).uop.unshard(axis), dtype=dtype, device=ref.device)

@functools.cache
def _custom_fused_qkv_rope_reference(q:UOp, k:UOp, v:UOp, xqkv:UOp, freqs_cis:UOp,
                                  device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int):
  group_size = H // H_KV
  q, k, v = q.reshape(B, N, H, D), k.reshape(B, N, H_KV, D), v.reshape(B, N, H_KV, D)
  xqkv = xqkv.reshape(B, N, H_KV, group_size + 2, D)
  b, n = UOp.range(B, 0), UOp.range(N, 1)
  pair = UOp.range(D // 2, 2)
  even = pair * 2
  c = freqs_cis[0, n, 0, pair, 0].cast(dtypes.float)
  s = freqs_cis[0, n, 0, pair, 1].cast(dtypes.float)
  ordered:UOp|None = None
  for kvh in range(H_KV):
    q_out, k_out, v_out = (x.after(ordered) if ordered is not None else x for x in (q, k, v))
    x_in = xqkv.after(ordered) if ordered is not None else xqkv
    stores:list[UOp] = []
    for rep in range(group_size):
      a = x_in[b, n, kvh, rep, even].cast(dtypes.float)
      bb = x_in[b, n, kvh, rep, even + 1].cast(dtypes.float)
      h = kvh * group_size + rep
      stores += [q_out[b, n, h, even].store((a * c - bb * s).cast(q.dtype)), q_out[b, n, h, even + 1].store((a * s + bb * c).cast(q.dtype))]
    a = x_in[b, n, kvh, group_size, even].cast(dtypes.float)
    bb = x_in[b, n, kvh, group_size, even + 1].cast(dtypes.float)
    stores += [k_out[b, n, kvh, even].store((a * c - bb * s).cast(k.dtype)),
               k_out[b, n, kvh, even + 1].store((a * s + bb * c).cast(k.dtype)),
               v_out[b, n, kvh, even].store(x_in[b, n, kvh, group_size + 1, even]),
               v_out[b, n, kvh, even + 1].store(x_in[b, n, kvh, group_size + 1, even + 1])]
    ordered = UOp.group(*stores)
  assert ordered is not None
  return ordered.end(pair, n, b).sink(arg=KernelInfo(name="fused_qkv_rope_forward"))

@functools.cache
def custom_fused_qkv_rope_forward(q:UOp, k:UOp, v:UOp, xqkv:UOp, freqs_cis:UOp, *bias:UOp,
                                  device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int, direct_save:bool=False):
  if bias:
    assert len(bias) == 1 and bias[0].shape == (5120,) and bias[0].dtype == dtypes.bfloat16
    assert GPTOSS_QKV_ROPE_PACKED and (B, N, H, H_KV, D) == (2, 8192, 64, 8, 64)
  if not (GPTOSS_QKV_ROPE_PACKED and (N, H, H_KV, D) == (8192, 64, 8, 64)):
    return _custom_fused_qkv_rope_reference(q,k,v,xqkv,freqs_cis,device,arch,B,N,H,H_KV,D)
  gptoss_packed = GPTOSS_QKV_ROPE_PACKED and (B, N, H, H_KV, D) == (2, 8192, 64, 8, 64)
  accesses = ()
  if direct_save:
    assert gptoss_packed and arch == "gfx950" and all(x.dtype == dtypes.bfloat16 for x in (q, k, v, xqkv, freqs_cis))
    zero = UOp.const(0, dtypes.int32)
    accesses = tuple(x.index(zero).store(UOp.const(0, dtypes.bfloat16)) for x in (q, k, v)) + \
               (xqkv.index(zero).load(), freqs_cis.index(zero).load(), *(b.index(zero).load() for b in bias))
  code = (pathlib.Path(__file__).parent / ("fused_qkv_rope_gptoss.cpp" if gptoss_packed else "fused_qkv_rope.cpp")).read_text()
  threads = 256
  thread_idx = UOp.special(threads, "lidx0")
  block_idx_x, block_idx_y = UOp.special(B, "gidx0"), UOp.special(N, "gidx1")
  sink = UOp.sink(q.base, k.base, v.base, xqkv.base, freqs_cis.base, *(b.base for b in bias),
                  thread_idx, block_idx_x, block_idx_y, *accesses,
                  arg=KernelInfo(name="fused_qkv_rope_forward_gptoss_bias" if bias else
                                     "fused_qkv_rope_forward_gptoss_packed" if gptoss_packed else "fused_qkv_rope_forward"))
  compile_args = ["-std=c++20", "-ffast-math", f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}",
                  f"-DATTN_H_KV={H_KV}", f"-DATTN_D={D}", f"-DTHREADS_PER_BLOCK={threads}"]
  if bias: compile_args.append("-DGPTOSS_QKV_BIAS=1")
  lib = HIPCCCompiler(arch, compile_args).compile_cached(code)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def custom_fused_qkv_rope_backward(dxqkv:UOp, dq:UOp, dk:UOp, dv:UOp, freqs_cis:UOp,
                                   device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int, heads_per_wg:int):
  # kernel is parametric on ATTN_H/H_KV/D (compile flags below); constraints are the tiling ones, not llama's exact shape
  assert D in (64, 128) and H % H_KV == 0 and N % 64 == 0, f"fused_qkv_rope_backward unsupported shape {(B,N,H,H_KV,D)=}"
  code = (pathlib.Path(__file__).parent / "fused_qkv_rope_bwd.cpp").read_text()
  threads = 256
  gptoss_qgroup = GPTOSS_QKV_ROPE_BWD_QGROUP and (B, N, H, H_KV, D) == (2, 8192, 64, 8, 64) and heads_per_wg in (2, 8)
  q_heads_per_wg = (8 if heads_per_wg == 2 else 2) if gptoss_qgroup else 1
  thread_idx = UOp.special(threads, "lidx0")
  gsz = (B, N // 64, H // q_heads_per_wg + 2 * H_KV)
  block_idx_x, block_idx_y, block_idx_z = (UOp.special(x, f"gidx{i}") for i, x in enumerate(gsz))
  sink = UOp.sink(dxqkv.base, dq.base, dk.base, dv.base, freqs_cis.base, thread_idx, block_idx_x, block_idx_y, block_idx_z,
                  arg=KernelInfo(name="fused_qkv_rope_backward_gptoss_qgroup" if gptoss_qgroup else "fused_qkv_rope_backward"))
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math", f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}",
                  f"-DATTN_H_KV={H_KV}", f"-DATTN_D={D}", f"-DBWD_HEADS_PER_WG={heads_per_wg}",
                  f"-DGPTOSS_BWD_DQ_CACHE={2 if gptoss_qgroup else 0}", f"-DGPTOSS_BWD_Q_HEADS_PER_WG={q_heads_per_wg}",
                  f"-DTHREADS_PER_BLOCK={threads}"]
  lib = HIPCCCompiler(arch, compile_args).compile_cached(code)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

def _fa_native_grads(dq:UOp, dk:UOp, dv:UOp) -> tuple[UOp, UOp, UOp, int]|None:
  def unwrap_partial(x:UOp) -> UOp|None:
    # Multiple partials retain the reduction/permute chain. GPT-OSS H8 has one native partial, so simplification
    # removes that no-op reduction and leaves only its cast/reshape chain.
    for expected in ((Ops.CAST, Ops.REDUCE, Ops.PERMUTE, Ops.CAST, Ops.RESHAPE, Ops.AFTER),
                     (Ops.CAST, Ops.RESHAPE, Ops.CAST, Ops.RESHAPE, Ops.AFTER)):
      cur = x
      for op in expected:
        if cur.op is not op: break
        if op is not Ops.AFTER: cur = cur.src[0]
      else: return cur
    return None
  dq_native, dk_partial, dv_partial = dq.base, unwrap_partial(dk), unwrap_partial(dv)
  if dq_native.op is not Ops.AFTER or dk_partial is None or dv_partial is None: return None
  B, N, H, D, H_KV = dq.shape[0], dq.shape[1], dq.shape[2], dq.shape[3], dk.shape[2]
  if dk_partial.shape[0] % B: return None
  partials = dk_partial.shape[0] // B
  if partials == 0 or (H // H_KV) % partials: return None
  heads_per_wg = (H // H_KV) // partials
  if dq_native.shape != (B, H, N, D) or dk_partial.shape != (B * partials, N, H_KV, D) or dv_partial.shape != dk_partial.shape: return None
  return dq_native, dk_partial, dv_partial, heads_per_wg

def _fused_qkv_rope_grad(dq_u:UOp, dk_u:UOp, dv_u:UOp, call:UOp) -> tuple[None, None, None, UOp, None]:
  dq, dk, dv = Tensor(dq_u, device=dq_u.device), Tensor(dk_u, device=dk_u.device), Tensor(dv_u, device=dv_u.device)
  xqkv_u, freqs_u = call.src[4], call.src[5]
  xqkv, freqs_cis = Tensor(xqkv_u, device=xqkv_u.device), Tensor(freqs_u, device=freqs_u.device)
  B, N, _ = xqkv.shape
  H, H_KV, D = dq.shape[2], dk.shape[2], dq.shape[3]
  num_devices = len(xqkv.device) if isinstance(xqkv.device, tuple) else 1
  is_dp, is_mp = xqkv.uop.axis == 0, xqkv.uop.axis == 2
  B_local = B // num_devices if is_dp else B
  H_local = H // num_devices if is_mp else H
  H_KV_local = H_KV // num_devices if is_mp else H_KV
  single_device = xqkv.device[0] if isinstance(xqkv.device, tuple) else xqkv.device
  arch = Device[single_device].renderer.target.arch
  fa_native = _fa_native_grads(dq_u, dk_u, dv_u)
  assert fa_native is not None, "fused QKV RoPE backward requires native Flash Attention gradients"
  dq, dk, dv = (Tensor(x, device=x.device) for x in fa_native[:3])
  heads_per_wg = fa_native[3]
  dxqkv = _sharded_empty_like(xqkv, axis=xqkv.uop.axis if isinstance(xqkv.device, tuple) else None)
  fxn = functools.partial(custom_fused_qkv_rope_backward, device=single_device, arch=arch,
                          B=B_local, N=N, H=H_local, H_KV=H_KV_local, D=D, heads_per_wg=heads_per_wg)
  dxqkv = Tensor.custom_kernel(dxqkv, dq, dk, dv, freqs_cis, fxn=fxn)[0]
  return None, None, None, dxqkv.uop, None

def _fused_qkv_rope_bias_grad(dq_u:UOp, dk_u:UOp, dv_u:UOp, call:UOp) -> tuple:
  from extra.llama_kernels.dense_bias import dense_bias_backward
  dx = _fused_qkv_rope_grad(dq_u, dk_u, dv_u, call)[3]
  dx, db = dense_bias_backward(dx, call.src[4], call.src[6])
  return None, None, None, dx, None, db

def gptoss_qkv_rope_saved_outputs(ref:Tensor, n_heads:int, n_kv_heads:int, head_dim:int, *, sliding:bool, save:bool
                                  ) -> tuple[Tensor, Tensor, Tensor]|None:
  # Allocate inside run_layer; contiguous destinations bind directly to the saved forward output slots.
  if not (getenv("GPTOSS_QKV_ROPE_DIRECT_SAVE", 0) and TRAINING and save and
          getenv("FUSED_QKV_ROPE", 0) and GPTOSS_QKV_ROPE_PACKED and getenv("HK_FLASH_ATTENTION", 0) and
          (not sliding or getenv("HK_FLASH_SLIDING", 1))): return None
  axis = ref.uop.axis if isinstance(ref.device, tuple) else None
  if isinstance(ref.device, tuple) and axis != 0: return None
  if ref.dtype != dtypes.bfloat16 or ref.uop.shard_shape != (2, 8192, 2880) or (n_heads, n_kv_heads, head_dim) != (64, 8, 64): return None
  single_device = ref.device[0] if isinstance(ref.device, tuple) else ref.device
  if Device[single_device].renderer.target.arch != "gfx950": return None
  B, N, _ = ref.shape
  q, k, v = (_sharded_empty((B, N, h, head_dim), ref, axis=axis, dtype=dtypes.bfloat16) for h in (n_heads, n_kv_heads, n_kv_heads))
  return q.clone(), k.clone(), v.clone()

def fused_qkv_rope(xqkv:Tensor, freqs_cis:Tensor, n_heads:int, n_kv_heads:int, head_dim:int, *,
                   out:tuple[Tensor, Tensor, Tensor]|None=None, bias:Tensor|None=None) -> tuple[Tensor, Tensor, Tensor]:
  B, N, packed_dim = xqkv.shape
  assert packed_dim == n_kv_heads * (n_heads // n_kv_heads + 2) * head_dim
  assert freqs_cis.dtype == dtypes.bfloat16, f"fused QKV RoPE requires bfloat16 frequencies, got {freqs_cis.dtype}"
  assert freqs_cis.shape == (1, freqs_cis.shape[1], 1, head_dim // 2, 2) and freqs_cis.shape[1] >= N, \
    f"invalid RoPE frequency shape {freqs_cis.shape} for sequence length {N} and head dimension {head_dim}"
  num_devices = len(xqkv.device) if isinstance(xqkv.device, tuple) else 1
  is_dp, is_mp = xqkv.uop.axis == 0, xqkv.uop.axis == 2
  B_local = B // num_devices if is_dp else B
  H_local = n_heads // num_devices if is_mp else n_heads
  H_KV_local = n_kv_heads // num_devices if is_mp else n_kv_heads
  assert H_local % H_KV_local == 0 and head_dim % 2 == 0 and head_dim <= 512
  single_device = xqkv.device[0] if isinstance(xqkv.device, tuple) else xqkv.device
  arch = Device[single_device].renderer.target.arch
  fuse_bias = bias is not None and getenv("GPTOSS_QKV_ROPE_BIAS", 0) and GPTOSS_QKV_ROPE_PACKED and arch == "gfx950" and \
              (B_local, N, H_local, H_KV_local, head_dim) == (2, 8192, 64, 8, 64)
  if bias is not None:
    assert bias.device == xqkv.device and bias.shape == (packed_dim,) and bias.dtype == xqkv.dtype == dtypes.bfloat16
    if not fuse_bias:
      from extra.llama_kernels.dense_bias import dense_bias_add
      xqkv = dense_bias_add(xqkv, bias)
  axis = 0 if is_dp else 2 if is_mp else None
  if out is None:
    q = _sharded_empty((B, N, n_heads, head_dim), xqkv, axis=axis, dtype=dtypes.bfloat16)
    k = _sharded_empty((B, N, n_kv_heads, head_dim), xqkv, axis=axis, dtype=dtypes.bfloat16)
    v = _sharded_empty((B, N, n_kv_heads, head_dim), xqkv, axis=axis, dtype=dtypes.bfloat16)
  else:
    assert GPTOSS_QKV_ROPE_PACKED and arch == "gfx950" and (B_local, N, H_local, H_KV_local, head_dim) == (2, 8192, 64, 8, 64)
    assert xqkv.dtype == dtypes.bfloat16 and (not isinstance(xqkv.device, tuple) or is_dp) and len(out) == 3
    for x, h in zip(out, (n_heads, n_kv_heads, n_kv_heads)):
      assert x.device == xqkv.device and x.dtype == dtypes.bfloat16 and x.uop.axis == xqkv.uop.axis
      assert x.shape == (B, N, h, head_dim) and x.uop.shard_shape == (B_local, N, h, head_dim)
    assert len({x.uop.buf_uop for x in (*out, xqkv, freqs_cis)}) == 5, "QKV saved outputs must own distinct storage"
    q, k, v = out
  fxn = functools.partial(custom_fused_qkv_rope_forward, device=single_device, arch=arch,
                          B=B_local, N=N, H=H_local, H_KV=H_KV_local, D=head_dim, direct_save=out is not None)
  q, k, v, *_ = Tensor.custom_kernel(q, k, v, xqkv, freqs_cis, *((bias,) if fuse_bias else ()), fxn=fxn,
                                    grad_fxn=_fused_qkv_rope_bias_grad if fuse_bias else _fused_qkv_rope_grad)
  return q, k, v

def _sharded_empty_like(ref:Tensor, axis:int|None=None) -> Tensor:
  return _sharded_empty(ref.shape, ref, axis)

@functools.cache
def _windowed_lse(xq:Tensor, xk:Tensor, sinks, W:int) -> Tensor:
  # Accurate fp32 windowed log-sum-exp (natural log), banded 2-block (mirrors _sliding_attention's softmax denom).
  # The flash forward's SAVED l_vec is biased ~0.1-0.8% at the KV-block straddle (bf16 online-softmax rescaling);
  # the backward's P=exp2(S-L) then carries that bias into dq/dk/dv, which diverges the sensitive sliding layers
  # (root-caused: feeding this exact L into a perfect backward reproduces the flash kernel's grad biases). Recomputing
  # L exactly here (fp32, no value matmul) keeps the fast bf16 backward while removing the bias. Only for window>0.
  B, N, H, hd = xq.shape
  H_KV = xk.shape[2]; R = H // H_KV; nb = N // W; sm = hd ** -0.5
  q = xq.reshape(B, N, H_KV, R, hd).permute(0, 2, 3, 1, 4).reshape(B, H_KV, R, nb, W, hd).float()
  k = xk.permute(0, 2, 1, 3).reshape(B, H_KV, 1, nb, W, hd).float()
  k_prev = k.pad((None, None, None, (1, 0), None, None))[:, :, :, :nb]
  sc_d = (q @ k.transpose(-1, -2)) * sm
  sc_p = (q @ k_prev.transpose(-1, -2)) * sm
  li, lj = Tensor.arange(W).reshape(W, 1), Tensor.arange(W).reshape(1, W)
  pv = (Tensor.arange(nb).reshape(nb, 1, 1) >= 1)
  sc_d = (lj <= li).where(sc_d, -float("inf"))
  sc_p = ((li < lj) & pv).where(sc_p, -float("inf"))
  m = sc_d.max(-1, keepdim=True).maximum(sc_p.max(-1, keepdim=True))
  if sinks is not None: m = m.maximum(sinks.reshape(1, H_KV, R, 1, 1, 1).float())
  denom = (sc_d - m).exp().sum(-1, keepdim=True) + (sc_p - m).exp().sum(-1, keepdim=True)
  if sinks is not None: denom = denom + (sinks.reshape(1, H_KV, R, 1, 1, 1).float() - m).exp()
  return (m + denom.log()).reshape(B, H, N).unsqueeze(2)  # (B, H, 1, N), matches saved l_vec

def _windowed_delta(xq:Tensor, xk:Tensor, xv:Tensor, do:Tensor, sinks, W:int) -> Tensor:
  # Accurate fp32 delta = rowsum_d(dO * O), banded 2-block. The flash's delta comes from the bf16 forward O while its
  # backward computes dP from V; for PEAKED attention delta~=dP_peak so (dP-delta) is a bf16-cancellation and dq/dk
  # blow up (up to 40% at max-attn-weight 0.9). Recomputing delta from the SAME fp32 softmax as O makes it consistent.
  B, N, H, hd = xq.shape
  H_KV = xk.shape[2]; R = H // H_KV; nb = N // W; sm = hd ** -0.5
  q = xq.reshape(B, N, H_KV, R, hd).permute(0, 2, 3, 1, 4).reshape(B, H_KV, R, nb, W, hd).float()
  k = xk.permute(0, 2, 1, 3).reshape(B, H_KV, 1, nb, W, hd).float()
  v = xv.permute(0, 2, 1, 3).reshape(B, H_KV, 1, nb, W, hd).float()
  dob = do.reshape(B, N, H_KV, R, hd).permute(0, 2, 3, 1, 4).reshape(B, H_KV, R, nb, W, hd).float()
  k_prev = k.pad((None, None, None, (1, 0), None, None))[:, :, :, :nb]
  v_prev = v.pad((None, None, None, (1, 0), None, None))[:, :, :, :nb]
  sc_d = (q @ k.transpose(-1, -2)) * sm
  sc_p = (q @ k_prev.transpose(-1, -2)) * sm
  li, lj = Tensor.arange(W).reshape(W, 1), Tensor.arange(W).reshape(1, W)
  pv = (Tensor.arange(nb).reshape(nb, 1, 1) >= 1)
  sc_d = (lj <= li).where(sc_d, -float("inf"))
  sc_p = ((li < lj) & pv).where(sc_p, -float("inf"))
  m = sc_d.max(-1, keepdim=True).maximum(sc_p.max(-1, keepdim=True))
  if sinks is not None: m = m.maximum(sinks.reshape(1, H_KV, R, 1, 1, 1).float())
  e_d, e_p = (sc_d - m).exp(), (sc_p - m).exp()
  denom = e_d.sum(-1, keepdim=True) + e_p.sum(-1, keepdim=True)
  if sinks is not None: denom = denom + (sinks.reshape(1, H_KV, R, 1, 1, 1).float() - m).exp()
  o = ((e_d / denom) @ v) + ((e_p / denom) @ v_prev)   # (B,H_KV,R,nb,W,hd) accurate output
  delta = (dob * o).sum(-1)                             # (B,H_KV,R,nb,W)  = rowsum_d(dO*O)
  return delta.reshape(B, H, N).unsqueeze(2)            # (B, H, 1, N), matches delta_vec

def _bwd_heads_per_wg(D:int, group_size:int, window:int) -> int:
  if D == 64:
    assert group_size == 8, "D=64 backward is specialized for GPT-OSS GQA8"
    if not window and getenv("FA_D64_GENERIC", 0): return 1
    # Full causal needs more workgroups to cover its triangular schedule; two heads still amortize K/V
    # and partial-gradient traffic without the load-balance loss measured with four or eight heads.
    return 8 if window else 2
  return 2 if D == 128 and group_size % 2 == 0 else 1

def _fa_grad_fxn(B, H, N, D, H_local, H_KV_local, H_KV, B_local, shard_axis, shard_axis_t, single_device, arch, has_sink, window=0):
  def grad(dou:UOp, ker:UOp) -> tuple:
    do = Tensor(dou, device=dou.device)
    attn = Tensor(ker.src[1].after(ker), device=ker.src[1].device)
    l_vec = Tensor(ker.src[2].after(ker), device=ker.src[2].device)
    xq = Tensor(ker.src[3], device=ker.src[3].device)
    xk = Tensor(ker.src[4], device=ker.src[4].device)
    xv = Tensor(ker.src[5], device=ker.src[5].device)
    # windowed forward saves a biased L -> substitute an exact fp32 windowed LSE so the fast bf16 backward gets a clean L
    if window and getenv("FA_WLSE", 0):  # DROPPED (default off): the fwd now scales scores AFTER the MMA in fp32 (WINDOW build,
      # fa_fwd_causal.cpp) instead of bf16(0.18*Q) — that bf16 rounding was the LSE bias that diverged peaked SWA (~step 210,
      # amplified 1e6x by the (dP-D) cancellation). With it fixed, the kernel's online L is unbiased. Verified: real training
      # to step 305+ stable (max grad_norm 4.97) with BOTH recomputes off. FA_WLSE=1 restores the old fp32 LSE recompute.
      l_vec = _windowed_lse(xq, xk, Tensor(ker.src[6], device=ker.src[6].device) if has_sink else None, window)

    dq = _sharded_empty((B, H, N, D), xq, axis=shard_axis_t)
    GROUP_SIZE = H_local // H_KV_local
    # Fuse all eight GPT-OSS GQA heads for the short sliding-window schedule. This amortizes K/V traffic and
    # removes the partial-gradient reductions; the longer full-causal schedule still loses load balance with H8.
    HEADS_PER_WG = _bwd_heads_per_wg(D, GROUP_SIZE, window)
    dk_partial = _sharded_empty((B * GROUP_SIZE // HEADS_PER_WG, N, H_KV, D), xk, axis=shard_axis)
    dv_partial = _sharded_empty((B * GROUP_SIZE // HEADS_PER_WG, N, H_KV, D), xv, axis=shard_axis)

    # delta_vec = (do * attn).sum(-1, dtype=dtypes.float32).transpose(1, 2).unsqueeze(-2).detach()
    delta_vec = _sharded_empty((B, H, 1, N), xq, dtype=dtypes.float32, axis=shard_axis_t)
    delta_vec, dq = Tensor.custom_kernel(delta_vec, dq, attn, do, fxn=functools.partial(custom_fa_backward_pre, device=single_device, arch=arch, B=B_local, N=N, H=H_local, H_KV=H_KV_local, D=D))[:2]
    # windowed: delta from bf16 forward-O vs backward dP-from-V don't cancel for PEAKED attn -> substitute a consistent
    # fp32 delta (rowsum dO*O from the same softmax). Fixes dq/dk blow-up on the sliding layers' peaked attention.
    if window and getenv("FA_WDELTA", 0):  # DROPPED (default off): real training to step 363 is stable+healthy on the stored bf16-O
      # delta (fa_backward_pre) — it's consistent with the bf16-QK backward P (FlashAttention's principle), so the fp32 O recompute
      # is unneeded (it was actually the inconsistent one). FA_WDELTA=1 restores the old recompute for debugging.
      delta_vec = _windowed_delta(xq, xk, xv, do, Tensor(ker.src[6], device=ker.src[6].device) if has_sink else None, window)

    dq, dk_partial, dv_partial = Tensor.custom_kernel(dq, dk_partial, dv_partial, do, xq, xk, xv, l_vec, delta_vec, fxn=functools.partial(custom_fa_backward, device=single_device, arch=arch, B=B_local, N=N, H=H_local, H_KV=H_KV_local, D=D, window=window))[:3]

    if D == 64:
      dq = dq.reshape(B, H, N//16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2).permute(0, 1, 2, 8, 9, 10, 11, 3, 4, 6, 7, 5, 12).reshape(B, H, N, D).transpose(1, 2)
    else:
      dq = dq.reshape(B, H, N//16, 4, 2, 2, D//32, 4, 4, 2).permute(0, 1, 2, 7, 8, 3, 4, 6, 5, 9).reshape(B, H, N, D).transpose(1, 2)

    # reduce partial dK/dV across GROUP_SIZE query heads
    dk = dk_partial.reshape(B, GROUP_SIZE // HEADS_PER_WG, N, H_KV, D).sum(1)
    dv = dv_partial.reshape(B, GROUP_SIZE // HEADS_PER_WG, N, H_KV, D).sum(1)

    if not has_sink: return None, None, dq.uop, dk.uop, dv.uop
    sinks = Tensor(ker.src[6], device=ker.src[6].device)
    p_sink = (sinks.reshape(1, H, 1, 1) - l_vec).exp()
    dsink = -(delta_vec.float() * p_sink).sum(axis=(0, 2, 3))

    return None, None, dq.uop, dk.uop, dv.uop, dsink.uop
  return grad

_cached_fa_grad_fxn = functools.cache(_fa_grad_fxn)

# TODO: remove write_flat once scheduler can remove reshapes between custom_kernel. TestCustomKernel.test_simple_reshape
def flash_attention(xq, xk, xv, attn_mask:Tensor|None=None, is_causal:bool=False, write_flat:bool=False, sinks:Tensor|None=None, window:int=0):
  assert attn_mask is None, "attn_mask not supported"
  assert is_causal, "only causal attention supported"

  B, N, H, D = xq.shape
  H_KV = xk.shape[2]
  assert D in (64, 128), "only D=64 or D=128 supported"
  has_sink = sinks is not None
  if has_sink: sinks = sinks.float()

  num_devices = len(xq.device) if isinstance(xq.device, tuple) else 1
  is_dp = xq.uop.axis == 0
  is_mp = xq.uop.axis == 2
  B_local = B // num_devices if is_dp else B
  H_local = H // num_devices if is_mp else H
  H_KV_local = H_KV // num_devices if is_mp else H_KV
  shard_axis = 0 if is_dp else 2 if is_mp else None
  shard_axis_t = 0 if is_dp else 1 if is_mp else None
  if DEBUG >= 2: print(f"Flash Attention {B=} {B_local=} {N=} {H=} {H_local=} {H_KV=} {H_KV_local=} {D=} on {num_devices} devices, {'DP' if is_dp else 'MP' if is_mp else 'no sharding'}")

  single_device = xq.device[0] if isinstance(xq.device, tuple) else xq.device
  arch = Device[single_device].renderer.target.arch

  attn = _sharded_empty_like(xq, axis=shard_axis)
  attn = _sharded_empty((B, N, H * D), xq, axis=shard_axis) if write_flat else _sharded_empty_like(xq, axis=shard_axis)
  l_vec = _sharded_empty((B, H, 1, N), xq, dtype=dtypes.float32, axis=shard_axis_t)

  grad = _cached_fa_grad_fxn(B, H, N, D, H_local, H_KV_local, H_KV, B_local, shard_axis, shard_axis_t,
                            single_device, arch, has_sink, window=window)

  attn, l_vec = (v.clone() for v in (attn, l_vec))
  fwd_inputs = (attn, l_vec, xq, xk, xv) + ((sinks,) if has_sink else ())
  attn, l_vec = Tensor.custom_kernel(*fwd_inputs, fxn=functools.partial(custom_fa_forward, device=single_device, arch=arch, B=B_local, N=N, H=H_local, H_KV=H_KV_local, D=D, has_sink=has_sink, window=window), grad_fxn=grad)[:2]

  return attn, attn, l_vec

def _fa_forward_pairs(n:int, window:int, warps:int, full_w4:bool=False) -> int:
  """Q/K pairs in executed MMA tiles, including masked lanes (not useful model attention pairs)."""
  pairs = 0
  for q_start in range(0, n, 32*warps):
    end = min((q_start+32*warps+63)//64, n//64)
    if full_w4: end = (end+3)//4*4  # full D64 keeps the four-KV-tile epilogue boundaries
    elif warps == 4: end = max(end, 4)  # window D64 pipeline needs at least four KV tiles
    start = max(0, min((q_start-window+1)//64, end-4)) if window else 0
    pairs += 32*warps*64*(end-start)
  return pairs

def _fa_backward_pairs(n:int, window:int, specialized:bool) -> int:
  # Each 256-key workgroup walks 64-query steps. Generic backward additionally flushes two masked window steps.
  return sum(256*64*(min((n-k)//64, (256+window)//64+(0 if specialized else 2)) if window else (n-k)//64)
             for k in range(0, n, 256))

@functools.cache
def custom_fa_forward(o:UOp, l_vec:UOp, q:UOp, k:UOp, v:UOp, sinks:UOp|None=None, *, device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int, has_sink:bool=True, window:int=0):
  # Four-wave GPT-OSS SWA avoids computing six KV tiles per wave for a 128-token window. Generic/Llama stays unchanged.
  w4 = bool(getenv("GPTOSS_FA_FWD_W4", 0) and arch == "gfx950" and has_sink and
            (B, N, H, H_KV, D, window) == (2, 8192, 64, 8, 64, 128))
  # Full causal uses a separate schedule that retains the original four-KV-tile epilogue boundaries for bit-exact O/LSE.
  full_w4 = bool(getenv("GPTOSS_FA_FULL_FWD_W4", 0) and arch == "gfx950" and has_sink and
                 (B, N, H, H_KV, D, window) == (2, 8192, 64, 8, 64, 0))
  full_lean = full_w4 and bool(getenv("GPTOSS_FA_FULL_FWD_LEAN", 0))
  source_name = "fa_fwd_causal_d64.cpp" if full_w4 else "fa_fwd_causal_d64_window.cpp" if w4 else "fa_fwd_causal.cpp"
  code = (pathlib.Path(__file__).parent / source_name).read_text()
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_H_KV={H_KV}", f"-DATTN_D={D}", f"-DATTN_SINK={int(has_sink)}", f"-DWINDOW={window}"]

  if full_lean: compile_args.append("-DGPTOSS_FA_FULL_FWD_LEAN=1")

  Q_BLOCK_SIZE = 32
  NUM_WARPS = 4 if w4 or full_w4 else 8
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (H, (math.ceil((N // Q_BLOCK_SIZE) / NUM_WARPS)), B)
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

  el = q.dtype.itemsize
  mem = (2*B*N*H*D + 2*B*N*H_KV*D) * el + B*H*N * l_vec.dtype.itemsize
  # Two matrix products, QK and PV; estimate dominant MMA work, not softmax/address instructions.
  estimates = Estimates(ops=4*B*H*D*_fa_forward_pairs(N, window, NUM_WARPS, full_w4), lds=mem, mem=mem)
  buf_inputs = (o.base, l_vec.base, q.base, k.base, v.base) + ((sinks.base,) if has_sink else ())
  name = "custom_fa_forward_d64_full_w4" if full_w4 else "custom_fa_forward_d64_window_w4" if w4 else "custom_fa_forward"
  if full_lean: name += "_lean1"
  sink = UOp.sink(*buf_inputs,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name=name, estimates=estimates))

  lib = HIPCCCompiler(arch, compile_args).compile_cached(code)
  if not getenv("NO_HIPCC"):
    lib = bytearray(lib)
    rodata_off = next(sh.header.sh_offset for sh in elf_loader(bytes(lib))[1] if sh.name == ".rodata")
    struct.pack_into('<I', lib, rodata_off, 32768 if w4 or full_w4 else 160000)
    lib = bytes(lib)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def custom_fa_backward_pre(delta_vec:UOp, dq:UOp, o:UOp, do:UOp, device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int):
  code = (pathlib.Path(__file__).parent / "fa_bwd_pre.cpp").read_text()
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_D={D}"]

  DOT_SLICE_QO = 16
  NUM_WARPS = 4
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (B, H, N // (DOT_SLICE_QO * NUM_WARPS))
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

  el = o.dtype.itemsize
  mem = 3*B*H*N*D * el + B*H*N * delta_vec.dtype.itemsize
  estimates = Estimates(ops=2*B*H*N*D, lds=mem, mem=mem)
  sink = UOp.sink(delta_vec.base, dq.base, o.base, do.base,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name="custom_fa_backward_pre", estimates=estimates))

  lib = HIPCCCompiler(arch, compile_args).compile_cached(code)
  if not getenv("NO_HIPCC"):
    lib = bytearray(lib)
    rodata_off = next(sh.header.sh_offset for sh in elf_loader(bytes(lib))[1] if sh.name == ".rodata")
    struct.pack_into('<I', lib, rodata_off, 0)
    lib = bytes(lib)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def custom_fa_backward(dq:UOp, dk:UOp, dv:UOp, do:UOp, q:UOp, k:UOp, v:UOp, l_vec:UOp, delta_vec:UOp, device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int, window:int=0):
  # Keep the performance-critical D=64 schedule independent from D=128 so its register allocation,
  # tiling, and instruction ordering can evolve without risking the D=128 kernel.
  # The generic D=64 kernel is a correctness baseline for full causal only; windowed H8 requires its dedicated schedule.
  use_d64_specialization = D == 64 and (window or not getenv("FA_D64_GENERIC", 0))
  source = ("fa_bwd_causal_d64_window.cpp" if window else "fa_bwd_causal_d64.cpp") if use_d64_specialization else "fa_bwd_causal.cpp"
  code = (pathlib.Path(__file__).parent / source).read_text()
  BLOCK_SIZE_KV, NUM_WARPS = 256, 4
  GROUP_SIZE = H // H_KV
  HEADS_PER_WG = _bwd_heads_per_wg(D, GROUP_SIZE, window)
  # Keep all LDS publication/reuse barriers, omitting only the intermediate wave-local dQ rendezvous.
  relax_dq_sync = bool(getenv("GPTOSS_FA_BWD_RELAX_DQ_SYNC", 0) and use_d64_specialization and arch == "gfx950" and
                       (B, N, H, H_KV, D, window, HEADS_PER_WG) == (2, 8192, 64, 8, 64, 0, 2))
  double_attn = relax_dq_sync and bool(getenv("GPTOSS_FA_BWD_DOUBLE_ATTN", 0))
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_H_KV={H_KV}", f"-DATTN_D={D}", f"-DWINDOW={window}",
                  f"-DBWD_BLOCK_SIZE_KV={BLOCK_SIZE_KV}", f"-DNUM_WARPS={NUM_WARPS}", f"-DBWD_HEADS_PER_WG={HEADS_PER_WG}"]
  if relax_dq_sync: compile_args.append("-DGPTOSS_FA_BWD_RELAX_DQ_SYNC=1")
  if double_attn: compile_args.append("-DGPTOSS_FA_BWD_DOUBLE_ATTN=1")
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (H // HEADS_PER_WG, N // BLOCK_SIZE_KV, B)
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

  el = q.dtype.itemsize
  mem = (3*B*H*N*D + 4*B*H_KV*N*D) * el + 2*B*H*N * l_vec.dtype.itemsize
  # Five products: recomputed QK, dP, dV, dK and dQ. Count masked tiles that the selected loop still executes.
  estimates = Estimates(ops=10*B*H*D*_fa_backward_pairs(N, window, use_d64_specialization), lds=mem, mem=mem)
  sink = UOp.sink(dq.base, dk.base, dv.base, do.base, q.base, k.base, v.base, l_vec.base, delta_vec.base,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name="custom_fa_backward_d64_full_rs1_ds2_vf1" if double_attn else
                                 "custom_fa_backward_d64_full_rs1" if relax_dq_sync else "custom_fa_backward", estimates=estimates))

  lib = HIPCCCompiler(arch, compile_args).compile_cached(code)
  if not getenv("NO_HIPCC"):
    lib = bytearray(lib)
    rodata_off = next(sh.header.sh_offset for sh in elf_loader(bytes(lib))[1] if sh.name == ".rodata")
    struct.pack_into('<I', lib, rodata_off, 160000)
    lib = bytes(lib)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def custom_fa_backward_post(dq_out:UOp, dq_in:UOp, device:str, arch:str, B:int, N:int, H:int, H_KV:int, D:int):
  code = (pathlib.Path(__file__).parent / "fa_bwd_post.cpp").read_text()
  compile_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                  f"-DATTN_B={B}", f"-DATTN_N={N}", f"-DATTN_H={H}", f"-DATTN_D={D}"]

  DOT_SLICE_QO = 16
  NUM_WARPS = 4
  NUM_THREADS = 64 * NUM_WARPS
  gsz = (B, H, N // (DOT_SLICE_QO * NUM_WARPS))
  lsz = (NUM_THREADS, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

  el = dq_out.dtype.itemsize
  mem = 2*B*H*N*D * el
  estimates = Estimates(lds=mem, mem=mem)
  sink = UOp.sink(dq_out.base, dq_in.base,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name="custom_fa_backward_post", estimates=estimates))

  lib = HIPCCCompiler(arch, compile_args).compile_cached(code)
  if not getenv("NO_HIPCC"):
    lib = bytearray(lib)
    rodata_off = next(sh.header.sh_offset for sh in elf_loader(bytes(lib))[1] if sh.name == ".rodata")
    struct.pack_into('<I', lib, rodata_off, 160000)
    lib = bytes(lib)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))
