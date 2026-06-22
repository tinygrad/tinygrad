# ThunderMittens (Metal) flash-attention kernel.
# Ported from the CUDA ThunderKittens kernel extra/thunder/cuda/fa.cu, built the same
# way extra/thunder/metal/gemm.py builds its GEMM. Two surfaces:
#   - the raw Metal C++ string (`fa`), compiled with Device["METAL"].compiler.compile.
#   - a tinygrad graph node: `custom_fa_forward_metal` returns an Ops.PROGRAM UOp wired
#     through Tensor.custom_kernel, mirroring extra/thunder/amd/fa.py:86-114. The graph
#     node survives TinyJit (the raw prg(...) form does not -- the manual buffers aren't a
#     graph node, so JIT replay errors KeyError 'METAL').
#
# Online-softmax (flash) attention, one simdgroup (warp) per threadgroup, one query block
# per threadgroup. Q/K/V are reshaped to [B*H, N, D] (contiguous) so a [ROWS, D] register
# tile loads with the natural row_stride = D (the global layout convention scales the rows
# coord by TILE::rows and uses row_stride = cols).
#
# BH (=B*H) and N are baked into the kernel as ATTN_BH / ATTN_N #defines (prepended per
# shape before compiling), mirroring how the AMD kernel bakes B/N/H via -DATTN_*. This drops
# the BH/N `const constant int&` buffer args so the kernel has ONLY O/Q/K/V `device bf16*`
# args at [[buffer(0..3)]] -- which is what the custom_kernel buffer-binding order needs.

import pathlib, math, functools, statistics
_INC = (pathlib.Path(__file__).parent / "include" / "tk.metal").as_posix()

def compute_rows(D:int) -> int: return (16 * 64) // D  # query/key block height for head dim D

fa = """
#include <metal_stdlib>
#include \"""" + _INC + """\"
using namespace mittens;

// D is the head dim (compile-time). ROWS = 16*(64/D): the query/key block height.
// ATTN_BH (batch*heads) and ATTN_N (sequence length) are baked in per shape via #define.
template<int D, int ROWS>
kernel void flash_attend(
    device bf16* O      [[buffer(0)]],
    device bf16* Q      [[buffer(1)]],
    device bf16* K      [[buffer(2)]],
    device bf16* V      [[buffer(3)]],
    uint3 tg_id            [[threadgroup_position_in_grid]],
    uint  simd_lane_id    [[thread_index_in_simdgroup]])
{
  const int BH = ATTN_BH;   // batch*heads (baked)
  const int N  = ATTN_N;    // sequence length, a multiple of ROWS (baked)

  // global layout: (batch=BH, depth=1, rows=N, cols=D). cols=D is compile-time.
  using gl_t = gl<bf16, -1, 1, -1, D>;
  gl_t Qg(Q, (size_t)BH, nullptr, (size_t)N, nullptr);
  gl_t Kg(K, (size_t)BH, nullptr, (size_t)N, nullptr);
  gl_t Vg(V, (size_t)BH, nullptr, (size_t)N, nullptr);
  gl_t Og(O, (size_t)BH, nullptr, (size_t)N, nullptr);

  const int bh    = (int)tg_id.y;          // which (batch, head)
  const int q_blk = (int)tg_id.x;          // which query block
  const short laneid = (short)simd_lane_id;

  // register tiles
  rt<bf16, ROWS, D,    ducks::rt_layout::row> q_reg;   // Q tile [queries, D]
  rt<bf16, ROWS, D,    ducks::rt_layout::col> k_reg;   // K tile [keys, D] in col layout for mma_ABt
  rt<bf16, ROWS, D,    ducks::rt_layout::row> v_reg;   // V tile [keys, D] in row layout for mma_AB
  rt<float, ROWS, D,   ducks::rt_layout::row> o_reg;   // output accumulator [queries, D]
  rt<float, ROWS, ROWS, ducks::rt_layout::row> att;    // attention scores [queries, keys]
  rt<bf16, ROWS, ROWS, ducks::rt_layout::row> att_bf;  // bf16 copy of att for the second mma

  typename rt<float, ROWS, ROWS, ducks::rt_layout::row>::col_vec max_vec, max_vec_last, norm_vec;

  // load Q tile, fold the softmax scale * log2(e) into Q so we can use exp2.
  load(q_reg, Qg, {bh, 0, q_blk, 0}, laneid);
  if (D == 64)       mul(q_reg, q_reg, (bf16)(0.125f        * 1.44269504089f));
  else if (D == 128) mul(q_reg, q_reg, (bf16)(0.08838834764f * 1.44269504089f));

  // running softmax state
  neg_infty(max_vec);
  zero(norm_vec);
  zero(o_reg);

  const int kv_blocks = N / ROWS;
  for (int kv = 0; kv < kv_blocks; kv++) {
    // att = Q @ K^T   (mma_ABt: b is col layout)
    load(k_reg, Kg, {bh, 0, kv, 0}, laneid);
    zero(att);
    mma_ABt(att, q_reg, k_reg, att);

    // online softmax update
    max_vec_last = max_vec;
    row_max(max_vec, att, max_vec, laneid);          // running row max (over keys)
    sub_row(att, att, max_vec);                       // att -= max_vec (broadcast over rows)
    exp2(att, att);                                   // att = exp2(att - max)
    sub(max_vec_last, max_vec_last, max_vec);         // max_last - max
    exp2(max_vec_last, max_vec_last);                 // corr = exp2(max_last - max)
    mul(norm_vec, norm_vec, max_vec_last);            // norm *= corr
    row_sum(norm_vec, att, norm_vec, laneid);         // norm += rowsum(att)

    copy(att_bf, att);                                // cast att -> bf16
    load(v_reg, Vg, {bh, 0, kv, 0}, laneid);
    mul_row(o_reg, o_reg, max_vec_last);              // o *= corr
    mma_AB(o_reg, att_bf, v_reg, o_reg);              // o += att @ V
  }

  div_row(o_reg, o_reg, norm_vec);                    // o /= norm
  store(Og, o_reg, {bh, 0, q_blk, 0}, laneid);
}

#define instantiate_fa(suffix, D, ROWS) \
  template [[host_name("flash_attend_" #suffix)]] [[kernel]] \
  void flash_attend<D, ROWS>( \
    device bf16*, device bf16*, device bf16*, device bf16*, \
    uint3, uint);

instantiate_fa(d64,  64,  16);
instantiate_fa(d128, 128, 8);
"""

from tinygrad import Device, Tensor
from tinygrad.dtype import dtypes
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import UOp, Ops, KernelInfo

# ---- tinygrad graph-node flash-attention ------------------------------------
# Returns an Ops.PROGRAM UOp that Tensor.custom_kernel wires into the graph (so it
# survives TinyJit), mirroring extra/thunder/amd/fa.py:86-114. There is no ELF rodata
# patching here -- that's AMD shared-mem-specific; this Metal flash kernel is register-only.

@functools.cache
def custom_fa_forward_metal(o:UOp, q:UOp, k:UOp, v:UOp, device:str, B:int, H:int, N:int, D:int):
  ROWS = compute_rows(D)
  assert N % ROWS == 0, (f"N={N} must be a multiple of ROWS={ROWS} for D={D}; partial query/kv "
    f"blocks need a right-fill -inf mask (see extra/thunder/cuda/fa.cu:84-85). FLUX has N=120, "
    f"ROWS=8 so the multiple-of-ROWS assert suffices for now.")
  BH = B * H
  # bake BH/N in as #defines so the kernel has only O/Q/K/V device-pointer args.
  code = f"#define ATTN_BH {BH}\n#define ATTN_N {N}\n" + fa

  gsz = (N // ROWS, BH, 1)
  lsz = (32, 1, 1)
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = UOp.special(gsz[0], "gidx0"), UOp.special(gsz[1], "gidx1"), UOp.special(gsz[2], "gidx2")

  el = q.dtype.itemsize
  mem = (3*BH*N*D + BH*N*D) * el   # Q + K + V read, O written
  estimates = Estimates(ops=2*B*H*N*N*D, lds=mem, mem=mem)
  # buffer order o,q,k,v MUST match the kernel's [[buffer(0..3)]] = O,Q,K,V.
  sink = UOp.sink(o.base, q.base, k.base, v.base,
                  threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name=f"flash_attend_d{D}", estimates=estimates))

  lib = Device["METAL"].compiler.compile(code)

  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)))

def flash_attention(q, k, v):
  """q,k,v: (B*H, N, D) tensors (already RoPE'd). D in {64,128}.
  N must be a multiple of ROWS = (16*64)//D. Returns (B*H, N, D) bf16.
  This is the graph-node version (survives TinyJit)."""
  BH, N, D = q.shape
  out = Tensor.empty(BH, N, D, dtype=dtypes.bfloat16)
  out, *_ = Tensor.custom_kernel(out, q.cast(dtypes.bfloat16), k.cast(dtypes.bfloat16), v.cast(dtypes.bfloat16),
                                 fxn=functools.partial(custom_fa_forward_metal, device=q.device, B=1, H=BH, N=N, D=D))
  return out

# ---- standalone validation + benchmark --------------------------------------
def reshape_bhnd(t):  # (B, N, H, D) -> (B*H, N, D) contiguous
  B, N, H, D = t.shape
  return t.permute(0, 2, 1, 3).reshape(B * H, N, D).contiguous()

def _rand_qkv(B, N, H, D):
  # (B,N,H,D) bf16 randoms + their (B*H,N,D) reshape; shared by test_shape/benchmark.
  Tensor.manual_seed(0)
  q4, k4, v4 = (Tensor.randn(B, N, H, D).cast("bfloat16") for _ in range(3))
  q, k, v = reshape_bhnd(q4), reshape_bhnd(k4), reshape_bhnd(v4)
  Tensor.realize(q, k, v)
  return q4, k4, v4, q, k, v

def reference_sdpa(q, k, v, scale):
  # q,k,v: (B*H, N, D). returns (B*H, N, D) float
  qf, kf, vf = q.cast("float"), k.cast("float"), v.cast("float")
  att = (qf @ kf.transpose(-1, -2)) * scale
  return (att.softmax(-1) @ vf)

def test_shape(B, N, H, D, label):
  ROWS = compute_rows(D)
  scale = 1.0 / math.sqrt(D)
  print(f"\n=== {label}: B={B} N={N} H={H} D={D} (ROWS={ROWS}) ===")
  assert N % ROWS == 0, f"test shape N={N} must be a multiple of ROWS={ROWS}"

  _, _, _, q, k, v = _rand_qkv(B, N, H, D)                         # (B*H, N, D)
  ref = reference_sdpa(q, k, v, scale).realize()  # (B*H, N, D) float
  out = flash_attention(q, k, v).realize()

  of, rf = out.cast("float").realize(), ref
  diff = (of - rf).abs()
  max_err = diff.max().item()
  mean_err = diff.mean().item()
  print(f"max abs err = {max_err:.6f}   mean abs err = {mean_err:.6f}")
  return max_err, mean_err

if __name__ == "__main__":
  import time as _t
  # correctness
  test_shape(1, 1024, 16, 64, "canonical D=64")
  test_shape(1, 576, 24, 128, "FLUX D=128")

  def benchmark(B, N, H, D, label):
    ROWS = compute_rows(D)
    assert N % ROWS == 0, "benchmark shape must be a multiple of ROWS"
    scale = 1.0 / math.sqrt(D)
    print(f"\n=== benchmark {label}: B={B} N={N} H={H} D={D} ===")
    q4, k4, v4, q, k, v = _rand_qkv(B, N, H, D)

    for _ in range(3): flash_attention(q, k, v).realize()  # warmup
    ts = []
    for _ in range(20):
      t0 = _t.perf_counter(); flash_attention(q, k, v).realize(); ts.append((_t.perf_counter()-t0)*1e3)
    ms = statistics.median(ts)
    attn_flops = 2 * B * H * N * N * D + 4 * B * H * N * N + 2 * B * H * N * N * D
    print(f"TK flash:        {ms:.4f} ms   ({attn_flops/(ms*1e-3)/1e9:.1f} GFLOPS)")

    for _ in range(3): reference_sdpa(q, k, v, scale).realize()
    ts = []
    for _ in range(20):
      t0 = _t.perf_counter(); reference_sdpa(q, k, v, scale).realize(); ts.append((_t.perf_counter()-t0)*1e3)
    print(f"tinygrad SDPA:   {statistics.median(ts):.4f} ms")

    try:
      import mlx.core as mx
      qm = mx.array(q4.permute(0,2,1,3).cast("float").tolist()).astype(mx.bfloat16)
      km = mx.array(k4.permute(0,2,1,3).cast("float").tolist()).astype(mx.bfloat16)
      vm = mx.array(v4.permute(0,2,1,3).cast("float").tolist()).astype(mx.bfloat16)
      def mlx_flash():
        o = mx.fast.scaled_dot_product_attention(qm, km, vm, scale=scale); mx.eval(o); return o
      for _ in range(3): mlx_flash()
      ts = []
      for _ in range(20):
        t0 = _t.perf_counter(); mlx_flash(); ts.append((_t.perf_counter()-t0)*1e3)
      print(f"MLX flash:       {statistics.median(ts):.4f} ms")
    except Exception as e:
      print(f"MLX flash:       unavailable ({type(e).__name__}: {e})")

  benchmark(1, 1024, 16, 64, "canonical D=64")
  benchmark(1, 576, 24, 128, "FLUX D=128")
