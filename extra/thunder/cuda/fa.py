# ThunderKittens (CUDA) flash-attention, wired as a tinygrad Ops.PROGRAM graph node so it survives
# TinyJit/graph capture (mirroring extra/thunder/metal/fa.py and extra/thunder/amd/fa.py). The kernel
# is fa.cu; B/N/H/D are baked per shape (constexpr substitution), it is compiled with NVCC (kittens.cuh
# needs the full toolchain, not nvrtc), and its 48KB dynamic shared memory is passed to the launch via
# ProgramInfo aux -> CUDAProgram.smem.
import pathlib, functools, re
from tinygrad import Device, Tensor
from tinygrad.dtype import dtypes
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_cuda import NVCCCompiler
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo

_DIR = pathlib.Path(__file__).parent
_INC = (_DIR / "include").as_posix()
NUM_WORKERS, WARP_THREADS = 4, 32

def compute_rows(D:int) -> int: return 16 if D >= 128 else (16 * 64) // D   # tile rows; kittens needs %16

def _kittens_define(arch:str) -> str:
  # ThunderKittens arch profile. sm_90(H100)->HOPPER, sm_80(A100)->A100, sm_86/89(A6000/4090)->4090.
  cc = int(arch.removeprefix("sm_"))
  if cc >= 90: return "-DKITTENS_HOPPER"
  if cc == 80: return "-DKITTENS_A100"
  return "-DKITTENS_4090"

def _bake(code:str, B:int, N:int, H:int, D:int) -> str:
  # fa.cu uses `constexpr int ATTN_x = <n>;` (not #define), so substitute the literals per shape.
  for name, val in (("ATTN_B", B), ("ATTN_N", N), ("ATTN_H", H), ("ATTN_D", D)):
    code = re.sub(rf"constexpr int {name} = \d+;", f"constexpr int {name} = {val};", code)
  return code

@functools.cache
def custom_fa_forward_cuda(o:UOp, q:UOp, k:UOp, v:UOp, device:str, arch:str, B:int, H:int, N:int, D:int):
  ROWS = compute_rows(D)
  assert N % (ROWS * NUM_WORKERS) == 0, (f"N={N} must be a multiple of ROWS*NUM_WORKERS={ROWS*NUM_WORKERS} "
    f"for D={D}; the right-fill -inf mask in fa.cu is disabled, so partial query/kv tiles are unsupported.")
  code = _bake((_DIR / "fa.cu").read_text(), B=B, N=N, H=H, D=D)
  kitten_args = [f"-I{_INC}", "-std=c++20", "--expt-relaxed-constexpr", _kittens_define(arch)]
  lib = NVCCCompiler(arch, ptx=False, extra_options=kitten_args).compile_cached(code)  # cubin (dynamic smem)

  gsz = (N // (ROWS * NUM_WORKERS), H, B)              # blockIdx.x/y/z = (q_blocks, head, batch)
  lsz = (NUM_WORKERS * WARP_THREADS, 1, 1)            # 128 threads (4 warps) -- matches __launch_bounds__
  threadIdx_x = UOp.special(lsz[0], "lidx0")
  blockIdx_x, blockIdx_y, blockIdx_z = (UOp.special(gsz[i], f"gidx{i}") for i in range(3))

  mem = (4 * B * H * N * D) * q.dtype.itemsize         # Q,K,V read + O written
  estimates = Estimates(ops=2 * B * H * N * N * D, lds=mem, mem=mem)
  # buffer order O,Q,K,V MUST match attend_ker(O_ptr,Q_ptr,K_ptr,V_ptr).
  sink = UOp.sink(o.base, q.base, k.base, v.base, threadIdx_x, blockIdx_x, blockIdx_y, blockIdx_z,
                  arg=KernelInfo(name="attend_ker", estimates=estimates))
  smem = 16384 * 3                                     # 49152B dynamic shared (fa.cu extern __shm[])
  prog = ProgramInfo.from_sink(sink, aux=(smem,))      # aux[0] -> CUDAProgram.smem (sharedMemBytes + setattr)
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                  UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)), arg=prog)

def flash_attention_cuda(q:Tensor, k:Tensor, v:Tensor) -> Tensor:
  """q,k,v: (B,H,S,D) (already RoPE'd). D in {64,128}, S a multiple of 16*(64//D)*4. Returns (B,H,S,D) bf16.
  fa.cu's global layout is (B,N,H,D), so we move H<->S in and out. Graph-node version (survives TinyJit)."""
  B, H, S, D = q.shape
  qn, kn, vn = (t.permute(0, 2, 1, 3).contiguous().cast(dtypes.bfloat16) for t in (q, k, v))  # (B,S,H,D)
  out = Tensor.empty(B, S, H, D, dtype=dtypes.bfloat16, device=q.device)
  arch = Device[q.device].compiler.arch
  out, *_ = Tensor.custom_kernel(out, qn, kn, vn,
              fxn=functools.partial(custom_fa_forward_cuda, device=q.device, arch=arch, B=B, H=H, N=S, D=D))
  return out.permute(0, 2, 1, 3)   # (B,S,H,D) -> (B,H,S,D)


if __name__ == "__main__":
  import time
  from tinygrad import GlobalCounters
  from tinygrad.engine.jit import TinyJit

  def test_shape(B, H, S, D, label):
    print(f"\n=== {label}: B={B} H={H} S={S} D={D} ===")
    Tensor.manual_seed(0)
    q, k, v = (Tensor.randn(B, H, S, D).cast("bfloat16") for _ in range(3))
    Tensor.realize(q, k, v)
    ref = q.scaled_dot_product_attention(k, v).float().realize()    # (B,H,S,D)
    out = flash_attention_cuda(q, k, v).float().realize()
    num, den = (out * ref).sum(), (out.square().sum().sqrt() * ref.square().sum().sqrt())
    print(f"  cosine {(num/den).item():.6f}   max abs err {(out-ref).abs().max().item():.6f}")
    # JIT replay must not KeyError / recompile (the whole reason for the graph node)
    jf = TinyJit(lambda a, b, c: flash_attention_cuda(a, b, c).realize())
    for _ in range(3): jf(q, k, v)
    Device[Device.DEFAULT].synchronize()
    t0 = time.perf_counter()
    for _ in range(50): jf(q, k, v)
    Device[Device.DEFAULT].synchronize()
    print(f"  JIT replay OK: {(time.perf_counter()-t0)/50*1e3:.4f} ms/call")

  test_shape(1, 16, 1024, 64, "canonical D=64")
  test_shape(1, 24, 576, 128, "FLUX D=128 (512 txt + 64 img)")
