import numpy as np, os
from tinygrad.helpers import getenv, flat_mv
from tinygrad import dtypes

script_dir = os.path.dirname(os.path.abspath(__file__))

dtype_in = dtypes.half if getenv("HALF") else dtypes.float
dtype_out = dtypes.half if getenv("HALF_OUT") else dtypes.float
# acc_dtype = dtypes.half if getenv("ACC_HALF") else None
N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)
CNT = getenv("CNT", 10)

ATOL = getenv("ATOL", 5e-3 if dtype_in == dtypes.float else 1e-2)
RTOL = getenv("RTOL", 1e-4 if dtype_in == dtypes.float else 1e-3)
FLOPS = M * N * K * 2
BW = 2 * ((M*K) + (K*N) + (M*N))

INPUT = getenv("INPUT", "RAND")

# algorithm variations
GEMM_VARIATION = getenv("GEMM_VARIATION", "generated_hcopt")

def randoms():
  if INPUT == "RAND":
    na = np.random.default_rng().normal(scale=1.0, size=(M,K)).astype(dtype=np.float32)
    nb = np.random.default_rng().normal(scale=1.0, size=(K,N)).astype(dtype=np.float32)
  elif INPUT == "IDENTITY" and M==N==K:
    na = np.identity(K, dtype=np.float32)
    nb = np.identity(K, dtype=np.float32)
  elif INPUT == "OUTPUTONES" and M==K:
    na = np.identity(K, dtype=np.float32)
    nb = np.ones((K,N), dtype=np.float32)
  else:
    na = np.ones((M,K), dtype=np.float32)
    nb = np.ones((K,N), dtype=np.float32)
  nc = np.zeros(M*N, np.float32)
  if dtype_in != dtypes.float:
    na = na.astype(np.bfloat16 if dtype_in == dtypes.bfloat16 else np.float16)
    nb = nb.astype(np.bfloat16 if dtype_in == dtypes.bfloat16 else np.float16)
  if dtype_out != dtypes.float:
    nc = nc.astype(np.bfloat16 if dtype_in == dtypes.bfloat16 else np.float16)
  return na, nb, nc

if __name__ == "__main__":
  print(f"gemm variation: {GEMM_VARIATION=} {M=} {N=} {K=} {dtype_in=} {dtype_out=}") # {acc_dtype=}")
  prog, global_size, local_size = None, None, None

  if getenv("CUDA") == 1:
    from tinygrad.runtime.ops_cuda import CUDAAllocator, CUDADevice, CUDAProgram, CUDACompiler
    device = CUDADevice("cuda:0")
    compiler = CUDACompiler(device.arch)
    cudaalloc = CUDAAllocator(device)

    a = cudaalloc.alloc(M*K*dtype_in.itemsize)
    b = cudaalloc.alloc(K*N*dtype_in.itemsize)
    c = cudaalloc.alloc(M*N*dtype_out.itemsize)

    if GEMM_VARIATION == "nv_triton" and (M%64)== 0 and (N%128)==0 and (K%64)==0 and dtype_in == dtypes.half and dtype_out == dtypes.float:
      print("Using CUDA and triton-generated kernel")
      # See nv_triton_gemm.annotated.ptx for PTX code which was generated from `PYTHONPATH=. DEBUG=6 CUDA=1 PTX=1 python3 extra/gemm/triton_nv_matmul.py`
      # this kernel with M=N=K=4096 does 162TFLOPS, vs torch at 144TFLOPS and BEAM=8 tinygrad at 138TFLOPS.  theo max is 165TFLOPS.

      # WMMA element size is (M, N, K) = (16, 8, 16)
      # warpgroup size in WMMA tiles is (B_M, B_N, B_K) = (2, 8, 4) so 64 HMMA calls per threadgroup reduce iteration
      # thread block size is (T_M, T_N, T_K) = (2, 2, 1), i.e. macro blocks in M and N, so 256 HMMA calls per kernel reduce iteration
      # kernel reduce iteration size in elements = (64, 128, 64)
      # single iteration SMEM_A = (64 * 64) * (2 bytes / half) =  8192 bytes, SMEM_B = (128 * 64) * (2 bytes / half) = 16384 bytes
      # double-buffer smem = (8192 + 16384) * 2 = 49152 bytes
      # reduce for_loop size = [1, 1, (4096 // 16 // 4)==64]
       # NOTE: T_K > 0 would be group_for_reduce
      prog = CUDAProgram(device, "wmma_example", compiler.compile(open(os.path.join(script_dir, 'max_kernels/nv_triton.cpp')).read()))
      args = (c, a, b)
      kwargs = {
        'global_size': [M//64, N//128, 1],
        'local_size': [128, 1, 1], # 4 warpgroups == (T_M:=2) * (T_N:=2)
        'wait': True,
        'vals': (N, K),
      }
    elif GEMM_VARIATION == "nv_hcopt" and M == N == K == 4096 and dtype_in == dtypes.half and dtype_out == dtypes.half:
      print("Using CUDA and generated hcopt")
      # [Opt(op=OptOps.TC, axis=0, amt=0), Opt(op=OptOps.UPCAST, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=1, amt=4)]
      prog = CUDAProgram(device, "wmma_example", compiler.compile(open(os.path.join(script_dir, 'max_kernels/nv_hcopt.cpp')).read()))
      args = (c, a, b)
      kwargs = {
        'global_size': [32, 64, 1],
        'local_size': [16, 2, 4], # 16,2 are warp, 4 workgroups upcasted to axis=1
        'wait': True,
      }
    else:
      raise RuntimeError(f"invalid gemm variation: {GEMM_VARIATION=} {M=} {N=} {K=} {dtype_in=} {dtype_out=}")

    tms = []
    na, nb, nc = randoms()
    cudaalloc.copyin(a, bytearray(na))
    cudaalloc.copyin(b, bytearray(nb))
    for i in range(CNT):
      tms.append(prog(*args, **kwargs))
    cudaalloc.copyout(flat_mv(nc.data), c)
    comp = na.astype(np.float32) @ nb.astype(np.float32)
    result = nc.reshape(M, N).astype(np.float32)

    print(f"{N*N:10d} {min(tms)*1e6:9.2f} us, would be {FLOPS*1e-9/min(tms):9.2f} GFLOPS matmul, {BW*1e-9/min(tms):.2f} GB/s")
    try:
      np.testing.assert_allclose(result, comp, atol=ATOL, rtol=RTOL)
    except AssertionError as e:
      if getenv("DEBUG_VALUES") > 0:
        indices = np.where(~np.isclose(result, comp, rtol=RTOL, atol=ATOL))
        non_matching_elements_result = result[indices]
        non_matching_elements_comp = comp[indices]
        print("valid       :", np.where(np.isclose(result, comp, rtol=RTOL, atol=ATOL)))
        print("invalid     :", indices)
        print("result      :", non_matching_elements_result)
        print("ground truth:", non_matching_elements_comp)
        print("result sum  :", np.sum(result))
        print("ground sum  :", np.sum(comp))
      raise e

    if getenv("DEBUG_VALUES") > 0:
      print(comp)
      print("ground sum  :", np.sum(comp))
      print(result)
      print("result sum  :", np.sum(result))

  elif getenv("AMD") == 1:
    # note: https://hipfft.readthedocs.io/en/rocm-6.1.2/how-to/fine-tuning-llms/optimizing-triton-kernel.html
    raise RuntimeError("invalid max_matmul device")

  else:
    raise RuntimeError("invalid max_matmul device")

