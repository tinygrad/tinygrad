import math
from dataclasses import dataclass
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import getenv

@dataclass(frozen=True)
class TensorCore: # D = A * B + C, A is (M x K), B is (K x N), C and D are (M x N)
  dims: tuple[int,int,int] # N, M, K
  threads: int # number of threads that construct the warp
  elements_per_thread: tuple[int, int, int] # elements per-thread to load/store from A/B/C
  dtype_in: DType # dtype for A and B
  dtype_out: DType # dtype for C and D
  opts: tuple[str, ...] # ordered tuple of "ux" or "lx" specifying kernel opts to perform. "ux" upcasts dim x and "lx" localizes dim x
  # (local_swizzle, reduce_swizzle, upcast_swizzle)
  swizzle: tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]
  def get_reduce_axes(self): return [(i, 2) for i in range(int(math.log2(self.dims[2])))]
  def get_upcast_axes(self): return [opt for opt in self.opts if opt[0] == "u"]
  def get_local_axes(self): return [opt for opt in self.opts if opt[0] == "l"]
  def __str__(self): return "_".join(["WMMA"] + list(map(str, self.dims)) + [self.dtype_in.name, self.dtype_out.name])
  def __post_init__(self):
    # all axes have size 2, <local> <reduce> <upcast> is the order
    local_axes, upcast_axes, reduce_axes = len(self.get_local_axes()), len(self.get_upcast_axes()), len(self.get_reduce_axes())
    assert self.dims[0] * self.dims[1] == 2**(local_axes + upcast_axes), \
      f"N({self.dims[0]}) x M({self.dims[1]}) != local({2**local_axes}) x upcast({2**upcast_axes}) with opts({self.opts})"
    assert 2**local_axes == self.threads, f"{self.threads} threads construct the warp but found {2**local_axes} in {self.opts}"
    assert 2**upcast_axes == self.elements_per_thread[2], \
      f"{self.elements_per_thread[2]} elements from C are processed per thread but found {2**upcast_axes} in {self.opts}"
    # check swizzle
    assert len(self.swizzle[0]) == 3 and len(self.swizzle[1]) == 3, "swizzle has wrong part count"
    assert len(self.swizzle[0][0]) == len(self.swizzle[1][0]) == local_axes, "local swizzle size is wrong"
    assert len(self.swizzle[0][1]) == len(self.swizzle[1][1]) == reduce_axes, "reduce swizzle size is wrong"
    assert len(self.swizzle[0][2]) == len(self.swizzle[1][2]) == upcast_axes, "reduce/upcast swizzle size is wrong"
    assert all(sorted(s[0] + s[1] + s[2]) == list(range(local_axes + reduce_axes + upcast_axes)) for s in self.swizzle), "swizzle missing some dims"

# ***** NVIDIA *****

cuda_tc_opts = ("u0","l0","l0","l1","l1","l1","u1")  # shared by all shapes with M=16 N=8

# https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-multiply-accumulate-instructions
cuda_81616 = [TensorCore(dims=(8,16,16), threads=32, elements_per_thread=(8,4,4), dtype_in=di, dtype_out=do, opts=cuda_tc_opts,
  swizzle=(((6,7,2,3,4),(0,1,9,5),(10,8)), ((6,7,9,0,1),(2,3,4,10),(5,8)))) for di,do in [(dtypes.half,dtypes.float), (dtypes.bfloat16,dtypes.float),
                                                                                          (dtypes.half,dtypes.half)]]
cuda_8168_f16 = [TensorCore(dims=(8,16,8), threads=32, elements_per_thread=(4,2,4), dtype_in=di, dtype_out=do, opts=cuda_tc_opts,
  swizzle=(((6,7,2,3,4),(0,1,8),(5,9)), ((6,7,8,0,1),(2,3,4),(9,5)))) for di,do in [(dtypes.half,dtypes.float), (dtypes.half,dtypes.half)]]
cuda_8168_tf32 = [TensorCore(dims=(8,16,8), threads=32, elements_per_thread=(4,2,4), dtype_in=dtypes.float, dtype_out=dtypes.float, opts=cuda_tc_opts,
  swizzle=(((5,6,2,3,4),(0,1,8),(9,7)), ((5,6,8,0,1),(2,3,4),(9,7))))]

cuda_sm80: list[TensorCore] = cuda_81616 + cuda_8168_f16
if getenv("ALLOW_TF32", 0): cuda_sm80 += cuda_8168_tf32
cuda_sm75: list[TensorCore] = cuda_8168_f16

# ***** AMD *****

# https://gpuopen.com/learn/wmma_on_rdna3/
amd_rdna3 = [TensorCore(dims=(16,16,16), threads=32, elements_per_thread=(16,16,8), dtype_in=di, dtype_out=do,
  opts=("l0","l0","l0","l0","l1","u1","u1","u1"), swizzle=(((4,9,10,11,0),(1,2,3,5),(6,7,8)), ((0,1,2,3,4),(9,10,11,5),(6,7,8))))
  for di,do in [(dtypes.half,dtypes.float),(dtypes.half,dtypes.half),(dtypes.bfloat16,dtypes.float)]]
amd_rdna4 = [TensorCore(dims=(16,16,16), threads=32, elements_per_thread=(8,8,8), dtype_in=di, dtype_out=do,
  opts=("l0","l0","l0","l0","u1","u1","u1","l1"), swizzle=(((9,10,11,4,7),(0,1,2,3),(5,6,8)),((0,1,2,3,7),(4,9,10,11),(5,6,8))))
  for di,do in [(dtypes.half,dtypes.float),(dtypes.half,dtypes.half),(dtypes.bfloat16,dtypes.float),(dtypes.bfloat16,dtypes.bfloat16)]]

# https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme
amd_cdna = [TensorCore(dims=(16,16,16), threads=64, elements_per_thread=(4,4,4), dtype_in=di, dtype_out=do,
  opts=("l0","l0","l0","l0","u1","u1","l1","l1"), swizzle=(((10,11,4,5,8,9),(0,1,2,3),(6,7)),((0,1,2,3,8,9),(4,5,10,11),(6,7))))
  for di,do in [(dtypes.half,dtypes.float),(dtypes.bfloat16,dtypes.float)]]

# ***** Apple Metal *****

metal = [TensorCore(dims=(8,8,8), threads=32, elements_per_thread=(2,2,2), dtype_in=di, dtype_out=do, opts=("u0","l0","l1","l1","l0","l1"),
  swizzle=(((6,1,2,7,4),(8,0,3),(5,)), ((0,5,6,3,7),(1,2,4),(8,)))) for di,do in [(dtypes.float,dtypes.float),(dtypes.half,dtypes.float),
  (dtypes.half,dtypes.half),(dtypes.bfloat16,dtypes.float),(dtypes.bfloat16,dtypes.bfloat16)]]

# ***** Apple AMX *****

amx = [TensorCore(dims=(sz,sz,1), threads=1, elements_per_thread=(sz,sz,sz*sz), dtype_in=dt, dtype_out=dt,
                  swizzle=(((),(),(0,1,2,3,4,5,6,7)), ((),(),(4,5,6,7,0,1,2,3))),
                  opts=("u0","u0","u0","u0","u1","u1","u1","u1")) for dt,sz in [(dt, 64 // dt.itemsize) for dt in [dtypes.float]]]

# ***** Intel ****

intel = [TensorCore(dims=(8,8,16), threads=8, elements_per_thread=(16,16,8), dtype_in=dtypes.half, dtype_out=dtypes.float,
  opts=("l0","l0","l0","u1","u1","u1"), swizzle=(((4,5,6),(0,1,2,3),(7,8,9)), ((0,1,2),(7,8,9,3),(4,5,6))))]
