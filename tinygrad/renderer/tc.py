import math
from dataclasses import dataclass
from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import PatternMatcher, UOp, UPat, Ops

@dataclass(frozen=True)
class TensorCore: # D = A * B + C, A is (M x K), B is (K x N), C and D are (M x N)
  dims: tuple[int,int,int] # N, M, K
  threads: int # number of threads that construct the warp
  elements_per_thread: tuple[int, int, int] # elements per-thread to load/store from A/B/C
  dtype_in: DType # dtype for A and B
  dtype_out: DType # dtype for C and D
  opts: tuple[str, ...] # ordered tuple of "ux" or "lx" specifying kernel opts to perform. "ux" upcasts dim x and "lx" localizes dim x
  # A and B fragments as (lane bits, element bits), least significant bit first, in tile bits m<i>/n<i>/k<i>. opts defines C
  # a foreign lane bit is broadcast. the k bits may be permuted, identically in A and B
  frag_a: tuple[tuple[str, ...], tuple[str, ...]]
  frag_b: tuple[tuple[str, ...], tuple[str, ...]]
  def axis_coords(self) -> list[str]:
    # tile bit of each tc axis in creation order, the opts then the k unrolls. split j of a dim is bit j
    bit, ret = [0, 0], []
    for opt in self.opts:
      ret.append("nm"[d:=int(opt[1])] + str(bit[d]))
      bit[d] += 1
    return ret + [f"k{i}" for i,_ in enumerate(self.get_reduce_axes())]
  def relabel(self) -> list[dict[int, int]]:
    # tc axis -> fragment slot axis, per operand
    coords, lanes = self.axis_coords(), [i for i,opt in enumerate(self.opts) if opt[0] == "l"]
    return [{coords.index(c): y for y,c in zip(lanes + self.base_upcast_axes()[:len(f[1])][::-1], f[0]+f[1])} for f in (self.frag_a, self.frag_b)]
  def get_reduce_axes(self): return [(i, 2) for i in range(int(math.log2(self.dims[2])))]
  def get_upcast_axes(self): return [opt for opt in self.opts if opt[0] == "u"]
  def get_local_axes(self): return [opt for opt in self.opts if opt[0] == "l"]
  def base_upcast_axes(self):
    # element slots, most significant bit first: upcast then reduce
    return (list(range(len(self.opts), len(self.opts)+len(self.get_reduce_axes()))) + [i for i,opt in enumerate(self.opts) if opt[0] == "u"])[::-1]
  def __str__(self): return "_".join(["WMMA"] + list(map(str, self.dims)) + [self.dtype_in.name, self.dtype_out.name])
  def __post_init__(self):
    # all axes have size 2
    local_axes, upcast_axes = len(self.get_local_axes()), len(self.get_upcast_axes())
    assert self.dims[0] * self.dims[1] == 2**(local_axes + upcast_axes), \
      f"N({self.dims[0]}) x M({self.dims[1]}) != local({2**local_axes}) x upcast({2**upcast_axes}) with opts({self.opts})"
    assert 2**local_axes == self.threads, f"{self.threads} threads construct the warp but found {2**local_axes} in {self.opts}"
    assert 2**upcast_axes == self.elements_per_thread[2], \
      f"{self.elements_per_thread[2]} elements from C are processed per thread but found {2**upcast_axes} in {self.opts}"
    # check dims match opts
    assert self.dims[0] == 2**len(gd:=[x for x in self.opts if x[1] == '0']), f"opts wrong on dims[0], {self.dims[0]} vs {gd}"
    assert self.dims[1] == 2**len(gd:=[x for x in self.opts if x[1] == '1']), f"opts wrong on dims[1], {self.dims[1]} vs {gd}"
    # NOTE: the K opts is implictly set by the dim
    # each own bit appears once, only lane bits may be foreign
    for i,(f,dims) in enumerate(zip((self.frag_a, self.frag_b), ("mk","kn"))):
      own = {c for c in self.axis_coords() if c[0] in dims}
      assert 2**len(f[0]) == self.threads and 2**len(f[1]) == self.elements_per_thread[i], f"fragment {f} has the wrong size"
      assert len(set(f[0]+f[1])) == len(f[0]+f[1]) and set(f[1]) <= own <= set(f[0]+f[1]) <= set(self.axis_coords()), \
        f"fragment {f} isn't distinct bits covering {dims}"

# ***** NVIDIA *****

cuda_tc_opts = ("u0","l0","l0","l1","l1","l1","u1")  # shared by all shapes with M=16 N=8

# https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-multiply-accumulate-instructions
cuda_81616 = [TensorCore(dims=(8,16,16), threads=32, elements_per_thread=(8,4,4), dtype_in=di, dtype_out=do, opts=cuda_tc_opts,
  frag_a=(("k1", "k2", "m0", "m1", "m2"), ("k0", "m3", "k3")), frag_b=(("k1", "k2", "n0", "n1", "n2"), ("k0", "k3")))
  for di,do in [(dtypes.half,dtypes.float), (dtypes.bfloat16,dtypes.float), (dtypes.half,dtypes.half)]]
cuda_81632_f8 = [TensorCore(dims=(8,16,32), threads=32, elements_per_thread=(16,8,4), dtype_in=di, dtype_out=do, opts=cuda_tc_opts,
  frag_a=(("k2", "k3", "m0", "m1", "m2"), ("k0", "k1", "m3", "k4")), frag_b=(("k2", "k3", "n0", "n1", "n2"), ("k0", "k1", "k4")))
  for di,do in [(dtypes.fp8e4m3,dtypes.float),(dtypes.fp8e5m2,dtypes.float)]]
cuda_8168_f16 = [TensorCore(dims=(8,16,8), threads=32, elements_per_thread=(4,2,4), dtype_in=di, dtype_out=do, opts=cuda_tc_opts,
  frag_a=(("k1", "k2", "m0", "m1", "m2"), ("k0", "m3")), frag_b=(("k1", "k2", "n0", "n1", "n2"), ("k0",)))
  for di,do in [(dtypes.half,dtypes.float), (dtypes.half,dtypes.half)]]
cuda_8168_tf32 = [TensorCore(dims=(8,16,8), threads=32, elements_per_thread=(4,2,4), dtype_in=dtypes.float, dtype_out=dtypes.float, opts=cuda_tc_opts,
  frag_a=(("k0", "k1", "m0", "m1", "m2"), ("m3", "k2")), frag_b=(("k0", "k1", "n0", "n1", "n2"), ("k2",)))]
cuda_sm75: list[TensorCore] = cuda_8168_f16
cuda_sm80: list[TensorCore] = cuda_81616 + cuda_8168_f16 + cuda_8168_tf32
cuda_sm89: list[TensorCore] = cuda_sm80 + cuda_81632_f8

def get_cuda(arch): return cuda_sm89 if (ver:=int(arch[3:])) >= 89 else cuda_sm80 if ver >= 80 else cuda_sm75 if ver >= 75 else []

# ***** AMD *****

# https://gpuopen.com/learn/wmma_on_rdna3/
amd_rdna3 = [TensorCore(dims=(16,16,16), threads=32, elements_per_thread=(16,16,8), dtype_in=di, dtype_out=do,
  opts=("l0","l0","l0","l0","l1","u1","u1","u1"),
  frag_a=(("m0", "m1", "m2", "m3", "n0"), ("k0", "k1", "k2", "k3")), frag_b=(("n0", "n1", "n2", "n3", "m0"), ("k0", "k1", "k2", "k3")))
  for di,do in [(dtypes.half,dtypes.float),(dtypes.half,dtypes.half),(dtypes.bfloat16,dtypes.float),(dtypes.int8,dtypes.int32)]]
amd_rdna4 = [TensorCore(dims=(16,16,16), threads=32, elements_per_thread=(8,8,8), dtype_in=di, dtype_out=do,
  opts=("l0","l0","l0","l0","u1","u1","u1","l1"),
  frag_a=(("m0", "m1", "m2", "m3", "k2"), ("k0", "k1", "k3")), frag_b=(("n0", "n1", "n2", "n3", "k2"), ("k0", "k1", "k3")))
  for di,do in [(dtypes.half,dtypes.float),(dtypes.half,dtypes.half),(dtypes.bfloat16,dtypes.float),(dtypes.bfloat16,dtypes.bfloat16)]]

# https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme
amd_cdna_161616 = [TensorCore(dims=(16,16,16), threads=64, elements_per_thread=(4,4,4), dtype_in=di, dtype_out=do,
  opts=("l0","l0","l0","l0","u1","u1","l1","l1"),
  frag_a=(("m0", "m1", "m2", "m3", "k2", "k3"), ("k0", "k1")), frag_b=(("n0", "n1", "n2", "n3", "k2", "k3"), ("k0", "k1")))
  for di,do in [(dtypes.half,dtypes.float),(dtypes.bfloat16,dtypes.float)]]

amd_cdna_161632 = [TensorCore(dims=(16,16,32), threads=64, elements_per_thread=(8,8,4), dtype_in=di, dtype_out=do,
  opts=("l0","l0","l0","l0","u1","u1","l1","l1"),
  frag_a=(("m0", "m1", "m2", "m3", "k3", "k4"), ("k2", "k0", "k1")), frag_b=(("n0", "n1", "n2", "n3", "k3", "k4"), ("k2", "k0", "k1")))
  for di,do in [(dtypes.fp8e5m2,dtypes.float),(dtypes.fp8e4m3,dtypes.float),(dtypes.half,dtypes.float),(dtypes.bfloat16,dtypes.float)]]

amd_cdna_1616128 = [TensorCore(dims=(16,16,128), threads=64, elements_per_thread=(32,32,4), dtype_in=di, dtype_out=do,
  opts=("l0","l0","l0","l0","u1","u1","l1","l1"),
  frag_a=(("m0", "m1", "m2", "m3", "k5", "k6"), ("k2", "k3", "k4", "k0", "k1")),
  frag_b=(("n0", "n1", "n2", "n3", "k5", "k6"), ("k2", "k3", "k4", "k0", "k1")))
  for di,do in [(dtypes.fp8e5m2,dtypes.float),(dtypes.fp8e4m3,dtypes.float)]]

amd_cdna3 = amd_cdna_161632[:2] + amd_cdna_161616

amd_cdna4 = amd_cdna_1616128 + amd_cdna_161632 + amd_cdna_161616

def get_amd(arch): return {"gfx942": amd_cdna3, "gfx950": amd_cdna4, "gfx1200": amd_rdna4, "gfx1201": amd_rdna4}.get(arch, amd_rdna3)

pm_validate_wmma_rdna3 = PatternMatcher([
  (UPat(Ops.WMMA, name="x", dtype=dtypes.int32), lambda x: x.replace(
    src=(x.src[0].bitcast(dtypes.uint32), x.src[1].bitcast(dtypes.uint32), x.src[2]))
    if x.src[0].dtype == dtypes.int8 and x.src[0].max_numel() == 16 else None),
  (UPat(Ops.WMMA, name="x", dtype=dtypes.half), lambda x: UOp(Ops.STACK, src=tuple(x.replace(
      src=(x.src[0], x.src[1], UOp(Ops.STACK, src=tuple(x.src[2].index(UOp.const(j//2, dtypes.int16))
      if j%2 == 0 else UOp.const(0.0, x.src[2].dtype)
      for j in range(x.max_numel()*2)))),
      arg=(*x.arg[:4], None)).index(UOp.const(i*2, dtypes.int16))
      for i in range(x.max_numel()))) if x.max_numel() == 8 else None),
  (UPat(Ops.WMMA, name="x"), lambda x: x.replace(
    src=(x.src[0].bitcast(dtypes.uint16), x.src[1].bitcast(dtypes.uint16), x.src[2]))
    if x.src[0].dtype == dtypes.bfloat16 and x.src[0].max_numel() == 16 else None),
])

pm_validate_wmma_rdna4 = PatternMatcher([
  (UPat(Ops.WMMA, name="x", dtype=dtypes.bfloat16), lambda x: x.replace(
    src=(x.src[0].bitcast(dtypes.uint16), x.src[1].bitcast(dtypes.uint16), x.src[2].bitcast(dtypes.uint16)))
      .bitcast(dtypes.bfloat16) if x.max_numel() == 8 and x.src[0].dtype == dtypes.bfloat16 and x.src[0].max_numel() == 8 else None),
  (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
    lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint16), x.src[1].bitcast(dtypes.uint16), x.src[2]))
    if x.max_numel() == 8 and x.src[0].dtype == dtypes.bfloat16 and x.src[0].max_numel() == 8 else None)
])

pm_validate_wmma_cdna = PatternMatcher([
  (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
    lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint32), x.src[1].bitcast(dtypes.uint32), x.src[2]))
    if x.arg[0][2] == 128 and x.src[0].dtype.itemsize <= 8 else None),
  (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
    lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint16), x.src[1].bitcast(dtypes.uint16), x.src[2]))
    if x.max_numel() == 4 and x.src[0].dtype == dtypes.bfloat16 and x.src[0].max_numel() == 4 else None),
  (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
    lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint64), x.src[1].bitcast(dtypes.uint64), x.src[2]))
    if x.max_numel() == 4 and x.src[0].dtype in dtypes.fp8_ocp and x.src[0].max_numel() == 8 else None),
])
# ***** Apple Metal *****

metal = [TensorCore(dims=(8,8,8), threads=32, elements_per_thread=(2,2,2), dtype_in=di, dtype_out=do,
  opts=("u0","l0","l1","l1","l0","l1"),
  frag_a=(("k1", "m0", "m1", "k2", "m2"), ("k0",)), frag_b=(("n1", "k0", "k1", "n2", "k2"), ("n0",)))
  for di,do in [(dtypes.float,dtypes.float),(dtypes.half,dtypes.float),
                (dtypes.half,dtypes.half),(dtypes.bfloat16,dtypes.float),(dtypes.bfloat16,dtypes.bfloat16)]]
