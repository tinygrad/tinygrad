import functools, math, platform, subprocess
from collections import namedtuple
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_cpu import ClangJITCompiler

# SIMD arch config. format strings: {} for single-arg, {w}/{x}/{c} for fma
Arch = namedtuple('Arch', 'inc vec width zero ld32 ld16 fma hsum helpers')
_ARCHS = {
  'avx512': Arch(
    inc='immintrin.h', vec='__m512', width=16,
    zero='_mm512_setzero_ps()',
    ld32='_mm512_loadu_ps({})',
    ld16='_mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)({})))',
    fma='_mm512_fmadd_ps({w},{x},{c})',
    hsum='_mm512_reduce_add_ps({})',
    helpers='',
  ),
  'avx2': Arch(
    inc='immintrin.h', vec='__m256', width=8,
    zero='_mm256_setzero_ps()',
    ld32='_mm256_loadu_ps({})',
    ld16='_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)({})))',
    fma='_mm256_fmadd_ps({w},{x},{c})',
    hsum='hsum256({})',
    helpers=('static inline float hsum256(__m256 v) {\n'
             '  __m128 hi = _mm256_extractf128_ps(v, 1);\n'
             '  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), hi);\n'
             '  lo = _mm_add_ps(lo, _mm_movehdup_ps(lo));\n'
             '  return _mm_cvtss_f32(_mm_add_ss(lo, _mm_movehl_ps(lo, lo)));\n}\n'),
  ),
  'neon': Arch(
    inc='arm_neon.h', vec='float32x4_t', width=4,
    zero='vdupq_n_f32(0)',
    ld32='vld1q_f32({})',
    ld16='vcvt_f32_f16(vld1_f16({}))',
    fma='vfmaq_f32({c},{w},{x})',
    hsum='vaddvq_f32({})',
    helpers='',
  ),
}

def _get_arch():
  m = platform.machine()
  if m in ('x86_64', 'AMD64'):
    try: return 'avx512' if b'avx512f' in subprocess.check_output(['grep', '-m1', 'flags', '/proc/cpuinfo']) else 'avx2'
    except Exception: return 'avx2'
  if m in ('arm64', 'aarch64'): return 'neon'
  return None

# this could be made more readable
def _gen_src(name: str, M: int, K: int, half: bool) -> str:
  a = _ARCHS[_get_arch()]
  ldw, wtype = (a.ld16, "__fp16") if half else (a.ld32, "float")
  fma_lines = [f'      a{i} = {a.fma.format(w=ldw.format(f"data1 + (m+{i})*{K} + k"), x="xv", c=f"a{i}")};' for i in range(8)]
  store_lines = [f'    data0[m+{i}] = {a.hsum.format(f"a{i}")};' for i in range(8)]
  lines = [f'#include <{a.inc}>', a.helpers +
    f'void {name}(float* restrict data0, {wtype}* restrict data1, float* restrict data2) {{',
    f'  for (int m = 0; m < {M}; m += 8) {{',
    f'    {a.vec} {", ".join(f"a{i}={a.zero}" for i in range(4))};',
    f'    {a.vec} {", ".join(f"a{i}={a.zero}" for i in range(4, 8))};',
    f'    for (int k = 0; k < {K}; k += {a.width}) {{',
    f'      {a.vec} xv = {a.ld32.format("data2 + k")};',
    *fma_lines, '    }', *store_lines, '  }', '}']
  return '\n'.join(lines) + '\n'

_compile_cache: dict[str, bytes] = {}
_compiler = ClangJITCompiler()

def _custom_cpu_matvec(C: UOp, A: UOp, B: UOp, dname: str) -> UOp:
  # build PROGRAM UOp for y[m] = sum_k W[m,k]*x[k]. C=out(1,M), A=W(M,K), B=x(K,)
  M, K, half = *A.shape, A.dtype.itemsize == 2
  name = f"matvec_{M}_{K}_{'f16' if half else 'f32'}"
  src = _gen_src(name, M, K, half)
  if name not in _compile_cache: _compile_cache[name] = _compiler.compile_cached(src)
  wmem = M * K * (2 if half else 4)
  sink = UOp.sink(C.base, A.base, B.base, arg=KernelInfo(name=name, estimates=Estimates(ops=2*M*K, mem=wmem + K*4 + M*4)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                                UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=_compile_cache[name])))

def can_use_cpu_matvec(a: Tensor, b: Tensor) -> bool:
  # check if a @ b can use custom CPU matvec
  arch = _get_arch()
  if arch is None or a.device != "CPU" or b.device != "CPU": return False
  if a.ndim < 2 or b.ndim != 2 or b.uop.op is not Ops.PERMUTE: return False
  if math.prod(a.shape[:-1]) != 1 or b.dtype not in {dtypes.half, dtypes.float}: return False
  K, M = b.shape
  return K % _ARCHS[arch].width == 0 and M % 8 == 0

def cpu_matvec(a: Tensor, b: Tensor) -> Tensor:
  # compute a=(...,K) @ b=(K,M) where b is PERMUTE view of contiguous (M,K) weight
  K, M = b.shape
  out = Tensor.empty(1, M, dtype=dtypes.float, device=a.device)
  out = Tensor.custom_kernel(out, Tensor(b.uop.src[0], device=b.device), a.reshape(K),
                             fxn=functools.partial(_custom_cpu_matvec, dname=a.device))[0]
  return out.reshape(*a.shape[:-1], M)
