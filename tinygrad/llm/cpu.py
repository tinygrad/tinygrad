from __future__ import annotations
import functools, platform, ctypes.util
from typing import TYPE_CHECKING
from tinygrad import Device, Tensor, nn, UOp, getenv, dtypes
from tinygrad.helpers import CPU_COUNT
from tinygrad.dtype import DType
from tinygrad.llm.gguf import _GGML_QUANT
from tinygrad.uop.ops import Ops, KernelInfo, ProgramInfo, AxisType
if TYPE_CHECKING:
  from tinygrad.llm.model import ExpertWeights, FFNBlock, Linear

SUPPORTED = platform.system() != "Windows"
_SCALAR_DOT = """
static inline int dot16(const signed char *a, const signed char *b) {
  int ret = 0; for (int i = 0; i < 16; i++) ret += a[i] * b[i]; return ret;
}
static inline int dot32(const signed char *a, const signed char *b) { return dot16(a, b) + dot16(a + 16, b + 16); }
static inline void dot_q6(const unsigned char *w, int subgroup, const signed char *xq, int *lo, int *hi) {
  signed char qvals[32];
  for (int pos = 0; pos < 32; pos++) {
    const int full_pos = subgroup * 32 + pos, within = full_pos & 127;
    const int low = (w[(full_pos >> 7) * 64 + (within & 63)] >> ((within >> 6) * 4)) & 15;
    const int high = (w[128 + (full_pos >> 7) * 32 + (within & 31)] >> ((within >> 5) * 2)) & 3;
    qvals[pos] = (signed char)((low | (high << 4)) - 32);
  }
  *lo = dot16(qvals, xq); *hi = dot16(qvals + 16, xq + 16);
}
"""
_SEM_POOL = """
extern void tiny_cpu_parallel(void (*)(void *, int, int, int, int), void *, int, int);
static void pool_worker(void *opaque, int id, int active, int begin, int end) {
  (void)id; (void)active;
  task_t task = *(task_t *)opaque;
  task.begin = begin; task.end = end;
  worker(&task);
}
static void dispatch(task_t task, int total, int requested) {
  tiny_cpu_parallel(pool_worker, &task, total, requested);
}
"""
_PARALLEL_SOURCE = r"""
#define MAX_THREADS 32
#define SPIN_COUNT __SPIN_COUNT__
typedef unsigned long pthread_t;
typedef void (*work_t)(void *, int, int, int, int);
extern int pthread_once(int *, void (*)(void));
extern int pthread_create(pthread_t *, const void *, void *(*)(void *), void *);
extern int pthread_detach(pthread_t);
extern int sem_init(unsigned long *, int, unsigned int);
extern int sem_wait(unsigned long *);
extern int sem_post(unsigned long *);
__PINNING__
static int once;
static pthread_t tids[MAX_THREADS - 1];
static unsigned long ready[MAX_THREADS - 1][4];
static int ids[MAX_THREADS - 1], workers;
static work_t current_work;
static void *current_opaque;
static int current_total, current_active;
static unsigned generation, completed_generation[MAX_THREADS - 1];
static int sleeping[MAX_THREADS - 1];
static void cpu_relax(void) { __CPU_RELAX__ }
static void *pool_worker(void *opaque) {
  const int id = *(int *)opaque;
  pin_index(id);
  unsigned seen = 0;
  for (;;) {
    unsigned next = seen;
    for (int spin = 0; spin < SPIN_COUNT; spin++) {
      next = __atomic_load_n(&generation, __ATOMIC_ACQUIRE);
      if (next != seen) break;
      cpu_relax();
    }
    if (next == seen) {
      __atomic_store_n(&sleeping[id - 1], 1, __ATOMIC_RELEASE);
      next = __atomic_load_n(&generation, __ATOMIC_ACQUIRE);
      if (next == seen) sem_wait(ready[id - 1]);
      __atomic_store_n(&sleeping[id - 1], 0, __ATOMIC_RELEASE);
      next = __atomic_load_n(&generation, __ATOMIC_ACQUIRE);
      if (next == seen) continue;
    }
    seen = next;
    const int active = current_active, total = current_total;
    if (id < active) current_work(current_opaque, id, active, total * id / active, total * (id + 1) / active);
    __atomic_store_n(&completed_generation[id - 1], seen, __ATOMIC_RELEASE);
  }
}
static void init_pool(void) {
  init_affinity();
  pin_index(0);
  for (int i = 0; i < thread_limit - 1; i++) {
    sem_init(ready[i], 0, 0);
    ids[i] = i + 1;
    if (pthread_create(&tids[i], (const void *)0, pool_worker, &ids[i])) break;
    pthread_detach(tids[i]);
    workers++;
  }
}
void tiny_cpu_parallel(work_t work, void *opaque, int total, int requested) {
  pthread_once(&once, init_pool);
  current_work = work;
  current_opaque = opaque;
  current_total = total;
  current_active = requested < workers + 1 ? requested : workers + 1;
  const unsigned job = __atomic_add_fetch(&generation, 1, __ATOMIC_RELEASE);
  for (int i = 0; i < workers; i++)
    if (__atomic_exchange_n(&sleeping[i], 0, __ATOMIC_ACQ_REL)) sem_post(ready[i]);
  work(opaque, 0, current_active, 0, total / current_active);
  for (int i = 0; i < workers; i++)
    while (__atomic_load_n(&completed_generation[i], __ATOMIC_ACQUIRE) != job) cpu_relax();
}
"""

def dot_source() -> str:
  if platform.machine().lower() not in ("x86_64", "amd64"): return _SCALAR_DOT
  return f"""
#if defined(__AVX2__)
#include <immintrin.h>
static inline int hsum8(__m256i x) {{
  __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
}}
static inline int hsum4(__m128i x) {{
  x = _mm_hadd_epi32(x, x);
  return _mm_cvtsi128_si32(_mm_hadd_epi32(x, x));
}}
static inline float hsum8f(__m256 x) {{
  __m128 sum = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}}
static inline int dot16(const signed char *a, const signed char *b) {{
  __m256i av = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)a));
  __m256i bv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)b));
  return hsum8(_mm256_madd_epi16(av, bv));
}}
static inline __m256i dot32_pairs(__m256i av, const signed char *b) {{
  const __m256i bv = _mm256_loadu_si256((const __m256i *)b);
  return _mm256_maddubs_epi16(_mm256_abs_epi8(av), _mm256_sign_epi8(bv, av));
}}
static inline __m256i dot32_parts(__m256i av, const signed char *b) {{
  return _mm256_madd_epi16(dot32_pairs(av, b), _mm256_set1_epi16(1));
}}
static inline int dot32v(__m256i av, const signed char *b) {{
  return hsum8(dot32_parts(av, b));
}}
static inline int dot32(const signed char *a, const signed char *b) {{
  return dot32v(_mm256_loadu_si256((const __m256i *)a), b);
}}
static inline __m256i unpack_q6(const unsigned char *w, int subgroup) {{
  const int lane = subgroup & 3, half = subgroup >> 2;
  __m256i low = _mm256_loadu_si256((const __m256i *)(w + half * 64 + (lane & 1) * 32));
  low = _mm256_and_si256(_mm256_srli_epi16(low, (lane >> 1) * 4), _mm256_set1_epi8(15));
  __m256i high = _mm256_loadu_si256((const __m256i *)(w + 128 + half * 32));
  high = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(high, lane * 2), _mm256_set1_epi8(3)), 4);
  return _mm256_sub_epi8(_mm256_or_si256(low, high), _mm256_set1_epi8(32));
}}
static inline void dot_q6(const unsigned char *w, int subgroup, const signed char *xq, int *lo, int *hi) {{
  const __m256i qvals = unpack_q6(w, subgroup);
  const __m128i qlo = _mm256_castsi256_si128(qvals), qhi = _mm256_extracti128_si256(qvals, 1);
  *lo = dot16((const signed char *)&qlo, xq); *hi = dot16((const signed char *)&qhi, xq + 16);
}}
static inline void dot_q6_vec(__m256i qvals, const signed char *xq, int *lo, int *hi) {{
  const __m256i xv = _mm256_loadu_si256((const __m256i *)xq);
  const __m256i pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(qvals), _mm256_sign_epi8(xv, qvals));
  __m256i sums = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
  sums = _mm256_hadd_epi32(sums, sums);
  sums = _mm256_hadd_epi32(sums, sums);
  *lo = _mm_cvtsi128_si32(_mm256_castsi256_si128(sums));
  *hi = _mm_cvtsi128_si32(_mm256_extracti128_si256(sums, 1));
}}
static inline void dot_q6_8(const unsigned char *w, const signed char *xq, int *lo, int *hi) {{
  const __m256i m15 = _mm256_set1_epi8(15), m3 = _mm256_set1_epi8(3), bias = _mm256_set1_epi8(32);
  for (int half = 0; half < 2; half++) {{
    const __m256i low0 = _mm256_loadu_si256((const __m256i *)(w + half * 64));
    const __m256i low1 = _mm256_loadu_si256((const __m256i *)(w + half * 64 + 32));
    const __m256i high = _mm256_loadu_si256((const __m256i *)(w + 128 + half * 32));
    const __m256i q0 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(low0, m15),
      _mm256_slli_epi16(_mm256_and_si256(high, m3), 4)), bias);
    const __m256i q1 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(low1, m15),
      _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(high, 2), m3), 4)), bias);
    const __m256i q2 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(low0, 4), m15),
      _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(high, 4), m3), 4)), bias);
    const __m256i q3 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(low1, 4), m15),
      _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(high, 6), m3), 4)), bias);
    dot_q6_vec(q0, xq + (half * 4 + 0) * 32, &lo[half * 4 + 0], &hi[half * 4 + 0]);
    dot_q6_vec(q1, xq + (half * 4 + 1) * 32, &lo[half * 4 + 1], &hi[half * 4 + 1]);
    dot_q6_vec(q2, xq + (half * 4 + 2) * 32, &lo[half * 4 + 2], &hi[half * 4 + 2]);
    dot_q6_vec(q3, xq + (half * 4 + 3) * 32, &lo[half * 4 + 3], &hi[half * 4 + 3]);
  }}
}}
#else
{_SCALAR_DOT}
static inline void dot_q6_8(const unsigned char *w, const signed char *xq, int *lo, int *hi) {{
  for (int subgroup = 0; subgroup < 8; subgroup++) dot_q6(w, subgroup, xq + subgroup * 32, &lo[subgroup], &hi[subgroup]);
}}
#endif
"""

def recurrent_decode_bucket(pos:int, max_context:int, device:str) -> int:
  short_decode_len = min(8192, max_context)
  # Fused CPU decode receives the full KV cache and applies start_pos itself, so one graph covers every position.
  # The short key is only a JIT cache identifier on CPU; it does not window or truncate attention.
  return short_decode_len if device.startswith("CPU") or pos < short_decode_len else max_context

@functools.cache
def parallel_runtime():
  from tinygrad.runtime.support.compiler_cpu import ClangCompiler
  from tinygrad.runtime.support.elf import elf_symbol_offsets, jit_loader
  pinning = r"""
extern int sched_getaffinity(int, unsigned long, void *);
extern int sched_setaffinity(int, unsigned long, const void *);
static unsigned long initial_affinity[16];
static int thread_limit = MAX_THREADS;
static void init_affinity(void) {
  for (int i = 0; i < 16; i++) initial_affinity[i] = 0;
  if (sched_getaffinity(0, sizeof(initial_affinity), initial_affinity)) return;
  int count = 0;
  for (int word = 0; word < 16; word++) for (int bit = 0; bit < 8 * (int)sizeof(unsigned long); bit++)
    count += (int)((initial_affinity[word] >> bit) & 1ul);
  if (count > 0 && count < thread_limit) thread_limit = count;
}
static void pin_index(int index) {
  int seen = 0;
  for (int word = 0; word < 16; word++) for (int bit = 0; bit < 8 * (int)sizeof(unsigned long); bit++)
    if ((initial_affinity[word] >> bit) & 1ul) {
      if (seen++ == index) {
        unsigned long selected[16];
        for (int i = 0; i < 16; i++) selected[i] = 0;
        selected[word] = 1ul << bit;
        sched_setaffinity(0, sizeof(selected), selected);
        return;
      }
    }
}
""" if platform.system() == "Linux" else \
    "static int thread_limit = MAX_THREADS;\nstatic void init_affinity(void) {}\nstatic void pin_index(int index) { (void)index; }\n"
  relax = "__builtin_ia32_pause();" if platform.machine().lower() in ("x86_64", "amd64") else \
          '__asm__ __volatile__("yield");' if platform.machine().lower() in ("aarch64", "arm64") else "(void)0;"
  src = _PARALLEL_SOURCE.replace("__PINNING__", pinning).replace("__CPU_RELAX__", relax).replace(
    "__SPIN_COUNT__", str(getenv("CPU_GGML_SPIN", 100000)))
  arch = {"amd64":"x86_64", "aarch64":"arm64"}.get(platform.machine().lower(), platform.machine().lower())
  obj = ClangCompiler([arch, "native"], cachekey="compile_cpu_parallel").compile_to_obj(src)
  link_libs = ["pthread", "c"]
  offset = elf_symbol_offsets(obj, link_libs=link_libs)["tiny_cpu_parallel"]
  runtime = Device["CPU"].runtime("tiny_cpu_parallel", jit_loader(obj, link_libs=link_libs), native=True)
  return runtime, runtime.addr + offset

@functools.cache
def _compile_cpu_ggml(src:str) -> bytes:
  from tinygrad.runtime.support.compiler_cpu import ClangCompiler
  from tinygrad.runtime.support.elf import jit_loader
  arch = {"amd64":"x86_64", "aarch64":"arm64"}.get(platform.machine().lower(), platform.machine().lower())
  obj = ClangCompiler([arch, "native"], cachekey="compile_cpu_ggml").compile_to_obj(src)
  link_libs = ["m", "pthread"] + (["mvec"] if "_ZGVdN8v_expf" in src else [])
  if "tiny_cpu_parallel" not in src: return jit_loader(obj, link_libs=link_libs)
  _, parallel = parallel_runtime()
  return jit_loader(obj, link_libs=link_libs, link_syms={"tiny_cpu_parallel": parallel})

@functools.cache
def attention_decode_program(batch:int, heads:int, kv_heads:int, head_dim:int, cache_len:int) -> tuple[str, bytes, str]:
  assert heads % kv_heads == 0
  name = f"cpu_attention_decode_{batch}_{heads}_{kv_heads}_{head_dim}_{cache_len}"
  threads = min(batch * heads, 32, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  vector_exp = platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64") and \
    heads // kv_heads == 8 and ctypes.util.find_library("mvec") is not None
  exp8_src = """
typedef float v8sf __attribute__((vector_size(32)));
extern v8sf _ZGVdN8v_expf(v8sf);
static inline __m256 exp8(__m256 x) { return (__m256)_ZGVdN8v_expf((v8sf)x); }
""" if vector_exp else ""
  src = f"""
extern float expf(float);
{dot_source()}
{exp8_src}
#define BATCH {batch}
#define HEADS {heads}
#define KV_HEADS {kv_heads}
#define HEAD_DIM {head_dim}
#define CACHE_LEN {cache_len}
#define THREADS {threads}
#define GROUP_HEADS (HEADS / KV_HEADS)
#define CHUNKS 16
#define MAX_CHUNK ((CACHE_LEN + CHUNKS - 1) / CHUNKS)
#define VECTOR_EXP {int(vector_exp)}
#define SCORE_BLOCK 64
typedef struct {{
  float *out; const float *q; const _Float16 *cache; int valid_len, stage, begin, end;
}} task_t;
static float partial_max[BATCH * HEADS * CHUNKS];
static float partial_denom[BATCH * HEADS * CHUNKS];
static float partial_acc[BATCH * HEADS * CHUNKS * HEAD_DIM];
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  if (task->stage == 0) for (int job = task->begin; job < task->end; job++) {{
    const int chunk = job % CHUNKS, bkv = job / CHUNKS;
    const int batch = bkv / KV_HEADS, kv_head = bkv - batch * KV_HEADS;
    const int token_begin = task->valid_len * chunk / CHUNKS;
    const int token_end = task->valid_len * (chunk + 1) / CHUNKS;
    const int count = token_end - token_begin;
    float scores[MAX_CHUNK][GROUP_HEADS], maxima[GROUP_HEADS];
    for (int gh = 0; gh < GROUP_HEADS; gh++) maxima[gh] = -__builtin_inff();
    for (int i = 0; i < count; i++) {{
      const _Float16 *key = task->cache + (((batch * KV_HEADS + kv_head) * CACHE_LEN + token_begin + i) * HEAD_DIM);
#if defined(__AVX2__)
      __m256 scorev[GROUP_HEADS];
      for (int gh = 0; gh < GROUP_HEADS; gh++) scorev[gh] = _mm256_setzero_ps();
      for (int dim = 0; dim < HEAD_DIM; dim += 8) {{
        const __m256 keyv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(key + dim)));
        for (int gh = 0; gh < GROUP_HEADS; gh++) {{
          const int head = kv_head * GROUP_HEADS + gh;
          const float *query = task->q + (batch * HEADS + head) * HEAD_DIM;
          scorev[gh] = _mm256_fmadd_ps(_mm256_loadu_ps(query + dim), keyv, scorev[gh]);
        }}
      }}
      for (int gh = 0; gh < GROUP_HEADS; gh++) {{
        const float score = hsum8f(scorev[gh]) * (1.0f / __builtin_sqrtf((float)HEAD_DIM));
        scores[i][gh] = score;
        if (score > maxima[gh]) maxima[gh] = score;
      }}
#else
      for (int gh = 0; gh < GROUP_HEADS; gh++) {{
        const int head = kv_head * GROUP_HEADS + gh;
        const float *query = task->q + (batch * HEADS + head) * HEAD_DIM;
        float score = 0.0f;
        for (int dim = 0; dim < HEAD_DIM; dim++) score += query[dim] * (float)key[dim];
        score *= 1.0f / __builtin_sqrtf((float)HEAD_DIM);
        scores[i][gh] = score;
        if (score > maxima[gh]) maxima[gh] = score;
      }}
#endif
    }}
#if VECTOR_EXP
    __m256 denomv = _mm256_setzero_ps(), maxv = _mm256_loadu_ps(maxima);
    for (int i = 0; i < count; i++) {{
      const __m256 weights = exp8(_mm256_sub_ps(_mm256_loadu_ps(scores[i]), maxv));
      _mm256_storeu_ps(scores[i], weights);
      denomv = _mm256_add_ps(denomv, weights);
    }}
    float denoms[GROUP_HEADS];
    _mm256_storeu_ps(denoms, denomv);
    for (int gh = 0; gh < GROUP_HEADS; gh++) {{
      const int head = kv_head * GROUP_HEADS + gh, bh = batch * HEADS + head;
      partial_max[bh * CHUNKS + chunk] = maxima[gh];
      partial_denom[bh * CHUNKS + chunk] = denoms[gh];
    }}
#else
    for (int gh = 0; gh < GROUP_HEADS; gh++) {{
      const int head = kv_head * GROUP_HEADS + gh, bh = batch * HEADS + head;
      float denom = 0.0f;
      for (int i = 0; i < count; i++) {{
        scores[i][gh] = expf(scores[i][gh] - maxima[gh]);
        denom += scores[i][gh];
      }}
      partial_max[bh * CHUNKS + chunk] = maxima[gh];
      partial_denom[bh * CHUNKS + chunk] = denom;
    }}
#endif
    for (int dim = 0; dim < HEAD_DIM; dim += 8) {{
#if defined(__AVX2__)
      __m256 acc[GROUP_HEADS];
      for (int gh = 0; gh < GROUP_HEADS; gh++) acc[gh] = _mm256_setzero_ps();
      for (int i = 0; i < count; i++) {{
        const _Float16 *value = task->cache +
          ((((BATCH + batch) * KV_HEADS + kv_head) * CACHE_LEN + token_begin + i) * HEAD_DIM + dim);
        const __m256 vv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)value));
        for (int gh = 0; gh < GROUP_HEADS; gh++)
          acc[gh] = _mm256_fmadd_ps(_mm256_set1_ps(scores[i][gh]), vv, acc[gh]);
      }}
      for (int gh = 0; gh < GROUP_HEADS; gh++) {{
        const int head = kv_head * GROUP_HEADS + gh, bh = batch * HEADS + head;
        _mm256_storeu_ps(partial_acc + (bh * CHUNKS + chunk) * HEAD_DIM + dim, acc[gh]);
      }}
#else
      for (int gh = 0; gh < GROUP_HEADS; gh++) {{
        const int head = kv_head * GROUP_HEADS + gh, bh = batch * HEADS + head;
        float *acc = partial_acc + (bh * CHUNKS + chunk) * HEAD_DIM + dim;
        for (int lane = 0; lane < 8; lane++) acc[lane] = 0.0f;
        for (int i = 0; i < count; i++) {{
          const _Float16 *value = task->cache +
            ((((BATCH + batch) * KV_HEADS + kv_head) * CACHE_LEN + token_begin + i) * HEAD_DIM + dim);
          for (int lane = 0; lane < 8; lane++) acc[lane] += scores[i][gh] * (float)value[lane];
        }}
      }}
#endif
    }}
  }}
  if (task->stage == 1) for (int bh = task->begin; bh < task->end; bh++) {{
    float maximum = -__builtin_inff(), denom = 0.0f;
    for (int chunk = 0; chunk < CHUNKS; chunk++)
      if (partial_max[bh * CHUNKS + chunk] > maximum) maximum = partial_max[bh * CHUNKS + chunk];
    for (int chunk = 0; chunk < CHUNKS; chunk++)
      denom += expf(partial_max[bh * CHUNKS + chunk] - maximum) * partial_denom[bh * CHUNKS + chunk];
    for (int dim = 0; dim < HEAD_DIM; dim += 8) {{
#if defined(__AVX2__)
      __m256 acc = _mm256_setzero_ps();
      for (int chunk = 0; chunk < CHUNKS; chunk++)
        acc = _mm256_fmadd_ps(_mm256_set1_ps(expf(partial_max[bh * CHUNKS + chunk] - maximum)),
          _mm256_loadu_ps(partial_acc + (bh * CHUNKS + chunk) * HEAD_DIM + dim), acc);
      _mm256_storeu_ps(task->out + bh * HEAD_DIM + dim, _mm256_div_ps(acc, _mm256_set1_ps(denom)));
#else
      for (int lane = 0; lane < 8; lane++) {{
        float acc = 0.0f;
        for (int chunk = 0; chunk < CHUNKS; chunk++)
          acc += expf(partial_max[bh * CHUNKS + chunk] - maximum) *
                 partial_acc[(bh * CHUNKS + chunk) * HEAD_DIM + dim + lane];
        task->out[bh * HEAD_DIM + dim + lane] = acc / denom;
      }}
#endif
    }}
  }}
  if (task->stage != -1) return (void *)0;
  for (int bh = task->begin; bh < task->end; bh++) {{
    const int batch = bh / HEADS, head = bh - batch * HEADS, kv_head = head / (HEADS / KV_HEADS);
    const float *query = task->q + bh * HEAD_DIM;
    float acc[HEAD_DIM], denom = 0.0f, max_score = -__builtin_inff();
    for (int dim = 0; dim < HEAD_DIM; dim++) acc[dim] = 0.0f;
    for (int base = 0; base < task->valid_len; base += SCORE_BLOCK) {{
      const int count = task->valid_len - base < SCORE_BLOCK ? task->valid_len - base : SCORE_BLOCK;
      float scores[SCORE_BLOCK], block_max = -__builtin_inff();
      for (int i = 0; i < count; i++) {{
        const _Float16 *key = task->cache + (((batch * KV_HEADS + kv_head) * CACHE_LEN + base + i) * HEAD_DIM);
        float score = 0.0f;
#if defined(__AVX2__)
        __m256 scorev = _mm256_setzero_ps();
        for (int dim = 0; dim < HEAD_DIM; dim += 8)
          scorev = _mm256_fmadd_ps(_mm256_loadu_ps(query + dim),
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(key + dim))), scorev);
        score = hsum8f(scorev);
#else
        for (int dim = 0; dim < HEAD_DIM; dim++) score += query[dim] * (float)key[dim];
#endif
        scores[i] = score * (1.0f / __builtin_sqrtf((float)HEAD_DIM));
        if (scores[i] > block_max) block_max = scores[i];
      }}
      const float next_max = max_score > block_max ? max_score : block_max;
      const float old_scale = max_score == -__builtin_inff() ? 0.0f : expf(max_score - next_max);
      denom *= old_scale;
      for (int dim = 0; dim < HEAD_DIM; dim += 8) {{
#if defined(__AVX2__)
        _mm256_storeu_ps(acc + dim, _mm256_mul_ps(_mm256_loadu_ps(acc + dim), _mm256_set1_ps(old_scale)));
#else
        for (int lane = 0; lane < 8; lane++) acc[dim + lane] *= old_scale;
#endif
      }}
      for (int i = 0; i < count; i++) {{
        const float weight = expf(scores[i] - next_max);
        const _Float16 *value = task->cache +
          ((((BATCH + batch) * KV_HEADS + kv_head) * CACHE_LEN + base + i) * HEAD_DIM);
        denom += weight;
        for (int dim = 0; dim < HEAD_DIM; dim += 8) {{
#if defined(__AVX2__)
          _mm256_storeu_ps(acc + dim, _mm256_fmadd_ps(_mm256_set1_ps(weight),
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(value + dim))), _mm256_loadu_ps(acc + dim)));
#else
          for (int lane = 0; lane < 8; lane++) acc[dim + lane] += weight * (float)value[dim + lane];
#endif
        }}
      }}
      max_score = next_max;
    }}
    for (int dim = 0; dim < HEAD_DIM; dim += 8) {{
#if defined(__AVX2__)
      _mm256_storeu_ps(task->out + bh * HEAD_DIM + dim,
        _mm256_div_ps(_mm256_loadu_ps(acc + dim), _mm256_set1_ps(denom)));
#else
      for (int lane = 0; lane < 8; lane++) task->out[bh * HEAD_DIM + dim + lane] = acc[dim + lane] / denom;
#endif
    }}
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}(float *out, const float *q, const _Float16 *cache, int start_pos) {{
  const int total = BATCH * HEADS, valid_len = start_pos + 1 < CACHE_LEN ? start_pos + 1 : CACHE_LEN;
  if (valid_len < 4096) {{
    const int requested = THREADS < 16 ? THREADS : 16;
    dispatch((task_t){{out, q, cache, valid_len, -1, 0, total}}, total, requested);
  }} else {{
    const int grouped_total = BATCH * KV_HEADS * CHUNKS;
    dispatch((task_t){{out, q, cache, valid_len, 0, 0, grouped_total}}, grouped_total, THREADS);
    dispatch((task_t){{out, q, cache, valid_len, 1, 0, total}}, total, THREADS);
  }}
}}
"""
  return src, _compile_cpu_ggml(src), name

def attention_decode(q:Tensor, cache:Tensor, start_pos:int|UOp) -> Tensor:
  batch, heads, query_len, head_dim = q.shape
  assert query_len == 1 and q.dtype == dtypes.float32 and cache.dtype == dtypes.float16
  kv_heads, cache_len = cache.shape[2], cache.shape[3]
  assert isinstance(cache_len, int) and head_dim % 8 == 0
  out = Tensor.empty(batch, heads, 1, head_dim, dtype=dtypes.float32, device=q.device)
  src, binary, name = attention_decode_program(batch, heads, kv_heads, head_dim, cache_len)
  pos = UOp.variable("cpu_attention_pos", 0, cache_len-1).bind(start_pos) if isinstance(start_pos, int) else start_pos
  pos_var = pos.unbind()[0]
  srcs = (out.uop, q.contiguous().uop, cache.uop)
  params = [UOp.placeholder_like(x, slot=i) for i,x in enumerate(srcs)]
  sink = UOp.sink(*params, pos_var, arg=KernelInfo(name=name))
  program = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
                arg=ProgramInfo(name=name, vars=(pos_var,), local_size=(1, 1, 1), globals=(0, 1, 2), outs=(0,), ins=(1, 2)))
  return Tensor(srcs[0].after(program.call(*srcs, pos)))

@functools.cache
def attention_prefill_program(batch:int, heads:int, tokens:int, kv_heads:int, head_dim:int,
                                   cache_len:int) -> tuple[str, bytes, str]:
  assert heads % kv_heads == 0
  name = f"cpu_attention_prefill_tiled_{batch}_{heads}_{tokens}_{kv_heads}_{head_dim}_{cache_len}"
  threads = min(batch * heads * ((tokens + 3) // 4), 32, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  src = f"""
extern float expf(float);
{dot_source()}
#define BATCH {batch}
#define HEADS {heads}
#define TOKENS {tokens}
#define KV_HEADS {kv_heads}
#define HEAD_DIM {head_dim}
#define CACHE_LEN {cache_len}
#define QUERY_TILE 4
#define QUERY_BLOCKS ((TOKENS + QUERY_TILE - 1) / QUERY_TILE)
#define SCORE_BLOCK 64
#define THREADS {threads}
typedef struct {{
  float *out; const float *q; const _Float16 *cache; unsigned *next_work; int start_pos, total_work, begin, end;
}} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  for (;;) {{
    const int job = (int)__atomic_fetch_add(task->next_work, 1, __ATOMIC_RELAXED);
    if (job >= task->total_work) break;
    const int query_block = job % QUERY_BLOCKS, bh = job / QUERY_BLOCKS;
    const int batch = bh / HEADS, head = bh - batch * HEADS, kv_head = head / (HEADS / KV_HEADS);
    const int query_base = query_block * QUERY_TILE;
    const int query_count = TOKENS - query_base < QUERY_TILE ? TOKENS - query_base : QUERY_TILE;
    float acc[QUERY_TILE][HEAD_DIM], denom[QUERY_TILE] = {{0}}, max_score[QUERY_TILE];
    for (int qi = 0; qi < query_count; qi++) {{
      max_score[qi] = -__builtin_inff();
      for (int dim = 0; dim < HEAD_DIM; dim++) acc[qi][dim] = 0.0f;
    }}
    const int max_valid = task->start_pos + query_base + query_count < CACHE_LEN ?
                          task->start_pos + query_base + query_count : CACHE_LEN;
    for (int base = 0; base < max_valid; base += SCORE_BLOCK) {{
      const int count = max_valid - base < SCORE_BLOCK ? max_valid - base : SCORE_BLOCK;
      float scores[QUERY_TILE][SCORE_BLOCK], block_max[QUERY_TILE];
      for (int qi = 0; qi < query_count; qi++) block_max[qi] = -__builtin_inff();
      for (int i = 0; i < count; i++) {{
        const int key_pos = base + i;
        const _Float16 *key = task->cache + (((batch * KV_HEADS + kv_head) * CACHE_LEN + key_pos) * HEAD_DIM);
        for (int qi = 0; qi < query_count; qi++) {{
          const int query_token = query_base + qi, valid = task->start_pos + query_token + 1;
          if (key_pos >= valid) continue;
          const float *query = task->q + ((batch * HEADS + head) * TOKENS + query_token) * HEAD_DIM;
          float score = 0.0f;
#if defined(__AVX2__)
          __m256 scorev = _mm256_setzero_ps();
          for (int dim = 0; dim < HEAD_DIM; dim += 8)
            scorev = _mm256_fmadd_ps(_mm256_loadu_ps(query + dim),
              _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(key + dim))), scorev);
          score = hsum8f(scorev);
#else
          for (int dim = 0; dim < HEAD_DIM; dim++) score += query[dim] * (float)key[dim];
#endif
          score *= 1.0f / __builtin_sqrtf((float)HEAD_DIM);
          scores[qi][i] = score;
          if (score > block_max[qi]) block_max[qi] = score;
        }}
      }}
      for (int qi = 0; qi < query_count; qi++) {{
        const float next_max = max_score[qi] > block_max[qi] ? max_score[qi] : block_max[qi];
        const float old_scale = max_score[qi] == -__builtin_inff() ? 0.0f : expf(max_score[qi] - next_max);
        denom[qi] *= old_scale;
        for (int dim = 0; dim < HEAD_DIM; dim++) acc[qi][dim] *= old_scale;
        const int valid = task->start_pos + query_base + qi + 1;
        for (int i = 0; i < count && base + i < valid; i++) {{
          scores[qi][i] = expf(scores[qi][i] - next_max);
          denom[qi] += scores[qi][i];
        }}
        max_score[qi] = next_max;
      }}
      for (int dim = 0; dim < HEAD_DIM; dim += 8) {{
#if defined(__AVX2__)
        __m256 av[QUERY_TILE];
        for (int qi = 0; qi < query_count; qi++) av[qi] = _mm256_loadu_ps(acc[qi] + dim);
        for (int i = 0; i < count; i++) {{
          const int key_pos = base + i;
          const _Float16 *value = task->cache +
            ((((BATCH + batch) * KV_HEADS + kv_head) * CACHE_LEN + key_pos) * HEAD_DIM + dim);
          const __m256 vv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)value));
          for (int qi = 0; qi < query_count; qi++)
            if (key_pos < task->start_pos + query_base + qi + 1)
              av[qi] = _mm256_fmadd_ps(_mm256_set1_ps(scores[qi][i]), vv, av[qi]);
        }}
        for (int qi = 0; qi < query_count; qi++) _mm256_storeu_ps(acc[qi] + dim, av[qi]);
#else
        for (int i = 0; i < count; i++) {{
          const int key_pos = base + i;
          const _Float16 *value = task->cache +
            ((((BATCH + batch) * KV_HEADS + kv_head) * CACHE_LEN + key_pos) * HEAD_DIM + dim);
          for (int qi = 0; qi < query_count; qi++)
            if (key_pos < task->start_pos + query_base + qi + 1)
              for (int lane = 0; lane < 8; lane++) acc[qi][dim + lane] += scores[qi][i] * (float)value[lane];
        }}
#endif
      }}
    }}
    for (int qi = 0; qi < query_count; qi++) {{
      float *out = task->out + ((batch * HEADS + head) * TOKENS + query_base + qi) * HEAD_DIM;
      for (int dim = 0; dim < HEAD_DIM; dim++) out[dim] = acc[qi][dim] / denom[qi];
    }}
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}(float *out, const float *q, const _Float16 *cache, int start_pos) {{
  const int total = BATCH * HEADS * QUERY_BLOCKS;
  unsigned next_work = 0;
  dispatch((task_t){{out, q, cache, &next_work, start_pos, total, 0, total}}, total, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def attention_prefill(q:Tensor, cache:Tensor, start_pos:int|UOp) -> Tensor:
  batch, heads, tokens, head_dim = q.shape
  assert tokens > 1 and q.dtype == dtypes.float32 and cache.dtype == dtypes.float16
  kv_heads, cache_len = cache.shape[2], cache.shape[3]
  assert isinstance(cache_len, int) and head_dim % 8 == 0
  out = Tensor.empty(batch, heads, tokens, head_dim, dtype=dtypes.float32, device=q.device)
  src, binary, name = attention_prefill_program(batch, heads, tokens, kv_heads, head_dim, cache_len)
  pos = UOp.variable("cpu_attention_prefill_pos", 0, cache_len-1).bind(start_pos) if isinstance(start_pos, int) else start_pos
  pos_var = pos.unbind()[0]
  srcs = (out.uop, q.contiguous().uop, cache.uop)
  params = [UOp.placeholder_like(x, slot=i) for i,x in enumerate(srcs)]
  sink = UOp.sink(*params, pos_var, arg=KernelInfo(name=name))
  program = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
                arg=ProgramInfo(name=name, vars=(pos_var,), local_size=(1, 1, 1), globals=(0, 1, 2), outs=(0,), ins=(1, 2)))
  return Tensor(srcs[0].after(program.call(*srcs, pos)))

@functools.cache
def gated_delta_program(batch:int, heads:int, dim:int, state_dtype:DType, normalize:bool=False,
                             norm_eps:float=0.0) -> tuple[str, bytes, str]:
  assert state_dtype in (dtypes.float16, dtypes.float32)
  state_ctype = "_Float16" if state_dtype == dtypes.float16 else "float"
  name = f"cpu_gated_delta{'_norm' if normalize else ''}_{batch}_{heads}_{dim}_{state_dtype.name}"
  threads = min(batch * heads, 32, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  src = f"""
{dot_source()}
#define BATCH {batch}
#define HEADS {heads}
#define DIM {dim}
#define THREADS {threads}
#define NORMALIZE {int(normalize)}
#define STATE_F16 {int(state_dtype == dtypes.float16)}
#if defined(__AVX2__)
static inline __m256 load_state(const {state_ctype} *p) {{
#if STATE_F16
  return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
#else
  return _mm256_loadu_ps(p);
#endif
}}
static inline void store_state({state_ctype} *p, __m256 value) {{
#if STATE_F16
  _mm_storeu_si128((__m128i *)p, _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#else
  _mm256_storeu_ps(p, value);
#endif
}}
#endif
typedef struct {{
  float *core; {state_ctype} *next_state; const float *q, *k, *v, *beta, *alpha; const {state_ctype} *state;
  const _Float16 *norm_weight; int begin, end;
}} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  for (int bh = task->begin; bh < task->end; bh++) {{
    const float *q = task->q + bh * DIM, *k = task->k + bh * DIM, *v = task->v + bh * DIM;
    const {state_ctype} *state = task->state + bh * DIM * DIM;
    float *core = task->core + bh * DIM;
    {state_ctype} *next_state = task->next_state + bh * DIM * DIM;
    const float alpha = task->alpha[bh], beta = task->beta[bh];
    float kq = 0.0f;
    for (int j = 0; j < DIM; j++) kq += k[j] * q[j];
    for (int i = 0; i < DIM; i++) {{
      const {state_ctype} *row = state + i * DIM;
#if defined(__AVX2__)
      __m256 state_k_vec = _mm256_setzero_ps(), state_q_vec = _mm256_setzero_ps();
      for (int j = 0; j < DIM; j += 8) {{
        const __m256 state_vec = load_state(row + j);
        state_k_vec = _mm256_fmadd_ps(state_vec, _mm256_loadu_ps(k + j), state_k_vec);
        state_q_vec = _mm256_fmadd_ps(state_vec, _mm256_loadu_ps(q + j), state_q_vec);
      }}
      const float state_k = hsum8f(state_k_vec), state_q = hsum8f(state_q_vec);
#else
      float state_k = 0.0f, state_q = 0.0f;
      for (int j = 0; j < DIM; j++) {{
        state_k += row[j] * k[j];
        state_q += row[j] * q[j];
      }}
#endif
      const float delta = (v[i] - state_k * alpha) * beta;
      core[i] = state_q * alpha + delta * kq;
#if defined(__AVX2__)
      const __m256 alpha_vec = _mm256_set1_ps(alpha), delta_vec = _mm256_set1_ps(delta);
      for (int j = 0; j < DIM; j += 8)
        store_state(next_state + i * DIM + j,
                    _mm256_fmadd_ps(delta_vec, _mm256_loadu_ps(k + j), _mm256_mul_ps(load_state(row + j), alpha_vec)));
#else
      for (int j = 0; j < DIM; j++) next_state[i * DIM + j] = row[j] * alpha + delta * k[j];
#endif
    }}
#if NORMALIZE
    float sum = 0.0f;
    for (int i = 0; i < DIM; i++) sum += core[i] * core[i];
    const float scale = 1.0f / __builtin_sqrtf(sum / (float)DIM + {norm_eps!r}f);
    for (int i = 0; i < DIM; i++) core[i] = core[i] * scale * (float)task->norm_weight[i];
#endif
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}(float *core, {state_ctype} *next_state, const float *q, const float *k, const float *v,
            const float *beta, const float *alpha, const {state_ctype} *state, const _Float16 *norm_weight) {{
  const int total = BATCH * HEADS;
  dispatch((task_t){{core, next_state, q, k, v, beta, alpha, state, norm_weight, 0, total}}, total, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def gated_delta(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor,
                     inplace:bool=False, norm_weight:Tensor|None=None, norm_eps:float=0.0) -> tuple[Tensor, Tensor]:
  batch, heads, dim = q.shape
  assert q.shape == k.shape == v.shape == (batch, heads, dim)
  assert beta.shape == alpha.shape == (batch, heads) and state.shape == (batch, heads, dim, dim)
  assert all(x.dtype == dtypes.float32 for x in (q, k, v, beta, alpha))
  assert state.dtype in (dtypes.float16, dtypes.float32)
  assert norm_weight is None or norm_weight.shape == (dim,) and norm_weight.dtype == dtypes.float16
  core, next_state = Tensor.empty_like(q), state if inplace else Tensor.empty_like(state)
  src, binary, name = gated_delta_program(batch, heads, dim, state.dtype, norm_weight is not None, norm_eps)
  norm_arg = state if norm_weight is None else norm_weight
  outputs = Tensor.custom_kernel(core, next_state, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(),
    alpha.contiguous(), state, norm_arg, fxn=lambda core,next_state,q,k,v,beta,alpha,state,norm_weight:
      gated_delta_kernel(core, next_state, q, k, v, beta, alpha, state, norm_weight, src, binary, name))
  return outputs[0], outputs[1]

def gated_delta_kernel(core:UOp, next_state:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, norm_weight:UOp,
                            src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(core.base, next_state.base, q.base, k.base, v.base, beta.base, alpha.base, state.base, norm_weight.base,
                  arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=tuple(range(9)), outs=(0, 1), ins=tuple(range(2, 9))))

@functools.cache
def gated_delta_prefill_program(batch:int, heads:int, tokens:int, dim:int, state_dtype:DType,
                                     norm_eps:float) -> tuple[str, bytes, str]:
  assert state_dtype in (dtypes.float16, dtypes.float32)
  state_ctype = "_Float16" if state_dtype == dtypes.float16 else "float"
  name = f"cpu_gated_delta_prefill_norm_{batch}_{heads}_{tokens}_{dim}_{state_dtype.name}"
  threads = min(batch * heads, 32, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  src = f"""
{dot_source()}
#define BATCH {batch}
#define HEADS {heads}
#define TOKENS {tokens}
#define DIM {dim}
#define THREADS {threads}
typedef struct {{
  float *core; {state_ctype} *next_state; const float *q, *k, *v, *beta, *alpha;
  const {state_ctype} *state; const _Float16 *norm_weight; int begin, end;
}} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  for (int bh = task->begin; bh < task->end; bh++) {{
    float current[DIM * DIM];
    const {state_ctype} *initial = task->state + bh * DIM * DIM;
    for (int i = 0; i < DIM * DIM; i++) current[i] = (float)initial[i];
    for (int token = 0; token < TOKENS; token++) {{
      const int offset = (bh * TOKENS + token) * DIM;
      const float *q = task->q + offset, *k = task->k + offset, *v = task->v + offset;
      float *core = task->core + offset;
      const float alpha = task->alpha[bh * TOKENS + token], beta = task->beta[bh * TOKENS + token];
      float kq = 0.0f;
      for (int j = 0; j < DIM; j++) kq += k[j] * q[j];
      for (int i = 0; i < DIM; i++) {{
        float *row = current + i * DIM;
#if defined(__AVX2__)
        __m256 state_k_vec = _mm256_setzero_ps(), state_q_vec = _mm256_setzero_ps();
        for (int j = 0; j < DIM; j += 8) {{
          const __m256 state_vec = _mm256_loadu_ps(row + j);
          state_k_vec = _mm256_fmadd_ps(state_vec, _mm256_loadu_ps(k + j), state_k_vec);
          state_q_vec = _mm256_fmadd_ps(state_vec, _mm256_loadu_ps(q + j), state_q_vec);
        }}
        const float state_k = hsum8f(state_k_vec), state_q = hsum8f(state_q_vec);
#else
        float state_k = 0.0f, state_q = 0.0f;
        for (int j = 0; j < DIM; j++) {{ state_k += row[j] * k[j]; state_q += row[j] * q[j]; }}
#endif
        const float delta = (v[i] - state_k * alpha) * beta;
        core[i] = state_q * alpha + delta * kq;
#if defined(__AVX2__)
        const __m256 alpha_vec = _mm256_set1_ps(alpha), delta_vec = _mm256_set1_ps(delta);
        for (int j = 0; j < DIM; j += 8)
          _mm256_storeu_ps(row + j, _mm256_fmadd_ps(delta_vec, _mm256_loadu_ps(k + j),
                                                    _mm256_mul_ps(_mm256_loadu_ps(row + j), alpha_vec)));
#else
        for (int j = 0; j < DIM; j++) row[j] = row[j] * alpha + delta * k[j];
#endif
      }}
      float sum = 0.0f;
      for (int i = 0; i < DIM; i++) sum += core[i] * core[i];
      const float scale = 1.0f / __builtin_sqrtf(sum / (float)DIM + {norm_eps!r}f);
      for (int i = 0; i < DIM; i++) core[i] = core[i] * scale * (float)task->norm_weight[i];
    }}
    {state_ctype} *next = task->next_state + bh * DIM * DIM;
    for (int i = 0; i < DIM * DIM; i++) next[i] = ({state_ctype})current[i];
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}(float *core, {state_ctype} *next_state, const float *q, const float *k, const float *v,
            const float *beta, const float *alpha, const {state_ctype} *state, const _Float16 *norm_weight) {{
  const int total = BATCH * HEADS;
  dispatch((task_t){{core, next_state, q, k, v, beta, alpha, state, norm_weight, 0, total}}, total, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor,
                             norm_weight:Tensor, norm_eps:float) -> tuple[Tensor, Tensor]:
  batch, heads, tokens, dim = q.shape
  assert q.shape == k.shape == v.shape == (batch, heads, tokens, dim)
  assert beta.shape == alpha.shape == (batch, heads, tokens) and state.shape == (batch, heads, dim, dim)
  assert all(x.dtype == dtypes.float32 for x in (q, k, v, beta, alpha))
  assert state.dtype in (dtypes.float16, dtypes.float32) and norm_weight.shape == (dim,) and norm_weight.dtype == dtypes.float16
  core, next_state = Tensor.empty_like(q), Tensor.empty_like(state)
  src, binary, name = gated_delta_prefill_program(batch, heads, tokens, dim, state.dtype, norm_eps)
  outputs = Tensor.custom_kernel(core, next_state, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(),
    alpha.contiguous(), state, norm_weight, fxn=lambda core,next_state,q,k,v,beta,alpha,state,norm_weight:
      gated_delta_kernel(core, next_state, q, k, v, beta, alpha, state, norm_weight, src, binary, name))
  return outputs[0], outputs[1]

@functools.cache
def rmsnorm_program(rows:int, dim:int, eps:float, x_dtype:DType, weight_dtype:DType) -> tuple[str, bytes, str]:
  assert x_dtype in (dtypes.float16, dtypes.float32) and weight_dtype in (dtypes.float16, dtypes.float32)
  x_ctype = "_Float16" if x_dtype == dtypes.float16 else "float"
  weight_ctype = "_Float16" if weight_dtype == dtypes.float16 else "float"
  out_dtype = dtypes.float16 if x_dtype == weight_dtype == dtypes.float16 else dtypes.float32
  out_ctype = "_Float16" if out_dtype == dtypes.float16 else "float"
  normalized = "_Float16 normalized = (_Float16)((float)x[i] * scale);" if x_dtype == dtypes.float16 else \
               "float normalized = x[i] * scale;"
  store = f"{normalized} out[i] = ({out_ctype})(normalized * task->weight[i]);"
  name = f"cpu_rmsnorm_{rows}_{dim}_{x_dtype.name}_{weight_dtype.name}_{str(eps).replace('.', '_').replace('-', 'm')}"
  threads = min(rows, 32 if rows > 32 else 8, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  src = f"""
#define ROWS {rows}
#define DIM {dim}
#define THREADS {threads}
typedef struct {{ {out_ctype} *out; const {x_ctype} *x; const {weight_ctype} *weight; int begin, end; }} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  for (int row = task->begin; row < task->end; row++) {{
    const {x_ctype} *x = task->x + row * DIM;
    {out_ctype} *out = task->out + row * DIM;
    float sum = 0.0f;
    for (int i = 0; i < DIM; i++) sum += (float)x[i] * (float)x[i];
    const float scale = 1.0f / __builtin_sqrtf(sum / (float)DIM + {eps!r}f);
    for (int i = 0; i < DIM; i++) {{ {store} }}
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}({out_ctype} *out, const {x_ctype} *x, const {weight_ctype} *weight) {{
  dispatch((task_t){{out, x, weight, 0, ROWS}}, ROWS, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def rmsnorm(norm:nn.RMSNorm, x:Tensor) -> Tensor:
  if not all(isinstance(s, int) for s in x.shape): return norm(x)
  rows, dim = int(x.numel()) // x.shape[-1], x.shape[-1]
  if not (SUPPORTED and str(x.device).startswith("CPU") and x.dtype in (dtypes.float16, dtypes.float32) and
          norm.weight is not None and norm.weight.dtype == dtypes.float16): return norm(x)
  out_dtype = dtypes.float16 if x.dtype == norm.weight.dtype == dtypes.float16 else dtypes.float32
  out = Tensor.empty(x.shape, dtype=out_dtype, device=x.device)
  src, binary, name = rmsnorm_program(rows, dim, norm.eps, x.dtype, norm.weight.dtype)
  return Tensor.custom_kernel(out, x.contiguous(), norm.weight, fxn=lambda out,x,weight:
    rmsnorm_kernel(out, x, weight, src, binary, name))[0]

def rmsnorm_kernel(out:UOp, x:UOp, weight:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out.base, x.base, weight.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=(0, 1, 2), outs=(0,), ins=(1, 2)))

@functools.cache
def shared_gate_program(rows:int, dim:int, x_dtype:DType, weight_dtype:DType) -> tuple[str, bytes, str]:
  assert x_dtype in (dtypes.float16, dtypes.float32) and weight_dtype in (dtypes.float16, dtypes.float32)
  x_ctype = "_Float16" if x_dtype == dtypes.float16 else "float"
  weight_ctype = "_Float16" if weight_dtype == dtypes.float16 else "float"
  out_dtype = dtypes.float16 if x_dtype == weight_dtype == dtypes.float16 else dtypes.float32
  out_ctype = "_Float16" if out_dtype == dtypes.float16 else "float"
  name = f"cpu_shared_gate_{rows}_{dim}_{x_dtype.name}_{weight_dtype.name}"
  src = f"""
extern float expf(float);
#define ROWS {rows}
#define DIM {dim}
void {name}({out_ctype} *out, const {x_ctype} *x, const {weight_ctype} *weight) {{
  for (int row = 0; row < ROWS; row++) {{
    float sum = 0.0f;
    for (int i = 0; i < DIM; i++) sum += (float)x[row * DIM + i] * (float)weight[i];
    out[row] = ({out_ctype})(1.0f / (1.0f + expf(-sum)));
  }}
}}
"""
  return src, _compile_cpu_ggml(src), name

def shared_gate(x:Tensor, weight:Tensor) -> Tensor:
  rows, dim = int(x.numel()) // x.shape[-1], x.shape[-1]
  if not (SUPPORTED and str(x.device).startswith("CPU") and rows <= 32 and
          x.dtype in (dtypes.float16, dtypes.float32) and weight.dtype in (dtypes.float16, dtypes.float32)):
    return (x * weight).sum(axis=-1, keepdim=True).sigmoid()
  out_dtype = dtypes.float16 if x.dtype == weight.dtype == dtypes.float16 else dtypes.float32
  out = Tensor.empty(*x.shape[:-1], 1, dtype=out_dtype, device=x.device)
  src, binary, name = shared_gate_program(rows, dim, x.dtype, weight.dtype)
  return Tensor.custom_kernel(out, x.contiguous(), weight, fxn=lambda out,x,weight:
    shared_gate_kernel(out, x, weight, src, binary, name))[0]

def shared_gate_kernel(out:UOp, x:UOp, weight:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out.base, x.base, weight.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=(0, 1, 2), outs=(0,), ins=(1, 2)))

def silu_mul(gate:Tensor, up:Tensor) -> Tensor:
  assert gate.shape == up.shape
  out = Tensor.empty(int(gate.numel()), dtype=(gate + up).dtype, device=gate.device)
  return Tensor.custom_kernel(out, gate.flatten().contiguous(), up.flatten().contiguous(), fxn=silu_mul_kernel)[0].reshape(*gate.shape)

def silu(x:Tensor) -> Tensor:
  out = Tensor.empty(int(x.numel()), dtype=x.dtype, device=x.device)
  return Tensor.custom_kernel(out, x.flatten().contiguous(), fxn=silu_kernel)[0].reshape(*x.shape)

@functools.cache
def causal_conv_silu_program(batch:int, tokens:int, channels:int, kernel_size:int, x_dtype:DType,
                                  weight_dtype:DType) -> tuple[str, bytes, str]:
  assert x_dtype in (dtypes.float16, dtypes.float32) and weight_dtype in (dtypes.float16, dtypes.float32)
  x_ctype = "_Float16" if x_dtype == dtypes.float16 else "float"
  weight_ctype = "_Float16" if weight_dtype == dtypes.float16 else "float"
  name = f"cpu_causal_conv_silu_{batch}_{tokens}_{channels}_{kernel_size}_{x_dtype.name}_{weight_dtype.name}"
  threads = min(batch * tokens * channels, 32, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  src = f"""
extern float expf(float);
#define TOKENS {tokens}
#define CHANNELS {channels}
#define KERNEL_SIZE {kernel_size}
#define THREADS {threads}
typedef struct {{ float *out; const {x_ctype} *window; const {weight_ctype} *weight; int begin, end; }} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  for (int idx = task->begin; idx < task->end; idx++) {{
    const int channel = idx % CHANNELS;
    const int token_batch = idx / CHANNELS, batch = token_batch / TOKENS, token = token_batch - batch * TOKENS;
    float sum = task->window[(batch * (TOKENS + KERNEL_SIZE - 1) + token) * CHANNELS + channel] *
                task->weight[channel * KERNEL_SIZE];
    for (int tap = 1; tap < KERNEL_SIZE; tap++)
      sum += task->window[(batch * (TOKENS + KERNEL_SIZE - 1) + token + tap) * CHANNELS + channel] *
             task->weight[channel * KERNEL_SIZE + tap];
    task->out[idx] = sum / (1.0f + expf(-sum));
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}(float *out, const {x_ctype} *window, const {weight_ctype} *weight) {{
  const int total = {batch} * TOKENS * CHANNELS;
  dispatch((task_t){{out, window, weight, 0, total}}, total, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def causal_conv_silu(window:Tensor, weight:Tensor, tokens:int) -> Tensor:
  assert len(window.shape) == 3 and weight.shape == (window.shape[2], window.shape[1] - tokens + 1)
  assert window.dtype in (dtypes.float16, dtypes.float32) and weight.dtype in (dtypes.float16, dtypes.float32)
  assert window.dtype == dtypes.float32 or weight.dtype == dtypes.float32
  batch, _, channels = window.shape
  out = Tensor.empty(batch, tokens, channels, dtype=dtypes.float32, device=window.device)
  src, binary, name = causal_conv_silu_program(batch, tokens, channels, weight.shape[1], window.dtype, weight.dtype)
  return Tensor.custom_kernel(out, window.contiguous(), weight.contiguous(), fxn=lambda out,window,weight:
    causal_conv_silu_kernel(out, window, weight, src, binary, name))[0]

def causal_conv_silu_kernel(out:UOp, window:UOp, weight:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out.base, window.base, weight.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=(0, 1, 2), outs=(0,), ins=(1, 2)))

@functools.cache
def gdn_qkv_program(batch:int, tokens:int, k_heads:int, v_heads:int, dim:int) -> tuple[str, bytes, str]:
  assert v_heads % k_heads == 0
  channels, q_dim = (2 * k_heads + v_heads) * dim, k_heads * dim
  name = f"cpu_gdn_qkv_{batch}_{tokens}_{k_heads}_{v_heads}_{dim}"
  threads = min(batch * v_heads * tokens, 32, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  src = f"""
#define TOKENS {tokens}
#define K_HEADS {k_heads}
#define V_HEADS {v_heads}
#define DIM {dim}
#define CHANNELS {channels}
#define Q_DIM {q_dim}
#define THREADS {threads}
typedef struct {{ float *q, *k, *v; const float *conv; int begin, end; }} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  for (int idx = task->begin; idx < task->end; idx++) {{
    const int token = idx % TOKENS, bh = idx / TOKENS, batch = bh / V_HEADS, head = bh - batch * V_HEADS;
    const int k_head = head % K_HEADS;
    const float *base = task->conv + (batch * TOKENS + token) * CHANNELS;
    const float *qin = base + k_head * DIM, *kin = base + Q_DIM + k_head * DIM;
    const float *vin = base + 2 * Q_DIM + head * DIM;
    float qsum = 0.0f, ksum = 0.0f;
    for (int i = 0; i < DIM; i++) {{ qsum += qin[i] * qin[i]; ksum += kin[i] * kin[i]; }}
    const float qscale = 1.0f / __builtin_sqrtf((qsum + 0.000001f) * (float)DIM);
    const float kscale = 1.0f / __builtin_sqrtf(ksum + 0.000001f);
    float *qout = task->q + idx * DIM, *kout = task->k + idx * DIM, *vout = task->v + idx * DIM;
    for (int i = 0; i < DIM; i++) {{
      qout[i] = qin[i] * qscale;
      kout[i] = kin[i] * kscale;
      vout[i] = vin[i];
    }}
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}(float *q, float *k, float *v, const float *conv) {{
  const int total = {batch} * V_HEADS * TOKENS;
  dispatch((task_t){{q, k, v, conv, 0, total}}, total, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def gdn_qkv(conv:Tensor, k_heads:int, v_heads:int, dim:int) -> tuple[Tensor, Tensor, Tensor]:
  batch, tokens, channels = conv.shape
  assert conv.dtype == dtypes.float32 and channels == (2 * k_heads + v_heads) * dim
  outputs = [Tensor.empty(batch, v_heads, tokens, dim, dtype=dtypes.float32, device=conv.device) for _ in range(3)]
  src, binary, name = gdn_qkv_program(batch, tokens, k_heads, v_heads, dim)
  ret = Tensor.custom_kernel(*outputs, conv.contiguous(), fxn=lambda q,k,v,conv:
    gdn_qkv_kernel(q, k, v, conv, src, binary, name))
  return ret[0], ret[1], ret[2]

def gdn_qkv_kernel(q:UOp, k:UOp, v:UOp, conv:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(q.base, k.base, v.base, conv.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=(0, 1, 2, 3), outs=(0, 1, 2), ins=(3,)))

def silu_mul_kernel(out:UOp, gate:UOp, up:UOp) -> UOp:
  elements = out.shape[0]
  idx = UOp.range(elements, 0, axis_type=AxisType.LOOP)
  value = gate[idx].load()
  return out[idx].store(value * value.sigmoid() * up[idx].load()).end(idx).sink(
    arg=KernelInfo(name=f"cpu_silu_mul_{elements}", opts_to_apply=()))

def silu_kernel(out:UOp, x:UOp) -> UOp:
  elements = out.shape[0]
  idx = UOp.range(elements, 0, axis_type=AxisType.LOOP)
  value = x[idx].load()
  return out[idx].store(value * value.sigmoid()).end(idx).sink(arg=KernelInfo(name=f"cpu_silu_{elements}", opts_to_apply=()))

@functools.cache
def biased_topk_program(outer:int, k:int, normalize:bool, x_dtype:DType, bias_dtype:DType) -> tuple[str, bytes, str]:
  assert x_dtype in (dtypes.float16, dtypes.float32) and bias_dtype in (dtypes.float16, dtypes.float32)
  x_ctype = "_Float16" if x_dtype == dtypes.float16 else "float"
  bias_ctype = "_Float16" if bias_dtype == dtypes.float16 else "float"
  score_ctype = "_Float16" if x_dtype == bias_dtype == dtypes.float16 else "float"
  normalize_src = f"""
    {x_ctype} denom = ({x_ctype})0.0f;
    for (int i = 0; i < K; i++) denom = ({x_ctype})(denom + top_prob[i]);
    for (int i = 0; i < K; i++) top_prob[i] = ({x_ctype})(top_prob[i] / denom);""" if normalize else ""
  name = f"cpu_biased_topk_{outer}_{k}_{int(normalize)}_{x_dtype.name}_{bias_dtype.name}"
  src = f"""
extern float expf(float);
#define OUTER {outer}
#define K {k}
void {name}({x_ctype} *out, int *sel, const {x_ctype} *x, const {bias_ctype} *bias) {{
  for (int row = 0; row < OUTER; row++) {{
    float top_score[K];
    {x_ctype} top_prob[K];
    int top_idx[K], count = 0;
    for (int index = 0; index < 256; index++) {{
      const {x_ctype} prob = ({x_ctype})(1.0f / (1.0f + expf(-(float)x[row * 256 + index])));
      const float score = (float)({score_ctype})((float)prob + (float)bias[index]);
      int pos = 0;
      while (pos < count && (top_score[pos] > score || (top_score[pos] == score && top_idx[pos] < index))) pos++;
      if (pos >= K) continue;
      const int end = count < K ? count++ : K - 1;
      for (int i = end; i > pos; i--) {{
        top_score[i] = top_score[i-1]; top_prob[i] = top_prob[i-1]; top_idx[i] = top_idx[i-1];
      }}
      top_score[pos] = score; top_prob[pos] = prob; top_idx[pos] = index;
    }}
{normalize_src}
    for (int i = 0; i < K; i++) {{
      out[row * K + i] = top_prob[K - 1 - i];
      sel[row * K + i] = top_idx[K - 1 - i];
    }}
  }}
}}
"""
  return src, _compile_cpu_ggml(src), name

def biased_topk(x:Tensor, bias:Tensor, k:int, normalize:bool) -> tuple[Tensor, Tensor]:
  outer = int(x.numel()) // 256
  values = Tensor.empty(outer, k, dtype=x.dtype, device=x.device)
  indices = Tensor.empty(outer, k, dtype=dtypes.int32, device=x.device)
  src, binary, name = biased_topk_program(outer, k, normalize, x.dtype, bias.dtype)
  outputs = Tensor.custom_kernel(values, indices, x.reshape(outer, 256).contiguous(), bias.contiguous(),
    fxn=lambda out,sel,x,bias:biased_topk_kernel(out, sel, x, bias, src, binary, name))
  return outputs[0], outputs[1]

def biased_topk_kernel(out:UOp, sel:UOp, x:UOp, bias:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out.base, sel.base, x.base, bias.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=(0, 1, 2, 3), outs=(0, 1), ins=(2, 3)))

@functools.cache
def topk_softmax_program(outer:int, k:int, dtype:DType) -> tuple[str, bytes, str]:
  assert dtype in (dtypes.float16, dtypes.float32)
  ctype = "_Float16" if dtype == dtypes.float16 else "float"
  name = f"cpu_topk_softmax_{outer}_{k}_{dtype.name}"
  src = f"""
extern float expf(float);
#define OUTER {outer}
#define K {k}
void {name}({ctype} *out, int *sel, const {ctype} *x) {{
  for (int row = 0; row < OUTER; row++) {{
    float top[K], probs[K];
    int top_idx[K], count = 0;
    for (int index = 0; index < 256; index++) {{
      const float value = (float)x[row * 256 + index];
      int pos = 0;
      while (pos < count && (top[pos] > value || (top[pos] == value && top_idx[pos] < index))) pos++;
      if (pos >= K) continue;
      const int end = count < K ? count++ : K - 1;
      for (int i = end; i > pos; i--) {{ top[i] = top[i-1]; top_idx[i] = top_idx[i-1]; }}
      top[pos] = value;
      top_idx[pos] = index;
    }}
    float denom = 0.0f;
    for (int i = 0; i < K; i++) {{ probs[i] = expf(top[i] - top[0]); denom += probs[i]; }}
    for (int i = 0; i < K; i++) {{
      out[row * K + i] = ({ctype})(probs[K - 1 - i] / denom);
      sel[row * K + i] = top_idx[K - 1 - i];
    }}
  }}
}}
"""
  return src, _compile_cpu_ggml(src), name

def topk_softmax(x:Tensor, k:int) -> tuple[Tensor, Tensor]:
  outer = int(x.numel()) // 256
  values = Tensor.empty(outer, k, dtype=x.dtype, device=x.device)
  indices = Tensor.empty(outer, k, dtype=dtypes.int32, device=x.device)
  src, binary, name = topk_softmax_program(outer, k, x.dtype)
  outputs = Tensor.custom_kernel(values, indices, x.reshape(outer, 256).contiguous(),
    fxn=lambda out,sel,x:topk_softmax_kernel(out, sel, x, src, binary, name))
  shape = (*x.shape[:-1], k)
  return outputs[0].reshape(*shape), outputs[1].reshape(*shape)

def topk_softmax_kernel(out:UOp, sel:UOp, x:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out.base, sel.base, x.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=(0, 1, 2), outs=(0, 1), ins=(2,)))

@functools.cache
def ggml_linear_program(ggml_type:int, tokens:int, out_features:int, in_features:int, dtype:DType) -> tuple[str, bytes, str]:
  assert ggml_type in (8, 14) and dtype in (dtypes.float16, dtypes.float32)
  token_tile = min(tokens, max(1, getenv("CPU_Q8_TOKEN_TILE", 8))) if ggml_type == 8 and tokens > 1 else tokens
  if tokens % token_tile: token_tile = 1
  name = f"cpu_ggml_linear_{ggml_type}_{tokens}_{out_features}_{in_features}_{dtype.name}" + \
         (f"_tt{token_tile}" if ggml_type == 8 and tokens > 1 else "")
  ctype = "_Float16" if dtype == dtypes.float16 else "float"
  weight_bytes = out_features * in_features * (34 / 32 if ggml_type == 8 else 210 / 256)
  thread_cap = 32 if tokens > 1 else 8 if weight_bytes <= 2 << 20 else 20 if weight_bytes <= 12 << 20 else 24
  threads = min(tokens * out_features, thread_cap, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  if ggml_type == 8:
    dot = """
    const unsigned char *w = raw + row * (IN_FEATURES / 32) * 34;
#if defined(__AVX2__)
    __m256 accv = _mm256_setzero_ps();
    for (int block = 0; block < IN_FEATURES / 32; block++, w += 34)
      accv = _mm256_fmadd_ps(_mm256_set1_ps(half_to_float(w) * xd[block]),
                            _mm256_cvtepi32_ps(dot32_parts(_mm256_loadu_si256((const __m256i *)(w + 2)),
                                                          xq + block * 32)), accv);
    acc = hsum8f(accv);
#else
    for (int block = 0; block < IN_FEATURES / 32; block++, w += 34) {
      acc += half_to_float(w) * xd[block] * (float)dot32((const signed char *)(w + 2), xq + block * 32);
    }
#endif"""
    batch_dot = f"""
    const unsigned char *w = task->raw + row * (IN_FEATURES / 32) * 34;
    for (int token_base = 0; token_base < TOKENS; token_base += {token_tile}) {{
#if defined(__AVX2__)
    __m256 accv[{token_tile}];
    for (int token = 0; token < {token_tile}; token++) accv[token] = _mm256_setzero_ps();
    const unsigned char *wb = w;
    for (int block = 0; block < IN_FEATURES / 32; block++, wb += 34) {{
      const __m256i wv = _mm256_loadu_si256((const __m256i *)(wb + 2));
      const float wd = half_to_float(wb);
      for (int token = 0; token < {token_tile}; token++) {{
        const int ti = token_base + token;
        accv[token] = _mm256_fmadd_ps(
          _mm256_set1_ps(wd * task->xd[block * TOKENS + ti]),
          _mm256_cvtepi32_ps(dot32_parts(wv, task->xq + (block * TOKENS + ti) * 32)), accv[token]);
      }}
    }}
    for (int token = 0; token < {token_tile}; token++) acc[token_base + token] = hsum8f(accv[token]);
#else
    const unsigned char *wb = w;
    for (int block = 0; block < IN_FEATURES / 32; block++, wb += 34) {{
      const float wd = half_to_float(wb);
      for (int token = 0; token < {token_tile}; token++) {{
        const int ti = token_base + token;
        acc[ti] += wd * task->xd[block * TOKENS + ti] *
          (float)dot32((const signed char *)(wb + 2), task->xq + (block * TOKENS + ti) * 32);
      }}
    }}
#endif
    }}"""
  else:
    dot = """
    const unsigned char *w = raw + row * (IN_FEATURES / 256) * 210;
    for (int block = 0; block < IN_FEATURES / 256; block++, w += 210) {
      const float d = half_to_float(w + 208);
      for (int subgroup = 0; subgroup < 8; subgroup++) {
        const int group = block * 8 + subgroup;
        int dot_lo, dot_hi;
        dot_q6(w, subgroup, xq + group * 32, &dot_lo, &dot_hi);
        const int scale0 = ((const signed char *)(w + 192))[subgroup * 2];
        const int scale1 = ((const signed char *)(w + 192))[subgroup * 2 + 1];
        acc += d * (float)scale0 * xd[group] * (float)(dot_lo + dot_hi);
        acc += d * (float)(scale1 - scale0) * xd[group] * (float)dot_hi;
      }
    }"""
    batch_dot = """
    const unsigned char *w = task->raw + row * (IN_FEATURES / 256) * 210;
    for (int block = 0; block < IN_FEATURES / 256; block++, w += 210) {
      const float d = half_to_float(w + 208);
      for (int subgroup = 0; subgroup < 8; subgroup++) {
        const int group = block * 8 + subgroup;
        for (int token = 0; token < TOKENS; token++) {
          const signed char *xq = task->xq + (group * TOKENS + token) * 32;
          const float xd = task->xd[group * TOKENS + token];
          int dot_lo, dot_hi;
          dot_q6(w, subgroup, xq, &dot_lo, &dot_hi);
          const int scale0 = ((const signed char *)(w + 192))[subgroup * 2];
          const int scale1 = ((const signed char *)(w + 192))[subgroup * 2 + 1];
          acc[token] += d * (float)scale0 * xd * (float)(dot_lo + dot_hi);
          acc[token] += d * (float)(scale1 - scale0) * xd * (float)dot_hi;
        }
      }
    }"""
  worker = f"""
  for (int row = task->begin; row < task->end; row++) {{
    float acc[TOKENS] = {{0}};
{batch_dot}
    for (int token = 0; token < TOKENS; token++) task->out[token * OUT_FEATURES + row] = ({ctype})acc[token];
  }}""" if tokens > 1 else f"""
  for (int row = task->begin; row < task->end; row++) {{
    const signed char *xq = task->xq;
    const float *xd = task->xd;
    const unsigned char *raw = task->raw;
    float acc = 0.0f;
{dot}
    task->out[row] = ({ctype})acc;
  }}"""
  dot_helpers = dot_source()
  src = f"""
typedef union {{ unsigned short u; _Float16 h; }} half_bits;
static inline float half_to_float(const unsigned char *p) {{
  half_bits v; v.u = (unsigned short)p[0] | ((unsigned short)p[1] << 8); return (float)v.h;
}}
{dot_helpers}
#define TOKENS {tokens}
#define OUT_FEATURES {out_features}
#define IN_FEATURES {in_features}
#define THREADS {threads}
typedef struct {{ {ctype} *out; const unsigned char *raw; const signed char *xq; const float *xd; int begin, end; }} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
{worker}
  return (void *)0;
}}
{_SEM_POOL}
void {name}({ctype} *out, const unsigned char *raw, const {ctype} *x) {{
  const int total = OUT_FEATURES;
  signed char xq[TOKENS * IN_FEATURES];
  float xd[TOKENS * (IN_FEATURES / 32)];
  for (int token = 0; token < TOKENS; token++) for (int group = 0; group < IN_FEATURES / 32; group++) {{
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {{
      float v = (float)x[token * IN_FEATURES + group * 32 + i];
      if (v < 0.0f) v = -v;
      if (v > amax) amax = v;
    }}
    const float d = amax > 0.00000001f ? amax / 127.0f : 0.00000001f;
    xd[group * TOKENS + token] = d;
    for (int i = 0; i < 32; i++) {{
      int q = (int)__builtin_roundf((float)x[token * IN_FEATURES + group * 32 + i] / d);
      if (q < -127) q = -127;
      if (q > 127) q = 127;
      xq[(group * TOKENS + token) * 32 + i] = (signed char)q;
    }}
  }}
  dispatch((task_t){{out, raw, xq, xd, 0, total}}, total, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def ggml_linear_kernel(out:UOp, raw:UOp, x:UOp, ggml_type:int) -> UOp:
  tokens, out_features, in_features = out.shape[0], out.shape[1], x.shape[1]
  src, binary, name = ggml_linear_program(ggml_type, tokens, out_features, in_features, x.dtype)
  sink = UOp.sink(out.base, raw.base, x.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=(0, 1, 2), outs=(0,), ins=(1, 2)))

@functools.cache
def q8_silu_linear_program(tokens:int, out_features:int, in_features:int) -> tuple[str, bytes, str]:
  token_tile = 8 if tokens % 8 == 0 else 1
  name = f"cpu_q8_silu_linear_{tokens}_{out_features}_{in_features}_tt{token_tile}"
  threads = min(out_features, 32, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  src = f"""
typedef union {{ unsigned short u; _Float16 h; }} half_bits;
static inline float half_to_float(const unsigned char *p) {{
  half_bits v; v.u = (unsigned short)p[0] | ((unsigned short)p[1] << 8); return (float)v.h;
}}
{dot_source()}
static inline float tiny_silu_mul(_Float16 gate, float up) {{
  const _Float16 x = gate * (_Float16)-1.4426950408889634f;
  const int is_nan = x != x;
  const _Float16 finite = x == (_Float16)-__builtin_inff() || x == (_Float16)__builtin_inff() || is_nan ? (_Float16)0.0f : x;
  const short rounded = (short)(finite + (finite < (_Float16)0.0f ? (_Float16)-0.5f : (_Float16)0.5f));
  const int negative = rounded < 0;
  const short half = (short)(((rounded + negative) >> 1) - (((rounded % 2) != 0) & negative));
  const _Float16 s = finite - (_Float16)rounded;
  const float poly = (((((((float)(_Float16)0.0001535920892f*s + (_Float16)0.001339262701f)*s +
    (_Float16)0.009618384764f)*s + (_Float16)0.05550347269f)*s + (_Float16)0.2402264476f)*s +
    (_Float16)0.6931471825f)*s + (_Float16)1.0f);
  half_bits hi = {{.u=(unsigned short)((half + 15) << 10)}};
  half_bits lo = {{.u=(unsigned short)(((rounded - half) + 15) << 10)}};
  _Float16 exp = !(x < (_Float16)23.0f) ? (_Float16)__builtin_inff() :
    x < (_Float16)-22.0f ? (_Float16)0.0f : (_Float16)(poly * (float)hi.h * (float)lo.h);
  if (is_nan) exp = (_Float16)__builtin_nanf("");
  return ((float)gate / (1.0f + (float)exp)) * up;
}}
#define TOKENS {tokens}
#define TOKEN_TILE {token_tile}
#define OUT_FEATURES {out_features}
#define IN_FEATURES {in_features}
#define GROUPS (IN_FEATURES / 32)
#define THREADS {threads}
typedef struct {{
  _Float16 *out; const unsigned char *raw; const _Float16 *gate; const float *up;
  signed char *xq; float *xd; int stage, begin, end;
}} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  if (task->stage == 0) for (int idx = task->begin; idx < task->end; idx++) {{
    const int token = idx / GROUPS, group = idx - token * GROUPS;
    _Float16 values[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {{
      const int pos = token * IN_FEATURES + group * 32 + i;
      const float value = (float)(values[i] = (_Float16)tiny_silu_mul(task->gate[pos], task->up[pos]));
      const float abs_value = value < 0.0f ? -value : value;
      if (abs_value > amax) amax = abs_value;
    }}
    const float d = amax > 0.00000001f ? amax / 127.0f : 0.00000001f;
    task->xd[group * TOKENS + token] = d;
    for (int i = 0; i < 32; i++) {{
      int value = (int)__builtin_roundf((float)values[i] / d);
      if (value < -127) value = -127;
      if (value > 127) value = 127;
      task->xq[(group * TOKENS + token) * 32 + i] = (signed char)value;
    }}
  }}
  if (task->stage == 1) for (int row = task->begin; row < task->end; row++) {{
    const unsigned char *w = task->raw + row * GROUPS * 34;
    for (int token_base = 0; token_base < TOKENS; token_base += TOKEN_TILE) {{
#if defined(__AVX2__)
      __m256 accv[TOKEN_TILE];
      for (int token = 0; token < TOKEN_TILE; token++) accv[token] = _mm256_setzero_ps();
      const unsigned char *wb = w;
      for (int block = 0; block < GROUPS; block++, wb += 34) {{
        const __m256i wv = _mm256_loadu_si256((const __m256i *)(wb + 2));
        const float wd = half_to_float(wb);
        for (int token = 0; token < TOKEN_TILE; token++) {{
          const int ti = token_base + token;
          accv[token] = _mm256_fmadd_ps(_mm256_set1_ps(wd * task->xd[block * TOKENS + ti]),
            _mm256_cvtepi32_ps(dot32_parts(wv, task->xq + (block * TOKENS + ti) * 32)), accv[token]);
        }}
      }}
      for (int token = 0; token < TOKEN_TILE; token++)
        task->out[(token_base + token) * OUT_FEATURES + row] = (_Float16)hsum8f(accv[token]);
#else
      for (int token = 0; token < TOKEN_TILE; token++) {{
        const int ti = token_base + token;
        float acc = 0.0f;
        const unsigned char *wb = w;
        for (int block = 0; block < GROUPS; block++, wb += 34)
          acc += half_to_float(wb) * task->xd[block * TOKENS + ti] *
            (float)dot32((const signed char *)(wb + 2), task->xq + (block * TOKENS + ti) * 32);
        task->out[ti * OUT_FEATURES + row] = (_Float16)acc;
      }}
#endif
    }}
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}(_Float16 *out, const unsigned char *raw, const _Float16 *gate, const float *up) {{
  signed char xq[TOKENS * IN_FEATURES];
  float xd[TOKENS * GROUPS];
  dispatch((task_t){{out, raw, gate, up, xq, xd, 0, 0, TOKENS * GROUPS}}, TOKENS * GROUPS, THREADS);
  dispatch((task_t){{out, raw, gate, up, xq, xd, 1, 0, OUT_FEATURES}}, OUT_FEATURES, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def q8_silu_linear(layer:Linear, gate:Tensor, up:Tensor) -> Tensor:
  assert layer.ggml_type == 8 and layer.bias is None and gate.shape == up.shape and gate.shape[-1] == layer.in_features
  assert gate.dtype == dtypes.float16 and up.dtype == dtypes.float32
  tokens = int(gate.numel()) // layer.in_features
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float16, device=gate.device)
  src, binary, name = q8_silu_linear_program(tokens, layer.out_features, layer.in_features)
  out = Tensor.custom_kernel(out, layer.weight, gate.reshape(tokens, layer.in_features).contiguous(),
    up.reshape(tokens, layer.in_features).contiguous(), fxn=lambda out,raw,gate,up:
      q8_silu_linear_kernel(out, raw, gate, up, src, binary, name))[0]
  return out.reshape(*gate.shape[:-1], layer.out_features)

def q8_silu_linear_kernel(out:UOp, raw:UOp, gate:UOp, up:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out.base, raw.base, gate.base, up.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=(0, 1, 2, 3), outs=(0,), ins=(1, 2, 3)))

@functools.cache
def f16_linear_program(tokens:int, out_features:int, in_features:int, x_dtype:DType) -> tuple[str, bytes, str]:
  assert x_dtype in (dtypes.float16, dtypes.float32)
  ctype = "_Float16" if x_dtype == dtypes.float16 else "float"
  name = f"cpu_f16_linear_{tokens}_{out_features}_{in_features}_{x_dtype.name}"
  threads = min(out_features, 32 if tokens > 1 else 16, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  src = f"""
{dot_source()}
#define OUT_FEATURES {out_features}
#define IN_FEATURES {in_features}
#define TOKENS {tokens}
#define THREADS {threads}
typedef struct {{ {ctype} *out; const {ctype} *x; const _Float16 *weight; int begin, end; }} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  for (int row = task->begin; row < task->end; row++) {{
    const _Float16 *weight = task->weight + row * IN_FEATURES;
#if defined(__AVX2__)
    for (int token_base = 0; token_base < TOKENS; token_base += 8) {{
      const int token_count = TOKENS - token_base < 8 ? TOKENS - token_base : 8;
      __m256 accv[8];
      for (int token = 0; token < token_count; token++) accv[token] = _mm256_setzero_ps();
      for (int i = 0; i < IN_FEATURES; i += 8) {{
        const __m256 w = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(weight + i)));
        for (int token = 0; token < token_count; token++) {{
          const {ctype} *x = task->x + (token_base + token) * IN_FEATURES;
          accv[token] = _mm256_fmadd_ps(
            {"_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i)))" if x_dtype == dtypes.float16 else "_mm256_loadu_ps(x + i)"},
            w, accv[token]);
        }}
      }}
      for (int token = 0; token < token_count; token++)
        task->out[(token_base + token) * OUT_FEATURES + row] = ({ctype})hsum8f(accv[token]);
    }}
#else
    float acc[TOKENS] = {{0}};
    for (int i = 0; i < IN_FEATURES; i++)
      for (int token = 0; token < TOKENS; token++)
        acc[token] += (float)task->x[token * IN_FEATURES + i] * (float)weight[i];
    for (int token = 0; token < TOKENS; token++) task->out[token * OUT_FEATURES + row] = ({ctype})acc[token];
#endif
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}({ctype} *out, const {ctype} *x, const _Float16 *weight) {{
  dispatch((task_t){{out, x, weight, 0, OUT_FEATURES}}, OUT_FEATURES, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def f16_matvec(x:Tensor, weight:Tensor) -> Tensor:
  assert weight.dtype == dtypes.float16 and len(weight.shape) == 2 and int(x.numel()) % weight.shape[1] == 0
  assert x.dtype in (dtypes.float16, dtypes.float32)
  tokens = int(x.numel()) // weight.shape[1]
  out = Tensor.empty(tokens, weight.shape[0], dtype=x.dtype, device=x.device)
  src, binary, name = f16_linear_program(tokens, weight.shape[0], weight.shape[1], x.dtype)
  return Tensor.custom_kernel(out, x.reshape(tokens, weight.shape[1]).contiguous(), weight, fxn=lambda out,x,weight:
    f16_linear_kernel(out, x, weight, src, binary, name))[0].reshape(*x.shape[:-1], weight.shape[0])

def f16_linear(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type is None and layer.bias is None and layer.weight.dtype == dtypes.float16
  assert int(x.numel()) % layer.in_features == 0
  return f16_matvec(x, layer.weight)

def f16_linear_kernel(out:UOp, x:UOp, weight:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out.base, x.base, weight.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=(0, 1, 2), outs=(0,), ins=(1, 2)))

@functools.cache
def q8_batched_pair_program(tokens:int, out0:int, out1:int, in_features:int, dtype:DType) -> tuple[str, bytes, str]:
  assert dtype in (dtypes.float16, dtypes.float32)
  ctype = "_Float16" if dtype == dtypes.float16 else "float"
  name = f"cpu_q8_batched_pair_{tokens}_{out0}_{out1}_{in_features}_{dtype.name}"
  threads = min(out0 + out1, 32, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  src = f"""
typedef union {{ unsigned short u; _Float16 h; }} half_bits;
static inline float half_to_float(const unsigned char *p) {{
  half_bits v; v.u = (unsigned short)p[0] | ((unsigned short)p[1] << 8); return (float)v.h;
}}
{dot_source()}
#define TOKENS {tokens}
#define OUT0 {out0}
#define OUT1 {out1}
#define IN_FEATURES {in_features}
#define GROUPS (IN_FEATURES / 32)
#define THREADS {threads}
typedef struct {{
  {ctype} *out0, *out1; const unsigned char *raw0, *raw1; const signed char *xq; const float *xd; int begin, end;
}} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  for (int idx = task->begin; idx < task->end; idx++) {{
    const int projection = idx >= OUT0, row = projection ? idx - OUT0 : idx;
    const unsigned char *w = (projection ? task->raw1 : task->raw0) + row * GROUPS * 34;
    float acc[TOKENS] = {{0}};
#if defined(__AVX2__)
    __m256 accv[TOKENS];
    for (int token = 0; token < TOKENS; token++) accv[token] = _mm256_setzero_ps();
    for (int block = 0; block < GROUPS; block++, w += 34) {{
      const __m256i wv = _mm256_loadu_si256((const __m256i *)(w + 2));
      const float wd = half_to_float(w);
      for (int token = 0; token < TOKENS; token++)
        accv[token] = _mm256_fmadd_ps(_mm256_set1_ps(wd * task->xd[block * TOKENS + token]),
          _mm256_cvtepi32_ps(dot32_parts(wv, task->xq + (block * TOKENS + token) * 32)), accv[token]);
    }}
    for (int token = 0; token < TOKENS; token++) acc[token] = hsum8f(accv[token]);
#else
    for (int block = 0; block < GROUPS; block++, w += 34) {{
      const float wd = half_to_float(w);
      for (int token = 0; token < TOKENS; token++)
        acc[token] += wd * task->xd[block * TOKENS + token] *
          (float)dot32((const signed char *)(w + 2), task->xq + (block * TOKENS + token) * 32);
    }}
#endif
    {ctype} *out = projection ? task->out1 : task->out0;
    const int out_features = projection ? OUT1 : OUT0;
    for (int token = 0; token < TOKENS; token++) out[token * out_features + row] = ({ctype})acc[token];
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}({ctype} *out0, {ctype} *out1, const unsigned char *raw0, const unsigned char *raw1, const {ctype} *x) {{
  signed char xq[TOKENS * IN_FEATURES];
  float xd[TOKENS * GROUPS];
  for (int token = 0; token < TOKENS; token++) for (int group = 0; group < GROUPS; group++) {{
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {{
      float value = (float)x[token * IN_FEATURES + group * 32 + i];
      if (value < 0.0f) value = -value;
      if (value > amax) amax = value;
    }}
    const float d = amax > 0.00000001f ? amax / 127.0f : 0.00000001f;
    xd[group * TOKENS + token] = d;
    for (int i = 0; i < 32; i++) {{
      int value = (int)__builtin_roundf((float)x[token * IN_FEATURES + group * 32 + i] / d);
      if (value < -127) value = -127;
      if (value > 127) value = 127;
      xq[(group * TOKENS + token) * 32 + i] = (signed char)value;
    }}
  }}
  const int total = OUT0 + OUT1;
  dispatch((task_t){{out0, out1, raw0, raw1, xq, xd, 0, total}}, total, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def q8_batched_pair(first:Linear, second:Linear, x:Tensor) -> tuple[Tensor, Tensor]:
  assert first.ggml_type == second.ggml_type == 8 and first.in_features == second.in_features == x.shape[-1]
  tokens = int(x.numel()) // x.shape[-1]
  assert tokens > 1 and x.dtype in (dtypes.float16, dtypes.float32)
  out0, out1 = (Tensor.empty(tokens, layer.out_features, dtype=x.dtype, device=x.device) for layer in (first, second))
  src, binary, name = q8_batched_pair_program(tokens, first.out_features, second.out_features, x.shape[-1], x.dtype)
  outputs = Tensor.custom_kernel(out0, out1, first.weight, second.weight, x.reshape(tokens, x.shape[-1]).contiguous(),
    fxn=lambda out0,out1,raw0,raw1,x:q8_batched_pair_kernel(out0, out1, raw0, raw1, x, src, binary, name))
  shape = x.shape[:-1]
  return outputs[0].reshape(*shape, first.out_features), outputs[1].reshape(*shape, second.out_features)

def q8_batched_pair_kernel(out0:UOp, out1:UOp, raw0:UOp, raw1:UOp, x:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out0.base, out1.base, raw0.base, raw1.base, x.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=tuple(range(5)), outs=(0, 1), ins=(2, 3, 4)))

@functools.cache
def q8_linear_pair_program(out_features0:int, out_features1:int, in_features:int, dtype:DType,
                                f16_features:int=0) -> tuple[str, bytes, str]:
  assert dtype in (dtypes.float16, dtypes.float32)
  assert not f16_features or dtype == dtypes.float16
  name = f"cpu_q8_linear_pair{'_f16' if f16_features else ''}_{out_features0}_{out_features1}_{f16_features}_{in_features}_{dtype.name}"
  ctype = "_Float16" if dtype == dtypes.float16 else "float"
  threads = min(out_features0 + out_features1 + f16_features, 32, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  extra_arg = ", _Float16 *out2, const _Float16 *raw2" if f16_features else ""
  task_init = "out0, out1, out2, raw0, raw1, raw2, x" if f16_features else \
              "out0, out1, (_Float16 *)out0, raw0, raw1, (const _Float16 *)x, x"
  src = f"""
typedef union {{ unsigned short u; _Float16 h; }} half_bits;
static inline float half_to_float(const unsigned char *p) {{
  half_bits v; v.u = (unsigned short)p[0] | ((unsigned short)p[1] << 8); return (float)v.h;
}}
{dot_source()}
#define OUT0 {out_features0}
#define OUT1 {out_features1}
#define OUT2 {f16_features}
#define IN_FEATURES {in_features}
#define THREADS {threads}
typedef struct {{
  {ctype} *out0, *out1; _Float16 *out2; const unsigned char *raw0, *raw1; const _Float16 *raw2;
  const {ctype} *x; const signed char *xq; const float *xd; int begin, end;
}} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  for (int idx = task->begin; idx < task->end; idx++) {{
    if (idx < OUT0 + OUT1) {{
      const int projection = idx >= OUT0, row = projection ? idx - OUT0 : idx;
      const unsigned char *w = (projection ? task->raw1 : task->raw0) + row * (IN_FEATURES / 32) * 34;
#if defined(__AVX2__)
      __m256 accv = _mm256_setzero_ps();
      for (int block = 0; block < IN_FEATURES / 32; block++, w += 34)
        accv = _mm256_fmadd_ps(_mm256_set1_ps(half_to_float(w) * task->xd[block]),
                              _mm256_cvtepi32_ps(dot32_parts(_mm256_loadu_si256((const __m256i *)(w + 2)),
                                                            task->xq + block * 32)), accv);
      const float acc = hsum8f(accv);
#else
      float acc = 0.0f;
      for (int block = 0; block < IN_FEATURES / 32; block++, w += 34)
        acc += half_to_float(w) * task->xd[block] * (float)dot32((const signed char *)(w + 2), task->xq + block * 32);
#endif
      (projection ? task->out1 : task->out0)[row] = ({ctype})acc;
    }} else {{
      const int row = idx - OUT0 - OUT1;
      const _Float16 *weight = task->raw2 + row * IN_FEATURES;
      float acc = 0.0f;
      for (int i = 0; i < IN_FEATURES; i += 4)
        acc = acc + (float)task->x[i] * (float)weight[i] + (float)task->x[i+1] * (float)weight[i+1] +
              (float)task->x[i+2] * (float)weight[i+2] + (float)task->x[i+3] * (float)weight[i+3];
      task->out2[row] = (_Float16)acc;
    }}
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}({ctype} *out0, {ctype} *out1, const unsigned char *raw0, const unsigned char *raw1, const {ctype} *x{extra_arg}) {{
  signed char xq[IN_FEATURES];
  float xd[IN_FEATURES / 32];
  for (int block = 0; block < IN_FEATURES / 32; block++) {{
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {{
      float value = (float)x[block * 32 + i]; if (value < 0.0f) value = -value; if (value > amax) amax = value;
    }}
    const float d = amax > 0.00000001f ? amax / 127.0f : 0.00000001f;
    xd[block] = d;
    for (int i = 0; i < 32; i++) {{
      int value = (int)__builtin_roundf((float)x[block * 32 + i] / d);
      if (value < -127) value = -127; if (value > 127) value = 127; xq[block * 32 + i] = (signed char)value;
    }}
  }}
  const int total = OUT0 + OUT1 + OUT2;
  dispatch((task_t){{{task_init}, xq, xd, 0, total}}, total, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def q8_linear_pair(first:Linear, second:Linear, x:Tensor) -> tuple[Tensor, Tensor]:
  assert first.ggml_type == second.ggml_type == 8 and first.in_features == second.in_features and int(x.numel()) == first.in_features
  out0, out1 = (Tensor.empty(layer.out_features, dtype=x.dtype, device=x.device) for layer in (first, second))
  src, binary, name = q8_linear_pair_program(first.out_features, second.out_features, first.in_features, x.dtype)
  outputs = Tensor.custom_kernel(out0, out1, first.weight, second.weight, x.flatten().contiguous(),
    fxn=lambda out0,out1,raw0,raw1,x:q8_linear_pair_kernel(out0, out1, raw0, raw1, x, src, binary, name))
  shape = x.shape[:-1]
  return outputs[0].reshape(*shape, first.out_features), outputs[1].reshape(*shape, second.out_features)

def q8_linear_pair_kernel(out0:UOp, out1:UOp, raw0:UOp, raw1:UOp, x:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out0.base, out1.base, raw0.base, raw1.base, x.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=tuple(range(5)), outs=(0, 1), ins=(2, 3, 4)))

def q8_gdn_projections(first:Linear, second:Linear, f16_weight:Tensor, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  assert first.ggml_type == second.ggml_type == 8 and first.in_features == second.in_features == x.shape[-1]
  assert x.dtype == f16_weight.dtype == dtypes.float16 and f16_weight.shape[1] == x.shape[-1] and int(x.numel()) == x.shape[-1]
  out0, out1 = (Tensor.empty(layer.out_features, dtype=x.dtype, device=x.device) for layer in (first, second))
  out2 = Tensor.empty(f16_weight.shape[0], dtype=dtypes.float16, device=x.device)
  src, binary, name = q8_linear_pair_program(first.out_features, second.out_features, first.in_features, x.dtype, f16_weight.shape[0])
  outputs = Tensor.custom_kernel(out0, out1, first.weight, second.weight, x.flatten().contiguous(), out2, f16_weight,
    fxn=lambda out0,out1,raw0,raw1,x,out2,raw2:q8_gdn_projections_kernel(out0, out1, raw0, raw1, x, out2, raw2,
                                                                            src, binary, name))
  shape = x.shape[:-1]
  return (outputs[0].reshape(*shape, first.out_features), outputs[1].reshape(*shape, second.out_features),
          outputs[5].reshape(*shape, f16_weight.shape[0]))

def q8_gdn_projections_kernel(out0:UOp, out1:UOp, raw0:UOp, raw1:UOp, x:UOp, out2:UOp, raw2:UOp,
                                   src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out0.base, out1.base, raw0.base, raw1.base, x.base, out2.base, raw2.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=tuple(range(7)), outs=(0, 1, 5), ins=(2, 3, 4, 6)))

@functools.cache
def q6_argmax_program(out_features:int, in_features:int, dtype:DType) -> tuple[str, bytes, str]:
  assert dtype in (dtypes.float16, dtypes.float32)
  name = f"cpu_q6_argmax_{out_features}_{in_features}_{dtype.name}"
  ctype = "_Float16" if dtype == dtypes.float16 else "float"
  threads = min(out_features, 32, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  dot_helpers = dot_source()
  src = f"""
typedef union {{ unsigned short u; _Float16 h; }} half_bits;
static inline float half_to_float(const unsigned char *p) {{
  half_bits v; v.u = (unsigned short)p[0] | ((unsigned short)p[1] << 8); return (float)v.h;
}}
{dot_helpers}
#define OUT_FEATURES {out_features}
#define IN_FEATURES {in_features}
#define THREADS {threads}
typedef struct {{ const unsigned char *raw; const signed char *xq; const float *xd; }} task_t;
static float best_values[THREADS];
static int best_indices[THREADS], active_workers;
static void run_part(const task_t *task, int id, int workers) {{
  const int begin = OUT_FEATURES * id / workers, end = OUT_FEATURES * (id + 1) / workers;
  float best = -__builtin_inff();
  int best_index = begin;
  for (int row = begin; row < end; row++) {{
    const unsigned char *w = task->raw + row * (IN_FEATURES / 256) * 210;
    float acc = 0.0f;
    for (int block = 0; block < IN_FEATURES / 256; block++, w += 210) {{
      const float d = half_to_float(w + 208);
      int dots_lo[8], dots_hi[8];
      dot_q6_8(w, task->xq + block * 256, dots_lo, dots_hi);
      int block_sum = 0;
      for (int subgroup = 0; subgroup < 8; subgroup++) {{
        const int scale0 = ((const signed char *)(w + 192))[subgroup * 2];
        const int scale1 = ((const signed char *)(w + 192))[subgroup * 2 + 1];
        block_sum += scale0 * dots_lo[subgroup] + scale1 * dots_hi[subgroup];
      }}
      acc += d * task->xd[block] * (float)block_sum;
    }}
    if (acc > best) {{ best = acc; best_index = row; }}
  }}
  best_values[id] = best;
  best_indices[id] = best_index;
}}
extern void tiny_cpu_parallel(void (*)(void *, int, int, int, int), void *, int, int);
static void pool_worker(void *opaque, int id, int active, int begin, int end) {{
  (void)begin; (void)end;
  if (id == 0) active_workers = active;
  run_part((const task_t *)opaque, id, active);
}}
void {name}(int *out, const unsigned char *raw, const {ctype} *x) {{
  signed char xq[IN_FEATURES];
  float xd[IN_FEATURES / 256];
  for (int block = 0; block < IN_FEATURES / 256; block++) {{
    float max = 0.0f, amax = 0.0f;
    for (int i = 0; i < 256; i++) {{
      const float value = (float)x[block * 256 + i];
      const float abs_value = value < 0.0f ? -value : value;
      if (abs_value > amax) {{ amax = abs_value; max = value; }}
    }}
    const float scale = amax > 0.0f ? -max / 127.0f : 0.0f;
    xd[block] = scale;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    for (int i = 0; i < 256; i++)
      xq[block * 256 + i] = (signed char)__builtin_roundf((float)x[block * 256 + i] * inverse);
  }}
  task_t task = (task_t){{raw, xq, xd}};
  tiny_cpu_parallel(pool_worker, &task, OUT_FEATURES, THREADS);
  float best = best_values[0];
  int best_index = best_indices[0];
  for (int i = 1; i < active_workers; i++)
    if (best_values[i] > best || (best_values[i] == best && best_indices[i] < best_index)) {{
      best = best_values[i]; best_index = best_indices[i];
    }}
  out[0] = best_index;
}}
"""
  return src, _compile_cpu_ggml(src), name

def q6_argmax(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type == 14 and int(x.numel()) == layer.in_features
  out = Tensor.empty(1, 1, dtype=dtypes.int32, device=x.device)
  xc = x.reshape(1, layer.in_features).contiguous()
  src, binary, name = q6_argmax_program(layer.out_features, layer.in_features, xc.dtype)
  return Tensor.custom_kernel(out, layer.weight, xc, fxn=lambda out,raw,x:
    q6_argmax_kernel(out, raw, x, src, binary, name))[0]

def q6_argmax_kernel(out:UOp, raw:UOp, x:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out.base, raw.base, x.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=(0, 1, 2), outs=(0,), ins=(1, 2)))

@functools.cache
def ggml_expert_program(ggml_type:int, routes:int, routes_per_input:int, out_features:int,
                             in_features:int, dtype:DType, projections:int=1, fuse_silu:bool=False) -> tuple[str, bytes, str]:
  assert ggml_type in (14, 21, 23) and dtype in (dtypes.float16, dtypes.float32)
  assert projections in (1, 2)
  assert not fuse_silu or projections == 2 and dtype == dtypes.float32
  from tinygrad.runtime.autogen import ggml_common
  use_q8k = ggml_type in (21, 23) and bool(getenv("CPU_EXPERT_Q8K", 1))
  expert_tile = max(1, getenv("CPU_EXPERT_TILE", 16))
  name = f"cpu_ggml_expert{'_silu' if fuse_silu else '_pair' if projections == 2 else ''}_{ggml_type}_{routes}_{routes_per_input}_" + \
         f"{out_features}_{in_features}_{dtype.name}{'_q8k' if use_q8k else ''}_t{expert_tile}"
  ctype = "_Float16" if dtype == dtypes.float16 else "float"
  threads = min(projections * routes * out_features, 32 if routes != routes_per_input or projections == 1 else 16,
                max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  if ggml_type == 21:
    sign_masks = [sum((0xff << (8*i)) for i in range(4) if signs & (1 << i)) for signs in range(16)]
    tables = "static const unsigned int grid[512] = {" + ",".join(hex(x) for x in ggml_common.iq3s_grid) + "};\n" + \
      "static const unsigned int sign_masks[16] = {" + ",".join(hex(x) for x in sign_masks) + r"""};
static unsigned int signed_grid[16][512];
static int signed_grid_initialized;
static void init_signed_grid(void) {
  if (signed_grid_initialized) return;
  for (int signs = 0; signs < 16; signs++) for (int value = 0; value < 512; value++) {
    const unsigned int mask = sign_masks[signs];
    signed_grid[signs][value] = (grid[value] ^ mask) + (mask & 0x01010101);
  }
  signed_grid_initialized = 1;
}
static inline void unpack_iq3(const unsigned char *qs, unsigned int qh, unsigned int signs, unsigned int *qwords) {
  for (int word = 0; word < 8; word++)
    qwords[word] = signed_grid[(signs >> (word * 4)) & 15][qs[word] | (((qh >> word) & 1) << 8)];
}
static inline int dot_iq3(const unsigned char *qs, unsigned int qh, unsigned int signs, const signed char *xq) {
  unsigned int qwords[8];
  unpack_iq3(qs, qh, signs, qwords);
  return dot32((const signed char *)qwords, xq);
}
#if defined(__AVX2__)
static const unsigned char iq3_mask1_data[32] = {
  0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3};
static const unsigned char iq3_mask2_data[32] = {
  1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,
  1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
static inline __m256i unpack_iq3_unsigned(const unsigned char *qs, unsigned int qh) {
  unsigned int qwords[8];
  for (int word = 0; word < 8; word++) qwords[word] = grid[qs[word] | (((qh >> word) & 1) << 8)];
  return _mm256_loadu_si256((const __m256i *)qwords);
}
static inline __m256i iq3_apply_signs(__m256i x, unsigned int signs) {
  const __m256i mask1 = _mm256_loadu_si256((const __m256i *)iq3_mask1_data);
  const __m256i mask2 = _mm256_loadu_si256((const __m256i *)iq3_mask2_data);
  __m256i bits = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)signs), mask1), mask2);
  const __m256i negative = _mm256_cmpeq_epi8(bits, mask2);
  return _mm256_sub_epi8(_mm256_xor_si256(negative, x), negative);
}
#endif"""
    batch_dot = """
    const unsigned char *w = row_raw;
    for (int block = 0; block < IN_FEATURES / 256; block++, w += 110) {
      const float d = half_to_float(w);
      const unsigned int scale_word = load_u32(w + 106);
#if defined(__AVX2__)
      __m256i block_accv[ROUTES];
      for (int m = 0; m < matched_count; m++) block_accv[m] = _mm256_setzero_si256();
      for (int pair = 0; pair < 4; pair++) {
        const int subgroup0 = pair * 2, subgroup1 = subgroup0 + 1;
        const unsigned int qh_word = load_u32(w + 66 + (pair >> 1) * 4);
        const unsigned int qh0 = (qh_word >> (8 * (subgroup0 & 3))) & 255;
        const unsigned int qh1 = (qh_word >> (8 * (subgroup1 & 3))) & 255;
        const __m256i q0 = unpack_iq3_unsigned(w + 2 + subgroup0 * 8, qh0);
        const __m256i q1 = unpack_iq3_unsigned(w + 2 + subgroup1 * 8, qh1);
        const unsigned int signs0 = load_u32(w + 74 + subgroup0 * 4);
        const unsigned int signs1 = load_u32(w + 74 + subgroup1 * 4);
        const int scale0 = 1 + 2 * ((scale_word >> (4 * subgroup0)) & 15);
        const int scale1 = 1 + 2 * ((scale_word >> (4 * subgroup1)) & 15);
        for (int m = 0; m < matched_count; m++) {
          const int input = matched[m] / ROUTES_PER_INPUT;
          const signed char *x = task->xq + input * IN_FEATURES + block * 256 + subgroup0 * 32;
          const __m256i x0 = iq3_apply_signs(_mm256_loadu_si256((const __m256i *)x), signs0);
          const __m256i x1 = iq3_apply_signs(_mm256_loadu_si256((const __m256i *)(x + 32)), signs1);
          const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(q0, x0), _mm256_set1_epi16((short)scale0));
          const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q1, x1), _mm256_set1_epi16((short)scale1));
          block_accv[m] = _mm256_add_epi32(block_accv[m], _mm256_add_epi32(p0, p1));
        }
      }
#else
      int block_acc[ROUTES];
      for (int m = 0; m < matched_count; m++) block_acc[m] = 0;
      for (int subgroup = 0; subgroup < 8; subgroup++) {
        const unsigned int qh_word = load_u32(w + 66 + (subgroup >> 2) * 4);
        const unsigned int qh = (qh_word >> (8 * (subgroup & 3))) & 255;
        const unsigned int signs = load_u32(w + 74 + subgroup * 4);
        const int scale = 1 + 2 * ((scale_word >> (4 * subgroup)) & 15);
        unsigned int qwords[8];
        unpack_iq3(w + 2 + subgroup * 8, qh, signs, qwords);
        for (int m = 0; m < matched_count; m++) {
          const int input = matched[m] / ROUTES_PER_INPUT;
          block_acc[m] += scale * dot32((const signed char *)qwords,
            task->xq + input * IN_FEATURES + block * 256 + subgroup * 32);
        }
      }
#endif
      for (int m = 0; m < matched_count; m++) {
        const int input = matched[m] / ROUTES_PER_INPUT;
        acc[m] += d * task->xd[input * (IN_FEATURES / 256) + block] *
#if defined(__AVX2__)
          (float)hsum8(block_accv[m]);
#else
          (float)block_acc[m];
#endif
      }
    }""" if use_q8k else """
    const unsigned char *w = row_raw;
    for (int block = 0; block < IN_FEATURES / 256; block++, w += 110) {
      const float d = half_to_float(w);
      const unsigned int scale_word = load_u32(w + 106);
      for (int subgroup = 0; subgroup < 8; subgroup++) {
        const unsigned int qh_word = load_u32(w + 66 + (subgroup >> 2) * 4);
        const unsigned int qh = (qh_word >> (8 * (subgroup & 3))) & 255;
        const unsigned int signs = load_u32(w + 74 + subgroup * 4);
        const int group = block * 8 + subgroup;
        const float wd = d * (float)(1 + 2 * ((scale_word >> (4 * subgroup)) & 15));
        unsigned int qwords[8];
        unpack_iq3(w + 2 + subgroup * 8, qh, signs, qwords);
        for (int m = 0; m < matched_count; m++) {
          const int input = matched[m] / ROUTES_PER_INPUT;
          acc[m] += wd * task->xd[input * (IN_FEATURES / 32) + group] *
            (float)dot32((const signed char *)qwords, task->xq + input * IN_FEATURES + group * 32);
        }
      }
    }"""
  elif ggml_type == 23:
    tables = "static const signed char kvalues[16] = {" + ",".join(str(x) for x in ggml_common.kvalues_iq4nl) + r"""};
static inline void unpack_iq4(const unsigned char *qs, signed char *qvals) {
#if defined(__AVX2__)
  const __m128i packed = _mm_loadu_si128((const __m128i *)qs);
  const __m128i mask = _mm_set1_epi8(15), lut = _mm_loadu_si128((const __m128i *)kvalues);
  const __m128i qlo = _mm_shuffle_epi8(lut, _mm_and_si128(packed, mask));
  const __m128i qhi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
  _mm_storeu_si128((__m128i *)qvals, qlo);
  _mm_storeu_si128((__m128i *)(qvals + 16), qhi);
#else
  for (int pos = 0; pos < 32; pos++) qvals[pos] = kvalues[(qs[pos & 15] >> (pos < 16 ? 0 : 4)) & 15];
#endif
}
static inline int dot_iq4(const unsigned char *qs, const signed char *xq) {
  signed char qvals[32];
  unpack_iq4(qs, qvals);
  return dot32(qvals, xq);
}"""
    batch_dot = """
    const unsigned char *w = row_raw;
    for (int block = 0; block < IN_FEATURES / 256; block++, w += 136) {
      const float d = half_to_float(w);
      const unsigned int high = (unsigned int)w[2] | ((unsigned int)w[3] << 8);
#if defined(__AVX2__)
      __m256i block_accv[ROUTES];
      for (int m = 0; m < matched_count; m++) block_accv[m] = _mm256_setzero_si256();
#else
      int block_acc[ROUTES];
      for (int m = 0; m < matched_count; m++) block_acc[m] = 0;
#endif
      for (int subgroup = 0; subgroup < 8; subgroup++) {
        const int low = (w[4 + (subgroup >> 1)] >> (4 * (subgroup & 1))) & 15;
        const int scale_bits = low | (((high >> (2 * subgroup)) & 3) << 4);
        const unsigned char *qs = w + 8 + subgroup * 16;
        signed char qvals[32];
        unpack_iq4(qs, qvals);
        for (int m = 0; m < matched_count; m++) {
          const int input = matched[m] / ROUTES_PER_INPUT;
#if defined(__AVX2__)
          const __m256i pairs = dot32_pairs(_mm256_loadu_si256((const __m256i *)qvals),
            task->xq + input * IN_FEATURES + block * 256 + subgroup * 32);
          block_accv[m] = _mm256_add_epi32(block_accv[m],
            _mm256_madd_epi16(pairs, _mm256_set1_epi16((short)(scale_bits - 32))));
#else
          block_acc[m] += (scale_bits - 32) * dot32(qvals,
            task->xq + input * IN_FEATURES + block * 256 + subgroup * 32);
#endif
        }
      }
      for (int m = 0; m < matched_count; m++) {
        const int input = matched[m] / ROUTES_PER_INPUT;
        acc[m] += d * task->xd[input * (IN_FEATURES / 256) + block] *
#if defined(__AVX2__)
          (float)hsum8(block_accv[m]);
#else
          (float)block_acc[m];
#endif
      }
    }""" if use_q8k else """
    const unsigned char *w = row_raw;
    for (int block = 0; block < IN_FEATURES / 256; block++, w += 136) {
      const float d = half_to_float(w);
      const unsigned int high = (unsigned int)w[2] | ((unsigned int)w[3] << 8);
      for (int subgroup = 0; subgroup < 8; subgroup++) {
        const int low = (w[4 + (subgroup >> 1)] >> (4 * (subgroup & 1))) & 15;
        const int scale_bits = low | (((high >> (2 * subgroup)) & 3) << 4);
        const unsigned char *qs = w + 8 + subgroup * 16;
        signed char qvals[32];
        unpack_iq4(qs, qvals);
        for (int m = 0; m < matched_count; m++) {
          const int input = matched[m] / ROUTES_PER_INPUT;
          const int group = block * 8 + subgroup;
          acc[m] += d * (float)(scale_bits - 32) * task->xd[input * (IN_FEATURES / 32) + group] *
            (float)dot32(qvals, task->xq + input * IN_FEATURES + group * 32);
        }
      }
    }"""
  else:
    tables = ""
    batch_dot = """
    const unsigned char *w = row_raw;
    for (int block = 0; block < IN_FEATURES / 256; block++, w += 210) {
      const float d = half_to_float(w + 208);
      for (int subgroup = 0; subgroup < 8; subgroup++) {
        const int group = block * 8 + subgroup;
        const int scale0 = ((const signed char *)(w + 192))[subgroup * 2];
        const int scale1 = ((const signed char *)(w + 192))[subgroup * 2 + 1];
#if defined(__AVX2__)
        const __m256i qvals = unpack_q6(w, subgroup);
#endif
        for (int m = 0; m < matched_count; m++) {
          const int input = matched[m] / ROUTES_PER_INPUT;
          const signed char *xq = task->xq + input * IN_FEATURES + group * 32;
          const float xd = task->xd[input * (IN_FEATURES / 32) + group];
          int dot_lo, dot_hi;
#if defined(__AVX2__)
          dot_q6_vec(qvals, xq, &dot_lo, &dot_hi);
#else
          dot_q6(w, subgroup, xq, &dot_lo, &dot_hi);
#endif
          acc[m] += d * (float)scale0 * xd * (float)(dot_lo + dot_hi);
          acc[m] += d * (float)(scale1 - scale0) * xd * (float)dot_hi;
        }
      }
    }"""
  type_size = _GGML_QUANT[ggml_type][1]
  worker = f"""
  for (int idx = task->begin; idx < task->end; idx++) {{
    const int projection = idx / task->projection_size, local_idx = idx - projection * task->projection_size;
    const int route = local_idx / OUT_FEATURES, row = local_idx - route * OUT_FEATURES, expert = task->sel[route];
    {ctype} *out = projection ? task->out1 : task->out0;
    const unsigned char *raw = projection ? task->raw1 : task->raw0;
    const int matched[1] = {{route}}, matched_count = 1;
    float acc[1] = {{0.0f}};
    const unsigned char *row_raw = raw + ((expert * OUT_FEATURES + row) * (IN_FEATURES / 256)) * TYPE_SIZE;
{batch_dot}
    out[local_idx] = ({ctype})acc[0];
  }}""" if routes == routes_per_input else f"""
  for (int idx = task->begin; idx < task->end; idx++) {{
    const int projection = idx / task->projection_size, local_idx = idx - projection * task->projection_size;
    const int unique_idx = local_idx / OUT_FEATURES, row = local_idx - unique_idx * OUT_FEATURES, expert = task->unique[unique_idx];
    {ctype} *out = projection ? task->out1 : task->out0;
    const unsigned char *raw = projection ? task->raw1 : task->raw0;
    int matched[ROUTES], matched_count = 0;
    float acc[ROUTES];
    for (int route = task->head[unique_idx]; route >= 0; route = task->next[route]) {{
      matched[matched_count] = route;
      acc[matched_count++] = 0.0f;
    }}
    const unsigned char *row_raw = raw + ((expert * OUT_FEATURES + row) * (IN_FEATURES / 256)) * TYPE_SIZE;
{batch_dot}
    for (int m = 0; m < matched_count; m++) out[matched[m] * OUT_FEATURES + row] = ({ctype})acc[m];
  }}"""
  if fuse_silu:
    batch_dot0 = "{\n" + batch_dot.replace("row_raw", "row_raw0").replace("acc[", "acc0[") + "\n}"
    batch_dot1 = "{\n" + batch_dot.replace("row_raw", "row_raw1").replace("acc[", "acc1[") + "\n}"
    worker = f"""
  for (int local_idx = task->begin; local_idx < task->end; local_idx++) {{
    const int route = local_idx / OUT_FEATURES, row = local_idx - route * OUT_FEATURES, expert = task->sel[route];
    const int matched[1] = {{route}}, matched_count = 1;
    float acc0[1] = {{0.0f}}, acc1[1] = {{0.0f}};
    const unsigned char *row_raw0 = task->raw0 + ((expert * OUT_FEATURES + row) * (IN_FEATURES / 256)) * TYPE_SIZE;
    const unsigned char *row_raw1 = task->raw1 + ((expert * OUT_FEATURES + row) * (IN_FEATURES / 256)) * TYPE_SIZE;
{batch_dot0}
{batch_dot1}
    task->out0[local_idx] = tiny_silu_mul(acc0[0], acc1[0]);
  }}""" if routes == routes_per_input else f"""
  for (int local_idx = task->begin; local_idx < task->end; local_idx++) {{
    const int unique_idx = local_idx / OUT_FEATURES, row = local_idx - unique_idx * OUT_FEATURES, expert = task->unique[unique_idx];
    int matched[ROUTES], matched_count = 0;
    float acc0[ROUTES], acc1[ROUTES];
    for (int route = task->head[unique_idx]; route >= 0; route = task->next[route]) {{
      matched[matched_count] = route;
      acc0[matched_count] = acc1[matched_count] = 0.0f;
      matched_count++;
    }}
    const unsigned char *row_raw0 = task->raw0 + ((expert * OUT_FEATURES + row) * (IN_FEATURES / 256)) * TYPE_SIZE;
    const unsigned char *row_raw1 = task->raw1 + ((expert * OUT_FEATURES + row) * (IN_FEATURES / 256)) * TYPE_SIZE;
{batch_dot0}
{batch_dot1}
    for (int m = 0; m < matched_count; m++)
      task->out0[matched[m] * OUT_FEATURES + row] = tiny_silu_mul(acc0[m], acc1[m]);
  }}"""
  if routes != routes_per_input:
    loop_start = "  for (int local_idx = task->begin; local_idx < task->end; local_idx++) {" if fuse_silu else \
                 "  for (int idx = task->begin; idx < task->end; idx++) {"
    loop_var = "local_idx" if fuse_silu else "idx"
    worker = worker.replace(loop_start, f"""  for (;;) {{
    const int tile_begin = __atomic_fetch_add(task->next_work, {expert_tile}, __ATOMIC_RELAXED);
    if (tile_begin >= task->total_work) break;
    const int tile_end = tile_begin + {expert_tile} < task->total_work ? tile_begin + {expert_tile} : task->total_work;
    for (int {loop_var} = tile_begin; {loop_var} < tile_end; {loop_var}++) {{""", 1) + "\n  }"
  dot_helpers = dot_source()
  entry_args = f"{ctype} *out0, const unsigned char *raw0, const unsigned char *raw1" if fuse_silu else \
               f"{ctype} *out0, {ctype} *out1, const unsigned char *raw0, const unsigned char *raw1" if projections == 2 else \
               f"{ctype} *out0, const unsigned char *raw0"
  task_pointers = "out0, out0, raw0, raw1" if fuse_silu else \
                  "out0, out1, raw0, raw1" if projections == 2 else "out0, out0, raw0, raw0"
  init_tables = "  init_signed_grid();\n" if ggml_type == 21 else ""
  silu_helper = r"""
static inline float tiny_silu_mul(float gate, float up) {
  const float x = gate * -1.4426950408889634f;
  const int is_nan = x != x;
  const float finite = x == -__builtin_inff() || x == __builtin_inff() || is_nan ? 0.0f : x;
  const int rounded = (int)(finite + (finite < 0.0f ? -0.5f : 0.5f));
  const int negative = rounded < 0;
  const int half = ((rounded + negative) >> 1) - (((rounded % 2) != 0) & negative);
  const float s = finite - (float)rounded;
  float exp = ((((((0.0001535920892f*s + 0.001339262701f)*s + 0.009618384764f)*s +
                  0.05550347269f)*s + 0.2402264476f)*s + 0.6931471825f)*s + 1.0f);
  union { unsigned int u; float f; } hi = {(unsigned int)((half + 127) << 23)};
  union { unsigned int u; float f; } lo = {(unsigned int)(((rounded - half) + 127) << 23)};
  exp *= hi.f * lo.f;
  exp = !(x < 128.0f) ? __builtin_inff() : x < -150.0f ? 0.0f : exp;
  if (is_nan) exp = __builtin_nanf("");
  return gate / (1.0f + exp) * up;
}
""" if fuse_silu else ""
  quantize_src = """
  signed char xq[INPUT_COUNT * IN_FEATURES];
  float xd[INPUT_COUNT * (IN_FEATURES / 256)];
  for (int input = 0; input < INPUT_COUNT; input++) for (int block = 0; block < IN_FEATURES / 256; block++) {
    float max = 0.0f, amax = 0.0f;
    for (int i = 0; i < 256; i++) {
      const float value = (float)x[input * IN_FEATURES + block * 256 + i];
      const float abs_value = value < 0.0f ? -value : value;
      if (abs_value > amax) { amax = abs_value; max = value; }
    }
    const float d = amax > 0.0f ? -max / 127.0f : 0.0f;
    xd[input * (IN_FEATURES / 256) + block] = d;
    const float inv = d != 0.0f ? 1.0f / d : 0.0f;
    for (int i = 0; i < 256; i++)
      xq[input * IN_FEATURES + block * 256 + i] =
        (signed char)__builtin_roundf((float)x[input * IN_FEATURES + block * 256 + i] * inv);
  }""" if use_q8k else """
  signed char xq[INPUT_COUNT * IN_FEATURES];
  float xd[INPUT_COUNT * (IN_FEATURES / 32)];
  for (int input = 0; input < INPUT_COUNT; input++) for (int group = 0; group < IN_FEATURES / 32; group++) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
      float v = (float)x[input * IN_FEATURES + group * 32 + i];
      if (v < 0.0f) v = -v;
      if (v > amax) amax = v;
    }
    const float d = amax > 0.00000001f ? amax / 127.0f : 0.00000001f;
    xd[input * (IN_FEATURES / 32) + group] = d;
    for (int i = 0; i < 32; i++) {
      int q = (int)__builtin_roundf((float)x[input * IN_FEATURES + group * 32 + i] / d);
      if (q < -127) q = -127;
      if (q > 127) q = 127;
      xq[input * IN_FEATURES + group * 32 + i] = (signed char)q;
    }
  }"""
  src = f"""
typedef union {{ unsigned short u; _Float16 h; }} half_bits;
static inline float half_to_float(const unsigned char *p) {{
  half_bits v; v.u = (unsigned short)p[0] | ((unsigned short)p[1] << 8); return (float)v.h;
}}
static inline unsigned int load_u32(const unsigned char *p) {{
  return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}}
{dot_helpers}
{tables}
{silu_helper}
#define ROUTES {routes}
#define ROUTES_PER_INPUT {routes_per_input}
#define INPUT_COUNT (ROUTES / ROUTES_PER_INPUT)
#define OUT_FEATURES {out_features}
#define IN_FEATURES {in_features}
#define TYPE_SIZE {type_size}
#define THREADS {threads}
#define PROJECTIONS {projections}
typedef struct {{
  {ctype} *out0, *out1; const unsigned char *raw0, *raw1; const int *sel, *unique, *head, *next; const {ctype} *x;
  const signed char *xq; const float *xd; unsigned *next_work; int total_work, projection_size, begin, end;
}} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
{worker}
  return (void *)0;
}}
{_SEM_POOL}
void {name}({entry_args}, const int *sel, const {ctype} *x) {{
{init_tables}\
  int unique[ROUTES], head[ROUTES], next[ROUTES], unique_count = 0;
  for (int route = 0; route < ROUTES; route++) {{
    int unique_idx = 0;
    while (unique_idx < unique_count && unique[unique_idx] != sel[route]) unique_idx++;
    if (unique_idx == unique_count) {{
      unique[unique_count] = sel[route];
      head[unique_count++] = -1;
    }}
    next[route] = head[unique_idx];
    head[unique_idx] = route;
  }}
  const int projection_size = INPUT_COUNT == 1 ? ROUTES * OUT_FEATURES : unique_count * OUT_FEATURES;
  const int total = projection_size * {1 if fuse_silu else projections};
{quantize_src}
  unsigned next_work = 0;
  dispatch((task_t){{{task_pointers}, sel, unique, head, next, x, xq, xd, &next_work, total, projection_size, 0, total}}, total, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def ggml_expert_kernel(out:UOp, raw:UOp, sel:UOp, x:UOp, ggml_type:int, routes_per_input:int) -> UOp:
  routes, out_features, in_features = out.shape[0], out.shape[1], x.shape[1]
  src, binary, name = ggml_expert_program(ggml_type, routes, routes_per_input, out_features, in_features, x.dtype)
  sink = UOp.sink(out.base, raw.base, sel.base, x.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=(0, 1, 2, 3), outs=(0,), ins=(1, 2, 3)))

def ggml_expert_pair_kernel(out0:UOp, out1:UOp, raw0:UOp, raw1:UOp, sel:UOp, x:UOp,
                                 ggml_type:int, routes_per_input:int) -> UOp:
  routes, out_features, in_features = out0.shape[0], out0.shape[1], x.shape[1]
  assert out1.shape == out0.shape
  src, binary, name = ggml_expert_program(ggml_type, routes, routes_per_input, out_features, in_features, x.dtype, projections=2)
  sink = UOp.sink(out0.base, out1.base, raw0.base, raw1.base, sel.base, x.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=tuple(range(6)), outs=(0, 1), ins=(2, 3, 4, 5)))

def ggml_expert_silu_kernel(out:UOp, raw0:UOp, raw1:UOp, sel:UOp, x:UOp, ggml_type:int, routes_per_input:int) -> UOp:
  routes, out_features, in_features = out.shape[0], out.shape[1], x.shape[1]
  src, binary, name = ggml_expert_program(ggml_type, routes, routes_per_input, out_features, in_features,
                                                x.dtype, projections=2, fuse_silu=True)
  sink = UOp.sink(out.base, raw0.base, raw1.base, sel.base, x.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=tuple(range(5)), outs=(0,), ins=(1, 2, 3, 4)))

@functools.cache
def moe_program(routes:int, dim:int, hidden:int, inputs:int=1) -> tuple[str, bytes, str]:
  from tinygrad.runtime.autogen import ggml_common
  assert dim % 256 == hidden % 256 == 0 and routes % inputs == 0
  threads = min(32, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  use_q8k = getenv("CPU_MOE_Q8K", 1)
  name = f"cpu_moe_iq3_iq4_q8{'_q8k' if use_q8k else ''}_{routes}_{inputs}_{dim}_{hidden}"
  grid = ",".join(hex(x) for x in ggml_common.iq3s_grid)
  kvalues = ",".join(str(x) for x in ggml_common.kvalues_iq4nl)
  sign_masks = ",".join(hex(sum((0xff << (8*i)) for i in range(4) if signs & (1 << i))) for signs in range(16))
  src = f"""
extern float expf(float);
typedef union {{ unsigned short u; _Float16 h; }} half_bits;
static inline float half_to_float(const unsigned char *p) {{
  half_bits v; v.u = (unsigned short)p[0] | ((unsigned short)p[1] << 8); return (float)v.h;
}}
static inline unsigned int load_u32(const unsigned char *p) {{
  return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}}
{dot_source()}
static const unsigned int grid[512] = {{{grid}}};
static const unsigned int sign_masks[16] = {{{sign_masks}}};
static const signed char kvalues[16] = {{{kvalues}}};
#if defined(__AVX2__)
static inline __m256i dot_iq3_parts(const unsigned char *qs, unsigned int qh, unsigned int signs, const signed char *xq) {{
  const __m256i qbytes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)qs));
  const __m256i qh_mask = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  const __m256i qh_zero = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32((int)qh), qh_mask), _mm256_setzero_si256());
  const __m256i indices = _mm256_or_si256(qbytes, _mm256_andnot_si256(qh_zero, _mm256_set1_epi32(256)));
  const __m256i qwords = _mm256_i32gather_epi32((const int *)grid, indices, 4);
  static const unsigned char mask1_data[32] = {{0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3}};
  static const unsigned char mask2_data[32] = {{1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,
                                                1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128}};
  const __m256i mask1 = _mm256_loadu_si256((const __m256i *)mask1_data);
  const __m256i mask2 = _mm256_loadu_si256((const __m256i *)mask2_data);
  __m256i expanded = _mm256_set1_epi32((int)signs);
  expanded = _mm256_and_si256(_mm256_shuffle_epi8(expanded, mask1), mask2);
  const __m256i negative = _mm256_cmpeq_epi8(expanded, mask2);
  const __m256i xv = _mm256_loadu_si256((const __m256i *)xq);
  const __m256i signed_x = _mm256_sub_epi8(_mm256_xor_si256(negative, xv), negative);
  const __m256i pairs = _mm256_maddubs_epi16(qwords, signed_x);
  return _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
}}
static inline int dot_iq3(const unsigned char *qs, unsigned int qh, unsigned int signs, const signed char *xq) {{
  return hsum8(dot_iq3_parts(qs, qh, signs, xq));
}}
static inline int dot_iq3_q8k(const unsigned char *w, const signed char *xq) {{
  static const unsigned char mask1_data[32] = {{0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3}};
  static const unsigned char mask2_data[32] = {{1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,
                                                1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128}};
  const __m256i mask1 = _mm256_loadu_si256((const __m256i *)mask1_data);
  const __m256i mask2 = _mm256_loadu_si256((const __m256i *)mask2_data);
  const __m256i idx_shift = _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1);
  typedef union {{ __m256i vec[2]; unsigned int index[16]; }} indices_t;
  const unsigned char *qs = w + 2, *qh = w + 66;
  const unsigned short *signs = (const unsigned short *)(w + 74);
  __m256i sum0 = _mm256_setzero_si256(), sum1 = _mm256_setzero_si256();
  indices_t indices;
  for (int subgroup = 0; subgroup < 8; subgroup += 2) {{
    const __m256i packed = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
    indices.vec[0] = _mm256_or_si256(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(packed)),
      _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[subgroup]), idx_shift), _mm256_set1_epi32(256)));
    indices.vec[1] = _mm256_or_si256(_mm256_cvtepu16_epi32(_mm256_extractf128_si256(packed, 1)),
      _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[subgroup+1]), idx_shift), _mm256_set1_epi32(256)));
    const __m256i values0 = _mm256_setr_epi32(grid[indices.index[0]], grid[indices.index[1]], grid[indices.index[2]],
      grid[indices.index[3]], grid[indices.index[4]], grid[indices.index[5]], grid[indices.index[6]], grid[indices.index[7]]);
    const __m256i values1 = _mm256_setr_epi32(grid[indices.index[8]], grid[indices.index[9]], grid[indices.index[10]],
      grid[indices.index[11]], grid[indices.index[12]], grid[indices.index[13]], grid[indices.index[14]], grid[indices.index[15]]);
    __m256i sign_bits = _mm256_set1_epi32(signs[0] | ((unsigned int)signs[1] << 16));
    sign_bits = _mm256_and_si256(_mm256_shuffle_epi8(sign_bits, mask1), mask2);
    const __m256i negative0 = _mm256_cmpeq_epi8(sign_bits, mask2);
    sign_bits = _mm256_set1_epi32(signs[2] | ((unsigned int)signs[3] << 16));
    sign_bits = _mm256_and_si256(_mm256_shuffle_epi8(sign_bits, mask1), mask2);
    const __m256i negative1 = _mm256_cmpeq_epi8(sign_bits, mask2);
    signs += 4;
    const __m256i xv0 = _mm256_loadu_si256((const __m256i *)(xq + subgroup * 32));
    const __m256i xv1 = _mm256_loadu_si256((const __m256i *)(xq + (subgroup + 1) * 32));
    const __m256i dot0 = _mm256_maddubs_epi16(values0, _mm256_sub_epi8(_mm256_xor_si256(negative0, xv0), negative0));
    const __m256i dot1 = _mm256_maddubs_epi16(values1, _mm256_sub_epi8(_mm256_xor_si256(negative1, xv1), negative1));
    const int scale0 = 1 + 2 * ((w[106 + subgroup / 2] >> (4 * (subgroup & 1))) & 15);
    const int scale1 = 1 + 2 * ((w[106 + (subgroup + 1) / 2] >> (4 * ((subgroup + 1) & 1))) & 15);
    sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(dot0, _mm256_set1_epi16(scale0)));
    sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(scale1)));
  }}
  return hsum8(_mm256_add_epi32(sum0, sum1));
}}
static inline float hsum_float8(__m256 x) {{
  __m128 sum = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
  sum = _mm_hadd_ps(sum, sum); return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}}
#else
static inline int dot_iq3(const unsigned char *qs, unsigned int qh, unsigned int signs, const signed char *xq) {{
  unsigned int qwords[8];
  for (int word = 0; word < 8; word++) qwords[word] = grid[qs[word] | (((qh >> word) & 1) << 8)];
  for (int word = 0; word < 8; word++) {{
    const unsigned int mask = sign_masks[(signs >> (word * 4)) & 15];
    qwords[word] = (qwords[word] ^ mask) + (mask & 0x01010101);
  }}
  return dot32((const signed char *)qwords, xq);
}}
static inline int dot_iq3_q8k(const unsigned char *w, const signed char *xq) {{
  int ret = 0;
  for (int subgroup = 0; subgroup < 8; subgroup++)
    ret += (1 + 2 * ((w[106 + subgroup / 2] >> (4 * (subgroup & 1))) & 15)) *
      dot_iq3(w + 2 + subgroup * 8, w[66 + subgroup], load_u32(w + 74 + subgroup * 4), xq + subgroup * 32);
  return ret;
}}
#endif
static inline int dot_iq4(const unsigned char *qs, const signed char *xq) {{
#if defined(__AVX2__)
  const __m128i packed = _mm_loadu_si128((const __m128i *)qs);
  const __m128i mask = _mm_set1_epi8(15), lut = _mm_loadu_si128((const __m128i *)kvalues);
  const __m128i qlo = _mm_shuffle_epi8(lut, _mm_and_si128(packed, mask));
  const __m128i qhi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
  return dot32v(_mm256_set_m128i(qhi, qlo), xq);
#else
  signed char qvals[32];
  for (int pos = 0; pos < 32; pos++) qvals[pos] = kvalues[(qs[pos & 15] >> (pos < 16 ? 0 : 4)) & 15];
  return dot32(qvals, xq);
#endif
}}
#if defined(__AVX2__)
static inline __m256i dot_iq4_parts(const unsigned char *qs, const signed char *xq) {{
  const __m128i packed = _mm_loadu_si128((const __m128i *)qs);
  const __m128i mask = _mm_set1_epi8(15), lut = _mm_loadu_si128((const __m128i *)kvalues);
  const __m128i qlo = _mm_shuffle_epi8(lut, _mm_and_si128(packed, mask));
  const __m128i qhi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
  return dot32_parts(_mm256_set_m128i(qhi, qlo), xq);
}}
#endif
static inline float tiny_silu_mul(float gate, float up) {{
  const float x = gate * -1.4426950408889634f;
  const int is_nan = x != x;
  const float finite = x == -__builtin_inff() || x == __builtin_inff() || is_nan ? 0.0f : x;
  const int rounded = (int)(finite + (finite < 0.0f ? -0.5f : 0.5f));
  const int negative = rounded < 0;
  const int half = ((rounded + negative) >> 1) - (((rounded % 2) != 0) & negative);
  const float s = finite - (float)rounded;
  float exp = ((((((0.0001535920892f*s + 0.001339262701f)*s + 0.009618384764f)*s +
                  0.05550347269f)*s + 0.2402264476f)*s + 0.6931471825f)*s + 1.0f);
  union {{ unsigned int u; float f; }} hi = {{(unsigned int)((half + 127) << 23)}};
  union {{ unsigned int u; float f; }} lo = {{(unsigned int)(((rounded - half) + 127) << 23)}};
  exp *= hi.f * lo.f;
  exp = !(x < 128.0f) ? __builtin_inff() : x < -150.0f ? 0.0f : exp;
  if (is_nan) exp = __builtin_nanf("");
  return gate / (1.0f + exp) * up;
}}
static void quantize(const float *x, int count, signed char *xq, float *xd) {{
  for (int group = 0; group < count / 32; group++) {{
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {{
      float value = x[group * 32 + i]; if (value < 0.0f) value = -value; if (value > amax) amax = value;
    }}
    const float d = amax > 0.00000001f ? amax / 127.0f : 0.00000001f;
    xd[group] = d;
    for (int i = 0; i < 32; i++) {{
      int value = (int)__builtin_roundf(x[group * 32 + i] / d);
      if (value < -127) value = -127; if (value > 127) value = 127; xq[group * 32 + i] = (signed char)value;
    }}
  }}
}}
static void quantize_q8k(const float *x, int count, signed char *xq, float *xd) {{
  for (int block = 0; block < count / 256; block++) {{
    float max = 0.0f, amax = 0.0f;
    for (int i = 0; i < 256; i++) {{
      const float value = x[block * 256 + i], abs_value = value < 0.0f ? -value : value;
      if (abs_value > amax) {{ amax = abs_value; max = value; }}
    }}
    const float scale = amax > 0.0f ? -max / 127.0f : 0.0f;
    xd[block] = scale;
    const float inv = scale != 0.0f ? 1.0f / scale : 0.0f;
    for (int i = 0; i < 256; i++) xq[block * 256 + i] = (signed char)__builtin_roundf(x[block * 256 + i] * inv);
  }}
}}
#define ROUTES {routes}
#define INPUTS {inputs}
#define ROUTES_PER_INPUT (ROUTES / INPUTS)
#define DIM {dim}
#define HIDDEN {hidden}
#define THREADS {threads}
#define USE_Q8K {int(use_q8k)}
typedef struct {{
  float *out, *rhidden, *shidden; const unsigned char *rgate, *rup, *rdown, *sgate, *sup, *sdown;
  const float *x, *probs; const int *sel, *unique, *head, *next; const signed char *xq, *rhq, *shq, *xkq, *rhkq;
  const float *xd, *rhd, *shd, *xkd, *rhkd;
  const float *shared_scale; int routed_work, stage, begin, end;
}} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  if (task->stage == 0) for (int idx = task->begin; idx < task->end; idx++) {{
    if (idx < task->routed_work) {{
      const int unique_idx = idx / HIDDEN, row = idx - unique_idx * HIDDEN, expert = task->unique[unique_idx];
      for (int route = task->head[unique_idx]; route >= 0; route = task->next[route]) {{
      const int input = route / ROUTES_PER_INPUT;
      const unsigned char *wg = task->rgate + ((expert * HIDDEN + row) * (DIM / 256)) * 110;
      const unsigned char *wu = task->rup + ((expert * HIDDEN + row) * (DIM / 256)) * 110;
#if defined(__AVX2__)
      __m256 gate_vec = _mm256_setzero_ps(), up_vec = _mm256_setzero_ps();
      float gate_k = 0.0f, up_k = 0.0f;
#else
      float gate = 0.0f, up = 0.0f, gate_k = 0.0f, up_k = 0.0f;
#endif
      for (int block = 0; block < DIM / 256; block++, wg += 110, wu += 110) {{
        const float dg = half_to_float(wg), du = half_to_float(wu);
        if (USE_Q8K) {{
          gate_k += dg * task->xkd[input * (DIM / 256) + block] *
            (float)dot_iq3_q8k(wg, task->xkq + input * DIM + block * 256);
          up_k += du * task->xkd[input * (DIM / 256) + block] *
            (float)dot_iq3_q8k(wu, task->xkq + input * DIM + block * 256);
          continue;
        }}
        const unsigned int sg = load_u32(wg + 106), su = load_u32(wu + 106);
        const unsigned long qhgs = (unsigned long)load_u32(wg + 66) | ((unsigned long)load_u32(wg + 70) << 32);
        const unsigned long qhus = (unsigned long)load_u32(wu + 66) | ((unsigned long)load_u32(wu + 70) << 32);
        for (int subgroup = 0; subgroup < 8; subgroup++) {{
          const int group = block * 8 + subgroup;
          const unsigned int qhg = (unsigned int)(qhgs >> (8 * subgroup)) & 255;
          const unsigned int qhu = (unsigned int)(qhus >> (8 * subgroup)) & 255;
#if defined(__AVX2__)
          const float scale = task->xd[input * (DIM / 32) + group];
          gate_vec = _mm256_fmadd_ps(_mm256_set1_ps(dg * (float)(1 + 2 * ((sg >> (4 * subgroup)) & 15)) * scale),
            _mm256_cvtepi32_ps(dot_iq3_parts(wg + 2 + subgroup * 8, qhg, load_u32(wg + 74 + subgroup * 4),
                                             task->xq + input * DIM + group * 32)), gate_vec);
          up_vec = _mm256_fmadd_ps(_mm256_set1_ps(du * (float)(1 + 2 * ((su >> (4 * subgroup)) & 15)) * scale),
            _mm256_cvtepi32_ps(dot_iq3_parts(wu + 2 + subgroup * 8, qhu, load_u32(wu + 74 + subgroup * 4),
                                             task->xq + input * DIM + group * 32)), up_vec);
#else
          gate += dg * (float)(1 + 2 * ((sg >> (4 * subgroup)) & 15)) * task->xd[input * (DIM / 32) + group] *
                  (float)dot_iq3(wg + 2 + subgroup * 8, qhg, load_u32(wg + 74 + subgroup * 4),
                                 task->xq + input * DIM + group * 32);
          up += du * (float)(1 + 2 * ((su >> (4 * subgroup)) & 15)) * task->xd[input * (DIM / 32) + group] *
                (float)dot_iq3(wu + 2 + subgroup * 8, qhu, load_u32(wu + 74 + subgroup * 4),
                               task->xq + input * DIM + group * 32);
#endif
        }}
      }}
#if defined(__AVX2__)
      const float gate = USE_Q8K ? gate_k : hsum_float8(gate_vec), up = USE_Q8K ? up_k : hsum_float8(up_vec);
#else
      if (USE_Q8K) {{ gate = gate_k; up = up_k; }}
#endif
      task->rhidden[route * HIDDEN + row] = tiny_silu_mul(gate, up);
      }}
    }} else {{
      const int shared_idx = idx - task->routed_work;
      const int input = shared_idx / HIDDEN, row = shared_idx - input * HIDDEN;
      const unsigned char *wg = task->sgate + row * (DIM / 32) * 34;
      const unsigned char *wu = task->sup + row * (DIM / 32) * 34;
#if defined(__AVX2__)
      __m256 gate_vec = _mm256_setzero_ps(), up_vec = _mm256_setzero_ps();
      for (int group = 0; group < DIM / 32; group++, wg += 34, wu += 34) {{
        const float scale = task->xd[input * (DIM / 32) + group];
        gate_vec = _mm256_fmadd_ps(_mm256_set1_ps(half_to_float(wg) * scale),
          _mm256_cvtepi32_ps(dot32_parts(_mm256_loadu_si256((const __m256i *)(wg + 2)),
                                         task->xq + input * DIM + group * 32)), gate_vec);
        up_vec = _mm256_fmadd_ps(_mm256_set1_ps(half_to_float(wu) * scale),
          _mm256_cvtepi32_ps(dot32_parts(_mm256_loadu_si256((const __m256i *)(wu + 2)),
                                         task->xq + input * DIM + group * 32)), up_vec);
      }}
      const float gate = hsum8f(gate_vec), up = hsum8f(up_vec);
#else
      float gate = 0.0f, up = 0.0f;
      for (int group = 0; group < DIM / 32; group++, wg += 34, wu += 34) {{
        const float scale = task->xd[input * (DIM / 32) + group];
        gate += half_to_float(wg) * scale *
                (float)dot32((const signed char *)(wg + 2), task->xq + input * DIM + group * 32);
        up += half_to_float(wu) * scale *
              (float)dot32((const signed char *)(wu + 2), task->xq + input * DIM + group * 32);
      }}
#endif
      task->shidden[shared_idx] = tiny_silu_mul(gate, up);
    }}
  }}
  if (task->stage == 1) for (int idx = task->begin; idx < task->end; idx++) {{
    const int row = idx / INPUTS, input = idx - row * INPUTS;
    float routed[ROUTES_PER_INPUT];
    for (int local_route = 0; local_route < ROUTES_PER_INPUT; local_route++) {{
      const int route = input * ROUTES_PER_INPUT + local_route;
      const int expert = task->sel[route];
      const unsigned char *w = task->rdown + ((expert * DIM + row) * (HIDDEN / 256)) * 136;
      float acc = 0.0f;
      for (int block = 0; block < HIDDEN / 256; block++, w += 136) {{
        const float d = half_to_float(w);
        const unsigned int high = (unsigned int)w[2] | ((unsigned int)w[3] << 8);
#if defined(__AVX2__)
        __m256i block_sum = _mm256_setzero_si256();
#else
        int block_sum = 0;
#endif
        for (int subgroup = 0; subgroup < 8; subgroup++) {{
          const int group = block * 8 + subgroup;
          const int low = (w[4 + (subgroup >> 1)] >> (4 * (subgroup & 1))) & 15;
          const int scale_bits = low | (((high >> (2 * subgroup)) & 3) << 4);
          if (USE_Q8K) {{
#if defined(__AVX2__)
            block_sum = _mm256_add_epi32(block_sum, _mm256_mullo_epi32(
            dot_iq4_parts(w + 8 + subgroup * 16, task->rhkq + route * HIDDEN + group * 32),
            _mm256_set1_epi32(scale_bits - 32)));
#else
            block_sum += (scale_bits - 32) *
              dot_iq4(w + 8 + subgroup * 16, task->rhkq + route * HIDDEN + group * 32);
#endif
          }} else acc += d * (float)(scale_bits - 32) * task->rhd[route * (HIDDEN / 32) + group] *
                        (float)dot_iq4(w + 8 + subgroup * 16, task->rhq + route * HIDDEN + group * 32);
        }}
#if defined(__AVX2__)
        if (USE_Q8K) acc += d * task->rhkd[route * (HIDDEN / 256) + block] * (float)hsum8(block_sum);
#else
        if (USE_Q8K) acc += d * task->rhkd[route * (HIDDEN / 256) + block] * (float)block_sum;
#endif
      }}
      routed[local_route] = acc;
    }}
    float routed_sum = routed[0] * task->probs[input * ROUTES_PER_INPUT];
    for (int route = 1; route < ROUTES_PER_INPUT; route++)
      routed_sum += routed[route] * task->probs[input * ROUTES_PER_INPUT + route];
    const unsigned char *w = task->sdown + row * (HIDDEN / 32) * 34;
#if defined(__AVX2__)
    __m256 shared_vec = _mm256_setzero_ps();
    for (int group = 0; group < HIDDEN / 32; group++, w += 34)
      shared_vec = _mm256_fmadd_ps(_mm256_set1_ps(half_to_float(w) * task->shd[input * (HIDDEN / 32) + group]),
        _mm256_cvtepi32_ps(dot32_parts(_mm256_loadu_si256((const __m256i *)(w + 2)),
                                       task->shq + input * HIDDEN + group * 32)), shared_vec);
    const float shared = hsum8f(shared_vec);
#else
    float shared = 0.0f;
    for (int group = 0; group < HIDDEN / 32; group++, w += 34)
      shared += half_to_float(w) * task->shd[input * (HIDDEN / 32) + group] *
                (float)dot32((const signed char *)(w + 2), task->shq + input * HIDDEN + group * 32);
#endif
    task->out[input * DIM + row] = routed_sum + shared * task->shared_scale[input];
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}(float *out, const unsigned char *rgate, const unsigned char *rup, const unsigned char *rdown,
            const unsigned char *sgate, const unsigned char *sup, const unsigned char *sdown,
            const float *x, const float *probs, const int *sel, const _Float16 *shared_gate) {{
  signed char xq[INPUTS * DIM], rhq[ROUTES * HIDDEN], shq[INPUTS * HIDDEN];
  float xd[INPUTS * (DIM / 32)], rhd[ROUTES * (HIDDEN / 32)], shd[INPUTS * (HIDDEN / 32)];
  signed char xkq[INPUTS * DIM], rhkq[ROUTES * HIDDEN];
  float xkd[INPUTS * (DIM / 256)], rhkd[ROUTES * (HIDDEN / 256)];
  float rhidden[ROUTES * HIDDEN], shidden[INPUTS * HIDDEN], shared_scale[INPUTS];
  int unique[ROUTES], head[ROUTES], next[ROUTES], unique_count = 0;
  for (int route = 0; route < ROUTES; route++) {{
    int unique_idx = 0;
    while (unique_idx < unique_count && unique[unique_idx] != sel[route]) unique_idx++;
    if (unique_idx == unique_count) {{
      unique[unique_count] = sel[route];
      head[unique_count++] = -1;
    }}
    next[route] = head[unique_idx];
    head[unique_idx] = route;
  }}
  for (int input = 0; input < INPUTS; input++) {{
    quantize(x + input * DIM, DIM, xq + input * DIM, xd + input * (DIM / 32));
    if (USE_Q8K) quantize_q8k(x + input * DIM, DIM, xkq + input * DIM, xkd + input * (DIM / 256));
    float gate_sum = 0.0f;
    for (int i = 0; i < DIM; i++) gate_sum += x[input * DIM + i] * (float)shared_gate[i];
    shared_scale[input] = 1.0f / (1.0f + expf(-gate_sum));
  }}
  task_t task = (task_t){{out, rhidden, shidden, rgate, rup, rdown, sgate, sup, sdown, x, probs, sel,
                           unique, head, next, xq, rhq, shq, xkq, rhkq, xd, rhd, shd, xkd, rhkd,
                           shared_scale, unique_count * HIDDEN, 0, 0, 0}};
  dispatch(task, task.routed_work + INPUTS * HIDDEN, THREADS);
  for (int route = 0; route < ROUTES; route++)
    quantize(rhidden + route * HIDDEN, HIDDEN, rhq + route * HIDDEN, rhd + route * (HIDDEN / 32));
  if (USE_Q8K) for (int route = 0; route < ROUTES; route++)
    quantize_q8k(rhidden + route * HIDDEN, HIDDEN, rhkq + route * HIDDEN, rhkd + route * (HIDDEN / 256));
  for (int input = 0; input < INPUTS; input++)
    quantize(shidden + input * HIDDEN, HIDDEN, shq + input * HIDDEN, shd + input * (HIDDEN / 32));
  task.stage = 1; dispatch(task, INPUTS * DIM, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def moe_kernel(out:UOp, rgate:UOp, rup:UOp, rdown:UOp, sgate:UOp, sup:UOp, sdown:UOp, x:UOp, probs:UOp,
                    sel:UOp, shared_gate:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out.base, rgate.base, rup.base, rdown.base, sgate.base, sup.base, sdown.base, x.base, probs.base, sel.base,
                  shared_gate.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=tuple(range(11)), outs=(0,), ins=tuple(range(1, 11))))

def expert_pair(first:ExpertWeights, second:ExpertWeights, sel:Tensor, x:Tensor) -> tuple[Tensor, Tensor]:
  assert first.ggml_type == second.ggml_type and first.ggml_type in (14, 21, 23)
  assert (first.in_features, first.out_features) == (second.in_features, second.out_features)
  input_count, routes = int(x.numel()) // first.in_features, int(sel.numel())
  routes_per_input = routes // input_count
  flat_sel, xc = sel.flatten().contiguous(), x.reshape(-1, first.in_features).contiguous()
  out0 = Tensor.empty(routes, first.out_features, dtype=x.dtype, device=x.device)
  out1 = Tensor.empty_like(out0)
  ggml_type = first.ggml_type
  outputs = Tensor.custom_kernel(out0, out1, first.weight, second.weight, flat_sel, xc,
    fxn=lambda out0,out1,raw0,raw1,sel,x:ggml_expert_pair_kernel(out0, out1, raw0, raw1, sel, x,
                                                                     ggml_type, routes_per_input))
  if len(sel.shape) == 1: return outputs[0], outputs[1]
  shape = (*sel.shape, first.out_features)
  return outputs[0].reshape(*shape), outputs[1].reshape(*shape)

def expert_silu(first:ExpertWeights, second:ExpertWeights, sel:Tensor, x:Tensor) -> Tensor:
  assert first.ggml_type == second.ggml_type and first.ggml_type in (14, 21, 23)
  assert (first.in_features, first.out_features) == (second.in_features, second.out_features) and x.dtype == dtypes.float32
  input_count, routes = int(x.numel()) // first.in_features, int(sel.numel())
  routes_per_input = routes // input_count
  flat_sel, xc = sel.flatten().contiguous(), x.reshape(-1, first.in_features).contiguous()
  out = Tensor.empty(routes, first.out_features, dtype=x.dtype, device=x.device)
  ggml_type = first.ggml_type
  out = Tensor.custom_kernel(out, first.weight, second.weight, flat_sel, xc,
    fxn=lambda out,raw0,raw1,sel,x:ggml_expert_silu_kernel(out, raw0, raw1, sel, x, ggml_type, routes_per_input))[0]
  return out if len(sel.shape) == 1 else out.reshape(*sel.shape, first.out_features)

@functools.cache
def weighted_sum_program(inputs:int, routes:int, dim:int, dtype:DType) -> tuple[str, bytes, str]:
  assert dtype in (dtypes.float16, dtypes.float32)
  name = f"cpu_weighted_sum_{inputs}_{routes}_{dim}_{dtype.name}"
  ctype = "_Float16" if dtype == dtypes.float16 else "float"
  threads = min(inputs * dim, 32, max(1, getenv("CPU_GGML_THREADS", CPU_COUNT.value)))
  src = f"""
#define INPUTS {inputs}
#define ROUTES {routes}
#define DIM {dim}
#define THREADS {threads}
typedef struct {{ {ctype} *out; const {ctype} *x, *probs; int begin, end; }} task_t;
static void *worker(void *opaque) {{
  task_t *task = (task_t *)opaque;
  for (int idx = task->begin; idx < task->end; idx++) {{
    const int input = idx / DIM, row = idx - input * DIM;
    float acc = 0.0f;
    for (int route = 0; route < ROUTES; route++)
      acc += (float)task->x[(input * ROUTES + route) * DIM + row] * (float)task->probs[input * ROUTES + route];
    task->out[idx] = ({ctype})acc;
  }}
  return (void *)0;
}}
{_SEM_POOL}
void {name}({ctype} *out, const {ctype} *x, const {ctype} *probs) {{
  dispatch((task_t){{out, x, probs, 0, INPUTS * DIM}}, INPUTS * DIM, THREADS);
}}
"""
  return src, _compile_cpu_ggml(src), name

def weighted_sum(x:Tensor, probs:Tensor) -> Tensor:
  inputs, routes, dim = int(probs.numel()) // probs.shape[-1], probs.shape[-1], x.shape[-1]
  assert x.shape[-2] == routes and x.dtype == probs.dtype and x.dtype in (dtypes.float16, dtypes.float32)
  out = Tensor.empty(inputs, dim, dtype=x.dtype, device=x.device)
  src, binary, name = weighted_sum_program(inputs, routes, dim, x.dtype)
  out = Tensor.custom_kernel(out, x.reshape(inputs, routes, dim).contiguous(), probs.reshape(inputs, routes).contiguous(),
    fxn=lambda out,x,probs:weighted_sum_kernel(out, x, probs, src, binary, name))[0]
  return out.reshape(*probs.shape[:-1], dim)

def weighted_sum_kernel(out:UOp, x:UOp, probs:UOp, src:str, binary:bytes, name:str) -> UOp:
  sink = UOp.sink(out.base, x.base, probs.base, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)),
             arg=ProgramInfo(name=name, local_size=(1, 1, 1), globals=(0, 1, 2), outs=(0,), ins=(1, 2)))

def moe_ffn(block:FFNBlock, x:Tensor, probs:Tensor, sel:Tensor) -> Tensor:
  routes, dim, hidden = int(sel.numel()), block.config.dim, block.config.hidden_dim
  inputs = int(x.numel()) // dim
  out = Tensor.empty(inputs, dim, dtype=dtypes.float32, device=x.device)
  src, binary, name = moe_program(routes, dim, hidden, inputs)
  tensors = (out, block.ffn_gate_exps.weight, block.ffn_up_exps.weight, block.ffn_down_exps.weight,
             block.ffn_gate_shexp.weight, block.ffn_up_shexp.weight, block.ffn_down_shexp.weight,
             x.flatten().contiguous(), probs.flatten().contiguous(), sel.flatten().contiguous(),
             block.ffn_gate_inp_shexp["weight"])
  out = Tensor.custom_kernel(*tensors, fxn=lambda out,rgate,rup,rdown,sgate,sup,sdown,x,probs,sel,shared_gate:
    moe_kernel(out, rgate, rup, rdown, sgate, sup, sdown, x, probs, sel, shared_gate, src, binary, name))[0]
  return out.reshape(*x.shape[:-1], dim)
