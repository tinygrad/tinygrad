#include <arm_neon.h>
#include <omp.h>
#include <stdint.h>

void set_threads(int n) { omp_set_num_threads(n); }

static inline void transform1v(float32x4_t out[7], const float32x4_t x[4], int side) {
  out[0] = vaddq_f32(x[0], x[3]);
  if (!side) {
    out[1] = vaddq_f32(x[2], x[3]); out[2] = x[0]; out[3] = x[3];
    out[4] = vaddq_f32(x[0], x[1]); out[5] = vsubq_f32(x[2], x[0]); out[6] = vsubq_f32(x[1], x[3]);
  } else {
    out[1] = x[0]; out[2] = vsubq_f32(x[1], x[3]); out[3] = vsubq_f32(x[2], x[0]);
    out[4] = x[3]; out[5] = vaddq_f32(x[0], x[1]); out[6] = vaddq_f32(x[2], x[3]);
  }
}

void transform2_f16(_Float16 *restrict out, const _Float16 *restrict in, int n, int side) {
  const int leaf = n / 4;
  #pragma omp parallel for schedule(dynamic, 1)
  for (int r = 0; r < leaf; r++) for (int c = 0; c < leaf; c += 4) {
    float32x4_t v[4][4], inner[4][7], outer[7];
    for (int q0 = 0; q0 < 4; q0++) for (int q1 = 0; q1 < 4; q1++) {
      const int rb = ((q0 >> 1) << 1) | (q1 >> 1);
      const int cb = ((q0 & 1) << 1) | (q1 & 1);
      const float16x4_t h = vld1_f16((const float16_t *)(in + (int64_t)(rb * leaf + r) * n + cb * leaf + c));
      v[q0][q1] = vcvt_f32_f16(h);
    }
    for (int q0 = 0; q0 < 4; q0++) transform1v(inner[q0], v[q0], side);
    for (int p1 = 0; p1 < 7; p1++) {
      float32x4_t column[4];
      for (int q0 = 0; q0 < 4; q0++) column[q0] = inner[q0][p1];
      transform1v(outer, column, side);
      for (int p0 = 0; p0 < 7; p0++) {
        const float16x4_t h = vcvt_f16_f32(outer[p0]);
        vst1_f16((float16_t *)(out + ((int64_t)p0 * 7 + p1) * leaf * leaf + (int64_t)r * leaf + c), h);
      }
    }
  }
}

static inline void combine1v(float32x4_t out[4], const float32x4_t m[7]) {
  out[0] = vaddq_f32(vsubq_f32(vaddq_f32(m[0], m[3]), m[4]), m[6]);
  out[1] = vaddq_f32(m[2], m[4]);
  out[2] = vaddq_f32(m[1], m[3]);
  out[3] = vaddq_f32(vaddq_f32(vsubq_f32(m[0], m[1]), m[2]), m[5]);
}

void combine2_f16(_Float16 *restrict out, const _Float16 *restrict in, int leaf) {
  const int n = leaf * 4;
  #pragma omp parallel for schedule(dynamic, 1)
  for (int r = 0; r < leaf; r++) for (int c = 0; c < leaf; c += 4) {
    float32x4_t inner[4][7];
    for (int p0 = 0; p0 < 7; p0++) {
      float32x4_t m[7], d[4];
      for (int p1 = 0; p1 < 7; p1++) {
        const float16x4_t h = vld1_f16((const float16_t *)(in + ((int64_t)p0 * 7 + p1) * leaf * leaf + (int64_t)r * leaf + c));
        m[p1] = vcvt_f32_f16(h);
      }
      combine1v(d, m);
      for (int u = 0; u < 4; u++) inner[u][p0] = d[u];
    }
    for (int u = 0; u < 4; u++) {
      float32x4_t d[4];
      combine1v(d, inner[u]);
      for (int v = 0; v < 4; v++) {
        const int br = (v >> 1) * 2 + (u >> 1);
        const int bc = (v & 1) * 2 + (u & 1);
        const float16x4_t h = vcvt_f16_f32(d[v]);
        vst1_f16((float16_t *)(out + (int64_t)(br * leaf + r) * n + bc * leaf + c), h);
      }
    }
  }
}

void cache_clean(void *ptr, int64_t size) {
  uintptr_t p = (uintptr_t)ptr & ~(uintptr_t)63;
  const uintptr_t end = ((uintptr_t)ptr + size + 63) & ~(uintptr_t)63;
  for (; p < end; p += 64) __asm__ volatile("dc cvac, %0" :: "r"(p) : "memory");
  __asm__ volatile("dsb sy" ::: "memory");
}

void cache_invalidate(void *ptr, int64_t size) {
  uintptr_t p = (uintptr_t)ptr & ~(uintptr_t)63;
  const uintptr_t end = ((uintptr_t)ptr + size + 63) & ~(uintptr_t)63;
  for (; p < end; p += 64) __asm__ volatile("dc civac, %0" :: "r"(p) : "memory");
  __asm__ volatile("dsb sy" ::: "memory");
}
