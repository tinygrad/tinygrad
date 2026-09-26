#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
  OP_NOP, OP_END, OP_BAR, OP_PREDT, OP_PREDF, OP_PREDE, OP_COV,
  OP_ADDU, OP_SUBU, OP_SHL, OP_SHR, OP_ASHR, OP_XOR, OP_AND, OP_OR, OP_MULL, OP_SHRG, OP_MADSH,
  OP_ADDF, OP_MULF, OP_MADF, OP_MAXF, OP_MINF, OP_FMOV, OP_RCP, OP_SQRT, OP_LOG2, OP_EXP2, OP_SIN, OP_COS, OP_FLOOR,
  OP_CMPU, OP_CMPS, OP_CMPF, OP_SEL, OP_LD, OP_ST, OP_ISAM, OP_STIB, OP_BR, OP_JUMP, OP_SWZ, OP_IMOV, OP_NOT, OP_BRAO, OP_BRAA,
  OP_MINU, OP_MAXU, OP_MINS, OP_MAXS, OP_SHLG, OP_SHRM, OP_SHLM, OP_ANDG, OP_RSQ, OP_CEIL, OP_TRUNC, OP_RNDNE,
  OP_MULS24, OP_MULU24, OP_CLZ, OP_SIGN,
};

typedef struct { uint32_t op, dst, extra, src_file, src_mod, src[4], ab; } Inst;

enum { F_FULL, F_HALF, F_CONST, F_IMM, F_PRED };
enum { TY_F16, TY_F32, TY_U16, TY_U32, TY_S16, TY_S32, TY_U8 };
#define NREG 1024
#define NHREG 1024
#define MAXTH 1024

static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float h2f(uint32_t h) { _Float16 x; uint16_t b = h; memcpy(&x, &b, 2); return (float)x; }
static uint32_t f2h(float f) { _Float16 x = (_Float16)f; uint16_t b; memcpy(&b, &x, 2); return b; }
static uint32_t apply_f(uint32_t b, uint32_t mod) { if (mod & 4) b &= 0x7fffffffu; if (mod & 2) b ^= 0x80000000u; return b; }
static uint32_t apply_i(uint32_t b, uint32_t mod) {
  if (mod & 4) { int32_t s = (int32_t)b; if (s < 0) b = (uint32_t)(-s); }
  if (mod & 2) b = (uint32_t)(-(int32_t)b);
  return b;
}

typedef struct { uint32_t *R, *C; uint16_t *H; uint8_t *P; int nconst; } Th;

static uint32_t raw(Th *t, const Inst *in, int s, int rep) {
  uint32_t mod = (in->src_mod >> (s * 8)) & 0xff, file = (in->src_file >> (s * 8)) & 0xff, idx = in->src[s] + ((mod & 1) ? (uint32_t)rep : 0);
  if (file == F_FULL) return idx < NREG ? t->R[idx] : 0;
  if (file == F_HALF) return idx < NHREG ? t->H[idx] : 0;
  if (file == F_CONST) return idx < (uint32_t)t->nconst ? t->C[idx] : 0;
  if (file == F_IMM) return in->src[s];
  return idx < 16 ? ((mod & 8) ? t->P[idx] ^ 1 : t->P[idx]) : (mod & 8 ? 1u : 0u);
}
static uint32_t ird(Th *t, const Inst *in, int s, int rep, int sext) {
  uint32_t mod = (in->src_mod >> (s * 8)) & 0xff, file = (in->src_file >> (s * 8)) & 0xff, v = raw(t, in, s, rep);
  if (file == F_HALF) v = sext ? (uint32_t)(int16_t)v : (v & 0xffff);
  return apply_i(v, mod);
}
static float frd(Th *t, const Inst *in, int s, int rep) {
  uint32_t mod = (in->src_mod >> (s * 8)) & 0xff, file = (in->src_file >> (s * 8)) & 0xff, v = raw(t, in, s, rep);
  if (file == F_HALF) v = f2u(h2f(v));
  return u2f(apply_f(v, mod));
}
static void wd(Th *t, uint32_t packed_dst, int rep, uint32_t v) {
  uint32_t dst = (packed_dst & 0xffff) + (uint32_t)rep, file = packed_dst >> 16;
  if (file == F_FULL) { if (dst < NREG) t->R[dst] = v; }
  else if (file == F_HALF) { if (dst < NHREG) t->H[dst] = (uint16_t)v; }
  else if (dst < 16) t->P[dst] = v & 1;
}
static void wf(Th *t, const Inst *in, int rep, float f) {
  if (in->op & 0x10000u) f = fminf(1.f, fmaxf(0.f, f));
  wd(t, in->dst, rep, (in->dst >> 16) == F_HALF ? f2h(f) : f2u(f));
}

static uint32_t cov(uint32_t v, int st, int dt) {
  double num = 0; int64_t inum = 0; int flt = st == TY_F16 || st == TY_F32;
  if (st == TY_F16) num = h2f(v);
  else if (st == TY_F32) num = u2f(v);
  else if (st == TY_U16) inum = v & 0xffff;
  else if (st == TY_U8) inum = (dt == TY_S16 || dt == TY_S32) ? (int8_t)v : (int64_t)(v & 0xff);
  else if (st == TY_S16) inum = (int16_t)v;
  else if (st == TY_S32) inum = (int32_t)v;
  else inum = v;
  if (dt == TY_F16) return f2h(flt ? (float)num : (float)inum);
  if (dt == TY_F32) return f2u(flt ? (float)num : (float)inum);
  int64_t r = flt ? (int64_t)trunc(num) : inum;
  if (dt == TY_U16 || dt == TY_S16) return (uint16_t)r;
  if (dt == TY_U8) return (uint8_t)r;
  return (uint32_t)r;
}
static int cmpk(int lt, int eq, int cond) {
  switch (cond & 7) {
    case 0: return lt; case 1: return lt || eq; case 2: return !lt && !eq; case 3: return !lt; case 4: return eq; default: return !eq;
  }
}

static uint64_t *g_maps; static int g_nmaps;
__attribute__((visibility("default"))) void ir3_set_maps(uint64_t *maps, int n) { g_maps = maps; g_nmaps = n; }
static int mapped(uint64_t a, uint64_t n) {
  if (!g_nmaps) return 1;
  for (int i = 0; i < g_nmaps; i++) {
    uint64_t b = g_maps[(size_t)i * 2], s = g_maps[(size_t)i * 2 + 1];
    if (a >= b && n <= s && a - b <= s - n) return 1;
  }
  return 0;
}

static uint64_t addr64(Th *t, uint32_t idx) {
  return (uint64_t)(idx < NREG ? t->R[idx] : 0) | ((uint64_t)(idx + 1 < NREG ? t->R[idx + 1] : 0) << 32);
}
static void ldn(Th *t, uint32_t packed_dst, int n, int el, uint64_t base) {
  uint32_t dst = packed_dst & 0xffff, file = packed_dst >> 16;
  for (int i = 0; i < n; i++) {
    uint32_t v = 0;
    if (el == 4) { memcpy(&v, (void *)(base + (uint64_t)i * 4), 4); if (file == F_HALF) v = f2h(u2f(v)); }
    else if (el == 2) { uint16_t h; memcpy(&h, (void *)(base + (uint64_t)i * 2), 2); v = (file == F_FULL) ? f2u(h2f(h)) : h; }
    else { uint8_t b; memcpy(&b, (void *)(base + i), 1); v = b; }
    if (file == F_HALF) { if (dst + i < NHREG) t->H[dst + i] = (uint16_t)v; }
    else if (dst + i < NREG) t->R[dst + i] = v;
  }
}
static void stn(Th *t, const Inst *in, int rep, int n, int el, uint64_t base) {
  uint32_t mod = in->src_mod & 0xff, file = in->src_file & 0xff, idx = in->src[0] + ((mod & 1) ? (uint32_t)rep : 0);
  for (int i = 0; i < n; i++) {
    uint32_t v = file == F_HALF ? (idx + i < NHREG ? t->H[idx + i] : 0) : (idx + i < NREG ? t->R[idx + i] : 0);
    uint64_t a = base + (uint64_t)i * el;
    if (el == 4 && file == F_HALF) { uint32_t b = f2u(h2f(v)); memcpy((void *)a, &b, 4); }
    else if (el == 2 && file == F_FULL) { uint16_t h = f2h(u2f(v)); memcpy((void *)a, &h, 2); }
    else if (el == 4) memcpy((void *)a, &v, 4);
    else if (el == 2) { uint16_t h = v; memcpy((void *)a, &h, 2); }
    else { uint8_t b = v; memcpy((void *)a, &b, 1); }
  }
}
static void zeron(Th *t, uint32_t packed_dst, int n) {
  for (int i = 0; i < n; i++) wd(t, (packed_dst & 0xffff0000u) | ((packed_dst & 0xffff) + i), 0, 0);
}

static uint32_t fld(uint32_t v, uint32_t mask, uint32_t shift) { return (v & mask) >> shift; }
static void desc(uint64_t table, int slot, uint64_t *base, uint32_t *w, uint32_t *h, uint32_t *pitch, int *el) {
  const uint32_t *d = (const uint32_t *)(table + (uint64_t)slot * 0x40);
  *w = fld(d[1], A6XX_TEX_CONST_1_WIDTH__MASK, A6XX_TEX_CONST_1_WIDTH__SHIFT);
  *h = fld(d[1], A6XX_TEX_CONST_1_HEIGHT__MASK, A6XX_TEX_CONST_1_HEIGHT__SHIFT);
  *pitch = fld(d[2], A6XX_TEX_CONST_2_PITCH__MASK, A6XX_TEX_CONST_2_PITCH__SHIFT);
  memcpy(base, d + 4, 8);
  *el = fld(d[0], A6XX_TEX_CONST_0_FMT__MASK, A6XX_TEX_CONST_0_FMT__SHIFT) == FMT6_16_16_16_16_FLOAT ? 2 : 4;
  if (!*pitch) *pitch = (*w) * 4 * (*el);
}
static void image(Th *t, const Inst *in, int rep, int store, uint64_t table) {
  int n = in->extra & 0xff, dim = (in->extra >> 8) & 0xff, el, slot = in->ab & 0xffff;
  uint64_t base; uint32_t w, h, pitch;
  if (!table) { if (!store) zeron(t, in->dst, n); return; }
  desc(table, slot, &base, &w, &h, &pitch, &el);
  int cs = store ? 1 : 0;
  uint32_t cidx = in->src[cs] + ((((in->src_mod >> (cs * 8)) & 1)) ? (uint32_t)rep : 0);
  uint32_t x = cidx < NREG ? t->R[cidx] : 0, y = (dim > 1 && cidx + 1 < NREG) ? t->R[cidx + 1] : 0;
  uint32_t texel = 4u * el;
  uint64_t a; int oob;
  if (dim <= 1) { oob = (int32_t)x < 0 || (w && h && x >= w * h); a = base + (uint64_t)x * texel; }
  else { oob = (int32_t)x < 0 || (int32_t)y < 0 || (w && x >= w) || (h && y >= h); a = base + (uint64_t)y * pitch + (uint64_t)x * texel; }
  if (oob || !mapped(a, (uint64_t)n * el)) { if (!store) zeron(t, in->dst, n); return; }
  if (store) stn(t, in, rep, n, el, a); else ldn(t, in->dst, n, el, a);
}

static int step(Th *t, const Inst *in, int rep, uint8_t *shared, uint32_t shsz, uint8_t *priv, uint32_t psz, uint64_t tex, uint64_t uav) {
  uint32_t op = in->op & 0xff;
  switch (op) {
    case OP_NOP: case OP_BAR: case OP_END: case OP_PREDT: case OP_PREDF: case OP_PREDE: return 0;
    case OP_COV: {
      int st = in->ab & 0xff, dt = (in->ab >> 8) & 0xff;
      wd(t, in->dst, rep, cov(st <= TY_F32 ? raw(t, in, 0, rep) : ird(t, in, 0, rep, 0), st, dt)); return 0;
    }
    case OP_ADDU: wd(t, in->dst, rep, ird(t, in, 0, rep, 0) + ird(t, in, 1, rep, 0)); return 0;
    case OP_SUBU: wd(t, in->dst, rep, ird(t, in, 0, rep, 0) - ird(t, in, 1, rep, 0)); return 0;
    case OP_SHL: wd(t, in->dst, rep, ird(t, in, 0, rep, 0) << (ird(t, in, 1, rep, 0) & 31)); return 0;
    case OP_SHR: wd(t, in->dst, rep, ird(t, in, 0, rep, 0) >> (ird(t, in, 1, rep, 0) & 31)); return 0;
    case OP_ASHR: wd(t, in->dst, rep, (uint32_t)((int32_t)ird(t, in, 0, rep, 1) >> (ird(t, in, 1, rep, 0) & 31))); return 0;
    case OP_XOR: wd(t, in->dst, rep, ird(t, in, 0, rep, 0) ^ ird(t, in, 1, rep, 0)); return 0;
    case OP_AND: wd(t, in->dst, rep, ird(t, in, 0, rep, 0) & ird(t, in, 1, rep, 0)); return 0;
    case OP_OR: wd(t, in->dst, rep, ird(t, in, 0, rep, 0) | ird(t, in, 1, rep, 0)); return 0;
    case OP_MULL: wd(t, in->dst, rep, (ird(t, in, 0, rep, 0) & 0xffff) * (ird(t, in, 1, rep, 0) & 0xffff)); return 0;
    case OP_MULU24: {
      uint32_t a = ird(t, in, 0, rep, 0) & 0xffffff, b = ird(t, in, 1, rep, 0) & 0xffffff;
      wd(t, in->dst, rep, (uint32_t)((uint64_t)a * b)); return 0;
    }
    case OP_MULS24: {
      uint32_t xa = ird(t, in, 0, rep, 1) & 0xffffff, xb = ird(t, in, 1, rep, 1) & 0xffffff;
      int32_t a = (xa & 0x800000) ? (int32_t)(xa | 0xff000000u) : (int32_t)xa;
      int32_t b = (xb & 0x800000) ? (int32_t)(xb | 0xff000000u) : (int32_t)xb;
      wd(t, in->dst, rep, (uint32_t)((int64_t)a * b)); return 0;
    }
    case OP_CLZ: { uint32_t v = ird(t, in, 0, rep, 0); wd(t, in->dst, rep, v ? __builtin_clz(v) : 0xffffffffu); return 0; }
    case OP_SHRG: case OP_SHLG: case OP_SHRM: case OP_SHLM: {
      uint32_t s = ird(t, in, 0, rep, 0) & 31, v = ird(t, in, 1, rep, 0), m = ird(t, in, 2, rep, 0);
      uint32_t sh = (op == OP_SHLG || op == OP_SHLM) ? v << s : v >> s;
      wd(t, in->dst, rep, (op == OP_SHRG || op == OP_SHLG) ? sh | m : sh & m); return 0;
    }
    case OP_ANDG: wd(t, in->dst, rep, (ird(t, in, 1, rep, 0) & ird(t, in, 0, rep, 0)) | ird(t, in, 2, rep, 0)); return 0;
    case OP_MADSH: {
      uint32_t x = ird(t, in, 0, rep, 0) & 0xffff, y = ird(t, in, 1, rep, 0) >> 16;
      wd(t, in->dst, rep, (uint32_t)((((uint64_t)x * y) << 16) + ird(t, in, 2, rep, 0))); return 0;
    }
    case OP_ADDF: wf(t, in, rep, frd(t, in, 0, rep) + frd(t, in, 1, rep)); return 0;
    case OP_MULF: wf(t, in, rep, frd(t, in, 0, rep) * frd(t, in, 1, rep)); return 0;
    case OP_MADF: wf(t, in, rep, fmaf(frd(t, in, 0, rep), frd(t, in, 1, rep), frd(t, in, 2, rep))); return 0;
    case OP_MAXF: wf(t, in, rep, fmaxf(frd(t, in, 0, rep), frd(t, in, 1, rep))); return 0;
    case OP_MINF: wf(t, in, rep, fminf(frd(t, in, 0, rep), frd(t, in, 1, rep))); return 0;
    case OP_MINU: { uint32_t a = ird(t, in, 0, rep, 0), b = ird(t, in, 1, rep, 0); wd(t, in->dst, rep, a < b ? a : b); return 0; }
    case OP_MAXU: { uint32_t a = ird(t, in, 0, rep, 0), b = ird(t, in, 1, rep, 0); wd(t, in->dst, rep, a > b ? a : b); return 0; }
    case OP_MINS: {
      int32_t a = (int32_t)ird(t, in, 0, rep, 1), b = (int32_t)ird(t, in, 1, rep, 1);
      wd(t, in->dst, rep, a < b ? (uint32_t)a : (uint32_t)b); return 0;
    }
    case OP_MAXS: {
      int32_t a = (int32_t)ird(t, in, 0, rep, 1), b = (int32_t)ird(t, in, 1, rep, 1);
      wd(t, in->dst, rep, a > b ? (uint32_t)a : (uint32_t)b); return 0;
    }
    case OP_FMOV: wf(t, in, rep, frd(t, in, 0, rep)); return 0;
    case OP_RCP: wf(t, in, rep, 1.f / frd(t, in, 0, rep)); return 0;
    case OP_SQRT: wf(t, in, rep, sqrtf(frd(t, in, 0, rep))); return 0;
    case OP_RSQ: wf(t, in, rep, 1.f / sqrtf(frd(t, in, 0, rep))); return 0;
    case OP_CEIL: wf(t, in, rep, ceilf(frd(t, in, 0, rep))); return 0;
    case OP_TRUNC: wf(t, in, rep, truncf(frd(t, in, 0, rep))); return 0;
    case OP_RNDNE: wf(t, in, rep, rintf(frd(t, in, 0, rep))); return 0;
    case OP_LOG2: wf(t, in, rep, log2f(frd(t, in, 0, rep))); return 0;
    case OP_EXP2: wf(t, in, rep, exp2f(frd(t, in, 0, rep))); return 0;
    case OP_SIN: wf(t, in, rep, sinf(frd(t, in, 0, rep))); return 0;
    case OP_COS: wf(t, in, rep, cosf(frd(t, in, 0, rep))); return 0;
    case OP_FLOOR: wf(t, in, rep, floorf(frd(t, in, 0, rep))); return 0;
    case OP_SIGN: { float s = frd(t, in, 0, rep); wf(t, in, rep, s > 0.f ? 1.f : s < 0.f ? -1.f : s); return 0; }
    case OP_CMPU: { uint32_t a = ird(t, in, 0, rep, 0), b = ird(t, in, 1, rep, 0); wd(t, in->dst, rep, cmpk(a < b, a == b, in->extra)); return 0; }
    case OP_CMPS: {
      int32_t a = (int32_t)ird(t, in, 0, rep, 1), b = (int32_t)ird(t, in, 1, rep, 1);
      wd(t, in->dst, rep, cmpk(a < b, a == b, in->extra)); return 0;
    }
    case OP_CMPF: { float a = frd(t, in, 0, rep), b = frd(t, in, 1, rep); wd(t, in->dst, rep, cmpk(a < b, a == b, in->extra)); return 0; }
    case OP_SEL: wd(t, in->dst, rep, ird(t, in, ird(t, in, 1, rep, 0) ? 0 : 2, rep, 0)); return 0;
    case OP_LD: case OP_ST: {
      int space = in->ab & 0xff, n = in->extra & 0xff, el = (in->ab >> 16) & 0xff, s = op == OP_ST ? 1 : 0;
      uint32_t idx = in->src[s] + (((in->src_mod >> (s * 8)) & 1) ? (uint32_t)rep : 0);
      uint64_t base = (space ? raw(t, in, s, rep) : addr64(t, idx)) + (in->extra >> 16);
      uint8_t *mem = space == 1 ? shared : space == 2 ? priv : NULL;
      uint32_t msz = space == 1 ? shsz : psz;
      if (mem) { if (base + (uint64_t)n * el > msz) return 0; base += (uint64_t)(uintptr_t)mem; }
      else if (!mapped(base, (uint64_t)n * el)) { if (op == OP_LD) zeron(t, in->dst + rep, n); return 0; }
      if (op == OP_LD) ldn(t, in->dst + rep, n, el, base); else stn(t, in, rep, n, el, base);
      return 0;
    }
    case OP_ISAM: image(t, in, rep, 0, tex); return 0;
    case OP_STIB: image(t, in, rep, 1, uav); return 0;
    case OP_IMOV: wd(t, in->dst, rep, ird(t, in, 0, rep, 1)); return 0;
    case OP_NOT: wd(t, in->dst, rep, ~ird(t, in, 0, rep, 0)); return 0;
    case OP_SWZ: {
      int st = in->ab & 0xff, dt = (in->ab >> 8) & 0xff;
      uint32_t a = cov(st <= TY_F32 ? raw(t, in, 0, rep) : ird(t, in, 0, rep, 0), st, dt);
      uint32_t b = cov(st <= TY_F32 ? raw(t, in, 1, rep) : ird(t, in, 1, rep, 0), st, dt);
      wd(t, in->dst, rep, a); wd(t, in->extra, rep, b); return 0;
    }
    default: return (int)op;
  }
}

static void preload(uint32_t *R, uint32_t lid, uint32_t wgid, uint32_t x, uint32_t y, uint32_t z, uint32_t gx, uint32_t gy, uint32_t gz) {
  if ((lid & 0xff) != 0xfc && lid + 2 < NREG) { R[lid] = x; R[lid + 1] = y; R[lid + 2] = z; }
  if ((wgid & 0xff) != 0xfc && wgid + 2 < NREG) { R[wgid] = gx; R[wgid + 1] = gy; R[wgid + 2] = gz; }
}

static int is_br(uint32_t op) { return op == OP_JUMP || op == OP_BR || op == OP_BRAO || op == OP_BRAA; }
static int taken(const Inst *in, Th *t) {
  uint32_t op = in->op & 0xff, a = raw(t, in, 0, 0) & 1, b = raw(t, in, 1, 0) & 1;
  if (op == OP_JUMP) return 1;
  if (op == OP_BR) return a;
  if (op == OP_BRAO) return a || b;
  if (op == OP_BRAA) return a && b;
  return 0;
}

static void pred(int op, uint8_t *mask, uint8_t *stk, int *sp, int stride, uint8_t p) {
  if (op == OP_PREDE) { if (*sp) *mask = stk[--(*sp) * stride] & 1; return; }
  uint8_t top = *sp ? stk[(*sp - 1) * stride] : 0; int take = op == OP_PREDT ? p : !p;
  if (*sp && !(top & 4) && ((top >> 1) & 1) == (op == OP_PREDT)) { stk[(*sp - 1) * stride] = top | 4; *mask = (top & 1) && take; }
  else if (*sp < 8) { stk[(*sp)++ * stride] = (*mask & 1) | (op == OP_PREDF ? 2 : 0); *mask = *mask && take; }
}

static int fiber_step(int *pc, uint8_t *mask, uint8_t *stk, int *sp, int nt, int tid, Th *t, const Inst *insts, int ninst,
                      uint8_t *shared, uint32_t shsz, uint8_t *priv, uint32_t psz, uint64_t tex, uint64_t uav) {
  if (*pc < 0 || *pc >= ninst) return 1;
  const Inst *in = &insts[*pc]; uint32_t op = in->op & 0xff;
  if (op == OP_END) return 1;
  if (op == OP_BAR) return 2;
  if (is_br(op)) { *pc += (*mask && taken(in, t)) ? (int)(int32_t)in->extra : 1; return 0; }
  if (op == OP_PREDT || op == OP_PREDF || op == OP_PREDE) { pred(op, mask, stk + tid, sp, nt, t->P[0]); (*pc)++; return 0; }
  if (*mask && op != OP_NOP)
    for (uint32_t rep = 0; rep <= (in->op >> 24); rep++) { int rc = step(t, in, (int)rep, shared, shsz, priv, psz, tex, uav); if (rc) return rc; }
  (*pc)++;
  return 0;
}

static int run_thread(const Inst *insts, int ninst, Th *t, uint8_t *shared, uint32_t shsz, uint8_t *priv, uint32_t psz, uint64_t tex, uint64_t uav) {
  uint8_t mask = 1, stk[8]; int sp = 0, pc = 0, steps = 0;
  while (pc >= 0 && pc < ninst) {
    if (++steps > 4000000) return -4;
    const Inst *in = &insts[pc]; uint32_t op = in->op & 0xff;
    if (op == OP_END) return 0;
    if (is_br(op)) { pc += (mask && taken(in, t)) ? (int)(int32_t)in->extra : 1; continue; }
    if (op == OP_PREDT || op == OP_PREDF || op == OP_PREDE) { pred(op, &mask, stk, &sp, 1, t->P[0]); pc++; continue; }
    if (mask && op != OP_BAR && op != OP_NOP)
      for (uint32_t rep = 0; rep <= (in->op >> 24); rep++) { int rc = step(t, in, (int)rep, shared, shsz, priv, psz, tex, uav); if (rc) return rc; }
    pc++;
  }
  return 0;
}

__attribute__((visibility("default")))
int ir3_launch(const Inst *insts, int ninst, uint32_t *consts, int nconst, uint32_t gx, uint32_t gy, uint32_t gz,
               uint32_t lx, uint32_t ly, uint32_t lz, uint32_t lid, uint32_t wgid, uint32_t shsz, uint32_t psz, uint64_t tex, uint64_t uav) {
  uint64_t nth = (uint64_t)lx * ly * lz;
  if (!nth || nth > MAXTH) return -2;
  int sync = 0, uses_p = 0, nt = (int)nth;
  for (int i = 0; i < ninst; i++) {
    uint32_t op = insts[i].op & 0xff, space = insts[i].ab & 0xff;
    if (op == OP_BAR || ((op == OP_LD || op == OP_ST) && space == 1)) sync = 1;
    if ((op == OP_LD || op == OP_ST) && space == 2) uses_p = 1;
  }
  if (uses_p && !psz) psz = 1u << 16;
  uint32_t item = uses_p ? psz : 0;
  if (!sync) {
    uint8_t *pmem = item ? calloc(item, 1) : NULL;
    if (item && !pmem) return -3;
    uint32_t R[NREG]; uint16_t H[NHREG]; uint8_t P[16]; Th t = {R, consts, H, P, nconst};
    int rc = 0;
    for (uint32_t z = 0; z < gz && !rc; z++) for (uint32_t y = 0; y < gy && !rc; y++) for (uint32_t x = 0; x < gx && !rc; x++)
      for (int tid = 0; tid < nt && !rc; tid++) {
        memset(R, 0, sizeof(R)); memset(H, 0, sizeof(H)); memset(P, 0, sizeof(P)); if (pmem) memset(pmem, 0, item);
        preload(R, lid, wgid, (uint32_t)(tid % lx), (uint32_t)((tid / lx) % ly), (uint32_t)(tid / (lx * ly)), x, y, z);
        rc = run_thread(insts, ninst, &t, NULL, 0, pmem, item, tex, uav);
      }
    free(pmem);
    return rc;
  }
  uint32_t *R = calloc((size_t)nt * NREG, 4); uint16_t *H = calloc((size_t)nt * NHREG, 2);
  uint8_t *P = calloc((size_t)nt * 16, 1), *M = malloc(nt), *stk = calloc((size_t)8 * nt, 1), *done = calloc(nt, 1);
  uint8_t *shared = calloc(shsz ? shsz : 1, 1), *pmem = item ? calloc((size_t)nt * item, 1) : NULL;
  int *pcs = calloc(nt, sizeof(int)), *sps = calloc(nt, sizeof(int));
  if (!R || !H || !P || !M || !stk || !done || !shared || !pcs || !sps || (item && !pmem)) {
    free(R); free(H); free(P); free(M); free(stk); free(done); free(shared); free(pmem); free(pcs); free(sps); return -3;
  }
  int rc = 0;
  for (uint32_t z = 0; z < gz && !rc; z++) for (uint32_t y = 0; y < gy && !rc; y++) for (uint32_t x = 0; x < gx && !rc; x++) {
    memset(R, 0, (size_t)nt * NREG * 4); memset(H, 0, (size_t)nt * NHREG * 2); memset(P, 0, (size_t)nt * 16);
    memset(shared, 0, shsz); memset(M, 1, nt); memset(done, 0, nt);
    memset(pcs, 0, (size_t)nt * sizeof(int)); memset(sps, 0, (size_t)nt * sizeof(int));
    if (pmem) memset(pmem, 0, (size_t)nt * item);
    for (int tid = 0; tid < nt; tid++)
      preload(R + (size_t)tid * NREG, lid, wgid, (uint32_t)(tid % lx), (uint32_t)((tid / lx) % ly), (uint32_t)(tid / (lx * ly)), x, y, z);
    int live = nt, steps = 0;
    while (live && !rc) {
      if (++steps > 4000000) { rc = -4; break; }
      int moved = 0, bars = 0;
      for (int tid = 0; tid < nt && !rc; tid++) if (!done[tid]) {
        Th t = {R + (size_t)tid * NREG, consts, H + (size_t)tid * NHREG, P + (size_t)tid * 16, nconst};
        int s = fiber_step(&pcs[tid], &M[tid], stk, &sps[tid], nt, tid, &t, insts, ninst, shared, shsz,
                           pmem ? pmem + (size_t)tid * item : NULL, item, tex, uav);
        if (s < 0) rc = s;
        else if (s == 1) { done[tid] = 1; live--; moved = 1; }
        else if (s == 2) bars++;
        else moved = 1;
      }
      if (!rc && !moved && bars) for (int tid = 0; tid < nt; tid++) if (!done[tid]) pcs[tid]++;
    }
  }
  free(R); free(H); free(P); free(M); free(stk); free(done); free(shared); free(pmem); free(pcs); free(sps);
  return rc;
}
