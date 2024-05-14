#include "common.h"
#include <stdint.h>

// https://github.com/JuliaMath/openlibm/blob/master/src/

static const double D_Lg1 = 6.666666666666735130e-01, /* 3FE55555 55555593 */
    D_Lg2 = 3.999999999940941908e-01,                 /* 3FD99999 9997FA04 */
    D_Lg3 = 2.857142874366239149e-01,                 /* 3FD24924 94229359 */
    D_Lg4 = 2.222219843214978396e-01,                 /* 3FCC71C5 1D8E78AF */
    D_Lg5 = 1.818357216161805012e-01,                 /* 3FC74664 96CB03DE */
    D_Lg6 = 1.531383769920937332e-01,                 /* 3FC39A09 D078C69F */
    D_Lg7 = 1.479819860511658591e-01;                 /* 3FC2F112 DF3E5244 */

/*
 * We always inline k_log1p(), since doing so produces a
 * substantial performance improvement (~40% on amd64).
 */
static inline double k_log1p(double f) {
  double hfsq, s, z, R, w, t1, t2;

  s = f / (2.0 + f);
  z = s * s;
  w = z * z;
  t1 = w * (D_Lg2 + w * (D_Lg4 + w * D_Lg6));
  t2 = z * (D_Lg1 + w * (D_Lg3 + w * (D_Lg5 + w * D_Lg7)));
  R = t2 + t1;
  hfsq = 0.5 * f * f;
  return s * (hfsq + R);
}

static const double d_two54 =
                        1.80143985094819840000e+16, /* 0x43500000, 0x00000000 */
    d_ivln2hi = 1.44269504072144627571e+00,         /* 0x3ff71547, 0x65200000 */
    d_ivln2lo = 1.67517131648865118353e-10;         /* 0x3de705fc, 0x2eefa200 */

static const double d_zero = 0.0;

double log2(double x) {
  double f, hfsq, hi, lo, r, val_hi, val_lo, w, y;
  int32_t i, k, hx;
  uint32_t lx;

  EXTRACT_WORDS(hx, lx, x);

  k = 0;
  if (hx < 0x00100000) { /* x < 2**-1022  */
    if (((hx & 0x7fffffff) | lx) == 0)
      return -d_two54 / d_zero; /* log(+-0)=-inf */
    if (hx < 0)
      return (x - x) / d_zero; /* log(-#) = NaN */
    k -= 54;
    x *= d_two54; /* subnormal number, scale up x */
    GET_HIGH_WORD(hx, x);
  }
  if (hx >= 0x7ff00000)
    return x + x;
  if (hx == 0x3ff00000 && lx == 0)
    return d_zero; /* log(1) = +0 */
  k += (hx >> 20) - 1023;
  hx &= 0x000fffff;
  i = (hx + 0x95f64) & 0x100000;
  SET_HIGH_WORD(x, hx | (i ^ 0x3ff00000)); /* normalize x or x/2 */
  k += (i >> 20);
  y = (double)k;
  f = x - 1.0;
  hfsq = 0.5 * f * f;
  r = k_log1p(f);

  /*
   * f-hfsq must (for args near 1) be evaluated in extra precision
   * to avoid a large cancellation when x is near sqrt(2) or 1/sqrt(2).
   * This is fairly efficient since f-hfsq only depends on f, so can
   * be evaluated in parallel with R.  Not combining hfsq with R also
   * keeps R small (though not as small as a true `lo' term would be),
   * so that extra precision is not needed for terms involving R.
   *
   * Compiler bugs involving extra precision used to break Dekker's
   * theorem for spitting f-hfsq as hi+lo, unless double_t was used
   * or the multi-precision calculations were avoided when double_t
   * has extra precision.  These problems are now automatically
   * avoided as a side effect of the optimization of combining the
   * Dekker splitting step with the clear-low-bits step.
   *
   * y must (for args near sqrt(2) and 1/sqrt(2)) be added in extra
   * precision to avoid a very large cancellation when x is very near
   * these values.  Unlike the above cancellations, this problem is
   * specific to base 2.  It is strange that adding +-1 is so much
   * harder than adding +-ln2 or +-log10_2.
   *
   * This uses Dekker's theorem to normalize y+val_hi, so the
   * compiler bugs are back in some configurations, sigh.  And I
   * don't want to used double_t to avoid them, since that gives a
   * pessimization and the support for avoiding the pessimization
   * is not yet available.
   *
   * The multi-precision calculations for the multiplications are
   * routine.
   */
  hi = f - hfsq;
  SET_LOW_WORD(hi, 0);
  lo = (f - hi) - hfsq + r;
  val_hi = hi * d_ivln2hi;
  val_lo = (lo + hi) * d_ivln2lo + lo * d_ivln2hi;

  /* spadd(val_hi, val_lo, y), except for not using double_t: */
  w = y + val_hi;
  val_lo += (y - w) + val_hi;
  val_hi = w;

  return val_lo + val_hi;
}

static const float
    /* |(log(1+s)-log(1-s))/s - Lg(s)| < 2**-34.24 (~[-4.95e-11, 4.97e-11]). */
    Lg1 = 0xaaaaaa.0p-24, /* 0.66666662693 */
    Lg2 = 0xccce13.0p-25, /* 0.40000972152 */
    Lg3 = 0x91e9ee.0p-25, /* 0.28498786688 */
    Lg4 = 0xf89e26.0p-26; /* 0.24279078841 */

static inline float k_log1pf(float f) {
  float hfsq, s, z, R, w, t1, t2;

  s = f / ((float)2.0 + f);
  z = s * s;
  w = z * z;
  t1 = w * (Lg2 + w * Lg4);
  t2 = z * (Lg1 + w * Lg3);
  R = t2 + t1;
  hfsq = (float)0.5 * f * f;
  return s * (hfsq + R);
}

static const float two25 = 3.3554432000e+07, /* 0x4c000000 */
    ivln2hi = 1.4428710938e+00,              /* 0x3fb8b000 */
    ivln2lo = -1.7605285393e-04;             /* 0xb9389ad4 */

static const float f_zero = 0.0;

float log2f(float x) {
  float f, hfsq, hi, lo, r, y;
  int32_t i, k, hx;

  GET_FLOAT_WORD(hx, x);

  k = 0;
  if (hx < 0x00800000) { /* x < 2**-126  */
    if ((hx & 0x7fffffff) == 0)
      return -two25 / f_zero; /* log(+-0)=-inf */
    if (hx < 0)
      return (x - x) / f_zero; /* log(-#) = NaN */
    k -= 25;
    x *= two25; /* subnormal number, scale up x */
    GET_FLOAT_WORD(hx, x);
  }
  if (hx >= 0x7f800000)
    return x + x;
  if (hx == 0x3f800000)
    return f_zero; /* log(1) = +0 */
  k += (hx >> 23) - 127;
  hx &= 0x007fffff;
  i = (hx + (0x4afb0d)) & 0x800000;
  SET_FLOAT_WORD(x, hx | (i ^ 0x3f800000)); /* normalize x or x/2 */
  k += (i >> 23);
  y = (float)k;
  f = x - (float)1.0;
  hfsq = (float)0.5 * f * f;
  r = k_log1pf(f);

  /*
   * We no longer need to avoid falling into the multi-precision
   * calculations due to compiler bugs breaking Dekker's theorem.
   * Keep avoiding this as an optimization.  See e_log2.c for more
   * details (some details are here only because the optimization
   * is not yet available in double precision).
   *
   * Another compiler bug turned up.  With gcc on i386,
   * (ivln2lo + ivln2hi) would be evaluated in float precision
   * despite runtime evaluations using double precision.  So we
   * must cast one of its terms to float_t.  This makes the whole
   * expression have type float_t, so return is forced to waste
   * time clobbering its extra precision.
   */
  if (sizeof(float) > sizeof(float))
    return (r - hfsq + f) * ((float)ivln2lo + ivln2hi) + y;

  hi = f - hfsq;
  GET_FLOAT_WORD(hx, hi);
  SET_FLOAT_WORD(hi, hx & 0xfffff000);
  lo = (f - hi) - hfsq + r;
  return (lo + hi) * ivln2lo + lo * ivln2hi + hi * ivln2hi + y;
}