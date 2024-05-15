#include "common.h"

static const double scbn_two54 =
                        1.80143985094819840000e+16, /* 0x43500000, 0x00000000 */
    scbn_twom54 = 5.55111512312578270212e-17,       /* 0x3C900000, 0x00000000 */
    scbn_huge = 1.0e+300, scbn_tiny = 1.0e-300;

double scalbn(double x, int n) {
  int32_t k, hx, lx;
  EXTRACT_WORDS(hx, lx, x);
  k = (hx & 0x7ff00000) >> 20; /* extract exponent */
  if (k == 0) {                /* 0 or subnormal x */
    if ((lx | (hx & 0x7fffffff)) == 0)
      return x; /* +-0 */
    x *= scbn_two54;
    GET_HIGH_WORD(hx, x);
    k = ((hx & 0x7ff00000) >> 20) - 54;
    if (n < -50000)
      return scbn_tiny * x; /*underflow*/
  }
  if (k == 0x7ff)
    return x + x; /* NaN or Inf */
  k = k + n;
  if (k > 0x7fe)
    return scbn_huge * copysign(scbn_huge, x); /* overflow  */
  if (k > 0)                                   /* normal result */
  {
    SET_HIGH_WORD(x, (hx & 0x800fffff) | (k << 20));
    return x;
  }
  if (k <= -54) {
    if (n > 50000) /* in case integer overflow in n+k */
      return scbn_huge * copysign(scbn_huge, x); /*overflow*/
    else
      return scbn_tiny * copysign(scbn_tiny, x); /*underflow*/
  }
  k += 54; /* subnormal result */
  SET_HIGH_WORD(x, (hx & 0x800fffff) | (k << 20));
  return x * scbn_twom54;
}

double floor(double x) {
#if defined(__aarch64__) || defined(_M_ARM64)
  __asm__("frintm %d0, %d1" : "=w"(x) : "w"(x));
#else
  union {
    double f;
    uint64_t i;
  } u = {x};
  int e = u.i >> 52 & 0x7ff;
  double y;

  if (e >= 0x3ff + 52 || x == 0)
    return x;
  /* y = int(x) - x, where int(x) is an integer neighbor of x */
  if (u.i >> 63)
    y = x - toint + toint - x;
  else
    y = x + toint - toint - x;
  /* special case because of non-nearest rounding modes */
  if (e <= 0x3ff - 1) {
    return u.i >> 63 ? -1 : 0;
  }
  if (y > 0)
    return x + y - 1;
#endif
  return x;
}

static const int init_jk[] = {3, 4, 4, 6}; /* initial value for jk */

/*
 * Table of constants for 2/pi, 396 Hex digits (476 decimal) of 2/pi
 *
 *		integer array, contains the (24*i)-th to (24*i+23)-th
 *		bit of 2/pi after binary point. The corresponding
 *		floating value is
 *
 *			ipio2[i] * 2^(-24(i+1)).
 *
 * NB: This table must have at least (e0-3)/24 + jk terms.
 *     For quad precision (e0 <= 16360, jk = 6), this is 686.
 */
static const int32_t ipio2[] = {
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C,
    0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649,
    0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129, 0xA73EE8, 0x8235F5, 0x2EBB44,
    0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C, 0x845F8B,
    0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D,
    0x367ECF, 0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
    0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330,
    0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3, 0x91615E, 0xE61B08,
    0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B,
};

static const double PIo2[] = {
    1.57079625129699707031e+00, /* 0x3FF921FB, 0x40000000 */
    7.54978941586159635335e-08, /* 0x3E74442D, 0x00000000 */
    5.39030252995776476554e-15, /* 0x3CF84698, 0x80000000 */
    3.28200341580791294123e-22, /* 0x3B78CC51, 0x60000000 */
    1.27065575308067607349e-29, /* 0x39F01B83, 0x80000000 */
    1.22933308981111328932e-36, /* 0x387A2520, 0x40000000 */
    2.73370053816464559624e-44, /* 0x36E38222, 0x80000000 */
    2.16741683877804819444e-51, /* 0x3569F31D, 0x00000000 */
};

static const double sd_zero = 0.0, sd_one = 1.0,
                    sd_two24 =
                        1.67772160000000000000e+07, /* 0x41700000, 0x00000000 */
    twon24 = 5.96046447753906250000e-08;            /* 0x3E700000, 0x00000000 */

int __kernel_rem_pio2(double *x, double *y, int e0, int nx, int prec) {
  int32_t jz, jx, jv, jp, jk, carry, n, iq[20], i, j, k, m, q0, ih;
  double z, fw, f[20], fq[20], q[20];

  /* initialize jk*/
  jk = init_jk[prec];
  jp = jk;

  /* determine jx,jv,q0, note that 3>q0 */
  jx = nx - 1;
  jv = (e0 - 3) / 24;
  if (jv < 0)
    jv = 0;
  q0 = e0 - 24 * (jv + 1);

  /* set up f[0] to f[jx+jk] where f[jx+jk] = ipio2[jv+jk] */
  j = jv - jx;
  m = jx + jk;
  for (i = 0; i <= m; i++, j++)
    f[i] = (j < 0) ? sd_zero : (double)ipio2[j];

  /* compute q[0],q[1],...q[jk] */
  for (i = 0; i <= jk; i++) {
    for (j = 0, fw = 0.0; j <= jx; j++)
      fw += x[j] * f[jx + i - j];
    q[i] = fw;
  }

  jz = jk;
recompute:
  /* distill q[] into iq[] reversingly */
  for (i = 0, j = jz, z = q[jz]; j > 0; i++, j--) {
    fw = (double)((int32_t)(twon24 * z));
    iq[i] = (int32_t)(z - sd_two24 * fw);
    z = q[j - 1] + fw;
  }

  /* compute n */
  z = scalbn(z, q0);           /* actual value of z */
  z -= 8.0 * floor(z * 0.125); /* trim off integer >= 8 */
  n = (int32_t)z;
  z -= (double)n;
  ih = 0;
  if (q0 > 0) { /* need iq[jz-1] to determine n */
    i = (iq[jz - 1] >> (24 - q0));
    n += i;
    iq[jz - 1] -= i << (24 - q0);
    ih = iq[jz - 1] >> (23 - q0);
  } else if (q0 == 0)
    ih = iq[jz - 1] >> 23;
  else if (z >= 0.5)
    ih = 2;

  if (ih > 0) { /* q > 0.5 */
    n += 1;
    carry = 0;
    for (i = 0; i < jz; i++) { /* compute 1-q */
      j = iq[i];
      if (carry == 0) {
        if (j != 0) {
          carry = 1;
          iq[i] = 0x1000000 - j;
        }
      } else
        iq[i] = 0xffffff - j;
    }
    if (q0 > 0) { /* rare case: chance is 1 in 12 */
      switch (q0) {
      case 1:
        iq[jz - 1] &= 0x7fffff;
        break;
      case 2:
        iq[jz - 1] &= 0x3fffff;
        break;
      }
    }
    if (ih == 2) {
      z = sd_one - z;
      if (carry != 0)
        z -= scalbn(sd_one, q0);
    }
  }

  /* check if recomputation is needed */
  if (z == sd_zero) {
    j = 0;
    for (i = jz - 1; i >= jk; i--)
      j |= iq[i];
    if (j == 0) { /* need recomputation */
      for (k = 1; iq[jk - k] == 0; k++)
        ; /* k = no. of terms needed */

      for (i = jz + 1; i <= jz + k; i++) { /* add q[jz+1] to q[jz+k] */
        f[jx + i] = (double)ipio2[jv + i];
        for (j = 0, fw = 0.0; j <= jx; j++)
          fw += x[j] * f[jx + i - j];
        q[i] = fw;
      }
      jz += k;
      goto recompute;
    }
  }

  /* chop off zero terms */
  if (z == 0.0) {
    jz -= 1;
    q0 -= 24;
    while (iq[jz] == 0) {
      jz--;
      q0 -= 24;
    }
  } else { /* break z into 24-bit if necessary */
    z = scalbn(z, -q0);
    if (z >= sd_two24) {
      fw = (double)((int32_t)(twon24 * z));
      iq[jz] = (int32_t)(z - sd_two24 * fw);
      jz += 1;
      q0 += 24;
      iq[jz] = (int32_t)fw;
    } else
      iq[jz] = (int32_t)z;
  }

  /* convert integer "bit" chunk to floating-point value */
  fw = scalbn(sd_one, q0);
  for (i = jz; i >= 0; i--) {
    q[i] = fw * (double)iq[i];
    fw *= twon24;
  }

  /* compute PIo2[0,...,jp]*q[jz,...,0] */
  for (i = jz; i >= 0; i--) {
    for (fw = 0.0, k = 0; k <= jp && k <= jz - i; k++)
      fw += PIo2[k] * q[i + k];
    fq[jz - i] = fw;
  }

  /* compress fq[] into y[] */
  switch (prec) {
  case 0:
    fw = 0.0;
    for (i = jz; i >= 0; i--)
      fw += fq[i];
    y[0] = (ih == 0) ? fw : -fw;
    break;
  case 1:
  case 2:
    fw = 0.0;
    for (i = jz; i >= 0; i--)
      fw += fq[i];
    STRICT_ASSIGN(double, fw, fw);
    y[0] = (ih == 0) ? fw : -fw;
    fw = fq[0] - fw;
    for (i = 1; i <= jz; i++)
      fw += fq[i];
    y[1] = (ih == 0) ? fw : -fw;
    break;
  case 3: /* painful */
    for (i = jz; i > 0; i--) {
      fw = fq[i - 1] + fq[i];
      fq[i] += fq[i - 1] - fw;
      fq[i - 1] = fw;
    }
    for (i = jz; i > 1; i--) {
      fw = fq[i - 1] + fq[i];
      fq[i] += fq[i - 1] - fw;
      fq[i - 1] = fw;
    }
    for (fw = 0.0, i = jz; i >= 2; i--)
      fw += fq[i];
    if (ih == 0) {
      y[0] = fq[0];
      y[1] = fq[1];
      y[2] = fw;
    } else {
      y[0] = -fq[0];
      y[1] = -fq[1];
      y[2] = -fw;
    }
  }
  return n & 7;
}

static const double rem_pio2_zero =
                        0.00000000000000000000e+00, /* 0x00000000, 0x00000000 */
    rem_pio2_two24 = 1.67772160000000000000e+07,    /* 0x41700000, 0x00000000 */
    invpio2 = 6.36619772367581382433e-01,           /* 0x3FE45F30, 0x6DC9C883 */
    pio2_1 = 1.57079632673412561417e+00,            /* 0x3FF921FB, 0x54400000 */
    pio2_1t = 6.07710050650619224932e-11,           /* 0x3DD0B461, 0x1A626331 */
    pio2_2 = 6.07710050630396597660e-11,            /* 0x3DD0B461, 0x1A600000 */
    pio2_2t = 2.02226624879595063154e-21,           /* 0x3BA3198A, 0x2E037073 */
    pio2_3 = 2.02226624871116645580e-21,            /* 0x3BA3198A, 0x2E000000 */
    pio2_3t = 8.47842766036889956997e-32;           /* 0x397B839A, 0x252049C1 */

int __ieee754_rem_pio2(double x, double *y) {
  double z, w, t, r, fn;
  double tx[3], ty[2];
  int32_t e0, i, j, nx, n, ix, hx;
  uint32_t low;

  GET_HIGH_WORD(hx, x); /* high word of x */
  ix = hx & 0x7fffffff;
#if 0 /* Must be handled in caller. */
	if(ix<=0x3fe921fb)   /* |x| ~<= pi/4 , no need for reduction */
	    {y[0] = x; y[1] = 0; return 0;}
#endif
  if (ix <= 0x400f6a7a) {          /* |x| ~<= 5pi/4 */
    if ((ix & 0xfffff) == 0x921fb) /* |x| ~= pi/2 or 2pi/2 */
      goto medium;                 /* cancellation -- use medium case */
    if (ix <= 0x4002d97c) {        /* |x| ~<= 3pi/4 */
      if (hx > 0) {
        z = x - pio2_1; /* one round good to 85 bits */
        y[0] = z - pio2_1t;
        y[1] = (z - y[0]) - pio2_1t;
        return 1;
      } else {
        z = x + pio2_1;
        y[0] = z + pio2_1t;
        y[1] = (z - y[0]) + pio2_1t;
        return -1;
      }
    } else {
      if (hx > 0) {
        z = x - 2 * pio2_1;
        y[0] = z - 2 * pio2_1t;
        y[1] = (z - y[0]) - 2 * pio2_1t;
        return 2;
      } else {
        z = x + 2 * pio2_1;
        y[0] = z + 2 * pio2_1t;
        y[1] = (z - y[0]) + 2 * pio2_1t;
        return -2;
      }
    }
  }
  if (ix <= 0x401c463b) {   /* |x| ~<= 9pi/4 */
    if (ix <= 0x4015fdbc) { /* |x| ~<= 7pi/4 */
      if (ix == 0x4012d97c) /* |x| ~= 3pi/2 */
        goto medium;
      if (hx > 0) {
        z = x - 3 * pio2_1;
        y[0] = z - 3 * pio2_1t;
        y[1] = (z - y[0]) - 3 * pio2_1t;
        return 3;
      } else {
        z = x + 3 * pio2_1;
        y[0] = z + 3 * pio2_1t;
        y[1] = (z - y[0]) + 3 * pio2_1t;
        return -3;
      }
    } else {
      if (ix == 0x401921fb) /* |x| ~= 4pi/2 */
        goto medium;
      if (hx > 0) {
        z = x - 4 * pio2_1;
        y[0] = z - 4 * pio2_1t;
        y[1] = (z - y[0]) - 4 * pio2_1t;
        return 4;
      } else {
        z = x + 4 * pio2_1;
        y[0] = z + 4 * pio2_1t;
        y[1] = (z - y[0]) + 4 * pio2_1t;
        return -4;
      }
    }
  }
  if (ix < 0x413921fb) { /* |x| ~< 2^20*(pi/2), medium size */
  medium:
    /* Use a specialized rint() to get fn.  Assume round-to-nearest. */
    STRICT_ASSIGN(double, fn, x *invpio2 + 0x1.8p52);
    fn = fn - 0x1.8p52;
#ifdef HAVE_EFFICIENT_IRINT
    n = irint(fn);
#else
    n = (int32_t)fn;
#endif
    r = x - fn * pio2_1;
    w = fn * pio2_1t; /* 1st round good to 85 bit */
    {
      uint32_t high;
      j = ix >> 20;
      y[0] = r - w;
      GET_HIGH_WORD(high, y[0]);
      i = j - ((high >> 20) & 0x7ff);
      if (i > 16) { /* 2nd iteration needed, good to 118 */
        t = r;
        w = fn * pio2_2;
        r = t - w;
        w = fn * pio2_2t - ((t - r) - w);
        y[0] = r - w;
        GET_HIGH_WORD(high, y[0]);
        i = j - ((high >> 20) & 0x7ff);
        if (i > 49) { /* 3rd iteration need, 151 bits acc */
          t = r;      /* will cover all possible cases */
          w = fn * pio2_3;
          r = t - w;
          w = fn * pio2_3t - ((t - r) - w);
          y[0] = r - w;
        }
      }
    }
    y[1] = (r - y[0]) - w;
    return n;
  }
  /*
   * all other (large) arguments
   */
  if (ix >= 0x7ff00000) { /* x is inf or NaN */
    y[0] = y[1] = x - x;
    return 0;
  }
  /* set z = scalbn(|x|,ilogb(x)-23) */
  GET_LOW_WORD(low, x);
  e0 = (ix >> 20) - 1046; /* e0 = ilogb(z)-23; */
  INSERT_WORDS(z, ix - ((int32_t)(e0 << 20)), low);
  for (i = 0; i < 2; i++) {
    tx[i] = (double)((int32_t)(z));
    z = (z - tx[i]) * rem_pio2_two24;
  }
  tx[2] = z;
  nx = 3;
  while (tx[nx - 1] == rem_pio2_zero)
    nx--; /* skip zero term */
  n = __kernel_rem_pio2(tx, ty, e0, nx, 1);
  if (hx < 0) {
    y[0] = -ty[0];
    y[1] = -ty[1];
    return -n;
  }
  y[0] = ty[0];
  y[1] = ty[1];
  return n;
}

static const double half =
                        5.00000000000000000000e-01, /* 0x3FE00000, 0x00000000 */
    S1 = -1.66666666666666324348e-01,               /* 0xBFC55555, 0x55555549 */
    S2 = 8.33333333332248946124e-03,                /* 0x3F811111, 0x1110F8A6 */
    S3 = -1.98412698298579493134e-04,               /* 0xBF2A01A0, 0x19C161D5 */
    S4 = 2.75573137070700676789e-06,                /* 0x3EC71DE3, 0x57B1FE7D */
    S5 = -2.50507602534068634195e-08,               /* 0xBE5AE5E6, 0x8A2B9CEB */
    S6 = 1.58969099521155010221e-10;                /* 0x3DE5D93A, 0x5ACFD57C */

double __kernel_sin(double x, double y, int iy) {
  double z, r, v, w;

  z = x * x;
  w = z * z;
  r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
  v = z * x;
  if (iy == 0)
    return x + v * (S1 + z * r);
  else
    return x - ((z * (half * y - v * r) - y) - v * S1);
}

static const double one =
                        1.00000000000000000000e+00, /* 0x3FF00000, 0x00000000 */
    C1 = 4.16666666666666019037e-02,                /* 0x3FA55555, 0x5555554C */
    C2 = -1.38888888888741095749e-03,               /* 0xBF56C16C, 0x16C15177 */
    C3 = 2.48015872894767294178e-05,                /* 0x3EFA01A0, 0x19CB1590 */
    C4 = -2.75573143513906633035e-07,               /* 0xBE927E4F, 0x809C52AD */
    C5 = 2.08757232129817482790e-09,                /* 0x3E21EE9E, 0xBDB4B1C4 */
    C6 = -1.13596475577881948265e-11;               /* 0xBDA8FAE9, 0xBE8838D4 */

double __kernel_cos(double x, double y) {
  double hz, z, r, w;

  z = x * x;
  w = z * z;
  r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
  hz = 0.5 * z;
  w = one - hz;
  return w + (((one - w) - hz) + (z * r - x * y));
}

double sin(double x) {
  double y[2], z = 0.0;
  int32_t n, ix;

  /* High word of x. */
  GET_HIGH_WORD(ix, x);

  /* |x| ~< pi/4 */
  ix &= 0x7fffffff;
  if (ix <= 0x3fe921fb) {
    if (ix < 0x3e500000) /* |x| < 2**-26 */
    {
      if ((int)x == 0)
        return x;
    } /* generate inexact */
    return __kernel_sin(x, z, 0);
  }

  /* sin(Inf or NaN) is NaN */
  else if (ix >= 0x7ff00000)
    return x - x;

  /* argument reduction needed */
  else {
    n = __ieee754_rem_pio2(x, y);
    switch (n & 3) {
    case 0:
      return __kernel_sin(y[0], y[1], 1);
    case 1:
      return __kernel_cos(y[0], y[1]);
    case 2:
      return -__kernel_sin(y[0], y[1], 1);
    default:
      return -__kernel_cos(y[0], y[1]);
    }
  }
}

float sinf(float x) { return (float)sin((double)x); }