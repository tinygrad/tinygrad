// https://github.com/JuliaMath/openlibm/blob/master/src/e_log2f.c

#define NAN (0.0f/0.0f)
#define INFINITY (1.0f/0.0f)
#define sqrt __builtin_sqrtf
#define uint32_t unsigned int
#define int32_t int

typedef union
{
  double value;
  struct
  {
    unsigned int msw;
    unsigned int lsw;
  } parts;
  struct
  {
    unsigned long w;
  } xparts;
} ieee_double_shape_type;

#define INSERT_WORDS(d,ix0,ix1)         \
do {                \
  ieee_double_shape_type iw_u;          \
  iw_u.parts.msw = (ix0);         \
  iw_u.parts.lsw = (ix1);         \
  (d) = iw_u.value;           \
} while (0)

#define STRICT_ASSIGN(type, lval, rval) ((lval) = (rval))
#define GET_FLOAT_WORD(i,f) i = *((int*)&(f))
#define SET_FLOAT_WORD(f,i) { int i2 = i; f = *((float*)&(i2)); }

const float
/* |(log(1+s)-log(1-s))/s - Lg(s)| < 2**-34.24 (~[-4.95e-11, 4.97e-11]). */
Lg1 =      0xaaaaaa.0p-24,  /* 0.66666662693 */
Lg2 =      0xccce13.0p-25,  /* 0.40000972152 */
Lg3 =      0x91e9ee.0p-25,  /* 0.28498786688 */
Lg4 =      0xf89e26.0p-26;  /* 0.24279078841 */

// ln(1+arg)
inline float k_log1pf(float f) {
  float hfsq,s,z,R,w,t1,t2;

  s = f/((float)2.0+f);
  z = s*s;
  w = z*z;
  t1= w*(Lg2+w*Lg4);
  t2= z*(Lg1+w*Lg3);
  R = t2+t1;
  hfsq=(float)0.5*f*f;
  return s*(hfsq+R);
}

// ***** log2 *****

const float
two25      =  3.3554432000e+07, /* 0x4c000000 */
ivln2hi    =  1.4428710938e+00, /* 0x3fb8b000 */
ivln2lo    = -1.7605285393e-04; /* 0xb9389ad4 */

const float zero   =  0.0;

inline float log2(float x) {
  float f,hfsq,hi,lo,r,y;
	int i,k,hx;
  GET_FLOAT_WORD(hx,x);

  k = 0;
	if (hx < 0x00800000) {			/* x < 2**-126  */
    if ((hx&0x7fffffff)==0) return -two25/zero;		/* log(+-0)=-inf */
    if (hx<0) return (x-x)/zero;	/* log(-#) = NaN */
    k -= 25; x *= two25; /* subnormal number, scale up x */
    GET_FLOAT_WORD(hx,x);
	}

  if (hx >= 0x7f800000) return x+x;
	if (hx == 0x3f800000) return zero;			/* log(1) = +0 */

  k += (hx>>23)-127;
	hx &= 0x007fffff;
	i = (hx+(0x4afb0d))&0x800000;
	SET_FLOAT_WORD(x,hx|(i^0x3f800000));	/* normalize x or x/2 */
	k += (i>>23);
	y = (float)k;
	f = x - (float)1.0;
	hfsq = (float)0.5*f*f;
	r = k_log1pf(f);

  hi = f - hfsq;
	GET_FLOAT_WORD(hx,hi);
	SET_FLOAT_WORD(hi,hx&0xfffff000);
	lo = (f - hi) - hfsq + r;
	return (lo+hi)*ivln2lo + lo*ivln2hi + hi*ivln2hi + y;
}

// ***** exp2 *****
// https://android.googlesource.com/platform/bionic/+/a27d2baa/libm/src/s_exp2f.c

#define	TBLBITS	4
#define	TBLSIZE	(1 << TBLBITS)

const float
    huge    = 0x1p100f,
    twom100 = 0x1p-100f,
    redux   = 0x1.8p23f / TBLSIZE,
    P1	    = 0x1.62e430p-1f,
    P2	    = 0x1.ebfbe0p-3f,
    P3	    = 0x1.c6b348p-5f,
    P4	    = 0x1.3b2c9cp-7f;
const double exp2ft[TBLSIZE] = {
	0x1.6a09e667f3bcdp-1,
	0x1.7a11473eb0187p-1,
	0x1.8ace5422aa0dbp-1,
	0x1.9c49182a3f090p-1,
	0x1.ae89f995ad3adp-1,
	0x1.c199bdd85529cp-1,
	0x1.d5818dcfba487p-1,
	0x1.ea4afa2a490dap-1,
	0x1.0000000000000p+0,
	0x1.0b5586cf9890fp+0,
	0x1.172b83c7d517bp+0,
	0x1.2387a6e756238p+0,
	0x1.306fe0a31b715p+0,
	0x1.3dea64c123422p+0,
	0x1.4bfdad5362a27p+0,
	0x1.5ab07dd485429p+0,
};

inline float exp2(float x) {
  double tv;
	float r, z;
	volatile float t;	/* prevent gcc from using too much precision */
	uint32_t hx, hr, ix, i0;
	int32_t k;
	/* Filter out exceptional cases. */
	GET_FLOAT_WORD(hx,x);
	ix = hx & 0x7fffffff;		/* high word of |x| */
	if(ix >= 0x43000000) {			/* |x| >= 128 */
		if(ix >= 0x7f800000) {
			if ((ix & 0x7fffff) != 0 || (hx & 0x80000000) == 0)
				return (x); 	/* x is NaN or +Inf */
			else
				return (0.0);	/* x is -Inf */
		}
		if(x >= 0x1.0p7f)
			return (huge * huge);	/* overflow */
		if(x <= -0x1.2cp7f)
			return (twom100 * twom100); /* underflow */
	} else if (ix <= 0x33000000) {		/* |x| <= 0x1p-25 */
		return (1.0f + x);
	}
	/* Reduce x, computing z, i0, and k. */
	t = x + redux;
	GET_FLOAT_WORD(i0, t);
	i0 += TBLSIZE / 2;
	k = (i0 >> TBLBITS) << 23;
	i0 &= TBLSIZE - 1;
	t -= redux;
	z = x - t;
	/* Compute r = exp2(y) = exp2ft[i0] * p(z). */
	tv = exp2ft[i0];
	r = tv + tv * (z * (P1 + z * (P2 + z * (P3 + z * P4))));
	/* Scale by 2**(k>>23). */
	if(k >= -(125 << 23)) {
		if (k != 0) {
			GET_FLOAT_WORD(hr, r);
			SET_FLOAT_WORD(r, hr + k);
		}
		return (r);
	} else {
		GET_FLOAT_WORD(hr, r);
		SET_FLOAT_WORD(r, hr + (k + (100 << 23)));
		return (r * twom100);
	}
}

// ***** sin *****

/* |sin(x)/x - s(x)| < 2**-37.5 (~[-4.89e-12, 4.824e-12]). */
const double
S1 = -0x15555554cbac77.0p-55,	/* -0.166666666416265235595 */
S2 =  0x111110896efbb2.0p-59,	/*  0.0083333293858894631756 */
S3 = -0x1a00f9e2cae774.0p-65,	/* -0.000198393348360966317347 */
S4 =  0x16cd878c3b46a7.0p-71;	/*  0.0000027183114939898219064 */

inline float sin(float x) {
  double r, s, w, z;

  /* Try to optimize for parallel evaluation as in k_tanf.c. */
  z = x*x;
  w = z*z;
  r = S3+z*S4;
  s = z*x;
  return (x + s*(S1+z*S2)) + s*w*r;
}