// https://github.com/JuliaMath/openlibm/blob/master/src/e_log2f.c

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
// https://stackoverflow.com/questions/65554112/fast-double-exp2-function-in-c

inline float exp2(float p) {
  if(p<-126.f) p= -126.f;
  int w=(int)p;
  float z=p-(float)w;
  if(p<0.f) z+= 1.f;
  union {unsigned int i; float f;} v={(unsigned int)((1<<23)*(p+121.2740575f+27.7280233f/(4.84252568f -z)-1.49012907f * z)) };
  return v.f;
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

#define NAN (0.0f/0.0f)
#define INFINITY (1.0f/0.0f)
#define sqrt __builtin_sqrtf