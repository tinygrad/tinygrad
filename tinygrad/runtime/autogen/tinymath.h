// tinymath - a tiny single-header math library for tinygrad
// a bunch of stuff here is taken from musl
#include <stdint.h> // only used for int8_t instead of char which is buggy for some reason

#define NAN __builtin_nanf("")
#define INFINITY __builtin_inff()

double sin(double);
float sinf(float);

double sqrt(double x) {
#if defined(__aarch64__) || defined(_M_ARM64)
  __asm__("fsqrt %d0, %d1" : "=w"(x) : "w"(x));
#elif defined(__x86_64__) || defined(_M_X64)
  __asm__("sqrtsd %1, %0" : "=x"(x) : "x"(x));
#else
#error only aarch64 and x86 is supported
#endif
  return x;
}

float sqrtf(float x) {
#if defined(__aarch64__) || defined(_M_ARM64)
  __asm__("fsqrt %s0, %s1" : "=w"(x) : "w"(x));
#elif defined(__x86_64__) || defined(_M_X64)
  __asm__("sqrtss %1, %0" : "=x"(x) : "x"(x));
#else
#error only aarch64 and x86 is supported
#endif
  return x;
}
double exp2(double);
float exp2f(float);
double log2(double);
float log2f(float);

#define __IS_FP(x) (sizeof((x) + 1ULL) == sizeof((x) + 1.0f))

/* if c then t else void */
#define __type1(c, t) __typeof__(*(0 ? (t *)0 : (void *)!(c)))
/* if c then t1 else t2 */
#define __type2(c, t1, t2)                                                     \
  __typeof__(*(0 ? (__type1(c, t1) *)0 : (__type1(!(c), t2) *)0))

#define __FLT(x) (__IS_FP(x) && sizeof(x) == sizeof(float))

#define __RETCAST(x) (__type2(__IS_FP(x), __typeof__(x), double))

#define __tg_real_nocast(fun, x) (__FLT(x) ? fun##f(x) : fun(x))

#define __tg_real(fun, x) (__RETCAST(x) __tg_real_nocast(fun, x))

#undef sin
#undef sqrt
#undef exp2
#undef log2
#define sin(x) __tg_real(sin, (x))
#define sqrt(x) __tg_real(sqrt, (x))
#define exp2(x) __tg_real(exp2, (x))
#define log2(x) __tg_real(log2, (x))