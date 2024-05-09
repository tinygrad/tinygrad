#include <stdint.h> // only used for int8_t instead of char which is buggy for some reason

double sin(double);
float sinf(float);
double sqrt(double);
float sqrtf(float);
double exp2(double);
float exp2f(float);
double log2(double);
float log2f(float);

// taken from musl's headers
#define NAN __builtin_nanf("")
#define INFINITY __builtin_inff()

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
#define sin(x) __tg_real(sin, (x))
#undef sqrt
#define sqrt(x) __tg_real(sqrt, (x))
#undef exp2
#define exp2(x) __tg_real(exp2, (x))
#undef log2
#define log2(x) __tg_real(log2, (x))