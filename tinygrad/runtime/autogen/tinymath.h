// tinymath - a tiny single-header math library for tinygrad
// a bunch of stuff here is taken from musl and
// https://github.com/JuliaMath/openlibm
#include "common.h"

double sin(double);
float sinf(float);
#include "exp2.h"
#include "log2.h"
#include "sqrt.h"

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

#define __tg_real_builtin(fun, x) __tg_real(__builtin_##fun, x)

#define sin(x) __tg_real_builtin(sin, (x))
#define sqrt(x) __tg_real_builtin(sqrt, (x))
#define exp2(x) __tg_real_builtin(exp2, (x))
#define log2(x) __tg_real_builtin(log2, (x))