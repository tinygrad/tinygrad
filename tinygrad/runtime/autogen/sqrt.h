#ifndef TINYMATH_SQRT_H
#define TINYMATH_SQRT_H

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

#endif