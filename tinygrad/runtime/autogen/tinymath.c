// taken from swift -
// https://github.com/apple/swift/blob/main/stdlib/public/runtime/Float16Support.cpp

static unsigned toEncodingF(float f) {
  unsigned e;
  // static_assert(sizeof e == sizeof f, "float and int must have the same
  // size");
  __builtin_memcpy(&e, &f, sizeof f);
  return e;
}

static float fromEncodingF(unsigned int e) {
  float f;
  // static_assert(sizeof f == sizeof e, "float and int must have the same
  // size");
  __builtin_memcpy(&f, &e, sizeof f);
  return f;
}

static __fp16 fromEncodingH(unsigned short s) {
  __fp16 f;
  // static_assert(sizeof s == sizeof f, "__fp16 and short must have the same
  // size");
  __builtin_memcpy(&f, &s, sizeof f);
  return f;
}

#if defined(__x86_64__) && defined(__F16C__)

// If we're compiling the runtime for a target that has the conversion
// instruction, we might as well just use those. In theory, we'd also be
// compiling Swift for that target and not need these builtins at all,
// but who knows what could go wrong, and they're tiny functions.
#include <immintrin.h>

short __gnu_f2h_ieee(float f) {
  return (unsigned short)_mm_cvtsi128_si32(
      _mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_CUR_DIRECTION));
}

#else
// Input in xmm0, result in di. We can get that calling convention in C++
// by returning int16 instead of Float16, which we don't have (or else
// we wouldn't need this function).
unsigned short __gnu_f2h_ieee(float f) {
  unsigned signbit = toEncodingF(f) & 0x80000000U;
  // Construct a "magic" rounding constant for f; this is a value that
  // we will add and subtract from f to force rounding to occur in the
  // correct position for half-precision. Half has 10 significand bits,
  // float has 23, so we need to add 2**(e+13) to get the desired rounding.
  float magic;
  unsigned exponent = toEncodingF(f) & 0x7f800000;
  // Subnormals all round in the same place as the minimum normal binade,
  // so treat anything below 0x1.0p-14 as 0x1.0p-14.
  if (exponent < 0x38800000)
    exponent = 0x38800000;
  // In the overflow, inf, and NaN cases, magic doesn't contribute, so we
  // just use zero for anything bigger than 0x1.0p16.
  if (exponent > 0x47000000)
    magic = fromEncodingF(signbit);
  else
    magic = fromEncodingF(signbit | exponent + 0x06800000);
  // Map anything with an exponent larger than 15 to infinity; this will
  // avoid special-casing overflow later on.
  f = 0x1.0p112f * f;
  f = 0x1.0p-112f * f + magic;
  f -= magic;
  // We've now rounded in the correct place. One more scaling and we have
  // all the bits we need (this multiply does not change anything for
  // normal results, but denormalizes tiny results exactly as needed).
  f *= 0x1.0p-112f;
  short magnitude = toEncodingF(f) >> 13 & 0x7fff;
  return (int)signbit >> 16 | magnitude;
}

#endif

__fp16 __truncdfhf2(double d) {
  // You can't just do (half)(float)x, because that makes the result
  // susceptible to double-rounding. Instead we need to make the first
  // rounding use round-to-odd, but that doesn't exist on x86, so we have
  // to fake it.
  float f = (float)d;
  // Double-rounding can only occur if the result of rounding to float is
  // an exact-halfway case for the subsequent rounding to float16. We
  // can check for that significand bit pattern quickly (though we need
  // to be careful about values that will result in a subnormal float16,
  // as those will round in a different position):
  unsigned e = toEncodingF(f);
  int exactHalfway = (e & 0x1fff) == 0x1000;
  double fabs = __builtin_fabsf(f);
  if (exactHalfway || __builtin_fabsf(f) < 0x1.0p-14f) {
    // We might be in a double-rounding case, so simulate sticky-rounding
    // by comparing f and x and adjusting as needed.
    double dabs = __builtin_fabs(d);
    if (fabs > dabs)
      e -= ~e & 1;
    if (fabs < dabs)
      e |= 1;
    f = fromEncodingF(e);
  }
  return fromEncodingH(__gnu_f2h_ieee(f));
}