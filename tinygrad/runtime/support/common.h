#ifndef TINYMATH_COMMON_H
#define TINYMATH_COMMON_H
#include <stdint.h>

// a bunch of stuff here is taken from musl and
// https://github.com/JuliaMath/openlibm

#define NAN __builtin_nanf("")
#define INFINITY __builtin_inff()
static const double toint = 1 / 2.22044604925031308085e-16;

#define STRICT_ASSIGN(type, lval, rval) ((lval) = (rval))

typedef union {
  double value;
  struct {
    uint32_t lsw;
    uint32_t msw;
  } parts;
  uint64_t word;
} ieee_double_shape_type;

#define EXTRACT_WORDS(ix0, ix1, d)                                             \
  do {                                                                         \
    ieee_double_shape_type ew_u;                                               \
    ew_u.value = (d);                                                          \
    (ix0) = ew_u.parts.msw;                                                    \
    (ix1) = ew_u.parts.lsw;                                                    \
  } while (0)

#define GET_HIGH_WORD(i, d)                                                    \
  do {                                                                         \
    ieee_double_shape_type gh_u;                                               \
    gh_u.value = (d);                                                          \
    (i) = gh_u.parts.msw;                                                      \
  } while (0)

#define GET_LOW_WORD(i, d)                                                     \
  do {                                                                         \
    ieee_double_shape_type gl_u;                                               \
    gl_u.value = (d);                                                          \
    (i) = gl_u.parts.lsw;                                                      \
  } while (0)

#define SET_HIGH_WORD(d, v)                                                    \
  do {                                                                         \
    ieee_double_shape_type sh_u;                                               \
    sh_u.value = (d);                                                          \
    sh_u.parts.msw = (v);                                                      \
    (d) = sh_u.value;                                                          \
  } while (0)

#define SET_LOW_WORD(d, v)                                                     \
  do {                                                                         \
    ieee_double_shape_type sl_u;                                               \
    sl_u.value = (d);                                                          \
    sl_u.parts.lsw = (v);                                                      \
    (d) = sl_u.value;                                                          \
  } while (0)

#define INSERT_WORDS(d, ix0, ix1)                                              \
  do {                                                                         \
    ieee_double_shape_type iw_u;                                               \
    iw_u.parts.msw = (ix0);                                                    \
    iw_u.parts.lsw = (ix1);                                                    \
    (d) = iw_u.value;                                                          \
  } while (0)

typedef union {
  float value;
  /* FIXME: Assumes 32 bit int.  */
  unsigned int word;
} ieee_float_shape_type;

#define GET_FLOAT_WORD(i, d)                                                   \
  do {                                                                         \
    ieee_float_shape_type gf_u;                                                \
    gf_u.value = (d);                                                          \
    (i) = gf_u.word;                                                           \
  } while (0)

#define SET_FLOAT_WORD(d, i)                                                   \
  do {                                                                         \
    ieee_float_shape_type sf_u;                                                \
    sf_u.word = (i);                                                           \
    (d) = sf_u.value;                                                          \
  } while (0)

double copysign(double x, double y) {
  uint32_t hx, hy;
  GET_HIGH_WORD(hx, x);
  GET_HIGH_WORD(hy, y);
  SET_HIGH_WORD(x, (hx & 0x7fffffff) | (hy & 0x80000000));
  return x;
}

#endif