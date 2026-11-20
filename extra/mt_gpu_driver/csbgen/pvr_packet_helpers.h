/*
 * Copyright © 2022 Imagination Technologies Ltd.
 *
 * based in part on anv driver which is:
 * Copyright © 2016 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef PVR_PACKET_HELPERS_H
#define PVR_PACKET_HELPERS_H

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifndef __pvr_validate_value
#   define __pvr_validate_value(x)
#endif

#ifdef NDEBUG
#   define NDEBUG_UNUSED __attribute__((unused))
#else
#   define NDEBUG_UNUSED
#endif

#ifndef __pvr_address_type
#   error #define __pvr_address_type before including this file
#endif

#ifndef __pvr_get_address
#   error #define __pvr_get_address before including this file
#endif

#ifndef __pvr_make_address
#   error #define __pvr_make_address before including this file
#endif

union __pvr_value {
   float f;
   uint32_t dw;
};

static inline __attribute__((always_inline)) uint64_t __pvr_mbo(uint32_t start,
                                                                uint32_t end)
{
   return (~0ull >> (64 - (end - start + 1))) << start;
}

static inline __attribute__((always_inline)) uint64_t
__pvr_uint(uint64_t v, uint32_t start, NDEBUG_UNUSED uint32_t end)
{
   __pvr_validate_value(v);

#ifndef NDEBUG
   const int width = end - start + 1;
   if (width < 64) {
      const uint64_t max = (1ull << width) - 1;
      assert(v <= max);
   }
#endif

   return v << start;
}

static inline __attribute__((always_inline)) uint64_t
__pvr_uint_unpack(uint64_t packed, uint32_t start, uint32_t end)
{
   const int width = end - start + 1;
   const uint64_t mask = ~0ull >> (64 - width);

   return (packed >> start) & mask;
}

static inline __attribute__((always_inline)) uint64_t
__pvr_sint(int64_t v, uint32_t start, uint32_t end)
{
   const int width = end - start + 1;

   __pvr_validate_value(v);

#ifndef NDEBUG
   if (width < 64) {
      const int64_t max = (1ll << (width - 1)) - 1;
      const int64_t min = -(1ll << (width - 1));
      assert(min <= v && v <= max);
   }
#endif

   const uint64_t mask = ~0ull >> (64 - width);

   return (v & mask) << start;
}

static inline __attribute__((always_inline)) int64_t
__pvr_sint_unpack(uint64_t packed, uint32_t start, uint32_t end)
{
   const int width = end - start + 1;
   const uint64_t mask = ~0ull >> (64 - width);

   return (int64_t)((packed >> start) & mask);
}

static inline __attribute__((always_inline)) uint64_t
__pvr_offset(uint64_t v,
             NDEBUG_UNUSED uint32_t start,
             NDEBUG_UNUSED uint32_t end)
{
   __pvr_validate_value(v);
#ifndef NDEBUG
   uint64_t mask = (~0ull >> (64 - (end - start + 1))) << start;

   assert((v & ~mask) == 0);
#endif

   return v;
}

static inline __attribute__((always_inline)) uint64_t
__pvr_offset_unpack(uint64_t packed,
                    NDEBUG_UNUSED uint32_t start,
                    NDEBUG_UNUSED uint32_t end)
{
#ifndef NDEBUG
   uint64_t mask = (~0ull >> (64 - (end - start + 1))) << start;

   assert((packed & ~mask) == 0);
#endif

   return packed;
}

static inline __attribute__((always_inline)) uint64_t
__pvr_address(__pvr_address_type address,
              uint32_t shift,
              uint32_t start,
              uint32_t end)
{
   uint64_t addr_u64 = __pvr_get_address(address);
   uint64_t mask = (~0ull >> (64 - (end - start + 1))) << start;

   return ((addr_u64 >> shift) << start) & mask;
}

static inline __attribute__((always_inline)) __pvr_address_type
__pvr_address_unpack(uint64_t packed,
                     uint32_t shift,
                     uint32_t start,
                     uint32_t end)
{
   uint64_t mask = (~0ull >> (64 - (end - start + 1))) << start;
   uint64_t addr_u64 = ((packed & mask) >> start) << shift;

   return __pvr_make_address(addr_u64);
}

static inline __attribute__((always_inline)) uint32_t __pvr_float(float v)
{
   __pvr_validate_value(v);
   return ((union __pvr_value){ .f = (v) }).dw;
}

static inline __attribute__((always_inline)) float
__pvr_float_unpack(uint32_t packed)
{
   return ((union __pvr_value){ .dw = (packed) }).f;
}

static inline __attribute__((always_inline)) uint64_t
__pvr_sfixed(float v, uint32_t start, uint32_t end, uint32_t fract_bits)
{
   __pvr_validate_value(v);

   const float factor = (1 << fract_bits);

#ifndef NDEBUG
   const float max = ((1 << (end - start)) - 1) / factor;
   const float min = -(1 << (end - start)) / factor;
   assert(min <= v && v <= max);
#endif

   const int64_t int_val = llroundf(v * factor);
   const uint64_t mask = ~0ull >> (64 - (end - start + 1));

   return (int_val & mask) << start;
}

static inline __attribute__((always_inline)) uint64_t
__pvr_ufixed(float v,
             uint32_t start,
             NDEBUG_UNUSED uint32_t end,
             uint32_t fract_bits)
{
   __pvr_validate_value(v);

   const float factor = (1 << fract_bits);

#ifndef NDEBUG
   const float max = ((1 << (end - start + 1)) - 1) / factor;
   const float min = 0.0f;
   assert(min <= v && v <= max);
#endif

   const uint64_t uint_val = llroundf(v * factor);

   return uint_val << start;
}

#undef NDEBUG_UNUSED

#endif /* PVR_PACKET_HELPERS_H */
