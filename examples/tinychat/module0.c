#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net0(float* buf_4, int* input0) {
  r_256_64_501_4_4_2(buf_0, (int *)bufs[0], input0, (signed char *)bufs[1], (float *)bufs[2]);
  r_128_64_2_4_4_2(buf_4, buf_0);
}

void net1(int* output0, float* buf_5, float* buf_352) {
  r_128_4_4(buf_346, buf_352);
  E_512_4n34(buf_16, buf_352, buf_346, (float *)bufs[260]);
  r_8016_512_4_4_4(buf_356, buf_16, (signed char *)bufs[1], (float *)bufs[2]);
  r_8_501_2_4_4(buf_357, buf_356);
  r_16_4_4(buf_346, buf_357);
  r_8_501_2_4_4n1(buf_357, buf_356, buf_346);
  r_64_4n3(buf_358, buf_357);
  E_42752_3n1(buf_359, buf_356, buf_346, buf_358);
  r_167_64_64_4_3_4(buf_360, buf_359);
  r_167_501_3(buf_361, buf_360);
  r_128_501_2(buf_362, buf_5, buf_360, buf_361);
  r_64_4n4(output0, buf_362);
}