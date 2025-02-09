#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_306, float* buf_285, int start_pos) {
  r_128_4_4(buf_5, buf_285);
  E_512_4n34(buf_288, buf_285, buf_5, (float *)bufs[212]);
  r_128_512_4_4(buf_290, buf_288, (signed char *)bufs[213], (float *)bufs[214]);
  r_128_512_4_4(buf_293, buf_288, (signed char *)bufs[215], (float *)bufs[216]);
  E_2_8_16_4(buf_296, buf_290, (float *)bufs[8], buf_293, start_pos);
  r_512_512_4_4n34(buf_19, buf_288, (signed char *)bufs[217], (float *)bufs[218]);
  E_32_16_2_2(buf_16, buf_19, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_16, buf_296, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_19, buf_23, buf_296, start_pos);
  r_512_2048_4(buf_16, buf_285, buf_19, (signed char *)bufs[219], (float *)bufs[220]);
  r_128_4_4(buf_5, buf_16);
  E_512_4n34(buf_19, buf_16, buf_5, (float *)bufs[221]);
  r_512_128_4_4_4_4(buf_27, buf_19, (signed char *)bufs[222], (float *)bufs[223]);
  r_1024_128_2_4_4_4(buf_30, buf_19, (signed char *)bufs[224], (float *)bufs[225], buf_27);
  r_256_8192_4_2(buf_306, buf_16, buf_30, (signed char *)bufs[226], (float *)bufs[227]);
}