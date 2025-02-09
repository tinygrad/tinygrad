#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_33, float* buf_4, int start_pos) {
  r_128_4_4(buf_5, buf_4);
  E_512_4n34(buf_6, buf_4, buf_5, (float *)bufs[3]);
  r_128_512_4_4(buf_8, buf_6, (signed char *)bufs[4], (float *)bufs[5]);
  r_128_512_4_4(buf_11, buf_6, (signed char *)bufs[6], (float *)bufs[7]);
  E_2_8_16_4(buf_14, buf_8, (float *)bufs[8], buf_11, start_pos);
  r_512_512_4_4n34(buf_16, buf_6, (signed char *)bufs[9], (float *)bufs[10]);
  E_32_16_2_2(buf_19, buf_16, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_19, buf_14, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_16, buf_23, buf_14, start_pos);
  r_512_2048_4(buf_19, buf_4, buf_16, (signed char *)bufs[11], (float *)bufs[12]);
  r_128_4_4(buf_5, buf_19);
  E_512_4n34(buf_16, buf_19, buf_5, (float *)bufs[13]);
  r_512_128_4_4_4_4(buf_27, buf_16, (signed char *)bufs[14], (float *)bufs[15]);
  r_1024_128_2_4_4_4(buf_30, buf_16, (signed char *)bufs[16], (float *)bufs[17], buf_27);
  r_256_8192_4_2(buf_33, buf_19, buf_30, (signed char *)bufs[18], (float *)bufs[19]);
}