#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_222, float* buf_201, int start_pos) {
  r_128_4_4(buf_5, buf_201);
  E_512_4n34(buf_204, buf_201, buf_5, (float *)bufs[148]);
  r_128_512_4_4(buf_206, buf_204, (signed char *)bufs[149], (float *)bufs[150]);
  r_128_512_4_4(buf_209, buf_204, (signed char *)bufs[151], (float *)bufs[152]);
  E_2_8_16_4(buf_212, buf_206, (float *)bufs[8], buf_209, start_pos);
  r_512_512_4_4n34(buf_19, buf_204, (signed char *)bufs[153], (float *)bufs[154]);
  E_32_16_2_2(buf_16, buf_19, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_16, buf_212, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_19, buf_23, buf_212, start_pos);
  r_512_2048_4(buf_16, buf_201, buf_19, (signed char *)bufs[155], (float *)bufs[156]);
  r_128_4_4(buf_5, buf_16);
  E_512_4n34(buf_19, buf_16, buf_5, (float *)bufs[157]);
  r_512_128_4_4_4_4(buf_27, buf_19, (signed char *)bufs[158], (float *)bufs[159]);
  r_1024_128_2_4_4_4(buf_30, buf_19, (signed char *)bufs[160], (float *)bufs[161], buf_27);
  r_256_8192_4_2(buf_222, buf_16, buf_30, (signed char *)bufs[162], (float *)bufs[163]);
}