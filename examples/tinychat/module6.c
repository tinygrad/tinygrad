#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_138, float* buf_117, int start_pos) {
  r_128_4_4(buf_5, buf_117);
  E_512_4n34(buf_120, buf_117, buf_5, (float *)bufs[84]);
  r_128_512_4_4(buf_122, buf_120, (signed char *)bufs[85], (float *)bufs[86]);
  r_128_512_4_4(buf_125, buf_120, (signed char *)bufs[87], (float *)bufs[88]);
  E_2_8_16_4(buf_128, buf_122, (float *)bufs[8], buf_125, start_pos);
  r_512_512_4_4n34(buf_19, buf_120, (signed char *)bufs[89], (float *)bufs[90]);
  E_32_16_2_2(buf_16, buf_19, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_16, buf_128, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_19, buf_23, buf_128, start_pos);
  r_512_2048_4(buf_16, buf_117, buf_19, (signed char *)bufs[91], (float *)bufs[92]);
  r_128_4_4(buf_5, buf_16);
  E_512_4n34(buf_19, buf_16, buf_5, (float *)bufs[93]);
  r_512_128_4_4_4_4(buf_27, buf_19, (signed char *)bufs[94], (float *)bufs[95]);
  r_1024_128_2_4_4_4(buf_30, buf_19, (signed char *)bufs[96], (float *)bufs[97], buf_27);
  r_256_8192_4_2(buf_138, buf_16, buf_30, (signed char *)bufs[98], (float *)bufs[99]);
}