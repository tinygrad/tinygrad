#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_96, float* buf_75, int start_pos) {
  r_128_4_4(buf_5, buf_75);
  E_512_4n34(buf_78, buf_75, buf_5, (float *)bufs[52]);
  r_128_512_4_4(buf_80, buf_78, (signed char *)bufs[53], (float *)bufs[54]);
  r_128_512_4_4(buf_83, buf_78, (signed char *)bufs[55], (float *)bufs[56]);
  E_2_8_16_4(buf_86, buf_80, (float *)bufs[8], buf_83, start_pos);
  r_512_512_4_4n34(buf_19, buf_78, (signed char *)bufs[57], (float *)bufs[58]);
  E_32_16_2_2(buf_16, buf_19, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_16, buf_86, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_19, buf_23, buf_86, start_pos);
  r_512_2048_4(buf_16, buf_75, buf_19, (signed char *)bufs[59], (float *)bufs[60]);
  r_128_4_4(buf_5, buf_16);
  E_512_4n34(buf_19, buf_16, buf_5, (float *)bufs[61]);
  r_512_128_4_4_4_4(buf_27, buf_19, (signed char *)bufs[62], (float *)bufs[63]);
  r_1024_128_2_4_4_4(buf_30, buf_19, (signed char *)bufs[64], (float *)bufs[65], buf_27);
  r_256_8192_4_2(buf_96, buf_16, buf_30, (signed char *)bufs[66], (float *)bufs[67]);
}