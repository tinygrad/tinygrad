#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_54, float* buf_33, int start_pos) {
  r_128_4_4(buf_5, buf_33);
  E_512_4n34(buf_36, buf_33, buf_5, (float *)bufs[20]);
  r_128_512_4_4(buf_38, buf_36, (signed char *)bufs[21], (float *)bufs[22]);
  r_128_512_4_4(buf_41, buf_36, (signed char *)bufs[23], (float *)bufs[24]);
  E_2_8_16_4(buf_44, buf_38, (float *)bufs[8], buf_41, start_pos);
  r_512_512_4_4n34(buf_19, buf_36, (signed char *)bufs[25], (float *)bufs[26]);
  E_32_16_2_2(buf_16, buf_19, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_16, buf_44, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_19, buf_23, buf_44, start_pos);
  r_512_2048_4(buf_16, buf_33, buf_19, (signed char *)bufs[27], (float *)bufs[28]);
  r_128_4_4(buf_5, buf_16);
  E_512_4n34(buf_19, buf_16, buf_5, (float *)bufs[29]);
  r_512_128_4_4_4_4(buf_27, buf_19, (signed char *)bufs[30], (float *)bufs[31]);
  r_1024_128_2_4_4_4(buf_30, buf_19, (signed char *)bufs[32], (float *)bufs[33], buf_27);
  r_256_8192_4_2(buf_54, buf_16, buf_30, (signed char *)bufs[34], (float *)bufs[35]);
}