#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_75, float* buf_54, int start_pos) {
  r_128_4_4(buf_5, buf_54);
  E_512_4n34(buf_57, buf_54, buf_5, (float *)bufs[36]);
  r_128_512_4_4(buf_59, buf_57, (signed char *)bufs[37], (float *)bufs[38]);
  r_128_512_4_4(buf_62, buf_57, (signed char *)bufs[39], (float *)bufs[40]);
  E_2_8_16_4(buf_65, buf_59, (float *)bufs[8], buf_62, start_pos);
  r_512_512_4_4n34(buf_16, buf_57, (signed char *)bufs[41], (float *)bufs[42]);
  E_32_16_2_2(buf_19, buf_16, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_19, buf_65, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_16, buf_23, buf_65, start_pos);
  r_512_2048_4(buf_19, buf_54, buf_16, (signed char *)bufs[43], (float *)bufs[44]);
  r_128_4_4(buf_5, buf_19);
  E_512_4n34(buf_16, buf_19, buf_5, (float *)bufs[45]);
  r_512_128_4_4_4_4(buf_27, buf_16, (signed char *)bufs[46], (float *)bufs[47]);
  r_1024_128_2_4_4_4(buf_30, buf_16, (signed char *)bufs[48], (float *)bufs[49], buf_27);
  r_256_8192_4_2(buf_75, buf_19, buf_30, (signed char *)bufs[50], (float *)bufs[51]);
}