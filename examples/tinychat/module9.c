#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_201, float* buf_180, int start_pos) {
  r_128_4_4(buf_5, buf_180);
  E_512_4n34(buf_183, buf_180, buf_5, (float *)bufs[132]);
  r_128_512_4_4(buf_185, buf_183, (signed char *)bufs[133], (float *)bufs[134]);
  r_128_512_4_4(buf_188, buf_183, (signed char *)bufs[135], (float *)bufs[136]);
  E_2_8_16_4(buf_191, buf_185, (float *)bufs[8], buf_188, start_pos);
  r_512_512_4_4n34(buf_16, buf_183, (signed char *)bufs[137], (float *)bufs[138]);
  E_32_16_2_2(buf_19, buf_16, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_19, buf_191, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_16, buf_23, buf_191, start_pos);
  r_512_2048_4(buf_19, buf_180, buf_16, (signed char *)bufs[139], (float *)bufs[140]);
  r_128_4_4(buf_5, buf_19);
  E_512_4n34(buf_16, buf_19, buf_5, (float *)bufs[141]);
  r_512_128_4_4_4_4(buf_27, buf_16, (signed char *)bufs[142], (float *)bufs[143]);
  r_1024_128_2_4_4_4(buf_30, buf_16, (signed char *)bufs[144], (float *)bufs[145], buf_27);
  r_256_8192_4_2(buf_201, buf_19, buf_30, (signed char *)bufs[146], (float *)bufs[147]);
}