#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_117, float* buf_96, int start_pos) {
  r_128_4_4(buf_5, buf_96);
  E_512_4n34(buf_99, buf_96, buf_5, (float *)bufs[68]);
  r_128_512_4_4(buf_101, buf_99, (signed char *)bufs[69], (float *)bufs[70]);
  r_128_512_4_4(buf_104, buf_99, (signed char *)bufs[71], (float *)bufs[72]);
  E_2_8_16_4(buf_107, buf_101, (float *)bufs[8], buf_104, start_pos);
  r_512_512_4_4n34(buf_16, buf_99, (signed char *)bufs[73], (float *)bufs[74]);
  E_32_16_2_2(buf_19, buf_16, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_19, buf_107, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_16, buf_23, buf_107, start_pos);
  r_512_2048_4(buf_19, buf_96, buf_16, (signed char *)bufs[75], (float *)bufs[76]);
  r_128_4_4(buf_5, buf_19);
  E_512_4n34(buf_16, buf_19, buf_5, (float *)bufs[77]);
  r_512_128_4_4_4_4(buf_27, buf_16, (signed char *)bufs[78], (float *)bufs[79]);
  r_1024_128_2_4_4_4(buf_30, buf_16, (signed char *)bufs[80], (float *)bufs[81], buf_27);
  r_256_8192_4_2(buf_117, buf_19, buf_30, (signed char *)bufs[82], (float *)bufs[83]);
}