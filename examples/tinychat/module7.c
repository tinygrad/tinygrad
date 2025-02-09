#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_159, float* buf_138, int start_pos) {
  r_128_4_4(buf_5, buf_138);
  E_512_4n34(buf_141, buf_138, buf_5, (float *)bufs[100]);
  r_128_512_4_4(buf_143, buf_141, (signed char *)bufs[101], (float *)bufs[102]);
  r_128_512_4_4(buf_146, buf_141, (signed char *)bufs[103], (float *)bufs[104]);
  E_2_8_16_4(buf_149, buf_143, (float *)bufs[8], buf_146, start_pos);
  r_512_512_4_4n34(buf_16, buf_141, (signed char *)bufs[105], (float *)bufs[106]);
  E_32_16_2_2(buf_19, buf_16, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_19, buf_149, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_16, buf_23, buf_149, start_pos);
  r_512_2048_4(buf_19, buf_138, buf_16, (signed char *)bufs[107], (float *)bufs[108]);
  r_128_4_4(buf_5, buf_19);
  E_512_4n34(buf_16, buf_19, buf_5, (float *)bufs[109]);
  r_512_128_4_4_4_4(buf_27, buf_16, (signed char *)bufs[110], (float *)bufs[111]);
  r_1024_128_2_4_4_4(buf_30, buf_16, (signed char *)bufs[112], (float *)bufs[113], buf_27);
  r_256_8192_4_2(buf_159, buf_19, buf_30, (signed char *)bufs[114], (float *)bufs[115]);
}