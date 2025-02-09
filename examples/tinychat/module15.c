#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_327, float* buf_306, int start_pos) {
  r_128_4_4(buf_5, buf_306);
  E_512_4n34(buf_309, buf_306, buf_5, (float *)bufs[228]);
  r_128_512_4_4(buf_311, buf_309, (signed char *)bufs[229], (float *)bufs[230]);
  r_128_512_4_4(buf_314, buf_309, (signed char *)bufs[231], (float *)bufs[232]);
  E_2_8_16_4(buf_317, buf_311, (float *)bufs[8], buf_314, start_pos);
  r_512_512_4_4n34(buf_16, buf_309, (signed char *)bufs[233], (float *)bufs[234]);
  E_32_16_2_2(buf_19, buf_16, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_19, buf_317, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_16, buf_23, buf_317, start_pos);
  r_512_2048_4(buf_19, buf_306, buf_16, (signed char *)bufs[235], (float *)bufs[236]);
  r_128_4_4(buf_5, buf_19);
  E_512_4n34(buf_16, buf_19, buf_5, (float *)bufs[237]);
  r_512_128_4_4_4_4(buf_27, buf_16, (signed char *)bufs[238], (float *)bufs[239]);
  r_1024_128_2_4_4_4(buf_30, buf_16, (signed char *)bufs[240], (float *)bufs[241], buf_27);
  r_256_8192_4_2(buf_327, buf_19, buf_30, (signed char *)bufs[242], (float *)bufs[243]);
}