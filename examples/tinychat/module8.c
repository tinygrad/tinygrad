#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_180, float* buf_159, int start_pos) {
  r_128_4_4(buf_5, buf_159);
  E_512_4n34(buf_162, buf_159, buf_5, (float *)bufs[116]);
  r_128_512_4_4(buf_164, buf_162, (signed char *)bufs[117], (float *)bufs[118]);
  r_128_512_4_4(buf_167, buf_162, (signed char *)bufs[119], (float *)bufs[120]);
  E_2_8_16_4(buf_170, buf_164, (float *)bufs[8], buf_167, start_pos);
  r_512_512_4_4n34(buf_19, buf_162, (signed char *)bufs[121], (float *)bufs[122]);
  E_32_16_2_2(buf_16, buf_19, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_16, buf_170, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_19, buf_23, buf_170, start_pos);
  r_512_2048_4(buf_16, buf_159, buf_19, (signed char *)bufs[123], (float *)bufs[124]);
  r_128_4_4(buf_5, buf_16);
  E_512_4n34(buf_19, buf_16, buf_5, (float *)bufs[125]);
  r_512_128_4_4_4_4(buf_27, buf_19, (signed char *)bufs[126], (float *)bufs[127]);
  r_1024_128_2_4_4_4(buf_30, buf_19, (signed char *)bufs[128], (float *)bufs[129], buf_27);
  r_256_8192_4_2(buf_180, buf_16, buf_30, (signed char *)bufs[130], (float *)bufs[131]);
}