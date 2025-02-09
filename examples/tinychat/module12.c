#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_264, float* buf_243, int start_pos) {
  r_128_4_4(buf_5, buf_243);
  E_512_4n34(buf_246, buf_243, buf_5, (float *)bufs[180]);
  r_128_512_4_4(buf_248, buf_246, (signed char *)bufs[181], (float *)bufs[182]);
  r_128_512_4_4(buf_251, buf_246, (signed char *)bufs[183], (float *)bufs[184]);
  E_2_8_16_4(buf_254, buf_248, (float *)bufs[8], buf_251, start_pos);
  r_512_512_4_4n34(buf_19, buf_246, (signed char *)bufs[185], (float *)bufs[186]);
  E_32_16_2_2(buf_16, buf_19, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_16, buf_254, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_19, buf_23, buf_254, start_pos);
  r_512_2048_4(buf_16, buf_243, buf_19, (signed char *)bufs[187], (float *)bufs[188]);
  r_128_4_4(buf_5, buf_16);
  E_512_4n34(buf_19, buf_16, buf_5, (float *)bufs[189]);
  r_512_128_4_4_4_4(buf_27, buf_19, (signed char *)bufs[190], (float *)bufs[191]);
  r_1024_128_2_4_4_4(buf_30, buf_19, (signed char *)bufs[192], (float *)bufs[193], buf_27);
  r_256_8192_4_2(buf_264, buf_16, buf_30, (signed char *)bufs[194], (float *)bufs[195]);
}