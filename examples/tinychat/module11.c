#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_243, float* buf_222, int start_pos) {
  r_128_4_4(buf_5, buf_222);
  E_512_4n34(buf_225, buf_222, buf_5, (float *)bufs[164]);
  r_128_512_4_4(buf_227, buf_225, (signed char *)bufs[165], (float *)bufs[166]);
  r_128_512_4_4(buf_230, buf_225, (signed char *)bufs[167], (float *)bufs[168]);
  E_2_8_16_4(buf_233, buf_227, (float *)bufs[8], buf_230, start_pos);
  r_512_512_4_4n34(buf_16, buf_225, (signed char *)bufs[169], (float *)bufs[170]);
  E_32_16_2_2(buf_19, buf_16, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_19, buf_233, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_16, buf_23, buf_233, start_pos);
  r_512_2048_4(buf_19, buf_222, buf_16, (signed char *)bufs[171], (float *)bufs[172]);
  r_128_4_4(buf_5, buf_19);
  E_512_4n34(buf_16, buf_19, buf_5, (float *)bufs[173]);
  r_512_128_4_4_4_4(buf_27, buf_16, (signed char *)bufs[174], (float *)bufs[175]);
  r_1024_128_2_4_4_4(buf_30, buf_16, (signed char *)bufs[176], (float *)bufs[177], buf_27);
  r_256_8192_4_2(buf_243, buf_19, buf_30, (signed char *)bufs[178], (float *)bufs[179]);
}