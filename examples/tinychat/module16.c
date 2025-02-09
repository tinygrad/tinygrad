#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

float* net(float* buf_352, float* buf_327, int start_pos) {
  r_128_4_4(buf_5, buf_327);
  E_512_4n34(buf_330, buf_327, buf_5, (float *)bufs[244]);
  r_128_512_4_4(buf_332, buf_330, (signed char *)bufs[245], (float *)bufs[246]);
  r_128_512_4_4(buf_335, buf_330, (signed char *)bufs[247], (float *)bufs[248]);
  E_2_8_16_4(buf_338, buf_332, (float *)bufs[8], buf_335, start_pos);
  E_n2(buf_339);
  r_512_512_4_4n34(buf_19, buf_330, (signed char *)bufs[249], (float *)bufs[250]);
  // buf_343 is probably random seeds
  E_n3(buf_342, buf_339, buf_343);
  E_32_16_2_2(buf_16, buf_19, (float *)bufs[8], start_pos);
  E_n4(buf_5, buf_339, buf_343);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_16, buf_338, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_19, buf_23, buf_338, start_pos);
  r_512_2048_4(buf_16, buf_327, buf_19, (signed char *)bufs[251], (float *)bufs[252]);
  r_128_4_4(buf_346, buf_16);
  E_512_4n34(buf_19, buf_16, buf_346, (float *)bufs[253]);
  r_512_128_4_4_4_4(buf_27, buf_19, (signed char *)bufs[254], (float *)bufs[255]);
  r_1024_128_2_4_4_4(buf_30, buf_19, (signed char *)bufs[256], (float *)bufs[257], buf_27);
  r_256_8192_4_2(buf_352, buf_16, buf_30, (signed char *)bufs[258], (float *)bufs[259]);
  return buf_5;
}