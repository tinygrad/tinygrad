#include <tgmath.h>
#include <stddef.h>
#include "kernels.h"

void net(float* buf_285, float* buf_264, int start_pos) {
  r_128_4_4(buf_5, buf_264);
  E_512_4n34(buf_267, buf_264, buf_5, (float *)bufs[196]);
  r_128_512_4_4(buf_269, buf_267, (signed char *)bufs[197], (float *)bufs[198]);
  r_128_512_4_4(buf_272, buf_267, (signed char *)bufs[199], (float *)bufs[200]);
  E_2_8_16_4(buf_275, buf_269, (float *)bufs[8], buf_272, start_pos);
  r_512_512_4_4n34(buf_16, buf_267, (signed char *)bufs[201], (float *)bufs[202]);
  E_32_16_2_2(buf_19, buf_16, (float *)bufs[8], start_pos);
  r_28start_pos2B129_4_64_4_2(buf_20, buf_19, buf_275, start_pos);
  r_2_28start_pos2B129_4_2_2(buf_21, buf_20, start_pos);
  r_2_28start_pos2B129_4_4(buf_22, buf_20, buf_21, start_pos);
  E_16_28start_pos2B129_2(buf_23, buf_20, buf_21, buf_22, start_pos);
  r_8_28start_pos2B129_4_16_4(buf_16, buf_23, buf_275, start_pos);
  r_512_2048_4(buf_19, buf_264, buf_16, (signed char *)bufs[203], (float *)bufs[204]);
  r_128_4_4(buf_5, buf_19);
  E_512_4n34(buf_16, buf_19, buf_5, (float *)bufs[205]);
  r_512_128_4_4_4_4(buf_27, buf_16, (signed char *)bufs[206], (float *)bufs[207]);
  r_1024_128_2_4_4_4(buf_30, buf_16, (signed char *)bufs[208], (float *)bufs[209], buf_27);
  r_256_8192_4_2(buf_285, buf_19, buf_30, (signed char *)bufs[210], (float *)bufs[211]);
}