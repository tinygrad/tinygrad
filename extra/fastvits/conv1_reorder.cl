//#pragma OPENCL EXTENSION cl_khr_fp16 : enable
//#define prec4 half4
//#define read_imagep read_imageh

#define prec4 float4
#define read_imagep read_imagef

/*float4 read_imagep(image2d_t data, sampler_t smp, int2 idx) {
  return read_imagef(data, smp, idx);
}*/

__kernel void r_32_16_16_64_4_4_4(write_only image2d_t data0, read_only image2d_t data1, read_only image2d_t data2) {
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int idx0 = get_global_id(2); /* 32 */
  int idx1 = get_global_id(1); /* 16 */
  int idx2 = get_global_id(0); /* 16 */

  float4 acc0 = 0.0f;
  float4 acc1 = 0.0f;
  float4 acc2 = 0.0f;
  float4 acc3 = 0.0f;

  // idx0 is a global
  // (32, 4096, 4)
  int2 imi0 = (int2)(((idx1*256)),idx0);
  int2 imi1 = (int2)(((idx1*256)+64),idx0);
  int2 imi2 = (int2)(((idx1*256)+128),idx0);
  int2 imi3 = (int2)(((idx1*256)+192),idx0);

  // idx2 is a local
  // (16, 256, 4)
  int2 imi4 = (int2)(0,idx2);
  int2 imi5 = (int2)(1,idx2);
  int2 imi6 = (int2)(2,idx2);
  int2 imi7 = (int2)(3,idx2);

  // 11% faster
  //for(;imi4.x < 256;) {
  //#pragma unroll(2)
  for (int ridx0 = 0; ridx0 < 64; ++ridx0) {
    prec4 val0 = read_imagep(data1, smp, imi0);
    prec4 val1 = read_imagep(data1, smp, imi1);
    prec4 val2 = read_imagep(data1, smp, imi2);
    prec4 val3 = read_imagep(data1, smp, imi3);
    prec4 val4 = read_imagep(data2, smp, imi4);
    prec4 val5 = read_imagep(data2, smp, imi5);
    prec4 val6 = read_imagep(data2, smp, imi6);
    prec4 val7 = read_imagep(data2, smp, imi7);
    imi0.x += 1;
    imi1.x += 1;
    imi2.x += 1;
    imi3.x += 1;
    imi4.x += 4;
    imi5.x += 4;
    imi6.x += 4;
    imi7.x += 4;

    /*(acc0).x = (((val0).x*(val4).x)+(acc0).x);
    (acc0).y = (((val0).x*(val4).y)+(acc0).y);
    (acc0).z = (((val0).x*(val4).z)+(acc0).z);
    (acc0).w = (((val0).x*(val4).w)+(acc0).w);
    (acc1).x = (((val1).x*(val4).x)+(acc1).x);
    (acc1).y = (((val1).x*(val4).y)+(acc1).y);
    (acc1).z = (((val1).x*(val4).z)+(acc1).z);
    (acc1).w = (((val1).x*(val4).w)+(acc1).w);
    (acc2).x = (((val2).x*(val4).x)+(acc2).x);
    (acc2).y = (((val2).x*(val4).y)+(acc2).y);
    (acc2).z = (((val2).x*(val4).z)+(acc2).z);
    (acc2).w = (((val2).x*(val4).w)+(acc2).w);
    (acc3).x = (((val3).x*(val4).x)+(acc3).x);
    (acc3).y = (((val3).x*(val4).y)+(acc3).y);
    (acc3).z = (((val3).x*(val4).z)+(acc3).z);
    (acc3).w = (((val3).x*(val4).w)+(acc3).w);*/

    //read_mem_fence(CLK_LOCAL_MEM_FENCE);

    /*(acc0).x = (((val0).x*(val4).x)+(acc0).x);
    (acc0).y = (((val0).x*(val4).y)+(acc0).y);
    (acc0).z = (((val0).x*(val4).z)+(acc0).z);
    (acc0).w = (((val0).x*(val4).w)+(acc0).w);*/

    acc0 = mad(val0.x, val4, acc0);
    acc0 = mad(val0.y, val5, acc0);
    acc0 = mad(val0.z, val6, acc0);
    acc0 = mad(val0.w, val7, acc0);

    acc1 = mad(val1.x, val4, acc1);
    acc1 = mad(val1.y, val5, acc1);
    acc1 = mad(val1.z, val6, acc1);
    acc1 = mad(val1.w, val7, acc1);

    acc2 = mad(val2.x, val4, acc2);
    acc2 = mad(val2.y, val5, acc2);
    acc2 = mad(val2.z, val6, acc2);
    acc2 = mad(val2.w, val7, acc2);

    acc3 = mad(val3.x, val4, acc3);
    acc3 = mad(val3.y, val5, acc3);
    acc3 = mad(val3.z, val6, acc3);
    acc3 = mad(val3.w, val7, acc3);

    /*acc0 = val0.x * val4 + acc0;
    acc1 = val1.x * val4 + acc1;
    acc2 = val2.x * val4 + acc2;
    acc3 = val3.x * val4 + acc3;*/

    /*acc0 = val0.y * val5 + acc0;
    acc1 = val1.y * val5 + acc1;
    acc2 = val2.y * val5 + acc2;
    acc3 = val3.y * val5 + acc3;
    acc0 = val0.z * val6 + acc0;
    acc1 = val1.z * val6 + acc1;
    acc2 = val2.z * val6 + acc2;
    acc3 = val3.z * val6 + acc3;
    acc0 = val0.w * val7 + acc0;
    acc1 = val1.w * val7 + acc1;
    acc2 = val2.w * val7 + acc2;
    acc3 = val3.w * val7 + acc3;*/

    /*(acc0).x = (((val0).y*(val5).x)+(acc0).x);
    (acc0).y = (((val0).y*(val5).y)+(acc0).y);
    (acc0).z = (((val0).y*(val5).z)+(acc0).z);
    (acc0).w = (((val0).y*(val5).w)+(acc0).w);
    (acc1).x = (((val1).y*(val5).x)+(acc1).x);
    (acc1).y = (((val1).y*(val5).y)+(acc1).y);
    (acc1).z = (((val1).y*(val5).z)+(acc1).z);
    (acc1).w = (((val1).y*(val5).w)+(acc1).w);
    (acc2).x = (((val2).y*(val5).x)+(acc2).x);
    (acc2).y = (((val2).y*(val5).y)+(acc2).y);
    (acc2).z = (((val2).y*(val5).z)+(acc2).z);
    (acc2).w = (((val2).y*(val5).w)+(acc2).w);
    (acc3).x = (((val3).y*(val5).x)+(acc3).x);
    (acc3).y = (((val3).y*(val5).y)+(acc3).y);
    (acc3).z = (((val3).y*(val5).z)+(acc3).z);
    (acc3).w = (((val3).y*(val5).w)+(acc3).w);
    (acc0).x = (((val0).z*(val6).x)+(acc0).x);
    (acc0).y = (((val0).z*(val6).y)+(acc0).y);
    (acc0).z = (((val0).z*(val6).z)+(acc0).z);
    (acc0).w = (((val0).z*(val6).w)+(acc0).w);
    (acc1).x = (((val1).z*(val6).x)+(acc1).x);
    (acc1).y = (((val1).z*(val6).y)+(acc1).y);
    (acc1).z = (((val1).z*(val6).z)+(acc1).z);
    (acc1).w = (((val1).z*(val6).w)+(acc1).w);
    (acc2).x = (((val2).z*(val6).x)+(acc2).x);
    (acc2).y = (((val2).z*(val6).y)+(acc2).y);
    (acc2).z = (((val2).z*(val6).z)+(acc2).z);
    (acc2).w = (((val2).z*(val6).w)+(acc2).w);
    (acc3).x = (((val3).z*(val6).x)+(acc3).x);
    (acc3).y = (((val3).z*(val6).y)+(acc3).y);
    (acc3).z = (((val3).z*(val6).z)+(acc3).z);
    (acc3).w = (((val3).z*(val6).w)+(acc3).w);
    (acc0).x = (((val0).w*(val7).x)+(acc0).x);
    (acc0).y = (((val0).w*(val7).y)+(acc0).y);
    (acc0).z = (((val0).w*(val7).z)+(acc0).z);
    (acc0).w = (((val0).w*(val7).w)+(acc0).w);
    (acc1).x = (((val1).w*(val7).x)+(acc1).x);
    (acc1).y = (((val1).w*(val7).y)+(acc1).y);
    (acc1).z = (((val1).w*(val7).z)+(acc1).z);
    (acc1).w = (((val1).w*(val7).w)+(acc1).w);
    (acc2).x = (((val2).w*(val7).x)+(acc2).x);
    (acc2).y = (((val2).w*(val7).y)+(acc2).y);
    (acc2).z = (((val2).w*(val7).z)+(acc2).z);
    (acc2).w = (((val2).w*(val7).w)+(acc2).w);
    (acc3).x = (((val3).w*(val7).x)+(acc3).x);
    (acc3).y = (((val3).w*(val7).y)+(acc3).y);
    (acc3).z = (((val3).w*(val7).z)+(acc3).z);
    (acc3).w = (((val3).w*(val7).w)+(acc3).w);*/
  }
  write_imagef(data0, (int2)(((idx1*64)+idx2),idx0), acc0); //(float4)((acc0).x,(acc0).y,(acc0).z,(acc0).w));
  write_imagef(data0, (int2)(((idx1*64)+idx2+16),idx0), acc1); //(float4)((acc1).x,(acc1).y,(acc1).z,(acc1).w));
  write_imagef(data0, (int2)(((idx1*64)+idx2+32),idx0), acc2); //(float4)((acc2).x,(acc2).y,(acc2).z,(acc2).w));
  write_imagef(data0, (int2)(((idx1*64)+idx2+48),idx0), acc3); //(float4)((acc3).x,(acc3).y,(acc3).z,(acc3).w));
}