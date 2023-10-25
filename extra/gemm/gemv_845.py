old = """__kernel void re_S256_16_8( write_only image2d_t data0, read_only image2d_t data1, read_only image2d_t data2, __global float* data3 ) {
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int idx2 = get_global_id(0); /* 4 */
  int idx1 = get_global_id(1); /* 16 */
  int idx0 = get_global_id(2); /* 256 */
  float acc0 = 0.0f;
  for (int idx3 = 0; idx3 < 8; idx3++) {
    float4 val1_0 = read_imagef(data1, smp, (int2)(((idx1*8)+idx3), 0)) /* (1, 128, 4) */;
    float4 val2_0 = read_imagef(data2, smp, (int2)(((idx1*32)+(idx3*4)+idx2), idx0)) /* (256, 512, 4) */;
    acc0+=(val1_0.x*val2_0.x);
    acc0+=(val1_0.y*val2_0.y);
    acc0+=(val1_0.z*val2_0.z);
    acc0+=(val1_0.w*val2_0.w);
  }
  __local float temp[64];
  temp[((idx1*4)+idx2)] = acc0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((idx1*4)+idx2) == 0) {
    float4 output0 = (float4)(0.0f,0.0f,0.0f,0.0f);
    for (int mid = 0; mid < 16; mid++) {
      float4 val5_0 = ((__local float4*)temp)[mid];
      output0.x+=val5_0.x;
      output0.y+=val5_0.y;
      output0.z+=val5_0.z;
      output0.w+=val5_0.w;
    }
    float4 val3_0 = ((__global float4*)data3)[idx0];
    write_imagef(data0, (int2)(idx0, 0), (float4)(max((output0.x+val3_0.x),(0.0f)),max((output0.y+val3_0.y),(0.0f)),max((output0.z+val3_0.z),(0.0f)),max((output0.w+val3_0.w),(0.0f))));  /* (1, 256, 4) */
  }
}"""

new = """__kernel void r_256_16_4_8_4(write_only image2d_t data0, read_only image2d_t data1, read_only image2d_t data2, const __global float* data3) {
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  __attribute__ ((aligned (16))) __local float temp[64];
  int gidx0 = get_group_id(0); /* 256 */
  int lidx1 = get_local_id(1); /* 16 */
  int lidx2 = get_local_id(0); /* 4 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 8; ++ridx0) {
    float4 val0 = read_imagef(data1, smp, (int2)(((lidx1*8)+ridx0),0));
    float4 val1 = read_imagef(data2, smp, (int2)(((lidx1*32)+lidx2+(ridx0*4)),gidx0));
    acc0 = (((val0).x*(val1).x)+acc0);
    acc0 = (((val0).y*(val1).y)+acc0);
    acc0 = (((val0).z*(val1).z)+acc0);
    acc0 = (((val0).w*(val1).w)+acc0);
  }
  temp[(lidx1*4)+lidx2] = acc0;
  barrier(CLK_LOCAL_MEM_FENCE);
  float4 acc1 = (float4)(0.0f,0.0f,0.0f,0.0f);
  for (int ridx1 = 0; ridx1 < 16; ++ridx1) {
    float4 val2 = (float4)(*((__local float4*)(temp+ridx1*4)));
    (acc1).x = ((val2).x+(acc1).x);
    (acc1).y = ((val2).y+(acc1).y);
    (acc1).z = ((val2).z+(acc1).z);
    (acc1).w = ((val2).w+(acc1).w);
  }
  float4 val3 = (float4)(*((__global float4*)(data3+gidx0*4)));
  write_imagef(data0, (int2)(gidx0,0), (float4)(max(((acc1).x+(val3).x),0.0f),max(((acc1).y+(val3).y),0.0f),max(((acc1).z+(val3).z),0.0f),max(((acc1).w+(val3).w),0.0f)));
}"""

from tinygrad.runtime.ops_gpu import CLBuffer, CLProgram
from tinygrad.helpers import dtypes, prod

if __name__ == "__main__":
  out = CLBuffer(prod((1, 128, 4)), dtypes.imageh((1,128,4)))
  x = CLBuffer(prod((1, 128, 4)), dtypes.imageh((1,128,4)))
  w = CLBuffer(prod((256, 512, 4)), dtypes.imageh((256, 512, 4)))
  b = CLBuffer(1024, dtypes.float)

  old = CLProgram("re_S256_16_8", old)
  new = CLProgram("r_256_16_4_8_4", new)

  old_tms = []
  new_tms = []

  for i in range(5):
    old_tms.append(old([1,1,256], [4,16,1], out, x, w, b, wait=True))
    new_tms.append(new([256,1,1], [4,16,1], out, x, w, b, wait=True))

  print(f"old: {min(old_tms)*1e6:.2f} us  new: {min(new_tms)*1e6:.2f} us")


