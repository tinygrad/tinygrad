from tinygrad.runtime.opencl import CLBuffer, CLProgram, CLImage, OSX_TIMING_RATIO
def benchmark(prog):
  e = prog()
  e.wait()
  return ((e.profile.end - e.profile.start) * OSX_TIMING_RATIO)
def mb(prog, N=10): return min([benchmark(prog) for _ in range(N)])

a = CLBuffer(2048*2048*4)
b = CLBuffer(2048*2048*4)
c = CLBuffer(2048*2048*4)

FLOPS = 2048*2048*2048*2

print("*** empty loop ***")
prog = CLProgram("test", """__kernel void test(__global float4 *a, __global float *data1, __global float *data2) {
  size_t idx1 = get_global_id(0); /* 512 */
  size_t idx0 = get_global_id(1); /* 512 */
  __local float ldata1[256];
  __local float ldata2[256];
  size_t lidx1 = get_local_id(0); /* 8 */
  size_t lidx0 = get_local_id(1); /* 8 */
  float4 acc = 0;
  for (int idx2 = 0; idx2 < 256; idx2++) {
    acc += idx2;
  }
  a[idx1*512+idx0] = acc;
}""")
tm = mb(lambda: prog([512,512], [8,8], a._cl, b._cl, c._cl))
print(f"{512*512:10d} {tm*1e-3:9.2f} us, would be {FLOPS/tm:.2f} GFLOPS matmul")

print("*** load global memory ***")
prog = CLProgram("test", """__kernel void test(__global float4 *a, __global float *data1, __global float *data2) {
  size_t idx1 = get_global_id(0); /* 512 */
  size_t idx0 = get_global_id(1); /* 512 */
  __local float ldata1[256];
  __local float ldata2[256];
  size_t lidx1 = get_local_id(0); /* 8 */
  size_t lidx0 = get_local_id(1); /* 8 */
  float4 acc = 0;
  for (int idx2 = 0; idx2 < 256; idx2++) {
    float val1_0 = data1[(lidx1+(idx0*8192)+(idx2*8))];
    float val1_2048 = data1[(lidx1+(idx0*8192)+(idx2*8)+2048)];
    float val1_4096 = data1[(lidx1+(idx0*8192)+(idx2*8)+4096)];
    float val1_6144 = data1[(lidx1+(idx0*8192)+(idx2*8)+6144)];
    float4 val2_0 = ((__global float4*)data2)[((lidx0*512)+idx1+(idx2*4096))];
    acc += (float4)(val1_0,val1_2048,val1_4096,val1_6144);
    acc += (float4)(val2_0.x,val2_0.y,val2_0.z,val2_0.w);
  }
  a[idx1*512+idx0] = acc;
}""")
tm = mb(lambda: prog([512,512], [8,8], a._cl, b._cl, c._cl))
print(f"{512*512:10d} {tm*1e-3:9.2f} us, would be {FLOPS/tm:.2f} GFLOPS matmul")

print("*** load global / store local memory ***")
prog = CLProgram("test", """__kernel void test(__global float4 *a, __global float *data1, __global float *data2) {
  size_t idx1 = get_global_id(0); /* 512 */
  size_t idx0 = get_global_id(1); /* 512 */
  __local float ldata1[256];
  __local float ldata2[256];
  size_t lidx1 = get_local_id(0); /* 8 */
  size_t lidx0 = get_local_id(1); /* 8 */
  for (int idx2 = 0; idx2 < 256; idx2++) {
    float val1_0 = data1[(lidx1+(idx0*8192)+(idx2*8))];
    float val1_2048 = data1[(lidx1+(idx0*8192)+(idx2*8)+2048)];
    float val1_4096 = data1[(lidx1+(idx0*8192)+(idx2*8)+4096)];
    float val1_6144 = data1[(lidx1+(idx0*8192)+(idx2*8)+6144)];
    //((__local float4*)ldata1)[((lidx0*8)+lidx1)] = (float4)(val1_0,val1_2048,val1_4096,val1_6144);
    ((__local float4*)ldata1)[(lidx0+(lidx1*8))] = (float4)(val1_0,val1_2048,val1_4096,val1_6144);
    float4 val2_0 = ((__global float4*)data2)[((lidx0*512)+idx1+(idx2*4096))];
    ((__local float4*)ldata2)[(lidx1+(lidx0*8))] = (float4)(val2_0.x,val2_0.y,val2_0.z,val2_0.w);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  a[idx1*512+idx0] = ldata1[lidx1] + ldata2[lidx0];
}""")
tm = mb(lambda: prog([512,512], [8,8], a._cl, b._cl, c._cl))
print(f"{512*512:10d} {tm*1e-3:9.2f} us, would be {FLOPS/tm:.2f} GFLOPS matmul")

print("*** load local memory ***")
prog = CLProgram("test", """__kernel void test(__global float4 *a, __global float4 *b, __global float4 *c) {
  size_t idx1 = get_global_id(0); /* 512 */
  size_t idx0 = get_global_id(1); /* 512 */
  __local float ldata1[256];
  __local float ldata2[256];
  size_t lidx1 = get_local_id(0); /* 8 */
  size_t lidx0 = get_local_id(1); /* 8 */
  ldata1[lidx0] = lidx0;
  ldata2[lidx1] = lidx1;
  barrier(CLK_LOCAL_MEM_FENCE);
  float4 acc = 0;
  for (int i = 0; i < 256; i++) {
    ldata1[((lidx0*8)+lidx1)] = lidx0;
    ldata2[(lidx1+(lidx0*8))] = lidx1;
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 lval1_0 = ((__local float4*)ldata1)[lidx0];
    float4 lval1_32 = ((__local float4*)ldata1)[(lidx0+8)];
    float4 lval1_64 = ((__local float4*)ldata1)[(lidx0+16)];
    float4 lval1_96 = ((__local float4*)ldata1)[(lidx0+24)];
    float4 lval1_128 = ((__local float4*)ldata1)[(lidx0+32)];
    float4 lval1_160 = ((__local float4*)ldata1)[(lidx0+40)];
    float4 lval1_192 = ((__local float4*)ldata1)[(lidx0+48)];
    float4 lval1_224 = ((__local float4*)ldata1)[(lidx0+56)];
    acc += lval1_0;
    acc += lval1_32;
    acc += lval1_64;
    acc += lval1_96;
    acc += lval1_128;
    acc += lval1_160;
    acc += lval1_192;
    acc += lval1_224;
    float4 lval2_0 = ((__local float4*)ldata2)[lidx1];
    float4 lval2_32 = ((__local float4*)ldata2)[(lidx1+8)];
    float4 lval2_64 = ((__local float4*)ldata2)[(lidx1+16)];
    float4 lval2_96 = ((__local float4*)ldata2)[(lidx1+24)];
    float4 lval2_128 = ((__local float4*)ldata2)[(lidx1+32)];
    float4 lval2_160 = ((__local float4*)ldata2)[(lidx1+40)];
    float4 lval2_192 = ((__local float4*)ldata2)[(lidx1+48)];
    float4 lval2_224 = ((__local float4*)ldata2)[(lidx1+56)];
    acc += lval2_0;
    acc += lval2_32;
    acc += lval2_64;
    acc += lval2_96;
    acc += lval2_128;
    acc += lval2_160;
    acc += lval2_192;
    acc += lval2_224;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  a[idx1*512+idx0] = acc;
}""")
tm = mb(lambda: prog([512,512], [8,8], a._cl, b._cl, c._cl))
print(f"{512*512:10d} {tm*1e-3:9.2f} us, would be {FLOPS/tm:.2f} GFLOPS matmul")

print("*** GEMM and store ***")
prog = CLProgram("test", """__kernel void test(__global float *data0, __global float *data1, __global float *data2) {
  size_t idx1 = get_global_id(0); /* 512 */
  size_t idx0 = get_global_id(1); /* 512 */
  __local float ldata1[256];
  __local float ldata2[256];
  size_t lidx1 = get_local_id(0); /* 8 */
  size_t lidx0 = get_local_id(1); /* 8 */
  float acc0 = 0.0;
  float acc1 = 0.0;
  float acc2 = 0.0;
  float acc3 = 0.0;
  float acc4 = 0.0;
  float acc5 = 0.0;                                                                                                                                                                             float acc6 = 0.0;
  float acc7 = 0.0;
  float acc8 = 0.0;
  float acc9 = 0.0;
  float acc10 = 0.0;
  float acc11 = 0.0;
  float acc12 = 0.0;
  float acc13 = 0.0;
  float acc14 = 0.0;
  float acc15 = 0.0;
  for (int i = 0; i < 256; i++) {
    ldata1[((lidx0*8)+lidx1)] = lidx0;
    ldata2[(lidx1+(lidx0*8))] = lidx1;
    barrier(CLK_LOCAL_MEM_FENCE);
    /*float4 lval1_0 = ((__local float4*)ldata1)[lidx0];
    float4 lval1_32 = ((__local float4*)ldata1)[(lidx0+8)];
    float4 lval1_64 = ((__local float4*)ldata1)[(lidx0+16)];
    float4 lval1_96 = ((__local float4*)ldata1)[(lidx0+24)];
    float4 lval1_128 = ((__local float4*)ldata1)[(lidx0+32)];
    float4 lval1_160 = ((__local float4*)ldata1)[(lidx0+40)];
    float4 lval1_192 = ((__local float4*)ldata1)[(lidx0+48)];
    float4 lval1_224 = ((__local float4*)ldata1)[(lidx0+56)];*/
    float4 lval1_0 = {lidx0, lidx1, 2.0, i};
    float4 lval1_32 = {lidx0, lidx1, 2.0, i};
    float4 lval1_64 = {lidx0, lidx1, 2.0, i};
    float4 lval1_96 = {lidx0, lidx1, 2.0, i};
    float4 lval1_128 = {lidx0, lidx1, 2.0, i};
    float4 lval1_160 = {lidx0, lidx1, 2.0, i};
    float4 lval1_192 = {lidx0, lidx1, 2.0, i};
    float4 lval1_224 = {lidx0, lidx1, 2.0, i};
    /*float4 lval2_0 = ((__local float4*)ldata2)[lidx1];
    float4 lval2_32 = ((__local float4*)ldata2)[(lidx1+8)];
    float4 lval2_64 = ((__local float4*)ldata2)[(lidx1+16)];
    float4 lval2_96 = ((__local float4*)ldata2)[(lidx1+24)];
    float4 lval2_128 = ((__local float4*)ldata2)[(lidx1+32)];
    float4 lval2_160 = ((__local float4*)ldata2)[(lidx1+40)];
    float4 lval2_192 = ((__local float4*)ldata2)[(lidx1+48)];
    float4 lval2_224 = ((__local float4*)ldata2)[(lidx1+56)];*/
    float4 lval2_0 = lidx1;
    float4 lval2_32 = lidx1;
    float4 lval2_64 = lidx1;
    float4 lval2_96 = lidx1;
    float4 lval2_128 = lidx1;
    float4 lval2_160 = lidx1;
    float4 lval2_192 = lidx1;
    float4 lval2_224 = lidx1;
    acc0+=(lval1_0.x*lval2_0.x);
    acc1+=(lval1_0.x*lval2_0.y);
    acc2+=(lval1_0.x*lval2_0.z);
    acc3+=(lval1_0.x*lval2_0.w);
    acc4+=(lval1_0.y*lval2_0.x);
    acc5+=(lval1_0.y*lval2_0.y);
    acc6+=(lval1_0.y*lval2_0.z);
    acc7+=(lval1_0.y*lval2_0.w);
    acc8+=(lval1_0.z*lval2_0.x);
    acc9+=(lval1_0.z*lval2_0.y);
    acc10+=(lval1_0.z*lval2_0.z);
    acc11+=(lval1_0.z*lval2_0.w);
    acc12+=(lval1_0.w*lval2_0.x);
    acc13+=(lval1_0.w*lval2_0.y);
    acc14+=(lval1_0.w*lval2_0.z);
    acc15+=(lval1_0.w*lval2_0.w);
    acc0+=(lval1_32.x*lval2_32.x);
    acc1+=(lval1_32.x*lval2_32.y);
    acc2+=(lval1_32.x*lval2_32.z);
    acc3+=(lval1_32.x*lval2_32.w);
    acc4+=(lval1_32.y*lval2_32.x);
    acc5+=(lval1_32.y*lval2_32.y);
    acc6+=(lval1_32.y*lval2_32.z);
    acc7+=(lval1_32.y*lval2_32.w);
    acc8+=(lval1_32.z*lval2_32.x);
    acc9+=(lval1_32.z*lval2_32.y);
    acc10+=(lval1_32.z*lval2_32.z);
    acc11+=(lval1_32.z*lval2_32.w);
    acc12+=(lval1_32.w*lval2_32.x);
    acc13+=(lval1_32.w*lval2_32.y);
    acc14+=(lval1_32.w*lval2_32.z);
    acc15+=(lval1_32.w*lval2_32.w);
    acc0+=(lval1_64.x*lval2_64.x);
    acc1+=(lval1_64.x*lval2_64.y);
    acc2+=(lval1_64.x*lval2_64.z);
    acc3+=(lval1_64.x*lval2_64.w);
    acc4+=(lval1_64.y*lval2_64.x);
    acc5+=(lval1_64.y*lval2_64.y);
    acc6+=(lval1_64.y*lval2_64.z);
    acc7+=(lval1_64.y*lval2_64.w);
    acc8+=(lval1_64.z*lval2_64.x);
    acc9+=(lval1_64.z*lval2_64.y);
    acc10+=(lval1_64.z*lval2_64.z);
    acc11+=(lval1_64.z*lval2_64.w);
    acc12+=(lval1_64.w*lval2_64.x);
    acc13+=(lval1_64.w*lval2_64.y);
    acc14+=(lval1_64.w*lval2_64.z);
    acc15+=(lval1_64.w*lval2_64.w);
    acc0+=(lval1_96.x*lval2_96.x);
    acc2+=(lval1_96.x*lval2_96.z);
    acc3+=(lval1_96.x*lval2_96.w);
    acc4+=(lval1_96.y*lval2_96.x);
    acc5+=(lval1_96.y*lval2_96.y);
    acc6+=(lval1_96.y*lval2_96.z);
    acc7+=(lval1_96.y*lval2_96.w);
    acc8+=(lval1_96.z*lval2_96.x);
    acc9+=(lval1_96.z*lval2_96.y);
    acc10+=(lval1_96.z*lval2_96.z);
    acc11+=(lval1_96.z*lval2_96.w);
    acc12+=(lval1_96.w*lval2_96.x);
    acc13+=(lval1_96.w*lval2_96.y);
    acc14+=(lval1_96.w*lval2_96.z);
    acc15+=(lval1_96.w*lval2_96.w);
    acc0+=(lval1_128.x*lval2_128.x);
    acc1+=(lval1_128.x*lval2_128.y);
    acc2+=(lval1_128.x*lval2_128.z);
    acc3+=(lval1_128.x*lval2_128.w);
    acc4+=(lval1_128.y*lval2_128.x);
    acc5+=(lval1_128.y*lval2_128.y);
    acc6+=(lval1_128.y*lval2_128.z);
    acc7+=(lval1_128.y*lval2_128.w);
    acc8+=(lval1_128.z*lval2_128.x);
    acc9+=(lval1_128.z*lval2_128.y);
    acc10+=(lval1_128.z*lval2_128.z);
    acc11+=(lval1_128.z*lval2_128.w);
    acc12+=(lval1_128.w*lval2_128.x);
    acc13+=(lval1_128.w*lval2_128.y);
    acc14+=(lval1_128.w*lval2_128.z);
    acc15+=(lval1_128.w*lval2_128.w);
    acc0+=(lval1_160.x*lval2_160.x);
    acc1+=(lval1_160.x*lval2_160.y);
    acc2+=(lval1_160.x*lval2_160.z);
    acc3+=(lval1_160.x*lval2_160.w);
    acc4+=(lval1_160.y*lval2_160.x);
    acc5+=(lval1_160.y*lval2_160.y);
    acc6+=(lval1_160.y*lval2_160.z);
    acc7+=(lval1_160.y*lval2_160.w);
    acc8+=(lval1_160.z*lval2_160.x);
    acc9+=(lval1_160.z*lval2_160.y);
    acc10+=(lval1_160.z*lval2_160.z);
    acc11+=(lval1_160.z*lval2_160.w);
    acc12+=(lval1_160.w*lval2_160.x);
    acc13+=(lval1_160.w*lval2_160.y);
    acc14+=(lval1_160.w*lval2_160.z);
    acc15+=(lval1_160.w*lval2_160.w);
    acc0+=(lval1_192.x*lval2_192.x);
    acc1+=(lval1_192.x*lval2_192.y);
    acc2+=(lval1_192.x*lval2_192.z);
    acc3+=(lval1_192.x*lval2_192.w);
    acc4+=(lval1_192.y*lval2_192.x);
    acc5+=(lval1_192.y*lval2_192.y);
    acc6+=(lval1_192.y*lval2_192.z);
    acc7+=(lval1_192.y*lval2_192.w);
    acc8+=(lval1_192.z*lval2_192.x);
    acc9+=(lval1_192.z*lval2_192.y);
    acc10+=(lval1_192.z*lval2_192.z);
    acc11+=(lval1_192.z*lval2_192.w);
    acc12+=(lval1_192.w*lval2_192.x);
    acc13+=(lval1_192.w*lval2_192.y);
    acc14+=(lval1_192.w*lval2_192.z);
    acc15+=(lval1_192.w*lval2_192.w);
    acc0+=(lval1_224.x*lval2_224.x);
    acc1+=(lval1_224.x*lval2_224.y);
    acc2+=(lval1_224.x*lval2_224.z);
    acc3+=(lval1_224.x*lval2_224.w);
    acc4+=(lval1_224.y*lval2_224.x);
    acc5+=(lval1_224.y*lval2_224.y);
    acc6+=(lval1_224.y*lval2_224.z);
    acc7+=(lval1_224.y*lval2_224.w);
    acc8+=(lval1_224.z*lval2_224.x);
    acc9+=(lval1_224.z*lval2_224.y);
    acc10+=(lval1_224.z*lval2_224.z);
    acc11+=(lval1_224.z*lval2_224.w);
    acc12+=(lval1_224.w*lval2_224.x);
    acc13+=(lval1_224.w*lval2_224.y);
    acc14+=(lval1_224.w*lval2_224.z);
    acc15+=(lval1_224.w*lval2_224.w);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  ((__global float4*)data0)[((idx0*2048)+idx1)] = (float4)(acc0,acc1,acc2,acc3);
  ((__global float4*)data0)[((idx0*2048)+idx1+512)] = (float4)(acc4,acc5,acc6,acc7);
  ((__global float4*)data0)[((idx0*2048)+idx1+1024)] = (float4)(acc8,acc9,acc10,acc11);
  ((__global float4*)data0)[((idx0*2048)+idx1+1536)] = (float4)(acc12,acc13,acc14,acc15);
}""")
tm = mb(lambda: prog([512,512], [8,8], a._cl, b._cl, c._cl))
print(f"{512*512:10d} {tm*1e-3:9.2f} us, would be {FLOPS/tm:.2f} GFLOPS matmul")

print("*** load local GEMM and store ***")
prog = CLProgram("test", """__kernel void test(__global float *data0, __global float *data1, __global float *data2) {
  size_t idx1 = get_global_id(0); /* 512 */
  size_t idx0 = get_global_id(1); /* 512 */
  __local float ldata1[256];
  __local float ldata2[256];
  size_t lidx1 = get_local_id(0); /* 8 */
  size_t lidx0 = get_local_id(1); /* 8 */
  float acc0 = 0.0;
  float acc1 = 0.0;
  float acc2 = 0.0;
  float acc3 = 0.0;
  float acc4 = 0.0;
  float acc5 = 0.0;                                                                                                                                                                             float acc6 = 0.0;
  float acc7 = 0.0;
  float acc8 = 0.0;
  float acc9 = 0.0;
  float acc10 = 0.0;
  float acc11 = 0.0;
  float acc12 = 0.0;
  float acc13 = 0.0;
  float acc14 = 0.0;
  float acc15 = 0.0;
  for (int i = 0; i < 256; i++) {
    ldata1[((lidx0*8)+lidx1)] = lidx0;
    ldata2[(lidx1+(lidx0*8))] = lidx1;
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 lval1_0 = ((__local float4*)ldata1)[lidx0];
    float4 lval1_32 = ((__local float4*)ldata1)[(lidx0+8)];
    float4 lval1_64 = ((__local float4*)ldata1)[(lidx0+16)];
    float4 lval1_96 = ((__local float4*)ldata1)[(lidx0+24)];
    float4 lval1_128 = ((__local float4*)ldata1)[(lidx0+32)];
    float4 lval1_160 = ((__local float4*)ldata1)[(lidx0+40)];
    float4 lval1_192 = ((__local float4*)ldata1)[(lidx0+48)];
    float4 lval1_224 = ((__local float4*)ldata1)[(lidx0+56)];
    float4 lval2_0 = ((__local float4*)ldata2)[lidx1];
    float4 lval2_32 = ((__local float4*)ldata2)[(lidx1+8)];
    float4 lval2_64 = ((__local float4*)ldata2)[(lidx1+16)];
    float4 lval2_96 = ((__local float4*)ldata2)[(lidx1+24)];
    float4 lval2_128 = ((__local float4*)ldata2)[(lidx1+32)];
    float4 lval2_160 = ((__local float4*)ldata2)[(lidx1+40)];
    float4 lval2_192 = ((__local float4*)ldata2)[(lidx1+48)];
    float4 lval2_224 = ((__local float4*)ldata2)[(lidx1+56)];
    acc0+=(lval1_0.x*lval2_0.x);
    acc1+=(lval1_0.x*lval2_0.y);
    acc2+=(lval1_0.x*lval2_0.z);
    acc3+=(lval1_0.x*lval2_0.w);
    acc4+=(lval1_0.y*lval2_0.x);
    acc5+=(lval1_0.y*lval2_0.y);
    acc6+=(lval1_0.y*lval2_0.z);
    acc7+=(lval1_0.y*lval2_0.w);
    acc8+=(lval1_0.z*lval2_0.x);
    acc9+=(lval1_0.z*lval2_0.y);
    acc10+=(lval1_0.z*lval2_0.z);
    acc11+=(lval1_0.z*lval2_0.w);
    acc12+=(lval1_0.w*lval2_0.x);
    acc13+=(lval1_0.w*lval2_0.y);
    acc14+=(lval1_0.w*lval2_0.z);
    acc15+=(lval1_0.w*lval2_0.w);
    acc0+=(lval1_32.x*lval2_32.x);
    acc1+=(lval1_32.x*lval2_32.y);
    acc2+=(lval1_32.x*lval2_32.z);
    acc3+=(lval1_32.x*lval2_32.w);
    acc4+=(lval1_32.y*lval2_32.x);
    acc5+=(lval1_32.y*lval2_32.y);
    acc6+=(lval1_32.y*lval2_32.z);
    acc7+=(lval1_32.y*lval2_32.w);
    acc8+=(lval1_32.z*lval2_32.x);
    acc9+=(lval1_32.z*lval2_32.y);
    acc10+=(lval1_32.z*lval2_32.z);
    acc11+=(lval1_32.z*lval2_32.w);
    acc12+=(lval1_32.w*lval2_32.x);
    acc13+=(lval1_32.w*lval2_32.y);
    acc14+=(lval1_32.w*lval2_32.z);
    acc15+=(lval1_32.w*lval2_32.w);
    acc0+=(lval1_64.x*lval2_64.x);
    acc1+=(lval1_64.x*lval2_64.y);
    acc2+=(lval1_64.x*lval2_64.z);
    acc3+=(lval1_64.x*lval2_64.w);
    acc4+=(lval1_64.y*lval2_64.x);
    acc5+=(lval1_64.y*lval2_64.y);
    acc6+=(lval1_64.y*lval2_64.z);
    acc7+=(lval1_64.y*lval2_64.w);
    acc8+=(lval1_64.z*lval2_64.x);
    acc9+=(lval1_64.z*lval2_64.y);
    acc10+=(lval1_64.z*lval2_64.z);
    acc11+=(lval1_64.z*lval2_64.w);
    acc12+=(lval1_64.w*lval2_64.x);
    acc13+=(lval1_64.w*lval2_64.y);
    acc14+=(lval1_64.w*lval2_64.z);
    acc15+=(lval1_64.w*lval2_64.w);
    acc0+=(lval1_96.x*lval2_96.x);
    acc2+=(lval1_96.x*lval2_96.z);
    acc3+=(lval1_96.x*lval2_96.w);
    acc4+=(lval1_96.y*lval2_96.x);
    acc5+=(lval1_96.y*lval2_96.y);
    acc6+=(lval1_96.y*lval2_96.z);
    acc7+=(lval1_96.y*lval2_96.w);
    acc8+=(lval1_96.z*lval2_96.x);
    acc9+=(lval1_96.z*lval2_96.y);
    acc10+=(lval1_96.z*lval2_96.z);
    acc11+=(lval1_96.z*lval2_96.w);
    acc12+=(lval1_96.w*lval2_96.x);
    acc13+=(lval1_96.w*lval2_96.y);
    acc14+=(lval1_96.w*lval2_96.z);
    acc15+=(lval1_96.w*lval2_96.w);
    acc0+=(lval1_128.x*lval2_128.x);
    acc1+=(lval1_128.x*lval2_128.y);
    acc2+=(lval1_128.x*lval2_128.z);
    acc3+=(lval1_128.x*lval2_128.w);
    acc4+=(lval1_128.y*lval2_128.x);
    acc5+=(lval1_128.y*lval2_128.y);
    acc6+=(lval1_128.y*lval2_128.z);
    acc7+=(lval1_128.y*lval2_128.w);
    acc8+=(lval1_128.z*lval2_128.x);
    acc9+=(lval1_128.z*lval2_128.y);
    acc10+=(lval1_128.z*lval2_128.z);
    acc11+=(lval1_128.z*lval2_128.w);
    acc12+=(lval1_128.w*lval2_128.x);
    acc13+=(lval1_128.w*lval2_128.y);
    acc14+=(lval1_128.w*lval2_128.z);
    acc15+=(lval1_128.w*lval2_128.w);
    acc0+=(lval1_160.x*lval2_160.x);
    acc1+=(lval1_160.x*lval2_160.y);
    acc2+=(lval1_160.x*lval2_160.z);
    acc3+=(lval1_160.x*lval2_160.w);
    acc4+=(lval1_160.y*lval2_160.x);
    acc5+=(lval1_160.y*lval2_160.y);
    acc6+=(lval1_160.y*lval2_160.z);
    acc7+=(lval1_160.y*lval2_160.w);
    acc8+=(lval1_160.z*lval2_160.x);
    acc9+=(lval1_160.z*lval2_160.y);
    acc10+=(lval1_160.z*lval2_160.z);
    acc11+=(lval1_160.z*lval2_160.w);
    acc12+=(lval1_160.w*lval2_160.x);
    acc13+=(lval1_160.w*lval2_160.y);
    acc14+=(lval1_160.w*lval2_160.z);
    acc15+=(lval1_160.w*lval2_160.w);
    acc0+=(lval1_192.x*lval2_192.x);
    acc1+=(lval1_192.x*lval2_192.y);
    acc2+=(lval1_192.x*lval2_192.z);
    acc3+=(lval1_192.x*lval2_192.w);
    acc4+=(lval1_192.y*lval2_192.x);
    acc5+=(lval1_192.y*lval2_192.y);
    acc6+=(lval1_192.y*lval2_192.z);
    acc7+=(lval1_192.y*lval2_192.w);
    acc8+=(lval1_192.z*lval2_192.x);
    acc9+=(lval1_192.z*lval2_192.y);
    acc10+=(lval1_192.z*lval2_192.z);
    acc11+=(lval1_192.z*lval2_192.w);
    acc12+=(lval1_192.w*lval2_192.x);
    acc13+=(lval1_192.w*lval2_192.y);
    acc14+=(lval1_192.w*lval2_192.z);
    acc15+=(lval1_192.w*lval2_192.w);
    acc0+=(lval1_224.x*lval2_224.x);
    acc1+=(lval1_224.x*lval2_224.y);
    acc2+=(lval1_224.x*lval2_224.z);
    acc3+=(lval1_224.x*lval2_224.w);
    acc4+=(lval1_224.y*lval2_224.x);
    acc5+=(lval1_224.y*lval2_224.y);
    acc6+=(lval1_224.y*lval2_224.z);
    acc7+=(lval1_224.y*lval2_224.w);
    acc8+=(lval1_224.z*lval2_224.x);
    acc9+=(lval1_224.z*lval2_224.y);
    acc10+=(lval1_224.z*lval2_224.z);
    acc11+=(lval1_224.z*lval2_224.w);
    acc12+=(lval1_224.w*lval2_224.x);
    acc13+=(lval1_224.w*lval2_224.y);
    acc14+=(lval1_224.w*lval2_224.z);
    acc15+=(lval1_224.w*lval2_224.w);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  ((__global float4*)data0)[((idx0*2048)+idx1)] = (float4)(acc0,acc1,acc2,acc3);
  ((__global float4*)data0)[((idx0*2048)+idx1+512)] = (float4)(acc4,acc5,acc6,acc7);
  ((__global float4*)data0)[((idx0*2048)+idx1+1024)] = (float4)(acc8,acc9,acc10,acc11);
  ((__global float4*)data0)[((idx0*2048)+idx1+1536)] = (float4)(acc12,acc13,acc14,acc15);
}""")
tm = mb(lambda: prog([512,512], [8,8], a._cl, b._cl, c._cl))
print(f"{512*512:10d} {tm*1e-3:9.2f} us, would be {FLOPS/tm:.2f} GFLOPS matmul")