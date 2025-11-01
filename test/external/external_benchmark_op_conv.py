# ruff: noqa: E501
from tinygrad import dtypes, Device
from tinygrad.uop.ops import UOp, AxisType, Ops
from tinygrad.codegen import full_rewrite
from tinygrad.renderer import ProgramSpec
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import dedup
from tinygrad.device import Buffer
from tinygrad.dtype import ImageDType

# PYTHONPATH="." DEBUG=5 DEV=QCOM FLOAT16=1 IMAGE=2 NOLOCALS=1 taskset -c 4-7 python3 examples/openpilot/compile3.py https://github.com/commaai/openpilot/raw/720392c9a5b986981fdbed1bb8c47a6c5573a50e/selfdrive/modeld/models/driving_vision.onnx
# kernel 672
# faster on d59d4cd, 50% slower with the new linearizer

""" d59d4cd
c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((32, 1024, 4)), arg=0, src=())
c1 = UOp.range(UOp.const(dtypes.index, 64), 3, AxisType.LOOP)
c2 = UOp.range(UOp.const(dtypes.index, 64), 4, AxisType.LOOP)
c3 = UOp.range(UOp.const(dtypes.index, 32), 2, AxisType.LOOP)
c4 = (((c1*UOp.const(dtypes.index, 64))+c2)+(c3*UOp.const(dtypes.index, 4096)))
c5 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((32, 1024, 4)), arg=1, src=())
c6 = c5.index(c4).load()
c7 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((32, 3072, 4)), arg=2, src=())
c8 = UOp.range(UOp.const(dtypes.index, 48), 0, AxisType.REDUCE)
c9 = UOp.range(UOp.const(dtypes.index, 4), 1, AxisType.REDUCE)
c10 = c7.index(((((c8*UOp.const(dtypes.index, 4))+c9)+(c1*UOp.const(dtypes.index, 192)))+(c3*UOp.const(dtypes.index, 12288)))).load()
c11 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((16, 192, 4)), arg=3, src=())
c12 = c11.index(((((c9*UOp.const(dtypes.index, 4))+(c2%UOp.const(dtypes.index, 4)))+(c8*UOp.const(dtypes.index, 16)))+((c2//UOp.const(dtypes.index, 4))*UOp.const(dtypes.index, 768)))).load()
c13 = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(64), arg=4, src=())
c14 = c13.index(c2).load()
c15 = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(64), arg=5, src=())
c16 = c15.index(c2).load()
c17 = (c6+(((c10*c12.cast(dtypes.float)).cast(dtypes.float).reduce(c8, c9, arg=Ops.ADD)+c14.cast(dtypes.float))*c16.cast(dtypes.float)))
c18 = c0.index(c4).store(c17, c3, c1, c2)
ast = c18.sink()
more upcast axis : [(3, 320, 0, 4)]
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void r_512_16_4_4_48_4(write_only image2d_t data0_131072, read_only image2d_t data1_131072, read_only image2d_t data2_393216, read_only image2d_t data3_12288, __global half* data4_64, __global half* data5_64) {
const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  float acc0[16];
  int idx0 = get_global_id(0); /* 16 */
  int idx1 = get_global_id(1); /* 512 */
  int alu0 = (idx1>>4);
  *(acc0+0) = 0.0f;
  *(acc0+1) = 0.0f;
  *(acc0+2) = 0.0f;
  *(acc0+3) = 0.0f;
  *(acc0+4) = 0.0f;
  *(acc0+5) = 0.0f;
  *(acc0+6) = 0.0f;
  *(acc0+7) = 0.0f;
  *(acc0+8) = 0.0f;
  *(acc0+9) = 0.0f;
  *(acc0+10) = 0.0f;
  *(acc0+11) = 0.0f;
  *(acc0+12) = 0.0f;
  *(acc0+13) = 0.0f;
  *(acc0+14) = 0.0f;
  *(acc0+15) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 48; Ridx0++) {
    int alu17 = ((idx1*192)+Ridx0);
    int alu18 = (alu17+48);
    int alu19 = (alu17+96);
    int alu20 = (alu17+144);
    int alu21 = (Ridx0<<2);
    float4 val0 = read_imagef(data3_12288, smp, (int2)(alu21,idx0));
    float4 val1 = read_imagef(data3_12288, smp, (int2)((alu21+1),idx0));
    float4 val2 = read_imagef(data3_12288, smp, (int2)((alu21+2),idx0));
    float4 val3 = read_imagef(data3_12288, smp, (int2)((alu21+3),idx0));
    float4 val4 = read_imagef(data2_393216, smp, (int2)((alu18-(3072*(((alu18>>10)*43)>>7))),alu0));
    float4 val5 = read_imagef(data2_393216, smp, (int2)((alu19-(3072*(((alu19>>10)*43)>>7))),alu0));
    float4 val6 = read_imagef(data2_393216, smp, (int2)((alu20-(3072*(((alu20>>10)*43)>>7))),alu0));
    float4 val7 = read_imagef(data2_393216, smp, (int2)((alu17-(3072*(((alu17>>10)*43)>>7))),alu0));
    *(acc0+1) = ((*(acc0+1))+(val4.x*val0.x)+(val4.y*val1.x)+(val4.z*val2.x)+(val4.w*val3.x));
    *(acc0+5) = ((*(acc0+5))+(val4.x*val0.y)+(val4.y*val1.y)+(val4.z*val2.y)+(val4.w*val3.y));
    *(acc0+9) = ((*(acc0+9))+(val4.x*val0.z)+(val4.y*val1.z)+(val4.z*val2.z)+(val4.w*val3.z));
    *(acc0+13) = ((*(acc0+13))+(val4.x*val0.w)+(val4.y*val1.w)+(val4.z*val2.w)+(val4.w*val3.w));
    *(acc0+2) = ((*(acc0+2))+(val5.x*val0.x)+(val5.y*val1.x)+(val5.z*val2.x)+(val5.w*val3.x));
    *(acc0+6) = ((*(acc0+6))+(val5.x*val0.y)+(val5.y*val1.y)+(val5.z*val2.y)+(val5.w*val3.y));
    *(acc0+10) = ((*(acc0+10))+(val5.x*val0.z)+(val5.y*val1.z)+(val5.z*val2.z)+(val5.w*val3.z));
    *(acc0+14) = ((*(acc0+14))+(val5.x*val0.w)+(val5.y*val1.w)+(val5.z*val2.w)+(val5.w*val3.w));
    *(acc0+3) = ((*(acc0+3))+(val6.x*val0.x)+(val6.y*val1.x)+(val6.z*val2.x)+(val6.w*val3.x));
    *(acc0+7) = ((*(acc0+7))+(val6.x*val0.y)+(val6.y*val1.y)+(val6.z*val2.y)+(val6.w*val3.y));
    *(acc0+11) = ((*(acc0+11))+(val6.x*val0.z)+(val6.y*val1.z)+(val6.z*val2.z)+(val6.w*val3.z));
    *(acc0+15) = ((*(acc0+15))+(val6.x*val0.w)+(val6.y*val1.w)+(val6.z*val2.w)+(val6.w*val3.w));
    *(acc0+0) = ((*(acc0+0))+(val7.x*val0.x)+(val7.y*val1.x)+(val7.z*val2.x)+(val7.w*val3.x));
    *(acc0+4) = ((*(acc0+4))+(val7.x*val0.y)+(val7.y*val1.y)+(val7.z*val2.y)+(val7.w*val3.y));
    *(acc0+8) = ((*(acc0+8))+(val7.x*val0.z)+(val7.y*val1.z)+(val7.z*val2.z)+(val7.w*val3.z));
    *(acc0+12) = ((*(acc0+12))+(val7.x*val0.w)+(val7.y*val1.w)+(val7.z*val2.w)+(val7.w*val3.w));
  }
  int alu39 = (idx0<<2);
  half4 val8 = (*((__global half4*)((data4_64+alu39))));
  half4 val9 = (*((__global half4*)((data5_64+alu39))));
  int alu40 = (idx0+(idx1<<6));
  int2 cast0 = (int2)((alu40&1023),alu0);
  float4 val10 = read_imagef(data1_131072, smp, cast0);
  int2 cast1 = (int2)(((alu40+16)&1023),alu0);
  float4 val11 = read_imagef(data1_131072, smp, cast1);
  int2 cast2 = (int2)(((alu40+32)&1023),alu0);
  float4 val12 = read_imagef(data1_131072, smp, cast2);
  int2 cast3 = (int2)(((alu40+48)&1023),alu0);
  float4 val13 = read_imagef(data1_131072, smp, cast3);
  float cast4 = ((float)(val8.x));
  float cast5 = ((float)(val9.x));
  float cast6 = ((float)(val8.y));
  float cast7 = ((float)(val9.y));
  float cast8 = ((float)(val8.z));
  float cast9 = ((float)(val9.z));
  float cast10 = ((float)(val8.w));
  float cast11 = ((float)(val9.w));
  write_imagef(data0_131072, cast0, (float4)((val10.x+(((*(acc0+0))+cast4)*cast5)),(val10.y+(((*(acc0+4))+cast6)*cast7)),(val10.z+(((*(acc0+8))+cast8)*cast9)),(val10.w+(((*(acc0+12))+cast10)*cast11))));
  write_imagef(data0_131072, cast1, (float4)((val11.x+(((*(acc0+1))+cast4)*cast5)),(val11.y+(((*(acc0+5))+cast6)*cast7)),(val11.z+(((*(acc0+9))+cast8)*cast9)),(val11.w+(((*(acc0+13))+cast10)*cast11))));
  write_imagef(data0_131072, cast2, (float4)((val12.x+(((*(acc0+2))+cast4)*cast5)),(val12.y+(((*(acc0+6))+cast6)*cast7)),(val12.z+(((*(acc0+10))+cast8)*cast9)),(val12.w+(((*(acc0+14))+cast10)*cast11))));
  write_imagef(data0_131072, cast3, (float4)((val13.x+(((*(acc0+3))+cast4)*cast5)),(val13.y+(((*(acc0+7))+cast6)*cast7)),(val13.z+(((*(acc0+11))+cast8)*cast9)),(val13.w+(((*(acc0+15))+cast10)*cast11))));
}
*** QCOM     672 r_512_16_4_4_48_4                              arg  6 mem   0.10 GB tm    322.55us/    77.83ms (    157 GFLOPS    4|160    GB/s) ['mul', '__add__', 'conv2d']
"""

""" master 99e76f33a0f4ec84c79c1271dbc955fe6b5a7778
c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((32, 1024, 4)), (), 0)
c2 = UOp.range(64, 3, AxisType.LOOP)
c4 = UOp.range(64, 4, AxisType.LOOP)
c7 = UOp.range(32, 2, AxisType.LOOP)
c10 = (((c2*64)+c4)+(c7*4096))
c12 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((32, 1024, 4)), (), 1)
c14 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((32, 3072, 4)), (), 2)
c16 = UOp.range(48, 0, AxisType.REDUCE)
c19 = UOp.range(4, 1, AxisType.REDUCE)
c28 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((16, 192, 4)), (), 3)
c40 = (c14.index(((((c16*4)+c19)+(c2*192))+(c7*12288)))*c28.index(((((c19*4)+(c4%4))+(c16*16))+((c4//4)*768))))
c42 = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(64), (), 4)
c46 = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(64), (), 5)
c50 = (c12.index(c10)+((c40.reduce(c16, c19, arg=Ops.ADD)+c42.index(c4).cast(dtypes.float))*c46.index(c4).cast(dtypes.float)))
c52 = c0.index(c10, ptr=True).store(c50).end(c7, c2, c4)
ast = c52.sink()
more upcast axis : [(3, 320, 0, 4)]
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void r_512_16_4_4_48_4(write_only image2d_t data0_131072, read_only image2d_t data1_131072, read_only image2d_t data2_393216, read_only image2d_t data3_12288, __global half* data4_64, __global half* data5_64) {
const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  float acc0[16];
  int idx0 = get_global_id(0); /* 16 */
  int idx1 = get_global_id(1); /* 512 */
  *(acc0+0) = 0.0f;
  *(acc0+1) = 0.0f;
  *(acc0+2) = 0.0f;
  *(acc0+3) = 0.0f;
  *(acc0+4) = 0.0f;
  *(acc0+5) = 0.0f;
  *(acc0+6) = 0.0f;
  *(acc0+7) = 0.0f;
  *(acc0+8) = 0.0f;
  *(acc0+9) = 0.0f;
  *(acc0+10) = 0.0f;
  *(acc0+11) = 0.0f;
  *(acc0+12) = 0.0f;
  *(acc0+13) = 0.0f;
  *(acc0+14) = 0.0f;
  *(acc0+15) = 0.0f;
  int alu16 = (idx0<<2);
  half4 val0 = (*((__global half4*)((data4_64+alu16))));
  half4 val1 = (*((__global half4*)((data5_64+alu16))));
  int alu17 = (idx0+(idx1<<6));
  int alu18 = (idx1>>4);
  int2 cast0 = (int2)((alu17&1023),alu18);
  float4 val2 = read_imagef(data1_131072, smp, cast0);
  int2 cast1 = (int2)(((alu17+16)&1023),alu18);
  float4 val3 = read_imagef(data1_131072, smp, cast1);
  int2 cast2 = (int2)(((alu17+32)&1023),alu18);
  float4 val4 = read_imagef(data1_131072, smp, cast2);
  int2 cast3 = (int2)(((alu17+48)&1023),alu18);
  float4 val5 = read_imagef(data1_131072, smp, cast3);
  for (int Ridx0 = 0; Ridx0 < 48; Ridx0++) {
    int alu19 = ((idx1*192)+Ridx0);
    int alu20 = (alu19+48);
    int alu21 = (alu19+96);
    int alu22 = (alu19+144);
    int alu23 = (Ridx0<<2);
    float4 val6 = read_imagef(data3_12288, smp, (int2)(alu23,idx0));
    float4 val7 = read_imagef(data3_12288, smp, (int2)((alu23+1),idx0));
    float4 val8 = read_imagef(data3_12288, smp, (int2)((alu23+2),idx0));
    float4 val9 = read_imagef(data3_12288, smp, (int2)((alu23+3),idx0));
    float4 val10 = read_imagef(data2_393216, smp, (int2)((alu20-(3072*(((alu20>>10)*43)>>7))),alu18));
    *(acc0+1) = ((*(acc0+1))+(val10.x*val6.x)+(val10.y*val7.x)+(val10.z*val8.x)+(val10.w*val9.x));
    *(acc0+5) = ((*(acc0+5))+(val10.x*val6.y)+(val10.y*val7.y)+(val10.z*val8.y)+(val10.w*val9.y));
    *(acc0+9) = ((*(acc0+9))+(val10.x*val6.z)+(val10.y*val7.z)+(val10.z*val8.z)+(val10.w*val9.z));
    *(acc0+13) = ((*(acc0+13))+(val10.x*val6.w)+(val10.y*val7.w)+(val10.z*val8.w)+(val10.w*val9.w));
    float4 val11 = read_imagef(data2_393216, smp, (int2)((alu21-(3072*(((alu21>>10)*43)>>7))),alu18));
    *(acc0+2) = ((*(acc0+2))+(val11.x*val6.x)+(val11.y*val7.x)+(val11.z*val8.x)+(val11.w*val9.x));
    *(acc0+6) = ((*(acc0+6))+(val11.x*val6.y)+(val11.y*val7.y)+(val11.z*val8.y)+(val11.w*val9.y));
    *(acc0+10) = ((*(acc0+10))+(val11.x*val6.z)+(val11.y*val7.z)+(val11.z*val8.z)+(val11.w*val9.z));
    *(acc0+14) = ((*(acc0+14))+(val11.x*val6.w)+(val11.y*val7.w)+(val11.z*val8.w)+(val11.w*val9.w));
    float4 val12 = read_imagef(data2_393216, smp, (int2)((alu22-(3072*(((alu22>>10)*43)>>7))),alu18));
    *(acc0+3) = ((*(acc0+3))+(val12.x*val6.x)+(val12.y*val7.x)+(val12.z*val8.x)+(val12.w*val9.x));
    *(acc0+7) = ((*(acc0+7))+(val12.x*val6.y)+(val12.y*val7.y)+(val12.z*val8.y)+(val12.w*val9.y));
    *(acc0+11) = ((*(acc0+11))+(val12.x*val6.z)+(val12.y*val7.z)+(val12.z*val8.z)+(val12.w*val9.z));
    *(acc0+15) = ((*(acc0+15))+(val12.x*val6.w)+(val12.y*val7.w)+(val12.z*val8.w)+(val12.w*val9.w));
    float4 val13 = read_imagef(data2_393216, smp, (int2)((alu19-(3072*(((alu19>>10)*43)>>7))),alu18));
    *(acc0+0) = ((*(acc0+0))+(val13.x*val6.x)+(val13.y*val7.x)+(val13.z*val8.x)+(val13.w*val9.x));
    *(acc0+4) = ((*(acc0+4))+(val13.x*val6.y)+(val13.y*val7.y)+(val13.z*val8.y)+(val13.w*val9.y));
    *(acc0+8) = ((*(acc0+8))+(val13.x*val6.z)+(val13.y*val7.z)+(val13.z*val8.z)+(val13.w*val9.z));
    *(acc0+12) = ((*(acc0+12))+(val13.x*val6.w)+(val13.y*val7.w)+(val13.z*val8.w)+(val13.w*val9.w));
  }
  float cast4 = ((float)(val0.x));
  float cast5 = ((float)(val1.x));
  float cast6 = ((float)(val0.y));
  float cast7 = ((float)(val1.y));
  float cast8 = ((float)(val0.z));
  float cast9 = ((float)(val1.z));
  float cast10 = ((float)(val0.w));
  float cast11 = ((float)(val1.w));
  write_imagef(data0_131072, cast0, (float4)((val2.x+(((*(acc0+0))+cast4)*cast5)),(val2.y+(((*(acc0+4))+cast6)*cast7)),(val2.z+(((*(acc0+8))+cast8)*cast9)),(val2.w+(((*(acc0+12))+cast10)*cast11))));
  write_imagef(data0_131072, cast1, (float4)((val3.x+(((*(acc0+1))+cast4)*cast5)),(val3.y+(((*(acc0+5))+cast6)*cast7)),(val3.z+(((*(acc0+9))+cast8)*cast9)),(val3.w+(((*(acc0+13))+cast10)*cast11))));
  write_imagef(data0_131072, cast2, (float4)((val4.x+(((*(acc0+2))+cast4)*cast5)),(val4.y+(((*(acc0+6))+cast6)*cast7)),(val4.z+(((*(acc0+10))+cast8)*cast9)),(val4.w+(((*(acc0+14))+cast10)*cast11))));
  write_imagef(data0_131072, cast3, (float4)((val5.x+(((*(acc0+3))+cast4)*cast5)),(val5.y+(((*(acc0+7))+cast6)*cast7)),(val5.z+(((*(acc0+11))+cast8)*cast9)),(val5.w+(((*(acc0+15))+cast10)*cast11))));
}
*** QCOM     672 r_512_16_4_4_48_4                              arg  6 mem   0.10 GB tm    527.97us/    78.94ms (     96 GFLOPS    3|98     GB/s) ['conv2d', 'mul', '__add__']
"""

c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((32, 1024, 4)), (), 0)
c2 = UOp.range(64, 3, AxisType.LOOP)
c4 = UOp.range(64, 4, AxisType.LOOP)
c7 = UOp.range(32, 2, AxisType.LOOP)
c10 = (((c2*64)+c4)+(c7*4096))
c12 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((32, 1024, 4)), (), 1)
c14 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((32, 3072, 4)), (), 2)
c16 = UOp.range(48, 0, AxisType.REDUCE)
c19 = UOp.range(4, 1, AxisType.REDUCE)
c28 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((16, 192, 4)), (), 3)
c40 = (c14.index(((((c16*4)+c19)+(c2*192))+(c7*12288)))*c28.index(((((c19*4)+(c4%4))+(c16*16))+((c4//4)*768))))
c42 = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(64), (), 4)
c46 = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(64), (), 5)
c50 = (c12.index(c10)+((c40.reduce(c16, c19, arg=Ops.ADD)+c42.index(c4).cast(dtypes.float))*c46.index(c4).cast(dtypes.float)))
c52 = c0.index(c10, ptr=True).store(c50).end(c7, c2, c4)
ast = c52.sink()

compiler = Device.default.compiler
renderer = Device.default.renderer
allocator = Device.default.allocator

uops = full_rewrite(ast, renderer)
src = renderer.render(uops)

# NOLOCALS=1 IMAGE=2 DEV=CL
lib = compiler.compile(src)
# r_64_8_16_4_4_48_4
# NOLOCALS: r_512_16_4_4_48_4
ps = ProgramSpec("r_512_16_4_4_48_4", src, Device.DEFAULT, ast, uops)
print(ps.src)
print(ps.applied_opts)
# (Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.NOLOCALS, axis=None, arg=None))
cr = CompiledRunner(ps, precompiled=lib)

gs = sorted(dedup([u for u in ast.toposort() if u.op is Ops.DEFINE_GLOBAL]), key=lambda u: u.arg)
print(len(gs))
print([g.dtype for g in gs])

bufs = [Buffer(ps.device, g.size, g.dtype if isinstance(g.dtype, ImageDType) else g.dtype._base).ensure_allocated() for g in gs]

t = cr(bufs, wait=True)
print(f"{t*1e6:.2f} us")