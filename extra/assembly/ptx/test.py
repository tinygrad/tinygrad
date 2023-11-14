#!/usr/bin/env python3
import numpy as np
from tinygrad.runtime.ops_cuda import CUDAProgram, RawCUDABuffer

def test_basic_operation():
  test = RawCUDABuffer.fromCPU(np.zeros(10, np.float32))
  prg = CUDAProgram("test", """
  .version 7.8
  .target sm_86
  .address_size 64
  .visible .entry test(.param .u64 x) {
    .reg .b32       %r<2>;
    .reg .b64       %rd<3>;

    ld.param.u64    %rd1, [x];
    cvta.to.global.u64      %rd2, %rd1;
    mov.u32         %r1, 0x40000000; // 2.0 in float
    st.global.u32   [%rd2], %r1;
    ret;
  }""", binary=True)
  prg([1], [1], test)
  print(test.toCPU())

def test_arithmetic_operations():
  arr1 = RawCUDABuffer.fromCPU(np.array([1.0, 2.0, 3.0, 4.0], np.float32))
  arr2 = RawCUDABuffer.fromCPU(np.array([5.0, 6.0, 7.0, 8.0], np.float32))
  output = RawCUDABuffer.fromCPU(np.zeros(4, np.float32))

  prg = CUDAProgram("arithmetic_op", """
  .version 7.8
  .target sm_86
  .address_size 64
  .visible .entry arithmetic_op(.param .u64 x, .param .u64 y, .param .u64 z) {
    .reg .b64 %rd[];
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [y];
    ld.param.u64 %rd3, [z];

    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;

    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    add.f32 %f3, %f1, %f2;

    st.global.f32 [%rd3], %f3;
    ret;
  }""", binary=True)

  prg([1], [1], arr1, arr2, output)
  print("Arithmetic Operation Test Result:", output.toCPU())


def test_memory_operations():
  input_data = RawCUDABuffer.fromCPU(np.array([9.0, 10.0, 11.0, 12.0], np.float32))
  output_data = RawCUDABuffer.fromCPU(np.zeros(4, np.float32))
  
  prg = CUDAProgram("memory_op", """
  .version 7.8
  .target sm_86
  .address_size 64
  .visible .entry memory_op(.param .u64 in, .param .u64 out) {
    .reg .b64 %rd[];
    .reg .f32 %f1;

    ld.param.u64 %rd1, [in];
    ld.param.u64 %rd2, [out];

    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;

    ld.global.f32 %f1, [%rd1];
    st.global.f32 [%rd2], %f1;
    ret;
  }""", binary=True)

  prg([1], [1], input_data, output_data)
  print("Memory Operation Test Result:", output_data.toCPU())

def main():
    test_basic_operation()
    test_arithmetic_operations()
    test_memory_operations()

if __name__ == "__main__":
    main()