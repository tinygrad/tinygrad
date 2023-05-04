#!/bin/bash
mkdir -p build/debs
cd build

# ROCT-Thunk-Interface (hsakmt)
if [ ! -f debs/hsakmt-roct-dev_5.5.0.99999-local_amd64.deb ]
then
  mkdir -p ROCT-Thunk-Interface
  cd ROCT-Thunk-Interface
  cmake ../../src/ROCT-Thunk-Interface
  make -j32 package
  cp hsakmt-roct-dev_5.5.0.99999-local_amd64.deb ../debs
  cd ../
fi

# ROCm-Device-Libs
if [ ! -f debs/rocm-device-libs_1.0.0.99999-local_amd64.deb ]
then
  mkdir -p ROCm-Device-Libs
  cd ROCm-Device-Libs
  cmake ../../src/ROCm-Device-Libs
  make -j32 package
  cp rocm-device-libs_1.0.0.99999-local_amd64.deb ../debs
fi

# build custom LLVM
mkdir -p llvm-project
cd llvm-project
cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="llvm;clang;lld" -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" ../../src/llvm-project/llvm
make -j32

# ROCm-CompilerSupport
#mkdir -p ROCm-CompilerSupport
#cd ROCm-CompilerSupport
#cmake ../../src/ROCm-CompilerSupport/lib/comgr
#make -j32 package

# ROCR-Runtime
#mkdir -p ROCR-Runtime
#cd ROCR-Runtime
#cmake ../../src/ROCR-Runtime/src
#make -j32 package
