#!/bin/bash -e
clang sniff.cc -Werror -shared -fPIC -I../src/ROCT-Thunk-Interface/include -I../src/ROCm-Device-Libs/ockl/inc -o sniff.so
#AMD_LOG_LEVEL=4 HSAKMT_DEBUG_LEVEL=7 LD_PRELOAD=$PWD/sniff.so /home/tiny/build/HIP-Examples/HIP-Examples-Applications/HelloWorld/HelloWorld
AMD_LOG_LEVEL=4 LD_PRELOAD=$PWD/sniff.so /home/tiny/build/HIP-Examples/HIP-Examples-Applications/HelloWorld/HelloWorld
#AMD_LOG_LEVEL=4 HSAKMT_DEBUG_LEVEL=7 LD_PRELOAD=$PWD/sniff.so rocm-bandwidth-test -s 0 -d 1 -m 1
#AMD_LOG_LEVEL=4 HSAKMT_DEBUG_LEVEL=7 LD_PRELOAD=$PWD/sniff.so rocm-bandwidth-test -s 1 -d 2 -m 1
