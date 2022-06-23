#!/bin/bash -e
if [[ $OSTYPE == 'darwin'* ]]; then
OP=$HOME/openpilot
LINK_CL="-framework opencl"
else
OP=/data/openpilot
LINK_CL="-lOpenCL"
fi

g++ -std=c++17 -I/opt/homebrew/include -I$OP/third_party/json11 -I/opt/homebrew/opt/opencl-headers/include -I$OP -I thneed \
  serialize.cc thneed.cc \
  $OP/third_party/json11/json11.cpp \
  $OP/common/clutil.cc \
  $OP/common/util.cc \
  run_thneed.cc -o run_thneed $LINK_CL
