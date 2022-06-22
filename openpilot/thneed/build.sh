#!/bin/bash -e
OP=/data/openpilot
g++ -std=c++17 -I/opt/homebrew/include -I$OP/third_party/json11 -I/opt/homebrew/opt/opencl-headers/include -I$OP -I thneed \
  serialize.cc thneed.cc \
  $OP/third_party/json11/json11.cpp \
  $OP/common/clutil.cc \
  $OP/common/util.cc \
  run_thneed.cc -o run_thneed -lOpenCL
