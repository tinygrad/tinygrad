#!/bin/bash -e
g++ -std=c++17 -framework OpenCL -I/opt/homebrew/include -I/Users/kafka/openpilot/third_party/json11 -I/opt/homebrew/opt/opencl-headers/include -I$HOME/openpilot -I thneed \
  thneed/serialize.cc thneed/thneed.cc \
  $HOME/openpilot/third_party/json11/json11.cpp \
  $HOME/openpilot/common/clutil.cc \
  $HOME/openpilot/common/util.cc \
  run_thneed.cc -o run_thneed
