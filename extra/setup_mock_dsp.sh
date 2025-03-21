#!/bin/bash -e

cd ./extra/dsp
docker build . -t qemu-hexagon --platform=linux/amd64
brew install llvm@19 lld
cd ../../
DEBUG=2 DSP=1 python test/test_tiny.py
