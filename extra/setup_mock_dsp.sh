#!/bin/bash -e

docker build ./extra/dsp -t qemu-hexagon
brew install llvm@19 lld
DSP=1 python test/test_tiny.py TestTiny.test_plus
