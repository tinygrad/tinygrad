#!/bin/bash -e

CWD=$(pwd)
LOCAL_DIR=$(dirname $(readlink -f $0))
cd $LOCAL_DIR

./patch.py AS_PCIE_201012_91_00_00.bin patched.raw_fw
./make_image.py patched.raw_fw -c ASM2362 -t fw -o patched.fw
./make_image.py patched.fw -c ASM2362 -t flash -o patched.flash

cd $CWD