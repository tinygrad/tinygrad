#!/bin/bash
SRC=/home/nimlgen/open-gpu-kernel-modules

clang2py \
  $SRC/src/nvidia/generated/g_allclasses.h \
  --clang-args="-I$SRC/src/common/sdk/nvidia/inc/" \
  -o class_ioctl.py -k cdefstum

# clang2py \
#   $SRC/src/common/sdk/nvidia/inc/class/clc6c0.h \
#   $SRC/kernel-open/nvidia-uvm/clc6b5.h \
#   $SRC/../include/clc6c0qmd.h \
#   --clang-args="-I$SRC/src/common/sdk/nvidia/inc/" \
#   -o nv_qcmds.py -k cdefstum

#   sed -i 's/# \(.*\) = MW ( \([0-9]*\) : \([0-9]*\) ) # macro/\1 = (\2, \3) # macro/' nv_qcmds.py

# clang2py \
#   $SRC/kernel-open/nvidia-uvm/uvm_ioctl.h \
#   $SRC/kernel-open/nvidia-uvm/uvm_linux_ioctl.h \
#   --clang-args="-I$SRC/src/common/inc -I$SRC/kernel-open/nvidia-uvm -I$SRC/kernel-open/common/inc" \
#   -o uvm_ioctl.py -k cdefstum

# exit

# clang2py $SRC/src/nvidia/arch/nvalloc/unix/include/nv_escape.h \
#  $SRC/src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h \
#  $SRC/src/nvidia/arch/nvalloc/unix/include/nv-ioctl-numbers.h \
#  $SRC/src/nvidia/arch/nvalloc/unix/include/nv-ioctl-numa.h \
#  $SRC/src/nvidia/arch/nvalloc/unix/include/nv-unix-nvos-params-wrappers.h \
#  $SRC/src/common/sdk/nvidia/inc/alloc/alloc_channel.h \
#  $SRC/src/common/sdk/nvidia/inc/nvos.h \
#  --clang-args="-I$SRC/src/common/sdk/nvidia/inc -I$SRC/src/nvidia/arch/nvalloc/unix/include -I$SRC/src/common/sdk/nvidia/inc/ctrl" \
#  -o esc_ioctl.py -k cdefstum

# clang2py \
#   $SRC/src/common/sdk/nvidia/inc/ctrl/ctrl0000/*.h \
#   $SRC/src/common/sdk/nvidia/inc/ctrl/ctrl0080/*.h \
#   $SRC/src/common/sdk/nvidia/inc/ctrl/ctrl2080/*.h \
#   $SRC/src/common/sdk/nvidia/inc/ctrl/ctrl83de/*.h \
#   $SRC/src/common/sdk/nvidia/inc/ctrl/ctrlc36f.h \
#   $SRC/src/common/sdk/nvidia/inc/ctrl/ctrlcb33.h \
#   $SRC/src/common/sdk/nvidia/inc/ctrl/ctrla06c.h \
#   --clang-args="-I$SRC/src/common/sdk/nvidia/inc -I$SRC/src/common/sdk/nvidia/inc/ctrl" \
#   -o ctrl_ioctl.py -k cdefstum
# sed -i "s\(0000000001)\1\g" ctrl_ioctl.py