#!/bin/bash
SRC=/home/kafka/build/open-gpu-kernel-modules

clang2py \
  $SRC/src/nvidia/generated/g_allclasses.h \
  -o class_ioctl.py -k cdefstum

exit

#clang2py $SRC/src/nvidia/arch/nvalloc/unix/include/nv_escape.h \
#  $SRC/src/nvidia/arch/nvalloc/unix/include/nv-ioctl-numbers.h \
#  $SRC/src/common/sdk/nvidia/inc/nvos.h \
#  --clang-args="-I $SRC/src/common/sdk/nvidia/inc -I $SRC/src/common/sdk/nvidia/inc/ctrl" \
#  -o esc_ioctl.py -k cdefstum

clang2py \
  $SRC/src/common/sdk/nvidia/inc/ctrl/ctrl0000/*.h \
  $SRC/src/common/sdk/nvidia/inc/ctrl/ctrl0080/*.h \
  $SRC/src/common/sdk/nvidia/inc/ctrl/ctrl2080/*.h \
  $SRC/src/common/sdk/nvidia/inc/ctrl/ctrl83de/*.h \
  $SRC/src/common/sdk/nvidia/inc/ctrl/ctrlc36f.h \
  $SRC/src/common/sdk/nvidia/inc/ctrl/ctrlcb33.h \
  $SRC/src/common/sdk/nvidia/inc/ctrl/ctrla06c.h \
  --clang-args="-I $SRC/src/common/sdk/nvidia/inc -I $SRC/src/common/sdk/nvidia/inc/ctrl" \
  -o ctrl_ioctl.py -k cdefstum
sed -i "s\(0000000001)\1\g" ctrl_ioctl.py