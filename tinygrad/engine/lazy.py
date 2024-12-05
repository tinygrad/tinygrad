from tinygrad.ops import UOp

# NOTE: this is imported by scheduler test
view_supported_devices = {"LLVM", "CLANG", "CUDA", "NV", "AMD", "METAL", "QCOM", "DSP", "DISK"}

# LazyBuffer is UOp! This is fundamental
LazyBuffer = UOp
