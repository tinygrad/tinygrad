# Python Panfrost backend

`DEV=MALI` uses tinygrad's HCQ2 queue and `panfrost.py` bindings over `/dev/dri/renderD128` (panthor). The runtime builds direct Valhall CSF command streams in Python, including compute state, descriptors, image surfaces, and runtime pointer updates, then submits them with `DRM_IOCTL_PANTHOR_GROUP_SUBMIT`. Mesa provides compilation and resource helpers, not a Gallium launch queue. There is no custom C backend, OpenCL, Rusticl, or `mali_kbase`.

Arm's compiler can optionally be used for supported non-shared-memory kernels; shared-memory, image-using, and floating-comparison kernels use Mesa. Mali uses explicit FP16/FP32 image shapes from `image_conv2d`; generic automatic image conversion is disabled until storage-image descriptors are supported:

```sh
export CMPBE_PATH="$HOME/mali-userspace/usr/lib/aarch64-linux-gnu/libmali.so.1.9.0"
export MALI_VENDOR=1
```

`MALI_VENDOR=1` is optional and affects compilation only; execution always uses panthor.

## Build

```sh
extra/mesa/panfrost/build.sh "$HOME/mesa-tinymesa"
export MESA_PATH="$HOME/mesa-tinymesa/build/src/tinymesa/libtinymesa.so"
```

The build clones Mesa 25.2.7, applies `mesa.patch`, and builds a stock `libtinymesa.so` with NIR and Panfrost enabled. Build dependencies include GCC/G++, Meson, Ninja, libdrm, expat, zlib/zstd, LLVM/Clang, libclc, SPIR-V tools, Wayland, and XCB development packages.

## Validate

```sh
DEV=MALI python3 - <<'PY'
from tinygrad import Tensor
x = Tensor([1., 2., 3.], device="MALI")
assert (x * 2).tolist() == [2., 4., 6.]
print("native MALI passed")
PY
```

OpenPilot uses the native image path:

```sh
PYTHONPATH=. DEV=MALI IMAGE=1 FLOAT16=1 python3 examples/openpilot/compile_onnx.py \
  "https://huggingface.co/commaai/openpilot-lfs/resolve/51b12d20a25b76fa8a5717c2e5af01f148e469e2/openpilot/selfdrive/modeld/models/driving_supercombo.onnx" \
  /tmp/openpilot_mali.pkl --benchmark-runs 3
```
