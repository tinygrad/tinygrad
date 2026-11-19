# Python Panfrost backend

`DEV=MALI` uses tinygrad's Python allocator/program classes and `panfrost.py` bindings to call Mesa's existing NIR and Gallium APIs over `/dev/dri/renderD128` (panthor). There is no custom C backend, OpenCL, Rusticl, or `mali_kbase`.

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
