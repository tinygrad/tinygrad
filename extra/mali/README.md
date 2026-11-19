# tinygrad native Mali backend

`DEV=MALI` uses tinygrad NIR and Mesa's Panfrost compiler/Gallium driver directly over `/dev/dri/renderD128` (panthor). It does not use OpenCL, Rusticl, or `mali_kbase`.

## Build

```sh
extra/mali/build.sh "$HOME/mesa-tinymali"
export MESA_PATH="$HOME/mesa-tinymali/build/src/tinymali/libtinymali.so"
export TINYMALI_PATH="$HOME/mesa-tinymali/build/src/tinymali/libtinymali.so"
```

The build script clones Mesa 25.2.7, applies `mesa.patch`, and builds `libtinymali.so`. Build dependencies include GCC/G++, Meson, Ninja, libdrm, expat, zlib/zstd, LLVM/Clang, libclc, SPIR-V tools, Wayland, and XCB development packages.

## Validate

```sh
cd ~/tinygrad
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

`ops_mali.py` is the Python runtime. `tinymali.c` contains the Gallium buffer/compiler/launch shim. `mesa.patch` exports NIR symbols, adds compiler compatibility fixes, and enables CSF compute waits/cache flushes instead of flushing every launch on the host.
