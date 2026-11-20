# Mali panthor backend

`DEV=MALI` supports the RK3588 Mali-G610 using mainline panthor. Python owns the VM, buffer objects, Valhall descriptors, CSF command streams, and `DRM_IOCTL_PANTHOR_GROUP_SUBMIT` calls through tinygrad's standard HCQ2 scheduler. No Gallium, NIR, OpenCL, Rusticl, `mali_kbase`, or custom C runtime is used.

Shaders are rendered as GLSL compute, translated to GLCompute SPIR-V with `glslangValidator`, and compiled directly to MBS2 machine code with Arm's `cmpbe_v2` compiler. Arm's library is used only for compilation; it is not used for device access or execution.

## Requirements

- Mainline panthor at `/dev/dri/renderD128`
- `glslangValidator`
- Arm's compiler-only `libmali.so`, selected with `CMPBE_PATH`

```sh
export CMPBE_PATH="$HOME/mali-userspace/usr/lib/aarch64-linux-gnu/libmali.so.1.9.0"
```

## Validate

```sh
DEV=MALI python3 - <<'PY'
from tinygrad import Tensor
x = Tensor([1., 2., 3.], device="MALI")
assert (x * 2).tolist() == [2., 4., 6.]
print("native MALI passed")
PY

PYTHONPATH=. DEV=MALI IMAGE=1 FLOAT16=1 python3 examples/openpilot/compile_onnx.py \
  "https://huggingface.co/commaai/openpilot-lfs/resolve/51b12d20a25b76fa8a5717c2e5af01f148e469e2/openpilot/selfdrive/modeld/models/driving_supercombo.onnx" \
  /tmp/openpilot_mali.pkl --benchmark-runs 3
```
