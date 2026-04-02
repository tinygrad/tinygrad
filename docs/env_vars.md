# List of environment variables that control tinygrad behavior.

This is a list of environment variable that control the runtime behavior of tinygrad and its examples.
Most of these are self-explanatory, and are usually used to set an option at runtime.

Example: `DEV=CL DEBUG=4 python3 -m pytest`

However you can also decorate a function to set a value only inside that function.

```python
# in tensor.py (probably only useful if you are a tinygrad developer)
@Context(DEBUG=4)
def numpy(self) -> ...
```

Or use contextmanager to temporarily set a value inside some scope:

```python
with Context(DEBUG=0):
  a = Tensor.ones(10, 10)
  a *= 2
```

## Global Variables
The columns of this list are are: Variable, Possible Value(s) and Description.

- A `#` means that the variable can take any integer value.

These control the behavior of core tinygrad even when used as a library.

### DEV variable

The `DEV` variable deserves special note due to its more nuanced syntax.
`DEV` is used to specify the target device and target renderer for said device, separated by colons.
Specifying the renderer is optional, omitting a preference will cause tinygrad to automatically select a renderer from those
available on the system. Some example values for `DEV` are: `AMD`, `AMD:LLVM`, `NV:PTX`, etc.

Variable | Possible Value(s) | Description
---|---|---
DEBUG               | [1-7]      | enable debugging output (operations, timings, speed, generated code and more)
DEV                 | [AMD, NV, ...] | enable a specific backend
BEAM                | [#]        | number of beams in kernel beam search
DEFAULT_FLOAT       | [HALF, ...]| specify the default float dtype (FLOAT32, HALF, BFLOAT16, FLOAT64, ...), default to FLOAT32
IMAGE               | [1-2]      | enable 2d specific optimizations
FLOAT16             | [1]        | use float16 for images instead of float32
HCQ_VISIBLE_DEVICES | [list[int]]| restricts the HCQ devices that are available. The format is a comma-separated list of identifiers (indexing starts with 0).
JIT                 | [0-2]      | 0=disabled, 1=[jit enabled](quickstart.md#jit) (default), 2=jit enabled, but graphs are disabled
VIZ                 | [1]        | 0=disabled, 1=[viz enabled](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/viz)
ALLOW_TF32          | [1]        | enable TensorFloat-32 tensor cores on Ampere or newer GPUs.
WEBGPU_BACKEND      | [WGPUBackendType_Metal, ...]          | Force select a backend for WebGPU (Metal, DirectX, OpenGL, Vulkan...)
CUDA_PATH           | str        | Use `CUDA_PATH/include` for CUDA headers for CUDA and NV backends. If not set, TinyGrad will use `/usr/local/cuda/include`, `/usr/include` and `/opt/cuda/include`.

## Debug breakdown

Variable | Value | Description
---|---|---
DEBUG               | >= 1       | Enables debugging and lists devices being used
DEBUG               | >= 2       | Provides performance metrics for operations, including timing, memory usage, bandwidth for each kernel execution
DEBUG               | >= 3       | Outputs buffers used for each kernel (shape, dtype and strides) and the applied optimizations at a kernel level
DEBUG               | >= 4       | Outputs the generated kernel code
DEBUG               | >= 5       | Displays the intermediate representation of the computation UOps (AST)
DEBUG               | >= 6       | Displays the intermediate representation of the computation UOps in a linearized manner, detailing the operation sequence
DEBUG               | >= 7       | Outputs the assembly code generated for the target hardware
