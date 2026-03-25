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

## `DEV` Variable

`DEV` is probably the most commonly used environment variable, and obeys an elaborate syntax.
In tinygrad, `DEV` is used to specify the default backend, which includes the interface, device runtime,
renderer, arch string, and additional arch parameters. The core of the `DEV` syntax is a `:`-delimited
triple (eg. `AMD:AMDLLVM:gfx1100`). In order, these fields specify the device runtime, renderer, and arch.
Omitting any of these fields will cause tinygrad to infer a value, which allows trailing `:`'s to be dropped.
That is, writing `DEV=AMD` is equivalent to writing `DEV=AMD::` (NB: you cannot write `DEV=gfx1100` as a
shorthand for `DEV=::gfx1100` and similarly `DEV=LLVM` is not a valid shorthand for `DEV=:LLVM`). The arch
section of the triple may also contain additional comma-separated parameters (eg. `DEV=CPU:LLVM:x86_64,avx512`).
To specify the interface, prepend `<iface>+` to rest of the target string (eg. `DEV=USB+AMD`). Finally, additional
target strings may be added, semicolon-separated, to `DEV` to specify desired settings for the non-default devices.

### Example Values

Value | Meaning
---|---
AMD                               | use the AMD device
AMD:AMDLLVM                       | use the AMD device with the AMDLLVM renderer
CPU:LLVM:x86_64,avx512            | use the CPU device with the LLVM renderer targeting x86_64, with additional parameter avx512
NV:CUDA:sm_70                     | use the NV device with the CUDA renderer targeting sm_70
NULL:QCOMCL:a630                  | use the NULL device with the QCOMCL renderer targeting a630
MOCK+AMD::gfx950                  | use the AMD device over the MOCK interface targeting gfx950
PYTHON::sm_89                     | use the PYTHON device targeting sm_89 (ie. emulate sm_89 tensor cores in python)
USB+AMD:AMDLLVM:gfx1201           | use the AMD device over the USB interface with the AMDLLVM renderer targeting gfx1201
REMOTE:localhost:1337+AMD:AMDLLVM | use the AMD device over the REMOTE interface (at localhost:1337) with the AMDLLVM renderer
PCI:0-2,4+AMD                     | use the AMD device over the PCI interface (physical devices 0-2,4)
AMD:AMDLLVM;QCOM:IR3              | default to using the AMD device with the AMDLLVM renderer, render any QCOM kernels with the IR3 renderer

## Global Variables
The columns of this list are are: Variable, Possible Value(s) and Description.

- A `#` means that the variable can take any integer value.

These control the behavior of core tinygrad even when used as a library.

Variable | Possible Value(s) | Description
---|---|---
DEBUG               | [1-7]      | enable debugging output (operations, timings, speed, generated code and more)
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
