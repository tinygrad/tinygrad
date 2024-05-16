# List of environment variables that control tinygrad behavior.

This is a list of environment variable that control the runtime behavior of tinygrad and its examples.
Most of these are self-explanatory, and are usually used to set an option at runtime.

Example: `GPU=1 DEBUG=4 python3 -m pytest`

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

Variable | Possible Value(s) | Description
---|---|---
DEBUG               | [1-6]      | enable debugging output, with 4 you get operations, timings, speed, generated code and more
GPU                 | [1]        | enable the GPU backend
CUDA                | [1]        | enable CUDA backend
HSA                 | [1]        | enable HSA backend
METAL               | [1]        | enable Metal backend (for Mac M1 and after)
METAL_XCODE         | [1]        | enable Metal using macOS Xcode SDK
CLANG               | [1]        | enable Clang backend
LLVM                | [1]        | enable LLVM backend
BEAM                | [#]        | number of beams in kernel beam search
GRAPH               | [1]        | create a graph of all operations (requires graphviz)
GRAPHUOPS           | [1]        | create a graph of uops (requires graphviz and saves at /tmp/uops.{svg,dot})
GRAPHPATH           | [/path/to] | where to put the generated graph
DEFAULT_FLOAT       | [HALF, ...]| specify the default float dtype (FLOAT32, HALF, BFLOAT16, FLOAT64, ...), default to FLOAT32
IMAGE               | [1-2]      | enable 2d specific optimizations
FLOAT16             | [1]        | use float16 for images instead of float32
PTX                 | [1]        | enable the specialized [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/) assembler for Nvidia GPUs. If not set, defaults to generic CUDA codegen backend.

## File Specific Variables

These are variables that control the behavior of a specific file, these usually don't affect the library itself.
Most of the time these will never be used, but they are here for completeness.

### accel/ane/2_compile/hwx_parse.py

Variable | Possible Value(s) | Description
---|---|---
PRINTALL | [1] | print all ANE registers

### extra/onnx.py

Variable | Possible Value(s) | Description
---|---|---
ONNXLIMIT | [#] | set a limit for ONNX
DEBUGONNX | [1] | enable ONNX debugging

### extra/thneed.py

Variable | Possible Value(s) | Description
---|---|---
DEBUGCL      | [1-4] | enable Debugging for OpenCL
PRINT_KERNEL | [1]   | Print OpenCL Kernels

### examples/vit.py

Variable | Possible Value(s) | Description
---|---|---
LARGE | [1] | enable larger dimension model

### examples/llama.py

Variable | Possible Value(s) | Description
---|---|---
WEIGHTS | [1] | enable loading weights

### examples/mlperf

Variable | Possible Value(s) | Description
---|---|---
MODEL | [resnet,retinanet,unet3d,rnnt,bert,maskrcnn] | what models to use

### examples/benchmark_train_efficientnet.py

Variable | Possible Value(s) | Description
---|---|---
CNT      | [10] | the amount of times to loop the benchmark
BACKWARD | [1]  | enable backward pass
TRAINING | [1]  | set Tensor.training
CLCACHE  | [1]  | enable cache for OpenCL

### examples/hlb_cifar10.py

Variable | Possible Value(s) | Description
---|---|---
TORCHWEIGHTS     | [1] | use torch to initialize weights
DISABLE_BACKWARD | [1] | don't do backward pass
DIST             | [1] | enable distributed training
STEPS            | [#] | number of steps

### examples/benchmark_train_efficientnet.py & examples/hlb_cifar10.py

Variable | Possible Value(s) | Description
---|---|---
ADAM | [1] | use the Adam optimizer

### examples/train_efficientnet.py

Variable | Possible Value(s) | Description
---|---|---
STEPS    | [# % 1024] | number of steps
TINY     | [1]        | use a tiny convolution network
IMAGENET | [1]        | use imagenet for training

### examples/train_efficientnet.py & examples/train_resnet.py

Variable | Possible Value(s) | Description
---|---|---
TRANSFER | [1] | enable to use pretrained data

### examples & test/external/external_test_opt.py

Variable | Possible Value(s) | Description
---|---|---
NUM | [18, 2] | what ResNet[18] / EfficientNet[2] to train

### test/test_ops.py

Variable | Possible Value(s) | Description
---|---|---
PRINT_TENSORS | [1] | print tensors
FORWARD_ONLY  | [1] | use forward operations only

### test/test_speed_v_torch.py

Variable | Possible Value(s) | Description
---|---|---
TORCHCUDA | [1] | enable the torch cuda backend

### test/external/external_test_gpu_ast.py

Variable | Possible Value(s) | Description
---|---|---
KCACHE | [1] | enable kernel cache

### test/external/external_test_opt.py

Variable | Possible Value(s) | Description
---|---|---
ENET_NUM | [-2,-1] | what EfficientNet to use

### test/test_dtype.py & test/extra/test_utils.py & extra/training.py

Variable | Possible Value(s) | Description
---|---|---
CI | [1] | disables some tests for CI

### examples & extra & test

Variable | Possible Value(s) | Description
---|---|---
BS | [8, 16, 32, 64, 128] | batch size to use

### extra/datasets/imagenet_download.py

Variable | Possible Value(s) | Description
---|---|---
IMGNET_TRAIN | [1] | download also training data with imagenet