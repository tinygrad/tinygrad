# List of environment variables that control tinygrad behavior.

This is a list of environment variable that control the runtime behavior of tinygrad and its examples.
Most of these are self-explanatory, and are usually used to set an option at runtime.

Example: `GPU=1 DEBUG=4 python3 -m pytest`

The columns are: Variable, Possible Value(s) and Description.

- A `#` means that the variable can take any integer value.

## Global Variables

These control the behavior of core tinygrad even when used as a library.

Variable | Possible Value(s) | Description
---|---|---
DEBUG               | [1-4]      | enable debugging output, with 4 you get operations, timings, speed, generated code and more
GPU                 | [1]        | enable the GPU backend
CPU                 | [1]        | enable CPU backend
MPS                 | [1]        | enable MPS device (for Mac M1 and after)
METAL               | [1]        | enable Metal backend (for Mac M1 and after)
METAL_XCODE         | [1]        | enable Metal using macOS Xcode SDK
TORCH               | [1]        | enable PyTorch backend
CLANG               | [1]        | enable Clang backend
LLVM                | [1]        | enable LLVM backend
LLVMOPT             | [1]        | enable slightly more expensive LLVM optimizations
LAZY                | [1]        | enable lazy operations (this is the default)
OPT                 | [1-4]      | optimization level
OPTLOCAL            | [1-2]      | enable local optimization
GRAPH               | [1]        | create a graph of all operations (requires graphviz)
GRAPHPATH           | [/path/to] | where to put the generated graph
PRUNEGRAPH          | [1]        | prune MovementOps and LoadOps from the graph
PRINT_PRG           | [1]        | print program code
IMAGE               | [1]        | enable 2d specific optimizations
FLOAT16             | [1]        | use float16 for images instead of float32
ENABLE_METHOD_CACHE | [1]        | enable method cache (this is the default)
EARLY_STOPPING      | [# > 0]  | stop after this many kernels
DISALLOW_ASSIGN     | [1]        | disallow assignment of tensors
NATIVE_EXPLOG       | [1]        | enable using native exp and log

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

### extra/kernel_search.py

Variable | Possible Value(s) | Description
---|---|---
OP            | [1-3] | different operations
NOTEST        | [1]   | enable not testing AST
DUMP          | [1]   | enable dumping of intervention cache
REDUCE        | [1]   | enable reduce operations
SIMPLE_REDUCE | [1]   | enable simpler reduce operations
BC            | [1]   | enable big conv operations
CONVW         | [1]   | enable convw operations
FASTCONV      | [1]   | enable faster conv operations
GEMM          | [1]   | enable general matrix multiply operations
BROKEN        | [1]   | enable a kind of operation
BROKEN3       | [1]   | enable a kind of operation

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

### examples/benchmark_train_efficientnet.py & examples/hlb_cifar10.py

Variable | Possible Value(s) | Description
---|---|---
ADAM | [1] | use the Adam optimizer

### examples/hlb_cifar10.py & xamples/hlb_cifar10_torch.py

Variable | Possible Value(s) | Description
---|---|---
STEPS    | [0-10] | number of steps
FAKEDATA | [1]    | enable to use random data

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
KOPT   | [1] | enable kernel optimization
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
