### Welcome to the tinygrad documentation

General instructions you will find in [README.md](https://github.com/geohot/tinygrad/blob/master/README.md)

[abstraction.py](https://github.com/geohot/tinygrad/blob/master/docs/abstractions.py) is a well documented showcase of the abstraction stack.

There are plenty of [tests](https://github.com/geohot/tinygrad/tree/master/test) you can read through
[Examples](https://github.com/geohot/tinygrad/tree/master/examples) contains tinygrad implementations of popular models (vision and language) and neural networks. LLama, Stable diffusion, GANs and Yolo to name a few

### Environment variables
Here is a list of environment variables you can use with tinygrad.
Most of these are self-explanatory, and used to enable an option at runtime.
Example : `GPU=1 DEBUG=4 python3 -m pytest`

The columns are: Variable, Value and Description
They are also grouped into either general tinygrad or specific files

##### General tinygrad
DEBUG: [1-4], enable debugging output, with 4 you get operations, timings, speed, generated code and more
GPU: [1], enable the GPU backend
CPU: [1], enable CPU backend
MPS: [1], emable MPS device (for Mac M1 and after)
METAL: [1], enable Metal backend (for Mac M1 and after)
METAL_XCODE: [1], enable Metal using MacOS Xcode sdk
TORCH: [1], enable Torch backend
CLANG: [1], enable Clang backend
LLVM: [1], enable LLVM backend
LLVMOPT: [1], enable LLVM optimization
LAZY: [1], enable lazy operations
OPT: [1-4], enable optimization
OPTLOCAL: [1], enable local optimization
JIT: [1], enable Jit
GRAPH: [1], Create a graph of all operations
GRAPHPATH: [/path/to], what path to generate the graph image
PRUNEGRAPH, [1], prune movementops and loadops from the graph
PRINT_PRG: [1], print program
FLOAT16: [1], use float16 instead of float32
ENABLE_METHOD_CACHE: [1], enable method cache
EARLY_STOPPING: [1], stop early
DISALLOW_ASSIGN: [1], enable not assigning the realized lazydata to the lazy output buffer

##### tinygrad/codegen/cstyle.py
NATIVE_EXPLOG: [1], enable using native explog

##### accel/ane/2_compile/hwx_parse.py
PRINTALL: [1], print all ane registers

##### extra/onnx.py
ONNXLIMIT: [ ], set a limit for Onnx
DEBUGONNX: [1], enable Onnx debugging

##### extra/thneed.py
DEBUGCL: [1-4], enable Debugging for OpenCL
PRINT_KERNEL: [1], Print OpenCL Kernels

##### extra/kernel_search.py
OP: [1-3], different operations
NOTEST: [1], enable not testing ast
DUMP: [1], enable dumping of intervention cache
REDUCE: [1], enable reduce operations
SIMPLE_REDUCE: [1], enable simpler reduce operations
BC: [1], enable big conv operations
CONVW: [1], enable convw operations
FASTCONV: [1], enable faster conv operations
GEMM: [1], enable general matrix multiply operations
BROKEN: [1], enable a kind of operation
BROKEN3: [1], enable a kind of operation

##### examples/vit.py
LARGE: [1], enable larger dimension model

##### examples/llama.py
WEIGHTS: [1], enable using weights

##### examples/mlperf
MODEL: [resnet,retinanet,unet3d,rnnt,bert,maskrcnn], what models to use

##### examples/benchmark_train_efficientnet.py
CNT: [10], the amount of times to loop the benchmark
BACKWARD: [1], enable backward call
TRAINING: [1], set Tensor.training
CLCACHE: [1], enable Cache for OpenCL

##### examples/hlb_cifar10.py
TORCHWEIGHTS: [1], use torch to initialize weights
DISABLE_BACKWARD: [1], dont use backward operations

##### examples/benchmark_train_efficientnet.py & examples/hlb_cifar10.py
ADAM: [1], enable Adam optimization

##### examples/hlb_cifar10.py & xamples/hlb_cifar10_torch.py
STEPS: [0-10], number of steps
FAKEDATA: [1], enable to use random data

##### examples/train_efficientnet.py
STEPS: [1024 dividable], number of steps
TINY: [1], use a tiny convolution network
IMAGENET: [1], use imagenet for training

##### examples/train_efficientnet.py & examples/train_resnet.py
TRANSFER: [1], enable to use pretrained data

##### examples & test/external/external_test_opt.py
NUM: [18, 2], what ResNet[18] / EfficientNet[2] to train

##### test/test_ops.py
PRINT_TENSORS: [1], print tensors
FORWARD_ONLY: [1], use forward operations only

##### test/test_speed_v_torch.py
TORCHCUDA: [1], enable the torch cuda backend

##### test/external/external_test_gpu_ast.py
KOPT: [1], enable kernel optimization
KCACHE: [1], enable kernel cache

##### test/external/external_test_opt.py
ENET_NUM: [-2,-1], what EfficientNet to use

##### test/test_dtype.py & test/extra/test_utils.py & extra/training.py
CI: [1], enable to avoid some tests to run in CI

##### examples & extra & test
BS: [8, 16, 32, 64, 128], bytesize

