# File-Specific Variables

File-specific environment variables control the behavior of a specific tinygrad file.

File-specific environment variables usually don't affect the library itself. Most of the time these will never be used, but they are here for completeness.

### accel/ane/2\_compile/hwx\_parse.py

| Variable | Possible Value(s) | Description             |
| -------- | ----------------- | ----------------------- |
| PRINTALL | \[1]              | print all ANE registers |

### extra/onnx.py

| Variable  | Possible Value(s) | Description           |
| --------- | ----------------- | --------------------- |
| ONNXLIMIT | \[#]              | set a limit for ONNX  |
| DEBUGONNX | \[1]              | enable ONNX debugging |

### extra/thneed.py

| Variable      | Possible Value(s) | Description                 |
| ------------- | ----------------- | --------------------------- |
| DEBUGCL       | \[1-4]            | enable Debugging for OpenCL |
| PRINT\_KERNEL | \[1]              | Print OpenCL Kernels        |

### extra/kernel\_search.py

| Variable       | Possible Value(s) | Description                               |
| -------------- | ----------------- | ----------------------------------------- |
| OP             | \[1-3]            | different operations                      |
| NOTEST         | \[1]              | enable not testing AST                    |
| DUMP           | \[1]              | enable dumping of intervention cache      |
| REDUCE         | \[1]              | enable reduce operations                  |
| SIMPLE\_REDUCE | \[1]              | enable simpler reduce operations          |
| BC             | \[1]              | enable big conv operations                |
| CONVW          | \[1]              | enable convw operations                   |
| FASTCONV       | \[1]              | enable faster conv operations             |
| GEMM           | \[1]              | enable general matrix multiply operations |
| BROKEN         | \[1]              | enable a kind of operation                |
| BROKEN3        | \[1]              | enable a kind of operation                |

### examples/vit.py

| Variable | Possible Value(s) | Description                   |
| -------- | ----------------- | ----------------------------- |
| LARGE    | \[1]              | enable larger dimension model |

### examples/llama.py

| Variable | Possible Value(s) | Description            |
| -------- | ----------------- | ---------------------- |
| WEIGHTS  | \[1]              | enable loading weights |

### examples/mlperf

| Variable | Possible Value(s)                             | Description        |
| -------- | --------------------------------------------- | ------------------ |
| MODEL    | \[resnet,retinanet,unet3d,rnnt,bert,maskrcnn] | what models to use |

### examples/benchmark\_train\_efficientnet.py

| Variable | Possible Value(s) | Description                               |
| -------- | ----------------- | ----------------------------------------- |
| CNT      | \[10]             | the amount of times to loop the benchmark |
| BACKWARD | \[1]              | enable backward pass                      |
| TRAINING | \[1]              | set Tensor.training                       |
| CLCACHE  | \[1]              | enable cache for OpenCL                   |

### examples/hlb\_cifar10.py

| Variable          | Possible Value(s) | Description                     |
| ----------------- | ----------------- | ------------------------------- |
| TORCHWEIGHTS      | \[1]              | use torch to initialize weights |
| DISABLE\_BACKWARD | \[1]              | don't do backward pass          |

### examples/benchmark\_train\_efficientnet.py & examples/hlb\_cifar10.py

| Variable | Possible Value(s) | Description            |
| -------- | ----------------- | ---------------------- |
| ADAM     | \[1]              | use the Adam optimizer |

### examples/hlb\_cifar10.py & xamples/hlb\_cifar10\_torch.py

| Variable | Possible Value(s) | Description               |
| -------- | ----------------- | ------------------------- |
| STEPS    | \[0-10]           | number of steps           |
| FAKEDATA | \[1]              | enable to use random data |

### examples/train\_efficientnet.py

| Variable | Possible Value(s) | Description                    |
| -------- | ----------------- | ------------------------------ |
| STEPS    | \[# % 1024]       | number of steps                |
| TINY     | \[1]              | use a tiny convolution network |
| IMAGENET | \[1]              | use imagenet for training      |

### examples/train\_efficientnet.py & examples/train\_resnet.py

| Variable | Possible Value(s) | Description                   |
| -------- | ----------------- | ----------------------------- |
| TRANSFER | \[1]              | enable to use pretrained data |

### examples & test/external/external\_test\_opt.py

| Variable | Possible Value(s) | Description                                  |
| -------- | ----------------- | -------------------------------------------- |
| NUM      | \[18, 2]          | what ResNet\[18] / EfficientNet\[2] to train |

### test/test\_ops.py

| Variable       | Possible Value(s) | Description                 |
| -------------- | ----------------- | --------------------------- |
| PRINT\_TENSORS | \[1]              | print tensors               |
| FORWARD\_ONLY  | \[1]              | use forward operations only |

### test/test\_speed\_v\_torch.py

| Variable  | Possible Value(s) | Description                   |
| --------- | ----------------- | ----------------------------- |
| TORCHCUDA | \[1]              | enable the torch cuda backend |

### test/external/external\_test\_gpu\_ast.py

| Variable | Possible Value(s) | Description                |
| -------- | ----------------- | -------------------------- |
| KOPT     | \[1]              | enable kernel optimization |
| KCACHE   | \[1]              | enable kernel cache        |

### test/external/external\_test\_opt.py

| Variable  | Possible Value(s) | Description              |
| --------- | ----------------- | ------------------------ |
| ENET\_NUM | \[-2,-1]          | what EfficientNet to use |

### test/test\_dtype.py & test/extra/test\_utils.py & extra/training.py

| Variable | Possible Value(s) | Description                |
| -------- | ----------------- | -------------------------- |
| CI       | \[1]              | disables some tests for CI |

### examples & extra & test

| Variable | Possible Value(s)     | Description       |
| -------- | --------------------- | ----------------- |
| BS       | \[8, 16, 32, 64, 128] | batch size to use |

### datasets/imagenet\_download.py

| Variable      | Possible Value(s) | Description                               |
| ------------- | ----------------- | ----------------------------------------- |
| IMGNET\_TRAIN | \[1]              | download also training data with ImageNet |
