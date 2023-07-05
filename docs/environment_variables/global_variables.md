# Global Variables

Global environment variables control core tinygrad behavior even when used as a library.

| Variable              | Possible Value(s) | Description                                                                                                                                                            |
| --------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DEBUG                 | \[1-4]            | enable debugging output, with 4 you get operations, timings, speed, generated code and more                                                                            |
| GPU                   | \[1]              | enable the GPU backend                                                                                                                                                 |
| CUDA                  | \[1]              | enable CUDA backend                                                                                                                                                    |
| CPU                   | \[1]              | enable CPU backend                                                                                                                                                     |
| MPS                   | \[1]              | enable MPS device (for Mac M1 and after)                                                                                                                               |
| METAL                 | \[1]              | enable Metal backend (for Mac M1 and after)                                                                                                                            |
| METAL\_XCODE          | \[1]              | enable Metal using macOS Xcode SDK                                                                                                                                     |
| TORCH                 | \[1]              | enable PyTorch backend                                                                                                                                                 |
| CLANG                 | \[1]              | enable Clang backend                                                                                                                                                   |
| LLVM                  | \[1]              | enable LLVM backend                                                                                                                                                    |
| LLVMOPT               | \[1]              | enable slightly more expensive LLVM optimizations                                                                                                                      |
| LAZY                  | \[1]              | enable lazy operations (this is the default)                                                                                                                           |
| OPT                   | \[1-4]            | optimization level                                                                                                                                                     |
| GRAPH                 | \[1]              | create a graph of all operations (requires graphviz)                                                                                                                   |
| GRAPHPATH             | \[/path/to]       | where to put the generated graph                                                                                                                                       |
| PRUNEGRAPH            | \[1]              | prune MovementOps and LoadOps from the graph                                                                                                                           |
| PRINT\_PRG            | \[1]              | print program code                                                                                                                                                     |
| IMAGE                 | \[1]              | enable 2d specific optimizations                                                                                                                                       |
| FLOAT16               | \[1]              | use float16 for images instead of float32                                                                                                                              |
| ENABLE\_METHOD\_CACHE | \[1]              | enable method cache (this is the default)                                                                                                                              |
| EARLY\_STOPPING       | \[# > 0]          | stop after this many kernels                                                                                                                                           |
| DISALLOW\_ASSIGN      | \[1]              | disallow assignment of tensors                                                                                                                                         |
| CL\_EXCLUDE           | \[name0,name1]    | comma-separated list of device names to exclude when using OpenCL GPU backend (like `CL_EXCLUDE=gfx1036`)                                                              |
| CL\_PLATFORM          | \[# >= 0]         | index of the OpenCL [platform](https://documen.tician.de/pyopencl/runtime\_platform.html#pyopencl.Platform) to run on. Defaults to 0.                                  |
| RDNA                  | \[1]              | enable the specialized [RDNA 3](https://en.wikipedia.org/wiki/RDNA\_3) assembler for AMD 7000-series GPUs. If not set, defaults to generic OpenCL codegen backend.     |
| PTX                   | \[1]              | enable the specialized [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/) assembler for Nvidia GPUs. If not set, defaults to generic CUDA codegen backend. |
