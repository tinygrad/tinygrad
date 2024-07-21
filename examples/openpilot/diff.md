# Documenting differences in openpilot kernel

To replicate:

on a823759dc57ca56c9a0d7eb0d5bd9d46baff4a79 (master as of writing)
```
PYTHONPATH="." DEBUG=4 ALLOWED_KERNEL_COUNT=208 FLOAT16=1 DEBUGCL=1 GPU=1 IMAGE=2 python examples/openpilot/compile2.py
```

on ae5d1407ee844a97a52ad3756835d38e7e2b9e1b
```
pip install requests pyopencl tqdm
PYTHONPATH="." DEBUG=4 ALLOWED_KERNEL_COUNT=208 FLOAT16=1 DEBUGCL=1 GPU=1 IMAGE=2 python openpilot/compile2.py
```

or see [the gist from George](https://gist.github.com/geohot/8d7edc7ac2fd9a31ea563c134b66cddb)

## I. Naive

a. [tinygrad_ae5d1407ee844a97a52ad3756835d38e7e2b9e1b.c -- from George](https://gist.github.com/geohot/8d7edc7ac2fd9a31ea563c134b66cddb)

b. [tinygrad_a823759dc57ca56c9a0d7eb0d5bd9d46baff4a79.c](https://gist.github.com/spikedoanz/b21ac4fe83d5c2f5d0d96cace2d4fb38)

c. [tinygrad_ae.. same hardware, see ## specs](https://gist.github.com/spikedoanz/834f214e9c840f4a97a8eb47774ef747)

### 1. line count (wc -l filename)
a. 20639 

b. 11618

### 2. unique kernel count (cat filename | grep __kernel | wc -l )
a. 291

b. 301

### 3. schedule items
a. 213 schedule items depend on the input, 598 don't

b. 214 schedule items depend on the input, 722 don't 

> master -1 input dependent, +124 not dependent

### 4. flops and memory
a.
```
total runtime: 14.63 ms   wall time: 21.53 ms
network using 1.49 GOPS with runtime 14.63 ms that's 101.66 GFLOPS
avg:   100.94 GFLOPS    14.04 GB/s           total:   207 kernels     1.49 GOPS     0.21 GB    14.73 ms
```

b. same hardware
```
avg:   234.69 GFLOPS    15.48 GB/s           total:   208 kernels     1.50 GOPS     0.10 GB     6.40 ms
```

c. same hardware
```
total runtime: 4.22 ms   wall time: 4.78 ms
avg:   316.95 GFLOPS    44.07 GB/s           total:   207 kernels     1.49 GOPS     0.21 GB     4.69 ms
```

> master is -82 GFLOPS, -15 GB/s

---

## II. Kernel analysis



---

## III. Specs

```
$ clinfo
Number of platforms                               1
  Platform Name                                   NVIDIA CUDA
  Platform Vendor                                 NVIDIA Corporation
  Platform Version                                OpenCL 3.0 CUDA 12.4.125
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts cl_nv_create_buffer cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_device_uuid cl_khr_pci_bus_info cl_khr_external_semaphore cl_khr_external_memory cl_khr_external_semaphore_opaque_fd cl_khr_external_memory_opaque_fd
  Platform Extensions with Version                cl_khr_global_int32_base_atomics                                 0x400000 (1.0.0)
                                                  cl_khr_global_int32_extended_atomics                             0x400000 (1.0.0)
                                                  cl_khr_local_int32_base_atomics                                  0x400000 (1.0.0)
                                                  cl_khr_local_int32_extended_atomics                              0x400000 (1.0.0)
                                                  cl_khr_fp64                                                      0x400000 (1.0.0)
                                                  cl_khr_3d_image_writes                                           0x400000 (1.0.0)
                                                  cl_khr_byte_addressable_store                                    0x400000 (1.0.0)
                                                  cl_khr_icd                                                       0x400000 (1.0.0)
                                                  cl_khr_gl_sharing                                                0x400000 (1.0.0)
                                                  cl_nv_compiler_options                                           0x400000 (1.0.0)
                                                  cl_nv_device_attribute_query                                     0x400000 (1.0.0)
                                                  cl_nv_pragma_unroll                                              0x400000 (1.0.0)
                                                  cl_nv_copy_opts                                                  0x400000 (1.0.0)
                                                  cl_nv_create_buffer                                              0x400000 (1.0.0)
                                                  cl_khr_int64_base_atomics                                        0x400000 (1.0.0)
                                                  cl_khr_int64_extended_atomics                                    0x400000 (1.0.0)
                                                  cl_khr_device_uuid                                               0x400000 (1.0.0)
                                                  cl_khr_pci_bus_info                                              0x400000 (1.0.0)
                                                  cl_khr_external_semaphore                                          0x9000 (0.9.0)
                                                  cl_khr_external_memory                                             0x9000 (0.9.0)
                                                  cl_khr_external_semaphore_opaque_fd                                0x9000 (0.9.0)
                                                  cl_khr_external_memory_opaque_fd                                   0x9000 (0.9.0)
  Platform Numeric Version                        0xc00000 (3.0.0)
  Platform Extensions function suffix             NV
  Platform Host timer resolution                  0ns

  Platform Name                                   NVIDIA CUDA
Number of devices                                 1
  Device Name                                     NVIDIA GeForce RTX 3090
  Device Vendor                                   NVIDIA Corporation
  Device Vendor ID                                0x10de
  Device Version                                  OpenCL 3.0 CUDA
  Device UUID                                     0158115e-ba1f-5312-697c-738cb9356e72
  Driver UUID                                     0158115e-ba1f-5312-697c-738cb9356e72
  Valid Device LUID                               No
  Device LUID                                     6d69-637300000000
  Device Node Mask                                0
  Device Numeric Version                          0xc00000 (3.0.0)
  Driver Version                                  550.67
  Device OpenCL C Version                         OpenCL C 1.2
  Device OpenCL C all versions                    OpenCL C                                                         0x400000 (1.0.0)
                                                  OpenCL C                                                         0x401000 (1.1.0)
                                                  OpenCL C                                                         0x402000 (1.2.0)
                                                  OpenCL C                                                         0xc00000 (3.0.0)
  Device OpenCL C features                        __opencl_c_fp64                                                  0xc00000 (3.0.0)
                                                  __opencl_c_images                                                0xc00000 (3.0.0)
                                                  __opencl_c_int64                                                 0xc00000 (3.0.0)
                                                  __opencl_c_3d_image_writes                                       0xc00000 (3.0.0)
  Latest comfornace test passed                   v2023-10-10-00
  Device Type                                     GPU
  Device Topology (NV)                            PCI-E, 0000:0b:00.0
  Device Profile                                  FULL_PROFILE
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Max compute units                               82
  Max clock frequency                             1695MHz
  Compute Capability (NV)                         8.6
  Device Partition                                (core)
    Max number of sub-devices                     1
    Supported partition types                     None
    Supported affinity domains                    (n/a)
  Max work item dimensions                        3
  Max work item sizes                             1024x1024x64
  Max work group size                             1024
  Preferred work group size multiple (device)     32
  Preferred work group size multiple (kernel)     32
  Warp size (NV)                                  32
  Max sub-groups per work group                   0
  Preferred / native vector sizes
    char                                                 1 / 1
    short                                                1 / 1
    int                                                  1 / 1
    long                                                 1 / 1
    half                                                 0 / 0        (n/a)
    float                                                1 / 1
    double                                               1 / 1        (cl_khr_fp64)
  Half-precision Floating-point support           (n/a)
  Single-precision Floating-point support         (core)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  Yes
  Double-precision Floating-point support         (cl_khr_fp64)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
  Address bits                                    64, Little-Endian
  Global memory size                              25422266368 (23.68GiB)
  Error Correction support                        No
  Max memory allocation                           6355566592 (5.919GiB)
  Unified memory for Host and Device              No
  Integrated memory (NV)                          No
  Shared Virtual Memory (SVM) capabilities        (core)
    Coarse-grained buffer sharing                 Yes
    Fine-grained buffer sharing                   No
    Fine-grained system sharing                   No
    Atomics                                       No
  Minimum alignment for any data type             128 bytes
  Alignment of base address                       4096 bits (512 bytes)
  Preferred alignment for atomics
    SVM                                           0 bytes
    Global                                        0 bytes
    Local                                         0 bytes
  Atomic memory capabilities                      relaxed, work-group scope
  Atomic fence capabilities                       relaxed, acquire/release, work-group scope
  Max size for global variable                    0
  Preferred total size of global vars             0
  Global Memory cache type                        Read/Write
  Global Memory cache size                        2351104 (2.242MiB)
  Global Memory cache line size                   128 bytes
  Image support                                   Yes
    Max number of samplers per kernel             32
    Max size for 1D images from buffer            268435456 pixels
    Max 1D or 2D image array size                 2048 images
    Max 2D image size                             32768x32768 pixels
    Max 3D image size                             16384x16384x16384 pixels
    Max number of read image args                 256
    Max number of write image args                32
    Max number of read/write image args           0
  Pipe support                                    No
  Max number of pipe args                         0
  Max active pipe reservations                    0
  Max pipe packet size                            0
  Local memory type                               Local
  Local memory size                               49152 (48KiB)
  Registers per block (NV)                        65536
  Max number of constant args                     9
  Max constant buffer size                        65536 (64KiB)
  Generic address space support                   No
  Max size of kernel argument                     32764 (32KiB)
  Queue properties (on host)
    Out-of-order execution                        Yes
    Profiling                                     Yes
  Device enqueue capabilities                     (n/a)
  Queue properties (on device)
    Out-of-order execution                        No
    Profiling                                     No
    Preferred size                                0
    Max size                                      0
  Max queues on device                            0
  Max events on device                            0
  Prefer user sync for interop                    No
  Profiling timer resolution                      1000ns
  Execution capabilities
    Run OpenCL kernels                            Yes
    Run native kernels                            No
    Non-uniform work-groups                       No
    Work-group collective functions               No
    Sub-group independent forward progress        No
    Kernel execution timeout (NV)                 Yes
  Concurrent copy and kernel execution (NV)       Yes
    Number of async copy engines                  2
    IL version                                    (n/a)
    ILs with version                              <printDeviceInfo:186: get CL_DEVICE_ILS_WITH_VERSION : error -30>
  printf() buffer size                            1048576 (1024KiB)
  Built-in kernels                                (n/a)
  Built-in kernels with version                   <printDeviceInfo:190: get CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION : error -30>
  Device Extensions                               cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts cl_nv_create_buffer cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_device_uuid cl_khr_pci_bus_info cl_khr_external_semaphore cl_khr_external_memory cl_khr_external_semaphore_opaque_fd cl_khr_external_memory_opaque_fd
  Device Extensions with Version                  cl_khr_global_int32_base_atomics                                 0x400000 (1.0.0)
                                                  cl_khr_global_int32_extended_atomics                             0x400000 (1.0.0)
                                                  cl_khr_local_int32_base_atomics                                  0x400000 (1.0.0)
                                                  cl_khr_local_int32_extended_atomics                              0x400000 (1.0.0)
                                                  cl_khr_fp64                                                      0x400000 (1.0.0)
                                                  cl_khr_3d_image_writes                                           0x400000 (1.0.0)
                                                  cl_khr_byte_addressable_store                   0x400000 (1.0.0)
                                                  cl_khr_icd                   0x400000 (1.0.0)
                                                  cl_khr_gl_sharing                   0x400000 (1.0.0)
                                                  cl_nv_compiler_options                   0x400000 (1.0.0)
                                                  cl_nv_device_attribute_query                   0x400000 (1.0.0)
                                                  cl_nv_pragma_unroll                   0x400000 (1.0.0)
                                                  cl_nv_copy_opts                   0x400000 (1.0.0)
                                                  cl_nv_create_buffer                   0x400000 (1.0.0)
                                                  cl_khr_int64_base_atomics                   0x400000 (1.0.0)
                                                  cl_khr_int64_extended_atomics                   0x400000 (1.0.0)
                                                  cl_khr_device_uuid                   0x400000 (1.0.0)
                                                  cl_khr_pci_bus_info                   0x400000 (1.0.0)
                                                  cl_khr_external_semaphore                     0x9000 (0.9.0)
                                                  cl_khr_external_memory                     0x9000 (0.9.0)
                                                  cl_khr_external_semaphore_opaque_fd                     0x9000 (0.9.0)
                                                  cl_khr_external_memory_opaque_fd                     0x9000 (0.9.0)

NULL platform behavior
  clGetPlatformInfo(NULL, CL_PLATFORM_NAME, ...)  No platform
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, ...)   No platform
  clCreateContext(NULL, ...) [default]            No platform
  clCreateContext(NULL, ...) [other]              Success [NV]
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_DEFAULT)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CUSTOM)  Invalid device type for platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL)  No platform
```
