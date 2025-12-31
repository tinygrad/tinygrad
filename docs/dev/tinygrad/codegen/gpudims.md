# GPU Dims Implementation Details

`tinygrad/codegen/gpudims.py` calculates optimal `global_size` and `local_size` for GPU kernels.

## 1. Goal
To utilize the GPU hardware efficiently by:
*   Maximizing occupancy (active warps).
*   Respecting hardware limits (max threads per block, max shared memory).
*   Aligning memory accesses (coalescing).

## 2. `GPUDims` Class

### 2.1 `optimize_local_size`
Calculates `local_size` (workgroup size) given `global_size` (total work items).

1.  **Factors**: `global_size` is usually factorized into `local_size * group_count`.
2.  **Heuristics**:
    *   Prefers larger local sizes (up to 256 or 1024) to hide latency.
    *   Prefers dimensions that align with `upcasted` loops (vectorization).
    *   Avoids local sizes that leave many threads idle (if global % local != 0).

### 2.2 `optimize_global_size`
Adjusts `global_size` if necessary (e.g., for image textures that require specific alignment).

## 3. Image Handling

If the kernel uses `ImageDType`:
*   GPUs access images using (x, y) coordinates.
*   The logic ensures `global_size` maps cleanly to these coordinates.
*   `OCL` (OpenCL) typically has hard limits on image dimensions.

## 4. Tensor Core Dimensions

If `Tensor Cores` (WMMA) are used:
*   The local size must respect the warp size (32 for NVIDIA, 64 for AMD).
*   The dimensions are often fixed (e.g., 16x16 blocks).
*   The logic hardcodes `local_size=[32, 1, 1]` or similar for TC kernels.
