# DType Implementation Details

`tinygrad/dtype.py` defines the type system. It is designed to be minimal but sufficient for deep learning.

## 1. `DType` Class

### 1.1 Attributes
*   **`priority`**: Used for type promotion. Higher priority types "win" (e.g., `float32` > `int32`).
*   **`itemsize`**: Size in bytes.
*   **`name`**: String name (e.g., "float32").
*   **`fmt`**: Python `struct` format character (for packing/unpacking).
*   **`count`**: Vectorization factor (1 for scalar, 4 for float4).
*   **`_scalar`**: Reference to the scalar base type if vectorized.

### 1.2 Methods
*   **`vec(sz)`**: Returns a vectorized version. Memoized.
*   **`ptr(size, addrspace)`**: Returns a `PtrDType`.

## 2. Special DTypes

### 2.1 `PtrDType`
Represents a pointer to a `DType`.
*   **`size`**: Size of the buffer pointed to (if known).
*   **`addrspace`**: `AddrSpace` enum (GLOBAL, LOCAL, REG). Used in codegen to specify memory bank.

### 2.2 `ImageDType`
Subclass of `PtrDType`. Represents an opaque image object (like `image2d_t` in OpenCL).
*   **`shape`**: Tuple (height, width, ...).
*   Used for optimizing memory access patterns on GPUs (texture cache).

## 3. The `dtypes` Namespace

Acts as a static registry of types.

### 3.1 Primitives
*   **Floats**: `half` (fp16), `float` (fp32), `double` (fp64), `bfloat16`.
*   **Ints**: `int8`, `int16`, `int32`, `int64`.
*   **UInts**: `uint8`, `uint16`, `uint32`, `uint64`.
*   **Bool**: `bool`.
*   **FP8**: `fp8e4m3`, `fp8e5m2`.

### 3.2 Constants & Sets
*   **`floats`**, **`ints`**, **`uints`**: Tuples grouping types for `isinstance` checks.
*   **`default_float`**: Defaults to `float32` (configurable via `DEFAULT_FLOAT` env var).

## 4. Type Promotion & Casting

### 4.1 `least_upper_dtype`
Determines the output type of a binary operation (e.g., `int32 + float32 -> float32`).
*   Uses a promotion lattice (`promo_lattice`).
*   Traverses recursive parents to find the lowest common ancestor.

### 4.2 `truncate`
Dictionary mapping DTypes to functions that truncate python values to fit the type (e.g., handling overflow, float->int conversion).

### 4.3 FP8 Conversion
*   **`float_to_fp8`**: Bitwise manipulation to pack float32 into 8 bits. Handles bias, denormals, NaN/Inf.
*   **`fp8_to_float`**: Unpacks 8 bits to float32.

## 5. Interop

*   **`_to_np_dtype`**: Maps tinygrad `DType` to `numpy.dtype`.
*   **`_from_np_dtype`**: Maps `numpy.dtype` to tinygrad `DType`.
*   **`_to_torch_dtype`**: Maps to `torch.dtype`.
