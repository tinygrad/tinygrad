# DType

`tinygrad/dtype.py` defines the data types used in tinygrad.

## `DType` Class
Represents a data type.
- Attributes: `priority`, `itemsize`, `name`, `fmt`, `count`.
- `vec(sz)`: Creates a vectorized version of the dtype (e.g., float4).
- `scalar()`: Returns the scalar version of a vectorized dtype.

## `PtrDType` and `ImageDType`
- `PtrDType`: Represents a pointer to a dtype, optionally with an address space.
- `ImageDType`: Special type for images, often used in GPU contexts.

## `dtypes` Namespace
Contains predefined constants for standard types:
- `bool`
- `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`
- `float16` (`half`), `bfloat16`, `float32` (`float`), `float64` (`double`)
- `fp8` types.

## Utilities
- `to_dtype`: Converts strings or classes to `DType`.
- `least_upper_dtype`: Finds the smallest common type that can represent a set of types.
- `sum_acc_dtype`: Determines the accumulator type for reductions.
- `float_to_fp8`, `fp8_to_float`: Conversion utilities for 8-bit floating point.
