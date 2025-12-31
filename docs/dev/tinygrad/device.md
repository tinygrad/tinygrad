# Device

`tinygrad/device.py` manages the compute devices, memory allocation, and compilation.

## `Device` Class
The `_Device` class (singleton `Device`) acts as a registry and factory for supported devices (e.g., METAL, CUDA, CPU, DISK).
- `Device[name]`: Returns a `Compiled` instance for the specified device.
- `Device.DEFAULT`: The default device used for tensors.

## `Buffer` Class
Represents a contiguous block of memory on a device.
- `allocate()`: Allocates memory using the device's allocator.
- `copyin()`: Copies data from a Python object/buffer to the device.
- `copyout()`: Copies data from the device to a Python object/buffer.
- `view()`: Creates a view into an existing buffer.

## `Allocator` Class
Abstract base class for memory allocators.
- `LRUAllocator`: Implements a Least Recently Used cache for buffers to reduce allocation overhead.

## `Compiled` Class
Represents a device that executes compiled code.
- `renderer`: Converts UOps to the target language (e.g., C++, CUDA, Metal).
- `compiler`: Compiles the rendered code to a binary.
- `runtime`: Executes the binary.

## Support Checks
`is_dtype_supported` checks if a specific data type is supported on a given device.
