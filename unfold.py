from tinygrad import Tensor, GlobalCounters
from icecream import install
install()

def unfold_stack(t:Tensor, dim:int, size:int, step:int):
    """
    Unfolds the tensor along dimension `dim` into overlapping windows.

    Each window has length `size` and begins every `step` elements of `t`.
    Returns the input tensor with dimension `dim` replaced by dims `(n_windows, size)` where `n_windows = (t.shape[dim] - size) // step + 1`.

    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(8).unfold(0,2,2)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(27).reshape(3,3,3).unfold(-1,2,3)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    """
    dim = t._resolve_dim(dim)
    if size < 0: raise RuntimeError(f'size must be >= 0 but got {size=}')
    if step <= 0: raise RuntimeError(f'step must be >0 but got {step=}')
    if size > t.shape[dim]: raise RuntimeError(f'maximum size for tensor at dimension {dim} is {t.shape[dim]} but size is {size}')

    n_windows = (t.shape[dim] - size) // step + 1
    slices = [t[(slice(None),)*dim + (slice(i*step, i*step+size),)] for i in range(n_windows)]
    ret = Tensor.stack(*slices, dim=dim)
    return ret

def unfold_gather(t:Tensor, dim:int, size:int, step:int):
    """
    Unfolds the tensor along dimension `dim` into overlapping windows.

    Each window has length `size` and begins every `step` elements of `t`.
    Returns the input tensor with dimension `dim` replaced by dims `(n_windows, size)` where `n_windows = (t.shape[dim] - size) // step + 1`.

    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(8).unfold(0,2,2)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(27).reshape(3,3,3).unfold(-1,2,3)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    """
    dim = t._resolve_dim(dim)
    if size < 0: raise RuntimeError(f'size must be >= 0 but got {size=}')
    if step <= 0: raise RuntimeError(f'step must be >0 but got {step=}')
    if size > t.shape[dim]: raise RuntimeError(f'maximum size for tensor at dimension {dim} is {t.shape[dim]} but size is {size}')
    ic(dim, size, step)
    ic(t.shape, t.numpy())

    n_windows = (t.shape[dim] - size) // step + 1
    idx = Tensor.arange(n_windows).unsqueeze(1) * step + Tensor.arange(size)
    idx = idx.reshape([1]*dim + [n_windows * size] + [1]*(t.ndim-dim-1)).expand(t.shape[:dim] + (n_windows * size,) + t.shape[dim+1:])
    ic(idx.shape, idx.numpy())
    return t.gather(dim, idx).reshape(t.shape[:dim] + (n_windows, size) + t.shape[dim+1:])

def unfold(t:Tensor, dim:int, size:int, step:int):
    """
    Unfolds the tensor along dimension `dim` into overlapping windows.

    Each window has length `size` and begins every `step` elements of `t`.
    Returns the input tensor with dimension `dim` replaced by dims `(n_windows, size)`
    where `n_windows = (t.shape[dim] - size) // step + 1`.

    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(8).unfold(0,2,2)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(27).reshape(3,3,3).unfold(-1,2,3)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    """
    dim = t._resolve_dim(dim)
    if size < 0: raise RuntimeError(f'size must be >= 0 but got {size=}')
    if step <= 0: raise RuntimeError(f'step must be >0 but got {step=}')
    if size > t.shape[dim]: raise RuntimeError(f'maximum size for tensor at dimension {dim} is {t.shape[dim]} but size is {size}')

    n_windows = (t.shape[dim] - size) // step + 1
    print(n_windows, size)
    print(t.shape)
    t2 = t.unsqueeze(0).repeat((n_windows,) + (1,)*t.ndim)
    print(t2.shape, t2.numpy())

    padding = [None if i == 0 else (0, t2.shape[i] % size) for i, shp in enumerate(t2.shape)]
    print(padding)
    t3 = t2.pad(padding)
    print(t3.shape, t3.numpy())

    # add +1 b/c we added a new dimension in the 0th dim, so shifts all dimensions by 1
    new_shp = [shp//size if i-1 == dim else shp for i, shp in enumerate(t3.shape)] + [size]
    ic(new_shp)
    t4 = t3.reshape(new_shp)
    print(t4.shape, t4.numpy())
    return t4

if __name__ == '__main__':
    t1 = Tensor.arange(16_384).reshape(1024, -1)
    # out = unfold_stack(t1, 1, 2, 1).realize()
    # out = unfold(t1, 1, 6, 1).realize()
    # print(f"{GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB, {GlobalCounters.time_sum_s*1e3:9.2f} ms, {GlobalCounters.kernel_count} kernels")

    # t1 = Tensor.arange(1., 8)
    # unfold(t1, 0, 2, 1).realize()
