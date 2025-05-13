import math
from tinygrad import Tensor
from tinygrad.helpers import argfix
from icecream import install
from typing import Sequence
install()

def unfold(t, dim:int, size:int, step:int):
    """
    Unfolds the tensor along dimension `dim` into overlapping blocks.
    Each block has length `size` and starts every `step` elements.
    Returns a tensor with an extra dimension of size `size`.

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
    if size > t.shape[dim]: raise RuntimeError(f'maximum size for tensor at dimension {dim} is {self.shape[dim]} but size is {size}')

    n_folds = (t.shape[dim] - size) // step + 1
    repeats = (n_folds,)+(1,)*t.ndim

    t = t.unsqueeze(0).repeat(repeats)
    ic(n_folds, repeats, t.shape, t.numpy())
    idxs = [(i*step, i*step+size) for i in range(n_folds)]
    ic(idxs)

    for i in range(n_folds):
        t2 = t[i, :, i:i+size:step]
        ic(t2.numpy())

    # for i in range(n_folds):
    #     t[i, :, dim].shrink((i*step, i*step+size))

    # t = t.shrink(idxs)
    ic(t.numpy())
    return t



def unfold2(t, dim:int, size:int, step:int):
    """
    Unfolds the tensor along dimension `dim` into overlapping blocks.
    Each block has length `size` and starts every `step` elements.
    Returns a tensor with an extra dimension of size `size`.

    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(8).unfold(0,2,2)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(27).reshape(3,3,3).unfold(-1,2,3)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```

    dim = d2
    t1.shape = (2, 5) = (d1, d2)
    t2.shape = (4, 2, 5) = (n_folds, d1, d2)
    t3.shape = (4, 2, 6) = (n_folds, d1 + d1%size, d2 + d2%size)
    t4.shape = (4, 2, 3, 2) = (n_folds, d1 + d1%size, d2//size, size)
    """
    dim = t._resolve_dim(dim)
    if step <= 0: raise RuntimeError(f'step must be >0 but got {step=}')
    if size < 0: raise RuntimeError(f'size must be >= 0 but got {size=}')
    if size > t.shape[dim]: raise RuntimeError(f'maximum size for tensor at dimension {dim} is {t.shape[dim]} but size is {size}')
    ic(dim, size, step)
    ic(t.shape, t.numpy())

    n_folds = (t.shape[dim] - size) // step + 1
    ic(n_folds, size)
    t2 = t.unsqueeze(0).repeat((n_folds,) + (1,)*t.ndim)
    ic(t2.shape, t2.numpy())

    padding = [None if i == 0 else (0, t2.shape[i] % size) for i, shp in enumerate(t2.shape)]
    ic(padding)
    t3 = t2.pad(padding)
    ic(t3.shape, t3.numpy())

    # add +1 b/c we added a new dimension in the 0th dim, so shifts all dimensions by 1
    new_shp = [shp//size if i-1 == dim else shp for i, shp in enumerate(t3.shape)] + [size]
    ic(new_shp)
    t4 = t3.reshape(new_shp)
    ic(t4.shape, t4.numpy())

    new_shp = t4.shape
    # new_shp
    # t5 =



def unfold3(t, dim:int, size:int, step:int):
    """
    Unfolds the tensor along dimension `dim` into overlapping blocks.
    Each block has length `size` and starts every `step` elements.
    Returns a tensor with an extra dimension of size `size`.

    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(8).unfold(0,2,2)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    unfolded = Tensor.arange(27).reshape(3,3,3).unfold(-1,2,3)
    print("\\n".join([repr(x.numpy()) for x in unfolded]))
    ```

    dim = d2
    t1.shape = (2, 5) = (d1, d2)
    t2.shape = (4, 2, 5) = (n_folds, d1, d2)
    t3.shape = (4, 2, 6) = (n_folds, d1 + d1%size, d2 + d2%size)
    t4.shape = (4, 2, 3, 2) = (n_folds, d1 + d1%size, d2//size, size)
    """
    dim = t._resolve_dim(dim)
    if step <= 0: raise RuntimeError(f'step must be >0 but got {step=}')
    if size < 0: raise RuntimeError(f'size must be >= 0 but got {size=}')
    if size > t.shape[dim]: raise RuntimeError(f'maximum size for tensor at dimension {dim} is {t.shape[dim]} but size is {size}')

    # n_folds = (t.shape[dim] - size) // step + 1
    # t = t.unsqueeze(0).repeat((n_folds,) + (1,)*t.ndim)
    # ic(t.shape, t.numpy())
    # mask = Tensor.zeros_like(t)
    # index = Tensor.cat(*[Tensor.arange(i, i+size, step).unsqueeze(0) for i in range(n_folds)]) # replace dim with (n_folds, size)
    # ic(index.shape, index.numpy())
    # return t.gather(dim+1, index)




    # n_folds = (t.shape[dim] - size) // step + 1

    # ic(t.shape, t.numpy())
    # # t = t.unsqueeze(0).repeat((n_folds,) + (1,)*t.ndim)
    # # ic(t.shape, t.numpy())

    # # Direct index calculation
    # idx = (Tensor.arange(n_folds).unsqueeze(1) * step + Tensor.arange(size).unsqueeze(0))
    # ic(idx.shape, idx.numpy())

    # # Reshape to match tensor dimensions
    # shape = [1] * t.ndim
    # shape[dim] = n_folds
    # shape.insert(dim+1, size)
    # ic(shape)
    # idx = idx.reshape(shape).expand(t.shape[:dim] + (n_folds, size) + t.shape[dim+1:])
    # ic(idx.shape, idx.numpy())

    # return t.gather(dim, idx)

    # n_folds = (t.shape[dim] - size) // step + 1
    # # create windowed indices
    # idx = Tensor.arange(n_folds).unsqueeze(1) * step + Tensor.arange(size)
    # # reshape for gather
    # idx = idx.reshape([1]*dim + [-1] + [1]*(t.ndim-dim-1)).expand(t.shape[:dim] + (n_folds * size,) + t.shape[dim+1:])
    # # gather + final reshape
    # return t.gather(dim, idx).reshape(t.shape[:dim] + (n_folds, size) + t.shape[dim+1:])





    # def _shp(lst:list, idx:int, new_val:int|list): return lst[:idx] + [new_val] if isinstance(new_val, int) else new_val + lst[idx+1:]

    # n_folds = (t.shape[dim] - size) // step + 1
    # shp1 = _shp([1]*t.ndim, dim, n_folds*size)
    # shp2 = _shp(t.shape, dim, n_folds*size)
    # shp3 = _shp(t.shape, dim, [n_folds, size])

    # # Create sliding window indices: [[0,1], [step,step+1], [2*step,2*step+1], ...]
    # idx = Tensor.arange(n_folds).unsqueeze(1) * step + Tensor.arange(size)

    # # Reshape to add singleton dims for broadcasting with input tensor
    # # Shape: [1, 1, ..., n_folds*size, ..., 1, 1]
    # idx = idx.reshape([1]*dim + [n_folds * size] + [1]*(t.ndim-dim-1))
    # idx = idx.expand(t.shape[:dim] + (n_folds * size,) + t.shape[dim+1:])

    # # Gather windows along dim and separate folds from each block
    # return t.gather(dim, idx).reshape(t.shape[:dim] + (n_folds, size) + t.shape[dim+1:])


    # n_folds = (t.shape[dim] - size) // step + 1
    # def _shp(lst:list, idx:int, new_val:int|list): return lst[:idx] + [new_val] if isinstance(new_val, int) else new_val + lst[idx+1:]
    # shp1, shp2, shp3 = _shp([1]*t.ndim, dim, n_folds*size), _shp(t.shape, dim, n_folds*size), _shp(t.shape, dim, [n_folds, size])

    # # Create sliding window indices (e.g. [[0,1], [step,step+1], [2*step,2*step+1], ...]) and reshape
    # idx = Tensor.arange(n_folds).unsqueeze(1) * step + Tensor.arange(size)
    # idx = idx.reshape(shp1).expand(shp2)
    # # Gather windows along dim and separate folds from each block
    # return t.gather(dim, idx).reshape(shp3)



    n_folds = (t.shape[dim] - size) // step + 1
    # Create sliding window indices: [[0,1], [step,step+1], [2*step,2*step+1], ...]
    idx = Tensor.arange(n_folds).unsqueeze(1) * step + Tensor.arange(size)
    idx = idx.reshape([1]*dim + [n_folds * size] + [1]*(t.ndim-dim-1)).expand(t.shape[:dim] + (n_folds * size,) + t.shape[dim+1:])
    # Gather windows along dim and separate folds from each block
    return t.gather(dim, idx).reshape(t.shape[:dim] + (n_folds, size) + t.shape[dim+1:])


if __name__ == '__main__':
    t1 = Tensor.arange(1,11).reshape(2,5)
    t2 = unfold3(t1, 1,2,1)
    ic(t2.shape, t2.numpy())


    t1 = Tensor.arange(1., 8)
    t2 = unfold3(t1, 0, 2, 1)
    ic(t2.shape, t2.numpy())
