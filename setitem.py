from tinygrad import Tensor
import functools

def gen_mask(self: Tensor, indices):
    masks = []
    for dim,(dim_size, idx) in enumerate(zip(self.shape, indices)):
        if isinstance(idx, int):
            # e.g. dim=1, idx=3, shape=(10,20,30,40,50)
            #              () -> (20,) -> (1,20,1,1,1) -> (10,20,30,40,50)
            mask = Tensor(idx).one_hot(dim_size).reshape((1,)*dim + (dim_size,) + (1,)*(self.ndim-dim-1)).expand(self.shape)
        elif isinstance(idx, slice):
            # e.g. dim=3, idx=slice(1,12,3) = [1,4,7,10], shape=(10,20,30,40,50)
            #              () -> (4,) -> (4,20) -> (20,) -> (1,20,1,1,1) -> (10,20,30,40,50)
            # Negative start/stop is handled by slice.indices, but negative step could be tricky
            start, stop, step = idx.indices(dim_size)
            mask = Tensor.arange(start, stop, step).one_hot(dim_size).sum(0).reshape((1,)*dim + (dim_size,) + (1,)*(self.ndim-dim-1)).expand(self.shape)
        elif idx is None:
            continue
        else:
            raise NotImplementedError(f"Indexing with {type(idx)} not supported")
        masks.append(mask)
    return functools.reduce(lambda x,y: x.mul(y), masks)

"""
e.g.
x : (10,20,30,40,50)
v : (10, 1, 30, 4, 50) or broadcastable to this shape
x[:, 3, ..., 1:12:3] = v
"""
def gen_index_shape(shape: tuple[int,...], indices):
    masks = []
    ndim = len(shape)
    indices = list(indices) + [None]*(ndim - len(indices))
    res_shape = []
    for (dim_size, idx) in zip(shape, indices):
        if isinstance(idx, int):
            res_shape.append(1)
        elif isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or dim_size
            step = idx.step or 1
            res_shape.append((stop - start + (step or 1) - 1) // (step or 1))
        elif idx is None:
            res_shape.append(dim_size)
        else:
            raise NotImplementedError(f"Indexing with {type(idx)} not supported")
    return tuple(res_shape)

def pad_values(v: Tensor, shape: tuple[int,...], indices):
    vshape = gen_index_shape(shape, indices)
    print(f"{vshape=}")
    ndim = len(shape)
    # e.g.
    # v.shape = (1, 1, 4, 1) -> (10, 1, 30, 4, 50)
    vb = v._broadcast_to(vshape)
    padding = []
    for dim,(dim_size, idx) in enumerate(zip(shape, indices)):
        print(f"dim={dim}, idx={idx}, dim_size={dim_size}")
        if isinstance(idx, int):
            pass
        elif isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or dim_size
            step = idx.step or 1
            if step > 1:
                # (10,1,30,4,50) -> (10,1,30,12,50) -> (10,1,30,(1+12+17),50)
                vb = vb.repeat_interleave(step, dim=dim)
                # e.g. dim=3, idx=slice(1,12,3) = [1,4,7,10], shape=(10,20,30,40,50)
            # pads = (None, None, (1, 17), 0, None)
            pad = (start, dim_size - (start + vshape[dim]*step ))
            pads = (None,) * dim + (pad, ) + (None,) * (ndim - dim - 1)
            print(f"{pads=}")
            vb = vb.pad(pads)
        elif idx is None:
            pass
        else:
            raise NotImplementedError(f"Indexing with {type(idx)} not supported")
    return vb

def setitem(x: Tensor, indices, v: Tensor):
    shape = x.shape
    mask = gen_mask(x, indices)
    vb = pad_values(v, shape, indices)
    print(f"{mask.numpy()=}")
    print(f"{vb.numpy()=}")
    return mask.where(vb, x)

def test_gen_mask():
    # single element
    x = Tensor.arange(6).reshape(2,3)
    mask = gen_mask(x, (1,2))
    print(f"{mask.numpy()=}")
    assert (mask == Tensor([[0,0,0],[0,0,1]])).sum().item() == 6

    # single element 2
    mask = gen_mask(x, (0,1))
    print(f"{mask.numpy()=}")
    assert (mask == Tensor([[0,1,0],[0,0,0]])).sum().item() == 6

    # partial indexing
    mask = gen_mask(x, (0,))
    print(f"{mask.numpy()=}")
    assert (mask == Tensor([[1,1,1],[0,0,0]])).sum().item() == 6

    # partial indexing
    mask = gen_mask(x, (None,1))
    print(f"{mask.numpy()=}")
    assert (mask == Tensor([[0,1,0],[0,1,0]])).sum().item() == 6

    # slice indexing
    x = Tensor.arange(24).reshape(2,3,4)
    mask = gen_mask(x, (1, slice(1,3), slice(0,4,2)))
    print(f"{mask.numpy()=}")
    assert (mask == Tensor([[[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,0,1,0],[1,0,1,0]]])).sum().item() == 24
    
def test_setitem():
    x = Tensor.zeros(2, 3, 4)
    v = Tensor.arange(1, 5).reshape((2,2))
    print(f"{x.numpy()=}")
    print(f"{v.numpy()=}")
    x = setitem(x, (None, slice(1,3), slice(1,4,2)), v)
    print(f"{x.numpy()=}")

