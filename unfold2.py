import math
from tinygrad import Tensor
from icecream import install
install()

def main1():
    # dim, size, step = 0, 2, 1
    # t = Tensor.arange(1., 8)

    dim, size, step = 1, 2, 1
    t = Tensor.arange(1., 11).reshape(5, 2)
    ic(dim, size, step)
    ic(t.shape, t.numpy())

    repeats = math.ceil(size / step)
    # n_windows = math.ceil((t.shape[dim] - size) / step)
    n_windows = (t.shape[dim] - size) // step + 1
    ic(repeats, n_windows)

    t = t.reshape(t.shape+(1,)).expand(t.shape+(repeats,))
    ic(t.shape, t.numpy())

    t = t.reshape(t.shape[0], size, n_windows+1)
    ic(t.shape, t.numpy())

    t = t.permute(2, 1, 0)
    ic(t.shape, t.numpy())


    t.numpy()

def unfold(t, dim, size, step):
    ic(dim, size, step)
    ic(t.shape, t.numpy())

    t = t.reshape(t.shape+(1,)).expand(-1, size)
    ic(t.shape, t.numpy())

    t = t.pad((0, 0, 0, 1))
    ic(t.shape, t.numpy())

    t = t.flatten()
    ic(t.shape, t.numpy())

    t = t.roll(1, 0)
    ic(t.shape, t.numpy())

    t = t.reshape(8, 2)
    ic(t.shape, t.numpy())

    t = t.shrink(((1, t.shape[0]-1), None))
    ic(t.shape, t.numpy())

    t = t.reshape(t.shape[:-1] + (step, -1))
    ic(t.shape, t.numpy())

    return t


def main():

    dim, size, step = 0, 2, 2
    t = Tensor.arange(1., 8, device='cpu')
    out = unfold(t, dim, size, step)

    # dim, size, step = 0, 2, 1
    # t = Tensor.arange(1., 8, device='cpu')
    # out = unfold(t, dim, size, step)
    # expected = Tensor([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.], [6., 7.]])
    # assert (out.numpy() == expected.numpy()).all()

    dim, size, step = 1, 2, 1
    t = Tensor.arange(1., 11).reshape(5, 2)

if __name__ == '__main__':
    main()
