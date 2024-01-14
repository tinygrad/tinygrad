# test cases are modified from pytorch test_indexing.py https://github.com/pytorch/pytorch/blob/597d3fb86a2f3b8d6d8ee067e769624dcca31cdb/test/test_indexing.py

import math, unittest, random, copy
# import warnings
import numpy as np

from tinygrad import Tensor, dtypes, Device
# from tinygrad import TinyJit
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.helpers import CI

random.seed(42)

def numpy_testing_assert_equal_helper(a, b):
  if isinstance(a, Tensor): a = a.numpy()
  if isinstance(b, Tensor): b = b.numpy()
  np.testing.assert_equal(a, b)

def consec(shape, start=1):
  return Tensor(np.arange(math.prod(shape)).reshape(shape)+start)

# creates strided tensor with base set to reference tensor's base, equivalent to torch.set_()
def set_(reference: Tensor, shape, strides, offset):
  if reference.lazydata.base.realized is None: reference.realize()
  assert reference.lazydata.base.realized, "base has to be realized before setting it to strided's base"
  # TODO: this shouldn't directly create a LazyBuffer
  strided = Tensor(LazyBuffer(device=reference.device,
                              st=ShapeTracker((View.create(shape=shape, strides=strides, offset=offset),)),
                              op=None, dtype=reference.dtype, srcs=(), base=reference.lazydata.base))
  assert strided.lazydata.st.real_strides() == strides, "real_strides should equal strides for strided"
  return strided

# TODO tries to mimic .detach().copy() or just .copy() behavior
# for torch.copy() the resulting tensor.data_ptr() is different than the original
# some random medium article says .clone() is a shallow copy, so maybe this is correct?
# https://discuss.pytorch.org/t/clone-and-detach-in-v0-4-0/16861/4
def clone(original:Tensor):
  ret = Tensor(copy.copy(original.lazydata), device=original.device, dtype=original.dtype, requires_grad=original.requires_grad)
  # TODO nah this isn't right either. maybe compare their loadops??? idk
  assert ret.lazydata is not original.lazydata, "clone should create a lazydata copy"
  return ret

# TODO torch.data_ptr
def data_ptr(tensor:Tensor):
  ...

# TODO torch.argsort()
def argsort(tensor:Tensor) -> Tensor:
  ...

# TODO torch.all()
# not sure if this is right
def all_(tensor:Tensor) -> Tensor:
  return tensor != 0

def diagonal(tensor:Tensor) -> Tensor:
  assert all(sh == sh[0] for sh in tensor.shape), "matrix should be square"
  assert tensor.ndim == 2, 'only support 2 ndim tensors'
  return (Tensor.eye(tensor.shape[0]) * tensor).sum(0)

# TODO torch.copy_()
def copy_(src:Tensor, other:Tensor) -> Tensor:
  ...

def unravel_index(tensor, shape):
  ...

def make_tensor(shape, dtype:dtypes, noncontiguous) -> Tensor:
  r"""Creates a tensor with the given :attr:`shape`, :attr:`device`, and :attr:`dtype`, and filled with
  values uniformly drawn from ``[low, high)``.

  If :attr:`low` or :attr:`high` are specified and are outside the range of the :attr:`dtype`'s representable
  finite values then they are clamped to the lowest or highest representable finite value, respectively.
  If ``None``, then the following table describes the default values for :attr:`low` and :attr:`high`,
  which depend on :attr:`dtype`.

  +---------------------------+------------+----------+
  | ``dtype``                 | ``low``    | ``high`` |
  +===========================+============+==========+
  | boolean type              | ``0``      | ``2``    |
  +---------------------------+------------+----------+
  | unsigned integral type    | ``0``      | ``10``   |
  +---------------------------+------------+----------+
  | signed integral types     | ``-9``     | ``10``   |
  +---------------------------+------------+----------+
  | floating types            | ``-9``     | ``9``    |
  +---------------------------+------------+----------+
  | complex types             | ``-9``     | ``9``    |
  +---------------------------+------------+----------+
  """
  contiguous = not noncontiguous # lol
  if dtype is dtypes.bool: return Tensor.randint(shape=shape, low=0, high=2, contiguous=contiguous).cast(dtypes.bool)
  elif dtype.is_unsigned(): return Tensor.randint(shape=shape, low=0, high=10, contiguous=contiguous).cast(dtype)
  elif dtype.is_int(): return Tensor.randint(shape=shape, low=-9, high=10, contiguous=contiguous).cast(dtype) # signed int
  elif dtype.is_float(): return Tensor.rand(shape=shape, low=-9, high=9, dtype=dtype, contiguous=contiguous)
  else: raise NotImplementedError(f"{dtype} not implemented")

class TestIndexing(unittest.TestCase):
  def test_index(self):

    reference = consec((3, 3, 3))

    numpy_testing_assert_equal_helper(reference[0], consec((3, 3)))
    numpy_testing_assert_equal_helper(reference[1], consec((3, 3), 10))
    numpy_testing_assert_equal_helper(reference[2], consec((3, 3), 19))
    numpy_testing_assert_equal_helper(reference[0, 1], consec((3,), 4))
    numpy_testing_assert_equal_helper(reference[0:2], consec((2, 3, 3)))
    numpy_testing_assert_equal_helper(reference[2, 2, 2], 27)
    numpy_testing_assert_equal_helper(reference[:], consec((3, 3, 3)))

    # indexing with Ellipsis
    numpy_testing_assert_equal_helper(reference[..., 2], np.array([[3., 6., 9.],[12., 15., 18.],[21., 24., 27.]]))
    numpy_testing_assert_equal_helper(reference[0, ..., 2], np.array([3., 6., 9.]))
    numpy_testing_assert_equal_helper(reference[..., 2], reference[:, :, 2])
    numpy_testing_assert_equal_helper(reference[0, ..., 2], reference[0, :, 2])
    numpy_testing_assert_equal_helper(reference[0, 2, ...], reference[0, 2])
    numpy_testing_assert_equal_helper(reference[..., 2, 2, 2], 27)
    numpy_testing_assert_equal_helper(reference[2, ..., 2, 2], 27)
    numpy_testing_assert_equal_helper(reference[2, 2, ..., 2], 27)
    numpy_testing_assert_equal_helper(reference[2, 2, 2, ...], 27)
    numpy_testing_assert_equal_helper(reference[...], reference)

    reference_5d = consec((3, 3, 3, 3, 3))
    numpy_testing_assert_equal_helper(reference_5d[..., 1, 0], reference_5d[:, :, :, 1, 0])
    numpy_testing_assert_equal_helper(reference_5d[2, ..., 1, 0], reference_5d[2, :, :, 1, 0])
    numpy_testing_assert_equal_helper(reference_5d[2, 1, 0, ..., 1], reference_5d[2, 1, 0, :, 1])
    numpy_testing_assert_equal_helper(reference_5d[...], reference_5d)

    # None indexing
    numpy_testing_assert_equal_helper(reference[2, None], reference[2].unsqueeze(0))
    numpy_testing_assert_equal_helper(reference[2, None, None], reference[2].unsqueeze(0).unsqueeze(0))
    numpy_testing_assert_equal_helper(reference[2:4, None], reference[2:4].unsqueeze(1))
    numpy_testing_assert_equal_helper(reference[None, 2, None, None], reference.unsqueeze(0)[:, 2].unsqueeze(0).unsqueeze(0))
    numpy_testing_assert_equal_helper(reference[None, 2:5, None, None], reference.unsqueeze(0)[:, 2:5].unsqueeze(2).unsqueeze(2))

    # indexing 0-length slice
    numpy_testing_assert_equal_helper(np.empty((0, 3, 3)), reference[slice(0)])
    numpy_testing_assert_equal_helper(np.empty((0, 3)), reference[slice(0), 2])
    numpy_testing_assert_equal_helper(np.empty((0, 3)), reference[2, slice(0)])
    numpy_testing_assert_equal_helper(np.empty([]), reference[2, 1:1, 2])

    # indexing with step
    reference = consec((10, 10, 10))
    numpy_testing_assert_equal_helper(reference[1:5:2], Tensor.stack([reference[1], reference[3]], 0))
    numpy_testing_assert_equal_helper(reference[1:6:2], Tensor.stack([reference[1], reference[3], reference[5]], 0))
    numpy_testing_assert_equal_helper(reference[1:9:4], Tensor.stack([reference[1], reference[5]], 0))
    numpy_testing_assert_equal_helper(reference[2:4, 1:5:2], Tensor.stack([reference[2:4, 1], reference[2:4, 3]], 1))
    numpy_testing_assert_equal_helper(reference[3, 1:6:2], Tensor.stack([reference[3, 1], reference[3, 3], reference[3, 5]], 0))
    numpy_testing_assert_equal_helper(reference[None, 2, 1:9:4], Tensor.stack([reference[2, 1], reference[2, 5]], 0).unsqueeze(0))
    numpy_testing_assert_equal_helper(reference[:, 2, 1:6:2], Tensor.stack([reference[:, 2, 1], reference[:, 2, 3], reference[:, 2, 5]], 1))

    lst = [list(range(i, i+10)) for i in range(0, 100, 10)]
    tensor = Tensor(lst)
    for _ in range(100):
      idx1_start = random.randrange(10)
      idx1_end = idx1_start + random.randrange(1, 10 - idx1_start + 1)
      idx1_step = random.randrange(1, 8)
      idx1 = slice(idx1_start, idx1_end, idx1_step)
      if random.randrange(2) == 0:
        idx2_start = random.randrange(10)
        idx2_end = idx2_start + random.randrange(1, 10 - idx2_start + 1)
        idx2_step = random.randrange(1, 8)
        idx2 = slice(idx2_start, idx2_end, idx2_step)
        lst_indexed = [l[idx2] for l in lst[idx1]]
        tensor_indexed = tensor[idx1, idx2]
      else:
        lst_indexed = lst[idx1]
        tensor_indexed = tensor[idx1]
      numpy_testing_assert_equal_helper(tensor_indexed, np.array(lst_indexed))

    self.assertRaises(ValueError, lambda: reference[1:9:0])
    # NOTE torch doesn't support this but numpy does so we should too. Torch raises ValueError
    # see test_slice_negative_strides in test_ops.py
    # self.assertRaises(ValueError, lambda: reference[1:9:-1])

    self.assertRaises(IndexError, lambda: reference[1, 1, 1, 1])
    self.assertRaises(IndexError, lambda: reference[1, 1, 1, 1:1])
    self.assertRaises(IndexError, lambda: reference[3, 3, 3, 3, 3, 3, 3, 3])

    self.assertRaises(IndexError, lambda: reference[0.0])
    self.assertRaises(TypeError, lambda: reference[0.0:2.0])
    self.assertRaises(IndexError, lambda: reference[0.0, 0.0:2.0])
    self.assertRaises(IndexError, lambda: reference[0.0, :, 0.0:2.0])
    self.assertRaises(IndexError, lambda: reference[0.0, ..., 0.0:2.0])
    self.assertRaises(IndexError, lambda: reference[0.0, :, 0.0])

    # TODO: delitem
    # def delitem(): del reference[0]
    # self.assertRaises(TypeError, delitem)

  # TODO: LLVM is quite fast, why are other compiled backends slow?
  @unittest.skipIf(CI and Device.DEFAULT in ["CLANG", "GPU", "METAL"], "slow")
  def test_advancedindex(self):
    # integer array indexing

    # pick a random valid indexer type
    def ri(indices):
      choice = random.randint(0, 2)
      if choice == 0: return Tensor(indices)
      if choice == 1: return list(indices)
      return tuple(indices)

    def validate_indexing(x):
      numpy_testing_assert_equal_helper(x[[0]], consec((1,)))
      numpy_testing_assert_equal_helper(x[ri([0]),], consec((1,)))
      numpy_testing_assert_equal_helper(x[ri([3]),], consec((1,), 4))
      numpy_testing_assert_equal_helper(x[[2, 3, 4]], consec((3,), 3))
      numpy_testing_assert_equal_helper(x[ri([2, 3, 4]),], consec((3,), 3))
      numpy_testing_assert_equal_helper(x[ri([0, 2, 4]),], np.array([1, 3, 5]))

    def validate_setting(x):
      pass
      # TODO: setitem
      '''
      x[[0]] = -2
      numpy_testing_assert_equal_helper(x[[0]], np.array([-2]))
      x[[0]] = -1
      numpy_testing_assert_equal_helper(x[ri([0]), ], np.array([-1]))
      x[[2, 3, 4]] = 4
      numpy_testing_assert_equal_helper(x[[2, 3, 4]], np.array([4, 4, 4]))
      x[ri([2, 3, 4]), ] = 3
      numpy_testing_assert_equal_helper(x[ri([2, 3, 4]), ], np.array([3, 3, 3]))
      x[ri([0, 2, 4]), ] = np.array([5, 4, 3])
      numpy_testing_assert_equal_helper(x[ri([0, 2, 4]), ], np.array([5, 4, 3]))
      '''

    # Case 1: Purely Integer Array Indexing
    reference = consec((10,))
    validate_indexing(reference)

    # setting values
    validate_setting(reference)

    # Tensor with stride != 1
    # strided is [1, 3, 5, 7]

    reference = consec((10,))
    strided = set_(reference, (4,), (2,), 0)

    numpy_testing_assert_equal_helper(strided[[0]], np.array([1]))
    numpy_testing_assert_equal_helper(strided[ri([0]), ], np.array([1]))
    numpy_testing_assert_equal_helper(strided[ri([3]), ], np.array([7]))
    numpy_testing_assert_equal_helper(strided[[1, 2]], np.array([3, 5]))
    numpy_testing_assert_equal_helper(strided[ri([1, 2]), ], np.array([3, 5]))
    numpy_testing_assert_equal_helper(strided[ri([[2, 1], [0, 3]]), ],
                      np.array([[5, 3], [1, 7]]))

    # stride is [4, 8]

    strided = set_(reference, (2,), (4,), offset=4)

    numpy_testing_assert_equal_helper(strided[[0]], np.array([5]))
    numpy_testing_assert_equal_helper(strided[ri([0]), ], np.array([5]))
    numpy_testing_assert_equal_helper(strided[ri([1]), ], np.array([9]))
    numpy_testing_assert_equal_helper(strided[[0, 1]], np.array([5, 9]))
    numpy_testing_assert_equal_helper(strided[ri([0, 1]), ], np.array([5, 9]))
    numpy_testing_assert_equal_helper(strided[ri([[0, 1], [1, 0]]), ],
                      np.array([[5, 9], [9, 5]]))

    # reference is 1 2
    #              3 4
    #              5 6
    reference = consec((3, 2))
    numpy_testing_assert_equal_helper(reference[ri([0, 1, 2]), ri([0])], np.array([1, 3, 5]))
    numpy_testing_assert_equal_helper(reference[ri([0, 1, 2]), ri([1])], np.array([2, 4, 6]))
    numpy_testing_assert_equal_helper(reference[ri([0]), ri([0])], consec((1,)))
    numpy_testing_assert_equal_helper(reference[ri([2]), ri([1])], consec((1,), 6))
    numpy_testing_assert_equal_helper(reference[[ri([0, 0]), ri([0, 1])]], np.array([1, 2]))
    numpy_testing_assert_equal_helper(reference[[ri([0, 1, 1, 0, 2]), ri([1])]], np.array([2, 4, 4, 2, 6]))
    numpy_testing_assert_equal_helper(reference[[ri([0, 0, 1, 1]), ri([0, 1, 0, 0])]], np.array([1, 2, 3, 3]))

    rows = ri([[0, 0],
               [1, 2]])
    columns = [0],
    numpy_testing_assert_equal_helper(reference[rows, columns], np.array([[1, 1],
                                                                          [3, 5]]))

    rows = ri([[0, 0],
               [1, 2]])
    columns = ri([1, 0])
    numpy_testing_assert_equal_helper(reference[rows, columns], np.array([[2, 1],
                                                                          [4, 5]]))
    rows = ri([[0, 0],
               [1, 2]])
    columns = ri([[0, 1],
                  [1, 0]])
    numpy_testing_assert_equal_helper(reference[rows, columns], np.array([[1, 2],
                                                                          [4, 5]]))

    # TODO: setitem
    '''
    # setting values
    reference[ri([0]), ri([1])] = -1
    numpy_testing_assert_equal_helper(reference[ri([0]), ri([1])], np.array([-1]))
    reference[ri([0, 1, 2]), ri([0])] = np.array([-1, 2, -4])
    numpy_testing_assert_equal_helper(reference[ri([0, 1, 2]), ri([0])],
                      np.array([-1, 2, -4]))
    reference[rows, columns] = np.array([[4, 6], [2, 3]])
    numpy_testing_assert_equal_helper(reference[rows, columns],
                      np.array([[4, 6], [2, 3]]))
    '''

    # Verify still works with Transposed (i.e. non-contiguous) Tensors

    reference = Tensor([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11]]).T

    # Transposed: [[0, 4, 8],
    #              [1, 5, 9],
    #              [2, 6, 10],
    #              [3, 7, 11]]

    numpy_testing_assert_equal_helper(reference[ri([0, 1, 2]), ri([0])], np.array([0, 1, 2]))
    numpy_testing_assert_equal_helper(reference[ri([0, 1, 2]), ri([1])], np.array([4, 5, 6]))
    numpy_testing_assert_equal_helper(reference[ri([0]), ri([0])], np.array([0]))
    numpy_testing_assert_equal_helper(reference[ri([2]), ri([1])], np.array([6]))
    numpy_testing_assert_equal_helper(reference[[ri([0, 0]), ri([0, 1])]], np.array([0, 4]))
    numpy_testing_assert_equal_helper(reference[[ri([0, 1, 1, 0, 3]), ri([1])]], np.array([4, 5, 5, 4, 7]))
    numpy_testing_assert_equal_helper(reference[[ri([0, 0, 1, 1]), ri([0, 1, 0, 0])]], np.array([0, 4, 1, 1]))

    rows = ri([[0, 0],
               [1, 2]])
    columns = [0],
    numpy_testing_assert_equal_helper(reference[rows, columns], np.array([[0, 0], [1, 2]]))

    rows = ri([[0, 0],
               [1, 2]])
    columns = ri([1, 0])
    numpy_testing_assert_equal_helper(reference[rows, columns], np.array([[4, 0], [5, 2]]))
    rows = ri([[0, 0],
               [1, 3]])
    columns = ri([[0, 1],
                  [1, 2]])
    numpy_testing_assert_equal_helper(reference[rows, columns], np.array([[0, 4], [5, 11]]))

    # TODO: setitem
    '''
    # setting values
    reference[ri([0]), ri([1])] = -1
    numpy_testing_assert_equal_helper(reference[ri([0]), ri([1])],
                      np.array([-1]))
    reference[ri([0, 1, 2]), ri([0])] = np.array([-1, 2, -4])
    numpy_testing_assert_equal_helper(reference[ri([0, 1, 2]), ri([0])],
                      np.array([-1, 2, -4]))
    reference[rows, columns] = np.array([[4, 6], [2, 3]])
    numpy_testing_assert_equal_helper(reference[rows, columns],
                      np.array([[4, 6], [2, 3]]))
    '''

    # stride != 1

    # strided is [[1 3 5 7],
    #             [9 11 13 15]]

    reference = Tensor.arange(0., 24).reshape(3, 8)
    strided = set_(reference, (2,4), (8,2), 1)

    numpy_testing_assert_equal_helper(strided[ri([0, 1]), ri([0])],
                      np.array([1, 9]))
    numpy_testing_assert_equal_helper(strided[ri([0, 1]), ri([1])],
                      np.array([3, 11]))
    numpy_testing_assert_equal_helper(strided[ri([0]), ri([0])],
                      np.array([1]))
    numpy_testing_assert_equal_helper(strided[ri([1]), ri([3])],
                      np.array([15]))
    numpy_testing_assert_equal_helper(strided[[ri([0, 0]), ri([0, 3])]],
                      np.array([1, 7]))
    numpy_testing_assert_equal_helper(strided[[ri([1]), ri([0, 1, 1, 0, 3])]],
                      np.array([9, 11, 11, 9, 15]))
    numpy_testing_assert_equal_helper(strided[[ri([0, 0, 1, 1]), ri([0, 1, 0, 0])]],
                      np.array([1, 3, 9, 9]))

    rows = ri([[0, 0],
                [1, 1]])
    columns = [0],
    numpy_testing_assert_equal_helper(strided[rows, columns],
                      np.array([[1, 1], [9, 9]]))

    rows = ri([[0, 1],
                [1, 0]])
    columns = ri([1, 2])
    numpy_testing_assert_equal_helper(strided[rows, columns],
                      np.array([[3, 13], [11, 5]]))
    rows = ri([[0, 0],
                [1, 1]])
    columns = ri([[0, 1],
                  [1, 2]])
    numpy_testing_assert_equal_helper(strided[rows, columns],
                      np.array([[1, 3], [11, 13]]))


    # setting values

    # strided is [[10, 11],
    #             [17, 18]]

    reference = Tensor.arange(0., 24).reshape(3, 8)
    strided = set_(reference, (2,2), (7,1), 10)

    numpy_testing_assert_equal_helper(strided[ri([0]), ri([1])],
                      np.array([11]))
    # TODO setitem
    '''
    strided[ri([0]), ri([1])] = -1
    numpy_testing_assert_equal_helper(strided[ri([0]), ri([1])],
                      Tensor([-1]))
    '''

    reference = Tensor.arange(0., 24).reshape(3, 8)
    strided = set_(reference, (2,2), (7,1), 10)

    numpy_testing_assert_equal_helper(strided[ri([0, 1]), ri([1, 0])],
                      np.array([11, 17]))
    # TODO setitem
    '''
    strided[ri([0, 1]), ri([1, 0])] = Tensor([-1, 2])
    numpy_testing_assert_equal_helper(strided[ri([0, 1]), ri([1, 0])],
                      Tensor([-1, 2]))
    '''

    reference = Tensor.arange(0., 24).realize().reshape(3, 8)
    strided = set_(reference, (2,2), (7,1), 10)

    rows = ri([[0],
                [1]])
    columns = ri([[0, 1],
                  [0, 1]])
    numpy_testing_assert_equal_helper(strided[rows, columns],
                      np.array([[10, 11], [17, 18]]))
    # TODO setitem
    '''
    strided[rows, columns] = Tensor([[4, 6], [2, 3]])
    numpy_testing_assert_equal_helper(strided[rows, columns],
                      Tensor([[4, 6], [2, 3]]))
    '''

    # Tests using less than the number of dims, and ellipsis

    # reference is 1 2
    #              3 4
    #              5 6
    reference = consec((3, 2))
    numpy_testing_assert_equal_helper(reference[ri([0, 2]),], np.array([[1, 2], [5, 6]]))
    numpy_testing_assert_equal_helper(reference[ri([1]), ...], np.array([[3, 4]]))
    numpy_testing_assert_equal_helper(reference[..., ri([1])], np.array([[2], [4], [6]]))

    # verify too many indices fails
    with self.assertRaises(IndexError): reference[ri([1]), ri([0, 2]), ri([3])]

    # test invalid index fails
    reference = Tensor.empty(10)
    for err_idx in (10, -11):
      with self.assertRaisesRegex(IndexError, r'out of'):
        reference[err_idx]
      # TODO: cannot check for out of bounds with Tensor indexing
      # see test_ops.py: test_slice_fancy_indexing_errors()
      '''
      with self.assertRaisesRegex(IndexError, r'out of'):
        reference[Tensor([err_idx], dtype=dtypes.int64)]
      with self.assertRaisesRegex(IndexError, r'out of'):
        reference[[err_idx]]
      '''

    def tensor_indices_to_np(tensor: Tensor, indices):
      npt = tensor.numpy()
      idxs = tuple(i.numpy().tolist() if isinstance(i, Tensor) and i.dtype is dtypes.int64 else
                  i for i in indices)
      return npt, idxs

    def get_numpy(tensor, indices):
      npt, idxs = tensor_indices_to_np(tensor, indices)
      return Tensor(npt[idxs])

    '''
    def set_numpy(tensor, indices, value):
        if not isinstance(value, int):
            if self.device_type != 'cpu':
                value = value.cpu()
            value = value.numpy()
        npt, idxs = tensor_indices_to_np(tensor, indices)
        npt[idxs] = value
        return npt
    '''
    def set_numpy(tensor:Tensor, indices, value):
      if not isinstance(value, int):
        value = value.numpy()
      npt, idxs = tensor_indices_to_np(tensor, indices)
      npt[idxs] = value
      return npt

    def assert_get_eq(tensor, indexer):
      numpy_testing_assert_equal_helper(tensor[indexer], get_numpy(tensor, indexer))

    '''
    def assert_set_eq(tensor, indexer, val):
        pyt = tensor.clone()
        numt = tensor.clone()
        pyt[indexer] = val
        numt = np.array(set_numpy(numt, indexer, val))
        numpy_testing_assert_equal_helper(pyt, numt)
    '''
    def assert_set_eq(tensor: Tensor, indexer, val):
      pyt = clone(tensor)
      numt = clone(tensor)
      pyt[indexer] = val
      numt = set_numpy(numt, indexer, val)
      numpy_testing_assert_equal_helper(pyt, numt)

    # NOTE: torch initiates the gradients using g0cpu (rand as gradients)
    def assert_backward_eq(tensor: Tensor, indexer):
      cpu = clone(tensor.float())
      cpu.requires_grad = True
      outcpu = cpu[indexer].sum()
      outcpu.backward()
      dev = cpu.detach()
      dev.requires_grad = True
      outdev = dev[indexer].sum()
      outdev.backward()
      numpy_testing_assert_equal_helper(cpu.grad, dev.grad)

    '''
    def get_set_tensor(indexed, indexer):
        set_size = indexed[indexer].size()
        set_count = indexed[indexer].numel()
        set_tensor = torch.randperm(set_count).view(set_size).double().to(device)
        return set_tensor
    '''
    def get_set_tensor(indexed: Tensor, indexer):
      set_size = indexed[indexer].shape
      set_count = indexed[indexer].numel()
      set_tensor = Tensor.randint(set_count, high=set_count).reshape(set_size).cast(dtypes.float64)
      return set_tensor

    # Tensor is  0  1  2  3  4
    #            5  6  7  8  9
    #           10 11 12 13 14
    #           15 16 17 18 19
    reference = Tensor.arange(0., 20).reshape(4, 5)

    indices_to_test = [
      # grab the second, fourth columns
      [slice(None), [1, 3]],

      # first, third rows,
      [[0, 2], slice(None)],

      # weird shape
      [slice(None), [[0, 1],
                      [2, 3]]],
      # negatives
      [[-1], [0]],
      [[0, 2], [-1]],
      [slice(None), [-1]],
    ]

    # only test dupes on gets
    get_indices_to_test = indices_to_test + [[slice(None), [0, 1, 1, 2, 2]]]

    for indexer in get_indices_to_test:
      assert_get_eq(reference, indexer)
      assert_backward_eq(reference, indexer)

    # TODO setitem
    '''
    for indexer in indices_to_test:
      assert_set_eq(reference, indexer, 44)
      assert_set_eq(reference, indexer, get_set_tensor(reference, indexer))
    '''

    reference = Tensor.arange(0., 160).reshape(4, 8, 5)

    indices_to_test = [
      [slice(None), slice(None), [0, 3, 4]],
      [slice(None), [2, 4, 5, 7], slice(None)],
      [[2, 3], slice(None), slice(None)],
      [slice(None), [0, 2, 3], [1, 3, 4]],
      [slice(None), [0], [1, 2, 4]],
      [slice(None), [0, 1, 3], [4]],
      [slice(None), [[0, 1], [1, 0]], [[2, 3]]],
      [slice(None), [[0, 1], [2, 3]], [[0]]],
      [slice(None), [[5, 6]], [[0, 3], [4, 4]]],
      [[0, 2, 3], [1, 3, 4], slice(None)],
      [[0], [1, 2, 4], slice(None)],
      [[0, 1, 3], [4], slice(None)],
      [[[0, 1], [1, 0]], [[2, 1], [3, 5]], slice(None)],
      [[[0, 1], [1, 0]], [[2, 3]], slice(None)],
      [[[0, 1], [2, 3]], [[0]], slice(None)],
      [[[2, 1]], [[0, 3], [4, 4]], slice(None)],
      [[[2]], [[0, 3], [4, 1]], slice(None)],
      # non-contiguous indexing subspace
      [[0, 2, 3], slice(None), [1, 3, 4]],

      # less dim, ellipsis
      [[0, 2], ],
      [[0, 2], slice(None)],
      [[0, 2], Ellipsis],
      [[0, 2], slice(None), Ellipsis],
      [[0, 2], Ellipsis, slice(None)],
      [[0, 2], [1, 3]],
      [[0, 2], [1, 3], Ellipsis],
      [Ellipsis, [1, 3], [2, 3]],
      [Ellipsis, [2, 3, 4]],
      [Ellipsis, slice(None), [2, 3, 4]],
      [slice(None), Ellipsis, [2, 3, 4]],

      # ellipsis counts for nothing
      [Ellipsis, slice(None), slice(None), [0, 3, 4]],
      [slice(None), Ellipsis, slice(None), [0, 3, 4]],
      [slice(None), slice(None), Ellipsis, [0, 3, 4]],
      [slice(None), slice(None), [0, 3, 4], Ellipsis],
      [Ellipsis, [[0, 1], [1, 0]], [[2, 1], [3, 5]], slice(None)],
      [[[0, 1], [1, 0]], [[2, 1], [3, 5]], Ellipsis, slice(None)],
      [[[0, 1], [1, 0]], [[2, 1], [3, 5]], slice(None), Ellipsis],
    ]

    for indexer in indices_to_test:
      assert_get_eq(reference, indexer)
      # TODO setitem
      '''
      assert_set_eq(reference, indexer, 212)
      assert_set_eq(reference, indexer, get_set_tensor(reference, indexer))
      '''
      assert_backward_eq(reference, indexer)

    reference = Tensor.arange(0., 1296).reshape(3, 9, 8, 6)

    indices_to_test = [
      [slice(None), slice(None), slice(None), [0, 3, 4]],
      [slice(None), slice(None), [2, 4, 5, 7], slice(None)],
      [slice(None), [2, 3], slice(None), slice(None)],
      [[1, 2], slice(None), slice(None), slice(None)],
      [slice(None), slice(None), [0, 2, 3], [1, 3, 4]],
      [slice(None), slice(None), [0], [1, 2, 4]],
      [slice(None), slice(None), [0, 1, 3], [4]],
      [slice(None), slice(None), [[0, 1], [1, 0]], [[2, 3]]],
      [slice(None), slice(None), [[0, 1], [2, 3]], [[0]]],
      [slice(None), slice(None), [[5, 6]], [[0, 3], [4, 4]]],
      [slice(None), [0, 2, 3], [1, 3, 4], slice(None)],
      [slice(None), [0], [1, 2, 4], slice(None)],
      [slice(None), [0, 1, 3], [4], slice(None)],
      [slice(None), [[0, 1], [3, 4]], [[2, 3], [0, 1]], slice(None)],
      [slice(None), [[0, 1], [3, 4]], [[2, 3]], slice(None)],
      [slice(None), [[0, 1], [3, 2]], [[0]], slice(None)],
      [slice(None), [[2, 1]], [[0, 3], [6, 4]], slice(None)],
      [slice(None), [[2]], [[0, 3], [4, 2]], slice(None)],
      [[0, 1, 2], [1, 3, 4], slice(None), slice(None)],
      [[0], [1, 2, 4], slice(None), slice(None)],
      [[0, 1, 2], [4], slice(None), slice(None)],
      [[[0, 1], [0, 2]], [[2, 4], [1, 5]], slice(None), slice(None)],
      [[[0, 1], [1, 2]], [[2, 0]], slice(None), slice(None)],
      [[[2, 2]], [[0, 3], [4, 5]], slice(None), slice(None)],
      [[[2]], [[0, 3], [4, 5]], slice(None), slice(None)],
      [slice(None), [3, 4, 6], [0, 2, 3], [1, 3, 4]],
      [slice(None), [2, 3, 4], [1, 3, 4], [4]],
      [slice(None), [0, 1, 3], [4], [1, 3, 4]],
      [slice(None), [6], [0, 2, 3], [1, 3, 4]],
      [slice(None), [2, 3, 5], [3], [4]],
      [slice(None), [0], [4], [1, 3, 4]],
      [slice(None), [6], [0, 2, 3], [1]],
      [slice(None), [[0, 3], [3, 6]], [[0, 1], [1, 3]], [[5, 3], [1, 2]]],
      [[2, 2, 1], [0, 2, 3], [1, 3, 4], slice(None)],
      [[2, 0, 1], [1, 2, 3], [4], slice(None)],
      [[0, 1, 2], [4], [1, 3, 4], slice(None)],
      [[0], [0, 2, 3], [1, 3, 4], slice(None)],
      [[0, 2, 1], [3], [4], slice(None)],
      [[0], [4], [1, 3, 4], slice(None)],
      [[1], [0, 2, 3], [1], slice(None)],
      [[[1, 2], [1, 2]], [[0, 1], [2, 3]], [[2, 3], [3, 5]], slice(None)],

      # less dim, ellipsis
      [Ellipsis, [0, 3, 4]],
      [Ellipsis, slice(None), [0, 3, 4]],
      [Ellipsis, slice(None), slice(None), [0, 3, 4]],
      [slice(None), Ellipsis, [0, 3, 4]],
      [slice(None), slice(None), Ellipsis, [0, 3, 4]],
      [slice(None), [0, 2, 3], [1, 3, 4]],
      [slice(None), [0, 2, 3], [1, 3, 4], Ellipsis],
      [Ellipsis, [0, 2, 3], [1, 3, 4], slice(None)],
      [[0], [1, 2, 4]],
      [[0], [1, 2, 4], slice(None)],
      [[0], [1, 2, 4], Ellipsis],
      [[0], [1, 2, 4], Ellipsis, slice(None)],
      [[1], ],
      [[0, 2, 1], [3], [4]],
      [[0, 2, 1], [3], [4], slice(None)],
      [[0, 2, 1], [3], [4], Ellipsis],
      [Ellipsis, [0, 2, 1], [3], [4]],
    ]

    for indexer in indices_to_test:
      assert_get_eq(reference, indexer)
      # TODO setitem
      '''
      assert_set_eq(reference, indexer, 1333)
      assert_set_eq(reference, indexer, get_set_tensor(reference, indexer))
      '''
    indices_to_test += [
      [slice(None), slice(None), [[0, 1], [1, 0]], [[2, 3], [3, 0]]],
      [slice(None), slice(None), [[2]], [[0, 3], [4, 4]]],
    ]
    for indexer in indices_to_test:
      assert_get_eq(reference, indexer)
      # TODO setitem
      '''
      assert_set_eq(reference, indexer, 1333)
      '''
      assert_backward_eq(reference, indexer)

  # TODO setitem
  '''
  def test_set_item_to_scalar_tensor(self):
    m = random.randint(1, 10)
    n = random.randint(1, 10)
    z = Tensor.randn([m, n])
    a = 1.0
    w = Tensor(a, requires_grad=True)
    z[:, 0] = w
    z.sum().backward()
    numpy_testing_assert_equal_helper(w.grad, m * a)
  '''

  def test_single_int(self):
    v = Tensor.randn(5, 7, 3)
    numpy_testing_assert_equal_helper(v[4].shape, (7, 3))

  def test_multiple_int(self):
    v = Tensor.randn(5, 7, 3)
    numpy_testing_assert_equal_helper(v[4].shape, (7, 3))
    numpy_testing_assert_equal_helper(v[4, :, 1].shape, (7,))

  def test_none(self):
    v = Tensor.randn(5, 7, 3)
    numpy_testing_assert_equal_helper(v[None].shape, (1, 5, 7, 3))
    numpy_testing_assert_equal_helper(v[:, None].shape, (5, 1, 7, 3))
    numpy_testing_assert_equal_helper(v[:, None, None].shape, (5, 1, 1, 7, 3))
    numpy_testing_assert_equal_helper(v[..., None].shape, (5, 7, 3, 1))

  def test_step(self):
    v = Tensor.arange(10)
    numpy_testing_assert_equal_helper(v[::1], v)
    numpy_testing_assert_equal_helper(v[::2], [0, 2, 4, 6, 8])
    numpy_testing_assert_equal_helper(v[::3], [0, 3, 6, 9])
    numpy_testing_assert_equal_helper(v[::11], [0])
    numpy_testing_assert_equal_helper(v[1:6:2], [1, 3, 5])

  # TODO setitem
  '''
  def test_step_assignment(self):
    v = Tensor.zeros(4, 4)
    v[0, 1::2] = Tensor([3., 4.])
    numpy_testing_assert_equal_helper(v[0].numpy().tolist(), [0, 3, 0, 4])
    numpy_testing_assert_equal_helper(v[1:].sum(), 0)
  '''

  # TODO bool indexing
  '''
  def test_bool_indices(self):
    v = Tensor.randn(5, 7, 3)
    boolIndices = Tensor([True, False, True, True, False], dtype=dtypes.bool)
    numpy_testing_assert_equal_helper(v[boolIndices].shape, (3, 7, 3))
    numpy_testing_assert_equal_helper(v[boolIndices], Tensor.stack([v[0], v[2], v[3]]))

    v = Tensor([True, False, True], dtype=dtypes.bool)
    boolIndices = Tensor([True, False, False], dtype=dtypes.bool)
    uint8Indices = Tensor([1, 0, 0], dtype=dtypes.uint8)
    with warnings.catch_warnings(record=True) as w:
      numpy_testing_assert_equal_helper(v[boolIndices].shape, v[uint8Indices].shape)
      numpy_testing_assert_equal_helper(v[boolIndices], v[uint8Indices])
      numpy_testing_assert_equal_helper(v[boolIndices], tensor([True], dtype=torch.bool))
      numpy_testing_assert_equal_helper(len(w), 2)
  '''

  # TODO setindex
  # TODO bool indexing
  '''
  def test_bool_indices_accumulate(self):
    mask = Tensor.zeros(size=(10, ), dtype=dtypes.bool)
    y = Tensor.ones(size=(10, 10))
    y.index_put_((mask, ), y[mask], accumulate=True)
    numpy_testing_assert_equal_helper(y, Tensor.ones(size=(10, 10)))
  '''

  # TODO bool indexing
  '''
  def test_multiple_bool_indices(self):
    v = Tensor.randn(5, 7, 3)
    # note: these broadcast together and are transposed to the first dim
    mask1 = Tensor([1, 0, 1, 1, 0], dtype=dtypes.bool)
    mask2 = Tensor([1, 1, 1], dtype=dtypes.bool)
    numpy_testing_assert_equal_helper(v[mask1, :, mask2].shape, (3, 7))
  '''

  # TODO bool indexing
  '''
  def test_byte_mask(self):
    v = Tensor.randn(5, 7, 3)
    mask = Tensor([1, 0, 1, 1, 0], dtype=dtypes.uint8)
    with warnings.catch_warnings(record=True) as w:
      numpy_testing_assert_equal_helper(v[mask].shape, (3, 7, 3))
      numpy_testing_assert_equal_helper(v[mask], Tensor.stack([v[0], v[2], v[3]]))
      numpy_testing_assert_equal_helper(len(w), 2)

    v = Tensor([1.])
    numpy_testing_assert_equal_helper(v[v == 0], Tensor([]))
  '''

  # TODO setitem
  # TODO bool indexing
  '''
  def test_byte_mask_accumulate(self):
    mask = Tensor.zeros(size=(10, ), dtype=dtypes.uint8)
    y = Tensor.ones(size=(10, 10))
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      y.index_put_((mask, ), y[mask], accumulate=True)
      numpy_testing_assert_equal_helper(y, Tensor.ones(size=(10, 10)))
      numpy_testing_assert_equal_helper(len(w), 2)
  '''

  # TODO setitem
  '''
  def test_index_put_accumulate_large_tensor(self):
    # This test is for tensors with number of elements >= INT_MAX (2^31 - 1).
    N = (1 << 31) + 5
    dt = dtypes.int8
    a = Tensor.ones(N, dtype=dt)
    indices = Tensor([-2, 0, -2, -1, 0, -1, 1], dtype=dtypes.int64)
    values = Tensor([6, 5, 6, 6, 5, 7, 11], dtype=dt)

    a.index_put_((indices, ), values, accumulate=True)

    numpy_testing_assert_equal_helper(a[0], 11)
    numpy_testing_assert_equal_helper(a[1], 12)
    numpy_testing_assert_equal_helper(a[2], 1)
    numpy_testing_assert_equal_helper(a[-3], 1)
    numpy_testing_assert_equal_helper(a[-2], 13)
    numpy_testing_assert_equal_helper(a[-1], 14)

    a = Tensor.ones((2, N), dtype=dt)
    indices0 = np.array([0, -1, 0, 1], dtype=dtypes.int64)
    indices1 = np.array([-2, -1, 0, 1], dtype=dtypes.int64)
    values = np.array([12, 13, 10, 11], dtype=dt)

    a.index_put_((indices0, indices1), values, accumulate=True)

    numpy_testing_assert_equal_helper(a[0, 0], 11)
    numpy_testing_assert_equal_helper(a[0, 1], 1)
    numpy_testing_assert_equal_helper(a[1, 0], 1)
    numpy_testing_assert_equal_helper(a[1, 1], 12)
    numpy_testing_assert_equal_helper(a[:, 2], Tensor.ones(2, dtype=dtypes.int8))
    numpy_testing_assert_equal_helper(a[:, -3], Tensor.ones(2, dtype=dtypes.int8))
    numpy_testing_assert_equal_helper(a[0, -2], 13)
    numpy_testing_assert_equal_helper(a[1, -2], 1)
    numpy_testing_assert_equal_helper(a[-1, -1], 14)
    numpy_testing_assert_equal_helper(a[0, -1], 1)
  '''

  @unittest.skip("pytorch device specific test")
  def test_index_put_accumulate_expanded_values(self, device):
    # checks the issue with cuda: https://github.com/pytorch/pytorch/issues/39227
    # and verifies consistency with CPU result
    t = Tensor.zeros((5, 2))
    t_dev = t.to(device)
    indices = [
      Tensor([0, 1, 2, 3]),
      Tensor([1, ]),
    ]
    indices_dev = [i.to(device) for i in indices]
    values0d = Tensor(1.0)
    values1d = Tensor([1.0, ])

    out_cuda = t_dev.index_put_(indices_dev, values0d.to(device), accumulate=True)
    out_cpu = t.index_put_(indices, values0d, accumulate=True)
    numpy_testing_assert_equal_helper(out_cuda.cpu(), out_cpu)

    out_cuda = t_dev.index_put_(indices_dev, values1d.to(device), accumulate=True)
    out_cpu = t.index_put_(indices, values1d, accumulate=True)
    numpy_testing_assert_equal_helper(out_cuda.cpu(), out_cpu)

    t = Tensor.zeros(4, 3, 2)
    t_dev = t.to(device)

    indices = [
      Tensor([0, ]),
      Tensor.arange(3)[:, None],
      Tensor.arange(2)[None, :],
    ]
    indices_dev = [i.to(device) for i in indices]
    values1d = np.array([-1.0, -2.0])
    values2d = np.array([[-1.0, -2.0], ])

    out_cuda = t_dev.index_put_(indices_dev, values1d.to(device), accumulate=True)
    out_cpu = t.index_put_(indices, values1d, accumulate=True)
    numpy_testing_assert_equal_helper(out_cuda.cpu(), out_cpu)

    out_cuda = t_dev.index_put_(indices_dev, values2d.to(device), accumulate=True)
    out_cpu = t.index_put_(indices, values2d, accumulate=True)
    numpy_testing_assert_equal_helper(out_cuda.cpu(), out_cpu)

  # TODO setitem
  @unittest.skip("pytorch device specific test")
  def test_index_put_accumulate_non_contiguous(self, device):
    t = Tensor.zeros((5, 2, 2))
    t_dev = t.to(device)
    t1 = t_dev[:, 0, :]
    t2 = t[:, 0, :]
    self.assertTrue(not t1.lazydata.st.contiguous)
    self.assertTrue(not t2.lazydata.st.contiguous)

    indices = [Tensor([0, 1]), ]
    indices_dev = [i.to(device) for i in indices]
    value = Tensor.randn(2, 2)
    out_cuda = t1.index_put_(indices_dev, value.to(device), accumulate=True)
    out_cpu = t2.index_put_(indices, value, accumulate=True)
    self.assertTrue(not t1.lazydata.st.contiguous)
    self.assertTrue(not t2.lazydata.st.contiguous)

    numpy_testing_assert_equal_helper(out_cuda.cpu(), out_cpu)

  # TODO setitem
  '''
  def test_index_put_accumulate_with_optional_tensors(self):
    # TODO: replace with a better solution.
    # Currently, here using torchscript to put None into indices.
    # on C++ it gives indices as a list of 2 optional tensors: first is null and
    # the second is a valid tensor.
    @TinyJit
    def func(x, i, v):
      idx = [None, i]
      x.index_put_(idx, v, accumulate=True)
      return x

    n = 4
    t = Tensor.arange(n * 2, dtype=dtypes.float32).reshape(n, 2)
    t_dev = t.to(device)
    indices = Tensor([1, 0])
    indices_dev = indices.to(device)
    value0d = Tensor(10.0)
    value1d = Tensor([1.0, 2.0])

    out_cuda = func(t_dev, indices_dev, value0d.cuda())
    out_cpu = func(t, indices, value0d)
    numpy_testing_assert_equal_helper(out_cuda.cpu(), out_cpu)

    out_cuda = func(t_dev, indices_dev, value1d.cuda())
    out_cpu = func(t, indices, value1d)
    numpy_testing_assert_equal_helper(out_cuda.cpu(), out_cpu)
  '''

  # TODO setindex
  '''
  def test_index_put_accumulate_duplicate_indices(self):
    for i in range(1, 512):
      # generate indices by random walk, this will create indices with
      # lots of duplicates interleaved with each other
      delta = Tensor.uniform(low=-1, high=1, dtype=dtypes.double)
      indices = delta.cumsum(0).cast(dtypes.int64)

      # input = torch.randn(indices.abs().max() + 1)
      input = Tensor.randn(indices.abs().max().item() + 1)
      # values = torch.randn(indices.size(0))
      values = Tensor.randn(indices.shape(0))
      output = input.index_put((indices,), values, accumulate=True)

      input_list = input.numpy().tolist()
      indices_list = indices.numpy().tolist()
      values_list = values.numpy().tolist()
      for i, v in zip(indices_list, values_list):
        input_list[i] += v

      numpy_testing_assert_equal_helper(output, input_list)
  '''

  def test_index_ind_dtype(self):
    x = Tensor.randn(4, 4)
    # ind_long = torch.randint(4, (4,), dtype=torch.long)
    # TODO should we spend an extra line to allow for randint other dtypes?
    # copied from randint
    ind_long = (Tensor.rand((4,),)*(4-0)+0).cast(dtypes.int64)
    # ind_int = ind_long.int()
    ind_int = (ind_long).cast(dtypes.int32)
    ref = x[ind_long, ind_long]
    res = x[ind_int, ind_int]
    numpy_testing_assert_equal_helper(ref, res)
    ref = x[ind_long, :]
    res = x[ind_int, :]
    numpy_testing_assert_equal_helper(ref, res)
    ref = x[:, ind_long]
    res = x[:, ind_int]
    numpy_testing_assert_equal_helper(ref, res)
    # no repeating indices for index_put
    # TODO setitem
    '''
    src = Tensor.randn(4)
    ind_long = Tensor.arange(4, dtype=dtypes.int64)
    ind_int = ind_long.cast(dtypes.int32)
    for accum in (True, False):
      inp_ref = clone(x)
      inp_res = clone(x)
      torch.index_put_(inp_ref, (ind_long, ind_long), src, accum)
      torch.index_put_(inp_res, (ind_int, ind_int), src, accum)
      numpy_testing_assert_equal_helper(inp_ref, inp_res)
    '''

  # TODO setitem
  '''
  def test_index_put_accumulate_empty(self):
    # Regression test for https://github.com/pytorch/pytorch/issues/94667
    input = Tensor.rand([], dtype=dtypes.float32)
    with self.assertRaises(RuntimeError):
      input.index_put([], np.array([1.0]), True)
  '''

  # TODO bool indexing
  '''
  def test_multiple_byte_mask(self):
    v = Tensor.randn(5, 7, 3)
    # note: these broadcast together and are transposed to the first dim
    mask1 = Tensor([1, 0, 1, 1, 0], dtype=dtypes.uint8)
    mask2 = Tensor([1, 1, 1], dtype=dtypes.uint8)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      numpy_testing_assert_equal_helper(v[mask1, :, mask2].shape, (3, 7))
      numpy_testing_assert_equal_helper(len(w), 2)
  '''

  # TODO bool indexing
  '''
  def test_byte_mask2d(self):
    v = Tensor.randn(5, 7, 3)
    c = Tensor.randn(5, 7)
    num_ones = (c > 0).sum()
    r = v[c > 0]
    numpy_testing_assert_equal_helper(r.shape, (num_ones, 3))
  '''

  # TODO setindex
  # TODO bool indexing
  '''
  def test_jit_indexing(self):
    def fn1(x):
      x[x < 50] = 1.0
      return x

    def fn2(x):
      x[0:50] = 1.0
      return x

    scripted_fn1 = TinyJit(fn1)
    scripted_fn2 = TinyJit(fn2)
    data = Tensor.arange(100, dtype=dtypes.float)
    out = scripted_fn1(clone(data))
    ref = Tensor(np.concatenate((np.ones(50), np.arange(50, 100))), dtype=dtypes.float)
    numpy_testing_assert_equal_helper(out, ref)
    out = scripted_fn2(clone(data))
    numpy_testing_assert_equal_helper(out, ref)
  '''

  def test_int_indices(self):
      v = Tensor.randn(5, 7, 3)
      numpy_testing_assert_equal_helper(v[[0, 4, 2]].shape, (3, 7, 3))
      numpy_testing_assert_equal_helper(v[:, [0, 4, 2]].shape, (5, 3, 3))
      numpy_testing_assert_equal_helper(v[:, [[0, 1], [4, 3]]].shape, (5, 2, 2, 3))

  # TODO setindex
  '''
  def test_index_put_src_datatype(self, dtype):
    src = Tensor.ones(3, 2, 4, dtype=dtype)
    vals = Tensor.ones(3, 2, 4, dtype=dtype)
    indices = (np.array([0, 2, 1]),)
    res = src.index_put_(indices, vals, accumulate=True)
    numpy_testing_assert_equal_helper(res.shape, src.shape)
  '''

  def test_index_src_datatype(self):
    src = Tensor.ones(3, 2, 4)
    # test index
    res = src[[0, 2, 1], :, :]
    numpy_testing_assert_equal_helper(res.shape, src.shape)
    # test index_put, no accum
    # TODO setindex
    '''
    src[[0, 2, 1], :, :] = res
    numpy_testing_assert_equal_helper(res.shape, src.shape)
    '''

  def test_int_indices2d(self):
    # From the NumPy indexing example
    x = Tensor.arange(0, 12).reshape(4, 3)
    rows = Tensor([[0, 0], [3, 3]])
    columns = Tensor([[0, 2], [0, 2]])
    numpy_testing_assert_equal_helper(x[rows, columns].numpy().tolist(), [[0, 2], [9, 11]])

  def test_int_indices_broadcast(self):
    # From the NumPy indexing example
    x = Tensor.arange(0, 12).reshape(4, 3)
    rows = Tensor([0, 3])
    columns = Tensor([0, 2])
    result = x[rows[:, None], columns]
    numpy_testing_assert_equal_helper(result.numpy().tolist(), [[0, 2], [9, 11]])

  # TODO setitem
  # TODO empty Tensor fancy index
  '''
  def test_empty_index(self):
    x = Tensor.arange(0, 12).reshape(4, 3)
    idx = Tensor([], dtype=dtypes.int64)
    numpy_testing_assert_equal_helper(x[idx].numel(), 0)

    # empty assignment should have no effect but not throw an exception
    y = clone(x)
    y[idx] = -1
    numpy_testing_assert_equal_helper(x, y)

    mask = Tensor.zeros(4, 3).cast(dtypes.bool)
    y[mask] = -1
    numpy_testing_assert_equal_helper(x, y)
  '''

  # TODO empty Tensor fancy index
  '''
  def test_empty_ndim_index(self):
    x = Tensor.randn(5)
    numpy_testing_assert_equal_helper(Tensor.empty(0, 2), x[Tensor.empty(0, 2, dtype=dtypes.int64)])

    x = Tensor.randn(2, 3, 4, 5)
    numpy_testing_assert_equal_helper(Tensor.empty(2, 0, 6, 4, 5),
                      x[:, Tensor.empty(0, 6, dtype=dtypes.int64)])

    x = Tensor.empty(10, 0)
    numpy_testing_assert_equal_helper(x[[1, 2]].shape, (2, 0))
    numpy_testing_assert_equal_helper(x[[], []].shape, (0,))
    with self.assertRaisesRegex(IndexError, 'for dimension with size 0'):
        x[:, [0, 1]]
  '''

  # TODO: should this fail?
  # def test_empty_ndim_index_bool(self):
  #   x = Tensor.randn(5)
  #   self.assertRaises(IndexError, lambda: x[Tensor.empty(0, 2, dtype=dtypes.uint8)])

  def test_empty_slice(self):
    x = Tensor.randn(2, 3, 4, 5)
    y = x[:, :, :, 1]
    z = y[:, 1:1, :]
    numpy_testing_assert_equal_helper((2, 0, 4), z.shape)
    # this isn't technically necessary, but matches NumPy stride calculations.
    # TODO not too sure about this
    # numpy_testing_assert_equal_helper((60, 20, 5), z.lazydata.st.real_strides())
    # TODO: should this be contiguous?
    # self.assertTrue(z.lazydata.st.contiguous)

  # TODO bool indexing
  # TODO data_ptr()
  '''
  def test_index_getitem_copy_bools_slices(self):
    true = Tensor(1, dtype=dtypes.uint8)
    false = Tensor(0, dtype=dtypes.uint8)

    tensors = [Tensor.randn(2, 3), Tensor(3.)]

    for a in tensors:
      self.assertNotEqual(a.data_ptr(), a[True].data_ptr())
      numpy_testing_assert_equal_helper(Tensor.empty(0, *a.shape), a[False])
      self.assertNotEqual(a.data_ptr(), a[true].data_ptr())
      numpy_testing_assert_equal_helper(Tensor.empty(0, *a.shape), a[false])
      numpy_testing_assert_equal_helper(a.data_ptr(), a[None].data_ptr())
      numpy_testing_assert_equal_helper(a.data_ptr(), a[...].data_ptr())
  '''

  # TODO setitem
  # TODO bool indexing
  '''
  def test_index_setitem_bools_slices(self):
    true = Tensor(1, dtype=dtypes.uint8)
    false = Tensor(0, dtype=dtypes.uint8)

    tensors = [Tensor.randn(2, 3), Tensor(3)]

    for a in tensors:
      # prefix with a 1,1, to ensure we are compatible with numpy which cuts off prefix 1s
      # (some of these ops already prefix a 1 to the size)
      neg_ones = Tensor.ones_like(a) * -1
      neg_ones_expanded = neg_ones.unsqueeze(0).unsqueeze(0)
      a[True] = neg_ones_expanded
      numpy_testing_assert_equal_helper(a, neg_ones)
      a[False] = 5
      numpy_testing_assert_equal_helper(a, neg_ones)
      a[true] = neg_ones_expanded * 2
      numpy_testing_assert_equal_helper(a, neg_ones * 2)
      a[false] = 5
      numpy_testing_assert_equal_helper(a, neg_ones * 2)
      a[None] = neg_ones_expanded * 3
      numpy_testing_assert_equal_helper(a, neg_ones * 3)
      a[...] = neg_ones_expanded * 4
      numpy_testing_assert_equal_helper(a, neg_ones * 4)
      if a.dim() == 0:
          with self.assertRaises(IndexError):
              a[:] = neg_ones_expanded * 5
  '''

  # TODO bool indexing
  '''
  def test_index_scalar_with_bool_mask(self):
    a = Tensor(1)
    uintMask = Tensor(True, dtype=dtypes.uint8)
    boolMask = Tensor(True, dtype=dtypes.bool)
    numpy_testing_assert_equal_helper(a[uintMask], a[boolMask])
    numpy_testing_assert_equal_helper(a[uintMask].dtype, a[boolMask].dtype)

    a = Tensor(True, dtype=dtypes.bool)
    numpy_testing_assert_equal_helper(a[uintMask], a[boolMask])
    numpy_testing_assert_equal_helper(a[uintMask].dtype, a[boolMask].dtype)
  '''

  # TODO setitem
  '''
  def test_setitem_expansion_error(self):
    true = Tensor(True)
    a = Tensor.randn(2, 3)
    # check prefix with  non-1s doesn't work
    # a_expanded = a.expand(torch.Size([5, 1]) + a.size())
    a_expanded = a.expand((5, 1) + a.shape)
    # NumPy: ValueError
    with self.assertRaises(RuntimeError):
      a[True] = a_expanded
    with self.assertRaises(RuntimeError):
      a[true] = a_expanded
  '''

  def test_getitem_scalars_simple(self):
    src = Tensor([[[1.,2.],[3.,4.]], [[1,1],[1,1]]])
    a = src[0].mul(src[1])
    self.assertEqual(a[0,1].item(), 2)

  def test_getitem_scalars(self):
    zero = Tensor(0, dtype=dtypes.int64)
    one = Tensor(1, dtype=dtypes.int64)

    # non-scalar indexed with scalars
    a = Tensor.randn(2, 3)
    numpy_testing_assert_equal_helper(a[0], a[zero])
    numpy_testing_assert_equal_helper(a[0][1], a[zero][one])
    numpy_testing_assert_equal_helper(a[0, 1], a[zero, one])
    numpy_testing_assert_equal_helper(a[0, one], a[zero, 1])

    # indexing by a scalar should slice (not copy)
    # TODO data ptr
    '''
    numpy_testing_assert_equal_helper(a[0, 1].data_ptr(), a[zero, one].data_ptr())
    numpy_testing_assert_equal_helper(a[1].data_ptr(), a[one.cast(dtypes.int32)].data_ptr())
    numpy_testing_assert_equal_helper(a[1].data_ptr(), a[one.cast(dtypes.int16)].data_ptr())
    '''

    # scalar indexed with scalar
    r = Tensor.randn()
    with self.assertRaises(IndexError):
      r[:]
    with self.assertRaises(IndexError):
      r[zero]
    numpy_testing_assert_equal_helper(r, r[...])

  # TODO setitem
  '''
  def test_setitem_scalars(self):
    zero = Tensor(0, dtype=dtypes.int64)

    # non-scalar indexed with scalars
    a = Tensor.randn(2, 3)
    a_set_with_number = clone(a)
    a_set_with_scalar = clone(a)
    b = Tensor.randn(3)

    a_set_with_number[0] = b
    a_set_with_scalar[zero] = b
    numpy_testing_assert_equal_helper(a_set_with_number, a_set_with_scalar)
    a[1, zero] = 7.7
    numpy_testing_assert_equal_helper(7.7, a[1, 0])

    # scalar indexed with scalars
    r = Tensor.randn()
    with self.assertRaises(IndexError):
      r[:] = 8.8
    with self.assertRaises(IndexError):
      r[zero] = 8.8
    r[...] = 9.9
    numpy_testing_assert_equal_helper(9.9, r)
    '''

  def test_basic_advanced_combined(self):
    # From the NumPy indexing example
    x = Tensor.arange(0, 12).reshape(4, 3)
    numpy_testing_assert_equal_helper(x[1:2, 1:3], x[1:2, [1, 2]])
    numpy_testing_assert_equal_helper(x[1:2, 1:3].numpy().tolist(), [[4, 5]])

    # Check that it is a copy
    unmodified = clone(x)
    # x[1:2, [1, 2]].zero_()
    x[1:2, [1, 2]].zeros_like()
    numpy_testing_assert_equal_helper(x, unmodified)

    # But assignment should modify the original
    # TODO setitem
    '''
    unmodified = clone(x)
    x[1:2, [1, 2]] = 0
    self.assertNotEqual(x, unmodified)
    '''

  # TODO setitem
  '''
  def test_int_assignment(self):
    x = Tensor.arange(0, 4).reshape(2, 2)
    x[1] = 5
    numpy_testing_assert_equal_helper(x.numpy().tolist(), [[0, 1], [5, 5]])

    x = Tensor.arange(0, 4).reshape(2, 2)
    x[1] = Tensor.arange(5, 7)
    numpy_testing_assert_equal_helper(x.numpy().tolist(), [[0, 1], [5, 6]])
  '''

  # TODO setitem
  '''
  def test_byte_tensor_assignment(self):
    x = Tensor.arange(0., 16).reshape(4, 4)
    b = Tensor([True, False, True, False], dtype=dtypes.uint8)
    value = Tensor([3., 4., 5., 6.])

    with warnings.catch_warnings(record=True) as w:
      x[b] = value
      numpy_testing_assert_equal_helper(len(w), 1)

    numpy_testing_assert_equal_helper(x[0], value)
    numpy_testing_assert_equal_helper(x[1], Tensor.arange(4., 8))
    numpy_testing_assert_equal_helper(x[2], value)
    numpy_testing_assert_equal_helper(x[3], Tensor.arange(12., 16))
  '''

  # TODO tensor unpacking
  '''
  def test_variable_slicing(self):
    x = Tensor.arange(0, 16).reshape(4, 4)
    indices = Tensor([0, 1], dtype=dtypes.int32)
    i, j = indices
    numpy_testing_assert_equal_helper(x[i:j], x[0:1])
  '''

  def test_ellipsis_tensor(self):
    x = Tensor.arange(0, 9).reshape(3, 3)
    idx = Tensor([0, 2])
    numpy_testing_assert_equal_helper(x[..., idx].numpy().tolist(), [[0, 2],
                                                                     [3, 5],
                                                                     [6, 8]])
    numpy_testing_assert_equal_helper(x[idx, ...].numpy().tolist(), [[0, 1, 2],
                                                                     [6, 7, 8]])

  # TODO unravel_index
  # def test_unravel_index_errors(self):
  #   with self.assertRaisesRegex(TypeError, r"expected 'indices' to be integer"):
  #     unravel_index(
  #       Tensor(0.5),
  #       (2, 2))

  #   with self.assertRaisesRegex(TypeError, r"expected 'indices' to be integer"):
  #     unravel_index(
  #       Tensor([]),
  #       (10, 3, 5))

  #   with self.assertRaisesRegex(TypeError, r"expected 'shape' to be int or sequence"):
  #     unravel_index(
  #       Tensor([1], dtype=dtypes.int64),
  #       Tensor([1, 2, 3]))

  #   with self.assertRaisesRegex(TypeError, r"expected 'shape' sequence to only contain ints"):
  #     unravel_index(
  #       Tensor([1], dtype=dtypes.int64),
  #       (1, 2, 2.0))

  #   with self.assertRaisesRegex(ValueError, r"'shape' cannot have negative values, but got \(2, -3\)"):
  #     unravel_index(
  #       Tensor(0),
  #       (2, -3))

  def test_invalid_index(self):
    x = Tensor.arange(0, 16).reshape(4, 4)
    self.assertRaisesRegex(TypeError, 'slice indices', lambda: x["0":"1"])

  def test_out_of_bound_index(self):
    x = Tensor.arange(0, 100).reshape(2, 5, 10)
    self.assertRaises(IndexError, lambda: x[0, 5])
    self.assertRaises(IndexError, lambda: x[4, 5])
    self.assertRaises(IndexError, lambda: x[0, 1, 15])
    self.assertRaises(IndexError, lambda: x[:, :, 12])

  # TODO .item() bug
  '''
  def test_zero_dim_index(self):
    x = Tensor(10)
    numpy_testing_assert_equal_helper(x, x.item())

    def runner():
      print(x[0])
      return x[0]

    self.assertRaisesRegex(IndexError, 'invalid index', runner)
  '''

  # TODO not too sure
  '''
  def test_invalid_device(self):
    idx = Tensor([0, 1])
    b = Tensor.zeros(5)
    c = Tensor([1., 2.], device="CPU")

    for accumulate in [True, False]:
      self.assertRaises(RuntimeError, lambda: torch.index_put_(b, (idx,), c, accumulate=accumulate))
  '''

  # TODO setitem
  '''
  def test_cpu_indices(self):
    idx = Tensor([0, 1])
    b = Tensor.zeros(2)
    x = Tensor.ones(10)
    x[idx] = b  # index_put_
    ref = Tensor.ones(10)
    ref[:2] = 0
    numpy_testing_assert_equal_helper(x, ref)
    out = x[idx]  # index
    numpy_testing_assert_equal_helper(out, Tensor.zeros(2))
  '''

  def test_take_along_dim(self):
    '''
    def _test_against_numpy(t, indices, dim):
        actual = torch.take_along_dim(t, indices, dim=dim)
        t_np = t.cpu().numpy()
        indices_np = indices.cpu().numpy()
        expected = np.take_along_axis(t_np, indices_np, axis=dim)
        numpy_testing_assert_equal_helper(actual, expected)
    '''
    def _test_against_numpy(t: Tensor, indices: Tensor, dim):
      actual = t.gather(indices, dim=dim)
      t_np = t.numpy()
      indices_np = indices.numpy()
      expected = np.take_along_axis(t_np, indices_np, axis=dim)
      numpy_testing_assert_equal_helper(actual, expected)

      # TODO argsort
      '''
      for shape in [(3, 2), (2, 3, 5), (2, 4, 0), (2, 3, 1, 4)]:
        for noncontiguous in [True, False]:
          for dtype in (dtypes.float32, dtypes.int64):
            t = make_tensor(shape, dtype=dtype, noncontiguous=noncontiguous)
            for dim in list(range(t.ndim)) + [None]:
              if dim is None:
                indices = argsort(t.reshape(-1))
              else:
                indices = argsort(t, dim=dim)

          _test_against_numpy(t, indices, dim)
      '''

      # test broadcasting
      t = Tensor.ones((3, 4, 1))
      indices = Tensor.ones((1, 2, 5), dtype=dtypes.int64)

      _test_against_numpy(t, indices, 1)

      # test empty indices
      t = Tensor.ones((3, 4, 5))
      indices = Tensor.ones((3, 0, 5), dtype=dtypes.int64)

      _test_against_numpy(t, indices, 1)

  # TODO argsort
  '''
  def test_take_along_dim_invalid(self):
    for dtype in (dtypes.int64, dtypes.float32):
      shape = (2, 3, 1, 4)
      dim = 0
      t = make_tensor(shape, dtype=dtype)
      indices = argsort(t, dim=dim)

      # dim of `t` and `indices` does not match
      with self.assertRaisesRegex(RuntimeError, "input and indices should have the same number of dimensions"):
        t.gather(indices[0], dim=0)

      # invalid `indices` dtype
      with self.assertRaisesRegex(RuntimeError, r"dtype of indices should be Long"):
        t.gather(indices.cast(dtypes.bool), dim=0)

      with self.assertRaisesRegex(RuntimeError, r"dtype of indices should be Long"):
        t.gather(indices.cast(dtypes.float32), dim=0)

      with self.assertRaisesRegex(RuntimeError, r"dtype of indices should be Long"):
        t.gather(indices.cast(dtypes.int32), dim=0)

      # invalid axis
      with self.assertRaisesRegex(IndexError, "Dimension out of range"):
        t.gather(indices, dim=-7)

      with self.assertRaisesRegex(IndexError, "Dimension out of range"):
        t.gather(t, indices, dim=7)
  '''

  # TODO Exception for different devices
  # TODO argsort
  '''
  def test_gather_take_along_dim_cross_device(self, dtype=dtypes.float32):
    shape = (2, 3, 1, 4)
    dim = 0
    t = make_tensor(shape, dtype=dtype)
    indices = argsort(t, dim=dim)

    with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
      torch.gather(t, 0, indices.cpu())

    with self.assertRaisesRegex(RuntimeError, r"Expected tensor to have .* but got tensor with .* torch.take_along_dim()"):
      torch.take_along_dim(t, indices.cpu(), dim=0)

    with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
      torch.gather(t.cpu(), 0, indices)

    with self.assertRaisesRegex(RuntimeError, r"Expected tensor to have .* but got tensor with .* torch.take_along_dim()"):
      torch.take_along_dim(t.cpu(), indices, dim=0)
  '''

  # TODO another torch specific test...
  '''
  def test_cuda_broadcast_index_use_deterministic_algorithms(self):
    with DeterministicGuard(True):
      idx1 = np.array([0])
      idx2 = np.array([2, 6])
      idx3 = np.array([1, 5, 7])

      tensor_a = torch.rand(13, 11, 12, 13, 12).cpu()
      tensor_b = tensor_a.to(device=device)
      tensor_a[idx1] = 1.0
      tensor_a[idx1, :, idx2, idx2, :] = 2.0
      tensor_a[:, idx1, idx3, :, idx3] = 3.0
      tensor_b[idx1] = 1.0
      tensor_b[idx1, :, idx2, idx2, :] = 2.0
      tensor_b[:, idx1, idx3, :, idx3] = 3.0
      numpy_testing_assert_equal_helper(tensor_a, tensor_b.cpu())

      tensor_a = torch.rand(10, 11).cpu()
      tensor_b = tensor_a.to(device=device)
      tensor_a[idx3] = 1.0
      tensor_a[idx2, :] = 2.0
      tensor_a[:, idx2] = 3.0
      tensor_a[:, idx1] = 4.0
      tensor_b[idx3] = 1.0
      tensor_b[idx2, :] = 2.0
      tensor_b[:, idx2] = 3.0
      tensor_b[:, idx1] = 4.0
      numpy_testing_assert_equal_helper(tensor_a, tensor_b.cpu())

      tensor_a = torch.rand(10, 10).cpu()
      tensor_b = tensor_a.to(device=device)
      tensor_a[[8]] = 1.0
      tensor_b[[8]] = 1.0
      numpy_testing_assert_equal_helper(tensor_a, tensor_b.cpu())

      tensor_a = torch.rand(10).cpu()
      tensor_b = tensor_a.to(device=device)
      tensor_a[6] = 1.0
      tensor_b[6] = 1.0
      numpy_testing_assert_equal_helper(tensor_a, tensor_b.cpu())
  '''

class TestNumpy(unittest.TestCase):
  def test_index_no_floats(self):
    a = Tensor([[[5.]]])

    self.assertRaises(IndexError, lambda: a[0.0])
    self.assertRaises(IndexError, lambda: a[0, 0.0])
    self.assertRaises(IndexError, lambda: a[0.0, 0])
    self.assertRaises(IndexError, lambda: a[0.0, :])
    self.assertRaises(IndexError, lambda: a[:, 0.0])
    self.assertRaises(IndexError, lambda: a[:, 0.0, :])
    self.assertRaises(IndexError, lambda: a[0.0, :, :])
    self.assertRaises(IndexError, lambda: a[0, 0, 0.0])
    self.assertRaises(IndexError, lambda: a[0.0, 0, 0])
    self.assertRaises(IndexError, lambda: a[0, 0.0, 0])
    self.assertRaises(IndexError, lambda: a[-1.4])
    self.assertRaises(IndexError, lambda: a[0, -1.4])
    self.assertRaises(IndexError, lambda: a[-1.4, 0])
    self.assertRaises(IndexError, lambda: a[-1.4, :])
    self.assertRaises(IndexError, lambda: a[:, -1.4])
    self.assertRaises(IndexError, lambda: a[:, -1.4, :])
    self.assertRaises(IndexError, lambda: a[-1.4, :, :])
    self.assertRaises(IndexError, lambda: a[0, 0, -1.4])
    self.assertRaises(IndexError, lambda: a[-1.4, 0, 0])
    self.assertRaises(IndexError, lambda: a[0, -1.4, 0])
    self.assertRaises(IndexError, lambda: a[0.0:, 0.0])
    self.assertRaises(IndexError, lambda: a[0.0:, 0.0,:])

  def test_none_index(self):
    # `None` index adds newaxis
    a = Tensor([1, 2, 3])
    numpy_testing_assert_equal_helper(a[None].ndim, a.ndim+1)

  def test_empty_tuple_index(self):
    # Empty tuple index creates a view
    a = Tensor([1, 2, 3])
    numpy_testing_assert_equal_helper(a[()], a)
    # TODO data_ptr
    '''
    numpy_testing_assert_equal_helper(a[()].data_ptr(), a.data_ptr())
    '''

  # TODO empty fancy index
  '''
  def test_empty_fancy_index(self):
    # Empty list index creates an empty array
    a = Tensor([1, 2, 3])
    numpy_testing_assert_equal_helper(a[[]], np.array([]))

    b = Tensor([]).cast(dtypes.int64)
    numpy_testing_assert_equal_helper(a[[]], np.array([]))

    # TODO fancy index dtype error
    b = Tensor([]).float()
    self.assertRaises(IndexError, lambda: a[b])
  '''

  def test_ellipsis_index(self):
    a = Tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    self.assertIsNot(a[...], a)
    numpy_testing_assert_equal_helper(a[...], a)
    # `a[...]` was `a` in numpy <1.9.
    # TODO data_ptr
    '''
    numpy_testing_assert_equal_helper(a[...].data_ptr(), a.data_ptr())
    '''

    # Slicing with ellipsis can skip an
    # arbitrary number of dimensions
    numpy_testing_assert_equal_helper(a[0, ...], a[0])
    numpy_testing_assert_equal_helper(a[0, ...], a[0, :])
    numpy_testing_assert_equal_helper(a[..., 0], a[:, 0])

    # In NumPy, slicing with ellipsis results in a 0-dim array. In PyTorch
    # we don't have separate 0-dim arrays and scalars.
    numpy_testing_assert_equal_helper(a[0, ..., 1], np.array(2))

    # Assignment with `(Ellipsis,)` on 0-d arrays
    b = np.array(1)
    b[(Ellipsis,)] = 2
    numpy_testing_assert_equal_helper(b, 2)

  def test_single_int_index(self):
    # Single integer index selects one row
    a = Tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

    numpy_testing_assert_equal_helper(a[0], [1, 2, 3])
    numpy_testing_assert_equal_helper(a[-1], [7, 8, 9])

    self.assertRaises(IndexError, a.__getitem__, 1 << 30)
    self.assertRaises(IndexError, a.__getitem__, 1 << 64)

  # TODO bool indexing
  '''
  def test_single_bool_index(self):
    # Single boolean index
    a = Tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

    numpy_testing_assert_equal_helper(a[True], a[None])
    numpy_testing_assert_equal_helper(a[False], a[None][0:0])
  '''

  # TODO bool indexing
  '''
  def test_boolean_shape_mismatch(self):
    arr = Tensor.ones((5, 4, 3))

    index = Tensor([True])
    self.assertRaisesRegex(IndexError, 'mask', lambda: arr[index])

    index = Tensor([False] * 6)
    self.assertRaisesRegex(IndexError, 'mask', lambda: arr[index])

    # index = torch.ByteTensor(4, 4).to(device).zero_()
    index = Tensor.zeros(4, 4, dtype=dtypes.uint8)
    self.assertRaisesRegex(IndexError, 'mask', lambda: arr[index])
    self.assertRaisesRegex(IndexError, 'mask', lambda: arr[(slice(None), index)])
  '''

  # TODO setitem
  # TODO bool indexing
  '''
  def test_boolean_indexing_onedim(self):
    # Indexing a 2-dimensional array with
    # boolean array of length one
    a = Tensor([[0., 0., 0.]])
    b = Tensor([True])
    numpy_testing_assert_equal_helper(a[b], a)
    # boolean assignment
    a[b] = 1.
    numpy_testing_assert_equal_helper(a, Tensor([[1., 1., 1.]]))
  '''

  # TODO setitem
  # TODO bool indexing
  '''
  def test_boolean_assignment_value_mismatch(self):
    # A boolean assignment should fail when the shape of the values
    # cannot be broadcast to the subscription. (see also gh-3458)
    a = Tensor.arange(0, 4)

    def f(a, v):
      a[a > -1] = Tensor(v)

    self.assertRaisesRegex(Exception, 'shape mismatch', f, a, [])
    self.assertRaisesRegex(Exception, 'shape mismatch', f, a, [1, 2, 3])
    self.assertRaisesRegex(Exception, 'shape mismatch', f, a[:1], [1, 2, 3])
  '''

  # TODO setitem
  # TODO bool indexing
  '''
  def test_boolean_indexing_twodim(self):
    # Indexing a 2-dimensional array with
    # 2-dimensional boolean array
    a = Tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    b = Tensor([[True, False, True],
                [False, True, False],
                [True, False, True]])
    numpy_testing_assert_equal_helper(a[b], Tensor([1, 3, 5, 7, 9]))
    numpy_testing_assert_equal_helper(a[b[1]], Tensor([[4, 5, 6]]))
    numpy_testing_assert_equal_helper(a[b[0]], a[b[2]])

    # boolean assignment
    a[b] = 0
    numpy_testing_assert_equal_helper(a, Tensor([[0, 2, 0],
                                                  [4, 0, 6],
                                                  [0, 8, 0]]))
  '''

  # TODO bool indexing
  '''
  def test_boolean_indexing_weirdness(self):
    # Weird boolean indexing things
    a = Tensor.ones((2, 3, 4))
    numpy_testing_assert_equal_helper((0, 2, 3, 4), a[False, True, ...].shape)
    numpy_testing_assert_equal_helper(Tensor.ones(1, 2), a[True, [0, 1], True, True, [1], [[2]]])
    self.assertRaises(IndexError, lambda: a[False, [0, 1], ...])
  '''

  # TODO bool indexing
  '''
  def test_boolean_indexing_weirdness_tensors(self):
    # Weird boolean indexing things
    false = np.array(False)
    true = np.array(True)
    a = torch.ones((2, 3, 4))
    numpy_testing_assert_equal_helper((0, 2, 3, 4), a[False, True, ...].shape)
    numpy_testing_assert_equal_helper(torch.ones(1, 2), a[true, [0, 1], true, true, [1], [[2]]])
    self.assertRaises(IndexError, lambda: a[false, [0, 1], ...])
  '''

  # TODO bool indexing
  '''
  def test_boolean_indexing_alldims(self):
    true = Tensor(True)
    a = Tensor.ones((2, 3))
    numpy_testing_assert_equal_helper((1, 2, 3), a[True, True].shape)
    numpy_testing_assert_equal_helper((1, 2, 3), a[true, true].shape)
  '''

  # TODO bool indexing
  # NOTE this is easier
  '''
  def test_boolean_list_indexing(self):
    # Indexing a 2-dimensional array with
    # boolean lists
    a = Tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    b = [True, False, False]
    c = [True, True, False]
    numpy_testing_assert_equal_helper(a[b], Tensor([[1, 2, 3]]))
    numpy_testing_assert_equal_helper(a[b, b], Tensor([1]))
    numpy_testing_assert_equal_helper(a[c], Tensor([[1, 2, 3], [4, 5, 6]]))
    numpy_testing_assert_equal_helper(a[c, c], Tensor([1, 5]))
  '''

  def test_everything_returns_views(self):
    # Before `...` would return a itself.
    a = Tensor([5])

    self.assertIsNot(a, a[()])
    self.assertIsNot(a, a[...])
    self.assertIsNot(a, a[:])

  def test_broaderrors_indexing(self):
    a = Tensor.zeros(5, 5)
    self.assertRaises(IndexError, a.__getitem__, ([0, 1], [0, 1, 2]))
    self.assertRaises(IndexError, a.__setitem__, ([0, 1], [0, 1, 2]), 0)

  # TODO setitem
  '''
  def test_trivial_fancy_out_of_bounds(self):
    a = Tensor.zeros(5)
    ind = Tensor.ones(20, dtype=dtypes.int64)
    ind[-1] = 10
    self.assertRaises(IndexError, a.__getitem__, ind)
    self.assertRaises(IndexError, a.__setitem__, ind, 0)
    ind = Tensor.ones(20, dtype=dtypes.int64)
    ind[0] = 11
    self.assertRaises(IndexError, a.__getitem__, ind)
    self.assertRaises(IndexError, a.__setitem__, ind, 0)
  '''

  # TODO setitem
  '''
  def test_index_is_larger(self):
    # Simple case of fancy index broadcasting of the index.
    a = Tensor.zeros((5, 5))
    a[[[0], [1], [2]], [0, 1, 2]] = Tensor([2., 3., 4.])

    self.assertTrue((a[:3, :3] == all_(Tensor([2., 3., 4.]))))
  '''

  # TODO setitem
  '''
  def test_broadcast_subspace(self):
    a = Tensor.zeros((100, 100))
    v = Tensor.arange(0., 100)[:, None]
    b = Tensor.arange(99, -1, -1).cast(dtypes.int64)
    a[b] = v
    expected = b.float().unsqueeze(1).expand(100, 100)
    numpy_testing_assert_equal_helper(a, expected)
  '''

  # TODO copy_
  '''
  def test_truncate_leading_1s(self):
    col_max = Tensor.randn(1, 4)
    kernel = col_max.T * col_max  # [4, 4] tensor
    kernel2 = clone(kernel)
    # Set the diagonal
    kernel[range(len(kernel)), range(len(kernel))] = col_max.square()
    kernel2 = diagonal(kernel2)
    # torch.diagonal(kernel2).copy_(torch.square(col_max.view(4)))
    kernel2 = copy_(kernel2, col_max.reshape(4).square())
    numpy_testing_assert_equal_helper(kernel, kernel2)
  '''

if __name__ == '__main__':
  unittest.main()