#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.shape.symbolic import Variable, NumNode
from tinygrad.ops import UOp, UOps, graph_rewrite
from tinygrad.codegen.uopgraph import constant_folder
from itertools import product

def shapetracker_getitem(st:ShapeTracker, val:int):
  idx, valid = st.reshape((st.size,)).to_indexed_uops([UOp.const(dtypes.pyint, val)])
  idx, valid = graph_rewrite(idx, constant_folder), graph_rewrite(valid, constant_folder)
  assert idx.op is UOps.CONST and valid.op is UOps.CONST
  return idx.arg, valid.arg

class CheckingShapeTracker:
  def __init__(self, shape):
    self.st = ShapeTracker.from_shape(shape)
    self.t = np.arange(prod(shape), dtype=np.int32).reshape(shape)

  @property
  def shape(self):
    return self.t.shape

  def simplify(self):
    self.st = self.st.simplify()
    return self

  def reshape(self, new_shape):
    self.st = self.st.reshape(new_shape)
    self.t = self.t.reshape(new_shape)
    return self

  def permute(self, axis):
    self.st = self.st.permute(axis)
    self.t = np.transpose(self.t, axis)
    return self

  def expand(self, new_shape):
    self.st = self.st.expand(new_shape)
    self.t = np.broadcast_to(self.t, new_shape)
    return self

  def flip(self, axis):
    self.st = self.st.stride(tuple(-1 if i in axis else 1 for i in range(len(self.shape))))
    self.t = np.flip(self.t, axis)
    return self

  def shrink(self, arg):
    self.st = self.st.shrink(arg)
    self.t = self.t[tuple([slice(x[0], x[1]) for x in arg])]
    return self

  def pad(self, arg):
    self.st = self.st.pad(arg)
    self.t = np.pad(self.t, arg, constant_values=-1)
    return self

  def stride(self, arg):
    self.st = self.st.stride(arg)
    self.t = self.t[tuple([slice(None, None, x) for x in arg])]
    return self

  def __getitem__(self, val):
    return self.t.flatten()[val]

  @property
  def views(self): return self.st.views

  @property
  def contiguous(self): return self.st.contiguous

  def assert_same(self):
    x = [(v[0] if (v:=shapetracker_getitem(self.st, i))[1] else -1) for i in range(prod(self.st.shape))]
    y = [self[i] for i in range(prod(self.shape))]
    assert self.st.shape == self.shape
    assert x == y, f"mismatch shapetracker:{x} real:{y}"

@unittest.skip("don't create shapetrackers with views")
class TestRealIssues(unittest.TestCase):
  def test_reshape_doesnt_multiview(self):
    self.st = ShapeTracker((View.create((256, 256, 2, 2, 2, 2, 2, 256, 8, 2), (0, 8, 0, 4, 0, 0, 2, 16384, 2048, 1), 0, None),))
    self.st.reshape((128, 2, 256, 2, 2, 2, 2, 2, 256, 8, 2))
    assert len(self.st.views) == 1

  def test_reshape_stable_diffusion(self):
    # regression test for https://github.com/tinygrad/tinygrad/pull/2616
    st = ShapeTracker((View((2, 1920, 32, 32), (1310720, 1024, 32, 1), 0, ((0, 2), (0, 1280), (0, 32), (0, 32)), False),))
    st = st.reshape((2, 32, 240, 256))
    assert len(st.views) == 2

  def test_reshape_trailing_invalid_ones(self):
    st = ShapeTracker((View(shape=(1, 1, 5), strides=(0, 0, 1), offset=-5, mask=((1, 1), (0, 1), (0, 5)), contiguous=False),))
    st = st.reshape((5,))
    assert len(st.views) == 1
    assert st.views[0].mask == ((0,0),)

class TestRealDoesntSimplify(unittest.TestCase):
  def tearDown(self):
    st = self.st.real_strides()
    print(st)
    self.st = self.st.simplify()
    assert len(self.st.views) != 1
    assert None in st

  def test_1(self):
    self.st = ShapeTracker((
      View.create((8, 3, 1, 2, 11, 1), (33, 11, 0, 0, 1, 0), 0, None),
      View.create((8, 6, 11), (66, 11, 1), 0, None)))
    self.assertEqual(self.st.real_strides(), (33, None, 1))

  def test_2(self):
    self.st = ShapeTracker((
      View.create((2, 2, 4, 3, 3), (72, 9, 18, -3, -1), 8, None),
      View.create((4, 4, 3, 3), (36, 9, 3, 1), 0, None)))
    self.assertEqual(self.st.real_strides(), (None, 18, -3, -1))

class TestRealStrides(unittest.TestCase):
  @unittest.expectedFailure
  def test_1(self):
    # TODO: find the correct rewrite rule to fix this
    self.st = ShapeTracker((
      View.create((2048,), (1,), 0, ((0, 512),)),
      View.create((16, 32, 4), (128, 4, 1), 0, None)))
    st = self.st.real_strides()
    print(self.st, st)
    assert st == (None, 4, 1)

class TestRealSimplifies(unittest.TestCase):
  def tearDown(self):
    st = self.st.real_strides()
    self.st = self.st.simplify()
    assert len(self.st.views) == 1
    print(self.st.views[-1].strides, st)
    self.assertEqual(self.st.views[-1].strides, st)

  def test_1(self):
    self.st = ShapeTracker((
      View.create((1, 3, 2, 11, 4, 28), (0, 308, 0, 28, 0, 1), 0, None),
      View.create((1, 3, 2, 11, 26, 1, 1, 3), (0, 2464, 0, 112, 1, 0, 0, 29), 0, None)))

  def test_2(self):
    self.st = ShapeTracker((
      View.create((8, 3, 3, 11, 2, 28), (924, 308, 0, 28, 0, 1), 0, None),
      View.create((8, 1, 6, 10, 28, 3, 2, 1), (5544, 0, 0, 56, 1, 1848, 672, 0), 0, None)))

class TestViewMinify(unittest.TestCase):
  def test_minifies(self):
    assert len(View.create((10,10)).minify().shape) == 1
    assert len(View.create((10,10)).permute((1,0)).minify().shape) == 2
    assert len(View.create((10,10,10,10)).permute((1,0,2,3)).minify().shape) == 3

class TestIndexExpressions2d(unittest.TestCase):

  def setUp(self):
    shapes = [(30, 5), (15, 10), (15, 1), (5, 10), (5, 1)] # Make sure dim0 is a multiple of 5, one of the tests divides this dimension by 5
    offsets = [0, 1, 15, 28, 10000]
    self.sts = [ShapeTracker.from_shape((prod(base_shape)+offset,)).shrink(((offset, offset+prod(base_shape)),)).\
                reshape(base_shape) for base_shape in shapes for offset in offsets]
    self.offset = [NumNode(offset) for base_shape in shapes for offset in offsets]
    self.shapes = [shape for shape in shapes for offset in offsets]
    self.idxs_exprs = []

  def tearDown(self):
    for st, offset, shape, idxs_expr in zip(self.sts, self.offset, self.shapes, self.idxs_exprs):
      numel = prod(shape)
      self.check_bounds(idxs_expr(self.default_idxs(st.shape)), offset, numel)
      idx0s = [(0,0), (0, min(1, st.shape[0]-1)), (0, st.shape[0]-1), (min(3, st.shape[0]-1), min(6, st.shape[0]-1)), (st.shape[0]-1, st.shape[0]-1)]
      idx1s = [(0,0), (0, min(1, st.shape[1]-1)), (0, st.shape[1]-1), (min(3, st.shape[1]-1), min(6, st.shape[1]-1)), (st.shape[1]-1, st.shape[1]-1)]
      idx2s = [(0,0), (0, min(1, st.shape[2]-1)), (0, st.shape[2]-1), (min(3, st.shape[2]-1), min(6, st.shape[2]-1)),
               (st.shape[2]-1, st.shape[2]-1)] if len(st.shape) == 3 else [None for _ in idx0s]
      for idx0, idx1, idx2 in product(idx0s, idx1s, idx2s):
        idxs = [Variable(f"idx{i}", idx[0], idx[1]) for i, idx in enumerate((idx0, idx1, idx2)) if idx is not None]
        self.check_bounds(idxs_expr(idxs), offset, numel)

  def default_idx(self, shape):
    return Variable("idx", 0, prod(shape)-1)

  def default_idxs(self, shape):
    return [Variable(f"idx{i}", 0, d-1) for i,d in enumerate(shape)]

  def check_bounds(self, expr, offset, numel):
    assert expr.min >= offset
    assert expr.max <= offset + numel - 1

  def test_noop(self):
    for st, base_shape, offset in zip(self.sts, self.shapes, self.offset):
      self.idxs_exprs.append(lambda idxs, base_shape=base_shape, offset=offset: idxs[0]*base_shape[1] + idxs[1] + offset)

  def test_permute(self):
    new_st = []
    for st, base_shape, offset in zip(self.sts, self.shapes, self.offset):
      st = st.permute((1, 0))
      self.idxs_exprs.append(lambda idxs, base_shape=base_shape, offset=offset: idxs[0] + idxs[1]*base_shape[1] + offset)
      new_st.append(st)
    self.sts = new_st

  def test_reshape(self):
    new_st = []
    for st, base_shape, offset in zip(self.sts, self.shapes, self.offset):
      st = st.reshape((base_shape[0], 1, base_shape[1]))
      self.idxs_exprs.append(lambda idxs, base_shape=base_shape, offset=offset: idxs[0]*base_shape[1] + idxs[2] + offset)
      new_st.append(st)
    self.sts = new_st

  def test_reshape_expand(self):
    new_st = []
    for st, base_shape, offset in zip(self.sts, self.shapes, self.offset):
      st = st.reshape((base_shape[0], 1, base_shape[1]))
      st = st.expand((base_shape[0], base_shape[1], base_shape[1]))
      self.idxs_exprs.append(lambda idxs, base_shape=base_shape, offset=offset: idxs[0]*base_shape[1] + idxs[2] + offset)
      new_st.append(st)
    self.sts = new_st

  def test_permute_reshape_1(self): # This tests multiple views
    new_st = []
    for st, base_shape, offset in zip(self.sts, self.shapes, self.offset):
      st = st.permute((1, 0))
      st = st.reshape((base_shape[0]//5, 1, base_shape[1]*5))
      self.idxs_exprs.append(lambda idxs, base_shape=base_shape, offset=offset: (idxs[0]*(base_shape[1]*5)+idxs[2])%base_shape[0]*base_shape[1] + \
                             (idxs[0]*(base_shape[1]*5)+idxs[2])//base_shape[0] + offset)
      new_st.append(st)
    self.sts = new_st

  def test_permute_reshape_2(self):
    new_st = []
    for st, base_shape, offset in zip(self.sts, self.shapes, self.offset):
      st = st.permute((1, 0))
      st = st.reshape((1, base_shape[0]//5, base_shape[1]*5))
      self.idxs_exprs.append(lambda idxs, base_shape=base_shape, offset=offset: (idxs[1]*(base_shape[1]*5)+idxs[2])%base_shape[0]*base_shape[1] + \
                             (idxs[1]*(base_shape[1]*5)+idxs[2])//base_shape[0] + offset)
      new_st.append(st)
    self.sts = new_st

  def test_reshaping_splitting(self):
    self.st = CheckingShapeTracker((5,10,5,10))
    self.st.permute((1, 0, 3, 2))
    self.st.pad(((0,0), (0,5), (0,0), (0,5)))
    self.st.reshape((10,2,5,10,2,5))
    assert len(self.st.views) == 1
    self.st.assert_same()

  def test_reshape_splitting_1(self):
    self.st = CheckingShapeTracker((1,10,1))
    self.st.pad(((0,4),(0,0),(1,0)))
    self.st.reshape((5,5,2,2))
    assert len(self.st.views) == 1
    self.st.assert_same()

  def test_reshape_combining_1(self):
    self.st = CheckingShapeTracker((2,1,10))
    self.st.pad(((2,6), (0,0), (0,0)))
    self.st.reshape((100,))
    assert len(self.st.views) == 1
    self.st.assert_same()

  def test_reshape_combining_2(self):
    self.st = CheckingShapeTracker((1,1,5))
    self.st.pad(((3,6), (0,0), (0,5)))
    self.st.reshape((100,))
    assert len(self.st.views) == 1
    self.st.assert_same()

  def test_reshape_combining_3(self):
    self.st = CheckingShapeTracker((1,1,4))
    self.st.pad(((3,6), (0,0), (1,5)))
    self.st.reshape((100,))
    assert len(self.st.views) == 1
    assert self.st.views[0].mask[0] == (31, 35)
    self.st.assert_same()

  def test_reshape_combining_4(self):
    # interestingly this one is quite slow
    self.st = CheckingShapeTracker((1,1,5,5,1,1,5))
    self.st.pad(((3,6), (0,0), (0,5), (0,0), (3,6), (0,0), (0,5)))
    self.st.reshape((100,5,100))
    assert len(self.st.views) == 1
    self.st.assert_same()

  def test_reshape_splitting_combining(self):
    self.st = CheckingShapeTracker((1,5,5))
    self.st.pad(((0,4), (0,5), (0,0)))
    self.st.reshape((10,25))
    assert len(self.st.views) == 1
    self.st.assert_same()

  def test_reshape_only_1s(self):
    self.st = CheckingShapeTracker((1, 1, 1, 4, 1, 3, 5, 1))
    self.st.pad(((0,4), (0,0), (0,0), (1,1), (0,0), (0,0), (0,0), (0,0)))
    self.st.reshape((5, 6, 3, 5))
    assert len(self.st.views) == 1
    self.st.assert_same()
    self.st.reshape((1, 1, 5, 6, 3, 5, 1, 1))
    assert len(self.st.views) == 1
    self.st.assert_same()
    self.st.reshape((1, 5, 6, 1, 3, 1, 5, 1))
    assert len(self.st.views) == 1
    self.st.assert_same()

  def test_zero_mask_1(self):
    self.st = CheckingShapeTracker((1, 3, 2))
    self.st.pad(((0,0), (0,3), (0,0)))
    self.st.shrink(((0,1), (3,6), (0,2)))
    self.st.reshape((3,2))
    assert len(self.st.views) == 1
    self.st.assert_same()
    self.st.reshape((1, 3, 1, 2, 1))
    assert len(self.st.views) == 1
    self.st.assert_same()

  def test_zero_mask_2(self):
    self.st = CheckingShapeTracker((1, 3, 2))
    self.st.pad(((0,2), (0,3), (0,0)))
    self.st.shrink(((2,3), (3,6), (0,2)))
    self.st.reshape((3,2))
    assert len(self.st.views) == 1
    self.st.assert_same()
    self.st.reshape((1, 3, 1, 2, 1))
    assert len(self.st.views) == 1
    self.st.assert_same()

  def test_expanded_reshaped(self):
    self.st = CheckingShapeTracker((1, 3, 2, 1))
    self.st.expand((5, 3, 2, 2))
    self.st.pad(((0,0), (0,3), (0,0), (0, 0)))
    self.st.reshape((5, 2, 3, 2, 2))
    assert len(self.st.views) == 1
    self.st.assert_same()

  def test_splitting_big(self):
    self.st = CheckingShapeTracker((1, 5, 1, 15, 1))
    self.st.pad(((0,0), (0,5), (0,0), (0,15), (0,0)))
    self.st.reshape((10, 1, 30))
    self.st.permute((2,1,0))
    self.st.reshape((2,3,5,2,5))
    assert len(self.st.views) == 1
    v = self.st.views[-1]
    assert v.strides == (0, 5, 1, 0, 15) and v.mask == ((0, 1), (0, 3), (0, 5), (0, 1), (0, 5))
    self.st.assert_same()

  def test_combining_big(self):
    self.st = CheckingShapeTracker((1,3,1,5,3,1))
    self.st.pad(((0,0),(2,2),(0,0),(0,0),(0,0),(0,0)))
    self.st.reshape((1,1,1,105,1,1))
    assert len(self.st.views) == 1
    v = self.st.views[-1]
    assert v.strides == (0, 0, 0, 1, 0, 0) and v.mask == ((0, 1), (0, 1), (0, 1), (30, 75), (0, 1), (0, 1)) and v.offset == -30
    self.st.assert_same()

  def test_pad_reshape(self):
    self.st = CheckingShapeTracker((4,))
    self.st.pad(((2,2),))
    self.st.reshape((4,2))
    assert len(self.st.views) == 1
    self.st.assert_same()

class TestSimplifyingShapeTracker(unittest.TestCase):
  def setUp(self):
    self.st = CheckingShapeTracker((1, 10))

  def tearDown(self):
    self.st.assert_same()

  # multiview simplify
  def test_expand_contract_simple(self):
    self.st = self.st.expand((10, 10))
    self.st = self.st.reshape((100,))
    print(self.st.views)
    assert (len(self.st.views) == 2)
    self.st = self.st.reshape((10, 10))
    print(self.st.views)

    self.st = self.st.simplify()
    print(self.st.views)
    assert (len(self.st.views) == 1)

  # multiview simplify
  def test_expand_contract_different_shape(self):
    self.st.expand((10, 10))
    self.st.reshape((100,))
    print(self.st.views)
    assert (len(self.st.views) == 2)
    self.st.reshape((2, 5, 2, 5))
    print(self.st.views)

    self.st = self.st.simplify()
    print(self.st.views)
    assert (len(self.st.views) == 1)

  # multiview simplify
  def test_expand_contract_still_complex(self):
    self.st.expand((10, 10))
    self.st.reshape((100,))
    print(self.st.views)
    assert (len(self.st.views) == 2)
    self.st.reshape((5, 20))

    self.st = self.st.simplify()
    print(self.st.views)
    assert (len(self.st.views) == 2)

# Tensor.zeros(2, 4).permute(1,0).reshape(2, 4)
# (d1*4 + d0%4), d1=x//4, d0=x%4 = ((x//4)*4) + (x%4)%4

class TestComplexShapeTracker(unittest.TestCase):
  def test_add_1s(self):
    self.st = CheckingShapeTracker((4, 4))
    self.st.permute((1,0))
    self.st.reshape((1,4,1,4,1))
    assert not self.st.contiguous
    self.st.permute((0,3,2,1,4))
    assert self.st.contiguous

  def test_permute_1s_simple(self):
    self.st = CheckingShapeTracker((1, 16, 9,9))
    self.st.permute((1,0,2,3))
    assert self.st.contiguous
    self.st = CheckingShapeTracker((2, 16, 9,9))
    self.st.permute((1,0,2,3))
    assert not self.st.contiguous

  def test_remove_1s_simple(self):
    self.st = CheckingShapeTracker((1, 16, 1, 1))
    self.st.reshape((16,))
    assert self.st.contiguous

  def test_remove_1s(self):
    self.st = CheckingShapeTracker((1, 4, 1, 4, 1))
    self.st.permute((0,3,2,1,4))
    self.st.reshape((4,4))
    assert not self.st.contiguous
    self.st.permute((1,0))
    assert self.st.contiguous

  def test_permute_reshape(self):
    self.st = CheckingShapeTracker((4, 4))
    self.st.permute((1,0))
    self.st.reshape((2, 2, 2, 2))
    # TODO: should also be tested by test_super_complex
    assert len(self.st.views) == 1

  def test_factorize_split(self):
    self.st = CheckingShapeTracker((4, 4))
    self.st.permute((1,0))
    self.st.reshape((2, 2, 2, 2))
    self.st.permute((2,3,0,1))
    assert self.st.contiguous

  def test_factorize_combine(self):
    self.st = CheckingShapeTracker((4, 4, 4))
    self.st.permute((2, 0, 1))
    self.st.reshape((4, 16))
    self.st.permute((1, 0))
    assert self.st.contiguous

  def test_factorize_combine_add_ones(self):
    self.st = CheckingShapeTracker((4, 4, 4))
    self.st.permute((2, 0, 1))
    self.st.reshape((4, 16, 1, 1))
    self.st.permute((1, 0, 2, 3))
    assert self.st.contiguous

  def test_fancy_factorize(self):
    self.st = CheckingShapeTracker((32, 3, 3, 1))
    self.st.reshape((8, 4, 3, 3))
    assert len(self.st.views) == 1

  def test_super_complex_2_fail(self):
    self.st = CheckingShapeTracker((4, 4, 4))
    self.st.permute((2, 0, 1))
    self.st.reshape((16, 4))
    assert len(self.st.views) != 1

  def test_work(self):
    self.st = CheckingShapeTracker((64, 1024, 4))
    self.st.reshape((1, 64, 128, 32))
    self.st.permute((0, 3, 1, 2))
    self.st.reshape((1, 32, 1, 64, 128))
    self.st.permute((0, 3, 4, 1, 2))
    assert self.st.contiguous

  def test_work2(self):
    self.st = CheckingShapeTracker((64, 1024, 4))
    self.st.reshape((1, 64, 128, 32))
    self.st.permute((0, 3, 1, 2))
    self.st.reshape((1, 1, 32, 64, 128))
    self.st.permute((0, 3, 4, 1, 2))
    self.st.reshape((64, 1024, 4))
    print(self.st.views)
    assert self.st.contiguous

class TestShapeTrackerEquality(unittest.TestCase):
  def test_simple_equals(self):
    self.assertEqual(ShapeTracker.from_shape((10,10)), ShapeTracker.from_shape((10,10)))
  def test_other_equals(self):
    st1 = ShapeTracker(views=(View(shape=(3,), strides=(1,), offset=0, mask=None, contiguous=True)))
    st2 = ShapeTracker(views=(View(shape=(3,), strides=(1,), offset=0, mask=None, contiguous=True)))
    self.assertEqual(st1, st2)

class TestSingleShapeTracker(unittest.TestCase):
  def setUp(self):
    self.st = CheckingShapeTracker((7,4))

  def tearDown(self):
    self.st.assert_same()

  def test_reshape(self):
    self.st.reshape((7,1,4))
    assert self.st.contiguous

  def test_permute(self):
    self.st.permute((1,0))
    assert not self.st.contiguous

  def test_shrink(self):
    self.st.shrink(((1,2), (0,4)))
    assert not self.st.contiguous

  def test_double_permute(self):
    self.st.permute((1,0))
    self.st.permute((1,0))
    assert self.st.contiguous

  def test_reshape_permute(self):
    self.st.reshape((7,1,4))
    self.st.permute((0,1,2))
    assert self.st.contiguous

  def test_reshape_permute_yes(self):
    self.st.reshape((7,1,4))
    self.st.permute((0,2,1))
    assert self.st.contiguous

  def test_reshape_permute_no(self):
    self.st.reshape((4,7))
    self.st.permute((1,0))
    assert not self.st.contiguous

class TestShapeTrackerFuzzFailures(unittest.TestCase):
  def setUp(self):
    self.st = CheckingShapeTracker((3,3,3))
  def tearDown(self):
    self.st.assert_same()
  def test_case_1(self):
    self.st.shrink(((1, 2), (1, 3), (1, 3)))
    self.st.reshape((1, 4))
    self.st.shrink(((0, 1), (1, 3)))
    print(self.st.st)
    self.st = self.st.simplify()
    print(self.st.st)
  def test_case_2(self):
    self.st.stride( (1, 1, -2) )
    self.st.reshape( (3, 6) )
    self.st.shrink( ((1, 2), (1, 5)) )
    self.st.stride( (1, -1) )
  def test_case_3(self):
    self.st.shrink( ((0, 2), (0, 2), (0, 1)) )
    self.st.permute( (1, 0, 2) )
    self.st.reshape( (4,) )
    self.st.shrink( ((0, 3),) )
    self.st.stride( (-1,) )
  def test_case_4(self):
    self.st.reshape( (3, 3, 3, 1) )
    self.st.pad( ((0, 0), (0, 0), (0, 0), (1, 1)) )
    self.st.shrink( ((0, 2), (1, 2), (0, 2), (0, 1)) )
    self.st.expand( (2, 1, 2, 3) )

class TestMaskedShapeTracker(unittest.TestCase):
  def test_pad_1x1(self):
    self.st = CheckingShapeTracker((1,1))
    self.st.pad(((1,1), (1,1)))
    self.st.assert_same()

  def test_pad_2x2(self):
    self.st = CheckingShapeTracker((2,2))
    self.st.pad(((1,1), (1,1)))
    self.st.assert_same()

  def test_pad_reshape(self):
    st1 = CheckingShapeTracker((1, 2))
    st1.pad(((1, 0), (0, 1)))
    st1.reshape((3, 2))
    st1.assert_same()

    st2 = CheckingShapeTracker((1, 2))
    st2.pad(((1, 1), (0, 2)))
    st2.reshape((4, 3))
    st2.assert_same()

    st3 = CheckingShapeTracker((1, 1, 1, 2))
    st3.pad(((0, 2), (1, 2), (2, 2), (0, 4)))
    st3.reshape((4, 3, 6, 5))
    st3.assert_same()

  def test_axis_is_masked(self):
    st = ShapeTracker.from_shape((100, 100, 100, 100)).pad(((0,1),(0,0),(2,0), (0,0)))
    assert st.axis_is_masked(0)
    assert not st.axis_is_masked(1)
    assert st.axis_is_masked(2)
    assert not st.axis_is_masked(3)

  def test_axis_is_masked_rw1(self):
    st = ShapeTracker(views=(View(shape=(1, 2, 1, 4, 4, 13, 4, 13), strides=(0, 324, 0, 81, 0, 9, 0, 1), offset=-20,
                                  mask=((0, 1), (0, 2), (0, 1), (0, 4), (0, 4), (2, 11), (0, 4), (2, 11)), contiguous=False),
                             View(shape=(2, 4, 11, 11, 4, 3, 3), strides=(10816, 0, 52, 1, 2704, 728, 14), offset=0,
                                  mask=None, contiguous=False)))
    assert not st.axis_is_masked(0)

class TestShapeTracker(unittest.TestCase):
  def setUp(self):
    self.st = CheckingShapeTracker((7,4))
    self.apply = lambda fxn: [fxn(x) for x in [self.st]]

  def tearDown(self):
    self.st.assert_same()

  def test_noop(self):
    pass

  def test_simple_split(self):
    self.test_permute()
    self.apply(lambda x: x.reshape((prod(self.st.shape), )))

  def test_simple_pad(self):
    self.st.pad(((1,1), (1,1)))

  def test_pad_shrink(self):
    self.st.pad(((1,1), (1,1)))
    self.st.shrink(((0,4), (0,4)))

  def test_pad_one_sided(self):
    self.st.pad(((0,1), (0,0)))

  def test_pad_reshape(self):
    self.st.pad(((0,1), (0,0)))
    self.st.reshape((8*4,))

  def test_pad_pad(self):
    self.st.pad(((1,1), (1,1)))
    self.st.pad(((1,1), (1,1)))

  def test_pad_permute(self):
    self.st.pad(((1,1), (2,2)))
    self.st.permute((1,0))

  def test_pad_expand(self):
    self.st.reshape((7,4,1))
    self.st.pad(((1,1), (1,1), (0,0)))
    self.st.expand((9,6,4))

  def test_pad_expand_alt(self):
    self.st.pad(((1,1), (1,1)))
    self.st.reshape((9,6,1))
    self.st.expand((9,6,4))

  def test_pad_stride(self):
    self.st.pad(((1,4), (1,3)))
    self.st.stride((2,2))

  def test_pad_stride_neg(self):
    self.st.pad(((1,2), (1,0)))
    self.st.stride((-1,-1))

  def test_pad_stride_both(self):
    self.st.pad(((1,2), (1,0)))
    self.st.stride((-2,-2))

  def test_shrink_pad(self):
    self.st.shrink(((0,4), (0,4)))
    self.st.pad(((1,1), (1,1)))

  def test_reshape(self):
    new_shape = self.st.shape[::-1]
    self.apply(lambda x: x.reshape(new_shape))

  def test_permute(self):
    if len(self.st.shape) == 2: self.apply(lambda x: x.permute((1,0)))
    elif len(self.st.shape) == 3: self.apply(lambda x: x.permute((2,0,1)))

  def test_reshape_with_1(self):
    new_shape = (self.st.shape[0], 1, self.st.shape[1])
    self.apply(lambda x: x.reshape(new_shape))

  def test_expand(self):
    self.test_reshape_with_1()
    new_shape = list(self.st.shape)
    new_shape[1] = 2
    self.apply(lambda x: x.expand(tuple(new_shape)))

  def test_flip_0(self):
    self.apply(lambda x: x.flip((0,)))

  def test_flip_1(self):
    self.apply(lambda x: x.flip((1,)))

  def test_flip_01(self):
    self.apply(lambda x: x.flip((0,1)))

  def test_slice_0(self):
    self.apply(lambda x: x.shrink(((1, x.shape[0]), (0, x.shape[1]))))

  def test_slice_1(self):
    self.apply(lambda x: x.shrink(((0, x.shape[0]), (1, x.shape[1]))))

  def test_slice_1c1(self):
    self.apply(lambda x: x.shrink(((0, 1), (0, 1))))

  def test_slice_1c2(self):
    self.apply(lambda x: x.shrink(((1, 2), (1, 2))))

  def test_double_permute(self):
    self.apply(lambda x: x.permute((1, 0)))
    self.apply(lambda x: x.permute((1, 0)))

  def test_slice_permute(self):
    self.apply(lambda x: x.shrink(((0, 2), (2, 4))))
    self.apply(lambda x: x.permute((1, 0)))

  def test_slice_expand(self):
    self.apply(lambda x: x.shrink(((0, 2), (3, 4))))
    self.apply(lambda x: x.expand((2, 10)))

  def test_double_stride(self):
    self.apply(lambda x: x.stride((1, 2)))
    self.apply(lambda x: x.stride((2, 1)))

  def test_stride(self): self.apply(lambda x: x.stride((2,1)))
  def test_stride_int(self): self.apply(lambda x: x.stride((1,2)))
  def test_stride_2(self): self.apply(lambda x: x.stride((2,2)))
  def test_stride_n(self): self.apply(lambda x: x.stride((-2,1)))
  def test_stride_int_n(self): self.apply(lambda x: x.stride((-1,2)))
  def test_stride_2_n(self): self.apply(lambda x: x.stride((-2,-2)))

  def test_reshape_then_permute(self):
    self.test_reshape()
    self.test_permute()

  def test_reshape_then_expand(self):
    self.test_reshape()
    self.test_expand()

  def test_permute_then_reshape(self):
    self.test_permute()
    self.test_reshape()

  def test_expand_then_reshape(self):
    self.test_expand()
    self.test_reshape()

  def test_combo(self):
    self.test_permute()
    self.test_reshape()
    self.test_slice_1()
    self.test_expand()
    self.test_permute()

class TestShapeTrackerSize(unittest.TestCase):
  def test_simple_size(self):
    st = ShapeTracker.from_shape((100, 100))
    self.assertEqual(st.real_size(), 100*100)

  def test_expand_size(self):
    st = ShapeTracker.from_shape((100, 100))
    st = st.reshape((100, 100, 1))
    st = st.expand((100, 100, 100))
    self.assertEqual(st.real_size(), 100*100)

  def test_expand_size_flatten(self):
    st = ShapeTracker.from_shape((100, 100))
    st = st.reshape((100, 100, 1))
    st = st.expand((100, 100, 100))
    st = st.reshape((100*100*100,))
    self.assertEqual(st.real_size(), 100*100)

  def test_shrink_size_axis_0(self):
    st = ShapeTracker.from_shape((100, 100))
    st = st.shrink(((0, 50), (0, 100)))
    self.assertEqual(st.real_size(), 50*100)

  def test_shrink_size_axis_0_variable(self):
    st = ShapeTracker.from_shape((100, 100))
    st = st.shrink(((0, Variable("a", 0, 50)), (0, 100)))
    self.assertEqual(st.real_size(), 50*100)

  def test_shrink_size_axis_1(self):
    st = ShapeTracker.from_shape((100, 100))
    st = st.shrink(((0, 100), (0, 50)))
    self.assertEqual(st.real_size(), 9950)    # careful here

  def test_size_variable(self):
    st = ShapeTracker(views=(View(shape=(1, 1, 1, (NumNode(1)+Variable('start_pos', 0, 8192)), 1, 8, 4, 128), strides=(0, 0, 0, 1024, 0, 128, 0, 1),
                                  offset=0, mask=None, contiguous=False), View(shape=(1, 32, 1, (NumNode(1)+Variable('start_pos', 0, 8192)), 128),
                                                                               strides=(0, 128, 0, 4096, 1), offset=0, mask=None, contiguous=False)))
    self.assertEqual(st.real_size(), 8389632)

class TestConsecutive(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    from tinygrad.tensor import Tensor  # easier test setup
    self.t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    self.const = Tensor(2)
    self.ones = Tensor.ones(2, 4)

  def test_unmodified(self):
    assert self.t.lazydata.st.consecutive
    assert self.t.reshape(4, 2).lazydata.st.consecutive
    assert self.t.reshape(1, 8).lazydata.st.consecutive

  def test_sliced(self):
    assert self.t[0].lazydata.st.consecutive
    assert self.t[0, 1:2].lazydata.st.consecutive
    assert self.t[1].lazydata.st.consecutive
    assert not self.t[:, 0].lazydata.st.consecutive
    assert not self.t[:, 1].lazydata.st.consecutive

  def test_padded(self):
    assert not self.t.pad(((1, 1), None)).lazydata.st.consecutive
    assert not self.t.pad((None, (1, 1))).lazydata.st.consecutive

  def test_const(self):
    assert self.const.lazydata.st.consecutive

  def test_ones(self):
    assert not self.ones.lazydata.st.consecutive
    assert not self.ones[0, :].lazydata.st.consecutive
    # consecutive if sliced into size 1
    assert self.ones[0, 0].lazydata.st.consecutive

if __name__ == '__main__':
  unittest.main()
