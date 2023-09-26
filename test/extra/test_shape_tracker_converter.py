import unittest
from math import prod

from extra.shape_tracker_converter import convert_st_to_movement_ops
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.ops import MovementOps

def apply_movement_op(st, movement_op):
  op_type, args = movement_op
  if op_type == MovementOps.EXPAND: return st.expand(args)
  elif op_type == MovementOps.PAD: return st.pad(args)
  elif op_type == MovementOps.PERMUTE: return st.permute(args)
  elif op_type == MovementOps.RESHAPE: return st.reshape(args)
  elif op_type == MovementOps.SHRINK: return st.shrink(args)
  elif op_type == MovementOps.STRIDE: return st.stride(args)

def helper_test_st_converter(test_case, shapes, movement_ops):
  for shape in shapes:
    st = ShapeTracker((View.create((prod(shape),)),View.create(shape)))
    for op in movement_ops: st = apply_movement_op(st, op)

    # ugh ugly hack since deepcopy doesn't work on Shapetracker.
    copy_st = ShapeTracker((View.create((prod(shape),)),View.create(shape)))
    for op in movement_ops: copy_st = apply_movement_op(copy_st, op)

    out = convert_st_to_movement_ops(copy_st)
    print(movement_ops, out)

    # apply movement ops generated and compare result
    new_st = ShapeTracker((View.create((prod(shape),)),View.create(shape)))
    for op in out: new_st = apply_movement_op(new_st, op)

    test_case.assertEqual(len(st.views), len(new_st.views))
    for old_view, gen_view in zip(st.views, new_st.views):
      test_case.assertEqual(old_view.mask, gen_view.mask)
      test_case.assertEqual(old_view.strides, gen_view.strides)
      test_case.assertEqual(old_view.offset, gen_view.offset)

class TestStConverter(unittest.TestCase):
  def test_single_ops(self):
    helper_test_st_converter(self, [(4,3,6,1)], [(MovementOps.EXPAND, (4,3,6,6))])
    helper_test_st_converter(self, [(4,3,6)], [(MovementOps.PAD, ((1,2),(3,0),(4,5)))])
    helper_test_st_converter(self, [(4,3,6)], [(MovementOps.PERMUTE, (2,0,1))])
    helper_test_st_converter(self, [(4,3,6)], [(MovementOps.PERMUTE, (2,0,1)),(MovementOps.RESHAPE, (12,6))])
    helper_test_st_converter(self, [(4,3,6)], [(MovementOps.SHRINK, ((1,3),(0,3),(2,5)))])
    helper_test_st_converter(self, [(4,3,6)], [(MovementOps.STRIDE, (-1,1,-1))])

  def test_multiple_ops(self):
    # EXPAND combinations
    helper_test_st_converter(self, [(4,3,6,1)], [(MovementOps.EXPAND, (4,3,6,6)),(MovementOps.PERMUTE, (3,0,1,2))])
    helper_test_st_converter(self, [(4,3,6,1)], [(MovementOps.EXPAND, (4,3,6,6)),(MovementOps.PAD, ((1,2),(0,5),(4,3),(4,7)))])
    helper_test_st_converter(self, [(4,3,6,1)], [(MovementOps.EXPAND, (4,3,6,6)),(MovementOps.RESHAPE, (12,2,3,6))])
    # helper_test_st_converter(self, [(4,3,6,1)], [(MovementOps.EXPAND, (4,3,6,6)),(MovementOps.SHRINK, ((1,2),(0,2),(4,6),(1,5)))])
    # helper_test_st_converter(self, [(4,3,6,1)], [(MovementOps.EXPAND, (4,3,6,6)),(MovementOps.STRIDE, (12,2,-3,6))])

    # PAD combinations
    helper_test_st_converter(self, [(4,3,6,1)], [(MovementOps.PAD, ((1,2),(3,0),(4,5),(0,0))),(MovementOps.EXPAND, (7,6,15,6))])
    helper_test_st_converter(self, [(4,3,6)], [(MovementOps.PAD, ((1,2),(3,0),(4,5))),(MovementOps.PERMUTE, (1,2,0))])
    helper_test_st_converter(self, [(4,3,6)], [(MovementOps.PAD, ((1,2),(3,0),(4,5))),(MovementOps.RESHAPE, (42,3,5))])
    # helper_test_st_converter(self, [(4,3,6)], [(MovementOps.PAD, ((1,2),(3,0),(4,5))),(MovementOps.SHRINK, ((0,1),(1,3),(5,10)))])
    # helper_test_st_converter(self, [(4,3,6)], [(MovementOps.PAD, ((1,2),(3,0),(4,5))), (MovementOps.STRIDE, (2,3,4))])


  @unittest.skip('Tests to highlight some known limitations of shapetracker.')
  def test_known_limitations(self):
    # first test passes but second doesn't. Shape tracker outputs are the same...
    helper_test_st_converter(self, [(4,3,6,1)], [(MovementOps.PERMUTE, (2,1,3,0))])
    helper_test_st_converter(self, [(4,1,3,6)], [(MovementOps.PERMUTE, (3,2,1,0))])

    # vanilla reshapes don't create new views.
    helper_test_st_converter(self, [(2,3,4)], [(MovementOps.RESHAPE, (6,4))])

    # first dimentsion of shrink does not affect output of st.
    # The function gets the right op (so test pass) but can't generate the right shape for the next calc.
    helper_test_st_converter(self, [(2,3)], [(MovementOps.SHRINK, ((0,1),(1,2)))])
    helper_test_st_converter(self, [(10,3)], [(MovementOps.SHRINK, ((0,1),(1,2)))])


if __name__ == '__main__':
  unittest.main()
