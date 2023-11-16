from tinygrad.tensor import Tensor
from tinygrad.shape.symbolic import Variable

# specify an output shapetracker

a = Tensor.ones(5, 5)
b = a.contiguous().realize()

# with output shapetracker ShapeTracker.from_shape(self.info.shape).shrink(((1, 4), (1, 4)))

# [[0. 0. 0. 0. 0.]
#  [0. 1. 1. 1. 0.]
#  [0. 1. 1. 1. 0.]
#  [0. 1. 1. 1. 0.]
#  [0. 0. 0. 0. 0.]]

# with output shapetracker ShapeTracker.from_shape(self.info.shape).stride((2, 2))

# [[1. 0. 1. 0. 1.]
#  [0. 0. 0. 0. 0.]
#  [1. 0. 1. 0. 1.]
#  [0. 0. 0. 0. 0.]
#  [1. 0. 1. 0. 1.]]

# with output shapetracker ShapeTracker.from_shape(self.info.shape).shrink(((2, 5), (2, 5))).stride((1, 2))

# [[0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 1.]
#  [0. 0. 1. 0. 1.]
#  [0. 0. 1. 0. 1.]]