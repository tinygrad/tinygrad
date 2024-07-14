# source: https://mesozoic-egg.github.io/tinygrad-notes/dotproduct.html

from tinygrad.tensor import Tensor
a = Tensor([1,2])
b = Tensor([3,4])
# c = a.dot(b)
# res = c.numpy()
res = a.dot(b).numpy()
print(res) # 11

# to run:
# DEBUG=5 NOOPT=1 python temp/dot_prod_eg.py
# NOOPT means no optimization to make it easier to visualize


"""
AST output:

0 ━┳ BufferOps.STORE MemBuffer(idx=0, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)))
1  ┗━┳ ReduceOps.SUM (0,)
2    ┗━┳ BinaryOps.MUL None
3      ┣━━ BufferOps.LOAD MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(2,), strides=(1,), offset=0, mask=None, contiguous=True),)))
4      ┗━━ BufferOps.LOAD MemBuffer(idx=2, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(2,), strides=(1,), offset=0, mask=None, contiguous=True),)))

  
linear steps of the above AST

   0 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (0, True)
   1 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (1, False)
   2 UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (2, False)
   3 UOps.CONST          : dtypes.int                []                               0
   4 UOps.CONST          : dtypes.int                []                               2
   5 UOps.DEFINE_ACC     : dtypes.int                [3, 6]                           (0, 0)
   6 UOps.RANGE          : dtypes.int                [3, 4]                           (2, 0, True)
   7 UOps.LOAD           : dtypes.int                [1, 6]                           None
   8 UOps.LOAD           : dtypes.int                [2, 6]                           None
   9 UOps.ALU            : dtypes.int                [7, 8]                           BinaryOps.MUL
  10 UOps.ALU            : dtypes.int                [9, 5]                           BinaryOps.ADD
  11 UOps.PHI            : dtypes.int                [5, 10]                          None
  12 UOps.ENDRANGE       :                           [6]                              None
  13 UOps.STORE          :                           [0, 3, 11]                       None

"""