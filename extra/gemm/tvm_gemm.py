# https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html#example-2-manually-optimizing-matrix-multiplication-with-te
import os
import tvm
from tvm import te
#print(tvm.target.Target.list_kinds())

M, N, K = 1024, 1024, 1024

"""
# c, opencl
target = tvm.target.Target(target="c")

# TVM Matrix Multiplication using TE
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

# Default schedule
s = te.create_schedule(C.op)
#print(tvm.lower(s, [A, B, C], simple_mode=True))

# Output C code
func = tvm.build(s, [A, B, C], target=target, name="mmult")
print(func.get_source())
"""

# tinygrad version

from tinygrad.tensor import Tensor

# define the compute
# TODO: the expand, realize, and keepdim won't be needed if we remove movement lazyops
A = Tensor.rand(M, K, device="clang")
B = Tensor.rand(K, N, device="clang")
C = (A.reshape(M, 1, K).expand(M, N, K).realize() * B.permute(1,0).reshape(1, N, K).expand(M, N, K).realize()).sum(axis=2, keepdim=True)

from extra.utils import print_tree
print_tree(C)

from tinygrad.codegen.linearizer import Linearizer

# default schedule
C.lazydata._prerealize()
print(C.lazydata.op.src)
out = Tensor.empty(M, N, 1, device="clang").realize()
s = Linearizer(C.lazydata.op, out.lazydata)
uops = s.linearize().uops

from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage
src = uops_to_cstyle(CStyleLanguage(), "mmult", uops)
print(src)

exit(0)

# capture the kernel. TODO: https://github.com/tinygrad/tinygrad/issues/1812
os.environ["NOOPT"] = "1"
from tinygrad.jit import CacheCollector
CacheCollector.start()
C.realize()
result = CacheCollector.finish()

print(result[0][0].prg)


