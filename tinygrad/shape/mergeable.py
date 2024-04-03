# DISABLE_CPU_CACHE=1 METAL=1 python3 tinygrad/shape/mergeable.py
# Alternates between incorrect and correct behavior when cache is enabled on M1 Mac...

from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, un1d
from tinygrad.shape.symbolic import Node, NumNode, Variable
from tinygrad import Tensor
from typing import Tuple, Iterable

def to_tensor(exp: Node, shape: Tuple[int], vars: Iterable[Variable] = None):
    if vars is None: vars = exp.vars()
    if len(vars) == 0: return None
    vars = sorted(list(vars), key=str)
    indices = [list() for _ in vars]
    indices[0] = [[x] for x in list(range(vars[0].min, vars[0].max + 1))]
    for i, v in enumerate(vars[1:]):
        for j in range(v.min, v.max + 1):
            indices[i+1] += [x + [j] for x in indices[i]]
    indices[-1].sort()
    ans = []
    for i in range(len(indices[-1])):
        ans.append(exp.substitute(dict(zip(vars, map(NumNode, indices[-1][i])))).b)
    return Tensor(ans).reshape(shape)

def is_single_view(s: ShapeTracker):
    x = to_tensor(s.expr_idxs()[0], s.shape, s.expr_idxs()[0].vars())
    m = to_tensor(s.expr_idxs()[1], s.shape, s.expr_idxs()[0].vars())
    u = m
    for i in range(N := len(s.shape)):
        u = u.cumsum(axis=i)
    lower_corner = tuple(un1d(m.shape, (u == 1).argmax().item()))
    mask = tuple(zip(lower_corner, [c + 1 for c in un1d(m.shape, u.argmax().item())]))
    strides = []
    for i in range(N):
        strides.append((x[tuple([lower_corner[i] if j != i else lower_corner[i] + 1 for j in range(N)])] - x[lower_corner]).item())
    
    # TODO: check if offset is right
    s2 = ShapeTracker(views=(View(shape=x.shape, strides=tuple(strides), offset=s.views[-1].offset, mask=mask, contiguous=False),))
    return (m * x == m * to_tensor(s2.expr_idxs()[0], s2.shape, s2.expr_idxs()[0].vars())).numpy().all()


s = ShapeTracker.from_shape((2,4)).permute((1,0)).reshape((2,4))
s1 = ShapeTracker.from_shape((2,4))
s2 = ShapeTracker(views=(View.create(shape=(3,3,3), strides=(9, 3,1), mask=((1,3), (1,3), (1,3)), offset=0),))
s3 = ShapeTracker(views=(View.create(shape=(3,3,3), strides=(9, 3,1), mask=None, offset=99),))
views = list(s.views)
views[1] = View.create(shape=views[1].shape, mask=((0,2), (0,2)))
s4 = ShapeTracker(views=tuple(views))

print(is_single_view(s))
print(is_single_view(s1))
print(is_single_view(s2))
print(is_single_view(s3))
print(is_single_view(s4))
