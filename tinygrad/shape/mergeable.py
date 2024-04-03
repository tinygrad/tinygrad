from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, un1d
from tinygrad.shape.symbolic import Node, NumNode, Variable, sint
from tinygrad import Tensor
from typing import Tuple, Optional, Set, List

def to_tensor(exp: Node, shape: Tuple[sint, ...], vars: Optional[Set[Variable]]):
    if vars is None: vars = exp.vars()
    if len(vars) == 0: return None
    vars2 = sorted(list(vars), key=str)
    indices: List[List[List[int]]] = [list() for _ in vars2]
    indices[0] = [[x] for x in list(range(vars2[0].min, vars2[0].max + 1))]
    for i, v in enumerate(vars2[1:]):
        for j in range(v.min, v.max + 1):
            indices[i+1] += [x + [j] for x in indices[i]]
    indices[-1].sort()
    ans = []
    for i in range(len(indices[-1])):
        ans.append(exp.substitute(dict(zip(vars2, map(NumNode, indices[-1][i])))).b)
    return Tensor(ans).reshape(shape)

def is_single_view(st: ShapeTracker):
    x = to_tensor(st.expr_idxs()[0], st.shape, st.expr_idxs()[0].vars())
    m = to_tensor(st.expr_idxs()[1], st.shape, st.expr_idxs()[0].vars())
    u = m
    for i in range(N := len(st.shape)):
        u = u.cumsum(axis=i)
    lower_corner = tuple(un1d(m.shape, (u == 1).argmax().item()))
    mask = tuple(zip(lower_corner, [c + 1 for c in un1d(m.shape, u.argmax().item())]))
    strides = []
    for i in range(N):
        strides.append((x[tuple([lower_corner[i] if j != i else lower_corner[i] + 1 for j in range(N)])] - x[lower_corner]).item())

    # TODO: check if offset is right
    st2 = ShapeTracker(views=(View(shape=x.shape, strides=tuple(strides), offset=st.views[-1].offset, mask=mask, contiguous=False),))
    return (m * x == m * to_tensor(st2.expr_idxs()[0], st2.shape, st2.expr_idxs()[0].vars())).numpy().all()


s = ShapeTracker.from_shape((2,4)).permute((1,0)).reshape((2,4))
s1 = ShapeTracker.from_shape((2,4))
s2 = ShapeTracker(views=(View.create(shape=(3,3,3), strides=(9, 3,1), mask=((1,3), (1,3), (1,3)), offset=0),))
s3 = ShapeTracker(views=(View.create(shape=(3,3,3), strides=(9, 3,1), mask=None, offset=99),))
views = list(s.views)
views[1] = View.create(shape=views[1].shape, mask=((0,2), (0,2)))
s4 = ShapeTracker(views=tuple(views))

assert(not is_single_view(s))
assert(is_single_view(s1))
assert(is_single_view(s2))
assert(is_single_view(s3))
assert(is_single_view(s4))
