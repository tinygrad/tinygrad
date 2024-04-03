from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, un1d
from tinygrad.shape.symbolic import Node, NumNode, Variable, sint
from tinygrad import Tensor
from typing import List, Tuple

def node_to_tensor(exp: Node, vars: List[Variable], shape: Tuple[sint, ...]) -> Tensor:
    if len(vars) == 0:
        assert isinstance(exp, NumNode), f'No vars in {exp}'
        return Tensor(exp.min) if shape is None else Tensor.full(shape, exp.min)
    indices: List[List[List[int]]] = [list() for _ in vars]
    indices[0] = [[x] for x in list(range(vars[0].min, vars[0].max + 1))]
    for i, v in enumerate(vars[1:]):
        for j in range(v.min, v.max + 1):
            indices[i+1] += [x + [j] for x in indices[i]]
    indices[-1].sort()
    ans = []
    for i in range(len(indices[-1])):
        ans.append(exp.substitute(dict(zip(vars, map(NumNode, indices[-1][i])))).b)
    return Tensor(ans).reshape(shape)

def st_to_tensors(st: ShapeTracker) -> Tuple[Tensor, Tensor]:
    vars = [Variable(f'idx{i}', 0, d-1) for i, d in enumerate(st.shape)]
    return node_to_tensor(st.expr_idxs()[0], vars, shape=st.shape), node_to_tensor(st.expr_idxs()[1], vars, shape=st.shape)

def st_equal(st1: ShapeTracker, st2: ShapeTracker) -> bool:
    if st1.shape != st2.shape: return False
    x1, m1 = st_to_tensors(st1)
    x2, m2 = st_to_tensors(st2)
    if len(set([x1.shape, m1.shape, x2.shape, m2.shape])) > 1: return False
    return bool((m1 == m2).numpy().all() and (m1 * x1 == m2 * x2).numpy().all())

def merge_views(st: ShapeTracker) -> ShapeTracker:
    x, m = st_to_tensors(st)
    u = m
    for i in range(N := len(m.shape)):
        u = u.cumsum(axis=i) if m.shape[i] > 1 else u
    lower_corner = tuple(un1d(m.shape, (u == 1).argmax().item()))
    mask = tuple([(0,0)]*N if m.sum().item() == 0 else zip(lower_corner, [c + 1 for c in un1d(m.shape, u.argmax().item())]))
    strides = []
    for i in range(N):
        if lower_corner[i] == x.shape[i] - 1:
            strides.append(0)
        else:
            strides.append(int((x[tuple([lower_corner[j] if j != i else lower_corner[j] + 1 for j in range(N)])] - x[lower_corner]).item()))
    offset = (x[lower_corner] - (Tensor(strides) * Tensor(list(lower_corner))).sum()).item()
    st2 = ShapeTracker(views=(View.create(shape=x.shape, strides=tuple(strides), offset=offset, mask=mask),))
    return st2 if st_equal(st, st2) else st

def simplify2(st: ShapeTracker) -> ShapeTracker:
    for i in range(2, len(st.views) + 1):
        if len((st2 := merge_views(ShapeTracker(st.views[-i:]))).views) < i:
            return simplify2(ShapeTracker(st.views[:-i] + st2.views))
    return st
