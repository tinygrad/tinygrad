from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, un1d
from tinygrad.shape.symbolic import Node, NumNode, Variable, sint
from numpy import ndarray, array
from typing import List, Tuple

def node_to_tensor(exp: Node, shape: Tuple[sint, ...]) -> ndarray:
    if len(shape) == 0: return array(exp.min)
    vars = [Variable(f'idx{i}', 0, d-1) for i, d in enumerate(shape)]
    indices: List[List[List[int]]] = [list() for _ in vars]
    indices[0] = [[x] for x in range(vars[0].max + 1)]
    for i, v in enumerate(vars[1:]):
        for j in range(v.max + 1):
            indices[i+1] += [x + [j] for x in indices[i]]
    return array([exp.substitute(dict(zip(vars, map(NumNode, i)))).b for i in sorted(indices[-1])]).reshape(shape)

def st_to_tensors(st: ShapeTracker) -> Tuple[ndarray, ndarray]:
    return node_to_tensor(st.expr_idxs()[0], shape=st.shape), node_to_tensor(st.expr_idxs()[1], shape=st.shape)

def st_equal(st1: ShapeTracker, st2: ShapeTracker) -> bool:
    if st1.shape != st2.shape: return False
    x1, m1, x2, m2 = st_to_tensors(st1) + st_to_tensors(st2)
    return bool((m1 == m2).all() and (m1 * x1 == m2 * x2).all())

def merge_views(st: ShapeTracker) -> ShapeTracker:
    x, m = st_to_tensors(st)
    u = m
    for i in range(N := len(m.shape)):
        u = u.cumsum(axis=i) if m.shape[i] > 1 else u
    lower_corner = tuple(un1d(m.shape, (u == 1).argmax().item()))
    mask = tuple([(0,0)]*N if m.sum().item() == 0 else zip(lower_corner, [c + 1 for c in un1d(m.shape, u.argmax().item())]))
    strides = [(x[tuple([lower_corner[j] if j != i else lower_corner[j] + 1 for j in range(N)])]
                - x[lower_corner]).item() if lower_corner[i] < m.shape[i] - 1 else 0 for i in range(N)]
    offset = (x[lower_corner] - (array(strides) * array(list(lower_corner))).sum()).item()
    st2 = ShapeTracker(views=(View.create(shape=x.shape, strides=tuple(strides), offset=int(offset), mask=mask),))
    return st2 if st_equal(st, st2) else st

def simplify2(st: ShapeTracker) -> ShapeTracker:
    for i in range(2, len(st.views) + 1):
        if len((st2 := merge_views(ShapeTracker(st.views[-i:]))).views) < i:
            return simplify2(ShapeTracker(st.views[:-i] + st2.views))
    return st
