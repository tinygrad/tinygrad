# DISABLE_CPU_CACHE=1 METAL=1 python3 tinygrad/shape/mergeable.py
# Alternates between incorrect and correct behavior when cache is enabled on M1 Mac...

from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Node, NumNode
from tinygrad import Tensor


def to_tensor(exp: Node, shape, vars = None):
    if vars is None:
        vars = list(exp.vars())

    if len(vars) == 0:
        return None

    indices = [list() for _ in vars]

    for i, v in enumerate(vars):
        if i == 0:
            indices[0] = [[x] for x in list(range(v.min, v.max + 1))]
            continue

        for j in range(v.min, v.max + 1):
            indices[i] += [x + [j] for x in indices[i-1]]
            
    indices[-1].sort()

    ans = []
    for i in range(len(indices[-1])):
        ans.append(exp.substitute(dict(zip(vars, map(NumNode, indices[-1][i])))).b)

    return Tensor(ans).reshape(shape).numpy()



def is_single_view(s: ShapeTracker, lower_corner=None):
    x = to_tensor(s.expr_idxs()[0], s.shape, s.expr_idxs()[0].vars())

    lower_corner = None
    N = len(x.shape)
    if lower_corner is None:
        lower_corner = tuple([0]*N)

    strides = []
    for i in range(len(x.shape)):
        strides.append(x[tuple([lower_corner[i] if j != i else lower_corner[i] + 1 for j in range(N)])] - x[lower_corner])

    offset = int((Tensor(strides) * Tensor(list(lower_corner))).sum().numpy())
    s2 = ShapeTracker(views=(View.create(shape=x.shape, strides=tuple(strides), offset=offset),))

    try:
        return (to_tensor(s.expr_idxs()[0], s.shape, s.expr_idxs()[0].vars()) == to_tensor(s2.expr_idxs()[0], s2.shape, s2.expr_idxs()[0].vars())).all()
    except Exception as e:
        #print(e)
        return False

def is_rectangular(x: Tensor):
    return True  #TODO



s = ShapeTracker.from_shape((2,4))
print(is_single_view(s))
print(is_single_view(ShapeTracker.from_shape((2,4)).permute((1,0)).reshape((2,4))))

