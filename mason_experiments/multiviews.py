from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad import Tensor
from pprint import pprint
from itertools import product
from functools import cache
import numpy as np
# Example of multiple views being created in the ShapeTracker

def realize_view(view):

    def dot(u: list[int], v: list[int]) -> int:
        return sum(ui*vi for ui,vi in zip(u,v))
    
    data = [i for i in range(view.size())]
    ret = np.zeros((view.shape))
    for indexes in product(*map(range,view.shape)):
        index = dot(indexes,view.strides) + view.offset
        #print(f"{index=} for {len(data)=}")
        ret[indexes] = data[index] if (not view.mask or all(iN in range(view.mask[iN][0],view.mask[iN[1]]) for iN in indexes)) else 0
    return ret

@cache
def all_possible_views(prod: int) -> list[ShapeTracker]:
    # splits the product into all possible contiguous divisions
    factors = []
    possible_factor = 2
    while possible_factor * possible_factor <= prod:
        if prod % possible_factor == 0:
            (left,right) = possible_factor, prod//possible_factor
            factors.append((left,right))
            factors.append((right,left))
        possible_factor += 1

    views = set()
    for left,right in factors:
        for l,r in product(all_possible_views(left),[right]):
            views.add(l + (r,))
        views.add((left,right))
        for l,r in product([left],all_possible_views(right)):
            views.add((l,) + r)
    return views

def one_to_four_to_one() -> ShapeTracker:
    "Returns ShapeTracker with len(views) going from 1 -> 2 -> 3 -> 4 -> 3 -> 2 -> 1"
    a = ShapeTracker.from_shape((10,10)); a = a.permute((1,0)) # 1
    a = a.reshape((25,4)); a = a.permute((1,0)) # 2
    a = a.reshape((50,2)); a = a.permute((1,0)) # 3

    a = a.reshape((10,10)) # 4

    a = a.reshape((2,50)); a = a.simplify(); a = a.permute((1,0)) # 3
    a = a.reshape((4,25)); a = a.simplify(); a = a.permute((1,0)) # 2
    a = a.reshape((10,10)); a = a.simplify(); a = a.permute((1,0)) # 1

    return a
