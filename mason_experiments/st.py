from dataclasses import dataclass
from functools import reduce
from operator import mul
"""
A ShapeTracker is a list of Views
A View is a view of a Tensor. The Key of a View is that assuming underlying Tensor data is accurate, it will accurately build the Tensor from the 
stream of entries the Tensor contains. 
A View needs then:
    shape
    strides
    offest
    mask
    contiguous
"""

def prod(shape: tuple[int]): return reduce(mul, shape, 1)

@dataclass
class Tensor:
    data: list
    
    @property
    def size(self) -> tuple[int,...]:
        return len(self.data)

    def reshape(self, shape: tuple[int]):
        it = iter(self.data)
        def helper(shape) -> list[list,...]:
            first, rest = shape[0], shape[1:]
            if rest: return [helper(rest) for _ in range(first))]
            else: return [next(it) for _ in range(first)]
        return Tensor(helper(shape))
        
    
a = range(24)
assert len(a) == 24

T = Tensor(a)
print(list(T.data))
T = T.reshape((6,2,2))
print(T.data)
