tinygrad is a bit bloated now, and there's several places where concerns should be seperated and they aren't.

tensor.py and mlops.py are great code. The interface going backward here is:

LazyBuffer.e (elementwise)
LazyBuffer.r (reduce)
reshape/permute/expand/stride/shrink/pad (movement)
