tinygrad is a bit bloated now, and there's several places where concerns should be seperated and they aren't.

tensor.py and mlops.py are great code. The interface going backward here is:

LazyBuffer.const (this creates a matching size buffer)
LazyBuffer.contiguous (tbis is not exactly elementwise)
LazyBuffer.e (elementwise)
LazyBuffer.r (reduce)
reshape/permute/expand/stride/shrink/pad (movement)

The lazy.py reordering engine has a lot of junk to deal with movementops that should be removed.

view.py is mostly great code, except it shouldn't have the rendering logic, and the int type should be parameterized to not import from symbolic.

LazyOp shouldn't have LazyBuffers as sources, just LazyOp LoadOps with a tuple of Views. Then the LazyOp uniquely determines the kernel and we don't have to do any replacement.

ShapeTracker probably shouldn't exist and just be a part of LazyBuffer. Most of the stuff in ShapeTracker should move to symbolic_view, which combines view and symbolic.
