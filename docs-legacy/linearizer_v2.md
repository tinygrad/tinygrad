At base, the Linearizer a function that takes an AST + opts -> uops
It should be rewritten like this. The AST can't be a LazyOp, because it should be able to have multiple outputs

We need a generic class to represent DAGs.
This refactor is probably a prereq for the new linearizer, and can be used on existing uops also.
Can this class also represent the large graph? The op graph is a subset of the large graph.

Currently the Linearizer is merging many concerns:

1. LocalBuffers are added. These should be added to the upper DAG, for both grouping and tensor cores. Some opts are used here. NOTE: currently reduce splitting is done in lazy.py and it shouldn't be
2. The ShapeTrackers at the edges are collected and modified according to the other opts.
3. The Ops are toposorted.
4. The Ops are lowered to UOps. This requires expansion and loop assignment, potentially to global dimensions
5. The indexes into the Tensor are computed from the shapetrackers

More generically, the whole network is a DAG. Ignore the forward/backward stuff, I'm fine with starting at the LazyBuffer level.

1. Is it possible to put an entire network in a single kernel? I think the answer has to be yes, but you may end up doing an absolutely crazy amount of recomputation. This should still be doable to check correctness.
2. You can use intermediate buffers, be they local or global, to do less compute.

This is a rewrite of a lot of tinygrad. I don't think continuing to support Interpreted backends is worth it, have to deal with disk in a smart way.

We keep the features and nn stuff = 793 lines
We keep the frontend (Tensor -> LazyBuffer): tensor.py + mlops.py + lazy.py + dtype.py = 1032 lines
We keep the shapetracker/symbolic (part of the frontend): shapetracker.py + view.py + symbolic.py = 603 lines
Codegen is all rewritten. realize.py is simpler with the new codegen
We keep the backend (uops renderer/runtime): cstyle.py/llvmir.py + device.py + ops_*.py = 1216 lines (less when we remove interpreted)
