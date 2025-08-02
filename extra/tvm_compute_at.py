import tvm
from tvm import te, tir

n   = te.var("n")
A   = te.placeholder((n,), name="A")
B   = te.compute((n,), lambda i: A[i] + 1, name="B")   # producer
C   = te.compute((n,), lambda i: B[i] * 2, name="C")   # consumer

prim_func = te.create_prim_func([A, C])
ir_mod    = tvm.IRModule({"main": prim_func})
sch = tir.Schedule(ir_mod, debug_mask="all")

blk_B = sch.get_block("B", func_name="main")
blk_C = sch.get_block("C", func_name="main")
i_loop = sch.get_loops(blk_C)[0]
sch.compute_at(blk_B, i_loop)

print(sch.mod.script())
