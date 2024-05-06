import unittest
from tinygrad import dtypes, Variable
from tinygrad.ops import BinaryOps, TernaryOps
from tinygrad.codegen.uops import UOpGraph, UOps

class TestUOpGraph(unittest.TestCase):
  def test_add_constant_fold(self):
    g = UOpGraph()
    c1 = g.add(UOps.CONST, dtypes.float, arg=1.0)
    c2 = g.add(UOps.CONST, dtypes.float, arg=2.0)
    out = g.add(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.ADD)
    g.remove_childless({out})
    self.assertEqual(len(g.uops), 1)
    self.assertEqual(out.uop, UOps.CONST)
    self.assertEqual(out.arg, 3.0)

  def test_where_same_fold(self):
    g = UOpGraph()
    v = g.add(UOps.DEFINE_VAR, dtypes.int, arg=Variable('tmp', 0, 1))
    c0 = g.add(UOps.CONST, dtypes.int, arg=0)
    vc = g.add(UOps.ALU, dtypes.bool, (v, c0), BinaryOps.CMPEQ)
    c1 = g.add(UOps.CONST, dtypes.float, arg=1.0)
    out = g.add(UOps.ALU, dtypes.float, (vc, c1, c1), TernaryOps.WHERE)
    g.remove_childless({out})
    self.assertEqual(len(g.uops), 1)
    self.assertEqual(out.uop, UOps.CONST)
    self.assertEqual(out.arg, 1.0)

  def test_where_const_fold(self):
    g = UOpGraph()
    bf = g.add(UOps.CONST, dtypes.bool, arg=False)
    c1 = g.add(UOps.CONST, dtypes.float, arg=1.0)
    c2 = g.add(UOps.CONST, dtypes.float, arg=2.0)
    out = g.add(UOps.ALU, dtypes.float, (bf, c1, c2), TernaryOps.WHERE)
    g.remove_childless({out})
    self.assertEqual(len(g.uops), 1)
    self.assertEqual(out.uop, UOps.CONST)
    self.assertEqual(out.arg, 2.0)

  def test_const_cast(self):
    g = UOpGraph()
    bf = g.add(UOps.CONST, dtypes.bool, arg=False)
    out = g.add(UOps.CAST, dtypes.int, (bf,))
    g.remove_childless({out})
    self.assertEqual(len(g.uops), 1)
    self.assertEqual(out.uop, UOps.CONST)
    self.assertEqual(out.arg, 0)

  def test_loop_scope_load_local(self):
    g = UOpGraph()
    # def indecies and buffers
    def_global = g.add(UOps.DEFINE_GLOBAL, dtypes.float, arg=None)
    def_local = g.add(UOps.DEFINE_LOCAL, dtypes.float, arg=None)
    idx_local = g.add(UOps.SPECIAL, dtypes.int, arg=None)

    # def constants
    one = g.add(UOps.CONST, dtypes.int, arg=1)
    N = g.add(UOps.CONST, dtypes.float, arg=1/64)
    four = g.add(UOps.SPECIAL, dtypes.int, arg=4)
    zero = g.add(UOps.CONST, dtypes.int, arg=0)
    sixteen = g.add(UOps.CONST, dtypes.int, arg=16)

    # first reduction
    acc0 = g.add(UOps.DEFINE_ACC, dtypes.float, arg=0.0, cachable=False)
    loop0 = g.add(UOps.LOOP, dtypes.int, (zero, four), cachable=False)
    load0 = g.add(UOps.LOAD, dtypes.float, (def_global, loop0))
    alu0 = g.add(UOps.ALU, dtypes.float, (acc0, load0), arg=BinaryOps.ADD)
    phi0 = g.add(UOps.PHI, dtypes.float, (acc0, alu0, loop0))

    # first reduction (local)
    store_local0 = g.add(UOps.STORE, dtypes.float, (def_local, idx_local, phi0))
    barrier0 = g.add(UOps.BARRIER, vin=(store_local0,), cachable=False)
    # barrier MUST be included in if_cond other it gets optimized away
    if_cond0 = g.add(UOps.IF, None, (g.add(UOps.ALU, dtypes.bool, (idx_local, one), arg=BinaryOps.CMPLT), barrier0), cachable=False)
    acc1 = g.add(UOps.DEFINE_ACC, dtypes.float, arg=0.0, cachable=False)
    loop1 = g.add(UOps.LOOP, dtypes.int, (zero, sixteen), cachable=False)
    # the if_cond MUST be included in global_load( ... barrier=if_cond), otherwise it gets optimized away
    load1 = g.add(UOps.LOAD, dtypes.float, (def_local, loop1, if_cond0))
    alu1 = g.add(UOps.ALU, dtypes.float, (acc1, load1), arg=BinaryOps.ADD)
    phi1 = g.add(UOps.PHI, dtypes.float, (acc1, alu1, loop1))

    # store & load back into every thread
    store_local1 = g.add(UOps.STORE, dtypes.float, (def_local, zero, phi1))
    endif0 = g.add(UOps.ENDIF, vin=(if_cond0,)) # need to put endif here manually, tacking it on the end wrecks scope
    barrier1 = g.add(UOps.BARRIER, vin=(store_local1,endif0), cachable=False) # endif has to be included here so it doesn't get optimized away
    load_local0 = g.add(UOps.LOAD, dtypes.float, (def_local, zero, barrier1)) # global_load with barrier=barrier1 again


    # second reduction
    acc2 = g.add(UOps.DEFINE_ACC, dtypes.float, arg=0.0, cachable=False)
    loop2 = g.add(UOps.LOOP, dtypes.int, (zero, four), cachable=False)
    load2 = g.add(UOps.LOAD, dtypes.float, (def_global, loop2))
    alu2 = g.add(UOps.ALU, dtypes.float, (load_local0, N), arg=BinaryOps.MUL) # this should get lifted out of the loop
    alu3 = g.add(UOps.ALU, dtypes.float, (load2, alu2), arg=BinaryOps.SUB) # this cannot get lifted out of the loop
    alu4 = g.add(UOps.ALU, dtypes.float, (acc2, alu3), arg=BinaryOps.ADD)
    phi2 = g.add(UOps.PHI, dtypes.float, (acc2, alu4, loop2))

    # second reduction (local)
    store_local2 = g.add(UOps.STORE, dtypes.float, (def_local, idx_local, phi2))
    barrier2 = g.add(UOps.BARRIER, vin=(store_local2,), cachable=False)
    if_cond1 = g.add(UOps.IF, None, (g.add(UOps.ALU, dtypes.bool, (idx_local, one), arg=BinaryOps.CMPLT), barrier2), cachable=False) # ""
    acc3 = g.add(UOps.DEFINE_ACC, dtypes.float, arg=0.0, cachable=False)
    loop3 = g.add(UOps.LOOP, dtypes.int, (zero, sixteen), cachable=False)
    load3 = g.add(UOps.LOAD, dtypes.float, (def_local, loop3, if_cond1)) # ""
    alu5 = g.add(UOps.ALU, dtypes.float, (acc3, load3), arg=BinaryOps.ADD)
    phi3 = g.add(UOps.PHI, dtypes.float, (acc3, alu5, loop3))

    # store & load back into every thread
    store_local3 = g.add(UOps.STORE, dtypes.float, (def_local, zero, phi3))
    endif1 = g.add(UOps.ENDIF, vin=(if_cond1,)) # ""
    barrier3 = g.add(UOps.BARRIER, vin=(store_local3, endif1), cachable=False) # ""
    load_local1 = g.add(UOps.LOAD, dtypes.float, (def_local, zero, barrier3)) # ""

    # third reduction
    acc4 = g.add(UOps.DEFINE_ACC, dtypes.float, arg=0.0, cachable=False)
    loop4 = g.add(UOps.LOOP, dtypes.int, (zero, four), cachable=False)
    load4 = g.add(UOps.LOAD, dtypes.float, (def_global, loop4))
    alu6 = g.add(UOps.ALU, dtypes.float, (load_local0, N), arg=BinaryOps.MUL) # should get merged with alu2 since alu2 gets lifted out of the loop
    alu7 = g.add(UOps.ALU, dtypes.float, (load_local1, N), arg=BinaryOps.MUL)
    alu8 = g.add(UOps.ALU, dtypes.float, (load4, alu6), arg=BinaryOps.SUB) # alu3 cannot get lifted out of it's loop, so this should not be merged
    alu9 = g.add(UOps.ALU, dtypes.float, (alu8, alu7), arg=BinaryOps.DIV)
    alu10 = g.add(UOps.ALU, dtypes.float, (acc4, alu9), arg=BinaryOps.ADD)
    phi4 = g.add(UOps.PHI, dtypes.float, (acc4, alu10, loop4))

    # third reduction (local)
    store_local4 = g.add(UOps.STORE, dtypes.float, (def_local, idx_local, phi4))
    barrier4 = g.add(UOps.BARRIER, vin=(store_local4,), cachable=False)
    if_cond3 = g.add(UOps.IF, None, (g.add(UOps.ALU, dtypes.bool, (idx_local, one), arg=BinaryOps.CMPLT), barrier4), cachable=False) # ""
    acc5 = g.add(UOps.DEFINE_ACC, dtypes.float, arg=0.0, cachable=False)
    loop5 = g.add(UOps.LOOP, dtypes.int, (zero, sixteen), cachable=False)
    load5 = g.add(UOps.LOAD, dtypes.float, (def_local, loop5, if_cond3)) # ""
    alu11 = g.add(UOps.ALU, dtypes.float, (acc5, load5), arg=BinaryOps.ADD)
    phi5 = g.add(UOps.PHI, dtypes.float, (acc5, alu11, loop5))
    g.add(UOps.STORE, dtypes.float, (def_global, zero, phi5))

    g.uoptimize()

    # first reduction locals and if_cond
    self.assertLess(g.uops.index(store_local0), g.uops.index(barrier0), "barrier must follow local store")
    self.assertLess(g.uops.index(barrier0), g.uops.index(if_cond0), "if statement must follow barrier and local store")
    self.assertLess(g.uops.index(if_cond0), g.uops.index(loop1), "local loop must be within the if statement")
    self.assertLess(g.uops.index(loop1), g.uops.index(load1), "loading the local buffer must take place within the local loop")

    # loading the result of the first local reduction back into every thread
    self.assertEqual(g.uops[g.uops.index(store_local1)-1].uop, UOps.ENDLOOP, \
                     "storing the result of the local reduction should follow the end of the loop")
    self.assertLess(g.uops.index(store_local1), g.uops.index(endif0), \
                    "should store the result of the local reduction before ending the if statement")
    self.assertLess(g.uops.index(endif0), g.uops.index(barrier1), "should put a barrier after the local store and in every thread")
    self.assertLess(g.uops.index(barrier1), g.uops.index(load_local0), "should load back the result of the local reduction after the barrier")

    # second reduction
    self.assertLess(g.uops.index(load_local0), g.uops.index(loop2), \
                    "the result of the first reduction should be loaded into every thread before begining any other reductions")
    self.assertIn(acc2, g.uops, "There should be individual accumulators for each reduction")
    self.assertLess(g.uops.index(alu2), g.uops.index(loop2), \
                    "uops on the result of a reduction and in the output shape should be lifted out of loops")
    self.assertLess(g.uops.index(loop2), g.uops.index(alu3), \
                    "uops on the result of a reduction but in the full shape shoud NOT get lifted out of loops")

    # second reduction (local)
    self.assertEqual(g.uops[g.uops.index(store_local2)-1].uop, UOps.ENDLOOP, "second store must be outside the end of the second loop")
    self.assertLess(g.uops.index(store_local2), g.uops.index(barrier2), "barrier must follow the local store")
    self.assertLess(g.uops.index(barrier2), g.uops.index(if_cond1), "if statement must follow barrier and local store")
    self.assertLess(g.uops.index(if_cond1), g.uops.index(loop3), "local loop must be within the if statement")
    self.assertLess(g.uops.index(loop2), g.uops.index(load3), "loading the local buffer must take place within the local loop")

    # loading the result of the second local reduction back into every thread
    self.assertEqual(g.uops[g.uops.index(store_local3)-1].uop, UOps.ENDLOOP, \
                     "storing the result of the local reduction should follow the end of the loop")
    self.assertLess(g.uops.index(store_local3), g.uops.index(endif1), \
                    "should store the result of the local reduction before ending the if statement")
    self.assertLess(g.uops.index(endif1), g.uops.index(barrier3), \
                    "should put a barrier after the local store and in every thread")
    self.assertLess(g.uops.index(barrier3), g.uops.index(load_local1), \
                    "should load back the result of the local reduction after the barrier")

    # third reduction
    self.assertLess(g.uops.index(load_local1), g.uops.index(loop4), \
                    "the result of the second reduction should be loaded into every thread before begining any other reductions")
    self.assertLess(g.uops.index(alu7), g.uops.index(loop4), \
                    "uops on the result of a reduction and in the output shape should be lifted out of loops")
    self.assertEqual(alu2, alu6, "alu6 should have been merged with alu2 after it got lifted out of it's loop")
    self.assertNotEqual(alu3, alu8, "alu8 should not get merged with alu3 because alu3 could not get lifted out of it's loop")
    self.assertIn(alu2, alu8.vin, "alu6 should have gotten merged with alu2 and replaced in the inputs for alu8")
    self.assertLess(g.uops.index(loop4), g.uops.index(alu8), "alu8 should still be within it's loop")
    self.assertLess(g.uops.index(loop4), g.uops.index(alu9), "operations with inputs from both in and out of the loop should be in the loop")

if __name__ == '__main__':
  unittest.main(verbosity=2)
