import unittest, math
from tinygrad import dtypes
from tinygrad.ops import UOp, UOps, BinaryOps, UnaryOps
from tinygrad.codegen.uopgraph import full_graph_rewrite

class TestGraphRewrite(unittest.TestCase):
  def test_full_graph_rewrite_division_by_zero(self):
    # Create CONST UOps
    const_uop_num = UOp(UOps.CONST, dtype=dtypes.float32, arg=10.0)
    const_uop_den = UOp(UOps.CONST, dtype=dtypes.float32, arg=0.0)
    # Create a DIV UOp
    div_uop = UOp(UOps.ALU, dtype=dtypes.float32, src=(const_uop_num, const_uop_den), arg=BinaryOps.IDIV)
    # Create a SINK UOp that depends on the division
    sink = UOp(UOps.SINK, dtypes.void, src=(div_uop,))
    # Apply full_graph_rewrite
    optimized_sink = full_graph_rewrite(sink)
    # Check that the division by zero is handled appropriately
    optimized_div_uop = optimized_sink.src[0]
    assert optimized_div_uop.op == UOps.CONST
    assert math.isinf(optimized_div_uop.arg) or math.isnan(optimized_div_uop.arg)

  def test_full_graph_rewrite_redundant_operations(self):
    # Create CONST UOps
    const_uop_1 = UOp(UOps.CONST, dtype=dtypes.float32, arg=10.0)
    const_zero = UOp(UOps.CONST, dtype=dtypes.float32, arg=0.0)
    const_one = UOp(UOps.CONST, dtype=dtypes.float32, arg=1.0)
    # Create redundant operations: (10.0 + 0.0) * 1.0
    add_uop = UOp(UOps.ALU, dtype=dtypes.float32, src=(const_uop_1, const_zero), arg=BinaryOps.ADD)
    mul_uop = UOp(UOps.ALU, dtype=dtypes.float32, src=(add_uop, const_one), arg=BinaryOps.MUL)
    # Create a SINK UOp that depends on the MUL
    sink = UOp(UOps.SINK, dtypes.void, src=(mul_uop,))
    # Apply full_graph_rewrite
    optimized_sink = full_graph_rewrite(sink)
    # Verify that the redundant operations are eliminated
    optimized_uop = optimized_sink.src[0]
    assert optimized_uop.op == UOps.CONST
    assert optimized_uop.arg == 10.0  # (10.0 + 0.0) * 1.0 = 10.0

  def test_full_graph_rewrite_large_graph(self):
    # Create a large chain of operations
    uop_chain = []
    # Initialize prev_uop to 0 instead of 1
    prev_uop = UOp(UOps.CONST, dtype=dtypes.int32, arg=0)
    for i in range(1, 101):  # Sum numbers from 1 to 100
        const_uop = UOp(UOps.CONST, dtype=dtypes.int32, arg=i)
        add_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(prev_uop, const_uop), arg=BinaryOps.ADD)
        uop_chain.append(add_uop)
        prev_uop = add_uop
    # Create a SINK UOp that depends on the last operation
    sink = UOp(UOps.SINK, dtypes.void, src=(prev_uop,))
    # Apply full_graph_rewrite
    optimized_sink = full_graph_rewrite(sink)
    # Check that the optimized graph contains a CONST with the correct accumulated value
    optimized_uop = optimized_sink.src[0]
    assert optimized_uop.op == UOps.CONST
    # The accumulated sum from 1 to 100
    expected_sum = sum(range(1, 101))  # Should be 5050
    assert optimized_uop.arg == expected_sum, f"Expected {expected_sum}, got {optimized_uop.arg}"

  def test_full_graph_rewrite_division_by_one(self):
    # Create CONST UOps
    const_uop_num = UOp(UOps.CONST, dtype=dtypes.float32, arg=42.0)
    const_uop_one = UOp(UOps.CONST, dtype=dtypes.float32, arg=1.0)
    # Create a DIV UOp
    div_uop = UOp(UOps.ALU, dtype=dtypes.float32, src=(const_uop_num, const_uop_one), arg=BinaryOps.IDIV)
    # Create a SINK UOp that depends on the division
    sink = UOp(UOps.SINK, dtypes.void, src=(div_uop,))
    # Apply full_graph_rewrite
    optimized_sink = full_graph_rewrite(sink)
    # Check that the division by one is simplified
    optimized_uop = optimized_sink.src[0]
    assert optimized_uop.op == UOps.CONST
    assert optimized_uop.arg == 42.0

  def test_full_graph_rewrite_modulo_by_one(self):
    # Create CONST UOps
    const_uop_num = UOp(UOps.CONST, dtype=dtypes.int32, arg=42)
    const_uop_one = UOp(UOps.CONST, dtype=dtypes.int32, arg=1)
    # Create a MOD UOp
    mod_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(const_uop_num, const_uop_one), arg=BinaryOps.MOD)
    # Create a SINK UOp that depends on the modulo
    sink = UOp(UOps.SINK, dtypes.void, src=(mod_uop,))
    # Apply full_graph_rewrite
    optimized_sink = full_graph_rewrite(sink)
    # Modulo by one should result in zero
    optimized_uop = optimized_sink.src[0]
    assert optimized_uop.op == UOps.CONST
    assert optimized_uop.arg == 0

  def test_full_graph_rewrite_transcendental_edge_cases(self):
    # Create a CONST UOp with a negative value
    const_neg = UOp(UOps.CONST, dtype=dtypes.float32, arg=-1.0)
    # Apply LOG2 to the negative constant
    log2_neg_uop = UOp(UOps.ALU, dtype=dtypes.float32, src=(const_neg,), arg=UnaryOps.LOG2)

    # Create a CONST UOp with zero
    const_zero = UOp(UOps.CONST, dtype=dtypes.float32, arg=0.0)
    # Apply RECIP to zero
    recip_zero_uop = UOp(UOps.ALU, dtype=dtypes.float32, src=(const_zero,), arg=UnaryOps.RECIP)

    # Combine both operations in a SINK
    sink = UOp(UOps.SINK, dtypes.void, src=(log2_neg_uop, recip_zero_uop))

    # Apply full_graph_rewrite
    optimized_sink = full_graph_rewrite(sink)

    # Extract the optimized UOps
    optimized_log2_neg = optimized_sink.src[0]
    optimized_recip_zero = optimized_sink.src[1]

    # Check that log2(-1.0) results in NaN
    assert optimized_log2_neg.op == UOps.CONST
    assert math.isnan(optimized_log2_neg.arg), f"Expected NaN for log2(-1.0), got {optimized_log2_neg.arg}"

    # Check that recip(0.0) results in infinity with the correct sign
    assert optimized_recip_zero.op == UOps.CONST
    assert math.isinf(optimized_recip_zero.arg), f"Expected infinity for recip(0.0), got {optimized_recip_zero.arg}"
    assert optimized_recip_zero.arg > 0, "Expected positive infinity for recip(0.0)"

  @unittest.skip("currently failing")
  def test_full_graph_rewrite_nested_loop_collapse(self):
    # Define constants for loop ranges
    outer_start = UOp.const(dtypes.int32, 0)
    outer_end = UOp.const(dtypes.int32, 8)
    inner_start = UOp.const(dtypes.int32, 0)
    inner_end = UOp.const(dtypes.int32, 4)

    # Create RANGE UOps for the loops
    outer_range = UOp(UOps.RANGE, src=(outer_start, outer_end), arg=(0,))
    inner_range = UOp(UOps.RANGE, src=(inner_start, inner_end), arg=(1,))

    # Indices for the loops
    idx_outer = outer_range
    idx_inner = inner_range

    # Compute operation: result = (idx_outer * 10) + idx_inner
    mul_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(idx_outer, UOp.const(dtypes.int32, 10)), arg=BinaryOps.MUL)
    add_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(mul_uop, idx_inner), arg=BinaryOps.ADD)

    # Reduce over both loops (sum all results)
    reduce_uop = UOp(UOps.REDUCE, dtype=dtypes.int32, src=(add_uop, outer_range, inner_range), arg=BinaryOps.ADD)

    # Create a SINK UOp that depends on the reduction
    sink = UOp(UOps.SINK, dtypes.void, src=(reduce_uop,))

    # Apply full_graph_rewrite
    optimized_sink = full_graph_rewrite(sink)

    # Extract the optimized UOp
    optimized_reduce_uop = optimized_sink.src[0]

    # Verify that the reduction is computed at compile time
    assert optimized_reduce_uop.op == UOps.CONST, "Expected reduction to be computed at compile time."

    # Manually compute the expected sum
    expected_sum = sum((i * 10) + j for i in range(8) for j in range(4))
    assert optimized_reduce_uop.arg == expected_sum, f"Expected sum {expected_sum}, got {optimized_reduce_uop.arg}"

  def test_full_graph_rewrite_modulo_folding_with_define_var(self):
    # Define a symbolic variable 'x' with known range
    x_var_uop = UOp(UOps.DEFINE_VAR, dtype=dtypes.int32, arg=('x', UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 100)))

    # Create an expression: (x * 4 + 2) % 4
    const_4 = UOp.const(dtypes.int32, 4)
    const_2 = UOp.const(dtypes.int32, 2)
    mul_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(x_var_uop, const_4), arg=BinaryOps.MUL)
    add_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(mul_uop, const_2), arg=BinaryOps.ADD)
    mod_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(add_uop, const_4), arg=BinaryOps.MOD)

    # Create a SINK UOp that depends on the modulo operation
    sink = UOp(UOps.SINK, dtypes.void, src=(mod_uop,))

    # Apply full_graph_rewrite
    optimized_sink = full_graph_rewrite(sink)

    # Extract the optimized UOp
    optimized_mod_uop = optimized_sink.src[0]

    # Expected behavior: (x * 4 + 2) % 4 simplifies to 2 % 4 == 2
    # Because (x * 4) % 4 == 0
    assert optimized_mod_uop.op == UOps.CONST, "Expected modulo operation to be simplified to a constant."
    assert optimized_mod_uop.arg == 2, f"Expected modulo result to be 2, got {optimized_mod_uop.arg}"

  def test_full_graph_rewrite_division_folding_with_define_var(self):
    # Define a symbolic variable 'n' with known range
    n_var_uop = UOp(UOps.DEFINE_VAR, dtype=dtypes.int32, arg=('n', UOp.const(dtypes.int32, 1), UOp.const(dtypes.int32, 1000)))

    # Create constants
    const_6 = UOp.const(dtypes.int32, 6)
    const_3 = UOp.const(dtypes.int32, 3)

    # Create an expression: (n * 6) // 3
    mul_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(n_var_uop, const_6), arg=BinaryOps.MUL)
    div_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(mul_uop, const_3), arg=BinaryOps.IDIV)

    # Create a SINK UOp that depends on the division
    sink = UOp(UOps.SINK, dtype=dtypes.void, src=(div_uop,))

    # Apply full_graph_rewrite
    optimized_sink = full_graph_rewrite(sink)

    # Extract the optimized UOp
    optimized_div_uop = optimized_sink.src[0]

    # Expected behavior: (n * 6) // 3 simplifies to n * 2
    # Because 6 // 3 == 2

    # Check that the optimizer simplified the division to a multiplication
    assert optimized_div_uop.op == UOps.ALU, "Expected the division to be simplified."
    assert optimized_div_uop.arg == BinaryOps.MUL, "Expected the operation to be a multiplication after simplification."
    # Check that the multiplication involves n_var_uop and a constant 2
    multiplier = optimized_div_uop.src[1]
    assert multiplier.op == UOps.CONST and multiplier.arg == 2, f"Expected multiplier to be 2, got {multiplier.arg}"

  def test_full_graph_rewrite_complex_mod_div_folding(self):
    # Define a symbolic variable 'k' with known range
    k_var_uop = UOp(UOps.DEFINE_VAR, dtype=dtypes.int32, arg=('k', UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 50)))

    # Create constants
    const_12 = UOp.const(dtypes.int32, 12)
    const_8 = UOp.const(dtypes.int32, 8)
    const_6 = UOp.const(dtypes.int32, 6)
    const_2 = UOp.const(dtypes.int32, 2)

    # Create an expression: ((k * 12 + 8) % 6) // 2
    mul_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(k_var_uop, const_12), arg=BinaryOps.MUL)
    add_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(mul_uop, const_8), arg=BinaryOps.ADD)
    mod_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(add_uop, const_6), arg=BinaryOps.MOD)
    div_uop = UOp(UOps.ALU, dtype=dtypes.int32, src=(mod_uop, const_2), arg=BinaryOps.IDIV)

    # Create a SINK UOp that depends on the division
    sink = UOp(UOps.SINK, dtype=dtypes.void, src=(div_uop,))

    # Apply full_graph_rewrite
    optimized_sink = full_graph_rewrite(sink)

    # Extract the optimized UOp
    optimized_div_uop = optimized_sink.src[0]

    # Expected behavior: ((k * 12 + 8) % 6) // 2 simplifies to 1
    # Because (k * 12) % 6 == 0, so the expression simplifies to (8 % 6) // 2 == 2 // 2 == 1

    # Verify that the optimizer simplified the expression to a constant
    assert optimized_div_uop.op == UOps.CONST, "Expected the entire expression to be simplified to a constant."
    assert optimized_div_uop.arg == 1, f"Expected result to be 1, got {optimized_div_uop.arg}"

if __name__ == '__main__':
  unittest.main()
