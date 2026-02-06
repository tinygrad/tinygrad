# run_shape_tests.py
print("=== Running shape_utils tests ===")

from tinygrad.shape_utils import normalize_shape, broadcast_shape, reduce_chomp, add_gpu_dims

def assert_equal(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: {a} != {b}")

def run_tests():
    print("=== Running shape_utils tests ===")

    # --- normalize_shape ---
    assert_equal(normalize_shape((1, 32, 1, 64)), (32, 64), "normalize removes 1s")
    assert_equal(normalize_shape((1, 1, 1)), (), "normalize all 1s -> empty tuple")

    # --- broadcast_shape ---
    assert_equal(broadcast_shape((3, 1), (1, 5)), (3, 5), "broadcast basic")
    assert_equal(broadcast_shape((2, 3, 1), (1, 1, 4)), (2, 3, 4), "broadcast rank align")
    assert_equal(broadcast_shape((4, 5), (4, 5)), (4, 5), "broadcast equal shapes")

    # --- reduce_chomp ---
    assert_equal(reduce_chomp((10, 20, 30), 1), (10, 20), "reduce rightmost 1 dim")
    assert_equal(reduce_chomp((5, 6, 7, 8), 2), (5, 6), "reduce rightmost 2 dims")
    try:
        reduce_chomp((2, 3), 3)
    except AssertionError:
        print("✅ reduce_chomp rejects invalid n")
    else:
        raise AssertionError("❌ reduce_chomp should raise for n > len(shape)")

    # --- add_gpu_dims ---
    assert_equal(add_gpu_dims((16, 32), (8, 4)), (16, 32, 8, 4), "append GPU dims")
    assert_equal(add_gpu_dims((), (2,)), (2,), "append GPU dims to scalar")

    # --- full pipeline combo ---
    shape = (1, 4, 1, 5)
    norm = normalize_shape(shape)
    reduced = reduce_chomp(norm, 1)
    gpu = add_gpu_dims(reduced, (8, 8))
    assert_equal(gpu, (4, 8, 8), "pipeline combo")

    print("\n🎉 All shape_utils tests passed!\n")

if __name__ == "__main__":
    run_tests()
