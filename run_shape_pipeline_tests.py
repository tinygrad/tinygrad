# run_shape_pipeline_tests.py
# -------------------------------------------------------
# Quick validation for tinygrad/shape_pipeline.py
# -------------------------------------------------------
from tinygrad.shape_pipeline import normalize_shape_pipeline

def test_case(shape, reduce_n=0, gpu_dims=()):
    meta = normalize_shape_pipeline(shape, reduce_n=reduce_n, gpu_dims=gpu_dims)
    print(f"Input shape: {shape}")
    print(f"  Logical (yellow): {meta.logical}")
    print(f"  Reduced (red):    {meta.reduced}")
    print(f"  Physical (green): {meta.physical}")
    print("-" * 50)

if __name__ == "__main__":
    print("=== Running shape pipeline validation ===\n")

    # 1️⃣ Simple case: remove 1s
    test_case((1, 32, 1, 64))

    # 2️⃣ Reduction over rightmost dims
    test_case((1, 32, 1, 64), reduce_n=1)

    # 3️⃣ Add GPU-special dimensions
    test_case((1, 32, 1, 64), reduce_n=1, gpu_dims=(8,))

    # 4️⃣ Multiple reductions and GPU dims
    test_case((1, 16, 1, 8, 1), reduce_n=2, gpu_dims=(4, 4))

    # 5️⃣ Edge case: no reduction, no 1s
    test_case((2, 3, 4))

