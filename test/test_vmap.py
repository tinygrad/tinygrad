from tinygrad.vmap import vmap
from tinygrad.tensor import Tensor

if __name__ == "__main__":
    print("--- Test 3: Multiple Outputs ---")
    
    @vmap(in_axes=(0, None))
    def power_and_diff(x, val):
        return x * val, x - val

    x = Tensor([1, 2, 3]).contiguous()
    val = Tensor([10]).contiguous()
    
    out_mul, out_sub = power_and_diff(x, val)
    
    res_mul = out_mul.realize().numpy()
    res_sub = out_sub.realize().numpy()
    
    print(f"Mul: {res_mul.flatten()}")
    print(f"Sub: {res_sub.flatten()}")
    
    assert res_mul[1] == 20
    assert res_sub[1] == -8
    print("Test 3 OK! (Tuples with multiple outputs)")