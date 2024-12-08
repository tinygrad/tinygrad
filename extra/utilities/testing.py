import math
import numpy as np
from extra.utilities.blake3 import create_flags, create_state, pairwise_concat, tensor_to_blake_input
from tinygrad import Variable
from tinygrad.dtype import dtypes
from tinygrad.engine import jit
from tinygrad.tensor import Tensor

PAD, DEFAULT_LEN, PERMS = 66, 65, Tensor([2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8], dtype=dtypes.uint32)
IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)

def compress_blocks(states: Tensor, data: Tensor, chain_vals: Tensor) -> Tensor:
  for i in range(7):
    for j, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
      mx, my = data[j * 2], data[j * 2 + 1]
      for m in (mx, my):
        states[a] = (states[a] + states[b] + m)
        states[d] = ((states[d] ^ states[a]) << (32 - (16 if m is mx else 8))) | ((states[d] ^ states[a]) >> (16 if m is mx else 8))
        states[c] = states[c] + states[d]
        states[b] = ((states[b] ^ states[c]) << (32 - (12 if m is mx else 7))) | ((states[b] ^ states[c]) >> (12 if m is mx else 7))
    if i < 6: data = data[PERMS]
  return (states[:8] ^ states[8:]).cat(chain_vals[:8] ^ states[8:]).realize()


# 3072 bytes
example_input_cvs_1 = Tensor(
 [[1147510364, 3089309531, 3191123761, 0], [3516130065, 2724725561, 4156599190, 0], [281526004, 2442481857, 727455631, 0], [1455216076, 3083962627, 1414001530, 0], [2098703209, 2790451990, 370737570, 0], [3054373473, 172872671, 1480840121, 0], [3925802129, 2720164518, 1875932204, 0], [2984817711, 2803660315, 2822509831, 0]], 
 dtype=dtypes.uint32
)
example_input_len_1 = 3
expected_1 = Tensor([4289760441,   62792502, 1027042098, 1376369126, 1681805592, 3710840561, 2915509541, 3539950622], dtype=dtypes.uint32)

# 2048 bytes
example_input_cvs_2 = Tensor(
  [[1147510364, 3089309531,          0 , 0],
       [3516130065, 2724725561,          0 , 0],
       [ 281526004, 2442481857,          0 , 0],
       [1455216076, 3083962627,          0 , 0],
       [2098703209, 2790451990,          0 , 0],
       [3054373473,  172872671,          0 , 0],
       [3925802129, 2720164518,          0 , 0],
       [2984817711, 2803660315,          0, 0]],
  dtype=dtypes.uint32
)
example_input_len_2 = 2
expected_2 = Tensor([  45512423,  718437516, 2191592269,  543342504, 1995779677, 2390982214, 2605904598, 1252195205], dtype=dtypes.uint32)

# tasks
# - [x] test that it works on real inputs
# - [] add it back into the original code, check tests pass

def tree_step(chain_vals: Tensor) -> Tensor:
    final_step = (chain_vals[:, :3].any(0).sum() <= 2)
    stacked = chain_vals.transpose().reshape(-1, 16).transpose().reshape(2, 8, -1)
    stacked_mask = stacked.any(1)
    pair_mask, remainder_mask = (stacked_mask[0] * stacked_mask[1]), (stacked_mask[0] ^ stacked_mask[1])
    paired, remainder = (stacked * pair_mask).reshape(16, -1), (stacked * remainder_mask).reshape(16, -1)[:8]
    # states
    iv = IV.reshape(1, 8, 1).expand(16, 8, paired.shape[-1])
    counts = Tensor.zeros((16, 2, paired.shape[-1]), dtype=dtypes.uint32)
    flags =  Tensor.full((16, 1, paired.shape[-1]), 4, dtype=dtypes.uint32)
    flags = final_step.where(12, flags)
    lengths = Tensor.full((16, 1, paired.shape[-1]), 64, dtype=dtypes.uint32)
    states = (iv.cat(iv[:, :4], counts, lengths, flags, dim=1))
    # compress
    compressed = (compress_blocks(states[-1].contiguous(), paired, iv[0]) * pair_mask)[:8]
    chain_vals = compressed + remainder
    return chain_vals

@jit.TinyJit
def tree_hash(chain_vals: Tensor, n_steps: Variable) -> Tensor:
    for _ in range(n_steps.val):
        chain_vals = tree_step(chain_vals)
    return chain_vals.realize()
    
# # Test runs
# print(test_add(Tensor.ones((2,))).numpy())

n_steps = Variable(min_val=0, max_val=5, name="n").bind(math.ceil(math.log2(max(example_input_len_1, 1))))
result = tree_hash(example_input_cvs_1, n_steps)
result = tree_hash(example_input_cvs_1, n_steps)
result = tree_hash(example_input_cvs_1, n_steps)
np.testing.assert_array_equal(result[:, 0].numpy(), expected_1.numpy())

n_steps = Variable(min_val=0, max_val=5, name="n").bind(math.ceil(math.log2(max(example_input_len_2, 1))))
result = tree_hash(example_input_cvs_2, n_steps)
np.testing.assert_array_equal(result[:, 0].numpy(), expected_2.numpy())


# input_len = 2**4 + 1 # 17
# input_t = Tensor(([1] * input_len))
# input_t = input_t.pad(((0, 64 - input_len),)).contiguous()
# n_steps = Variable(min_val=0, max_val=5, name="n").bind(math.ceil(math.log2(max(input_len, 1))))
# x = tree_hash(input_t, n_steps)
# x = tree_hash(input_t, n_steps)
# x = tree_hash(input_t, n_steps)
# result1 = x.numpy()
# np.testing.assert_equal(result1[0], input_len + 1)

# input_len = 2**3 + 1 # 9
# input_t = Tensor(([1] * input_len))
# input_t = input_t.pad(((0, 64 - input_len),)).contiguous()
# n_steps = Variable(min_val=0, max_val=5, name="n").bind(math.ceil(math.log2(max(input_len, 1))))
# y = tree_hash(input_t, n_steps)
# # y = tree_hash(input_t, n_steps)
# # y = tree_hash(input_t, n_steps)
# result2 = y.numpy()
# np.testing.assert_equal(result2[0], input_len + 1)

# n_steps = Variable(min_val=0, max_val=5, name="n").bind(0)
# y = tree_hash(Tensor([5]).pad(((0, 64 - 1),)).contiguous(), n_steps)
# np.testing.assert_equal(y.numpy()[0], 5)