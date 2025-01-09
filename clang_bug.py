# run this script with BEAM=2 on master branch commit 83a8217cbf36f663cdfd0aa08c91c5349f21c01a
# silent exception at line 80 of tinygrad.runtime.ops_clang.py, set breakpoint with condition: bufs[0]._length_ == 2097152 and bufs[1]._length_ == 8192 and bufs[2]._length_ == 430062 and bufs[3]._length_ == 2 and bufs[4]._length_ == 4 and bufs[5]._length_ == 32768

from tinygrad import Tensor, dtypes, Device
from extra.export_model import jit_model
from typing import NamedTuple, List, Any


# from nn.state.ggml_data_to_tensor
def q_to_uint8(t: Tensor, b: int) -> Tensor:
  # TODO: rewrite with arange?
  shift_tensor, bitmask = Tensor.stack(*[ Tensor(2**(i*b), device=t.device, dtype=t.dtype) for i in range(8//b) ]), 0xff >> (8 - b)
  return t.unsqueeze(-1).expand((*t.shape,8//b)).idiv(shift_tensor).bitwise_and(bitmask).transpose(-1, -2).flatten(-2)

# adapted from nn.state.ggml_data_to_tensor
# uint8 (256 elements per 210 bytes) to float32
def q6k_to_f32(x: Tensor) -> Tensor:
  blocks = x.reshape((-1, 210))
  xl, xh = q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4), q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
  scales = blocks[:,192:208].bitcast(dtypes.int8).unsqueeze(-1).expand((-1, 16, 16)).reshape((-1, 256))
  d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32).expand((-1, 256))
  return (d * (xl.bitwise_or(xh).bitcast(dtypes.int8) - 32).flatten(-2) * scales).cast(dtypes.float32).flatten()

def clang_q6k_to_f32():
  Device.DEFAULT="CLANG"
  class Step(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None

  step = Step(
    name = "q6k_to_f32", 
    input = [Tensor.randn(430_080, dtype=dtypes.uint8).realize()], # llama-3.2-1B Q6_K weights are multiples of 430_080 bytes
    forward = q6k_to_f32,
  )
  
  run, special_names = jit_model(step, *step.input)

if __name__=="__main__":
  # run this script with BEAM=2 on master branch commit 83a8217cbf36f663cdfd0aa08c91c5349f21c01a
  # silent exception at line 80 of tinygrad.runtime.ops_clang.py, set breakpoint with condition: bufs[0]._length_ == 2097152 and bufs[1]._length_ == 8192 and bufs[2]._length_ == 430062 and bufs[3]._length_ == 2 and bufs[4]._length_ == 4 and bufs[5]._length_ == 32768
  clang_q6k_to_f32()