from examples.tinychat.compile import q6k_to_f32
import os
from examples.llama3 import build_transformer
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor
from tinygrad import Device
from tinygrad.helpers import fetch
import ctypes
import numpy as np

"""
# from inspecting llama-3.2-1B q6_k gguf weights
q6k_min_size = 860160
q6k_max_size = 215470080 # token_embd.weight
q6k_size = Variable("q6k_size", q6k_min_size, q6k_max_size).bind(q6k_max_size)
q6k_shape1_min = 512
q6k_shape1_max = 128256
q6k_shape1 = Variable("q6k_shape1", q6k_shape1_min, q6k_shape1_max).bind(128256) # belongs to q6k_max_size (token_embd.weight)
q6k_shape0_min = 2048
q6k_shape0_max = 8192
q6k_shape0 = Variable("q6k_shape0", q6k_shape0_min, q6k_shape0_max).bind(2048) # belongs to q6k_max_size (token_embd.weight)
"""
# let sizes be a list of num_elements for each of the q6k weights:
# functools.reduce(math.gcd, sizes) = 524288
# functools.reduce(math.gcd, sizes) // 256 = 2048
# functools.reduce(math.gcd, sizes) // 256 * 210 = 430080
# therefore, jit the q6k_to_f32 with a uint8 Tensor of 430_080 bytes, should work to decompress all the q6k weights we have

if __name__=="__main__":
  model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf", "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct")
  model_size="1B"
  Tensor.no_grad = True
  max_context=1024

  # byte location of token_embd.weight in the model_path file
  data_start = 7831552
  start = data_start + 128
  end = data_start + 215470208
  x = bytes(open(model_path, 'rb').read())[start:end]
  y = Tensor.stack(*[q6k_to_f32(Tensor(x[i: i+430080], device="WEBGPU"), target="WEBGPU").realize() for i in range(0, len(x), 430080)])

  """
  # For CLANG
  lib_path = os.path.join(os.path.dirname(__file__), "q6k_to_f32.so")
  lib = ctypes.CDLL(lib_path)
  lib.net.restype = None
  lib.net.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(ctypes.c_float)
  ]

  input_chunk_size = 430080
  output_chunk_size = 524288
  num_chunks = len(x) // input_chunk_size
  final_output = np.zeros(num_chunks * output_chunk_size, dtype=np.float32)

  for i in range(num_chunks):
    input_chunk = np.frombuffer(x[i * input_chunk_size:(i + 1) * input_chunk_size], dtype=np.uint8)
    output_chunk = np.zeros(output_chunk_size, dtype=np.float32)
    input_ptr = input_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    output_ptr = output_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.net(input_ptr, output_ptr)
    final_output[i * output_chunk_size:(i + 1) * output_chunk_size] = output_chunk
  """

  # say "hi" to the model
  toks = [128000, 128006, 882, 128007, 271, 6151, 128009, 128006, 78191, 128007, 271]
  #toks = toks + (max_context - len(toks)) * [1_000_000]

  model = build_transformer(model_path, model_size=model_size, max_context=max_context)
  out1 = model.forward(Tensor([toks]), 0, 1e-7, 0, 0.0, 0.0, 0.0).item() # low temp for non-random output
  assert out1 == 4438

  emb_shape = model.tok_embeddings.weight.shape
  y = y.reshape(emb_shape).to(Device.DEFAULT)
  model.tok_embeddings.weight.assign(y).realize()
  out2 = model.forward(Tensor([toks]), 0, 1e-7, 0, 0.0, 0.0, 0.0).item() # low temp for non-random output
  assert out2 == out1

  # confirm we get different output when reassigning with bad weights:
  #model.tok_embeddings.weight.assign(Tensor.zeros(emb_shape, dtype=dtypes.float32).realize()).realize()
  #out3 = model.forward(Tensor([toks]), 0, 1e-7, 0, 0.0, 0.0, 0.0, True).item() # low temp for non-random output
  #assert out3 != out1