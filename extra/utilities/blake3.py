import math
from typing import Callable, Tuple
from tinygrad.dtype import dtypes
from tinygrad.engine import jit
from tinygrad.helpers import ceildiv
from tinygrad.tensor import Tensor

PAD, DEFAULT_LEN = 66, 65 # symbolic constants for masks

@jit.TinyJit
def permute_data(data: Tensor) -> Tensor: data[:] = data[[2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]]

@jit.TinyJit
def mix(states: Tensor, chunks: Tensor) -> Tensor:
  def rotr(x: Tensor, n: int) -> Tensor: return ((x << (32 - n)) | (x >> n))
  for i, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
    mx, my = chunks[i * 2], chunks[i * 2 + 1]
    for m in (mx, my):
      states[a] = states[a] + states[b] + m
      states[d] = rotr(states[d] ^ states[a], 16 if m is mx else 8)
      states[c] = states[c] + states[d]
      states[b] = rotr(states[b] ^ states[c], 12 if m is mx else 7)
  return states.realize()

def compress_blocks(states: Tensor, data: Tensor, chain_vals: Tensor) -> Tensor:
  for _ in range(6):
    mix(states, data)
    permute_data(data)
  mix(states, data)
  states[:8] = states[:8] ^ states[8:]
  states[8:] = chain_vals[:8] ^ states[8:]
  return states.realize()

def compress_chunks(states: Tensor, data: Tensor, chain_vals: Tensor, info: Tensor) -> Tensor:
  for i in range(16): # parallel over chunks, sequential over blocks
    compressed = compress_blocks(states[i].contiguous(), data[i].contiguous(), chain_vals[i].contiguous()).clone().realize()
    if i < data.shape[1] - 1:
      states[i, :] = compressed
      states[i + 1, :8] = compressed[:8] # propagate chaining values
  states = states * (info < PAD)
  end_block = (states * (info < DEFAULT_LEN)).sum(0) # pick out partial chunk end block
  return (states[-1, :] | end_block)[:8] # last block of each chunk + partial chunk end block

def tensor_to_blake_data(tensor: Tensor, max_memory: int) -> Tuple[Tensor, Tensor]:
  assert max_memory % 1024 == 0, "max_memory must be divisible by 1024"
  data = tensor.flatten().pad(((0, (max_memory // tensor.element_size()) - tensor.shape[0],),), value=0)
  data = data.bitcast(dtypes.uint32).reshape(-1, 16, 16).permute(1, 2, 0).contiguous()
  final_chunk_len = 0 if tensor.nbytes() == 0 else (tensor.nbytes() % 1024 or 1024)
  n_end_blocks = ceildiv(final_chunk_len, 64) or 1
  n_chunks = max(1, ceildiv(tensor.nbytes(), 1024))
  info = Tensor.full((16, 1, data.shape[-1]), fill_value=DEFAULT_LEN, dtype=dtypes.uint32).contiguous()
  info[n_end_blocks - 1, :, n_chunks - 1] = 0 if tensor.nbytes() == 0 else (tensor.nbytes() % 64) or 64
  info[n_end_blocks:, :, n_chunks - 1:] = info[:, :, n_chunks:] = PAD
  n_steps = math.ceil(math.log2(max(n_chunks, 1)))
  return data.contiguous(), info.contiguous(), n_steps

@jit.TinyJit
def pairwise_concat(chain_vals: Tensor) -> Tuple[Tensor, Tensor]:
  cv_pairs_with_leftover = chain_vals.permute(1, 0).reshape(-1, 16).transpose()
  paired = chain_vals.any(0).reshape(-1, 2)
  paired_mask, leftover_mask = (paired[:, 0] * paired[:, 1]), (paired[:, 0] ^ paired[:, 1])
  cv_pairs = (cv_pairs_with_leftover * paired_mask).pad(((0, 0), (0, paired.shape[0])), value=0)
  leftover = (cv_pairs_with_leftover * leftover_mask).max(1)[:8].reshape(-1, 1)
  return cv_pairs, leftover

def create_state(iv: Tensor, counts: Tensor, info: Tensor, flags: Tensor) -> Tensor:
  counts = counts.cat(Tensor.zeros(iv.shape[0], 1, iv.shape[-1], dtype=dtypes.uint32), dim=1)
  lengths = (info == DEFAULT_LEN).where(64, info) # set default lengths
  states = iv.cat(iv[:, :4], counts, lengths, flags, dim=1) # create states
  return states * (info < PAD).cast(dtypes.uint32) # zero out padding

def create_flags(info: Tensor, parents: Tensor, final_step: Tensor) -> Tensor:
  flags = Tensor.zeros((16, 1, info.shape[-1]), dtype=dtypes.uint32).contiguous()
  flags[-1, 0] = flags[-1, 0] + 2 # chunk end flag
  flags = (flags + 2 * ((flags != 2) * (info < DEFAULT_LEN))) # chunk end flag for partial final chunk
  flags[0] = flags[0] + 1 # chunk start flag
  flags = parents.where(4, flags) # parent flag
  flags = (flags + (8 * (((info < PAD).sum() <= 16) * (info < DEFAULT_LEN)))) # add root flag if <= 1 chunk
  flags = final_step.where(12, flags) * (info < PAD) # add final step flag and zero out padding
  return flags.cast(dtypes.uint32)

def compress(data: Tensor, compressor: Callable, counts: Tensor, info: Tensor, parents: Tensor, final_step: Tensor) -> Tensor:
  IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)
  iv = IV.reshape(1, 8, 1).expand(16, 8, info.shape[-1]).contiguous()
  states = create_state(iv, counts, info, create_flags(info, parents, final_step))
  return compressor(states, data, iv, info)

def compress_tree(states: Tensor, data: Tensor, iv: Tensor, _) -> Tensor: return compress_blocks(states[-1].contiguous(), data, iv[0])

def blake3(tensor: Tensor, max_memory: int = 1024**3) -> str:
  data, info, n_steps = tensor_to_blake_data(tensor, max_memory)
  parents = final_step = Tensor.zeros((1,), dtype=dtypes.bool).contiguous()
  counts = Tensor.arange(0, data.shape[-1], dtype=dtypes.uint32).reshape(-1, 1).expand(-1, 16).reshape(-1, 16, 1).permute(1, 2, 0)
  chain_vals = compress(data, compress_chunks, counts, info, parents, final_step)
  info = (info < DEFAULT_LEN).where(64, info)
  counts, parents = Tensor.zeros((16, 1, data.shape[-1]), dtype=dtypes.uint32), Tensor.ones((1,), dtype=dtypes.bool)
  final_step = chain_vals.any(0).sum(-1) == 2
  for _ in range(n_steps): # tree-hash chain value pairs ~halving them in each step
    chain_vals, leftover_chain_val = pairwise_concat(chain_vals)
    valid = chain_vals.any(0)
    chain_vals = compress(chain_vals.contiguous(), compress_tree, counts, info, parents, final_step)
    chain_vals = (chain_vals[:8] * valid)
    insertion_mask = (valid ^ valid.roll(1, -1))
    insertion_mask[0] = 0
    chain_vals = insertion_mask.where(leftover_chain_val, chain_vals)
    final_step = chain_vals.any(0).sum(-1) == 2
  return chain_vals[:, 0].flatten().bitcast(dtypes.uint8).data().tobytes().hex()