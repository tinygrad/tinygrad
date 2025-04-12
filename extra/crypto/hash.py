from tinygrad.dtype import DType
from tinygrad.ops import sint
from tinygrad import Tensor, dtypes

# TODO: Remove
import numpy as np
np.set_printoptions(formatter={'int':lambda x:hex(int(x))})

# TODO: Get rid of useless contiguous
# TODO: Get rid of the comments
# TODO: Reduce LOC
# TODO: Optional, other modes, out_len, not for ml
# TODO: Op reduction
# TODO: Better log test + Graph maybe
# TODO: Bandwidth test/analysis (TinyJit warmup)

# **************** Constants ****************

# *** Blake3 ***

IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
            0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)
MSG_PERM = Tensor([2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8], dtype=dtypes.uint32)
CHUNK_START, CHUNK_END = 0x01, 0x02
NODE_PARENT, NODE_ROOT = 0x04, 0x08
COLUMNS = [(0,4,8,12,0,1), (1,5,9,13,2,3), (2,6,10,14,4,5), (3,7,11,15,6,7)]
DIAGONALS = [(0,5,10,15,8,9), (1,6,11,12,10,11), (2,7,8,13,12,13), (3,4,9,14,14,15)]

# **************** Helper Functions ****************

# *** Blake3 ***

def check_tensor(*requirements:tuple[Tensor, tuple[sint, ...], DType, str]) -> None:
  def check(x, shape, dtype, name): assert x.shape == shape and x.dtype == dtype, f"{name} requirements: {shape}.{dtype} does not match {x.shape}.{x.dtype}"
  list(map(lambda r: check(r[0], r[1], r[2], r[3]), requirements))
def rotate_right(x:Tensor, bits:int) -> Tensor: return (x.rshift(bits)) | (x.lshift(x.dtype.itemsize * 8 - bits))

def g(state:Tensor, a:int, b:int, c:int, d:int, x:Tensor, y:Tensor) -> Tensor:
  assert len(state.shape); check_tensor((state, (bsize:=state.shape[0], 16), dtypes.uint32, "state"),
    (x, (bsize,), dtypes.uint32, "x"), (y, (bsize,), dtypes.uint32, "y"))

  for z in (x, y):
    state[:, a] = state[:, a] + state[:, b] + z
    state[:, d] = rotate_right(state[:, d] ^ state[:, a], 16 if z is x else 8)
    state[:, c] = state[:, c] + state[:, d]
    state[:, b] = rotate_right(state[:, b] ^ state[:, c], 12 if z is x else 7)
  return state # (bsize,16).uint32

def compress_block(block:Tensor, chain_val:Tensor, counter_low:Tensor, counter_high:Tensor, block_len:Tensor, flags:Tensor) -> Tensor:
  assert len(block.shape); check_tensor((block, (bsize:=block.shape[0], 16), dtypes.uint32, "block"), (chain_val, (bsize, 8), dtypes.uint32, "chain_val"),
    (counter_low, (bsize,), dtypes.uint32, "counter_low"), (counter_high, (bsize,), dtypes.uint32, "counter_high"),
    (block_len, (bsize,), dtypes.uint32, "block_len"), (flags, (bsize,), dtypes.uint32, "flags"))

  initials, configs = IV[:4].reshape(1, 4).expand(bsize,4), Tensor.stack(counter_low, counter_high, block_len, flags, dim=1)
  state = Tensor.cat(chain_val, initials, configs, dim=1).cast(dtypes.uint32)
  for r in range(7):
    for qr, (a,b,c,d,xi,yi) in enumerate([*COLUMNS, *DIAGONALS]): state = g(state, a, b, c, d, block[:, xi], block[:, yi])
    block = block[:, MSG_PERM]
  state[:, :8] = state[:, :8] ^ state[:, 8:16]
  state[:, 8:16] = state[:, 8:16] ^ chain_val
  return state.contiguous() # (bsize,16).uint32

def pack_blocks(blocks:Tensor) -> Tensor:
  assert len(blocks.shape); check_tensor((blocks, (num_blocks:=blocks.shape[0], 64), dtypes.uint8, "blocks"))

  block_words = Tensor.zeros(num_blocks, 16, dtype=dtypes.uint32).contiguous()
  for j in range(64): k = j//4; block_words[:, k] = block_words[:, k] | blocks[:, j].lshift((j % 4) * 8)
  return block_words # (num_blocks,16).uint32

def process_chunk(chunk:Tensor, chain_val:Tensor, chunk_idx:Tensor, flags:Tensor, root:bool=False) -> Tensor:
  assert len(chunk.shape) == 2 and (csize:=chunk.shape[1]) <= 1024
  check_tensor((chunk, (bsize:=chunk.shape[0], csize), dtypes.uint8, "chunk"), (chain_val, (bsize, 8), dtypes.uint32, "chain_val"),
    (chunk_idx, (bsize,), dtypes.uint32, "chunk_idx"), (flags, (bsize,), dtypes.uint32, "flags"))

  padding_size = (64 - csize % 64) % 64  # This gives 0 when csize is a multiple of 64
  # padded_chunk = chunk.pad((0, 0, 0, padding_size), value=0)
  padded_chunk = chunk.pad((0, padding_size), value=0)
  num_blocks = padded_chunk.shape[1] // 64
  block_words = pack_blocks(padded_chunk.reshape(bsize * num_blocks, 64)).reshape(bsize, num_blocks, 16)
  chain_val_out = chain_val.clone() # TODO: Consider reassigning chain_val instead (cache, mem, and kernel advantage ?)
  for i in range(num_blocks):
    block_flags = flags
    if root and i == num_blocks - 1: block_flags = block_flags | NODE_ROOT
    block_flags = block_flags | (CHUNK_START if i == 0 else 0)
    block_flags = block_flags | (CHUNK_END if i == num_blocks - 1 else 0)
    block_len = Tensor.full((bsize,), min(64, csize - i * 64), dtype=dtypes.uint32)
    chain_val_out = compress_block(block_words[:, i], chain_val_out, chunk_idx, Tensor.zeros(bsize, dtype=dtypes.uint32), block_len, block_flags)[:, :8]
  return chain_val_out # (bsize,8).uint32

def process_parent(left:Tensor, right:Tensor, flags:Tensor) -> Tensor:
  check_tensor((left, (bsize:=left.shape[0], 8), dtypes.uint32, "left"), (right, (bsize, 8), dtypes.uint32, "right"), (flags, (bsize,), dtypes.uint32, "flags"))

  blocks = Tensor.cat(left, right, dim=1).cast(dtypes.uint32)
  return compress_block(blocks, IV.expand(bsize, 8), Tensor.zeros(bsize, dtype=dtypes.uint32), Tensor.zeros(bsize, dtype=dtypes.uint32),
                        Tensor.full((bsize,), 64, dtype=dtypes.uint32), flags | NODE_PARENT)[:, :8].contiguous() # (bsize,8).uint32

# **************** Hash Functions ****************

def blake3(msg:Tensor, max_batch_size:int=None) -> Tensor:
  if not isinstance(msg, Tensor): msg = Tensor(np.frombuffer(msg, dtype=np.uint8))
  if msg.dtype != dtypes.uint8: msg = msg.cast(dtypes.uint8)
  assert len(msg.shape) == 1, "blake3 does not support batched inputs"

  num_chunks = (msg.shape[0] + 1023) // 1024
  chunks = msg.pad((0, num_chunks * 1024 - msg.shape[0]), value=0).reshape(num_chunks, 1024)
  chunk_hashes = Tensor.zeros((num_chunks, 8), dtype=dtypes.uint32).contiguous()
  chunk_flags = Tensor.zeros((num_chunks,), dtype=dtypes.uint32)
  chunk_indices = Tensor.arange(num_chunks, dtype=dtypes.uint32)

  last_chunk_complete = (msg.shape[0] % 1024 == 0)
  last_chunk_idx = num_chunks - 1

  if max_batch_size is None: batch_size = num_chunks if last_chunk_complete and num_chunks > 1 else last_chunk_idx
  else: batch_size = min(max_batch_size, num_chunks if last_chunk_complete and num_chunks > 1 else last_chunk_idx)

  for i in range(0, last_chunk_idx if batch_size else 0, batch_size if batch_size else 1):
    batch_end = min(i + batch_size, last_chunk_idx)
    batch_chunks = chunks[i:batch_end]
    batch_indices = chunk_indices[i:batch_end]
    batch_flags = chunk_flags[i:batch_end]
    batch_chain_vals = IV.expand(batch_end - i, 8)
    chunk_hashes[i:batch_end] = process_chunk(batch_chunks, batch_chain_vals, batch_indices, batch_flags)

  if (not last_chunk_complete and num_chunks > 0) or num_chunks == 1:
    last_chunk_size = msg.shape[0] % 1024 if not last_chunk_complete else 1024
    last_chunk = chunks[last_chunk_idx:last_chunk_idx+1, :last_chunk_size]
    last_index = chunk_indices[last_chunk_idx:last_chunk_idx+1]
    last_flag = chunk_flags[last_chunk_idx:last_chunk_idx+1]
    last_chain_val = IV.expand(1, 8)
    chunk_hashes[last_chunk_idx:last_chunk_idx+1] = process_chunk(last_chunk, last_chain_val, last_index, last_flag, num_chunks==1)#[0] TODO: Enable when only one chunk?

  while chunk_hashes.shape[0] > 1:
    num_parents = (chunk_hashes.shape[0] + 1) // 2
    parents = Tensor.zeros((num_parents, 8), dtype=dtypes.uint32).contiguous()
    pair_count = chunk_hashes.shape[0] // 2
    batch_left = chunk_hashes[0:-1:2] if chunk_hashes.shape[0] % 2 else chunk_hashes[::2]
    batch_right = chunk_hashes[1::2]
    parent_flags = Tensor.full((pair_count,), NODE_ROOT if num_parents == 1 else 0, dtype=dtypes.uint32)
    parents[:pair_count] = process_parent(batch_left, batch_right, parent_flags)
    if chunk_hashes.shape[0] % 2: parents[-1] = chunk_hashes[-1]
    chunk_hashes = parents

  assert chunk_hashes.shape[0] == 1, "Final result should be a single hash"
  return chunk_hashes[0]
