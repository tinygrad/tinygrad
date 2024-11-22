import math
import os
import tempfile
from typing import Optional, Tuple, Union
import unittest
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

MD_LEN = 32
BLOCK_BYTES = 64
CHUNK_BYTES = 1024
CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
PARENT = 1 << 2
ROOT = 1 << 3
IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)
MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

def rotr(x: Tensor, n: int) -> Tensor: return ((x << (32 - n)) | (x >> n))

def mix(states: Tensor, a: int, b: int, c: int, d: int, mx: Tensor, my: Tensor) -> Tensor:
  states[:, a] = states[:, a] + (states[:, b] + mx)
  states[:, d] = rotr(states[:, d] ^ states[:, a], 16)
  states[:, c] = states[:, c] + states[:, d]
  states[:, b] = rotr(states[:, b] ^ states[:, c], 12)
  states[:, a] = states[:, a] + (states[:, b] + my)
  states[:, d] = rotr(states[:, d] ^ states[:, a], 8)
  states[:, c] = states[:, c] + states[:, d]
  states[:, b] = rotr(states[:, b] ^ states[:, c], 7)

def mix_round(states: Tensor, chunks: Tensor) -> Tensor:
  # mix columns
  mix(states, 0, 4, 8, 12, chunks[:, 0], chunks[:, 1])
  mix(states, 1, 5, 9, 13, chunks[:, 2], chunks[:, 3])
  mix(states, 2, 6, 10, 14, chunks[:, 4], chunks[:, 5])
  mix(states, 3, 7, 11, 15, chunks[:, 6], chunks[:, 7])
  # mix diagonals
  mix(states, 0, 5, 10, 15, chunks[:, 8], chunks[:, 9])
  mix(states, 1, 6, 11, 12, chunks[:, 10], chunks[:, 11])
  mix(states, 2, 7, 8, 13, chunks[:, 12], chunks[:, 13])
  mix(states, 3, 4, 9, 14, chunks[:, 14], chunks[:, 15])

def compress_chunks(states: Tensor, chunks: Tensor, chain_vals: Tensor, final_chunk_n_blocks: int):
  n_blocks = chunks.shape[1]
  for i in range(n_blocks):
    # parallel over chunks, sequential over blocks
    # must be sequential over blocks because of chain value dependencies between blocks
    compressed = compress_blocks(states[:, i].contiguous(), chunks[:, i].contiguous(), chain_vals[:, i].contiguous())
    next_chain_vals = compressed[:, :8]
    if i < n_blocks - 1:
      # add chain value dependencies to the next block
      states[:, i + 1, :8] = next_chain_vals
  if final_chunk_n_blocks == 16:
    return next_chain_vals
  else:
    # handle partial final chunk by returning latest chain values except for the last partial chunk
    cvs_for_full_chunks = next_chain_vals[:-1]
    partial_chunk_end_idx = final_chunk_n_blocks - 1 if chunks.shape[0] == 1 else final_chunk_n_blocks
    cv_for_partial_chunk = states[-1:, partial_chunk_end_idx, :8]
    return cvs_for_full_chunks.cat(cv_for_partial_chunk)

def compress_blocks(states: Tensor, chunks: Tensor, chain_vals: Tensor) -> Tensor:
  for _ in range(6):
    mix_round(states, chunks)
    chunks.replace(chunks[:, MSG_PERMUTATION])
  mix_round(states, chunks)
  states[:, :8] = states[:, :8] ^ states[:, 8:]
  states[:, 8:] = chain_vals[:, :8] ^ states[:, 8:]
  return states

def bytes_to_chunks(text_bytes: bytes) -> Tuple[Tensor, int, int]:
  n_bytes = len(text_bytes)
  n_chunks = max((n_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES, 1)
  n_blocks = CHUNK_BYTES // BLOCK_BYTES
  n_words = BLOCK_BYTES // 4
  chunks = Tensor.zeros(n_chunks, n_blocks, n_words, dtype=dtypes.uint32).contiguous()
  unpadded_len = 0
  # chunks
  for i in range(0, max(len(text_bytes), 1), CHUNK_BYTES):
    chunk = text_bytes[i: i + CHUNK_BYTES]
    unpadded_len = len(chunk)
    chunk = chunk.ljust(CHUNK_BYTES, b"\0")
    # blocks
    for j in range(16):
      block_start = j * BLOCK_BYTES
      bw_bytes = chunk[block_start:block_start + BLOCK_BYTES]
      # words
      block_words = [int.from_bytes(bw_bytes[i: i + 4], "little") for i in range(0, len(bw_bytes), 4)]
      chunks[i // CHUNK_BYTES, j] = Tensor(block_words, dtype=dtypes.uint32)
  n_end_blocks = max(1, (unpadded_len // BLOCK_BYTES) + (1 if unpadded_len % BLOCK_BYTES else 0))
  end_block_len = BLOCK_BYTES if unpadded_len % BLOCK_BYTES == 0 and unpadded_len else unpadded_len % BLOCK_BYTES
  return chunks, n_end_blocks, end_block_len

def pairwise_concat(chain_vals: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
  if chain_vals.shape[0] % 2 != 0:
    leftover_chunk = chain_vals[-1:]
    chain_vals = chain_vals[:-1]
  else:
    leftover_chunk = None
  return chain_vals.reshape(-1, 16), leftover_chunk

def create_state(iv: Tensor, count: Optional[int], end_block_len: Optional[int], n_end_blocks: Optional[int], flags: Tensor) -> Tensor:
  n_chunks, n_blocks = iv.shape[0], iv.shape[1]
  if count is not None:
    counts = Tensor.arange(count, count + n_chunks, dtype=dtypes.uint32)
    counts = counts.reshape(n_chunks, 1).expand(n_chunks, n_blocks).reshape(n_chunks, n_blocks, 1)
    counts = counts.cat(Tensor.zeros(n_chunks, n_blocks, 1, dtype=dtypes.uint32), dim=-1)
  else:
    counts = Tensor.zeros(n_chunks, n_blocks, 2, dtype=dtypes.uint32)
  lengths = Tensor.full((n_chunks, n_blocks, 1), fill_value=BLOCK_BYTES, dtype=dtypes.uint32).contiguous()
  if end_block_len is not None: lengths[-1, n_end_blocks - 1] = end_block_len
  states = iv.cat(iv[:, :, :4], counts, lengths, flags, dim=-1)
  return states

def create_flags(chunks: Tensor, n_end_blocks: Optional[int], is_root: bool, are_parents: bool = False) -> Tensor:
  n_chunks, n_blocks = chunks.shape[0], chunks.shape[1]
  flags = Tensor.zeros((n_chunks, n_blocks, 1), dtype=dtypes.uint32).contiguous()
  flags[:, 0] = flags[:, 0] + CHUNK_START
  end_idx = n_end_blocks - 1 if n_end_blocks is not None else -1
  flags[:-1, -1] = flags[:-1, -1] + CHUNK_END
  flags[-1, end_idx:] = flags[-1, end_idx:] + CHUNK_END
  if are_parents: flags[:, :, :] = PARENT
  if is_root: flags[0, end_idx] = flags[0, end_idx] + ROOT
  return flags

def init_compress(input: Union[str, bytes], counter = 0, not_root: bool = False) -> Tensor:
  input_bytes = input.encode("utf-8") if isinstance(input, str) else input if isinstance(input, bytes) else b""
  chunks, n_end_blocks, end_block_len = bytes_to_chunks(input_bytes)
  n_chunks, n_blocks, _ = chunks.shape
  init_chain_vals = IV.expand(n_chunks, n_blocks, -1).contiguous()
  flags = create_flags(chunks, n_end_blocks, is_root=n_chunks == 1 and not not_root)
  states = create_state(init_chain_vals, counter, end_block_len, n_end_blocks, flags)
  return compress_chunks(states, chunks, init_chain_vals, n_end_blocks)

def init_compress_file(file: str, bufsize: int) -> Tensor:
  chain_vals = Tensor.zeros(0, 8, dtype=dtypes.uint32)
  file_size = os.path.getsize(file)
  with open(file, "rb") as f:
    while chunk_bytes := f.read(bufsize):
      counter = chain_vals.shape[0]
      chain_vals = chain_vals.cat(init_compress(chunk_bytes, counter, not_root=file_size > CHUNK_BYTES))
  return chain_vals

def blake3(text: Optional[Union[str, bytes]] = None, file: Optional[str] = None, bufsize: int = 8 * 1024 * 1024) -> str:
  """
  Hash an input string, bytes, or file-like object in parallel using the BLAKE3 hashing algorithm.
  When hashing a file, the file is read in chunks of `bufsize` bytes.
  """
  assert text is not None or file is not None, "Either text or a file must be provided"
  assert bufsize % CHUNK_BYTES == 0, f"bufsize must be a multiple of {CHUNK_BYTES}"
  # compress input chunks into an initial set of chain values
  chain_vals = init_compress(text) if isinstance(text, (str, bytes)) else init_compress_file(file, bufsize)
  tree_levels = math.ceil(math.log2(max(chain_vals.shape[0], 1)))
  # tree hash pairs of chain values until only one remains
  for i in range(tree_levels):
    chain_vals, leftover_chain_val = pairwise_concat(chain_vals)
    n_chain_vals, n_blocks = chain_vals.shape[0], chain_vals.shape[1]
    init_chain_vals = IV.expand(n_chain_vals, n_blocks, -1).contiguous()
    flags = create_flags(chain_vals, None, are_parents=True, is_root=i == tree_levels - 1)
    states = create_state(init_chain_vals, None, None, None, flags)[:, -1].contiguous()
    chain_vals = compress_blocks(states, chain_vals, init_chain_vals[:, 0])[:, :8]
    if leftover_chain_val is not None: chain_vals = chain_vals.cat(leftover_chain_val, dim=0)
  return chain_vals[0].flatten().numpy().tobytes()[:MD_LEN].hex()

class TestBLAKE3(unittest.TestCase):
    def setUp(self):
        self.vectors = [
          {
            "input_len": 0,
            "hash": "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
          },
          {
            "input_len": 1,
            "hash": "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"
          },
          {
            "input_len": 2,
            "hash": "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63"
          },
          {
            "input_len": 3,
            "hash": "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f"
          },
          {
            "input_len": 4,
            "hash": "f30f5ab28fe047904037f77b6da4fea1e27241c5d132638d8bedce9d40494f32"
          },
          {
            "input_len": 5,
            "hash": "b40b44dfd97e7a84a996a91af8b85188c66c126940ba7aad2e7ae6b385402aa2"
          },
          {
            "input_len": 6,
            "hash": "06c4e8ffb6872fad96f9aaca5eee1553eb62aed0ad7198cef42e87f6a616c844"
          },
          {
            "input_len": 7,
            "hash": "3f8770f387faad08faa9d8414e9f449ac68e6ff0417f673f602a646a891419fe"
          },
          {
            "input_len": 8,
            "hash": "2351207d04fc16ade43ccab08600939c7c1fa70a5c0aaca76063d04c3228eaeb"
          },
          {
            "input_len": 63,
            "hash": "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b"
          },
          {
            "input_len": 64,
            "hash": "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"
          },
          {
            "input_len": 65,
            "hash": "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee"
          },
          {
            "input_len": 127,
            "hash": "d81293fda863f008c09e92fc382a81f5a0b4a1251cba1634016a0f86a6bd640d"
          },
          {
            "input_len": 128,
            "hash": "f17e570564b26578c33bb7f44643f539624b05df1a76c81f30acd548c44b45ef"
          },
          {
            "input_len": 129,
            "hash": "683aaae9f3c5ba37eaaf072aed0f9e30bac0865137bae68b1fde4ca2aebdcb12"
          },
          {
            "input_len": 1023,
            "hash": "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"
          },
          {
            "input_len": 1024,
            "hash": "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"
          },
          {
            "input_len": 2048,
            "hash": "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a"
          },
          {
            "input_len": 2049,
            "hash": "5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b6879522563030"
          },
          {
            "input_len": 3072,
            "hash": "b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd2"
          },
          {
            "input_len": 3073,
            "hash": "7124b49501012f81cc7f11ca069ec9226cecb8a2c850cfe644e327d22d3e1cd3"
          },
          {
            "input_len": 4096,
            "hash": "015094013f57a5277b59d8475c0501042c0b642e531b0a1c8f58d2163229e969"
          },
          {
            "input_len": 4097,
            "hash": "9b4052b38f1c5fc8b1f9ff7ac7b27cd242487b3d890d15c96a1c25b8aa0fb995"
          },
          {
            "input_len": 5120,
            "hash": "9cadc15fed8b5d854562b26a9536d9707cadeda9b143978f319ab34230535833"
          },
          {
            "input_len": 5121,
            "hash": "628bd2cb2004694adaab7bbd778a25df25c47b9d4155a55f8fbd79f2fe154cff"
          },
          {
            "input_len": 6144,
            "hash": "3e2e5b74e048f3add6d21faab3f83aa44d3b2278afb83b80b3c35164ebeca205"
          },
          {
            "input_len": 6145,
            "hash": "f1323a8631446cc50536a9f705ee5cb619424d46887f3c376c695b70e0f0507f"
          },
          {
            "input_len": 7168,
            "hash": "61da957ec2499a95d6b8023e2b0e604ec7f6b50e80a9678b89d2628e99ada77a"
          },
          {
            "input_len": 7169,
            "hash": "a003fc7a51754a9b3c7fae0367ab3d782dccf28855a03d435f8cfe74605e7817"
          },
          {
            "input_len": 8192,
            "hash": "aae792484c8efe4f19e2ca7d371d8c467ffb10748d8a5a1ae579948f718a2a63"
          },
          {
            "input_len": 8193,
            "hash": "bab6c09cb8ce8cf459261398d2e7aef35700bf488116ceb94a36d0f5f1b7bc3b"
          },
          {
            "input_len": 16384,
            "hash": "f875d6646de28985646f34ee13be9a576fd515f76b5b0a26bb324735041ddde4"
          },
          {
            "input_len": 31744,
            "hash": "62b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47"
          },
          {
            "input_len": 102400,
            "hash": "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085"
          }
        ]

    def generate_input(self, length: int) -> bytes:
      return bytes(i % 251 for i in range(length))

    def test_official_vectors(self):
      """Test against the official test vectors from: https://github.com/BLAKE3-team/BLAKE"""
      for vector in self.vectors:
          input_len = vector["input_len"]
          expected = vector["hash"]
          input = self.generate_input(input_len)
          actual = blake3(input)
          self.assertEqual(actual, expected)

    def test_file_input(self):
      # all full-length chunks
      with tempfile.NamedTemporaryFile(delete=True) as file:
        file.write(self.generate_input(102400))
        file.flush()
        actual = blake3(file=file.name, bufsize=1024 * 50) # expect 2 reads
        self.assertEqual(actual, "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085")
      # partial last chunk
      with tempfile.NamedTemporaryFile(delete=True) as file:
        file.write(self.generate_input(4097))
        file.flush()
        actual = blake3(file=file.name, bufsize=1024) # expect 5 reads
        self.assertEqual(actual, "9b4052b38f1c5fc8b1f9ff7ac7b27cd242487b3d890d15c96a1c25b8aa0fb995")

if __name__ == "__main__":
  unittest.main()
