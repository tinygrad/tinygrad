# the V3.1 token_embd (IQ4_XS): the prefill jits look tokens up in a table in host memory. The lazily dequantized IQ4_XS table is a jit intermediate
# (2.4 GiB in float: the serving start's MemoryError in make_arenas); the half table the loader makes (qwen4exp._dequant_rows_half) is read as it is.
import types, unittest
from unittest import mock
from tinygrad import Tensor, UOp, dtypes, nn
from tinygrad.llm import qwen4exp
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_QUANT

V, D = 4096, 2560

def _big_buffers(out:Tensor, weight:Tensor) -> list[int]:
  table = V*D*2
  sizes = {}
  for call in out.schedule_linear().src:
    for b in call.src[1:]:
      try: sizes[b.buffer] = b.buffer.nbytes
      except Exception: pass
  wb = weight.uop.base.buffer if weight.uop.base.op.name == "BUFFER" else None
  return sorted(n for b, n in sizes.items() if b is not wb and n >= table)

class TestIQ4XSGather(unittest.TestCase):
  def test_lazy_dequant_materializes_the_table(self):
    block, nb = _GGML_QUANT[23]
    w = ggml_data_to_tensor(Tensor.empty(V*D//block*nb, dtype=dtypes.uint8), V*D, 23).reshape(V, D).cast(dtypes.float16)
    e = nn.Embedding(V, D)
    e.weight = w
    self.assertTrue(_big_buffers(e(Tensor.empty(1, 64, dtype=dtypes.int32)).float(), w))  # (the bug: the float table is a buffer)

  def test_half_table_is_read_as_it_is(self):
    w = Tensor.empty(V, D, dtype=dtypes.float16)
    e = nn.Embedding(V, D)
    e.weight = w
    self.assertEqual(_big_buffers(e(Tensor.empty(1, 64, dtype=dtypes.int32)).float(), w), [])

  def test_decode_rows_come_from_the_iq4xs_kernel(self):
    # (decode / draft steps: static token counts -> iq4xs_embed; a symbolic count -> the generic lookup of the (half) table)
    m = qwen4exp.Transformer.__new__(qwen4exp.Transformer)
    m.token_embd = nn.Embedding(512, 64)
    m.config = types.SimpleNamespace(dim=64)
    m._embd_iq4xs = Tensor.empty(100, dtype=dtypes.uint8)
    calls = []
    def fake(raw, ids, dim):
      calls.append((raw, tuple(ids.shape), dim))
      return Tensor.empty(int(ids.shape[0]), dim)
    with mock.patch.object(qwen4exp, "amd_custom_kernels_supported", lambda d: True), mock.patch.object(qwen4exp, "iq4xs_embed", fake):
      out = m._embed(Tensor.empty(3, 5, dtype=dtypes.int32))
      self.assertEqual((calls[0][1], calls[0][2], out.shape), ((15,), 64, (3, 5, 64)))
      nt = UOp.variable("toks", 1, 16).bind(5)
      m._embed(Tensor.empty(1, 16, dtype=dtypes.int32)[:, :nt])
      self.assertEqual(len(calls), 1)

  def test_half_host_buffer_on_several_devices(self):
    # (the loader's buffer for the half table: bytes in, a replicated half tensor the lookup reads)
    for devs in (("NULL",), ("NULL:0", "NULL:1")):
      with self.subTest(devices=devs):
        mv, half = qwen4exp._host_buffer(devs, 2*V*D, dtypes.float16)
        self.assertEqual((mv.format, len(mv), half.reshape(V, D).shape, half.dtype), ("B", 2*V*D, (V, D), dtypes.float16))

if __name__ == "__main__": unittest.main()
