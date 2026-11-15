# VISION for GLM 5.3 Flash (glm_vision's tower, glm5v sizing, glm5next's vision variant of a prefill chunk): what doesn't need the model
import array, io, types, unittest
import numpy as np
from tinygrad import Tensor, Context
from tinygrad.llm import qwen_vision
from tinygrad.llm.glm_vision import canvas_size, swiglu_clamp, FACTOR
from tinygrad.llm.qwen_vision import picture_ids

PAD, VS, VE = 154854, 154830, 154831

class TestGlmSizing(unittest.TestCase):
  def test_canvas(self):
    # (llama.cpp mtmd_image_preprocessor_glm5v) a Game Boy screenshot isn't resized: a 336 x 308 canvas, 12 x 11 rows
    self.assertEqual(canvas_size(288, 320), (308, 336, 288, 320))
    # (below the minimum: scaled up, ceil-aligned)
    self.assertEqual(canvas_size(64, 64), (112, 112, 112, 112))
    # (over the budget: the largest aligned canvas within it, the picture scaled down into it, never up)
    for h, w, mx in ((3000, 4000, 1024 * 28 * 28), (1080, 1920, 300 * 28 * 28), (50, 9000, 64 * 28 * 28)):
      ch, cw, th, tw = canvas_size(h, w, max_pixels=mx)
      self.assertTrue(ch % FACTOR == 0 and cw % FACTOR == 0 and ch * cw <= mx and th <= ch and tw <= cw and th <= h and tw <= w, (h, w, ch, cw))

  def test_load_picture_on_a_black_canvas(self):
    from PIL import Image
    im = Image.new("RGBA", (320, 288), (200, 100, 50, 0))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    old = qwen_vision.SIZING[0]
    qwen_vision.SIZING[0] = "glm5v"
    try: pic = qwen_vision.load_picture(buf.getvalue(), max_tokens=1024)
    finally: qwen_vision.SIZING[0] = old
    self.assertEqual((pic.rgb.shape, pic.grid, pic.tokens, pic.factor), ((308, 336, 3), (11, 12), 132, 28))
    # (alpha dropped as llama.cpp's: the stored colour, not white; the padding black)
    self.assertEqual(pic.rgb[0, 0].tolist(), [200, 100, 50])
    self.assertEqual(pic.rgb[300, 330].tolist(), [0, 0, 0])
    # (the default sizing is Qwen's, unchanged)
    self.assertEqual(qwen_vision.load_picture(buf.getvalue(), max_tokens=1024).factor, 32)

  def test_swiglu_clamp(self):
    with Context(DEV="CPU"):
      g, u = Tensor([-20.0, 0.5, 20.0]), Tensor([-20.0, 2.0, 20.0])
      ref = np.minimum([-20.0, 0.5, 20.0], 10) / (1 + np.exp(-np.minimum([-20.0, 0.5, 20.0], 10))) * np.clip([-20.0, 2.0, 20.0], -10, 10)
      np.testing.assert_allclose(swiglu_clamp(g, u, 10.0).numpy(), ref, rtol=1e-5)

class TestGlmVisionPrefill(unittest.TestCase):
  def _model(self, sizes:list[int]):
    from tinygrad.llm.glm5next import Transformer
    m = Transformer.__new__(Transformer)
    m.vision, m.image_pad_id, m.config = True, PAD, types.SimpleNamespace(dim=4, qsa=False)
    calls:list = []
    m._finish_load = m._seq_init = lambda: None
    m.token_embd = types.SimpleNamespace(weight=types.SimpleNamespace(device="NULL"))
    m._token_host = array.array('i', [0] * 4096)
    m._seq_copy_call = lambda *a, **k: None
    m._seq_pf_host_temp = array.array('f', [0.0] * 8)
    m._seq_chunks = lambda n: [min(s, n - sum(sizes[:i])) for i, s in enumerate(sizes) if sum(sizes[:i]) < n]
    m._seq_chunk = lambda base, pos, n, **kw: calls.append((pos, n, kw.get("vis")))
    m._wait_host = lambda get, what: 7
    m.seq_rows = lambda: 4096
    m.REGION_ALIGN = 1
    return m, calls

  def test_only_chunks_with_picture_rows_take_the_vision_variant(self):
    # (no rope: a chunk after a picture is text as before; the token buffer has the image pad at a picture's rows)
    m, calls = self._model([8] * 20)
    text = [5, 6, PAD, 7, 8, 9, 10]
    self.assertEqual(m.seq_prefill(0, 0, text, 0, 0), 7)
    self.assertEqual(calls, [(0, 7, None)])
    self.assertEqual(m._token_host[:len(text)].tolist(), text)
    calls.clear()
    pic = picture_ids(b"\x01" * 32, 2, 6)   # (12 rows: rows 2..13 of the prompt)
    ids = [5, VS] + pic + [VE] + list(range(1, 12))
    m.seq_prefill(0, 0, ids, 0, 0, pictures={pic[0]: lambda: None})
    self.assertEqual([(p, n, v is not None) for p, n, v in calls], [(0, 8, True), (8, 8, True), (16, 8, False), (24, 2, False)])
    self.assertEqual(calls[0][2][0], [(2, 12, 6, pic[0])])
    self.assertEqual(m._token_host[:len(ids)].tolist(), [5, VS] + [PAD] * 12 + [VE] + list(range(1, 12)))
    # (a prompt may end right after its picture: no rope window to keep)
    m.seq_prefill(0, 0, [5, VS] + pic + [VE], 0, 0, pictures={pic[0]: lambda: None})

  def test_vis_fill_and_embed(self):
    from tinygrad.llm.glm5next import Transformer
    m, _ = self._model([8])
    M, D = 8, 4
    m._vis_rows_max = M
    m._vis_host_rows, m._vis_host_mask = memoryview(bytearray(4 * M * D)).cast('f'), memoryview(bytearray(4 * M)).cast('i')
    m.token_embd = types.SimpleNamespace(weight=types.SimpleNamespace(device="CPU"))
    pic = picture_ids(b"\x02" * 32, 1, 3)
    rows = np.arange(3 * D, dtype=np.float32).reshape(3, D) + 100
    # (the chunk of rows 4..11 of a prompt whose picture is at rows 2..4: its last 1 row in the chunk's row 0)
    with Context(DEV="CPU"):
      Transformer._vis_fill(m, [(2, 3, 3, pic[0])], {pic[0]: lambda: rows}, 4, 8)
      self.assertEqual(m._vis_host_mask.tolist(), [1, 0, 0, 0, 0, 0, 0, 0])
      m._vis_rows, m._vis_mask = Tensor(np.array(m._vis_host_rows, np.float32)), Tensor(np.array([m._vis_host_mask.tolist()], np.int32))
      emb = Tensor.ones(1, 5, D)
      out = Transformer._vis_embed(m, emb).numpy()
    np.testing.assert_array_equal(out[0, 0], rows[2])
    np.testing.assert_array_equal(out[0, 1:], np.ones((4, D), np.float32))

if __name__ == "__main__":
  unittest.main()
