# VISION for qwen4exp (tinygrad/llm/qwen_vision.py, the vision variant's rope in qwen4exp.py), on the CPU: the tower's picture size against HF's
# smart_resize, a picture's ids in the token lists, the M-RoPE positions against HF's get_rope_index rule, the interleaved rope rows against
# ggml's IMROPE rule, and the window's rope rows of a chunk: a text row's are the table's own row
import itertools, math, unittest
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.llm import qwen4exp as q
from tinygrad.llm.qwen4exp import precompute_freqs_cis, imrope_mix, _rope_rows, _vis_pools, VIS_PAD
from tinygrad.llm.qwen_vision import smart_resize, picture_ids, picture_runs, rope_positions, rope_delta, patches, bucket

SECTIONS = (11, 11, 10)

def hf_rope_index(kinds:list[int], grids:list[tuple[int, int]]) -> np.ndarray:
  """transformers' Qwen3VLModel.get_rope_index (5.13) for one sequence: kinds 0 text / 1 a picture's row, grids its (rows, cols) merged"""
  pos, cur, it = [], 0, iter(grids)
  for kind, grp in itertools.groupby(kinds):
    n = len(list(grp))
    if kind == 0:
      pos.append(np.arange(n)[None].repeat(3, 0) + cur)
      cur += n
    else:
      gh, gw = next(it)
      assert gh * gw == n
      hh, ww = np.meshgrid(np.arange(gh) + cur, np.arange(gw) + cur, indexing="ij")
      pos.append(np.stack([np.full(n, cur), hh.reshape(-1), ww.reshape(-1)]))
      cur += max(gh, gw)
  return np.concatenate(pos, axis=1)

class TestSmartResize(unittest.TestCase):
  def test_against_hf(self):
    # (transformers 5.13 qwen2_vl smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=16777216 / 4194304), run on teeny 2026-11-14)
    hf = {(64, 64): ((256, 256), (256, 256)), (224, 320): ((224, 320), (224, 320)), (700, 1000): ((704, 992), (704, 992)),
          (1080, 1920): ((1088, 1920), (1088, 1920)), (2160, 3840): ((2176, 3840), (1536, 2720)), (4320, 7680): ((3072, 5440), (1536, 2720)),
          (48, 4000): ((64, 4000), (64, 4000)), (255, 257): ((256, 256), (256, 256)), (16, 16): ((256, 256), (256, 256)),
          (10000, 10000): ((4096, 4096), (2048, 2048)), (208, 208): ((256, 256), (256, 256)), (240, 240): ((256, 256), (256, 256))}
    for (h, w), (a, b) in hf.items():
      self.assertEqual(smart_resize(h, w), a, (h, w))
      self.assertEqual(smart_resize(h, w, max_pixels=4194304), b, (h, w))

  def test_aspect_refused(self):
    with self.assertRaisesRegex(ValueError, "aspect"): smart_resize(10, 2011)

class TestPatches(unittest.TestCase):
  def test_merge_groups_contiguous(self):
    # a picture whose pixel value is its 16 x 16 patch's (row, col): the patches come in 2 x 2 merge groups, row major over the groups
    gh, gw = 4, 6
    rgb = np.zeros((gh * 16, gw * 16, 3), np.uint8)
    for r in range(gh):
      for c in range(gw): rgb[r*16:(r+1)*16, c*16:(c+1)*16] = (r * 10, c * 10, 0)
    px, grid = patches(rgb)
    self.assertEqual(grid, (gh, gw))
    self.assertEqual(px.shape, (gh * gw, 3 * 2 * 16 * 16))
    got = [(round((px[i, 0] * 0.5 + 0.5) * 255 / 10), round((px[i, 2 * 256] * 0.5 + 0.5) * 255 / 10)) for i in range(gh * gw)]
    want = [(2*R + dr, 2*C + dc) for R in range(gh // 2) for C in range(gw // 2) for dr in range(2) for dc in range(2)]
    self.assertEqual(got, want)

  def test_bucket(self):
    self.assertEqual([bucket(n) for n in (1, 256, 257, 1024, 1025, 16384)], [256, 256, 512, 1024, 2048, 16384])

class TestPictureIds(unittest.TestCase):
  def test_runs(self):
    a, b = picture_ids(b"\x01" * 32, 8, 9), picture_ids(b"\x02" * 32, 8, 9)
    self.assertTrue(all(t < 0 for t in a) and len(set(a)) == 72 and not set(a) & set(b))
    self.assertEqual(a, picture_ids(b"\x01" * 32, 8, 9))
    ids = [5, 248053] + a + [248054, 7, 248053] + b + [248054, 1, 2, 3, 4]
    self.assertEqual(picture_runs(ids), [(2, 72, 9, a[0]), (2 + 72 + 3, 72, 9, b[0])])
    # (the same picture twice in a row, markers between: two runs)
    self.assertEqual(len(picture_runs([248053] + a + [248054, 248053] + a + [248054])), 2)

  def test_size_in_ids(self):
    self.assertNotEqual(picture_ids(b"\x01" * 32, 8, 9)[0], picture_ids(b"\x01" * 32, 9, 8)[0])
    with self.assertRaises(AssertionError): picture_ids(b"\x01" * 32, 1, 1 << 14)

class TestRopePositions(unittest.TestCase):
  def test_against_hf(self):
    a, b = picture_ids(b"\x01" * 32, 8, 9), picture_ids(b"\x02" * 32, 12, 5)
    ids = list(range(17)) + [248053] + a + [248054] + list(range(30)) + [248053] + b + [248054] + list(range(9))
    kinds = [1 if t < 0 else 0 for t in ids]
    runs = picture_runs(ids)
    ref = hf_rope_index(kinds, [(8, 9), (12, 5)])
    np.testing.assert_array_equal(rope_positions(runs, 0, len(ids)), ref)
    # (a window: rows before 0 at 0, rows past the prompt go on as text)
    got = rope_positions(runs, -4, len(ids) + 6)
    np.testing.assert_array_equal(got[:, :4], 0)
    np.testing.assert_array_equal(got[:, 4:4 + len(ids)], ref)
    np.testing.assert_array_equal(got[:, -6:], ref[:, -1:] + np.arange(1, 7))
    # the delta: a decoding row's position is its row minus it
    self.assertEqual(rope_delta(runs), (72 - 9) + (60 - 12))
    self.assertEqual(int(ref[0, -1]), len(ids) - 1 - rope_delta(runs))

  def test_picture_at_row_0(self):
    # a picture at the very start and the window of a chunk at row 0 (rows -4 .. -1: at 0, never read as a picture's)
    a = picture_ids(b"\x03" * 32, 4, 6)
    runs = picture_runs(a + [1, 2, 3, 4, 5])
    got = rope_positions(runs, -VIS_PAD, 24 + 5 + 3)
    np.testing.assert_array_equal(got[:, :VIS_PAD], 0)
    np.testing.assert_array_equal(got[:, VIS_PAD:VIS_PAD + 24], hf_rope_index([1] * 24 + [0] * 5, [(4, 6)])[:, :24])
    # the text after it: at max(4, 6) = 6 on
    np.testing.assert_array_equal(got[:, VIS_PAD + 24:], np.arange(6, 6 + 8)[None].repeat(3, 0))

  def test_text_only(self):
    np.testing.assert_array_equal(rope_positions([], 10, 20), np.arange(10, 20)[None].repeat(3, 0))

class TestIMRope(unittest.TestCase):
  def test_mix_against_ggml_rule(self):
    dim, theta = 64, 10_000_000.0
    table = precompute_freqs_cis(dim, 512, theta).realize()  # (512, 64): cos of the 32 pairs, then sin
    pos = np.array([[3, 3, 3], [40, 47, 52], [40, 40, 61], [100, 107, 100]], dtype=np.int32)
    t, h, w = (Tensor(table.numpy()[pos[:, i]]) for i in range(3))
    got = imrope_mix(t, h, w, SECTIONS).numpy()
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    for r, (pt, ph, pw) in enumerate(pos):
      for p in range(dim // 2):
        # (ggml_mrope_cache_init, is_imrope)
        x = ph if p % 3 == 1 and p < 3 * SECTIONS[1] else pw if p % 3 == 2 and p < 3 * SECTIONS[2] else pt
        self.assertAlmostEqual(got[r, p], math.cos(x * inv[p]), places=5)
        self.assertAlmostEqual(got[r, dim // 2 + p], math.sin(x * inv[p]), places=5)
    # a text row (three equal positions): its table row, the same floats
    np.testing.assert_array_equal(got[0], table.numpy()[3])

  def test_sections_cover_every_pair(self):
    x = Tensor.zeros(2, 64)
    with self.assertRaisesRegex(AssertionError, "sections"): imrope_mix(x, x, x, (8, 8, 8))

class TestVisWindow(unittest.TestCase):
  def tearDown(self): q._vis_rope[0] = None

  def test_text_window_equals_table(self):
    # the window of a chunk of text rows (positions = rows): the chunk's rope rows and its pools' are the table's, at a symbolic start
    table = precompute_freqs_cis(64, 4096, 10000.0).realize()
    tn = table.numpy()
    for start, n in ((1001, 100), (1000, 512), (3, 77), (0, 64)):
      Tq, W = 512, 512 + 3 * VIS_PAD
      pos = rope_positions([], start - VIS_PAD, start - VIS_PAD + W)
      win = Tensor(tn[pos[0]])
      sp, T = UOp.variable("start_pos", 0, 4095).bind(start), UOp.variable("toks", 1, Tq).bind(n)
      want_rows = _rope_rows(table, sp, T, Tq).numpy()
      first, new = sp // 4, (sp % 4 + T + 3) // 4
      want_pools = table.reshape(-1, 4, 64)[:, 0].shrink(((first, first + new), None)).pad_to((Tq // 4 + 1, 64)).numpy()
      q._vis_rope[0] = (win, sp)
      np.testing.assert_array_equal(_rope_rows(table, sp, T, Tq).numpy(), want_rows, err_msg=f"{start} {n}")
      got_pools = _vis_pools(4, first, new).pad_to((Tq // 4 + 1, 64)).numpy()
      q._vis_rope[0] = None
      np.testing.assert_array_equal(got_pools, want_pools, err_msg=f"{start} {n}")

if __name__ == "__main__": unittest.main()
