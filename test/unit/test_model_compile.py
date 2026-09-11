import io, pickle, unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.compile import compile_jit, dump_pickle, load_pickle
from tinygrad.nn.warp_compile import NV12Frame, compile_warp, make_frame_prepare, make_luma_warp


class TestCompile(unittest.TestCase):
  def test_pickle_formats(self):
    data = np.arange(100, dtype=np.float32)
    for out_of_band in (False, True):
      with self.subTest(out_of_band=out_of_band), io.BytesIO() as f:
        dump_pickle([data, data], f, out_of_band=out_of_band)
        f.seek(0)
        a, b = load_pickle(f, out_of_band=out_of_band)
        self.assertIs(a, b)
        np.testing.assert_array_equal(a, data)

  def test_stateful_callable(self):
    def function(x, state):
      x = x.to(state.device)
      state.assign(state+x).realize()
      return {'sum': state.clone(), 'twice': x*2}
    def make_inputs(seed):
      data = np.random.default_rng(seed).standard_normal(4).astype(np.float32)
      return (), {'x': Tensor(data, device='NPY').realize(), 'state': Tensor.zeros(4).contiguous().realize()}
    jit = compile_jit(function, make_inputs, 3, out_of_band=True)
    with io.BytesIO() as f:
      dump_pickle(jit, f, out_of_band=True)
      f.seek(0)
      loaded = load_pickle(f, out_of_band=True)
    _, inputs = make_inputs(7)
    expected = inputs['x'].numpy().copy()
    for i in range(1, 4):
      output = loaded(**inputs)
      np.testing.assert_allclose(output['sum'].numpy(), expected*i)
      np.testing.assert_array_equal(output['twice'].numpy(), expected*2)


class TestWarpCompile(unittest.TestCase):
  frame = NV12Frame(8, 8, 12, 10, 6, 224)

  def test_luma_border_and_stride(self):
    data = np.arange(self.frame.size, dtype=np.uint8)
    transform = np.array([[1, 0, -1], [0, 1, 1], [0, 0, 1]], dtype=np.float32)
    warp = make_luma_warp(self.frame, 8, 8)
    out = warp(Tensor(data), Tensor(transform)).numpy().reshape(8, 8)
    expected = np.full((8, 8), 16, dtype=np.uint8)
    expected[:7, 1:] = data[:8*12].reshape(8, 12)[1:, :7]
    np.testing.assert_array_equal(out, expected)

  def test_yuv_planes_and_padding(self):
    data = np.arange(self.frame.size, dtype=np.uint8)
    warp = make_frame_prepare(self.frame, 8, 8)
    out = warp(Tensor(data), Tensor(np.eye(3, dtype=np.float32))).numpy()
    y = data[:8*12].reshape(8, 12)[:, :8]
    uv = data[12*10:12*10+4*12].reshape(4, 12)[:, :8]
    expected = np.stack([y[::2, ::2], y[1::2, ::2], y[::2, 1::2], y[1::2, 1::2], uv[:, ::2], uv[:, 1::2]])
    np.testing.assert_array_equal(out, expected)

  def test_standalone_warp_pickle(self):
    for layout in ('luma', 'yuv420'):
      with self.subTest(layout=layout):
        jit = compile_warp(self.frame, (8, 8), layout=layout, benchmark_runs=1)
        loaded = pickle.loads(pickle.dumps(jit))
        data = Tensor(np.arange(self.frame.size, dtype=np.uint8)).realize()
        transform = Tensor(np.eye(3, dtype=np.float32), device='NPY').realize()
        out = loaded(data, transform).numpy()
        self.assertEqual(out.shape, (1, 64) if layout == 'luma' else (6, 4, 4))


if __name__ == '__main__': unittest.main()
