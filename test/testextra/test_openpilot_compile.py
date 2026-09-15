import pickle, tempfile, unittest
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper
from tinygrad import Device, Tensor
from examples.openpilot.compile_onnx import compile_onnx, make_input_queues, read_file_chunked_to_disk
from examples.openpilot.compile_warp import NV12Frame, compile_warp
from examples.openpilot.helpers import load_oob


class TestOnnxCompile(unittest.TestCase):
  def test_stateful_roundtrip(self):
    with tempfile.TemporaryDirectory() as directory:
      path, output = Path(directory)/'model.onnx', Path(directory)/'model.pkl'
      inputs = [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 2]),
                helper.make_tensor_value_info('counter', TensorProto.INT32, [1])]
      outputs = [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', 2]),
                 helper.make_tensor_value_info('next_counter', TensorProto.INT32, [1])]
      nodes = [helper.make_node('Cast', ['counter'], ['count_float'], to=TensorProto.FLOAT),
               helper.make_node('Add', ['x', 'count_float'], ['y']), helper.make_node('Add', ['counter', 'one'], ['next_counter'])]
      graph = helper.make_graph(nodes, 'stateful', inputs, outputs, [helper.make_tensor('one', TensorProto.INT32, [1], [1])])
      onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)]), path)
      compile_onnx(path, output)
      with output.open('rb') as f: model = load_oob(f)
      self.assertEqual(model['state_pairs'], {'counter': 'next_counter'})
      self.assertIsNone(model['metadata']['output_slices'])
      inputs = make_input_queues(model['input_shapes'], Device.DEFAULT)
      for i in range(3):
        inputs['x'].assign(Tensor([[i, -i]], dtype='float32', device=Device.DEFAULT)).realize()
        result, = model['run_model'](**inputs)
        np.testing.assert_array_equal(result.numpy(), [[2*i, 0]])
        np.testing.assert_array_equal(inputs['counter'].numpy(), [i+1])

  def test_chunked_input(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'model.onnx'
      Path(f'{path}.chunkmanifest').write_text('2')
      Path(f'{path}.chunk01of02').write_bytes(b'first')
      Path(f'{path}.chunk02of02').write_bytes(b'second')
      self.assertEqual(Path(read_file_chunked_to_disk(str(path))).read_bytes(), b'firstsecond')


class TestWarpCompile(unittest.TestCase):
  def check_warp(self, layout, frames):
    width, height, stride, uv_offset = 32, 24, 48, 48*32
    nv12 = NV12Frame(width, height, stride, uv_offset, uv_offset + stride*16)
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'warp.pkl'
      compile_warp(nv12, 16, 16, path, layout, border_fill=16, frames=frames, transform_device='NPY')
      with path.open('rb') as f: warp = pickle.load(f)
      rng = np.random.default_rng(42)
      for step in range(3):
        data = rng.integers(0, 256, (frames, nv12.size), dtype=np.uint8)
        transforms = np.tile(np.eye(3, dtype=np.float32), (frames, 1, 1))
        transforms[:, 0, 2], transforms[:, 1, 2] = -4 + 2*step, 2*step
        expected = []
        for frame, matrix in zip(data, transforms, strict=True):
          def sample(plane, w, h, scale):
            yy, xx = np.indices((h, w))
            sx, sy = xx + int(matrix[0, 2]/scale), yy + int(matrix[1, 2]/scale)
            valid = (sx >= 0) & (sx < plane.shape[1]) & (sy >= 0) & (sy < plane.shape[0])
            return np.where(valid, plane[sy.clip(0, plane.shape[0]-1), sx.clip(0, plane.shape[1]-1)], 16).astype(np.uint8)
          y = sample(frame[:height*stride].reshape(height, stride)[:, :width], 16, 16, 1)
          if layout == 'luma': expected.append(y.reshape(1, -1))
          else:
            uv = frame[uv_offset:uv_offset+stride*(height//2)].reshape(height//2, stride)[:, :width]
            u, v = (sample(uv[:, offset::2], 8, 8, 2) for offset in (0, 1))
            expected.append(np.stack((y[::2, ::2], y[1::2, ::2], y[::2, 1::2], y[1::2, 1::2], u, v)))
        frame_arg, transform_arg = (data[0], transforms[0]) if frames == 1 else (data, transforms)
        actual = warp(Tensor(frame_arg, device=Device.DEFAULT).realize(), Tensor(transform_arg, device='NPY')).numpy()
        np.testing.assert_array_equal(actual, expected[0] if frames == 1 else np.stack(expected))

  def test_luma(self): self.check_warp('luma', 1)
  def test_batched_luma(self): self.check_warp('luma', 3)
  def test_batched_yuv420(self): self.check_warp('yuv420', 2)


if __name__ == '__main__': unittest.main()
