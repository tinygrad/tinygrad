import io, pickle, tempfile, unittest
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper
from tinygrad import Tensor, Device
from tinygrad.nn.compile import compile_onnx, dump_pickle, load_pickle, onnx_metadata
from tinygrad.nn.warp_compile import NV12Frame


class TestOnnxCompile(unittest.TestCase):
  def test_onnx_multi_output_and_metadata(self):
    graph = helper.make_graph([
      helper.make_node('Add', ['x', 'x'], ['twice']),
      helper.make_node('Identity', ['mask'], ['flags']),
    ], 'mixed', [helper.make_tensor_value_info('x', TensorProto.FLOAT16, ['batch', 4]),
                 helper.make_tensor_value_info('mask', TensorProto.BOOL, ['batch', 4])],
       [helper.make_tensor_value_info('twice', TensorProto.FLOAT16, ['batch', 4]),
        helper.make_tensor_value_info('flags', TensorProto.BOOL, ['batch', 4])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    helper.set_model_props(model, {'checkpoint': 'example', 'custom': 'retained'})
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'model.onnx'
      onnx.save(model, path)
      metadata = onnx_metadata(path)
      self.assertEqual(metadata['input_shapes']['x'], (0, 4))
      self.assertEqual(metadata['metadata'], {'checkpoint': 'example', 'custom': 'retained'})
      jit = compile_onnx(path, benchmark_runs=1)
      result = jit(mask=Tensor([[True, False, True, False]], device='NPY').realize(),
                   x=Tensor(np.arange(4, dtype=np.float16).reshape(1, 4), device='NPY').realize())
      np.testing.assert_array_equal(result['twice'].numpy(), [[0, 2, 4, 6]])
      np.testing.assert_array_equal(result['flags'].numpy(), [[True, False, True, False]])
      jit = compile_onnx(path, float32=True, output_name='twice', device_inputs=['x'], benchmark_runs=1)
      result = jit(mask=Tensor([[False]*4], device='NPY').realize(), x=Tensor([[1., 2., 3., 4.]]).realize())
      np.testing.assert_array_equal(result.numpy(), [[2, 4, 6, 8]])


class TestDrivingCompile(unittest.TestCase):
  def test_fused_model_and_input_layout(self):
    from examples.openpilot.compile_modeld import compile_model, make_input_queues
    shapes = {'img': (1, 12, 2, 4), 'big_img': (1, 12, 2, 4), 'features_buffer': (1, 2, 2),
              'desire_pulse': (1, 3, 8), 'traffic_convention': (1, 2), 'action_t': (1, 2)}
    nodes = [helper.make_node('Flatten', ['features_buffer'], ['features']),
             helper.make_node('Cast', ['img'], ['float_img'], to=TensorProto.FLOAT),
             helper.make_node('ReduceMean', ['float_img'], ['mean'], keepdims=0),
             helper.make_node('Cast', ['big_img'], ['float_big_img'], to=TensorProto.FLOAT),
             helper.make_node('ReduceMean', ['float_big_img'], ['big_mean'], keepdims=0),
             helper.make_node('ReduceMean', ['desire_pulse'], ['desire_mean'], keepdims=0),
             helper.make_node('Add', ['mean', 'big_mean'], ['images']),
             helper.make_node('Add', ['images', 'desire_mean'], ['context']),
             helper.make_node('Add', ['features', 'context'], ['outputs'])]
    graph = helper.make_graph(nodes, 'driving', [helper.make_tensor_value_info(k, TensorProto.UINT8 if 'img' in k else TensorProto.FLOAT, s)
                                              for k, s in shapes.items()], [helper.make_tensor_value_info('outputs', TensorProto.FLOAT, (1, 4))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    import codecs
    helper.set_model_props(model, {'model_checkpoint': 'synthetic', 'output_slices': codecs.encode(pickle.dumps({'hidden_state': slice(0, 2)}),
                                                                                                'base64').decode()})
    frames = [NV12Frame(8, 8, 12, 10, 6, 224), NV12Frame(12, 8, 16, 10, 6, 288)]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'model.onnx'
      onnx.save(model, path)
      compiled = compile_model(path, frames, (8, 4), 2, 1)
      with io.BytesIO() as f:
        dump_pickle(compiled, f, out_of_band=True)
        f.seek(0)
        loaded = load_pickle(f, out_of_band=True)
      self.assertEqual(set(loaded['run_model']), {(8, 8), (12, 8)})
      for frame in frames:
        specs = loaded['input_specs'][(frame.width, frame.height)]
        old, _, _ = make_input_queues(shapes, 2, Device.DEFAULT, frame.copy_size)
        self.assertEqual({k: t.shape for k, t in old.items()}, {k: spec[0] for k, spec in specs.items()})
        for _ in range(2):
          buffers = {k: np.zeros(s, dtype=d) for k, (s, d, _) in specs.items()}
          inputs = {k: Tensor(buffers[k], device=dev).realize() for k, (_, _, dev) in specs.items()}
          n = sum(np.prod(s) for s in loaded['npy_shapes'].values())*4
          transforms = buffers['packed_npy_inputs'][:18*4].view(np.float32).reshape(2, 3, 3)
          transforms[:] = np.eye(3, dtype=np.float32)
          buffers['packed_npy_inputs'][n:] = 64
          outputs, = loaded['run_model'][(frame.width, frame.height)](**inputs)
          np.testing.assert_array_equal(outputs.numpy(), np.full((1, 4), 64, dtype=np.float32))


if __name__ == '__main__': unittest.main()
