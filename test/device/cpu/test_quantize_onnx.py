import unittest, tempfile
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import Ops
from tinygrad.dtype import AddrSpace
from tinygrad.codegen import to_program
from test.device.dsp.test_quantize_onnx import get_quantized_model

def get_quantized_conv_model(cin=16, cout=32, hw=14, group=1, per_channel=False):
  import onnx
  from onnx import helper, numpy_helper, TensorProto
  from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader
  x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, cin, hw, hw])
  y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, cout, hw, hw])
  w = numpy_helper.from_array(np.random.randn(cout, cin//group, 3, 3).astype(np.float32), name="W")
  b = numpy_helper.from_array(np.random.randn(cout).astype(np.float32), name="B")
  conv = helper.make_node("Conv", ["input", "W", "B"], ["conv"], pads=[1,1,1,1], group=group)
  relu = helper.make_node("Relu", ["conv"], ["relu"])
  add = helper.make_node("Add", ["relu", "conv"], ["output"])
  graph = helper.make_graph([conv, relu, add], "qconv", [x], [y], initializer=[w, b])
  tmpdir = tempfile.mkdtemp()
  onnx.save_model(helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 13)]), f"{tmpdir}/in.onnx")
  class DataReader(CalibrationDataReader):
    def __init__(self): self.cnt = 0
    def get_next(self) -> dict|None:
      self.cnt += 1
      return None if self.cnt == 20 else {"input": np.random.uniform(-1, 1, size=(1, cin, hw, hw)).astype(np.float32)}
  quantize_static(f"{tmpdir}/in.onnx", f"{tmpdir}/out.onnx", DataReader(), quant_format=QuantFormat.QDQ, per_channel=per_channel,
                  activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8, extra_options={"ActivationSymmetric": False})
  return f"{tmpdir}/out.onnx"

def quantize_graph(nodes, inputs, initializers, gen, per_channel=False, weight_type="QUInt8", opset=13):
  # a float graph (output "output") quantized like onnxruntime does, calibrated on inputs from gen()
  import onnx
  from onnx import helper, TensorProto
  from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader
  out = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)
  graph = helper.make_graph(nodes, "q", inputs, [out], initializer=initializers)
  tmpdir = tempfile.mkdtemp()
  onnx.save_model(helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", opset)]), f"{tmpdir}/in.onnx")
  class DataReader(CalibrationDataReader):
    def __init__(self): self.cnt = 0
    def get_next(self) -> dict|None:
      self.cnt += 1
      return None if self.cnt == 20 else {"input": gen()}
  quantize_static(f"{tmpdir}/in.onnx", f"{tmpdir}/out.onnx", DataReader(), quant_format=QuantFormat.QDQ, per_channel=per_channel,
                  activation_type=QuantType.QUInt8, weight_type=getattr(QuantType, weight_type), extra_options={"ActivationSymmetric": False})
  return f"{tmpdir}/out.onnx"

def conv_nodes(cin, cout, x="input", out="conv"):
  from onnx import helper, numpy_helper
  w = numpy_helper.from_array(np.random.randn(cout, cin, 3, 3).astype(np.float32), name=f"W_{out}")
  return [helper.make_node("Conv", [x, f"W_{out}"], [out], pads=[1,1,1,1])], [w]

@unittest.skipIf(Device.DEFAULT != "CPU", "only tests for CPU")
class TestQuantizeOnnxCPU(unittest.TestCase):
  def setUp(self):
    try:
      import onnx, onnxruntime # noqa: F401 # pylint: disable=unused-import
    except ImportError: raise unittest.SkipTest()

  def _test_matches_ort(self, fn:str, shape:tuple[int, ...], inp:np.ndarray|None=None, fused:tuple[str, ...]=()):
    import onnxruntime as ort
    from tinygrad.nn.onnx import OnnxRunner
    if inp is None: inp = np.random.uniform(-1, 1, size=shape).astype(np.float32)
    ref = ort.InferenceSession(fn).run(None, {"input": inp})[0]
    runner = OnnxRunner(fn)
    for op in fused: self.assertIn(op, [n.op for n in runner.graph_nodes])
    # the QDQ model computes in integers like onnxruntime, so it matches up to the rounding of the requantize
    step = np.diff(np.unique(ref)).min()
    np.testing.assert_allclose(runner({"input": Tensor(inp)})["output"].numpy(), ref, atol=step*1.01, rtol=0)

  def test_quant_gemm(self): self._test_matches_ort(get_quantized_model(64), (64, 64))
  def test_quant_conv(self): self._test_matches_ort(get_quantized_conv_model(), (1, 16, 14, 14))
  def test_quant_conv_per_channel(self): self._test_matches_ort(get_quantized_conv_model(per_channel=True), (1, 16, 14, 14))
  def test_quant_conv_depthwise(self): self._test_matches_ort(get_quantized_conv_model(16, 16, group=16), (1, 16, 14, 14))

  def test_quant_conv_per_channel_int8(self):
    from onnx import helper, TensorProto
    nodes, inits = conv_nodes(16, 32)
    fn = quantize_graph(nodes + [helper.make_node("Relu", ["conv"], ["output"])], [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                        [1, 16, 14, 14])], inits, lambda: np.random.uniform(-1, 1, (1, 16, 14, 14)).astype(np.float32), True, "QInt8")
    self._test_matches_ort(fn, (1, 16, 14, 14), fused=("QLinearConv",))

  def _test_lookup(self, op:str, **opts):
    from onnx import helper, TensorProto
    nodes, inits = conv_nodes(8, 8)
    fn = quantize_graph(nodes + [helper.make_node(op, ["conv"], ["act"], **opts)] + conv_nodes(8, 8, "act", "output")[0],
                        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 10, 10])], inits + conv_nodes(8, 8, "act", "output")[1],
                        lambda: np.random.uniform(-1, 1, (1, 8, 10, 10)).astype(np.float32), opset=20)
    self._test_matches_ort(fn, (1, 8, 10, 10), fused=("QLinearLookup",))
  def test_lookup_gelu(self): self._test_lookup("Gelu")
  def test_lookup_gelu_tanh(self): self._test_lookup("Gelu", approximate="tanh")
  def test_lookup_sigmoid(self): self._test_lookup("Sigmoid")

  def test_lookup_float_reduce_mean(self):
    # an op without a QuantizeLinear after it is a lookup of floats
    from onnx import helper, numpy_helper, TensorProto
    nodes, inits = conv_nodes(8, 8)
    nodes += [helper.make_node("Gelu", ["conv"], ["act"]), helper.make_node("ReduceMean", ["act", "axes"], ["output"], keepdims=1)]
    fn = quantize_graph(nodes, [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 10, 10])],
                        inits + [numpy_helper.from_array(np.array([2, 3], np.int64), name="axes")],
                        lambda: np.random.uniform(-1, 1, (1, 8, 10, 10)).astype(np.float32), opset=20)
    self._test_matches_ort(fn, (1, 8, 10, 10), fused=("QLinearLookup",))

  def test_reduce_mean(self):
    from onnx import helper, numpy_helper, TensorProto
    nodes, inits = conv_nodes(8, 8)
    nodes += [helper.make_node("ReduceMean", ["conv", "axes"], ["mean"], keepdims=1)] + conv_nodes(8, 8, "mean", "output")[0]
    fn = quantize_graph(nodes, [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 10, 10])],
                        inits + conv_nodes(8, 8, "mean", "output")[1] + [numpy_helper.from_array(np.array([2, 3], np.int64), name="axes")],
                        lambda: np.random.uniform(-1, 1, (1, 8, 10, 10)).astype(np.float32), opset=20)
    self._test_matches_ort(fn, (1, 8, 10, 10), fused=("QLinearReduceMean",))

  def test_lookup_input(self):
    # the ops on a uint8 input (and movement ops) quantized are a lookup
    from onnx import helper, numpy_helper, TensorProto
    nodes, inits = conv_nodes(8, 8, "x")
    nodes = [helper.make_node("Cast", ["input"], ["f"], to=TensorProto.FLOAT), helper.make_node("Div", ["f", "c"], ["d"]),
             helper.make_node("Transpose", ["d"], ["x"], perm=[0, 1, 3, 2])] + nodes + [helper.make_node("Relu", ["conv"], ["output"])]
    def gen(): return np.random.randint(0, 256, (1, 8, 10, 12), dtype=np.uint8)
    fn = quantize_graph(nodes, [helper.make_tensor_value_info("input", TensorProto.UINT8, [1, 8, 10, 12])],
                        inits + [numpy_helper.from_array(np.array(255, np.float32), name="c")], gen)
    self._test_matches_ort(fn, (1, 8, 10, 12), gen(), fused=("QLinearLookup",))

  def test_quant_128_int_acc(self, sz=128):
    from tinygrad.nn.onnx import OnnxRunner
    run_onnx = OnnxRunner(get_quantized_model(sz))
    linear = run_onnx({"input":Tensor(np.random.uniform(size=(sz, sz)).astype(np.float32))})["output"].schedule_linear()
    for call in linear.src:
      if call.src[0].op is not Ops.SINK: continue
      prg = to_program(call.src[0], renderer=Device[Device.DEFAULT].renderer)
      # all accumulators are int
      daccs = [u for u in tuple(prg.src[1].src) if u.op is Ops.BUFFER and u.addrspace is AddrSpace.REG]
      assert all(u.dtype is dtypes.int for u in daccs), [u.dtype for u in daccs]

if __name__ == "__main__": unittest.main()
