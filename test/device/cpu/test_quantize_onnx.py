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

@unittest.skipIf(Device.DEFAULT != "CPU", "only tests for CPU")
class TestQuantizeOnnxCPU(unittest.TestCase):
  def setUp(self):
    try:
      import onnx, onnxruntime # noqa: F401 # pylint: disable=unused-import
    except ImportError: raise unittest.SkipTest()

  def _test_matches_ort(self, fn:str, shape:tuple[int, ...]):
    import onnxruntime as ort
    from tinygrad.nn.onnx import OnnxRunner
    inp = np.random.uniform(-1, 1, size=shape).astype(np.float32)
    ref = ort.InferenceSession(fn).run(None, {"input": inp})[0]
    # the QDQ model computes in integers like onnxruntime, so it matches up to the rounding of the requantize
    step = np.diff(np.unique(ref)).min()
    np.testing.assert_allclose(OnnxRunner(fn)({"input": Tensor(inp)})["output"].numpy(), ref, atol=step*1.01, rtol=0)

  def test_quant_gemm(self): self._test_matches_ort(get_quantized_model(64), (64, 64))
  def test_quant_conv(self): self._test_matches_ort(get_quantized_conv_model(), (1, 16, 14, 14))
  def test_quant_conv_per_channel(self): self._test_matches_ort(get_quantized_conv_model(per_channel=True), (1, 16, 14, 14))
  def test_quant_conv_depthwise(self): self._test_matches_ort(get_quantized_conv_model(16, 16, group=16), (1, 16, 14, 14))

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
