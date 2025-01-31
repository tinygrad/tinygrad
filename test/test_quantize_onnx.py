import numpy as np
import onnx, unittest
from onnx import helper, numpy_helper, TensorProto
from examples.benchmark_onnx import load_onnx_model
from tinygrad import Tensor, Context

N = 1024

def create_gemm_model(model_path, in_size=N, out_size=N):
  # Define input and output
  input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, in_size])
  output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, out_size])

  # Create random weights and bias
  W_data = np.random.randn(in_size, out_size).astype(np.float32)
  B_data = np.random.randn(out_size).astype(np.float32)

  W_init = numpy_helper.from_array(W_data, name="W")
  B_init = numpy_helper.from_array(B_data, name="B")

  gemm_node = helper.make_node("Gemm", inputs=["input", "W", "B"], outputs=["output"], alpha=1.0, beta=1.0, transB=0)
  graph_def = helper.make_graph([gemm_node], "SingleGemmGraph", [input_tensor], [output_tensor], initializer=[W_init, B_init])

  # Create and save the model
  model_def = helper.make_model(graph_def, producer_name="single_gemm_example")
  onnx.save_model(model_def, model_path)
  return model_path

class TestQuantizeOnnx(unittest.TestCase):
  def test_quant(self):
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader
    class FakeDataReader(CalibrationDataReader):
      def __init__(self): self.cnt = 0
      def get_next(self) -> dict:
        self.cnt += 1
        if self.cnt == 100: return None
        return {"input": np.random.uniform(size=(1, N)).astype(np.float32)}
    out_file = "/tmp/test_out.onnx"
    quantize_static(create_gemm_model("/tmp/test_in.onnx"), out_file,
                    FakeDataReader(), quant_format=QuantFormat.QDQ, per_channel=False,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    extra_options={"ActivationSymmetric": True})
    run_onnx_jit, _, _ = load_onnx_model(out_file)
    with Context(NOOPT=1):
      run_onnx_jit(input=Tensor(np.random.uniform(size=(1, N)).astype(np.float32)))

if __name__ == "__main__":
  unittest.main()
