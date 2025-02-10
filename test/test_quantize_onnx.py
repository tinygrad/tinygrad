import numpy as np
import unittest
from tinygrad import Tensor, Context, Device
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.engine.realize import CompiledRunner, ExecItem

N = 1024

def create_gemm_model(model_path:str, in_size=N, out_size=N):
  import onnx
  from onnx import helper, numpy_helper, TensorProto
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

def sexec(out:Tensor, opts:list[Opt]):
  si = out.schedule()[-1]
  k = Kernel(si.ast, opts=Device[Device.DEFAULT].renderer)
  #opts = [Opt(op=OptOps.UPCAST, axis=0, arg=128)] #, Opt(op=OptOps.UNROLL, axis=0, arg=4)]
  for opt in opts: k.apply_opt(opt)
  prg = k.to_program()
  ei = ExecItem(CompiledRunner(prg), [x.ensure_allocated() for x in si.bufs], si.metadata)
  for _ in range(3): ei.run(wait=True)

@unittest.skipIf(Device.DEFAULT != "DSP", "only tests for DSP")
class TestQuantizeOnnx(unittest.TestCase):
  def test_quant(self):
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader
    from examples.benchmark_onnx import load_onnx_model
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
    run_onnx_jit, _ = load_onnx_model(out_file)
    with Context(DONT_REALIZE_EXPAND=1):
      run_onnx_jit(input=Tensor(np.random.uniform(size=(1, N)).astype(np.float32)))

  def test_prequant_conv2d_1x1(self):
    X = Tensor(np.random.uniform(0, 255, size=(1, 32, 128, 128)).astype(np.uint8))
    W = Tensor(np.random.uniform(0, 255, size=(64, 32, 1, 1)).astype(np.uint8))
    out = X.conv2d(W, acc_dtype=X.dtype)
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=128), Opt(op=OptOps.UNROLL, axis=0, arg=4)]
    sexec(out, opts)

  def test_prequant_gemm(self):
    N = 512
    # ugh, it's so broken with those casts. need DONT_REALIZE_EXPAND=1 python3 test/test_quantize_onnx.py TestQuantizeOnnx.test_prequant
    X = Tensor(np.random.uniform(0, 255, size=(N,N)).astype(np.uint8))
    W = Tensor(np.random.uniform(0, 255, size=(N,N)).astype(np.uint8))
    out = X.matmul(W, acc_dtype=X.dtype)
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=128), Opt(op=OptOps.UNROLL, axis=0, arg=4)]
    sexec(out, opts)

  def test_prequant_gemm_intacc(self):
    N = 512
    # ugh, it's so broken with those casts. need DONT_REALIZE_EXPAND=1 python3 test/test_quantize_onnx.py TestQuantizeOnnx.test_prequant
    X = Tensor(np.random.uniform(0, 255, size=(N,N)).astype(np.uint8))
    W = Tensor(np.random.uniform(0, 255, size=(N,N)).astype(np.uint8))
    out = X.matmul(W)
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=128), Opt(op=OptOps.UNROLL, axis=0, arg=4)]
    sexec(out, opts)

  def test_prequant_gemv(self):
    N = 2048
    # ugh, it's so broken with those casts. need DONT_REALIZE_EXPAND=1 python3 test/test_quantize_onnx.py TestQuantizeOnnx.test_prequant
    X = Tensor(np.random.uniform(0, 255, size=(1,N)).astype(np.uint8))
    W = Tensor(np.random.uniform(0, 255, size=(N,N)).astype(np.uint8))
    #out = X.cast(dtypes.int) @ W.cast(dtypes.int)
    #out = X @ W
    out = X.matmul(W, acc_dtype=X.dtype)
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=128), Opt(op=OptOps.UNROLL, axis=0, arg=4)]
    sexec(out, opts)

if __name__ == "__main__":
  unittest.main()
