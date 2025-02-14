import numpy as np
import unittest
from dataclasses import replace
from tinygrad import Tensor, Context, Device
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.engine.realize import CompiledRunner, ExecItem

N = 512

def create_gemm_model(model_path:str, in_size=N, out_size=N, bias=False):
  import onnx
  from onnx import helper, numpy_helper, TensorProto
  # Define input and output
  input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [N, in_size])
  output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [N, out_size])

  # Create random weights and bias
  W_data = np.random.randn(in_size, out_size).astype(np.float32)
  W_init = numpy_helper.from_array(W_data, name="W")

  if bias:
    B_data = np.random.randn(out_size).astype(np.float32)
    B_init = numpy_helper.from_array(B_data, name="B")
    gemm_node = helper.make_node("Gemm", inputs=["input", "W", "B"], outputs=["output"], alpha=1.0, beta=1.0, transB=0)
    graph_def = helper.make_graph([gemm_node], "SingleGemmGraph", [input_tensor], [output_tensor], initializer=[W_init, B_init])
  else:
    gemm_node = helper.make_node("Gemm", inputs=["input", "W"], outputs=["output"], alpha=1.0, beta=1.0, transB=0)
    graph_def = helper.make_graph([gemm_node], "SingleGemmGraph", [input_tensor], [output_tensor], initializer=[W_init])

  # Create and save the model
  model_def = helper.make_model(graph_def, producer_name="single_gemm_example")
  onnx.save_model(model_def, model_path)
  return model_path

def sexec(out:Tensor, opts:list[Opt], replace_src=None):
  si = out.schedule()[-1]
  k = Kernel(si.ast, opts=Device[Device.DEFAULT].renderer)
  #opts = [Opt(op=OptOps.UPCAST, axis=0, arg=128)] #, Opt(op=OptOps.UNROLL, axis=0, arg=4)]
  for opt in opts: k.apply_opt(opt)
  prg = k.to_program()
  if replace_src is not None: prg = replace(prg, src=replace_src + "\nstatic long syscall" + prg.src.split("static long syscall")[1])
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
        return {"input": np.random.uniform(size=(N, N)).astype(np.float32)}
    out_file = "/tmp/test_out.onnx"
    # divide is ~1500-2000 without reduce_range, 750-900 with it
    quantize_static(create_gemm_model("/tmp/test_in.onnx"), out_file,
                    FakeDataReader(), quant_format=QuantFormat.QDQ, per_channel=False, reduce_range=False,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    extra_options={"ActivationSymmetric": False})
    run_onnx_jit, _ = load_onnx_model(out_file)
    with Context(DONT_REALIZE_EXPAND=1):
      run_onnx_jit(input=Tensor(np.random.uniform(size=(N, N)).astype(np.float32)))

  def test_prequant_conv2d_1x1(self):
    X = Tensor(np.random.uniform(0, 255, size=(1, 32, 128, 128)).astype(np.uint8))
    W = Tensor(np.random.uniform(0, 255, size=(64, 32, 1, 1)).astype(np.uint8))
    out = X.conv2d(W, acc_dtype=X.dtype)
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=128), Opt(op=OptOps.UNROLL, axis=0, arg=4)]
    sexec(out, opts)

  def test_prequant_gemm(self):
    N = 512
    X = Tensor(np.random.uniform(0, 255, size=(N,N)).astype(np.uint8))
    W = Tensor(np.random.uniform(0, 255, size=(N,N)).astype(np.uint8))
    out = X.matmul(W, acc_dtype=X.dtype)
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=128), Opt(op=OptOps.UNROLL, axis=0, arg=4)]
    sexec(out, opts)

  # TODO: this has to work
  def test_prequant_gemm_intacc_early(self, xi=np.int8, wi=np.int8):
    N = 512
    X = Tensor(np.random.uniform(0, 255, size=(N,N)).astype(xi))
    W = Tensor(np.random.uniform(0, 255, size=(N,N)).astype(wi))
    with Context(DONT_REALIZE_EXPAND=1):
      # this divide is interesting and forces the accumulator to actually be an int
      out = (X.cast("int").matmul(W.cast("int"))//1000).cast("int8")
      opts = [Opt(op=OptOps.UPCAST, axis=1, arg=128), Opt(op=OptOps.UNROLL, axis=0, arg=4)]
      sexec(out, opts)

  def test_prequant_gemm_handcode(self):
    src = """typedef int int128 __attribute__((aligned(512),vector_size(512)));
    typedef int int32 __attribute__((aligned(128),vector_size(128)));
    typedef int int64 __attribute__((aligned(256),vector_size(256)));
    typedef unsigned char unsigned_char4 __attribute__((aligned(4),vector_size(4)));
    typedef signed char signed_char128 __attribute__((aligned(128),vector_size(128)));
    typedef unsigned char unsigned_char128 __attribute__((aligned(128),vector_size(128)));
    __attribute__((noinline)) void r_512_4_128_128_4(unsigned char* restrict __attribute__((align_value(128))) data0, unsigned char* restrict __attribute__((align_value(128))) data1, signed char* restrict __attribute__((align_value(128))) data2) {
      for (int ridx0 = 0; ridx0 < 512; ridx0++) {
        int alu0 = (ridx0<<9);
        for (int ridx1 = 0; ridx1 < 4; ridx1++) {
          int alu1 = (ridx1<<7);
          //int128 acc0 = (int128){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
          int32 acc0 = __builtin_HEXAGON_V6_vd0_128B();
          int32 acc1 = __builtin_HEXAGON_V6_vd0_128B();
          int32 acc2 = __builtin_HEXAGON_V6_vd0_128B();
          int32 acc3 = __builtin_HEXAGON_V6_vd0_128B();

          for (int ridx2 = 0; ridx2 < 128; ridx2++) {
            unsigned_char4 val0 = *((unsigned_char4*)((data1+(alu0+(ridx2<<2)))));
            int val0_int = *(int*)(&val0);
            int alu2 = (alu1+(ridx2<<11));
            signed_char128 val1 = *((signed_char128*)((data2+alu2)));
            signed_char128 val2 = *((signed_char128*)((data2+(alu2+512))));
            signed_char128 val3 = *((signed_char128*)((data2+(alu2+1024))));
            signed_char128 val4 = *((signed_char128*)((data2+(alu2+1536))));
            acc0 = __builtin_HEXAGON_V6_vrmpybus_acc_128B(acc0, val1, val0_int);
            acc1 = __builtin_HEXAGON_V6_vrmpybus_acc_128B(acc1, val2, val0_int);
            acc2 = __builtin_HEXAGON_V6_vrmpybus_acc_128B(acc2, val3, val0_int);
            acc3 = __builtin_HEXAGON_V6_vrmpybus_acc_128B(acc3, val4, val0_int);
          }
          acc0 /= 1000;
          acc1 /= 1000;
          acc2 /= 1000;
          acc3 /= 1000;
          // ','.join([f"acc{j}[{i}]" for j in range(4) for i in range(32)])
          *((unsigned_char128*)((data0+(alu0+alu1)))) = (unsigned_char128){acc0[0],acc0[1],acc0[2],acc0[3],acc0[4],acc0[5],acc0[6],acc0[7],acc0[8],acc0[9],acc0[10],acc0[11],acc0[12],acc0[13],acc0[14],acc0[15],acc0[16],acc0[17],acc0[18],acc0[19],acc0[20],acc0[21],acc0[22],acc0[23],acc0[24],acc0[25],acc0[26],acc0[27],acc0[28],acc0[29],acc0[30],acc0[31],acc1[0],acc1[1],acc1[2],acc1[3],acc1[4],acc1[5],acc1[6],acc1[7],acc1[8],acc1[9],acc1[10],acc1[11],acc1[12],acc1[13],acc1[14],acc1[15],acc1[16],acc1[17],acc1[18],acc1[19],acc1[20],acc1[21],acc1[22],acc1[23],acc1[24],acc1[25],acc1[26],acc1[27],acc1[28],acc1[29],acc1[30],acc1[31],acc2[0],acc2[1],acc2[2],acc2[3],acc2[4],acc2[5],acc2[6],acc2[7],acc2[8],acc2[9],acc2[10],acc2[11],acc2[12],acc2[13],acc2[14],acc2[15],acc2[16],acc2[17],acc2[18],acc2[19],acc2[20],acc2[21],acc2[22],acc2[23],acc2[24],acc2[25],acc2[26],acc2[27],acc2[28],acc2[29],acc2[30],acc2[31],acc3[0],acc3[1],acc3[2],acc3[3],acc3[4],acc3[5],acc3[6],acc3[7],acc3[8],acc3[9],acc3[10],acc3[11],acc3[12],acc3[13],acc3[14],acc3[15],acc3[16],acc3[17],acc3[18],acc3[19],acc3[20],acc3[21],acc3[22],acc3[23],acc3[24],acc3[25],acc3[26],acc3[27],acc3[28],acc3[29],acc3[30],acc3[31]};
        }
      }
    }"""
    self.test_prequant_gemm_intacc(np.uint8, np.int8, src)

  def test_prequant_gemm_intacc(self, xi=np.uint8, wi=np.uint8, replace_src=None):
    N = 512
    X = Tensor(np.random.uniform(0, 255, size=(N,N)).astype(xi))
    W = Tensor(np.random.uniform(0, 255, size=(N,N)).astype(wi))
    # ugh, it's so broken with those casts. need DONT_REALIZE_EXPAND=1 python3 test/test_quantize_onnx.py TestQuantizeOnnx.test_prequant
    with Context(DONT_REALIZE_EXPAND=1):
      out = (X.int().matmul(W.int())//1000).cast('int8' if xi == np.int8 else 'uint8')
      opts = [Opt(op=OptOps.UPCAST, axis=1, arg=128), Opt(op=OptOps.UNROLL, axis=0, arg=4)]
      sexec(out, opts, replace_src)

  def test_prequant_gemm_intacc_wi(self): self.test_prequant_gemm_intacc(wi=np.int8)
  def test_prequant_gemm_intacc_xiwi(self): self.test_prequant_gemm_intacc(xi=np.int8, wi=np.int8)

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
