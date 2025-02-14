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

def sexec(out:Tensor, opts:list[Opt], replace_src=None, run_count=3):
  si = out.schedule()[-1]
  k = Kernel(si.ast, opts=Device[Device.DEFAULT].renderer)
  #opts = [Opt(op=OptOps.UPCAST, axis=0, arg=128)] #, Opt(op=OptOps.UNROLL, axis=0, arg=4)]
  for opt in opts: k.apply_opt(opt)
  prg = k.to_program()
  if replace_src is not None: prg = replace(prg, src=replace_src + "/* DSP boilerplate */" + prg.src.split("/* DSP boilerplate */")[1])
  ei = ExecItem(CompiledRunner(prg), [x.ensure_allocated() for x in si.bufs], si.metadata)
  for _ in range(run_count): ei.run(wait=True)

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
    typedef unsigned char unsigned_char256 __attribute__((aligned(256),vector_size(256)));
    union V256 {
      unsigned_char256 vec256;
      struct {
        unsigned_char128 lo128;
        unsigned_char128 hi128;
      };
    };
    __attribute__((noinline)) void r_512_4_128_128_4(unsigned char* restrict __attribute__((align_value(128))) data0, unsigned char* restrict __attribute__((align_value(128))) data1, signed char* restrict __attribute__((align_value(128))) data2) {
      for (int ridx0 = 0; ridx0 < 512; ridx0++) {
        int alu0 = (ridx0<<9);
        for (int ridx1 = 0; ridx1 < 4; ridx1++) {
          int alu1 = (ridx1<<7);
          int32 acc0 = __builtin_HEXAGON_V6_vd0_128B();
          int32 acc1 = __builtin_HEXAGON_V6_vd0_128B();
          int32 acc2 = __builtin_HEXAGON_V6_vd0_128B();
          int32 acc3 = __builtin_HEXAGON_V6_vd0_128B();

          for (int ridx2 = 0; ridx2 < 128; ridx2++) {
            unsigned_char4 val0 = *((unsigned_char4*)((data1+(alu0+(ridx2<<2)))));
            int alu2 = (alu1+(ridx2<<11));
            signed_char128 x0 = *((signed_char128*)((data2+alu2)));
            signed_char128 x1 = *((signed_char128*)((data2+(alu2+512))));
            signed_char128 x2 = *((signed_char128*)((data2+(alu2+1024))));
            signed_char128 x3 = *((signed_char128*)((data2+(alu2+1536))));

            union V256 ss01;
            // ss01.lo128 = (x0[0], x1[0], x0[2], x1[2], x0[4], x1[4], ...)
            // ss01.hi128 = (x0[1], x1[1], x0[3], x1[3], x0[5], x1[5], ...)
            ss01.vec256 = __builtin_HEXAGON_V6_vshufoeb_128B(x1, x0);

            union V256 ss23;
            // ss23.lo128 = (x2[0], x3[0], x2[2], x3[2], x2[4], x3[4], ...)
            // ss23.hi128 = (x2[1], x3[1], x2[3], x3[3], x2[5], x3[5], ...)
            ss23.vec256 = __builtin_HEXAGON_V6_vshufoeb_128B(x3, x2);

            union V256 sslo;
            // sslo.lo128 = (x0[0], x1[0], x2[0], x3[0], x0[4], x1[4], ...)
            // sslo.hi128 = (x0[2], x1[2], x2[2], x3[2], x0[6], x1[6], ...)
            sslo.vec256 = __builtin_HEXAGON_V6_vdealvdd_128B(ss23.lo128, ss01.lo128, 2);

            union V256 sshi;
            // sshi.lo128 = (x0[1], x1[1], x2[1], x3[1], x0[5], x1[5], ...)
            // sshi.hi128 = (x0[3], x1[3], x2[3], x3[3], x0[7], x1[7], ...)
            sshi.vec256 = __builtin_HEXAGON_V6_vdealvdd_128B(ss23.hi128, ss01.hi128, 2);

            //signed_char128 sw0 = (signed_char128){x0[0],x1[0],x2[0],x3[0],x0[1],x1[1],x2[1],x3[1],x0[2],x1[2],x2[2],x3[2],x0[3],x1[3],x2[3],x3[3],x0[4],x1[4],x2[4],x3[4],x0[5],x1[5],x2[5],x3[5],x0[6],x1[6],x2[6],x3[6],x0[7],x1[7],x2[7],x3[7],x0[8],x1[8],x2[8],x3[8],x0[9],x1[9],x2[9],x3[9],x0[10],x1[10],x2[10],x3[10],x0[11],x1[11],x2[11],x3[11],x0[12],x1[12],x2[12],x3[12],x0[13],x1[13],x2[13],x3[13],x0[14],x1[14],x2[14],x3[14],x0[15],x1[15],x2[15],x3[15],x0[16],x1[16],x2[16],x3[16],x0[17],x1[17],x2[17],x3[17],x0[18],x1[18],x2[18],x3[18],x0[19],x1[19],x2[19],x3[19],x0[20],x1[20],x2[20],x3[20],x0[21],x1[21],x2[21],x3[21],x0[22],x1[22],x2[22],x3[22],x0[23],x1[23],x2[23],x3[23],x0[24],x1[24],x2[24],x3[24],x0[25],x1[25],x2[25],x3[25],x0[26],x1[26],x2[26],x3[26],x0[27],x1[27],x2[27],x3[27],x0[28],x1[28],x2[28],x3[28],x0[29],x1[29],x2[29],x3[29],x0[30],x1[30],x2[30],x3[30],x0[31],x1[31],x2[31],x3[31]};
            unsigned_char128 w0 = (unsigned_char128){val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3],val0[0],val0[1],val0[2],val0[3]};
            acc0 = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(acc0, w0, sslo.lo128);
            acc1 = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(acc1, w0, sshi.lo128);
            acc2 = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(acc2, w0, sslo.hi128);
            acc3 = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(acc3, w0, sshi.hi128);
          }
          acc0 /= 1000;
          acc1 /= 1000;
          acc2 /= 1000;
          acc3 /= 1000;
          // ','.join([f"acc{j}[{i}]" for i in range(32) for j in range(4)])
          *((unsigned_char128*)((data0+(alu0+alu1)))) = (unsigned_char128){acc0[0],acc1[0],acc2[0],acc3[0],acc0[1],acc1[1],acc2[1],acc3[1],acc0[2],acc1[2],acc2[2],acc3[2],acc0[3],acc1[3],acc2[3],acc3[3],acc0[4],acc1[4],acc2[4],acc3[4],acc0[5],acc1[5],acc2[5],acc3[5],acc0[6],acc1[6],acc2[6],acc3[6],acc0[7],acc1[7],acc2[7],acc3[7],acc0[8],acc1[8],acc2[8],acc3[8],acc0[9],acc1[9],acc2[9],acc3[9],acc0[10],acc1[10],acc2[10],acc3[10],acc0[11],acc1[11],acc2[11],acc3[11],acc0[12],acc1[12],acc2[12],acc3[12],acc0[13],acc1[13],acc2[13],acc3[13],acc0[14],acc1[14],acc2[14],acc3[14],acc0[15],acc1[15],acc2[15],acc3[15],acc0[16],acc1[16],acc2[16],acc3[16],acc0[17],acc1[17],acc2[17],acc3[17],acc0[18],acc1[18],acc2[18],acc3[18],acc0[19],acc1[19],acc2[19],acc3[19],acc0[20],acc1[20],acc2[20],acc3[20],acc0[21],acc1[21],acc2[21],acc3[21],acc0[22],acc1[22],acc2[22],acc3[22],acc0[23],acc1[23],acc2[23],acc3[23],acc0[24],acc1[24],acc2[24],acc3[24],acc0[25],acc1[25],acc2[25],acc3[25],acc0[26],acc1[26],acc2[26],acc3[26],acc0[27],acc1[27],acc2[27],acc3[27],acc0[28],acc1[28],acc2[28],acc3[28],acc0[29],acc1[29],acc2[29],acc3[29],acc0[30],acc1[30],acc2[30],acc3[30],acc0[31],acc1[31],acc2[31],acc3[31]};
          // ','.join([f"acc{j}[{i}]" for j in range(4) for i in range(32)])
          //*((unsigned_char128*)((data0+(alu0+alu1)))) = (unsigned_char128){acc0[0],acc0[1],acc0[2],acc0[3],acc0[4],acc0[5],acc0[6],acc0[7],acc0[8],acc0[9],acc0[10],acc0[11],acc0[12],acc0[13],acc0[14],acc0[15],acc0[16],acc0[17],acc0[18],acc0[19],acc0[20],acc0[21],acc0[22],acc0[23],acc0[24],acc0[25],acc0[26],acc0[27],acc0[28],acc0[29],acc0[30],acc0[31],acc1[0],acc1[1],acc1[2],acc1[3],acc1[4],acc1[5],acc1[6],acc1[7],acc1[8],acc1[9],acc1[10],acc1[11],acc1[12],acc1[13],acc1[14],acc1[15],acc1[16],acc1[17],acc1[18],acc1[19],acc1[20],acc1[21],acc1[22],acc1[23],acc1[24],acc1[25],acc1[26],acc1[27],acc1[28],acc1[29],acc1[30],acc1[31],acc2[0],acc2[1],acc2[2],acc2[3],acc2[4],acc2[5],acc2[6],acc2[7],acc2[8],acc2[9],acc2[10],acc2[11],acc2[12],acc2[13],acc2[14],acc2[15],acc2[16],acc2[17],acc2[18],acc2[19],acc2[20],acc2[21],acc2[22],acc2[23],acc2[24],acc2[25],acc2[26],acc2[27],acc2[28],acc2[29],acc2[30],acc2[31],acc3[0],acc3[1],acc3[2],acc3[3],acc3[4],acc3[5],acc3[6],acc3[7],acc3[8],acc3[9],acc3[10],acc3[11],acc3[12],acc3[13],acc3[14],acc3[15],acc3[16],acc3[17],acc3[18],acc3[19],acc3[20],acc3[21],acc3[22],acc3[23],acc3[24],acc3[25],acc3[26],acc3[27],acc3[28],acc3[29],acc3[30],acc3[31]};
        }
      }
    }"""
    self.test_prequant_gemm_intacc(np.uint8, np.int8, src)

  def test_prequant_gemm_intacc_128(self): self.test_prequant_gemm_intacc(np.uint8, np.int8, N=128)
  def test_prequant_gemm_intacc_256(self): self.test_prequant_gemm_intacc(np.uint8, np.int8, N=256)
  def test_prequant_gemm_intacc(self, xi=np.uint8, wi=np.uint8, replace_src=None, N=512):
    X = Tensor(m1:=(np.random.uniform(0, 255, size=(N,N)).astype(xi))).realize()
    W = Tensor(m2:=(np.random.uniform(0, 255, size=(N,N)).astype(wi))).realize()
    # ugh, it's so broken with those casts. need DONT_REALIZE_EXPAND=1 python3 test/test_quantize_onnx.py TestQuantizeOnnx.test_prequant
    with Context(DONT_REALIZE_EXPAND=1):
      out = (X.int().matmul(W.int())//1000).cast('int8' if xi == np.int8 else 'uint8')
      opts = [Opt(op=OptOps.UPCAST, axis=1, arg=128), Opt(op=OptOps.UNROLL, axis=0, arg=4)]
      sexec(out, opts, replace_src, run_count=1)
    tout = out.numpy()
    mout = ((m1.astype(np.int32) @ m2.astype(np.int32)) / 1000).astype(xi)
    print(tout)
    print(mout)
    np.testing.assert_equal(tout, mout)

  def test_prequant_gemm_intacc_wi(self): self.test_prequant_gemm_intacc(wi=np.int8)
  def test_prequant_gemm_intacc_xiwi(self): self.test_prequant_gemm_intacc(xi=np.int8, wi=np.int8)

  def test_prequant_gemv(self):
    N = 2048
    # ugh, it's so broken with those casts. need DONT_REALIZE_EXPAND=1 python3 test/test_quantize_onnx.py TestQuantizeOnnx.test_prequant
    X = Tensor(np.random.uniform(0, 255, size=(1,N)).astype(np.uint8)).realize()
    W = Tensor(np.random.uniform(0, 255, size=(N,N)).astype(np.uint8)).realize()
    #out = X.cast(dtypes.int) @ W.cast(dtypes.int)
    #out = X @ W
    out = X.matmul(W, acc_dtype=X.dtype)
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=128), Opt(op=OptOps.UNROLL, axis=0, arg=4)]
    sexec(out, opts)

if __name__ == "__main__":
  unittest.main()
