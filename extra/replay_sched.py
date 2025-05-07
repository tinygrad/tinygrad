import pickle, sys
from tinygrad import TinyJit, Context, Device
from tinygrad.helpers import colored
from tinygrad.engine.realize import CompiledRunner
from tinygrad.codegen.kernel import Kernel

# for master:
# generates /tmp/model.quant.onnx
# CNT=2 NOOPT=1 DEBUG=2 CPU=1 QUANT=1 python3 examples/test_onnx_imagenet.py https://github.com/xamcat/mobcat-samples/raw/refs/heads/master/onnx_runtime/InferencingSample/InferencingSample/mobilenetv2-7.onnx

# run the model on metal, CPU fails.
# CNT=2 NOOPT=1 DEBUG=2 METAL=1 JIT_BATCH_SIZE=0 ALIGNED=0 QUANTIZE=1 DONT_GROUP_REDUCES=1 NHWC=1 IGNORE_OOB=1 DEVECTORIZE=0 DONT_REALIZE_EXPAND=1 python3 examples/test_onnx_imagenet.py /tmp/model.quant.onnx
# CPU=1 QUANTIZE=1 python ./extra/replay_sched.py /tmp/im.pkl

if __name__ == "__main__":
  with Context(DEBUG=0):
    with open(sys.argv[1], "rb") as f:
      fxn: TinyJit = pickle.load(f)
      print(f"{f.tell()/1e6:.2f}M loaded")
    print(type(fxn))

  failed = 0
  for ei in fxn.captured.jit_cache:
    # skip the copy and the first kernel
    if not isinstance(ei.prg, CompiledRunner) or any(x is None for x in ei.bufs): continue
    k = Kernel(ei.prg.p.ast, opts=Device[Device.DEFAULT].renderer)
    try:
      k.to_program()
    except:
      failed += 1
      continue

  print(f"{failed} kernels failed to linearize")
