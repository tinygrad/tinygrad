import plotly.express as px
import pandas as pd
from extra.utils import print_tree
from tinygrad.helpers import ansilen
from tinygrad.runtime.ops_metal import renderer, MetalProgram, RawMetalBuffer
from pathlib import Path
from examples.llama import LLaMa, Transformer, MODEL_PARAMS

from pathlib import Path
from extra.gguf import gguf_load
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.ops import LoadOps
from tinygrad.runtime.ops_metal import RawMetalBuffer
from tinygrad.tensor import Tensor

"""
mdl = Transformer(**MODEL_PARAMS["2"]["7B"]["args"])
seen = set()
mdl(tokens=Tensor.empty(64, 128), start_pos=0).lazydata.schedule(seen)
"""
model_path = Path("./weights/llama2-7b-q4/llama-2-7b.Q4_0.gguf")
weights, params = gguf_load(model_path)
load_ops = []
for k,(v, _dtype) in weights.items():
  # if k != "output.weight": continue
  total_tm = 0
  sched = [x for x in v.lazydata.schedule() if x[0].op not in LoadOps]
  for i, (op,out,inp) in enumerate(sched):
    lin = Linearizer(op, LinearizerOptions(device="METAL"))
    lin.process()
    lin.hand_coded_optimizations(use_tensor_cores=1)
    lin.linearize()

    code = renderer(lin.function_name, lin.uops)
    prg = MetalProgram(lin.function_name, code)
    rout = RawMetalBuffer(out.st.size(), out.dtype)
    rin = [RawMetalBuffer(x.st.size(), x.dtype) for x in inp]
    tm = prg(lin.global_size, lin.local_size, rout, *rin, wait=True)
    gflops = lin.info.flops*1e-9/tm
    print(f"kernel {i:2d} {lin.display_name+' '*(37-ansilen(lin.display_name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS")
    load_ops.append((f"{k} ({_dtype})", op.op, tm*1000, gflops, code))

  print(f"--- tensor {k} total {total_tm*1000:7.2f} ms")

loads = pd.DataFrame(load_ops, columns=["layer", "op", "time", "gflops", "code"])
loads.to_csv("loads.csv")
print(loads.head())
