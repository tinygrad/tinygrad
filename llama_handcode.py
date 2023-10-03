import plotly.express as px
import pandas as pd
from extra.utils import print_tree
from tinygrad.helpers import ansilen
from tinygrad.runtime.ops_metal import renderer, MetalProgram, RawMetalBuffer
from pathlib import Path
from examples.llama import LLaMa
# llama = LLaMa.build(model_path=, tokenizer_path=Path("./weights/LLaMA/7B/tokenizer.model"))

# 1. it shouldn't fuck up your computer when you load in the weights (esp if you're also playing music')

from pathlib import Path
from extra.gguf import gguf_load
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.ops import LoadOps
from tinygrad.runtime.ops_metal import RawMetalBuffer

model = Path("./weights/llama2-7b-q4/llama-2-7b.Q4_0.gguf")
weights, params = gguf_load(model)
load_times = []
for k,v in weights.items():
  total_tm = 0; total_gflops = 0
  sched = [x for x in v.lazydata.schedule() if x[0].op not in LoadOps]
  for i, (op,out,inp) in enumerate(sched):
    lin = Linearizer(op, LinearizerOptions(device="METAL"))
    lin.process()
    lin.hand_coded_optimizations(use_tensor_cores=1)
    lin.linearize()

    code = renderer(lin.function_name, lin.uops)
    # print(code)
    prg = MetalProgram(lin.function_name, code)
    rout = RawMetalBuffer(out.st.size(), out.dtype)
    rin = [RawMetalBuffer(x.st.size(), x.dtype) for x in inp]
    tm = prg(lin.global_size, lin.local_size, rout, *rin, wait=True)
    gflops = lin.info.flops*1e-9/tm
    total_tm += tm; total_gflops += gflops
    print(f"kernel {i:2d} {lin.display_name+' '*(37-ansilen(lin.display_name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS")
  print(f"--- tensor {k} total {total_tm*1000:7.2f} ms")
  load_times.append((k, total_tm*1000, total_gflops))

loads = pd.DataFrame(load_times, columns=["layer", "time", "gflops"]).sort_values("time", ascending=False)
fig = px.bar(loads, x="layer", y="gflops")
fig.show()
