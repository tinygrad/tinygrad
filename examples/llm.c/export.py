#!/usr/bin/env python3
import os
if "NOOPT" not in os.environ: os.environ["NOOPT"] = "1"
from tinygrad import Device, nn, Tensor, dtypes
Device.DEFAULT = "CLANG"
from train_gpt2 import GPT, GPTConfig
from tinygrad.helpers import dedup, to_function_name, flatten, getenv, GRAPH, GlobalCounters, ansilen, to_function_name
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import memory_planner, run_schedule
from tinygrad.ops import BufferOps, LoadOps
from tinygrad.runtime.ops_clang import CLANG_PROGRAM_HEADER

if __name__ == "__main__":
  model = GPT(GPTConfig(n_layer=getenv("NLAYER", 12), n_head=12, n_embd=768))
  #model.load_pretrained()
  for p in nn.state.get_parameters(model): p.replace(Tensor.empty(p.shape, dtype=p.dtype)) # fake load pretrained

  seen = set()
  #early_sched = create_schedule([x.lazydata for x in nn.state.get_parameters(model)], seen)
  #print(f"built model {len(early_sched)}")

  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-4)
  warmup_count = getenv("WARMUP", 3)
  for i in range(warmup_count):  # TODO: why does it take three and not two to stablize
    if i == warmup_count-1: GRAPH.value = getenv("LATEGRAPH")
    GlobalCounters.reset()
    X = Tensor.empty(4, 64, dtype=dtypes.int)
    Y = Tensor.empty(4, 64, dtype=dtypes.int)
    _, loss = model(X, Y)
    optimizer.zero_grad()
    if getenv("BACKWARD", 1):
      loss.backward()
      tensors = optimizer.schedule_step()
    else:
      tensors = []
    sched = create_schedule([loss.lazydata] + [x.lazydata for x in tensors], seen)
    print(f"calls {i}:", len(sched))
    #run_schedule(sched[:])
  del seen  # free the LazyBuffers
  sched = memory_planner(sched)
  ast_dedup = dedup([si.ast for si in sched if si.ast[0].op is BufferOps.STORE])
  srcs = {}
  for ast in ast_dedup:
    k = Device["CLANG"].get_linearizer(*ast)
    k.linearize()
    src = Device["CLANG"].compiler.render(to_function_name(k.name), k.uops).strip(CLANG_PROGRAM_HEADER)
    srcs[ast] = (k.name, src)
  print("functions:", len(srcs))
  used_buffers = dedup(flatten([si.outputs+si.inputs for si in sched]))
  numbered_bufs = {x:i for i,x in enumerate(used_buffers)}
  print("buffers:", len(numbered_bufs))

  state_dict = nn.state.get_state_dict(model)
  state_dict.update({'X': X, 'Y': Y, 'loss': loss})
  for k,v in state_dict.items():
    if v.lazydata.base.buffer not in used_buffers: print(f"UNUSED: {k}")
  state_dict.update({'adam_b1': optimizer.b1, 'adam_b2': optimizer.b2, 'adam_t': optimizer.t, 'adam_lr': optimizer.lr})
  inverse_state_dict = {v:k for k,v in state_dict.items()}
  for p,m,v in zip(optimizer.params, optimizer.m, optimizer.v):
    nm = inverse_state_dict[p]
    state_dict["adam_m_"+nm] = m
    state_dict["adam_v_"+nm] = v
  named_buffers = {v.lazydata.base.buffer:k.replace(".", "_") for k,v in state_dict.items()}

  c_code = [CLANG_PROGRAM_HEADER]
  c_code += [x[1] for x in srcs.values()]

  main = ["int main() {"]
  all_bufs = []
  for i,si in enumerate(sched):
    bufs = [(named_buffers.get(b, f"b{numbered_bufs[b]}"), b) for b in si.outputs+si.inputs]
    all_bufs += bufs
    if si.ast[0].op is not BufferOps.STORE:
      print(f"// {si.ast[0].op}", bufs)
    else:
      print(f"{srcs[si.ast][0]}({', '.join([x[0] for x in bufs])})")
      main.append(f"  {to_function_name(srcs[si.ast][0])}({', '.join([x[0] for x in bufs])});")
      #call = f"{srcs[si.ast][0]}({', '.join(bufs)})"
      #call += " "*(80-ansilen(call))
      #print(f"{call} // {i+1}")
      #print(srcs[si.ast][1])
  main.append("}")

  for n,b in dedup(all_bufs):
    c_code.append(f"{b.dtype.name} {n}[{b.size}];")

  with open("out.c", "w") as f: f.write('\n'.join(c_code+main))