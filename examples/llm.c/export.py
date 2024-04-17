#!/usr/bin/env python3
import os
os.environ["NOOPT"] = "1"
from tinygrad import Device, nn, Tensor, dtypes
Device.DEFAULT = "CLANG"
from train_gpt2 import GPT, GPTConfig
from tinygrad.helpers import dedup, to_function_name
from tinygrad.engine.schedule import create_schedule
from tinygrad.ops import BufferOps, LoadOps
from tinygrad.runtime.ops_clang import CLANG_PROGRAM_HEADER

if __name__ == "__main__":
  model = GPT(GPTConfig(n_layer=12, n_head=12, n_embd=768))
  seen = set()
  early_sched = create_schedule([x.lazydata for x in nn.state.get_parameters(model)], seen)
  print(f"built model {len(early_sched)}")

  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-4)
  x = Tensor.empty(4, 64, dtype=dtypes.int)
  y = Tensor.empty(4, 64, dtype=dtypes.int)
  _, loss = model(x, y)
  optimizer.zero_grad()
  loss.backward()
  tensors = optimizer._step()
  sched = create_schedule([loss.lazydata] + [x.lazydata for x in tensors], seen)
  print(len(sched))
  ast_dedup = dedup([si.ast for si in sched if si.ast[0].op is BufferOps.STORE])
  print(len(ast_dedup))
  srcs = {}
  for ast in ast_dedup:
    k = Device["CLANG"].get_linearizer(*ast)
    k.linearize()
    src = Device["CLANG"].compiler.render(to_function_name(k.name), k.uops).strip(CLANG_PROGRAM_HEADER)
    srcs[ast] = (k.name, src)
  print(len(srcs))

  for si in sched:
    if si.ast[0].op == LoadOps.EMPTY: continue
    print(srcs[si.ast][0])