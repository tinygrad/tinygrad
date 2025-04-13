from tinygrad import Tensor, Device, TinyJit
from tinygrad.engine.memory import _internal_memory_planner
from tinygrad.engine.realize import ExecItem
Device.DEFAULT="WEBGPU"

def print_jit_memory(run: TinyJit):
  num_bufs = len(set(buf for ji in run.jit_cache for buf in ji.bufs if buf is not None and buf._base is None)) + \
    len(set(buf._base for ji in run.jit_cache for buf in ji.bufs if buf is not None and buf._base is not None)) 
  print(f"  -- {num_bufs} buffers are used by jit")

  jit_mem = sum(buf.nbytes for buf in set(b for ji in run.jit_cache for b in ji.bufs if b is not None and b._base is None)) + \
    sum(buf.nbytes for buf in set(b._base for ji in run.jit_cache for b in ji.bufs if b is not None and b._base is not None))
  print(f"  -- {jit_mem} bytes are used by jit buffers\n")

def fxn() -> Tensor:
  acc = Tensor.zeros(5)
  for _ in range(100):
    a = Tensor.arange(0, 5)
    acc = (acc + a).realize()
  return acc

run = TinyJit(fxn)
for _ in range(2): run()

print("BEFORE new memory planning in JIT")
print_jit_memory(run)

[b.deallocate() for ji in run.jit_cache for b in ji.bufs if b is not None and b.is_allocated() and b.lb_refcount == 0]
assigned = _internal_memory_planner([item.bufs for item in run.jit_cache]) 
run.captured._jit_cache = run.captured.jit_cache = \
  [ExecItem(item.prg, [assigned.get(b,b).ensure_allocated() for b in item.bufs if b is not None]) for item in run.jit_cache] 

print("AFTER new memory planning in JIT")
print_jit_memory(run)

assert run().tolist() == fxn().tolist()