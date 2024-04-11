# abstractions2 goes from back to front, here we will go from front to back
from typing import Tuple, Dict, List
from dataclasses import dataclass

# *****
# 0. Load mnist on the device

from tinygrad.features.datasets import mnist
X_train, Y_train, X_test, Y_test = mnist()

# *****
# 1. Define an MNIST model.

from tinygrad import Tensor

l1 = Tensor.kaiming_uniform(784, 128)
l2 = Tensor.kaiming_uniform(128, 10)
def model(x): return x.flatten(1).dot(l1).relu().dot(l2)

# *****
# 2. Choose a batch for training and do the backward pass.

from tinygrad.nn.optim import SGD
optim = SGD([l1, l2])

X, Y = X_train[samples:=Tensor.randint(512, high=X_train.shape[0])], Y_train[samples]
optim.zero_grad()
model(X).sparse_categorical_crossentropy(Y).backward()
optim._step()

# *****
# 3. Create a schedule.

# The weight Tensors have been assigned to, but not yet realized. Everything is still lazy at this point
# l1.lazydata and l2.lazydata define a computation graph

from tinygrad.engine.schedule import create_schedule_with_vars
schedule, var_vals = create_schedule_with_vars([l1.lazydata, l2.lazydata])

# Once scheduled, all the computation is put in a line.

from tinygrad.ops import LazyOp
from tinygrad.buffer import Buffer

@dataclass(frozen=True)
class ScheduleItem:
  ast: Tuple[LazyOp, ...]
  outputs: Tuple[Buffer, ...]    # NOTE: if a buffer is both read from and written to, it appears in both outputs and inputs
  inputs: Tuple[Buffer, ...]

print(f"The schedule contains {len(schedule)} items.")
for si in schedule: print(str(si)[:80])

# *****************************
# TODO: below this point isn't the real API yet, but it needs to change from the current mess
# *****************************

# *****
# 4. Lower a schedule.

from tinygrad.device import JITRunner, Device, Compiled, BufferXfer, BufferCopy
from tinygrad.shape.symbolic import sint
from tinygrad.ops import LoadOps, BufferOps
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.helpers import to_function_name
from tinygrad.engine.realize import CustomOp, EmptyOp

@dataclass(frozen=True)
class ExecItem:
  prg: JITRunner               # wrapper class does [x._buf for x in ei.buffers], [sym_infer(x, var_vals) for x in vals]
  buffers: List[Buffer]        # NOTE: this can have some outputs/inputs removed and will always be deduped
  vals: Tuple[sint, ...] = ()  # NOTE: this includes global_size and local_size as the first 6

class CompiledKernel(JITRunner):
  def __init__(self, name:str, prg:str, dname:str, op_estimate:sint=0, mem_estimate:sint=0):
    super().__init__()

def lower_schedule_item(si:ScheduleItem) -> ExecItem:
  assert len(set(x.device for x in si.outputs+si.inputs)) == 1 or si.ast[0].op is LoadOps.COPY
  buffers, dname = list(si.outputs+si.inputs), si.outputs[0].device
  if si.ast[0].op is BufferOps.STORE:
    k = Linearizer(*si.ast, opts=Device[dname].compiler.compiler_opts)
    k.hand_coded_optimizations()
    k.linearize()
    ck = CompiledKernel(k.name, Device[dname].compiler.render(to_function_name(k.name), k.uops), dname, *k.uops.flops_mem())
    return ExecItem(ck, buffers, k.global_size+k.local_size+k.uops.vars())
  out, ast = si.outputs[0], si.ast[0]
  if ast.op is LoadOps.COPY:
    use_xfer = hasattr(Device[out.device].allocator, 'transfer') and out.device.split(":")[0] == si.inputs[0].device.split(":")[0]
    return ExecItem(BufferXfer() if use_xfer else BufferCopy(), buffers)
  if ast.op is LoadOps.CUSTOM: return ExecItem(CustomOp(ast.arg), buffers)
  if ast.op is LoadOps.EMPTY: return ExecItem(EmptyOp(), buffers)
  raise RuntimeError(f"don't know how to lower {ast}")

# To call: ei.prg(ei.buffers, ei.vals, var_vals: Dict[Variable, int])
lowered: List[ExecItem] = [lower_schedule_item(si) for si in schedule]

# *****
# 5. (Optional) Group several ExecItem into a single ExecItem with a "graph" API

# *****
# 6. Run the schedule

# NOTE: var_vals changes based on calls to the JIT, ei.buffers can change too
# TODO: separate the concept of input/intermediate/output buffers? is the idea if they are bound to LazyBuffers or not?

for ei in lowered:
  for b in ei.buffers: b.ensure_allocated()
  ei.prg(ei.buffers, ei.vals, var_vals)
