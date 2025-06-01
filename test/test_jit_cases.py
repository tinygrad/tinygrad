import unittest
from tinygrad import TinyJit, Tensor
from tinygrad.renderer.graph import GraphRenderer
from tinygrad.uop.ops import UOp, Ops
from tinygrad.engine.jit import get_input_replace
from tinygrad.device import Buffer
from typing import cast

# The JIT functions as a "capturing" JIT.
# Whatever kernels ran in the JIT the second run through the function will be the kernels that will run from then on.
# Explicit inputs to the function are updated in the JIT graph to the new inputs.

# JITs have four tensor types
#  1. Tensors that are explicit in the input, aka what's passed in. TODO: support lists/dicts/classes, anything get_state works on
#  2. Tensors that are explicit in the output, aka what's returned. TODO: same as above
#  3. Tensors that are implicit in the input as a closure.
#  4. Tensors that are implicit in the output because they were assigned to and realized.

# explicit inputs and outputs are realized on their way in and out of the JIT
# there's a whole bunch of edge cases and weirdness here that needs to be tested and clarified.

# This prototype jit is for testing whether GraphRenderer captures graphs correctly
def jit_v2(fxn, *args, **kwargs):
  r = GraphRenderer(fxn, *args, **kwargs)

  r_args = [arg.buffer if arg.op is Ops.BUFFER else arg for arg in r.inputs]
  r_buf_args = [arg for arg in r_args if isinstance(arg, Buffer)]
  map_input_replace_idx_to_original_idx = {i: r_args.index(buf) for i, buf in enumerate(r_buf_args)}
  input_replace = get_input_replace(r.eis, r_buf_args)
  input_replace.update({k: map_input_replace_idx_to_original_idx[v] for k,v in input_replace.items()})

  explicit_output_j_i: dict[int, tuple[int, int]] = dict()
  for idx, out in enumerate(r.outputs):
    for j, ei in enumerate(r.eis):
      for i, buf in enumerate(ei.bufs):
        if buf == out.buffer:
          explicit_output_j_i[idx] = (j, i)

  # NOTE: The graph captured by GraphRenderer only takes positional Tensor/Variable args (no kwargs)
  def captured_jit_v2(*args: Tensor|UOp):
    if args: Tensor.realize(*[arg for arg in args if isinstance(arg, Tensor)])
    var_vals = dict(v.unbind() for v in args if isinstance(v, UOp))
    for (j,i),input_idx in input_replace.items():
      assert isinstance(args[input_idx], Tensor)
      b = cast(Tensor, args[input_idx]).lazydata.base.realized
      assert isinstance(b, Buffer)
      r.eis[j].bufs[i] = b
    for ei in r.eis: ei.run(var_vals)
    # NOTE: GraphRenderer captures only ret buffers, not Tensors, so we need to wrap the ret buffer in a Tensor to be consistent with TinyJit
    assert all(isinstance(b, Buffer) for ei in r.eis for b in ei.bufs)

    # Implicit input buffers can change during execution, so we have to copy the data from the new buffer back to the original buffer
    for dest, src in r.implicit_input_copies:
      dest.buffer.copyin(src.buffer.as_buffer())

    ret = [Tensor(bytes((b:=cast(Buffer, r.eis[j].bufs[i])).as_buffer()), b.device, b.dtype) for j,i in explicit_output_j_i.values()]
    return ret[0] if len(ret) == 1 else ret

  return captured_jit_v2

def check_realize(t:Tensor, v2=False):
  if not v2: return t.realize()
  else: return t

class TestJitCases(unittest.TestCase):
  def test_explicit(self, v2=False):
    # this function has an explicit input and an explicit output
    def f(x:Tensor):
      ret:Tensor = x*2
      return ret
    f = TinyJit(f) if not v2 else jit_v2(f, Tensor([0]))

    for i in range(5):
      out = f(Tensor([i]))
      self.assertEqual(out.item(), i*2)

  def test_explicit_v2(self):
    self.test_explicit(v2=True)

  def test_implicit_input(self, v2=False):
    # x is the implicit input (like a weight)
    x = Tensor([0])

    # this function has an implicit input and an explicit output
    def f():
      ret:Tensor = x*2
      return ret
    f = TinyJit(f) if not v2 else jit_v2(f)

    for i in range(5):
      # NOTE: this must be realized here, otherwise the update doesn't happen
      # if we were explicitly tracking the implicit input Tensors, we might not need this realize
      x.assign(Tensor([i])).realize()
      out = f()
      self.assertEqual(out.item(), i*2)

  def test_implicit_input_v2(self):
    self.test_implicit_input(v2=True)

  def test_implicit_output(self, v2=False):
    # out is the implicit output (it's assigned to)
    out = Tensor([0])

    # this function has an explicit input and an implicit output
    def f(x:Tensor):
      # NOTE: this must be realized here
      # if we were explicitly tracking the implicit output Tensors, we might not need this realize
      check_realize(out.assign(x*2), v2)
    f = TinyJit(f) if not v2 else jit_v2(f, Tensor([0]))

    for i in range(5):
      f(Tensor([i]))
      self.assertEqual(out.item(), i*2)

  def test_implicit_output_v2(self):
    self.test_implicit_output(v2=True)

  def test_implicit_io(self, v2=False):
    # x is the implicit input (like a weight)
    # out is the implicit output (it's assigned to)
    x = Tensor([0])
    out = Tensor([0])

    # this function has an implicit input and an implicit output
    def f():
      check_realize(out.assign(x*2), v2) # NOTE: this must be realized here
    f = TinyJit(f) if not v2 else jit_v2(f)

    for i in range(5):
      x.assign(Tensor([i])).realize()
      f()
      self.assertEqual(out.item(), i*2)

  def test_implicit_io_v2(self):
    self.test_implicit_io(v2=True)

  def test_mutate_implicit_input(self, v2=False):
    w = Tensor([0])

    def f(x:Tensor):
      check_realize(w.assign(w + 1), v2)
      x = check_realize(x + w, v2)
      check_realize(w.assign(w + 1), v2)
      return x

    f = TinyJit(f) if not v2 else jit_v2(f, Tensor([123]))

    for i in range(5):
      out = f(Tensor([42]))
      self.assertEqual(out.item(), 42 + 2*i + 1)
      self.assertEqual(w.item(), 2*(i + 1))

  def test_mutate_implicit_input_v2(self):
    self.test_mutate_implicit_input(v2=True)

if __name__ == '__main__':
  unittest.main()
