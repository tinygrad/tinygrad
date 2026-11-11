import unittest, pickle, types, subprocess, sys
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ContextVar, Context
from tinygrad.uop.ops import PatternMatcher, UPat, UOp, deconstruct_function
from tinygrad.uop.uir import parse_uir

class TestPickle(unittest.TestCase):
  def test_pickle_code_object(self):
    y = lambda x: x*2  # noqa: E731
    code_str = pickle.dumps(y.__code__)
    fxn = types.FunctionType(pickle.loads(code_str), globals())
    self.assertEqual(fxn(2), 4)

  def test_deconstruct_function_nested_comprehension(self):
    # pre PEP 709, each comprehension is its own code object, so dtypes here is referenced two code objects deep
    def fxn(): return [[dtypes.int for _ in range(2)] for _ in range(2)]
    self.assertEqual(types.FunctionType(*deconstruct_function(fxn))(), fxn())

  def test_pickle_pattern_matcher(self):
    pm = PatternMatcher([(UPat.cvar('x'), lambda x: x*2)])
    sink = UOp.const(2)
    tt = pm.rewrite(sink)
    pm_str = pickle.dumps(pm)
    pm2 = pickle.loads(pm_str)
    self.assertEqual(pm2.rewrite(sink).key, tt.key)

  def test_pickle_main_pattern_matcher(self):
    from tinygrad.uop.symbolic import sym
    ssym = pickle.dumps(sym)
    dsym = pickle.loads(ssym)
    self.assertEqual(dsym.patterns[0][0].location, sym.patterns[0][0].location)

  def test_pickle_context_var(self):
    v = ContextVar("test_var", 0)
    with Context(test_var=1):
      vs = pickle.dumps(v)
    v2 = pickle.loads(vs)
    self.assertEqual(v2.value, 1)

  def test_pickle_schedule(self):
    a = Tensor([1,2])
    out = a + 2
    sched = out.schedule_linear()
    pk = pickle.dumps(sched)
    sched_pk = pickle.loads(pk)
    self.assertEqual(sched_pk.src[-1].src[0], sched.src[-1].src[0])

  def test_pickle_storage_free_buffers_in_subprocess(self):
    root = parse_uir("%a = buffer 4 slot=-1 dtype=f32 device=NULL\n%b = buffer 4 slot=-1 dtype=i32 device=NULL\n%out = sink %a %b")
    code = '''
import pickle, sys
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
if sys.argv[1] == 'before': live = UOp.new_buffer('NULL', 4, dtypes.float32)
a, b = pickle.loads(sys.stdin.buffer.read()).src
if sys.argv[1] == 'after': live = UOp.new_buffer('NULL', 4, dtypes.float32)
assert a.arg.slot == b.arg.slot < 0
assert a.slot_buffer is None and b.slot_buffer is None
assert live.arg.slot >= 0 and live.buffer is not None
assert a is not live and b is not live
'''
    for order in ('before', 'after'):
      with self.subTest(order=order):
        subprocess.run([sys.executable, '-c', code, order], input=pickle.dumps(root), check=True, capture_output=True)

  def test_pickle_renderer(self):
    from tinygrad.device import Device
    pk = pickle.dumps(Device.default.renderer)
    pickle.loads(pk)

if __name__ == '__main__':
  unittest.main()
