import unittest
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.schedule.allreduce import create_allreduce_function, handle_allreduce, _is_stable_custom_output, is_allreduce_linear_output
from tinygrad.schedule.prepare import prepare_rangeify, _accumulate_linear_allreduce, _accumulate_linear_replicated, walk_mop
from tinygrad.schedule.rangeify import no_indexing_calls
from test.helpers import KernelCountException
from tinygrad.engine.realize import run_linear

class TestRingAllReduce(unittest.TestCase):
  def test_classify_linear_allreduce_output(self):
    devices = ("NULL", "NULL:1")
    out = UOp.param(0, dtypes.float, (8,), device=devices)
    slices = [out.mselect(i).shrink(((4*i, 4*i+4),)).rtag(("allreduce",)) for i in range(2)]
    self.assertTrue(slices[0].has_buffer_identity())
    linear = UOp(Ops.LINEAR, src=tuple(UOp(Ops.CALL, src=(UOp(Ops.SINK), x)) for x in slices))
    self.assertTrue(is_allreduce_linear_output(linear, 0))
    ordinary = UOp(Ops.LINEAR, src=(UOp(Ops.CALL, src=(UOp(Ops.SINK), out)),))
    self.assertFalse(is_allreduce_linear_output(ordinary, 0))

  def test_physical_view_offset_uses_base_dtype_units(self):
    base = UOp.new_buffer("NULL", 64, dtypes.uint8)
    view = base.bitcast(dtypes.float).shrink(((3, 7),)).rtag(("allreduce",))
    self.assertEqual(view.contiguous_view(), (base, 12))

  def test_copy_keeps_zero_offset_physical_view(self):
    src = UOp.param(1, dtypes.float, (64,), device="NULL")
    view = src.shrink(((0, 16),)).rtag(("allreduce",))
    copy_call = UOp(Ops.COPY, src=(UOp.param(1, dtypes.float, (16,), device="NULL"),), arg="NULL:1").call(view, view)
    self.assertEqual(no_indexing_calls(copy_call).src[1:], (view, view))
    sink_call = UOp(Ops.SINK).call(view)
    self.assertEqual(no_indexing_calls(sink_call).src[1:], (view,))

  def test_walk_mop_keeps_physical_view(self):
    src = UOp.param(1, dtypes.float, (64,), device="NULL")
    view = src.shrink(((8, 24),)).rtag(("allreduce",))
    self.assertIs(walk_mop(view), view)

  def test_classify_linear_allreduce_output_with_direct_write(self):
    devices = ("NULL", "NULL:1")
    out = UOp.param(5, dtypes.float, (8,), device=devices)
    slices = [out.mselect(i).shrink(((4*i, 4*i+4),)).rtag(("allreduce",)) for i in range(2)]
    calls = [UOp(Ops.CALL, src=(UOp(Ops.SINK), x)) for x in slices]
    for i in range(2):
      p, idx = UOp.param(0, dtypes.float, (8,)), UOp.const(0)
      sink = UOp.sink(p.index(idx).store(UOp.const(0, dtypes.float)))
      calls.append(UOp(Ops.CALL, src=(sink, out.mselect(i))))
      copy = UOp(Ops.COPY, src=(UOp.param(1, dtypes.float, (4,), device=devices[(i+1)%2]),), arg=devices[i])
      calls.append(UOp(Ops.CALL, dtypes.void, src=(copy, out.mselect(i), slices[(i+1)%2])))
    self.assertTrue(is_allreduce_linear_output(UOp(Ops.LINEAR, src=tuple(calls)), 5))

    p, idx = UOp.param(0, dtypes.float, (8,)), UOp.const(0)
    read_sink = UOp.sink(p.index(idx).load())
    calls.append(UOp(Ops.CALL, src=(read_sink, out.mselect(0))))
    self.assertFalse(is_allreduce_linear_output(UOp(Ops.LINEAR, src=tuple(calls)), 5))

    copy = UOp(Ops.COPY, src=(UOp.param(1, dtypes.float, (4,), device=devices[0]),), arg=devices[1])
    bad_copy = UOp(Ops.CALL, dtypes.void, src=(copy, slices[0], out.mselect(0)))
    self.assertFalse(is_allreduce_linear_output(UOp(Ops.LINEAR, src=tuple(calls[:-1])+(bad_copy,)), 5))

  def test_accumulate_replicated_linear_output_preserves_cast(self):
    devices = ("NULL", "NULL:1")
    output = UOp.param(5, dtypes.bfloat16, (8,), device=devices)
    dest = UOp.param(0, dtypes.bfloat16, (8,), device=devices)
    reduced = UOp.param(1, dtypes.float, (8,), device=devices)
    idx = UOp.range(8, 0)
    cast = reduced.index(idx).cast(dtypes.bfloat16)
    body = UOp.sink(dest.index(idx).store(cast).end(idx))
    linear = UOp(Ops.LINEAR, src=(body.call(output, UOp.param(6, dtypes.float, (8,), device=devices)),))
    accumulated = _accumulate_linear_replicated(linear, 5)
    assert accumulated is not None
    store = next(x for x in accumulated.toposort() if x.op is Ops.STORE)
    self.assertEqual(store.src[1].tag, ("allreduce_accumulate",))
    self.assertIs(store.src[1].src[1], cast)

  def test_accumulate_linear_allreduce_changes_only_owners(self):
    devices = ("NULL", "NULL:1")
    output = UOp.param(5, dtypes.bfloat16, (8,), device=devices)
    views = [[output.mselect(rank).shrink(((4*chunk, 4*chunk+4),)).rtag(("allreduce",)) for chunk in range(2)] for rank in range(2)]
    calls = []
    for rank,chunk in ((0, 0), (1, 1)):
      dest, reduced, idx = UOp.param(0, dtypes.bfloat16, (4,)), UOp.param(1, dtypes.bfloat16, (4,)), UOp.range(4, 0)
      calls.append(UOp.sink(dest.index(idx).store(reduced.index(idx)).end(idx)).call(views[rank][chunk], reduced))
    copy = UOp(Ops.COPY, src=(UOp.param(1, dtypes.bfloat16, (4,), device="NULL"),), arg="NULL:1")
    calls += [copy.call(views[1][0], views[0][0]), copy.call(views[0][1], views[1][1])]
    accumulated = _accumulate_linear_allreduce(UOp(Ops.LINEAR, src=tuple(calls)), 5)
    assert accumulated is not None
    stores = [x for x in accumulated.toposort() if x.op is Ops.STORE]
    self.assertTrue(all(x.src[1].tag == ("allreduce_accumulate",) for x in stores))
    self.assertEqual(sum(any(x.op is Ops.STORE and x.src[1].tag == ("allreduce_accumulate",) for x in call.src[0].toposort())
                         for call in accumulated.src if call.src[0].op is Ops.SINK), 2)
    self.assertEqual(sum(x.op is Ops.COPY for x in accumulated.toposort()), 1)

  def test_accumulate_linear_allreduce_rejects_other_reader(self):
    devices = ("NULL", "NULL:1")
    output = UOp.param(5, dtypes.bfloat16, (8,), device=devices)
    views = [[output.mselect(rank).shrink(((4*chunk, 4*chunk+4),)).rtag(("allreduce",)) for chunk in range(2)] for rank in range(2)]
    calls = []
    for rank,chunk in ((0, 0), (1, 1)):
      dest, reduced, idx = UOp.param(0, dtypes.bfloat16, (4,)), UOp.param(1, dtypes.bfloat16, (4,)), UOp.range(4, 0)
      calls.append(UOp.sink(dest.index(idx).store(reduced.index(idx)).end(idx)).call(views[rank][chunk], reduced))
    copy = UOp(Ops.COPY, src=(UOp.param(1, dtypes.bfloat16, (4,), device="NULL"),), arg="NULL:1")
    calls += [copy.call(views[1][0], views[0][0]), copy.call(views[0][1], views[1][1])]
    dest, inp, idx = UOp.param(0, dtypes.bfloat16, (4,)), UOp.param(1, dtypes.bfloat16, (4,)), UOp.range(4, 0)
    calls.append(UOp.sink(dest.index(idx).store(inp.index(idx)).end(idx)).call(UOp.param(7, dtypes.bfloat16, (4,)), views[0][0]))
    self.assertIsNone(_accumulate_linear_allreduce(UOp(Ops.LINEAR, src=tuple(calls)), 5))

  def test_write_only_custom_output_is_stable(self):
    devices = tuple(f"NULL:{i}" for i in range(4))
    out, inp = (UOp.param(i, dtypes.float, (4096,), device=devices) for i in range(2))
    p0, p1, idx = UOp.param(0, dtypes.float, (4096,)), UOp.param(1, dtypes.float, (4096,)), UOp.const(0)
    sink = UOp.sink(p0.index(idx).store(UOp.const(0, dtypes.float)), p1.index(idx).load(), arg=KernelInfo("opaque_write"))
    produced = out.after(UOp(Ops.PROGRAM, src=(sink,)).call(out, inp))
    self.assertTrue(_is_stable_custom_output(produced))
    with Context(ALL2ALL=2): reduced = handle_allreduce(produced, produced.allreduce(Ops.ADD, devices))
    assert reduced is not None
    self.assertFalse(any(x.op is Ops.STORE and x.src[0].base.op is Ops.BUFFER for x in reduced.toposort()))

    rw_sink = UOp.sink(p0.index(idx).store(p0.index(idx).load()), p1.index(idx).load(), arg=KernelInfo("opaque_readwrite"))
    readwrite = out.after(UOp(Ops.PROGRAM, src=(rw_sink,)).call(out, inp))
    self.assertFalse(_is_stable_custom_output(readwrite))
    with Context(ALL2ALL=2): reduced = handle_allreduce(readwrite, readwrite.allreduce(Ops.ADD, devices))
    assert reduced is not None
    self.assertTrue(any(x.op is Ops.STORE and x.src[0].base.op is Ops.BUFFER for x in reduced.toposort()))

  def test_precompiled_input_staged_once(self):
    devices = tuple(f"NULL:{i}" for i in range(4))
    buf = UOp.new_buffer(devices, 4096, dtypes.float) + 1
    ret = create_allreduce_function(buf, buf.allreduce(Ops.ADD, devices))
    assert ret is not None
    contiguous = [x for x in ret.toposort() if x.op is Ops.CONTIGUOUS]
    self.assertEqual(len(contiguous), 1)
    call = next(x for x in ret.toposort() if x.op is Ops.CALL)
    self.assertIs(call.src[-1], contiguous[0])

  def test_schedule_ring(self):
    with Context(RING=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.empty(N, N*100).shard(ds, axis=0).realize()
      linear = t.sum(0).linear_with_vars()[0]
      copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
      pairs = [(c.src[1].buffer.device, c.src[2].buffer.device) for c in copies]
      # N*(N-1) scatter reduce, and N*(N-1) allgather
      if len(pairs) != N*(N-1)*2: raise KernelCountException(N*(N-1)*2, len(pairs))
      # copy topology forms a ring
      self.assertEqual(len(set(pairs)), N)

  def test_schedule_all2all(self):
    with Context(ALL2ALL=2):
      N = 4
      M = N*100
      ds = tuple(f"CPU:{i}" for i in range(N))
      x = Tensor.arange(N*M, dtype=dtypes.float).reshape(N, M)
      t = (x*x).clone().shard(ds, axis=0).realize()
      out = t.sum(0).mul(2.).contiguous()
      linear, var_vals = out.linear_with_vars()
      copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
      sinks = [si for si in linear.src if si.src[0].op is Ops.SINK]
      # N*(N-1) copies for input and output
      copy_count = N*(N-1)*2
      if len(copies) != copy_count: raise KernelCountException(copy_count, len(copies))
      # direct physical views avoid the former zero-offset staging kernels
      if len(sinks) != 11: raise KernelCountException(11, len(sinks))
      # correctness
      run_linear(linear, var_vals)
      expected = [2*sum((d*M+i)**2 for d in range(N)) for i in range(M)]
      dev_nums = Tensor.arange(1, N+1, dtype=dtypes.float).reshape(N, 1).expand(N, M).shard(ds, axis=0)
      shards = out.reshape(1, M).expand(N, M)+dev_nums
      self.assertListEqual(shards.tolist(), [[x+d+1 for x in expected] for d in range(N)])

  def test_correct_all2all_direct_slices(self):
    with Context(ALL2ALL=2):
      N, W = 4, 512
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = (Tensor.arange(N*W).reshape(N, W).shard(ds, axis=0) * 2 + 1).contiguous().realize()
      self.assertListEqual(t.sum(0).tolist(), [8*i + 4*W*3 + 4 for i in range(W)])

  def _persistent_accumulate_tensors(self, packed=False):
    ndev, width = 4, 512
    devices = tuple(f"CPU:{i}" for i in range(ndev))
    persistent = ((Tensor.arange(width+256)*0.125-17).cast(dtypes.bfloat16).clone(device="CPU").realize()
                  .shard(devices, axis=None).realize())
    target = persistent[128:128+width] if packed else persistent[:width]
    grads = (((Tensor.arange(ndev*width)%97)*0.0625-2).reshape(ndev, width).cast(dtypes.bfloat16)
             .clone(device="CPU").realize().shard(devices, axis=0).realize())
    return persistent, target, grads

  @Context(ALL2ALL=2, RING=0)
  def test_schedule_persistent_allreduce_accumulate(self):
    _, target, grads = self._persistent_accumulate_tensors()
    target.assign(target+grads.sum(0))
    prepared = prepare_rangeify(target.uop.sink())
    stores = [x for x in prepared.toposort() if x.op is Ops.STORE]
    self_stores = [x for x in stores if x.src[0] in x.src[1].toposort(enter_calls=False)]
    self.assertEqual(len(self_stores), 4)
    self.assertTrue(all(x.src[1].tag == ("allreduce_accumulate",) for x in self_stores))
    self.assertEqual(sum(x.op is Ops.COPY for x in prepared.toposort()), 24)

  @Context(ALL2ALL=2, RING=0)
  def test_correct_persistent_allreduce_accumulate_bf16(self):
    _, target, grads = self._persistent_accumulate_tensors()
    target.assign(target+grads.sum(0)).realize()

    _, reference, reference_grads = self._persistent_accumulate_tensors()
    reduced = reference_grads.sum(0).realize()  # preserve the pre-accumulation BF16 store boundary
    reference.assign(reference+reduced).realize()
    self.assertListEqual(target.bitcast(dtypes.uint16).tolist(), reference.bitcast(dtypes.uint16).tolist())

  @Context(ALL2ALL=2, RING=0)
  def test_correct_packed_persistent_allreduce_accumulate(self):
    persistent, target, grads = self._persistent_accumulate_tensors(packed=True)
    before = persistent.bitcast(dtypes.uint16).tolist()
    target_offset = target.uop.contiguous_view_offset()
    assert target_offset is not None
    target.assign(target+grads.sum(0))
    prepared = prepare_rangeify(target.uop.sink())
    owners = [x for x in prepared.toposort() if x.op is Ops.STORE and x.src[1].tag == ("allreduce_accumulate",)]
    owner_offsets = [x.src[0].contiguous_view_offset() for x in owners]
    self.assertFalse(any(x is None for x in owner_offsets))
    self.assertEqual(sorted(target_offset+x for x in owner_offsets if x is not None), [128, 256, 384, 512])
    target.realize()
    after = persistent.bitcast(dtypes.uint16).tolist()
    self.assertListEqual(after[:128], before[:128])
    self.assertListEqual(after[640:], before[640:])

  @Context(RING=0, ALL2ALL=0)
  def test_schedule_naive(self):
    N = 4
    ds = tuple(f"NULL:{i}" for i in range(N))
    t = Tensor.empty(N, 4096).shard(ds, axis=0).realize()
    linear = t.sum(0).linear_with_vars()[0]

    copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
    sinks = [si for si in linear.src if si.src[0].op is Ops.SINK]
    pairs = [(c.src[1].buffer.device, c.src[2].buffer.device) for c in copies]

    if len(pairs) != N*(N-1): raise KernelCountException(N*(N-1), len(pairs))
    if len(sinks) != 2: raise KernelCountException(2, len(sinks))
    self.assertTrue(all(dst != src for dst, src in pairs))

  def test_symbolic_shape(self):
    rows = UOp.variable("rows", 1, 4).bind(3)
    t = Tensor.ones(4, 4).shard(("CPU:0", "CPU:1"), axis=1).realize()
    out = t[:rows].sum(1).realize()
    self.assertEqual(out.shape, (rows,))
    self.assertTrue((out == 4).all().item())

  def test_correct_ring(self):
    with Context(RING=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.ones(N, N*100).contiguous().shard(ds, axis=0).realize()
      out = t.sum(0)
      self.assertListEqual(out.tolist(), [4]*N*100)

class TestAllreduceCast(unittest.TestCase):
  def _get_copy_dtypes(self, dtype, allreduce_cast):
    ds = tuple(f"CPU:{i}" for i in range(2))
    with Context(ALLREDUCE_CAST=allreduce_cast, RING=0, SCACHE=0):
      t = Tensor.empty(4, 4, dtype=dtype).shard(ds, axis=0)
      linear = t.sum(0).linear_with_vars()[0]
      return {si.src[1].buffer.dtype for si in linear.src if si.src[0].op is Ops.COPY}

  def test_allreduce_cast_bf16(self):
    # with ALLREDUCE_CAST, allreduce copies stay in bfloat16 instead of promoting to float32
    self.assertNotIn(dtypes.float, self._get_copy_dtypes(dtypes.bfloat16, allreduce_cast=1))
    self.assertIn(dtypes.float, self._get_copy_dtypes(dtypes.bfloat16, allreduce_cast=0))

  def test_allreduce_cast_half(self):
    self.assertNotIn(dtypes.float, self._get_copy_dtypes(dtypes.half, allreduce_cast=1))
    self.assertIn(dtypes.float, self._get_copy_dtypes(dtypes.half, allreduce_cast=0))

  def test_allreduce_cast_float32_noop(self):
    # float32 should not be affected by ALLREDUCE_CAST (no promotion happens)
    dtypes_on = self._get_copy_dtypes(dtypes.float, allreduce_cast=1)
    dtypes_off = self._get_copy_dtypes(dtypes.float, allreduce_cast=0)
    self.assertEqual(dtypes_on, dtypes_off)

if __name__ == '__main__':
  unittest.main()
