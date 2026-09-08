import ctypes, itertools, pathlib, shutil, struct, subprocess, tempfile, unittest
from types import SimpleNamespace
from unittest.mock import patch

from tinygrad import Context, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.device import TinyELF
from tinygrad.helpers import Target, mv_address, to_mv
from tinygrad.runtime.support.hcq import CLikeArgsState, HCQBuffer, HWQueue
from tinygrad.runtime.support.memory import MMIOInterface


def signature(order):
  return tuple((name, i * 2, dtypes.int if name.startswith("v") else dtypes.float,
                () if name.startswith("v") else (1,), AddrSpace.ALU if name.startswith("v") else AddrSpace.GLOBAL)
               for i,name in enumerate(order))


def argument_buffer(size=4096):
  backing = bytearray(size)
  address = mv_address(memoryview(backing))
  return backing, HCQBuffer(address, size, view=MMIOInterface(address, size))


class TestHCQArgumentTransport(unittest.TestCase):
  def test_mixed_layouts_and_prefix(self):
    for order, offsets in ((('b0', 'v0', 'b1'), (0, 8, 16)), (('v0', 'b0', 'v1', 'b1'), (0, 8, 16, 24)),
                           (('b0', 'b1', 'v0'), (0, 8, 16)), (('b0', 'b1'), (0, 8)), (('v0',), (0,)), ((), ())):
      for prefix in (None, [0x12345678] * 88):
        with self.subTest(order=order, prefix=prefix is not None):
          backing, buf = argument_buffer()
          values = {'b0': 0x1122334455667788, 'b1': 0x2233445566778899, 'v0': 7, 'v1': 11}
          bufs = tuple(HCQBuffer(values[n], 4) for n in order if n.startswith('b'))
          vals = tuple(values[n] for n in order if n.startswith('v'))
          state = CLikeArgsState(buf, SimpleNamespace(signature=signature(order)), bufs, vals, prefix=prefix)
          HWQueue().bind_args_state(state)
          base = len(prefix or []) * 4
          if prefix: self.assertEqual(struct.unpack_from('I', backing)[0], prefix[0])
          for name, off in zip(order, offsets):
            self.assertEqual(struct.unpack_from('i' if name.startswith('v') else 'Q', backing, base + off)[0], values[name])

  def test_symbolic_buffer_and_scalar_replay(self):
    backing, buf = argument_buffer()
    address = UOp.variable('address', 0, 0xffffffffffffffff, dtypes.uint64)
    value = UOp.variable('value', 0, 100)
    state = CLikeArgsState(buf, SimpleNamespace(signature=signature(('b0', 'v0', 'b1'))),
                          (HCQBuffer(address, 4), HCQBuffer(0x2222, 4)), (value,))
    queue = HWQueue()
    queue.bind_args_state(state)
    for addr, val in ((0x1111, 3), (0x3333, 9), (0x3333, 9)):
      queue._apply_var_vals({'address': addr, 'value': val})
      self.assertEqual(struct.unpack_from('Qi4xQ', backing), (addr, val, 0x2222))


class TestQCOMArgumentTransport(unittest.TestCase):
  def test_ir3_compiler_image_mapping_matches_runtime_bindings(self):
    from tinygrad.codegen import to_program
    from tinygrad.renderer.nir import IR3Renderer
    from tinygrad.runtime.autogen import mesa
    from tinygrad.runtime.ops_qcom import QCOMProgram, QCOMArgsState
    from tinygrad.uop.ops import KernelInfo
    if 'mesa' not in mesa.dll._loaded_: self.skipTest('tinymesa required for IR3 compilation')
    renderer = IR3Renderer(Target('QCOM', 'IR3', 'a630,IMAGE_PITCH_ALIGNMENT=64'))
    storage = []
    def allocate(size, *args):
      backing, buf = argument_buffer(size)
      storage.append(backing)
      return buf
    dev = SimpleNamespace(renderer=renderer, device='QCOM', prof_prg_counter=itertools.count(), _ensure_stack_size=lambda _: None,
                          allocator=SimpleNamespace(alloc=allocate, free=lambda *args: None))
    for reverse in (False, True):
      with self.subTest(reverse=reverse), Context(IMAGE=2):
        out = UOp.param(2 if reverse else 0, dtypes.float, 256)
        value = UOp.param(1, dtypes.int, (), vmin_vmax=(0, 100), name='value', addrspace=AddrSpace.ALU)
        inp = UOp.param(0 if reverse else 2, dtypes.float, 256)
        idx = UOp.range(256, 0)
        sink = out[idx].store(inp[idx].load() + value.cast(dtypes.float)).end(idx).sink(arg=KernelInfo('image_mixed'))
        elf = to_program(sink, renderer).to_elf()
        self.assertEqual([s[3] for s in elf.signature], [(1, 64, 4), (), (1, 64, 4)])
        prg = QCOMProgram(dev, elf)
        self.assertEqual((prg.ibo_cnt, prg.tex_cnt), (2, 1))
        self.assertEqual(prg.tex_to_image[0], 0 if reverse else 1)
        backing, buf = argument_buffer(prg.kernargs_alloc_size)
        state = QCOMArgsState(buf, prg, (HCQBuffer(0x10000, 1024), HCQBuffer(0x20000, 1024)), (7,))
        HWQueue().bind_args_state(state)
        self.assertEqual(struct.unpack_from('i', backing, prg.buf_off)[0], 7)
        self.assertEqual(struct.unpack_from('Q', backing, prg.tex_off + 16)[0], 0x10000 if reverse else 0x20000)
        for i,addr in enumerate((0x10000, 0x20000)):
          self.assertEqual(struct.unpack_from('Q', backing, prg.ibo_off + i*64 + 16)[0], addr)

  def program(self, sig, nir, **kwargs):
    return SimpleNamespace(signature=sig, NIR=nir, kernargs_alloc_size=4096, consts_info=[(0x1234, 1000, 4)],
                           samp_cnt=0, tex_cnt=0, ibo_cnt=0, tex_to_image=[], image_types=[], buf_off=64, buf_offs=[64, 88, 112, 136],
                           tex_off=2048, ibo_off=2560, samp_off=3072, samplers=[], **kwargs)

  def test_scalar_first_interleaved_and_unused_slots(self):
    from tinygrad.runtime.ops_qcom import QCOMArgsState
    for nir in (False, True):
      with self.subTest(nir=nir):
        backing, buf = argument_buffer()
        sig = signature(('v0', 'b0', 'v1', 'b1'))
        prg = self.program(sig, nir)
        state = QCOMArgsState(buf, prg, (HCQBuffer(0x1111, 4), HCQBuffer(0x2222, 4)), (7, 11))
        HWQueue().bind_args_state(state)
        offsets = (64, 72, 80, 88) if nir else (64, 88, 112, 136)
        for fmt, offset, value in zip(('i', 'Q', 'i', 'Q'), offsets, (7, 0x1111, 11, 0x2222)):
          self.assertEqual(struct.unpack_from(fmt, backing, offset)[0], value)
        self.assertEqual(struct.unpack_from('I', backing, 1000)[0], 0x1234)

  def test_images_keep_resource_indices_separate_from_argument_slots(self):
    from tinygrad.runtime.ops_qcom import QCOMArgsState
    sig = (("v", 0, dtypes.int, (), AddrSpace.ALU), (None, 2, dtypes.float, (1, 16, 4), AddrSpace.GLOBAL),
           (None, 4, dtypes.float, (1,), AddrSpace.GLOBAL), (None, 7, dtypes.float, (1, 16, 4), AddrSpace.GLOBAL))
    for nir in (False, True):
      with self.subTest(nir=nir):
        backing, buf = argument_buffer()
        prg = self.program(sig, nir)
        prg.buf_offs = [64, 88]
        prg.ibo_cnt, prg.tex_cnt, prg.tex_to_image, prg.image_types = 2 if nir else 1, 1, [1], [2, 1]
        prg.samp_cnt, prg.samplers = 1, [11, 22, 33, 44]
        state = QCOMArgsState(buf, prg, tuple(HCQBuffer(addr, 1024) for addr in (0x10000, 0x20000, 0x30000)), (7,))
        HWQueue().bind_args_state(state)
        self.assertEqual(struct.unpack_from('i', backing, 64)[0], 7)
        self.assertEqual(struct.unpack_from('Q', backing, 72 if nir else 88)[0], 0x20000)
        self.assertEqual(struct.unpack_from('Q', backing, prg.ibo_off + 16)[0], 0x10000)
        self.assertEqual(struct.unpack_from('Q', backing, prg.tex_off + 16)[0], 0x30000)
        self.assertEqual(struct.unpack_from('4I', backing, prg.samp_off), tuple(prg.samplers))

  def test_cl_texture_before_output_image_uses_compiler_resource_types(self):
    from tinygrad.runtime.ops_qcom import QCOMArgsState, BUFTYPE_TEX, BUFTYPE_IBO
    sig = ((None, 0, dtypes.float, (1, 16, 4), AddrSpace.GLOBAL), ("v", 2, dtypes.int, (), AddrSpace.ALU),
           (None, 4, dtypes.float, (1, 16, 4), AddrSpace.GLOBAL))
    prg = self.program(sig, False)
    prg.buf_offs, prg.image_types, prg.ibo_cnt, prg.tex_cnt = [64], [BUFTYPE_TEX, BUFTYPE_IBO], 1, 1
    backing, buf = argument_buffer()
    state = QCOMArgsState(buf, prg, (HCQBuffer(0x10000, 1024), HCQBuffer(0x20000, 1024)), (7,))
    HWQueue().bind_args_state(state)
    self.assertEqual(struct.unpack_from('Q', backing, prg.tex_off + 16)[0], 0x10000)
    self.assertEqual(struct.unpack_from('Q', backing, prg.ibo_off + 16)[0], 0x20000)
    self.assertEqual(struct.unpack_from('i', backing, 64)[0], 7)


class TestNVArgumentTransport(unittest.TestCase):
  def test_mock_and_native_layouts(self):
    from tinygrad.runtime.ops_nv import NVArgsState, MOCKIface
    for mock in (False, True):
      with self.subTest(mock=mock):
        backing, buf = argument_buffer()
        prg = SimpleNamespace(signature=signature(('v0', 'b0', 'v1', 'b1')), cbuf_0=[0]*88,
                              dev=SimpleNamespace(iface=object.__new__(MOCKIface) if mock else None))
        state = NVArgsState(buf, prg, (HCQBuffer(0x1111, 4), HCQBuffer(0x2222, 4)), (-7, 11))
        HWQueue().bind_args_state(state)
        self.assertEqual(struct.unpack_from('qQqQ' if mock else 'i4xQi4xQ', backing, 0x160), (-7, 0x1111, 11, 0x2222))
        if mock: self.assertEqual(struct.unpack_from('2I', backing, 80*4), (2, 2))


class TestCUDAGraphArgumentTransport(unittest.TestCase):
  def test_driver_boundary_preserves_layout_dependencies_and_replay(self):
    from tinygrad.codegen import to_program
    from tinygrad.device import Buffer
    from tinygrad.engine.jit import GraphRunner
    from tinygrad.runtime.graph.cuda import CUDAGraph
    from tinygrad.runtime.ops_python import PythonRenderer
    from tinygrad.runtime.autogen import cuda
    from tinygrad.uop.ops import KernelInfo, Ops
    out, inp = UOp.param(3, dtypes.float, 1), UOp.param(2, dtypes.float, 1)
    val = UOp.param(0, dtypes.int, (), vmin_vmax=(0, 100), name='value', addrspace=AddrSpace.ALU)
    sink = out[0].store(inp[0].load() + val.cast(dtypes.float)).sink(arg=KernelInfo('cuda_graph_mixed'))
    program = to_program(sink, PythonRenderer(Target('PYTHON')))
    output = Buffer('NULL', 1, dtypes.float, opaque=0x2000)
    first = Buffer('NULL', 1, dtypes.float, opaque=0x1000)
    unused = Buffer('NULL', 1, dtypes.float, opaque=0xDEAD)
    call = program.call(UOp.variable('value', 0, 100).bind(3), UOp.from_buffer(unused),
                        UOp.param(0, dtypes.float, 1, 'NULL'), UOp.from_buffer(output))
    runtime = SimpleNamespace(prg=cuda.CUfunction(), smem=0, signature=program.to_elf().signature)
    writes, launches, param_blocks = [], [], []
    def add_node(node, graph, deps, count, params):
      p = ctypes.cast(params, ctypes.POINTER(cuda.CUDA_KERNEL_NODE_PARAMS_v1)).contents
      param_blocks.append(p.extra[1])
      return 0
    def launch(*args):
      launches.append(struct.unpack('i4xQQ', bytes(to_mv(param_blocks[0], 24))))
      return 0
    original_access = GraphRunner._access_resources
    def access(graph, bufs, written, new_dependency):
      writes.append(tuple(written))
      return original_access(graph, bufs, written, new_dependency)
    mocked = dict(cuGraphCreate=lambda *a: 0, cuGraphAddKernelNode=add_node, cuGraphInstantiate_v2=lambda *a: 0,
                  cuGraphExecKernelNodeSetParams=lambda *a: 0, cuGraphLaunch=launch, cuGraphDestroy=lambda *a: 0, cuGraphExecDestroy=lambda *a: 0)
    with patch('tinygrad.engine.jit.get_runtime', return_value=runtime), patch.object(GraphRunner, '_access_resources', access), \
         patch.multiple(cuda, **mocked):
      graph = CUDAGraph(UOp.custom_function('graph', UOp(Ops.LINEAR, src=(call,))), input_uops=(UOp.from_buffer(first),))
      self.assertEqual(writes, [(1,)])
      for address, value in ((0x3000, 7), (0x4000, 11)):
        replacement = Buffer('NULL', 1, dtypes.float, opaque=address)
        graph((UOp.from_buffer(replacement),), {'value': value})
      self.assertEqual(launches, [(7, 0x3000, 0x2000), (11, 0x4000, 0x2000)])
      graph.__del__()
      del graph.graph, graph.instance


class TestDSPArgumentTransport(unittest.TestCase):
  @unittest.skipUnless(shutil.which('clang'), "host clang required to execute the generated RPC wrapper")
  def test_rpc_runtime_and_generated_wrapper_execute_together(self):
    from tinygrad.runtime.ops_dsp import DSPRenderer, DSPProgram, DSPBuffer
    renderer = object.__new__(DSPRenderer)
    sig = (("v0", 0, dtypes.int, (), AddrSpace.ALU), (None, 2, dtypes.float, (1,), AddrSpace.GLOBAL),
           ("v1", 4, dtypes.int64, (), AddrSpace.ALU), (None, 6, dtypes.float, (1,), AddrSpace.GLOBAL))
    params = [(name or f'buf{slot}', (UOp.param(slot, dt, shape, addrspace=space), False)) for name,slot,dt,shape,space in sig]
    source = '\n'.join(renderer._render_defines([])) + '''
static void *buffers[2];
void set_buffers(void *out, void *in) { buffers[0]=out; buffers[1]=in; }
int HAP_power_set(void *handle, void *req) { return 0; }
void *HAP_mmap(void *addr, int len, int prot, int flags, int fd, long off) { return buffers[fd == 20]; }
int HAP_munmap(void *addr, int len) { return 0; }
unsigned long long HAP_perf_get_time_us(void) { return 25; }
void mixed(int v0, float *out, long long v1, float *in) { *out = *in + v0 + (v1 >> 32); }
''' + renderer._render_entry('mixed', params)
    with tempfile.TemporaryDirectory() as tmp:
      path = pathlib.Path(tmp) / 'wrapper.so'
      result = subprocess.run(['clang', '-shared', '-fPIC', '-x', 'c', '-', '-o', str(path)], input=source.encode(), capture_output=True)
      self.assertEqual(result.returncode, 0, result.stderr.decode())
      lib = ctypes.CDLL(str(path))
      lib.set_buffers.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
      lib.entry.argtypes = (ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_void_p)
      def receive(binary, sc, pra, fds, attrs):
        for i in range(3, len(fds)): pra[i].dma.fd = fds[i]
        self.assertEqual(lib.entry(0, sc, pra), 0)
      prg = DSPProgram(SimpleNamespace(exec_lib=receive), TinyELF(b'', 'mixed', Target('DSP'), sig))
      for value, scalar, wide in ((5, 3, 4 << 32), (9, -2, 7 << 32)):
        output, inp = ctypes.c_float(), ctypes.c_float(value)
        lib.set_buffers(ctypes.byref(output), ctypes.byref(inp))
        bufs = tuple(DSPBuffer(ctypes.addressof(b), 4, SimpleNamespace(fd=fd)) for b,fd in ((output, 10), (inp, 20)))
        prg(*bufs, vals=(scalar, wide))
        self.assertEqual(output.value, value + scalar + (wide >> 32))

  def test_rpc_preserves_signature_order_and_compact_fd_offsets(self):
    from tinygrad.runtime.ops_dsp import DSPProgram, DSPBuffer
    for order in (('b0', 'v0', 'b1'), ('v0', 'b0', 'v1', 'b1'), ('b0', 'b1', 'v0')):
      with self.subTest(order=order):
        values = {'b0': 32, 'b1': 64, 'v0': 7, 'v1': 11}
        def receive(lib, sc, pra, fds, attrs):
          payload = bytes(to_mv(pra[0].buf.pv, pra[0].buf.len))
          self.assertEqual(pra[0].buf.len, len(order)*8)
          self.assertEqual([struct.unpack_from('i', payload, i*8)[0] for i in range(len(order))], [values[n] for n in order])
          self.assertEqual(list(to_mv(pra[1].buf.pv, pra[1].buf.len).cast('I')), [4, 8])
          self.assertEqual(list(fds), [-1, -1, -1, 10, 20])
          to_mv(pra[2].buf.pv, 8).cast('Q')[0] = 25
        bufs = (DSPBuffer(0x1000, 32, SimpleNamespace(fd=10), 4), DSPBuffer(0x2000, 64, SimpleNamespace(fd=20), 8))
        prg = DSPProgram(SimpleNamespace(exec_lib=receive), TinyELF(b'', 'mixed', Target('DSP'), signature(order)))
        self.assertEqual(prg(*bufs, vals=tuple(values[n] for n in order if n.startswith('v'))), 25 / 1e6)

  def test_rpc_wrapper_uses_compact_buffer_indices(self):
    from tinygrad.runtime.ops_dsp import DSPRenderer
    renderer = object.__new__(DSPRenderer)
    order = ('v0', 'b0', 'v1', 'b1')
    params = [(n, (UOp.param(i, dtypes.int, () if n.startswith('v') else 1,
                              addrspace=AddrSpace.ALU if n.startswith('v') else AddrSpace.GLOBAL), False)) for i,n in enumerate(order)]
    source = renderer._render_entry('mixed', params)
    self.assertIn('off1 = ((int*)pra[1].buf.pv)[0]', source)
    self.assertIn('off3 = ((int*)pra[1].buf.pv)[1]', source)
    self.assertIn('pra[3].dma.fd', source)
    self.assertIn('pra[4].dma.fd', source)
    self.assertNotIn('pra[6].dma.fd', source)
    self.assertIn('mixed(sz_or_val_0, buf_1, sz_or_val_2, buf_3)', source)

  def test_mock_serializes_in_receiver_order_and_copies_back_buffers(self):
    from tinygrad.runtime.ops_dsp import MockDSPProgram, DSPBuffer
    order = ('v0', 'b0', 'v1', 'b1')
    first, second = bytearray(b'abcd'), bytearray(b'efgh')
    bufs = tuple(DSPBuffer(mv_address(memoryview(x)), 4, None) for x in (first, second))
    def receive(command, **kwargs):
      self.assertEqual(kwargs['input'], struct.pack('i', 7) + b'abcd' + struct.pack('i', 11) + b'efgh')
      return SimpleNamespace(stdout=struct.pack('I', 25) + b'ABCD' + b'EFGH')
    prg = MockDSPProgram(None, TinyELF(b'', 'mixed', Target('DSP'), signature(order)))
    with patch('tinygrad.runtime.ops_dsp.subprocess.run', side_effect=receive): self.assertEqual(prg(*bufs, vals=(7, 11)), 25 / 1e9)
    self.assertEqual((first, second), (b'ABCD', b'EFGH'))


if __name__ == '__main__': unittest.main()
