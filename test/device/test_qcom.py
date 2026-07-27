import ctypes, itertools, platform, struct, unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Device, Tensor
from tinygrad.device import Buffer, BufferSpec, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.helpers import mv_address, Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.support.hcq import HCQBuffer, HWQueue, MMIOInterface
from tinygrad.runtime.support.memory import BumpAllocator

class FakeAllocator:
  def __init__(self): self.allocations = []

  def alloc(self, size, options):
    gpu_memory, cpu_memory = bytearray([0xaa] * size), bytearray([0xaa] * size)
    buf = HCQBuffer(mv_address(memoryview(gpu_memory)), size, view=MMIOInterface(mv_address(memoryview(cpu_memory)), size))
    self.allocations.append((gpu_memory, cpu_memory, buf))
    return buf

  def free(self, *args): pass

class RecordingIface:
  def __init__(self): self.submissions, self.sleeps = [], []

  def submit(self, command, size, buffers, _var_vals=None):
    self.submissions.append((command, size, buffers))
    return 42

  def sleep(self, timeout): self.sleeps.append(timeout)

class RecordingMemoryIface:
  def __init__(self):
    self.allocated, self.mapped, self.freed, self.maps = HCQBuffer(0x1000, 16), HCQBuffer(0x2000, 16), [], []

  def alloc(self, size, uncached=False): return self.allocated
  def map(self, ptr, size, fd=None, offset=0):
    self.maps.append((ptr, size, fd, offset))
    return self.mapped
  def free(self, buf): self.freed.append(buf)

class TestQCOM(unittest.TestCase):
  def test_args_use_cpu_view(self):
    from tinygrad.runtime.ops_qcom import QCOMArgsState

    gpu_memory, cpu_memory = bytearray([0xaa] * 64), bytearray([0xaa] * 64)
    args = HCQBuffer(mv_address(memoryview(gpu_memory)), len(gpu_memory),
                     view=MMIOInterface(mv_address(memoryview(cpu_memory)), len(cpu_memory)))
    data = HCQBuffer(0x123456789abcdef0, 16)
    prg = SimpleNamespace(kernargs_alloc_size=64, signature=((None, 0, dtypes.float32, (1,)),),
                          ibo_cnt=0, tex_cnt=0, samp_cnt=0, NIR=True, tex_to_image=[], consts_info=[(0x12345678, 24, 4)],
                          buf_off=8, tex_off=64, ibo_off=64, samplers=[])

    state = QCOMArgsState(args, prg, (data,), vals=(0x87654321,))
    HWQueue().bind_args_state(state)

    self.assertEqual(cpu_memory[:8], bytes(8))
    self.assertEqual(int.from_bytes(cpu_memory[8:16], "little"), data.va_addr)
    self.assertEqual(int.from_bytes(cpu_memory[16:20], "little"), 0x87654321)
    self.assertEqual(int.from_bytes(cpu_memory[24:28], "little"), 0x12345678)
    self.assertEqual(cpu_memory[28:], bytes(36))
    self.assertEqual(gpu_memory, bytes([0xaa] * 64))

  def test_program_upload_uses_cpu_view(self):
    from tinygrad.runtime.ops_qcom import QCOMProgram

    lib, image, image_offset, image_desc_offset, reg_desc_offset = bytearray(0x500), b"\x12\x34\x56\x78", 0x400, 0x180, 0x300
    struct.pack_into("I", lib, 0x100, len(image))
    struct.pack_into("I", lib, 0xc0, image_offset)
    struct.pack_into("I", lib, 0x110, image_desc_offset)
    struct.pack_into("I", lib, 0x34, reg_desc_offset)
    struct.pack_into("I", lib, reg_desc_offset + 0x14, 1)
    lib[image_offset:image_offset+len(image)] = image

    allocator = FakeAllocator()
    dev = SimpleNamespace(device="QCOM", renderer=object(), allocator=allocator, prof_prg_counter=itertools.count(),
                          _ensure_stack_size=lambda size: None)
    QCOMProgram(dev, TinyELF(bytes(lib), "test", Target("QCOM"), ()))

    gpu_memory, cpu_memory, _ = allocator.allocations[0]
    self.assertEqual(cpu_memory, image)
    self.assertEqual(gpu_memory, bytes([0xaa] * len(image)))

  def test_workgroup_size_uses_cpu_view(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue

    gpu_memory, cpu_memory = bytearray([0xaa] * 32), bytearray([0xaa] * 32)
    args = HCQBuffer(mv_address(memoryview(gpu_memory)), len(gpu_memory),
                     view=MMIOInterface(mv_address(memoryview(cpu_memory)), len(cpu_memory)))
    prg = SimpleNamespace(NIR=True, wgsz=1, hregs=0, fregs=0, brnchstck=0, shared_size=1, prg_offset=0,
                          lib_gpu=HCQBuffer(0x200000, 128), pvtmem_size_per_item=0, pvtmem_size_total=0, hw_stack_offset=0,
                          image_size=128, samp_cnt=0, tex_cnt=0, ibo_cnt=0, wgid=0xfc, lid=0xfc)
    dummy = HCQBuffer(0x300000, 4096)
    dev = SimpleNamespace(gpu_id=(6, 0, 0), dummy_buf=dummy, dummy_addr=dummy.va_addr, _stack=HCQBuffer(0x400000, 4096),
                          border_color_buf=HCQBuffer(0x500000, 4096))
    prg.dev = dev

    QCOMComputeQueue(dev).exec(prg, SimpleNamespace(bind_data=[], buf=args, prg=prg, bufs=()), (1, 1, 1), (2, 3, 4))

    self.assertEqual(cpu_memory[4:16], struct.pack("III", 2, 3, 4))
    self.assertEqual(gpu_memory, bytes([0xaa] * len(gpu_memory)))

  def test_shader_instruction_size_uses_128_byte_units(self):
    from tinygrad.runtime.autogen import mesa
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue, pkt4_hdr

    image_size = 0x3280
    args = HCQBuffer(0x100000, 4096)
    prg = SimpleNamespace(NIR=True, wgsz=0xfc, hregs=0, fregs=0, brnchstck=0, shared_size=1, prg_offset=0,
                          lib_gpu=HCQBuffer(0x200000, image_size), pvtmem_size_per_item=0, pvtmem_size_total=0,
                          hw_stack_offset=0, image_size=image_size, samp_cnt=0, tex_cnt=0, ibo_cnt=0, wgid=0xfc, lid=0xfc)
    dummy = HCQBuffer(0x300000, 4096)
    dev = SimpleNamespace(gpu_id=(6, 0, 0), dummy_buf=dummy, dummy_addr=dummy.va_addr, _stack=HCQBuffer(0x400000, 4096),
                          border_color_buf=HCQBuffer(0x500000, 4096))
    prg.dev = dev

    queue = QCOMComputeQueue(dev).exec(prg, SimpleNamespace(bind_data=[], buf=args, prg=prg, bufs=()), (1, 1, 1), (1, 1, 1))
    register_packet = pkt4_hdr(mesa.REG_A6XX_SP_CS_INSTR_SIZE, 1)

    self.assertEqual(queue._q[queue._q.index(register_packet) + 1], image_size // 128)

  def test_queue_bind_uses_cpu_view(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue

    allocator = FakeAllocator()
    queue = QCOMComputeQueue(SimpleNamespace(allocator=allocator, ctx=1))
    queue.q(0x12345678, 0x9abcdef0)
    queue.bind(queue.dev)

    gpu_memory, cpu_memory, _ = allocator.allocations[0]
    self.assertEqual(cpu_memory, struct.pack("II", 0x12345678, 0x9abcdef0))
    self.assertEqual(gpu_memory, bytes([0xaa] * len(gpu_memory)))

  def test_queue_submits_through_interface(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue

    allocator, iface = FakeAllocator(), RecordingIface()
    cmd_buf = allocator.alloc(64, None)
    dev = SimpleNamespace(iface=iface, cmd_buf=cmd_buf,
                          cmd_buf_allocator=BumpAllocator(cmd_buf.size, base=int(cmd_buf.va_addr), wrap=True))
    queue = QCOMComputeQueue(dev)
    queue.q(0x12345678, 0x9abcdef0)

    queue.submit(dev)

    command, size, buffers = iface.submissions[0]
    self.assertEqual(command.cpu_view()[:size], struct.pack("II", 0x12345678, 0x9abcdef0))
    self.assertEqual(buffers, set())
    self.assertEqual(dev.last_cmd, 42)

  def test_queue_submits_referenced_buffers(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue

    allocator, iface = FakeAllocator(), RecordingIface()
    cmd_buf = allocator.alloc(4096, None)
    args = allocator.alloc(32, None)
    data, lib, stack, border, dummy, signal = [HCQBuffer(addr, 4096) for addr in range(0x100000, 0x700000, 0x100000)]
    dev = SimpleNamespace(iface=iface, allocator=allocator, cmd_buf=cmd_buf,
                          cmd_buf_allocator=BumpAllocator(cmd_buf.size, base=int(cmd_buf.va_addr), wrap=True),
                          gpu_id=(6, 0, 0), _stack=stack, border_color_buf=border, dummy_buf=dummy, dummy_addr=dummy.va_addr)
    prg = SimpleNamespace(dev=dev, NIR=True, wgsz=0xfc, hregs=0, fregs=0, brnchstck=0, shared_size=1, prg_offset=0,
                          lib_gpu=lib, pvtmem_size_per_item=0, pvtmem_size_total=0, hw_stack_offset=0, image_size=128,
                          samp_cnt=0, tex_cnt=0, ibo_cnt=0, wgid=0xfc, lid=0xfc)
    state = SimpleNamespace(bind_data=[], buf=args, prg=prg, bufs=(data,))
    queue = QCOMComputeQueue(dev).exec(prg, state, (1, 1, 1), (1, 1, 1))
    queue.signal(SimpleNamespace(value_addr=signal.va_addr, base_buf=signal), 1).submit(dev)

    self.assertEqual(iface.submissions[0][2], {args, data, lib, stack, dummy, signal})

  def test_allocator_uses_interface(self):
    from tinygrad.runtime.ops_qcom import QCOMAllocator

    iface = RecordingMemoryIface()
    allocator = object.__new__(QCOMAllocator)
    allocator.dev = SimpleNamespace(iface=iface)

    self.assertIs(allocator._alloc(16, BufferSpec()), iface.allocated)
    self.assertIs(allocator._alloc(16, BufferSpec(external_ptr=0x1234)), iface.mapped)
    self.assertIs(allocator._alloc(16, BufferSpec(external_ptr=0x5678, external_fd=7, external_offset=0x100)), iface.mapped)
    self.assertEqual(iface.maps, [(0x1234, 16, None, 0), (0x5678, 16, 7, 0x100)])

    allocator._do_free(iface.allocated, BufferSpec())
    self.assertEqual(iface.freed, [iface.allocated])

  def test_tensor_from_blob_passes_external_fd_and_offset(self):
    with patch.object(Buffer, "allocate", autospec=True) as allocate:
      Tensor.from_blob(0x1234, (4,), dtype=dtypes.int, device="CPU", fd=7, offset=0x234)

    self.assertEqual(allocate.call_args.kwargs, {"external_ptr": 0x1234, "external_fd": 7, "external_offset": 0x234})

  def test_signal_sleep_uses_interface(self):
    from tinygrad.runtime.ops_qcom import QCOMSignal

    memory, iface = bytearray(16), RecordingIface()
    owner = SimpleNamespace(iface=iface)
    signal = QCOMSignal(HCQBuffer(0x1000, 16, view=MMIOInterface(mv_address(memoryview(memory)), 16)),
                        owner=owner, is_timeline=True, virt=True)

    signal._sleep(7)

    self.assertEqual(iface.sleeps, [7])

  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)

if __name__ == '__main__':
  unittest.main()
