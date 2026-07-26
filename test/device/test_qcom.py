import ctypes, itertools, platform, struct, unittest
from types import SimpleNamespace
from tinygrad import Device
from tinygrad.device import TinyELF
from tinygrad.dtype import dtypes
from tinygrad.helpers import mv_address, Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.support.hcq import HCQBuffer, HWQueue, MMIOInterface

class TestAllocator:
  def __init__(self): self.allocations = []

  def alloc(self, size, options):
    gpu_memory, cpu_memory = bytearray([0xaa] * size), bytearray([0xaa] * size)
    buf = HCQBuffer(mv_address(memoryview(gpu_memory)), size, view=MMIOInterface(mv_address(memoryview(cpu_memory)), size))
    self.allocations.append((gpu_memory, cpu_memory, buf))
    return buf

  def free(self, *args): pass

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

    allocator = TestAllocator()
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
    dev = SimpleNamespace(gpu_id=(6, 0, 0), dummy_addr=0x300000, _stack=HCQBuffer(0x400000, 4096),
                          border_color_buf=HCQBuffer(0x500000, 4096))
    prg.dev = dev

    QCOMComputeQueue(dev).exec(prg, SimpleNamespace(bind_data=[], buf=args, prg=prg), (1, 1, 1), (2, 3, 4))

    self.assertEqual(cpu_memory[4:16], struct.pack("III", 2, 3, 4))
    self.assertEqual(gpu_memory, bytes([0xaa] * len(gpu_memory)))

  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)

if __name__ == '__main__':
  unittest.main()
