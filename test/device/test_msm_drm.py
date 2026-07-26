import ctypes, errno, mmap, unittest
from types import SimpleNamespace
from unittest.mock import patch

from tinygrad.helpers import Context, mv_address
from tinygrad.runtime.autogen import msm_drm
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer, MMIOInterface
from tinygrad.runtime.support.memory import BumpAllocator


def ioctl_number(ioctl):
  direction, base, number, struct_type = ioctl.args
  return direction << 30 | ctypes.sizeof(struct_type) << 16 | base << 8 | number


class RecordingMSMFile(FileIOInterface):
  def __init__(self):
    self.memory = bytearray([0xaa] * mmap.PAGESIZE)
    self.cpu_addr = mv_address(memoryview(self.memory))
    self.requests, self.mmaps, self.unmaps, self.closed_handles = [], [], [], []
    self.submissions, self.waits, self.closed_queues = [], [], []
    self.new_queues = []
    self.fail_mmap, self.wait_errno, self.close_errno = False, None, None
    self.is_msm, self.chip_id, self.munmap_result = True, 0x06030000, 0

  def __del__(self): pass

  def ioctl(self, request, arg):
    self.requests.append(request)
    if request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_NEW): arg.handle = 17
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GET_PARAM):
      if not self.is_msm: raise OSError(errno.ENOTTY, "not msm")
      arg.value = 630 if arg.param == msm_drm.MSM_PARAM_GPU_ID else self.chip_id
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_INFO):
      arg.value = 0x1234_0000 if arg.info == msm_drm.MSM_INFO_GET_IOVA else 0x8000
    elif request == ioctl_number(msm_drm.DRM_IOCTL_GEM_CLOSE):
      self.closed_handles.append(arg.handle)
      if self.close_errno is not None: raise OSError(self.close_errno, "close failed")
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT):
      bos_type = msm_drm.struct_drm_msm_gem_submit_bo * arg.nr_bos
      cmds_type = msm_drm.struct_drm_msm_gem_submit_cmd * arg.nr_cmds
      bos = [(bo.flags, bo.handle, bo.presumed) for bo in bos_type.from_address(arg.bos)]
      cmds = [(cmd.type, cmd.submit_idx, cmd.submit_offset, cmd.size) for cmd in cmds_type.from_address(arg.cmds)]
      self.submissions.append((arg.flags, arg.queueid, bos, cmds))
      arg.fence = 42
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_WAIT_FENCE):
      self.waits.append((arg.fence, arg.flags, arg.timeout.tv_sec, arg.timeout.tv_nsec, arg.queueid))
      if self.wait_errno is not None: raise OSError(self.wait_errno, "wait failed")
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_NEW):
      self.new_queues.append((arg.flags, arg.prio))
      arg.id = 3
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_CLOSE): self.closed_queues.append(arg.value)
    return 0

  def mmap(self, start, size, prot, flags, offset):
    self.mmaps.append((start, size, prot, flags, offset))
    if self.fail_mmap: raise OSError(errno.ENOMEM, "mmap failed")
    return self.cpu_addr

  def munmap(self, addr, size):
    self.unmaps.append((addr, size))
    return self.munmap_result


def make_iface(fd):
  from tinygrad.runtime.ops_qcom import MSMIface

  iface = object.__new__(MSMIface)
  iface.dev, iface.fd, iface.queue_id = SimpleNamespace(last_cmd=0), fd, 3
  return iface


def make_buffer(handle, gpu_addr, size, cpu=False):
  from tinygrad.runtime.ops_qcom import MSMAllocation

  memory = bytearray(size) if cpu else None
  view = MMIOInterface(mv_address(memoryview(memory)), size) if memory is not None else None
  buf = HCQBuffer(gpu_addr, size, meta=MSMAllocation(handle, size), view=view)
  buf.test_memory = memory
  return buf


class TestMSMDRMUAPI(unittest.TestCase):
  def test_struct_layouts(self):
    layouts = {
      msm_drm.struct_drm_msm_timespec: (16, (0, 8)),
      msm_drm.struct_drm_msm_param: (24, (0, 4, 8, 16, 20)),
      msm_drm.struct_drm_msm_gem_new: (16, (0, 8, 12)),
      msm_drm.struct_drm_msm_gem_info: (24, (0, 4, 8, 16, 20)),
      msm_drm.struct_drm_msm_gem_submit_cmd: (32, (0, 4, 8, 12, 16, 20, 24, 24)),
      msm_drm.struct_drm_msm_gem_submit_bo: (16, (0, 4, 8)),
      msm_drm.struct_drm_msm_gem_submit: (72, (0, 4, 8, 12, 16, 24, 32, 36, 40, 48, 56, 60, 64, 68)),
      msm_drm.struct_drm_msm_wait_fence: (32, (0, 4, 8, 24)),
      msm_drm.struct_drm_msm_submitqueue: (12, (0, 4, 8)),
    }

    for struct_type, (size, offsets) in layouts.items():
      with self.subTest(struct=struct_type.__name__):
        self.assertEqual(struct_type.SIZE, size)
        self.assertEqual(ctypes.sizeof(struct_type), size)
        self.assertEqual(tuple(field[2] for field in struct_type._real_fields_), offsets)

  def test_ioctl_numbers_include_linux_struct_sizes(self):
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GET_PARAM), 0xC0186440)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT), 0xC0486446)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_WAIT_FENCE), 0x40206447)


class TestMSMIface(unittest.TestCase):
  def test_init_requires_explicit_interface(self):
    from tinygrad.runtime.ops_qcom import MSMIface

    with Context(DEV="QCOM:IR3"):
      with self.assertRaisesRegex(RuntimeError, "MSM\\+QCOM:IR3"): MSMIface(SimpleNamespace(), 0)

  def test_init_probes_render_nodes_and_creates_a6xx_queue(self):
    from tinygrad.runtime.ops_qcom import MSMIface

    foreign, fd, dev = RecordingMSMFile(), RecordingMSMFile(), SimpleNamespace()
    foreign.is_msm = False
    with Context(DEV="MSM+QCOM:IR3"), \
         patch("tinygrad.runtime.ops_qcom.glob.glob", return_value=["/dev/dri/renderD128", "/dev/dri/renderD129"]), \
         patch("tinygrad.runtime.ops_qcom.FileIOInterface", side_effect=[foreign, fd]):
      iface = MSMIface(dev, 0)

    self.assertIs(iface.fd, fd)
    self.assertIs(iface.dev, dev)
    self.assertEqual((iface.chip_id, iface.gpu_id, iface.queue_id), (0x06030000, (6, 3, 0), 3))
    self.assertEqual(fd.new_queues, [(0, 0)])

  def test_init_rejects_pre_a6xx_before_creating_queue(self):
    from tinygrad.runtime.ops_qcom import MSMIface

    fd = RecordingMSMFile()
    fd.chip_id = 0x05040000
    with Context(DEV="MSM+QCOM:IR3"), \
         patch("tinygrad.runtime.ops_qcom.glob.glob", return_value=["/dev/dri/renderD128"]), \
         patch("tinygrad.runtime.ops_qcom.FileIOInterface", return_value=fd), \
         self.assertRaisesRegex(RuntimeError, "A6xx"):
      MSMIface(SimpleNamespace(), 0)

    self.assertEqual(fd.new_queues, [])

  def test_qcom_exposes_msm_as_an_interface(self):
    from tinygrad.renderer.nir import IR3Renderer
    from tinygrad.runtime.ops_qcom import MSMIface, QCOMDevice

    self.assertIn(MSMIface, QCOMDevice.ifaces)
    self.assertEqual(MSMIface.renderers, [IR3Renderer])

  def test_alloc_maps_cpu_and_gpu_addresses(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)

    buf = iface.alloc(17, fill_zeroes=True)

    self.assertEqual(buf.va_addr, 0x1234_0000)
    self.assertEqual(buf.size, 17)
    self.assertEqual(buf.cpu_view().addr, fd.cpu_addr)
    self.assertNotEqual(buf.va_addr, buf.cpu_view().addr)
    self.assertEqual((buf.meta.handle, buf.meta.mapped_size), (17, mmap.PAGESIZE))
    self.assertEqual(fd.mmaps, [(0, mmap.PAGESIZE, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0x8000)])
    self.assertEqual(fd.memory[:17], bytes(17))

  def test_alloc_closes_handle_if_mmap_fails(self):
    fd = RecordingMSMFile()
    fd.fail_mmap = True
    iface = make_iface(fd)

    with self.assertRaisesRegex(OSError, "mmap failed"): iface.alloc(17)

    self.assertEqual(fd.closed_handles, [17])

  def test_alloc_reports_cleanup_failure(self):
    fd = RecordingMSMFile()
    fd.fail_mmap, fd.close_errno = True, errno.EIO

    with self.assertRaisesRegex(RuntimeError, "allocation cleanup failed"): make_iface(fd).alloc(17)

  def test_external_pointer_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "external pointers"): make_iface(RecordingMSMFile()).map(0x1000, 16)

  def test_free_unmaps_then_closes_handle(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)
    buf = iface.alloc(17)

    iface.free(buf)

    self.assertEqual(fd.unmaps, [(fd.cpu_addr, mmap.PAGESIZE)])
    self.assertEqual(fd.closed_handles, [17])

  def test_free_closes_handle_if_unmap_fails(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)
    buf = iface.alloc(17)
    fd.munmap_result = -1

    with self.assertRaisesRegex(RuntimeError, "Failed to unmap"): iface.free(buf)

    self.assertEqual(fd.closed_handles, [17])

  def test_submit_includes_command_and_referenced_buffers(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)
    command_base = make_buffer(11, 0x1000_0000, 0x1000)
    command = command_base.offset(0x40, 0x80)
    data = make_buffer(12, 0x2000_0000, 0x1000)

    fence = iface.submit(command, 0x20, {data, command_base})

    self.assertEqual(fence, 42)
    flags, queue_id, bos, cmds = fd.submissions[0]
    self.assertEqual((flags, queue_id), (msm_drm.MSM_PIPE_3D0, 3))
    self.assertEqual({(handle, presumed) for _,handle,presumed in bos}, {(11, 0x1000_0000), (12, 0x2000_0000)})
    self.assertTrue(all(flags == msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE for flags,_,_ in bos))
    command_idx = next(i for i,(_,handle,_) in enumerate(bos) if handle == 11)
    self.assertEqual(cmds, [(msm_drm.MSM_SUBMIT_CMD_BUF, command_idx, 0x40, 0x20)])

  def test_qcom_queue_submits_through_msm_interface(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue

    fd = RecordingMSMFile()
    iface = make_iface(fd)
    cmd_buf = make_buffer(11, 0x1000_0000, 0x1000, cpu=True)
    dev = SimpleNamespace(iface=iface, cmd_buf=cmd_buf,
                          cmd_buf_allocator=BumpAllocator(cmd_buf.size, base=int(cmd_buf.va_addr), wrap=True))
    iface.dev = dev

    queue = QCOMComputeQueue(dev)
    queue.q(0x12345678, 0x9abcdef0)
    queue.submit(dev)

    self.assertEqual(dev.last_cmd, 42)
    self.assertEqual(fd.submissions[0][3], [(msm_drm.MSM_SUBMIT_CMD_BUF, 0, 0, 8)])

  def test_submit_rejects_invalid_command_size(self):
    with self.assertRaisesRegex(ValueError, "multiple of 4"):
      make_iface(RecordingMSMFile()).submit(make_buffer(11, 0x1000_0000, 0x1000), 3, set())

  def test_submit_rejects_foreign_buffers(self):
    command = HCQBuffer(0x1000_0000, 0x1000)
    with self.assertRaisesRegex(RuntimeError, "not allocated by the MSM DRM interface"):
      make_iface(RecordingMSMFile()).submit(command, 4, set())

  def test_wait_fence_uses_absolute_monotonic_timeout(self):
    from tinygrad.runtime.ops_qcom import MSM_WAIT_SLICE_NS

    fd = RecordingMSMFile()
    fd.wait_errno = errno.ETIMEDOUT
    iface = make_iface(fd)
    iface.dev.last_cmd = 41

    with patch("tinygrad.runtime.ops_qcom.time.monotonic_ns", return_value=5_000_000_123): iface.sleep(0)

    deadline = 5_000_000_123 + MSM_WAIT_SLICE_NS
    self.assertEqual(fd.waits, [(41, 0, deadline // 1_000_000_000, deadline % 1_000_000_000, 3)])

  def test_wait_fence_reports_driver_errors(self):
    fd = RecordingMSMFile()
    fd.wait_errno = errno.EIO
    iface = make_iface(fd)
    iface.dev.last_cmd = 41

    with self.assertRaisesRegex(RuntimeError, "MSM fence wait failed"): iface.sleep(0)

  def test_device_fini_closes_submitqueue_once(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)

    iface.device_fini()
    iface.device_fini()

    self.assertEqual(fd.closed_queues, [3])


if __name__ == "__main__":
  unittest.main()
