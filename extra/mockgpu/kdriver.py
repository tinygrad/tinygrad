import pathlib, re, ctypes, mmap, collections
import tinygrad.runtime.autogen.kfd as kfd
from tinygrad.helpers import from_mv
from extra.mockgpu.gpu import AMDGPU

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

def ioctls_from_header():
  hdrpy = (pathlib.Path(__file__).parent.parent.parent / "tinygrad" / "runtime" / "autogen" / "kfd.py").read_text()
  pattern = r'# (AMDKFD_IOC_[A-Z0-9_]+)\s=\s_(IOW?R?).*\(( 0x[0-9a-fA-F]+) ,\s+struct\s([A-Za-z0-9_]+)\s+\)'
  matches = re.findall(pattern, hdrpy, re.MULTILINE)
  return type("KFD_IOCTLS", (object, ), {name: int(nr, 0x10) for name, _, nr, _ in matches}), \
         {int(nr, 0x10): getattr(kfd, "struct_"+sname) for name, idir, nr, sname in matches}
kfd_ioctls, kfd_headers = ioctls_from_header()

class KDriver:
  def open(self, name, flags, mode): raise NotImplementedError()
  def ioctl(self, fd, request, argp): raise NotImplementedError()

gpus_id = [49600]
class AMDDriver(KDriver):
  def __init__(self, gpus=6):
    self.gpus = {}
    self.files = [r'/dev/kfd', r'/dev/dri/renderD*']
    self.watched_addresses = []
    self.fds = []
    self.next_fd = 0x80
    self.next_handle = 1
    self.next_event = 1

    self.read_funcs = {}
    self.ioctl_funcs = {}
    self.mmap_funcs = {}
    
    self.object_by_handle = {}
    self.doorbells = {}
    self.next_doorbell = collections.defaultdict(int)
    self.event_come = []

    for i in gpus_id: self._prepare_gpu(i)

  def _alloc_fd(self):
    my_fd = self.next_fd
    self.next_fd = self.next_fd + 1
    return my_fd

  def _alloc_handle(self):
    handle = self.next_handle
    self.next_handle += 1
    return handle

  def _alloc_next_event_slot(self):
    ev = self.next_event
    self.next_event += 1
    return ev

  def _alloc_doorbell(self, gpu_id):
    x = ctypes.addressof(from_mv(self.doorbells[gpu_id])) + self.next_doorbell[gpu_id] * 8
    self.next_doorbell[gpu_id] += 1
    return x

  def _prepare_gpu(self, gpu_id):
    self.doorbells[gpu_id] = memoryview(bytearray(0x2000))
    self.gpus[gpu_id] = AMDGPU(gpu_id)

  def open(self, name, flags, mode):
    fd = self._alloc_fd()
    self.fds.append(fd)

    if name.decode() == "/dev/kfd":
      self.ioctl_funcs[fd] = self._kfd_ioctl
      self.mmap_funcs[fd] = self._kfd_mmap
    if name.decode().startswith("/dev/dri/renderD"):
      self.mmap_funcs[fd] = self._gpu_mmap

    return fd

  def read(self, fd, buf, sz):
    if fd in self.read_funcs: return self.read_funcs[fd](fd, buf, sz)
    return -1

  def ioctl(self, fd, request, argp):
    if fd in self.ioctl_funcs: return self.ioctl_funcs[fd](fd, request, argp)
    return -1

  def mmap(self, start, sz, prot, flags, fd, offset):
    if fd in self.mmap_funcs: return self.mmap_funcs[fd](start, sz, prot, flags, fd, offset)
    return -1

  def _kfd_read(self, fd, buf, sz): return -1
  def _kfd_ioctl(self, fd, req, argp):
    nr = req & 0xFF
    struct = kfd_headers[nr].from_address(argp)

    if nr == kfd_ioctls.AMDKFD_IOC_ACQUIRE_VM: pass
    elif nr == kfd_ioctls.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:
      if struct.gpu_id not in self.gpus: return -1
      struct.handle = self._alloc_handle()
      self.object_by_handle[struct.handle] = struct # save memory struct to know what mem it is
    elif nr == kfd_ioctls.AMDKFD_IOC_MAP_MEMORY_TO_GPU:
      dev_ids = (ctypes.c_int32 * struct.n_devices).from_address(struct.device_ids_array_ptr)
      for i in range(struct.n_devices):
        gpu = self.gpus[dev_ids[i]]
        mem_obj = self.object_by_handle[struct.handle]
        gpu.map_range(mem_obj.va_addr, mem_obj.size)
        struct.n_success = i + 1
    elif nr == kfd_ioctls.AMDKFD_IOC_CREATE_EVENT:
      struct.event_slot_index = self._alloc_next_event_slot()
      struct.event_id = struct.event_slot_index
    elif nr == kfd_ioctls.AMDKFD_IOC_CREATE_QUEUE:
      gpu = self.gpus[struct.gpu_id]
      struct.doorbell_offset = self._alloc_doorbell(struct.gpu_id)
      if struct.queue_type == kfd.KFD_IOC_QUEUE_TYPE_SDMA:
        queue_id = gpu.add_sdma_queue(struct.ring_base_address, struct.ring_size, struct.read_pointer_address, struct.write_pointer_address)
        self.watched_addresses.append(
          (struct.doorbell_offset, struct.doorbell_offset + 8, lambda mv,off: None, lambda mv,off: gpu.execute(0)))
      elif struct.queue_type == kfd.KFD_IOC_QUEUE_TYPE_COMPUTE:
        queue_id = gpu.add_pm4_queue(struct.ring_base_address, struct.ring_size, struct.read_pointer_address, struct.write_pointer_address)
        self.watched_addresses.append(
          (struct.doorbell_offset, struct.doorbell_offset + 8, lambda mv,off: None, lambda mv,off: gpu.execute(1)))
      else: raise RuntimeError("Unsuported, queue")
    elif nr == kfd_ioctls.AMDKFD_IOC_WAIT_EVENTS:
      pass
    else:
      name = "unknown"
      for k,v in kfd_ioctls.__dict__.items():
        if nr == v: name = k
      assert False, f"unknown kfd ioctl, {nr} {name}"
      exit(1)

    return 0

  def _kfd_mmap(self, start, sz, prot, flags, fd, offset):
    return offset
    # pass
    # if offset in self.doorbells.values():
    # return -1

  def _gpu_mmap(self, start, sz, prot, flags, fd, offset):
    # Fake mmap of gpu
    return libc.mmap(start, sz, prot, flags|mmap.MAP_ANONYMOUS, -1, 0)
