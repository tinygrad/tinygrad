from __future__ import annotations
from typing import Tuple, List
import ctypes, fcntl, os, mmap, tempfile, pathlib, functools, array
from tinygrad.device import BufferOptions, LRUAllocator, Compiled, Allocator
from tinygrad.helpers import from_mv, getenv, DEBUG, round_up, mv_address, to_mv
from tinygrad.runtime.ops_clang import ClangCompiler
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.dtype import DType
from tinygrad.ops import UOp

from tinygrad.runtime.autogen import libc, ion, msm_ion, adsprpc, qcom_dsp
# ION_IOC_ALLOC = 0
# ION_IOC_FREE = 1
# ION_IOC_SHARE = 4
# ION_IOC_SYNC = 7
# ION_IOC_CLEAN_INV_CACHES = 2

if getenv("IOCTL"): import extra.dsp.run # noqa: F401 # pylint: disable=unused-import

# adsp = ctypes.CDLL(ctypes.util.find_library("adsprpc"))

def rpc_invoke(rpcfd, handle, method, ins=None, outs=None):
  if ins or outs:
    ins = ins or list()
    outs = outs or list()
    pra = (qcom_dsp.union_remote_arg * (len(ins) + len(outs)))()
    for i,mv in enumerate(ins + outs):
      if isinstance(mv, memoryview):
        pra[i].buf.pv = mv_address(mv) if mv.nbytes > 0 else 0
        pra[i].buf.len = mv.nbytes
      else: assert False, "not supported"
    # pra = (qcom_dsp.union_remote_arg * (len(ins) + len(outs))).from_address(ctypes.addressof(pra))
  else:
    pra = None
    ins = ins or list()
    outs = outs or list()

  sc = (method << 24) | (len(ins) << 16) | (len(outs) << 8)
  return qcom_dsp.FASTRPC_IOCTL_INVOKE(rpcfd, handle=handle, sc=sc, pra=pra)

class DSPProgram:
  def __init__(self, device, name:str, lib:bytes):
    self.device = device

    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      output_file.write(lib) # TODO: not need to write this
      output_file.flush()
      if DEBUG >= 6: os.system(f"llvm-objdump -mhvx -d {output_file.name}")
      # self.handle = ctypes.c_int64(-1)
      fp = f"file:///{output_file.name}?entry&_modver=1.0&_dom=cdsp\0"

      a1 = memoryview(array.array('I', [len(fp), 0xff]))
      a2 = memoryview(bytearray(f"{fp}".encode()))
      o1 = memoryview(bytearray(0x8))
      o2 = memoryview(bytearray(0xff))
      z = rpc_invoke(self.device.rpc_fd, handle=0, method=0, ins=[a1, a2], outs=[o1, o2])

      self.handle = o1.cast('I')[0]
      print("Handle", hex(self.handle))

      # adsp.remote_handle64_open(ctypes.create_string_buffer(fp.encode()), ctypes.byref(self.handle))
      # print("OPEN", self.handle.value)
      # assert self.handle.value != -1, "load failed"

  def __del__(self): pass # TODO: fix this
    # if self.handle.value != -1:
    #   x = adsp.remote_handle64_close(self.handle)
    #   assert x == 0, "CLOSE failed"

  def __call__(self, *bufs, vals:Tuple[int, ...]=(), wait=False):
    assert len(vals) == 0
    assert len(bufs) < 16
    pra = (qcom_dsp.union_remote_arg * (2+len(bufs)))()
    test = (ctypes.c_int32 * len(bufs) * 2)()
    time_est = ctypes.c_uint64(0)
    pra[0].buf.pv = ctypes.addressof(test)
    pra[0].buf.len = 4 * len(bufs) * 2
    pra[1].buf.pv = ctypes.addressof(time_est)
    pra[1].buf.len = 8
    
    fds = (ctypes.c_int32 * (2+len(bufs)))()
    attrs = (ctypes.c_uint32 * (2+len(bufs)))()
    fds[0] = -1
    fds[1] = -1
    attrs[0] = 0
    attrs[1] = 0

    for i,b in enumerate(bufs):
      test[i * 2] = b.size
      test[i * 2 + 1] = b.share_info.fd
      fds[i + 2] = b.share_info.fd
      attrs[i + 2] = 1

      pra[i+2].dma.fd = 0x0
      pra[i+2].dma.len = 0x0

    # fds = (ctypes.c_int32 * (2+len(bufs)))(-1, -1, *[pra[i+2].dma.fd for i in range(len(bufs))])
    # attrs = (ctypes.c_uint32 * (2+len(bufs)))(0, 0, *[1 for i in range(len(bufs))])

    # print([-1, -1, *[pra[i+2].dma.fd for i in range(len(bufs))]])

    qcom_dsp.FASTRPC_IOCTL_INVOKE_ATTRS(self.device.rpc_fd, inv=qcom_dsp.struct_fastrpc_ioctl_invoke(handle=self.handle, sc=(1<<24) | (1<<16) | (1<<8) | len(bufs), pra=pra), fds=fds, attrs=attrs)

    # ret = adsp.remote_handle64_invoke(self.handle, (1<<24) | (1<<16) | (1<<8) | len(bufs), pra)
    # print("invoke", ret)
    # assert ret == 0, f"!!! invoke returned {ret}"
    # if ret != 0:
    #   print("errr,wow", ret)
    #   time.sleep(10)
    #return time_est.value / 19_200_000
    return time_est.value / 1e6

# def ion_iowr(fd, nr, args, magc=ion.ION_IOC_MAGIC):
#   ret = fcntl.ioctl(fd, (3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord(magc) & 0xFF) << 8 | (nr & 0xFF), args)
#   if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

class DSPBuffer:
  def __init__(self, va_addr:int, size:int, share_info:Any): self.va_addr, self.size, self.share_info = va_addr, size, share_info

class DSPAllocator(Allocator):
  def __init__(self, device):
    self.device = device
    # self.ion_fd = os.open("/dev/ion", os.O_RDWR | os.O_CLOEXEC)
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    # arg3 = ion.struct_ion_allocation_data(len=size, align=0x200, heap_id_mask=1<<msm_ion.ION_SYSTEM_HEAP_ID, flags=ion.ION_FLAG_CACHED)
    # ion_iowr(self.ion_fd, ION_IOC_ALLOC, arg3)
    # ion_iowr(self.ion_fd, ION_IOC_SHARE, arg2:=ion.struct_ion_fd_data(handle=arg3.handle))
    # res = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, arg2.fd, 0)
    # ion_iowr(self.ion_fd, ION_IOC_CLEAN_INV_CACHES, msm_ion.struct_ion_flush_data(handle=arg3.handle, fd=arg2.fd, vaddr=res, len=size), magc='M')

    alloc = qcom_dsp.ION_IOC_ALLOC(self.device.ion_fd, len=size, align=0x200, heap_id_mask=1<<msm_ion.ION_SYSTEM_HEAP_ID, flags=ion.ION_FLAG_CACHED)
    share_info = qcom_dsp.ION_IOC_SHARE(self.device.ion_fd, handle=alloc.handle)
    va_addr = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, share_info.fd, 0)
    return DSPBuffer(va_addr, size, share_info)

  def _free(self, opaque, options:BufferOptions):
    libc.munmap(opaque.va_addr, opaque.size)
    os.close(opaque.share_info.fd)
    qcom_dsp.ION_IOC_FREE(self.device.ion_fd, handle=opaque.share_info.handle)

  def copyin(self, dest, src:memoryview): ctypes.memmove(dest.va_addr, from_mv(src), src.nbytes)
  def copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src.va_addr, dest.nbytes)

class DSPRenderer(CStyleLanguage):
  device = "DSP"
  supports_float4 = False
  has_local = False
  buffer_suffix = " restrict __attribute__((align_value(128)))"
  kernel_prefix = "__attribute__((noinline)) "

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    ret = super().render_kernel(function_name, kernel, bufs, uops, prefix)
    prefix = ['#define max(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })', 'typedef int bool;']
    msrc = ['typedef struct { int fd; unsigned int offset; } remote_dma_handle;',
            'typedef struct { void *pv; unsigned int len; } remote_buf;',
            '#include "HAP_power.h"',
            'unsigned long long HAP_perf_get_time_us(void);',
            'typedef union { remote_buf buf; remote_dma_handle dma; } remote_arg;',
            'void* HAP_mmap(void *addr, int len, int prot, int flags, int fd, long offset);',
            'int HAP_munmap(void *addr, int len);',
            'int entry(unsigned long long handle, unsigned int sc, remote_arg* pra) {']
    msrc += ["""
    // Set client class
    {
      HAP_power_request_t request = {0};
      request.type = HAP_power_set_apptype;
      request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
      int retval = HAP_power_set((void*)handle, &request);
      if (retval) return 42;
    }
    // Set to turbo and disable DCVS
    {
      HAP_power_request_t request = {0};
      request.type = HAP_power_set_DCVS_v2;
      request.dcvs_v2.dcvs_enable = FALSE;
      request.dcvs_v2.set_dcvs_params = TRUE;
      request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_DISABLE;
      request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_DISABLE;
      request.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
      request.dcvs_v2.set_latency = TRUE;
      request.dcvs_v2.latency = 100;
      int retval = HAP_power_set((void*)handle, &request);
      if (retval) return 42;
    }
    // Vote for HVX power
    {
      HAP_power_request_t request = {0};
      request.type = HAP_power_set_HVX;
      request.hvx.power_up = TRUE;
      int retval = HAP_power_set((void*)handle, &request);
      if (retval) return 42;
    }
    """]
    msrc.append('if (sc>>24 == 1) {')
    msrc += [f'  void *buf_{i} = HAP_mmap(0, ((int*)pra[0].buf.pv)[{i}], 3, 0, pra[{i+2}].dma.fd, 0);' for i in range(len(bufs))]
    #msrc.append("  unsigned long long start = HAP_perf_get_time_us();")
    msrc.append("  unsigned long long start, end;")
    msrc.append("  start = HAP_perf_get_time_us();")
    #msrc.append('  asm volatile ("%0 = C15:14" : "=r"(start));')
    msrc.append(f"  {function_name}({', '.join([f'buf_{i}' for i in range(len(bufs))])});")
    msrc.append("  end = HAP_perf_get_time_us();")
    #msrc.append('  asm volatile ("%0 = C15:14" : "=r"(end));')
    msrc.append("  *(unsigned long long *)(pra[1].buf.pv) = end - start;")
    msrc += [f'  HAP_munmap(buf_{i}, ((int*)pra[0].buf.pv)[{i}]);' for i in range(len(bufs))]
    # msrc.append("return ((int*)buf_0)[0];")
    msrc.append("}")
    msrc.append("return 0;")
    msrc.append("}")
    return '\n'.join(prefix) + '\n' + ret + '\n' + '\n'.join(msrc)

class DSPDevice(Compiled):
  def __init__(self, device:str=""):
    self.ion_fd = os.open('/dev/ion', os.O_RDONLY)
    self.rpc_fd = os.open('/dev/adsprpc-smd', os.O_RDONLY | os.O_NONBLOCK)

    qcom_dsp.FASTRPC_IOCTL_GETINFO(self.rpc_fd, 3)

    fastrpc_shell = memoryview(bytearray(pathlib.Path('/dsp/cdsp/fastrpc_shell_3').read_bytes()))
    shell_mem = qcom_dsp.ION_IOC_ALLOC(self.ion_fd, len=round_up(fastrpc_shell.nbytes, 0x1000), align=0x1000, heap_id_mask=0x2000000, flags=0x1)
    shell_mapped = qcom_dsp.ION_IOC_MAP(self.ion_fd, handle=shell_mem.handle)
    fastrpc_shell_addr = libc.mmap(0, shell_mem.len, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, shell_mapped.fd, 0)
    ctypes.memmove(fastrpc_shell_addr, mv_address(fastrpc_shell), fastrpc_shell.nbytes)

    qcom_dsp.FASTRPC_IOCTL_CONTROL(self.rpc_fd, req=0x3)
    qcom_dsp.FASTRPC_IOCTL_INIT(self.rpc_fd, flags=0x1, file=fastrpc_shell_addr, filelen=fastrpc_shell.nbytes, filefd=shell_mapped.fd)

    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0x3, sc=(0x3 << 24)) # TODO: remove hardcoded 0x3

    # Start a serve thread for now
    from threading import Thread
    self.worker = Thread(target=self.listner_worker)
    self.worker.start()

    compiler_args = ["--target=hexagon", "-mcpu=hexagonv65", "-fuse-ld=lld", "-nostdlib", "-mhvx=v65", "-mhvx-length=128b",
                     "-I/data/home/nimlgen/tinygrad/extra/dsp/include"]
    super().__init__(device, DSPAllocator(self), DSPRenderer(), ClangCompiler("compile_dsp", args=compiler_args), functools.partial(DSPProgram, self))

  def listner_worker(self):
    # TODO: cleanup
    context = 0
    handle = 0xffffffff
    msg_send = memoryview(bytearray(0x10)).cast('I')
    msg_recv = memoryview(bytearray(0x10)).cast('I')
    out_buf = memoryview(bytearray(0x1000)).cast('I')
    in_buf = memoryview(bytearray(0x1000)).cast('I')

    prev_res = 0xffffffff
    out_buf_size = 0

    req_args = (qcom_dsp.union_remote_arg * 4)()
    req_args[0].buf = qcom_dsp.struct_remote_buf(pv=mv_address(msg_send), len=0x10)
    req_args[1].buf = qcom_dsp.struct_remote_buf(pv=mv_address(out_buf), len=0x1000)
    req_args[2].buf = qcom_dsp.struct_remote_buf(pv=mv_address(msg_recv), len=0x10)
    req_args[3].buf = qcom_dsp.struct_remote_buf(pv=mv_address(in_buf), len=0x1000)

    while True:
      msg_send[0] = context
      msg_send[1] = prev_res
      msg_send[2] = out_buf_size
      msg_send[3] = 0x1000

      req_args[1].buf.len = out_buf_size
      qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0x3, sc=0x04020200, pra=req_args) # listener

      # print("dai")

      context = msg_recv[0]
      handle = msg_recv[1]
      sc = msg_recv[2]
      inbufs = (sc >> 16) & 0xff
      outbufs = (sc >> 8) & 0xff

      in_args, out_args = [], []
      ptr = mv_address(in_buf)
      for i in range(inbufs):
        sz = to_mv(ptr, 4).cast('I')[0]
        obj_ptr = round_up(ptr + 4, 8)
        in_args.append(to_mv(obj_ptr, sz))
        ptr = obj_ptr + sz

      ctypes.memset(mv_address(out_buf), 0, 0x1000)
      ptr_out = mv_address(out_buf)
      for i in range(outbufs):
        sz = to_mv(ptr, 4).cast('I')[0]
        ptr += 4

        to_mv(ptr_out, 4).cast('I')[0] = sz
        obj_ptr = round_up(ptr_out + 4, 8)

        out_args.append(to_mv(obj_ptr, sz))
        ptr_out = obj_ptr + sz

      out_buf_size = ptr_out - mv_address(out_buf)

      if sc == 0x20200: # greating?
        prev_res = 0
      elif sc == 0x13050100: # open
        # for a in in_args: hexdump(a)
        try:
          fd = os.open(in_args[3].tobytes()[:-1].decode(), os.O_RDONLY)
          out_args[0].cast('I')[0] = fd
          prev_res = 0
        except: prev_res = 2
      elif sc == 0x9010000: # seek
        res = os.lseek(in_args[0].cast('I')[0], in_args[0].cast('I')[1], in_args[0].cast('I')[2])
        prev_res = 0 if res >= 0 else res
      elif sc == 0x4010200: # read
        buf = os.read(in_args[0].cast('I')[0], in_args[0].cast('I')[1])
        out_args[1][:len(buf)] = buf
        out_args[0].cast('I')[0] = len(buf)
        out_args[0].cast('I')[1] = int(len(buf) == 0)
        prev_res = 0
      elif sc == 0x3010000: # close
        os.close(in_args[0].cast('I')[0])
        prev_res = 0
      elif sc == 0x1f020100: # stat
        # try:
        stat = os.stat(in_args[1].tobytes()[:-1].decode())
        out_stat = out_args[0].cast('Q')
        out_stat[1] = stat.st_dev
        out_stat[2] = stat.st_ino
        out_stat[3] = stat.st_mode | (stat.st_nlink << 32)
        out_stat[4] = stat.st_rdev
        out_stat[5] = stat.st_size
        # print(stat, stat.st_rdev)
        # assert False
        prev_res = 0
        # except: prev_res = 2
      elif sc == 0x2010100:
        heapid = in_args[0].cast('I')[0]
        lflags = in_args[0].cast('I')[1]
        rflags = in_args[0].cast('I')[2]
        assert rflags == 0x1000

        # print(in_args[0])

        # print("WOOW", in_args[0].cast('Q')[2])
        # print("WOOW2", in_args[0].cast('Q')[2])
        # print("WOOW3", in_args[0].cast('Q')[3])
        # print("WOOW3", in_args[0].cast('Q')[3])

        vin = in_args[0].cast('Q')[2]
        sz = in_args[0].cast('Q')[3]
        # vin = to_mv(in_args[0].cast('Q')[2], 8).cast('Q')[0]
        # sz = to_mv(in_args[0].cast('Q')[3], 8).cast('Q')[0]

        st = qcom_dsp.FASTRPC_IOCTL_MMAP(self.rpc_fd, fd=-1, flags=rflags, vaddrin=0, size=sz)
        out_args[0].cast('Q')[0] = 0
        out_args[0].cast('Q')[1] = st.vaddrout
        prev_res = 0
      else: raise RuntimeError(f"Unknown {sc=:X}")
