from __future__ import annotations
from typing import Tuple, Dict, Any
import ctypes, os, mmap, tempfile, pathlib, array, functools, threading
from tinygrad.device import BufferOptions, Compiled, Allocator
from tinygrad.helpers import from_mv, getenv, DEBUG, round_up, mv_address, to_mv
from tinygrad.runtime.ops_clang import ClangCompiler
from tinygrad.renderer.cstyle import DSPRenderer
from tinygrad.runtime.autogen import libc, qcom_dsp
if getenv("IOCTL"): import extra.dsp.run # noqa: F401 # pylint: disable=unused-import

def qcom_sc(method=0, ins=0, outs=0, fd=0): return (method << 24) | (ins << 16) | (outs << 8)

def qcom_build_pra(ins=None, outs=None, in_fds=None):
  ins, outs, in_fds = ins or list(), outs or list(), in_fds or list()

  pra = (qcom_dsp.union_remote_arg * (len(ins) + len(outs) + len(in_fds)))()
  fds = (ctypes.c_int32 * (len(ins) + len(outs) + len(in_fds)))(*([-1] * (len(ins) + len(outs))), *in_fds)
  attrs = (ctypes.c_uint32 * (len(ins) + len(outs) + len(in_fds)))(*([0] * (len(ins) + len(outs))), *([1] * (len(in_fds))))

  for i, mv in enumerate(ins + outs): pra[i].buf.pv, pra[i].buf.len = mv_address(mv) if mv.nbytes > 0 else 0, mv.nbytes
  return pra, fds, attrs

class DSPProgram:
  def __init__(self, device:DSPDevice, name:str, lib:bytes):
    self.device, self.lib = device, lib

    # TODO: Remove lib flush to FS.
    with tempfile.NamedTemporaryFile(delete=False) as output_file:
      output_file.write(lib)
      output_file.flush()
      if DEBUG >= 6: os.system(f"llvm-objdump -mhvx -d {output_file.name}")
      self.filepath = output_file.name

  def __del__(self): os.remove(self.filepath)

  def __call__(self, *bufs, vals:Tuple[int, ...]=(), wait=False):
    handle = self.device.open_lib(self.filepath)

    timer = memoryview(bytearray(8)).cast('Q')
    var_vals_mv = memoryview(bytearray((len(bufs) + len(vals)) * 4))
    var_vals_mv.cast('i')[:] = array.array('i', tuple(b.size for b in bufs) + vals)

    pra, fds, attrs = qcom_build_pra(ins=[var_vals_mv], outs=[timer], in_fds=[b.share_info.fd for b in bufs])
    qcom_dsp.FASTRPC_IOCTL_INVOKE_ATTRS(self.device.rpc_fd, fds=fds, attrs=attrs,
                                        inv=qcom_dsp.struct_fastrpc_ioctl_invoke(handle=handle, sc=(2<<24)|(1<<16)|(1<<8)|len(bufs), pra=pra))

    self.device.close_lib(handle)
    return timer[0] / 1e6

class DSPBuffer:
  def __init__(self, va_addr:int, size:int, share_info:Any): self.va_addr, self.size, self.share_info = va_addr, size, share_info

class DSPAllocator(Allocator):
  def __init__(self, device:DSPDevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    b = qcom_dsp.ION_IOC_ALLOC(self.device.ion_fd, len=size, align=0x200, heap_id_mask=1<<qcom_dsp.ION_SYSTEM_HEAP_ID, flags=qcom_dsp.ION_FLAG_CACHED)
    share_info = qcom_dsp.ION_IOC_SHARE(self.device.ion_fd, handle=b.handle)
    va_addr = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, share_info.fd, 0)
    return DSPBuffer(va_addr, size, share_info)

  def _free(self, opaque:DSPBuffer, options:BufferOptions):
    libc.munmap(opaque.va_addr, opaque.size)
    os.close(opaque.share_info.fd)
    qcom_dsp.ION_IOC_FREE(self.device.ion_fd, handle=opaque.share_info.handle)

  def as_buffer(self, src:DSPBuffer) -> memoryview: return to_mv(src.va_addr, src.size)
  def copyin(self, dest:DSPBuffer, src:memoryview): ctypes.memmove(dest.va_addr, from_mv(src), src.nbytes)
  def copyout(self, dest:memoryview, src:DSPBuffer): ctypes.memmove(from_mv(dest), src.va_addr, dest.nbytes)

class DSPDevice(Compiled):
  def __init__(self, device:str=""):
    self.ion_fd = os.open('/dev/ion', os.O_RDONLY)
    self.rpc_fd = os.open('/dev/adsprpc-smd', os.O_RDONLY | os.O_NONBLOCK)
    self.libs: Dict[str, bytes] = {}

    qcom_dsp.FASTRPC_IOCTL_GETINFO(self.rpc_fd, 3)

    fastrpc_shell = memoryview(bytearray(pathlib.Path('/dsp/cdsp/fastrpc_shell_3').read_bytes()))
    shell_mem = qcom_dsp.ION_IOC_ALLOC(self.ion_fd, len=round_up(fastrpc_shell.nbytes, 0x1000), align=0x1000, heap_id_mask=0x2000000, flags=0x1)
    shell_mapped = qcom_dsp.ION_IOC_MAP(self.ion_fd, handle=shell_mem.handle)
    fastrpc_shell_addr = libc.mmap(0, shell_mem.len, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, shell_mapped.fd, 0)
    ctypes.memmove(fastrpc_shell_addr, mv_address(fastrpc_shell), fastrpc_shell.nbytes)

    qcom_dsp.FASTRPC_IOCTL_CONTROL(self.rpc_fd, req=0x3)
    qcom_dsp.FASTRPC_IOCTL_INIT(self.rpc_fd, flags=0x1, file=fastrpc_shell_addr, filelen=fastrpc_shell.nbytes, filefd=shell_mapped.fd)

    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0x3, sc=(0x3 << 24))

    self.listener = RPCListner(self)
    self.listener.start()

    compiler_args = ["--target=hexagon", "-mcpu=hexagonv65", "-fuse-ld=lld", "-nostdlib", "-mhvx=v65", "-mhvx-length=128B"]
    super().__init__(device, DSPAllocator(self), DSPRenderer(), ClangCompiler("compile_dsp", args=compiler_args), functools.partial(DSPProgram, self))

  def open_lib(self, filepath):
    fp = f"file:///{filepath}?entry&_modver=1.0&_dom=cdsp\0"

    a1 = memoryview(array.array('I', [len(fp), 0xff]))
    a2 = memoryview(bytearray(f"{fp}".encode()))
    o1, o2 = memoryview(bytearray(0x8)), memoryview(bytearray(0xff))
    pra, _, _ = qcom_build_pra(ins=[a1, a2], outs=[o1, o2])
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0, sc=qcom_sc(method=0, ins=2, outs=2), pra=pra)
    return o1.cast('Q')[0]

  def close_lib(self, handle):
    a1 = memoryview(array.array('I', [handle, 0xff]))
    o1, o2 = memoryview(bytearray(0x8)), memoryview(bytearray(0xff))
    pra, _, _ = qcom_build_pra(ins=[a1], outs=[o1, o2])
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0, sc=qcom_sc(method=1, ins=1, outs=2), pra=pra)

class RPCListner(threading.Thread):
  def __init__(self, device:DSPDevice):
    super().__init__()
    self.device, self.daemon = device, True

  def run(self):
    # Setup initial request arguments.
    context, status, out_buf_size = 0, 0xffffffff, 0

    # Buffers for comm.
    msg_send = memoryview(bytearray(0x10)).cast('I')
    msg_recv = memoryview(bytearray(0x10)).cast('I')
    out_buf = memoryview(bytearray(0x10000)).cast('I')
    in_buf = memoryview(bytearray(0x10000)).cast('I')
    req_args, _, _ = qcom_build_pra(ins=[msg_send, out_buf], outs=[msg_recv, in_buf])

    while True:
      # Update message request and send it.
      msg_send[:] = array.array('I', [context, status, out_buf_size, in_buf.nbytes])
      req_args[1].buf.len = out_buf_size
      qcom_dsp.FASTRPC_IOCTL_INVOKE(self.device.rpc_fd, handle=0x3, sc=0x04020200, pra=req_args)

      context, _, sc = msg_recv[:3]
      inbufs, outbufs = (sc >> 16) & 0xff, (sc >> 8) & 0xff

      in_ptr, out_ptr, objs = mv_address(in_buf), mv_address(out_buf), []
      for i in range(inbufs + outbufs):
        obj_size = to_mv(in_ptr, 4).cast('I')[0]
        obj_ptr = round_up(in_ptr + 4, 8) if i < inbufs else round_up(out_ptr + 4, 8)
        objs.append(to_mv(obj_ptr, obj_size))
        if i < inbufs: in_ptr = obj_ptr + obj_size
        else:
          to_mv(out_ptr, 4).cast('I')[0] = obj_size
          in_ptr += 4
          out_ptr = obj_ptr + obj_size

      in_args, out_args = objs[:inbufs], objs[inbufs:]
      out_buf_size = out_ptr - mv_address(out_buf)

      status = 0 # reset status, will set if error
      if sc == 0x20200: pass # greating
      elif sc == 0x13050100: # open
        try: out_args[0].cast('I')[0] = os.open(in_args[3].tobytes()[:-1].decode(), os.O_RDONLY)
        except OSError: status = 1
      elif sc == 0x9010000: # seek
        res = os.lseek(in_args[0].cast('I')[0], in_args[0].cast('I')[1], in_args[0].cast('I')[2])
        status = 0 if res >= 0 else res
      elif sc == 0x4010200: # read
        buf = os.read(in_args[0].cast('I')[0], in_args[0].cast('I')[1])
        out_args[1][:len(buf)] = buf
        out_args[0].cast('I')[0] = len(buf)
        out_args[0].cast('I')[1] = int(len(buf) == 0)
      elif sc == 0x3010000: os.close(in_args[0].cast('I')[0])
      elif sc == 0x1f020100: # stat
        stat = os.stat(in_args[1].tobytes()[:-1].decode())
        out_stat = qcom_dsp.struct_apps_std_STAT.from_address(mv_address(out_args[0]))
        for f in out_stat._fields_: out_stat.__setattr__(f[0], int(getattr(stat, f"st_{f[0]}", 0)))
      elif sc == 0x2010100: # mmap
        st = qcom_dsp.FASTRPC_IOCTL_MMAP(self.device.rpc_fd, fd=-1, flags=in_args[0].cast('I')[2], vaddrin=0, size=in_args[0].cast('Q')[3])
        out_args[0].cast('Q')[0] = 0
        out_args[0].cast('Q')[1] = st.vaddrout
      else: raise RuntimeError(f"Unknown op: {sc=:X}")
