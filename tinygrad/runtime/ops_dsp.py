from __future__ import annotations
import ctypes, os, mmap, tempfile, pathlib, array, threading, contextlib, subprocess, struct, hashlib
from tinygrad.device import BufferStorage, MMIOInterface, BufferSpec, Compiled, Allocator, Program
from tinygrad.renderer import Compiler, TinyELF
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import Ops, UOp
from tinygrad.helpers import getenv, round_up, mv_address, to_mv, cpu_objdump, system, DEBUG, suppress_finalizing, Target, unwrap
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.autogen import libc, qcom_dsp
from tinygrad.runtime.support.memory import TLSFAllocator
if getenv("IOCTL"): import extra.dsp.run # noqa: F401 # pylint: disable=unused-import

# a lib that stays loaded: it maps buffers on the DSP, loads the kernels and runs them (writing back and dropping the DSP's cache around them)
DSP_MAP, DSP_LOAD, DSP_RUN = range(3)
DSP_UTIL = f"""typedef union {{ struct {{ void *pv; unsigned int len; }} buf; struct {{ int fd; unsigned int offset; }} dma; }} remote_arg;
struct dcvs_v2_req {{ int type; int _pad; _Bool dcvs_enable; char dcvs_option; _Bool set_latency; int latency; _Bool set_dcvs_params; short _pad2;
  char target_corner; char min_corner; char max_corner; int _pad3[3]; }};
int HAP_power_set(void*, void*); unsigned long long HAP_perf_get_time_us(void);
void* HAP_mmap(void *addr, int len, int prot, int flags, int fd, long offset);
int qurt_mem_cache_clean(unsigned int, unsigned int, int, int); void *dlopen(const char*, int); void *dlsym(void*, const char*);
int entry(unsigned long long handle, unsigned int sc, remote_arg* pra) {{
  unsigned int *in = pra[0].buf.pv, *out = pra[1].buf.pv;
  if (in[0] == {DSP_MAP}) out[0] = (unsigned int)HAP_mmap(0, in[1], 3, 0, pra[2].dma.fd, 0);
  if (in[0] == {DSP_LOAD}) {{
    struct dcvs_v2_req req = {{.type=7, .dcvs_enable=0, .set_latency=1, .latency=100, .set_dcvs_params=1, .target_corner=6 /* TURBO */}};
    HAP_power_set((void*)handle, (void*)&req);
    void *lib = dlopen((const char*)(in+1), 2 /* RTLD_NOW */);
    out[0] = lib ? (unsigned int)dlsym(lib, "kernel_run") : 0;
  }}
  if (in[0] == {DSP_RUN}) {{
    qurt_mem_cache_clean(0, 0, 4 /* QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL */, 1 /* QURT_MEM_DCACHE */);
    unsigned long long start = HAP_perf_get_time_us();
    ((void (*)(const unsigned char*))in[1])((const unsigned char*)(in+2));
    out[0] = HAP_perf_get_time_us() - start;
    qurt_mem_cache_clean(0, 0, 4, 1);
  }}
  return 0;
}}"""

class DSPRenderer(ClangRenderer):
  buffer_suffix = " restrict __attribute__((align_value(128)))"
  kernel_typedef = "__attribute__((noinline)) void"
  packed_args = False
  # (HVX v65 has no float)
  vector_lanes, vector_floats, channels_last = 128, False, True
  type_map = { **ClangRenderer.type_map, dtypes.uint64: "unsigned long long", dtypes.int64: "long long" }
  code_for_op = {k:v for k,v in ClangRenderer.code_for_op.items() if k != Ops.SQRT}

  def __init__(self, target:Target): self.target, self.compiler = target, DSPCompiler()

  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str:
    # (the args are 8 bytes each: values and the buffers' addresses on the DSP, see DSP_UTIL)
    tys = [self._render_dtype(b[1][0].dtype) if b[1][0].addrspace == AddrSpace.ALU else "void*" for b in bufs]
    return f"void kernel_run(const unsigned char *a) {{ {function_name}({', '.join(f'*({t}*)(a+{i*8})' for i,t in enumerate(tys))}); }}"

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s+(dtypes.bfloat16,)}

def rpc_sc(method=0, ins=0, outs=0, fds=0): return (method << 24) | (ins << 16) | (outs << 8) | fds
def rpc_prep_args(ins=None, outs=None, in_fds=None):
  ins, outs, in_fds = ins or list(), outs or list(), in_fds or list()

  pra = (qcom_dsp.union_remote_arg * (len(ins) + len(outs) + len(in_fds)))()
  fds = (ctypes.c_int32 * (len(ins) + len(outs) + len(in_fds)))(*([-1] * (len(ins) + len(outs))), *in_fds)
  attrs = (ctypes.c_uint32 * (len(ins) + len(outs) + len(in_fds)))(*([0] * (len(ins) + len(outs))), *([1] * (len(in_fds))))

  for i, mv in enumerate(ins + outs): pra[i].buf.pv, pra[i].buf.len = ctypes.c_void_p(mv_address(mv) if mv.nbytes > 0 else 0), mv.nbytes
  return pra, fds, attrs, (ins, outs)

class DSPProgram(Program['DSPDevice']):
  def __init__(self, dev:DSPDevice, obj:TinyELF, links=None):
    self.dev, self.signature = dev, obj.signature
    # (the util lib loads it by a name of its own, the listener reads it: the DSP's dlopen of a loaded name is that lib)
    dev.binded_lib, dev.binded_lib_off = obj.lib, 0
    self.fxn = dev.util(DSP_LOAD, data=f"tinylib{hashlib.sha256(obj.lib).hexdigest()[:16]}\0".encode())
    if self.fxn == 0: raise RuntimeError(f"the DSP can't load {obj.name}")

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), wait=False, **kw):
    # (8 bytes each: values and the buffers' addresses on the DSP)
    vals = [struct.pack(f"<{unwrap(dt.fmt)}", x) if a is AddrSpace.ALU else struct.pack("<I", x.dsp_addr)
            for x,(_,a,dt,_) in zip(args, self.signature)]
    return self.dev.util(DSP_RUN, self.fxn, data=b''.join(v.ljust(8, b'\0') for v in vals)) / 1e6

class DSPBuffer:
  def __init__(self, va_addr:int, size:int, share_info, offset:int=0, dsp_addr:int=0):
    self.va_addr, self.size, self.share_info, self.offset, self.dsp_addr = va_addr, size, share_info, offset, dsp_addr

class DSPAllocator(Allocator['DSPDevice']):
  # the DSP maps few buffers: a buffer is a part of an arena, mapped on the DSP once (with room for the 128 bytes an unaligned vector load reads
  # past it). they aren't cached on the host, the DSP writes back its cache (see DSP_UTIL)
  def __init__(self, dev:DSPDevice):
    super().__init__(dev)
    self.arenas: list[tuple[DSPBuffer, TLSFAllocator]] = []

  def _ion(self, size:int) -> DSPBuffer:
    if getenv("MOCKDSP"): fd, share_info, flags = -1, None, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS
    else:
      b = qcom_dsp.ION_IOC_ALLOC(self.dev.ion_fd, len=size+128, align=0x1000, heap_id_mask=1<<qcom_dsp.ION_SYSTEM_HEAP_ID, flags=0)
      fd, flags = (share_info:=qcom_dsp.ION_IOC_SHARE(self.dev.ion_fd, handle=b.handle)).fd, mmap.MAP_SHARED
    return DSPBuffer(libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, flags, fd, 0), size, share_info)

  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    # (the mock's buffers, and the fastrpc shell's before the util lib is loaded, are their own)
    if getenv("MOCKDSP") or not hasattr(self.dev, "util_handle"): return BufferStorage(buf:=self._ion(size), None, MMIOInterface(buf.va_addr, size))
    for arena, parts in self.arenas:
      with contextlib.suppress(MemoryError):
        return BufferStorage(buf:=self._offset(arena, size, off:=parts.alloc(size, 128)), (parts, off), MMIOInterface(buf.va_addr, size))
    arena = self._ion(asize:=max(32 << 20, round_up(size, 0x1000)))
    arena.dsp_addr = self.dev.util(DSP_MAP, asize+128, fd=arena.share_info.fd)
    self.arenas.append((arena, TLSFAllocator(asize)))
    return self._alloc(size, options)

  @suppress_finalizing
  def _free(self, storage:BufferStorage, options:BufferSpec):
    if storage.meta is not None: return storage.meta[0].free(storage.meta[1])
    libc.munmap(storage.buf.va_addr, storage.buf.size)
    if storage.buf.share_info is not None:
      os.close(storage.buf.share_info.fd)
      qcom_dsp.ION_IOC_FREE(self.dev.ion_fd, handle=storage.buf.share_info.handle)

  def _as_buffer(self, src:DSPBuffer) -> memoryview: return to_mv(src.va_addr, src.size)
  def _copyin(self, dest:DSPBuffer, src:memoryview): ctypes.memmove(dest.va_addr, mv_address(src), src.nbytes)
  def _copyout(self, dest:memoryview, src:DSPBuffer): ctypes.memmove(mv_address(dest), src.va_addr, dest.nbytes)
  def _offset(self, buf, size:int, offset:int): return DSPBuffer(buf.va_addr+offset, size, buf.share_info, buf.offset+offset, buf.dsp_addr+offset)

class DSPCompiler(Compiler):
  def __init__(self, mock:bool=False):
    self.mock, compiler_args = mock, "--target=hexagon -mcpu=hexagonv65 -fuse-ld=lld -nostdlib -mhvx=v65 -mhvx-length=128b"
    if mock: self.args = f"-static {compiler_args}"
    else:
      # Generate link script to pass into clang. Aligning all used sections to 4k fixes invoke problem.
      sections = ['text', 'rela.plt', 'rela.dyn', 'plt', 'data', 'bss', 'hash', 'dynamic',
                  'got', 'got.plt', 'dynsym', 'dynstr', 'symtab', 'shstrtab', 'strtab']
      sections_link = '\n'.join([f'.{n} : ALIGN(4096) {{ *(.{n}) }}' for n in sections])
      with tempfile.NamedTemporaryFile(delete=False) as self.link_ld:
        self.link_ld.write(f"SECTIONS {{ . = 0x0; {sections_link}\n /DISCARD/ : {{ *(.note .note.* .gnu.hash .comment) }} }}".encode())
        self.link_ld.flush()

      self.args = f"-shared {compiler_args} -T{self.link_ld.name}"

    super().__init__(None if mock else "compile_dsp")

  def __del__(self):
    if not self.mock: os.unlink(self.link_ld.name)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as f:
      system(f"{getenv('CC','clang')} {self.args} -O2 -Wall -Werror -fno-stack-protector -x c -fPIC " +
             f"-ffreestanding -nostdlib - -o {f.name}", input=src.encode())
      return pathlib.Path(f.name).read_bytes()

  def disassemble(self, lib:bytes): return cpu_objdump(lib, "llvm-objdump")


class MockDSPRenderer(DSPRenderer):
  def __init__(self, target:Target): self.target, self.compiler = target, DSPCompiler(mock=True)
  def _render_defines(self, uops) -> list[str]: return ClangRenderer._render_defines(self, uops)
  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[UOp,bool]]]) -> str:
    # https://gpages.juszkiewicz.com.pl/syscalls-table/syscalls.html
    # control register 21 is HEX_REG_QEMU_INSN_CNT, 0x6a15c000 loads it
    msrc = [mockdsp_boilerplate, 'void _start(void) {']
    for i,b in enumerate(bufs):
      if b[1][0].addrspace == AddrSpace.GLOBAL:
        sz = b[1][0].max_numel()*b[1][0].dtype.itemsize
        # for loop for big reads
        msrc.append(f"void *buf{i} = mmap2(0, {sz}, 3, 0x21, -1, 0); for(int rd = 0; rd < {sz}; rd += read(0, buf{i}+rd, {sz}-rd));")
      else:
        msrc.append(f"{self._render_dtype(b[1][0].dtype)} val{i}; read(0, &val{i}, {b[1][0].dtype.itemsize});")
    msrc.append("unsigned int st = inscount();")
    params = [(f'(void*)buf{i}' if b[1][0].addrspace == AddrSpace.GLOBAL else f'val{i}') for i,b in enumerate(bufs)]
    msrc.append(f"{function_name}({', '.join(params)});")
    msrc.append("unsigned int et = inscount() - st; write(1, &et, sizeof(et));")
    for i,b in enumerate(bufs):
      if b[1][0].addrspace == AddrSpace.GLOBAL: msrc.append(f"write(1, buf{i}, {b[1][0].max_numel()*b[1][0].dtype.itemsize});")
    msrc.append('exit(0); }')
    return '\n'.join(msrc)

class DSPDevice(Compiled):
  renderers = [MockDSPRenderer] if getenv("MOCKDSP") else [DSPRenderer]
  def __init__(self, device:str=""):
    if getenv("MOCKDSP"): super().__init__(device, DSPAllocator(self), MockDSPProgram)
    else:
      self.ion_fd = os.open('/dev/ion', os.O_RDONLY)
      super().__init__(device, DSPAllocator(self), DSPProgram)
      fastrpc_shell = memoryview(bytearray(pathlib.Path('/dsp/cdsp/fastrpc_shell_3').read_bytes()))
      self.shell_buf = self.allocator.alloc(round_up(fastrpc_shell.nbytes, 0x1000), BufferSpec(nolru=True)).buf
      ctypes.memmove(self.shell_buf.va_addr, mv_address(fastrpc_shell), fastrpc_shell.nbytes)

      self.init_dsp()
      RPCListener(self).start()
      self.util_handle = self.open_lib(DSPCompiler().compile_cached(DSP_UTIL))

  def open_lib(self, lib):
    self.binded_lib, self.binded_lib_off = lib, 0
    fp = "file:///tinylib?entry&_modver=1.0&_dom=cdsp\0"
    pra, _, _, _ = rpc_prep_args(ins=[memoryview(array.array('I', [len(fp), 0xff])), memoryview(bytearray(fp.encode()))],
                                 outs=[o1:=memoryview(bytearray(0x8)), o2:=memoryview(bytearray(0xff))])
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0, sc=rpc_sc(method=0, ins=2, outs=2), pra=pra)
    if o1.cast('i')[1] < 0: raise RuntimeError(f"Cannot open lib: {o2.tobytes().decode()}")
    return o1.cast('I')[0]

  def util(self, op:int, *vals:int, data:bytes=b"", fd:int|None=None) -> int:
    pra, fds, attrs, _ = rpc_prep_args(ins=[memoryview(bytearray(array.array('I', [op, *vals]).tobytes() + data))],
                                       outs=[out:=memoryview(bytearray(4)).cast('I')], in_fds=[fd] if fd is not None else [])
    qcom_dsp.FASTRPC_IOCTL_INVOKE_ATTRS(self.rpc_fd, fds=fds, attrs=attrs, inv=qcom_dsp.struct_fastrpc_ioctl_invoke(handle=self.util_handle,
                                        sc=rpc_sc(method=2, ins=1, outs=1, fds=len(fds)-2), pra=pra))
    return out[0]

  def init_dsp(self):
    if hasattr(self, 'rpc_fd'):
      with contextlib.suppress(OSError):
        qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=4, sc=rpc_sc(method=2, ins=0, outs=0)) # pylint: disable=access-member-before-definition
      os.close(self.rpc_fd) # pylint: disable=access-member-before-definition

    self.rpc_fd: int = os.open('/dev/adsprpc-smd', os.O_RDONLY | os.O_NONBLOCK)
    qcom_dsp.FASTRPC_IOCTL_GETINFO(self.rpc_fd, 3)
    qcom_dsp.FASTRPC_IOCTL_CONTROL(self.rpc_fd, req=0x3)
    qcom_dsp.FASTRPC_IOCTL_INIT(self.rpc_fd, flags=0x1, file=self.shell_buf.va_addr, filelen=self.shell_buf.size, filefd=self.shell_buf.share_info.fd)
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=3, sc=rpc_sc(method=3, ins=0, outs=0))

class RPCListener(threading.Thread):
  def __init__(self, device:DSPDevice):
    super().__init__()
    self.device, self.daemon = device, True

  def run(self):
    # Setup initial request arguments.
    context, status, TINYFD = 0, 0xffffffff, 0xffff
    req_args, _, _, _ = rpc_prep_args(ins=[msg_send:=memoryview(bytearray(0x10)).cast('I'), out_buf:=memoryview(bytearray(0x10000)).cast('I')],
                                      outs=[msg_recv:=memoryview(bytearray(0x10)).cast('I'), in_buf:=memoryview(bytearray(0x10000)).cast('I')])
    req_args[1].buf.len = 0

    while True:
      # Update message request and send it.
      msg_send[:] = array.array('I', [context, status, req_args[1].buf.len, in_buf.nbytes])

      try: qcom_dsp.FASTRPC_IOCTL_INVOKE(self.device.rpc_fd, handle=0x3, sc=0x04020200, pra=req_args)
      except OSError: continue # retry

      context, inbufs, outbufs = msg_recv[0], ((sc:=msg_recv[2]) >> 16) & 0xff, (msg_recv[2] >> 8) & 0xff

      in_ptr, out_ptr, objs = mv_address(in_buf), mv_address(out_buf), []
      for i in range(inbufs + outbufs):
        obj_ptr = round_up(in_ptr + 4, 8) if i < inbufs else round_up(out_ptr + 4, 8)
        objs.append(to_mv(obj_ptr, obj_size:=to_mv(in_ptr, 4).cast('I')[0]))
        if i < inbufs: in_ptr = obj_ptr + obj_size
        else:
          to_mv(out_ptr, 4).cast('I')[0] = obj_size
          out_ptr = obj_ptr + obj_size
          in_ptr += 4

      in_args, out_args = objs[:inbufs], objs[inbufs:]
      req_args[1].buf.len = out_ptr - mv_address(out_buf)

      status = 0 # reset status, will set if error
      if sc == 0x20200: pass # greating
      elif sc == 0x13050100: # open
        try: out_args[0].cast('I')[0] = TINYFD if (name:=in_args[3].tobytes()[:-1].decode()).startswith("tinylib") else os.open(name, os.O_RDONLY)
        except OSError: status = 1
      elif sc == 0x3010000:
        if (fd:=in_args[0].cast('I')[0]) != TINYFD: os.close(fd)
      elif sc == 0x9010000: # seek
        if (fd:=in_args[0].cast('I')[0]) == TINYFD:
          assert in_args[0].cast('I')[2] == qcom_dsp.APPS_STD_SEEK_SET, "Supported only SEEK_SET"
          res, self.device.binded_lib_off = 0, in_args[0].cast('I')[1]
        else: res = os.lseek(fd, in_args[0].cast('I')[1], in_args[0].cast('I')[2])
        status = 0 if res >= 0 else res
      elif sc == 0x4010200: # read
        if (fd:=in_args[0].cast('I')[0]) == TINYFD:
          buf = self.device.binded_lib[self.device.binded_lib_off:self.device.binded_lib_off+in_args[0].cast('I')[1]]
          self.device.binded_lib_off += len(buf)
        else: buf = os.read(fd, in_args[0].cast('I')[1])
        out_args[1][:len(buf)] = buf
        out_args[0].cast('I')[0:2] = array.array('I', [len(buf), int(len(buf) == 0)])
      elif sc == 0x1f020100: # stat
        stat = os.stat(in_args[1].tobytes()[:-1].decode())
        out_stat = qcom_dsp.struct_apps_std_STAT.from_address(mv_address(out_args[0]))
        for f in out_stat._real_fields_: out_stat.__setattr__(f[0], int(getattr(stat, f"st_{f[0]}", 0)))
      elif sc == 0x2010100: # mmap
        st = qcom_dsp.FASTRPC_IOCTL_MMAP(self.device.rpc_fd, fd=-1, flags=in_args[0].cast('I')[2], vaddrin=0, size=in_args[0].cast('Q')[3])
        out_args[0].cast('Q')[0:2] = array.array('Q', [0, st.vaddrout])
      else: raise RuntimeError(f"Unknown op: {sc=:X}")

# ***** mock DSP *****

mockdsp_boilerplate = '''/* DSP boilerplate */ static long syscall(long r0, long r1, long r2, long r3, long r4, long r5, long r6) {
long retval; __asm__ volatile("r0 = %1; r1 = %2; r2 = %3; r3 = %4; r4 = %5; r5 = %6; r6 = %7; trap0(#1); %0 = r0" : "=r" (retval)
  : "r" (r0), "r" (r1), "r" (r2), "r" (r3), "r" (r4), "r" (r5), "r" (r6) : "r0", "r1", "r2", "r3", "r4", "r5", "r6"); return retval; }
static int read(int fd, void* buf, int len) {{ return syscall(fd, (long)buf, len, 0, 0, 0, 63); }}
static int write(int fd, void* buf, int len) {{ return syscall(fd, (long)buf, len, 0, 0, 0, 64); }}
static int exit(int ret) {{ return syscall(ret, 0, 0, 0, 0, 0, 93); }}
static unsigned int inscount(void) {{ unsigned int ret; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r" (ret) : : "r0"); return ret; }}
static void *mmap2(void *addr, unsigned int length, int prot, int flags, int fd, unsigned long offset) {{
return (void*)syscall((long)addr, length, prot, flags, fd, offset, 222); }}'''

class MockDSPProgram(Program[DSPDevice]):
  def __init__(self, dev:DSPDevice, obj:TinyELF, links=None): self.lib, self.signature = obj.lib, obj.signature
  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), wait=False, **kw):
    with tempfile.NamedTemporaryFile(suffix=".out") as dsp_lib:
      dsp_lib.write(self.lib)
      dsp_lib.flush()
      os.chmod(dsp_lib.name, 0o0777)
      proc = subprocess.run(["qemu-hexagon-static", *(['-strace'] if DEBUG >= 5 else []), dsp_lib.name],
        input=b''.join(struct.pack(unwrap(dt.fmt), x) if a is AddrSpace.ALU else bytes(to_mv(x.va_addr, x.size))
                       for x,(_,a,dt,_) in zip(args, self.signature)),
        stdout=subprocess.PIPE, check=True)
    offset = 4
    for x in (x for x,(_,a,_,_) in zip(args, self.signature) if a is not AddrSpace.ALU):
      to_mv(x.va_addr, x.size)[:] = proc.stdout[offset:offset+x.size]
      offset += x.size
    assert offset == len(proc.stdout)
    return struct.unpack("I", proc.stdout[0:4])[0] / 1e9  # pretend it's 1 Ghz, but this is an inscount, not a time
