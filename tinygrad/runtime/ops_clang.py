import ctypes, subprocess, pathlib, tempfile, mmap
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import OSX, cpu_time_execution, DEBUG, cpu_objdump, mv_address
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import libc, mac, macho

def macho_loader(blob: bytes) -> memoryview:
  header, curr_loc, sections = macho.struct_mach_header_64.from_buffer_copy(blob), ctypes.sizeof(macho.struct_mach_header_64), []
  for _ in range(header.ncmds):
    cmd = macho.struct_load_command.from_buffer_copy(blob, curr_loc)
    if cmd.cmd == macho.LC_SEGMENT_64:
      seg = macho.struct_segment_command_64.from_buffer_copy(blob, curr_loc)
      sections += (macho.struct_section_64 * seg.nsects).from_buffer_copy(blob, curr_loc + ctypes.sizeof(macho.struct_segment_command_64))
    curr_loc += cmd.cmdsize
  image = bytearray(max([sh.addr + sh.size for sh in sections]))
  for sh in sections: image[sh.addr:sh.addr+sh.size] = blob[sh.offset:sh.offset+sh.size]
  return memoryview(image)

class ClangCompiler(Compiler):
  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      # need to specify entrypoint so that ld doesn't complain (it's not fatal, so do we care?)
      name = ('_' if OSX else '') + src[(start:=src.rfind('void')+5):src[start:].find('(')+start]
      subprocess.check_output(['clang', '-static', '-march=native', '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib',
                               '-fno-math-errno', '-e', name, '-Wl,-segaddr,text,0,-pagezero_size,0,-preload' if OSX else '-Wl,-Ttext=0', '-', '-o',
                               str(output_file.name)], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    if DEBUG >= 6: cpu_objdump(lib)
    self.name, self.lib = name, lib
    image = macho_loader(lib) if OSX else elf_loader(lib)[0]
    addr = libc.mmap(0, len(image), mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_ANON | mmap.MAP_PRIVATE, -1, 0)
    assert addr != 0xffffffffffffffff
    ctypes.memmove(addr, mv_address(image), len(image))
    if OSX: mac.sys_icache_invalidate(addr, len(image))
    assert libc.mprotect(addr, len(image), mmap.PROT_READ | mmap.PROT_EXEC) != 0xffffffffffffffff
    self.fxn = ctypes.CFUNCTYPE(None)(addr)

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler("compile_clang"), ClangProgram, ClangGraph)
