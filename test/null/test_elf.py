import unittest, subprocess, platform, ctypes, mmap
from tinygrad.helpers import mv_address
from tinygrad.runtime.support.compiler_cpu import ClangCompiler
from tinygrad.runtime.support.elf import elf_loader, elf_symbol_offsets, jit_loader

ARCH = {"AMD64":"x86_64", "aarch64":"arm64"}.get(platform.machine(), platform.machine())

class TestElfLoader(unittest.TestCase):
  @staticmethod
  def load_function(obj:bytes, signature, **kwargs):
    lib = jit_loader(obj, **kwargs)
    mem = mmap.mmap(-1, len(lib), mmap.MAP_ANON|mmap.MAP_PRIVATE, mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)
    mem.write(lib)
    return mem, signature(mv_address(mem))

  def test_load_clang_jit_strtab(self):
    src = '''
      int something; // will be a load from a relocation (needed for .rela.text to exist)
      int test(int x) {
        return something + x;
      }
    '''
    args = ('-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native', '-fPIC', '-O2', '-ffreestanding', '-nostdlib')
    obj = subprocess.check_output(('clang',) + args + ('-', '-o', '-'), input=src.encode('utf-8'))
    _, sections, _ = elf_loader(obj)
    section_names = [sh.name for sh in sections]
    assert '.text' in section_names and '.rela.text' in section_names, str(section_names)
  def test_clang_jit_compiler_external_raise(self):
    src = '''
      int evil_external_function(int);
      int test(int x) {
        return evil_external_function(x+2)*2;
      }
    '''
    with self.assertRaisesRegex(RuntimeError, 'evil_external_function'):
      ClangCompiler([{'AMD64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine(), m), "native"]).compile(src)
  def test_link(self):
    src = '''
      float powf(float, float); // from libm
      float test(float x, float y) { return powf(x, y); }
    '''
    args = ('-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native', '-fPIC', '-O2', '-ffreestanding', '-nostdlib')
    obj = subprocess.check_output(('clang',) + args + ('-', '-o', '-'), input=src.encode())
    with self.assertRaisesRegex(RuntimeError, 'powf'): elf_loader(obj)
    elf_loader(obj, link_libs=['m'])
    _, test = self.load_function(obj, ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float, ctypes.c_float), link_libs=['m'])
    self.assertEqual(test(2, 3), 8)

  def test_bss(self):
    obj = ClangCompiler([ARCH, "native"]).compile_to_obj("static int counter; int test(void) { return ++counter; }")
    _, test = self.load_function(obj, ctypes.CFUNCTYPE(ctypes.c_int))
    self.assertEqual((test(), test()), (1, 2))

  def test_symbol_offsets(self):
    src = "static __attribute__((noinline)) int helper(int x) { return x + 2; } int test(int x) { return helper(x) + 1; }"
    obj = ClangCompiler([ARCH, "native"]).compile_to_obj(src)
    offsets = elf_symbol_offsets(obj)
    lib = jit_loader(obj)
    mem = mmap.mmap(-1, len(lib), mmap.MAP_ANON|mmap.MAP_PRIVATE, mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)
    mem.write(lib)
    test = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(mv_address(mem) + offsets["test"])
    self.assertEqual(test(4), 7)

  def test_link_symbol_address(self):
    obj = ClangCompiler([ARCH, "native"]).compile_to_obj("extern int injected(int); int test(int x) { return injected(x); }")
    callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(lambda x: x + 3)
    address = ctypes.cast(callback, ctypes.c_void_p).value
    self.assertIsNotNone(address)
    _, test = self.load_function(obj, ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int), link_syms={"injected": int(address)})
    self.assertEqual(test(4), 7)

if __name__ == '__main__':
  unittest.main()
