import unittest, subprocess, platform
from tinygrad.runtime.support.compiler_cpu import ClangCompiler
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.c import DLL

class TestElfLoader(unittest.TestCase):
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
      elf_loader(ClangCompiler([{'AMD64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine(), m), "native"]).compile(src))
  def test_link_symbols(self):
    src = "extern void helper(int *); void caller(int *x) { helper(x); }"
    obj = ClangCompiler([{'AMD64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine(), m), "native"]).compile(src)
    _, _, relocs = elf_loader(obj, link_syms={"helper":0x12340000})
    self.assertTrue(any(target == 0x12340000 for _,target,_,_ in relocs))
    with self.assertRaisesRegex(RuntimeError, "helper"): elf_loader(obj)

  def test_link(self):
    src = '''
      float powf(float, float); // from libm
      float test(float x, float y) { return powf(x, y); }
    '''
    args = ('-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native', '-fPIC', '-O2', '-ffreestanding', '-nostdlib')
    obj = subprocess.check_output(('clang',) + args + ('-', '-o', '-'), input=src.encode())
    with self.assertRaisesRegex(RuntimeError, 'powf'): elf_loader(obj)
    elf_loader(obj, link_libs=[DLL('m', 'm')])

if __name__ == '__main__':
  unittest.main()
