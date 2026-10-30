import unittest, subprocess, platform
from unittest.mock import patch
from tinygrad.runtime.support.compiler_cpu import ClangCompiler
from tinygrad.runtime.support.elf import elf_loader, jit_loader
from tinygrad.runtime.support.c import DLL

class TestElfLoader(unittest.TestCase):
  def test_darwin_jit_no_stack_protector(self):
    src = 'extern void fill(long *); long test(void) { long buf[2]; fill(buf); return buf[0] + buf[1]; }'
    check_output = subprocess.check_output
    # Model Apple Clang's default stack protection even when running on Linux.
    def compile_with_stack_protector(args, **kwargs): return check_output([args[0], '-fstack-protector-all', *args[1:]], **kwargs)
    with patch('tinygrad.runtime.support.compiler_cpu.OSX', True), patch('subprocess.check_output', compile_with_stack_protector):
      obj = ClangCompiler(['arm64', 'generic']).compile(src)
    jit_loader(obj, base=0x100000000, link_syms={'fill': 0x200000000, '__stack_chk_guard': 0x200000008, '__stack_chk_fail': 0x200000010})
    self.assertNotIn(b'__stack_chk', obj)

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
