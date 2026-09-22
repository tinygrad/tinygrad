import unittest, subprocess, platform, struct
from unittest.mock import patch
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.compiler_cpu import ClangCompiler
from tinygrad.runtime.support.elf import elf_loader, jit_loader
from tinygrad.runtime.support.c import DLL

class TestElfLoader(unittest.TestCase):
  def test_x86_plt32_range(self):
    base, code = 1 << 40, b'\xe8\x00\x00\x00\x00\xc3'
    for displacement in (-2**32, -2**31-1, -2**31, -16, 16, 2**31-1, 2**31, 2**32):
      with self.subTest(displacement=displacement):
        target = base + 5 + displacement
        relocs = [(1, target, libc.R_X86_64_PLT32, -4)]
        with patch('tinygrad.runtime.support.elf.elf_loader', return_value=(memoryview(code), [], relocs)):
          linked = jit_loader(b'', base=base)
        dest = base + 5 + struct.unpack_from('<i', linked, 1)[0]
        if -2**31 <= displacement < 2**31:
          self.assertEqual(dest, target)
          self.assertEqual(len(linked), len(code))
        else:
          self.assertEqual(dest, base + len(code))
          self.assertEqual(linked[len(code):len(code)+6], b'\xff\x25\x00\x00\x00\x00')
          self.assertEqual(struct.unpack_from('<Q', linked, len(code)+6)[0], target)

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
