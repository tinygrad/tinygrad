import unittest, subprocess, platform
from tinygrad.runtime.support.compiler_cpu import ClangCompiler, link_objects
from tinygrad.runtime.support.elf import elf_loader, jit_loader, elf_rename_symbols, elf_sections, elf_symbols
from tinygrad.runtime.support.c import DLL
from tinygrad.runtime.autogen import libc

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
  def test_link(self):
    src = '''
      float powf(float, float); // from libm
      float test(float x, float y) { return powf(x, y); }
    '''
    args = ('-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native', '-fPIC', '-O2', '-ffreestanding', '-nostdlib')
    obj = subprocess.check_output(('clang',) + args + ('-', '-o', '-'), input=src.encode())
    with self.assertRaisesRegex(RuntimeError, 'powf'): elf_loader(obj)
    elf_loader(obj, link_libs=[DLL('m', 'm')])
    caller, helper = [subprocess.check_output(('clang',) + args + ('-', '-o', '-'), input=src.encode()) for src in
                      ('int helper(int); int test(int x) { return helper(x)+2; }', 'int helper(int x) { return x+1; }')]
    linked = link_objects([(caller, "", ()), (helper, "", ())])
    renamed = []
    names = {"helper": "a_much_longer_helper_name", "test": "entry"}
    for obj in (caller, helper):
      renamed.append(elf_rename_symbols(obj, names))
      old_symbols, new_symbols = [elf_symbols(*elf_sections(lib)) for lib in (obj, renamed[-1])]
      self.assertEqual([names.get(name, name) for name, _ in old_symbols], [name for name, _ in new_symbols])
      for (_, old), (_, new) in zip(old_symbols, new_symbols):
        old.st_name = new.st_name
        self.assertEqual(bytes(old), bytes(new))
    self.assertEqual(jit_loader(linked), jit_loader(link_objects([(obj, "", ()) for obj in renamed])))
    # Internal PC-relative calls have the same displacement wherever the complete image is loaded.
    self.assertEqual(jit_loader(linked, base=0x10000000), jit_loader(linked, base=0x20000000))
    image = jit_loader(linked, base=0x10000000)
    for pos, target, typ, addend in elf_loader(linked)[2]:
      if typ == libc.R_AARCH64_CALL26:
        offset = int.from_bytes(image[pos:pos+4], 'little') & 0x3ffffff
        self.assertEqual((offset - (1<<26) if offset & (1<<25) else offset)*4, target+addend-pos)

if __name__ == '__main__':
  unittest.main()
