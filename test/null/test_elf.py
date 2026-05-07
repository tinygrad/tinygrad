import unittest, subprocess, platform, importlib, ctypes
from unittest.mock import patch
import tinygrad.runtime.support.compiler_cpu as compiler_cpu
import tinygrad.runtime.ops_cpu as ops_cpu
from tinygrad.runtime.support.compiler_cpu import ClangJITCompiler
from tinygrad.runtime.support.elf import elf_loader

class TestElfLoader(unittest.TestCase):
  def test_windows_arm64_aliases(self):
    try:
      with patch("platform.machine", return_value="ARM64"):
        importlib.reload(compiler_cpu)
        importlib.reload(ops_cpu)
        self.assertEqual(compiler_cpu.LLVMCompiler.target_arch, "AArch64")

        with patch("subprocess.check_output", return_value=b"") as check:
          compiler_cpu.ClangJITCompiler().compile_to_obj("int test(void) { return 0; }")
        self.assertIn("--target=arm64-none-unknown-elf", check.call_args.args[0])
        self.assertIn("-ffixed-x18", check.call_args.args[0])

        seen = []
        class Prg:
          runtimevars = {}
          def fxn(self, *args): seen.extend(type(arg) for arg in args)

        queue = object.__new__(ops_cpu.CPUComputeQueue)
        queue._exec(0, Prg(), 1, 1, 2)
        self.assertIs(seen[1], ctypes.c_int64)
    finally:
      importlib.reload(compiler_cpu)
      importlib.reload(ops_cpu)

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
      ClangJITCompiler().compile(src)
  def test_link(self):
    src = '''
      float powf(float, float); // from libm
      float test(float x, float y) { return powf(x, y); }
    '''
    args = ('-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native', '-fPIC', '-O2', '-ffreestanding', '-nostdlib')
    obj = subprocess.check_output(('clang',) + args + ('-', '-o', '-'), input=src.encode())
    with self.assertRaisesRegex(RuntimeError, 'powf'): elf_loader(obj)
    elf_loader(obj, link_libs=['m'])

if __name__ == '__main__':
  unittest.main()
