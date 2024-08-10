import unittest, os
from tinygrad.runtime.support.elf import elf_loader
from typing import List, Any

def helper_test_loader(file: str, img: str, relocs_expected: List[Any]):
  with open(f'{os.path.dirname(__file__)}/blobs/{file}', 'rb') as fd: blob = fd.read()
  image, _, relocs = elf_loader(blob)
  with open(f'{os.path.dirname(__file__)}/blobs/{img}', 'rb') as fd: image_expected = fd.read()
  assert bytes(image) == image_expected, f'Image missmatch\nexpected: {image_expected!r}\ngot: {bytes(image)!r}'
  assert relocs == relocs_expected, f'Relocs missmatch\nexpected: {[]}\ngot: {relocs_expected}'

class TestZeroCopy(unittest.TestCase):
  def test_load_amd_hip(self): helper_test_loader('hip.elf', 'hip.img', [])
  def test_load_nvidia_cuda(self): helper_test_loader('cuda.cubin', 'cuda.img', [(68, 512, 2, 0)])
  def test_load_clang_jit_simple(self): helper_test_loader('clang.o', 'clang.img', [])
  def test_load_clang_jit_reloc(self): helper_test_loader('clang_reloc.o', 'clang_reloc.img', [(4, 0, 282, 0)])

if __name__ == '__main__':
  unittest.main()
