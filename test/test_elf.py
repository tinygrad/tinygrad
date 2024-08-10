import unittest, os
from tinygrad.runtime.support.elf import elf_loader
from typing import Dict, List, Any

def helper_test_loader(file: str, img: str, relocs_expected: List[Any], exports_expected: Dict[str, int]):
  with open(f'{os.path.dirname(__file__)}/blobs/{file}', 'rb') as fd: blob = fd.read()
  image, _, relocs, exports = elf_loader(blob)
  with open(f'{os.path.dirname(__file__)}/blobs/{img}', 'rb') as fd: image_expected = fd.read()
  assert bytes(image) == image_expected, f'Image missmatch\nexpected: {image_expected!r}\ngot: {bytes(image)!r}'
  assert relocs == relocs_expected, f'Relocs missmatch\nexpected: {[]}\ngot: {relocs_expected}'
  assert exports == exports_expected, f'Exports missmatch\nexpected: {exports_expected}\ngot: {exports}'

class TestZeroCopy(unittest.TestCase):
  def test_load_amd_hip(self): helper_test_loader('hip.elf', 'hip.img', [], {'r_32_16_8_16_256_4_4_4': 11264})
  def test_load_nvidia_cuda(self): helper_test_loader('cuda.cubin', 'cuda.img', [(68, 512, 2, 0)], {'r_32_16_8_16_256_4_4_4': 512})
  def test_load_clang_jit_simple(self): helper_test_loader('clang.o', 'clang.img', [], {'r_256_256_256_4_4_4': 0})
  def test_load_clang_jit_reloc(self): helper_test_loader('clang_reloc.o', 'clang_reloc.img', [(4, 0, 282, 0)], {'entry': 0})

if __name__ == '__main__':
  unittest.main()
