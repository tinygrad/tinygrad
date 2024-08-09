import unittest, os
from tinygrad.runtime.support.elf import elf_loader

class TestZeroCopy(unittest.TestCase):
  def test_load_amd_hip(self):
    with open(f'{os.path.dirname(__file__)}/../extra/datasets/hip.elf', 'rb') as fd: blob = fd.read()
    image, sections, relocs, exports = elf_loader(blob)
    with open(f'{os.path.dirname(__file__)}/../extra/datasets/hip.img', 'rb') as fd: image_expected = fd.read()
    relocs_expected = []
    exports_expected = {'r_32_16_8_16_256_4_4_4': 11264}
    assert bytes(image) == image_expected, f'Image missmatch\nexpected: {image_expected!r}\ngot: {bytes(image)!r}'
    assert relocs == relocs_expected, f'Relocs missmatch\nexpected: {[]}\ngot: {relocs_expected}'
    assert exports == exports_expected, f'Exports missmatch\nexpected: {exports_expected}\ngot: {exports}'

if __name__ == '__main__':
  unittest.main()
