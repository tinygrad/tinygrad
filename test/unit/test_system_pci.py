import unittest
from unittest.mock import Mock, patch
from tinygrad.runtime.support.system import PCIDevice

class TestResizeBar(unittest.TestCase):
  def test_already_maximum(self): self._test_resize(16 << 30, False)
  def test_resize_small_bar(self): self._test_resize(256 << 20, True)

  def _test_resize(self, current_size, writes):
    device = Mock(pcibus='0000:86:00.0')
    device.bar_info.return_value = (0x10000000000, current_size)
    with patch('tinygrad.runtime.support.system.FileIOInterface') as fileio:
      fileio.return_value.read.return_value = '00007f00\n'  # 256 MiB through 16 GiB
      PCIDevice.resize_bar(device, 0)
      device.bar_info.assert_called_once_with(0)
      if writes:
        fileio.return_value.write.assert_called_once_with('14')
        device.bar_info.cache_clear.assert_called_once()
      else:
        fileio.return_value.write.assert_not_called()
        device.bar_info.cache_clear.assert_not_called()

if __name__ == '__main__': unittest.main()
