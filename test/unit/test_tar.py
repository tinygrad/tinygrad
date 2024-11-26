import unittest, tarfile, io, os, pathlib
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import tar_extract

class TestTarExtractPAX(unittest.TestCase):
  tar_format = tarfile.PAX_FORMAT
  max_link_len = 1000_000
  test_files = {
    'a/file1.txt': b'Hello, World!',
    'a/b/file2.bin': b'\x00\x01\x02\x03\x04',
    'empty_file.txt': b'',
    '512file': b'a' * 512,
    'long_file': b'some data' * 100,
    'very' * 15 + '/' + 'very' * 15 + '_long_filename.txt': b'Hello, World!!',
    'very' * 200 + '_long_filename.txt': b'Hello, World!!!',
  }

  def create_tar_tensor(self):
    fobj = io.BytesIO()
    test_dirs = set(os.path.dirname(k) for k in self.test_files.keys()).difference({ '' })
    with tarfile.open(fileobj=fobj, mode='w', format=self.tar_format) as tar:
      for dirname in test_dirs:
        dir_info = tarfile.TarInfo(name=dirname)
        dir_info.type = tarfile.DIRTYPE
        tar.addfile(dir_info)

      for filename, content in self.test_files.items():
        file_info = tarfile.TarInfo(name=filename)
        file_info.size = len(content)
        tar.addfile(file_info, io.BytesIO(content))

        if len(filename) < self.max_link_len:
          link_info = tarfile.TarInfo(name=filename + '.lnk')
          link_info.type = tarfile.SYMTYPE
          link_info.linkname = filename
          tar.addfile(link_info)
    return Tensor(fobj.getvalue())

  def test_tar_extract_returns_dict(self):
    result = tar_extract(self.create_tar_tensor())
    self.assertIsInstance(result, dict)

  def test_tar_extract_correct_keys(self):
    result = tar_extract(self.create_tar_tensor())
    self.assertEqual(set(result.keys()), set(self.test_files.keys()))

  def test_tar_extract_content_size(self):
    result = tar_extract(self.create_tar_tensor())
    for filename, content in self.test_files.items():
      self.assertEqual(len(result[filename]), len(content))

  def test_tar_extract_content_values(self):
    result = tar_extract(self.create_tar_tensor())
    for filename, content in self.test_files.items():
      np.testing.assert_array_equal(result[filename].numpy(), np.frombuffer(content, dtype=np.uint8))

  def test_tar_extract_empty_file(self):
    result = tar_extract(self.create_tar_tensor())
    self.assertEqual(len(result['empty_file.txt']), 0)

  def test_tar_extract_non_existent_file(self):
    with self.assertRaises(FileNotFoundError):
      tar_extract(Tensor(pathlib.Path('non_existent_file.tar')))

  def test_tar_extract_invalid_file(self):
    with self.assertRaises(tarfile.ReadError):
      tar_extract(Tensor(b'This is not a valid tar file'))

  def test_tar_extract_invalid_file_long(self):
    with self.assertRaises(tarfile.HeaderError):
      tar_extract(Tensor(b'This is not a valid tar file'*100))

class TestTarExtractUSTAR(TestTarExtractPAX):
  tar_format = tarfile.USTAR_FORMAT
  max_link_len = 100
  test_files = {k: v for k, v in TestTarExtractPAX.test_files.items() if len(k) < 256}

class TestTarExtractGNU(TestTarExtractPAX):
  tar_format = tarfile.GNU_FORMAT

if __name__ == '__main__':
  unittest.main()
