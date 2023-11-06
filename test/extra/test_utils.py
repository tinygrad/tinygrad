#!/usr/bin/env python
import io, unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

import torch
import numpy as np
from tinygrad.helpers import CI
from extra.utils import fetch, temp, download_file
from tinygrad.nn.state import torch_load
from PIL import Image

@unittest.skipIf(CI, "no internet tests in CI")
class TestFetch(unittest.TestCase):
  def test_fetch_bad_http(self):
    self.assertRaises(AssertionError, fetch, 'http://httpstat.us/500')
    self.assertRaises(AssertionError, fetch, 'http://httpstat.us/404')
    self.assertRaises(AssertionError, fetch, 'http://httpstat.us/400')

  def test_fetch_small(self):
    assert(len(fetch('https://google.com'))>0)

  def test_fetch_img(self):
    img = fetch("https://media.istockphoto.com/photos/hen-picture-id831791190")
    pimg = Image.open(io.BytesIO(img))
    assert pimg.size == (705, 1024)

class TestFetchRelative(unittest.TestCase):
  def setUp(self):
    self.working_dir = os.getcwd()
    self.tempdir = tempfile.TemporaryDirectory()
    os.chdir(self.tempdir.name)
    with open('test_file.txt', 'x') as f:
      f.write("12345")

  def tearDown(self):
    os.chdir(self.working_dir)
    self.tempdir.cleanup()

  #test ./
  def test_fetch_relative_dotslash(self):
    self.assertEqual(b'12345', fetch("./test_file.txt"))

  #test ../
  def test_fetch_relative_dotdotslash(self):
    os.mkdir('test_file_path')
    os.chdir('test_file_path')
    self.assertEqual(b'12345', fetch("../test_file.txt"))

class TestDownloadFile(unittest.TestCase):
  def setUp(self):
    from pathlib import Path
    self.test_file = Path(temp("test_download_file/test_file.txt"))

  def tearDown(self):
    os.remove(self.test_file)
    os.removedirs(self.test_file.parent)

  @patch('requests.get')
  def test_download_file_with_mkdir(self, mock_requests):
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b'1234', b'5678']
    mock_response.status_code = 200
    mock_response.headers = {'content-length': '8'}
    mock_requests.return_value = mock_response
    self.assertFalse(self.test_file.parent.exists())
    download_file("https://www.mock.com/fake.txt", self.test_file, skip_if_exists=False)
    self.assertTrue(self.test_file.parent.exists())
    self.assertTrue(self.test_file.is_file())
    self.assertEqual('12345678', self.test_file.read_text())

class TestUtils(unittest.TestCase):
  def test_fake_torch_load_zipped(self): self._test_fake_torch_load_zipped()
  def test_fake_torch_load_zipped_float16(self): self._test_fake_torch_load_zipped(isfloat16=True)
  def _test_fake_torch_load_zipped(self, isfloat16=False):
    class LayerWithOffset(torch.nn.Module):
      def __init__(self):
        super(LayerWithOffset, self).__init__()
        d = torch.randn(16)
        self.param1 = torch.nn.Parameter(
          d.as_strided([2, 2], [1, 2], storage_offset=5)
        )
        self.param2 = torch.nn.Parameter(
          d.as_strided([2, 2], [1, 2], storage_offset=4)
        )

    model = torch.nn.Sequential(
      torch.nn.Linear(4, 8),
      torch.nn.Linear(8, 3),
      LayerWithOffset()
    )
    if isfloat16: model = model.half()

    path = temp(f"test_load_{isfloat16}.pt")
    torch.save(model.state_dict(), path)
    model2 = torch_load(path)

    for name, a in model.state_dict().items():
      b = model2[name]
      a, b = a.numpy(), b.numpy()
      assert a.shape == b.shape
      assert a.dtype == b.dtype
      assert np.array_equal(a, b)
if __name__ == '__main__':
  unittest.main()
