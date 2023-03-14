#!/usr/bin/env python
import io
import unittest
from extra.utils import fetch
from PIL import Image

class TestUtils(unittest.TestCase):  
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

if __name__ == '__main__':
  unittest.main()