import unittest
from extra.models import resnet

class TestBert(unittest.TestCase):
  def test_model_load(self):
    model = resnet.ResNeXt50_32X4D()
    model.load_from_pretrained()

if __name__ == '__main__':
  unittest.main()