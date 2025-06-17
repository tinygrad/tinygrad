import unittest
from tinygrad import Tensor, Context
from extra.models import resnet

class TestResnet(unittest.TestCase):
  def test_model_load(self):
    model = resnet.ResNet18()
    model.load_from_pretrained()

    model = resnet.ResNeXt50_32X4D()
    model.load_from_pretrained()

  def test_model_load_no_fc_layer(self):
    model = resnet.ResNet18(num_classes=None)
    model.load_from_pretrained()

    model = resnet.ResNeXt50_32X4D(num_classes=None)
    model.load_from_pretrained()

  def test_model_run(self):
    model = resnet.ResNet18()
    model.load_from_pretrained()
    x = Tensor.empty(1,3,224,224)
    model(x).realize()

  def test_model_fuse(self):
    with Context(DEBUG=0):
      model = resnet.ResNet18()
      model.load_from_pretrained()
      x = Tensor.empty(1,3,224,224)

    out = model.bn1(model.conv1(x)).relu()
    out = out.pad([1,1,1,1]).max_pool2d((3,3), 2)
    #out = out.fuse().contiguous().sequential(model.layer1)
    print(out.fuse().realize())

if __name__ == '__main__':
  unittest.main()