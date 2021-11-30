import ast
import pathlib
import sys
import unittest

import numpy as np
from PIL import Image

from models.efficientnet import EfficientNet
from tinygrad.tensor import Tensor

def _load_labels():
  labels_filename = pathlib.Path(__file__).parent / 'efficientnet/imagenet1000_clsidx_to_labels.txt'
  return ast.literal_eval(labels_filename.read_text())


_LABELS = _load_labels()

def _infer(model: EfficientNet, img):
  # preprocess image
  aspect_ratio = img.size[0] / img.size[1]
  img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))

  img = np.array(img)
  y0, x0 =(np.asarray(img.shape)[:2] - 224) // 2
  img = img[y0: y0 + 224, x0: x0 + 224]

  # low level preprocess
  img = np.moveaxis(img, [2, 0, 1], [0, 1, 2])
  img = img.astype(np.float32)[:3].reshape(1, 3, 224, 224)
  img /= 255.0
  img -= np.array([0.485, 0.456, 0.406]).reshape((1, -1, 1, 1))
  img /= np.array([0.229, 0.224, 0.225]).reshape((1, -1, 1, 1))

  # run the net
  out = model.forward(Tensor(img)).cpu()
  class_id = np.argmax(out.data)
  return _LABELS[np.argmax(out.data)]


class TestEfficientNet(unittest.TestCase):
  def test_chicken(self):
    chicken_img = Image.open(pathlib.Path(__file__).parent / 'efficientnet/Chicken.jpg')
    model = EfficientNet(number=0)
    model.load_from_pretrained()
    label = _infer(model, chicken_img)
    self.assertEqual(label, "hen", f"Expected hen but got {label} for number=0")


if __name__ == '__main__':
  unittest.main()
