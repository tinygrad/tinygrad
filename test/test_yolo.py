import io
import unittest
from pathlib import Path

import requests
import numpy as np
from PIL import Image

from tinygrad.tensor import Tensor
from examples.yolov3 import Darknet, show_labels
from extra.utils import fetch

chicken_img = Image.open(str(Path(__file__).parent / 'efficientnet/Chicken.jpg'))
car_img = Image.open(str(Path(__file__).parent / 'efficientnet/car.jpg'))
dog_url = "https://github.com/ayooshkathuria/pytorch-yolo-v3/raw/master/dog-cycle-car.png"
dog_img = Image.open(io.BytesIO(requests.get(dog_url).content))

def infer(model, img):
  img = np.array(img.resize((608, 608)))
  img = img[:,:,::-1].transpose((2,0,1))
  img = img[np.newaxis,:,:,:]/255.0
  prediction = model.forward(Tensor(img.astype(np.float32)))
  return prediction

class TestYOLO(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cfg = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg')
    cls.model = Darknet(cfg)
    print("Loading weights file (237MB). This might take a while…")
    cls.model.load_weights('https://pjreddie.com/media/files/yolov3.weights')

  @classmethod
  def tearDownClass(cls):
    del cls.model

  def test_chicken(self):
    labels = show_labels(infer(self.model, chicken_img), confidence=0.8)
    self.assertEqual(labels, ["bird"])

  def test_car(self):
    labels = show_labels(infer(self.model, car_img), confidence=0.8)
    self.assertEqual(labels, ["car"])

  def test_dog(self):
    labels = show_labels(infer(self.model, dog_img), confidence=0.8)
    self.assertEqual(labels, ["bicycle", "truck", "dog"])

if __name__ == '__main__':
  unittest.main()
