import io
import unittest
from pathlib import Path

import cv2
import numpy as np

from tinygrad.tensor import Tensor
from extra.utils import fetch
from examples.yolov3 import Darknet, infer, show_labels

chicken_img = cv2.imread(str(Path(__file__).parent / 'efficientnet/Chicken.jpg'))
car_img = cv2.imread(str(Path(__file__).parent / 'efficientnet/car.jpg'))
dog_url = "https://github.com/ayooshkathuria/pytorch-yolo-v3/raw/master/dog-cycle-car.png"
dog_img = cv2.imdecode(np.frombuffer(io.BytesIO(fetch(dog_url)).read(), np.uint8), 1)

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
    labels = show_labels(infer(self.model, chicken_img))
    self.assertEqual(", ".join(labels), "person, bird")

  def test_car(self):
    labels = show_labels(infer(self.model, car_img))
    self.assertEqual(", ".join(labels), "car")

  def test_dog(self):
    labels = show_labels(infer(self.model, dog_img))
    self.assertEqual(", ".join(labels), "bicycle, truck, dog")

if __name__ == '__main__':
  unittest.main()
