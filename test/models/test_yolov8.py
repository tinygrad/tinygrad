import pathlib
import unittest

import numpy as np
from PIL import Image
import cv2

from extra.models.yolov8 import YOLOv8, preprocess, postprocess
from extra.utils import fetch

def _load_labels():
  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')
  return class_labels.decode('utf-8').split('\n')[:-1] # last is newline only

_LABELS = _load_labels()

def _infer(model: YOLOv8, img):
  image = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
  img_pre = preprocess(image)
  # run the net
  out = model(img_pre)
  pred = postprocess(preds=out, img=img_pre, orig_imgs=image)
  return _LABELS[int(pred[0][0,5])] # label of the first predictions bounding box

cap = cv2.VideoCapture(str(pathlib.Path(__file__).parent / 'efficientnet/Chicken.jpg'))
ret, chicken_img = cap.read()
cap = cv2.VideoCapture(str(pathlib.Path(__file__).parent / 'efficientnet/car.jpg'))
ret, car_img = cap.read()

class TestEfficientNet(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model = YOLOv8('s', len(_LABELS)) # 'n', 's', 'm', 'l', or 'x'
    cls.model.load_from_pretrained()

  @classmethod
  def tearDownClass(cls):
    del cls.model

  def test_chicken(self):
    label = _infer(self.model, chicken_img)
    self.assertEqual(label, "bird")

  def test_car(self):
    print('car')
    label = _infer(self.model, car_img)
    self.assertEqual(label, "car")

if __name__ == '__main__':
  unittest.main()
