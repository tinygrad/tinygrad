import numpy as np
from extra.utils import fetch, download_file, get_child
from examples.yolov8 import YOLOv8, get_variant_multiples, preprocess, postprocess, label_predictions
from pathlib import Path
import torch
import unittest
import io
from tinygrad.nn import Tensor
import cv2


class TestYOLOv8(unittest.TestCase):
  
  def setUp(self):
    self.yolo_variants = ['n', 's', 'm', 'l', 'x']

  def test_all_load_weights(self):
    for variant in self.yolo_variants:
      weights_location = Path(__file__).parent.parent.parent / "weights" / f'yolov8{variant}.pt'
      download_file(f'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8{variant}.pt', weights_location)
      
      depth, width, ratio = get_variant_multiples(variant) 
      TinyYolov8 = YOLOv8(w=width, r=ratio, d=depth, num_classes=80) 
      TinyYolov8.load_weights(weights_location, variant)
      
      state_dict = torch.load(weights_location)
      weights = state_dict['model'].state_dict().items()
      all_trainable_weights = TinyYolov8.return_all_trainable_modules()
      
      for k, v in weights:
        k = k.split('.')
        for i in all_trainable_weights:
          if int(k[1]) in i and k[-1] != "num_batches_tracked":
            child_key = '.'.join(k[2:]) if k[2] != 'm' else 'bottleneck.' + '.'.join(k[3:])
            obj = get_child(i[1], child_key)
            weight = v.numpy().astype(np.float32)
            assert obj.shape == weight.shape
            np.testing.assert_allclose(v.numpy(), obj.cpu().numpy(), atol=5e-4, rtol=1e-5)

  def test_predictions(self):
    test_image_urls = ['https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg', 'https://www.aljazeera.com/wp-content/uploads/2022/10/2022-04-28T192650Z_1186456067_UP1EI4S1I0P14_RTRMADP_3_SOCCER-ENGLAND-MUN-CHE-REPORT.jpg']
    variant = 'n'
    weights_location = Path(__file__).parent.parent.parent / "weights" / f'yolov8{variant}.pt'
    depth, width, ratio = get_variant_multiples(variant) 
    TinyYolov8 = YOLOv8(w=width, r=ratio, d=depth, num_classes=80) 
    TinyYolov8.load_weights(weights_location, variant)
    
    for i in range(len(test_image_urls)):
      img_stream = io.BytesIO(fetch(test_image_urls[i]))
      img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
      test_image = preprocess([img])
      predictions = TinyYolov8.forward(Tensor(test_image.astype(np.float32)))
      post_predictions = postprocess(preds=predictions, img=test_image, orig_imgs=[img])
      labels = label_predictions(post_predictions)
      assert labels == {5: 1, 0: 4, 11: 1} if i == 0 else labels == {0: 13, 29: 1, 32: 1}


if __name__ == '__main__':
    unittest.main()
    