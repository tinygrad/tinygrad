import copy,sys
import os
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from tinygrad import Tensor, dtypes

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
class ConvertCocoPolysToMask(object):
  def __init__(self, filter_iscrowd=True):
    self.filter_iscrowd = filter_iscrowd

  def __call__(self, image, target):
    h, w = image.size

    image_id = target["image_id"]
    image_id = Tensor([image_id])

    anno = target["annotations"]

    if self.filter_iscrowd:
      anno = [obj for obj in anno if obj['iscrowd'] == 0]

    boxes = [obj["bbox"] for obj in anno]
    print('BOXES:HHHHH', boxes)
    # guard against no boxes via resizing
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    print('BOXES:POSTT', boxes)
    boxes[:, 2:] += boxes[:, :2]
    # boxes[:, 0::2].clip(min_=0, max_=w)
    # boxes[:, 1::2].clip(min_=0, max_=h)
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
    print('BOXES:POSTPOST', boxes)
    classes = [obj["category_id"] for obj in anno]
    classes = np.array(classes, dtype=np.int64)

    keypoints = None
    if anno and "keypoints" in anno[0]:
      keypoints = [obj["keypoints"] for obj in anno]
      keypoints = Tensor(keypoints, dtype=dtypes.float)
      num_keypoints = keypoints.shape[0]
      if num_keypoints:
        keypoints = keypoints.reshape(num_keypoints, -1, 3)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    print('BOXES:KEPPPOST', boxes)
    classes = classes[keep]

    target = {}
    target["boxes"] = Tensor(boxes)
    print('BOXES:TENSCONV', target["boxes"].numpy())
    target["labels"] = Tensor(classes)
    target["image_id"] = image_id

    # for conversion to coco api
    area = Tensor([obj["area"] for obj in anno])
    iscrowd = Tensor([obj["iscrowd"] for obj in anno])
    target["area"] = area
    target["iscrowd"] = iscrowd

    return image, target

class CocoDetection:
  def __init__(self, img_folder, ann_file, transforms=None):
    self.root = img_folder
    self.coco = COCO(ann_file)
    self.ids = list(sorted(self.coco.imgs.keys()))
    # super(CocoDetection, self).__init__(img_folder, ann_file)
    self._transforms = transforms
  def _load_image(self, id: int) -> Image.Image:
    path = self.coco.loadImgs(id)[0]["file_name"]
    return Image.open(os.path.join(self.root, path)).convert("RGB")

  def _load_target(self, id: int) -> List[Any]:
    return self.coco.loadAnns(self.coco.getAnnIds(id))
  def __len__(self) -> int:
    return len(self.ids)
  def __getitem__(self, index):

    id = self.ids[index]
    img = self._load_image(id)
    orig_size = img.size
    # sys.exit()
    target = self._load_target(id)

    # if self.transforms is not None:
    #   img, target = self.transforms(img, target)

    image_id = self.ids[index]
    target = dict(image_id=image_id, annotations=target)
    if self._transforms is not None:
      img, target = self._transforms(img, target)
      target['image_size'] = orig_size
    return img, target
  
def get_openimages(name, root, image_set, transforms=None):
  PATHS = {
      "train": os.path.join(root, "train"),
      "val":   os.path.join(root, "validation"),
  }

  t = ConvertCocoPolysToMask(filter_iscrowd=False)

  # if transforms is not None:
  #     t.append(transforms)
  # transforms = T.Compose(t)

  img_folder = os.path.join(PATHS[image_set], "data")
  ann_file = os.path.join(PATHS[image_set], "labels", f"{name}.json")

  dataset = CocoDetection(img_folder, ann_file, transforms=t)

  return dataset

def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
  ratios = [
      Tensor(s, dtype=dtypes.float32) / Tensor(s_orig, dtype=dtypes.float32)
      for s, s_orig in zip(new_size, original_size)
  ]
  ratio_height, ratio_width = ratios
  print('resize_boxes boxes.shape:',boxes.shape, original_size, new_size)
  print(boxes.numpy())
  boxes = boxes.permute(1,0)
  print('post permute', boxes.shape)
  print(boxes.numpy())
  print(boxes[0].numpy())
  xmin, ymin, xmax, ymax = boxes[0], boxes[1], boxes[2], boxes[3]
  xmin = boxes[0]
  # xmin, ymin, xmax, ymax = boxes.split(1, dim=1)
  # print('temp t LEN:',t)
  # xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  print('UNBIND SHAPE_PRE:', xmin.shape, xmin.numpy())

  xmin = xmin * ratio_width
  xmax = xmax * ratio_width
  ymin = ymin * ratio_height
  ymax = ymax * ratio_height
  print('UNBIND SHAPE_POST:', xmin.shape)
  ans = Tensor.cat(*(xmin, ymin, xmax, ymax), dim=1)
  print('RESIZE_ANS', ans.shape)
  return ans
import torchvision.transforms.functional as F

def iterate(coco, bs=8):
  for i in range(8, len(coco.ids), bs):
    X, targets= [], []
    for img_id in coco.ids[i:i+bs]:
      x,t = coco.__getitem__(img_id)
      

      xNew = F.resize(x, size=(800, 800))
      xNew = np.array(xNew)
      X.append(xNew)
      bbox = t['boxes']
      print('ITERATE_PRE_RESIZE', bbox.shape)
      bbox = resize_boxes(bbox, x.size, (800,800))
      print('ITERATE_POST_RESIZE', bbox.shape)
      t['boxes'] = bbox
      targets.append(t)
    yield Tensor(X), targets