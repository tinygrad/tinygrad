import copy
import os
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple, Union

from tinygrad import Tensor, dtypes

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
class ConvertCocoPolysToMask(object):
  def __init__(self, filter_iscrowd=True):
    self.filter_iscrowd = filter_iscrowd

  def __call__(self, image, target):
    w, h = image.size

    image_id = target["image_id"]
    image_id = Tensor([image_id])

    anno = target["annotations"]

    if self.filter_iscrowd:
      anno = [obj for obj in anno if obj['iscrowd'] == 0]

    boxes = [obj["bbox"] for obj in anno]
    # guard against no boxes via resizing
    boxes = Tensor(boxes, dtype=dtypes.float).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp(min=0, max=w)
    boxes[:, 1::2].clamp(min=0, max=h)

    classes = [obj["category_id"] for obj in anno]
    classes = Tensor(classes, dtype=dtypes.int64)

    keypoints = None
    if anno and "keypoints" in anno[0]:
      keypoints = [obj["keypoints"] for obj in anno]
      keypoints = Tensor(keypoints, dtype=dtypes.float)
      num_keypoints = keypoints.shape[0]
      if num_keypoints:
        keypoints = keypoints.reshape(num_keypoints, -1, 3)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    classes = classes[keep]

    target = {}
    target["boxes"] = boxes
    target["labels"] = classes
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
    target = self._load_target(id)

    # if self.transforms is not None:
    #   img, target = self.transforms(img, target)

    image_id = self.ids[index]
    target = dict(image_id=image_id, annotations=target)
    if self._transforms is not None:
      img, target = self._transforms(img, target)
    return img, target
  
def get_openimages(name, root, image_set, transforms=None):
  PATHS = {
      "train": os.path.join(root, "train"),
      "val":   os.path.join(root, "validation"),
  }

  t = [ConvertCocoPolysToMask(filter_iscrowd=False)]

  # if transforms is not None:
  #     t.append(transforms)
  # transforms = T.Compose(t)

  img_folder = os.path.join(PATHS[image_set], "data")
  ann_file = os.path.join(PATHS[image_set], "labels", f"{name}.json")

  dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

  return dataset
def iterate(coco, bs=8):
  for i in range(0, len(coco.ids), bs):
    X, targets= [], []
    for img_id in coco.ids[i:i+bs]:
      x,t = coco.__getitem__(img_id)
      X.append(x)
      targets.append(t)
    yield X, targets