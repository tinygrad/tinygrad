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
    # w, h = image.size
    h, w = image.size

    image_id = target["image_id"]
    image_id = Tensor([image_id], requires_grad=False)

    anno = target["annotations"]

    if self.filter_iscrowd:
      anno = [obj for obj in anno if obj['iscrowd'] == 0]

    boxes = [obj["bbox"] for obj in anno]
    # print('BOXES:HHHHH', boxes)
    # guard against no boxes via resizing
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    # print('BOXES:POSTT', boxes)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
    # print('BOXES:POSTPOST', boxes)
    classes = [obj["category_id"] for obj in anno]
    classes = np.array(classes, dtype=np.int64)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    # print('BOXES:KEPPPOST', boxes)
    classes = classes[keep]

    target = {}
    target["boxes"] = Tensor(boxes, requires_grad=False)#.realize()
    # print('BOXES:TENSCONV', target["boxes"].numpy())
    target["labels"] = Tensor(classes, requires_grad=False)#.realize()
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
    # print('iterate orig size', orig_size, id)
    # sys.exit()
    target = self._load_target(id)

    # if self.transforms is not None:
    #   img, target = self.transforms(img, target)

    image_id = self.ids[index]
    target = dict(image_id=image_id, annotations=target)
    if self._transforms is not None:
      # print('HIT')
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

  # print('resize_boxes boxes.shape:',boxes.shape, original_size, new_size)
  # print(boxes.numpy())

  # bnp = boxes.numpy()
  # xmin, ymin, xmax, ymax = Tensor(bnp[:, 0]), Tensor(bnp[:, 1]), \
  #                         Tensor(bnp[:, 2]), Tensor(bnp[:, 3])
  xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], \
                          boxes[:, 2], boxes[:, 3]
  # print('temp t LEN:',t)
  # print('UNBIND SHAPE_PRE:', xmin.shape, xmin.numpy())

  xmin = xmin * ratio_width
  xmax = xmax * ratio_width
  ymin = ymin * ratio_height
  ymax = ymax * ratio_height
  # print('UNBIND SHAPE_POST:', xmin.shape, xmax.shape, ymin.shape, ymax.shape)
  ans = Tensor.stack((xmin, ymin, xmax, ymax), dim=1)#.cast(dtypes.float32)
  # print('RESIZE_ANS', ans.shape)
  ans.requires_grad = False
  return ans

import torchvision.transforms.functional as F
# SIZE = (400,400)
SIZE = (800, 800)

image_std = Tensor([0.229, 0.224, 0.225]).reshape(-1,1,1)
image_mean = Tensor([0.485, 0.456, 0.406]).reshape(-1,1,1)
def normalize(x):
  x = x.permute((2,0,1)) / 255.0
  # x = x.permute((2,1,0)) / 255.0
  x -= image_mean
  x /= image_std
  return x#.realize()
  
def iterate(coco, bs=8):
  i = 0
  i_sub = 0
  rem =0
  while(i+bs+rem<len(coco.ids)):
    # print('iterate', i)
    i_sub = 0
    rem =0
    X, target_boxes, target_labels, target_boxes_padded, target_labels_padded = [], [], [], [], []
    while(i_sub<bs and i+bs+rem<len(coco.ids)):
      # print(i_sub)
      x_orig,t = coco.__getitem__(i+i_sub+rem)
      # print('X_ORIG_SIZE', x_orig.size)
      # print('DATLOAD_ITER', t['boxes'].shape)

      # Training not done on empty targets
      if(t['boxes'].shape[0]<=0):
        rem+=1
      else:
        # xNew_pre = normalize(Tensor(np.array(x_orig)))
        xNew_tor = F.resize(x_orig, size=SIZE)
        # print('X_NEW', xNew.shape)
        xNew = normalize(Tensor(np.array(xNew_tor)))

        # print('X_MEAN_NORM',xNew.shape, xNew.mean().numpy())
        X.append(xNew)
        bbox = t['boxes']
        # print('ITERATE_PRE_RESIZE', bbox.shape)
        # bbox = resize_boxes(bbox, (x_orig.size[1],x_orig.size[0]), SIZE)
        bbox = resize_boxes(bbox, x_orig.size[::-1], SIZE)
        # print('ITERATE_POST_RESIZE', bbox.shape)
        t['boxes'] = bbox.realize()
        # max_pad = 120087
        max_pad = 500
        n = t['boxes'].shape[0]
        boxes_padded = t['boxes'].pad((((0,max_pad-n), None)),0)
        labels_padded = t['labels'].reshape(-1,1).pad((((0,max_pad-n), None)),-1).reshape(-1)
        # print('ITERATE', xNew.shape, t['boxes'].shape, t['labels'].shape)
        # print('ITERATE_PADDED', xNew.shape, boxes_padded.shape, labels_padded.shape)
        target_boxes.append(t['boxes'])
        target_labels.append(t['labels'].realize())
        # print('lABEL LOAD CHEK', target_labels[-1].shape)
        # print('PADDING LOGIC', labels_padded.numpy())
        # print(boxes_padded.numpy())
        target_boxes_padded.append(boxes_padded)
        target_labels_padded.append(labels_padded)
        i_sub+=1

    yield Tensor.stack(X), target_boxes, target_labels, Tensor.stack(target_boxes_padded), Tensor.stack(target_labels_padded)
    i= i+bs+rem

def iterate_val(coco, bs=8):
  for i in range(0, len(coco.ids)-bs, bs):
  # for i in range(0, 10, bs):
    X, targets = [], []
    for img_id in coco.ids[i:i+bs]:
      x_orig, t = coco.__getitem__(img_id)
      xNew_tor = F.resize(x_orig, size=SIZE)
      xNew = normalize(Tensor(np.array(xNew_tor)))
      X.append(xNew)
      # bbox = t['boxes']
      # t['boxes'] = resize_boxes(bbox, x_orig.size, SIZE).numpy()
      # t['labels'] = t['labels'].numpy()
      # t['image_id'] = t['image_id'].item()
      tNew = {'image_size' : t['image_size'][::-1], 
              'image_id' : t['image_id'].item()}
      targets.append(tNew)
    yield Tensor.stack(X), targets