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
    # print('BOXES:HHHHH', boxes)
    # guard against no boxes via resizing
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    # print('BOXES:POSTT', boxes)
    boxes[:, 2:] += boxes[:, :2]
    # boxes[:, 0::2].clip(min_=0, max_=w)
    # boxes[:, 1::2].clip(min_=0, max_=h)
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
    # print('BOXES:POSTPOST', boxes)
    classes = [obj["category_id"] for obj in anno]
    classes = np.array(classes, dtype=np.int64)

    keypoints = None
    if anno and "keypoints" in anno[0]:
      keypoints = [obj["keypoints"] for obj in anno]
      keypoints = Tensor(keypoints, dtype=dtypes.float32)
      num_keypoints = keypoints.shape[0]
      if num_keypoints:
        keypoints = keypoints.reshape(num_keypoints, -1, 3)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    # print('BOXES:KEPPPOST', boxes)
    classes = classes[keep]

    target = {}
    target["boxes"] = Tensor(boxes)#.realize()
    # print('BOXES:TENSCONV', target["boxes"].numpy())
    target["labels"] = Tensor(classes)#.realize()
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

  # print('resize_boxes boxes.shape:',boxes.shape, original_size, new_size)
  # print(boxes.numpy())


  # boxes = boxes.permute(1,0)
  # print('post permute', boxes.shape)
  # print(boxes.numpy())
  # print(boxes[0].numpy())
  # xmin, ymin, xmax, ymax = boxes[0], boxes[1], boxes[2], boxes[3]
  # xmin = boxes[0]
  bnp = boxes.numpy()
  xmin, ymin, xmax, ymax = Tensor(bnp[:, 0]), Tensor(bnp[:, 1]), \
                          Tensor(bnp[:, 2]), Tensor(bnp[:, 3])

  # xmin, ymin, xmax, ymax = boxes.split(1, dim=1)
  # print('temp t LEN:',t)
  # xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  # print('UNBIND SHAPE_PRE:', xmin.shape, xmin.numpy())

  xmin = xmin * ratio_width
  xmax = xmax * ratio_width
  ymin = ymin * ratio_height
  ymax = ymax * ratio_height
  # print('UNBIND SHAPE_POST:', xmin.shape, xmax.shape, ymin.shape, ymax.shape)
  # ans = Tensor([xmin, ymin, xmax, ymax])


  # ans = Tensor.cat(*(xmin, ymin, xmax, ymax), dim=1)
  ans = Tensor.stack((xmin, ymin, xmax, ymax), dim=1)#.cast(dtypes.float32)
  # print('RESIZE_ANS', ans.shape)
  return ans
import torchvision.transforms.functional as F
import torch
# SIZE = (400,400)
SIZE = (800, 800)
# image_std = torch.tensor([0.229, 0.224, 0.225])
# image_mean = torch.tensor([0.485, 0.456, 0.406])
# def normalize(image):
#         if not image.is_floating_point():
#             raise TypeError(
#                 f"Expected input images to be of floating type (in range [0, 1]), "
#                 f"but found type {image.dtype} instead"
#             )
#         # dtype, device = image.dtype, image.device
#         # mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
#         # std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
#         return (image - image_mean[:, None, None]) / image_std[:, None, None]
image_std = Tensor([0.229, 0.224, 0.225]).reshape(-1,1,1)
image_mean = Tensor([0.485, 0.456, 0.406]).reshape(-1,1,1)
def normalize(x):
  x = x.permute((2,0,1)) / 255.0
  x -= image_mean
  x /= image_std
  return x.realize()
# def iterate(coco, bs=8):
#   for i in range(0, 800, bs):
#   # for i in range(8, len(coco.ids), bs):
#     X, targets= [], []
#     for img_id in coco.ids[i:i+bs]:
#       x,t = coco.__getitem__(img_id)
      

#       xNew = F.resize(x, size=SIZE)
#       xNew = np.array(xNew)
#       X.append(xNew)
#       bbox = t['boxes']
#       # print('ITERATE_PRE_RESIZE', bbox.shape)
#       bbox = resize_boxes(bbox, x.size, SIZE)
#       # print('ITERATE_POST_RESIZE', bbox.shape)
#       t['boxes'] = bbox.realize()
#       targets.append(t)
#     yield Tensor(X), targets
  
def iterate (coco, bs=8):
  
  i = 0
  while(i<800):
    i_sub = 0
    rem =0
    X, targets = [], []
    while(i_sub<bs):
      # print(i_sub)
      x_orig,t = coco.__getitem__(i+i_sub+rem)
      if(t['boxes'].shape[0]==0):
        rem+=1
        # pass
      else:
        # temp = np.array(x_orig)
        # print('TEMP', temp.shape)
        # x_torch = torch.as_tensor(x_orig, dtype=torch.float)
        # print('xTORCH', x_torch.shape)
        # x_torch =x_torch.unsqueeze(0)
        # x_torch = x_torch.to(memory_format=torch.channels_last)
        # x_torch =x_torch.squeeze(0)
        # # x_torch = x_torch.permute(2, 0, 1)
        # print('POST xTORCH', x_torch.shape)
        # x_torch = normalize(x_torch)
        # print(x_torch)
        xNew = F.resize(x_orig, size=SIZE)
        # print('POSt RESIZE')
        
        # xNew = xNew.numpy()

        # xNew = Tensor(np.array(xNew))
        xNew = normalize(Tensor(np.array(xNew)))

        # print('X_NEW',xNew.shape, xNew)
        X.append(xNew)
        bbox = t['boxes']
        # print('ITERATE_PRE_RESIZE', bbox.shape)
        bbox = resize_boxes(bbox, x_orig.size, SIZE)
        # print('ITERATE_POST_RESIZE', bbox.shape)
        t['boxes'] = bbox#.realize()
        targets.append(t)
        i_sub+=1
    # yield Tensor(X), targets
    yield Tensor.stack(X), targets
    i= i+bs+rem

