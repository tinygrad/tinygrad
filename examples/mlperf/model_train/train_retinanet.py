from tinygrad.tensor import Tensor
from tinygrad.nn import optim
import numpy as np
from models.resnet import ResNeXt50_32X4D
from models.retinanet import RetinaNet
from typing import List

# Compute the IoU between all <box,query_box> pairs
# boxes: (N, 4), query_boxes: (K, 4), returns: (N, K)
# TODO: make this faster.
def compute_overlap(boxes: np.ndarray, query_boxes: np.ndarray):
  overlaps = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float32)
  for k in range(query_boxes.shape[0]):
    query_box_area = ((query_boxes[k, 2] - query_boxes[k, 0]) * (query_boxes[k, 3] - query_boxes[k, 1]))
    for n in range(boxes.shape[0]):
      intersection_width = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]))
      if intersection_width <= 0: continue
      intersection_height = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]))
      if intersection_height <= 0: continue
      box_area = np.float32((boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1]))
      union_area = box_area + query_box_area - intersection_width * intersection_height
      overlaps[n, k] = intersection_width * intersection_height / union_area
  return overlaps

# anchors: (batch_size, N, 4). Each anchor is (x1, y1, x2, y2).
def compute_batch_targets(pred_anchors: np.ndarray, annotations: List, num_classes, negative_overlap=0.4, positive_overlap=0.5):
  batch_size = len(annotations)
  regression_targets = np.zeros((batch_size, pred_anchors.shape[0], 4), dtype=np.float32)
  classif_targets = np.zeros((batch_size, pred_anchors.shape[0], num_classes), dtype=np.float32)
  for index, ann in enumerate(annotations):
    if not ann['boxes'].shape[0]: continue
    overlaps = compute_overlap(pred_anchors.astype(np.float32), ann['boxes'].astype(np.float32))  # TODO: fix segmentation fault
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]
    positive_indices = max_overlaps >= positive_overlap
    # TODO: exclude ignore_indices from training
    # ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices
    regression_targets[index, :, :] = ann['boxes'][argmax_overlaps_inds, :]
    classif_targets[index, positive_indices, ann['labels'][argmax_overlaps_inds[positive_indices]].astype(int)] = 1
  return np.concatenate((regression_targets, classif_targets), axis=2)

def train_retinanet():
  mdl = RetinaNet(ResNeXt50_32X4D(), num_classes=91, num_anchors=9)

  input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
  def input_fixup(x):
    x = x.permute([0,3,1,2]) / 255.0
    x -= input_mean
    x /= input_std
    return x

  from datasets.openimages import openimages, iterate
  from pycocotools.coco import COCO
  coco = COCO(openimages())

  params = optim.get_parameters(mdl)
  optimizer = optim.SGD(params)
  n, BS = 0, 1
  for epoch in range(1, 11):
    for x, annotations in iterate(coco, BS):
      dat = Tensor(x.astype(np.float32))
      outs = mdl(input_fixup(dat)).numpy()
      anchors = outs[:, :, :4]
      classif = outs[:, :, 4:]

      targets = compute_batch_targets(anchors, annotations, mdl.num_classes)

      # TODO: compute L1 loss between anchors and targets[:, :, :4] + focal loss between classif and targets[:, :, 4:]

      # TODO: ...then loss.backward() and optimizer.zero_grad()

      # TODO: evaluate results on test set with COCOeval

      n += len(annotations)
      print(f"[Epoch {epoch}, {n}/{len(coco.imgs)}]")


if __name__ == "__main__":
  Tensor.training = True
  print('training retinanet')
  train_retinanet()
