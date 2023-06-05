from tinygrad.tensor import Tensor
from tinygrad.nn import optim
import numpy as np
from models.resnet import ResNeXt50_32X4D
from models.retinanet import RetinaNet
from typing import List
from examples.mlperf.train_retinanet.utils.bbox_transformations import bbox_transform, resize_box_based_on_new_image_size
from extra.training import smooth_l1_loss, focal_loss
import random

NUM_CLASSES = 264
INPUT_IMG_SHAPE = (800, 800, 3)


# Compute the IoU between all <box,query_box> pairs. A box is represented as (x1, y1, x2, y2).
# boxes: (N, 4), query_boxes: (K, 4), returns: (N, K)
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

# anchors: (A, 4). Every anchor is represented as (x1, y1, x2, y2). The anchors are always the same.
# `annotations` is a list of length BS. One annotation for every image in the batch. One annotation can have multiple labeled boxes.
def compute_batch_targets(anchors: np.ndarray, annotations: List, negative_overlap=0.4, positive_overlap=0.5):
  batch_size = len(annotations)
  regression_targets = np.zeros((batch_size, len(anchors), 4), dtype=np.float32)
  classif_targets = np.zeros((batch_size, len(anchors), NUM_CLASSES), dtype=np.float32)
  positive_anchors_mask = np.zeros((batch_size, len(anchors)), dtype=bool)  # Anchors that shouldn't be used for training.
  negative_anchors_mask = np.zeros((batch_size, len(anchors)), dtype=bool)  # Anchors that should only be used for learning classification, not bbox regression.
  for i in range(batch_size):
    ann_boxes, ann_labels = annotations[i]['boxes'], annotations[i]['labels']
    assert len(ann_boxes) > 0
    ann_boxes = np.array([resize_box_based_on_new_image_size(box, img_old_size=annotations[i]['image_size'], img_new_size=INPUT_IMG_SHAPE[:2]) for box in ann_boxes])
    overlaps = compute_overlap(anchors, ann_boxes)
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    positive_anchors_idxs = max_overlaps >= positive_overlap  # The object might be in one of these anchors, so we'll learn to regress to them and to classify the label as such.
    negative_anchors_idxs = (max_overlaps < negative_overlap) & ~positive_anchors_idxs  # There are definitely no objects in these anchors, so we use them for learning to classify object absence (i.e., all classes to 0).

    regression_targets[i, :, :] = bbox_transform(anchors, ann_boxes[argmax_overlaps_inds, :])  # Calculate the spatial transformations from the anchors to the GT boxes.
    classif_targets[i, positive_anchors_idxs, ann_labels[argmax_overlaps_inds[positive_anchors_idxs]].astype(int)] = 1
    positive_anchors_mask[i, positive_anchors_idxs] = True
    negative_anchors_mask[i, negative_anchors_idxs] = True
  return {"regression_targets": regression_targets, "classif_targets": classif_targets, "positive_anchors_mask": positive_anchors_mask, "negative_anchors_mask": negative_anchors_mask}

def train_retinanet():
  model = RetinaNet(ResNeXt50_32X4D(), num_classes=NUM_CLASSES, num_anchors=9)
  model.backbone.body.fc = None  # it's not used by RetinaNet and would break the training loop because of .requires_grad

  anchors = model.anchor_gen(input_size=INPUT_IMG_SHAPE[:2])
  anchors = np.array([a for sublist in anchors for a in sublist])
  print('Number of anchors:', len(anchors))

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

  optimizer = optim.SGD(optim.get_parameters(model), lr=0.001, weight_decay=0.0001, momentum=0.9)
  n, BS = 0, 4
  for epoch in range(1, 11):
    Tensor.training = True
    for imgs, annotations in iterate(coco, BS):
      dat = Tensor(imgs.astype(np.float32))

      optimizer.zero_grad()
      outs = model(input_fixup(dat))
      regression_preds, classif_preds = outs[:, :, :4], outs[:, :, 4:]

      target = compute_batch_targets(anchors, annotations, model.num_classes)
      regression_targets, classif_targets = Tensor(target['regression_targets']), Tensor(target['classif_targets'])

      # TODO: something's still wrong with the loss calculation.
      regression_mask = Tensor(np.repeat(target['positive_anchors_mask'][:, :, np.newaxis], repeats=4, axis=2))
      regression_preds = regression_mask.where(regression_preds, regression_targets)
      regression_loss = smooth_l1_loss(regression_preds, regression_targets, reduction='sum')

      classif_mask = Tensor(np.repeat((target['positive_anchors_mask'] + target['negative_anchors_mask'])[:, :, np.newaxis], repeats=NUM_CLASSES, axis=2))
      classif_preds = classif_mask.where(classif_preds, classif_targets)
      classif_loss = focal_loss(classif_preds, classif_targets, reduction='sum')

      loss = regression_loss + classif_loss  # alternative option: lambda * regression_loss + classif_loss
      loss = loss / max(1, np.count_nonzero(target['positive_anchors_mask']))
      loss.backward()

      optimizer.step()

      n += len(annotations)
      print(f"[Epoch {epoch}, {n}/{len(coco.imgs)}]")
      print('Training loss for last batch:', loss.numpy())

    # At the end of each epoch:
    # TODO: evaluate results on test set with COCOeval


if __name__ == "__main__":
  Tensor.manual_seed(0)
  np.random.seed(0)
  random.seed(0)
  print('training retinanet')
  train_retinanet()
