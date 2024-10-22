# https://github.com/mlcommons/training/blob/cdd928d4596c142c15a7d86b2eeadbac718c8da2/single_stage_detector/ssd/coco_utils.py

import copy
import os
from PIL import Image

import random
import numpy as np
import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from test.external.mlperf_retinanet import transforms as T
from test.external.mlperf_retinanet.boxes import box_iou
from test.external.mlperf_retinanet.utils import Matcher


class ConvertCocoPolysToMask(object):
    def __init__(self, filter_iscrowd=True):
        self.filter_iscrowd = filter_iscrowd

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        if self.filter_iscrowd:
            anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_openimages(name, root, image_set, transforms):
    PATHS = {
        "train": os.path.join(root, "train"),
        "val":   os.path.join(root, "validation"),
    }

    t = [ConvertCocoPolysToMask(filter_iscrowd=False)]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    img_folder = os.path.join(PATHS[image_set], "data")
    ann_file = os.path.join(PATHS[image_set], "labels", f"{name}.json")

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    return dataset

# This applies the filtering in https://github.com/mlcommons/training/blob/cdd928d4596c142c15a7d86b2eeadbac718c8da2/single_stage_detector/ssd/model/retinanet.py#L117
# and https://github.com/mlcommons/training/blob/cdd928d4596c142c15a7d86b2eeadbac718c8da2/single_stage_detector/ssd/model/retinanet.py#L203 to match with tinygrad's dataloader implementation.
def postprocess_targets(targets, anchors):
    proposal_matcher, matched_idxs = Matcher(0.5, 0.4, allow_low_quality_matches=True), []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        if targets_per_image['boxes'].numel() == 0:
            matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64,
                                            device=anchors_per_image.device))
            continue

        match_quality_matrix = box_iou(targets_per_image['boxes'], anchors_per_image)
        matched_idxs.append(proposal_matcher(match_quality_matrix))

    for targets_per_image,  matched_idxs_per_image in zip(targets, matched_idxs):
        foreground_idxs_per_image = matched_idxs_per_image >= 0
        targets_per_image["boxes"] = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
        targets_per_image["labels"] = targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]]

    return targets