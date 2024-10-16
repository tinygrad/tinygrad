# https://github.com/mlcommons/training/blob/cdd928d4596c142c15a7d86b2eeadbac718c8da2/single_stage_detector/ssd/model/boxes.py

import torch
from torch import Tensor
from typing import Tuple
import torchvision
from torchvision.extension import _assert_has_ops


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    _assert_has_ops()
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    # Ideally for GPU we'd use a higher threshold
    if boxes.numel() > 4_000 and not torchvision._is_tracing():
        return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
    else:
        return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)


@torch.jit._script_if_tracing
def _batched_nms_coordinate_trick(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    # strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


@torch.jit._script_if_tracing
def _batched_nms_vanilla(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]


def clip_boxes_to_image(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    Clip boxes so that they lie inside an image of size `size`.

    Args:
        boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        size (Tuple[height, width]): size of the image

    Returns:
        Tensor[N, 4]: clipped boxes
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou