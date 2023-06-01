import re
import math
import numpy as np
from pathlib import Path
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from extra.utils import get_child, download_file, fake_torch_load
from models.resnet import ResNet
from torch.nn import functional as F
import torch

from maskrcnn_benchmark import _C

_box_nms = _C.nms

class LastLevelMaxPool:
  def __call__(self, x):
    return [Tensor.max_pool2d(x, 1, 2)]


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


def permute_and_flatten(layer:Tensor, N, A, C, H, W):
  layer = layer.reshape(N, -1, C, H, W)
  layer = layer.permute(0, 3, 4, 1, 2)
  layer = layer.reshape(N, -1, C)
  return layer

class BoxList(object):
  """
  This class represents a set of bounding boxes.
  The bounding boxes are represented as a Nx4 Tensor.
  In order to uniquely determine the bounding boxes with respect
  to an image, we also store the corresponding image dimensions.
  They can contain extra information that is specific to each bounding box, such as
  labels.
  """

  def __init__(self, bbox, image_size, mode="xyxy"):
    if not isinstance(bbox, Tensor):
      bbox = Tensor(bbox)
    if bbox.ndim != 2:
      raise ValueError(
        "bbox should have 2 dimensions, got {}".format(bbox.ndim)
      )
    if bbox.shape[-1] != 4:
      raise ValueError(
        "last dimenion of bbox should have a "
        "size of 4, got {}".format(bbox.shape[-1])
      )
    if mode not in ("xyxy", "xywh"):
      raise ValueError("mode should be 'xyxy' or 'xywh'")

    self.bbox = bbox
    self.size = image_size  # (image_width, image_height)
    self.mode = mode
    self.extra_fields = {}

  def add_field(self, field, field_data):
    self.extra_fields[field] = field_data

  def get_field(self, field):
    return self.extra_fields[field]

  def has_field(self, field):
    return field in self.extra_fields

  def fields(self):
    return list(self.extra_fields.keys())

  def _copy_extra_fields(self, bbox):
    for k, v in bbox.extra_fields.items():
      self.extra_fields[k] = v

  def convert(self, mode):
    if mode not in ("xyxy", "xywh"):
      raise ValueError("mode should be 'xyxy' or 'xywh'")
    if mode == self.mode:
      return self
    # we only have two modes, so don't need to check
    # self.mode
    xmin, ymin, xmax, ymax = self._split_into_xyxy()
    if mode == "xyxy":
      bbox = Tensor.cat(*(xmin, ymin, xmax, ymax), dim=-1)
      bbox = BoxList(bbox, self.size, mode=mode)
    else:
      TO_REMOVE = 1
      bbox = Tensor.cat(
        *(xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
      )
      bbox = BoxList(bbox, self.size, mode=mode)
    bbox._copy_extra_fields(self)
    return bbox

  def _split_into_xyxy(self):
    if self.mode == "xyxy":
      xmin, ymin, xmax, ymax = self.bbox.split(4, dim=-1)
      return xmin, ymin, xmax, ymax
    elif self.mode == "xywh":
      TO_REMOVE = 1
      xmin, ymin, w, h = self.bbox.split(4, dim=-1)
      return (
        xmin,
        ymin,
        xmin + (w - TO_REMOVE).clamp(min=0),
        ymin + (h - TO_REMOVE).clamp(min=0),
      )
    else:
      raise RuntimeError("Should not be here")

  def resize(self, size, *args, **kwargs):
    """
    Returns a resized copy of this bounding box

    :param size: The requested size in pixels, as a 2-tuple:
        (width, height).
    """

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
    if ratios[0] == ratios[1]:
      ratio = ratios[0]
      scaled_box = self.bbox * ratio
      bbox = BoxList(scaled_box, size, mode=self.mode)
      # bbox._copy_extra_fields(self)
      for k, v in self.extra_fields.items():
        if not isinstance(v, Tensor):
          v = v.resize(size, *args, **kwargs)
        bbox.add_field(k, v)
      return bbox

    ratio_width, ratio_height = ratios
    xmin, ymin, xmax, ymax = self._split_into_xyxy()
    scaled_xmin = xmin * ratio_width
    scaled_xmax = xmax * ratio_width
    scaled_ymin = ymin * ratio_height
    scaled_ymax = ymax * ratio_height
    scaled_box = Tensor.cat(
      *(scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
    )
    bbox = BoxList(scaled_box, size, mode="xyxy")
    # bbox._copy_extra_fields(self)
    for k, v in self.extra_fields.items():
      if not isinstance(v, Tensor):
        v = v.resize(size, *args, **kwargs)
      bbox.add_field(k, v)

    return bbox.convert(self.mode)

  def transpose(self, method):
    """
    Transpose bounding box (flip or rotate in 90 degree steps)
    :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
      :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
      :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
      :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
    """
    if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
      raise NotImplementedError(
        "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
      )

    image_width, image_height = self.size
    xmin, ymin, xmax, ymax = self._split_into_xyxy()
    if method == FLIP_LEFT_RIGHT:
      TO_REMOVE = 1
      transposed_xmin = image_width - xmax - TO_REMOVE
      transposed_xmax = image_width - xmin - TO_REMOVE
      transposed_ymin = ymin
      transposed_ymax = ymax
    elif method == FLIP_TOP_BOTTOM:
      transposed_xmin = xmin
      transposed_xmax = xmax
      transposed_ymin = image_height - ymax
      transposed_ymax = image_height - ymin

    transposed_boxes = Tensor.cat(
      *(transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
    )
    bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
    # bbox._copy_extra_fields(self)
    for k, v in self.extra_fields.items():
      if not isinstance(v, Tensor):
        v = v.transpose(method)
      bbox.add_field(k, v)
    return bbox.convert(self.mode)

  def clip_to_image(self, remove_empty=True):
    TO_REMOVE = 1
    self.bbox[:, 0].clip(min_=0, max_=self.size[0] - TO_REMOVE)
    self.bbox[:, 1].clip(min_=0, max_=self.size[1] - TO_REMOVE)
    self.bbox[:, 2].clip(min_=0, max_=self.size[0] - TO_REMOVE)
    self.bbox[:, 3].clip(min_=0, max_=self.size[1] - TO_REMOVE)
    if remove_empty:
      box = self.bbox
      keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
      return self[keep]
    return self

  def __getitem__(self, item):
    bbox = BoxList(self.bbox.numpy()[item], self.size, self.mode)
    for k, v in self.extra_fields.items():
      bbox.add_field(k, Tensor(v.numpy()[item]))
    return bbox

  def __len__(self):
    return self.bbox.shape[0]


def cat_boxlist(bboxes):
  """
  Concatenates a list of BoxList (having the same image size) into a
  single BoxList

  Arguments:
      bboxes (list[BoxList])
  """
  assert isinstance(bboxes, (list, tuple))
  assert all(isinstance(bbox, BoxList) for bbox in bboxes)

  size = bboxes[0].size
  assert all(bbox.size == size for bbox in bboxes)

  mode = bboxes[0].mode
  assert all(bbox.mode == mode for bbox in bboxes)

  fields = set(bboxes[0].fields())
  assert all(set(bbox.fields()) == fields for bbox in bboxes)

  cat_boxes = BoxList(Tensor.cat(*(bbox.bbox for bbox in bboxes), dim=0), size, mode)

  for field in fields:
    data = Tensor.cat(*[bbox.get_field(field) for bbox in bboxes], dim=0)
    cat_boxes.add_field(field, data)

  return cat_boxes


class FPN:
  def __init__(self, in_channels_list, out_channels):
    self.inner_blocks, self.layer_blocks = [], []
    for in_channels in in_channels_list:
      self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
      self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    self.top_block = LastLevelMaxPool()

  def __call__(self, x: Tensor):
    """
    Arguments:
        x (list[Tensor]): feature maps for each feature level.
    Returns:
        results (tuple[Tensor]): feature maps after FPN layers.
            They are ordered from highest resolution first.
    """
    last_inner = self.inner_blocks[-1](x[-1])
    results = []
    results.append(self.layer_blocks[-1](last_inner))
    for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
    ):
      if not inner_block:
        continue
      # TODO: remove torch
      inner_top_down = Tensor(F.interpolate(torch.tensor(last_inner.numpy()), scale_factor=2, mode="nearest").numpy())
      inner_lateral = inner_block(feature)
      last_inner = inner_lateral + inner_top_down
      layer_result = layer_block(last_inner)
      results.insert(0, layer_result)
    last_results = self.top_block(results[-1])
    results.extend(last_results)

    return tuple(results)


class ResNetFPN:
  def __init__(self, resnet, out_channels=256):
    self.out_channels = out_channels
    self.body = resnet
    in_channels_stage2 = 256
    in_channels_list = [
      in_channels_stage2,
      in_channels_stage2 * 2,
      in_channels_stage2 * 4,
      in_channels_stage2 * 8,
    ]
    self.fpn = FPN(in_channels_list, out_channels)

  def __call__(self, x):
    x = self.body(x)
    return self.fpn(x)


class AnchorGenerator:
  def __init__(
          self,
          sizes=(32, 64, 128, 256, 512),
          aspect_ratios=(0.5, 1.0, 2.0),
          anchor_strides=(4, 8, 16, 32, 64),
          straddle_thresh=0,
  ):
    if len(anchor_strides) == 1:
      anchor_stride = anchor_strides[0]
      cell_anchors = [
        generate_anchors(anchor_stride, sizes, aspect_ratios)
      ]
    else:
      if len(anchor_strides) != len(sizes):
        raise RuntimeError("FPN should have #anchor_strides == #sizes")

      cell_anchors = [
        generate_anchors(
          anchor_stride,
          size if isinstance(size, (tuple, list)) else (size,),
          aspect_ratios
        )
        for anchor_stride, size in zip(anchor_strides, sizes)
      ]
    self.strides = anchor_strides
    self.cell_anchors = [Tensor(a) for a in cell_anchors]
    self.straddle_thresh = straddle_thresh

  def num_anchors_per_location(self):
    return [cell_anchors.shape[0] for cell_anchors in self.cell_anchors]

  def grid_anchors(self, grid_sizes):
    anchors = []
    for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.cell_anchors
    ):
      grid_height, grid_width = size
      device = base_anchors.device
      shifts_x = Tensor.arange(
        start=0, stop=grid_width * stride, step=stride, dtype=dtypes.float32, device=device
      )
      shifts_y = Tensor.arange(
        start=0, stop=grid_height * stride, step=stride, dtype=dtypes.float32, device=device
      )
      shift_y, shift_x = Tensor.meshgrid(shifts_y, shifts_x)
      shift_x = shift_x.reshape(-1)
      shift_y = shift_y.reshape(-1)
      shifts = Tensor.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

      anchors.append(
        (shifts.reshape(-1, 1, 4) + base_anchors.reshape(1, -1, 4)).reshape(-1, 4)
      )

    return anchors

  def add_visibility_to(self, boxlist):
    image_width, image_height = boxlist.size
    anchors = boxlist.bbox
    if self.straddle_thresh >= 0:
      inds_inside = (
              (anchors[:, 0] >= -self.straddle_thresh)
              * (anchors[:, 1] >= -self.straddle_thresh)
              * (anchors[:, 2] < image_width + self.straddle_thresh)
              * (anchors[:, 3] < image_height + self.straddle_thresh)
      )
    else:
      device = anchors.device
      inds_inside = Tensor.ones(anchors.shape[0], dtype=dtypes.uint8, device=device)
    boxlist.add_field("visibility", inds_inside)

  def __call__(self, image_list, feature_maps):
    grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
    anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
    anchors = []
    for i, (image_height, image_width) in enumerate(image_list.image_sizes):
      anchors_in_image = []
      for anchors_per_feature_map in anchors_over_all_feature_maps:
        boxlist = BoxList(
          anchors_per_feature_map, (image_width, image_height), mode="xyxy"
        )
        self.add_visibility_to(boxlist)
        anchors_in_image.append(boxlist)
      anchors.append(anchors_in_image)
    return anchors


def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
  return _generate_anchors(stride, np.array(sizes, dtype=np.float32) / stride, np.array(aspect_ratios, dtype=np.float32))


def _generate_anchors(base_size, scales, aspect_ratios):
  anchor = np.array([1, 1, base_size, base_size], dtype=np.float32) - 1
  anchors = _ratio_enum(anchor, aspect_ratios)
  anchors = np.vstack(
    [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
  )
  return anchors


def _whctrs(anchor):
  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
  ws = ws[:, np.newaxis]
  hs = hs[:, np.newaxis]
  anchors = np.hstack((
    x_ctr - 0.5 * (ws - 1),
    y_ctr - 0.5 * (hs - 1),
    x_ctr + 0.5 * (ws - 1),
    y_ctr + 0.5 * (hs - 1),
  ))
  return anchors


def _ratio_enum(anchor, ratios):
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  size = w * h
  size_ratios = size / ratios
  ws = np.round(np.sqrt(size_ratios))
  hs = np.round(ws * ratios)
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


def _scale_enum(anchor, scales):
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  ws = w * scales
  hs = h * scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


class RPNHead:
  def __init__(self, in_channels, num_anchors):
    self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
    self.cls_logits = nn.Conv2d(256, num_anchors, kernel_size=1)
    self.bbox_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=1)

  def __call__(self, x):
    logits = []
    bbox_reg = []
    for feature in x:
        t = Tensor.relu(self.conv(feature))
        logits.append(self.cls_logits(t))
        bbox_reg.append(self.bbox_pred(t))
    return logits, bbox_reg


class BoxCoder(object):
  """
  This class encodes and decodes a set of bounding boxes into
  the representation used for training the regressors.
  """

  def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
    """
    Arguments:
        weights (4-element tuple)
        bbox_xform_clip (float)
    """
    self.weights = weights
    self.bbox_xform_clip = bbox_xform_clip

  def encode(self, reference_boxes, proposals):
    """
    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
    """

    TO_REMOVE = 1  # TODO remove
    ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
    ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
    ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
    ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

    gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
    gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
    gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = self.weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * Tensor.log(gt_widths / ex_widths)
    targets_dh = wh * Tensor.log(gt_heights / ex_heights)

    targets = Tensor.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets

  def decode(self, rel_codes, boxes):
    """
    From a set of original boxes and encoded relative box offsets,
    get the decoded boxes.

    Arguments:
        rel_codes (Tensor): encoded boxes
        boxes (Tensor): reference boxes.
    """

    boxes = boxes.cast(rel_codes.dtype).numpy()
    rel_codes = rel_codes.numpy()

    TO_REMOVE = 1  # TODO remove
    widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
    heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = self.weights
    dx = rel_codes[:, 0::4] / wx
    dy = rel_codes[:, 1::4] / wy
    dw = rel_codes[:, 2::4] / ww
    dh = rel_codes[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    dw = np.clip(dw, a_min=dw.min(), a_max=self.bbox_xform_clip)
    dh = np.clip(dh, a_min=dh.min(), a_max=self.bbox_xform_clip)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = np.exp(dw) * widths[:, None]
    pred_h = np.exp(dh) * heights[:, None]
    pred_boxes = np.zeros_like(rel_codes)
    # x1
    pred_boxes[:, 0::4] += pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] += pred_ctr_y - 0.5 * pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] += pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] += pred_ctr_y + 0.5 * pred_h - 1

    return Tensor(pred_boxes)


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
  """
  Performs non-maximum suppression on a boxlist, with scores specified
  in a boxlist field via score_field.

  Arguments:
      boxlist(BoxList)
      nms_thresh (float)
      max_proposals (int): if > 0, then only the top max_proposals are kept
          after non-maximum suppression
      score_field (str)
  """
  if nms_thresh <= 0:
    return boxlist
  mode = boxlist.mode
  boxlist = boxlist.convert("xyxy")
  boxes = boxlist.bbox
  score = boxlist.get_field(score_field)
  # TODO: remove torch
  keep = _box_nms(torch.tensor(boxes.numpy()), torch.tensor(score.numpy()), nms_thresh)
  if max_proposals > 0:
    keep = keep[: max_proposals]
  boxlist = boxlist[keep]
  return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
  """
  Only keep boxes with both sides >= min_size

  Arguments:
      boxlist (Boxlist)
      min_size (int)
  """
  # TODO maybe add an API for querying the ws / hs
  xywh_boxes = boxlist.convert("xywh").bbox
  _, _, ws, hs = xywh_boxes.split(4, dim=1)
  keep = ((
          (ws.numpy() >= min_size) * (hs.numpy() >= min_size)
  ) > 0).squeeze(1)
  return boxlist[keep.tolist()]


class RPNPostProcessor:
  """
  Performs post-processing on the outputs of the RPN boxes, before feeding the
  proposals to the heads
  """

  def __init__(
          self,
          pre_nms_top_n,
          post_nms_top_n,
          nms_thresh,
          min_size,
          box_coder=None,
          fpn_post_nms_top_n=None,
  ):
    """
    Arguments:
        pre_nms_top_n (int)
        post_nms_top_n (int)
        nms_thresh (float)
        min_size (int)
        box_coder (BoxCoder)
        fpn_post_nms_top_n (int)
    """
    self.pre_nms_top_n = pre_nms_top_n
    self.post_nms_top_n = post_nms_top_n
    self.nms_thresh = nms_thresh
    self.min_size = min_size

    if box_coder is None:
      box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    self.box_coder = box_coder

    if fpn_post_nms_top_n is None:
      fpn_post_nms_top_n = post_nms_top_n
    self.fpn_post_nms_top_n = fpn_post_nms_top_n

  def forward_for_single_feature_map(self, anchors, objectness, box_regression):
    """
    Arguments:
        anchors: list[BoxList]
        objectness: tensor of size N, A, H, W
        box_regression: tensor of size N, A * 4, H, W
    """
    device = objectness.device
    N, A, H, W = objectness.shape

    # put in the same format as anchors
    objectness = permute_and_flatten(objectness, N, A, 1, H, W).reshape(N, -1)
    objectness = objectness.sigmoid()

    box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)

    num_anchors = A * H * W

    pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
    objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

    batch_idx = Tensor.arange(N, device=device)[:, None]
    box_regression = Tensor(box_regression.numpy()[batch_idx.numpy().astype(np.intc), topk_idx.astype(np.intc)])

    image_shapes = [box.size for box in anchors]
    concat_anchors = Tensor.cat(*[a.bbox for a in anchors], dim=0)
    concat_anchors = Tensor(concat_anchors.reshape(N, -1, 4).numpy()[batch_idx.numpy().astype(np.intc), topk_idx.astype(np.intc)])

    proposals = self.box_coder.decode(
      box_regression.reshape(-1, 4), concat_anchors.reshape(-1, 4)
    )

    proposals = proposals.reshape(N, -1, 4)

    result = []
    for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
      boxlist = BoxList(proposal, im_shape, mode="xyxy")
      boxlist.add_field("objectness", score)
      boxlist = boxlist.clip_to_image(remove_empty=False)
      boxlist = remove_small_boxes(boxlist, self.min_size)
      boxlist = boxlist_nms(
        boxlist,
        self.nms_thresh,
        max_proposals=self.post_nms_top_n,
        score_field="objectness",
      )
      result.append(boxlist)
    return result

  def __call__(self, anchors, objectness, box_regression):
    """
    Arguments:
        anchors: list[list[BoxList]]
        objectness: list[tensor]
        box_regression: list[tensor]

    Returns:
        boxlists (list[BoxList]): the post-processed anchors, after
            applying box decoding and NMS
    """
    sampled_boxes = []
    num_levels = len(objectness)
    anchors = list(zip(*anchors))
    for a, o, b in zip(anchors, objectness, box_regression):
      sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

    boxlists = list(zip(*sampled_boxes))
    boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

    if num_levels > 1:
      boxlists = self.select_over_all_levels(boxlists)

    return boxlists

  def select_over_all_levels(self, boxlists):
    num_images = len(boxlists)
    for i in range(num_images):
      objectness = boxlists[i].get_field("objectness")
      post_nms_top_n = min(self.fpn_post_nms_top_n, objectness.shape[0])
      _, inds_sorted = objectness.topk(
        post_nms_top_n, dim=0, sorted=True
      )
      boxlists[i] = boxlists[i][inds_sorted]
    return boxlists


class RPN:
  def __init__(self, in_channels):
    self.anchor_generator = AnchorGenerator()

    in_channels = 256
    head = RPNHead(
      in_channels, self.anchor_generator.num_anchors_per_location()[0]
    )
    rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    box_selector_test = RPNPostProcessor(
        pre_nms_top_n=1000,
        post_nms_top_n=1000,
        nms_thresh=0.7,
        min_size=0,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=1000
    )
    self.head = head
    self.box_selector_test = box_selector_test

  def __call__(self, images, features, targets=None):
    objectness, rpn_box_regression = self.head(features)
    anchors = self.anchor_generator(images, features)
    boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
    return boxes, {}


def make_conv3x3(
  in_channels,
  out_channels,
  dilation=1,
  stride=1,
  use_gn=False,
):
  conv = nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=stride,
    padding=dilation,
    dilation=dilation,
    bias=False if use_gn else True
  )
  return conv


class MaskRCNNFPNFeatureExtractor:
  def __init__(self):
    resolution = 14
    scales = (1.0 / 16,)
    sampling_ratio = 0
    pooler = Pooler(
      output_size=(resolution, resolution),
      scales=scales,
      sampling_ratio=sampling_ratio,
    )
    input_size = 256
    self.pooler = pooler

    use_gn = False
    layers = (256, 256, 256, 256)
    dilation = 1

    next_feature = input_size
    self.blocks = []
    for layer_idx, layer_features in enumerate(layers, 1):
      layer_name = "mask_fcn{}".format(layer_idx)
      module = make_conv3x3(next_feature, layer_features,
                            dilation=dilation, stride=1, use_gn=use_gn
                            )
      exec(f"self.{layer_name} = module")
      next_feature = layer_features
      self.blocks.append(layer_name)


class MaskRCNNC4Predictor:
  def __init__(self):
    num_classes = 81
    dim_reduced = 256
    num_inputs = dim_reduced
    self.conv5_mask = nn.ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
    self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)


class RoIBoxFeatureExtractor:
  def __init__(self, in_channels):
    self.fc6 = nn.Linear(12544, 1024)
    self.fc7 = nn.Linear(1024, 1024)
    self.pooler = Pooler(0, 0, 0)


class Pooler:
  def __init__(self, output_size, scales, sampling_ratio):
    self.output_size = output_size
    self.scales = scales
    self.sampling_ratio = sampling_ratio


class Predictor:
  def __init__(self, ):
    self.cls_score = nn.Linear(1024, 81)
    self.bbox_pred = nn.Linear(1024, 324)


class RoIBoxHead:
  def __init__(self, in_channels):
    self.feature_extractor = RoIBoxFeatureExtractor(in_channels)
    self.predictor = Predictor()


class Mask:
  def __init__(self):
    self.feature_extractor = MaskRCNNFPNFeatureExtractor()
    self.predictor = MaskRCNNC4Predictor()


class RoIHeads:
  def __init__(self, in_channels, num_classes):
    self.box = RoIBoxHead(in_channels)
    self.mask = Mask()


class ImageList(object):
  """
  Structure that holds a list of images (of possibly
  varying sizes) as a single tensor.
  This works by padding the images to the same size,
  and storing in a field the original sizes of each image
  """

  def __init__(self, tensors, image_sizes):
    """
    Arguments:
        tensors (tensor)
        image_sizes (list[tuple[int, int]])
    """
    self.tensors = tensors
    self.image_sizes = image_sizes

  def to(self, *args, **kwargs):
    cast_tensor = self.tensors.to(*args, **kwargs)
    return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=0):
  """
  tensors can be an ImageList, a torch.Tensor or
  an iterable of Tensors. It can't be a numpy array.
  When tensors is an iterable of Tensors, it pads
  the Tensors with zeros so that they have the same
  shape
  """
  if isinstance(tensors, Tensor) and size_divisible > 0:
    tensors = [tensors]

  if isinstance(tensors, ImageList):
    return tensors
  elif isinstance(tensors, Tensor):
    # single tensor shape can be inferred
    assert tensors.ndim == 4
    image_sizes = [tensor.shape[-2:] for tensor in tensors]
    return ImageList(tensors, image_sizes)
  elif isinstance(tensors, (tuple, list)):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

    # TODO Ideally, just remove this and let me model handle arbitrary
    if size_divisible > 0:
      stride = size_divisible
      max_size = list(max_size)
      max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
      max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
      max_size = tuple(max_size)

    batch_shape = (len(tensors),) + max_size
    batched_imgs = tensors[0].new(*batch_shape).zero_()
    for img, pad_img in zip(tensors, batched_imgs):
      pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    image_sizes = [im.shape[-2:] for im in tensors]

    return ImageList(batched_imgs, image_sizes)
  else:
    raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


class MaskRCNN:
  def __init__(self, backbone: ResNet):
    self.backbone = ResNetFPN(backbone, out_channels=256)
    self.rpn = RPN(self.backbone.out_channels)
    self.roi_heads = RoIHeads(self.backbone.out_channels, 91)

  def load_from_pretrained(self):
    fn = Path('./') / "weights/maskrcnn.pt"
    download_file("https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth", fn)

    with open(fn, "rb") as f:
      state_dict = fake_torch_load(f.read())['model']
    loaded_keys = []
    for k, v in state_dict.items():
      if "module." in k:
        k = k.replace("module.", "")
      if "stem." in k:
        k = k.replace("stem.", "")
      if "fpn_inner" in k:
        block_index = int(re.search(r"fpn_inner(\d+)", k).group(1))
        k = re.sub(r"fpn_inner\d+", f"inner_blocks.{block_index - 1}", k)
      if "fpn_layer" in k:
        block_index = int(re.search(r"fpn_layer(\d+)", k).group(1))
        k = re.sub(r"fpn_layer\d+", f"layer_blocks.{block_index - 1}", k)
      print(k)
      loaded_keys.append(k)
      get_child(self, k).assign(v.numpy()).realize()
    return loaded_keys

  def __call__(self, images):
    # TODO: add __call__ for all children
    image_list = to_image_list(images)
    features = self.backbone(images)
    proposals = self.rpn(image_list, features, None)
    # detections = self.roi_heads(features, proposals)
    return proposals


if __name__ == '__main__':
  resnet = resnet = ResNet(50, num_classes=None)
  model = MaskRCNN(backbone=resnet)
  model.load_from_pretrained()
