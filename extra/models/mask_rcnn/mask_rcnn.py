import re
import math
import numpy as np
import torch
import extra.models.mask_rcnn as mask_rcnn
from pathlib import Path
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, get_child, fetch
from tinygrad.nn.state import torch_load
from typing import Tuple
from extra.models.resnet import ResNet

def _make_conv3x3(
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

def _bilinear_interpolate(
  input,  # [N, C, H, W]
  roi_batch_ind,  # [K]
  y,  # [K, PH, IY]
  x,  # [K, PW, IX]
  ymask,  # [K, IY]
  xmask,  # [K, IX]
):
  _, channels, height, width = input.shape
  y = y.clip(min_=0.0, max_=float(height-1))
  x = x.clip(min_=0.0, max_=float(width-1))

  # Tensor.where doesnt work well with int32 data so cast to float32
  y_low = y.cast(dtypes.int32).contiguous().float().contiguous()
  x_low = x.cast(dtypes.int32).contiguous().float().contiguous()

  y_high = Tensor.where(y_low >= height - 1, float(height - 1), y_low + 1)
  y_low = Tensor.where(y_low >= height - 1, float(height - 1), y_low)

  x_high = Tensor.where(x_low >= width - 1, float(width - 1), x_low + 1)
  x_low = Tensor.where(x_low >= width - 1, float(width - 1), x_low)

  ly = y - y_low
  lx = x - x_low
  hy = 1.0 - ly
  hx = 1.0 - lx

  def masked_index(
    y,  # [K, PH, IY]
    x,  # [K, PW, IX]
  ):
    if ymask is not None:
      assert xmask is not None
      y = Tensor.where(ymask[:, None, :], y, 0)
      x = Tensor.where(xmask[:, None, :], x, 0)
    key1 = roi_batch_ind[:, None, None, None, None, None]
    key2 = Tensor.arange(channels, device=input.device)[None, :, None, None, None, None]
    key3 = y[:, None, :, None, :, None]
    key4 = x[:, None, None, :, None, :]
    return mask_rcnn.tensor_getitem(input,key1,key2,key3,key4)  # [K, C, PH, PW, IY, IX]

  v1 = masked_index(y_low, x_low)
  v2 = masked_index(y_low, x_high)
  v3 = masked_index(y_high, x_low)
  v4 = masked_index(y_high, x_high)

  # all ws preemptively [K, C, PH, PW, IY, IX]
  def outer_prod(y, x):
    return y[:, None, :, None, :, None] * x[:, None, None, :, None, :]

  w1 = outer_prod(hy, hx)
  w2 = outer_prod(hy, lx)
  w3 = outer_prod(ly, hx)
  w4 = outer_prod(ly, lx)

  val = w1*v1 + w2*v2 + w3*v3 + w4*v4
  return val

#https://pytorch.org/vision/main/_modules/torchvision/ops/roi_align.html#roi_align
def _roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
  orig_dtype = input.dtype
  _, _, height, width = input.shape
  ph = Tensor.arange(pooled_height, device=input.device)
  pw = Tensor.arange(pooled_width, device=input.device)

  roi_batch_ind = rois[:, 0].cast(dtypes.int32).contiguous()
  offset = 0.5 if aligned else 0.0
  roi_start_w = rois[:, 1] * spatial_scale - offset
  roi_start_h = rois[:, 2] * spatial_scale - offset
  roi_end_w = rois[:, 3] * spatial_scale - offset
  roi_end_h = rois[:, 4] * spatial_scale - offset

  roi_width = roi_end_w - roi_start_w
  roi_height = roi_end_h - roi_start_h
  if not aligned:
    roi_width = roi_width.maximum(1.0)
    roi_height = roi_height.maximum(1.0)

  bin_size_h = roi_height / pooled_height
  bin_size_w = roi_width / pooled_width

  exact_sampling = sampling_ratio > 0
  roi_bin_grid_h = sampling_ratio if exact_sampling else (roi_height / pooled_height).ceil()
  roi_bin_grid_w = sampling_ratio if exact_sampling else (roi_width / pooled_width).ceil()

  if exact_sampling:
    count = max(roi_bin_grid_h * roi_bin_grid_w, 1)
    iy = Tensor.arange(roi_bin_grid_h, device=input.device)
    ix = Tensor.arange(roi_bin_grid_w, device=input.device)
    ymask = None
    xmask = None
  else:
    count = (roi_bin_grid_h * roi_bin_grid_w).maximum(1)
    iy = Tensor.arange(height, device=input.device)
    ix = Tensor.arange(width, device=input.device)
    ymask = iy[None, :] < roi_bin_grid_h[:, None]
    xmask = ix[None, :] < roi_bin_grid_w[:, None]

  def from_K(t):
    return t[:, None, None]

  y = (
    from_K(roi_start_h)
    + ph[None, :, None] * from_K(bin_size_h)
    + (iy[None, None, :] + 0.5) * from_K(bin_size_h / roi_bin_grid_h)
  )
  x = (
    from_K(roi_start_w)
    + pw[None, :, None] * from_K(bin_size_w)
    + (ix[None, None, :] + 0.5) * from_K(bin_size_w / roi_bin_grid_w)
  )

  val = _bilinear_interpolate(input, roi_batch_ind, y, x, ymask, xmask)
  if not exact_sampling:
    val = ymask[:, None, None, None, :, None].where(val, 0)
    val = xmask[:, None, None, None, None, :].where(val, 0)

  output = val.sum((-1, -2))
  if isinstance(count, Tensor):
    output = output / count[:, None, None, None]
  else:
    output = output / count

  output = output.cast(orig_dtype)
  return output

def _keep_only_positive_boxes(boxes):
  """
  Given a set of BoxList containing the `labels` field,
  return a set of BoxList for which `labels > 0`.

  Arguments:
      boxes (list of BoxList)
  """
  assert isinstance(boxes, (list, tuple))
  assert isinstance(boxes[0], mask_rcnn.BoxList)
  assert boxes[0].has_field("labels")
  positive_boxes = []
  positive_inds = []
  num_boxes = 0
  for boxes_per_image in boxes:
    labels = boxes_per_image.get_field("labels")
    inds_mask = labels > 0
    inds = mask_rcnn.nonzero(inds_mask)
    positive_boxes.append(boxes_per_image[inds])
    positive_inds.append(inds_mask)
  return positive_boxes, positive_inds



class LastLevelMaxPool:
  def __call__(self, x): return [Tensor.max_pool2d(x, 1, 2)]


class FPN:
  def __init__(self, in_channels_list, out_channels):
    self.inner_blocks, self.layer_blocks = [], []
    for in_channels in in_channels_list:
      self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
      self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    self.top_block = LastLevelMaxPool()

  def __call__(self, x: Tensor):
    last_inner = self.inner_blocks[-1](x[-1])
    results = []
    results.append(self.layer_blocks[-1](last_inner))
    for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
    ):
      if not inner_block:
        continue
      inner_top_down = mask_rcnn.nearest_interpolate(last_inner, scale_factor=2)
      inner_lateral = inner_block(feature)
      last_inner = inner_lateral + inner_top_down
      layer_result = layer_block(last_inner)
      results.insert(0, layer_result)
    last_results = self.top_block(results[-1])
    results.extend(last_results)

    return tuple(results)


class ResNetFPN:
  def __init__(self, resnet, out_channels=256, training=True):
    self.out_channels = out_channels
    self.training = training
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
    # NOTE: this ensures that `BatchNorm2d` behaves similar to mlperf's implementation of FrozenBatchNorm2d
    with Tensor.train(val=not self.training): x = self.body(x)
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
        mask_rcnn.generate_anchors(anchor_stride, sizes, aspect_ratios)
      ]
    else:
      if len(anchor_strides) != len(sizes):
        raise RuntimeError("FPN should have #anchor_strides == #sizes")

      cell_anchors = [
        mask_rcnn.generate_anchors(
          anchor_stride,
          size if isinstance(size, (tuple, list)) else (size,),
          aspect_ratios
        )
        for anchor_stride, size in zip(anchor_strides, sizes)
      ]
    self.strides = anchor_strides
    self.cell_anchors = cell_anchors
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
      shift_y, shift_x = mask_rcnn.meshgrid(shifts_y, shifts_x)
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
    for (image_height, image_width) in image_list.image_sizes:
      anchors_in_image = []
      for anchors_per_feature_map in anchors_over_all_feature_maps:
        boxlist = mask_rcnn.BoxList(
          anchors_per_feature_map, (image_width, image_height), mode="xyxy"
        )
        self.add_visibility_to(boxlist)
        anchors_in_image.append(boxlist)
      anchors.append(anchors_in_image)
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


class RPNPostProcessor:
  # Not used in Loss calculation
  def __init__(
          self,
          pre_nms_top_n,
          post_nms_top_n,
          nms_thresh,
          min_size,
          box_coder=None,
          fpn_post_nms_top_n=None,
  ):
    self.pre_nms_top_n = pre_nms_top_n
    self.post_nms_top_n = post_nms_top_n
    self.nms_thresh = nms_thresh
    self.min_size = min_size

    if box_coder is None:
      box_coder = mask_rcnn.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    self.box_coder = box_coder

    if fpn_post_nms_top_n is None:
      fpn_post_nms_top_n = post_nms_top_n
    self.fpn_post_nms_top_n = fpn_post_nms_top_n

  def forward_for_single_feature_map(self, anchors, objectness, box_regression):
    device = objectness.device
    N, A, H, W = objectness.shape
    objectness = mask_rcnn.permute_and_flatten(objectness, N, A, 1, H, W).reshape(N, -1)
    objectness = objectness.sigmoid()

    box_regression = mask_rcnn.permute_and_flatten(box_regression, N, A, 4, H, W)

    num_anchors = A * H * W

    pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
    objectness, topk_idx = mask_rcnn.topk(objectness, pre_nms_top_n, dim=1, sorted=False)
    concat_anchors = Tensor.cat(*[a.bbox for a in anchors], dim=0).reshape(N, -1, 4)
    image_shapes = [box.size for box in anchors]

    box_regression_list = []
    concat_anchors_list = []
    for batch_idx in range(N):
      box_regression_list.append(mask_rcnn.tensor_gather(box_regression[batch_idx], topk_idx[batch_idx]))
      concat_anchors_list.append(mask_rcnn.tensor_gather(concat_anchors[batch_idx], topk_idx[batch_idx]))

    box_regression = Tensor.stack(box_regression_list)
    concat_anchors = Tensor.stack(concat_anchors_list)

    proposals = self.box_coder.decode(
      box_regression.reshape(-1, 4), concat_anchors.reshape(-1, 4)
    )

    proposals = proposals.reshape(N, -1, 4)

    result = []
    for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
      boxlist = mask_rcnn.BoxList(proposal, im_shape, mode="xyxy")
      boxlist.add_field("objectness", score)
      boxlist = boxlist.clip_to_image(remove_empty=False)
      boxlist = mask_rcnn.remove_small_boxes(boxlist, self.min_size)
      boxlist = mask_rcnn.boxlist_nms(
        boxlist,
        self.nms_thresh,
        max_proposals=self.post_nms_top_n,
        score_field="objectness",
      )
      result.append(boxlist)
    return result

  def __call__(self, anchors, objectness, box_regression, targets=None):
    sampled_boxes = []
    num_levels = len(objectness)
    anchors = list(zip(*anchors))
    for a, o, b in zip(anchors, objectness, box_regression):
      sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

    boxlists = list(zip(*sampled_boxes))
    boxlists = [mask_rcnn.cat_boxlist(boxlist) for boxlist in boxlists]

    if num_levels > 1:
      boxlists = self.select_over_all_levels(boxlists)

    if targets is not None:
      boxlists = self.add_gt_proposals(boxlists, targets)

    return boxlists

  def select_over_all_levels(self, boxlists):
    num_images = len(boxlists)
    for i in range(num_images):
      objectness = boxlists[i].get_field("objectness")
      post_nms_top_n = min(self.fpn_post_nms_top_n, objectness.shape[0])
      _, inds_sorted = mask_rcnn.topk(objectness,
        post_nms_top_n, dim=0, sorted=False
      )
      boxlists[i] = boxlists[i][inds_sorted]
    return boxlists
 
  def add_gt_proposals(self, proposals, targets):
    gt_boxes = [target.copy_with_fields([]) for target in targets]

    # later cat of bbox requires all fields to be present for all bbox
    # so we need to add a dummy for objectness that's missing
    for gt_box in gt_boxes:
      gt_box.add_field("objectness", Tensor.ones(len(gt_box)))

    proposals = [mask_rcnn.cat_boxlist((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]
    return proposals


class RPN:
  def __init__(self, in_channels):
    self.anchor_generator = AnchorGenerator()

    in_channels = 256
    head = RPNHead(
      in_channels, self.anchor_generator.num_anchors_per_location()[0]
    )
    box_coder = mask_rcnn.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    box_selector_train = RPNPostProcessor(
      pre_nms_top_n=2000,
      post_nms_top_n=2000,
      nms_thresh=0.7,
      min_size=0,
      box_coder=box_coder,
      fpn_post_nms_top_n=2000
    )
    box_selector_test = RPNPostProcessor(
        pre_nms_top_n=1000,
        post_nms_top_n=1000,
        nms_thresh=0.7,
        min_size=0,
        box_coder=box_coder,
        fpn_post_nms_top_n=1000
    )
    self.head = head
    self.box_selector_train = box_selector_train
    self.box_selector_test = box_selector_test
    self.loss_evaluator = mask_rcnn.create_rpn_loss_evaluator(box_coder)

  def __call__(self, images, features, targets=None):
    objectness, rpn_box_regression = self.head(features)
    anchors = self.anchor_generator(images, features)
    if targets is not None:
      with Tensor.train(val=False):
        boxes = self.box_selector_train(anchors, objectness, rpn_box_regression, targets=targets)

      loss_objectness, loss_rpn_box_neg = self.loss_evaluator(anchors, objectness, rpn_box_regression, targets)
      return boxes, {"loss_objectness": loss_objectness, "loss_rpn_box_neg": loss_rpn_box_neg}
    else:
      boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
      return boxes, {}


class MaskRCNNFPNFeatureExtractor:
  def __init__(self):
    resolution = 14
    scales = (0.25, 0.125, 0.0625, 0.03125)
    sampling_ratio = 2
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
    self.mask_fcn1 = _make_conv3x3(input_size, layers[0], dilation=dilation, stride=1, use_gn=use_gn)
    self.mask_fcn2 = _make_conv3x3(layers[0], layers[1], dilation=dilation, stride=1, use_gn=use_gn)
    self.mask_fcn3 = _make_conv3x3(layers[1], layers[2], dilation=dilation, stride=1, use_gn=use_gn)
    self.mask_fcn4 = _make_conv3x3(layers[2], layers[3], dilation=dilation, stride=1, use_gn=use_gn)
    self.blocks = [self.mask_fcn1, self.mask_fcn2, self.mask_fcn3, self.mask_fcn4]

  def __call__(self, x, proposals):
    x = self.pooler(x, proposals)
    for layer in self.blocks:
      if x is not None:
        x = Tensor.relu(layer(x))
    return x


class MaskRCNNC4Predictor:
  def __init__(self):
    num_classes = 81
    dim_reduced = 256
    num_inputs = dim_reduced
    self.conv5_mask = nn.ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
    self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)

  def __call__(self, x):
    x = Tensor.relu(self.conv5_mask(x))
    return self.mask_fcn_logits(x)


class FPN2MLPFeatureExtractor:
  def __init__(self, cfg):
    resolution = 7
    scales = (0.25, 0.125, 0.0625, 0.03125)
    sampling_ratio = 2
    pooler = Pooler(
      output_size=(resolution, resolution),
      scales=scales,
      sampling_ratio=sampling_ratio,
    )
    input_size = 256 * resolution ** 2
    representation_size = 1024
    self.pooler = pooler
    self.fc6 = nn.Linear(input_size, representation_size)
    self.fc7 = nn.Linear(representation_size, representation_size)

  def __call__(self, x, proposals):
    x = self.pooler(x, proposals)
    x = x.reshape(x.shape[0], -1)
    x = Tensor.relu(self.fc6(x))
    x = Tensor.relu(self.fc7(x))
    return x


class ROIAlign:
  def __init__(self, output_size, spatial_scale, sampling_ratio):
    self.output_size = output_size
    self.spatial_scale = spatial_scale
    self.sampling_ratio = sampling_ratio

  def __call__(self, input, rois):
    output = _roi_align(
      input, rois, self.spatial_scale, self.output_size[0], self.output_size[1], self.sampling_ratio, aligned=False
    )
    return output


class LevelMapper:
  def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
    self.k_min = k_min
    self.k_max = k_max
    self.s0 = canonical_scale
    self.lvl0 = canonical_level
    self.eps = eps

  def __call__(self, boxlists):
    s = Tensor.sqrt(Tensor.cat(*[boxlist.area() for boxlist in boxlists]))
    target_lvls = (self.lvl0 + Tensor.log2(s / self.s0 + self.eps)).floor()
    target_lvls = target_lvls.clip(min_=self.k_min, max_=self.k_max)
    return target_lvls - self.k_min


class Pooler:
  def __init__(self, output_size, scales, sampling_ratio):
    self.output_size = output_size
    self.scales = scales
    self.sampling_ratio = sampling_ratio
    poolers = []
    for scale in scales:
      poolers.append(
        ROIAlign(
          output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
        )
      )
    self.poolers = poolers
    self.output_size = output_size
    lvl_min = -math.log2(scales[0])
    lvl_max = -math.log2(scales[-1])
    self.map_levels = LevelMapper(lvl_min, lvl_max)

  def convert_to_roi_format(self, boxes):
    concat_boxes = Tensor.cat(*[b.bbox for b in boxes], dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = Tensor.cat(
      *[
        Tensor.full((len(b), 1), i, dtype=dtype, device=device)
        for i, b in enumerate(boxes)
      ],
      dim=0,
    )
    if concat_boxes.shape[0] != 0:
      rois = Tensor.cat(*[ids, concat_boxes], dim=1)
      return rois

  def __call__(self, x, boxes):
    num_levels = len(self.poolers)
    rois = self.convert_to_roi_format(boxes)
    if rois:
      if num_levels == 1:
        return self.poolers[0](x[0], rois)

      levels = self.map_levels(boxes)
      results = []
      all_idxs = []
      for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
        # this is fine because no grad will flow through index
        idx_in_level = (levels.numpy() == level).nonzero()[0]
        if len(idx_in_level) > 0:
          rois_per_level = mask_rcnn.tensor_gather(rois, idx_in_level)
          pooler_output = pooler(per_level_feature, rois_per_level)
          all_idxs.extend(idx_in_level)
          results.append(pooler_output)

      return mask_rcnn.tensor_gather(Tensor.cat(*results), [x[0] for x in sorted({i:idx for i, idx in enumerate(all_idxs)}.items(), key=lambda x: x[1])])


class FPNPredictor:
  def __init__(self):
    num_classes = 81
    representation_size = 1024
    self.cls_score = nn.Linear(representation_size, num_classes)
    num_bbox_reg_classes = num_classes
    self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

  def __call__(self, x):
    scores = self.cls_score(x)
    bbox_deltas = self.bbox_pred(x)
    return scores, bbox_deltas


class PostProcessor:
  # Not used in training
  def __init__(
          self,
          score_thresh=0.05,
          nms=0.5,
          detections_per_img=100,
          box_coder=None,
          cls_agnostic_bbox_reg=False
  ):
    self.score_thresh = score_thresh
    self.nms = nms
    self.detections_per_img = detections_per_img
    if box_coder is None:
      box_coder = mask_rcnn.BoxCoder(weights=(10., 10., 5., 5.))
    self.box_coder = box_coder
    self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

  def __call__(self, x, boxes):
    class_logits, box_regression = x
    class_prob = Tensor.softmax(class_logits, -1)
    image_shapes = [box.size for box in boxes]
    boxes_per_image = [len(box) for box in boxes]
    concat_boxes = Tensor.cat(*[a.bbox for a in boxes], dim=0)

    if self.cls_agnostic_bbox_reg:
      box_regression = box_regression[:, -4:]
    proposals = self.box_coder.decode(
      box_regression.reshape(sum(boxes_per_image), -1), concat_boxes
    )
    if self.cls_agnostic_bbox_reg:
      proposals = proposals.repeat([1, class_prob.shape[1]])
    num_classes = class_prob.shape[1]
    proposals = proposals.unsqueeze(0)
    class_prob = class_prob.unsqueeze(0)
    results = []
    for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
    ):
      boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
      boxlist = boxlist.clip_to_image(remove_empty=False)
      boxlist = self.filter_results(boxlist, num_classes)
      results.append(boxlist)
    return results

  def prepare_boxlist(self, boxes, scores, image_shape):
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    boxlist = mask_rcnn.BoxList(boxes, image_shape, mode="xyxy")
    boxlist.add_field("scores", scores)
    return boxlist

  def filter_results(self, boxlist, num_classes):
    boxes = boxlist.bbox.reshape(-1, num_classes * 4)
    scores = boxlist.get_field("scores").reshape(-1, num_classes)

    device = scores.device
    result = []
    scores = scores.numpy()
    boxes = boxes.numpy()
    inds_all = scores > self.score_thresh
    for j in range(1, num_classes):
      inds = inds_all[:, j].nonzero()[0]
      # This needs to be done in numpy because it can create empty arrays
      scores_j = scores[inds, j]
      boxes_j = boxes[inds, j * 4: (j + 1) * 4]
      boxes_j = Tensor(boxes_j)
      scores_j = Tensor(scores_j)
      boxlist_for_class = mask_rcnn.BoxList(boxes_j, boxlist.size, mode="xyxy")
      boxlist_for_class.add_field("scores", scores_j)
      if len(boxlist_for_class):
        boxlist_for_class = mask_rcnn.boxlist_nms(
          boxlist_for_class, self.nms
        )
      num_labels = len(boxlist_for_class)
      boxlist_for_class.add_field(
        "labels", Tensor.full((num_labels,), j, device=device)
      )
      result.append(boxlist_for_class)

    result = mask_rcnn.at_boxlist(result)
    number_of_detections = len(result)

    if number_of_detections > self.detections_per_img > 0:
      cls_scores = result.get_field("scores")
      image_thresh, _ = mask_rcnn.topk(cls_scores, k=self.detections_per_img)
      image_thresh = image_thresh.numpy()[-1]
      keep = (cls_scores.numpy() >= image_thresh).nonzero()[0]
      result = result[keep]
    return result


class RoIBoxHead:
  def __init__(self, in_channels):
    box_coder = mask_rcnn.BoxCoder((10., 10., 5., 5.))
    self.feature_extractor = FPN2MLPFeatureExtractor(in_channels)
    self.predictor = FPNPredictor()
    self.post_processor = PostProcessor(
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=box_coder,
        cls_agnostic_bbox_reg=False
    )
    self.loss_evaluator = mask_rcnn.create_fast_rcnn_loss_evaluator(box_coder)

  def __call__(self, features, proposals, targets=None):
    if targets is not None:
      with Tensor.train(val=False):
        proposals = self.loss_evaluator.subsample(proposals, targets)

    x = self.feature_extractor(features, proposals)
    class_logits, box_regression = self.predictor(x)

    if not Tensor.training:
      result = self.post_processor((class_logits, box_regression), proposals)
      return x, result, {}
    
    loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression])
    return x, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)


class MaskPostProcessor:
  # Not used in loss calculation
  def __call__(self, x, boxes):
    mask_prob = x.sigmoid().numpy()
    num_masks = x.shape[0]
    labels = [bbox.get_field("labels") for bbox in boxes]
    labels = Tensor.cat(*labels).numpy().astype(np.int32)
    index = np.arange(num_masks)
    mask_prob = mask_prob[index, labels][:, None]
    boxes_per_image, cumsum = [], 0
    for box in boxes:
      cumsum += len(box)
      boxes_per_image.append(cumsum)
    # using numpy here as Tensor.chunk doesnt have custom chunk sizes
    mask_prob = np.split(mask_prob, boxes_per_image, axis=0)
    results = []
    for prob, box in zip(mask_prob, boxes):
      bbox = mask_rcnn.BoxList(box.bbox, box.size, mode="xyxy")
      for field in box.fields():
        bbox.add_field(field, box.get_field(field))
      prob = Tensor(prob)
      bbox.add_field("mask", prob)
      results.append(bbox)

    return results


class Mask:
  def __init__(self):
    self.feature_extractor = MaskRCNNFPNFeatureExtractor()
    self.predictor = MaskRCNNC4Predictor()
    self.post_processor = MaskPostProcessor()
    self.loss_evaluator = mask_rcnn.create_mask_rcnn_loss_evaluator()

  def __call__(self, features, proposals, targets=None):
    if targets is not None:
      all_proposals = proposals
      proposals, _ = _keep_only_positive_boxes(proposals)

    x = self.feature_extractor(features, proposals)
    if x:
      mask_logits = self.predictor(x)
      # TODO: Fix this issue when we start to introduce SegmentationMasks
      # if targets is not None:
      #   loss_mask = self.loss_evaluator(proposals, mask_logits, targets)
      #   return x, all_proposals, dict(loss_mask=loss_mask)
      # else:
      result = self.post_processor(mask_logits, proposals)
      return x, result, {}


class RoIHeads:
  def __init__(self, in_channels):
    self.box = RoIBoxHead(in_channels)
    self.mask = Mask()

  def __call__(self, features, proposals, targets=None):
    x, detections, loss_box = self.box(features, proposals, targets)
    x, detections, loss_mask = self.mask(features, detections, targets)
    return x, detections, dict(loss_box=loss_box, loss_mask=loss_mask)


class ImageList(object):
  def __init__(self, tensors, image_sizes):
    self.tensors = tensors
    self.image_sizes = image_sizes

  def to(self, *args, **kwargs):
    cast_tensor = self.tensors.to(*args, **kwargs)
    return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=32):
  # Preprocessing
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
    if size_divisible > 0:

      stride = size_divisible
      max_size = list(max_size)
      max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
      max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
      max_size = tuple(max_size)

    batch_shape = (len(tensors),) + max_size
    batched_imgs = np.zeros(batch_shape, dtype=tensors[0].numpy().dtype)
    for img, pad_img in zip(tensors, batched_imgs):
      pad_img[: img.shape[0], : img.shape[1], : img.shape[2]] += img.numpy()

    batched_imgs = Tensor(batched_imgs)
    image_sizes = [im.shape[-2:] for im in tensors]

    return ImageList(batched_imgs, image_sizes)
  else:
    raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


class MaskRCNN:
  def __init__(self, backbone: ResNet):
    self.backbone = ResNetFPN(backbone, out_channels=256)
    self.rpn = RPN(self.backbone.out_channels)
    self.roi_heads = RoIHeads(self.backbone.out_channels)

  def load_from_pretrained(self):
    fn = Path('./') / "weights/maskrcnn.pt"
    fetch("https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth", fn)

    state_dict = torch_load(fn)['model']
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
      loaded_keys.append(k)
      get_child(self, k).assign(v.numpy()).realize()
    return loaded_keys

  def __call__(self, images, targets=None):
    if Tensor.training and targets is None:
      raise ValueError("In training mode, targets should be passed")

    images = to_image_list(images)
    features = self.backbone(images.tensors)
    proposals, proposal_losses = self.rpn(images, features, targets=targets)
    x, result, detector_losses = self.roi_heads(features, proposals, targets=targets)

    if targets is not None: return dict(detector_losses=detector_losses, proposal_losses=proposal_losses)
    return result


if __name__ == '__main__':
  resnet = resnet = ResNet(50, num_classes=None, stride_in_1x1=True)
  model = MaskRCNN(backbone=resnet)
  model.load_from_pretrained()
