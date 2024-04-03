import math
import sys
from tinygrad import Tensor, dtypes
from tinygrad.helpers import colored, flatten, get_child
import tinygrad.nn as nn
from extra.models.resnet import ResNet
import numpy as np

def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    mask: Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
):
  # print(colored(f'ENTERED SIMOID_LOSS {mask.shape} {mask.sum().numpy()} {inputs.shape} {targets.shape}', 'magenta'))
  # print(inputs.numpy())
  # print(mask.numpy())
  # p = inputs.sigmoid()
  p = Tensor.sigmoid(inputs)
  ce_loss = inputs.binary_crossentropy_logits(targets)
  p_t = p * targets + (1 - p) * (1 - targets)
  loss = ce_loss * ((1 - p_t) ** gamma)

  if alpha >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
  # print(colored(f'ENTERED SIMOID_LOSS {loss.shape}', 'green'))
  
  # loss = loss * mask
  # Reducing with sum instead of mean
  loss = loss.sum()

  # print('sigmoid_focal_loss:', loss.shape, loss.item())
  # sys.exit()
  return loss
def l1_loss(x1:Tensor, x2:Tensor) -> Tensor:
  # print('l1 loss inputs', x1.numpy(), x2.numpy())
  ans = (x1 - x2).abs().mean().sum()
  # print('l1 loss', ans.numpy())
  return ans
def _sum(x):
  # List[Tensor]
  res = x[0]
  for i in x[1:]:
    res = res + i
  
  return res
def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
  # print('BOX_IOU Arguements', boxes1.shape, boxes2.shape)

  def box_area(boxes): return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
  area1 = box_area(boxes1)
  area2 = box_area(boxes2)
  lt = Tensor.maximum(boxes1[:, :2].unsqueeze(1), boxes2[:, :2])  # [N,M,2]
  rb = Tensor.minimum(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:])  # [N,M,2]
  # wh = (rb - lt).clip(min_=0)  # [N,M,2]
  wh = (rb - lt).maximum(0)  # [N,M,2]
  inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
  union = area1[:, None] + area2 - inter
  iou = inter / union
  # print('BOx_IOU Ret Shape:', iou.shape)

  return iou.realize()

def nms(boxes, scores, thresh=0.5):
  x1, y1, x2, y2 = np.rollaxis(boxes, 1)
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  to_process, keep = scores.argsort()[::-1], []
  while to_process.size > 0:
    cur, to_process = to_process[0], to_process[1:]
    keep.append(cur)
    inter_x1 = np.maximum(x1[cur], x1[to_process])
    inter_y1 = np.maximum(y1[cur], y1[to_process])
    inter_x2 = np.minimum(x2[cur], x2[to_process])
    inter_y2 = np.minimum(y2[cur], y2[to_process])
    inter_area = np.maximum(0, inter_x2 - inter_x1 + 1) * np.maximum(0, inter_y2 - inter_y1 + 1)
    iou = inter_area / (areas[cur] + areas[to_process] - inter_area)
    to_process = to_process[np.where(iou <= thresh)[0]]
  return keep

def decode_bbox(offsets, anchors):
  dx, dy, dw, dh = np.rollaxis(offsets, 1)
  widths, heights = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
  cx, cy = anchors[:, 0] + 0.5 * widths, anchors[:, 1] + 0.5 * heights
  pred_cx, pred_cy = dx * widths + cx, dy * heights + cy
  pred_w, pred_h = np.exp(dw) * widths, np.exp(dh) * heights
  pred_x1, pred_y1 = pred_cx - 0.5 * pred_w, pred_cy - 0.5 * pred_h
  pred_x2, pred_y2 = pred_cx + 0.5 * pred_w, pred_cy + 0.5 * pred_h
  return np.stack([pred_x1, pred_y1, pred_x2, pred_y2], axis=1, dtype=np.float32)
def encode_boxes(reference_boxes, proposals, weights = (1.0,1.0,1.0,1.0)):
  # print('Encode BOx', reference_boxes.shape, proposals.shape)
  # print(reference_boxes.numpy())
  # print(proposals.numpy())
  
  # sys.exit()
  # type: (Tensor, Tensor, Tensor) -> Tensor
  """
  Encode a set of proposals with respect to some
  reference boxes

  Args:
      reference_boxes (Tensor): reference boxes
      proposals (Tensor): boxes to be encoded
      weights (Tensor[4]): the weights for ``(x, y, w, h)``
  """

  # perform some unpacking to make it JIT-fusion friendly
  wx = weights[0]
  wy = weights[1]
  ww = weights[2]
  wh = weights[3]

  proposals_x1 = proposals[:, 0].unsqueeze(1)
  proposals_y1 = proposals[:, 1].unsqueeze(1)
  proposals_x2 = proposals[:, 2].unsqueeze(1)
  proposals_y2 = proposals[:, 3].unsqueeze(1)
  # print('proposals:', proposals_x1.numpy(), proposals_x2.numpy(), proposals_y1.numpy(),
  #       proposals_y2.numpy())

  reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
  reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
  reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
  reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)
  # print('reference boxes:', reference_boxes_x1.numpy(), reference_boxes_x2.numpy(),
        # reference_boxes_y1.numpy(), reference_boxes_y2.numpy())

  # implementation starts here
  ex_widths = proposals_x2 - proposals_x1
  ex_heights = proposals_y2 - proposals_y1
  ex_ctr_x = proposals_x1 + 0.5 * ex_widths
  ex_ctr_y = proposals_y1 + 0.5 * ex_heights
  # print('exw,exh,excx,excy:',ex_widths.numpy(),ex_heights.numpy(),ex_ctr_x.numpy(),ex_ctr_y.numpy())

  gt_widths = reference_boxes_x2 - reference_boxes_x1
  gt_heights = reference_boxes_y2 - reference_boxes_y1
  gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
  gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights
  # when these all are 0, results in inf
  # print('gtw,gth,gtcx,gtcy', gt_widths.numpy(), gt_heights.numpy(), gt_ctr_x.numpy(), gt_ctr_y.numpy())

  # if gt_widths.item() == 0.0:
  #   print('replace gt_width/height hack hit')
  #   gt_widths = Tensor(3.)
  #   gt_heights = Tensor(3.)
  targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
  targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
  targets_dw = ww * (gt_widths / ex_widths).log()
  targets_dh = wh * (gt_heights / ex_heights).log()

  targets = Tensor.cat(*(targets_dx, targets_dy, targets_dw, targets_dh), dim=1)#.realize()
  # print('Encode BOX RETURN', targets.shape)
  return targets
def generate_anchors(input_size, grid_sizes, scales, aspect_ratios):
  assert len(scales) == len(aspect_ratios) == len(grid_sizes)
  anchors = []
  for s, ar, gs in zip(scales, aspect_ratios, grid_sizes):
    s, ar = np.array(s), np.array(ar)
    h_ratios = np.sqrt(ar)
    w_ratios = 1 / h_ratios
    ws = (w_ratios[:, None] * s[None, :]).reshape(-1)
    hs = (h_ratios[:, None] * s[None, :]).reshape(-1)
    base_anchors = (np.stack([-ws, -hs, ws, hs], axis=1) / 2).round()
    stride_h, stride_w = input_size[0] // gs[0], input_size[1] // gs[1]
    shifts_x, shifts_y = np.meshgrid(np.arange(gs[1]) * stride_w, np.arange(gs[0]) * stride_h)
    shifts_x = shifts_x.reshape(-1)
    shifts_y = shifts_y.reshape(-1)
    shifts = np.stack([shifts_x, shifts_y, shifts_x, shifts_y], axis=1, dtype=np.float32)
    anchors.append((shifts[:, None] + base_anchors[None, :]).reshape(-1, 4))
  return anchors

import torch
class Matcher(object):
  """
  This class assigns to each predicted "element" (e.g., a box) a ground-truth
  element. Each predicted element will have exactly zero or one matches; each
  ground-truth element may be assigned to zero or more predicted elements.

  Matching is based on the MxN match_quality_matrix, that characterizes how well
  each (ground-truth, predicted)-pair match. For example, if the elements are
  boxes, the matrix may contain box IoU overlap values.

  The matcher returns a tensor of size N containing the index of the ground-truth
  element m that matches to prediction n. If there is no match, a negative value
  is returned.
  """

  BELOW_LOW_THRESHOLD = -1
  BETWEEN_THRESHOLDS = -2

  __annotations__ = {
      'BELOW_LOW_THRESHOLD': int,
      'BETWEEN_THRESHOLDS': int,
  }

  def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
    # type: (float, float, bool) -> None
    """
    Args:
        high_threshold (float): quality values greater than or equal to
            this value are candidate matches.
        low_threshold (float): a lower quality threshold used to stratify
            matches into three levels:
            1) matches >= high_threshold
            2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
            3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
        allow_low_quality_matches (bool): if True, produce additional matches
            for predictions that have only low-quality match candidates. See
            set_low_quality_matches_ for more details.
    """
    self.BELOW_LOW_THRESHOLD = -1
    self.BETWEEN_THRESHOLDS = -2
    assert low_threshold <= high_threshold
    self.high_threshold = high_threshold
    self.low_threshold = low_threshold
    self.allow_low_quality_matches = allow_low_quality_matches

  def __call__(self, match_quality_matrix):
    # print('MATCHER ARG SIZE:', match_quality_matrix.shape)
    # with torch.no_grad():
    # match_quality_matrix_np = torch.as_tensor(match_quality_matrix.numpy())
    match_quality_matrix_np = match_quality_matrix.numpy()

    """
    Args:
        match_quality_matrix (Tensor[float]): an MxN tensor, containing the
        pairwise quality between M ground-truth elements and N predicted elements.

    Returns:
        matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
        [0, M - 1] or a negative value indicating that prediction i could not
        be matched.
    """
    if match_quality_matrix.numel() == 0:
      # empty targets or proposals not supported during training
      if match_quality_matrix.shape[0] == 0:
        raise ValueError(
            "No ground-truth boxes available for one of the images "
            "during training")
      else:
        raise ValueError(
            "No proposal boxes available for one of the images "
              "during training")

    # match_quality_matrix is M (gt) x N (predicted)
    # Max over gt elements (dim 0) to find best gt candidate for each prediction
    matched_vals = match_quality_matrix_np.max(axis=0)
    matches = match_quality_matrix_np.argmax(axis=0)
    if self.allow_low_quality_matches:
      # all_matches = matches.clone()
      all_matches = np.copy(matches)
    else:
      all_matches = None

    # Assign candidate matches with low quality to negative (unassigned) values
    below_low_threshold = matched_vals < self.low_threshold
    between_thresholds = (matched_vals >= self.low_threshold) & (
        matched_vals < self.high_threshold
    )
    matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
    matches[between_thresholds] = self.BETWEEN_THRESHOLDS

    if self.allow_low_quality_matches:
      # assert all_matches is not None
      matches = self.set_low_quality_matches_(matches, all_matches, match_quality_matrix_np)
    matches_tens = Tensor(matches).realize()
    del all_matches, matches, match_quality_matrix_np, below_low_threshold, between_thresholds
    return matches_tens

  def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
    """
    Produce additional matches for predictions that have only low-quality matches.
    Specifically, for each ground-truth find the set of predictions that have
    maximum overlap with it (including ties); for each prediction in that set, if
    it is unmatched, then match it to the ground-truth with which it has the highest
    quality value.
    """
    # For each gt, find the prediction with which it has highest quality
    highest_quality_foreach_gt = match_quality_matrix.max(axis=1)
    # Find highest quality match available, even if it is low, including ties
    
    gt_pred_pairs_of_highest_quality = np.nonzero(
        match_quality_matrix == highest_quality_foreach_gt[:, None]
    )
    # Example gt_pred_pairs_of_highest_quality:
    #   tensor([[    0, 39796],
    #           [    1, 32055],
    #           [    1, 32070],
    #           [    2, 39190],
    #           [    2, 40255],
    #           [    3, 40390],
    #           [    3, 41455],
    #           [    4, 45470],
    #           [    5, 45325],
    #           [    5, 46390]])
    # Each row is a (gt index, prediction index)
    # Note how gt items 1, 2, 3, and 5 each have two ties

    pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
    matches[pred_inds_to_update] = all_matches[pred_inds_to_update]
    del gt_pred_pairs_of_highest_quality, highest_quality_foreach_gt
    return matches
class RetinaNet:
  def __init__(self, backbone: ResNet, num_classes=264, num_anchors=9, scales=None, aspect_ratios=None,fg_iou_thresh=0.5, bg_iou_thresh=0.4):
    assert isinstance(backbone, ResNet)
    scales = tuple((i, int(i*2**(1/3)), int(i*2**(2/3))) for i in 2**np.arange(5, 10)) if scales is None else scales
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(scales) if aspect_ratios is None else aspect_ratios
    self.num_anchors, self.num_classes = num_anchors, num_classes
    assert len(scales) == len(aspect_ratios) and all(self.num_anchors == len(s) * len(ar) for s, ar in zip(scales, aspect_ratios))

    self.backbone = ResNetFPN(backbone)
    self.head = RetinaHead(self.backbone.out_channels, num_anchors=num_anchors, num_classes=num_classes)
    self.anchor_gen = lambda input_size: generate_anchors(input_size, self.backbone.compute_grid_sizes(input_size), scales, aspect_ratios)
    self.proposal_matcher =  Matcher(
                    fg_iou_thresh,
                    bg_iou_thresh,
                    allow_low_quality_matches=True,
                )
  def __call__(self, x):
    
    b = self.backbone(x)
    r, c = self.head(b)
    if Tensor.training:
      # l = self.loss_temp(c)
      # l = self.loss(r, c, Y, anchor_gen(x, b))
      # return l 
      return b, r, c
    else: 
      return r.cat(c.sigmoid(), dim=-1)
  def forward(self, x):
    return self.head(self.backbone(x))
  def loss_temp(self, logits) -> Tensor:
    return logits.sum()
  def loss(self, logits_reg, logits_class, targets, anchors) -> Tensor:
    # print(colored(f'RNET_LOSS SHAPES {logits_reg.shape} {logits_class.shape}','green'))
    matched_idxs = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
      if targets_per_image['boxes'].numel() == 0:
        print('NUMEL==0 HIT!!!')
        matched_idxs.append(Tensor.full((anchors_per_image.shape[0],), -1, dtype=dtypes.int64,))
                                        # device=anchors_per_image.device))
        continue

      match_quality_matrix = box_iou(targets_per_image['boxes'], anchors_per_image)
      # print('match_quality_matrix', match_quality_matrix.shape)
      # print(match_quality_matrix.numpy())
      matched_idxs.append(self.proposal_matcher(match_quality_matrix))
      # print(colored(f'PROP MATCHER {matched_idxs[-1].shape}', 'magenta'))
      # print('matcher apppend:', matched_idxs[-1].shape)

      # sys.exit()
      # matched_idxs.append(self.proposal_matcher(match_quality_matrix))
    # return logits_class.sum()
    loss_class = self.head.classification_head.loss(logits_class, targets, matched_idxs)
    # return loss_class
    loss_reg = self.head.regression_head.loss(logits_reg, targets, anchors, matched_idxs)
    # return loss_reg
    # print(colored(f'FINISHED CLASS LOSS FINAL COMPUTE {loss_reg}|||', 'green'))
    # https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/engine.py#L36
    
    print(colored(f'loss_reg {loss_reg.numpy()}', 'green'))
    print(colored(f'loss_class {loss_class.numpy()}', 'green'))
    # return loss_class
    # return loss_reg
    return (loss_reg+loss_class)
  def load_from_pretrained(self):
    model_urls = {
      (50, 1, 64): "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
      (50, 32, 4): "https://zenodo.org/record/6605272/files/retinanet_model_10.zip",
    }
    self.url = model_urls[(self.backbone.body.num, self.backbone.body.groups, self.backbone.body.base_width)]
    from torch.hub import load_state_dict_from_url
    state_dict = load_state_dict_from_url(self.url, progress=True, map_location='cpu')
    state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
    for k, v in state_dict.items():
      obj = get_child(self, k)
      dat = v.detach().numpy()
      assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
      obj.assign(dat)

  # predictions: (BS, (H1W1+...+HmWm)A, 4 + K)
  def postprocess_detections(self, predictions, input_size=(800, 800), image_sizes=None, orig_image_sizes=None, score_thresh=0.05, topk_candidates=1000, nms_thresh=0.5):
    anchors = self.anchor_gen(input_size)
    grid_sizes = self.backbone.compute_grid_sizes(input_size)
    split_idx = np.cumsum([int(self.num_anchors * sz[0] * sz[1]) for sz in grid_sizes[:-1]])
    detections = []
    for i, predictions_per_image in enumerate(predictions):
      h, w = input_size if image_sizes is None else image_sizes[i]

      predictions_per_image = np.split(predictions_per_image, split_idx)
      offsets_per_image = [br[:, :4] for br in predictions_per_image]
      scores_per_image = [cl[:, 4:] for cl in predictions_per_image]

      image_boxes, image_scores, image_labels = [], [], []
      for offsets_per_level, scores_per_level, anchors_per_level in zip(offsets_per_image, scores_per_image, anchors):
        # remove low scoring boxes
        scores_per_level = scores_per_level.flatten()
        keep_idxs = scores_per_level > score_thresh
        scores_per_level = scores_per_level[keep_idxs]

        # keep topk
        topk_idxs = np.where(keep_idxs)[0]
        num_topk = min(len(topk_idxs), topk_candidates)
        sort_idxs = scores_per_level.argsort()[-num_topk:][::-1]
        topk_idxs, scores_per_level = topk_idxs[sort_idxs], scores_per_level[sort_idxs]

        # bbox coords from offsets
        anchor_idxs = topk_idxs // self.num_classes
        labels_per_level = topk_idxs % self.num_classes
        boxes_per_level = decode_bbox(offsets_per_level[anchor_idxs], anchors_per_level[anchor_idxs])
        # clip to image size
        clipped_x = boxes_per_level[:, 0::2].clip(0, w)
        clipped_y = boxes_per_level[:, 1::2].clip(0, h)
        boxes_per_level = np.stack([clipped_x, clipped_y], axis=2).reshape(-1, 4)

        image_boxes.append(boxes_per_level)
        image_scores.append(scores_per_level)
        image_labels.append(labels_per_level)

      image_boxes = np.concatenate(image_boxes)
      image_scores = np.concatenate(image_scores)
      image_labels = np.concatenate(image_labels)

      # nms for each class
      keep_mask = np.zeros_like(image_scores, dtype=bool)
      for class_id in np.unique(image_labels):
        curr_indices = np.where(image_labels == class_id)[0]
        curr_keep_indices = nms(image_boxes[curr_indices], image_scores[curr_indices], nms_thresh)
        keep_mask[curr_indices[curr_keep_indices]] = True
      keep = np.where(keep_mask)[0]
      keep = keep[image_scores[keep].argsort()[::-1]]

      # resize bboxes back to original size
      image_boxes = image_boxes[keep]
      if orig_image_sizes is not None:
        resized_x = image_boxes[:, 0::2] * orig_image_sizes[i][1] / w
        resized_y = image_boxes[:, 1::2] * orig_image_sizes[i][0] / h
        image_boxes = np.stack([resized_x, resized_y], axis=2).reshape(-1, 4)
      # xywh format
      image_boxes = np.concatenate([image_boxes[:, :2], image_boxes[:, 2:] - image_boxes[:, :2]], axis=1)

      detections.append({"boxes":image_boxes, "scores":image_scores[keep], "labels":image_labels[keep]})
    return detections

class ClassificationHead:
  def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01):
    self.num_classes = num_classes
    self.conv = flatten([(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), lambda x: x.relu()) for _ in range(4)])
    # print(len(self.conv))
    for i in self.conv:
      if isinstance(i, nn.Conv2d):
        i.weight = Tensor.normal(i.weight.shape, std=0.01).realize()
        i.bias = Tensor.full(i.bias.shape, 0.0).realize()

    self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
    print('CLASS ININTTTTTTT')
    # print(self.cls_logits.weight.numpy())
    self.cls_logits.weight = Tensor.normal(self.cls_logits.weight.shape, std=0.01).realize()
    self.cls_logits.bias = Tensor.full(self.cls_logits.bias.shape, -math.log((1 - prior_probability) / prior_probability)).realize()
    # print(self.cls_logits.bias.shape)
    # print(self.cls_logits.bias.numpy())
    # print(self.cls_logits.weight.numpy())

  def __call__(self, x):
    out = [self.cls_logits(feat.sequential(self.conv)).permute(0, 2, 3, 1).reshape(feat.shape[0], -1, self.num_classes) for feat in x]
    return out[0].cat(*out[1:], dim=1)#.sigmoid()
  def loss(self, logits, targets, matched_idxs):
    losses = []
    for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, logits, matched_idxs):
      # print(colored(f'matched_idxs_per_image CLASS {matched_idxs_per_image.shape} {cls_logits_per_image.shape}', 'red' ))
      # determine only the foreground
      foreground_idxs_per_image = matched_idxs_per_image >= 0
      num_foreground = foreground_idxs_per_image.sum()
      
      print('num_foreground:',num_foreground.shape, num_foreground.numpy())
      # Idk if this hack works
      foreground_idxs_per_image_np = np.nonzero(foreground_idxs_per_image.numpy())[0]
      # if foreground_idxs_per_image.shape==(0,):
      #   print(colored('empty forground idx in class head', 'red'))
      #   foreground_idxs_per_image = Tensor([0])
      #   # foreground_idxs_per_image = Tensor([])
      # else:
      foreground_idxs_per_image = Tensor(foreground_idxs_per_image_np)
      del foreground_idxs_per_image_np
      # print(colored(f'foreground_idxs_per_image CLASS{foreground_idxs_per_image.shape}', 'red' ))

      
      # create the target classification

      # print('cls_logits_per_image:', cls_logits_per_image.shape)
      gt_classes_target_np = np.zeros_like(cls_logits_per_image.numpy())

      # gt_classes_target = torch.zeros(cls_logits_per_image.shape)
      gt_classes_target_np[
          foreground_idxs_per_image.numpy(),
          targets_per_image['labels'].numpy()[matched_idxs_per_image.numpy()[foreground_idxs_per_image.numpy()]]
      ] = 1.0
      # print('gt_classes_target shape NP VERSION:', gt_classes_target.shape)
      # gt_classes_target = Tensor(gt_classes_target)

      # print(colored(f'TORCH gt_classes_target {type(gt_classes_target_np)} {gt_classes_target_np.shape}','yellow'))
      
      # gt_classes_target = Tensor(gt_classes_target.numpy())
      gt_classes_target = Tensor(gt_classes_target_np)#.realize()
      del gt_classes_target_np

      # print('Class_head matched_idxs_per_image', matched_idxs_per_image.numpy())
      valid_idxs_per_image = matched_idxs_per_image != Matcher.BETWEEN_THRESHOLDS
      # print(colored(f'PREE valid idx {valid_idxs_per_image.shape} {valid_idxs_per_image.numpy()}', 'yellow'))
      
      valid_idxs_per_image = valid_idxs_per_image.reshape(-1,1)
      # print(colored(f'valid idx {valid_idxs_per_image.shape} {valid_idxs_per_image.numpy()}', 'yellow'))
      # print(cls_logits_per_image.numpy())
      s = sigmoid_focal_loss(cls_logits_per_image, 
                                       gt_classes_target, valid_idxs_per_image)
      losses.append(s/max(1, num_foreground.item()))
      # print('CLASS LOSS ARRAY', losses[-1].numpy(), '||', num_foreground.item(), s.numpy())

    # print(colored(f'FINISHED CLASS LOSS APPEND{losses}', 'cyan'))
    # print(losses[0].shape)
    # return losses[0]+losses[1]
    return Tensor.stack(losses).mean()
    return _sum(losses) / len(targets)
  

class RegressionHead:
  def __init__(self, in_channels, num_anchors):
    self.conv = flatten([(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), lambda x: x.relu()) for _ in range(4)])
    for i in self.conv:
      if isinstance(i, nn.Conv2d):
        i.weight = Tensor.normal(i.weight.shape, std=0.01).realize()
        i.bias = Tensor.full(i.bias.shape, 0.0).realize()
    self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
    self.bbox_reg.weight = Tensor.normal(self.bbox_reg.weight.shape, std=0.01).realize()
    self.bbox_reg.bias = Tensor.full(self.bbox_reg.bias.shape, 0.0).realize()
    
  def __call__(self, x):
    out = [self.bbox_reg(feat.sequential(self.conv)).permute(0, 2, 3, 1).reshape(feat.shape[0], -1, 4) for feat in x]
    return out[0].cat(*out[1:], dim=1)
  def loss(self, logits, targets, anchors, matched_idxs):
    losses = []

    for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in \
                zip(targets, logits, anchors, matched_idxs):
      b = targets_per_image['boxes']

      # print(colored(f'targets_per_image {b.shape}', 'blue' ))
      # print(colored(f'matched_idxs_per_image {matched_idxs_per_image.shape}', 'blue' ))

      # print(matched_idxs_per_image.numpy())
      # determine only the foreground indices, ignore the rest
      # foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
      # Hack for now
      # print('Regression head match idx type:', matched_idxs_per_image.numpy())
      foreground_idxs_per_image = matched_idxs_per_image >= 0
      foreground_idxs_per_image_np = np.nonzero(foreground_idxs_per_image.numpy())[0]
      # print('np forground REGRESSION:', foreground_idxs_per_image, foreground_idxs_per_image.shape)
      # if foreground_idxs_per_image.shape==(0,):
      #   print(colored('empty forground idx in regression head', 'blue'))
      #   foreground_idxs_per_image = Tensor([0])
      # else:
      foreground_idxs_per_image = Tensor(foreground_idxs_per_image_np)
      del foreground_idxs_per_image_np
      # print(colored(f'foreground_idxs_per_image {foreground_idxs_per_image.shape}', 'blue' ))

      # print(foreground_idxs_per_image, matched_idxs_per_image.numpy())
      # print('Regression Foreground idx per img: ', foreground_idxs_per_image.shape, foreground_idxs_per_image.numpy())
      num_foreground = foreground_idxs_per_image.numel()

      # print('target img selection')
      # print(targets_per_image['boxes'].numpy())
      # print(matched_idxs_per_image.numpy())
      # print(foreground_idxs_per_image.numpy())

      # select only the foreground boxes
      matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image[foreground_idxs_per_image]]
      bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
      anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

      # compute the regression targets
      target_regression = encode_boxes(matched_gt_boxes_per_image, anchors_per_image)

      # compute the loss
      losses.append(l1_loss(
        bbox_regression_per_image,
        target_regression,
      ) / max(1, num_foreground))
    
    # print('REgression head loss num/dem', _sum(losses).numpy(),  max(1, len(targets)))
    # print('regression loss length', len(losses))
    # for i in losses:
    #   print(i.numpy())
    return Tensor.stack(losses).mean()
    return _sum(losses) / max(1, len(targets))
class RetinaHead:
  def __init__(self, in_channels, num_anchors, num_classes):
    self.classification_head = ClassificationHead(in_channels, num_anchors, num_classes)
    self.regression_head = RegressionHead(in_channels, num_anchors)
  def __call__(self, x):
    pred_bbox, pred_class = self.regression_head(x), self.classification_head(x)
    return pred_bbox, pred_class
    out = pred_bbox.cat(pred_class, dim=-1)
    return out

class ResNetFPN:
  def __init__(self, resnet, out_channels=256, returned_layers=[2, 3, 4]):
    self.out_channels = out_channels
    self.body = resnet
    in_channels_list = [(self.body.in_planes // 8) * 2 ** (i - 1) for i in returned_layers]
    self.fpn = FPN(in_channels_list, out_channels)

  # this is needed to decouple inference from postprocessing (anchors generation)
  def compute_grid_sizes(self, input_size):
    return np.ceil(np.array(input_size)[None, :] / 2 ** np.arange(3, 8)[:, None])

  def __call__(self, x):
    out = self.body.bn1(self.body.conv1(x)).relu()
    out = out.pad2d([1,1,1,1]).max_pool2d((3,3), 2)
    out = out.sequential(self.body.layer1)
    p3 = out.sequential(self.body.layer2)
    p4 = p3.sequential(self.body.layer3)
    p5 = p4.sequential(self.body.layer4)
    return self.fpn([p3, p4, p5])

class ExtraFPNBlock:
  def __init__(self, in_channels, out_channels):
    self.p6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    self.p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    self.use_P5 = in_channels == out_channels

  def __call__(self, p, c):
    p5, c5 = p[-1], c[-1]
    x = p5 if self.use_P5 else c5
    p6 = self.p6(x)
    p7 = self.p7(p6.relu())
    p.extend([p6, p7])
    return p

class FPN:
  def __init__(self, in_channels_list, out_channels, extra_blocks=None):
    self.inner_blocks, self.layer_blocks = [], []
    for in_channels in in_channels_list:
      self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
      self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    self.extra_blocks = ExtraFPNBlock(256, 256) if extra_blocks is None else extra_blocks

  def __call__(self, x):
    last_inner = self.inner_blocks[-1](x[-1])
    results = [self.layer_blocks[-1](last_inner)]
    for idx in range(len(x) - 2, -1, -1):
      inner_lateral = self.inner_blocks[idx](x[idx])

      # upsample to inner_lateral's shape
      (ih, iw), (oh, ow), prefix = last_inner.shape[-2:], inner_lateral.shape[-2:], last_inner.shape[:-2]
      eh, ew = math.ceil(oh / ih), math.ceil(ow / iw)
      inner_top_down = last_inner.reshape(*prefix, ih, 1, iw, 1).expand(*prefix, ih, eh, iw, ew).reshape(*prefix, ih*eh, iw*ew)[:, :, :oh, :ow]

      last_inner = inner_lateral + inner_top_down
      results.insert(0, self.layer_blocks[idx](last_inner))
    if self.extra_blocks is not None:
      results = self.extra_blocks(results, x)
    return results

if __name__ == "__main__":
  from extra.models.resnet import ResNeXt50_32X4D
  backbone = ResNeXt50_32X4D()
  retina = RetinaNet(backbone)
  retina.load_from_pretrained()