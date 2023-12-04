import torch

from extra.models.mask_rcnn import cat_boxlist, BoxList, permute_and_flatten
from typing import Tuple
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes

class RPNLossComputation:
  def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, generate_labels_func):
    self.proposal_matcher = proposal_matcher
    self.fg_bg_sampler = fg_bg_sampler
    self.box_coder = box_coder
    self.generate_labels_func = generate_labels_func
    self.discard_cases = ["not_visibility", "between_thresholds"]

  def __call__(self, anchors, objectness, box_regression, targets) -> Tuple[Tensor, ...]:
    anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
    labels, regression_targets = self.prepare_targets(anchors, targets)
    sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
    sampled_pos_inds, sampled_neg_inds = nonzero(Tensor.cat(sampled_pos_inds)), nonzero(Tensor.cat(sampled_neg_inds))
    sampled_inds = sampled_pos_inds.cat(sampled_neg_inds)

    objectness, box_regression = concat_box_prediction_layers(objectness, box_regression)
    objectness = objectness.squeeze()
    labels, regression_targets = Tensor.cat(*labels), Tensor.cat(*regression_targets)

    box_loss = smooth_l1_loss(box_regression[sampled_pos_inds], regression_targets[sampled_pos_inds], size_average=False) / (sampled_inds.numel())
    objectness_loss = objectness[sampled_inds].binary_crossentropy_logits(labels[sampled_inds])
    return objectness_loss, box_loss

  def match_targets_to_anchors(self, anchor, target):
    match_quality_matrix = boxlist_iou(target, anchor)
    matched_idxs = self.proposal_matcher(match_quality_matrix)
    matched_targets = target[matched_idxs.maximum(0)]
    return matched_targets, matched_idxs

  def prepare_targets(self, anchors, targets):
    labels, regression_targets = [], []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
      matched_targets, matched_idxs = self.match_targets_to_anchors(anchors_per_image, targets_per_image)

      labels_per_images = self.generate_labels_func(matched_idxs)
      labels_per_images = labels_per_images.cast(dtypes.float32)
      labels_per_images = (matched_idxs == -1).where(0, labels_per_images)

      if "not_visibility" in self.discard_cases:
        mask = tilde(anchors_per_image.bbox).cast(dtypes.bool)
        labels_per_images = mask.where(-1, labels_per_images)

      if "between_thresholds" in self.discard_cases:
        labels_per_images = (matched_idxs == -2).where(-1, labels_per_images)

      regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, anchors_per_image.bbox)
      labels.append(labels_per_images)
      regression_targets.append(regression_targets_per_image)

    return labels, regression_targets

def smooth_l1_loss(self:Tensor, Y:Tensor, beta:float = 1./9, size_average:bool = True) -> Tensor:
  n = (self-Y).abs()
  cond = n < beta
  loss = cond.where(0.5 * n ** 2 / beta, n - 0.5 * beta)
  if size_average: return loss.mean()
  return loss.sum()

def boxlist_iou(boxlist1:BoxList, boxlist2:BoxList) -> Tensor:
  assert boxlist1.size == boxlist2.size, "boxlists should have the same size"
  area1, area2 = boxlist1.area(), boxlist2.area()
  box1, box2 = boxlist1.bbox, boxlist2.bbox
  lt = Tensor.maximum(box1[:, None, :2], box2[:, :2])
  rb = Tensor.minimum(box1[:, None, 2:], box2[:, 2:])
  TO_REMOVE = 1
  wh = (rb - lt + TO_REMOVE).maximum(0)
  inter = wh[:, :, 0] * wh[:, :, 1]
  return inter / (area1[:, None] + area2 - inter)

def concat_box_prediction_layers(box_cls, box_regression) -> Tuple[Tensor, ...]:
  box_cls_flattened, box_regression_flattened = [], []
  # for each feature level, permute the outputs to make them be in the
  # same format as the labels. Note that the labels are computed for
  # all feature levels concatenated, so we keep the same representation
  # for the objectness and the box_regression
  for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
    N, AxC, H, W = box_cls_per_level.shape
    Ax4 = box_regression_per_level.shape[1]
    A = Ax4 // 4
    C = AxC // A
    box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
    box_cls_flattened.append(box_cls_per_level)

    box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
    box_regression_flattened.append(box_regression_per_level)
  # concatenate on the first dimension (representing the feature levels), to
  # take into account the way the labels were generated (with all feature maps
  # being concatenated as well)
  box_cls = Tensor.cat(*box_cls_flattened, dim=1).reshape(-1, C)
  box_regression = Tensor.cat(*box_regression_flattened, dim=1).reshape(-1, 4)
  return box_cls, box_regression

def generate_rpn_labels(matched_idxs): return matched_idxs >= 0

# NOTE: implement this in tinygrad
def nonzero(self:Tensor) -> Tensor: return Tensor(torch.from_numpy(self.numpy()).nonzero().squeeze(1).numpy())

def tilde(x: Tensor) -> Tensor:
  if x.dtype == dtypes.bool: return (1 - x).cast(dtypes.bool)
  return (x + 1) * -1  # this seems to be what the ~ operator does in pytorch for non bool
