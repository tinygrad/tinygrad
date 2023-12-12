import extra.models.mask_rcnn as mask_rcnn
import math
import torch

from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor
from typing import Tuple

def _project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
  """ Given segmentation masks and the bounding boxes corresponding
  to the location of the masks in the image, this function
  crops and resizes the masks in the position defined by the
  boxes. This prepares the masks for them to be fed to the
  loss computation as the targets.

  Arguments:
      segmentation_masks: an instance of SegmentationMask
      proposals: an instance of BoxList
  """
  masks = []
  M = discretization_size
  device = proposals.bbox.device
  proposals = proposals.convert("xyxy")
  assert segmentation_masks.size == proposals.size, "{}, {}".format(
    segmentation_masks, proposals
  )
  # TODO put the proposals on the CPU, as the representation for the
  # masks is not efficient GPU-wise (possibly several small tensors for
  # representing a single instance mask)
  proposals = proposals.bbox
  for segmentation_mask, proposal in zip(segmentation_masks, proposals):
    # crop the masks, resize them to the desired resolution and
    # then convert them to the tensor representation,
    # instead of the list representation that was used
    cropped_mask = segmentation_mask.crop(proposal)
    scaled_mask = cropped_mask.resize((M, M))
    mask = scaled_mask.convert(mode="mask")
    masks.append(mask)
  if len(masks) == 0:
    return Tensor.empty(0, dtype=dtypes.float32)
  return masks.stack(dim=0).cast(dtypes.float32)


class Matcher:
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

  def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
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
    assert low_threshold <= high_threshold
    self.high_threshold = high_threshold
    self.low_threshold = low_threshold
    self.allow_low_quality_matches = allow_low_quality_matches

  def __call__(self, match_quality_matrix):
    """
    Args:
        match_quality_matrix (Tensor[float]): an MxN tensor, containing the
        pairwise quality between M ground-truth elements and N predicted elements.

    Returns:
        matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
        [0, M - 1] or a negative value indicating that prediction i could not
        be matched.
    """
    match_quality_matrix = torch.from_numpy(match_quality_matrix.numpy())
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
    matched_vals, matches = match_quality_matrix.max(dim=0)
    # matched_vals, matches = Tensor(matched_vals.numpy()), Tensor(matches.numpy())
    if self.allow_low_quality_matches:
      all_matches = matches.clone()

    # Assign candidate matches with low quality to negative (unassigned) values
    below_low_threshold = matched_vals < self.low_threshold
    between_thresholds = (matched_vals >= self.low_threshold) & (
        matched_vals < self.high_threshold
    )
    matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
    matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS

    if self.allow_low_quality_matches:
        self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

    return Tensor(matches.numpy())

  def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
    """
    Produce additional matches for predictions that have only low-quality matches.
    Specifically, for each ground-truth find the set of predictions that have
    maximum overlap with it (including ties); for each prediction in that set, if
    it is unmatched, then match it to the ground-truth with which it has the highest
    quality value.
    """
    # For each gt, find the prediction with which it has highest quality
    highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
    # Find highest quality match available, even if it is low, including ties
    gt_pred_pairs_of_highest_quality = torch.nonzero(
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

    pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
    matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


class BalancedPositiveNegativeSampler:
  """
  This class samples batches, ensuring that they contain a fixed proportion of positives
  """

  def __init__(self, batch_size_per_image, positive_fraction):
    """
    Arguments:
        batch_size_per_image (int): number of elements to be selected per image
        positive_fraction (float): percentage of positive elements per batch
    """
    self.batch_size_per_image = batch_size_per_image
    self.positive_fraction = positive_fraction

  def __call__(self, matched_idxs):
    """
    Arguments:
        matched idxs: list of tensors containing -1, 0 or positive values.
            Each tensor corresponds to a specific image.
            -1 values are ignored, 0 are considered as negatives and > 0 as
            positives.

    Returns:
        pos_idx (list[tensor])
        neg_idx (list[tensor])

    Returns two lists of binary masks for each image.
    The first list contains the positive elements that were selected,
    and the second list the negative example.
    """
    pos_idx = []
    neg_idx = []
    # TODO: optimize some of the ops to be tinygrad native
    for matched_idxs_per_image in matched_idxs:
      positive = torch.nonzero(torch.from_numpy(matched_idxs_per_image.numpy()) >= 1).squeeze(1)
      negative = torch.nonzero(torch.from_numpy(matched_idxs_per_image.numpy()) == 0).squeeze(1)

      num_pos = int(self.batch_size_per_image * self.positive_fraction)
      # protect against not enough positive examples
      num_pos = min(positive.numel(), num_pos)
      num_neg = self.batch_size_per_image - num_pos
      # protect against not enough negative examples
      num_neg = min(negative.numel(), num_neg)

      # randomly select positive and negative examples
      perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
      perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

      pos_idx_per_image = positive[perm1]
      neg_idx_per_image = negative[perm2]

      # create binary mask from indices
      pos_idx_per_image_mask = torch.zeros_like(
          torch.from_numpy(matched_idxs_per_image.numpy()), dtype=torch.bool
      )
      neg_idx_per_image_mask = torch.zeros_like(
          torch.from_numpy(matched_idxs_per_image.numpy()), dtype=torch.bool
      )
      pos_idx_per_image_mask[pos_idx_per_image] = 1
      neg_idx_per_image_mask[neg_idx_per_image] = 1

      pos_idx.append(Tensor(pos_idx_per_image_mask.numpy()))
      neg_idx.append(Tensor(neg_idx_per_image_mask.numpy()))

    return pos_idx, neg_idx


class BoxCoder:
  def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
    self.weights = weights
    self.bbox_xform_clip = bbox_xform_clip

  def encode(self, reference_boxes, proposals):
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
    boxes = boxes.cast(rel_codes.dtype)
    rel_codes = rel_codes

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

    # Prevent sending too large values into Tensor.exp()
    dw = dw.clip(min_=dw.min(), max_=self.bbox_xform_clip)
    dh = dh.clip(min_=dh.min(), max_=self.bbox_xform_clip)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = dw.exp() * widths[:, None]
    pred_h = dh.exp() * heights[:, None]
    x = pred_ctr_x - 0.5 * pred_w
    y = pred_ctr_y - 0.5 * pred_h
    w = pred_ctr_x + 0.5 * pred_w - 1
    h = pred_ctr_y + 0.5 * pred_h - 1
    pred_boxes = Tensor.stack([x, y, w, h]).permute(1,2,0).reshape(rel_codes.shape[0], rel_codes.shape[1])
    return pred_boxes


class RPNLossComputation:
  def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, generate_labels_func):
    self.proposal_matcher = proposal_matcher
    self.fg_bg_sampler = fg_bg_sampler
    self.box_coder = box_coder
    self.generate_labels_func = generate_labels_func
    self.discard_cases = ["not_visibility", "between_thresholds"]

  def __call__(self, anchors, objectness, box_regression, targets) -> Tuple[Tensor, ...]:
    anchors = [mask_rcnn.cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
    labels, regression_targets = self.prepare_targets(anchors, targets)
    sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
    sampled_pos_inds, sampled_neg_inds = mask_rcnn.nonzero(Tensor.cat(*sampled_pos_inds)), mask_rcnn.nonzero(Tensor.cat(*sampled_neg_inds))
    sampled_inds = sampled_pos_inds.cat(sampled_neg_inds)

    objectness, box_regression = mask_rcnn.concat_box_prediction_layers(objectness, box_regression)
    objectness = objectness.squeeze()
    labels, regression_targets = Tensor.cat(*labels), Tensor.cat(*regression_targets)

    box_loss = smooth_l1_loss(box_regression[sampled_pos_inds], regression_targets[sampled_pos_inds], size_average=False) / (sampled_inds.numel())
    objectness_loss = objectness[sampled_inds].binary_crossentropy_logits(labels[sampled_inds])
    return objectness_loss, box_loss

  def match_targets_to_anchors(self, anchor, target):
    match_quality_matrix = mask_rcnn.boxlist_iou(target, anchor)
    matched_idxs = self.proposal_matcher(match_quality_matrix)
    matched_targets = target[matched_idxs.maximum(0)]
    return matched_targets, matched_idxs

  def prepare_targets(self, anchors, targets):
    labels, regression_targets = [], []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
      matched_targets, matched_idxs = self.match_targets_to_anchors(anchors_per_image, targets_per_image)

      labels_per_image = self.generate_labels_func(matched_idxs)
      labels_per_image = labels_per_image.cast(dtypes.float32)
      labels_per_image = (matched_idxs == -1).where(0, labels_per_image)

      if "not_visibility" in self.discard_cases:
        mask = mask_rcnn.tilde(anchors_per_image.get_field("visibility").cast(dtypes.bool))
        labels_per_image = mask.where(-1, labels_per_image)

      if "between_thresholds" in self.discard_cases:
        labels_per_image = (matched_idxs == -2).where(-1, labels_per_image)

      regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, anchors_per_image.bbox)
      labels.append(labels_per_image)
      regression_targets.append(regression_targets_per_image)

    return labels, regression_targets


class FastRCNNLossComputation:
  def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, cls_agnostic_bbox_reg=False):
    self.proposal_matcher = proposal_matcher
    self.fg_bg_sampler = fg_bg_sampler
    self.box_coder = box_coder
    self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

  def match_targets_to_proposals(self, proposal, target):
    match_quality_matrix = mask_rcnn.boxlist_iou(target, proposal)
    matched_idxs = self.proposal_matcher(match_quality_matrix)
    # Fast RCNN only need "labels" field for selecting the targets
    target = target.copy_with_fields("labels")
    # get the targets corresponding GT for each proposal
    # NB: need to clamp the indices because we can have a single
    # GT in the image, and matched_idxs can be -2, which goes
    # out of bounds
    matched_targets = target[matched_idxs.maximum(0)]
    return matched_targets, matched_idxs
  
  def prepare_targets(self, proposals, targets):
    labels, regression_targets = [], []
    for proposals_per_image, targets_per_image in zip(proposals, targets):
      matched_targets, matched_idxs = self.match_targets_to_proposals(proposals_per_image, targets_per_image)

      labels_per_images = matched_targets.get_field("labels").cast(dtypes.int64)

      # Label background (below the low threshold)
      labels_per_images = (matched_idxs == -1).where(0, labels_per_images)

      # Label ignore proposals (between low and high thresholds)
      labels_per_images = (matched_idxs == -2).where(-1, labels_per_images)

      # compute regression targets
      regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, proposals_per_image.bbox)

      labels.append(labels_per_images)
      regression_targets.append(regression_targets_per_image)

    return labels, regression_targets
  
  def subsample(self, proposals, targets):
    labels, regression_targets = self.prepare_targets(proposals, targets)
    sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

    proposals = list(proposals)
    # add corresponding label and regression_targets information to the bounding boxes
    for labels_per_image, regression_targets_per_image, proposals_per_image in zip(labels, regression_targets, proposals):
      proposals_per_image.add_field("labels", labels_per_image)
      proposals_per_image.add_field("regression_targets", regression_targets_per_image)

    # distributed sampled proposals, that were obtained on all feature maps
    # concatenated via the fg_bg_sampler, into individual feature map levels
    for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
      # TODO: optimize this to be in tinygrad
      pos_inds_img, neg_inds_img = torch.from_numpy(pos_inds_img.numpy()), torch.from_numpy(neg_inds_img.numpy())
      img_sampled_inds = Tensor(torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1).numpy())
      proposals_per_image = proposals[img_idx][img_sampled_inds]
      proposals[img_idx] = proposals_per_image

    self._proposals = proposals
    return proposals
  
  def __call__(self, class_logits, box_regression):
    class_logits = Tensor.cat(*class_logits)
    box_regression = Tensor.cat(*box_regression)
    device = class_logits.device

    if not hasattr(self, "_proposals"):
      raise RuntimeError("subsample needs to be called before")
    
    proposals = self._proposals

    labels = Tensor.cat(*[proposal.get_field("labels") for proposal in proposals])
    regression_targets = Tensor.cat(*[proposal.get_field("regression_targets") for proposal in proposals])

    # TODO: figure this out
    classification_loss = torch.nn.functional.cross_entropy(torch.from_numpy(class_logits.numpy()), torch.from_numpy(labels.numpy()).long())

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = mask_rcnn.nonzero(labels > 0)
    labels_pos = labels[sampled_pos_inds_subset]
    if self.cls_agnostic_bbox_reg:
      map_inds = Tensor([4, 5, 6, 7], device=device)
    else:
      map_inds = 4 * labels_pos[:, None] + Tensor([0, 1, 2, 3], device=device)

    box_loss = mask_rcnn.smooth_l1_loss(
      box_regression[sampled_pos_inds_subset[:, None], map_inds],
      regression_targets[sampled_pos_inds_subset],
      beta=1,
      size_average=False
    )
    box_loss = box_loss / labels.numel()
    return classification_loss, box_loss


class MaskRCNNLossComputation:
  def __init__(self, proposal_matcher, discretization_size):
    self.proposal_matcher = proposal_matcher
    self.discretization_size = discretization_size

  def match_targets_to_proposals(self, proposal, target):
    match_quality_matrix = mask_rcnn.boxlist_iou(target, proposal)
    matched_idxs = self.proposal_matcher(match_quality_matrix)
    # Fast RCNN only need "labels" field for selecting the targets
    target = target.copy_with_fields("labels")
    # get the targets corresponding GT for each proposal
    # NB: need to clamp the indices because we can have a single
    # GT in the image, and matched_idxs can be -2, which goes
    # out of bounds
    matched_targets = target[matched_idxs.maximum(0)]
    return matched_targets, matched_idxs
  
  def prepare_targets(self, proposals, targets):
    labels, masks = [], []
    for proposals_per_image, targets_per_image in zip(proposals, targets):
      matched_targets, matched_idxs = self.match_targets_to_proposals(proposals_per_image, targets_per_image)

      labels_per_image = matched_targets.get_field("labels").cast(dtypes.int64)
      labels_per_image = (matched_idxs == Matcher.BELOW_LOW_THRESHOLD).where(0, labels_per_image)

      positive_inds = mask_rcnn.nonzero(labels_per_image > 0)

      segmentation_masks = matched_targets.get_field("masks")
      segmentation_masks = segmentation_masks[positive_inds]

      positive_proposals = proposals_per_image[positive_inds]

      masks_per_image = _project_masks_on_boxes(segmentation_masks, positive_proposals, self.discretization_size)

      labels.append(labels_per_image)
      masks.append(masks_per_image)

    return labels, masks
  
  def __call__(self, proposals, mask_logits, targets):
    labels, mask_targets = self.prepare_targets(proposals, targets)
    labels, mask_targets = Tensor.cat(*labels), Tensor.cat(*mask_targets)

    positive_inds = mask_rcnn.nonzero(labels > 0)
    labels_pos = labels[positive_inds]

    if mask_targets.numel() == 0: return mask_logits.sum() * 0

    return Tensor.binary_crossentropy_logits(mask_logits[positive_inds, labels_pos], mask_targets)


def smooth_l1_loss(self:Tensor, Y:Tensor, beta:float = 1./9, size_average:bool = True) -> Tensor:
  n = (self-Y).abs()
  cond = n < beta
  loss = cond.where(0.5 * n ** 2 / beta, n - 0.5 * beta)
  if size_average: return loss.mean()
  return loss.sum()

def create_rpn_loss_evaluator(box_coder:BoxCoder) -> RPNLossComputation:
  matcher = Matcher(0.7, 0.3, allow_low_quality_matches=True)
  fg_bg_sampler = BalancedPositiveNegativeSampler(256, 0.5)
  return RPNLossComputation(matcher, fg_bg_sampler, box_coder, lambda x: x >= 0)

def create_fast_rcnn_loss_evaluator(box_coder:BoxCoder) -> FastRCNNLossComputation:
  matcher = Matcher(0.5, 0.5, allow_low_quality_matches=False)
  fg_bg_sampler = BalancedPositiveNegativeSampler(512, 0.25)
  return FastRCNNLossComputation(matcher, fg_bg_sampler, box_coder, cls_agnostic_bbox_reg=False)

def create_mask_rcnn_loss_evaluator() -> MaskRCNNLossComputation:
  matcher = Matcher(0.5, 0.5, allow_low_quality_matches=False)
  return MaskRCNNLossComputation(matcher, 28)
