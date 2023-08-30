# RCNN-specific loss functions

from models.mask_rcnn import BoxList
from tinygrad.tensor import Tensor

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications

def test_boxlist_iou():
  a = boxlist_iou(BoxList(Tensor([[0, 0, 10, 10]]), image_size = (50, 50)), BoxList(Tensor([[0, 0, 5, 5]]), image_size = (50, 50)))
  assert all(((a == .25)[0]).numpy())


def boxlist_iou(boxlist1: BoxList, boxlist2: BoxList) -> Tensor:
  # Compute the intersection over union of two set of boxes.
  assert boxlist1.size == boxlist2.size, f"boxlists should have same image size, got {boxlist1}, {boxlist2}"
  N, M = len(boxlist1), len(boxlist2)
  area1, area2 = boxlist1.area(), boxlist2.area()
  box1, box2 = boxlist1.bbox, boxlist2.bbox
  lt = Tensor.maximum(box1[:, None, :2], box2[:, :2])  # [N,M,2]
  rb = Tensor.minimum(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
  wh = (rb - lt).maximum(0)  # [N,M,2]
  inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

  iou = inter / (area1[:, None] + area2 - inter)
  return iou


def make_rpn_loss_evaluator(cfg, box_coder):
  # just, "is it in range", another overthought class construct that could be a lambda
  matcher = Matcher(
      0.7,
      0.3,
      allow_low_quality_matches=True,
  )

  fg_bg_sampler = BalancedPositiveNegativeSampler(
      cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
  )

  loss_evaluator = RPNLossComputation(
      matcher,
      fg_bg_sampler,
      box_coder,
      generate_rpn_labels
  )
  return loss_evaluator

class RPNLossComputation:
  def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
              generate_labels_func):
    """
    Arguments:
        proposal_matcher (Matcher)
        fg_bg_sampler (BalancedPositiveNegativeSampler)
        box_coder (BoxCoder)
    """
    # self.target_preparator = target_preparator
    self.proposal_matcher = proposal_matcher
    self.fg_bg_sampler = fg_bg_sampler
    self.box_coder = box_coder
    self.copied_fields = []
    self.generate_labels_func = generate_labels_func
    self.discard_cases = ['not_visibility', 'between_thresholds']

  def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
    match_quality_matrix = boxlist_iou(target, anchor)
    matched_idxs = self.proposal_matcher(match_quality_matrix)
    # RPN doesn't need any fields from target
    # for creating the labels, so clear them all
    target = target.copy_with_fields(copied_fields)
    # get the targets corresponding GT for each anchor
    # NB: need to clamp the indices because we can have a single
    # GT in the image, and matched_idxs can be -2, which goes
    # out of bounds
    matched_targets = target[matched_idxs.clamp(min=0)]
    matched_targets.add_field("matched_idxs", matched_idxs)
    return matched_targets

  def prepare_targets(self, anchors, targets):
    labels = []
    regression_targets = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
      matched_targets = self.match_targets_to_anchors(
          anchors_per_image, targets_per_image, self.copied_fields
      )

      matched_idxs = matched_targets.get_field("matched_idxs")
      labels_per_image = self.generate_labels_func(matched_targets)
      labels_per_image = labels_per_image.to(dtype=torch.float32)

      # Background (negative examples)
      bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
      labels_per_image[bg_indices] = 0

      # discard anchors that go out of the boundaries of the image
      if "not_visibility" in self.discard_cases:
          labels_per_image[~anchors_per_image.get_field("visibility")] = -1

      # discard indices that are between thresholds
      if "between_thresholds" in self.discard_cases:
          inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
          labels_per_image[inds_to_discard] = -1

      # compute regression targets
      regression_targets_per_image = self.box_coder.encode(
          matched_targets.bbox, anchors_per_image.bbox
      )

      labels.append(labels_per_image)
      regression_targets.append(regression_targets_per_image)

    return labels, regression_targets


  def __call__(self, anchors, objectness, box_regression, targets):
    """
    Arguments:
        anchors (list[BoxList])
        objectness (list[Tensor])
        box_regression (list[Tensor])
        targets (list[BoxList])

    Returns:
        objectness_loss (Tensor)
        box_loss (Tensor
    """
    anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
    labels, regression_targets = self.prepare_targets(anchors, targets)
    sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
    sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
    sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

    sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    objectness, box_regression = \
            concat_box_prediction_layers(objectness, box_regression)

    objectness = objectness.squeeze()

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds],
        regression_targets[sampled_pos_inds],
        beta=1.0 / 9,
        size_average=False,
    ) / (sampled_inds.numel())

    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness[sampled_inds], labels[sampled_inds]
    )

    return objectness_loss, box_loss

if __name__ == "__main__":
  test_boxlist_iou()
