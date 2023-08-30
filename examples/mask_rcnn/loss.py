# RCNN-specific loss functions

from models.mask_rcnn import BoxList
from tinygrad.tensor import Tensor
from tinygrad.tensor import dtypes
from typing import List, Callable

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


def test_match_eval():
  fn1 = make_match_evaluation_fn(0.7, 0.4)

  match_quality_matrix = Tensor([[0.9, 0.7, 0.8, 0.9], # gt 1
                                [0.1, 0.8, 0.1, 0.2],  # gt 2
                                [0.1, 0.5, 0.1, 0.2],  # gt 2
                                [0.1, 0.2, 0.2, 0.3]]) # gt 3
  # 1. test that it works
  a = fn1(match_quality_matrix)
  assert all(((a == Tensor([[ 0.9],[ 0.8],[-2.],[-1.]]))[:, 0]).numpy())
   

def make_match_evaluation_fn(high: float, low: float, allow_low_qual: bool = False) -> Callable[[Tensor], Tensor]:
  # TODO
  def set_low_quality_matches_(preds: Tensor):
    """
    Produce additional matches for predictions that have only low-quality matches.
    Specifically, for each ground-truth find the set of predictions that have
    maximum overlap with it (including ties); for each prediction in that set, if
    it is unmatched, then match it to the ground-truth with which it has the highest
    quality value.
    """
    # For each gt, find the prediction with which it has highest quality
    highest_quality_foreach_gt, _ = match_quality_matrix.max(axis=1)
    # Find highest quality match available, even if it is low, including ties
    gt_pred_pairs_of_highest_quality = Tensor.nonzero(
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

  def loss_eval_fn(match_quality_matrix: Tensor) -> Tensor:
    if match_quality_matrix.numel() == 0:
      if match_quality_matrix.shape[0] == 0:
        raise ValueError(
          "No ground-truth boxes available for one of the images "
          "during training")
      else:
        raise ValueError(
          "No proposal boxes available for one of the images "
          "during training")

    # find best gt candidate for each prediction
    preds = match_quality_matrix.max(axis=1, keepdim=True)
    if allow_low_qual:
      all_matches = Tensor.zeros(preds.shape, dtype=dtypes.int8)
    preds = Tensor(
       [[0.9],
        [0.8],
        [0.5],
        [0.3]]
    )
    above_high_threshold = preds >= high
    below_low_threshold = preds < low
    between_thresholds = (preds >= low) * (preds < high)
    if allow_low_qual:
      set_low_quality_matches_(preds)

    return above_high_threshold*preds - between_thresholds*2 - below_low_threshold*1
  return loss_eval_fn
   

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
  test_match_eval()





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

        return matches

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
