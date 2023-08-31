# RCNN-specific loss functions

from models.mask_rcnn import BoxList
from tinygrad.tensor import Tensor
from tinygrad.tensor import dtypes
from typing import List, Callable, Tuple

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

  match_quality_matrix = Tensor([[0.9, 0.7, 0.8, 0.9], # gt 1, .9
                                [0.1, 0.5, 0.1, 0.2],  # gt 2, .5
                                [0.1, 0.2, 0.2, 0.3]]) # gt 3, .3
  # 1. test that it works
  a = fn1(match_quality_matrix)
  assert all(((a == Tensor([[ 0.9],[-2.],[-1.]]))[:, 0]).numpy())
   

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

    above_high_threshold = preds >= high
    below_low_threshold = preds < low
    between_thresholds = (preds >= low) * (preds < high)
    
    if allow_low_qual:
      set_low_quality_matches_(preds)

    return above_high_threshold*preds - between_thresholds*2 - below_low_threshold*1
  return loss_eval_fn
   
def test_balanced_sampler():
  fn1 = make_balanced_sampler_fn(10, 0.5)
  a = fn1([Tensor([1, 0, 1, 1, 1, 1, 0, 1, 1, 0])])
  assert all(((a[0] == Tensor([1, 1, 1, 1, 1]))).numpy())

def test_randperm():
  # [.2, .6, .8, .2, 0, 0, 0, 0, .8, .9]
  # [1,  1,  1,  1,  0, 0, 0, 0,  1, 1] # 6 positives
  # [.2, .6, .8, .2, .8, .9] # positives only

  # TODO perf
  def rand_sample(t: Tensor, mask: Tensor, take: int) -> Tensor:
    # TODO bool masks would be nice
    # t = t[mask.cast(dtypes.bool)]
    t = t.numpy()[mask.numpy().astype("bool")]
    print("Masked:", t)
    ind, r = Tensor.arange(t.shape[0]), Tensor.rand(t.shape[0])
    pairs = Tensor.stack([ind,r],dim=1)
    xx = Tensor(pairs[:,1].numpy().argsort().reshape(pairs.shape[0], 1)[:take])
    print("here")
    print(t.numpy())
    print(xx.numpy())
    return pairs.gather(Tensor(pairs[:,1].numpy().argsort().reshape(pairs.shape[0], 1)[:take]), dim=1)

  t = Tensor([.2, .6, .8, .2, 0, 0, 0, 0, .8, .9])
  x = rand_sample(t, Tensor([1, 1, 1, 1, 0, 0, 0, 0, 1, 1]), 3)
  print(x.numpy())
  # [1, 2, 3, 4, 9, 10]
  # [.5,.1,.5,.2,.1,.4] # rand
  # stack, then sort and take

  def bool_mask(t: Tensor, mask: Tensor) -> Tensor:

    pairs = Tensor.stack([idx,t],dim=1).numpy()
    return pairs[pairs[:,1].argsort()]

  # [[1,0], [1,1] .. [0,4] .. [1, 9]]
  # [[2], [1], [10]] # sample 3
  mask = Tensor([1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
  print("Mask:", mask.numpy())
  r = Tensor.rand(10)
  print("Random:", r.numpy())
  print("masked:", r[mask * Tensor.arange(10)].numpy())
  randperm(10, Tensor([1, 0, 1, 1, 1, 1, 0, 1, 1, 0]))

def randperm(size: int, idx: Tensor) -> Tensor:
  tensor = Tensor([.2, .6, .8, .2, 0, 0, 0, 0, .8, .9])
  # Sort the tensor
  sorted_tensor, _ = Tensor.sort(tensor)
  # Find the index where zeros end
  non_zero_start_idx = torch.sum(sorted_tensor == 0).item()
  # Slice the tensor from that index to the end
  non_zero_tensor = sorted_tensor[non_zero_start_idx:]
  pairs = Tensor.stack([idx,Tensor.rand(size)],dim=1).numpy()
  return pairs[pairs[:,1].argsort()]

def make_balanced_sampler_fn(batch_size_per_image: int, positive_fraction: float) -> Callable[[Tensor], Tuple[List[Tensor], List[Tensor]]]:
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

  def sampler_fn(image_matches: List[Tensor], numpy_fancy: bool) -> (Tensor, Tensor):
    pos_idx = []
    neg_idx = []
    for matches in image_matches:
      positive = matches.where(matches >= 1, 0) # TODO this was 1 in the example, should be >0? or threshold
      print("Positive matches:", positive.numpy())
      negative = matches.where(matches == 0, 0)

      num_pos = int(batch_size_per_image * positive_fraction)
      # protect against not enough positive examples
      pos_numel, neg_numel = positive.sum().numpy().item(), negative.sum().numpy().item()
      num_pos = min(pos_numel, num_pos)
      num_neg = min(neg_numel, batch_size_per_image - num_pos)

      pos_pairs = randperm(num_pos)
      neg_pairs = randperm(num_neg)
      
      pos_index = pos_pairs[:, 0].astype("int32")
      print("Random permutation of indices:", pos_index)


      neg_index,neg_r = Tensor.arange(neg_numel),Tensor.rand(neg_numel)
      neg_pairs = Tensor.stack([neg_index,neg_r],dim=1)

      neg_pairs = neg_pairs.numpy()[neg_pairs[:,1].numpy().argsort()]
      neg_index = neg_pairs[:, 0].astype("int32")


      pos_idx_per_image = positive.numpy()[pos_index]
      print("Positive indices:", pos_idx_per_image)
      neg_idx_per_image = negative.numpy()[neg_index]
      

      # # create binary mask from indices
      # pos_idx_per_image_mask = torch.zeros_like(
      #     matched_idxs_per_image, dtype=torch.uint8
      # )
      # neg_idx_per_image_mask = torch.zeros_like(
      #     matched_idxs_per_image, dtype=torch.uint8
      # )
      # pos_idx_per_image_mask[pos_idx_per_image] = 1
      # neg_idx_per_image_mask[neg_idx_per_image] = 1

      # pos_idx.append(pos_idx_per_image_mask)
      # neg_idx.append(neg_idx_per_image_mask)

    return pos_idx, neg_idx
  return sampler_fn

def make_rpn_loss_evaluator(cfg, box_coder):
  matcher = make_match_evaluation_fn(.7, .3)
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
  #PLAYGROUND
  data = Tensor([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9],
               [10, 11, 12]])
  idx = Tensor([0, 2, 1, 1]).reshape(4, 1)
  result = data.gather(idx, dim=1)

  test_randperm()
  test_boxlist_iou()
  test_match_eval()
  test_balanced_sampler()
  

  # boolmask
  # ind = Tensor.arange(mask.shape[0])
  # nz = mask.sum().numpy().item()
  # print("Nonzero:", nz)
  # mask = mask * ind
  # masked = mask.numpy().argsort()[-int(nz):]
  # print("Nonzero indices:", masked)