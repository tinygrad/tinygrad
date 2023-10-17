# RCNN-specific loss functions

from models.mask_rcnn import *
from tinygrad.tensor import Tensor, dtypes, Function
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import DEBUG
import tinygrad.nn.optim as optim
import numpy as np
from typing import List, Callable, Tuple
from train import build_transforms
from pycocotools.coco import COCO
from PIL import Image
from extra.datasets.coco import BASEDIR, download_train
import os
# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications

import pycuda.driver as cuda
import pycuda.autoinit  # This is needed to initialize CUDA driver
def print_gpu_memory(label):
  """Get the GPU memory usage."""
  free = cuda.mem_get_info()[0]
  total = cuda.mem_get_info()[1]
  used = total - free
  print(f"Used memory at {label}: {used / (1024**2):.2f} MB")

def test_boxlist_iou():
  a = boxlist_iou(BoxList(Tensor([[0, 0, 10, 10], [5, 5, 10, 10]]), image_size = (50, 50)), BoxList(Tensor([[0, 0, 5, 5], [0, 0, 10, 10], [4, 4, 8, 8]]), image_size = (50, 50)))
  assert np.allclose(a.numpy(), Tensor([[0.25, 1., 0.16], [0., 0.25, 0.28125]]).numpy())


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
  fn1, _ = make_match_fn(0.7, 0.4)
  match_quality_matrix = Tensor([[0.9, 0.7, 0.8, 0.9, .1],
                                [0.1, 0.5, 0.1, 0.2, .1],
                                [0.1, 0.2, 0.85, 0.3, .1],
                                [0.1, 0.9, 0.2, 0.3, .3],
                                [0.1, 0.9, 0.2, 0.8, .1]])
  a = fn1(match_quality_matrix)
  assert all(((a == Tensor([3, -1, 2, 1, 1]))).numpy())

def test_low_qual_match():
  matches = Tensor([
    [.3, .4, .1], #gt 1
    [.4, .5, .6],
    [.5, .8, .7],
    [.6, .1, .0],
    [.2, .1, .0],
    [.8, .0, .9]
  ])
  hq_fn, lq_fn = make_match_fn(.7, .3)
  first_pass = hq_fn(matches)
  second_pass = (first_pass == -1).where(lq_fn(matches), first_pass) # merges in low quality matches
  assert all(((second_pass == Tensor([1, 2, 1, 0, -1, 2]))).numpy())
  # tests where matches are either hq or negative (weak signals filtered)
  low_signal = ((matches >= .3) * (matches <= .7)).sum(axis=1) == matches.shape[1]
  res = low_signal.where(-2, first_pass)
  assert all(res == Tensor([-1, -2, 1, -1, -1, 2]).numpy())

def make_match_fn(high: float, low: float) -> Callable[[Tensor], Tensor]:
  # for the tensor of M*N
  # where M is the index of the gt and N is the prediction's quality for that gt
  # returns a tensor of N length, where N[i] is the gt that best matched this prediction
  # N[i] is a negative value if there is no match
  def hq_match_fn(preds: Tensor) -> Tensor:
    assert preds.numel() > 0, "must be scoring something"
    # drops lq+mid matches early
    hq = (preds >= high) * preds
    max_vals = hq.max(axis=1)
    max_vals = (max_vals == 0).where(-1, max_vals).unsqueeze(1) # -1 when pred == 0 for all gt
    best_matches = (hq == max_vals).float()
    # todo divide and concur cumsums
    best_gt = (best_matches * Tensor(Tensor.ones_like(best_matches).numpy().cumsum(axis=1))).max(axis=1)
    return best_gt - 1
  
  # Returns matches that were greater than low and less than high
  # TODO dry
  def lq_match_fn(preds: Tensor) -> Tensor:
    assert preds.numel() > 0, "must be scoring something"
    # drops lq+mid matches early
    lq = (preds < high) * (preds >= low) * preds
    max_vals = lq.max(axis=1)
    max_vals = (max_vals == 0).where(-1, max_vals).unsqueeze(1) # -1 when pred == 0 for all gt
    best_matches = (lq == max_vals).float()
    best_gt = (best_matches * Tensor.ones_like(best_matches).cumsum(axis=1)).max(axis=1)
    return best_gt - 1

  return hq_match_fn, lq_match_fn

def test_rind():
  x = rind(Tensor([1, 1, 1, 1, 0, 0, 0, 0, 1, 1]).numpy(), 3)
  assert x.ndim == 1
  assert x.shape[0] == 3
  assert np.isin(x, [0, 1, 2, 3, 8, 9]).all()

# TODO perf
# returns random indices of a mask
def rind(mask: np.ndarray, take: int) -> Tensor:
  assert mask.ndim == 1 and mask.shape[0] >= take
  masked = (np.arange(mask.shape[0]) * mask)[mask.astype(bool)]
  stacked = np.stack([masked,np.random.rand(masked.shape[0])],axis=0)
  return stacked[0, stacked[1].argsort()[:take]]

def test_balanced_sampler():
  fn1 = make_balanced_sampler_fn(10, 0.5)
  t1 = Tensor([1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
  a1 = np.arange(t1.shape[0])
  a, b = fn1([t1])
  assert np.isin(a[0], t1.numpy() * a1).all()
  assert np.isin(b[0], (t1 == 0).numpy() * a1).all()

# returns random mask of positive and negative examples
def make_balanced_sampler_fn(batch_size_per_image: int, positive_fraction: float) -> Callable[[Tensor], Tuple[List[Tensor], List[Tensor]]]:
  def sampler_fn(image_matches: List[Tensor]) -> (Tensor, Tensor):
    pos_inds = []
    neg_inds = []
    for matches in image_matches:
      # expected that positive samples are amped to 1
      if DEBUG > 0: print("matches", matches.numpy())
      positive, negative = matches == 1, matches == 0 
      num_pos = int(batch_size_per_image * positive_fraction)

      # protect against not enough positive examples
      if DEBUG > 0: print("positive", positive.numpy(), "negative", negative.numpy())
      pos_numel, neg_numel = positive.sum().numpy().item(), negative.sum().numpy().item()
      num_pos = int(min(pos_numel, num_pos))
      num_neg = int(min(neg_numel, int(batch_size_per_image * (1 - positive_fraction))))
      
      # option .. return a mask or return gather indices, which is more efficient?
      pos_inds.append(rind(positive.numpy(), num_pos).astype(int))
      neg_inds.append(rind(negative.numpy(), num_neg).astype(int))

    return pos_inds, neg_inds
  return sampler_fn

# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_idxs: Tensor) -> Tensor:
    labels_per_image = matched_idxs >= 0
    return labels_per_image

def test_concat_box_prediction_layers():
  channels=256
  anchor_generator = AnchorGenerator()
  head = RPNHead(
    channels, anchor_generator.num_anchors_per_location()[0]
  )
  backbone = ResNetFPN(ResNet(50, num_classes=None, stride_in_1x1=True), out_channels=256)
  img_id = 387655
  img = [Tensor(build_transforms()(Image.open(BASEDIR/f'train2017/000000{img_id}.jpg').convert("RGB")).numpy())] # TODO this uses torch to transform
  images = to_image_list(img)
  features = backbone(images.tensors)
  objectness, regression = head(features)
  objectness_concat, regression_concat = concat_box_prediction_layers(objectness, regression)
  assert np.allclose(objectness[0][0, :, 0, 0].numpy(), objectness_concat[0:3].squeeze().numpy())
  x = flat_idx(objectness, 2, 5, 3)
  assert np.allclose(objectness[2][0, :, 5, 3].numpy(), objectness_concat[flat_idx(objectness, 2, 5, 3):flat_idx(objectness, 2, 5, 3)+3].squeeze().numpy())
  assert np.allclose(regression[2][0, 0:4, 5, 3].squeeze().numpy(), regression_concat[flat_idx(regression, 2, 5, 3, 4)].numpy())

def flat_idx(F: list[Tensor], l: int, h: int, w: int, p: int = 1) -> int:
  acc=0
  for i in range(0, l):
    _, A, H, W = F[i].shape
    acc+=A/p*H*W
  _, A, _, W = F[l].shape
  return int(acc+h*A/p*W+w*A/p)

def concat_box_prediction_layers(box_cls, box_regression):
  box_cls_flattened = []
  box_regression_flattened = []
  # for each feature level, permute the outputs to make them be in the
  # same format as the labels. Note that the labels are computed for
  # all feature levels concatenated, so we keep the same representation
  # for the objectness and the box_regression
  for box_cls_per_level, box_regression_per_level in zip(
      box_cls, box_regression
  ):
      N, AxC, H, W = box_cls_per_level.shape
      Ax4 = box_regression_per_level.shape[1]
      A = Ax4 // 4
      C = AxC // A
      box_cls_per_level = permute_and_flatten(
          box_cls_per_level, N, A, C, H, W
      )
      box_cls_flattened.append(box_cls_per_level)

      box_regression_per_level = permute_and_flatten(
          box_regression_per_level, N, A, 4, H, W
      )
      box_regression_flattened.append(box_regression_per_level)
  # concatenate on the first dimension (representing the feature levels), to
  # take into account the way the labels were generated (with all feature maps
  # being concatenated as well)
  box_cls = Tensor.cat(*box_cls_flattened, dim=1).reshape(-1, C)
  box_regression = Tensor.cat(*box_regression_flattened, dim=1).reshape(-1, 4)
  return box_cls, box_regression

class SmoothL1Loss(Function):
  def abs(self, x: LazyBuffer) -> LazyBuffer:
    _x = x.e(UnaryOps.NEG)
    return x.e(BinaryOps.MAX, x.const(0)).e(BinaryOps.ADD, _x.e(BinaryOps.MAX, _x.const(0))) #abs

  def forward(self, x, t, beta=1. / 9, size_average=True): # (abs(x - t) < beta).where( 0.5 * abs(x - t) ** 2 / beta, abs(x - t) - 0.5 * beta))
    self.dif = x.e(BinaryOps.SUB, t)
    self.beta = beta
    n = self.abs(self.dif)
    cond = n.e(BinaryOps.CMPLT, n.const(beta))
    a = n.e(BinaryOps.MUL, n).e(BinaryOps.MUL, n.const(.5 / beta))
    loss = cond.e(TernaryOps.WHERE, a, n.e(BinaryOps.SUB, n.const(.5 * beta)))
    loss = Tensor(loss) #todo
    if size_average:
        return loss.mean().lazydata
    return loss.sum().lazydata

  def backward(self, grad_output): # (abs(x - t) < beta).where((x-t)/beta, (x-t).sign())
    cond = self.abs(self.dif).e(BinaryOps.CMPLT, self.dif.const(self.beta))
    grad_input = cond.e(TernaryOps.WHERE, self.dif.e(BinaryOps.DIV, self.dif.const(self.beta)),
      Tensor(self.dif).sign().lazydata) #todo sign
    return grad_input.e(BinaryOps.MUL, grad_output.reshape(tuple(1 for _ in grad_input.shape)).expand(grad_input.shape)), None

def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    return SmoothL1Loss.apply(input, target, beta=beta, size_average=size_average)

def test_fork_grad():
  optimizer = optim.SGD(Tensor([0,0,0,0,0]), lr=0.001, momentum=0.9, weight_decay=0.0005)
  res = smooth_l1_loss(
      Tensor([1, 4, 1, 9, 2], requires_grad=True),
      Tensor([2, 8, 1, 1, 4]),
      beta=2
  )
  assert res.numpy() == 2.25
  optimizer.zero_grad()
  res.backward()
  optimizer.step()
  

def test_match_targets_to_anchors():
  anchors = BoxList(Tensor([[0, 0, 10, 10], [0, 0, 5, 5]]), image_size = (50, 50)) # preds
  targets = BoxList(Tensor([[0, 0, 5, 5], [0, 0, 10, 10]]), image_size = (50, 50))
  hq_fn, _ = make_match_fn(0.7, 0.4)
  loss = RPNLossComputation(hq_fn, None, None, generate_rpn_labels)
  matched_targets, _ = loss.match_targets_to_anchors(anchors, targets)
  result = Tensor([[0, 0, 10, 10], [0, 0, 5, 5]])
  assert (matched_targets.bbox == result).numpy().all()

def test_prepare_targets():
  hq_fn, _ = make_match_fn(0.7, 0.4)
  sampler = make_balanced_sampler_fn(10, 0.5)
  rpn = RPNLossComputation(hq_fn, sampler, BoxCoder(weights=(1.0, 1.0, 1.0, 1.0)), generate_rpn_labels)
  labels,regression_targets = rpn.prepare_targets(
    [BoxList(Tensor([[0, 0, 10, 10], [0, 0, 5, 5], [12, 12, 14, 14]]), image_size = (50, 50))],
    [BoxList(Tensor([[0, 0, 5, 5], [0, 0, 10, 10]]), image_size = (50, 50))]
  )
  assert (labels[0] == Tensor([1, 1, 0])).numpy().all() # good matches, fg, bad matches, bg
  assert np.allclose(
    rpn.box_coder.decode(
      regression_targets[0],
      Tensor([[0, 0, 10, 10], [0, 0, 5, 5], [12, 12, 14, 14]])
    ).numpy(),
    Tensor([[0, 0, 10, 10], [0, 0, 5, 5], [0, 0, 5, 5]]).numpy(),
    atol=1e-6 ## TODO currently drift is 1e-7, why?
  )

def test_loss():
  hq_fn, _ = make_match_fn(0.7, 0.4)
  sampler = make_balanced_sampler_fn(10, 0.5)
  coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
  loss = RPNLossComputation(hq_fn, sampler, coder, generate_rpn_labels)
  channels=256
  anchor_generator = AnchorGenerator()
  backbone = ResNetFPN(ResNet(50, num_classes=None, stride_in_1x1=True), out_channels=256)
  rpn = RPNHead(
    channels, anchor_generator.num_anchors_per_location()[0]
  )
  optimizer = optim.SGD(backbone.parameters() + rpn.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
  img_id = 387655
  img = [Tensor(build_transforms()(Image.open(BASEDIR/f'train2017/000000{img_id}.jpg').convert("RGB")).numpy(), requires_grad=True)] # TODO this uses torch to transform
  images = to_image_list(img)
  features = backbone(images.tensors)
  objectness, rpn_box_regression = rpn(features)
  anchors = [anchor for anchor in anchor_generator(images, features)]
  coco = COCO(os.path.join(BASEDIR, 'annotations', 'instances_train2017.json'))
  annotations = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
  gt = []
  for annotation in annotations:
      bbox = annotation['bbox']  # [x,y,width,height]
      x, y, width, height = bbox
      gt.append([x, y, x + width, y + height])
  targets = [BoxList(Tensor(gt), image_size=anchors[0][0].size)]
  objectness_loss, regression_loss = loss(anchors, objectness, rpn_box_regression, targets)
  total_loss = objectness_loss + regression_loss
  optimizer.zero_grad()
  total_loss.backward()
  optimizer.step()

def binary_cross_entropy(pred: Tensor, y: Tensor): return -(pred.log()*y + (1-y)*(1-pred).log()).mean()
# todo eps is a bit big
def binary_cross_entropy_with_logits(x: Tensor, y: Tensor, eps: float=1e-7): return binary_cross_entropy(x.sigmoid().minimum(1-eps).maximum(eps), y)

def test_binary_cross_entropy_with_logits():
  x = Tensor([[ 2.3611, -0.8813, -0.5006, -0.2178],[0.0419, 0.0763, -1.0457, -1.6692]])
  y = Tensor([[0., 1., 0., 0.],[0., 1., 0., 0.]])
   # this test was built from a torch example and the numbers match
  assert np.allclose(binary_cross_entropy_with_logits(x, y).numpy(), 0.8233704)

# the version of this function in models.mask_rcnn has changed the parameter type to `Boxlist`, not `list[Boxlist]`
def cat_boxlist(bboxes: list[BoxList]):
  """Concatenates a list of BoxList (having the same image size) into a single BoxList"""
  assert isinstance(bboxes, (list, tuple))
  assert all(isinstance(bbox, BoxList) for bbox in bboxes)

  size = bboxes[0].size
  assert all(bbox.size == size for bbox in bboxes)
  mode = bboxes[0].mode
  assert all(bbox.mode == mode for bbox in bboxes)
  fields = set(bboxes[0].fields())
  assert all(set(bbox.fields()) == fields for bbox in bboxes)

  cat_boxes = BoxList(Tensor.cat(*[bbox.bbox for bbox in bboxes], dim=0), size, mode)
  for field in fields:
      data = Tensor.cat(*[bbox.get_field(field) for bbox in bboxes], dim=0)
      cat_boxes.add_field(field, data)
  return cat_boxes
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
    self.generate_labels_func = generate_labels_func
    self.discard_cases = ['not_visibility', 'between_thresholds']

  def match_targets_to_anchors(self, anchors: BoxList, targets: BoxList):
    match_quality_matrix = boxlist_iou(anchors, targets)
    matched_idxs = self.proposal_matcher(match_quality_matrix)
    if DEBUG > 0: print("matched_idxs", matched_idxs.numpy())
    matched_targets = targets[matched_idxs.maximum(0)] # drop negatives
    if DEBUG > 0: print("matched_targets", matched_targets.bbox.numpy())
    return matched_targets, matched_idxs

  def prepare_targets(self, anchors: List[BoxList], targets: List[BoxList]):
    labels = []
    regression_targets = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
      matched_targets, matched_idxs = self.match_targets_to_anchors(
          anchors_per_image, targets_per_image
      )

      # TODO this has fp errors
      regression_targets_per_image = self.box_coder.encode(
          matched_targets.bbox, anchors_per_image.bbox
      )
      # all matches become 1 (roi head) (.7 and above amplified)
      labels_per_image = self.generate_labels_func(matched_idxs)
      labels_per_image = labels_per_image.cast(dtype=dtypes.float32)

      # negative samples are labeled 0
      labels_per_image = (matched_idxs == -1).where(0, labels_per_image)

      # TODO: discard anchors that go out of the boundaries of the image
      # labels_per_image[~anchors_per_image.get_field("visibility")] = -1

      # discards weak signals (when lq matches is False), -1 is ignored by fg_bg_sampler
      labels_per_image = (matched_idxs == -2).where(-1, labels_per_image)
      labels.append(labels_per_image)
      regression_targets.append(regression_targets_per_image)

    return labels, regression_targets

  def __call__(self, anchors: list[list[BoxList]], objectness: list[Tensor],
               box_regression: list[Tensor], targets: list[BoxList]):
    """
    Arguments:
        anchors (list[list[BoxList]]) 
          - For each image, the anchors separated into different boxlists of different aspect ratios
        objectness (list[Tensor])
        box_regression (list[Tensor])
        targets (list[BoxList])

    Returns:
        objectness_loss (Tensor)
        box_loss (Tensor
    """
    if DEBUG > 0:
      anchors[0][0].bbox.realize(), targets[0].bbox.realize()
      print_gpu_memory("loss_start")
      if DEBUG > 1:
        print("anchors", anchors[0][0].bbox.numpy())
        print("targets", targets[0].bbox.numpy())
    anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
    labels, regression_targets = self.prepare_targets(anchors, targets)
    sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
    if DEBUG > 0:
      print_gpu_memory("after_sampling")
    if len(sampled_pos_inds[0]) == 0: return None, None # todo negative mining
    sampled_pos_inds, sampled_neg_inds = Tensor(sampled_pos_inds).squeeze(0), Tensor(sampled_neg_inds).squeeze(0)
    sampled_inds = Tensor.cat(sampled_pos_inds, sampled_neg_inds, dim=0)
    objectness, box_regression = \
            concat_box_prediction_layers(objectness, box_regression)
    objectness = objectness.squeeze() 
    labels, regression_targets = Tensor.cat(*labels, dim=0), Tensor.cat(*regression_targets, dim=0)
    if DEBUG > 0:
      box_regression[sampled_pos_inds].realize(), regression_targets[sampled_pos_inds].realize()
      print_gpu_memory("after_cats")
      if DEBUG > 1:
        print("box_reg", box_regression[sampled_pos_inds].numpy(), "reg_targets", regression_targets[sampled_pos_inds].numpy())
    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds],
        regression_targets[sampled_pos_inds],
        beta=1.0 / 9,
        size_average=False,
    ) / sampled_inds.numel()
    if DEBUG > 0:
      box_loss.realize(), objectness[sampled_inds].realize(), labels[sampled_inds].realize()
      print_gpu_memory("after_box_loss")
      if DEBUG > 1:
        print("box_loss", box_loss.numpy(), "objectness", objectness[sampled_inds].numpy(), "objectness gt", labels[sampled_inds].numpy())
    objectness_loss = binary_cross_entropy_with_logits(
        objectness[sampled_inds], labels[sampled_inds]
    )
    del objectness, labels, regression_targets, sampled_pos_inds, sampled_neg_inds, anchors, box_regression
    if DEBUG > 0: print_gpu_memory("after_cleanup")
    return objectness_loss, box_loss

if __name__ == "__main__":
  test_loss()
  test_fork_grad()
  download_train()
  test_concat_box_prediction_layers()
  test_binary_cross_entropy_with_logits()
  test_prepare_targets()
  test_boxlist_iou()
  test_match_eval()
  test_low_qual_match()
  test_rind()
  test_balanced_sampler()
  test_match_targets_to_anchors()
