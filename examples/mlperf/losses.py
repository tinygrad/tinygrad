from tinygrad import Tensor
from examples.mlperf.metrics import dice_score

def dice_ce_loss(pred, tgt):
  ce = pred.permute(0, 2, 3, 4, 1).sparse_categorical_crossentropy(tgt.squeeze(1))
  dice = (1.0 - dice_score(pred, tgt, argmax=False, to_one_hot_x=False)).mean()
  return (dice + ce) / 2

def sigmoid_focal_loss(inputs:Tensor, targets:Tensor, mask:Tensor, alpha = 0.25, gamma = 2.0):
  def cust_bin_cross_logits(inputs:Tensor, targets:Tensor): 
    return inputs.maximum(0) - targets * inputs + (1 + inputs.abs().neg().exp()).log()

  p = Tensor.sigmoid(inputs) * mask
  ce_loss = cust_bin_cross_logits(inputs, targets)
  p_t = p * targets + (1 - p) * (1 - targets)
  loss = ce_loss * ((1 - p_t) ** gamma)

  if alpha >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

  loss = loss * mask
  loss = loss.sum(-1)
  loss = loss.sum(-1)
  return loss

def l1_loss(x1:Tensor, x2:Tensor):
  ans = (x1 - x2).abs().sum(-1)
  ans = ans.sum(-1)
  return ans