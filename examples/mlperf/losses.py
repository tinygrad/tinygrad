from examples.mlperf.metrics import dice_score


def cross_entropy_loss(x, y, reduction="mean", label_smoothing=0.0):
  divisor = y.shape[1]
  assert isinstance(divisor, int), "only supported int divisor"
  y = (1 - label_smoothing) * y + label_smoothing / divisor
  ret = -x.log_softmax(axis=1).mul(y).sum(axis=1)
  if reduction == "none": return ret
  if reduction == "sum": return ret.sum()
  if reduction == "mean": return ret.mean()
  raise NotImplementedError(reduction)

def dice_ce_loss(pred, tgt, gpus=[]):
  ce = cross_entropy_loss(pred, tgt)
  dice = (1.0 - dice_score(pred, tgt, argmax=False, to_one_hot_x=False, gpus=gpus)).mean()
  return (dice + ce) / 2
