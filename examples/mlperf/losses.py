from examples.mlperf.metrics import dice_score

def cross_entropy_loss(x, y, reduction='mean', label_smoothing=0.0):
  divisor = y.shape[1]
  y = (1 - label_smoothing)*y + label_smoothing / divisor
  if reduction == "none": return -x.log_softmax(axis=1).mul(y).sum(axis=1)
  if reduction == "sum": return -x.log_softmax(axis=1).mul(y).sum(axis=1).sum()
  return -x.log_softmax(axis=1).mul(y).sum(axis=1).mean()

def dice_ce_loss(pred, tgt):
  ce = cross_entropy_loss(pred, tgt)
  dice = (1.0 - dice_score(pred, tgt, argmax=False, to_one_hot_x=False)).mean()
  return (dice + ce) / 2
