from examples.mlperf.metrics import dice_score

def dice_ce_loss(pred, tgt):
  ce = pred.reshape(0, 2, 3, 4, 1).sparse_categorical_crossentropy(tgt)
  dice = (1.0 - dice_score(pred, tgt, argmax=False, one_hot_pred=False)).mean()
  return (dice + ce) / 2
