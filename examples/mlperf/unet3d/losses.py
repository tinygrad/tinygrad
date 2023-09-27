
import torch.nn.functional as F
import torch
import numpy as np

from tinygrad.helpers import dtypes

from tinygrad.tensor import Tensor


def to_one_hot_old(array, layout, channel_axis):
  if len(array.shape) >= 5:
    array = torch.squeeze(array, dim=channel_axis)
  array = F.one_hot(array.long(), num_classes=3)
  if layout == "NCDHW":
    array = array.permute(0, 4, 1, 2, 3).float()
  return array

class Dice:
    def __init__(self,
                 to_onehot_x: bool = False,
                 to_onehot_y: bool = True,
                 use_softmax: bool = True,
                 use_argmax: bool = False,
                 include_background: bool = False,
                 layout: str = "NCDHW"):
        self.include_background = include_background
        self.to_onehot_x = to_onehot_x
        self.to_onehot_y = to_onehot_y
        self.use_softmax = use_softmax
        self.use_argmax = use_argmax
        self.smooth_nr = 1e-6
        self.smooth_dr = 1e-6
        self.layout = layout

    def __call__(self, prediction, target):
        """
        Input (default):
            prediction: (B, n_channels, H, W, D) logit tensor
            target: (B, n_channels, H, W, D) label tensor
        """
        if self.layout == "NCDHW":
            channel_axis = 1
            reduce_axis = list(range(2, len(prediction.shape)))
        else:
            channel_axis = -1
            reduce_axis = list(range(1, len(prediction.shape) - 1))
        
        num_pred_ch = prediction.shape[channel_axis]

        if self.use_softmax:
            prediction = prediction.softmax(axis=channel_axis)
        elif self.use_argmax:
            prediction = prediction.argmax(axis=channel_axis)

        if self.to_onehot_y:
          target = to_one_hot(target, self.layout, channel_axis)

        if self.to_onehot_x:
          prediction = to_one_hot(prediction, self.layout, channel_axis)

        if not self.include_background:
            assert num_pred_ch > 1, \
                f"To exclude background the prediction needs more than one channel. Got {num_pred_ch}."
            if self.layout == "NCDHW":
                target = target[:, 1:]
                prediction = prediction[:, 1:]
            else:
                target = target[..., 1:]
                prediction = prediction[..., 1:]

        assert (target.shape == prediction.shape), \
            f"Target and prediction shape do not match. Target: ({target.shape}), prediction: ({prediction.shape})."
            
        intersection = (target * prediction).sum(axis=reduce_axis)
        target_sum = target.sum(axis=reduce_axis)
        prediction_sum = prediction.sum(axis=reduce_axis)

        return (2.0 * intersection + self.smooth_nr) / (target_sum + prediction_sum + self.smooth_dr)

# >>> F.one_hot(torch.arange(0, 5) % 3)
# tensor([[1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1],
#         [1, 0, 0],
#         [0, 1, 0]])
# >>> F.one_hot(torch.arange(0, 5) % 3, num_classes=5)
# tensor([[1, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0],
#         [0, 0, 1, 0, 0],
#         [1, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0]])

def to_one_hot(array: Tensor, layout="NCDHW", channel_axis=1):
  # https://github.com/mlcommons/training/blob/00f04c57d589721aabce4618922780d29f73cf4e/image_segmentation/pytorch/model/losses.py#L63
  array = array.squeeze(dim=channel_axis) #todo could move squeeze up
  num_classes = 3
  array = Tensor.eye(num_classes, dtype=dtypes.int32, device=array.device)[array] # this is the F.one_hot function
  array = array.permute(0, 4, 1, 2, 3)
  return array

def cross_entropy(y_pred, y_true_onehot):
    y_pred = y_pred.softmax(axis=1)
    return -((y_true_onehot * y_pred.log()).sum(axis=1)).mean()

class DiceCELoss:
  def __init__(self, to_onehot_y=True, use_softmax=True, layout="NCDHW", include_background=False):
    self.dice = Dice(to_onehot_y=to_onehot_y, use_softmax=use_softmax, layout=layout,
                     include_background=include_background)

  def __call__(self, y_pred, y_true):
    ce = cross_entropy(y_pred, to_one_hot(y_true))

    dice = (1.0 - self.dice(y_pred, y_true)).mean()
    return (dice + ce) / 2


class DiceScore:
    def __init__(self, to_onehot_y: bool = True, use_argmax: bool = True, layout: str = "NCDHW",
                 include_background: bool = False):
        self.dice = Dice(to_onehot_y=to_onehot_y, to_onehot_x=True, use_softmax=False,
                         use_argmax=use_argmax, layout=layout, include_background=include_background)

    def __call__(self, y_pred, y_true):
        return self.dice(y_pred, y_true).mean(axis=0).realize()

# tests

def test_to_one_hot():
    labels = Tensor(np.arange(2*2*2*2).reshape(2, 1, 2, 2, 2) % 3).cast(dtypes.int32)
    assert labels.shape == (2, 1, 2, 2, 2)
    assert labels.dtype == dtypes.int32
    assert labels.numpy()[0, 0, 1, 0, 0] == 1
    
    one_hot = to_one_hot(labels, channel_axis=1)
    assert one_hot.shape == (2, 3, 2, 2, 2)
    assert one_hot.dtype == dtypes.float
    assert one_hot.numpy()[0, :, 1, 0, 0].tolist() == [0, 1, 0]
    
def test_cross_entropy():
    import torch.nn.functional as F
    
    y_pred = np.random.rand(2, 3, 128, 128, 128).astype(np.float32)
    y_true = np.random.randint(0, 3, size=(2, 1, 128, 128, 128))
    
    loss_torch = F.cross_entropy(torch.from_numpy(y_pred), torch.from_numpy(y_true).squeeze(dim=1).long())
    
    loss_our = cross_entropy(Tensor(y_pred), to_one_hot(Tensor(y_true)))
    
    assert np.allclose(loss_torch.item(), loss_our.numpy())
    
def test_dice():
    np.random.seed(42)
    size = 128
    y_pred = Tensor(np.random.rand(2, 3, size, size, size).astype(np.float32))
    y_true = Tensor(np.random.randint(0, 3, size=(2, 1, size, size, size)))
    dice = Dice()
    dice_score = dice(y_pred, y_true)
    assert dice_score.shape == (2, 2)
    assert np.allclose(dice_score.numpy(), np.ones((2,2))/3, atol=0.001)
    
def test_dice_ce_loss():
    np.random.seed(42) # important for reference number
    size = 128
    y_pred = Tensor(np.random.rand(2, 3, size, size, size).astype(np.float32))
    y_true = Tensor(np.random.randint(0, 3, size=(2, 1, size, size, size)))
    dice_loss = DiceCELoss(to_onehot_y=True, use_softmax=True, layout="NCDHW", include_background=False)
    
    loss = dice_loss(y_pred, y_true)
    assert np.allclose(loss.numpy(), 0.8964, atol=0.001) # computed from reference implementation
    
    
if __name__ == "__main__":
    test_to_one_hot()
    test_cross_entropy()
    test_dice()
    test_dice_ce_loss()