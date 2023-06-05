import numpy as np
from tqdm import trange
from tinygrad.tensor import Tensor, Device
from tinygrad.helpers import getenv

def sparse_categorical_crossentropy(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten().astype(np.int32)
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  # correct loss for NLL, torch NLL loss returns one per row
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y)
  return out.mul(y).mean()

# TODO: use the log-sum-exp trick to improve the numerical stability of cross-entropy.
def focal_loss(out, target, alpha:float=0.25, gamma=2, reduction='mean'):
  out, target = out.float(), target.float()
  p_t = out * target + (1.0 - out) * (1.0 - target)
  ce_loss = -(p_t + 1e-10).log()  # adding an epsilon in order to avoid log(0) case
  loss = ce_loss * ((1.0 - p_t) ** gamma)
  if alpha >= 0:
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * loss
  assert reduction in ['mean', 'sum']
  return loss.sum() if reduction == 'sum' else loss.mean()

def smooth_l1_loss(out: Tensor, target: Tensor, beta:float=1.0, reduction='mean') -> Tensor:
  out, target = out.float(), target.float()
  n = (out - target).abs()
  loss = ((1 - (n - beta).sign()) * 0.5 * n**2 / beta + (1 + (n - beta).sign()) * (n - 0.5*beta)) / 2
  if loss.numel() <= 0: return 0.0 * loss.sum()
  assert reduction in ['mean', 'sum']
  return loss.sum() if reduction == 'sum' else loss.mean()

def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=sparse_categorical_crossentropy, 
        transform=lambda x: x, target_transform=lambda x: x, noloss=False):
  Tensor.training = True
  losses, accuracies = [], []
  for i in (t := trange(steps, disable=getenv('CI', False))):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    x = Tensor(transform(X_train[samp]), requires_grad=False)
    y = target_transform(Y_train[samp])

    # network
    out = model.forward(x) if hasattr(model, 'forward') else model(x)

    loss = lossfn(out, y)
    optim.zero_grad()
    loss.backward()
    if noloss: del loss
    optim.step()

    # printing
    if not noloss:
      cat = np.argmax(out.cpu().numpy(), axis=-1)
      accuracy = (cat == y).mean()

      loss = loss.detach().cpu().numpy()
      losses.append(loss)
      accuracies.append(accuracy)
      t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
    

def evaluate(model, X_test, Y_test, num_classes=None, BS=128, return_predict=False, transform=lambda x: x, 
             target_transform=lambda y: y):
  Tensor.training = False
  def numpy_eval(Y_test, num_classes):
    Y_test_preds_out = np.zeros(list(Y_test.shape)+[num_classes])
    for i in trange((len(Y_test)-1)//BS+1, disable=getenv('CI', False)):
      x = Tensor(transform(X_test[i*BS:(i+1)*BS]))
      out = model.forward(x) if hasattr(model, 'forward') else model(x)
      Y_test_preds_out[i*BS:(i+1)*BS] = out.cpu().numpy()
    Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
    Y_test = target_transform(Y_test)
    return (Y_test == Y_test_preds).mean(), Y_test_preds

  if num_classes is None: num_classes = Y_test.max().astype(int)+1
  acc, Y_test_pred = numpy_eval(Y_test, num_classes)
  print("test set accuracy is %f" % acc)
  return (acc, Y_test_pred) if return_predict else acc

