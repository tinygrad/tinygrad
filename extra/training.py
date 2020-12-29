import os
import numpy as np
from tqdm import trange
from extra.utils import get_parameters
from tinygrad.tensor import Tensor, GPU, Device

def sparse_categorical_crossentropy(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten()
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  # correct loss for NLL, torch NLL loss returns one per row
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y, device=out.device)
  return out.mul(y).mean()

def train(model, X_train, Y_train, optim, steps, BS=128, device=Device.CPU, lossfn=sparse_categorical_crossentropy):
  Tensor.training = True
  if device == Device.GPU: [x.gpu_() for x in get_parameters([model, optim])]
  elif device == Device.ANE: [x.ane_() for x in get_parameters([model, optim])]
  losses, accuracies = [], []
  for i in (t := trange(steps, disable=os.getenv('CI') is not None)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))

    x = Tensor(X_train[samp], device=device)
    y = Y_train[samp]

    # network
    out = model.forward(x)

    loss = lossfn(out, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    cat = np.argmax(out.cpu().data, axis=-1)
    accuracy = (cat == y).mean()

    # printing
    loss = loss.cpu().data
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

def evaluate(model, X_test, Y_test, num_classes=None, device=Device.CPU, BS=128):
  Tensor.training = False
  def numpy_eval(num_classes):
    Y_test_preds_out = np.zeros(list(Y_test.shape)+[num_classes])
    for i in trange(len(Y_test)//BS, disable=os.getenv('CI') is not None):
      Y_test_preds_out[i*BS:(i+1)*BS] = model.forward(Tensor(X_test[i*BS:(i+1)*BS], device=device)).cpu().data
    Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
    return (Y_test == Y_test_preds).mean()

  if num_classes is None: num_classes = Y_test.max().astype(int)+1
  accuracy = numpy_eval(num_classes)
  print("test set accuracy is %f" % accuracy)
  return accuracy

