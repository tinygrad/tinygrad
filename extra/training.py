import os
import numpy as np
from tqdm import trange
from tinygrad.utils import fetch, get_parameters
from tinygrad.tensor import Tensor, GPU

def train(model, X_train, Y_train, optim, steps, num_classes=None, BS=128, gpu=False, lossfn = lambda out,y: out.mul(y).mean()):
  if gpu is True: [x.cuda_() for x in get_parameters([model, optim])]
  if num_classes is None: num_classes = Y_train.max().astype(int)+1
  losses, accuracies = [], []
  for i in (t := trange(steps, disable=os.getenv('CI') is not None)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))

    x = Tensor(X_train[samp].reshape((-1, 28*28)).astype(np.float32), gpu=gpu)
    Y = Y_train[samp]
    y = np.zeros((len(samp),num_classes), np.float32)
    # correct loss for NLL, torch NLL loss returns one per row
    y[range(y.shape[0]),Y] = -1.0*num_classes
    y = Tensor(y, gpu=gpu)

    # network
    out = model.forward(x)

    # NLL loss function
    loss = lossfn(out, y) 
    optim.zero_grad()
    loss.backward()
    optim.step()

    cat = np.argmax(out.cpu().data, axis=1)
    accuracy = (cat == Y).mean()

    # printing
    loss = loss.cpu().data
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

def evaluate(model, X_test, Y_test, num_classes=None, gpu=False, BS=128):
  def numpy_eval(num_classes):
    Y_test_preds_out = np.zeros((len(Y_test),num_classes))
    for i in trange(len(Y_test)//BS, disable=os.getenv('CI') is not None):
      Y_test_preds_out[i*BS:(i+1)*BS] = model.forward(Tensor(X_test[i*BS:(i+1)*BS].reshape((-1, 28*28)).astype(np.float32), gpu=gpu)).cpu().data
    Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
    return (Y_test == Y_test_preds).mean()

  if num_classes is None: num_classes = Y_test.max().astype(int)+1
  accuracy = numpy_eval(num_classes)
  print("test set accuracy is %f" % accuracy)
  return accuracy 
