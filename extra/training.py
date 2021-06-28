import os
import numpy as np
from tqdm import trange
from extra.utils import get_parameters
from tinygrad.tensor import Tensor, GPU, Device
from tqdm import tqdm

def sparse_categorical_crossentropy(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten()
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  # correct loss for NLL, torch NLL loss returns one per row
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y)
  return out.mul(y).mean()

def train(model, optim, dl_train, dl_test=None, epochs=1, lossfn=sparse_categorical_crossentropy):
  losses, accuracies = {'train': [], 'eval': []}, {'train': [], 'eval': []}
  dls = {'train': dl_train, 'eval': dl_test}
  for epoch in range(epochs):
    for mode in ['train', 'eval']:
      if dls[mode] is not None:
        out = single_datalaoder_pass(model, optim, dls[mode], mode=mode, lossfn=lossfn)
        losses[mode] += [out['loss']]
        accuracies[mode] += [out['acc']]
  return losses, accuracies

def evaluate(model, dataloader, lossfn=sparse_categorical_crossentropy, return_predict=False):
  return single_datalaoder_pass(model, None, dataloader, mode='eval', return_predict=return_predict, lossfn=lossfn)

def single_datalaoder_pass(model, optim, dataloader, mode='train', return_predict=False,
                           lossfn=sparse_categorical_crossentropy):
  Tensor.training = mode == 'train'
  losses, accuracies, preds = [], [], []
  for x, y in (pbar := tqdm(dataloader)):
    x = Tensor(x)
    out = model.forward(x)
    if return_predict:
      preds += [out]
    loss = lossfn(out, y)
    if optim is not None:
      optim.zero_grad()
      loss.backward()
      optim.step()
    cat = np.argmax(out.cpu().data, axis=-1)
    accuracy = (cat == y).mean()

    # printing
    loss = loss.cpu().data
    losses += [loss[0]]
    accuracies += [accuracy]
    pbar.set_description(f'{mode} - loss {loss[0]:.2f}, accuracy {accuracy:.2f}')
  pbar.set_description(f'{mode} - loss {np.mean(losses):.2f}, accuracy {np.mean(accuracies):.2f}')
  pbar.close()
  outs = {'loss': np.mean(losses), 'acc': np.mean(accuracies)}
  if return_predict:
    outs['preds'] = np.concatenate(preds, 0)
  return outs
