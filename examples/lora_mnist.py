from collections import namedtuple
import sys

import numpy as np

from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from extra.datasets import fetch_mnist
from extra.training import train, evaluate


GPU = getenv("GPU")
Dataset = namedtuple("Dataset", ["X", "Y"])


def loraize(model, rank=1, alpha=0.5):
  # freeeeezzzzeeeee
  original_params = model.parameters()
  for par in original_params:
    par.requires_grad = False

  in_l1, out_l1 = model.l1.shape
  in_l2, out_l2 = model.l2.shape
  in_l3, out_l3 = model.l3.shape

  # add lora params
  model.loraB1 = Tensor.zeros(in_l1, rank, requires_grad=True)
  model.loraA1 = Tensor.randn(rank, out_l1, requires_grad=True) * (alpha / rank)
  model.loraB2 = Tensor.zeros(in_l2, rank, requires_grad=True)
  model.loraA2 = Tensor.randn(rank, out_l2, requires_grad=True) * (alpha / rank)
  model.loraB3 = Tensor.zeros(in_l3, rank, requires_grad=True)
  model.loraA3 = Tensor.randn(rank, out_l3, requires_grad=True) * (alpha / rank)

  return model


def lora_forward(self, x):
  x = x.dot(self.l1 + self.loraB1.dot(self.loraA1)).relu().dropout(0.3)
  x = x.dot(self.l2 + self.loraB2.dot(self.loraA2)).relu().dropout(0.3)
  x = x.dot(self.l3 + self.loraB3.dot(self.loraA3)).log_softmax()
  return x


class BigClassifierNetwork:
  def __init__(self, hidden_size_1=1000, hidden_size_2=2000) -> None:
    self.hidden_size_1 = hidden_size_1
    self.hidden_size_2 = hidden_size_2

    self.l1 = Tensor.scaled_uniform(28 * 28, self.hidden_size_1)
    self.l2 = Tensor.scaled_uniform(self.hidden_size_1, self.hidden_size_2)
    self.l3 = Tensor.scaled_uniform(self.hidden_size_2, 10)

  def forward(self, x):
    x = x.dot(self.l1).relu().dropout(0.3)
    x = x.dot(self.l2).relu().dropout(0.3)
    x = x.dot(self.l3).log_softmax()
    return x

  def parameters(self):
    return get_parameters(self)

  def save(self, filename):
    with open(filename + ".npy", "wb") as f:
      for par in get_parameters(self):
        np.save(f, par.numpy())

  def load(self, filename):
    with open(filename + ".npy", "rb") as f:
      for par in get_parameters(self):
        try:
          par.numpy()[:] = np.load(f)
          if GPU:
            par.gpu()
        except:
          print("Could not load parameter")


def filter_data_by_class(X, Y, class_label):
  class_indices = np.where(Y == class_label)[0]

  filtered_X = X[class_indices]
  filtered_Y = Y[class_indices]

  return filtered_X, filtered_Y


def _get_mislabeled_counts(y, y_pred) -> dict[int, float]:
  mislabeled_counts_dict: dict[int, float] = {cls: -np.inf for cls in range(10)}
  for cls in range(10):
    total_predictions = np.sum(y == cls)
    incorrect_predictions = np.sum((y == cls) & (y != y_pred))
    if total_predictions > 0:
      mislabeled_count = incorrect_predictions
    else:
      mislabeled_count = -np.inf
    mislabeled_counts_dict[cls] = mislabeled_count
  return mislabeled_counts_dict


def _pretty_print_mislabeled_counts(mislabeled_counts: dict[int, float]) -> None:
  for cls in mislabeled_counts.keys():
    print(f"Class {cls}: Missing {mislabeled_counts[cls]}")


def _pretty_print_parameters(parameters):
  pars = [par for par in parameters]
  no_pars = 0
  for i, par in enumerate(pars):
    print(f"layer {i + 1}: {par.shape}")
    no_pars += np.prod(par.shape)
  print("no of parameters", no_pars)
  return pars


if __name__ == "__main__":
  # Inspiration from [1] & [2]
  # [1] https://www.youtube.com/watch?v=PXWYUTMt-AU
  # [2] https://colab.research.google.com/drive/13okPgkUeK8BrSMz5PXwQ_FXgUZgWxYLp
  print("Simulating a pre-trained model, with one epoch..")
  lrs = [1e-3]
  epochss = [1]
  BS = 128

  lossfn = lambda out, y: out.sparse_categorical_crossentropy(y)
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  train_data = Dataset(X=X_train, Y=Y_train)
  test_data = Dataset(X=X_test, Y=Y_test)
  steps = len(X_train) // BS
  np.random.seed(222)

  model = BigClassifierNetwork()

  if GPU:
    params = get_parameters(model)
    [x.gpu_() for x in params]

  for lr, epochs in zip(lrs, epochss):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
      train(
        model,
        train_data.X,
        train_data.Y,
        optimizer,
        steps=steps,
        lossfn=lossfn,
        BS=BS,
      )

  print("After pre-training our model..")
  accuracy, Y_test_pred = evaluate(
    model, test_data.X, test_data.Y, BS=BS, return_predict=True
  )
  # pretrained_file_path = f"examples/pretrained{accuracy * 1e6:.0f}"
  # model.save(pretrained_file_path)

  mislabeled_counts = _get_mislabeled_counts(test_data.Y, Y_test_pred)
  _pretty_print_mislabeled_counts(mislabeled_counts)

  params = model.parameters()
  _pretty_print_parameters(params)

  FROZEN_PARAMS = params.copy()

  worst_class = max(mislabeled_counts, key=lambda k: mislabeled_counts[k])
  print(f"Worst class: {worst_class}")

  print("Lora-izing the model..")
  lora_model = loraize(model)
  setattr(lora_model, "forward", lambda x: lora_forward(lora_model, x))

  print(f"Fine-tuning the worst class, {worst_class}..")
  lrs = [1e-5]
  epochss = [1]
  BS = 128

  X_train, Y_train = filter_data_by_class(train_data.X, train_data.Y, worst_class)
  filtered_data = Dataset(X=X_train, Y=Y_train)

  if GPU:
    params = get_parameters(lora_model)
    [x.gpu_() for x in params if x.requires_grad is True]

  for lr, epochs in zip(lrs, epochss):
    optimizer = optim.Adam(lora_model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
      train(
        lora_model,
        filtered_data.X,
        filtered_data.Y,
        optimizer,
        steps=200,
        lossfn=lossfn,
        BS=BS,
      )

  print("Here's your fine-tuned model..")
  accuracy, Y_test_pred = evaluate(
    lora_model, test_data.X, test_data.Y, BS=BS, return_predict=True
  )
  # checkpoint_file_path = f"examples/checkpoint{accuracy * 1e6:.0f}"
  # lora_model.save(checkpoint_file_path)
  mislabeled_counts = _get_mislabeled_counts(test_data.Y, Y_test_pred)
  _pretty_print_mislabeled_counts(mislabeled_counts)

  params = lora_model.parameters()
  _pretty_print_parameters(params)

  frozen_params = [par for par in params if par.requires_grad is False]
  assert all(
    np.array_equal(par1.numpy(), par2.numpy())
    for par1, par2 in zip(frozen_params, FROZEN_PARAMS)
  )
  print("Congratulations on your success, champ..")
