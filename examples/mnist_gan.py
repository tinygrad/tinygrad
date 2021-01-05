#!/usr/bin/env python
import os
import sys
import numpy as np
from tqdm import tqdm
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'test'))

from tinygrad.tensor import Tensor, Function, register
from extra.utils import get_parameters
import tinygrad.optim as optim
from test_mnist import X_train
from torchvision.utils import make_grid, save_image
import torch
GPU = os.getenv("GPU") is not None
class LinearGen:
  def __init__(self):
    lv = 128
    self.l1 = Tensor.uniform(128, 256)
    self.l2 = Tensor.uniform(256, 512)
    self.l3 = Tensor.uniform(512, 1024)
    self.l4 = Tensor.uniform(1024, 784)

  def forward(self, x):
    x = x.dot(self.l1).leakyrelu(0.2)
    x = x.dot(self.l2).leakyrelu(0.2)
    x = x.dot(self.l3).leakyrelu(0.2)
    x = x.dot(self.l4).tanh()
    return x

class LinearDisc:
  def __init__(self):
    in_sh = 784
    self.l1 = Tensor.uniform(784, 1024)
    self.l2 = Tensor.uniform(1024, 512)
    self.l3 = Tensor.uniform(512, 256)
    self.l4 = Tensor.uniform(256, 2)

  def forward(self, x, train=True):
    x = x.dot(self.l1).leakyrelu(0.2)
    if train:
      x = x.dropout(0.3)
    x = x.dot(self.l2).leakyrelu(0.2)
    if train:
      x = x.dropout(0.3)
    x = x.dot(self.l3).leakyrelu(0.2)
    if train:
      x = x.dropout(0.3)
    x = x.dot(self.l4).logsoftmax()
    return x

if __name__ == "__main__":
  generator = LinearGen()
  discriminator = LinearDisc()
  batch_size = 512
  k = 1
  epochs = 300
  generator_params = get_parameters(generator)
  discriminator_params = get_parameters(discriminator)
  gen_loss = []
  disc_loss = []
  output_folder = "outputs"
  os.makedirs(output_folder, exist_ok=True)
  train_data_size = len(X_train)
  ds_noise = Tensor(np.random.randn(64,128).astype(np.float32), requires_grad=False)
  n_steps = int(train_data_size/batch_size)
  if GPU:
    [x.gpu_() for x in generator_params+discriminator_params]
  # optimizers
  optim_g = optim.Adam(generator_params,lr=0.0002, b1=0.5) # 0.0002 for equilibrium!
  optim_d = optim.Adam(discriminator_params,lr=0.0002, b1=0.5)

  def regularization_l2(model, a=1e-4):
      #TODO: l2 reg loss
    pass

  def generator_batch():
    idx = np.random.randint(0, X_train.shape[0], size=(batch_size))
    image_b = X_train[idx].reshape(-1, 28*28).astype(np.float32)/255.
    image_b = (image_b - 0.5)/0.5
    return Tensor(image_b)

  def real_label(bs):
    y = np.zeros((bs,2), np.float32)
    y[range(bs), [1]*bs] = -2.0
    real_labels = Tensor(y)
    return real_labels

  def fake_label(bs):
    y = np.zeros((bs,2), np.float32)
    y[range(bs), [0]*bs] = -2.0 # Can we do label smoothin? i.e -2.0 changed to -1.98789.
    fake_labels = Tensor(y)
    return fake_labels

  def train_discriminator(optimizer, data_real, data_fake):
    real_labels = real_label(batch_size)
    fake_labels = fake_label(batch_size)

    optimizer.zero_grad()

    output_real = discriminator.forward(data_real)
    loss_real = (output_real * real_labels).mean()

    output_fake = discriminator.forward(data_fake)
    loss_fake = (output_fake * fake_labels).mean()

    loss_real.backward()
    loss_fake.backward()
    optimizer.step()
    return loss_real.cpu().data + loss_fake.cpu().data

  def train_generator(optimizer, data_fake):
    real_labels = real_label(batch_size)
    optimizer.zero_grad()
    output = discriminator.forward(data_fake)
    loss = (output * real_labels).mean()
    loss.backward()
    optimizer.step()
    return loss.cpu().data

  for epoch in tqdm(range(epochs)):
    loss_g = 0.0
    loss_d = 0.0
    print(f"Epoch {epoch} of {epochs}")
    for i in tqdm(range(n_steps)):
      image = generator_batch()
      for step in range(k): # Try with k = 5 or 7.
        noise = Tensor(np.random.randn(batch_size,128))
        data_fake = generator.forward(noise).detach()
        data_real = image
        loss_d_step = train_discriminator(optim_d, data_real, data_fake)
        loss_d += loss_d_step
      noise = Tensor(np.random.randn(batch_size,128))
      data_fake = generator.forward(noise)
      loss_g_step = train_generator(optim_g, data_fake)
      loss_g += loss_g_step
    fake_images = generator.forward(ds_noise).detach().cpu().data
    fake_images = (fake_images.reshape(-1, 1, 28, 28)+ 1) / 2 # 0 - 1 range.
    fake_images = make_grid(torch.tensor(fake_images))
    save_image(fake_images, os.path.join(output_folder,f"image_{epoch}.jpg"))
    epoch_loss_g = loss_g / n_steps
    epoch_loss_d = loss_d / n_steps
    print(f"EPOCH: Generator loss: {epoch_loss_g}, Discriminator loss: {epoch_loss_d}")
  else:
    print("Training Completed!")
