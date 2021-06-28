#!/usr/bin/env python3
import numpy as np
import random
from PIL import Image

from tinygrad.optim import Adam
from datasets import MNIST, CIFAR10
import datasets.transforms as T
from extra.utils import get_parameters
from extra.training import train
from models.resnet import ResNet18, IMAGENET_MEAN, IMAGENET_STD


if __name__ == "__main__":
  for ds_cls in [MNIST, CIFAR10]:
    print(f'Train on {ds_cls}')
    model = ResNet18(num_classes=10, pretrained=True)
    transform = T.Compose([T.Normalize(IMAGENET_MEAN, IMAGENET_STD), T.G2RGB()])
    dl_train = ds_cls(transform=transform, train=True).dataloader(batch_size=16, shuffle=True, steps=100)
    dl_test = ds_cls(transform=transform, train=False).dataloader(batch_size=16, steps=100)
    train(model, Adam(model.parameters(), dl_train, dl_test, lr=1e-4), epochs=1)
