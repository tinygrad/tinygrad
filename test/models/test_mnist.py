#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import Tensor, Device
from tinygrad.nn import optim, BatchNorm2d
from extra.training import train, evaluate
from datasets import fetch_mnist
import multiprocessing

# Preprocess and save the MNIST dataset in a more efficient format
def preprocess_mnist():
    X_train, Y_train, X_test, Y_test = fetch_mnist()
    np.savez('mnist_data.npz', X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

# Load the preprocessed MNIST dataset
def load_mnist():
    data = np.load('mnist_data.npz')
    return data['X_train'], data['Y_train'], data['X_test'], data['Y_test']

# Create a model
class TinyBobNet:
    def __init__(self):
        self.l1 = Tensor.scaled_uniform(784, 128)
        self.l2 = Tensor.scaled_uniform(128, 10)

    def parameters(self):
        return optim.get_parameters(self)

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).log_softmax()

class TestMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Preprocess and save the MNIST dataset only once
        preprocess_mnist()

        # Load the preprocessed dataset
        cls.X_train, cls.Y_train, cls.X_test, cls.Y_test = load_mnist()

    def test_sgd_onestep(self):
        np.random.seed(1337)
        model = TinyBobNet()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        train(model, self.X_train, self.Y_train, optimizer, BS=69, steps=1)
        for p in model.parameters():
            p.realize()

    def test_sgd_threestep(self):
        np.random.seed(1337)
        model = TinyBobNet()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        train(model, self.X_train, self.Y_train, optimizer, BS=69, steps=3)

    def test_sgd_sixstep(self):
        np.random.seed(1337)
        model = TinyBobNet()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        train(model, self.X_train, self.Y_train, optimizer, BS=69, steps=6, noloss=True)

    def test_adam_onestep(self):
        np.random.seed(1337)
        model = TinyBobNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, self.X_train, self.Y_train, optimizer, BS=69, steps=1)
        for p in model.parameters():
            p.realize()

    def test_adam_threestep(self):
        np.random.seed(1337)
        model = TinyBobNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, self.X_train, self.Y_train, optimizer, BS=69, steps=3)

    def test_conv_onestep(self):
        np.random.seed(1337)
        model = TinyConvNet()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        train(model, self.X_train, self.Y_train, optimizer, BS=69, steps=1, noloss=True)
        for p in model.parameters():
            p.realize()

    def test_conv(self):
        np.random.seed(1337)
        model = TinyConvNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, self.X_train, self.Y_train, optimizer, steps=100)
        assert evaluate(model, self.X_test, self.Y_test) > 0.94   # torch gets 0.9415 sometimes

    def test_conv_with_bn(self):
        np.random.seed(1337)
        model = TinyConvNet(has_batchnorm=True)
        optimizer = optim.AdamW(model.parameters(), lr=0.003)
        train(model, self.X_train, self.Y_train, optimizer, steps=200)
        assert evaluate(model, self.X_test, self.Y_test) > 0.94

    def test_sgd(self):
        np.random.seed(1337)
        model = TinyBobNet()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        train(model, self.X_train, self.Y_train, optimizer, steps=600)
        assert evaluate(model, self.X_test, self.Y_test) > 0.94   # CPU gets 0.9494 sometimes

if __name__ == '__main__':
    # Use multiprocessing to speed up test execution
    multiprocessing.set_start_method('fork')
    unittest.main()
