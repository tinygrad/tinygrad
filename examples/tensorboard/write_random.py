import numpy as np
from extra.tensorboard.writer import TinySummaryWriter
from tinygrad.tensor import Tensor

if __name__ == "__main__":
  writer = TinySummaryWriter()
  for n_iter in range(10):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    writer.add_image('images', Tensor.randn(3, 100, 100), n_iter)
    writer.add_histogram('histogram', np.random.random(1000), n_iter)
  writer.close()