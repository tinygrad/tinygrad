from tinygrad import Tensor
from matplotlib import pyplot as plt

def imshow(x):
    if isinstance(x,Tensor): x = x.numpy()
    while len(x.shape) > 2: x = x[:,:,0]
    plt.imshow(x[:,:])
    plt.show()