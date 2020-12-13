import numpy as np
from PIL import Image
import random
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'test'))
from test_mnist import fetch_mnist
from tqdm import trange

def augment_img(X):
  Xaug = np.zeros_like(X)
  for i in trange(len(X)):
    im = Image.fromarray(X[i])
    im = im.rotate(random.randint(-10,10))
    Xaug[i] = im
  return Xaug

if __name__ == "__main__":
  from test_mnist import fetch_mnist
  import matplotlib.pyplot as plt
  n=10
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  X = X_train[:n] #random.randint(0,len(X)-1,2)]
  fig, a = plt.subplots(2,n)
  Xaug = augment_img(X)
  for i in range(n):
    a[0][i].imshow(X[i])
    a[1][i].imshow(Xaug[i])
  plt.show()
