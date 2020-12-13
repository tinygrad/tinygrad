import numpy as np
from PIL import Image
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
    im = im.rotate(np.random.randint(-10,10), resample=Image.BICUBIC)
    w, h = X.shape[1:]
    #upper left, lower left, lower right, upper right
    quad = np.random.randint(-3,3,size=(8)) + np.array([0,0,0,h,w,h,w,0])
    im = im.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)
    Xaug[i] = im
  return Xaug

if __name__ == "__main__":
  from test_mnist import fetch_mnist
  import matplotlib.pyplot as plt
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  X = np.vstack([X_train[:1]]*10+[X_train[1:2]]*10)
  fig, a = plt.subplots(2,len(X))
  Xaug = augment_img(X)
  for i in range(len(X)):
    a[0][i].imshow(X[i])
    a[1][i].imshow(Xaug[i])
    a[0][i].axis('off')
    a[1][i].axis('off')
  plt.show()
