#from extra.datasets.imagenet import iterate
from extra.datasets.dataloader import cross_process, iterate
from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
input_mean = np.array(mean).reshape(-1, 1, 1)
input_std = np.array(std).reshape(-1, 1, 1)

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop, Normalize

@TinyJit
def normalize(X: Tensor):
  input_mean = Tensor(mean).reshape(-1,1,1)
  input_std = Tensor(std).reshape(-1,1,1)
  X = X.permute(0,3,1,2) / 255.0
  X -= input_mean
  X /= input_std
  return X.realize()

# BS=16,W=8,decoder=jpeg_decode
# total = 32

# BS=16,W=8,decoder=PIL
# total = 37

# BS=32,W=8
# norm batched
# F.norm = 20
# total = 77

# BS=32,W=4
# F.norm = 20
# data = 85
from models.resnet import ResNet50

if __name__ == '__main__':
  model = ResNet50()
  BS,W=16,8
  import time
  n = Normalize(mean, std)
  s = time.perf_counter()
 # for i, (X,Y) in enumerate(cross_process(lambda: iterate(bs=32,val=False,shuffle=True))):
  norm,data = [],[]
  for i, (X,Y) in enumerate(cross_process(lambda: iterate(bs=BS,val=False,shuffle=True,num_workers=W))):
    n1 = time.perf_counter()
    # TODO: move to gpu, may be faster. 
    #X = torch.from_numpy(X.transpose([0,3,1,2])) / 255.0
    #X = F.normalize(X, mean, std)
    X = normalize(Tensor(X, requires_grad=False))
    # do i need to move to cpu? 
    n2 = time.perf_counter()
    s1 = time.perf_counter()  
    t = s1-s
    s = time.perf_counter()
    #print(X.shape)
    print(f'total data {(t)*1000:7.2f}ms norm {(n2-n1)*1000:7.2f}ms')
    data.append(t)
    norm.append(n2-n1)
    if i!= 0 and i % (300) == 0: break
  data,norm = data[1:],norm[1:]
  print(f'data avg tm {(sum(data)/len(data))*1000:7.2f} avg tm {(sum(norm)/len(norm))*1000:7.2f}')
  