import torch
from torch.nn import functional as F
from tinygrad import getenv
from icecream import ic

if getenv("TINY_BACKEND"): import tinygrad.frontend.torch
device = torch.device("tiny" if getenv("TINY_BACKEND") else "mps")

# dims from hlb_cifar10 train image batch
# x = torch.ones(50000, 3, 32, 32, device=device)
x = torch.ones(4,1,1,1, device=device)
pad = (4,4,4,4)
out = F.pad(x, pad, 'reflect')
ic(out.cpu())


# x = torch.ones((4, 4), device=device)
# ic(x.cpu())
# pad2 = (2,2)
# out = F.pad(x, pad2, mode='constant', value=3)
# ic(out.cpu())

# out = F.pad(x, pad2, mode='reflect')
# ic(out.cpu())

# out = F.pad(x, pad2, mode='replicate')
# ic(out.cpu())
