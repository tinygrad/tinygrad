from tinygrad import Tensor
import torch

x = Tensor([[1,1,1,1],
            [1,1,1,1],
            [1,1,1,1],
            [1,1,1,1]]).unsqueeze(0)
xt = torch.tensor(x.numpy())

k_ = (2,2)

m, m_i = Tensor.max_pool2d(x, k_, return_indices=True)
mt, mt_i = torch.nn.functional.max_pool2d(xt, k_, return_indices=True)

print(m_i.numpy())
print(mt_i.numpy())
