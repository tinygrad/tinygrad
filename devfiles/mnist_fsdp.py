# model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters, Device
from tinygrad.helpers import getenv, trange
from tinygrad.nn.datasets import mnist
from tinygrad.helpers import prod
from tinygrad.tensor import Function
from typing import Type, Callable
from tinygrad.tensor import _METADATA 


# Function to shard the parameters of the optimizer (including the model itself)
def fsdp(obj, devices: tuple[str]):
  for name, param in nn.state.get_state_dict(obj).items():
    print(f"\n {name} {param.dtype.itemsize} {param.shape} \n ")
    if(param.shape[0] == 1 or prod(param.shape) <= 1):
      param.to_(devices)
    else:
      param.shard_(devices, axis=0)
      
  return obj

class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Linear(784, 2500, bias=False), Tensor.relu,
      nn.Linear(2500, 2000, bias=False), Tensor.relu,
      nn.Linear(2000, 1500, bias=False), Tensor.relu,
      nn.Linear(1500, 1000, bias=False), Tensor.relu,
      nn.Linear(1000, 500, bias=False), Tensor.relu,
      nn.Linear(500, 10, bias=False),
    ]

  def __call__(self, x:Tensor) -> Tensor: 
    x = x.flatten(1)
    for i, layer in enumerate(self.layers):
      print(f"\nLinear {i}, Input shape: {x.shape} \n ")
      if(i == 0 or i % 2):
        x = layer(x)
      else:
        x = layer(x)
      print(f"In Memory: {GlobalCounters.global_device_mem['CUDA'] //1000/1000:.1f} MB")
      print("---")
    
    return x

if __name__ == "__main__":
  Device.DEFAULT = "CUDA"
  #GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 2)))
  GPUS = ("CLANG", "CUDA")

  X_train, Y_train, X_test, Y_test = mnist()
  # we shard the test data on axis 0
  X_test.shard_(GPUS, axis=0)
  Y_test.shard_(GPUS, axis=0)
  Device.DEFAULT = "CLANG" #initialize the parameters on CPU
  print(f"\n Model \n ")
  model = Model()
  opt = fsdp(nn.optim.Adam(nn.state.get_parameters(model)), GPUS)
  print(f"\n End of modle init \n ")
  Device.DEFAULT = "CUDA"
  

  def train_step() -> Tensor:
    with Tensor.train():
      for param in opt.params:
        param.lazydata.placement = "replicate"
      opt.zero_grad()
      samples = Tensor.randint(getenv("BS", 64), high=X_train.shape[0])
      Xt, Yt = X_train[samples].shard_(GPUS, axis=0), Y_train[samples].shard_(GPUS, axis=0)  # we shard the data on axis 0
      loss = model(Xt).sparse_categorical_crossentropy(Yt)
      loss.backward(retain_graph=True)
      print(f"In Memory: {GlobalCounters.global_device_mem['CUDA'] //1000/1000:.1f} MB")
      # for n, t in reversed(list(nn.state.get_state_dict(opt).items())):
      #   if(t.requires_grad):
      #     print(f"Gradient {n}: {t.grad.shape}")
      #     t.grad.realize()
      #     print(f"In Memory: {GlobalCounters.global_device_mem['CUDA'] //1000/1000:.1f} MB")
      #     print("---")
      opt.step()
      return loss
    
  def get_test_acc() -> Tensor: return (model(X_test).argmax(axis=1) == Y_test).mean()*100

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 70))):
    loss = train_step()
    t.set_description(f"loss: {loss.item():6.2f}")

  print(f"Test Accuracy: {get_test_acc().item():6.2f}%")