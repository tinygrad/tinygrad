import functools
from tinygrad import Tensor, nn
from tinygrad.helpers import unwrap, dedup
from examples.mlperf.helpers import dedup_dict

class DistributedDataParallel:
  def __init__(self, model, ib:str, sender:bool):
    self.params: dict[str, Tensor] = {k:v for k,v in dedup_dict(nn.state.get_state_dict(model)).items() if v.requires_grad in {True, None}}
    self.ib, self.sender = ib, sender
  def sync(self):
    my_grads = {k:unwrap(p.grad).contiguous() for k,p in self.params.items()}
    other_grads = {}
    Tensor.realize(*my_grads.values())
    for name,my_grad in my_grads.items():
      if self.sender:
        my_grad.to('AMD').to(f'{self.ib}/{name}').realize()
        other_grads[name] = Tensor.empty(*my_grad.shape, device=f'{self.ib}/{name}', dtype=my_grad.dtype).to('AMD').realize()
      else:
        other_grads[name] = Tensor.empty(*my_grad.shape, device=f'{self.ib}/{name}', dtype=my_grad.dtype).to('AMD').realize()
        my_grad.to('AMD').to(f'{self.ib}/{name}').realize()
    for k,v in self.params.items():
      v.grad = (my_grads[k]+other_grads[k].to(my_grads[k].device))/2
    Tensor.realize(*(unwrap(p.grad) for p in self.params.values()))
