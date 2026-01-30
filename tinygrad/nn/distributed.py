from tinygrad import Tensor
from tinygrad.uop.ops import UOp
from tinygrad import nn
from typing import Any

class FSDP:
  def __init__(self, module:Any, devices:tuple[str, ...], axis:int=0):
    self.module, self.devices, self.axis, self.ndev = module, devices, axis, len(devices)
    params = nn.state.get_state_dict(self.module)
    self.logical_shapes:dict[str, tuple[int|UOp, ...]] = {}
    self.sharded_params:dict[str, Tensor] = {}
    for name, param in params.items():
      # save the original shapes of the params before padding and sharding for the all-gather step
      self.logical_shapes[name] = param.shape
      # NOTE: we do padding before sharding to make sure that the shape of the parameter is appropriate for the sharding
      rem = param.shape[axis] % self.ndev
      pad_width = int((self.ndev - rem) % self.ndev)
      padding = tuple((0, pad_width) if i == axis else (0, 0) for i in range(param.ndim))
      padded_param = param.pad(padding).contiguous().to(self.devices[0]).realize()
      sharded_param = padded_param.shard(self.devices, self.axis).realize()
      sharded_param.requires_grad = param.requires_grad
      sharded_param.grad = Tensor.zeros(padded_param.shape, device=self.devices[0]).shard(self.devices, self.axis).realize()
      self.sharded_params[name] = sharded_param
      # replace the module tensors with the sharded versions
      self._set_attr(name, sharded_param)

  def _get_p(self, name:str) -> tuple[Any, str]:
    """Search for the parameter in the module"""
    submod = self.module
    parts = name.split('.')
    for p in parts[:-1]: submod = submod[int(p)] if p.isdigit() else getattr(submod, p)
    return submod, parts[-1]

  def _set_attr(self, name:str, val:Tensor):
    """Set parameters in the module using it's name in the state_dict"""
    p, a = self._get_p(name)
    if a.isdigit(): p[int(a)] = val
    else: setattr(p, a, val)

  def __call__(self, x:Tensor) -> Tensor:
    # all gathering with slices to make sure that we use the orignal tensor without the padded zeros for efficiency
    for name, sharded_param in self.sharded_params.items():
      gathered = sharded_param.to(self.devices[0]).shard(self.devices, axis=None)
      logical_shape = self.logical_shapes[name]
      if gathered.shape != logical_shape:
        slices = tuple(slice(0, s) for s in logical_shape)
        gathered = gathered[slices]
      self._set_attr(name, gathered)
    return self.module(x)

  def sync_grad(self):
    # reduce-scatter
    for name, sharded in self.sharded_params.items():
      if sharded.requires_grad and sharded.grad is not None:
        sharded.grad.assign(sharded.grad / self.ndev)
    self._set_attr(name, sharded)
