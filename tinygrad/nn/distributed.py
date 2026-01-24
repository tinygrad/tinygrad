from tinygrad import Tensor
from tinygrad import nn
from typing import Any

class FSDP:
  def __init__(self, module:Any, devices:tuple[str, ...], axis:int=0):
    self.module = module
    self.devices = devices
    self.axis = axis
    self.ndev = len(self.devices)
    self.units = []
    self.logical_shapes:dict[str, tuple] = {}
    self.unit_prefix:dict[Any, str] = {}
    self.to_sync:list[tuple[Tensor, Tensor, tuple[int, ...]]] = []
    for name, param in nn.state.get_state_dict(self.module).items(): self.shard_param(name, param)
    for unit in self.units: self.wrap_call(unit)

  def shard_param(self, name:str, param:Tensor):
    # save the correct shape of the parameter and calculate the padded shape to use in sharding.
    self.logical_shapes[name] = param.shape
    rem = param.shape[self.axis] % self.ndev
    pad_width = (self.ndev - rem) % self.ndev
    padding = [(0, pad_width) if i == self.axis else (0, 0) for i in range(param.ndim)]
    # shard the parameter across training devices
    param_padded = param.pad(padding)
    param_sharded = param_padded.shard(self.devices, self.axis)
    param_sharded.requires_grad = True
    # get the basic building units of the model
    parts = name.split(".")
    submod = self.module
    for i, p in enumerate(parts[:-1]):
      submod = getattr(submod, p)
      if i == len(parts) - 2 and submod not in self.units:
        self.units.append(submod)
        self.unit_prefix[submod] = ".".join(parts[:-1])
    setattr(submod, parts[-1], param_sharded)

  def gather_param(self, shape:tuple[int], param:Tensor) -> Tensor:
    parts = []
    for i in range(self.ndev):
      # extract shard, ensure single device
      chunk = param.chunk(self.ndev, self.axis)[i].to(self.devices[i]).contiguous()
      parts.append(chunk.shard(self.devices, axis=None))
    full_padded = Tensor.cat(*parts, dim=self.axis)
    slices = tuple(slice(0, s) for s in shape)
    gathered = full_padded[slices]
    # detach to separate forward graph from backward communication
    out = gathered.detach()
    out.requires_grad = True
    self.to_sync.append((param, out, full_padded.shape))
    return out

  def wrap_call(self, unit:Any):
    ctx = self
    class Wrapper(unit.__class__):
      def __call__(self, *args, **kwargs):
        # save sharded params so after finishing the forward pass of the unit over the gathered parameters, we get back the sharded ones
        sharded_refs = {}
        unit_params = nn.state.get_state_dict(self).items()
        # get the unit params to do the forward pass on it
        for name, param in unit_params:
          prefix = ctx.unit_prefix.get(self)
          full_name = f"{prefix}.{name}" if prefix else name
          if full_name in ctx.logical_shapes:
            sharded_refs[name] = param
            shape = ctx.logical_shapes[full_name]
            setattr(self, name, ctx.gather_param(shape, param))
        try:
          return super().__call__(*args, **kwargs)
        finally:
          for name, param in sharded_refs.items(): setattr(self, name, param)
    unit.__class__ = Wrapper

  def __call__(self, x:Tensor) -> Tensor:
    self.to_sync = []
    if not isinstance(x.device, tuple):
      x = x.shard_(self.devices, self.axis)
    return self.module(x)

  def sync_grad(self):
    for sharded_param, gathered_out, padded_shape in self.to_sync:
      if gathered_out.grad is None: continue
      full_grad = gathered_out.grad
      # pad gradient back to the physically sharded shape
      padding = []
      for i in range(full_grad.ndim):
        padding.append((0, padded_shape[i] - full_grad.shape[i]))
      grad_padded = full_grad.pad(padding).contiguous()
      grad_sharded = grad_padded.to(self.devices[0]).shard(self.devices, self.axis)
      # assign the grad to the model sharded parameter
      sharded_param.grad = grad_sharded
    self.to_sync.clear()

