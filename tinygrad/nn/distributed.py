from tinygrad import Tensor
from tinygrad import nn
from typing import Any
from functools import wraps


class FSDP:
	def __init__(self, module:Any, devices:tuple[str, ...], axis:int=0):
		self.module = module
		self.devices:tuple[str, ...] = devices
		self.axis = axis
		self.ndev = len(self.devices)
		self.units =  []

		for name, param in nn.state.get_state_dict(self.module).items(): self.shard_param(name, param)
		for unit in self.units: self.wrap_call(unit)
	
	def shard_param(self, name:str, param:Tensor):
		# get the padding width needed to be able to shard the tensor around the sharding axis
		rem = param.shape[self.axis] % self.ndev
		pad_width = (self.ndev - rem) % self.ndev
		padding = [(0, pad_width) if i == self.axis else (0, 0) for i in range(param.ndim)]
		# pad the tensor then replace it with the sharded version
		param_padded = param.pad(padding)
		param_sharded = param_padded.shard_(self.devices, self.axis)
		# get parameters and submodules of the model
		parts = name.split(".")
		submod = self.module
		# replace all the model parameters with the new sharded ones
		for i, p in enumerate(parts[:-1]):
			submod = getattr(submod, p)
			# if this is a leaf module in the model, append to the units to modify its __call__ using wrappers
			if i == len(parts) - 2 and submod not in self.units:
				self.units.append(submod)
		setattr(submod, parts[-1], param_sharded)

	def gather_param(self, param:Tensor) -> Tensor:
		# Creating a full copy of the param on all devices and do one tensor that have copies across devices
		return Tensor.cat([param.to(device) for device in self.devices])

	def wrap_call(self, unit:Any):
		orig_call = getattr(unit, "__call__")	
		@wraps(orig_call)
		def wrapped(*args, **kwargs):
			sharded_params = {}
			unit_params = nn.state.get_state_dict(unit).items()
			# gathering the params of the unit before doing the call
			for name, param in unit_params:
				sharded_params[name] = param.clone()
				setattr(unit, name, self.gather_param(param))
			out = orig_call(*args, **kwargs)
			# sharding the params of the units again
			for name, param in unit_params: setattr(unit, name, sharded_params[name])
			return out
		setattr(unit, "__call__", wrapped)

	def __call__(self, x:Tensor) -> Tensor:
		return self.module(x)

	def sync_gradient(self):
		pass
