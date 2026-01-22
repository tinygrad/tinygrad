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
		self.units = []
		self.logical_shapes:dict[str, tuple] = {}
		self.unit_prefix:dict[Any, str] = {}

		for name, param in nn.state.get_state_dict(self.module).items(): self.shard_param(name, param)
		for unit in self.units: self.wrap_call(unit)
	
	def shard_param(self, name:str, param:Tensor):
		self.logical_shapes[name] = param.shape
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
				self.unit_prefix[submod] = ".".join(parts[:-1])
		setattr(submod, parts[-1], param_sharded)

	def gather_param(self, shape:tuple[int], param:Tensor) -> Tensor:
		# creating a full copy of the param on all devices and do one tensor that have copies across devices
		# we use logical shapes to get the correct tensors without the padding for shape matching and avoiding unnessecary math ops on padded elements
		gathered = param.to(self.devices[0])
		slices = tuple(slice(0, s) for s in shape)
		return gathered[slices].shard(self.devices, axis=None)

	def wrap_call(self, unit:Any):
		ctx = self

		class Wrapper(unit.__class__):
			def __call__(self, *args, **kwargs):
				sharded_refs = {}
				unit_params = nn.state.get_state_dict(self).items()
				for name, param in unit_params:
					key = f"{ctx.unit_prefix[self]}.{name}"
					sharded_refs[name] = param
					shape = ctx.logical_shapes[key]
					setattr(self, name, ctx.gather_param(shape, param))
				try:
					return super().__call__(*args, **kwargs)
				finally:
					for name, param in sharded_refs.items(): setattr(self, name, param)
		unit.__class__ = Wrapper
					

	def __call__(self, x:Tensor) -> Tensor:
		return self.module(x)

	def sync_grad(self):
		pass
