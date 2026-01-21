from tinygrad import Tensor
from tinygrad import nn
from typing import Any

def get_fsdp_units(fsdp_model:Any, units:list[Any]) -> Any:
	pass


class FSDP:
	def __init__(self, module:Any, devices:tuple[str, ...], axis:int=0):
		self.module = module
		self.devices:tuple[str, ...] = devices
		self.axis = axis
		self.ndev = len(self.devices)
		self.units =  []

		self.shard_model_()
		get_fsdp_units(self, self.units)
	
	def shard_model_(self):
		for name, param in nn.state.get_state_dict(self.module).items():
			self.shard_param(name, param)

	def shard_param(self, name:str, param:Tensor):
		# get the padding width needed to be able to shard the tensor around the sharding axis
		rem = param.shape[self.axis] % self.ndev
		pad_width = (self.ndev - rem) % self.ndev
		padding = [(0, pad_width) if i == self.axis else (0, 0) for i in range(param.ndim)]
		# pad the tensor then replace it with the sharded version
		param_padded = param.pad(padding)
		param_sharded = param_padded.shard_(self.devices, self.axis)
		parts = name.split(".")
		submod = self.module
		# replace all the model parameters with the new sharded ones
		for p in parts[:-1]:
			submod = getattr(submod, p)
		setattr(submod, parts[-1], param_sharded)

	def gather_param(self, param:Tensor) -> Tensor:
		# Creating a full copy of the param on all devices and do one tensor that have copies across devices
		return Tensor.cat([param.to(device) for device in self.devices])

	def gather_params(self, unit:Any):
		pass

	def sync_gradient(self):
		pass
