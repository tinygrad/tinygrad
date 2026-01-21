from tinygrad import Tensor
from tinygrad import nn
import functools
import math
from typing import Any


class FSDP:
    def __init__(self, module:Any, devices:tuple[str, ...], axis:int=0):
        self.module = module
        self.devices = devices
        self.axis = axis
        self.ndev = len(self.devices)
        self.sharded_params = []

        self.shard_params()
    
    def shard_params(self):
        for name, param in nn.state.get_state_dict(self.module).items():
            rem = param.shape[self.axis] % self.ndev
            pad_width = (self.ndev - rem) % self.ndev
            padding = [
                (0, pad_width) if i == self.axis else (0, 0)
                for i in range(param.ndim)
            ]
            param_padded = param.pad(padding)
            param_sharded = param_padded.shard_(self.devices, self.axis)

            parts = name.split(".")
            submod = self.module
            for p in parts[:-1]:
                submod = getattr(submod, p)
            setattr(submod, parts[-1], param_sharded)


    def gather_params(self):
        pass

    def sync_gradient(self):
        pass
