from tinygrad import nn
import functools

def fsdp(devices:tuple[str, ...], shard_axis:int=0):
    def decorator(cls):
        init = cls.__init__
        @functools.wraps(init)
        def wrap(self, *args, **kwargs):
            init(self, *args, **kwargs)
            for param in nn.state.get_parameters(self):
                param = param.shard_(devices, shard_axis)
        cls.__init__ = wrap
        return cls
    return decorator
