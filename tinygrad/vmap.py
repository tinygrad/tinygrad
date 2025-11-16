from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp

def vmap(func=None, in_axes=0):
    """
    vmap implementation for tinygrad.
    Supports:
    - @vmap decorator
    - arguments broadcasting (in_axes=(0, None))
    - multiple outputs (tuples)
    """
    if func is None:
        return lambda f: vmap(f, in_axes=in_axes)

    def wrapper(*args):
        # --- 1. SETUP ---
        if isinstance(in_axes, int):
            axes = [in_axes] * len(args)
        else:
            axes = in_axes 

        batch_size = None
        for arg, axis in zip(args, axes):
            if axis is not None:
                batch_size = arg.shape[axis]
                break
        
        if batch_size is None:
            raise ValueError("At least one argument must be mapped.")
        
        idxs = UOp.range(batch_size, -1)

        vmapped_args = []
        for arg, axis in zip(args, axes):
            if axis is None:
                vmapped_args.append(arg)
            else:
                sl = [slice(None)] * len(arg.shape)
                sl[axis] = idxs
                vmapped_args.append(arg[tuple(sl)])

        out = func(*vmapped_args)

        def _vectorize_output(o):
            if not isinstance(o, Tensor):
                return o
            
            reshape_dim = (1,) + o.shape
            expand_dim = (idxs,) + o.shape
            return o.reshape(reshape_dim).expand(expand_dim).contiguous()

        if isinstance(out, tuple):
            return tuple(_vectorize_output(o) for o in out)
        else:
            return _vectorize_output(out)
        
    return wrapper