# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
import torch
import torch._dynamo
from torch._dynamo.backends.registry import register_backend
from torch._functorch.aot_autograd import aot_module_simplified
from torch._functorch._aot_autograd.graph_compile import make_boxed_func

from tinygrad import Tensor, TinyJit
from tinygrad.engine.jit import JitError
from tinygrad.uop.ops import Ops

def register_tiny_backend(wrap, unwrap):
  if "tiny" in torch._dynamo.list_backends(): return

  @register_backend
  def tiny(gm:torch.fx.GraphModule, sample_inputs):
    def my_compiler(gm:torch.fx.GraphModule, sample_inputs):
      out_node = next(n for n in gm.graph.nodes if n.op == "output")
      outs = out_node.args[0]
      if not isinstance(outs, (tuple, list)): outs = (outs,)
      out_mask = tuple(x is not None for x in outs)
      tensor_only_outs = True
      for o in outs:
        if o is None: continue
        if not isinstance(o, torch.fx.Node): tensor_only_outs = False; break
        if not isinstance(o.meta.get("val", None), torch.Tensor): tensor_only_outs = False; break
      gm_run = torch._dynamo.disable(gm)
      jit_ok = True

      @TinyJit
      def tiny_function(*args):
        outs = gm_run(*[wrap(x) if isinstance(x, Tensor) else x for x in args])
        if not isinstance(outs, (tuple, list)): outs = (outs,)
        outs = tuple(unwrap(x).realize() for x in outs if x is not None)
        return outs

      def torch_function(*args:torch.Tensor):
        nonlocal jit_ok
        targs = []
        for x in args:
          if not isinstance(x, torch.Tensor):
            targs.append(x)
            continue
          tx = unwrap(x) if x.device.type == "tiny" else unwrap(x.tiny())
          targs.append(tx.contiguous() if tx.uop.base.op is Ops.CONST else tx)
        if not tensor_only_outs:
          return gm_run(*[wrap(x) if isinstance(x, Tensor) else x for x in targs])
        try:
          touts = tiny_function(*targs) if jit_ok else tiny_function.fxn(*targs)
        except JitError:
          # Some graphs have no kernels (pure views / metadata ops). TinyJit can't capture them safely, so fall back.
          jit_ok = False
          touts = tiny_function.fxn(*targs)
        it = iter(touts)
        outs = [wrap(next(it)) if m else None for m in out_mask]
        return outs[0] if len(outs) == 1 else tuple(outs)
      return make_boxed_func(torch_function)

    return aot_module_simplified(gm, sample_inputs, decompositions={}, fw_compiler=my_compiler)