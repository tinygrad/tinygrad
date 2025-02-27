import extra.nv_gpu_driver.hook_cuda as hook_cuda

import ctypes, struct, platform, pathlib, os, binascii, itertools
from hexdump import hexdump
from tinygrad.helpers import to_mv, DEBUG, getenv, colored, time_to_str
from tinygrad.runtime.autogen import libc, cuda
from tinygrad.device import CPUProgram, Device
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.ops_cuda import cu_time_execution
from tinygrad import Tensor

TINY_MIRROR = getenv("TINY_MIRROR", 1)
TINY_REALIZE = getenv("TINY_REALIZE", 1)

if DEBUG >= 1: print("importing torch...")
import torch
if DEBUG >= 1: print("importing torch done:", torch.__version__, torch.__file__)

if TINY_MIRROR:
  print("importing tiny torch")
  import extra.torch_backend.backend as tiny_torch
  print("importing tiny torch done")

device = "cuda"
torch.set_default_device(device)
hook_cuda.push_prefix(colored("cuda", "cyan"))

cuda_to_tiny_mappings = {}

enumerator_aten_calls = itertools.count(0)
from torch.utils._python_dispatch import TorchDispatchMode
class DispatchLog(TorchDispatchMode):
  def __torch_dispatch__(self, func, types, args, kwargs=None):
    txt_args = []
    should_call_tiny = kwargs.get('device') is not None and kwargs['device'].type == "cuda"
    for arg in args:
      if torch.is_tensor(arg):
        if arg.device.type == "cuda": should_call_tiny = True
        txt_args.append(f"tensor({arg.shape} {arg.device} {arg.dtype})")
      else: txt_args.append(f"{arg}")
    for k,v in (kwargs or {}).items():
      if torch.is_tensor(v):
        if arg.device.type == "cuda": should_call_tiny = True
        txt_args.append(f"{k}:tensor({v.shape} {v.device} {v.dtype})")
      else: txt_args.append(f"{k}:{v}")

    # magenta-colored kerenls mirrored to tiny backend.
    print(colored(f"#{next(enumerator_aten_calls)} {func}", "magenta" if should_call_tiny else "cyan") + "("+", ".join(txt_args)+")", flush=True)

    orig_x = func(*args, **(kwargs or {}))
    if TINY_MIRROR and should_call_tiny:
      # replace with tiny tensor
      tiny_args, tiny_kwargs = [], {}
      for arg in args:
        if torch.is_tensor(arg): tiny_args.append(cuda_to_tiny_mappings[arg])
        else: tiny_args.append(arg)

      for k,v in (kwargs or {}).items():
        if torch.is_tensor(v): tiny_kwargs[k] = cuda_to_tiny_mappings[v]
        else: tiny_kwargs[k] = v
      if 'device' in tiny_kwargs and kwargs['device'].type == "cuda":
        tiny_kwargs['device'] = torch.device("tiny")

      if TINY_REALIZE: 
        torch.cuda.synchronize()
        cuda_events = hook_cuda.collect_events(clear=True)

      tiny_x = func(*tiny_args, **tiny_kwargs)
      
      # TODO: this is a hack, any way to do this better?
      if TINY_REALIZE:
        hook_cuda.push_prefix(colored("tiny", "magenta"))
        tiny_x.cpu()
        hook_cuda.pop_prefix()
        tiny_events = hook_cuda.collect_events(clear=True)

        def print_events(evs, name, out_addr):
          for ev in evs:
            if isinstance(ev, hook_cuda.HookKernelCallEvent):
              txt_params = []
              for param in ev.params:
                if isinstance(param, hook_cuda.HookTensorParamEvent):
                  is_out = param.cuda_address == out_addr
                  txt_params += [f"{'out' if is_out else 'in'} tensor{param.enum}({param.cuda_address:#x}, off={param.offset:#x})"]
              print(f"\t {name} kernel {ev.name[:24]} {ev.grid} {ev.block} {ev.ptm}\n\t\t({', '.join(txt_params)})")
            else: print("\t", name, ev)

        print_events(cuda_events, colored("cuda", "cyan"), orig_x.data_ptr())
        print_events(tiny_events, colored("tiny", "magenta"), 0x0)

      cuda_to_tiny_mappings[orig_x] = tiny_x
    return orig_x
DispatchLog().__enter__()

if __name__ == "__main__":
  if getenv("RESNET"):
    import torchvision.models as models
    model = models.resnet18(pretrained=True)
    model = model.cuda()
    model.eval()

    if getenv("COMPILE"): model = torch.compile(model)

    X = torch.rand(getenv("BS", 1), 3, 288, 288, device='cuda')
    model(X)

    print("\n\n\n****** second run ******\n")
    model(X)
  else:
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    a += 1
    b += 2
    a = a.exp2()
    b = b.exp2()
    a += b
    c = a @ b
    print("tensor math done", c.cpu().numpy())
