from tinygrad import Tensor
from typing import List
from tinygrad.multi import MultiLazyBuffer

def get_size(tensors: List[Tensor]): return sum([t.nbytes() if isinstance(t, Tensor) else t.size for t in tensors])

def print_size(name, *tensors: Tensor):
    size = get_size(tensors)
    for unit in ['bytes', 'KB', 'MB', 'GB']:
        if size < 1000 or unit == 'GB': break
        size /= 1000
    print(f'{name} size: {size:.2f} {unit}')

def print_lb(lb):
  seen = set()
  def repr_lb(lb, indent=0):
    text = "\u001b[2m" if lb in seen else ""
    seen.add(lb)
    def _indent(): return " " * indent
    if isinstance(lb, MultiLazyBuffer):
      text += _indent() + "MLB "
      if lb.axis is not None:
        text += _indent() + f"shape {lb.shape} axis {lb.axis}"
        text += _indent() + f" bounds {lb.bounds}"
      text += "\n"
      indent += 4
      text += _indent()
      for i, _lb in enumerate(lb.lbs):
        text += "\u001b[0m"
        text += repr_lb(_lb, indent)
        if i < len(lb.lbs) - 1: text += _indent()
    else:
      text += "LB"
      view = lb.st.views[0]
      text += f" shape {view.shape}"
      text += f" strides {view.strides}"
      text += f" offset {view.offset}" if view.offset != 0 else ""
      text += f" mask {view.mask}" if view.mask is not None else ""
      text += "\n"
      indent += 3
      if hasattr(lb, "_base") and lb._base is not None:
        text += _indent() + "_base: "
        text += repr_lb(lb._base, indent + 7)
        return text
      else:
        text += _indent() + f"\u001b[35m{lb.op}\u001b[30m "
        text += f"arg: {lb.arg} " if lb.arg is not None else ""
        text += f"size: {lb.size} "
        text += f"device: {lb.device} \n"
        if hasattr(lb, "buffer") and hasattr(lb.buffer, "_buf"):
          text += _indent() + f"buf: {lb.buffer.nbytes} bytes\n"
          return text
        elif len(lb.srcs) > 0:
          text += _indent() + "srcs: "
          indent += 6
          for i, src in enumerate(lb.srcs):
            text += "\u001b[0m"
            text += repr_lb(src, indent)
            if i < len(lb.srcs) - 1: text += _indent()
    text += "\u001b[0m"
    return text
  print(repr_lb(lb))

