def get_available_llops():
  import os, importlib, inspect
  _buffers, DEFAULT = {}, "CPU"
  for op in [os.path.splitext(x)[0] for x in sorted(os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "llops"))) if x.startswith("ops_")]:
    name = op[len("ops_"):].upper()
    DEFAULT = name if os.environ.get(name, 0) == "1" else DEFAULT
    try:
      _buffers[name] = [cls for cname, cls in inspect.getmembers(importlib.import_module('tinygrad.llops.'+op), inspect.isclass) if (cname.upper() == name + "BUFFER")][0]
    except ImportError as e:  # NOTE: this can't be put on one line due to mypy issue
      print(op, "not available", e)
  return _buffers, DEFAULT

class Device:
  _buffers, DEFAULT = get_available_llops()
  for name in _buffers.keys():
    vars()[name] = name