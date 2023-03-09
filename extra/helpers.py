import time

class Timing(object):
  def __init__(self, prefix="", on_exit=None): self.prefix, self.on_exit = prefix, on_exit
  def __enter__(self): self.st = time.perf_counter_ns()
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.et = time.perf_counter_ns() - self.st
    print(f"{self.prefix}{self.et*1e-6:.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))
