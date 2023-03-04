import time

class Timing(object):
  def __enter__(self): self.st = time.monotonic_ns()
  def __exit__(self, exc_type, exc_val, exc_tb): print(f"{(time.monotonic_ns()-self.st)*1e-6:.2f} ms")
