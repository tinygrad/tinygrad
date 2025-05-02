import random, os
from tinygrad.helpers import Timing, Profiling
from tinygrad import Device

if __name__ == "__main__":
  am = Device["AMD"]
  assert am.is_am(), "should run AM"

  with Profiling("allocation 127.7mb"):
    am.dev_iface.adev.mm.valloc(int(127.7*1024*1024))
