import array
from hexdump import hexdump
from tinygrad.runtime.support.am.usb import USBConnector

usb = USBConnector("")

def print_cfg(bus, dev):
  cfg = []
  for i in range(0, 256, 4):
    cfg.append(usb.pcie_cfg_req(i, bus=bus, dev=dev, fn=0, value=None, size=4))
  
  print("bus={}, dev={}".format(bus, dev))
  hexdump(bytearray(array.array('I', cfg)))

try: print_cfg(0, 0)
except: print("bus=0, dev=0 failed")

try: print_cfg(1, 0)
except: print("bus=1, dev=0 failed")

try: print_cfg(2, 0)
except: print("bus=2, dev=0 failed")
