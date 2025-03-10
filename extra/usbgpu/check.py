import array, time
from hexdump import hexdump
from tinygrad.runtime.support.am.usb import USBConnector
from tinygrad.runtime.autogen import pci

usb = USBConnector("")

def print_cfg(bus, dev):
  cfg = []
  for i in range(0, 256, 4):
    #print("cfg", i)
    cfg.append(usb.pcie_cfg_req(i, bus=bus, dev=dev, fn=0, value=None, size=4))

  print("bus={}, dev={}".format(bus, dev))
  dmp = bytearray(array.array('I', cfg))
  hexdump(dmp)
  return dmp

def rescan_bus(bus, gpu_bus):
  print("set PCI_SUBORDINATE_BUS bus={} to {}".format(bus, gpu_bus))
  usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=bus, dev=0, fn=0, value=gpu_bus, size=1)
  usb.pcie_cfg_req(pci.PCI_SECONDARY_BUS, bus=bus, dev=0, fn=0, value=bus+1, size=1)
  usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=max(0, bus-1), size=1)

  print("rescan bus={}".format(bus))
  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_BUS_RESET, size=1)
  time.sleep(0.1)
  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_PARITY|pci.PCI_BRIDGE_CTL_SERR, size=1)

  usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x1000, size=2)
  usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0x2000, size=2)
  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x2000, size=2)
  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xffff, size=2)

print_cfg(0, 0)
#exit(0)

rescan_bus(0, gpu_bus=4)

print_cfg(1, 0)
#try:
#except: print("bus=1, dev=0 failed")

rescan_bus(1, gpu_bus=4)

# sleep after we rescan the bus
time.sleep(0.1)

print_cfg(2, 0)
#try
#except: print("bus=2, dev=0 failed")
#exit(0)

def setup_bus(bus, gpu_bus):
  print("setup bus={}".format(bus))
  usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=bus, dev=0, fn=0, value=gpu_bus, size=1)
  usb.pcie_cfg_req(pci.PCI_SECONDARY_BUS, bus=bus, dev=0, fn=0, value=bus+1, size=1)
  usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=max(0, bus-1), size=1)

  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_BUS_RESET, size=1)
  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_PARITY|pci.PCI_BRIDGE_CTL_SERR, size=1)
  usb.pcie_cfg_req(pci.PCI_COMMAND, bus=bus, dev=0, fn=0, value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)

  usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x1000, size=2)
  usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0x2000, size=2)

  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x2000, size=2)
  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xffff, size=2)

setup_bus(2, gpu_bus=4)
print_cfg(3, 0)
#try:
#except: print("bus=3, dev=0 failed")

setup_bus(3, gpu_bus=4)
dmp = print_cfg(4, 0)
print(dmp[0:4])
assert dmp[0:4] == b"\x02\x10\x80\x74", "GPU NOT FOUND!"
#try:
#except: print("bus=4, dev=0 failed")
