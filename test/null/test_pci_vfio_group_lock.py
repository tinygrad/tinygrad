import os, sys, threading, types
import pytest

@pytest.mark.skipif(sys.platform != "linux", reason="uses linux sysfs layout")
def test_vfio_group_bind_and_attach_under_one_lock(monkeypatch):
  # two cards of one iommu group opened by two threads at once (ts5 Nov 14 15:23: 89 + 8c of group 10): every bind / unbind of a card and the
  # group's attach must happen under PCIDevice._vfio_lock, else the group can be attached while a sibling is between drivers (EPERM). The
  # attach is retried on EPERM; the group is opened once
  import tinygrad.runtime.support.system as system
  bound:dict[str, str|None] = {"0000:89:00.0": None, "0000:8c:00.0": None}
  log:list[tuple[str, bool]] = []
  gdir = "/sys/kernel/iommu_groups/10/devices"

  class FakeFIO:
    def __init__(self, path="", flags=0, fd=None): self.path, self.fd = path, (fd if fd is not None else os.open("/dev/null", os.O_RDONLY))
    def write(self, content, *a, **k):
      log.append((self.path, system.PCIDevice._vfio_lock.locked()))
      if self.path.endswith("drivers_probe"): bound[content] = "vfio-pci"
      if self.path.endswith("/driver/unbind"): bound[content] = None
    @staticmethod
    def exists(path):
      if path.endswith("/driver"): return bound.get(path.split("/")[-2]) is not None
      return path.endswith("/iommu_group") or path.endswith("/enable")
    @staticmethod
    def readlink(path): return "../../../kernel/iommu_groups/10"
    @staticmethod
    def eventfd(initval, flags=None): return FakeFIO(fd=os.eventfd(0, 0))

  attaches = {"n": 0}
  def set_container(grp, fd):
    log.append(("SET_CONTAINER", system.PCIDevice._vfio_lock.locked()))
    attaches["n"] += 1
    if attaches["n"] < 3: raise PermissionError(1, "Operation not permitted")  # (not viable for a moment: retried)
  fake_vfio = types.SimpleNamespace(VFIO_GROUP_SET_CONTAINER=set_container, VFIO_SET_IOMMU=lambda *a: 0, VFIO_TYPE1v2_IOMMU=3,
                                    VFIO_GROUP_GET_DEVICE_FD=lambda grp, name: os.open("/dev/null", os.O_RDONLY),
                                    struct_vfio_irq_set=lambda **k: type("irq", (), {"argsz": 24, "__bytes__": lambda self: bytes(20)})(),
                                    VFIO_PCI_MSI_IRQ_INDEX=1,
                                    VFIO_IRQ_SET_DATA_EVENTFD=4, VFIO_IRQ_SET_ACTION_TRIGGER=32, VFIO_DEVICE_SET_IRQS=lambda *a: 0)
  real_exists, real_listdir = os.path.exists, os.listdir
  monkeypatch.setattr(system, "FileIOInterface", FakeFIO)
  monkeypatch.setattr(system, "vfio", fake_vfio)
  monkeypatch.setattr(system.System, "flock_acquire", lambda name: None)
  monkeypatch.setitem(system.System.__dict__, "vfio", FakeFIO(fd=os.open("/dev/null", os.O_RDONLY)))
  monkeypatch.setattr(system.os, "listdir", lambda p: list(bound) if p == gdir else real_listdir(p))
  def exists(p):
    p = str(p)
    return bound.get(p.split("/")[-2]) is not None if p.startswith(gdir) and p.endswith("/driver") else real_exists(p)
  monkeypatch.setattr(system.os.path, "exists", exists)
  monkeypatch.setattr(system.ctypes, "sizeof", lambda x: 4)
  import builtins
  real_open = builtins.open
  monkeypatch.setattr(builtins, "open", lambda p, *a, **k: types.SimpleNamespace(read=lambda: "0x030000") if str(p).startswith(gdir) else
                      real_open(p, *a, **k))
  monkeypatch.setattr(system.PCIDevice, "_vfio_groups", {})
  errs:list[BaseException] = []
  def go(bus):
    try: system.PCIDevice("am", bus)
    except BaseException as e: errs.append(e)
  ts = [threading.Thread(target=go, args=(b,)) for b in bound]
  for t in ts: t.start()
  for t in ts: t.join()
  assert not errs, errs
  binds = [x for x in log if "driver_override" in x[0] or "drivers_probe" in x[0] or x[0] == "SET_CONTAINER"]
  assert binds and all(locked for _, locked in binds), log
  assert attaches["n"] == 3 and set(system.PCIDevice._vfio_groups) == {"10"}
  assert all(v == "vfio-pci" for v in bound.values())
