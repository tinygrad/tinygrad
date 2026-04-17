import ctypes, socket
import pytest
import tinygrad.runtime.support.system as system

def test_list_devices_osx_uses_usb4_indices(monkeypatch):
  monkeypatch.setattr(system, "OSX", True)
  monkeypatch.setattr(system.System, "pci_scan_bus", lambda vendor, devices, base_class=None: ["10de:2b85", "10de:2b04"])

  devs = system.System.list_devices(0x10de, devices=((0xff00, (0x2b00, 0x2d00)),), base_class=0x03)
  assert devs == [(system.APLRemotePCIDevice, "usb4:0"), (system.APLRemotePCIDevice, "usb4:1")]

def test_pci_scan_bus_osx_respects_base_class(monkeypatch):
  monkeypatch.setattr(system, "OSX", True)

  services = iter([1, 2, 0])
  monkeypatch.setattr(system.iokit, "IOServiceGetMatchingServices", lambda *args: 0)
  monkeypatch.setattr(system.iokit, "IOServiceMatching", lambda name: name)
  monkeypatch.setattr(system.iokit, "IOIteratorNext", lambda iterator: next(services))

  props = {
    1: {"vendor-id": 0x10de, "device-id": 0x22e8, "class-code": 0x00040300},
    2: {"vendor-id": 0x10de, "device-id": 0x2b85, "class-code": 0x00030000},
  }
  refs: dict[int, tuple[int, str]] = {}
  next_ref = iter(range(1, 100))

  def fake_create_cf_property(svc, key, *_args):
    ref = next(next_ref)
    refs[ref] = (svc, ctypes.cast(key, ctypes.c_char_p).value.decode())
    return ctypes.c_void_p(ref)

  monkeypatch.setattr(system.iokit, "IORegistryEntryCreateCFProperty", fake_create_cf_property)
  monkeypatch.setattr(system.corefoundation, "CFStringCreateWithCString",
                      lambda _alloc, raw, _enc: ctypes.c_char_p(raw))
  monkeypatch.setattr(system.corefoundation, "CFDataGetLength", lambda ref: 4)
  monkeypatch.setattr(system.corefoundation, "CFDataGetBytes",
                      lambda ref, _rng, buf: ctypes.memmove(buf, props[refs[ctypes.cast(ref, ctypes.c_void_p).value][0]][refs[ctypes.cast(ref, ctypes.c_void_p).value][1]].to_bytes(4, "little"), 4))

  scanned = system.System.pci_scan_bus(0x10de, ((0xff00, (0x2200, 0x2b00)),), base_class=0x03)
  assert scanned == ["10de:2b85"]

def test_apl_remote_pcidevice_preserves_pcibus(monkeypatch):
  monkeypatch.setattr(system.APLRemotePCIDevice, "ensure_app", classmethod(lambda cls: None))
  monkeypatch.setattr(system, "temp", lambda name: f"/tmp/{name}")

  class FakeSocket:
    def connect(self, path): assert path == "/tmp/tinygpu.sock"
    def setsockopt(self, *args, **kwargs): pass
    def getpeername(self): return ("peer",)

  called = {}
  monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: FakeSocket())
  monkeypatch.setattr(system.RemotePCIDevice, "__init__", lambda self, devpref, pcibus, sock: called.update(devpref=devpref, pcibus=pcibus, sock=sock))

  system.APLRemotePCIDevice("NV", "usb4:3")
  assert called["devpref"] == "NV"
  assert called["pcibus"] == "usb4:3"

def test_apl_remote_pcidevice_fails_fast_on_all_ones_config(monkeypatch):
  monkeypatch.setattr(system.APLRemotePCIDevice, "ensure_app", classmethod(lambda cls: None))
  monkeypatch.setattr(system, "temp", lambda name: f"/tmp/{name}")

  class FakeSocket:
    def connect(self, path): pass
    def setsockopt(self, *args, **kwargs): pass
    def getpeername(self): return ("peer",)

  monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: FakeSocket())
  monkeypatch.setattr(system.RemotePCIDevice, "__init__", lambda self, devpref, pcibus, sock: setattr(self, "dev_id", 0))
  monkeypatch.setattr(system.APLRemotePCIDevice, "read_config", lambda self, offset, size: 0xffffffff)

  with pytest.raises(RuntimeError, match="TinyGPU returned 0xffffffff for PCI config"):
    system.APLRemotePCIDevice("NV", "usb4:0")
