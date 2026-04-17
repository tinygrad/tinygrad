import socket
import tinygrad.runtime.support.system as system

def test_list_devices_osx_uses_usb4_indices(monkeypatch):
  monkeypatch.setattr(system, "OSX", True)
  monkeypatch.setattr(system.System, "pci_scan_bus", lambda vendor, devices, base_class=None: ["10de:2b85", "10de:2b04"])

  devs = system.System.list_devices(0x10de, devices=((0xff00, (0x2b00, 0x2d00)),), base_class=0x03)
  assert devs == [(system.APLRemotePCIDevice, "usb4:0"), (system.APLRemotePCIDevice, "usb4:1")]

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
