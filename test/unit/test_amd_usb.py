import pytest

from tinygrad.runtime import ops_amd


def test_usb_iface_discovers_current_vid(monkeypatch):
  calls = []

  def list_devices(vid, pid):
    calls.append((vid, pid))
    return [(object(), "current")]

  monkeypatch.setattr(ops_amd.USB3, "list_devices", staticmethod(list_devices))
  monkeypatch.setattr(ops_amd, "hcq_filter_visible_devices", lambda visible, _prefix: visible)

  with pytest.raises(RuntimeError, match=r"AMD:1 does not exist \(1 device available\)"):
    ops_amd.USBIface(None, 1)

  assert calls == [(0x3801, 0x0001)]
