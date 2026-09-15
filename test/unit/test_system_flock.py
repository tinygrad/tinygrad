import os, sys
import pytest

@pytest.mark.skipif(sys.platform != "linux", reason="uses flock and /proc/self/fd")
def test_flock_acquire_closes_its_fd_when_contended(monkeypatch, tmp_path):
  import fcntl
  import tinygrad.runtime.support.system as system

  monkeypatch.setattr(system, "temp", lambda name: str(tmp_path / name))
  def contended(*args): raise OSError("would block")
  monkeypatch.setattr(fcntl, "flock", contended)

  before = set(os.listdir("/proc/self/fd"))
  with pytest.raises(RuntimeError): system.System.flock_acquire("tinygrad_test_flock.lock")
  assert set(os.listdir("/proc/self/fd")) <= before, "lock file descriptor leaked on a contended flock"
