import pathlib
import tempfile
import unittest
from unittest import mock

from tinygrad.runtime.support.c import DLL


class TestCDLLFindlib(unittest.TestCase):
  def test_findlib_does_not_require_path_on_linux(self):
    with tempfile.TemporaryDirectory() as td:
      td_path = pathlib.Path(td)
      libc = td_path / "libc.so.6"
      libc.write_bytes(b"\x7fELF" + bytes([2]) + b"ok")

      orig_iterdir = pathlib.Path.iterdir
      def fake_iterdir(path_obj):
        if path_obj == td_path: return iter([libc])
        return orig_iterdir(path_obj)

      with mock.patch("tinygrad.runtime.support.c.os.name", "posix"), \
           mock.patch("tinygrad.runtime.support.c.sys.platform", "linux"), \
           mock.patch("tinygrad.runtime.support.c.sys.maxsize", 2**63 - 1), \
           mock.patch("tinygrad.runtime.support.c.sysconfig.get_config_var", return_value=None), \
           mock.patch.dict("tinygrad.runtime.support.c.os.environ", {}, clear=True), \
           mock.patch("tinygrad.runtime.support.c.pathlib.Path.is_dir", new=lambda self: self == td_path), \
           mock.patch("tinygrad.runtime.support.c.pathlib.Path.iterdir", new=fake_iterdir):
        found = DLL.findlib("libc", ["c"], extra_paths=[str(td_path)])

      self.assertEqual(found, str(libc))

  def test_findlib_skips_wrong_elf_class_and_uses_matching_candidate(self):
    with tempfile.TemporaryDirectory() as td:
      td_path = pathlib.Path(td)
      wrong = td_path / "libc.so.6"
      right = td_path / "libc.so.7"

      wrong.write_bytes(b"\x7fELF" + bytes([1]) + b"wrong-class")
      right.write_bytes(b"\x7fELF" + bytes([2]) + b"right-class")

      orig_iterdir = pathlib.Path.iterdir
      def fake_iterdir(path_obj):
        if path_obj == td_path: return iter([wrong, right])
        return orig_iterdir(path_obj)

      with mock.patch("tinygrad.runtime.support.c.os.name", "posix"), \
           mock.patch("tinygrad.runtime.support.c.sys.platform", "linux"), \
           mock.patch("tinygrad.runtime.support.c.sys.maxsize", 2**63 - 1), \
           mock.patch("tinygrad.runtime.support.c.sysconfig.get_config_var", return_value=None), \
           mock.patch.dict("tinygrad.runtime.support.c.os.environ", {"LD_LIBRARY_PATH": ""}, clear=False), \
           mock.patch("tinygrad.runtime.support.c.pathlib.Path.is_dir", new=lambda self: self == td_path), \
           mock.patch("tinygrad.runtime.support.c.pathlib.Path.iterdir", new=fake_iterdir):
        found = DLL.findlib("libc", ["c"], extra_paths=[str(td_path)])

      self.assertEqual(found, str(right))


if __name__ == "__main__":
  unittest.main()
