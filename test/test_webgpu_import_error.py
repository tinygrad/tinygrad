import importlib, sys, types, unittest
from contextlib import contextmanager

import tinygrad.runtime.autogen as autogen


class _Dummy:
  def __init__(self, *args, **kwargs): pass


class _FakeWebGPU(types.ModuleType):
  def __init__(self):
    super().__init__("tinygrad.runtime.autogen.webgpu")
    self.missing = set()
    self.enum_WGPUBackendType = {0: "WGPUBackendType_Undefined"}
    self.instance = _Dummy()

  def __getattribute__(self, nm):
    if nm not in {"__dict__", "__class__", "missing"} and nm in self.__dict__.get("missing", set()): raise AttributeError(nm)
    return super().__getattribute__(nm)

  def wgpuCreateInstance(self, descriptor): return self.instance

  def __getattr__(self, nm):
    if nm in self.missing: raise AttributeError(nm)
    if nm.startswith("enum_"): return {}
    if nm.startswith("WGPU"): return _Dummy
    if nm.startswith("wgpu"):
      def fn(*args, **kwargs): return _Dummy()
      fn.__name__ = nm
      fn.argtypes = ()
      return fn
    raise AttributeError(nm)


@contextmanager
def _patched_webgpu(webgpu):
  old_ops_webgpu = sys.modules.pop("tinygrad.runtime.ops_webgpu", None)
  old_webgpu = sys.modules.get("tinygrad.runtime.autogen.webgpu")
  old_autogen_webgpu = autogen.__dict__.get("webgpu")
  had_autogen_webgpu = "webgpu" in autogen.__dict__
  sys.modules["tinygrad.runtime.autogen.webgpu"] = autogen.webgpu = webgpu
  try:
    yield
  finally:
    sys.modules.pop("tinygrad.runtime.ops_webgpu", None)
    if old_ops_webgpu is not None: sys.modules["tinygrad.runtime.ops_webgpu"] = old_ops_webgpu
    if old_webgpu is not None: sys.modules["tinygrad.runtime.autogen.webgpu"] = old_webgpu
    else: sys.modules.pop("tinygrad.runtime.autogen.webgpu", None)
    if had_autogen_webgpu: autogen.webgpu = old_autogen_webgpu
    else: autogen.__dict__.pop("webgpu", None)


class TestWebGPUImportError(unittest.TestCase):
  def test_missing_dawn_symbols_raise_install_message(self):
    for missing in ("enum_WGPUBackendType", "wgpuCreateInstance"):
      with self.subTest(missing=missing):
        webgpu = _FakeWebGPU()
        webgpu.missing.add(missing)
        with _patched_webgpu(webgpu):
          with self.assertRaisesRegex(RuntimeError, "(?i)dawn.*install"):
            importlib.import_module("tinygrad.runtime.ops_webgpu")

  def test_import_succeeds_when_dawn_symbols_present(self):
    webgpu = _FakeWebGPU()
    with _patched_webgpu(webgpu):
      ops_webgpu = importlib.import_module("tinygrad.runtime.ops_webgpu")
    self.assertIs(ops_webgpu.instance, webgpu.instance)
    self.assertEqual(ops_webgpu.backend_types, {"WGPUBackendType_Undefined": 0})


if __name__ == "__main__":
  unittest.main()
