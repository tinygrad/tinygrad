import os

from tinygrad.tensor import _pool_impl_mode


def test_pool_impl_mode_defaults_to_main():
  old = os.environ.pop("POOL_IMPL", None)
  try:
    assert _pool_impl_mode() == "MAIN"
  finally:
    if old is not None: os.environ["POOL_IMPL"] = old


def test_pool_impl_mode_reads_override():
  old = os.environ.get("POOL_IMPL")
  try:
    os.environ["POOL_IMPL"] = "ALT"
    assert _pool_impl_mode() == "ALT"
  finally:
    if old is None: os.environ.pop("POOL_IMPL", None)
    else: os.environ["POOL_IMPL"] = old
