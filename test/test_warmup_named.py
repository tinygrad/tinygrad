import os, unittest
from unittest.mock import patch
from tinygrad.engine.jit import warmup_select
from tinygrad.helpers import getenv

class TestWarmupNamed(unittest.TestCase):
  def setUp(self):
    getenv.cache_clear()
    self.call_jit, self.capture_only, self.other = object(), object(), object()
    self.calls = {"step1": (self.call_jit, lambda: None)}
    self.par = [(self.call_jit, lambda: None), (self.capture_only, lambda: None), (self.other, lambda: None)]
    self.named = {"stash0_state": [self.capture_only], "stash1_state": [self.other]}

  def tearDown(self): getenv.cache_clear()

  def test_default_unchanged(self):
    calls, par = warmup_select(self.calls, self.par, self.named)
    self.assertEqual(calls, self.calls)
    self.assertEqual(par, self.par)

  def test_capture_only_name_selects_without_calling(self):
    with patch.dict(os.environ, {"WARMUP_JITS": "step1,stash0_state"}):
      calls, par = warmup_select(self.calls, self.par, self.named)
    self.assertEqual(calls, self.calls)
    self.assertEqual(par, self.par[:2])

  def test_named_list_can_hold_multiple_jits(self):
    named = {"stash0": [self.capture_only, self.other]}
    with patch.dict(os.environ, {"WARMUP_JITS": "stash0"}):
      calls, par = warmup_select(self.calls, self.par, named)
    self.assertEqual(calls, {})
    self.assertEqual(par, self.par[1:])

  def test_unselected_named_allocates_no_warm_call_or_capture(self):
    with patch.dict(os.environ, {"WARMUP_JITS": "none"}):
      calls, par = warmup_select(self.calls, self.par, self.named)
    self.assertEqual(calls, {})
    self.assertEqual(par, [])

if __name__ == "__main__": unittest.main()
