#!/usr/bin/env python
import os
import unittest
from tinygrad.helpers import Target, getenv
from tinygrad.renderer import Renderer
from tinygrad.codegen.opt.postrange import should_skip_hand_coded_optimizations

class TestPostrangeMitigations(unittest.TestCase):
  def setUp(self):
    os.environ.pop("NAK_SM120_HAND_CODED_OPTS", None)
    getenv.cache_clear()

  def tearDown(self):
    os.environ.pop("NAK_SM120_HAND_CODED_OPTS", None)
    getenv.cache_clear()

  def test_skip_hand_coded_optimizations_for_nak_sm120(self):
    self.assertTrue(should_skip_hand_coded_optimizations(Renderer(Target("NV", "NAK", "sm_120"))))

  def test_override_enables_hand_coded_optimizations_for_nak_sm120(self):
    os.environ["NAK_SM120_HAND_CODED_OPTS"] = "1"
    getenv.cache_clear()
    self.assertFalse(should_skip_hand_coded_optimizations(Renderer(Target("NV", "NAK", "sm_120"))))

  def test_other_targets_are_unchanged(self):
    self.assertFalse(should_skip_hand_coded_optimizations(Renderer(Target("NV", "NAK", "sm_89"))))
    self.assertFalse(should_skip_hand_coded_optimizations(Renderer(Target("NV", "CUDA", "sm_120"))))
    self.assertFalse(should_skip_hand_coded_optimizations(Renderer(Target("AMD", "LLVM", "gfx1201"))))

if __name__ == "__main__":
  unittest.main()