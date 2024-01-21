#!/usr/bin/env python
import unittest
from tinygrad.device import Device

class TestDevice(unittest.TestCase):
  def test_canonicalize(self):
    assert Device.canonicalize(None) == Device.DEFAULT
    assert Device.canonicalize("CPU") == "CPU"
    assert Device.canonicalize("cpu") == "CPU"
    assert Device.canonicalize("GPU") == "GPU"
    assert Device.canonicalize("GPU:0") == "GPU"
    assert Device.canonicalize("gpu:0") == "GPU"
    assert Device.canonicalize("GPU:1") == "GPU:1"
    assert Device.canonicalize("gpu:1") == "GPU:1"
    assert Device.canonicalize("GPU:2") == "GPU:2"
    assert Device.canonicalize("disk:/dev/shm/test") == "DISK:/dev/shm/test"

if __name__ == "__main__":
  unittest.main()
