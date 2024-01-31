#!/usr/bin/env python
import unittest
from unittest.mock import patch
import os
from tinygrad.device import Device
from tinygrad.helpers import getenv


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

    def test_default_device_from_env(self):
        def reset_caches():
            getenv.cache_clear()
            if "DEFAULT" in Device.__dict__:
                del Device.__dict__["DEFAULT"]

        with patch.dict(os.environ, {"CUDA": "1"}, clear=True):
            reset_caches()
            self.assertEqual(Device.canonicalize(None), "CUDA")

        # this is commonly already set on MacOS
        with patch.dict(
            os.environ, {"CLANG": "arm64-apple-darwin20.0.0-clang"}, clear=True
        ):
            reset_caches()
            self.assertRaisesRegex(
                ValueError,
                "environment variable 'CLANG' cannot be casted to <class 'int'>. Please check docs/env_vars.md for reference.",
                Device.canonicalize,
                None,
            )


if __name__ == "__main__":
    unittest.main()
