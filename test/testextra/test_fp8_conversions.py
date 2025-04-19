import unittest
import math
import numpy as np
from extra.fp8_conversions import float_to_fp8, fp8_to_float

class TestFp8sConversions(unittest.TestCase):
    def test_float_to_fp8e4m3(self):
        test_values = [(0.0, 0),(0.1, 29),(0.2, 37),(0.3, 42),(0.4, 45),(0.5, 48),(1, 56),(2, 64),(5, 74),(math.nan, math.nan)
                        (5.1, 74),(5.2, 74),(5.3, 75),(448, 126),(100000, 126),(-1, 184),(-2, 192),(-3, 196),(-100000, 254),]
        for value, expected_fp8 in test_values:
            fp8 = float_to_fp8(value, "E4M3")
            np.testing.assert_equal(fp8, expected_fp8)

    def test_float_to_fp8e5m2(self):
        test_values = [(0.0, 0),(0.1, 46),(0.2, 50),(0.3, 53),(0.4, 54),(0.5, 56),(1, 60),(2, 64),(5, 69),(math.nan, math.nan)
                         (5.1, 69),(5.2, 69),(5.3, 69),(57344, 123),(500000, 123),(-1, 188),(-2, 192),(-3, 194),(-100000, 251),]
        for value, expected_fp8 in test_values:
            fp8 = float_to_fp8(value, "E5M2")
            np.testing.assert_equal(fp8, expected_fp8)

    def test_fp8e4m3_to_float(self):
        test_values = [(0, 0.0),(29, 0.1015625),(37, 0.203125),(42, 0.3125),(45, 0.40625),(48, 0.5),(56, 1.0),(64, 2.0),(74, 5.0),
                        (75, 5.5),(126, 448.0),(254, -448.0),(math.nan, math.nan)]
        for fp8_value, expected_float in test_values:
            float_value = fp8_to_float(fp8_value, "E4M3")
            np.testing.assert_equal(float_value, expected_float)

    def test_fp8e5m2_to_float(self):
        test_values = [(0, 0.0),(46, 0.09375),(50, 0.1875),(53, 0.3125),(54, 0.375),(56, 0.5),(60, 1.0),(64, 2.0),(69, 5.0),
                        (123, 57344.0),(188, -1.0),(251, -57344.0),(math.nan, math.nan)]
        for fp8_value, expected_float in test_values:
            float_value = fp8_to_float(fp8_value, "E5M2")
            np.testing.assert_equal(float_value, expected_float)