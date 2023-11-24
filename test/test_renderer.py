import unittest
from tinygrad.helpers import dtypes

from tinygrad.renderer.hip import HIPLanguage

class TestRenderer(unittest.TestCase):
  def test_render_cast(self):
    self.assertEqual(HIPLanguage().render_cast(["data0"], dtypes.half), "(half)(data0)")
    self.assertEqual(HIPLanguage().render_cast(["data0", "data1", "data2", "data3"], dtypes.float.vec(4)), "make_float4(data0,data1,data2,data3)")
    self.assertEqual(HIPLanguage().render_cast(["data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7"], dtypes.float.vec(8)), "{data0,data1,data2,data3,data4,data5,data6,data7}")
    self.assertEqual(HIPLanguage().render_cast(["data0", "data1", "data2", "data3"], dtypes.half.vec(4)), "{(half)data0,(half)data1,(half)data2,(half)data3}")
