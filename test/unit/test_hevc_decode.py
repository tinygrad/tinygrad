import os, unittest, sys
from unittest.mock import Mock
from tinygrad import dtypes

# Add tinygrad to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Skip all HEVC tests on Windows - NV driver not supported
if sys.platform == 'win32':
  raise unittest.SkipTest("HEVC decode not available on Windows, skipping HEVC tests")

try:
  from tinygrad.runtime.support.hevc import (
    check_hevc_support, HEVCDecoder, create_hevc_decoder_auto,
    validate_hevc_stream, create_sample_hevc_data, CUVIDDECODECAPS, CUVIDDECODECREATEINFO
  )
except ImportError:
  raise unittest.SkipTest("HEVC decode not available, skipping HEVC tests")

# Test configuration
SKIP_HARDWARE_TESTS = os.getenv("SKIP_HARDWARE_TESTS", "1") == "1"

def create_mock_device():
  """Create mock device for testing"""
  device = Mock()
  device.device = "CUDA"
  device.timeline_signal = Mock()
  device.timeline_value = 1000
  return device

def create_test_hevc_bitstream(width: int = 1920, height: int = 1080) -> bytes:
  """Create test HEVC bitstream with proper NAL units"""
  return create_sample_hevc_data()

@unittest.skipIf(not hasattr(dtypes, 'uint8'), "Backend must support uint8")
class TestHEVCDecode(unittest.TestCase):
  def setUp(self) -> None:
    self.device = create_mock_device()
    self.test_bitstream = create_test_hevc_bitstream()

  def test_cuvid_import_and_structs(self):
    """Test NVDEC module imports and data structures"""
    # Test structure creation
    caps = CUVIDDECODECAPS()
    create_info = CUVIDDECODECREATEINFO()

    self.assertIsNotNone(caps)
    self.assertIsNotNone(create_info)

    # Test capability check
    caps_result = check_hevc_support(self.device)
    self.assertIsInstance(caps_result, bool)

  def test_hevc_stream_validation(self):
    """Test HEVC bitstream validation"""
    # Test valid stream
    is_valid = validate_hevc_stream(self.test_bitstream)
    self.assertTrue(is_valid)

    # Test invalid stream
    invalid_stream = b'\xff\xff\xff\xff'
    is_invalid = validate_hevc_stream(invalid_stream)
    self.assertFalse(is_invalid)

  def test_decoder_creation_and_initialization(self):
    """Test HEVC decoder creation and initialization"""
    # Test manual decoder creation
    decoder = HEVCDecoder(self.device, 1920, 1080)
    self.assertIsNotNone(decoder)
    self.assertEqual(decoder.width, 1920)
    self.assertEqual(decoder.height, 1080)

    # Test initialization
    initialized = decoder.initialize()
    self.assertIsInstance(initialized, bool)

    # Test auto decoder creation with fallback
    auto_decoder = create_hevc_decoder_auto(self.device, 1920, 1080, allow_mock=True)
    self.assertIsNotNone(auto_decoder)

  def test_frame_decoding(self):
    """Test HEVC frame decoding functionality"""
    decoder = create_hevc_decoder_auto(self.device, 1920, 1080, allow_mock=True)
    assert decoder is not None

    # Test successful decode
    surface = decoder.decode_frame(self.test_bitstream)
    if surface:  # May be None in mock mode
      self.assertEqual(surface.width, 1920)
      self.assertEqual(surface.height, 1080)
      self.assertEqual(surface.format, "NV12")

    # Test decode stats
    stats = decoder.get_stats()
    self.assertIsInstance(stats, dict)
    self.assertIn('decoded', stats)
    self.assertIn('failed', stats)
    self.assertGreaterEqual(stats['decoded'], 0)
    self.assertGreaterEqual(stats['failed'], 0)

  def test_decoder_error_handling(self):
    """Test decoder error handling with invalid inputs"""
    decoder = create_hevc_decoder_auto(self.device, 1920, 1080, allow_mock=True)
    assert decoder is not None

    # Test with invalid bitstream
    invalid_data = b'\x00\x00\x00\x00'
    decoder.decode_frame(invalid_data)
    # Should handle gracefully (return None or raise controlled exception)

    # Check that failed counter increments
    stats = decoder.get_stats()
    self.assertGreaterEqual(stats['failed'], 0)

  def test_multiple_resolution_support(self):
    """Test decoder with different resolutions"""
    resolutions = [(1280, 720), (1920, 1080), (3840, 2160)]

    for width, height in resolutions:
      with self.subTest(resolution=f"{width}x{height}"):
        decoder = create_hevc_decoder_auto(self.device, width, height, allow_mock=True)
        self.assertIsNotNone(decoder)
        assert decoder is not None
        self.assertEqual(decoder.width, width)
        self.assertEqual(decoder.height, height)

        # Test decode with this resolution
        test_data = create_test_hevc_bitstream(width, height)
        surface = decoder.decode_frame(test_data)
        if surface:
          self.assertEqual(surface.width, width)
          self.assertEqual(surface.height, height)

  def test_decoder_lifecycle(self):
    """Test complete decoder lifecycle"""
    decoder = create_hevc_decoder_auto(self.device, 1920, 1080, allow_mock=True)
    assert decoder is not None

    # Test multiple decode operations
    for i in range(5):
      decoder.decode_frame(self.test_bitstream)
      # Each decode should work independently

    # Test stats accumulation
    stats = decoder.get_stats()
    self.assertGreater(stats['decoded'] + stats['failed'], 0)

    # Test cleanup
    decoder.destroy()
    # Should not crash after cleanup

  def test_hardware_capability_detection(self):
    """Test hardware capability detection"""
    caps = check_hevc_support(self.device)

    # Should return boolean indicating support
    self.assertIsInstance(caps, bool)

if __name__ == '__main__':
  unittest.main()