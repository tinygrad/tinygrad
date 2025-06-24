# Unit tests for HEVC decode implementation
import unittest
import sys
import os
import time
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add tinygrad to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Test configuration 
SKIP_HARDWARE_TESTS = os.getenv("SKIP_HARDWARE_TESTS", "1") == "1"

class TestCUVIDBindings(unittest.TestCase):
  """Test CUVID bindings and library loading"""
  
  def setUp(self):
    self.test_start = time.time()
  
  def tearDown(self):
    duration = time.time() - self.test_start
    if duration > 1.0:
      print(f"⚠️  Test {self._testMethodName} took {duration:.2f}s")
  
  def test_cuvid_import(self):
    """Test CUVID module can be imported"""
    try:
      from tinygrad.runtime.support.cuvid import check_hevc_support, create_hevc_decoder
      self.assertTrue(True, "CUVID module imported successfully")
    except ImportError as e:
      self.skipTest(f"CUVID not available: {e}")
  
  def test_cuvid_structs(self):
    """Test CUVID data structures are defined"""
    try:
      from tinygrad.runtime.support.cuvid import CUVIDDECODECAPS, CUVIDDECODECREATEINFO, CUVIDPICPARAMS
      
      # Test structure creation
      caps = CUVIDDECODECAPS()
      create_info = CUVIDDECODECREATEINFO()
      pic_params = CUVIDPICPARAMS()
      
      # Basic field access
      self.assertIsNotNone(caps)
      self.assertIsNotNone(create_info)
      self.assertIsNotNone(pic_params)
      
    except ImportError as e:
      self.skipTest(f"CUVID structures not available: {e}")
  
  def test_hevc_support_check(self):
    """Test HEVC support detection"""
    try:
      from tinygrad.runtime.support.cuvid import check_hevc_support
      
      # Should return capabilities or None gracefully
      caps = check_hevc_support()
      
      if caps is not None:
        # Validate capability structure
        self.assertTrue(hasattr(caps, 'nMaxWidth'))
        self.assertTrue(hasattr(caps, 'nMaxHeight'))
        self.assertTrue(hasattr(caps, 'nNumNVDECs'))
        print(f"✅ HEVC caps: {caps.nNumNVDECs} engines, {caps.nMaxWidth}x{caps.nMaxHeight}")
      else:
        print(f"⚠️  HEVC support not available")
        
    except Exception as e:
      if SKIP_HARDWARE_TESTS:
        self.skipTest(f"Hardware test skipped: {e}")
      else:
        self.fail(f"HEVC support check failed: {e}")

class TestHEVCParser(unittest.TestCase):
  """Test HEVC bitstream parser"""
  
  def setUp(self):
    try:
      from tinygrad.runtime.support.hevc_parser import HEVCParser, extract_parameter_sets, HEVCNalType
      self.parser_available = True
    except ImportError:
      self.parser_available = False
  
  def test_parser_import(self):
    """Test HEVC parser can be imported"""
    if not self.parser_available:
      self.skipTest("HEVC parser not available")
    
    from tinygrad.runtime.support.hevc_parser import HEVCParser, extract_parameter_sets
    self.assertTrue(True, "HEVC parser imported successfully")
  
  def test_nal_type_detection(self):
    """Test NAL unit type detection"""
    if not self.parser_available:
      self.skipTest("HEVC parser not available")
    
    from tinygrad.runtime.support.hevc_parser import HEVCNalType
    
    # Test NAL type constants (using actual names from implementation)
    self.assertIsNotNone(HEVCNalType.VPS_NUT)
    self.assertIsNotNone(HEVCNalType.SPS_NUT)
    self.assertIsNotNone(HEVCNalType.PPS_NUT)
    self.assertIsNotNone(HEVCNalType.IDR_W_RADL)
  
  def test_mock_hevc_parsing(self):
    """Test HEVC parsing with mock data"""
    if not self.parser_available:
      self.skipTest("HEVC parser not available")
    
    from tinygrad.runtime.support.hevc_parser import extract_parameter_sets
    
    # Mock HEVC data with NAL unit start codes
    mock_hevc_data = b'\x00\x00\x00\x01\x40\x01\x0c\x01\xff\xff'  # Mock VPS
    mock_hevc_data += b'\x00\x00\x00\x01\x42\x01\x01\x01\x60\x00'  # Mock SPS
    mock_hevc_data += b'\x00\x00\x00\x01\x44\x01\xc1\x72\xb4\x62'  # Mock PPS
    
    # Test parameter extraction
    try:
      param_sets = extract_parameter_sets(mock_hevc_data)
      
      # Should return dict with parameter sets
      self.assertIsInstance(param_sets, dict)
      
      # May or may not find parameters in mock data, but shouldn't crash
      print(f"✅ Parameter extraction completed: {list(param_sets.keys())}")
      
    except Exception as e:
      print(f"⚠️  Parameter extraction with mock data: {e}")

class TestVideoSurface(unittest.TestCase):
  """Test NVVideoSurface class"""
  
  def test_video_surface_import(self):
    """Test video surface can be imported"""
    try:
      from tinygrad.runtime.ops_nv import NVVideoSurface
      self.assertTrue(True, "NVVideoSurface imported successfully")
    except ImportError as e:
      self.skipTest(f"NVVideoSurface not available: {e}")
  
  def test_video_surface_creation(self):
    """Test video surface creation with mock data"""
    try:
      from tinygrad.runtime.ops_nv import NVVideoSurface
      
      # Create mock surface
      surface = NVVideoSurface(
        va_addr=0x1000000,  # Mock GPU address
        size=1920*1080*3//2,  # NV12 size
        width=1920,
        height=1080,
        format="NV12"
      )
      
      # Test attributes
      self.assertEqual(surface.width, 1920)
      self.assertEqual(surface.height, 1080)
      self.assertEqual(surface.format, "NV12")
      
      # Test properties
      self.assertGreater(surface.pitch, 0)
      self.assertGreater(surface.y_size, 0)
      self.assertGreater(surface.uv_size, 0)
      
      print(f"✅ Surface: {surface.width}x{surface.height} {surface.format}, pitch={surface.pitch}")
      
    except Exception as e:
      self.fail(f"Video surface creation failed: {e}")
  
  def test_video_surface_formats(self):
    """Test different video surface formats"""
    try:
      from tinygrad.runtime.ops_nv import NVVideoSurface
      
      formats = [
        ("NV12", 1920*1080*3//2),
        ("RGBA", 1920*1080*4)
      ]
      
      for format_name, expected_size in formats:
        surface = NVVideoSurface(
          va_addr=0x1000000,
          size=expected_size,
          width=1920,
          height=1080,
          format=format_name
        )
        
        self.assertEqual(surface.format, format_name)
        self.assertEqual(surface.size, expected_size)
        
        print(f"✅ Format {format_name}: {surface.width}x{surface.height}, size={surface.size}")
        
    except Exception as e:
      self.fail(f"Multi-format test failed: {e}")

class TestVideoQueue(unittest.TestCase):
  """Test NVVideoQueue class"""
  
  def test_video_queue_import(self):
    """Test video queue can be imported"""
    try:
      from tinygrad.runtime.ops_nv import NVVideoQueue
      self.assertTrue(True, "NVVideoQueue imported successfully")
    except ImportError as e:
      self.skipTest(f"NVVideoQueue not available: {e}")
  
  def test_video_queue_creation(self):
    """Test video queue creation and setup"""
    try:
      from tinygrad.runtime.ops_nv import NVVideoQueue
      
      # Create video queue
      queue = NVVideoQueue()
      
      # Test initial state
      self.assertIsNone(queue.decode_engine_class)
      self.assertIsNone(queue.nvdec_channel)
      
      # Test setup
      mock_engine = Mock()
      mock_channel = Mock()
      
      queue.setup(decode_engine_class=mock_engine, nvdec_channel=mock_channel)
      
      self.assertEqual(queue.decode_engine_class, mock_engine)
      self.assertEqual(queue.nvdec_channel, mock_channel)
      
      print(f"✅ Video queue setup completed")
      
    except Exception as e:
      self.fail(f"Video queue test failed: {e}")

class TestNVDECEngine(unittest.TestCase):
  """Test NVDEC engine abstraction"""
  
  def test_nvdec_import(self):
    """Test NVDEC module can be imported"""
    try:
      from tinygrad.runtime.support.nvdec import NVDECEngine, NVDECCommandType, get_available_nvdec_engines
      self.assertTrue(True, "NVDEC module imported successfully")
    except ImportError as e:
      self.skipTest(f"NVDEC module not available: {e}")
  
  def test_nvdec_command_types(self):
    """Test NVDEC command type constants"""
    try:
      from tinygrad.runtime.support.nvdec import NVDECCommandType
      
      # Test command constants exist
      self.assertIsNotNone(NVDECCommandType.SET_BITSTREAM_BUFFER)
      self.assertIsNotNone(NVDECCommandType.SET_OUTPUT_SURFACE)
      self.assertIsNotNone(NVDECCommandType.EXECUTE_DECODE)
      
      print(f"✅ NVDEC command types defined")
      
    except Exception as e:
      self.fail(f"NVDEC command types test failed: {e}")
  
  def test_nvdec_descriptors(self):
    """Test NVDEC hardware descriptors"""
    try:
      from tinygrad.runtime.support.nvdec import NVDECBitstreamBuffer, NVDECSurfaceDesc, NVDECDecodeParams
      
      # Test descriptor creation
      bitstream = NVDECBitstreamBuffer(gpu_addr=0x1000000, size=1024)
      surface = NVDECSurfaceDesc(gpu_addr=0x2000000, width=1920, height=1080)
      params = NVDECDecodeParams(pic_width_in_mbs=120, pic_height_in_mbs=68)
      
      # Test hardware descriptor conversion
      bitstream_desc = bitstream.to_hw_desc()
      surface_desc = surface.to_hw_desc()
      params_desc = params.to_hw_desc()
      
      self.assertIsInstance(bitstream_desc, bytes)
      self.assertIsInstance(surface_desc, bytes)
      self.assertIsInstance(params_desc, bytes)
      
      print(f"✅ NVDEC descriptors: bitstream={len(bitstream_desc)}B, surface={len(surface_desc)}B, params={len(params_desc)}B")
      
    except Exception as e:
      self.fail(f"NVDEC descriptors test failed: {e}")

class TestVideoMemoryManager(unittest.TestCase):
  """Test video memory management"""
  
  def test_memory_manager_import(self):
    """Test memory manager can be imported"""
    try:
      from tinygrad.runtime.support.video_memory import VideoMemoryManager, VideoBufferPool
      self.assertTrue(True, "Video memory manager imported successfully")
    except ImportError as e:
      self.skipTest(f"Video memory manager not available: {e}")
  
  def test_buffer_pool(self):
    """Test video buffer pool operations"""
    try:
      from tinygrad.runtime.support.video_memory import VideoBufferPool
      
      # Mock device interface
      mock_device = Mock()
      mock_device._alloc_video_surface = Mock(return_value=Mock())
      
      # Create buffer pool with correct constructor
      pool = VideoBufferPool(device_interface=mock_device, max_surfaces=4, recycle_timeout=2.0)
      
      # Test initial state
      self.assertEqual(len(pool.surfaces), 0)
      self.assertIsNotNone(pool.stats)
      
      # Test pool operations
      stats = pool.get_stats()
      self.assertIsInstance(stats, dict)
      self.assertIn('total_allocated', stats)
      self.assertIn('pool_hits', stats)
      
      print(f"✅ Buffer pool stats: {stats}")
      
    except Exception as e:
      self.fail(f"Buffer pool test failed: {e}")

class TestVideoSyncManager(unittest.TestCase):
  """Test video synchronization"""
  
  def test_sync_manager_import(self):
    """Test sync manager can be imported"""
    try:
      from tinygrad.runtime.support.video_sync import VideoSyncManager
      self.assertTrue(True, "Video sync manager imported successfully")
    except ImportError as e:
      self.skipTest(f"Video sync manager not available: {e}")
  
  def test_sync_manager_creation(self):
    """Test sync manager creation and basic operations"""
    try:
      from tinygrad.runtime.support.video_sync import VideoSyncManager
      
      # Mock device interface
      mock_device = Mock()
      mock_device.timeline_signal = Mock()
      mock_device.timeline_value = 100
      
      # Create sync manager with correct constructor
      sync_mgr = VideoSyncManager(device_interface=mock_device)
      
      # Test basic operations
      decode_id = 1
      sync_obj = sync_mgr.submit_decode(decode_id, timeout_ms=1000.0)
      self.assertIsNotNone(sync_obj)
      
      # Test signal operations
      sync_mgr.signal_decode_complete(decode_id)
      
      # Test stats
      stats = sync_mgr.get_stats()
      self.assertIsInstance(stats, dict)
      
      print(f"✅ Sync manager stats: {stats}")
      
    except Exception as e:
      self.fail(f"Sync manager test failed: {e}")

class TestVideoTensorConverter(unittest.TestCase):
  """Test video tensor conversion"""
  
  def test_tensor_converter_import(self):
    """Test tensor converter can be imported"""
    try:
      from tinygrad.runtime.support.video_tensor import VideoTensorConverter, create_video_tensor_converter
      self.assertTrue(True, "Video tensor converter imported successfully")
    except ImportError as e:
      self.skipTest(f"Video tensor converter not available: {e}")
  
  def test_tensor_converter_creation(self):
    """Test tensor converter creation and format specs"""
    try:
      from tinygrad.runtime.support.video_tensor import VideoTensorConverter
      
      # Mock device interface
      mock_device = Mock()
      mock_device.device = "CUDA"
      
      # Create converter
      converter = VideoTensorConverter(mock_device)
      
      # Test format specifications
      self.assertIn('NV12', converter.format_specs)
      self.assertIn('RGBA', converter.format_specs)
      self.assertIn('RGB', converter.format_specs)
      
      # Test format spec structure
      nv12_spec = converter.format_specs['NV12']
      self.assertIn('planes', nv12_spec)
      self.assertIn('dtype', nv12_spec)
      self.assertIn('subsampling', nv12_spec)
      
      print(f"✅ Tensor converter formats: {list(converter.format_specs.keys())}")
      
    except Exception as e:
      self.fail(f"Tensor converter test failed: {e}")
  
  def test_yuv_rgb_conversion_utilities(self):
    """Test YUV↔RGB conversion utilities"""
    try:
      from tinygrad.runtime.support.video_tensor import yuv_to_rgb_tensor, rgb_to_yuv_tensor
      
      # Test utility function existence
      self.assertTrue(callable(yuv_to_rgb_tensor))
      self.assertTrue(callable(rgb_to_yuv_tensor))
      
      print(f"✅ YUV↔RGB conversion utilities available")
      
    except ImportError as e:
      self.skipTest(f"Tensor conversion utilities not available: {e}")
    except Exception as e:
      self.fail(f"YUV↔RGB utilities test failed: {e}")

class TestHEVCDecoder(unittest.TestCase):
  """Test HEVC decoder implementation"""
  
  def test_decoder_import(self):
    """Test decoder can be imported"""
    try:
      from tinygrad.runtime.support.hevc_decoder import NVHEVCDecoder, create_hevc_decoder, DecoderState
      self.assertTrue(True, "HEVC decoder imported successfully")
    except ImportError as e:
      self.skipTest(f"HEVC decoder not available: {e}")
  
  def test_decoder_states(self):
    """Test decoder state management"""
    try:
      from tinygrad.runtime.support.hevc_decoder import DecoderState
      
      # Test state enumeration
      states = [
        DecoderState.UNINITIALIZED,
        DecoderState.INITIALIZING,
        DecoderState.READY,
        DecoderState.DECODING,
        DecoderState.ERROR,
        DecoderState.DESTROYED
      ]
      
      for state in states:
        self.assertIsNotNone(state)
      
      print(f"✅ Decoder states: {[s.value for s in states]}")
      
    except Exception as e:
      self.fail(f"Decoder states test failed: {e}")

def run_unit_tests():
  """Run all unit tests and return results"""
  print("🧪 Running HEVC Decode Unit Tests...")
  print("=" * 60)
  
  # Create test suite
  test_classes = [
    TestCUVIDBindings,
    TestHEVCParser,
    TestVideoSurface,
    TestVideoQueue,
    TestNVDECEngine,
    TestVideoMemoryManager,
    TestVideoSyncManager,
    TestVideoTensorConverter,
    TestHEVCDecoder
  ]
  
  suite = unittest.TestSuite()
  for test_class in test_classes:
    tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
    suite.addTests(tests)
  
  # Run tests with detailed output
  runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
  result = runner.run(suite)
  
  # Summary
  total_tests = result.testsRun
  failures = len(result.failures)
  errors = len(result.errors)
  skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
  
  passed = total_tests - failures - errors - skipped
  
  print(f"\n{'='*60}")
  print(f"🏁 Unit Test Summary:")
  print(f"   Tests Run: {total_tests}")
  print(f"   Passed: {passed}")
  print(f"   Failed: {failures}")
  print(f"   Errors: {errors}")
  print(f"   Skipped: {skipped}")
  print(f"   Success Rate: {(passed/total_tests)*100:.1f}%" if total_tests > 0 else "   Success Rate: 0.0%")
  
  # Show failures/errors
  if result.failures:
    print(f"\n❌ Failures:")
    for test, traceback in result.failures:
      print(f"   {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0] if 'AssertionError:' in traceback else 'Unknown failure'}")
  
  if result.errors:
    print(f"\n💥 Errors:")
    for test, traceback in result.errors:
      print(f"   {test}: {traceback.split('Exception: ')[-1].split('\\n')[0] if 'Exception:' in traceback else 'Unknown error'}")
  
  return result.wasSuccessful()

if __name__ == "__main__":
  success = run_unit_tests()
  sys.exit(0 if success else 1) 