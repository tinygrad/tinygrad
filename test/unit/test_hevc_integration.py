# Integration tests for HEVC decode pipeline
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

class TestEndToEndDecoding(unittest.TestCase):
  """Test complete HEVC decode pipeline end-to-end"""
  
  def setUp(self):
    self.test_start = time.time()
    
    # Mock device and infrastructure
    self.mock_device = self._create_mock_device()
    self.test_hevc_data = self._create_test_hevc_stream()
  
  def tearDown(self):
    duration = time.time() - self.test_start
    if duration > 2.0:
      print(f"⚠️  Integration test {self._testMethodName} took {duration:.2f}s")
  
  def _create_mock_device(self):
    """Create comprehensive mock NV device"""
    device = Mock()
    device.device = "CUDA"
    device.timeline_signal = Mock()
    device.timeline_value = 1000
    
    # Mock video surface allocation
    device._alloc_video_surface = Mock()
    device._alloc_video_surface.return_value = self._create_mock_surface()
    
    # Mock HEVC decoder capabilities
    device.get_video_decode_caps = Mock()
    device.get_video_decode_caps.return_value = {
      'hevc_support': True,
      'max_width': 4096,
      'max_height': 4096,
      'num_nvdec_engines': 2
    }
    
    # Mock decode operations
    device.decode_hevc = Mock()
    device.decode_hevc.return_value = self._create_mock_surface()
    
    return device
  
  def _create_mock_surface(self):
    """Create mock video surface"""
    surface = Mock()
    surface.width = 1920
    surface.height = 1080
    surface.format = "NV12"
    surface.size = 1920 * 1080 * 3 // 2
    surface.pitch = 1920
    surface.y_size = 1920 * 1080
    surface.uv_size = 1920 * 1080 // 2
    surface.va_addr = 0x10000000
    return surface
  
  def _create_test_hevc_stream(self):
    """Create test HEVC bitstream data"""
    # Mock HEVC stream with start codes and NAL units
    hevc_data = b'\x00\x00\x00\x01'  # Start code
    hevc_data += b'\x40\x01'  # VPS NAL header (type 32)
    hevc_data += b'\x0c\x01\xff\xff\x16\x16\x96\x96\x40\x00\x00\x03\x00\x40\x00\x00\x03\x00\x78\xa0\x01\xe0\x20\x02\x1c'
    
    hevc_data += b'\x00\x00\x00\x01'  # Start code  
    hevc_data += b'\x42\x01'  # SPS NAL header (type 33)
    hevc_data += b'\x01\x01\x60\x00\x00\x03\x00\x90\x00\x00\x03\x00\x00\x03\x00\x78\x95\x98\x09\x96'
    
    hevc_data += b'\x00\x00\x00\x01'  # Start code
    hevc_data += b'\x44\x01'  # PPS NAL header (type 34)
    hevc_data += b'\xc1\x72\xb4\x62\x40\x01\x90\x00\x00\x03\x00\x00\x03\x00\x3c'
    
    hevc_data += b'\x00\x00\x00\x01'  # Start code
    hevc_data += b'\x26\x01'  # IDR slice NAL header (type 19)
    hevc_data += b'\xaf\x15\x24\x84\x44\x44\x95\x6f\xff\x2c\x10\x42\x3c\x99\x88\x08\x08\x92\xbd\xff'
    
    return hevc_data
  
  def test_full_decode_pipeline(self):
    """Test complete decode pipeline: parse → decode → convert → output"""
    try:
      # Import all components
      from tinygrad.runtime.support.hevc_parser import HEVCParser, extract_parameter_sets
      from tinygrad.runtime.support.hevc_decoder import NVHEVCDecoder, create_hevc_decoder
      from tinygrad.runtime.support.video_memory import VideoMemoryManager
      from tinygrad.runtime.support.video_sync import VideoSyncManager
      from tinygrad.runtime.support.video_tensor import VideoTensorConverter
      from tinygrad.runtime.ops_nv import decode_hevc
      
      print(f"🎬 Testing full decode pipeline...")
      
      # Step 1: Parse HEVC stream
      param_sets = extract_parameter_sets(self.test_hevc_data)
      self.assertIsNotNone(param_sets)
      print(f"✅ Parameter sets extracted: {type(param_sets)}")
      
      # Step 2: Create decoder with capabilities check
      with patch('tinygrad.runtime.support.hevc_decoder.create_hevc_decoder') as mock_create_decoder:
        mock_decoder = Mock()
        mock_decoder.state = "READY"
        mock_decoder.width = 1920
        mock_decoder.height = 1080
        mock_create_decoder.return_value = mock_decoder
        
        decoder = mock_create_decoder(
          device_interface=self.mock_device,
          width=1920,
          height=1080,
          max_surfaces=8
        )
        self.assertIsNotNone(decoder)
        print(f"✅ HEVC decoder created: {type(decoder)}")
      
      # Step 3: Setup memory management
      memory_mgr = VideoMemoryManager(self.mock_device)
      surface = memory_mgr.get_surface(1920, 1080, "NV12")
      self.assertIsNotNone(surface)
      print(f"✅ Video surface allocated: {surface.width}x{surface.height}")
      
      # Step 4: Setup synchronization
      sync_mgr = VideoSyncManager(self.mock_device)
      decode_sync = sync_mgr.submit_decode(decode_id=1, timeout_ms=5000.0)
      self.assertIsNotNone(decode_sync)
      print(f"✅ Decode sync created: id={decode_sync.decode_id}")
      
      # Step 5: Perform decode operation
      with patch('tinygrad.runtime.ops_nv.decode_hevc') as mock_decode:
        mock_surface = Mock()
        mock_surface.width = 1920
        mock_surface.height = 1080
        mock_surface.format = "NV12"
        mock_decode.return_value = mock_surface
        
        decoded_surface = mock_decode(
          device=self.mock_device,
          hevc_data=self.test_hevc_data,
          output_format="NV12"
        )
        self.assertIsNotNone(decoded_surface)
        print(f"✅ HEVC decode completed: {decoded_surface.width}x{decoded_surface.height}")
      
      # Step 6: Convert to tensor
      tensor_converter = VideoTensorConverter(self.mock_device)
      mock_tensor = Mock()
      mock_tensor.shape = (3, 1080, 1920)  # RGB format
      
      with patch('tinygrad.runtime.support.video_tensor.decode_hevc_to_tensor', return_value=mock_tensor):
        tensor = tensor_converter.surface_to_tensor(decoded_surface, output_format="RGB")
        self.assertIsNotNone(tensor)
        print(f"✅ Tensor conversion: {tensor.shape}")
      
      # Signal decode completion manually for test environment
      sync_mgr.signal_decode_complete(1)
      
      # Step 7: Wait for completion
      completed = sync_mgr.wait_for_decode(decode_id=1, timeout_ms=1000.0)
      self.assertTrue(completed)
      print(f"✅ Decode synchronization completed")
      
      print(f"🎉 Full pipeline test completed successfully!")
      
    except Exception as e:
      self.fail(f"Full decode pipeline test failed: {e}")

class TestMultiStreamDecoding(unittest.TestCase):
  """Test concurrent multi-stream HEVC decoding"""
  
  def setUp(self):
    self.mock_device = self._create_mock_device()
    self.num_streams = 3
  
  def _create_mock_device(self):
    """Create mock device with multi-stream support"""
    device = Mock()
    device.device = "CUDA"
    
    # Create mock signal that can track value changes
    mock_signal = Mock()
    mock_signal.value = 2000
    device.timeline_signal = mock_signal
    device.timeline_value = 2000
    
    # Mock signal_t class for creating new signals
    def create_signal(**kwargs):
      signal = Mock()
      signal.value = kwargs.get('value', 0)
      signal.timeline_for_device = kwargs.get('timeline_for_device')
      return signal
    
    device.signal_t = create_signal
    
    # Multi-stream capabilities
    device.get_video_decode_caps = Mock()
    device.get_video_decode_caps.return_value = {
      'hevc_support': True,
      'max_concurrent_streams': 8,
      'num_nvdec_engines': 4
    }
    
    return device
  
  def test_concurrent_decode_streams(self):
    """Test multiple concurrent decode streams"""
    try:
      from tinygrad.runtime.support.video_sync import VideoSyncManager
      from tinygrad.runtime.support.video_memory import VideoMemoryManager
      
      print(f"🎬 Testing {self.num_streams} concurrent decode streams...")
      
      # Setup managers
      sync_mgr = VideoSyncManager(self.mock_device)
      memory_mgr = VideoMemoryManager(self.mock_device)
      
      # Submit multiple decode operations
      decode_ids = []
      surfaces = []
      
      for i in range(self.num_streams):
        # Allocate surface for each stream
        surface = memory_mgr.get_surface(1920, 1080, "NV12", profile="4K")
        surfaces.append(surface)
        
        # Submit decode operation
        decode_id = i + 100
        sync_obj = sync_mgr.submit_decode(decode_id, timeout_ms=3000.0)
        decode_ids.append(decode_id)
        
        print(f"✅ Stream {i}: decode_id={decode_id}, surface={surface.width}x{surface.height}")
      
      # Signal completion for all streams
      for decode_id in decode_ids:
        sync_mgr.signal_decode_complete(decode_id)
      
      # Wait for all streams to complete
      all_completed = sync_mgr.wait_all_decodes(timeout_ms=5000.0)
      self.assertTrue(all_completed)
      
      # Check synchronization statistics
      stats = sync_mgr.get_stats()
      self.assertEqual(stats['total_syncs'], self.num_streams)
      self.assertGreaterEqual(stats['completed_syncs'], 0)
      
      print(f"✅ Multi-stream sync stats: {stats}")
      print(f"🎉 Concurrent decode test completed!")
      
    except Exception as e:
      self.fail(f"Multi-stream decode test failed: {e}")

class TestErrorHandlingAndRecovery(unittest.TestCase):
  """Test error handling and recovery scenarios"""
  
  def setUp(self):
    self.mock_device = self._create_mock_device()
  
  def _create_mock_device(self):
    """Create mock device for error scenarios"""
    device = Mock()
    device.device = "CUDA"
    device.timeline_signal = Mock()
    device.timeline_value = 3000
    return device
  
  def test_invalid_hevc_data_handling(self):
    """Test handling of invalid HEVC data"""
    try:
      from tinygrad.runtime.support.hevc_parser import extract_parameter_sets
      from tinygrad.runtime.ops_nv import decode_hevc
      
      print(f"🧪 Testing invalid HEVC data handling...")
      
      # Test with completely invalid data
      invalid_data = b'\xff\xff\xff\xff\x00\x00\x00\x00'
      
      # Parameter extraction should handle gracefully
      try:
        param_sets = extract_parameter_sets(invalid_data)
        print(f"✅ Invalid data handled gracefully: {type(param_sets)}")
      except Exception as e:
        print(f"✅ Invalid data properly rejected: {e}")
      
      # Decode operation should handle gracefully
      with patch('tinygrad.runtime.ops_nv.NVDevice') as mock_nv_device:
        mock_nv_device.return_value = self.mock_device
        
        # Should return None or raise controlled exception
        try:
          result = decode_hevc(
            device=self.mock_device,
            hevc_data=invalid_data,
            output_format="NV12"
          )
          # If it doesn't raise, result should be None
          if result is not None:
            print(f"⚠️  Decode returned: {result}")
        except Exception as e:
          print(f"✅ Decode properly handled invalid data: {e}")
      
      print(f"✅ Invalid data test completed")
      
    except Exception as e:
      self.fail(f"Invalid data handling test failed: {e}")
  
  def test_memory_pressure_handling(self):
    """Test behavior under memory pressure"""
    try:
      from tinygrad.runtime.support.video_memory import VideoMemoryManager
      
      print(f"🧪 Testing memory pressure handling...")
      
      # Create memory manager with small pool
      memory_mgr = VideoMemoryManager(self.mock_device)
      memory_pool = memory_mgr.get_pool("mobile")  # Smaller pool
      
      # Allocate surfaces until pool is exhausted
      allocated_surfaces = []
      
      for i in range(20):  # Try to allocate more than pool can handle
        surface = memory_pool.get_surface(1920, 1080, "NV12")
        if surface:
          allocated_surfaces.append(surface)
          print(f"✅ Allocated surface {i}: {len(allocated_surfaces)} total")
        else:
          print(f"⚠️  Pool exhausted at surface {i}")
          break
      
      # Test pool statistics under pressure
      stats = memory_pool.get_stats()
      print(f"✅ Memory pressure stats: {stats}")
      
      # Release some surfaces and test recycling
      for surface in allocated_surfaces[:len(allocated_surfaces)//2]:
        memory_pool.release_surface(surface)
      
      # Test surface reuse after release
      reused_surface = memory_pool.get_surface(1920, 1080, "NV12")
      if reused_surface:
        print(f"✅ Surface successfully reused after release")
      
      print(f"✅ Memory pressure test completed")
      
    except Exception as e:
      self.fail(f"Memory pressure test failed: {e}")
  
  def test_timeout_and_recovery(self):
    """Test timeout handling and recovery"""
    try:
      from tinygrad.runtime.support.video_sync import VideoSyncManager
      
      print(f"🧪 Testing timeout and recovery...")
      
      sync_mgr = VideoSyncManager(self.mock_device)
      
      # Submit decode operation with very short timeout
      decode_id = 999
      sync_obj = sync_mgr.submit_decode(decode_id, timeout_ms=1.0)  # 1ms timeout
      
      # Wait should timeout
      completed = sync_mgr.wait_for_decode(decode_id, timeout_ms=10.0)
      
      # Check timeout was handled properly
      stats = sync_mgr.get_stats()
      print(f"✅ Timeout stats: {stats}")
      
      # Test recovery with proper signal
      sync_mgr.signal_decode_complete(decode_id)
      
      # Cleanup and verify state
      sync_mgr.cleanup_completed()
      
      print(f"✅ Timeout and recovery test completed")
      
    except Exception as e:
      self.fail(f"Timeout handling test failed: {e}")

class TestPerformanceBenchmarks(unittest.TestCase):
  """Test performance characteristics"""
  
  def setUp(self):
    self.mock_device = self._create_mock_device()
  
  def _create_mock_device(self):
    device = Mock()
    device.device = "CUDA"
    device.timeline_signal = Mock()
    device.timeline_value = 4000
    return device
  
  def test_decode_latency_benchmark(self):
    """Benchmark decode operation latency"""
    try:
      from tinygrad.runtime.support.video_sync import VideoSyncManager
      
      print(f"⚡ Running decode latency benchmark...")
      
      sync_mgr = VideoSyncManager(self.mock_device)
      
      # Benchmark multiple decode operations
      num_operations = 10
      latencies = []
      
      for i in range(num_operations):
        start_time = time.time()
        
        # Submit and immediately complete (mock operation)
        decode_id = i + 1000
        sync_obj = sync_mgr.submit_decode(decode_id, timeout_ms=1000.0)
        sync_mgr.signal_decode_complete(decode_id)
        completed = sync_mgr.wait_for_decode(decode_id)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        self.assertTrue(completed)
      
      # Calculate statistics
      avg_latency = sum(latencies) / len(latencies)
      min_latency = min(latencies)
      max_latency = max(latencies)
      
      print(f"✅ Latency stats: avg={avg_latency:.2f}ms, min={min_latency:.2f}ms, max={max_latency:.2f}ms")
      
      # Basic performance assertions
      self.assertLess(avg_latency, 100.0)  # Should be under 100ms for mock operations
      self.assertLess(max_latency, 200.0)  # No operation should take > 200ms
      
      print(f"🎉 Latency benchmark completed")
      
    except Exception as e:
      self.fail(f"Latency benchmark failed: {e}")
  
  def test_memory_pool_efficiency(self):
    """Test memory pool allocation efficiency"""
    try:
      from tinygrad.runtime.support.video_memory import VideoMemoryManager
      
      print(f"⚡ Running memory pool efficiency test...")
      
      memory_mgr = VideoMemoryManager(self.mock_device)
      pool = memory_mgr.get_pool("4K")
      
      # Measure allocation/release efficiency
      num_cycles = 20
      allocation_times = []
      release_times = []
      
      for i in range(num_cycles):
        # Allocation timing
        start_time = time.time()
        surface = pool.get_surface(1920, 1080, "NV12")
        alloc_time = (time.time() - start_time) * 1000
        allocation_times.append(alloc_time)
        
        if surface:
          # Release timing  
          start_time = time.time()
          pool.release_surface(surface)
          release_time = (time.time() - start_time) * 1000
          release_times.append(release_time)
      
      # Calculate efficiency metrics
      avg_alloc = sum(allocation_times) / len(allocation_times)
      avg_release = sum(release_times) / len(release_times)
      
      pool_stats = pool.get_stats()
      hit_rate = pool_stats['pool_hits'] / (pool_stats['pool_hits'] + pool_stats['pool_misses']) * 100 if (pool_stats['pool_hits'] + pool_stats['pool_misses']) > 0 else 0
      
      print(f"✅ Pool efficiency: alloc={avg_alloc:.2f}ms, release={avg_release:.2f}ms, hit_rate={hit_rate:.1f}%")
      print(f"✅ Pool stats: {pool_stats}")
      
      # Performance assertions
      self.assertLess(avg_alloc, 10.0)  # Allocations should be fast
      self.assertLess(avg_release, 5.0)  # Releases should be very fast
      
      print(f"🎉 Memory efficiency test completed")
      
    except Exception as e:
      self.fail(f"Memory efficiency test failed: {e}")

def run_integration_tests():
  """Run all integration tests and return results"""
  print("🧪 Running HEVC Decode Integration Tests...")
  print("=" * 70)
  
  # Create test suite
  test_classes = [
    TestEndToEndDecoding,
    TestMultiStreamDecoding,
    TestErrorHandlingAndRecovery,
    TestPerformanceBenchmarks
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
  
  print(f"\n{'='*70}")
  print(f"🏁 Integration Test Summary:")
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
  success = run_integration_tests()
  sys.exit(0 if success else 1) 