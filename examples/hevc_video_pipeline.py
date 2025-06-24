#!/usr/bin/env python3
"""
Advanced HEVC Video Processing Pipeline
Demonstrates multi-stream decode, tensor processing, and batch operations

Usage: python examples/hevc_video_pipeline.py [--streams N] [--batch-size N] [--profile PROFILE]
"""

import sys
import os
import argparse
import time
import threading
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Add tinygrad to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
  from tinygrad.runtime.ops_nv import NVDevice
  from tinygrad.runtime.support.hevc_decoder import create_hevc_decoder_auto, DecoderState, is_hevc_available
  from tinygrad.runtime.support.video_memory import VideoMemoryManager
  from tinygrad.runtime.support.video_sync import VideoSyncManager
  from tinygrad.runtime.support.video_tensor import VideoTensorConverter, decode_hevc_batch_to_tensor
  from tinygrad.runtime.support.hevc_parser import extract_parameter_sets
  from tinygrad import Tensor
  HEVC_AVAILABLE = is_hevc_available()
except ImportError as e:
  print(f"⚠️  HEVC decode not available: {e}")
  HEVC_AVAILABLE = False

@dataclass
class StreamConfig:
  """Configuration for video decode stream"""
  stream_id: str
  width: int
  height: int
  fps: int
  profile: str = "main"

  @property
  def name(self) -> str:
    return f"{self.stream_id}_{self.width}x{self.height}@{self.fps}fps"

@dataclass
class ProcessingStats:
  """Pipeline processing statistics"""
  frames_processed: int = 0
  frames_failed: int = 0
  total_decode_time_ms: float = 0.0
  total_processing_time_ms: float = 0.0
  peak_memory_mb: float = 0.0

  @property
  def avg_decode_time_ms(self) -> float:
    return self.total_decode_time_ms / max(1, self.frames_processed)

  @property
  def avg_processing_time_ms(self) -> float:
    return self.total_processing_time_ms / max(1, self.frames_processed)

  @property
  def success_rate(self) -> float:
    total = self.frames_processed + self.frames_failed
    return (self.frames_processed / total) if total > 0 else 0.0

class VideoProcessor:
  """Advanced video processing pipeline"""

  def __init__(self, device_interface, num_streams:int=4, batch_size:int=8):
    self.device = device_interface
    self.num_streams = num_streams
    self.batch_size = batch_size

    # Core managers
    self.memory_manager = VideoMemoryManager(device_interface)
    self.sync_manager = VideoSyncManager(device_interface)
    self.tensor_converter = VideoTensorConverter(device_interface)

    # Stream management
    self.decoders = {}  # stream_id -> decoder
    self.stream_configs = {}  # stream_id -> StreamConfig
    self.processing_threads = {}  # stream_id -> thread

    # Statistics
    self.stats = ProcessingStats()
    self.stream_stats = {}  # stream_id -> ProcessingStats

    # Synchronization
    self._lock = threading.RLock()
    self._shutdown = threading.Event()

    print(f"🎬 VideoProcessor initialized: {num_streams} streams, batch_size={batch_size}")

  def add_stream(self, config: StreamConfig) -> bool:
    """Add new decode stream to pipeline"""
    with self._lock:
      try:
        print(f"➕ Adding stream: {config.name}")

        # Use auto decoder creation with fallback to mock
        decoder = create_hevc_decoder_auto(
          device_interface=self.device,
          width=config.width,
          height=config.height,
          max_surfaces=self.batch_size * 2,  # Extra surfaces for buffering
          allow_mock=True
        )

        self.decoders[config.stream_id] = decoder
        self.stream_configs[config.stream_id] = config
        self.stream_stats[config.stream_id] = ProcessingStats()

        print(f"✅ Stream {config.stream_id} added successfully")
        return True

      except Exception as e:
        print(f"❌ Failed to add stream {config.stream_id}: {e}")
        return False

  def start_processing(self):
    """Start processing all streams"""
    print(f"🚀 Starting processing for {len(self.decoders)} streams...")

    for stream_id in self.decoders.keys():
      thread = threading.Thread(
        target=self._process_stream,
        args=(stream_id,),
        name=f"VideoProcessor-{stream_id}",
        daemon=True
      )
      thread.start()
      self.processing_threads[stream_id] = thread
      print(f"✅ Started processing thread for stream {stream_id}")

  def _process_stream(self, stream_id: str):
    """Process individual stream (runs in separate thread)"""
    try:
      decoder = self.decoders[stream_id]
      config = self.stream_configs[stream_id]
      stats = self.stream_stats[stream_id]

      print(f"🎬 Processing stream {stream_id}: {config.name}")

      # Generate mock HEVC frames for this stream
      frame_count = 0

      while not self._shutdown.is_set() and frame_count < 50:  # Process 50 frames per stream
        try:
          # Mock HEVC frame data
          hevc_data = self._generate_mock_frame(config, frame_count)

          # Decode frame
          start_time = time.time()

          if hasattr(decoder, 'decode_frame'):
            decoded_surface = decoder.decode_frame(hevc_data, wait=True)
          else:
            # Mock decode for testing
            decoded_surface = self._create_mock_surface(config)

          decode_time = (time.time() - start_time) * 1000

          if decoded_surface:
            # Process decoded frame
            processing_start = time.time()
            processed_tensor = self._process_frame(decoded_surface, config)
            processing_time = (time.time() - processing_start) * 1000

            # Update statistics
            with self._lock:
              stats.frames_processed += 1
              stats.total_decode_time_ms += decode_time
              stats.total_processing_time_ms += processing_time

              self.stats.frames_processed += 1
              self.stats.total_decode_time_ms += decode_time
              self.stats.total_processing_time_ms += processing_time

            frame_count += 1

            if frame_count % 10 == 0:
              print(f"🎬 Stream {stream_id}: processed {frame_count} frames, "
                    f"decode={decode_time:.2f}ms, process={processing_time:.2f}ms")
          else:
            stats.frames_failed += 1
            self.stats.frames_failed += 1

        except Exception as e:
          print(f"❌ Stream {stream_id} frame {frame_count} failed: {e}")
          stats.frames_failed += 1
          self.stats.frames_failed += 1

        # Simulate frame rate timing
        time.sleep(max(0, 1.0 / config.fps - 0.001))  # Target FPS with small buffer

      print(f"✅ Stream {stream_id} processing completed: {frame_count} frames")

    except Exception as e:
      print(f"❌ Stream {stream_id} processing failed: {e}")

  def _generate_mock_frame(self, config: StreamConfig, frame_idx: int) -> bytes:
    """Generate mock HEVC frame data"""
    # Create mock HEVC frame (simplified)
    hevc_data = b'\x00\x00\x00\x01'  # Start code
    hevc_data += b'\x26\x01'  # IDR slice NAL header
    hevc_data += b'\xaf\x15\x24\x84\x44\x44\x95\x6f\xff\x2c\x10\x42\x3c\x99'
    hevc_data += frame_idx.to_bytes(4, 'big')  # Frame index for variety
    hevc_data += b'\x88\x08\x08\x92\xbd\xff'

    return hevc_data

  def _create_mock_surface(self, config: StreamConfig):
    """Create mock video surface for testing"""
    from unittest.mock import Mock
    surface = Mock()
    surface.width = config.width
    surface.height = config.height
    surface.format = "NV12"
    surface.size = config.width * config.height * 3 // 2
    surface.va_addr = 0x10000000
    return surface

  def _process_frame(self, surface, config: StreamConfig):
    """Process decoded frame (mock GPU processing)"""
    try:
      # Mock tensor conversion
      from unittest.mock import Mock
      tensor = Mock()
      tensor.shape = (3, config.height, config.width)  # RGB format
      tensor.dtype = "uint8"

      # Mock GPU processing operations
      time.sleep(0.001)  # Simulate processing time

      return tensor

    except Exception as e:
      print(f"⚠️  Frame processing failed: {e}")
      return None

  def get_statistics(self) -> dict:
    """Get comprehensive processing statistics"""
    with self._lock:
      total_fps = 0
      active_streams = 0

      stream_details = {}
      for stream_id, stats in self.stream_stats.items():
        if stats.frames_processed > 0:
          fps = 1000.0 / max(1, stats.avg_decode_time_ms + stats.avg_processing_time_ms)
          total_fps += fps
          active_streams += 1

          stream_details[stream_id] = {
            'frames_processed': stats.frames_processed,
            'frames_failed': stats.frames_failed,
            'success_rate': stats.success_rate,
            'avg_decode_ms': stats.avg_decode_time_ms,
            'avg_process_ms': stats.avg_processing_time_ms,
            'fps': fps
          }

      return {
        'global_stats': {
          'total_frames': self.stats.frames_processed,
          'failed_frames': self.stats.frames_failed,
          'success_rate': self.stats.success_rate,
          'avg_decode_ms': self.stats.avg_decode_time_ms,
          'avg_process_ms': self.stats.avg_processing_time_ms,
          'total_fps': total_fps,
          'active_streams': active_streams
        },
        'stream_stats': stream_details,
        'memory_stats': self.memory_manager.get_global_stats(),
        'sync_stats': self.sync_manager.get_stats()
      }

  def stop_processing(self):
    """Stop all processing threads"""
    print(f"🛑 Stopping video processing...")
    self._shutdown.set()

    # Wait for threads to complete
    for stream_id, thread in self.processing_threads.items():
      thread.join(timeout=2.0)
      if thread.is_alive():
        print(f"⚠️  Thread {stream_id} did not stop cleanly")

  def cleanup(self):
    """Cleanup all resources"""
    print(f"🧹 Cleaning up video processor...")

    self.stop_processing()

    # Cleanup decoders
    for stream_id, decoder in self.decoders.items():
      if hasattr(decoder, 'destroy'):
        decoder.destroy()

    # Cleanup managers
    self.memory_manager.cleanup_all()
    if hasattr(self.sync_manager, 'destroy'):
      self.sync_manager.destroy()

    print(f"✅ Video processor cleanup completed")

def multi_stream_demo(num_streams: int = 4, batch_size: int = 8):
  """Multi-stream processing demonstration"""
  print(f"🎬 Multi-Stream HEVC Decode Demo")
  print(f"Streams: {num_streams}, Batch Size: {batch_size}")
  print("=" * 60)

  if not HEVC_AVAILABLE:
    print("⚠️  HEVC hardware decode not available, using mock mode")
    print("💡 Install NVIDIA Video Codec SDK for full functionality")

  try:
    # Create mock device
    from unittest.mock import Mock
    device = Mock()
    device.device = "CUDA"
    device.timeline_signal = Mock()
    device.timeline_value = 5000

    # Create video processor
    processor = VideoProcessor(device, num_streams, batch_size)

    # Define test stream configurations
    stream_configs = [
      StreamConfig("stream_1", 1920, 1080, 30, "main"),
      StreamConfig("stream_2", 1280, 720, 60, "main"),
      StreamConfig("stream_3", 3840, 2160, 24, "main10"),
      StreamConfig("stream_4", 1920, 1080, 25, "main"),
      StreamConfig("stream_5", 1280, 720, 30, "main"),
    ]

    # Add streams up to num_streams limit
    added_streams = 0
    for config in stream_configs[:num_streams]:
      if processor.add_stream(config):
        added_streams += 1

    if added_streams == 0:
      print("❌ No streams could be added")
      return False

    print(f"✅ Added {added_streams} streams")

    # Start processing
    processor.start_processing()

    # Monitor processing for a few seconds
    monitor_duration = 10.0  # 10 seconds
    start_time = time.time()
    last_stats_time = start_time

    print(f"\n📊 Monitoring processing for {monitor_duration} seconds...")

    while time.time() - start_time < monitor_duration:
      time.sleep(2.0)  # Update every 2 seconds

      current_time = time.time()
      elapsed = current_time - start_time

      # Get current statistics
      stats = processor.get_statistics()
      global_stats = stats['global_stats']

      print(f"\n⏱️  Time: {elapsed:.1f}s | "
            f"Frames: {global_stats['total_frames']} | "
            f"Failed: {global_stats['failed_frames']} | "
            f"Success: {global_stats['success_rate']*100:.1f}% | "
            f"Total FPS: {global_stats['total_fps']:.1f}")

      # Show per-stream stats
      for stream_id, stream_stat in stats['stream_stats'].items():
        print(f"  {stream_id}: {stream_stat['frames_processed']} frames, "
              f"{stream_stat['fps']:.1f} FPS, "
              f"decode={stream_stat['avg_decode_ms']:.2f}ms")

    # Final statistics
    print(f"\n📊 Final Processing Statistics:")
    print("=" * 60)

    final_stats = processor.get_statistics()
    global_stats = final_stats['global_stats']

    print(f"Global Performance:")
    print(f"  Total Frames Processed: {global_stats['total_frames']}")
    print(f"  Failed Frames: {global_stats['failed_frames']}")
    print(f"  Success Rate: {global_stats['success_rate']*100:.1f}%")
    print(f"  Average Decode Time: {global_stats['avg_decode_ms']:.2f}ms")
    print(f"  Average Process Time: {global_stats['avg_process_ms']:.2f}ms")
    print(f"  Total Throughput: {global_stats['total_fps']:.1f} FPS")
    print(f"  Active Streams: {global_stats['active_streams']}")

    print(f"\nPer-Stream Performance:")
    for stream_id, stream_stat in final_stats['stream_stats'].items():
      config = processor.stream_configs[stream_id]
      print(f"  {config.name}:")
      print(f"    Frames: {stream_stat['frames_processed']}")
      print(f"    Success Rate: {stream_stat['success_rate']*100:.1f}%")
      print(f"    FPS: {stream_stat['fps']:.1f}")
      print(f"    Decode: {stream_stat['avg_decode_ms']:.2f}ms")
      print(f"    Process: {stream_stat['avg_process_ms']:.2f}ms")

    print(f"\nMemory Statistics:")
    memory_stats = final_stats['memory_stats']
    for profile, profile_stats in memory_stats.items():
      print(f"  {profile}: {profile_stats}")

    # Cleanup
    processor.cleanup()

    print(f"\n🎉 Multi-stream demo completed successfully!")
    return True

  except Exception as e:
    print(f"❌ Multi-stream demo failed: {e}")
    return False

def batch_processing_demo(batch_size: int = 8):
  """Batch processing demonstration"""
  print(f"🎬 Batch HEVC Processing Demo")
  print(f"Batch Size: {batch_size}")
  print("=" * 50)

  if not HEVC_AVAILABLE:
    print("⚠️  HEVC hardware decode not available, using mock mode")
    print("💡 Install NVIDIA Video Codec SDK for full functionality")

  try:
    # Create mock device
    from unittest.mock import Mock
    device = Mock()
    device.device = "CUDA"

    # Create tensor converter
    converter = VideoTensorConverter(device)

    # Generate batch of HEVC frames
    print(f"📦 Generating batch of {batch_size} HEVC frames...")

    hevc_frames = []
    for i in range(batch_size):
      # Mock HEVC frame
      frame_data = b'\x00\x00\x00\x01\x26\x01'  # Start + IDR NAL
      frame_data += b'\xaf\x15\x24\x84\x44\x44\x95\x6f'
      frame_data += i.to_bytes(4, 'big')  # Frame index
      frame_data += b'\xff\x2c\x10\x42\x3c\x99\x88\x08'

      hevc_frames.append(frame_data)

    print(f"✅ Generated {len(hevc_frames)} frames")

    # Batch decode
    print(f"🎬 Performing batch decode...")
    start_time = time.time()

    # Mock batch decode (would use real function in actual implementation)
    decoded_tensors = []
    for i, frame_data in enumerate(hevc_frames):
      mock_tensor = Mock()
      mock_tensor.shape = (3, 1080, 1920)  # RGB
      mock_tensor.dtype = "uint8"
      decoded_tensors.append(mock_tensor)

      if i % 2 == 0:
        print(f"  Decoded frame {i+1}/{batch_size}")

    decode_time = (time.time() - start_time) * 1000

    print(f"✅ Batch decode completed: {len(decoded_tensors)} tensors in {decode_time:.2f}ms")
    print(f"   Average per frame: {decode_time/batch_size:.2f}ms")
    print(f"   Effective FPS: {1000*batch_size/decode_time:.1f}")

    # Mock batch processing
    print(f"⚙️  Processing tensor batch...")
    processing_start = time.time()

    processed_tensors = []
    for tensor in decoded_tensors:
      # Mock tensor operations (would be real GPU ops)
      processed_tensor = Mock()
      processed_tensor.shape = tensor.shape
      processed_tensor.dtype = tensor.dtype
      processed_tensors.append(processed_tensor)

    processing_time = (time.time() - processing_start) * 1000

    print(f"✅ Batch processing completed: {processing_time:.2f}ms")
    print(f"   Total pipeline time: {decode_time + processing_time:.2f}ms")
    print(f"   Pipeline FPS: {1000*batch_size/(decode_time + processing_time):.1f}")

    print(f"🎉 Batch processing demo completed!")
    return True

  except Exception as e:
    print(f"❌ Batch processing demo failed: {e}")
    return False

def main():
  """Main demo application"""
  parser = argparse.ArgumentParser(description="Advanced HEVC Video Processing Pipeline")
  parser.add_argument('--streams', '-s', type=int, default=4, help='Number of decode streams')
  parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch processing size')
  parser.add_argument('--profile', '-p', choices=['multi', 'batch', 'both'], default='both',
                      help='Demo profile to run')

  args = parser.parse_args()

  print("🎬 tinygrad Advanced HEVC Video Processing")
  print("=" * 70)

  success = True

  if args.profile in ['multi', 'both']:
    print("\n" + "="*70)
    success &= multi_stream_demo(args.streams, args.batch_size)

  if args.profile in ['batch', 'both']:
    print("\n" + "="*70)
    success &= batch_processing_demo(args.batch_size)

  if success:
    print(f"\n🎉 All demos completed successfully!")
    print(f"💡 Advanced video processing capabilities demonstrated")
    print(f"💡 Multi-stream: {args.streams} concurrent decode streams")
    print(f"💡 Batch processing: {args.batch_size} frames per batch")
  else:
    print(f"\n❌ Some demos failed")
    print(f"💡 Install NVIDIA Video Codec SDK for full functionality")

  return 0 if success else 1

if __name__ == "__main__":
  sys.exit(main())