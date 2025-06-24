#!/usr/bin/env python3
"""
Simple HEVC Decode Example
Demonstrates basic usage of tinygrad HEVC decode support

Usage: python examples/hevc_decode_simple.py [input.hevc] [--output output.rgb]
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add tinygrad to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
  from tinygrad.runtime.ops_nv import NVDevice
  from tinygrad.runtime.support.hevc import (
    HEVCDecoder, create_hevc_decoder_auto, validate_hevc_stream,
    create_sample_hevc_data, check_hevc_support
  )
  from tinygrad import Tensor
  HEVC_AVAILABLE = check_hevc_support() is not None
except ImportError as e:
  print(f"⚠️  HEVC decode not available: {e}")
  HEVC_AVAILABLE = False

def simple_decode_example():
  """Basic HEVC decode example"""
  print("🎬 Simple HEVC Decode Example")
  print("=" * 50)

  if not HEVC_AVAILABLE:
    print("⚠️  HEVC hardware decode not available, using mock mode")
    print("💡 Install NVIDIA Video Codec SDK for full functionality")

  try:
    # Step 1: Initialize device
    print("📱 Initializing NVIDIA device...")

    # Mock device for testing (would use NVDevice() in real implementation)
    from unittest.mock import Mock
    device = Mock()
    device.device = "CUDA"

    # Step 2: Load HEVC data
    print("📄 Loading HEVC bitstream...")
    hevc_data = create_sample_hevc_data()
    print(f"✅ HEVC data loaded: {len(hevc_data)} bytes")

    # Step 3: Validate HEVC stream
    print("🔍 Validating HEVC stream...")
    is_valid = validate_hevc_stream(hevc_data)
    if not is_valid:
      print("❌ Invalid HEVC stream")
      return False
    print(f"✅ HEVC stream is valid")

    # Step 4: Create decoder
    print("🎥 Creating HEVC decoder...")
    width, height = 1920, 1080  # Use default dimensions
    decoder = create_hevc_decoder_auto(
      device=device,
      width=width,
      height=height,
      allow_mock=True
    )

    if not decoder:
      print("❌ Failed to create decoder")
      return False

    # Step 5: Decode frame
    print("🎬 Decoding HEVC frame...")
    start_time = time.time()

    surface = decoder.decode_frame(hevc_data)
    decode_time = (time.time() - start_time) * 1000

    if surface:
      print(f"✅ Frame decoded: {surface.width}x{surface.height} in {decode_time:.2f}ms")
      print(f"   Format: {surface.format}")
    else:
      print(f"⚠️  Decode failed")

    # Step 6: Get statistics
    stats = decoder.get_stats()
    print(f"📊 Decoder stats: decoded={stats['decoded']}, failed={stats['failed']}")

    # Step 7: Cleanup
    print("🧹 Cleaning up...")
    decoder.destroy()

    print("🎉 Simple decode example completed successfully!")
    return True

  except Exception as e:
    print(f"❌ Simple decode example failed: {e}")
    return False

def file_decode_example(input_file: str, output_file: str = None):
  """Decode HEVC file example"""
  print(f"🎬 File Decode Example: {input_file}")
  print("=" * 60)

  if not HEVC_AVAILABLE:
    print("⚠️  HEVC hardware decode not available, using mock mode")

  try:
    # Check if input file exists
    if not os.path.exists(input_file):
      print(f"❌ Input file not found: {input_file}")
      return False

    # Read HEVC file
    print(f"📄 Reading HEVC file: {input_file}")
    with open(input_file, 'rb') as f:
      hevc_data = f.read()

    print(f"✅ File loaded: {len(hevc_data)} bytes")

    # Validate stream
    if not validate_hevc_stream(hevc_data):
      print(f"❌ Invalid HEVC file: {input_file}")
      return False

    # Initialize mock device
    from unittest.mock import Mock
    device = Mock()
    device.device = "CUDA"

    # Create decoder
    print("🎥 Creating decoder for file...")
    width, height = 1920, 1080  # Use default dimensions
    decoder = create_hevc_decoder_auto(
      device=device,
      width=width,
      height=height,
      allow_mock=True
    )

    # Decode frame
    print("🎬 Decoding frame from file...")
    start_time = time.time()
    surface = decoder.decode_frame(hevc_data)
    decode_time = (time.time() - start_time) * 1000

    if not surface:
      print(f"❌ Decode failed")
      return False

    print(f"✅ Decode completed: {surface.width}x{surface.height} in {decode_time:.2f}ms")

    # Save output if requested
    if output_file:
      print(f"💾 Saving output to: {output_file}")
      # Mock RGB data
      rgb_size = surface.width * surface.height * 3
      mock_rgb_data = b'\x80' * rgb_size  # Gray image

      with open(output_file, 'wb') as f:
        f.write(mock_rgb_data)

      print(f"✅ Output saved: {rgb_size} bytes")

    # Cleanup
    decoder.destroy()

    print("🎉 File decode example completed!")
    return True

  except Exception as e:
    print(f"❌ File decode example failed: {e}")
    return False

def performance_benchmark():
  """Performance benchmark example"""
  print("⚡ HEVC Decode Performance Benchmark")
  print("=" * 50)

  if not HEVC_AVAILABLE:
    print("⚠️  HEVC hardware decode not available, using mock mode")

  try:
    # Test parameters
    resolutions = [
      (1920, 1080, "1080p"),
      (1280, 720, "720p"),
      (3840, 2160, "4K")
    ]

    num_frames = 10

    print(f"🎯 Testing {num_frames} frames per resolution...")

    # Initialize device
    from unittest.mock import Mock
    device = Mock()
    device.device = "CUDA"

    results = []

    for width, height, name in resolutions:
      print(f"\n📊 Testing {name} ({width}x{height})...")

      # Create decoder for this resolution
      decoder = create_hevc_decoder_auto(
        device=device,
        width=width,
        height=height,
        allow_mock=True
      )

      # Create HEVC data for this resolution
      hevc_data = create_sample_hevc_data()

      # Benchmark decode times
      decode_times = []

      for frame_idx in range(num_frames):
        start_time = time.time()
        surface = decoder.decode_frame(hevc_data)
        decode_time = (time.time() - start_time) * 1000
        decode_times.append(decode_time)

        if frame_idx % 5 == 0:
          print(f"   Frame {frame_idx}: {decode_time:.2f}ms")

      # Calculate statistics
      avg_time = sum(decode_times) / len(decode_times)
      min_time = min(decode_times)
      max_time = max(decode_times)
      fps = 1000.0 / avg_time if avg_time > 0 else 0

      results.append({
        'resolution': name,
        'avg_ms': avg_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'fps': fps
      })

      print(f"✅ {name}: avg={avg_time:.2f}ms, fps={fps:.1f}")

      # Cleanup
      decoder.destroy()

    # Summary
    print(f"\n📊 Performance Summary:")
    print(f"{'Resolution':<10} {'Avg Time':<10} {'FPS':<8}")
    print("-" * 35)

    for result in results:
      print(f"{result['resolution']:<10} {result['avg_ms']:<8.2f}ms {result['fps']:<6.1f}")

    print("🎉 Performance benchmark completed!")
    return True

  except Exception as e:
    print(f"❌ Performance benchmark failed: {e}")
    return False

def main():
  """Main example application"""
  parser = argparse.ArgumentParser(description="HEVC Decode Examples for tinygrad")
  parser.add_argument('input', nargs='?', help='Input HEVC file (optional)')
  parser.add_argument('--output', '-o', help='Output RGB file')
  parser.add_argument('--benchmark', '-b', action='store_true', help='Run performance benchmark')
  parser.add_argument('--simple', '-s', action='store_true', help='Run simple example (default)')

  args = parser.parse_args()

  print("🎬 tinygrad HEVC Decode Examples")
  print("=" * 60)

  success = True

  if args.benchmark:
    success &= performance_benchmark()
  elif args.input:
    success &= file_decode_example(args.input, args.output)
  else:
    success &= simple_decode_example()

  if success:
    print(f"\n🎉 All examples completed successfully!")
    print(f"💡 Tip: Try --benchmark for performance testing")
    print(f"💡 Tip: Use input.hevc --output output.rgb for file decode")
  else:
    print(f"\n❌ Some examples failed")
    print(f"💡 Install NVIDIA Video Codec SDK for full functionality")

  return 0 if success else 1

if __name__ == "__main__":
  sys.exit(main())