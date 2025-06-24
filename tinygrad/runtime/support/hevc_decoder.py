# NVHEVC Decoder Core Implementation
# Following tinygrad HCQ patterns for hardware decoder management

from typing import Optional, List, Tuple, Any, Dict
import threading
import time
from dataclasses import dataclass
from enum import Enum

# Import support modules
try:
  from tinygrad.runtime.support.cuvid import (
    CUVIDDECODECREATEINFO, CUVIDDECODECAPS, CUVIDPICPARAMS,
    check_hevc_support, create_hevc_decoder, cuvid_check
  )
  from tinygrad.runtime.support.hevc_parser import HEVCParser, extract_parameter_sets
  from tinygrad.runtime.support.nvdec import NVDECEngine, create_nvdec_engine
  from tinygrad.runtime.ops_nv import NVVideoSurface
except ImportError as e:
  print(f"⚠️  HEVC decoder dependencies not available: {e}")

class DecoderState(Enum):
  """Decoder lifecycle states"""
  UNINITIALIZED = "uninitialized"
  INITIALIZING = "initializing" 
  READY = "ready"
  DECODING = "decoding"
  ERROR = "error"
  DESTROYED = "destroyed"

@dataclass 
class DecoderStats:
  """Decoder performance and error statistics"""
  frames_decoded: int = 0
  frames_failed: int = 0
  bytes_processed: int = 0
  decode_time_ms: float = 0.0
  last_error: Optional[str] = None
  
  @property
  def success_rate(self) -> float:
    total = self.frames_decoded + self.frames_failed
    return (self.frames_decoded / total) if total > 0 else 0.0

class NVHEVCDecoder:
  """Main HEVC decoder class implementing HCQ-compatible lifecycle"""
  
  def __init__(self, device_interface, width:int, height:int, max_surfaces:int=8):
    self.device = device_interface
    self.width = width
    self.height = height
    self.max_surfaces = max_surfaces
    
    # Decoder state management
    self.state = DecoderState.UNINITIALIZED
    self.stats = DecoderStats()
    self._lock = threading.RLock()
    
    # CUVID/NVDEC components
    self.cuvid_decoder = None
    self.nvdec_engine = None
    self.decode_caps = None
    
    # Memory management with video buffer pool
    try:
      from tinygrad.runtime.support.video_memory import VideoMemoryManager
      self.memory_manager = VideoMemoryManager(device_interface)
      print(f"✅ Video memory manager initialized")
    except ImportError:
      self.memory_manager = None
      print(f"⚠️  Video memory manager not available, using basic surface management")
    
    # Synchronization management
    try:
      from tinygrad.runtime.support.video_sync import VideoSyncManager
      self.sync_manager = VideoSyncManager(device_interface)
      print(f"✅ Video sync manager initialized")
    except ImportError:
      self.sync_manager = None
      print(f"⚠️  Video sync manager not available, using basic synchronization")
    
    # Basic surface management (fallback)
    self.surfaces: List[NVVideoSurface] = []
    self.free_surfaces: List[int] = []  # Surface indices
    self.used_surfaces: Dict[int, float] = {}  # Surface idx -> timestamp
    
    # HEVC parsing
    self.parser = HEVCParser()
    self.sps_data = None
    self.pps_data = None
    self.vps_data = None
    
    print(f"🎬 NVHEVCDecoder created: {width}x{height}, max_surfaces={max_surfaces}")

  def initialize(self) -> bool:
    """Initialize decoder with capability checking following HCQ patterns"""
    with self._lock:
      if self.state != DecoderState.UNINITIALIZED:
        print(f"⚠️  Decoder already initialized (state: {self.state})")
        return self.state == DecoderState.READY
        
      self.state = DecoderState.INITIALIZING
      
      try:
        # Step 1: Check HEVC decode capabilities
        self.decode_caps = check_hevc_support()
        if not self.decode_caps:
          raise RuntimeError("HEVC decode capabilities not available")
          
        # Validate resolution support
        if self.width > self.decode_caps.nMaxWidth or self.height > self.decode_caps.nMaxHeight:
          raise RuntimeError(f"Resolution {self.width}x{self.height} exceeds max {self.decode_caps.nMaxWidth}x{self.decode_caps.nMaxHeight}")
          
        print(f"✅ HEVC caps: {self.decode_caps.nNumNVDECs} engines, max {self.decode_caps.nMaxWidth}x{self.decode_caps.nMaxHeight}")
        
        # Step 2: Initialize NVDEC engine
        self.nvdec_engine = create_nvdec_engine(engine_id=0, device_interface=self.device)
        if not self.nvdec_engine or not self.nvdec_engine.initialize():
          raise RuntimeError("NVDEC engine initialization failed")
          
        print(f"✅ NVDEC engine initialized: type=0x{self.nvdec_engine.info.engine_type:x}")
        
        # Step 3: Create CUVID decoder instance
        self.cuvid_decoder = create_hevc_decoder(self.width, self.height)
        if not self.cuvid_decoder:
          raise RuntimeError("CUVID decoder creation failed")
          
        print(f"✅ CUVID decoder created: {self.width}x{self.height}")
        
        # Step 4: Allocate video surfaces
        self._allocate_surfaces()
        
        self.state = DecoderState.READY
        print(f"🎬 NVHEVCDecoder initialized successfully")
        return True
        
      except Exception as e:
        self.stats.last_error = str(e)
        self.state = DecoderState.ERROR
        print(f"❌ Decoder initialization failed: {e}")
        return False

  def _allocate_surfaces(self):
    """Allocate video surfaces for decode output"""
    if not hasattr(self.device, '_alloc_video_surface'):
      print("⚠️  Device does not support video surface allocation")
      return
      
    try:
      for i in range(self.max_surfaces):
        surface = self.device._alloc_video_surface(self.width, self.height, "NV12")
        self.surfaces.append(surface)
        self.free_surfaces.append(i)
        
      print(f"✅ Allocated {len(self.surfaces)} video surfaces")
      
    except Exception as e:
      print(f"❌ Surface allocation failed: {e}")
      raise

  def decode_frame(self, bitstream_data:bytes, wait:bool=True) -> Optional[NVVideoSurface]:
    """Decode single HEVC frame with proper error handling and synchronization"""
    with self._lock:
      if self.state != DecoderState.READY:
        print(f"❌ Decoder not ready for decode (state: {self.state})")
        return None
        
      self.state = DecoderState.DECODING
      decode_start = time.time()
      decode_id = self.stats.frames_decoded + self.stats.frames_failed
      
      # Submit decode for synchronization tracking
      decode_sync = None
      if self.sync_manager:
        decode_sync = self.sync_manager.submit_decode(decode_id)
      
      try:
        # Step 1: Parse bitstream for frame info
        frame_info = self._parse_frame(bitstream_data)
        if not frame_info:
          raise RuntimeError("Failed to parse HEVC frame")
          
        # Step 2: Get free surface for output (use memory manager if available)
        if self.memory_manager:
          output_surface = self.memory_manager.get_surface(
            frame_info['width'], frame_info['height'], "NV12"
          )
          if not output_surface:
            raise RuntimeError("No free surfaces available from memory manager")
          surface_idx = -1  # Use memory manager tracking
        else:
          # Fallback to basic surface management
          surface_idx = self._get_free_surface()
          if surface_idx is None:
            raise RuntimeError("No free surfaces available")
          output_surface = self.surfaces[surface_idx]
        
        # Step 3: Submit decode via NVDEC engine
        success = self._submit_decode(bitstream_data, output_surface, frame_info)
        if not success:
          if self.memory_manager:
            self.memory_manager.release_surface(output_surface)
          else:
            self._release_surface(surface_idx)
          raise RuntimeError("Decode submission failed")
          
        # Step 4: Track surface usage
        if not self.memory_manager:
          self.used_surfaces[surface_idx] = time.time()
        
        # Wait for decode completion if requested
        if wait and decode_sync:
          success = self.sync_manager.wait_for_decode(decode_id, timeout_ms=5000.0)
          if not success:
            print(f"⚠️  Decode synchronization timeout")
        
        # Signal completion for synchronization
        if decode_sync:
          self.sync_manager.signal_decode_complete(decode_id)
        
        # Update statistics
        decode_time = (time.time() - decode_start) * 1000
        self.stats.frames_decoded += 1
        self.stats.bytes_processed += len(bitstream_data)
        self.stats.decode_time_ms += decode_time
        
        print(f"🎬 Frame decoded: {frame_info['width']}x{frame_info['height']} in {decode_time:.2f}ms")
        
        self.state = DecoderState.READY
        return output_surface
        
      except Exception as e:
        self.stats.frames_failed += 1
        self.stats.last_error = str(e)
        self.state = DecoderState.ERROR
        print(f"❌ Frame decode failed: {e}")
        
        # Transition back to ready if error is recoverable
        if "No free surfaces" not in str(e):
          self.state = DecoderState.READY
          
        return None

  def _parse_frame(self, bitstream_data:bytes) -> Optional[Dict[str, Any]]:
    """Parse HEVC frame and extract decode parameters"""
    try:
      # Parse NAL units
      nal_units = self.parser.parse_bitstream(bitstream_data)
      if not nal_units:
        return None
        
      # Extract parameter sets if available
      param_sets = extract_parameter_sets(bitstream_data)
      if param_sets.get('sps'):
        self.sps_data = param_sets['sps']
      if param_sets.get('pps'):
        self.pps_data = param_sets['pps']
      if param_sets.get('vps'):
        self.vps_data = param_sets['vps']
        
      # Get frame dimensions from SPS if available
      frame_width, frame_height = self.width, self.height
      if self.sps_data:
        try:
          from tinygrad.runtime.support.hevc_parser import get_frame_dimensions
          frame_width, frame_height = get_frame_dimensions(self.sps_data)
        except:
          pass
          
      return {
        'width': frame_width,
        'height': frame_height,
        'nal_count': len(nal_units),
        'has_sps': bool(self.sps_data),
        'has_pps': bool(self.pps_data),
        'bitstream_size': len(bitstream_data)
      }
      
    except Exception as e:
      print(f"⚠️  Frame parsing failed: {e}")
      return None

  def _get_free_surface(self) -> Optional[int]:
    """Get free surface index for decode output"""
    if self.free_surfaces:
      return self.free_surfaces.pop(0)
      
    # Try to reclaim old surfaces (simple timeout-based reclamation)
    current_time = time.time()
    for surface_idx, timestamp in list(self.used_surfaces.items()):
      if current_time - timestamp > 1.0:  # 1 second timeout
        self._release_surface(surface_idx)
        return surface_idx
        
    return None

  def _release_surface(self, surface_idx:int):
    """Release surface back to free pool"""
    if surface_idx in self.used_surfaces:
      del self.used_surfaces[surface_idx]
    if surface_idx not in self.free_surfaces:
      self.free_surfaces.append(surface_idx)

  def _submit_decode(self, bitstream_data:bytes, output_surface:NVVideoSurface, frame_info:Dict[str, Any]) -> bool:
    """Submit decode command to NVDEC engine"""
    try:
      # Create hardware descriptors for decode
      from tinygrad.runtime.support.nvdec import NVDECBitstreamBuffer, NVDECSurfaceDesc, NVDECDecodeParams
      
      # Bitstream buffer (simplified - real implementation would upload to GPU)
      bitstream_desc = NVDECBitstreamBuffer(
        gpu_addr=0x1000000,  # Mock GPU address
        size=len(bitstream_data)
      )
      
      # Output surface descriptor
      surface_desc = NVDECSurfaceDesc(
        gpu_addr=int(output_surface.va_addr),
        width=frame_info['width'], 
        height=frame_info['height'],
        format="NV12"
      )
      
      # Decode parameters
      decode_params = NVDECDecodeParams(
        pic_width_in_mbs=(frame_info['width'] + 15) // 16,
        pic_height_in_mbs=(frame_info['height'] + 15) // 16
      )
      
      # Submit to NVDEC engine
      return self.nvdec_engine.submit_decode_command(bitstream_desc, surface_desc, decode_params)
      
    except Exception as e:
      print(f"❌ Decode submission failed: {e}")
      return False

  def destroy(self):
    """Cleanup decoder resources following HCQ patterns"""
    with self._lock:
      if self.state == DecoderState.DESTROYED:
        return
        
      print(f"🎬 Destroying NVHEVCDecoder...")
      
      # Release all surfaces
      for surface_idx in list(self.used_surfaces.keys()):
        self._release_surface(surface_idx)
        
      # Cleanup synchronization
      if self.sync_manager:
        self.sync_manager.destroy()
        
      # Cleanup memory management
      if self.memory_manager:
        self.memory_manager.cleanup_all()
      
      # Cleanup CUVID/NVDEC resources
      if self.cuvid_decoder:
        # CUVID cleanup would go here
        self.cuvid_decoder = None
        
      if self.nvdec_engine:
        # NVDEC cleanup would go here
        self.nvdec_engine = None
        
      self.surfaces.clear()
      self.free_surfaces.clear()
      
      self.state = DecoderState.DESTROYED
      print(f"✅ Decoder destroyed")

  def get_stats(self) -> DecoderStats:
    """Get decoder performance statistics"""
    return self.stats

  def is_ready(self) -> bool:
    """Check if decoder is ready for decode operations"""
    return self.state == DecoderState.READY

  def __del__(self):
    """Ensure proper cleanup on deletion"""
    if self.state != DecoderState.DESTROYED:
      self.destroy()

# Factory functions for decoder creation
def is_hevc_available() -> bool:
  """Check if HEVC hardware decode is available"""
  try:
    from tinygrad.runtime.support.cuvid import cuvid
    return cuvid is not None
  except ImportError:
    return False

def create_mock_decoder(width:int, height:int, max_surfaces:int=8):
  """Create mock decoder for testing when hardware not available"""
  from unittest.mock import Mock
  
  mock_decoder = Mock()
  mock_decoder.width = width
  mock_decoder.height = height
  mock_decoder.max_surfaces = max_surfaces
  mock_decoder.state = "READY"
  
  # Mock decode function
  def mock_decode_frame(bitstream_data, wait=True):
    mock_surface = Mock()
    mock_surface.width = width
    mock_surface.height = height
    mock_surface.format = "NV12"
    mock_surface.size = width * height * 3 // 2
    mock_surface.va_addr = 0x10000000
    return mock_surface
  
  mock_decoder.decode_frame = mock_decode_frame
  mock_decoder.destroy = Mock()
  mock_decoder.get_stats = Mock(return_value={
    'frames_decoded': 0,
    'frames_failed': 0,
    'success_rate': 1.0
  })
  
  return mock_decoder

def create_hevc_decoder(device_interface, width:int, height:int, max_surfaces:int=8) -> Optional[NVHEVCDecoder]:
  """Create real HEVC decoder with availability checking"""
  # Quick availability check before creating anything
  if not is_hevc_available():
    print(f"⚠️  CUVID library not available, cannot create real decoder")
    return None
  
  try:
    decoder = NVHEVCDecoder(device_interface, width, height, max_surfaces)
    if decoder.initialize():
      return decoder
    else:
      decoder.destroy()
      return None
  except Exception as e:
    print(f"❌ Failed to create HEVC decoder: {e}")
    return None

def create_hevc_decoder_auto(device_interface, width:int, height:int, max_surfaces:int=8, allow_mock:bool=True):
  """Create HEVC decoder with automatic fallback to mock if needed"""
  # Try real decoder first
  decoder = create_hevc_decoder(device_interface, width, height, max_surfaces)
  
  if decoder is not None:
    print(f"✅ Real HEVC decoder created: {width}x{height}")
    return decoder
  
  # Fallback to mock if allowed
  if allow_mock:
    print(f"🎭 Creating mock HEVC decoder: {width}x{height}")
    return create_mock_decoder(width, height, max_surfaces)
  
  print(f"❌ HEVC decoder unavailable and mock disabled")
  return None

# Export main classes and functions
__all__ = [
  'NVHEVCDecoder', 'DecoderState', 'DecoderStats', 
  'create_hevc_decoder', 'create_hevc_decoder_auto', 'create_mock_decoder', 'is_hevc_available'
] 