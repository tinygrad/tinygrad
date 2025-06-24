# Video Memory Management for HEVC Decode
# Following tinygrad HCQ allocator patterns

from typing import Optional, List, Dict, Set, Tuple
import threading
import time
import weakref
from dataclasses import dataclass
from enum import Enum

try:
  from tinygrad.runtime.ops_nv import NVVideoSurface
  from tinygrad.runtime.support.hcq import HCQBuffer
except ImportError:
  print("⚠️  HCQ/NVDevice imports not available")

class SurfaceState(Enum):
  """Video surface lifecycle states"""
  FREE = "free"
  ALLOCATED = "allocated"
  IN_USE = "in_use"
  PENDING_RELEASE = "pending_release"

@dataclass
class SurfaceInfo:
  """Surface metadata for pool management"""
  surface: NVVideoSurface
  state: SurfaceState
  allocated_time: float
  last_used_time: float
  use_count: int = 0
  width: int = 0
  height: int = 0
  format: str = "NV12"

class VideoBufferPool:
  """Video surface buffer pool with automatic recycling"""
  
  def __init__(self, device_interface, max_surfaces:int=16, recycle_timeout:float=2.0):
    self.device = device_interface
    self.max_surfaces = max_surfaces
    self.recycle_timeout = recycle_timeout
    
    # Pool management
    self._lock = threading.RLock()
    self.surfaces: Dict[int, SurfaceInfo] = {}  # surface_id -> SurfaceInfo
    self.free_by_format: Dict[Tuple[int, int, str], List[int]] = {}  # (w,h,fmt) -> surface_ids
    self.surface_counter = 0
    
    # Statistics
    self.stats = {
      'total_allocated': 0,
      'pool_hits': 0,
      'pool_misses': 0,
      'recycled_surfaces': 0,
      'peak_usage': 0
    }
    
    print(f"🎬 VideoBufferPool created: max={max_surfaces}, timeout={recycle_timeout}s")

  def get_surface(self, width:int, height:int, format:str="NV12") -> Optional[NVVideoSurface]:
    """Get video surface from pool or allocate new one"""
    with self._lock:
      key = (width, height, format)
      
      # Try to reuse existing free surface
      if key in self.free_by_format and self.free_by_format[key]:
        surface_id = self.free_by_format[key].pop(0)
        surface_info = self.surfaces[surface_id]
        
        # Update surface state
        surface_info.state = SurfaceState.IN_USE
        surface_info.last_used_time = time.time()
        surface_info.use_count += 1
        
        self.stats['pool_hits'] += 1
        print(f"🔄 Reused surface {surface_id}: {width}x{height} {format}")
        
        return surface_info.surface
      
      # Check if we can allocate new surface
      if len(self.surfaces) >= self.max_surfaces:
        # Try to recycle old surfaces
        recycled = self._recycle_old_surfaces()
        if recycled and key in self.free_by_format and self.free_by_format[key]:
          return self.get_surface(width, height, format)  # Recursive retry
        
        print(f"⚠️  Surface pool exhausted: {len(self.surfaces)}/{self.max_surfaces}")
        return None
      
      # Allocate new surface
      surface = self._allocate_new_surface(width, height, format)
      if surface:
        self.stats['pool_misses'] += 1
        self.stats['peak_usage'] = max(self.stats['peak_usage'], len(self.surfaces))
        
      return surface

  def _allocate_new_surface(self, width:int, height:int, format:str) -> Optional[NVVideoSurface]:
    """Allocate new video surface"""
    try:
      if not hasattr(self.device, '_alloc_video_surface'):
        print("❌ Device does not support video surface allocation")
        return None
        
      surface = self.device._alloc_video_surface(width, height, format)
      if not surface:
        return None
        
      # Add to pool management
      surface_id = self.surface_counter
      self.surface_counter += 1
      
      surface_info = SurfaceInfo(
        surface=surface,
        state=SurfaceState.IN_USE,
        allocated_time=time.time(),
        last_used_time=time.time(),
        width=width,
        height=height,
        format=format
      )
      
      self.surfaces[surface_id] = surface_info
      self.stats['total_allocated'] += 1
      
      print(f"✅ Allocated new surface {surface_id}: {width}x{height} {format}")
      return surface
      
    except Exception as e:
      print(f"❌ Surface allocation failed: {e}")
      return None

  def release_surface(self, surface:NVVideoSurface):
    """Release surface back to pool for reuse"""
    with self._lock:
      # Find surface in pool
      surface_id = None
      for sid, info in self.surfaces.items():
        if info.surface is surface:
          surface_id = sid
          break
          
      if surface_id is None:
        print(f"⚠️  Surface not found in pool for release")
        return
        
      surface_info = self.surfaces[surface_id]
      
      # Update surface state
      surface_info.state = SurfaceState.FREE
      surface_info.last_used_time = time.time()
      
      # Add to free list by format
      key = (surface_info.width, surface_info.height, surface_info.format)
      if key not in self.free_by_format:
        self.free_by_format[key] = []
      self.free_by_format[key].append(surface_id)
      
      print(f"🔄 Released surface {surface_id} to pool")

  def _recycle_old_surfaces(self) -> int:
    """Recycle old unused surfaces to free up pool space"""
    recycled_count = 0
    current_time = time.time()
    
    # Find surfaces eligible for recycling
    to_recycle = []
    for surface_id, info in self.surfaces.items():
      if (info.state == SurfaceState.FREE and 
          current_time - info.last_used_time > self.recycle_timeout):
        to_recycle.append(surface_id)
    
    # Recycle surfaces (oldest first)
    to_recycle.sort(key=lambda sid: self.surfaces[sid].last_used_time)
    
    for surface_id in to_recycle[:self.max_surfaces//4]:  # Recycle up to 25% at once
      self._recycle_surface(surface_id)
      recycled_count += 1
      
    if recycled_count > 0:
      self.stats['recycled_surfaces'] += recycled_count
      print(f"♻️  Recycled {recycled_count} old surfaces")
      
    return recycled_count

  def _recycle_surface(self, surface_id:int):
    """Recycle specific surface and free its memory"""
    if surface_id not in self.surfaces:
      return
      
    surface_info = self.surfaces[surface_id]
    
    # Remove from free lists
    key = (surface_info.width, surface_info.height, surface_info.format)
    if key in self.free_by_format and surface_id in self.free_by_format[key]:
      self.free_by_format[key].remove(surface_id)
      
    # Free GPU memory (would call device.allocator.free in real implementation)
    # For now, just mark as recycled
    surface_info.state = SurfaceState.PENDING_RELEASE
    
    # Remove from pool
    del self.surfaces[surface_id]

  def get_stats(self) -> Dict[str, any]:
    """Get pool statistics"""
    with self._lock:
      current_stats = dict(self.stats)
      current_stats.update({
        'active_surfaces': len(self.surfaces),
        'free_surfaces': sum(len(ids) for ids in self.free_by_format.values()),
        'memory_formats': list(self.free_by_format.keys())
      })
      return current_stats

  def cleanup(self):
    """Cleanup all surfaces in pool"""
    with self._lock:
      print(f"🧹 Cleaning up video buffer pool...")
      
      # Recycle all surfaces
      for surface_id in list(self.surfaces.keys()):
        self._recycle_surface(surface_id)
        
      self.free_by_format.clear()
      
      print(f"✅ Pool cleanup complete")

  def __del__(self):
    """Ensure cleanup on deletion"""
    self.cleanup()

class VideoMemoryManager:
  """High-level video memory management"""
  
  def __init__(self, device_interface):
    self.device = device_interface
    self._pools: Dict[str, VideoBufferPool] = {}  # profile -> pool
    self._lock = threading.RLock()
    
    # Default pool for common video formats
    self.default_pool = VideoBufferPool(device_interface, max_surfaces=16)
    
  def get_pool(self, profile:str="default") -> VideoBufferPool:
    """Get buffer pool for specific profile"""
    with self._lock:
      if profile == "default":
        return self.default_pool
        
      if profile not in self._pools:
        # Create pool for specific profile (e.g., "4K", "HD", "mobile")
        pool_config = self._get_pool_config(profile)
        self._pools[profile] = VideoBufferPool(
          self.device, 
          max_surfaces=pool_config['max_surfaces'],
          recycle_timeout=pool_config['recycle_timeout']
        )
        
      return self._pools[profile]

  def _get_pool_config(self, profile:str) -> Dict[str, any]:
    """Get pool configuration for profile"""
    configs = {
      "4K": {"max_surfaces": 8, "recycle_timeout": 1.0},
      "HD": {"max_surfaces": 16, "recycle_timeout": 2.0}, 
      "mobile": {"max_surfaces": 4, "recycle_timeout": 3.0}
    }
    return configs.get(profile, {"max_surfaces": 12, "recycle_timeout": 2.0})

  def get_surface(self, width:int, height:int, format:str="NV12", profile:str="default") -> Optional[NVVideoSurface]:
    """Get video surface from appropriate pool"""
    pool = self.get_pool(profile)
    return pool.get_surface(width, height, format)

  def release_surface(self, surface:NVVideoSurface, profile:str="default"):
    """Release surface back to pool"""
    pool = self.get_pool(profile)
    pool.release_surface(surface)

  def get_global_stats(self) -> Dict[str, any]:
    """Get statistics for all pools"""
    stats = {"default": self.default_pool.get_stats()}
    for profile, pool in self._pools.items():
      stats[profile] = pool.get_stats()
    return stats

  def cleanup_all(self):
    """Cleanup all memory pools"""
    print(f"🧹 Cleaning up all video memory pools...")
    self.default_pool.cleanup()
    for pool in self._pools.values():
      pool.cleanup()
    self._pools.clear()

# Utility functions for memory management
def estimate_surface_memory(width:int, height:int, format:str="NV12") -> int:
  """Estimate memory usage for video surface"""
  if format == "NV12":
    return width * height * 3 // 2  # Y + UV/2
  elif format == "RGBA":
    return width * height * 4
  elif format == "YUV420":
    return width * height * 3 // 2
  else:
    return width * height * 2  # Default assumption

def get_optimal_pool_size(resolution_profile:str) -> int:
  """Get optimal pool size for resolution profile"""
  profiles = {
    "4K": 8,      # 3840x2160
    "2K": 12,     # 2560x1440  
    "1080p": 16,  # 1920x1080
    "720p": 20,   # 1280x720
    "480p": 24    # 854x480
  }
  return profiles.get(resolution_profile, 16)

# Export main classes and functions
__all__ = [
  'VideoBufferPool', 'VideoMemoryManager', 'SurfaceState', 'SurfaceInfo',
  'estimate_surface_memory', 'get_optimal_pool_size'
] 