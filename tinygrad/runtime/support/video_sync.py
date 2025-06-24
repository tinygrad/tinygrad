# Video Decode Synchronization
# Following tinygrad HCQ timeline signal patterns

from typing import Optional, Dict, List, Callable, Any
import threading
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum

try:
  from tinygrad.runtime.support.hcq import HCQSignal
  from tinygrad.runtime.ops_nv import NVSignal, NVDevice
except ImportError:
  print("⚠️  HCQ/NVDevice imports not available")

class SyncState(Enum):
  """Synchronization states for decode operations"""
  PENDING = "pending"
  SUBMITTED = "submitted" 
  EXECUTING = "executing"
  COMPLETED = "completed"
  FAILED = "failed"
  TIMEOUT = "timeout"

@dataclass
class DecodeSync:
  """Decode operation synchronization tracking"""
  decode_id: int
  signal: Any  # HCQSignal
  target_value: int
  submitted_time: float
  state: SyncState = SyncState.PENDING
  completion_time: Optional[float] = None
  timeout_ms: float = 5000.0  # 5 second timeout
  
  @property
  def elapsed_ms(self) -> float:
    end_time = self.completion_time or time.time()
    return (end_time - self.submitted_time) * 1000

  @property
  def is_completed(self) -> bool:
    return self.state in [SyncState.COMPLETED, SyncState.FAILED, SyncState.TIMEOUT]

class VideoSyncManager:
  """Video decode synchronization manager following HCQ patterns"""
  
  def __init__(self, device_interface):
    self.device = device_interface
    self._lock = threading.RLock()
    
    # Decode operation tracking
    self.active_decodes: Dict[int, DecodeSync] = {}
    self.decode_counter = 0
    
    # Timeline signal integration
    self.timeline_signal = None
    self.timeline_value = 0
    
    # Multi-stream support
    self.stream_signals: Dict[str, Any] = {}  # stream_id -> signal
    self.stream_timelines: Dict[str, int] = {}  # stream_id -> timeline_value
    
    # Performance tracking
    self.stats = {
      'total_syncs': 0,
      'completed_syncs': 0,
      'failed_syncs': 0,
      'timeout_syncs': 0,
      'avg_completion_ms': 0.0
    }
    
    self._initialize_signals()
    print(f"🎬 VideoSyncManager initialized")

  def _initialize_signals(self):
    """Initialize timeline signals for video decode synchronization"""
    try:
      if hasattr(self.device, 'timeline_signal'):
        self.timeline_signal = self.device.timeline_signal
        self.timeline_value = getattr(self.device, 'timeline_value', 0)
        print(f"✅ Using device timeline signal: value={self.timeline_value}")
      else:
        # Create dedicated video timeline signal
        if hasattr(self.device, 'signal_t'):
          self.timeline_signal = self.device.signal_t(value=0, timeline_for_device=self.device)
          self.timeline_value = 1
          print(f"✅ Created dedicated video timeline signal")
        else:
          print(f"⚠️  No signal support available")
          
    except Exception as e:
      print(f"⚠️  Signal initialization failed: {e}")

  def submit_decode(self, decode_id:int, timeout_ms:float=5000.0) -> DecodeSync:
    """Submit decode operation for synchronization tracking"""
    with self._lock:
      if self.timeline_signal is None:
        # Create mock sync for testing
        sync = DecodeSync(
          decode_id=decode_id,
          signal=None,
          target_value=0,
          submitted_time=time.time(),
          timeout_ms=timeout_ms
        )
        sync.state = SyncState.SUBMITTED
        self.active_decodes[decode_id] = sync
        self.stats['total_syncs'] += 1
        return sync
        
      # Create signal for this decode operation
      target_value = self._next_timeline_value()
      
      sync = DecodeSync(
        decode_id=decode_id,
        signal=self.timeline_signal,
        target_value=target_value,
        submitted_time=time.time(),
        timeout_ms=timeout_ms
      )
      
      sync.state = SyncState.SUBMITTED
      self.active_decodes[decode_id] = sync
      self.stats['total_syncs'] += 1
      
      print(f"🎬 Decode sync submitted: id={decode_id}, target={target_value}")
      return sync

  def wait_for_decode(self, decode_id:int, timeout_ms:Optional[float]=None) -> bool:
    """Wait for specific decode operation to complete"""
    with self._lock:
      if decode_id not in self.active_decodes:
        print(f"⚠️  Decode ID {decode_id} not found")
        return False
        
      sync = self.active_decodes[decode_id]
      
    # Release lock for waiting
    return self._wait_for_sync(sync, timeout_ms)

  def _wait_for_sync(self, sync:DecodeSync, timeout_ms:Optional[float]=None) -> bool:
    """Wait for synchronization object to complete"""
    timeout = timeout_ms or sync.timeout_ms
    start_time = time.time()
    
    try:
      if sync.signal is None:
        # Mock wait for testing
        time.sleep(0.001)  # 1ms mock decode time
        self._complete_sync(sync, SyncState.COMPLETED)
        return True
        
      # Use HCQ signal waiting
      if hasattr(sync.signal, 'wait'):
        # Set timeout and wait
        end_time = start_time + (timeout / 1000.0)
        
        while time.time() < end_time:
          if sync.signal.value >= sync.target_value:
            self._complete_sync(sync, SyncState.COMPLETED)
            return True
          time.sleep(0.001)  # 1ms polling interval
          
        # Timeout occurred
        self._complete_sync(sync, SyncState.TIMEOUT)
        return False
      else:
        # Signal doesn't support waiting, mark as completed
        self._complete_sync(sync, SyncState.COMPLETED)
        return True
        
    except Exception as e:
      print(f"❌ Sync wait failed: {e}")
      self._complete_sync(sync, SyncState.FAILED)
      return False

  def _complete_sync(self, sync:DecodeSync, state:SyncState):
    """Mark synchronization as completed with given state"""
    with self._lock:
      sync.state = state
      sync.completion_time = time.time()
      
      # Update statistics
      if state == SyncState.COMPLETED:
        self.stats['completed_syncs'] += 1
      elif state == SyncState.FAILED:
        self.stats['failed_syncs'] += 1
      elif state == SyncState.TIMEOUT:
        self.stats['timeout_syncs'] += 1
        
      # Update average completion time
      if state == SyncState.COMPLETED:
        total_completed = self.stats['completed_syncs']
        old_avg = self.stats['avg_completion_ms']
        new_time = sync.elapsed_ms
        self.stats['avg_completion_ms'] = ((old_avg * (total_completed - 1)) + new_time) / total_completed
      
      print(f"🎬 Decode sync {state.value}: id={sync.decode_id}, time={sync.elapsed_ms:.2f}ms")

  def signal_decode_complete(self, decode_id:int):
    """Signal that decode operation is complete (for manual signaling)"""
    with self._lock:
      if decode_id not in self.active_decodes:
        return
        
      sync = self.active_decodes[decode_id]
      if sync.signal and hasattr(sync.signal, 'value'):
        # Update signal value to trigger completion
        sync.signal.value = sync.target_value
        print(f"🎬 Signaled decode complete: id={decode_id}")

  def wait_all_decodes(self, timeout_ms:float=10000.0) -> bool:
    """Wait for all active decode operations to complete"""
    start_time = time.time()
    
    while True:
      with self._lock:
        active_syncs = [s for s in self.active_decodes.values() if not s.is_completed]
        
      if not active_syncs:
        print(f"✅ All decodes completed")
        return True
        
      if (time.time() - start_time) * 1000 > timeout_ms:
        print(f"⚠️  Timeout waiting for all decodes")
        return False
        
      time.sleep(0.001)  # 1ms polling

  def cleanup_completed(self):
    """Remove completed decode synchronizations"""
    with self._lock:
      completed = [decode_id for decode_id, sync in self.active_decodes.items() if sync.is_completed]
      for decode_id in completed:
        del self.active_decodes[decode_id]
        
      if completed:
        print(f"🧹 Cleaned up {len(completed)} completed syncs")

  def get_stream_signal(self, stream_id:str):
    """Get timeline signal for specific stream"""
    with self._lock:
      if stream_id not in self.stream_signals:
        try:
          if hasattr(self.device, 'signal_t'):
            signal = self.device.signal_t(value=0, timeline_for_device=self.device)
            self.stream_signals[stream_id] = signal
            self.stream_timelines[stream_id] = 1
            print(f"✅ Created stream signal: {stream_id}")
          else:
            self.stream_signals[stream_id] = None
            self.stream_timelines[stream_id] = 0
        except Exception as e:
          print(f"⚠️  Stream signal creation failed: {e}")
          self.stream_signals[stream_id] = None
          self.stream_timelines[stream_id] = 0
          
      return self.stream_signals[stream_id]

  def synchronize_streams(self, stream_ids:List[str], timeout_ms:float=5000.0) -> bool:
    """Synchronize multiple video streams"""
    start_time = time.time()
    
    # Get all stream signals
    stream_signals = []
    for stream_id in stream_ids:
      signal = self.get_stream_signal(stream_id)
      if signal:
        stream_signals.append((stream_id, signal, self.stream_timelines.get(stream_id, 0)))
    
    # Wait for all streams to reach their timeline values
    while (time.time() - start_time) * 1000 < timeout_ms:
      all_ready = True
      for stream_id, signal, target_value in stream_signals:
        if hasattr(signal, 'value') and signal.value < target_value:
          all_ready = False
          break
          
      if all_ready:
        print(f"✅ Stream synchronization complete: {stream_ids}")
        return True
        
      time.sleep(0.001)
      
    print(f"⚠️  Stream synchronization timeout: {stream_ids}")
    return False

  def _next_timeline_value(self) -> int:
    """Get next timeline value for synchronization"""
    self.timeline_value += 1
    return self.timeline_value

  def get_stats(self) -> Dict[str, Any]:
    """Get synchronization statistics"""
    with self._lock:
      current_stats = dict(self.stats)
      current_stats.update({
        'active_decodes': len(self.active_decodes),
        'success_rate': (self.stats['completed_syncs'] / max(self.stats['total_syncs'], 1)) * 100,
        'active_streams': len(self.stream_signals)
      })
      return current_stats

  def destroy(self):
    """Cleanup synchronization resources"""
    with self._lock:
      print(f"🧹 Destroying VideoSyncManager...")
      
      # Wait for active operations with short timeout
      if self.active_decodes:
        print(f"⏳ Waiting for {len(self.active_decodes)} active decodes...")
        self.wait_all_decodes(timeout_ms=1000.0)
      
      # Cleanup resources
      self.active_decodes.clear()
      self.stream_signals.clear()
      self.stream_timelines.clear()
      
      print(f"✅ VideoSyncManager destroyed")

# Utility functions for synchronization
def create_decode_barrier(sync_manager:VideoSyncManager, decode_ids:List[int], timeout_ms:float=5000.0) -> bool:
  """Create synchronization barrier for multiple decode operations"""
  start_time = time.time()
  
  while (time.time() - start_time) * 1000 < timeout_ms:
    completed = 0
    for decode_id in decode_ids:
      if decode_id in sync_manager.active_decodes:
        if sync_manager.active_decodes[decode_id].is_completed:
          completed += 1
      else:
        completed += 1  # Not tracked = already completed
        
    if completed == len(decode_ids):
      return True
      
    time.sleep(0.001)
    
  return False

def estimate_decode_time(width:int, height:int, codec:str="HEVC") -> float:
  """Estimate decode time for synchronization planning"""
  # Simple estimation based on resolution and codec
  pixels = width * height
  
  base_times = {
    "HEVC": 0.5,  # 0.5ms per megapixel
    "H264": 0.3,  # 0.3ms per megapixel  
    "AV1": 1.0    # 1.0ms per megapixel
  }
  
  base_time = base_times.get(codec, 0.5)
  return (pixels / 1_000_000) * base_time  # milliseconds

# Export main classes and functions
__all__ = [
  'VideoSyncManager', 'DecodeSync', 'SyncState',
  'create_decode_barrier', 'estimate_decode_time'
] 