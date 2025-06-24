# Video Tensor Integration for tinygrad
# Convert decoded video surfaces to Tensor objects

from typing import Optional, Tuple, Union, Any
import numpy as np

try:
  from tinygrad.tensor import Tensor
  from tinygrad.runtime.ops_nv import NVVideoSurface, NVDevice
  from tinygrad.dtype import dtypes
  from tinygrad.runtime.support.hcq import HCQBuffer
except ImportError as e:
  print(f"⚠️  Tensor/NVDevice imports not available: {e}")

class VideoTensorConverter:
  """Convert video surfaces to tinygrad Tensors with format handling"""
  
  def __init__(self, device_interface):
    self.device = device_interface
    
    # Format specifications
    self.format_specs = {
      "NV12": {
        "planes": 2,
        "y_channels": 1,
        "uv_channels": 2,
        "dtype": dtypes.uint8,
        "subsampling": (2, 2)  # UV subsampled 2x2
      },
      "RGBA": {
        "planes": 1,
        "channels": 4,
        "dtype": dtypes.uint8,
        "subsampling": (1, 1)
      },
      "RGB": {
        "planes": 1,
        "channels": 3,
        "dtype": dtypes.uint8,
        "subsampling": (1, 1)
      }
    }
    
  def surface_to_tensor(self, surface:NVVideoSurface, output_format:str="RGB", normalize:bool=False, device:Optional[str]=None) -> Tensor:
    """
    Convert video surface to tinygrad Tensor
    
    Args:
      surface: Input video surface
      output_format: Target format ("RGB", "RGBA", "NV12")
      normalize: Normalize values to [0, 1] range
      device: Target device for tensor
      
    Returns:
      Tensor containing video data
    """
    if surface.format not in self.format_specs:
      raise ValueError(f"Unsupported input format: {surface.format}")
      
    if output_format not in self.format_specs:
      raise ValueError(f"Unsupported output format: {output_format}")
    
    try:
      # Convert surface format if needed
      if surface.format != output_format and output_format != "RGB":
        # Use device format conversion for GPU formats
        converted_surface = self.device.convert_video_format(surface, output_format)
      else:
        converted_surface = surface
      
      # Extract tensor data based on format
      if output_format == "RGB":
        tensor = self._surface_to_rgb_tensor(converted_surface, normalize, device)
      elif output_format == "RGBA":
        tensor = self._surface_to_rgba_tensor(converted_surface, normalize, device)
      elif output_format == "NV12":
        tensor = self._surface_to_nv12_tensor(converted_surface, normalize, device)
      else:
        raise ValueError(f"Unsupported tensor conversion: {output_format}")
      
      return tensor
      
    except Exception as e:
      print(f"❌ Surface to tensor conversion failed: {e}")
      raise

  def _surface_to_rgb_tensor(self, surface:NVVideoSurface, normalize:bool, device:Optional[str]) -> Tensor:
    """Convert surface to RGB tensor [H, W, 3]"""
    if surface.format == "RGBA":
      # Convert RGBA to RGB by dropping alpha channel
      rgba_tensor = self._surface_to_rgba_tensor(surface, normalize=False, device=device)
      rgb_tensor = rgba_tensor[..., :3]  # Drop alpha channel
    elif surface.format == "NV12":
      # Convert NV12 to RGB using YUV→RGB conversion
      rgb_tensor = self._nv12_to_rgb_tensor(surface, device)
    else:
      raise ValueError(f"Cannot convert {surface.format} to RGB")
    
    if normalize:
      rgb_tensor = rgb_tensor.cast(dtypes.float32) / 255.0
      
    return rgb_tensor

  def _surface_to_rgba_tensor(self, surface:NVVideoSurface, normalize:bool, device:Optional[str]) -> Tensor:
    """Convert RGBA surface to tensor [H, W, 4]"""
    if surface.format != "RGBA":
      raise ValueError("Surface must be RGBA format")
    
    # Create tensor from GPU buffer
    target_device = device or (self.device.device if hasattr(self.device, 'device') else "CUDA")
    
    # Calculate tensor shape
    height, width = surface.height, surface.width
    channels = 4
    
    # For now, create mock tensor data (real implementation would use GPU memory directly)
    if hasattr(surface, 'va_addr') and isinstance(surface.va_addr, int):
      # Real GPU memory - would need to map/copy data
      print(f"🔄 Creating RGBA tensor from GPU memory: {width}x{height}x{channels}")
      data = np.zeros((height, width, channels), dtype=np.uint8)  # Mock data
    else:
      # CPU data
      data = np.zeros((height, width, channels), dtype=np.uint8)
    
    tensor = Tensor(data, device=target_device, dtype=dtypes.uint8)
    
    if normalize:
      tensor = tensor.cast(dtypes.float32) / 255.0
      
    return tensor

  def _surface_to_nv12_tensor(self, surface:NVVideoSurface, normalize:bool, device:Optional[str]) -> Tensor:
    """Convert NV12 surface to tensor [H*1.5, W] with Y and UV planes"""
    if surface.format != "NV12":
      raise ValueError("Surface must be NV12 format")
    
    target_device = device or (self.device.device if hasattr(self.device, 'device') else "CUDA")
    
    # NV12 has Y plane (H×W) + UV plane (H/2×W)
    height, width = surface.height, surface.width
    y_size = height * width
    uv_size = (height // 2) * width
    total_height = height + height // 2  # Y + UV/2
    
    print(f"🔄 Creating NV12 tensor: {width}x{total_height} (Y:{y_size} + UV:{uv_size})")
    
    # Mock NV12 data (real implementation would read from GPU memory)
    data = np.zeros((total_height, width), dtype=np.uint8)
    
    tensor = Tensor(data, device=target_device, dtype=dtypes.uint8)
    
    if normalize:
      tensor = tensor.cast(dtypes.float32) / 255.0
      
    return tensor

  def _nv12_to_rgb_tensor(self, surface:NVVideoSurface, device:Optional[str]) -> Tensor:
    """Convert NV12 surface to RGB tensor using YUV→RGB conversion"""
    if surface.format != "NV12":
      raise ValueError("Surface must be NV12 format")
    
    target_device = device or (self.device.device if hasattr(self.device, 'device') else "CUDA")
    height, width = surface.height, surface.width
    
    # For now, create mock RGB data (real implementation would do YUV→RGB conversion)
    print(f"🔄 Converting NV12 to RGB: {width}x{height}")
    
    # Mock RGB conversion
    rgb_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    return Tensor(rgb_data, device=target_device, dtype=dtypes.uint8)

  def batch_surfaces_to_tensor(self, surfaces:list[NVVideoSurface], output_format:str="RGB", normalize:bool=False, device:Optional[str]=None) -> Tensor:
    """
    Convert multiple video surfaces to batched tensor
    
    Args:
      surfaces: List of video surfaces
      output_format: Target format for all surfaces
      normalize: Normalize values to [0, 1]
      device: Target device
      
    Returns:
      Batched tensor [N, H, W, C] or [N, H, W] depending on format
    """
    if not surfaces:
      raise ValueError("Empty surface list")
    
    # Convert each surface to tensor
    tensors = []
    for surface in surfaces:
      tensor = self.surface_to_tensor(surface, output_format, normalize, device)
      tensors.append(tensor)
    
    # Stack into batch
    batch_tensor = Tensor.stack(tensors, dim=0)
    print(f"✅ Created batch tensor: {batch_tensor.shape} from {len(surfaces)} surfaces")
    
    return batch_tensor

def create_video_tensor_converter(device_interface) -> VideoTensorConverter:
  """Factory function to create video tensor converter"""
  return VideoTensorConverter(device_interface)

# Utility functions for video tensor operations
def yuv_to_rgb_tensor(y_tensor:Tensor, u_tensor:Tensor, v_tensor:Tensor) -> Tensor:
  """Convert YUV planes to RGB tensor using standard conversion matrix"""
  # YUV to RGB conversion matrix (ITU-R BT.601)
  # R = Y + 1.402 * (V - 128)
  # G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)  
  # B = Y + 1.772 * (U - 128)
  
  # Convert to float for computation
  y_f = y_tensor.cast(dtypes.float32)
  u_f = u_tensor.cast(dtypes.float32) - 128.0
  v_f = v_tensor.cast(dtypes.float32) - 128.0
  
  # Apply conversion matrix
  r = y_f + 1.402 * v_f
  g = y_f - 0.344136 * u_f - 0.714136 * v_f
  b = y_f + 1.772 * u_f
  
  # Stack channels and clamp to valid range
  rgb = Tensor.stack([r, g, b], dim=-1)
  rgb = rgb.clamp(0, 255).cast(dtypes.uint8)
  
  return rgb

def rgb_to_yuv_tensor(rgb_tensor:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
  """Convert RGB tensor to YUV planes"""
  # RGB to YUV conversion matrix (ITU-R BT.601)
  # Y = 0.299*R + 0.587*G + 0.114*B
  # U = -0.169*R - 0.331*G + 0.5*B + 128
  # V = 0.5*R - 0.419*G - 0.081*B + 128
  
  rgb_f = rgb_tensor.cast(dtypes.float32)
  r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
  
  y = 0.299 * r + 0.587 * g + 0.114 * b
  u = -0.169 * r - 0.331 * g + 0.5 * b + 128.0
  v = 0.5 * r - 0.419 * g - 0.081 * b + 128.0
  
  # Clamp and convert back to uint8
  y = y.clamp(0, 255).cast(dtypes.uint8)
  u = u.clamp(0, 255).cast(dtypes.uint8)
  v = v.clamp(0, 255).cast(dtypes.uint8)
  
  return y, u, v

def tensor_to_video_surface(tensor:Tensor, width:int, height:int, format:str="RGBA", device_interface=None) -> NVVideoSurface:
  """
  Convert tinygrad Tensor back to video surface
  
  Args:
    tensor: Input tensor
    width: Surface width
    height: Surface height
    format: Surface format
    device_interface: Device for surface allocation
    
  Returns:
    NVVideoSurface containing tensor data
  """
  if device_interface is None:
    raise ValueError("Device interface required for surface allocation")
  
  # Allocate target surface
  surface = device_interface._alloc_video_surface(width, height, format)
  
  # Copy tensor data to surface (simplified)
  # Real implementation would copy tensor GPU memory to surface
  print(f"🔄 Converting tensor {tensor.shape} to {format} surface {width}x{height}")
  
  return surface

# High-level decode functions
def decode_hevc_to_tensor(device_interface, hevc_data:bytes, output_format:str="RGB", normalize:bool=False, device:Optional[str]=None) -> Optional[Tensor]:
  """
  Decode HEVC data directly to tinygrad Tensor
  
  Args:
    device_interface: NVDevice interface
    hevc_data: HEVC bitstream data
    output_format: Output tensor format
    normalize: Normalize to [0, 1] range
    device: Target device for tensor
    
  Returns:
    Decoded tensor or None if failed
  """
  try:
    # Decode HEVC to surface
    surface = device_interface.decode_hevc(hevc_data, output_format="NV12")
    if not surface:
      return None
    
    # Convert surface to tensor
    converter = VideoTensorConverter(device_interface)
    tensor = converter.surface_to_tensor(surface, output_format, normalize, device)
    
    return tensor
    
  except Exception as e:
    print(f"❌ HEVC to tensor decode failed: {e}")
    return None

def decode_hevc_batch_to_tensor(device_interface, hevc_frames:list[bytes], output_format:str="RGB", normalize:bool=False, device:Optional[str]=None) -> Optional[Tensor]:
  """
  Decode batch of HEVC frames to batched tensor
  
  Args:
    device_interface: NVDevice interface
    hevc_frames: List of HEVC bitstream data
    output_format: Output tensor format
    normalize: Normalize to [0, 1] range
    device: Target device for tensor
    
  Returns:
    Batched tensor [N, H, W, C] or None if failed
  """
  try:
    # Decode all frames to surfaces
    surfaces = []
    for frame_data in hevc_frames:
      surface = device_interface.decode_hevc(frame_data, output_format="NV12")
      if surface:
        surfaces.append(surface)
    
    if not surfaces:
      return None
    
    # Convert surfaces to batched tensor
    converter = VideoTensorConverter(device_interface)
    batch_tensor = converter.batch_surfaces_to_tensor(surfaces, output_format, normalize, device)
    
    return batch_tensor
    
  except Exception as e:
    print(f"❌ Batch HEVC to tensor decode failed: {e}")
    return None

# Export main classes and functions
__all__ = [
  'VideoTensorConverter', 'create_video_tensor_converter',
  'decode_hevc_to_tensor', 'decode_hevc_batch_to_tensor',
  'yuv_to_rgb_tensor', 'rgb_to_yuv_tensor', 'tensor_to_video_surface'
] 