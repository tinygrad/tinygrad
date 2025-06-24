"""
HEVC (H.265) bitstream parser for video decode support
Follows tinygrad conventions: minimal, readable, focused
"""
import struct
from typing import NamedTuple, Optional, Iterator
from dataclasses import dataclass

# HEVC NAL unit types
class HEVCNalType:
  TRAIL_N = 0    # Coded slice segment of a non-TSA, non-STSA trailing picture
  TRAIL_R = 1    # Coded slice segment of a non-TSA, non-STSA trailing picture
  TSA_N = 2      # Coded slice segment of a TSA picture
  TSA_R = 3      # Coded slice segment of a TSA picture
  STSA_N = 4     # Coded slice segment of an STSA picture
  STSA_R = 5     # Coded slice segment of an STSA picture
  RADL_N = 6     # Coded slice segment of a RADL picture
  RADL_R = 7     # Coded slice segment of a RADL picture
  RASL_N = 8     # Coded slice segment of a RASL picture
  RASL_R = 9     # Coded slice segment of a RASL picture
  BLA_W_LP = 16  # Coded slice segment of a BLA picture
  BLA_W_RADL = 17 # Coded slice segment of a BLA picture
  BLA_N_LP = 18  # Coded slice segment of a BLA picture
  IDR_W_RADL = 19 # Coded slice segment of an IDR picture
  IDR_N_LP = 20  # Coded slice segment of an IDR picture
  CRA_NUT = 21   # Coded slice segment of a CRA picture
  VPS_NUT = 32   # Video parameter set
  SPS_NUT = 33   # Sequence parameter set
  PPS_NUT = 34   # Picture parameter set
  AUD_NUT = 35   # Access unit delimiter
  EOS_NUT = 36   # End of sequence
  EOB_NUT = 37   # End of bitstream
  FD_NUT = 38    # Filler data
  PREFIX_SEI_NUT = 39  # Supplemental enhancement information
  SUFFIX_SEI_NUT = 40  # Supplemental enhancement information

@dataclass
class HEVCNalUnit:
  """HEVC NAL unit structure"""
  nal_type: int
  layer_id: int
  temporal_id: int
  data: bytes
  
  @property
  def is_vcl(self) -> bool:
    """Check if this is a Video Coding Layer (VCL) NAL unit"""
    return self.nal_type <= 31
  
  @property
  def is_irap(self) -> bool:
    """Check if this is an Intra Random Access Point (IRAP) picture"""
    return 16 <= self.nal_type <= 23

@dataclass
class HEVCVPS:
  """Video Parameter Set (simplified)"""
  vps_id: int
  max_layers: int
  max_sub_layers: int
  temporal_id_nesting_flag: bool

@dataclass
class HEVCSPS:
  """Sequence Parameter Set (simplified)"""
  sps_id: int
  vps_id: int
  max_sub_layers: int
  temporal_id_nesting_flag: bool
  profile_tier_level: dict
  chroma_format_idc: int
  pic_width_in_luma_samples: int
  pic_height_in_luma_samples: int
  conformance_window_flag: bool
  bit_depth_luma: int
  bit_depth_chroma: int

@dataclass
class HEVCPPS:
  """Picture Parameter Set (simplified)"""
  pps_id: int
  sps_id: int
  dependent_slice_segments_enabled_flag: bool
  output_flag_present_flag: bool
  num_extra_slice_header_bits: int

class HEVCBitstreamReader:
  """Bitstream reader with emulation prevention byte handling"""
  def __init__(self, data: bytes):
    self.data = self._remove_emulation_prevention(data)
    self.pos = 0
    self.bit_pos = 0

  def _remove_emulation_prevention(self, data: bytes) -> bytes:
    """Remove emulation prevention bytes (0x03 after 0x00 0x00)"""
    result = bytearray()
    i = 0
    while i < len(data):
      if i < len(data) - 2 and data[i:i+3] == b'\x00\x00\x03':
        result.extend(data[i:i+2])  # Add 0x00 0x00, skip 0x03
        i += 3
      else:
        result.append(data[i])
        i += 1
    return bytes(result)

  def read_bits(self, n: int) -> int:
    """Read n bits from bitstream"""
    if n == 0: return 0
    
    result = 0
    bits_read = 0
    
    while bits_read < n:
      if self.pos >= len(self.data):
        raise ValueError("Unexpected end of bitstream")
      
      byte = self.data[self.pos]
      available_bits = 8 - self.bit_pos
      needed_bits = n - bits_read
      
      if needed_bits >= available_bits:
        # Read all remaining bits from current byte
        mask = (1 << available_bits) - 1
        bits = (byte >> (8 - self.bit_pos - available_bits)) & mask
        result = (result << available_bits) | bits
        bits_read += available_bits
        self.pos += 1
        self.bit_pos = 0
      else:
        # Read only needed bits from current byte
        shift = available_bits - needed_bits
        mask = (1 << needed_bits) - 1
        bits = (byte >> shift) & mask
        result = (result << needed_bits) | bits
        bits_read += needed_bits
        self.bit_pos += needed_bits
    
    return result

  def read_ue(self) -> int:
    """Read unsigned exponential Golomb code"""
    leading_zeros = 0
    while self.read_bits(1) == 0:
      leading_zeros += 1
    
    if leading_zeros == 0:
      return 0
    
    return (1 << leading_zeros) - 1 + self.read_bits(leading_zeros)

  def read_se(self) -> int:
    """Read signed exponential Golomb code"""
    ue = self.read_ue()
    return (ue + 1) // 2 if ue % 2 == 1 else -(ue // 2)

class HEVCParser:
  """HEVC bitstream parser for decode pipeline"""
  def __init__(self):
    self.vps_list: dict[int, HEVCVPS] = {}
    self.sps_list: dict[int, HEVCSPS] = {}
    self.pps_list: dict[int, HEVCPPS] = {}

  def find_nal_units(self, data: bytes) -> Iterator[HEVCNalUnit]:
    """Find and parse NAL units in bitstream"""
    start_codes = []
    
    # Find all start codes (0x000001 or 0x00000001)
    i = 0
    while i < len(data) - 3:
      if data[i:i+3] == b'\x00\x00\x01':
        start_codes.append(i + 3)
        i += 3
      elif i < len(data) - 4 and data[i:i+4] == b'\x00\x00\x00\x01':
        start_codes.append(i + 4)
        i += 4
      else:
        i += 1
    
    # Extract NAL units between start codes
    for i in range(len(start_codes)):
      start = start_codes[i]
      end = start_codes[i + 1] if i + 1 < len(start_codes) else len(data)
      
      if start < len(data):
        nal_header = data[start]
        nal_type = (nal_header >> 1) & 0x3F
        layer_id = ((data[start] & 0x01) << 5) | ((data[start + 1] >> 3) & 0x1F)
        temporal_id = (data[start + 1] & 0x07) - 1
        
        yield HEVCNalUnit(
          nal_type=nal_type,
          layer_id=layer_id, 
          temporal_id=temporal_id,
          data=data[start:end]
        )

  def parse_vps(self, nal_data: bytes) -> HEVCVPS:
    """Parse Video Parameter Set"""
    reader = HEVCBitstreamReader(nal_data[2:])  # Skip NAL header
    
    vps_id = reader.read_bits(4)
    reader.read_bits(2)  # vps_base_layer_internal_flag, vps_base_layer_available_flag
    max_layers = reader.read_bits(6) + 1
    max_sub_layers = reader.read_bits(3) + 1
    temporal_id_nesting_flag = reader.read_bits(1) == 1
    
    vps = HEVCVPS(vps_id, max_layers, max_sub_layers, temporal_id_nesting_flag)
    self.vps_list[vps_id] = vps
    return vps

  def parse_sps(self, nal_data: bytes) -> HEVCSPS:
    """Parse Sequence Parameter Set"""
    reader = HEVCBitstreamReader(nal_data[2:])  # Skip NAL header
    
    vps_id = reader.read_bits(4)
    max_sub_layers = reader.read_bits(3) + 1
    temporal_id_nesting_flag = reader.read_bits(1) == 1
    
    # Parse profile_tier_level (simplified)
    profile_tier_level = {}
    profile_tier_level['general_profile_space'] = reader.read_bits(2)
    profile_tier_level['general_tier_flag'] = reader.read_bits(1)
    profile_tier_level['general_profile_idc'] = reader.read_bits(5)
    
    # Skip profile compatibility flags
    reader.read_bits(32)
    
    # Skip constraint flags
    reader.read_bits(48)
    
    profile_tier_level['general_level_idc'] = reader.read_bits(8)
    
    sps_id = reader.read_ue()
    chroma_format_idc = reader.read_ue()
    
    if chroma_format_idc == 3:
      reader.read_bits(1)  # separate_colour_plane_flag
    
    pic_width_in_luma_samples = reader.read_ue()
    pic_height_in_luma_samples = reader.read_ue()
    
    conformance_window_flag = reader.read_bits(1) == 1
    if conformance_window_flag:
      reader.read_ue()  # conf_win_left_offset
      reader.read_ue()  # conf_win_right_offset
      reader.read_ue()  # conf_win_top_offset
      reader.read_ue()  # conf_win_bottom_offset
    
    bit_depth_luma = reader.read_ue() + 8
    bit_depth_chroma = reader.read_ue() + 8
    
    sps = HEVCSPS(sps_id, vps_id, max_sub_layers, temporal_id_nesting_flag,
                  profile_tier_level, chroma_format_idc, pic_width_in_luma_samples,
                  pic_height_in_luma_samples, conformance_window_flag, 
                  bit_depth_luma, bit_depth_chroma)
    
    self.sps_list[sps_id] = sps
    return sps

  def parse_pps(self, nal_data: bytes) -> HEVCPPS:
    """Parse Picture Parameter Set"""
    reader = HEVCBitstreamReader(nal_data[2:])  # Skip NAL header
    
    pps_id = reader.read_ue()
    sps_id = reader.read_ue()
    dependent_slice_segments_enabled_flag = reader.read_bits(1) == 1
    output_flag_present_flag = reader.read_bits(1) == 1
    num_extra_slice_header_bits = reader.read_bits(3)
    
    pps = HEVCPPS(pps_id, sps_id, dependent_slice_segments_enabled_flag,
                  output_flag_present_flag, num_extra_slice_header_bits)
    
    self.pps_list[pps_id] = pps
    return pps

  def parse_frame(self, data: bytes) -> dict:
    """Parse HEVC frame and return decode information"""
    frame_info = {
      'nal_units': [],
      'is_keyframe': False,
      'width': 0,
      'height': 0,
      'profile': 'unknown',
      'level': 0,
      'chroma_format': 'unknown',
      'bit_depth': 8
    }
    
    for nal_unit in self.find_nal_units(data):
      frame_info['nal_units'].append(nal_unit)
      
      if nal_unit.nal_type == HEVCNalType.VPS_NUT:
        self.parse_vps(nal_unit.data)
      elif nal_unit.nal_type == HEVCNalType.SPS_NUT:
        sps = self.parse_sps(nal_unit.data)
        frame_info['width'] = sps.pic_width_in_luma_samples
        frame_info['height'] = sps.pic_height_in_luma_samples
        frame_info['bit_depth'] = sps.bit_depth_luma
        frame_info['chroma_format'] = {0: 'mono', 1: 'yuv420', 2: 'yuv422', 3: 'yuv444'}.get(sps.chroma_format_idc, 'unknown')
        frame_info['level'] = sps.profile_tier_level.get('general_level_idc', 0)
      elif nal_unit.nal_type == HEVCNalType.PPS_NUT:
        self.parse_pps(nal_unit.data)
      elif nal_unit.is_irap:
        frame_info['is_keyframe'] = True
    
    return frame_info

  def validate_stream(self, data: bytes) -> bool:
    """Validate HEVC bitstream format"""
    try:
      frame_info = self.parse_frame(data)
      return len(frame_info['nal_units']) > 0 and frame_info['width'] > 0 and frame_info['height'] > 0
    except Exception:
      return False

# Helper functions for decode pipeline
def extract_parameter_sets(data: bytes) -> tuple[bytes, bytes, bytes]:
  """Extract VPS, SPS, PPS from HEVC bitstream"""
  parser = HEVCParser()
  vps_data, sps_data, pps_data = b'', b'', b''
  
  for nal_unit in parser.find_nal_units(data):
    if nal_unit.nal_type == HEVCNalType.VPS_NUT:
      vps_data = nal_unit.data
    elif nal_unit.nal_type == HEVCNalType.SPS_NUT:
      sps_data = nal_unit.data
    elif nal_unit.nal_type == HEVCNalType.PPS_NUT:
      pps_data = nal_unit.data
  
  return vps_data, sps_data, pps_data

def get_frame_dimensions(data: bytes) -> tuple[int, int]:
  """Extract frame dimensions from HEVC SPS"""
  parser = HEVCParser()
  frame_info = parser.parse_frame(data)
  return frame_info['width'], frame_info['height']

def is_keyframe(data: bytes) -> bool:
  """Check if HEVC frame is a keyframe (IRAP)"""
  parser = HEVCParser()
  frame_info = parser.parse_frame(data)
  return frame_info['is_keyframe'] 