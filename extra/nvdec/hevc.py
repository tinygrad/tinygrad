from enum import IntEnum

# H.265 specification
# https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-H.265-201802-S!!PDF-E&type=items

NAL_UNIT_START_CODE = b"\x00\x00\x01"
NAL_UNIT_START_CODE_SIZE = len(NAL_UNIT_START_CODE)
NAL_UNIT_HEADER_SIZE = 2

class HevcNalUnitType(IntEnum):
  TRAIL_N = 0         # RBSP structure: slice_segment_layer_rbsp( )
  TRAIL_R = 1         # RBSP structure: slice_segment_layer_rbsp( )
  TSA_N = 2           # RBSP structure: slice_segment_layer_rbsp( )
  TSA_R = 3           # RBSP structure: slice_segment_layer_rbsp( )
  STSA_N = 4          # RBSP structure: slice_segment_layer_rbsp( )
  STSA_R = 5          # RBSP structure: slice_segment_layer_rbsp( )
  RADL_N = 6          # RBSP structure: slice_segment_layer_rbsp( )
  RADL_R = 7          # RBSP structure: slice_segment_layer_rbsp( )
  RASL_N = 8          # RBSP structure: slice_segment_layer_rbsp( )
  RASL_R = 9          # RBSP structure: slice_segment_layer_rbsp( )
  RSV_VCL_N10 = 10
  RSV_VCL_R11 = 11
  RSV_VCL_N12 = 12
  RSV_VCL_R13 = 13
  RSV_VCL_N14 = 14
  RSV_VCL_R15 = 15
  BLA_W_LP = 16       # RBSP structure: slice_segment_layer_rbsp( )
  BLA_W_RADL = 17     # RBSP structure: slice_segment_layer_rbsp( )
  BLA_N_LP = 18       # RBSP structure: slice_segment_layer_rbsp( )
  IDR_W_RADL = 19     # RBSP structure: slice_segment_layer_rbsp( )
  IDR_N_LP = 20       # RBSP structure: slice_segment_layer_rbsp( )
  CRA_NUT = 21        # RBSP structure: slice_segment_layer_rbsp( )
  RSV_IRAP_VCL22 = 22
  RSV_IRAP_VCL23 = 23
  RSV_VCL24 = 24
  RSV_VCL25 = 25
  RSV_VCL26 = 26
  RSV_VCL27 = 27
  RSV_VCL28 = 28
  RSV_VCL29 = 29
  RSV_VCL30 = 30
  RSV_VCL31 = 31
  VPS_NUT = 32        # RBSP structure: video_parameter_set_rbsp( )
  SPS_NUT = 33        # RBSP structure: seq_parameter_set_rbsp( )
  PPS_NUT = 34        # RBSP structure: pic_parameter_set_rbsp( )
  AUD_NUT = 35
  EOS_NUT = 36
  EOB_NUT = 37
  FD_NUT = 38
  PREFIX_SEI_NUT = 39
  SUFFIX_SEI_NUT = 40
  RSV_NVCL41 = 41
  RSV_NVCL42 = 42
  RSV_NVCL43 = 43
  RSV_NVCL44 = 44
  RSV_NVCL45 = 45
  RSV_NVCL46 = 46
  RSV_NVCL47 = 47
  UNSPEC48 = 48
  UNSPEC49 = 49
  UNSPEC50 = 50
  UNSPEC51 = 51
  UNSPEC52 = 52
  UNSPEC53 = 53
  UNSPEC54 = 54
  UNSPEC55 = 55
  UNSPEC56 = 56
  UNSPEC57 = 57
  UNSPEC58 = 58
  UNSPEC59 = 59
  UNSPEC60 = 60
  UNSPEC61 = 61
  UNSPEC62 = 62
  UNSPEC63 = 63

# B.2.2 Byte stream NAL unit semantics
# - The nal_unit_type within the nal_unit( ) syntax structure is equal to VPS_NUT, SPS_NUT or PPS_NUT.
# - The byte stream NAL unit syntax structure contains the first NAL unit of an access unit in decoding
#   order, as specified in clause 7.4.2.4.4.
HEVC_PARAMETER_SET_NAL_UNITS = (
  HevcNalUnitType.VPS_NUT,
  HevcNalUnitType.SPS_NUT,
  HevcNalUnitType.PPS_NUT,
)

# 3.29 coded slice segment NAL unit: A NAL unit that has nal_unit_type in the range of TRAIL_N to RASL_R,
# inclusive, or in the range of BLA_W_LP to RSV_IRAP_VCL23, inclusive, which indicates that the NAL unit
# contains a coded slice segment
HEVC_CODED_SLICE_SEGMENT_NAL_UNITS = (
  HevcNalUnitType.TRAIL_N,
  HevcNalUnitType.TRAIL_R,
  HevcNalUnitType.TSA_N,
  HevcNalUnitType.TSA_R,
  HevcNalUnitType.STSA_N,
  HevcNalUnitType.STSA_R,
  HevcNalUnitType.RADL_N,
  HevcNalUnitType.RADL_R,
  HevcNalUnitType.RASL_N,
  HevcNalUnitType.RASL_R,
  HevcNalUnitType.BLA_W_LP,
  HevcNalUnitType.BLA_W_RADL,
  HevcNalUnitType.BLA_N_LP,
  HevcNalUnitType.IDR_W_RADL,
  HevcNalUnitType.IDR_N_LP,
  HevcNalUnitType.CRA_NUT,
)

class BitReader:
  def __init__(self, data:bytes): self.reader, self.current_bits, self.bits, self.read_bits, self.total = iter(data), 0, 0, 0, len(data) * 8
  def empty(self): return self.read_bits == self.total and self.current_bits == 0
  def peak_bits(self, n):
    while self.current_bits < n:
      self.bits = (self.bits << 8) | next(self.reader)
      self.current_bits += 8
      self.read_bits += 8
    return (self.bits >> (self.current_bits - n)) & ((1 << n) - 1)
  def _next_bits(self, n):
    val = self.peak_bits(n)
    self.bits &= (1 << (self.current_bits - n)) - 1
    self.current_bits -= n
    return val
  def u(self, n): return self._next_bits(n)
  def ue_v(self):
    leading_zero_bits = -1
    bits = []
    while True:
      bit = self.u(1)
      bits.append(bit)
      leading_zero_bits += 1
      if bit == 1: break

    part = self.u(leading_zero_bits)

    if leading_zero_bits == 0: return 0
    return (1 << leading_zero_bits) - 1 + part
  def se_v(self):
    k = self.ue_v()
    return (-1 ** (k + 1)) * (k // 2)