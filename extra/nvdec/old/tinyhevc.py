import argparse, pathlib
import os
import struct
from enum import IntEnum
from tinygrad.helpers import DEBUG

from tinygrad.runtime.ops_nv import NVDevice, NVCommandQueue, nv_gpu

from extra.nv_gpu_driver.nv_ioctl import _dump_gpfifo, dump_struct_ext

NAL_UNIT_START_CODE = b"\x00\x00\x01"
NAL_UNIT_START_CODE_SIZE = len(NAL_UNIT_START_CODE)
NAL_UNIT_HEADER_SIZE = 2

class VideoFileInvalid(Exception): pass

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

class HEVCReader:
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

class NVVidDecoder:
  def __init__(self, dev):
    self.desc = nv_gpu.nvdec_hevc_pic_s(gptimer_timeout_value=81600000, tileformat=1, sw_start_code_e=1)

  def stream_slice(self, nal_unit_type, data):
    reader = HEVCReader(data)
    if nal_unit_type == HevcNalUnitType.SPS_NUT:
      sps_video_parameter_set_id = reader.u(4)
      sps_max_sub_layers_minus1 = reader.u(3)
      sps_temporal_id_nesting_flag = reader.u(1)
      # 7.3.3 Profile, tier and level syntax
      if True: # profile parsing
        assert sps_max_sub_layers_minus1 == 0, "no sublayers supported"
        smth44 = reader.u(88)
        general_level_idc = reader.u(8)
      sps_seq_parameter_set_id = reader.ue_v()
      self.desc.chroma_format_idc = reader.ue_v()
      self.desc.pic_width_in_luma_samples = reader.ue_v()
      self.desc.pic_height_in_luma_samples = reader.ue_v()
      if conformance_window_flag := reader.u(1):
        conf_win_left_offset = reader.ue_v()
        conf_win_right_offset = reader.ue_v()
        conf_win_top_offset = reader.ue_v()
        conf_win_bottom_offset = reader.ue_v()
      self.desc.bit_depth_luma = reader.ue_v() + 8
      self.desc.bit_depth_chroma = reader.ue_v() + 8
      self.desc.ecdma_cfg.log2_max_pic_order_cnt_lsb_minus4 = reader.ue_v()
      sps_sub_layer_ordering_info_present_flag = reader.u(1)
      for i in range((0 if sps_sub_layer_ordering_info_present_flag else sps_max_sub_layers_minus1), sps_max_sub_layers_minus1 + 1):
        sps_max_dec_pic_buffering_minus1 = reader.ue_v()
        sps_max_num_reorder_pics = reader.ue_v()
        sps_max_latency_increase_plus1 = reader.ue_v()
      self.desc.log2_min_luma_coding_block_size = reader.ue_v() + 3
      self.desc.log2_max_luma_coding_block_size = self.desc.log2_min_luma_coding_block_size + reader.ue_v()
      self.desc.log2_min_transform_block_size = reader.ue_v() + 2
      self.desc.log2_max_transform_block_size = self.desc.log2_min_transform_block_size + reader.ue_v()
      max_transform_hierarchy_depth_inter = reader.ue_v()
      max_transform_hierarchy_depth_intra = reader.ue_v()
      if scaling_list_enabled_flag := reader.u(1):
        if sps_scaling_list_data_present_flag := reader.u(1): assert False, "scaling_list_data parsing not implemented"
      self.desc.amp_enabled_flag = reader.u(1)
      self.desc.sample_adaptive_offset_enabled_flag = reader.u(1)
      self.desc.pcm_enabled_flag = reader.u(1)
      assert self.desc.pcm_enabled_flag == 0, "pcm not implemented"
      num_short_term_ref_pic_sets = reader.ue_v()
      assert num_short_term_ref_pic_sets == 0, "ref pic sets parsing not implemented"
      if long_term_ref_pics_present_flag := reader.u(1): assert False, "long_term_ref_pics parsing not implemented"
      self.desc.sps_temporal_mvp_enabled_flag = reader.u(1)
      self.desc.strong_intra_smoothing_enabled_flag = reader.u(1)
      # not parsed any further
    elif nal_unit_type == HevcNalUnitType.PPS_NUT:
      pps_pic_parameter_set_id = reader.ue_v()
      pps_seq_parameter_set_id = reader.ue_v()
      dependent_slice_segments_enabled_flag = reader.u(1)
      output_flag_present_flag = reader.u(1)
      num_extra_slice_header_bits = reader.u(3)
      self.desc.sign_data_hiding_enabled_flag = reader.u(1)
      cabac_init_present_flag = reader.u(1)
      self.desc.num_ref_idx_l0_default_active = reader.ue_v() + 1
      self.desc.num_ref_idx_l1_default_active = reader.ue_v() + 1
      self.desc.init_qp = reader.se_v() + 26
      constrained_intra_pred_flag = reader.u(1)
      transform_skip_enabled_flag = reader.u(1)
      self.desc.cu_qp_delta_enabled_flag = reader.u(1)
      if self.desc.cu_qp_delta_enabled_flag: self.desc.diff_cu_qp_delta_depth = reader.ue_v()
      
      self.desc.pps_cb_qp_offset = reader.se_v()
      self.desc.pps_cr_qp_offset = reader.se_v()
      self.desc.pps_slice_chroma_qp_offsets_present_flag = reader.u(1)
      self.desc.weighted_pred_flag = reader.u(1)
      self.desc.weighted_bipred_flag = reader.u(1)
      self.desc.transquant_bypass_enabled_flag = reader.u(1)
      self.desc.tiles_enabled_flag = reader.u(1)
      self.desc.entropy_coding_sync_enabled_flag = reader.u(1)
      if self.desc.tiles_enabled_flag: assert False, "tiles parsing not implemented"
      self.desc.loop_filter_across_slices_enabled_flag = reader.u(1)
      self.desc.deblocking_filter_control_present_flag = reader.u(1)
      if self.desc.deblocking_filter_control_present_flag: assert False, "deblocking_filter parsing not implemented"
      self.desc.scaling_list_data_present_flag = reader.u(1)
      if self.desc.scaling_list_data_present_flag: assert False, "scaling_list_data parsing not implemented"
      self.desc.lists_modification_present_flag = reader.u(1)
      self.desc.log2_parallel_merge_level = reader.ue_v() + 2
    elif nal_unit_type == HevcNalUnitType.IDR_N_LP:
      self.desc.IDR_picture_flag = 1
      self.desc.RAP_picture_flag = 1
      self.desc.pattern_id = 2
      self.desc.stream_len = len(data)
      self.curr_pic_idx = pass

      # buf_in_data = dev.allocator.alloc(0x3fc000)
      # luma_chroma_bufs = [dev.allocator.alloc(0x2fd000) for i in range(3)]
      # coloc_buf = dev.allocator.alloc(0x400000)
      # filter_buf = dev.allocator.alloc(0xa00000)
      # scaling_list = dev.allocator.alloc(0x1000)
      # tile_sizes = dev.allocator.alloc(0x1000)
      # status_buf = dev.allocator.alloc(0x1000)
      # status_desc = nv_gpu.nvdec_status_s.from_address(status_buf.cpu_view().addr)

def require_nal_unit_start(dat: bytes, nal_unit_start: int) -> None:
  if nal_unit_start < 1:
    raise ValueError("start index must be greater than zero")

  if dat[nal_unit_start:nal_unit_start + NAL_UNIT_START_CODE_SIZE] != NAL_UNIT_START_CODE:
    raise VideoFileInvalid("data must begin with start code")

def get_hevc_nal_unit_length(dat: bytes, nal_unit_start: int) -> int:
  try: pos = dat.index(NAL_UNIT_START_CODE, nal_unit_start + NAL_UNIT_START_CODE_SIZE)
  except ValueError: pos = -1

  # length of NAL unit is byte count up to next NAL unit start index
  nal_unit_len = (pos if pos != -1 else len(dat)) - nal_unit_start
  if DEBUG: print("  nal_unit_len:", nal_unit_len)
  return nal_unit_len

def get_hevc_nal_unit_type(dat: bytes, nal_unit_start: int) -> HevcNalUnitType:
  # 7.3.1.2 NAL unit header syntax
  # nal_unit_header( ) {    // descriptor
  #   forbidden_zero_bit    f(1)
  #   nal_unit_type         u(6)
  #   nuh_layer_id          u(6)
  #   nuh_temporal_id_plus1 u(3)
  # }
  header_start = nal_unit_start + NAL_UNIT_START_CODE_SIZE
  nal_unit_header = dat[header_start:header_start + NAL_UNIT_HEADER_SIZE]
  
  if len(nal_unit_header) != 2:
    raise VideoFileInvalid("data to short to contain nal unit header")

  reader = HEVCReader(nal_unit_header)
  reader.u(1)  # forbidden_zero_bit
  nal_unit_type = HevcNalUnitType(reader.u(6))
  nal_unit_type_2 = HevcNalUnitType((nal_unit_header[0] >> 1) & 0x3F)
  assert nal_unit_type == nal_unit_type_2, f"nal_unit_type parsing mismatch {nal_unit_type} != {nal_unit_type_2}"

  if DEBUG: print("  nal_unit_type:", nal_unit_type.name, f"({nal_unit_type.value})")
  return nal_unit_type

def hevc_index(hevc_file_name: str, allow_corrupt: bool=False) -> tuple[list, int, bytes]:
  with pathlib.Path(hevc_file_name).open("rb") as f:
    dat = f.read()

  if len(dat) < NAL_UNIT_START_CODE_SIZE + 1:
    raise VideoFileInvalid("data is too short")

  if dat[0] != 0x00:
    raise VideoFileInvalid("first byte must be 0x00")

  prefix_dat = b""
  frame_types = list()

  decoder = NVVidDecoder()

  i = 1 # skip past first byte 0x00
  try:
    while i < len(dat):
      require_nal_unit_start(dat, i)
      nal_unit_len = get_hevc_nal_unit_length(dat, i)
      nal_unit_type = get_hevc_nal_unit_type(dat, i)

      rbsp = bytes()
      reader = HEVCReader(dat[i+3+2:i+3+nal_unit_len])
      # 7.3.1.1 General NAL unit syntax
      j = 2
      while j < nal_unit_len:
        print(nal_unit_len-j, reader.total - reader.read_bits + reader.current_bits)
        if j + 2 < nal_unit_len and reader.peak_bits(24) == 0x000003:
          rbsp += bytes([reader.u(8), reader.u(8)])
          assert 3 == reader.u(8) # skip emulation_prevention_three_byte
          j += 3
        else:
          rbsp += bytes([reader.u(8)])
          j += 1

      decoder.stream_slice(nal_unit_type, rbsp)

      # if nal_unit_type == HevcNalUnitType.VPS_NUT: pass
      # if nal_unit_type in HEVC_PARAMETER_SET_NAL_UNITS:
      #   prefix_dat += dat[i:i+nal_unit_len]
      elif nal_unit_type in HEVC_CODED_SLICE_SEGMENT_NAL_UNITS:
        slice_type, is_first_slice = get_hevc_slice_type(dat, i, nal_unit_type)
        if is_first_slice:
          frame_types.append((slice_type, i))
      i += nal_unit_len
  except Exception as e:
    if not allow_corrupt:
      raise
    print(f"ERROR: NAL unit skipped @ {i}\n", str(e))

  return frame_types, len(dat), prefix_dat

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input_file", type=str)
  args = parser.parse_args()

  frame_types, dat_len, prefix_dat = hevc_index(args.input_file)
  with open(args.output_prefix_file, "wb") as f:
    f.write(prefix_dat)

  with open(args.output_index_file, "wb") as f:
    for ft, fp in frame_types:
      f.write(struct.pack("<II", ft, fp))
    f.write(struct.pack("<II", 0xFFFFFFFF, dat_len))

if __name__ == "__main__":
  main()