# mypy: ignore-errors
from __future__ import annotations
import ctypes
from typing import Annotated
from tinygrad.runtime.support.c import DLL, record, CEnum, _IO, _IOW, _IOR, _IOWR, init_records
enum_HEVCNALUnitType = CEnum(ctypes.c_uint32)
HEVC_NAL_TRAIL_N = enum_HEVCNALUnitType.define('HEVC_NAL_TRAIL_N', 0)
HEVC_NAL_TRAIL_R = enum_HEVCNALUnitType.define('HEVC_NAL_TRAIL_R', 1)
HEVC_NAL_TSA_N = enum_HEVCNALUnitType.define('HEVC_NAL_TSA_N', 2)
HEVC_NAL_TSA_R = enum_HEVCNALUnitType.define('HEVC_NAL_TSA_R', 3)
HEVC_NAL_STSA_N = enum_HEVCNALUnitType.define('HEVC_NAL_STSA_N', 4)
HEVC_NAL_STSA_R = enum_HEVCNALUnitType.define('HEVC_NAL_STSA_R', 5)
HEVC_NAL_RADL_N = enum_HEVCNALUnitType.define('HEVC_NAL_RADL_N', 6)
HEVC_NAL_RADL_R = enum_HEVCNALUnitType.define('HEVC_NAL_RADL_R', 7)
HEVC_NAL_RASL_N = enum_HEVCNALUnitType.define('HEVC_NAL_RASL_N', 8)
HEVC_NAL_RASL_R = enum_HEVCNALUnitType.define('HEVC_NAL_RASL_R', 9)
HEVC_NAL_VCL_N10 = enum_HEVCNALUnitType.define('HEVC_NAL_VCL_N10', 10)
HEVC_NAL_VCL_R11 = enum_HEVCNALUnitType.define('HEVC_NAL_VCL_R11', 11)
HEVC_NAL_VCL_N12 = enum_HEVCNALUnitType.define('HEVC_NAL_VCL_N12', 12)
HEVC_NAL_VCL_R13 = enum_HEVCNALUnitType.define('HEVC_NAL_VCL_R13', 13)
HEVC_NAL_VCL_N14 = enum_HEVCNALUnitType.define('HEVC_NAL_VCL_N14', 14)
HEVC_NAL_VCL_R15 = enum_HEVCNALUnitType.define('HEVC_NAL_VCL_R15', 15)
HEVC_NAL_BLA_W_LP = enum_HEVCNALUnitType.define('HEVC_NAL_BLA_W_LP', 16)
HEVC_NAL_BLA_W_RADL = enum_HEVCNALUnitType.define('HEVC_NAL_BLA_W_RADL', 17)
HEVC_NAL_BLA_N_LP = enum_HEVCNALUnitType.define('HEVC_NAL_BLA_N_LP', 18)
HEVC_NAL_IDR_W_RADL = enum_HEVCNALUnitType.define('HEVC_NAL_IDR_W_RADL', 19)
HEVC_NAL_IDR_N_LP = enum_HEVCNALUnitType.define('HEVC_NAL_IDR_N_LP', 20)
HEVC_NAL_CRA_NUT = enum_HEVCNALUnitType.define('HEVC_NAL_CRA_NUT', 21)
HEVC_NAL_RSV_IRAP_VCL22 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_IRAP_VCL22', 22)
HEVC_NAL_RSV_IRAP_VCL23 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_IRAP_VCL23', 23)
HEVC_NAL_RSV_VCL24 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL24', 24)
HEVC_NAL_RSV_VCL25 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL25', 25)
HEVC_NAL_RSV_VCL26 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL26', 26)
HEVC_NAL_RSV_VCL27 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL27', 27)
HEVC_NAL_RSV_VCL28 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL28', 28)
HEVC_NAL_RSV_VCL29 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL29', 29)
HEVC_NAL_RSV_VCL30 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL30', 30)
HEVC_NAL_RSV_VCL31 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_VCL31', 31)
HEVC_NAL_VPS = enum_HEVCNALUnitType.define('HEVC_NAL_VPS', 32)
HEVC_NAL_SPS = enum_HEVCNALUnitType.define('HEVC_NAL_SPS', 33)
HEVC_NAL_PPS = enum_HEVCNALUnitType.define('HEVC_NAL_PPS', 34)
HEVC_NAL_AUD = enum_HEVCNALUnitType.define('HEVC_NAL_AUD', 35)
HEVC_NAL_EOS_NUT = enum_HEVCNALUnitType.define('HEVC_NAL_EOS_NUT', 36)
HEVC_NAL_EOB_NUT = enum_HEVCNALUnitType.define('HEVC_NAL_EOB_NUT', 37)
HEVC_NAL_FD_NUT = enum_HEVCNALUnitType.define('HEVC_NAL_FD_NUT', 38)
HEVC_NAL_SEI_PREFIX = enum_HEVCNALUnitType.define('HEVC_NAL_SEI_PREFIX', 39)
HEVC_NAL_SEI_SUFFIX = enum_HEVCNALUnitType.define('HEVC_NAL_SEI_SUFFIX', 40)
HEVC_NAL_RSV_NVCL41 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL41', 41)
HEVC_NAL_RSV_NVCL42 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL42', 42)
HEVC_NAL_RSV_NVCL43 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL43', 43)
HEVC_NAL_RSV_NVCL44 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL44', 44)
HEVC_NAL_RSV_NVCL45 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL45', 45)
HEVC_NAL_RSV_NVCL46 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL46', 46)
HEVC_NAL_RSV_NVCL47 = enum_HEVCNALUnitType.define('HEVC_NAL_RSV_NVCL47', 47)
HEVC_NAL_UNSPEC48 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC48', 48)
HEVC_NAL_UNSPEC49 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC49', 49)
HEVC_NAL_UNSPEC50 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC50', 50)
HEVC_NAL_UNSPEC51 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC51', 51)
HEVC_NAL_UNSPEC52 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC52', 52)
HEVC_NAL_UNSPEC53 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC53', 53)
HEVC_NAL_UNSPEC54 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC54', 54)
HEVC_NAL_UNSPEC55 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC55', 55)
HEVC_NAL_UNSPEC56 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC56', 56)
HEVC_NAL_UNSPEC57 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC57', 57)
HEVC_NAL_UNSPEC58 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC58', 58)
HEVC_NAL_UNSPEC59 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC59', 59)
HEVC_NAL_UNSPEC60 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC60', 60)
HEVC_NAL_UNSPEC61 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC61', 61)
HEVC_NAL_UNSPEC62 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC62', 62)
HEVC_NAL_UNSPEC63 = enum_HEVCNALUnitType.define('HEVC_NAL_UNSPEC63', 63)

enum_HEVCSliceType = CEnum(ctypes.c_uint32)
HEVC_SLICE_B = enum_HEVCSliceType.define('HEVC_SLICE_B', 0)
HEVC_SLICE_P = enum_HEVCSliceType.define('HEVC_SLICE_P', 1)
HEVC_SLICE_I = enum_HEVCSliceType.define('HEVC_SLICE_I', 2)

_anonenum0 = CEnum(ctypes.c_uint32)
HEVC_MAX_LAYERS = _anonenum0.define('HEVC_MAX_LAYERS', 63)
HEVC_MAX_SUB_LAYERS = _anonenum0.define('HEVC_MAX_SUB_LAYERS', 7)
HEVC_MAX_LAYER_SETS = _anonenum0.define('HEVC_MAX_LAYER_SETS', 1024)
HEVC_MAX_LAYER_ID = _anonenum0.define('HEVC_MAX_LAYER_ID', 63)
HEVC_MAX_NUH_LAYER_ID = _anonenum0.define('HEVC_MAX_NUH_LAYER_ID', 62)
HEVC_MAX_VPS_COUNT = _anonenum0.define('HEVC_MAX_VPS_COUNT', 16)
HEVC_MAX_SPS_COUNT = _anonenum0.define('HEVC_MAX_SPS_COUNT', 16)
HEVC_MAX_PPS_COUNT = _anonenum0.define('HEVC_MAX_PPS_COUNT', 64)
HEVC_MAX_DPB_SIZE = _anonenum0.define('HEVC_MAX_DPB_SIZE', 16)
HEVC_MAX_REFS = _anonenum0.define('HEVC_MAX_REFS', 16)
HEVC_MAX_SHORT_TERM_REF_PIC_SETS = _anonenum0.define('HEVC_MAX_SHORT_TERM_REF_PIC_SETS', 64)
HEVC_MAX_LONG_TERM_REF_PICS = _anonenum0.define('HEVC_MAX_LONG_TERM_REF_PICS', 32)
HEVC_MIN_LOG2_CTB_SIZE = _anonenum0.define('HEVC_MIN_LOG2_CTB_SIZE', 4)
HEVC_MAX_LOG2_CTB_SIZE = _anonenum0.define('HEVC_MAX_LOG2_CTB_SIZE', 6)
HEVC_MAX_CPB_CNT = _anonenum0.define('HEVC_MAX_CPB_CNT', 32)
HEVC_MAX_LUMA_PS = _anonenum0.define('HEVC_MAX_LUMA_PS', 35651584)
HEVC_MAX_WIDTH = _anonenum0.define('HEVC_MAX_WIDTH', 16888)
HEVC_MAX_HEIGHT = _anonenum0.define('HEVC_MAX_HEIGHT', 16888)
HEVC_MAX_TILE_ROWS = _anonenum0.define('HEVC_MAX_TILE_ROWS', 22)
HEVC_MAX_TILE_COLUMNS = _anonenum0.define('HEVC_MAX_TILE_COLUMNS', 20)
HEVC_MAX_SLICE_SEGMENTS = _anonenum0.define('HEVC_MAX_SLICE_SEGMENTS', 600)
HEVC_MAX_ENTRY_POINT_OFFSETS = _anonenum0.define('HEVC_MAX_ENTRY_POINT_OFFSETS', 2700)
HEVC_MAX_PALETTE_PREDICTOR_SIZE = _anonenum0.define('HEVC_MAX_PALETTE_PREDICTOR_SIZE', 128)

enum_HEVCScalabilityMask = CEnum(ctypes.c_uint32)
HEVC_SCALABILITY_DEPTH = enum_HEVCScalabilityMask.define('HEVC_SCALABILITY_DEPTH', 32768)
HEVC_SCALABILITY_MULTIVIEW = enum_HEVCScalabilityMask.define('HEVC_SCALABILITY_MULTIVIEW', 16384)
HEVC_SCALABILITY_SPATIAL = enum_HEVCScalabilityMask.define('HEVC_SCALABILITY_SPATIAL', 8192)
HEVC_SCALABILITY_AUXILIARY = enum_HEVCScalabilityMask.define('HEVC_SCALABILITY_AUXILIARY', 4096)
HEVC_SCALABILITY_MASK_MAX = enum_HEVCScalabilityMask.define('HEVC_SCALABILITY_MASK_MAX', 65535)

enum_HEVCAuxId = CEnum(ctypes.c_uint32)
HEVC_AUX_ALPHA = enum_HEVCAuxId.define('HEVC_AUX_ALPHA', 1)
HEVC_AUX_DEPTH = enum_HEVCAuxId.define('HEVC_AUX_DEPTH', 2)

@record
class struct_H265RawNALUnitHeader:
  SIZE = 3
  nal_unit_type: Annotated[uint8_t, 0]
  nuh_layer_id: Annotated[uint8_t, 1]
  nuh_temporal_id_plus1: Annotated[uint8_t, 2]
uint8_t = ctypes.c_ubyte
H265RawNALUnitHeader = struct_H265RawNALUnitHeader
@record
class struct_H265RawProfileTierLevel:
  SIZE = 422
  general_profile_space: Annotated[uint8_t, 0]
  general_tier_flag: Annotated[uint8_t, 1]
  general_profile_idc: Annotated[uint8_t, 2]
  general_profile_compatibility_flag: Annotated[(uint8_t* 32), 3]
  general_progressive_source_flag: Annotated[uint8_t, 35]
  general_interlaced_source_flag: Annotated[uint8_t, 36]
  general_non_packed_constraint_flag: Annotated[uint8_t, 37]
  general_frame_only_constraint_flag: Annotated[uint8_t, 38]
  general_max_12bit_constraint_flag: Annotated[uint8_t, 39]
  general_max_10bit_constraint_flag: Annotated[uint8_t, 40]
  general_max_8bit_constraint_flag: Annotated[uint8_t, 41]
  general_max_422chroma_constraint_flag: Annotated[uint8_t, 42]
  general_max_420chroma_constraint_flag: Annotated[uint8_t, 43]
  general_max_monochrome_constraint_flag: Annotated[uint8_t, 44]
  general_intra_constraint_flag: Annotated[uint8_t, 45]
  general_one_picture_only_constraint_flag: Annotated[uint8_t, 46]
  general_lower_bit_rate_constraint_flag: Annotated[uint8_t, 47]
  general_max_14bit_constraint_flag: Annotated[uint8_t, 48]
  general_inbld_flag: Annotated[uint8_t, 49]
  general_level_idc: Annotated[uint8_t, 50]
  sub_layer_profile_present_flag: Annotated[(uint8_t* 7), 51]
  sub_layer_level_present_flag: Annotated[(uint8_t* 7), 58]
  sub_layer_profile_space: Annotated[(uint8_t* 7), 65]
  sub_layer_tier_flag: Annotated[(uint8_t* 7), 72]
  sub_layer_profile_idc: Annotated[(uint8_t* 7), 79]
  sub_layer_profile_compatibility_flag: Annotated[((uint8_t* 32)* 7), 86]
  sub_layer_progressive_source_flag: Annotated[(uint8_t* 7), 310]
  sub_layer_interlaced_source_flag: Annotated[(uint8_t* 7), 317]
  sub_layer_non_packed_constraint_flag: Annotated[(uint8_t* 7), 324]
  sub_layer_frame_only_constraint_flag: Annotated[(uint8_t* 7), 331]
  sub_layer_max_12bit_constraint_flag: Annotated[(uint8_t* 7), 338]
  sub_layer_max_10bit_constraint_flag: Annotated[(uint8_t* 7), 345]
  sub_layer_max_8bit_constraint_flag: Annotated[(uint8_t* 7), 352]
  sub_layer_max_422chroma_constraint_flag: Annotated[(uint8_t* 7), 359]
  sub_layer_max_420chroma_constraint_flag: Annotated[(uint8_t* 7), 366]
  sub_layer_max_monochrome_constraint_flag: Annotated[(uint8_t* 7), 373]
  sub_layer_intra_constraint_flag: Annotated[(uint8_t* 7), 380]
  sub_layer_one_picture_only_constraint_flag: Annotated[(uint8_t* 7), 387]
  sub_layer_lower_bit_rate_constraint_flag: Annotated[(uint8_t* 7), 394]
  sub_layer_max_14bit_constraint_flag: Annotated[(uint8_t* 7), 401]
  sub_layer_inbld_flag: Annotated[(uint8_t* 7), 408]
  sub_layer_level_idc: Annotated[(uint8_t* 7), 415]
H265RawProfileTierLevel = struct_H265RawProfileTierLevel
@record
class struct_H265RawSubLayerHRDParameters:
  SIZE = 544
  bit_rate_value_minus1: Annotated[(uint32_t* 32), 0]
  cpb_size_value_minus1: Annotated[(uint32_t* 32), 128]
  cpb_size_du_value_minus1: Annotated[(uint32_t* 32), 256]
  bit_rate_du_value_minus1: Annotated[(uint32_t* 32), 384]
  cbr_flag: Annotated[(uint8_t* 32), 512]
uint32_t = ctypes.c_uint32
H265RawSubLayerHRDParameters = struct_H265RawSubLayerHRDParameters
@record
class struct_H265RawHRDParameters:
  SIZE = 7672
  nal_hrd_parameters_present_flag: Annotated[uint8_t, 0]
  vcl_hrd_parameters_present_flag: Annotated[uint8_t, 1]
  sub_pic_hrd_params_present_flag: Annotated[uint8_t, 2]
  tick_divisor_minus2: Annotated[uint8_t, 3]
  du_cpb_removal_delay_increment_length_minus1: Annotated[uint8_t, 4]
  sub_pic_cpb_params_in_pic_timing_sei_flag: Annotated[uint8_t, 5]
  dpb_output_delay_du_length_minus1: Annotated[uint8_t, 6]
  bit_rate_scale: Annotated[uint8_t, 7]
  cpb_size_scale: Annotated[uint8_t, 8]
  cpb_size_du_scale: Annotated[uint8_t, 9]
  initial_cpb_removal_delay_length_minus1: Annotated[uint8_t, 10]
  au_cpb_removal_delay_length_minus1: Annotated[uint8_t, 11]
  dpb_output_delay_length_minus1: Annotated[uint8_t, 12]
  fixed_pic_rate_general_flag: Annotated[(uint8_t* 7), 13]
  fixed_pic_rate_within_cvs_flag: Annotated[(uint8_t* 7), 20]
  elemental_duration_in_tc_minus1: Annotated[(uint16_t* 7), 28]
  low_delay_hrd_flag: Annotated[(uint8_t* 7), 42]
  cpb_cnt_minus1: Annotated[(uint8_t* 7), 49]
  nal_sub_layer_hrd_parameters: Annotated[(H265RawSubLayerHRDParameters* 7), 56]
  vcl_sub_layer_hrd_parameters: Annotated[(H265RawSubLayerHRDParameters* 7), 3864]
uint16_t = ctypes.c_uint16
H265RawHRDParameters = struct_H265RawHRDParameters
@record
class struct_H265RawVUI:
  SIZE = 7736
  aspect_ratio_info_present_flag: Annotated[uint8_t, 0]
  aspect_ratio_idc: Annotated[uint8_t, 1]
  sar_width: Annotated[uint16_t, 2]
  sar_height: Annotated[uint16_t, 4]
  overscan_info_present_flag: Annotated[uint8_t, 6]
  overscan_appropriate_flag: Annotated[uint8_t, 7]
  video_signal_type_present_flag: Annotated[uint8_t, 8]
  video_format: Annotated[uint8_t, 9]
  video_full_range_flag: Annotated[uint8_t, 10]
  colour_description_present_flag: Annotated[uint8_t, 11]
  colour_primaries: Annotated[uint8_t, 12]
  transfer_characteristics: Annotated[uint8_t, 13]
  matrix_coefficients: Annotated[uint8_t, 14]
  chroma_loc_info_present_flag: Annotated[uint8_t, 15]
  chroma_sample_loc_type_top_field: Annotated[uint8_t, 16]
  chroma_sample_loc_type_bottom_field: Annotated[uint8_t, 17]
  neutral_chroma_indication_flag: Annotated[uint8_t, 18]
  field_seq_flag: Annotated[uint8_t, 19]
  frame_field_info_present_flag: Annotated[uint8_t, 20]
  default_display_window_flag: Annotated[uint8_t, 21]
  def_disp_win_left_offset: Annotated[uint16_t, 22]
  def_disp_win_right_offset: Annotated[uint16_t, 24]
  def_disp_win_top_offset: Annotated[uint16_t, 26]
  def_disp_win_bottom_offset: Annotated[uint16_t, 28]
  vui_timing_info_present_flag: Annotated[uint8_t, 30]
  vui_num_units_in_tick: Annotated[uint32_t, 32]
  vui_time_scale: Annotated[uint32_t, 36]
  vui_poc_proportional_to_timing_flag: Annotated[uint8_t, 40]
  vui_num_ticks_poc_diff_one_minus1: Annotated[uint32_t, 44]
  vui_hrd_parameters_present_flag: Annotated[uint8_t, 48]
  hrd_parameters: Annotated[H265RawHRDParameters, 52]
  bitstream_restriction_flag: Annotated[uint8_t, 7724]
  tiles_fixed_structure_flag: Annotated[uint8_t, 7725]
  motion_vectors_over_pic_boundaries_flag: Annotated[uint8_t, 7726]
  restricted_ref_pic_lists_flag: Annotated[uint8_t, 7727]
  min_spatial_segmentation_idc: Annotated[uint16_t, 7728]
  max_bytes_per_pic_denom: Annotated[uint8_t, 7730]
  max_bits_per_min_cu_denom: Annotated[uint8_t, 7731]
  log2_max_mv_length_horizontal: Annotated[uint8_t, 7732]
  log2_max_mv_length_vertical: Annotated[uint8_t, 7733]
H265RawVUI = struct_H265RawVUI
class struct_H265RawExtensionData(ctypes.Structure): pass
H265RawExtensionData = struct_H265RawExtensionData
class struct_H265RawVPS(ctypes.Structure): pass
H265RawVPS = struct_H265RawVPS
@record
class struct_H265RawSTRefPicSet:
  SIZE = 136
  inter_ref_pic_set_prediction_flag: Annotated[uint8_t, 0]
  delta_idx_minus1: Annotated[uint8_t, 1]
  delta_rps_sign: Annotated[uint8_t, 2]
  abs_delta_rps_minus1: Annotated[uint16_t, 4]
  used_by_curr_pic_flag: Annotated[(uint8_t* 16), 6]
  use_delta_flag: Annotated[(uint8_t* 16), 22]
  num_negative_pics: Annotated[uint8_t, 38]
  num_positive_pics: Annotated[uint8_t, 39]
  delta_poc_s0_minus1: Annotated[(uint16_t* 16), 40]
  used_by_curr_pic_s0_flag: Annotated[(uint8_t* 16), 72]
  delta_poc_s1_minus1: Annotated[(uint16_t* 16), 88]
  used_by_curr_pic_s1_flag: Annotated[(uint8_t* 16), 120]
H265RawSTRefPicSet = struct_H265RawSTRefPicSet
@record
class struct_H265RawScalingList:
  SIZE = 1632
  scaling_list_pred_mode_flag: Annotated[((uint8_t* 6)* 4), 0]
  scaling_list_pred_matrix_id_delta: Annotated[((uint8_t* 6)* 4), 24]
  scaling_list_dc_coef_minus8: Annotated[((int16_t* 6)* 4), 48]
  scaling_list_delta_coeff: Annotated[(((int8_t* 64)* 6)* 4), 96]
int16_t = ctypes.c_int16
int8_t = ctypes.c_byte
H265RawScalingList = struct_H265RawScalingList
class struct_H265RawSPS(ctypes.Structure): pass
H265RawSPS = struct_H265RawSPS
class struct_H265RawPPS(ctypes.Structure): pass
H265RawPPS = struct_H265RawPPS
@record
class struct_H265RawAUD:
  SIZE = 4
  nal_unit_header: Annotated[H265RawNALUnitHeader, 0]
  pic_type: Annotated[uint8_t, 3]
H265RawAUD = struct_H265RawAUD
@record
class struct_H265RawSliceHeader:
  SIZE = 11772
  nal_unit_header: Annotated[H265RawNALUnitHeader, 0]
  first_slice_segment_in_pic_flag: Annotated[uint8_t, 3]
  no_output_of_prior_pics_flag: Annotated[uint8_t, 4]
  slice_pic_parameter_set_id: Annotated[uint8_t, 5]
  dependent_slice_segment_flag: Annotated[uint8_t, 6]
  slice_segment_address: Annotated[uint16_t, 8]
  slice_reserved_flag: Annotated[(uint8_t* 8), 10]
  slice_type: Annotated[uint8_t, 18]
  pic_output_flag: Annotated[uint8_t, 19]
  colour_plane_id: Annotated[uint8_t, 20]
  slice_pic_order_cnt_lsb: Annotated[uint16_t, 22]
  short_term_ref_pic_set_sps_flag: Annotated[uint8_t, 24]
  short_term_ref_pic_set: Annotated[H265RawSTRefPicSet, 26]
  short_term_ref_pic_set_idx: Annotated[uint8_t, 162]
  num_long_term_sps: Annotated[uint8_t, 163]
  num_long_term_pics: Annotated[uint8_t, 164]
  lt_idx_sps: Annotated[(uint8_t* 16), 165]
  poc_lsb_lt: Annotated[(uint8_t* 16), 181]
  used_by_curr_pic_lt_flag: Annotated[(uint8_t* 16), 197]
  delta_poc_msb_present_flag: Annotated[(uint8_t* 16), 213]
  delta_poc_msb_cycle_lt: Annotated[(uint32_t* 16), 232]
  slice_temporal_mvp_enabled_flag: Annotated[uint8_t, 296]
  slice_sao_luma_flag: Annotated[uint8_t, 297]
  slice_sao_chroma_flag: Annotated[uint8_t, 298]
  num_ref_idx_active_override_flag: Annotated[uint8_t, 299]
  num_ref_idx_l0_active_minus1: Annotated[uint8_t, 300]
  num_ref_idx_l1_active_minus1: Annotated[uint8_t, 301]
  ref_pic_list_modification_flag_l0: Annotated[uint8_t, 302]
  list_entry_l0: Annotated[(uint8_t* 16), 303]
  ref_pic_list_modification_flag_l1: Annotated[uint8_t, 319]
  list_entry_l1: Annotated[(uint8_t* 16), 320]
  mvd_l1_zero_flag: Annotated[uint8_t, 336]
  cabac_init_flag: Annotated[uint8_t, 337]
  collocated_from_l0_flag: Annotated[uint8_t, 338]
  collocated_ref_idx: Annotated[uint8_t, 339]
  luma_log2_weight_denom: Annotated[uint8_t, 340]
  delta_chroma_log2_weight_denom: Annotated[int8_t, 341]
  luma_weight_l0_flag: Annotated[(uint8_t* 16), 342]
  chroma_weight_l0_flag: Annotated[(uint8_t* 16), 358]
  delta_luma_weight_l0: Annotated[(int8_t* 16), 374]
  luma_offset_l0: Annotated[(int16_t* 16), 390]
  delta_chroma_weight_l0: Annotated[((int8_t* 2)* 16), 422]
  chroma_offset_l0: Annotated[((int16_t* 2)* 16), 454]
  luma_weight_l1_flag: Annotated[(uint8_t* 16), 518]
  chroma_weight_l1_flag: Annotated[(uint8_t* 16), 534]
  delta_luma_weight_l1: Annotated[(int8_t* 16), 550]
  luma_offset_l1: Annotated[(int16_t* 16), 566]
  delta_chroma_weight_l1: Annotated[((int8_t* 2)* 16), 598]
  chroma_offset_l1: Annotated[((int16_t* 2)* 16), 630]
  five_minus_max_num_merge_cand: Annotated[uint8_t, 694]
  use_integer_mv_flag: Annotated[uint8_t, 695]
  slice_qp_delta: Annotated[int8_t, 696]
  slice_cb_qp_offset: Annotated[int8_t, 697]
  slice_cr_qp_offset: Annotated[int8_t, 698]
  slice_act_y_qp_offset: Annotated[int8_t, 699]
  slice_act_cb_qp_offset: Annotated[int8_t, 700]
  slice_act_cr_qp_offset: Annotated[int8_t, 701]
  cu_chroma_qp_offset_enabled_flag: Annotated[uint8_t, 702]
  deblocking_filter_override_flag: Annotated[uint8_t, 703]
  slice_deblocking_filter_disabled_flag: Annotated[uint8_t, 704]
  slice_beta_offset_div2: Annotated[int8_t, 705]
  slice_tc_offset_div2: Annotated[int8_t, 706]
  slice_loop_filter_across_slices_enabled_flag: Annotated[uint8_t, 707]
  num_entry_point_offsets: Annotated[uint16_t, 708]
  offset_len_minus1: Annotated[uint8_t, 710]
  entry_point_offset_minus1: Annotated[(uint32_t* 2700), 712]
  slice_segment_header_extension_length: Annotated[uint16_t, 11512]
  slice_segment_header_extension_data_byte: Annotated[(uint8_t* 256), 11514]
H265RawSliceHeader = struct_H265RawSliceHeader
class struct_H265RawSlice(ctypes.Structure): pass
H265RawSlice = struct_H265RawSlice
@record
class struct_H265RawSEIBufferingPeriod:
  SIZE = 1048
  bp_seq_parameter_set_id: Annotated[uint8_t, 0]
  irap_cpb_params_present_flag: Annotated[uint8_t, 1]
  cpb_delay_offset: Annotated[uint32_t, 4]
  dpb_delay_offset: Annotated[uint32_t, 8]
  concatenation_flag: Annotated[uint8_t, 12]
  au_cpb_removal_delay_delta_minus1: Annotated[uint32_t, 16]
  nal_initial_cpb_removal_delay: Annotated[(uint32_t* 32), 20]
  nal_initial_cpb_removal_offset: Annotated[(uint32_t* 32), 148]
  nal_initial_alt_cpb_removal_delay: Annotated[(uint32_t* 32), 276]
  nal_initial_alt_cpb_removal_offset: Annotated[(uint32_t* 32), 404]
  vcl_initial_cpb_removal_delay: Annotated[(uint32_t* 32), 532]
  vcl_initial_cpb_removal_offset: Annotated[(uint32_t* 32), 660]
  vcl_initial_alt_cpb_removal_delay: Annotated[(uint32_t* 32), 788]
  vcl_initial_alt_cpb_removal_offset: Annotated[(uint32_t* 32), 916]
  use_alt_cpb_params_flag: Annotated[uint8_t, 1044]
H265RawSEIBufferingPeriod = struct_H265RawSEIBufferingPeriod
@record
class struct_H265RawSEIPicTiming:
  SIZE = 3624
  pic_struct: Annotated[uint8_t, 0]
  source_scan_type: Annotated[uint8_t, 1]
  duplicate_flag: Annotated[uint8_t, 2]
  au_cpb_removal_delay_minus1: Annotated[uint32_t, 4]
  pic_dpb_output_delay: Annotated[uint32_t, 8]
  pic_dpb_output_du_delay: Annotated[uint32_t, 12]
  num_decoding_units_minus1: Annotated[uint16_t, 16]
  du_common_cpb_removal_delay_flag: Annotated[uint8_t, 18]
  du_common_cpb_removal_delay_increment_minus1: Annotated[uint32_t, 20]
  num_nalus_in_du_minus1: Annotated[(uint16_t* 600), 24]
  du_cpb_removal_delay_increment_minus1: Annotated[(uint32_t* 600), 1224]
H265RawSEIPicTiming = struct_H265RawSEIPicTiming
@record
class struct_H265RawSEIPanScanRect:
  SIZE = 60
  pan_scan_rect_id: Annotated[uint32_t, 0]
  pan_scan_rect_cancel_flag: Annotated[uint8_t, 4]
  pan_scan_cnt_minus1: Annotated[uint8_t, 5]
  pan_scan_rect_left_offset: Annotated[(int32_t* 3), 8]
  pan_scan_rect_right_offset: Annotated[(int32_t* 3), 20]
  pan_scan_rect_top_offset: Annotated[(int32_t* 3), 32]
  pan_scan_rect_bottom_offset: Annotated[(int32_t* 3), 44]
  pan_scan_rect_persistence_flag: Annotated[uint16_t, 56]
int32_t = ctypes.c_int32
H265RawSEIPanScanRect = struct_H265RawSEIPanScanRect
@record
class struct_H265RawSEIRecoveryPoint:
  SIZE = 4
  recovery_poc_cnt: Annotated[int16_t, 0]
  exact_match_flag: Annotated[uint8_t, 2]
  broken_link_flag: Annotated[uint8_t, 3]
H265RawSEIRecoveryPoint = struct_H265RawSEIRecoveryPoint
@record
class struct_H265RawFilmGrainCharacteristics:
  SIZE = 10774
  film_grain_characteristics_cancel_flag: Annotated[uint8_t, 0]
  film_grain_model_id: Annotated[uint8_t, 1]
  separate_colour_description_present_flag: Annotated[uint8_t, 2]
  film_grain_bit_depth_luma_minus8: Annotated[uint8_t, 3]
  film_grain_bit_depth_chroma_minus8: Annotated[uint8_t, 4]
  film_grain_full_range_flag: Annotated[uint8_t, 5]
  film_grain_colour_primaries: Annotated[uint8_t, 6]
  film_grain_transfer_characteristics: Annotated[uint8_t, 7]
  film_grain_matrix_coeffs: Annotated[uint8_t, 8]
  blending_mode_id: Annotated[uint8_t, 9]
  log2_scale_factor: Annotated[uint8_t, 10]
  comp_model_present_flag: Annotated[(uint8_t* 3), 11]
  num_intensity_intervals_minus1: Annotated[(uint8_t* 3), 14]
  num_model_values_minus1: Annotated[(uint8_t* 3), 17]
  intensity_interval_lower_bound: Annotated[((uint8_t* 256)* 3), 20]
  intensity_interval_upper_bound: Annotated[((uint8_t* 256)* 3), 788]
  comp_model_value: Annotated[(((int16_t* 6)* 256)* 3), 1556]
  film_grain_characteristics_persistence_flag: Annotated[uint8_t, 10772]
H265RawFilmGrainCharacteristics = struct_H265RawFilmGrainCharacteristics
@record
class struct_H265RawSEIDisplayOrientation:
  SIZE = 10
  display_orientation_cancel_flag: Annotated[uint8_t, 0]
  hor_flip: Annotated[uint8_t, 1]
  ver_flip: Annotated[uint8_t, 2]
  anticlockwise_rotation: Annotated[uint16_t, 4]
  display_orientation_repetition_period: Annotated[uint16_t, 6]
  display_orientation_persistence_flag: Annotated[uint8_t, 8]
H265RawSEIDisplayOrientation = struct_H265RawSEIDisplayOrientation
@record
class struct_H265RawSEIActiveParameterSets:
  SIZE = 83
  active_video_parameter_set_id: Annotated[uint8_t, 0]
  self_contained_cvs_flag: Annotated[uint8_t, 1]
  no_parameter_set_update_flag: Annotated[uint8_t, 2]
  num_sps_ids_minus1: Annotated[uint8_t, 3]
  active_seq_parameter_set_id: Annotated[(uint8_t* 16), 4]
  layer_sps_idx: Annotated[(uint8_t* 63), 20]
H265RawSEIActiveParameterSets = struct_H265RawSEIActiveParameterSets
@record
class struct_H265RawSEIDecodedPictureHash:
  SIZE = 68
  hash_type: Annotated[uint8_t, 0]
  picture_md5: Annotated[((uint8_t* 16)* 3), 1]
  picture_crc: Annotated[(uint16_t* 3), 50]
  picture_checksum: Annotated[(uint32_t* 3), 56]
H265RawSEIDecodedPictureHash = struct_H265RawSEIDecodedPictureHash
@record
class struct_H265RawSEITimeCode:
  SIZE = 60
  num_clock_ts: Annotated[uint8_t, 0]
  clock_timestamp_flag: Annotated[(uint8_t* 3), 1]
  units_field_based_flag: Annotated[(uint8_t* 3), 4]
  counting_type: Annotated[(uint8_t* 3), 7]
  full_timestamp_flag: Annotated[(uint8_t* 3), 10]
  discontinuity_flag: Annotated[(uint8_t* 3), 13]
  cnt_dropped_flag: Annotated[(uint8_t* 3), 16]
  n_frames: Annotated[(uint16_t* 3), 20]
  seconds_value: Annotated[(uint8_t* 3), 26]
  minutes_value: Annotated[(uint8_t* 3), 29]
  hours_value: Annotated[(uint8_t* 3), 32]
  seconds_flag: Annotated[(uint8_t* 3), 35]
  minutes_flag: Annotated[(uint8_t* 3), 38]
  hours_flag: Annotated[(uint8_t* 3), 41]
  time_offset_length: Annotated[(uint8_t* 3), 44]
  time_offset_value: Annotated[(int32_t* 3), 48]
H265RawSEITimeCode = struct_H265RawSEITimeCode
@record
class struct_H265RawSEIAlphaChannelInfo:
  SIZE = 12
  alpha_channel_cancel_flag: Annotated[uint8_t, 0]
  alpha_channel_use_idc: Annotated[uint8_t, 1]
  alpha_channel_bit_depth_minus8: Annotated[uint8_t, 2]
  alpha_transparent_value: Annotated[uint16_t, 4]
  alpha_opaque_value: Annotated[uint16_t, 6]
  alpha_channel_incr_flag: Annotated[uint8_t, 8]
  alpha_channel_clip_flag: Annotated[uint8_t, 9]
  alpha_channel_clip_type_flag: Annotated[uint8_t, 10]
H265RawSEIAlphaChannelInfo = struct_H265RawSEIAlphaChannelInfo
@record
class struct_H265RawSEI3DReferenceDisplaysInfo:
  SIZE = 358
  prec_ref_display_width: Annotated[uint8_t, 0]
  ref_viewing_distance_flag: Annotated[uint8_t, 1]
  prec_ref_viewing_dist: Annotated[uint8_t, 2]
  num_ref_displays_minus1: Annotated[uint8_t, 3]
  left_view_id: Annotated[(uint16_t* 32), 4]
  right_view_id: Annotated[(uint16_t* 32), 68]
  exponent_ref_display_width: Annotated[(uint8_t* 32), 132]
  mantissa_ref_display_width: Annotated[(uint8_t* 32), 164]
  exponent_ref_viewing_distance: Annotated[(uint8_t* 32), 196]
  mantissa_ref_viewing_distance: Annotated[(uint8_t* 32), 228]
  additional_shift_present_flag: Annotated[(uint8_t* 32), 260]
  num_sample_shift_plus512: Annotated[(uint16_t* 32), 292]
  three_dimensional_reference_displays_extension_flag: Annotated[uint8_t, 356]
H265RawSEI3DReferenceDisplaysInfo = struct_H265RawSEI3DReferenceDisplaysInfo
@record
class struct_H265RawSEI:
  SIZE = 24
  nal_unit_header: Annotated[H265RawNALUnitHeader, 0]
  message_list: Annotated[SEIRawMessageList, 8]
@record
class struct_SEIRawMessageList:
  SIZE = 16
  messages: Annotated[ctypes.POINTER(SEIRawMessage), 0]
  nb_messages: Annotated[ctypes.c_int32, 8]
  nb_messages_allocated: Annotated[ctypes.c_int32, 12]
SEIRawMessageList = struct_SEIRawMessageList
@record
class struct_SEIRawMessage:
  SIZE = 40
  payload_type: Annotated[uint32_t, 0]
  payload_size: Annotated[uint32_t, 4]
  payload: Annotated[ctypes.POINTER(None), 8]
  payload_ref: Annotated[ctypes.POINTER(None), 16]
  extension_data: Annotated[ctypes.POINTER(uint8_t), 24]
  extension_bit_length: Annotated[size_t, 32]
SEIRawMessage = struct_SEIRawMessage
size_t = ctypes.c_uint64
H265RawSEI = struct_H265RawSEI
@record
class struct_H265RawFiller:
  SIZE = 8
  nal_unit_header: Annotated[H265RawNALUnitHeader, 0]
  filler_size: Annotated[uint32_t, 4]
H265RawFiller = struct_H265RawFiller
class struct_CodedBitstreamH265Context(ctypes.Structure): pass
CodedBitstreamH265Context = struct_CodedBitstreamH265Context
init_records()
