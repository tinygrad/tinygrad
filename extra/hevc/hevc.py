import dataclasses, enum, argparse, os
from typing import Any
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import DEBUG
from tinygrad.nn.state import TensorIO

# H.265 specification
# https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-H.265-201802-S!!PDF-E&type=items

# NAL_UNIT_START_CODE = b"\x00\x00\x01"
# NAL_UNIT_START_CODE_SIZE = len(NAL_UNIT_START_CODE)
# NAL_UNIT_HEADER_SIZE = 2

class HevcNalUnitType(enum.IntEnum):
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

class HevcSliceType(enum.IntEnum):
  B = 0
  P = 1
  I = 2

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

def _hevc_get_rbsp(dat:bytes, off=0) -> bytes:
  # 7.3.1.1 General NAL unit syntax
  rbsp = bytes()
  while off < len(dat):
    if off + 2 < len(dat) and dat[off:off+3] == b'\x00\x00\x03':
      rbsp += bytes([0, 0])
      off += 3
    else:
      rbsp += bytes([dat[off]])
      off += 1
  return rbsp

def _hevc_profile_tier_level(r:BitReader, enable:bool, max_sub_layers:int):
  assert enable and max_sub_layers == 0, "no sublayers supported"
  return {'_notimpl_profile_tier_level': r.u(88), 'general_level_idc': r.u(8)}

def _hevc_seq_parameter_set_rbsp(r:BitReader):
  f = {'sps_video_parameter_set_id': r.u(4), 'sps_max_sub_layers_minus1': r.u(3), 'sps_temporal_id_nesting_flag': r.u(1)}
  f.update(_hevc_profile_tier_level(r, True, f['sps_max_sub_layers_minus1']))
  f.update({'sps_seq_parameter_set_id': r.ue_v(), 'chroma_format_idc': r.ue_v()})
  if f['chroma_format_idc'] == 3: f.update({'sps_separate_colour_plane_flag': r.u(1)})
  f.update({'pic_width_in_luma_samples': r.ue_v(), 'pic_height_in_luma_samples': r.ue_v(), 'conformance_window_flag': r.u(1)})
  if f['conformance_window_flag']:
    f.update({'conf_win_left_offset': r.ue_v(), 'conf_win_right_offset': r.ue_v(), 'conf_win_top_offset': r.ue_v(), 'conf_win_bottom_offset': r.ue_v()})
  f.update({'bit_depth_luma_minus8': r.ue_v(), 'bit_depth_chroma_minus8': r.ue_v(), 'log2_max_pic_order_cnt_lsb_minus4': r.ue_v()})
  f.update({'sps_sub_layer_ordering_info_present_flag': r.u(1), 'sps_max_dec_pic_buffering_minus1': [], 'sps_max_num_reorder_pics': [], 
            'sps_max_latency_increase_plus1': []})
  for i in range((0 if f['sps_sub_layer_ordering_info_present_flag'] else f['sps_max_sub_layers_minus1']), f['sps_max_sub_layers_minus1'] + 1):
    f['sps_max_dec_pic_buffering_minus1'].append(r.ue_v())
    f['sps_max_num_reorder_pics'].append(r.ue_v())
    f['sps_max_latency_increase_plus1'].append(r.ue_v())
  f.update({'log2_min_luma_coding_block_size_minus3': r.ue_v(), 'log2_diff_max_min_luma_coding_block_size': r.ue_v(),
            'log2_min_luma_transform_block_size_minus2': r.ue_v(), 'log2_diff_max_min_luma_transform_block_size': r.ue_v()})
  f.update({'max_transform_hierarchy_depth_inter': r.ue_v(), 'max_transform_hierarchy_depth_intra': r.ue_v(), 'scaling_list_enabled_flag': r.u(1)})
  if f['scaling_list_enabled_flag']:
    f.update({'sps_scaling_list_data_present_flag': r.u(1)})
    if f['sps_scaling_list_data_present_flag']: assert False, "scaling_list_data parsing not implemented"
  f.update({'amp_enabled_flag': r.u(1), 'sample_adaptive_offset_enabled_flag': r.u(1), 'pcm_enabled_flag': r.u(1)})
  assert f['pcm_enabled_flag'] == 0, "pcm not implemented"
  f.update({'pps_num_short_term_ref_pic_sets': r.ue_v()})
  assert f['pps_num_short_term_ref_pic_sets'] == 0, "ref pic sets parsing not implemented"
  f.update({'pps_long_term_ref_pics_present_flag': r.u(1)})
  if f['pps_long_term_ref_pics_present_flag']: assert False, "long_term_ref_pics parsing not implemented"
  f.update({'sps_temporal_mvp_enabled_flag': r.u(1), 'strong_intra_smoothing_enabled_flag': r.u(1)})

  # NOTE: not all fields are parsed
  return f

def _hevc_pic_parameter_set_rbsp(r:BitReader):
  f = {'pps_pic_parameter_set_id': r.ue_v(), 'pps_seq_parameter_set_id': r.ue_v(), 'dependent_slice_segments_enabled_flag': r.u(1),
        'pps_output_flag_present_flag': r.u(1), 'pps_num_extra_slice_header_bits': r.u(3), 'sign_data_hiding_enabled_flag': r.u(1),
        'cabac_init_present_flag': r.u(1), 'num_ref_idx_l0_default_active_minus1': r.ue_v(), 'num_ref_idx_l1_default_active_minus1': r.ue_v(),
        'init_qp_minus26': r.se_v(), 'constrained_intra_pred_flag': r.u(1), 'transform_skip_enabled_flag': r.u(1),'cu_qp_delta_enabled_flag': r.u(1)}
  if f['cu_qp_delta_enabled_flag']: f.update({'diff_cu_qp_delta_depth': r.ue_v()})
  f.update({'pps_cb_qp_offset': r.se_v(), 'pps_cr_qp_offset': r.se_v(), 'pps_slice_chroma_qp_offsets_present_flag': r.u(1),
            'weighted_pred_flag': r.u(1), 'weighted_bipred_flag': r.u(1), 'transquant_bypass_enabled_flag': r.u(1),
            'tiles_enabled_flag': r.u(1), 'entropy_coding_sync_enabled_flag': r.u(1)})
  if f['tiles_enabled_flag']: assert False, "tiles parsing not implemented"
  f.update({'loop_filter_across_slices_enabled_flag': r.u(1), 'deblocking_filter_control_present_flag': r.u(1)})
  if f['deblocking_filter_control_present_flag']: assert False, "deblocking_filter parsing not implemented"
  f.update({'scaling_list_data_present_flag': r.u(1)})
  if f['scaling_list_data_present_flag']: assert False, "scaling_list_data parsing not implemented"
  f.update({'lists_modification_present_flag': r.u(1), 'log2_parallel_merge_level_minus2': r.ue_v()})
  return f

@dataclasses.dataclass
class HEVCCtx:
  pps:dict[str, tuple[int, int]] = dataclasses.field(default_factory=dict)
  sps:dict[str, tuple[int, int]] = dataclasses.field(default_factory=dict)

  def __hash__(self) -> int: return id(self) # TODO: fix this

class HEVCDecoder:
  def __init__(self, t:Tensor):
    self.io = TensorIO(t)
    self.context = HEVCCtx()

  def frames(self) -> Tensor:
    nal_unit_start = 1

    while nal_unit_start < self.io._tensor.shape[0]:
      self.io.seek(nal_unit_start)
      assert self.io.read(3) == b"\x00\x00\x01", "NAL unit start code not found"

      nal_unit_len, dat = self.io._tensor.shape[0] - nal_unit_start, bytearray()
      while tmpbuf:=self.io.read(4096):
        dat.extend(tmpbuf)
        if (off:=dat.find(b"\x00\x00\x01", 3)) != -1:
          nal_unit_len = off + 3
          break

      # 7.3.1.1 General NAL unit syntax
      nal_unit_type = (dat[0] >> 1) & 0x3F

      if nal_unit_type == HevcNalUnitType.SPS_NUT: self.context.sps = _hevc_seq_parameter_set_rbsp(BitReader(_hevc_get_rbsp(dat, off=2)))
      elif nal_unit_type == HevcNalUnitType.PPS_NUT: self.context.pps = _hevc_pic_parameter_set_rbsp(BitReader(_hevc_get_rbsp(dat, off=2)))
      elif nal_unit_type in {HevcNalUnitType.IDR_N_LP, HevcNalUnitType.TRAIL_R, HevcNalUnitType.TRAIL_N}:
        img = Tensor.from_hevc(nal_unit_type, self.io._tensor[nal_unit_start:nal_unit_start+nal_unit_len], self.context, device=Device.DEFAULT)
        yield img
        # print(img.uop)
        # print(img.tolist())
        exit(0)

      print(f"NAL unit type: {nal_unit_type}, length: {nal_unit_len}")
      nal_unit_start += nal_unit_len

def _addr_table(h, w):
  GOB_W, GOB_H = 64, 8          # pixels
  GOB_SIZE = 512            # bytes  (64 * 8 * 1-byte pixels)
  BLOCK_H_GOBS = 2

  xs = Tensor.arange(w, dtype=dtypes.uint32).reshape(1, w)
  ys = Tensor.arange(h, dtype=dtypes.uint32).reshape(h, 1)

  gob_x = xs // GOB_W
  gob_y = ys // GOB_H
  super_block_y  = gob_y // BLOCK_H_GOBS
  gob_y_in_block = gob_y  % BLOCK_H_GOBS
  stride_gobs    = w // GOB_W

  base = ((super_block_y * stride_gobs + gob_x) * BLOCK_H_GOBS + gob_y_in_block) * GOB_SIZE

  lx, ly = xs % GOB_W, ys % GOB_H
  swiz = (
      (lx & 0x0F) |
      ((ly & 0x03) << 4) |
      ((lx & 0x10) << 2) |
      ((ly & 0x04) << 5) |
      ((lx & 0x20) << 3)
  )
  return (base + swiz).reshape(-1)

def untile(luma_chroma_buf):
  src = Tensor.from_blob(luma_chroma_buf.va_addr, (0x3fe000,), dtype=dtypes.uint8) # TODO: remove this
  return bytes(src[_addr_table(1080, 1920)].reshape(-1).cat(src[_addr_table(540, 1920) + 0x1fe000].reshape(-1)).tolist())

if __name__ == "__main__":
  import cv2, numpy as np

  parser = argparse.ArgumentParser()
  parser.add_argument("input_file", type=str)
  args = parser.parse_args()

  hevc_tensor = Tensor.empty(os.stat(args.input_file).st_size, dtype=dtypes.uint8, device=f"disk:{args.input_file}").to("CPU")
  for i, src in enumerate(HEVCDecoder(hevc_tensor).frames()):
    img1 = bytes(src[_addr_table(1080, 1920)].reshape(-1).cat(src[_addr_table(540, 1920) + 0x1fe000].reshape(-1)).tolist())
    w, h, ch = 1920, 1080, 540
    total = h + ch
    frame = np.frombuffer(img1, dtype=np.uint8).reshape((total, w))
    cv2.imwrite(f"extra/nvdec/out/nvp_frame_{i}.png", cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12))
