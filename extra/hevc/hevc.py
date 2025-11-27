import dataclasses, enum, argparse, os, itertools, time, ctypes
from typing import Any
from tinygrad import Tensor, dtypes, Device, TinyJit
from tinygrad.helpers import DEBUG, round_up, ceildiv, Timing, prod
from tinygrad.runtime.autogen import avcodec, nv_570 as nv_gpu

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

  # 9.2 Parsing process for 0-th order Exp-Golomb codes
  def ue_v(self):
    leading_zero_bits = -1
    while True:
      bit = self.u(1)
      leading_zero_bits += 1
      if bit == 1: break

    part = self.u(leading_zero_bits)

    if leading_zero_bits == 0: return 0
    return (1 << leading_zero_bits) - 1 + part

  # 9.2.2 Mapping process for signed Exp-Golomb codes
  def se_v(self):
    k = self.ue_v()
    return (-1 ** (k + 1)) * (k // 2)

# 7.3.1.1 General NAL unit syntax
def _hevc_get_rbsp(dat:bytes, off=0) -> bytes:
  rbsp = bytes()
  while off < len(dat):
    if off + 2 < len(dat) and dat[off:off+3] == b'\x00\x00\x03':
      rbsp += bytes([0, 0])
      off += 3
    else:
      rbsp += bytes([dat[off]])
      off += 1
  return rbsp

class HevcSlice:
  # 7.3.3 Profile, tier and level syntax
  def profile_tier_level(self, r:BitReader, enable:bool, max_sub_layers:int):
    assert enable and max_sub_layers == 0, "no sublayers supported"
    self._notimpl_profile_tier_level = r.u(88)
    self.general_level_idc = r.u(8)

  # 7.3.7 Short-term reference picture set syntax
  def st_ref_pic_set(self, r:BitReader, stRpsIdx:int, num_short_term_ref_pic_sets:int=0):
    inter_ref_pic_set_prediction_flag = r.u(1) if stRpsIdx != 0 else 0

    if inter_ref_pic_set_prediction_flag:
      if stRpsIdx == num_short_term_ref_pic_sets:
        delta_idx_minus1 = r.ue_v()
      delta_rps_sign = r.u(1)
      abs_delta_rps_minus1 = r.ue_v()
      # for( j = 0; j <= NumDeltaPocs[ RefRpsIdx ]; j++ ) {

    else:
      num_negative_pics = r.ue_v()
      num_positive_pics = r.ue_v()
      for i in range(num_negative_pics):
        delta_poc_s0_minus1 = r.ue_v()
        used_by_curr_pic_s0_flag = r.u(1)
      for i in range(num_positive_pics):
        delta_poc_s1_minus1 = r.ue_v()
        used_by_curr_pic_s1_flag = r.u(1)

# 7.3.2.2 Sequence parameter set RBSP syntax
class SPS(HevcSlice):
  def __init__(self, r:BitReader):
    self.sps_video_parameter_set_id = r.u(4)
    self.sps_max_sub_layers_minus1 = r.u(3)
    self.sps_temporal_id_nesting_flag = r.u(1)

    self.profile_tier_level(r, True, self.sps_max_sub_layers_minus1)

    self.sps_seq_parameter_set_id = r.ue_v()
    self.chroma_format_idc = r.ue_v()
    self.separate_colour_plane_flag = r.u(1) if self.chroma_format_idc == 3 else 0
    self.pic_width_in_luma_samples = r.ue_v()
    self.pic_height_in_luma_samples = r.ue_v()
    self.conformance_window_flag = r.u(1)

    if self.conformance_window_flag:
      self.conf_win_left_offset = r.ue_v()
      self.conf_win_right_offset = r.ue_v()
      self.conf_win_top_offset = r.ue_v()
      self.conf_win_bottom_offset = r.ue_v()
    else: self.conf_win_left_offset = self.conf_win_right_offset = self.conf_win_top_offset = self.conf_win_bottom_offset = 0

    self.bit_depth_luma = r.ue_v() + 8
    self.bit_depth_chroma = r.ue_v() + 8
    self.log2_max_pic_order_cnt_lsb_minus4 = r.ue_v()
    self.sps_sub_layer_ordering_info_present_flag = r.u(1)
    self.sps_max_dec_pic_buffering, self.sps_max_num_reorder_pics, self.sps_max_latency_increase_plus1 = [], [], []
    for i in range((0 if self.sps_sub_layer_ordering_info_present_flag else self.sps_max_sub_layers_minus1), self.sps_max_sub_layers_minus1 + 1):
      self.sps_max_dec_pic_buffering.append(r.ue_v() + 1)
      self.sps_max_num_reorder_pics.append(r.ue_v())
      self.sps_max_latency_increase_plus1.append(r.ue_v())
    self.log2_min_luma_coding_block_size = r.ue_v() + 3
    self.log2_max_luma_coding_block_size = self.log2_min_luma_coding_block_size + r.ue_v()
    self.log2_min_transform_block_size = r.ue_v() + 2
    self.log2_max_transform_block_size = self.log2_min_transform_block_size + r.ue_v()
    self.max_transform_hierarchy_depth_inter = r.ue_v()
    self.max_transform_hierarchy_depth_intra = r.ue_v()
    if scaling_list_enabled_flag := r.u(1):
      if sps_scaling_list_data_present_flag := r.u(1): assert False, "scaling_list_data parsing not implemented"
    self.amp_enabled_flag = r.u(1)
    self.sample_adaptive_offset_enabled_flag = r.u(1)
    self.pcm_enabled_flag = r.u(1)
    assert self.pcm_enabled_flag == 0, "pcm not implemented"
    self.num_short_term_ref_pic_sets = r.ue_v()
    for i in range(self.num_short_term_ref_pic_sets):
      self.st_ref_pic_set(r, i, self.num_short_term_ref_pic_sets)
    self.long_term_ref_pics_present_flag = r.u(1)
    if self.long_term_ref_pics_present_flag: assert False, "long_term_ref_pics parsing not implemented"
    self.sps_temporal_mvp_enabled_flag = r.u(1)
    self.strong_intra_smoothing_enabled_flag = r.u(1)

# 7.3.2.3 Picture parameter set RBSP syntax
class PPS(HevcSlice):
  def __init__(self, r:BitReader):
    self.pps_pic_parameter_set_id = r.ue_v()
    self.pps_seq_parameter_set_id = r.ue_v()
    self.dependent_slice_segments_enabled_flag = r.u(1)
    self.output_flag_present_flag = r.u(1)
    self.num_extra_slice_header_bits = r.u(3)
    self.sign_data_hiding_enabled_flag = r.u(1)
    self.cabac_init_present_flag = r.u(1)
    self.num_ref_idx_l0_default_active = r.ue_v() + 1
    self.num_ref_idx_l1_default_active = r.ue_v() + 1
    self.init_qp = r.se_v() + 26
    self.constrained_intra_pred_flag = r.u(1)
    self.transform_skip_enabled_flag = r.u(1)
    self.cu_qp_delta_enabled_flag = r.u(1)
    if self.cu_qp_delta_enabled_flag: self.diff_cu_qp_delta_depth = r.ue_v()

    self.pps_cb_qp_offset = r.se_v()
    self.pps_cr_qp_offset = r.se_v()
    self.pps_slice_chroma_qp_offsets_present_flag = r.u(1)
    self.weighted_pred_flag = r.u(1)
    self.weighted_bipred_flag = r.u(1)
    self.transquant_bypass_enabled_flag = r.u(1)
    self.tiles_enabled_flag = r.u(1)
    self.entropy_coding_sync_enabled_flag = r.u(1)
    if self.tiles_enabled_flag:
      self.num_tile_columns_minus1 = r.ue_v()
      self.num_tile_rows_minus1 = r.ue_v()
      self.uniform_spacing_flag = r.u(1)
      self.column_width_minus1, self.row_height_minus1 = [], []
      if not self.uniform_spacing_flag:
        for i in range(self.num_tile_columns_minus1): self.column_width_minus1.append(r.ue_v())
        for i in range(self.num_tile_rows_minus1): self.row_height_minus1.append(r.ue_v())
      self.loop_filter_across_tiles_enabled_flag = r.u(1)
    self.loop_filter_across_slices_enabled_flag = r.u(1)
    self.deblocking_filter_control_present_flag = r.u(1)
    if self.deblocking_filter_control_present_flag: assert False, "deblocking_filter parsing not implemented"
    self.scaling_list_data_present_flag = r.u(1)
    if self.scaling_list_data_present_flag: assert False, "scaling_list_data parsing not implemented"
    self.lists_modification_present_flag = r.u(1)
    self.log2_parallel_merge_level = r.ue_v() + 2

# 7.3.6 Slice segment header syntax
class SliceSegment(HevcSlice):
  def __init__(self, r:BitReader, nal_unit_type:int, sps:SPS, pps:PPS):
    self.first_slice_segment_in_pic_flag = r.u(1)
    if nal_unit_type >= avcodec.HEVC_NAL_BLA_W_LP and nal_unit_type <= avcodec.HEVC_NAL_RSV_IRAP_VCL23:
      self.no_output_of_prior_pics_flag = r.u(1)
    self.slice_pic_parameter_set_id = r.ue_v()
    if not self.first_slice_segment_in_pic_flag:
      if pps.dependent_slice_segments_enabled_flag:
        self.dependent_slice_segment_flag = r.u(1)
      self.slice_segment_address = r.ue_v()
    self.dependent_slice_segment_flag = 0
    if not self.dependent_slice_segment_flag:
      r.u(pps.num_extra_slice_header_bits) # extra bits ignored
      self.slice_type = r.ue_v()

      self.sw_skip_start = r.read_bits - r.current_bits
      self.pic_output_flag = r.u(1) if pps.output_flag_present_flag else 0
      self.colour_plane_id = r.u(2) if sps.separate_colour_plane_flag else 0

      if nal_unit_type != avcodec.HEVC_NAL_IDR_W_RADL and nal_unit_type != avcodec.HEVC_NAL_IDR_N_LP:
        self.slice_pic_order_cnt_lsb = r.u(sps.log2_max_pic_order_cnt_lsb_minus4 + 4)

        self.short_term_ref_pic_set_sps_flag = r.u(1)
        if not self.short_term_ref_pic_set_sps_flag:
          self.short_term_ref_pics_in_slice_start = r.read_bits - r.current_bits
          self.st_ref_pic_set(r, sps.num_short_term_ref_pic_sets)
          self.short_term_ref_pics_in_slice_end = r.read_bits - r.current_bits
        elif sps.num_short_term_ref_pic_sets > 1: assert False, "short_term_ref_pic_set parsing not implemented"

        if sps.long_term_ref_pics_present_flag: assert False, "long_term_ref_pics parsing not implemented"

        self.sw_skip_end = r.read_bits - r.current_bits
        self.slice_temporal_mvp_enabled_flag = r.u(1) if sps.sps_temporal_mvp_enabled_flag else 0
      else: self.slice_pic_order_cnt_lsb, self.sw_skip_end = 0, self.sw_skip_start

      if sps.sample_adaptive_offset_enabled_flag:
        slice_sao_luma_flag = r.u(1)
        ChromaArrayType = sps.chroma_format_idc if sps.separate_colour_plane_flag == 0 else 0
        slice_sao_chroma_flag = r.u(1) if ChromaArrayType != 0 else 0

      if self.slice_type in {avcodec.HEVC_SLICE_B, avcodec.HEVC_SLICE_B}:
        if num_ref_idx_active_override_flag := r.u(1):
          num_ref_idx_l0_active_minus1 = r.ue_v()
          num_ref_idx_l1_active_minus1 = r.ue_v() if self.slice_type == avcodec.HEVC_SLICE_B else 0

def fill_sps_into_dev_context(device_ctx, sps:SPS):
  device_ctx.chroma_format_idc = sps.chroma_format_idc
  device_ctx.pic_width_in_luma_samples = sps.pic_width_in_luma_samples
  device_ctx.pic_height_in_luma_samples = sps.pic_height_in_luma_samples
  device_ctx.bit_depth_luma = sps.bit_depth_luma
  device_ctx.bit_depth_chroma = sps.bit_depth_chroma
  device_ctx.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4
  device_ctx.log2_min_luma_coding_block_size = sps.log2_min_luma_coding_block_size
  device_ctx.log2_max_luma_coding_block_size = sps.log2_max_luma_coding_block_size
  device_ctx.log2_min_transform_block_size = sps.log2_min_transform_block_size
  device_ctx.log2_max_transform_block_size = sps.log2_max_transform_block_size
  device_ctx.amp_enabled_flag = sps.amp_enabled_flag
  device_ctx.pcm_enabled_flag = sps.pcm_enabled_flag
  device_ctx.sample_adaptive_offset_enabled_flag = sps.sample_adaptive_offset_enabled_flag
  device_ctx.sps_temporal_mvp_enabled_flag = sps.sps_temporal_mvp_enabled_flag
  device_ctx.strong_intra_smoothing_enabled_flag = sps.strong_intra_smoothing_enabled_flag

def fill_pps_into_dev_context(device_ctx, pps:PPS):
  device_ctx.sign_data_hiding_enabled_flag = pps.sign_data_hiding_enabled_flag
  device_ctx.cabac_init_present_flag = pps.cabac_init_present_flag
  device_ctx.num_ref_idx_l0_default_active = pps.num_ref_idx_l0_default_active
  device_ctx.num_ref_idx_l1_default_active = pps.num_ref_idx_l1_default_active
  device_ctx.init_qp = pps.init_qp
  device_ctx.cu_qp_delta_enabled_flag = pps.cu_qp_delta_enabled_flag
  device_ctx.diff_cu_qp_delta_depth = getattr(pps, 'diff_cu_qp_delta_depth', 0)
  device_ctx.pps_cb_qp_offset = pps.pps_cb_qp_offset
  device_ctx.pps_cr_qp_offset = pps.pps_cr_qp_offset
  device_ctx.pps_slice_chroma_qp_offsets_present_flag = pps.pps_slice_chroma_qp_offsets_present_flag
  device_ctx.weighted_pred_flag = pps.weighted_pred_flag
  device_ctx.weighted_bipred_flag = pps.weighted_bipred_flag
  device_ctx.transquant_bypass_enabled_flag = pps.transquant_bypass_enabled_flag
  device_ctx.tiles_enabled_flag = pps.tiles_enabled_flag
  device_ctx.entropy_coding_sync_enabled_flag = pps.entropy_coding_sync_enabled_flag
  device_ctx.loop_filter_across_slices_enabled_flag = pps.loop_filter_across_slices_enabled_flag
  device_ctx.deblocking_filter_control_present_flag = pps.deblocking_filter_control_present_flag
  device_ctx.scaling_list_data_present_flag = pps.scaling_list_data_present_flag
  device_ctx.lists_modification_present_flag = pps.lists_modification_present_flag
  device_ctx.log2_parallel_merge_level = pps.log2_parallel_merge_level

def parse_hevc_file_headers(dat:bytes):
  res = []
  nal_unit_start = 1
  history:list[tuple[int, int, int]] = []
  device_ctx = nv_gpu.nvdec_hevc_pic_s(gptimer_timeout_value=81600000, tileformat=1, sw_start_code_e=1, pattern_id=2)

  while nal_unit_start < len(dat):
    assert dat[nal_unit_start:nal_unit_start+3] == b"\x00\x00\x01", "NAL unit start code not found"

    pos = dat.find(b"\x00\x00\x01", nal_unit_start + 3)
    nal_unit_len = (pos if pos != -1 else len(dat)) - nal_unit_start

    # 7.3.1.1 General NAL unit syntax
    nal_unit_type = (dat[nal_unit_start+3] >> 1) & 0x3F
    slice_dat = dat[nal_unit_start+5:nal_unit_start+nal_unit_len]

    if nal_unit_type == avcodec.HEVC_NAL_SPS:
      sps = SPS(BitReader(_hevc_get_rbsp(slice_dat)))
      fill_sps_into_dev_context(device_ctx, sps)
    elif nal_unit_type == avcodec.HEVC_NAL_PPS:
      pps = PPS(BitReader(_hevc_get_rbsp(slice_dat)))
      fill_pps_into_dev_context(device_ctx, pps)
    elif nal_unit_type in {avcodec.HEVC_NAL_IDR_N_LP, avcodec.HEVC_NAL_IDR_W_RADL, avcodec.HEVC_NAL_TRAIL_R, avcodec.HEVC_NAL_TRAIL_N}:
      hdr = SliceSegment(BitReader(slice_dat), nal_unit_type, sps, pps)

      device_ctx.curr_pic_idx = next(i for i in range(16) if all(d[0] != i for d in history))

      if nal_unit_type in {avcodec.HEVC_NAL_IDR_W_RADL, avcodec.HEVC_NAL_IDR_N_LP}:
        history = []

      device_ctx.num_ref_frames = len(history)
      device_ctx.IDR_picture_flag = int(nal_unit_type in {avcodec.HEVC_NAL_IDR_W_RADL, avcodec.HEVC_NAL_IDR_N_LP})
      device_ctx.RAP_picture_flag = int(nal_unit_type >= avcodec.HEVC_NAL_BLA_W_LP and nal_unit_type <= avcodec.HEVC_NAL_RSV_IRAP_VCL23)
      device_ctx.RefDiffPicOrderCnts=(ctypes.c_int16 * 16)()
      device_ctx.colMvBuffersize = (round_up(sps.pic_width_in_luma_samples, 64) * round_up(sps.pic_height_in_luma_samples, 64) // 16) // 256
      device_ctx.framestride=(ctypes.c_uint32 * 2)(round_up(sps.pic_width_in_luma_samples, 64), round_up(sps.pic_width_in_luma_samples, 64))
      device_ctx.sw_hdr_skip_length = hdr.sw_skip_end - hdr.sw_skip_start
      device_ctx.num_bits_short_term_ref_pics_in_slice = max(0, device_ctx.sw_hdr_skip_length - 9)
      device_ctx.stream_len = nal_unit_len

      before_list, after_list = [], []
      for pic_idx, poc, _ in history:
        device_ctx.RefDiffPicOrderCnts[pic_idx] = hdr.slice_pic_order_cnt_lsb - poc
        if hdr.slice_pic_order_cnt_lsb < poc: after_list.append((poc - hdr.slice_pic_order_cnt_lsb, pic_idx))
        else: before_list.append((hdr.slice_pic_order_cnt_lsb - poc, pic_idx))
      before_list.sort()
      after_list.sort()

      device_ctx.initreflistidxl0 = (ctypes.c_uint8 * 16)(*[idx for _,idx in before_list + after_list])
      if hdr.slice_type == avcodec.HEVC_SLICE_B: device_ctx.initreflistidxl1 = (ctypes.c_uint8 * 16)(*[idx for _,idx in after_list + before_list])

      ctx_bytes = bytes(device_ctx)
      ctx_bytes += bytes(0x200 - len(ctx_bytes)) # pad to 512 bytes

      # append tile sizes 0x200
      ctx_bytes += ceildiv(sps.pic_width_in_luma_samples, (1 << sps.log2_max_luma_coding_block_size)).to_bytes(2, "little")
      ctx_bytes += ceildiv(sps.pic_height_in_luma_samples, (1 << sps.log2_max_luma_coding_block_size)).to_bytes(2, "little")

      ctx = Tensor(ctx_bytes, dtype=dtypes.uint8)
      luma_size = round_up(sps.pic_width_in_luma_samples, 64) * round_up(sps.pic_height_in_luma_samples, 64)
      chroma_size = round_up(sps.pic_width_in_luma_samples, 64) * round_up((sps.pic_height_in_luma_samples + 1) // 2, 64)
      is_hist = nal_unit_type in {avcodec.HEVC_NAL_TRAIL_R, avcodec.HEVC_NAL_IDR_N_LP, avcodec.HEVC_NAL_IDR_W_RADL}

      res.append((nal_unit_start, nal_unit_len, (sps.pic_height_in_luma_samples, sps.pic_width_in_luma_samples), ctx, device_ctx.curr_pic_idx, len(history), is_hist))

      if nal_unit_type in {avcodec.HEVC_NAL_TRAIL_R, avcodec.HEVC_NAL_IDR_N_LP, avcodec.HEVC_NAL_IDR_W_RADL}:
        history.append((device_ctx.curr_pic_idx, hdr.slice_pic_order_cnt_lsb, None))

      if len(history) >= sps.sps_max_dec_pic_buffering[0]:
        # remove the oldest poc
        history.pop(0)

    if DEBUG >= 4: print(f"NAL unit type: {nal_unit_type}, length: {nal_unit_len}")
    nal_unit_start += nal_unit_len

  w = sps.pic_width_in_luma_samples - 2 * (sps.conf_win_left_offset + sps.conf_win_right_offset)
  h = sps.pic_height_in_luma_samples - 2 * (sps.conf_win_top_offset  + sps.conf_win_bottom_offset)
  return w, h, res

def _addr_table(h, w, w_aligned):
  GOB_W, GOB_H = 64, 8
  GOB_SIZE = GOB_W * GOB_H
  BLOCK_H_GOBS = 2

  xs = Tensor.arange(w, dtype=dtypes.uint32).reshape(1, w)
  ys = Tensor.arange(h, dtype=dtypes.uint32).reshape(h, 1)

  gob_x = xs // GOB_W
  gob_y = ys // GOB_H
  super_block_y = gob_y // BLOCK_H_GOBS
  gob_y_in_block = gob_y  % BLOCK_H_GOBS
  stride_gobs = w_aligned // GOB_W

  base = ((super_block_y * stride_gobs + gob_x) * BLOCK_H_GOBS + gob_y_in_block) * GOB_SIZE

  lx, ly = xs % GOB_W, ys % GOB_H
  swiz = (lx & 0x0F) | ((ly & 0x03) << 4) | ((lx & 0x10) << 2) | ((ly & 0x04) << 5) | ((lx & 0x20) << 3)
  return (base + swiz).reshape(-1)

def nv12_to_bgr_from_planes(luma: Tensor, chroma: Tensor, h: int, w: int) -> Tensor:
  Y = luma.reshape(h, w).cast(dtypes.float32)

  uv = chroma.reshape(h // 2, w // 2, 2).cast(dtypes.float32)
  U_small = uv[..., 0]
  V_small = uv[..., 1]

  U = U_small.reshape(h // 2, 1, w // 2, 1).expand(h // 2, 2, w // 2, 2).reshape(h, w)
  V = V_small.reshape(h // 2, 1, w // 2, 1).expand(h // 2, 2, w // 2, 2).reshape(h, w)

  C = Y - 16.0
  D = U - 128.0
  E = V - 128.0

  R = 1.1643835616438356 * C + 1.5960267857142858 * E
  G = 1.1643835616438356 * C - 0.39176229009491365 * D - 0.8129676472377708 * E
  B = 1.1643835616438356 * C + 2.017232142857143  * D

  R = R.maximum(0.0).minimum(255.0)
  G = G.maximum(0.0).minimum(255.0)
  B = B.maximum(0.0).minimum(255.0)

  return Tensor.stack([B, G, R], dim=2).cast(dtypes.uint8)

if __name__ == "__main__":
  import cv2

  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", type=str, default="")
  parser.add_argument("--output_dir", type=str, default="extra/hevc/out")
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  if args.input_file == "":
    url = "https://github.com/commaai/comma2k19/raw/refs/heads/master/Example_1/b0c9d2329ad1606b%7C2018-08-02--08-34-47/40/video.hevc"
    hevc_tensor = Tensor.from_url(url, device="CPU")
  else:
    hevc_tensor = Tensor.empty(os.stat(args.input_file).st_size, dtype=dtypes.uint8, device=f"disk:{args.input_file}").to("CPU")

  with Timing("prep infos: "):
    dat = bytes(hevc_tensor.data())
    dat_nv = hevc_tensor.to("NV")
    w, h, frame_info = parse_hevc_file_headers(dat)

  all_slices = []
  all_datas = []
  all_bufs_out = []
  all_pics = []
  with Timing("prep slices to gpu: "):
    for i, (offset, sz, shape, opaque, frame_pos, history_sz, _) in enumerate(frame_info):
      all_slices.append(dat_nv[offset:offset+sz].contiguous().realize())
      all_datas.append(opaque.contiguous().realize())
      
      chroma_off = round_up(shape[0], 64) * round_up(shape[1], 64)
      shape = (round_up(shape[0] + (shape[0] + 1) // 2, 64), round_up(shape[1], 64))
      bufout = Tensor.empty(shape, dtype=dtypes.uint8).realize()
      bufout.uop.buffer.allocate()
      all_bufs_out.append(bufout)

      pic = Tensor.empty(h, w, 3, dtype=dtypes.uint8).realize()
      pic.uop.buffer.allocate()
      all_pics.append(pic)
    Device.default.synchronize()

  @TinyJit
  def untile_nv12(src:Tensor, out:Tensor):
    luma = src.reshape(-1)[_addr_table(h, w, round_up(w, 64))]
    chroma = src.reshape(-1)[chroma_off:][_addr_table((h + 1) // 2, w, round_up(w, 64))]
    x = nv12_to_bgr_from_planes(luma, chroma, h, w)
    out.assign(x).realize()
    return x.realize()
  
  # warm up
  for i in range(3): x = untile_nv12(all_bufs_out[0], all_pics[0])

  from tinygrad.runtime.ops_nv import NVVideoQueue
  # from extra.nv_gpu_driver.nv_ioctl import dump_struct_ext
  dev = Device.default
  dev._ensure_has_vid_hw(w, h)

  # from hexdump import hexdump

  history = []
  prev_timeline_wait = dev.timeline_value - 1
  with Timing("decoding whole file (no tg): "):
    for i, (offset, sz, shape, opaque, frame_pos, history_sz, is_hist) in enumerate(frame_info):
      history = history[-history_sz:] if history_sz > 0 else []

      bufin_hcq = all_slices[i].uop.buffer._buf
      desc_buf_hcq = all_datas[i].uop.buffer._buf
      bufout_hcq = all_bufs_out[i].uop.buffer._buf

      # dump_struct_ext(nv_gpu.nvdec_hevc_pic_s.from_buffer(opaque.data()))
      # assert frame_pos not in [(frame_pos-x) % (len(history)+1) for x in range(len(history), 0, -1)]
      # print(frame_pos, [(frame_pos-x) % (len(history)+1) for x in range(len(history), 0, -1)], [(frame_pos-x) % 5 for x in range(len(history), 0, -1)])

      NVVideoQueue().wait(dev.timeline_signal, prev_timeline_wait) \
                    .decode_hevc_chunk(i, desc_buf_hcq, bufin_hcq, bufout_hcq, frame_pos, history,
                                       [(frame_pos-x) % (len(history) + 1) for x in range(len(history), 0, -1)],
                                       chroma_off, dev.vid_coloc_buf, dev.vid_filter_buf, dev.intra_top_off, dev.vid_status_buf) \
                    .signal(dev.timeline_signal, prev_timeline_wait:=dev.next_timeline()).submit(dev)

      untile_nv12(all_bufs_out[i], all_pics[i])
      if is_hist: history.append(bufout_hcq)
    Device.default.synchronize()

  import cv2
  for i, src in enumerate(all_pics):
    if i > 230:
      cv2.imwrite(f"{args.output_dir}/nv_frame_{i}.png", all_pics[i].numpy())
  # for i, src in enumerate(all_bufs_out):
  #   w_aligned = round_up(w, 64)
  #   luma = src.reshape(-1)[_addr_table(h, w, w_aligned)]
  #   chroma = src.reshape(-1)[chroma_off:][_addr_table((h + 1) // 2, w, w_aligned)]
  #   cv2.imwrite(f"{args.output_dir}/nv_frame_{i}.png", cv2.cvtColor(luma.cat(chroma).reshape((h + (h + 1) // 2, w)).numpy(), cv2.COLOR_YUV2BGR_NV12))
  #   cv2.imwrite(f"{args.output_dir}/c_nv_frame_{i}.png", cv2.cvtColor(chroma.reshape(((h + 1) // 2, w)).numpy(), cv2.COLOR_YUV2BGR_NV12))
