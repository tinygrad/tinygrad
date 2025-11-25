import dataclasses, enum, argparse, os, itertools
from typing import Any
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import DEBUG, EncDecCtx, HEVCFrameCtx, HEVCRawSPS, HEVCRawPPS, HEVCRawSlice
from tinygrad.nn.state import TensorIO
from tinygrad.runtime.autogen import avcodec

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

class HEVCDecoderState:
  def __init__(self):
    self.dpb:list[tuple[int, Tensor]] = []  # (poc, frame)
    self.next_frame = itertools.count()

  def profile_tier_level(self, r:BitReader, enable:bool, max_sub_layers:int):
    assert enable and max_sub_layers == 0, "no sublayers supported"
    self._notimpl_profile_tier_level = r.u(88)
    self.general_level_idc = r.u(8)

  def seq_parameter_set_rbsp(self, r:BitReader):
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
    self.pps_num_short_term_ref_pic_sets = r.ue_v()
    assert self.pps_num_short_term_ref_pic_sets == 0, "ref pic sets parsing not implemented"
    self.pps_long_term_ref_pics_present_flag = r.u(1)
    if self.pps_long_term_ref_pics_present_flag: assert False, "long_term_ref_pics parsing not implemented"
    self.sps_temporal_mvp_enabled_flag = r.u(1)
    self.strong_intra_smoothing_enabled_flag = r.u(1)

    self.sps = HEVCRawSPS(chroma_format_idc=self.chroma_format_idc, pic_width_in_luma_samples=self.pic_width_in_luma_samples,
      pic_height_in_luma_samples=self.pic_height_in_luma_samples, bit_depth_luma=self.bit_depth_luma, bit_depth_chroma=self.bit_depth_chroma,
      log2_max_pic_order_cnt_lsb_minus4=self.log2_max_pic_order_cnt_lsb_minus4, log2_min_luma_coding_block_size=self.log2_min_luma_coding_block_size,
      log2_max_luma_coding_block_size=self.log2_max_luma_coding_block_size, log2_min_transform_block_size=self.log2_min_transform_block_size,
      log2_max_transform_block_size=self.log2_max_transform_block_size, amp_enabled_flag=self.amp_enabled_flag,pcm_enabled_flag=self.pcm_enabled_flag,
      sample_adaptive_offset_enabled_flag=self.sample_adaptive_offset_enabled_flag, sps_temporal_mvp_enabled_flag=self.sps_temporal_mvp_enabled_flag,
      strong_intra_smoothing_enabled_flag=self.strong_intra_smoothing_enabled_flag)

  def pic_parameter_set_rbsp(self, r:BitReader):
    self.pps_pic_parameter_set_id = r.ue_v()
    self.pps_seq_parameter_set_id = r.ue_v()
    self.dependent_slice_segments_enabled_flag = r.u(1)
    self.pps_output_flag_present_flag = r.u(1)
    self.pps_num_extra_slice_header_bits = r.u(3)
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
    if self.tiles_enabled_flag: assert False, "tiles parsing not implemented"
    self.loop_filter_across_slices_enabled_flag = r.u(1)
    self.deblocking_filter_control_present_flag = r.u(1)
    if self.deblocking_filter_control_present_flag: assert False, "deblocking_filter parsing not implemented"
    self.scaling_list_data_present_flag = r.u(1)
    if self.scaling_list_data_present_flag: assert False, "scaling_list_data parsing not implemented"
    self.lists_modification_present_flag = r.u(1)
    self.log2_parallel_merge_level = r.ue_v() + 2

    self.pps = HEVCRawPPS(sign_data_hiding_enabled_flag=self.sign_data_hiding_enabled_flag,
      num_ref_idx_l0_default_active=self.num_ref_idx_l0_default_active, num_ref_idx_l1_default_active=self.num_ref_idx_l1_default_active,
      init_qp=self.init_qp, cu_qp_delta_enabled_flag=self.cu_qp_delta_enabled_flag, diff_cu_qp_delta_depth=getattr(self, 'diff_cu_qp_delta_depth', 0),
      pps_cb_qp_offset=self.pps_cb_qp_offset, pps_cr_qp_offset=self.pps_cr_qp_offset, 
      pps_slice_chroma_qp_offsets_present_flag=self.pps_slice_chroma_qp_offsets_present_flag, weighted_pred_flag=self.weighted_pred_flag,
      weighted_bipred_flag=self.weighted_bipred_flag, transquant_bypass_enabled_flag=self.transquant_bypass_enabled_flag,
      tiles_enabled_flag=self.tiles_enabled_flag, entropy_coding_sync_enabled_flag=self.entropy_coding_sync_enabled_flag,
      loop_filter_across_slices_enabled_flag=self.loop_filter_across_slices_enabled_flag,
      deblocking_filter_control_present_flag=self.deblocking_filter_control_present_flag,
      scaling_list_data_present_flag=self.scaling_list_data_present_flag, lists_modification_present_flag=self.lists_modification_present_flag,
      log2_parallel_merge_level=self.log2_parallel_merge_level)

  def st_ref_pic_set(self, reader, stRpsIdx):
    if stRpsIdx != 0: assert False, "st_ref_pic_set parsing not implemented"
    else:
      num_negative_pics = reader.ue_v()
      num_positive_pics = reader.ue_v()
      for i in range(num_negative_pics):
        delta_poc_s0_minus1 = reader.ue_v()
        used_by_curr_pic_s0_flag = reader.u(1)
      for i in range(num_positive_pics):
        delta_poc_s1_minus1 = reader.ue_v()
        used_by_curr_pic_s1_flag = reader.u(1)

  def slice_fragment_header(self, nal_unit_type, r:BitReader):
    self.first_slice_segment_in_pic_flag = r.u(1)
    if nal_unit_type >= avcodec.HEVC_NAL_BLA_W_LP and nal_unit_type <= avcodec.HEVC_NAL_RSV_IRAP_VCL23:
      self.no_output_of_prior_pics_flag = r.u(1)
    self.slice_pic_parameter_set_id = r.ue_v()
    if not self.first_slice_segment_in_pic_flag: assert False, "dependent slices not implemented"
    self.dependent_slice_segment_flag = 0
    if not self.dependent_slice_segment_flag:
      r.u(self.pps_num_extra_slice_header_bits) # extra bits ignored
      self.slice_type = r.ue_v()

      self.sw_skip_start = r.read_bits - r.current_bits
      self.pic_output_flag = r.u(1) if self.pps_output_flag_present_flag else 0
      self.colour_plane_id = r.u(2) if self.separate_colour_plane_flag else 0

      if nal_unit_type != avcodec.HEVC_NAL_IDR_W_RADL and nal_unit_type != avcodec.HEVC_NAL_IDR_N_LP:
        self.slice_pic_order_cnt_lsb = r.u(self.log2_max_pic_order_cnt_lsb_minus4 + 4)

        self.short_term_ref_pic_set_sps_flag = r.u(1)
        if not self.short_term_ref_pic_set_sps_flag:
          self.short_term_ref_pics_in_slice_start = r.read_bits - r.current_bits
          self.st_ref_pic_set(r, self.pps_num_short_term_ref_pic_sets)
          self.short_term_ref_pics_in_slice_end = r.read_bits - r.current_bits
        elif self.pps_num_short_term_ref_pic_sets > 1: assert False, "short_term_ref_pic_set parsing not implemented"

        if self.pps_long_term_ref_pics_present_flag: assert False, "long_term_ref_pics parsing not implemented"

        self.sw_skip_end = r.read_bits - r.current_bits
        self.slice_temporal_mvp_enabled_flag = r.u(1) if self.sps_temporal_mvp_enabled_flag else 0
      else: self.slice_pic_order_cnt_lsb, self.sw_skip_end = 0, self.sw_skip_start

      if self.sample_adaptive_offset_enabled_flag:
        slice_sao_luma_flag = r.u(1)
        ChromaArrayType = self.chroma_format_idc if self.separate_colour_plane_flag == 0 else 0
        slice_sao_chroma_flag = r.u(1) if ChromaArrayType != 0 else 0
      
      if self.slice_type in {avcodec.HEVC_SLICE_B, avcodec.HEVC_SLICE_B}:
        if num_ref_idx_active_override_flag := r.u(1):
          num_ref_idx_l0_active_minus1 = r.ue_v()
          num_ref_idx_l1_active_minus1 = r.ue_v() if self.slice_type == avcodec.HEVC_SLICE_B else 0

  def free_pic_index(self):
    for i in range(16):
      if all(dpb_entry[0] != i for dpb_entry in self.dpb): return i

  def state_for_slice(self, nal_unit_type, r:BitReader) -> EncDecCtx:
    self.slice_fragment_header(nal_unit_type, r)
  
    # Update history
    if nal_unit_type in {avcodec.HEVC_NAL_IDR_W_RADL, avcodec.HEVC_NAL_IDR_N_LP}:
      self.dpb = [] # reset DPB on IDR

    diff_poc = [0] * 16
    ref_idx_l0 = [0] * 16
    ref_idx_l1 = [0] * 16
    self.cur_pic_idx = self.free_pic_index()
    hist = [x[2] for x in self.dpb]
    hist_order = tuple([x[0] for x in self.dpb])

    before_list, after_list = [], []
    for pic_idx, poc, _ in self.dpb:
      diff_poc[pic_idx] = self.slice_pic_order_cnt_lsb - poc

      if self.slice_pic_order_cnt_lsb < poc: after_list.append((poc - self.slice_pic_order_cnt_lsb, pic_idx))
      else: before_list.append((self.slice_pic_order_cnt_lsb - poc, pic_idx))

    before_list.sort()
    after_list.sort()

    for i, (_, pic_idx) in enumerate(before_list + after_list): ref_idx_l0[i] = pic_idx
    if self.slice_type == avcodec.HEVC_SLICE_B:
      for i, (_, pic_idx) in enumerate(after_list + before_list): ref_idx_l1[i] = pic_idx

    frame_info = HEVCRawSlice(idr=nal_unit_type in {avcodec.HEVC_NAL_IDR_W_RADL, avcodec.HEVC_NAL_IDR_N_LP},
      rap=nal_unit_type >= avcodec.HEVC_NAL_BLA_W_LP and nal_unit_type <= avcodec.HEVC_NAL_RSV_IRAP_VCL23, pic_idx=self.cur_pic_idx,
      sw_hdr_skip=self.sw_skip_end - self.sw_skip_start, diff_poc=tuple(diff_poc), ref_idx_l0=tuple(ref_idx_l0), ref_idx_l1=tuple(ref_idx_l1))

    return EncDecCtx(hevc=HEVCFrameCtx(idx=next(self.next_frame), hist_order=hist_order, sps=self.sps, pps=self.pps, frame=frame_info)), hist

  def add_ref_frame(self, nal_unit_type, img):
    if nal_unit_type in {avcodec.HEVC_NAL_TRAIL_R, avcodec.HEVC_NAL_IDR_N_LP}:
      self.dpb.append((self.cur_pic_idx, self.slice_pic_order_cnt_lsb, img))

    if len(self.dpb) >= self.sps_max_dec_pic_buffering[0]:
      # remove the oldest poc
      self.dpb.pop(0)

class HEVCDecoder:
  def __init__(self, t:Tensor):
    self.io = TensorIO(t)
    self.decoder = HEVCDecoderState()

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

      if nal_unit_type == avcodec.HEVC_NAL_SPS: self.decoder.seq_parameter_set_rbsp(BitReader(_hevc_get_rbsp(dat, off=2)))
      elif nal_unit_type == avcodec.HEVC_NAL_PPS: self.decoder.pic_parameter_set_rbsp(BitReader(_hevc_get_rbsp(dat, off=2)))
      elif nal_unit_type in {avcodec.HEVC_NAL_IDR_N_LP, avcodec.HEVC_NAL_TRAIL_R, avcodec.HEVC_NAL_TRAIL_N}:
        ctx, ref_frames = self.decoder.state_for_slice(nal_unit_type, BitReader(dat[2:]))
        size = 0x3fe000 #ctx.hevc.sps.pic_width_in_luma_samples * (ctx.hevc.sps.pic_height_in_luma_samples+ctx.hevc.sps.pic_height_in_luma_samples//2)
        img = Tensor.from_hevc(self.io._tensor[nal_unit_start:nal_unit_start+nal_unit_len], ref_frames, size=size, ctx=ctx, device=Device.DEFAULT)
        self.decoder.add_ref_frame(nal_unit_type, img)
        yield img
        # exit(0)

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
    if i > 200:
      img1 = bytes(src[_addr_table(1080, 1920)].reshape(-1).cat(src[_addr_table(540, 1920) + 0x1fe000].reshape(-1)).tolist())
      w, h, ch = 1920, 1080, 540
      total = h + ch
      frame = np.frombuffer(img1, dtype=np.uint8).reshape((total, w))
      cv2.imwrite(f"extra/nvdec/out/nvp_frame_{i}.png", cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12))
