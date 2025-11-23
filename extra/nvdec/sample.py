import pathlib, argparse, ctypes, numpy as np, cv2, itertools
from extra.nvdec.hevc import NAL_UNIT_START_CODE, NAL_UNIT_START_CODE_SIZE, NAL_UNIT_HEADER_SIZE, HevcNalUnitType, BitReader, HevcSliceType
from tinygrad.runtime.ops_nv import NVCommandQueue, NVVideoQueue, nv_gpu, BufferSpec
from tinygrad.helpers import DEBUG, round_up
from tinygrad import Tensor, dtypes, Device

from extra.nv_gpu_driver.nv_ioctl import _dump_gpfifo, dump_struct_ext

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

class NVVidCtx:
  def __init__(self, dev):
    self.dev = dev
    self.buf_in_data = dev.allocator.alloc(0x3fc000, BufferSpec(cpu_access=True))
    self.luma_chroma_bufs = [dev.allocator.alloc(0x2fd000, BufferSpec(cpu_access=True)) for i in range(6)]
    self.coloc_buf = dev.allocator.alloc(0x400000)
    self.filter_buf = dev.allocator.alloc(0xa00000)
    self.scaling_list = dev.allocator.alloc(0x1000, BufferSpec(cpu_access=True))
    self.tile_sizes = dev.allocator.alloc(0x1000, BufferSpec(cpu_access=True))
    self.status_buf = dev.allocator.alloc(0x1000, BufferSpec(cpu_access=True))
    self.status_desc = nv_gpu.nvdec_status_s.from_address(self.status_buf.cpu_view().addr)

    self.tile_cnt = 63*16
    self.scaling_list.cpu_view()[:self.tile_cnt] = bytes([0x10]*self.tile_cnt)
    self.scaling_list.cpu_view()[8:16] = bytes([0x0]*8)

    # 1E 00 11 00 00 00 00 00  00 00 00 00 00 00 00 00
    self.tile_sizes.cpu_view()[:4] = bytes([0x1e,0x0,0x11,0x0])

    self.pic_desc = dev.allocator.alloc(0x1000, BufferSpec(cpu_access=True))

class NVVidDecoder:
  def __init__(self, dev):
    self.ctx = NVVidCtx(dev)
    self.desc = nv_gpu.nvdec_hevc_pic_s(gptimer_timeout_value=81600000, tileformat=1, sw_start_code_e=1)
    self.pic_counter = itertools.count()
    self.dpb:list[tuple[int, int, int]] = []  # (nv_pic_idx, poc, is_long_term)

  def free_pic_index(self):
    for i in range(16):
      if all(dpb_entry[0] != i for dpb_entry in self.dpb): return i

  def _hevc_st_ref_pic_set(self, reader, stRpsIdx):
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

  def stream_slice(self, nal_unit_type, data):
    if nal_unit_type == HevcNalUnitType.SPS_NUT:
      reader = BitReader(hevc_get_rbsp(data, off=5))
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
      self.sps_separate_colour_plane_flag = reader.u(1) if self.desc.chroma_format_idc == 3 else 0
      self.desc.pic_width_in_luma_samples = reader.ue_v()
      self.desc.pic_height_in_luma_samples = reader.ue_v()
      self.desc.framestride[0] = self.desc.pic_width_in_luma_samples
      self.desc.framestride[1] = self.desc.pic_width_in_luma_samples
      self.desc.colMvBuffersize = (round_up(self.desc.pic_width_in_luma_samples, 64) * round_up(self.desc.pic_height_in_luma_samples, 64) // 16) // 256
      # self.desc.HevcSaoBufferOffset=2584
      # self.desc.HevcBsdCtrlOffset=23256
      if conformance_window_flag := reader.u(1):
        conf_win_left_offset = reader.ue_v()
        conf_win_right_offset = reader.ue_v()
        conf_win_top_offset = reader.ue_v()
        conf_win_bottom_offset = reader.ue_v()
      self.desc.bit_depth_luma = reader.ue_v() + 8
      self.desc.bit_depth_chroma = reader.ue_v() + 8
      self.desc.log2_max_pic_order_cnt_lsb_minus4 = reader.ue_v()
      sps_sub_layer_ordering_info_present_flag = reader.u(1)
      self.sps_max_dec_pic_buffering = []
      for i in range((0 if sps_sub_layer_ordering_info_present_flag else sps_max_sub_layers_minus1), sps_max_sub_layers_minus1 + 1):
        self.sps_max_dec_pic_buffering.append(reader.ue_v() + 1)
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
      self.pps_num_short_term_ref_pic_sets = reader.ue_v()
      assert self.pps_num_short_term_ref_pic_sets == 0, "ref pic sets parsing not implemented"
      self.pps_long_term_ref_pics_present_flag = reader.u(1)
      if self.pps_long_term_ref_pics_present_flag: assert False, "long_term_ref_pics parsing not implemented"
      self.desc.sps_temporal_mvp_enabled_flag = reader.u(1)
      self.desc.strong_intra_smoothing_enabled_flag = reader.u(1)
      # not parsed any further
    elif nal_unit_type == HevcNalUnitType.PPS_NUT:
      reader = BitReader(hevc_get_rbsp(data, off=5))
      pps_pic_parameter_set_id = reader.ue_v()
      pps_seq_parameter_set_id = reader.ue_v()
      dependent_slice_segments_enabled_flag = reader.u(1)
      self.pps_output_flag_present_flag = reader.u(1)
      self.pps_num_extra_slice_header_bits = reader.u(3)
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
      self.desc.curr_pic_idx = self.free_pic_index()
      global_pic_idx = next(self.pic_counter)

      desc = bytes(self.desc)
      self.ctx.pic_desc.cpu_view()[:len(desc)] = desc
      self.ctx.buf_in_data.cpu_view()[:len(data)] = data

      _dump_gpfifo(f"before decode {self.desc.curr_pic_idx}", iden=True)
      print("Decoding IDR")
      NVVideoQueue().wait(self.ctx.dev.timeline_signal, self.ctx.dev.timeline_value - 1) \
                    .decode_hevc_chunk(self.ctx.buf_in_data, self.ctx.luma_chroma_bufs, self.ctx.coloc_buf, self.ctx.filter_buf,
                                       self.ctx.scaling_list, self.ctx.tile_sizes, self.ctx.status_buf, self.ctx.pic_desc, global_pic_idx) \
                    .signal(self.ctx.dev.timeline_signal, self.ctx.dev.next_timeline()).submit(self.ctx.dev)
      _dump_gpfifo(f"aft decode {self.desc.curr_pic_idx}", iden=True)
      
      self.ctx.dev.synchronize()

      self.dpb = [(self.desc.curr_pic_idx, 0, 0)]  # reset DPB on IDR

      return self.ctx.luma_chroma_bufs[self.desc.curr_pic_idx]
    elif nal_unit_type in {HevcNalUnitType.TRAIL_R, HevcNalUnitType.TRAIL_N}:
      global_pic_idx = next(self.pic_counter)
      
      reader = BitReader(hevc_get_rbsp(data, off=5))
      first_slice_segment_in_pic_flag = reader.u(1)
      if nal_unit_type >= HevcNalUnitType.BLA_W_LP and nal_unit_type <= HevcNalUnitType.RSV_IRAP_VCL23:
        no_output_of_prior_pics_flag = reader.u(1)
      slice_pic_parameter_set_id = reader.ue_v()
      if not first_slice_segment_in_pic_flag: assert False, "dependent slices not implemented"
      dependent_slice_segment_flag = 0
      if not dependent_slice_segment_flag:
        reader.u(self.pps_num_extra_slice_header_bits) # extra bits ignored
        slice_type = reader.ue_v()

        skip_start = reader.read_bits - reader.current_bits
        pic_output_flag = reader.u(1) if self.pps_output_flag_present_flag else 0
        colour_plane_id = reader.u(2) if self.sps_separate_colour_plane_flag else 0

        if nal_unit_type != HevcNalUnitType.IDR_W_RADL and nal_unit_type != HevcNalUnitType.IDR_N_LP:
          slice_pic_order_cnt_lsb = reader.u(self.desc.log2_max_pic_order_cnt_lsb_minus4 + 4)

          short_term_ref_pic_set_sps_flag = reader.u(1)
          if not short_term_ref_pic_set_sps_flag:
            short_term_ref_pics_in_slice_start = reader.read_bits - reader.current_bits
            self._hevc_st_ref_pic_set(reader, self.pps_num_short_term_ref_pic_sets)
            short_term_ref_pics_in_slice_end = reader.read_bits - reader.current_bits
          elif self.pps_num_short_term_ref_pic_sets > 1: assert False, "short_term_ref_pic_set parsing not implemented"

          if self.pps_long_term_ref_pics_present_flag: assert False, "long_term_ref_pics parsing not implemented"

          skip_end = reader.read_bits - reader.current_bits
          slice_temporal_mvp_enabled_flag = reader.u(1) if self.desc.sps_temporal_mvp_enabled_flag else 0
        
        if self.desc.sample_adaptive_offset_enabled_flag:
          slice_sao_luma_flag = reader.u(1)
          ChromaArrayType = self.desc.chroma_format_idc if self.sps_separate_colour_plane_flag == 0 else 0
          slice_sao_chroma_flag = reader.u(1) if ChromaArrayType != 0 else 0
        
        if slice_type in {HevcSliceType.P, HevcSliceType.B}:
          if num_ref_idx_active_override_flag := reader.u(1):
            num_ref_idx_l0_active_minus1 = reader.ue_v()
            num_ref_idx_l1_active_minus1 = reader.ue_v() if slice_type == HevcSliceType.B else 0

          ## TODO: to be continud

        # print('l0-l1', num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1)

        # print('slice_pic_order_cnt_lsb', slice_pic_order_cnt_lsb)
        # print(reader.read_bits - reader.current_bits)

      self.desc.IDR_picture_flag = 0
      self.desc.RAP_picture_flag = 0
      self.desc.pattern_id = 2
      self.desc.stream_len = len(data)
      self.desc.curr_pic_idx = self.free_pic_index()
      self.desc.num_ref_frames = len(self.dpb)

      self.desc.num_bits_short_term_ref_pics_in_slice = short_term_ref_pics_in_slice_end - short_term_ref_pics_in_slice_start
      self.desc.sw_hdr_skip_length = skip_end - skip_start
      self.desc.loop_filter_across_tiles_enabled_flag = 1
      self.desc.v3.slice_ec_mv_type = 1
      # self.desc.v1.hevc_main10_444_ext.HevcFltAboveOffset = 23902
      # self.desc.v1.hevc_main10_444_ext.HevcSaoAboveOffset = 32130
      # self.desc.v1.hevc_main10_444_ext.log2MaxTransformSkipSize = 2
      # self.desc.v3.HevcSliceEdgeOffset = 32130

      for i in range(16): self.desc.RefDiffPicOrderCnts[i] = 0
      
      before_list, after_list = [], []
      for pic_idx, poc, is_long_term in self.dpb:
        self.desc.RefDiffPicOrderCnts[pic_idx] = slice_pic_order_cnt_lsb - poc
        if slice_pic_order_cnt_lsb < poc: after_list.append((poc - slice_pic_order_cnt_lsb, pic_idx))
        else: before_list.append((slice_pic_order_cnt_lsb - poc, pic_idx))

      before_list.sort()
      after_list.sort()

      for i, (_, pic_idx) in enumerate(before_list + after_list): self.desc.initreflistidxl0[i] = pic_idx
      if slice_type == HevcSliceType.B:
        for i, (_, pic_idx) in enumerate(after_list + before_list): self.desc.initreflistidxl1[i] = pic_idx

      desc = bytes(self.desc)
      self.ctx.pic_desc.cpu_view()[:len(desc)] = desc
      self.ctx.buf_in_data.cpu_view()[:len(data)] = data

      _dump_gpfifo(f"before decode {self.desc.curr_pic_idx}", iden=True)
      # print("Decoding non-IDR")
      NVVideoQueue().wait(self.ctx.dev.timeline_signal, self.ctx.dev.timeline_value - 1) \
                    .decode_hevc_chunk(self.ctx.buf_in_data, self.ctx.luma_chroma_bufs, self.ctx.coloc_buf, self.ctx.filter_buf,
                                       self.ctx.scaling_list, self.ctx.tile_sizes, self.ctx.status_buf, self.ctx.pic_desc, global_pic_idx) \
                    .signal(self.ctx.dev.timeline_signal, self.ctx.dev.next_timeline()).submit(self.ctx.dev)
      _dump_gpfifo(f"aft decode {self.desc.curr_pic_idx}", iden=True)
      self.ctx.dev.synchronize()

      if nal_unit_type == HevcNalUnitType.TRAIL_R:
        self.dpb.append((self.desc.curr_pic_idx, slice_pic_order_cnt_lsb, 0))

      if len(self.dpb) >= self.sps_max_dec_pic_buffering[0]:
        # remove the oldest poc
        self.dpb.pop(0)

      return self.ctx.luma_chroma_bufs[self.desc.curr_pic_idx]

def hevc_get_rbsp(dat:bytes, off=0) -> bytes:
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

def hevc_decode(dat:bytes, nal_unit_start=0):
  next_img_counter = 0
  decoder = NVVidDecoder(Device.default)

  while nal_unit_start < len(dat):
    assert dat[nal_unit_start:nal_unit_start + NAL_UNIT_START_CODE_SIZE] == NAL_UNIT_START_CODE

    pos = dat.find(NAL_UNIT_START_CODE, nal_unit_start + NAL_UNIT_START_CODE_SIZE)
    nal_unit_len = (pos if pos != -1 else len(dat)) - nal_unit_start

    # 7.3.1.1 General NAL unit syntax
    nal_unit_type = (dat[nal_unit_start + NAL_UNIT_START_CODE_SIZE] >> 1) & 0x3F
    # rbsp = hevc_get_rbsp()

    img = decoder.stream_slice(nal_unit_type, dat[nal_unit_start:nal_unit_start+nal_unit_len])
    if img is not None:
      img1 = untile(img)
      w, h, ch = 1920, 1080, 540
      total = h + ch
      frame = np.frombuffer(img1, dtype=np.uint8).reshape((total, w))
      cv2.imwrite(f"extra/nvdec/out/nvn_frame_{next_img_counter}.png", cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12))
      next_img_counter += 1
      # if next_img_counter == 2: exit(0)

    # if DEBUG >= 4: print(f"NAL unit type: {nal_unit_type}, length: {nal_unit_len}")
    nal_unit_start += nal_unit_len

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input_file", type=str)
  args = parser.parse_args()

  with pathlib.Path(args.input_file).open("rb") as f: dat = f.read()
  frame_types, dat_len, prefix_dat = hevc_decode(dat[1:])

  with open(args.output_prefix_file, "wb") as f:
    f.write(prefix_dat)

  with open(args.output_index_file, "wb") as f:
    for ft, fp in frame_types:
      f.write(struct.pack("<II", ft, fp))
    f.write(struct.pack("<II", 0xFFFFFFFF, dat_len))

