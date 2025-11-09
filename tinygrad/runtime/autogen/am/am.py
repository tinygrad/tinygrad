# mypy: ignore-errors
import ctypes
from tinygrad.helpers import Struct, CEnum, _IO, _IOW, _IOR, _IOWR, unwrap
class struct_v11_gfx_mqd(Struct): pass
struct_v11_gfx_mqd._fields_ = [
  ('shadow_base_lo', ctypes.c_uint),
  ('shadow_base_hi', ctypes.c_uint),
  ('gds_bkup_base_lo', ctypes.c_uint),
  ('gds_bkup_base_hi', ctypes.c_uint),
  ('fw_work_area_base_lo', ctypes.c_uint),
  ('fw_work_area_base_hi', ctypes.c_uint),
  ('shadow_initialized', ctypes.c_uint),
  ('ib_vmid', ctypes.c_uint),
  ('reserved_8', ctypes.c_uint),
  ('reserved_9', ctypes.c_uint),
  ('reserved_10', ctypes.c_uint),
  ('reserved_11', ctypes.c_uint),
  ('reserved_12', ctypes.c_uint),
  ('reserved_13', ctypes.c_uint),
  ('reserved_14', ctypes.c_uint),
  ('reserved_15', ctypes.c_uint),
  ('reserved_16', ctypes.c_uint),
  ('reserved_17', ctypes.c_uint),
  ('reserved_18', ctypes.c_uint),
  ('reserved_19', ctypes.c_uint),
  ('reserved_20', ctypes.c_uint),
  ('reserved_21', ctypes.c_uint),
  ('reserved_22', ctypes.c_uint),
  ('reserved_23', ctypes.c_uint),
  ('reserved_24', ctypes.c_uint),
  ('reserved_25', ctypes.c_uint),
  ('reserved_26', ctypes.c_uint),
  ('reserved_27', ctypes.c_uint),
  ('reserved_28', ctypes.c_uint),
  ('reserved_29', ctypes.c_uint),
  ('reserved_30', ctypes.c_uint),
  ('reserved_31', ctypes.c_uint),
  ('reserved_32', ctypes.c_uint),
  ('reserved_33', ctypes.c_uint),
  ('reserved_34', ctypes.c_uint),
  ('reserved_35', ctypes.c_uint),
  ('reserved_36', ctypes.c_uint),
  ('reserved_37', ctypes.c_uint),
  ('reserved_38', ctypes.c_uint),
  ('reserved_39', ctypes.c_uint),
  ('reserved_40', ctypes.c_uint),
  ('reserved_41', ctypes.c_uint),
  ('reserved_42', ctypes.c_uint),
  ('reserved_43', ctypes.c_uint),
  ('reserved_44', ctypes.c_uint),
  ('reserved_45', ctypes.c_uint),
  ('reserved_46', ctypes.c_uint),
  ('reserved_47', ctypes.c_uint),
  ('reserved_48', ctypes.c_uint),
  ('reserved_49', ctypes.c_uint),
  ('reserved_50', ctypes.c_uint),
  ('reserved_51', ctypes.c_uint),
  ('reserved_52', ctypes.c_uint),
  ('reserved_53', ctypes.c_uint),
  ('reserved_54', ctypes.c_uint),
  ('reserved_55', ctypes.c_uint),
  ('reserved_56', ctypes.c_uint),
  ('reserved_57', ctypes.c_uint),
  ('reserved_58', ctypes.c_uint),
  ('reserved_59', ctypes.c_uint),
  ('reserved_60', ctypes.c_uint),
  ('reserved_61', ctypes.c_uint),
  ('reserved_62', ctypes.c_uint),
  ('reserved_63', ctypes.c_uint),
  ('reserved_64', ctypes.c_uint),
  ('reserved_65', ctypes.c_uint),
  ('reserved_66', ctypes.c_uint),
  ('reserved_67', ctypes.c_uint),
  ('reserved_68', ctypes.c_uint),
  ('reserved_69', ctypes.c_uint),
  ('reserved_70', ctypes.c_uint),
  ('reserved_71', ctypes.c_uint),
  ('reserved_72', ctypes.c_uint),
  ('reserved_73', ctypes.c_uint),
  ('reserved_74', ctypes.c_uint),
  ('reserved_75', ctypes.c_uint),
  ('reserved_76', ctypes.c_uint),
  ('reserved_77', ctypes.c_uint),
  ('reserved_78', ctypes.c_uint),
  ('reserved_79', ctypes.c_uint),
  ('reserved_80', ctypes.c_uint),
  ('reserved_81', ctypes.c_uint),
  ('reserved_82', ctypes.c_uint),
  ('reserved_83', ctypes.c_uint),
  ('checksum_lo', ctypes.c_uint),
  ('checksum_hi', ctypes.c_uint),
  ('cp_mqd_query_time_lo', ctypes.c_uint),
  ('cp_mqd_query_time_hi', ctypes.c_uint),
  ('reserved_88', ctypes.c_uint),
  ('reserved_89', ctypes.c_uint),
  ('reserved_90', ctypes.c_uint),
  ('reserved_91', ctypes.c_uint),
  ('cp_mqd_query_wave_count', ctypes.c_uint),
  ('cp_mqd_query_gfx_hqd_rptr', ctypes.c_uint),
  ('cp_mqd_query_gfx_hqd_wptr', ctypes.c_uint),
  ('cp_mqd_query_gfx_hqd_offset', ctypes.c_uint),
  ('reserved_96', ctypes.c_uint),
  ('reserved_97', ctypes.c_uint),
  ('reserved_98', ctypes.c_uint),
  ('reserved_99', ctypes.c_uint),
  ('reserved_100', ctypes.c_uint),
  ('reserved_101', ctypes.c_uint),
  ('reserved_102', ctypes.c_uint),
  ('reserved_103', ctypes.c_uint),
  ('control_buf_addr_lo', ctypes.c_uint),
  ('control_buf_addr_hi', ctypes.c_uint),
  ('disable_queue', ctypes.c_uint),
  ('reserved_107', ctypes.c_uint),
  ('reserved_108', ctypes.c_uint),
  ('reserved_109', ctypes.c_uint),
  ('reserved_110', ctypes.c_uint),
  ('reserved_111', ctypes.c_uint),
  ('reserved_112', ctypes.c_uint),
  ('reserved_113', ctypes.c_uint),
  ('reserved_114', ctypes.c_uint),
  ('reserved_115', ctypes.c_uint),
  ('reserved_116', ctypes.c_uint),
  ('reserved_117', ctypes.c_uint),
  ('reserved_118', ctypes.c_uint),
  ('reserved_119', ctypes.c_uint),
  ('reserved_120', ctypes.c_uint),
  ('reserved_121', ctypes.c_uint),
  ('reserved_122', ctypes.c_uint),
  ('reserved_123', ctypes.c_uint),
  ('reserved_124', ctypes.c_uint),
  ('reserved_125', ctypes.c_uint),
  ('reserved_126', ctypes.c_uint),
  ('reserved_127', ctypes.c_uint),
  ('cp_mqd_base_addr', ctypes.c_uint),
  ('cp_mqd_base_addr_hi', ctypes.c_uint),
  ('cp_gfx_hqd_active', ctypes.c_uint),
  ('cp_gfx_hqd_vmid', ctypes.c_uint),
  ('reserved_131', ctypes.c_uint),
  ('reserved_132', ctypes.c_uint),
  ('cp_gfx_hqd_queue_priority', ctypes.c_uint),
  ('cp_gfx_hqd_quantum', ctypes.c_uint),
  ('cp_gfx_hqd_base', ctypes.c_uint),
  ('cp_gfx_hqd_base_hi', ctypes.c_uint),
  ('cp_gfx_hqd_rptr', ctypes.c_uint),
  ('cp_gfx_hqd_rptr_addr', ctypes.c_uint),
  ('cp_gfx_hqd_rptr_addr_hi', ctypes.c_uint),
  ('cp_rb_wptr_poll_addr_lo', ctypes.c_uint),
  ('cp_rb_wptr_poll_addr_hi', ctypes.c_uint),
  ('cp_rb_doorbell_control', ctypes.c_uint),
  ('cp_gfx_hqd_offset', ctypes.c_uint),
  ('cp_gfx_hqd_cntl', ctypes.c_uint),
  ('reserved_146', ctypes.c_uint),
  ('reserved_147', ctypes.c_uint),
  ('cp_gfx_hqd_csmd_rptr', ctypes.c_uint),
  ('cp_gfx_hqd_wptr', ctypes.c_uint),
  ('cp_gfx_hqd_wptr_hi', ctypes.c_uint),
  ('reserved_151', ctypes.c_uint),
  ('reserved_152', ctypes.c_uint),
  ('reserved_153', ctypes.c_uint),
  ('reserved_154', ctypes.c_uint),
  ('reserved_155', ctypes.c_uint),
  ('cp_gfx_hqd_mapped', ctypes.c_uint),
  ('cp_gfx_hqd_que_mgr_control', ctypes.c_uint),
  ('reserved_158', ctypes.c_uint),
  ('reserved_159', ctypes.c_uint),
  ('cp_gfx_hqd_hq_status0', ctypes.c_uint),
  ('cp_gfx_hqd_hq_control0', ctypes.c_uint),
  ('cp_gfx_mqd_control', ctypes.c_uint),
  ('reserved_163', ctypes.c_uint),
  ('reserved_164', ctypes.c_uint),
  ('reserved_165', ctypes.c_uint),
  ('reserved_166', ctypes.c_uint),
  ('reserved_167', ctypes.c_uint),
  ('reserved_168', ctypes.c_uint),
  ('reserved_169', ctypes.c_uint),
  ('cp_num_prim_needed_count0_lo', ctypes.c_uint),
  ('cp_num_prim_needed_count0_hi', ctypes.c_uint),
  ('cp_num_prim_needed_count1_lo', ctypes.c_uint),
  ('cp_num_prim_needed_count1_hi', ctypes.c_uint),
  ('cp_num_prim_needed_count2_lo', ctypes.c_uint),
  ('cp_num_prim_needed_count2_hi', ctypes.c_uint),
  ('cp_num_prim_needed_count3_lo', ctypes.c_uint),
  ('cp_num_prim_needed_count3_hi', ctypes.c_uint),
  ('cp_num_prim_written_count0_lo', ctypes.c_uint),
  ('cp_num_prim_written_count0_hi', ctypes.c_uint),
  ('cp_num_prim_written_count1_lo', ctypes.c_uint),
  ('cp_num_prim_written_count1_hi', ctypes.c_uint),
  ('cp_num_prim_written_count2_lo', ctypes.c_uint),
  ('cp_num_prim_written_count2_hi', ctypes.c_uint),
  ('cp_num_prim_written_count3_lo', ctypes.c_uint),
  ('cp_num_prim_written_count3_hi', ctypes.c_uint),
  ('reserved_186', ctypes.c_uint),
  ('reserved_187', ctypes.c_uint),
  ('reserved_188', ctypes.c_uint),
  ('reserved_189', ctypes.c_uint),
  ('mp1_smn_fps_cnt', ctypes.c_uint),
  ('sq_thread_trace_buf0_base', ctypes.c_uint),
  ('sq_thread_trace_buf0_size', ctypes.c_uint),
  ('sq_thread_trace_buf1_base', ctypes.c_uint),
  ('sq_thread_trace_buf1_size', ctypes.c_uint),
  ('sq_thread_trace_wptr', ctypes.c_uint),
  ('sq_thread_trace_mask', ctypes.c_uint),
  ('sq_thread_trace_token_mask', ctypes.c_uint),
  ('sq_thread_trace_ctrl', ctypes.c_uint),
  ('sq_thread_trace_status', ctypes.c_uint),
  ('sq_thread_trace_dropped_cntr', ctypes.c_uint),
  ('sq_thread_trace_finish_done_debug', ctypes.c_uint),
  ('sq_thread_trace_gfx_draw_cntr', ctypes.c_uint),
  ('sq_thread_trace_gfx_marker_cntr', ctypes.c_uint),
  ('sq_thread_trace_hp3d_draw_cntr', ctypes.c_uint),
  ('sq_thread_trace_hp3d_marker_cntr', ctypes.c_uint),
  ('reserved_206', ctypes.c_uint),
  ('reserved_207', ctypes.c_uint),
  ('cp_sc_psinvoc_count0_lo', ctypes.c_uint),
  ('cp_sc_psinvoc_count0_hi', ctypes.c_uint),
  ('cp_pa_cprim_count_lo', ctypes.c_uint),
  ('cp_pa_cprim_count_hi', ctypes.c_uint),
  ('cp_pa_cinvoc_count_lo', ctypes.c_uint),
  ('cp_pa_cinvoc_count_hi', ctypes.c_uint),
  ('cp_vgt_vsinvoc_count_lo', ctypes.c_uint),
  ('cp_vgt_vsinvoc_count_hi', ctypes.c_uint),
  ('cp_vgt_gsinvoc_count_lo', ctypes.c_uint),
  ('cp_vgt_gsinvoc_count_hi', ctypes.c_uint),
  ('cp_vgt_gsprim_count_lo', ctypes.c_uint),
  ('cp_vgt_gsprim_count_hi', ctypes.c_uint),
  ('cp_vgt_iaprim_count_lo', ctypes.c_uint),
  ('cp_vgt_iaprim_count_hi', ctypes.c_uint),
  ('cp_vgt_iavert_count_lo', ctypes.c_uint),
  ('cp_vgt_iavert_count_hi', ctypes.c_uint),
  ('cp_vgt_hsinvoc_count_lo', ctypes.c_uint),
  ('cp_vgt_hsinvoc_count_hi', ctypes.c_uint),
  ('cp_vgt_dsinvoc_count_lo', ctypes.c_uint),
  ('cp_vgt_dsinvoc_count_hi', ctypes.c_uint),
  ('cp_vgt_csinvoc_count_lo', ctypes.c_uint),
  ('cp_vgt_csinvoc_count_hi', ctypes.c_uint),
  ('reserved_230', ctypes.c_uint),
  ('reserved_231', ctypes.c_uint),
  ('reserved_232', ctypes.c_uint),
  ('reserved_233', ctypes.c_uint),
  ('reserved_234', ctypes.c_uint),
  ('reserved_235', ctypes.c_uint),
  ('reserved_236', ctypes.c_uint),
  ('reserved_237', ctypes.c_uint),
  ('reserved_238', ctypes.c_uint),
  ('reserved_239', ctypes.c_uint),
  ('reserved_240', ctypes.c_uint),
  ('reserved_241', ctypes.c_uint),
  ('reserved_242', ctypes.c_uint),
  ('reserved_243', ctypes.c_uint),
  ('reserved_244', ctypes.c_uint),
  ('reserved_245', ctypes.c_uint),
  ('reserved_246', ctypes.c_uint),
  ('reserved_247', ctypes.c_uint),
  ('reserved_248', ctypes.c_uint),
  ('reserved_249', ctypes.c_uint),
  ('reserved_250', ctypes.c_uint),
  ('reserved_251', ctypes.c_uint),
  ('reserved_252', ctypes.c_uint),
  ('reserved_253', ctypes.c_uint),
  ('reserved_254', ctypes.c_uint),
  ('reserved_255', ctypes.c_uint),
  ('reserved_256', ctypes.c_uint),
  ('reserved_257', ctypes.c_uint),
  ('reserved_258', ctypes.c_uint),
  ('reserved_259', ctypes.c_uint),
  ('reserved_260', ctypes.c_uint),
  ('reserved_261', ctypes.c_uint),
  ('reserved_262', ctypes.c_uint),
  ('reserved_263', ctypes.c_uint),
  ('reserved_264', ctypes.c_uint),
  ('reserved_265', ctypes.c_uint),
  ('reserved_266', ctypes.c_uint),
  ('reserved_267', ctypes.c_uint),
  ('vgt_strmout_buffer_filled_size_0', ctypes.c_uint),
  ('vgt_strmout_buffer_filled_size_1', ctypes.c_uint),
  ('vgt_strmout_buffer_filled_size_2', ctypes.c_uint),
  ('vgt_strmout_buffer_filled_size_3', ctypes.c_uint),
  ('reserved_272', ctypes.c_uint),
  ('reserved_273', ctypes.c_uint),
  ('reserved_274', ctypes.c_uint),
  ('reserved_275', ctypes.c_uint),
  ('vgt_dma_max_size', ctypes.c_uint),
  ('vgt_dma_num_instances', ctypes.c_uint),
  ('reserved_278', ctypes.c_uint),
  ('reserved_279', ctypes.c_uint),
  ('reserved_280', ctypes.c_uint),
  ('reserved_281', ctypes.c_uint),
  ('reserved_282', ctypes.c_uint),
  ('reserved_283', ctypes.c_uint),
  ('reserved_284', ctypes.c_uint),
  ('reserved_285', ctypes.c_uint),
  ('reserved_286', ctypes.c_uint),
  ('reserved_287', ctypes.c_uint),
  ('it_set_base_ib_addr_lo', ctypes.c_uint),
  ('it_set_base_ib_addr_hi', ctypes.c_uint),
  ('reserved_290', ctypes.c_uint),
  ('reserved_291', ctypes.c_uint),
  ('reserved_292', ctypes.c_uint),
  ('reserved_293', ctypes.c_uint),
  ('reserved_294', ctypes.c_uint),
  ('reserved_295', ctypes.c_uint),
  ('reserved_296', ctypes.c_uint),
  ('reserved_297', ctypes.c_uint),
  ('reserved_298', ctypes.c_uint),
  ('reserved_299', ctypes.c_uint),
  ('reserved_300', ctypes.c_uint),
  ('reserved_301', ctypes.c_uint),
  ('reserved_302', ctypes.c_uint),
  ('reserved_303', ctypes.c_uint),
  ('reserved_304', ctypes.c_uint),
  ('reserved_305', ctypes.c_uint),
  ('reserved_306', ctypes.c_uint),
  ('reserved_307', ctypes.c_uint),
  ('reserved_308', ctypes.c_uint),
  ('reserved_309', ctypes.c_uint),
  ('reserved_310', ctypes.c_uint),
  ('reserved_311', ctypes.c_uint),
  ('reserved_312', ctypes.c_uint),
  ('reserved_313', ctypes.c_uint),
  ('reserved_314', ctypes.c_uint),
  ('reserved_315', ctypes.c_uint),
  ('reserved_316', ctypes.c_uint),
  ('reserved_317', ctypes.c_uint),
  ('reserved_318', ctypes.c_uint),
  ('reserved_319', ctypes.c_uint),
  ('reserved_320', ctypes.c_uint),
  ('reserved_321', ctypes.c_uint),
  ('reserved_322', ctypes.c_uint),
  ('reserved_323', ctypes.c_uint),
  ('reserved_324', ctypes.c_uint),
  ('reserved_325', ctypes.c_uint),
  ('reserved_326', ctypes.c_uint),
  ('reserved_327', ctypes.c_uint),
  ('reserved_328', ctypes.c_uint),
  ('reserved_329', ctypes.c_uint),
  ('reserved_330', ctypes.c_uint),
  ('reserved_331', ctypes.c_uint),
  ('reserved_332', ctypes.c_uint),
  ('reserved_333', ctypes.c_uint),
  ('reserved_334', ctypes.c_uint),
  ('reserved_335', ctypes.c_uint),
  ('reserved_336', ctypes.c_uint),
  ('reserved_337', ctypes.c_uint),
  ('reserved_338', ctypes.c_uint),
  ('reserved_339', ctypes.c_uint),
  ('reserved_340', ctypes.c_uint),
  ('reserved_341', ctypes.c_uint),
  ('reserved_342', ctypes.c_uint),
  ('reserved_343', ctypes.c_uint),
  ('reserved_344', ctypes.c_uint),
  ('reserved_345', ctypes.c_uint),
  ('reserved_346', ctypes.c_uint),
  ('reserved_347', ctypes.c_uint),
  ('reserved_348', ctypes.c_uint),
  ('reserved_349', ctypes.c_uint),
  ('reserved_350', ctypes.c_uint),
  ('reserved_351', ctypes.c_uint),
  ('reserved_352', ctypes.c_uint),
  ('reserved_353', ctypes.c_uint),
  ('reserved_354', ctypes.c_uint),
  ('reserved_355', ctypes.c_uint),
  ('spi_shader_pgm_rsrc3_ps', ctypes.c_uint),
  ('spi_shader_pgm_rsrc3_vs', ctypes.c_uint),
  ('spi_shader_pgm_rsrc3_gs', ctypes.c_uint),
  ('spi_shader_pgm_rsrc3_hs', ctypes.c_uint),
  ('spi_shader_pgm_rsrc4_ps', ctypes.c_uint),
  ('spi_shader_pgm_rsrc4_vs', ctypes.c_uint),
  ('spi_shader_pgm_rsrc4_gs', ctypes.c_uint),
  ('spi_shader_pgm_rsrc4_hs', ctypes.c_uint),
  ('db_occlusion_count0_low_00', ctypes.c_uint),
  ('db_occlusion_count0_hi_00', ctypes.c_uint),
  ('db_occlusion_count1_low_00', ctypes.c_uint),
  ('db_occlusion_count1_hi_00', ctypes.c_uint),
  ('db_occlusion_count2_low_00', ctypes.c_uint),
  ('db_occlusion_count2_hi_00', ctypes.c_uint),
  ('db_occlusion_count3_low_00', ctypes.c_uint),
  ('db_occlusion_count3_hi_00', ctypes.c_uint),
  ('db_occlusion_count0_low_01', ctypes.c_uint),
  ('db_occlusion_count0_hi_01', ctypes.c_uint),
  ('db_occlusion_count1_low_01', ctypes.c_uint),
  ('db_occlusion_count1_hi_01', ctypes.c_uint),
  ('db_occlusion_count2_low_01', ctypes.c_uint),
  ('db_occlusion_count2_hi_01', ctypes.c_uint),
  ('db_occlusion_count3_low_01', ctypes.c_uint),
  ('db_occlusion_count3_hi_01', ctypes.c_uint),
  ('db_occlusion_count0_low_02', ctypes.c_uint),
  ('db_occlusion_count0_hi_02', ctypes.c_uint),
  ('db_occlusion_count1_low_02', ctypes.c_uint),
  ('db_occlusion_count1_hi_02', ctypes.c_uint),
  ('db_occlusion_count2_low_02', ctypes.c_uint),
  ('db_occlusion_count2_hi_02', ctypes.c_uint),
  ('db_occlusion_count3_low_02', ctypes.c_uint),
  ('db_occlusion_count3_hi_02', ctypes.c_uint),
  ('db_occlusion_count0_low_03', ctypes.c_uint),
  ('db_occlusion_count0_hi_03', ctypes.c_uint),
  ('db_occlusion_count1_low_03', ctypes.c_uint),
  ('db_occlusion_count1_hi_03', ctypes.c_uint),
  ('db_occlusion_count2_low_03', ctypes.c_uint),
  ('db_occlusion_count2_hi_03', ctypes.c_uint),
  ('db_occlusion_count3_low_03', ctypes.c_uint),
  ('db_occlusion_count3_hi_03', ctypes.c_uint),
  ('db_occlusion_count0_low_04', ctypes.c_uint),
  ('db_occlusion_count0_hi_04', ctypes.c_uint),
  ('db_occlusion_count1_low_04', ctypes.c_uint),
  ('db_occlusion_count1_hi_04', ctypes.c_uint),
  ('db_occlusion_count2_low_04', ctypes.c_uint),
  ('db_occlusion_count2_hi_04', ctypes.c_uint),
  ('db_occlusion_count3_low_04', ctypes.c_uint),
  ('db_occlusion_count3_hi_04', ctypes.c_uint),
  ('db_occlusion_count0_low_05', ctypes.c_uint),
  ('db_occlusion_count0_hi_05', ctypes.c_uint),
  ('db_occlusion_count1_low_05', ctypes.c_uint),
  ('db_occlusion_count1_hi_05', ctypes.c_uint),
  ('db_occlusion_count2_low_05', ctypes.c_uint),
  ('db_occlusion_count2_hi_05', ctypes.c_uint),
  ('db_occlusion_count3_low_05', ctypes.c_uint),
  ('db_occlusion_count3_hi_05', ctypes.c_uint),
  ('db_occlusion_count0_low_06', ctypes.c_uint),
  ('db_occlusion_count0_hi_06', ctypes.c_uint),
  ('db_occlusion_count1_low_06', ctypes.c_uint),
  ('db_occlusion_count1_hi_06', ctypes.c_uint),
  ('db_occlusion_count2_low_06', ctypes.c_uint),
  ('db_occlusion_count2_hi_06', ctypes.c_uint),
  ('db_occlusion_count3_low_06', ctypes.c_uint),
  ('db_occlusion_count3_hi_06', ctypes.c_uint),
  ('db_occlusion_count0_low_07', ctypes.c_uint),
  ('db_occlusion_count0_hi_07', ctypes.c_uint),
  ('db_occlusion_count1_low_07', ctypes.c_uint),
  ('db_occlusion_count1_hi_07', ctypes.c_uint),
  ('db_occlusion_count2_low_07', ctypes.c_uint),
  ('db_occlusion_count2_hi_07', ctypes.c_uint),
  ('db_occlusion_count3_low_07', ctypes.c_uint),
  ('db_occlusion_count3_hi_07', ctypes.c_uint),
  ('db_occlusion_count0_low_10', ctypes.c_uint),
  ('db_occlusion_count0_hi_10', ctypes.c_uint),
  ('db_occlusion_count1_low_10', ctypes.c_uint),
  ('db_occlusion_count1_hi_10', ctypes.c_uint),
  ('db_occlusion_count2_low_10', ctypes.c_uint),
  ('db_occlusion_count2_hi_10', ctypes.c_uint),
  ('db_occlusion_count3_low_10', ctypes.c_uint),
  ('db_occlusion_count3_hi_10', ctypes.c_uint),
  ('db_occlusion_count0_low_11', ctypes.c_uint),
  ('db_occlusion_count0_hi_11', ctypes.c_uint),
  ('db_occlusion_count1_low_11', ctypes.c_uint),
  ('db_occlusion_count1_hi_11', ctypes.c_uint),
  ('db_occlusion_count2_low_11', ctypes.c_uint),
  ('db_occlusion_count2_hi_11', ctypes.c_uint),
  ('db_occlusion_count3_low_11', ctypes.c_uint),
  ('db_occlusion_count3_hi_11', ctypes.c_uint),
  ('db_occlusion_count0_low_12', ctypes.c_uint),
  ('db_occlusion_count0_hi_12', ctypes.c_uint),
  ('db_occlusion_count1_low_12', ctypes.c_uint),
  ('db_occlusion_count1_hi_12', ctypes.c_uint),
  ('db_occlusion_count2_low_12', ctypes.c_uint),
  ('db_occlusion_count2_hi_12', ctypes.c_uint),
  ('db_occlusion_count3_low_12', ctypes.c_uint),
  ('db_occlusion_count3_hi_12', ctypes.c_uint),
  ('db_occlusion_count0_low_13', ctypes.c_uint),
  ('db_occlusion_count0_hi_13', ctypes.c_uint),
  ('db_occlusion_count1_low_13', ctypes.c_uint),
  ('db_occlusion_count1_hi_13', ctypes.c_uint),
  ('db_occlusion_count2_low_13', ctypes.c_uint),
  ('db_occlusion_count2_hi_13', ctypes.c_uint),
  ('db_occlusion_count3_low_13', ctypes.c_uint),
  ('db_occlusion_count3_hi_13', ctypes.c_uint),
  ('db_occlusion_count0_low_14', ctypes.c_uint),
  ('db_occlusion_count0_hi_14', ctypes.c_uint),
  ('db_occlusion_count1_low_14', ctypes.c_uint),
  ('db_occlusion_count1_hi_14', ctypes.c_uint),
  ('db_occlusion_count2_low_14', ctypes.c_uint),
  ('db_occlusion_count2_hi_14', ctypes.c_uint),
  ('db_occlusion_count3_low_14', ctypes.c_uint),
  ('db_occlusion_count3_hi_14', ctypes.c_uint),
  ('db_occlusion_count0_low_15', ctypes.c_uint),
  ('db_occlusion_count0_hi_15', ctypes.c_uint),
  ('db_occlusion_count1_low_15', ctypes.c_uint),
  ('db_occlusion_count1_hi_15', ctypes.c_uint),
  ('db_occlusion_count2_low_15', ctypes.c_uint),
  ('db_occlusion_count2_hi_15', ctypes.c_uint),
  ('db_occlusion_count3_low_15', ctypes.c_uint),
  ('db_occlusion_count3_hi_15', ctypes.c_uint),
  ('db_occlusion_count0_low_16', ctypes.c_uint),
  ('db_occlusion_count0_hi_16', ctypes.c_uint),
  ('db_occlusion_count1_low_16', ctypes.c_uint),
  ('db_occlusion_count1_hi_16', ctypes.c_uint),
  ('db_occlusion_count2_low_16', ctypes.c_uint),
  ('db_occlusion_count2_hi_16', ctypes.c_uint),
  ('db_occlusion_count3_low_16', ctypes.c_uint),
  ('db_occlusion_count3_hi_16', ctypes.c_uint),
  ('db_occlusion_count0_low_17', ctypes.c_uint),
  ('db_occlusion_count0_hi_17', ctypes.c_uint),
  ('db_occlusion_count1_low_17', ctypes.c_uint),
  ('db_occlusion_count1_hi_17', ctypes.c_uint),
  ('db_occlusion_count2_low_17', ctypes.c_uint),
  ('db_occlusion_count2_hi_17', ctypes.c_uint),
  ('db_occlusion_count3_low_17', ctypes.c_uint),
  ('db_occlusion_count3_hi_17', ctypes.c_uint),
  ('reserved_492', ctypes.c_uint),
  ('reserved_493', ctypes.c_uint),
  ('reserved_494', ctypes.c_uint),
  ('reserved_495', ctypes.c_uint),
  ('reserved_496', ctypes.c_uint),
  ('reserved_497', ctypes.c_uint),
  ('reserved_498', ctypes.c_uint),
  ('reserved_499', ctypes.c_uint),
  ('reserved_500', ctypes.c_uint),
  ('reserved_501', ctypes.c_uint),
  ('reserved_502', ctypes.c_uint),
  ('reserved_503', ctypes.c_uint),
  ('reserved_504', ctypes.c_uint),
  ('reserved_505', ctypes.c_uint),
  ('reserved_506', ctypes.c_uint),
  ('reserved_507', ctypes.c_uint),
  ('reserved_508', ctypes.c_uint),
  ('reserved_509', ctypes.c_uint),
  ('reserved_510', ctypes.c_uint),
  ('reserved_511', ctypes.c_uint),
]
class struct_v11_sdma_mqd(Struct): pass
struct_v11_sdma_mqd._fields_ = [
  ('sdmax_rlcx_rb_cntl', ctypes.c_uint),
  ('sdmax_rlcx_rb_base', ctypes.c_uint),
  ('sdmax_rlcx_rb_base_hi', ctypes.c_uint),
  ('sdmax_rlcx_rb_rptr', ctypes.c_uint),
  ('sdmax_rlcx_rb_rptr_hi', ctypes.c_uint),
  ('sdmax_rlcx_rb_wptr', ctypes.c_uint),
  ('sdmax_rlcx_rb_wptr_hi', ctypes.c_uint),
  ('sdmax_rlcx_rb_rptr_addr_hi', ctypes.c_uint),
  ('sdmax_rlcx_rb_rptr_addr_lo', ctypes.c_uint),
  ('sdmax_rlcx_ib_cntl', ctypes.c_uint),
  ('sdmax_rlcx_ib_rptr', ctypes.c_uint),
  ('sdmax_rlcx_ib_offset', ctypes.c_uint),
  ('sdmax_rlcx_ib_base_lo', ctypes.c_uint),
  ('sdmax_rlcx_ib_base_hi', ctypes.c_uint),
  ('sdmax_rlcx_ib_size', ctypes.c_uint),
  ('sdmax_rlcx_skip_cntl', ctypes.c_uint),
  ('sdmax_rlcx_context_status', ctypes.c_uint),
  ('sdmax_rlcx_doorbell', ctypes.c_uint),
  ('sdmax_rlcx_doorbell_log', ctypes.c_uint),
  ('sdmax_rlcx_doorbell_offset', ctypes.c_uint),
  ('sdmax_rlcx_csa_addr_lo', ctypes.c_uint),
  ('sdmax_rlcx_csa_addr_hi', ctypes.c_uint),
  ('sdmax_rlcx_sched_cntl', ctypes.c_uint),
  ('sdmax_rlcx_ib_sub_remain', ctypes.c_uint),
  ('sdmax_rlcx_preempt', ctypes.c_uint),
  ('sdmax_rlcx_dummy_reg', ctypes.c_uint),
  ('sdmax_rlcx_rb_wptr_poll_addr_hi', ctypes.c_uint),
  ('sdmax_rlcx_rb_wptr_poll_addr_lo', ctypes.c_uint),
  ('sdmax_rlcx_rb_aql_cntl', ctypes.c_uint),
  ('sdmax_rlcx_minor_ptr_update', ctypes.c_uint),
  ('sdmax_rlcx_rb_preempt', ctypes.c_uint),
  ('sdmax_rlcx_midcmd_data0', ctypes.c_uint),
  ('sdmax_rlcx_midcmd_data1', ctypes.c_uint),
  ('sdmax_rlcx_midcmd_data2', ctypes.c_uint),
  ('sdmax_rlcx_midcmd_data3', ctypes.c_uint),
  ('sdmax_rlcx_midcmd_data4', ctypes.c_uint),
  ('sdmax_rlcx_midcmd_data5', ctypes.c_uint),
  ('sdmax_rlcx_midcmd_data6', ctypes.c_uint),
  ('sdmax_rlcx_midcmd_data7', ctypes.c_uint),
  ('sdmax_rlcx_midcmd_data8', ctypes.c_uint),
  ('sdmax_rlcx_midcmd_data9', ctypes.c_uint),
  ('sdmax_rlcx_midcmd_data10', ctypes.c_uint),
  ('sdmax_rlcx_midcmd_cntl', ctypes.c_uint),
  ('sdmax_rlcx_f32_dbg0', ctypes.c_uint),
  ('sdmax_rlcx_f32_dbg1', ctypes.c_uint),
  ('reserved_45', ctypes.c_uint),
  ('reserved_46', ctypes.c_uint),
  ('reserved_47', ctypes.c_uint),
  ('reserved_48', ctypes.c_uint),
  ('reserved_49', ctypes.c_uint),
  ('reserved_50', ctypes.c_uint),
  ('reserved_51', ctypes.c_uint),
  ('reserved_52', ctypes.c_uint),
  ('reserved_53', ctypes.c_uint),
  ('reserved_54', ctypes.c_uint),
  ('reserved_55', ctypes.c_uint),
  ('reserved_56', ctypes.c_uint),
  ('reserved_57', ctypes.c_uint),
  ('reserved_58', ctypes.c_uint),
  ('reserved_59', ctypes.c_uint),
  ('reserved_60', ctypes.c_uint),
  ('reserved_61', ctypes.c_uint),
  ('reserved_62', ctypes.c_uint),
  ('reserved_63', ctypes.c_uint),
  ('reserved_64', ctypes.c_uint),
  ('reserved_65', ctypes.c_uint),
  ('reserved_66', ctypes.c_uint),
  ('reserved_67', ctypes.c_uint),
  ('reserved_68', ctypes.c_uint),
  ('reserved_69', ctypes.c_uint),
  ('reserved_70', ctypes.c_uint),
  ('reserved_71', ctypes.c_uint),
  ('reserved_72', ctypes.c_uint),
  ('reserved_73', ctypes.c_uint),
  ('reserved_74', ctypes.c_uint),
  ('reserved_75', ctypes.c_uint),
  ('reserved_76', ctypes.c_uint),
  ('reserved_77', ctypes.c_uint),
  ('reserved_78', ctypes.c_uint),
  ('reserved_79', ctypes.c_uint),
  ('reserved_80', ctypes.c_uint),
  ('reserved_81', ctypes.c_uint),
  ('reserved_82', ctypes.c_uint),
  ('reserved_83', ctypes.c_uint),
  ('reserved_84', ctypes.c_uint),
  ('reserved_85', ctypes.c_uint),
  ('reserved_86', ctypes.c_uint),
  ('reserved_87', ctypes.c_uint),
  ('reserved_88', ctypes.c_uint),
  ('reserved_89', ctypes.c_uint),
  ('reserved_90', ctypes.c_uint),
  ('reserved_91', ctypes.c_uint),
  ('reserved_92', ctypes.c_uint),
  ('reserved_93', ctypes.c_uint),
  ('reserved_94', ctypes.c_uint),
  ('reserved_95', ctypes.c_uint),
  ('reserved_96', ctypes.c_uint),
  ('reserved_97', ctypes.c_uint),
  ('reserved_98', ctypes.c_uint),
  ('reserved_99', ctypes.c_uint),
  ('reserved_100', ctypes.c_uint),
  ('reserved_101', ctypes.c_uint),
  ('reserved_102', ctypes.c_uint),
  ('reserved_103', ctypes.c_uint),
  ('reserved_104', ctypes.c_uint),
  ('reserved_105', ctypes.c_uint),
  ('reserved_106', ctypes.c_uint),
  ('reserved_107', ctypes.c_uint),
  ('reserved_108', ctypes.c_uint),
  ('reserved_109', ctypes.c_uint),
  ('reserved_110', ctypes.c_uint),
  ('reserved_111', ctypes.c_uint),
  ('reserved_112', ctypes.c_uint),
  ('reserved_113', ctypes.c_uint),
  ('reserved_114', ctypes.c_uint),
  ('reserved_115', ctypes.c_uint),
  ('reserved_116', ctypes.c_uint),
  ('reserved_117', ctypes.c_uint),
  ('reserved_118', ctypes.c_uint),
  ('reserved_119', ctypes.c_uint),
  ('reserved_120', ctypes.c_uint),
  ('reserved_121', ctypes.c_uint),
  ('reserved_122', ctypes.c_uint),
  ('reserved_123', ctypes.c_uint),
  ('reserved_124', ctypes.c_uint),
  ('reserved_125', ctypes.c_uint),
  ('sdma_engine_id', ctypes.c_uint),
  ('sdma_queue_id', ctypes.c_uint),
]
class struct_v11_compute_mqd(Struct): pass
struct_v11_compute_mqd._fields_ = [
  ('header', ctypes.c_uint),
  ('compute_dispatch_initiator', ctypes.c_uint),
  ('compute_dim_x', ctypes.c_uint),
  ('compute_dim_y', ctypes.c_uint),
  ('compute_dim_z', ctypes.c_uint),
  ('compute_start_x', ctypes.c_uint),
  ('compute_start_y', ctypes.c_uint),
  ('compute_start_z', ctypes.c_uint),
  ('compute_num_thread_x', ctypes.c_uint),
  ('compute_num_thread_y', ctypes.c_uint),
  ('compute_num_thread_z', ctypes.c_uint),
  ('compute_pipelinestat_enable', ctypes.c_uint),
  ('compute_perfcount_enable', ctypes.c_uint),
  ('compute_pgm_lo', ctypes.c_uint),
  ('compute_pgm_hi', ctypes.c_uint),
  ('compute_dispatch_pkt_addr_lo', ctypes.c_uint),
  ('compute_dispatch_pkt_addr_hi', ctypes.c_uint),
  ('compute_dispatch_scratch_base_lo', ctypes.c_uint),
  ('compute_dispatch_scratch_base_hi', ctypes.c_uint),
  ('compute_pgm_rsrc1', ctypes.c_uint),
  ('compute_pgm_rsrc2', ctypes.c_uint),
  ('compute_vmid', ctypes.c_uint),
  ('compute_resource_limits', ctypes.c_uint),
  ('compute_static_thread_mgmt_se0', ctypes.c_uint),
  ('compute_static_thread_mgmt_se1', ctypes.c_uint),
  ('compute_tmpring_size', ctypes.c_uint),
  ('compute_static_thread_mgmt_se2', ctypes.c_uint),
  ('compute_static_thread_mgmt_se3', ctypes.c_uint),
  ('compute_restart_x', ctypes.c_uint),
  ('compute_restart_y', ctypes.c_uint),
  ('compute_restart_z', ctypes.c_uint),
  ('compute_thread_trace_enable', ctypes.c_uint),
  ('compute_misc_reserved', ctypes.c_uint),
  ('compute_dispatch_id', ctypes.c_uint),
  ('compute_threadgroup_id', ctypes.c_uint),
  ('compute_req_ctrl', ctypes.c_uint),
  ('reserved_36', ctypes.c_uint),
  ('compute_user_accum_0', ctypes.c_uint),
  ('compute_user_accum_1', ctypes.c_uint),
  ('compute_user_accum_2', ctypes.c_uint),
  ('compute_user_accum_3', ctypes.c_uint),
  ('compute_pgm_rsrc3', ctypes.c_uint),
  ('compute_ddid_index', ctypes.c_uint),
  ('compute_shader_chksum', ctypes.c_uint),
  ('compute_static_thread_mgmt_se4', ctypes.c_uint),
  ('compute_static_thread_mgmt_se5', ctypes.c_uint),
  ('compute_static_thread_mgmt_se6', ctypes.c_uint),
  ('compute_static_thread_mgmt_se7', ctypes.c_uint),
  ('compute_dispatch_interleave', ctypes.c_uint),
  ('compute_relaunch', ctypes.c_uint),
  ('compute_wave_restore_addr_lo', ctypes.c_uint),
  ('compute_wave_restore_addr_hi', ctypes.c_uint),
  ('compute_wave_restore_control', ctypes.c_uint),
  ('reserved_53', ctypes.c_uint),
  ('reserved_54', ctypes.c_uint),
  ('reserved_55', ctypes.c_uint),
  ('reserved_56', ctypes.c_uint),
  ('reserved_57', ctypes.c_uint),
  ('reserved_58', ctypes.c_uint),
  ('reserved_59', ctypes.c_uint),
  ('reserved_60', ctypes.c_uint),
  ('reserved_61', ctypes.c_uint),
  ('reserved_62', ctypes.c_uint),
  ('reserved_63', ctypes.c_uint),
  ('reserved_64', ctypes.c_uint),
  ('compute_user_data_0', ctypes.c_uint),
  ('compute_user_data_1', ctypes.c_uint),
  ('compute_user_data_2', ctypes.c_uint),
  ('compute_user_data_3', ctypes.c_uint),
  ('compute_user_data_4', ctypes.c_uint),
  ('compute_user_data_5', ctypes.c_uint),
  ('compute_user_data_6', ctypes.c_uint),
  ('compute_user_data_7', ctypes.c_uint),
  ('compute_user_data_8', ctypes.c_uint),
  ('compute_user_data_9', ctypes.c_uint),
  ('compute_user_data_10', ctypes.c_uint),
  ('compute_user_data_11', ctypes.c_uint),
  ('compute_user_data_12', ctypes.c_uint),
  ('compute_user_data_13', ctypes.c_uint),
  ('compute_user_data_14', ctypes.c_uint),
  ('compute_user_data_15', ctypes.c_uint),
  ('cp_compute_csinvoc_count_lo', ctypes.c_uint),
  ('cp_compute_csinvoc_count_hi', ctypes.c_uint),
  ('reserved_83', ctypes.c_uint),
  ('reserved_84', ctypes.c_uint),
  ('reserved_85', ctypes.c_uint),
  ('cp_mqd_query_time_lo', ctypes.c_uint),
  ('cp_mqd_query_time_hi', ctypes.c_uint),
  ('cp_mqd_connect_start_time_lo', ctypes.c_uint),
  ('cp_mqd_connect_start_time_hi', ctypes.c_uint),
  ('cp_mqd_connect_end_time_lo', ctypes.c_uint),
  ('cp_mqd_connect_end_time_hi', ctypes.c_uint),
  ('cp_mqd_connect_end_wf_count', ctypes.c_uint),
  ('cp_mqd_connect_end_pq_rptr', ctypes.c_uint),
  ('cp_mqd_connect_end_pq_wptr', ctypes.c_uint),
  ('cp_mqd_connect_end_ib_rptr', ctypes.c_uint),
  ('cp_mqd_readindex_lo', ctypes.c_uint),
  ('cp_mqd_readindex_hi', ctypes.c_uint),
  ('cp_mqd_save_start_time_lo', ctypes.c_uint),
  ('cp_mqd_save_start_time_hi', ctypes.c_uint),
  ('cp_mqd_save_end_time_lo', ctypes.c_uint),
  ('cp_mqd_save_end_time_hi', ctypes.c_uint),
  ('cp_mqd_restore_start_time_lo', ctypes.c_uint),
  ('cp_mqd_restore_start_time_hi', ctypes.c_uint),
  ('cp_mqd_restore_end_time_lo', ctypes.c_uint),
  ('cp_mqd_restore_end_time_hi', ctypes.c_uint),
  ('disable_queue', ctypes.c_uint),
  ('reserved_107', ctypes.c_uint),
  ('gds_cs_ctxsw_cnt0', ctypes.c_uint),
  ('gds_cs_ctxsw_cnt1', ctypes.c_uint),
  ('gds_cs_ctxsw_cnt2', ctypes.c_uint),
  ('gds_cs_ctxsw_cnt3', ctypes.c_uint),
  ('reserved_112', ctypes.c_uint),
  ('reserved_113', ctypes.c_uint),
  ('cp_pq_exe_status_lo', ctypes.c_uint),
  ('cp_pq_exe_status_hi', ctypes.c_uint),
  ('cp_packet_id_lo', ctypes.c_uint),
  ('cp_packet_id_hi', ctypes.c_uint),
  ('cp_packet_exe_status_lo', ctypes.c_uint),
  ('cp_packet_exe_status_hi', ctypes.c_uint),
  ('gds_save_base_addr_lo', ctypes.c_uint),
  ('gds_save_base_addr_hi', ctypes.c_uint),
  ('gds_save_mask_lo', ctypes.c_uint),
  ('gds_save_mask_hi', ctypes.c_uint),
  ('ctx_save_base_addr_lo', ctypes.c_uint),
  ('ctx_save_base_addr_hi', ctypes.c_uint),
  ('reserved_126', ctypes.c_uint),
  ('reserved_127', ctypes.c_uint),
  ('cp_mqd_base_addr_lo', ctypes.c_uint),
  ('cp_mqd_base_addr_hi', ctypes.c_uint),
  ('cp_hqd_active', ctypes.c_uint),
  ('cp_hqd_vmid', ctypes.c_uint),
  ('cp_hqd_persistent_state', ctypes.c_uint),
  ('cp_hqd_pipe_priority', ctypes.c_uint),
  ('cp_hqd_queue_priority', ctypes.c_uint),
  ('cp_hqd_quantum', ctypes.c_uint),
  ('cp_hqd_pq_base_lo', ctypes.c_uint),
  ('cp_hqd_pq_base_hi', ctypes.c_uint),
  ('cp_hqd_pq_rptr', ctypes.c_uint),
  ('cp_hqd_pq_rptr_report_addr_lo', ctypes.c_uint),
  ('cp_hqd_pq_rptr_report_addr_hi', ctypes.c_uint),
  ('cp_hqd_pq_wptr_poll_addr_lo', ctypes.c_uint),
  ('cp_hqd_pq_wptr_poll_addr_hi', ctypes.c_uint),
  ('cp_hqd_pq_doorbell_control', ctypes.c_uint),
  ('reserved_144', ctypes.c_uint),
  ('cp_hqd_pq_control', ctypes.c_uint),
  ('cp_hqd_ib_base_addr_lo', ctypes.c_uint),
  ('cp_hqd_ib_base_addr_hi', ctypes.c_uint),
  ('cp_hqd_ib_rptr', ctypes.c_uint),
  ('cp_hqd_ib_control', ctypes.c_uint),
  ('cp_hqd_iq_timer', ctypes.c_uint),
  ('cp_hqd_iq_rptr', ctypes.c_uint),
  ('cp_hqd_dequeue_request', ctypes.c_uint),
  ('cp_hqd_dma_offload', ctypes.c_uint),
  ('cp_hqd_sema_cmd', ctypes.c_uint),
  ('cp_hqd_msg_type', ctypes.c_uint),
  ('cp_hqd_atomic0_preop_lo', ctypes.c_uint),
  ('cp_hqd_atomic0_preop_hi', ctypes.c_uint),
  ('cp_hqd_atomic1_preop_lo', ctypes.c_uint),
  ('cp_hqd_atomic1_preop_hi', ctypes.c_uint),
  ('cp_hqd_hq_status0', ctypes.c_uint),
  ('cp_hqd_hq_control0', ctypes.c_uint),
  ('cp_mqd_control', ctypes.c_uint),
  ('cp_hqd_hq_status1', ctypes.c_uint),
  ('cp_hqd_hq_control1', ctypes.c_uint),
  ('cp_hqd_eop_base_addr_lo', ctypes.c_uint),
  ('cp_hqd_eop_base_addr_hi', ctypes.c_uint),
  ('cp_hqd_eop_control', ctypes.c_uint),
  ('cp_hqd_eop_rptr', ctypes.c_uint),
  ('cp_hqd_eop_wptr', ctypes.c_uint),
  ('cp_hqd_eop_done_events', ctypes.c_uint),
  ('cp_hqd_ctx_save_base_addr_lo', ctypes.c_uint),
  ('cp_hqd_ctx_save_base_addr_hi', ctypes.c_uint),
  ('cp_hqd_ctx_save_control', ctypes.c_uint),
  ('cp_hqd_cntl_stack_offset', ctypes.c_uint),
  ('cp_hqd_cntl_stack_size', ctypes.c_uint),
  ('cp_hqd_wg_state_offset', ctypes.c_uint),
  ('cp_hqd_ctx_save_size', ctypes.c_uint),
  ('cp_hqd_gds_resource_state', ctypes.c_uint),
  ('cp_hqd_error', ctypes.c_uint),
  ('cp_hqd_eop_wptr_mem', ctypes.c_uint),
  ('cp_hqd_aql_control', ctypes.c_uint),
  ('cp_hqd_pq_wptr_lo', ctypes.c_uint),
  ('cp_hqd_pq_wptr_hi', ctypes.c_uint),
  ('reserved_184', ctypes.c_uint),
  ('reserved_185', ctypes.c_uint),
  ('reserved_186', ctypes.c_uint),
  ('reserved_187', ctypes.c_uint),
  ('reserved_188', ctypes.c_uint),
  ('reserved_189', ctypes.c_uint),
  ('reserved_190', ctypes.c_uint),
  ('reserved_191', ctypes.c_uint),
  ('iqtimer_pkt_header', ctypes.c_uint),
  ('iqtimer_pkt_dw0', ctypes.c_uint),
  ('iqtimer_pkt_dw1', ctypes.c_uint),
  ('iqtimer_pkt_dw2', ctypes.c_uint),
  ('iqtimer_pkt_dw3', ctypes.c_uint),
  ('iqtimer_pkt_dw4', ctypes.c_uint),
  ('iqtimer_pkt_dw5', ctypes.c_uint),
  ('iqtimer_pkt_dw6', ctypes.c_uint),
  ('iqtimer_pkt_dw7', ctypes.c_uint),
  ('iqtimer_pkt_dw8', ctypes.c_uint),
  ('iqtimer_pkt_dw9', ctypes.c_uint),
  ('iqtimer_pkt_dw10', ctypes.c_uint),
  ('iqtimer_pkt_dw11', ctypes.c_uint),
  ('iqtimer_pkt_dw12', ctypes.c_uint),
  ('iqtimer_pkt_dw13', ctypes.c_uint),
  ('iqtimer_pkt_dw14', ctypes.c_uint),
  ('iqtimer_pkt_dw15', ctypes.c_uint),
  ('iqtimer_pkt_dw16', ctypes.c_uint),
  ('iqtimer_pkt_dw17', ctypes.c_uint),
  ('iqtimer_pkt_dw18', ctypes.c_uint),
  ('iqtimer_pkt_dw19', ctypes.c_uint),
  ('iqtimer_pkt_dw20', ctypes.c_uint),
  ('iqtimer_pkt_dw21', ctypes.c_uint),
  ('iqtimer_pkt_dw22', ctypes.c_uint),
  ('iqtimer_pkt_dw23', ctypes.c_uint),
  ('iqtimer_pkt_dw24', ctypes.c_uint),
  ('iqtimer_pkt_dw25', ctypes.c_uint),
  ('iqtimer_pkt_dw26', ctypes.c_uint),
  ('iqtimer_pkt_dw27', ctypes.c_uint),
  ('iqtimer_pkt_dw28', ctypes.c_uint),
  ('iqtimer_pkt_dw29', ctypes.c_uint),
  ('iqtimer_pkt_dw30', ctypes.c_uint),
  ('iqtimer_pkt_dw31', ctypes.c_uint),
  ('reserved_225', ctypes.c_uint),
  ('reserved_226', ctypes.c_uint),
  ('reserved_227', ctypes.c_uint),
  ('set_resources_header', ctypes.c_uint),
  ('set_resources_dw1', ctypes.c_uint),
  ('set_resources_dw2', ctypes.c_uint),
  ('set_resources_dw3', ctypes.c_uint),
  ('set_resources_dw4', ctypes.c_uint),
  ('set_resources_dw5', ctypes.c_uint),
  ('set_resources_dw6', ctypes.c_uint),
  ('set_resources_dw7', ctypes.c_uint),
  ('reserved_236', ctypes.c_uint),
  ('reserved_237', ctypes.c_uint),
  ('reserved_238', ctypes.c_uint),
  ('reserved_239', ctypes.c_uint),
  ('queue_doorbell_id0', ctypes.c_uint),
  ('queue_doorbell_id1', ctypes.c_uint),
  ('queue_doorbell_id2', ctypes.c_uint),
  ('queue_doorbell_id3', ctypes.c_uint),
  ('queue_doorbell_id4', ctypes.c_uint),
  ('queue_doorbell_id5', ctypes.c_uint),
  ('queue_doorbell_id6', ctypes.c_uint),
  ('queue_doorbell_id7', ctypes.c_uint),
  ('queue_doorbell_id8', ctypes.c_uint),
  ('queue_doorbell_id9', ctypes.c_uint),
  ('queue_doorbell_id10', ctypes.c_uint),
  ('queue_doorbell_id11', ctypes.c_uint),
  ('queue_doorbell_id12', ctypes.c_uint),
  ('queue_doorbell_id13', ctypes.c_uint),
  ('queue_doorbell_id14', ctypes.c_uint),
  ('queue_doorbell_id15', ctypes.c_uint),
  ('control_buf_addr_lo', ctypes.c_uint),
  ('control_buf_addr_hi', ctypes.c_uint),
  ('control_buf_wptr_lo', ctypes.c_uint),
  ('control_buf_wptr_hi', ctypes.c_uint),
  ('control_buf_dptr_lo', ctypes.c_uint),
  ('control_buf_dptr_hi', ctypes.c_uint),
  ('control_buf_num_entries', ctypes.c_uint),
  ('draw_ring_addr_lo', ctypes.c_uint),
  ('draw_ring_addr_hi', ctypes.c_uint),
  ('reserved_265', ctypes.c_uint),
  ('reserved_266', ctypes.c_uint),
  ('reserved_267', ctypes.c_uint),
  ('reserved_268', ctypes.c_uint),
  ('reserved_269', ctypes.c_uint),
  ('reserved_270', ctypes.c_uint),
  ('reserved_271', ctypes.c_uint),
  ('reserved_272', ctypes.c_uint),
  ('reserved_273', ctypes.c_uint),
  ('reserved_274', ctypes.c_uint),
  ('reserved_275', ctypes.c_uint),
  ('reserved_276', ctypes.c_uint),
  ('reserved_277', ctypes.c_uint),
  ('reserved_278', ctypes.c_uint),
  ('reserved_279', ctypes.c_uint),
  ('reserved_280', ctypes.c_uint),
  ('reserved_281', ctypes.c_uint),
  ('reserved_282', ctypes.c_uint),
  ('reserved_283', ctypes.c_uint),
  ('reserved_284', ctypes.c_uint),
  ('reserved_285', ctypes.c_uint),
  ('reserved_286', ctypes.c_uint),
  ('reserved_287', ctypes.c_uint),
  ('reserved_288', ctypes.c_uint),
  ('reserved_289', ctypes.c_uint),
  ('reserved_290', ctypes.c_uint),
  ('reserved_291', ctypes.c_uint),
  ('reserved_292', ctypes.c_uint),
  ('reserved_293', ctypes.c_uint),
  ('reserved_294', ctypes.c_uint),
  ('reserved_295', ctypes.c_uint),
  ('reserved_296', ctypes.c_uint),
  ('reserved_297', ctypes.c_uint),
  ('reserved_298', ctypes.c_uint),
  ('reserved_299', ctypes.c_uint),
  ('reserved_300', ctypes.c_uint),
  ('reserved_301', ctypes.c_uint),
  ('reserved_302', ctypes.c_uint),
  ('reserved_303', ctypes.c_uint),
  ('reserved_304', ctypes.c_uint),
  ('reserved_305', ctypes.c_uint),
  ('reserved_306', ctypes.c_uint),
  ('reserved_307', ctypes.c_uint),
  ('reserved_308', ctypes.c_uint),
  ('reserved_309', ctypes.c_uint),
  ('reserved_310', ctypes.c_uint),
  ('reserved_311', ctypes.c_uint),
  ('reserved_312', ctypes.c_uint),
  ('reserved_313', ctypes.c_uint),
  ('reserved_314', ctypes.c_uint),
  ('reserved_315', ctypes.c_uint),
  ('reserved_316', ctypes.c_uint),
  ('reserved_317', ctypes.c_uint),
  ('reserved_318', ctypes.c_uint),
  ('reserved_319', ctypes.c_uint),
  ('reserved_320', ctypes.c_uint),
  ('reserved_321', ctypes.c_uint),
  ('reserved_322', ctypes.c_uint),
  ('reserved_323', ctypes.c_uint),
  ('reserved_324', ctypes.c_uint),
  ('reserved_325', ctypes.c_uint),
  ('reserved_326', ctypes.c_uint),
  ('reserved_327', ctypes.c_uint),
  ('reserved_328', ctypes.c_uint),
  ('reserved_329', ctypes.c_uint),
  ('reserved_330', ctypes.c_uint),
  ('reserved_331', ctypes.c_uint),
  ('reserved_332', ctypes.c_uint),
  ('reserved_333', ctypes.c_uint),
  ('reserved_334', ctypes.c_uint),
  ('reserved_335', ctypes.c_uint),
  ('reserved_336', ctypes.c_uint),
  ('reserved_337', ctypes.c_uint),
  ('reserved_338', ctypes.c_uint),
  ('reserved_339', ctypes.c_uint),
  ('reserved_340', ctypes.c_uint),
  ('reserved_341', ctypes.c_uint),
  ('reserved_342', ctypes.c_uint),
  ('reserved_343', ctypes.c_uint),
  ('reserved_344', ctypes.c_uint),
  ('reserved_345', ctypes.c_uint),
  ('reserved_346', ctypes.c_uint),
  ('reserved_347', ctypes.c_uint),
  ('reserved_348', ctypes.c_uint),
  ('reserved_349', ctypes.c_uint),
  ('reserved_350', ctypes.c_uint),
  ('reserved_351', ctypes.c_uint),
  ('reserved_352', ctypes.c_uint),
  ('reserved_353', ctypes.c_uint),
  ('reserved_354', ctypes.c_uint),
  ('reserved_355', ctypes.c_uint),
  ('reserved_356', ctypes.c_uint),
  ('reserved_357', ctypes.c_uint),
  ('reserved_358', ctypes.c_uint),
  ('reserved_359', ctypes.c_uint),
  ('reserved_360', ctypes.c_uint),
  ('reserved_361', ctypes.c_uint),
  ('reserved_362', ctypes.c_uint),
  ('reserved_363', ctypes.c_uint),
  ('reserved_364', ctypes.c_uint),
  ('reserved_365', ctypes.c_uint),
  ('reserved_366', ctypes.c_uint),
  ('reserved_367', ctypes.c_uint),
  ('reserved_368', ctypes.c_uint),
  ('reserved_369', ctypes.c_uint),
  ('reserved_370', ctypes.c_uint),
  ('reserved_371', ctypes.c_uint),
  ('reserved_372', ctypes.c_uint),
  ('reserved_373', ctypes.c_uint),
  ('reserved_374', ctypes.c_uint),
  ('reserved_375', ctypes.c_uint),
  ('reserved_376', ctypes.c_uint),
  ('reserved_377', ctypes.c_uint),
  ('reserved_378', ctypes.c_uint),
  ('reserved_379', ctypes.c_uint),
  ('reserved_380', ctypes.c_uint),
  ('reserved_381', ctypes.c_uint),
  ('reserved_382', ctypes.c_uint),
  ('reserved_383', ctypes.c_uint),
  ('reserved_384', ctypes.c_uint),
  ('reserved_385', ctypes.c_uint),
  ('reserved_386', ctypes.c_uint),
  ('reserved_387', ctypes.c_uint),
  ('reserved_388', ctypes.c_uint),
  ('reserved_389', ctypes.c_uint),
  ('reserved_390', ctypes.c_uint),
  ('reserved_391', ctypes.c_uint),
  ('reserved_392', ctypes.c_uint),
  ('reserved_393', ctypes.c_uint),
  ('reserved_394', ctypes.c_uint),
  ('reserved_395', ctypes.c_uint),
  ('reserved_396', ctypes.c_uint),
  ('reserved_397', ctypes.c_uint),
  ('reserved_398', ctypes.c_uint),
  ('reserved_399', ctypes.c_uint),
  ('reserved_400', ctypes.c_uint),
  ('reserved_401', ctypes.c_uint),
  ('reserved_402', ctypes.c_uint),
  ('reserved_403', ctypes.c_uint),
  ('reserved_404', ctypes.c_uint),
  ('reserved_405', ctypes.c_uint),
  ('reserved_406', ctypes.c_uint),
  ('reserved_407', ctypes.c_uint),
  ('reserved_408', ctypes.c_uint),
  ('reserved_409', ctypes.c_uint),
  ('reserved_410', ctypes.c_uint),
  ('reserved_411', ctypes.c_uint),
  ('reserved_412', ctypes.c_uint),
  ('reserved_413', ctypes.c_uint),
  ('reserved_414', ctypes.c_uint),
  ('reserved_415', ctypes.c_uint),
  ('reserved_416', ctypes.c_uint),
  ('reserved_417', ctypes.c_uint),
  ('reserved_418', ctypes.c_uint),
  ('reserved_419', ctypes.c_uint),
  ('reserved_420', ctypes.c_uint),
  ('reserved_421', ctypes.c_uint),
  ('reserved_422', ctypes.c_uint),
  ('reserved_423', ctypes.c_uint),
  ('reserved_424', ctypes.c_uint),
  ('reserved_425', ctypes.c_uint),
  ('reserved_426', ctypes.c_uint),
  ('reserved_427', ctypes.c_uint),
  ('reserved_428', ctypes.c_uint),
  ('reserved_429', ctypes.c_uint),
  ('reserved_430', ctypes.c_uint),
  ('reserved_431', ctypes.c_uint),
  ('reserved_432', ctypes.c_uint),
  ('reserved_433', ctypes.c_uint),
  ('reserved_434', ctypes.c_uint),
  ('reserved_435', ctypes.c_uint),
  ('reserved_436', ctypes.c_uint),
  ('reserved_437', ctypes.c_uint),
  ('reserved_438', ctypes.c_uint),
  ('reserved_439', ctypes.c_uint),
  ('reserved_440', ctypes.c_uint),
  ('reserved_441', ctypes.c_uint),
  ('reserved_442', ctypes.c_uint),
  ('reserved_443', ctypes.c_uint),
  ('reserved_444', ctypes.c_uint),
  ('reserved_445', ctypes.c_uint),
  ('reserved_446', ctypes.c_uint),
  ('reserved_447', ctypes.c_uint),
  ('gws_0_val', ctypes.c_uint),
  ('gws_1_val', ctypes.c_uint),
  ('gws_2_val', ctypes.c_uint),
  ('gws_3_val', ctypes.c_uint),
  ('gws_4_val', ctypes.c_uint),
  ('gws_5_val', ctypes.c_uint),
  ('gws_6_val', ctypes.c_uint),
  ('gws_7_val', ctypes.c_uint),
  ('gws_8_val', ctypes.c_uint),
  ('gws_9_val', ctypes.c_uint),
  ('gws_10_val', ctypes.c_uint),
  ('gws_11_val', ctypes.c_uint),
  ('gws_12_val', ctypes.c_uint),
  ('gws_13_val', ctypes.c_uint),
  ('gws_14_val', ctypes.c_uint),
  ('gws_15_val', ctypes.c_uint),
  ('gws_16_val', ctypes.c_uint),
  ('gws_17_val', ctypes.c_uint),
  ('gws_18_val', ctypes.c_uint),
  ('gws_19_val', ctypes.c_uint),
  ('gws_20_val', ctypes.c_uint),
  ('gws_21_val', ctypes.c_uint),
  ('gws_22_val', ctypes.c_uint),
  ('gws_23_val', ctypes.c_uint),
  ('gws_24_val', ctypes.c_uint),
  ('gws_25_val', ctypes.c_uint),
  ('gws_26_val', ctypes.c_uint),
  ('gws_27_val', ctypes.c_uint),
  ('gws_28_val', ctypes.c_uint),
  ('gws_29_val', ctypes.c_uint),
  ('gws_30_val', ctypes.c_uint),
  ('gws_31_val', ctypes.c_uint),
  ('gws_32_val', ctypes.c_uint),
  ('gws_33_val', ctypes.c_uint),
  ('gws_34_val', ctypes.c_uint),
  ('gws_35_val', ctypes.c_uint),
  ('gws_36_val', ctypes.c_uint),
  ('gws_37_val', ctypes.c_uint),
  ('gws_38_val', ctypes.c_uint),
  ('gws_39_val', ctypes.c_uint),
  ('gws_40_val', ctypes.c_uint),
  ('gws_41_val', ctypes.c_uint),
  ('gws_42_val', ctypes.c_uint),
  ('gws_43_val', ctypes.c_uint),
  ('gws_44_val', ctypes.c_uint),
  ('gws_45_val', ctypes.c_uint),
  ('gws_46_val', ctypes.c_uint),
  ('gws_47_val', ctypes.c_uint),
  ('gws_48_val', ctypes.c_uint),
  ('gws_49_val', ctypes.c_uint),
  ('gws_50_val', ctypes.c_uint),
  ('gws_51_val', ctypes.c_uint),
  ('gws_52_val', ctypes.c_uint),
  ('gws_53_val', ctypes.c_uint),
  ('gws_54_val', ctypes.c_uint),
  ('gws_55_val', ctypes.c_uint),
  ('gws_56_val', ctypes.c_uint),
  ('gws_57_val', ctypes.c_uint),
  ('gws_58_val', ctypes.c_uint),
  ('gws_59_val', ctypes.c_uint),
  ('gws_60_val', ctypes.c_uint),
  ('gws_61_val', ctypes.c_uint),
  ('gws_62_val', ctypes.c_uint),
  ('gws_63_val', ctypes.c_uint),
]
class struct_v12_gfx_mqd(Struct): pass
class struct_v12_sdma_mqd(Struct): pass
class struct_v12_compute_mqd(Struct): pass
enum_amdgpu_vm_level = CEnum(ctypes.c_uint)
AMDGPU_VM_PDB2 = enum_amdgpu_vm_level.define('AMDGPU_VM_PDB2', 0)
AMDGPU_VM_PDB1 = enum_amdgpu_vm_level.define('AMDGPU_VM_PDB1', 1)
AMDGPU_VM_PDB0 = enum_amdgpu_vm_level.define('AMDGPU_VM_PDB0', 2)
AMDGPU_VM_PTB = enum_amdgpu_vm_level.define('AMDGPU_VM_PTB', 3)

table = CEnum(ctypes.c_uint)
IP_DISCOVERY = table.define('IP_DISCOVERY', 0)
GC = table.define('GC', 1)
HARVEST_INFO = table.define('HARVEST_INFO', 2)
VCN_INFO = table.define('VCN_INFO', 3)
MALL_INFO = table.define('MALL_INFO', 4)
NPS_INFO = table.define('NPS_INFO', 5)
TOTAL_TABLES = table.define('TOTAL_TABLES', 6)

class struct_table_info(Struct): pass
table_info = struct_table_info
class struct_binary_header(Struct): pass
binary_header = struct_binary_header
class struct_die_info(Struct): pass
die_info = struct_die_info
class struct_ip_discovery_header(Struct): pass
class _anonunion0(ctypes.Union): pass
class _anonstruct1(Struct): pass
ip_discovery_header = struct_ip_discovery_header
class struct_ip(Struct): pass
ip = struct_ip
class struct_ip_v3(Struct): pass
ip_v3 = struct_ip_v3
class struct_ip_v4(Struct): pass
ip_v4 = struct_ip_v4
class struct_die_header(Struct): pass
die_header = struct_die_header
class struct_ip_structure(Struct): pass
class struct_die(Struct): pass
class struct_die_0(ctypes.Union): pass
struct_die_0._fields_ = [
  ('ip_list', ctypes.POINTER(ip)),
  ('ip_v3_list', ctypes.POINTER(ip_v3)),
  ('ip_v4_list', ctypes.POINTER(ip_v4)),
]
struct_die._anonymous_ = ['_0']
struct_die._fields_ = [
  ('die_header', ctypes.POINTER(die_header)),
  ('_0', struct_die_0),
]
struct_ip_structure._fields_ = [
  ('header', ctypes.POINTER(ip_discovery_header)),
  ('die', struct_die),
]
ip_structure = struct_ip_structure
class struct_gpu_info_header(Struct): pass
class struct_gc_info_v1_0(Struct): pass
class struct_gc_info_v1_1(Struct): pass
class struct_gc_info_v1_2(Struct): pass
class struct_gc_info_v1_3(Struct): pass
class struct_gc_info_v2_0(Struct): pass
class struct_gc_info_v2_1(Struct): pass
class struct_harvest_info_header(Struct): pass
harvest_info_header = struct_harvest_info_header
class struct_harvest_info(Struct): pass
harvest_info = struct_harvest_info
class struct_harvest_table(Struct): pass
harvest_table = struct_harvest_table
class struct_mall_info_header(Struct): pass
class struct_mall_info_v1_0(Struct): pass
class struct_mall_info_v2_0(Struct): pass
class struct_vcn_info_header(Struct): pass
class struct_vcn_instance_info_v1_0(Struct): pass
class union__fuse_data(ctypes.Union): pass
class _anonstruct2(Struct): pass
class struct_vcn_info_v1_0(Struct): pass
class struct_nps_info_header(Struct): pass
class struct_nps_instance_info_v1_0(Struct): pass
class struct_nps_info_v1_0(Struct): pass
enum_amd_hw_ip_block_type = CEnum(ctypes.c_uint)
GC_HWIP = enum_amd_hw_ip_block_type.define('GC_HWIP', 1)
HDP_HWIP = enum_amd_hw_ip_block_type.define('HDP_HWIP', 2)
SDMA0_HWIP = enum_amd_hw_ip_block_type.define('SDMA0_HWIP', 3)
SDMA1_HWIP = enum_amd_hw_ip_block_type.define('SDMA1_HWIP', 4)
SDMA2_HWIP = enum_amd_hw_ip_block_type.define('SDMA2_HWIP', 5)
SDMA3_HWIP = enum_amd_hw_ip_block_type.define('SDMA3_HWIP', 6)
SDMA4_HWIP = enum_amd_hw_ip_block_type.define('SDMA4_HWIP', 7)
SDMA5_HWIP = enum_amd_hw_ip_block_type.define('SDMA5_HWIP', 8)
SDMA6_HWIP = enum_amd_hw_ip_block_type.define('SDMA6_HWIP', 9)
SDMA7_HWIP = enum_amd_hw_ip_block_type.define('SDMA7_HWIP', 10)
LSDMA_HWIP = enum_amd_hw_ip_block_type.define('LSDMA_HWIP', 11)
MMHUB_HWIP = enum_amd_hw_ip_block_type.define('MMHUB_HWIP', 12)
ATHUB_HWIP = enum_amd_hw_ip_block_type.define('ATHUB_HWIP', 13)
NBIO_HWIP = enum_amd_hw_ip_block_type.define('NBIO_HWIP', 14)
MP0_HWIP = enum_amd_hw_ip_block_type.define('MP0_HWIP', 15)
MP1_HWIP = enum_amd_hw_ip_block_type.define('MP1_HWIP', 16)
UVD_HWIP = enum_amd_hw_ip_block_type.define('UVD_HWIP', 17)
VCN_HWIP = enum_amd_hw_ip_block_type.define('VCN_HWIP', 17)
JPEG_HWIP = enum_amd_hw_ip_block_type.define('JPEG_HWIP', 17)
VCN1_HWIP = enum_amd_hw_ip_block_type.define('VCN1_HWIP', 18)
VCE_HWIP = enum_amd_hw_ip_block_type.define('VCE_HWIP', 19)
VPE_HWIP = enum_amd_hw_ip_block_type.define('VPE_HWIP', 20)
DF_HWIP = enum_amd_hw_ip_block_type.define('DF_HWIP', 21)
DCE_HWIP = enum_amd_hw_ip_block_type.define('DCE_HWIP', 22)
OSSSYS_HWIP = enum_amd_hw_ip_block_type.define('OSSSYS_HWIP', 23)
SMUIO_HWIP = enum_amd_hw_ip_block_type.define('SMUIO_HWIP', 24)
PWR_HWIP = enum_amd_hw_ip_block_type.define('PWR_HWIP', 25)
NBIF_HWIP = enum_amd_hw_ip_block_type.define('NBIF_HWIP', 26)
THM_HWIP = enum_amd_hw_ip_block_type.define('THM_HWIP', 27)
CLK_HWIP = enum_amd_hw_ip_block_type.define('CLK_HWIP', 28)
UMC_HWIP = enum_amd_hw_ip_block_type.define('UMC_HWIP', 29)
RSMU_HWIP = enum_amd_hw_ip_block_type.define('RSMU_HWIP', 30)
XGMI_HWIP = enum_amd_hw_ip_block_type.define('XGMI_HWIP', 31)
DCI_HWIP = enum_amd_hw_ip_block_type.define('DCI_HWIP', 32)
PCIE_HWIP = enum_amd_hw_ip_block_type.define('PCIE_HWIP', 33)
ISP_HWIP = enum_amd_hw_ip_block_type.define('ISP_HWIP', 34)
MAX_HWIP = enum_amd_hw_ip_block_type.define('MAX_HWIP', 35)

class struct_common_firmware_header(Struct): pass
struct_common_firmware_header._fields_ = [
  ('size_bytes', ctypes.c_uint),
  ('header_size_bytes', ctypes.c_uint),
  ('header_version_major', ctypes.c_ushort),
  ('header_version_minor', ctypes.c_ushort),
  ('ip_version_major', ctypes.c_ushort),
  ('ip_version_minor', ctypes.c_ushort),
  ('ucode_version', ctypes.c_uint),
  ('ucode_size_bytes', ctypes.c_uint),
  ('ucode_array_offset_bytes', ctypes.c_uint),
  ('crc32', ctypes.c_uint),
]
class struct_mc_firmware_header_v1_0(Struct): pass
struct_mc_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('io_debug_size_bytes', ctypes.c_uint),
  ('io_debug_array_offset_bytes', ctypes.c_uint),
]
class struct_smc_firmware_header_v1_0(Struct): pass
struct_smc_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('ucode_start_addr', ctypes.c_uint),
]
class struct_smc_firmware_header_v2_0(Struct): pass
struct_smc_firmware_header_v2_0._fields_ = [
  ('v1_0', struct_smc_firmware_header_v1_0),
  ('ppt_offset_bytes', ctypes.c_uint),
  ('ppt_size_bytes', ctypes.c_uint),
]
class struct_smc_soft_pptable_entry(Struct): pass
struct_smc_soft_pptable_entry._fields_ = [
  ('id', ctypes.c_uint),
  ('ppt_offset_bytes', ctypes.c_uint),
  ('ppt_size_bytes', ctypes.c_uint),
]
class struct_smc_firmware_header_v2_1(Struct): pass
struct_smc_firmware_header_v2_1._fields_ = [
  ('v1_0', struct_smc_firmware_header_v1_0),
  ('pptable_count', ctypes.c_uint),
  ('pptable_entry_offset', ctypes.c_uint),
]
class struct_psp_fw_legacy_bin_desc(Struct): pass
struct_psp_fw_legacy_bin_desc._fields_ = [
  ('fw_version', ctypes.c_uint),
  ('offset_bytes', ctypes.c_uint),
  ('size_bytes', ctypes.c_uint),
]
class struct_psp_firmware_header_v1_0(Struct): pass
struct_psp_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('sos', struct_psp_fw_legacy_bin_desc),
]
class struct_psp_firmware_header_v1_1(Struct): pass
struct_psp_firmware_header_v1_1._fields_ = [
  ('v1_0', struct_psp_firmware_header_v1_0),
  ('toc', struct_psp_fw_legacy_bin_desc),
  ('kdb', struct_psp_fw_legacy_bin_desc),
]
class struct_psp_firmware_header_v1_2(Struct): pass
struct_psp_firmware_header_v1_2._fields_ = [
  ('v1_0', struct_psp_firmware_header_v1_0),
  ('res', struct_psp_fw_legacy_bin_desc),
  ('kdb', struct_psp_fw_legacy_bin_desc),
]
class struct_psp_firmware_header_v1_3(Struct): pass
struct_psp_firmware_header_v1_3._fields_ = [
  ('v1_1', struct_psp_firmware_header_v1_1),
  ('spl', struct_psp_fw_legacy_bin_desc),
  ('rl', struct_psp_fw_legacy_bin_desc),
  ('sys_drv_aux', struct_psp_fw_legacy_bin_desc),
  ('sos_aux', struct_psp_fw_legacy_bin_desc),
]
class struct_psp_fw_bin_desc(Struct): pass
struct_psp_fw_bin_desc._fields_ = [
  ('fw_type', ctypes.c_uint),
  ('fw_version', ctypes.c_uint),
  ('offset_bytes', ctypes.c_uint),
  ('size_bytes', ctypes.c_uint),
]
enum_psp_fw_type = CEnum(ctypes.c_uint)
PSP_FW_TYPE_UNKOWN = enum_psp_fw_type.define('PSP_FW_TYPE_UNKOWN', 0)
PSP_FW_TYPE_PSP_SOS = enum_psp_fw_type.define('PSP_FW_TYPE_PSP_SOS', 1)
PSP_FW_TYPE_PSP_SYS_DRV = enum_psp_fw_type.define('PSP_FW_TYPE_PSP_SYS_DRV', 2)
PSP_FW_TYPE_PSP_KDB = enum_psp_fw_type.define('PSP_FW_TYPE_PSP_KDB', 3)
PSP_FW_TYPE_PSP_TOC = enum_psp_fw_type.define('PSP_FW_TYPE_PSP_TOC', 4)
PSP_FW_TYPE_PSP_SPL = enum_psp_fw_type.define('PSP_FW_TYPE_PSP_SPL', 5)
PSP_FW_TYPE_PSP_RL = enum_psp_fw_type.define('PSP_FW_TYPE_PSP_RL', 6)
PSP_FW_TYPE_PSP_SOC_DRV = enum_psp_fw_type.define('PSP_FW_TYPE_PSP_SOC_DRV', 7)
PSP_FW_TYPE_PSP_INTF_DRV = enum_psp_fw_type.define('PSP_FW_TYPE_PSP_INTF_DRV', 8)
PSP_FW_TYPE_PSP_DBG_DRV = enum_psp_fw_type.define('PSP_FW_TYPE_PSP_DBG_DRV', 9)
PSP_FW_TYPE_PSP_RAS_DRV = enum_psp_fw_type.define('PSP_FW_TYPE_PSP_RAS_DRV', 10)
PSP_FW_TYPE_PSP_IPKEYMGR_DRV = enum_psp_fw_type.define('PSP_FW_TYPE_PSP_IPKEYMGR_DRV', 11)
PSP_FW_TYPE_MAX_INDEX = enum_psp_fw_type.define('PSP_FW_TYPE_MAX_INDEX', 12)

class struct_psp_firmware_header_v2_0(Struct): pass
struct_psp_firmware_header_v2_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('psp_fw_bin_count', ctypes.c_uint),
  ('psp_fw_bin', (struct_psp_fw_bin_desc * 1)),
]
class struct_psp_firmware_header_v2_1(Struct): pass
struct_psp_firmware_header_v2_1._fields_ = [
  ('header', struct_common_firmware_header),
  ('psp_fw_bin_count', ctypes.c_uint),
  ('psp_aux_fw_bin_index', ctypes.c_uint),
  ('psp_fw_bin', (struct_psp_fw_bin_desc * 1)),
]
class struct_ta_firmware_header_v1_0(Struct): pass
struct_ta_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('xgmi', struct_psp_fw_legacy_bin_desc),
  ('ras', struct_psp_fw_legacy_bin_desc),
  ('hdcp', struct_psp_fw_legacy_bin_desc),
  ('dtm', struct_psp_fw_legacy_bin_desc),
  ('securedisplay', struct_psp_fw_legacy_bin_desc),
]
enum_ta_fw_type = CEnum(ctypes.c_uint)
TA_FW_TYPE_UNKOWN = enum_ta_fw_type.define('TA_FW_TYPE_UNKOWN', 0)
TA_FW_TYPE_PSP_ASD = enum_ta_fw_type.define('TA_FW_TYPE_PSP_ASD', 1)
TA_FW_TYPE_PSP_XGMI = enum_ta_fw_type.define('TA_FW_TYPE_PSP_XGMI', 2)
TA_FW_TYPE_PSP_RAS = enum_ta_fw_type.define('TA_FW_TYPE_PSP_RAS', 3)
TA_FW_TYPE_PSP_HDCP = enum_ta_fw_type.define('TA_FW_TYPE_PSP_HDCP', 4)
TA_FW_TYPE_PSP_DTM = enum_ta_fw_type.define('TA_FW_TYPE_PSP_DTM', 5)
TA_FW_TYPE_PSP_RAP = enum_ta_fw_type.define('TA_FW_TYPE_PSP_RAP', 6)
TA_FW_TYPE_PSP_SECUREDISPLAY = enum_ta_fw_type.define('TA_FW_TYPE_PSP_SECUREDISPLAY', 7)
TA_FW_TYPE_MAX_INDEX = enum_ta_fw_type.define('TA_FW_TYPE_MAX_INDEX', 8)

class struct_ta_firmware_header_v2_0(Struct): pass
struct_ta_firmware_header_v2_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('ta_fw_bin_count', ctypes.c_uint),
  ('ta_fw_bin', (struct_psp_fw_bin_desc * 1)),
]
class struct_gfx_firmware_header_v1_0(Struct): pass
struct_gfx_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('ucode_feature_version', ctypes.c_uint),
  ('jt_offset', ctypes.c_uint),
  ('jt_size', ctypes.c_uint),
]
class struct_gfx_firmware_header_v2_0(Struct): pass
struct_gfx_firmware_header_v2_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('ucode_feature_version', ctypes.c_uint),
  ('ucode_size_bytes', ctypes.c_uint),
  ('ucode_offset_bytes', ctypes.c_uint),
  ('data_size_bytes', ctypes.c_uint),
  ('data_offset_bytes', ctypes.c_uint),
  ('ucode_start_addr_lo', ctypes.c_uint),
  ('ucode_start_addr_hi', ctypes.c_uint),
]
class struct_mes_firmware_header_v1_0(Struct): pass
struct_mes_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('mes_ucode_version', ctypes.c_uint),
  ('mes_ucode_size_bytes', ctypes.c_uint),
  ('mes_ucode_offset_bytes', ctypes.c_uint),
  ('mes_ucode_data_version', ctypes.c_uint),
  ('mes_ucode_data_size_bytes', ctypes.c_uint),
  ('mes_ucode_data_offset_bytes', ctypes.c_uint),
  ('mes_uc_start_addr_lo', ctypes.c_uint),
  ('mes_uc_start_addr_hi', ctypes.c_uint),
  ('mes_data_start_addr_lo', ctypes.c_uint),
  ('mes_data_start_addr_hi', ctypes.c_uint),
]
class struct_rlc_firmware_header_v1_0(Struct): pass
struct_rlc_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('ucode_feature_version', ctypes.c_uint),
  ('save_and_restore_offset', ctypes.c_uint),
  ('clear_state_descriptor_offset', ctypes.c_uint),
  ('avail_scratch_ram_locations', ctypes.c_uint),
  ('master_pkt_description_offset', ctypes.c_uint),
]
class struct_rlc_firmware_header_v2_0(Struct): pass
struct_rlc_firmware_header_v2_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('ucode_feature_version', ctypes.c_uint),
  ('jt_offset', ctypes.c_uint),
  ('jt_size', ctypes.c_uint),
  ('save_and_restore_offset', ctypes.c_uint),
  ('clear_state_descriptor_offset', ctypes.c_uint),
  ('avail_scratch_ram_locations', ctypes.c_uint),
  ('reg_restore_list_size', ctypes.c_uint),
  ('reg_list_format_start', ctypes.c_uint),
  ('reg_list_format_separate_start', ctypes.c_uint),
  ('starting_offsets_start', ctypes.c_uint),
  ('reg_list_format_size_bytes', ctypes.c_uint),
  ('reg_list_format_array_offset_bytes', ctypes.c_uint),
  ('reg_list_size_bytes', ctypes.c_uint),
  ('reg_list_array_offset_bytes', ctypes.c_uint),
  ('reg_list_format_separate_size_bytes', ctypes.c_uint),
  ('reg_list_format_separate_array_offset_bytes', ctypes.c_uint),
  ('reg_list_separate_size_bytes', ctypes.c_uint),
  ('reg_list_separate_array_offset_bytes', ctypes.c_uint),
]
class struct_rlc_firmware_header_v2_1(Struct): pass
struct_rlc_firmware_header_v2_1._fields_ = [
  ('v2_0', struct_rlc_firmware_header_v2_0),
  ('reg_list_format_direct_reg_list_length', ctypes.c_uint),
  ('save_restore_list_cntl_ucode_ver', ctypes.c_uint),
  ('save_restore_list_cntl_feature_ver', ctypes.c_uint),
  ('save_restore_list_cntl_size_bytes', ctypes.c_uint),
  ('save_restore_list_cntl_offset_bytes', ctypes.c_uint),
  ('save_restore_list_gpm_ucode_ver', ctypes.c_uint),
  ('save_restore_list_gpm_feature_ver', ctypes.c_uint),
  ('save_restore_list_gpm_size_bytes', ctypes.c_uint),
  ('save_restore_list_gpm_offset_bytes', ctypes.c_uint),
  ('save_restore_list_srm_ucode_ver', ctypes.c_uint),
  ('save_restore_list_srm_feature_ver', ctypes.c_uint),
  ('save_restore_list_srm_size_bytes', ctypes.c_uint),
  ('save_restore_list_srm_offset_bytes', ctypes.c_uint),
]
class struct_rlc_firmware_header_v2_2(Struct): pass
struct_rlc_firmware_header_v2_2._fields_ = [
  ('v2_1', struct_rlc_firmware_header_v2_1),
  ('rlc_iram_ucode_size_bytes', ctypes.c_uint),
  ('rlc_iram_ucode_offset_bytes', ctypes.c_uint),
  ('rlc_dram_ucode_size_bytes', ctypes.c_uint),
  ('rlc_dram_ucode_offset_bytes', ctypes.c_uint),
]
class struct_rlc_firmware_header_v2_3(Struct): pass
struct_rlc_firmware_header_v2_3._fields_ = [
  ('v2_2', struct_rlc_firmware_header_v2_2),
  ('rlcp_ucode_version', ctypes.c_uint),
  ('rlcp_ucode_feature_version', ctypes.c_uint),
  ('rlcp_ucode_size_bytes', ctypes.c_uint),
  ('rlcp_ucode_offset_bytes', ctypes.c_uint),
  ('rlcv_ucode_version', ctypes.c_uint),
  ('rlcv_ucode_feature_version', ctypes.c_uint),
  ('rlcv_ucode_size_bytes', ctypes.c_uint),
  ('rlcv_ucode_offset_bytes', ctypes.c_uint),
]
class struct_rlc_firmware_header_v2_4(Struct): pass
struct_rlc_firmware_header_v2_4._fields_ = [
  ('v2_3', struct_rlc_firmware_header_v2_3),
  ('global_tap_delays_ucode_size_bytes', ctypes.c_uint),
  ('global_tap_delays_ucode_offset_bytes', ctypes.c_uint),
  ('se0_tap_delays_ucode_size_bytes', ctypes.c_uint),
  ('se0_tap_delays_ucode_offset_bytes', ctypes.c_uint),
  ('se1_tap_delays_ucode_size_bytes', ctypes.c_uint),
  ('se1_tap_delays_ucode_offset_bytes', ctypes.c_uint),
  ('se2_tap_delays_ucode_size_bytes', ctypes.c_uint),
  ('se2_tap_delays_ucode_offset_bytes', ctypes.c_uint),
  ('se3_tap_delays_ucode_size_bytes', ctypes.c_uint),
  ('se3_tap_delays_ucode_offset_bytes', ctypes.c_uint),
]
class struct_sdma_firmware_header_v1_0(Struct): pass
struct_sdma_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('ucode_feature_version', ctypes.c_uint),
  ('ucode_change_version', ctypes.c_uint),
  ('jt_offset', ctypes.c_uint),
  ('jt_size', ctypes.c_uint),
]
class struct_sdma_firmware_header_v1_1(Struct): pass
struct_sdma_firmware_header_v1_1._fields_ = [
  ('v1_0', struct_sdma_firmware_header_v1_0),
  ('digest_size', ctypes.c_uint),
]
class struct_sdma_firmware_header_v2_0(Struct): pass
struct_sdma_firmware_header_v2_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('ucode_feature_version', ctypes.c_uint),
  ('ctx_ucode_size_bytes', ctypes.c_uint),
  ('ctx_jt_offset', ctypes.c_uint),
  ('ctx_jt_size', ctypes.c_uint),
  ('ctl_ucode_offset', ctypes.c_uint),
  ('ctl_ucode_size_bytes', ctypes.c_uint),
  ('ctl_jt_offset', ctypes.c_uint),
  ('ctl_jt_size', ctypes.c_uint),
]
class struct_vpe_firmware_header_v1_0(Struct): pass
struct_vpe_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('ucode_feature_version', ctypes.c_uint),
  ('ctx_ucode_size_bytes', ctypes.c_uint),
  ('ctx_jt_offset', ctypes.c_uint),
  ('ctx_jt_size', ctypes.c_uint),
  ('ctl_ucode_offset', ctypes.c_uint),
  ('ctl_ucode_size_bytes', ctypes.c_uint),
  ('ctl_jt_offset', ctypes.c_uint),
  ('ctl_jt_size', ctypes.c_uint),
]
class struct_umsch_mm_firmware_header_v1_0(Struct): pass
struct_umsch_mm_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('umsch_mm_ucode_version', ctypes.c_uint),
  ('umsch_mm_ucode_size_bytes', ctypes.c_uint),
  ('umsch_mm_ucode_offset_bytes', ctypes.c_uint),
  ('umsch_mm_ucode_data_version', ctypes.c_uint),
  ('umsch_mm_ucode_data_size_bytes', ctypes.c_uint),
  ('umsch_mm_ucode_data_offset_bytes', ctypes.c_uint),
  ('umsch_mm_irq_start_addr_lo', ctypes.c_uint),
  ('umsch_mm_irq_start_addr_hi', ctypes.c_uint),
  ('umsch_mm_uc_start_addr_lo', ctypes.c_uint),
  ('umsch_mm_uc_start_addr_hi', ctypes.c_uint),
  ('umsch_mm_data_start_addr_lo', ctypes.c_uint),
  ('umsch_mm_data_start_addr_hi', ctypes.c_uint),
]
class struct_sdma_firmware_header_v3_0(Struct): pass
struct_sdma_firmware_header_v3_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('ucode_feature_version', ctypes.c_uint),
  ('ucode_offset_bytes', ctypes.c_uint),
  ('ucode_size_bytes', ctypes.c_uint),
]
class struct_gpu_info_firmware_v1_0(Struct): pass
struct_gpu_info_firmware_v1_0._fields_ = [
  ('gc_num_se', ctypes.c_uint),
  ('gc_num_cu_per_sh', ctypes.c_uint),
  ('gc_num_sh_per_se', ctypes.c_uint),
  ('gc_num_rb_per_se', ctypes.c_uint),
  ('gc_num_tccs', ctypes.c_uint),
  ('gc_num_gprs', ctypes.c_uint),
  ('gc_num_max_gs_thds', ctypes.c_uint),
  ('gc_gs_table_depth', ctypes.c_uint),
  ('gc_gsprim_buff_depth', ctypes.c_uint),
  ('gc_parameter_cache_depth', ctypes.c_uint),
  ('gc_double_offchip_lds_buffer', ctypes.c_uint),
  ('gc_wave_size', ctypes.c_uint),
  ('gc_max_waves_per_simd', ctypes.c_uint),
  ('gc_max_scratch_slots_per_cu', ctypes.c_uint),
  ('gc_lds_size', ctypes.c_uint),
]
class struct_gpu_info_firmware_v1_1(Struct): pass
struct_gpu_info_firmware_v1_1._fields_ = [
  ('v1_0', struct_gpu_info_firmware_v1_0),
  ('num_sc_per_sh', ctypes.c_uint),
  ('num_packer_per_sc', ctypes.c_uint),
]
class struct_gpu_info_firmware_header_v1_0(Struct): pass
struct_gpu_info_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('version_major', ctypes.c_ushort),
  ('version_minor', ctypes.c_ushort),
]
class struct_dmcu_firmware_header_v1_0(Struct): pass
struct_dmcu_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('intv_offset_bytes', ctypes.c_uint),
  ('intv_size_bytes', ctypes.c_uint),
]
class struct_dmcub_firmware_header_v1_0(Struct): pass
struct_dmcub_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('inst_const_bytes', ctypes.c_uint),
  ('bss_data_bytes', ctypes.c_uint),
]
class struct_imu_firmware_header_v1_0(Struct): pass
struct_imu_firmware_header_v1_0._fields_ = [
  ('header', struct_common_firmware_header),
  ('imu_iram_ucode_size_bytes', ctypes.c_uint),
  ('imu_iram_ucode_offset_bytes', ctypes.c_uint),
  ('imu_dram_ucode_size_bytes', ctypes.c_uint),
  ('imu_dram_ucode_offset_bytes', ctypes.c_uint),
]
class union_amdgpu_firmware_header(ctypes.Union): pass
union_amdgpu_firmware_header._fields_ = [
  ('common', struct_common_firmware_header),
  ('mc', struct_mc_firmware_header_v1_0),
  ('smc', struct_smc_firmware_header_v1_0),
  ('smc_v2_0', struct_smc_firmware_header_v2_0),
  ('psp', struct_psp_firmware_header_v1_0),
  ('psp_v1_1', struct_psp_firmware_header_v1_1),
  ('psp_v1_3', struct_psp_firmware_header_v1_3),
  ('psp_v2_0', struct_psp_firmware_header_v2_0),
  ('psp_v2_1', struct_psp_firmware_header_v2_0),
  ('ta', struct_ta_firmware_header_v1_0),
  ('ta_v2_0', struct_ta_firmware_header_v2_0),
  ('gfx', struct_gfx_firmware_header_v1_0),
  ('gfx_v2_0', struct_gfx_firmware_header_v2_0),
  ('rlc', struct_rlc_firmware_header_v1_0),
  ('rlc_v2_0', struct_rlc_firmware_header_v2_0),
  ('rlc_v2_1', struct_rlc_firmware_header_v2_1),
  ('rlc_v2_2', struct_rlc_firmware_header_v2_2),
  ('rlc_v2_3', struct_rlc_firmware_header_v2_3),
  ('rlc_v2_4', struct_rlc_firmware_header_v2_4),
  ('sdma', struct_sdma_firmware_header_v1_0),
  ('sdma_v1_1', struct_sdma_firmware_header_v1_1),
  ('sdma_v2_0', struct_sdma_firmware_header_v2_0),
  ('sdma_v3_0', struct_sdma_firmware_header_v3_0),
  ('gpu_info', struct_gpu_info_firmware_header_v1_0),
  ('dmcu', struct_dmcu_firmware_header_v1_0),
  ('dmcub', struct_dmcub_firmware_header_v1_0),
  ('imu', struct_imu_firmware_header_v1_0),
  ('raw', (ctypes.c_ubyte * 256)),
]
enum_AMDGPU_UCODE_ID = CEnum(ctypes.c_uint)
AMDGPU_UCODE_ID_CAP = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CAP', 0)
AMDGPU_UCODE_ID_SDMA0 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SDMA0', 1)
AMDGPU_UCODE_ID_SDMA1 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SDMA1', 2)
AMDGPU_UCODE_ID_SDMA2 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SDMA2', 3)
AMDGPU_UCODE_ID_SDMA3 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SDMA3', 4)
AMDGPU_UCODE_ID_SDMA4 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SDMA4', 5)
AMDGPU_UCODE_ID_SDMA5 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SDMA5', 6)
AMDGPU_UCODE_ID_SDMA6 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SDMA6', 7)
AMDGPU_UCODE_ID_SDMA7 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SDMA7', 8)
AMDGPU_UCODE_ID_SDMA_UCODE_TH0 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SDMA_UCODE_TH0', 9)
AMDGPU_UCODE_ID_SDMA_UCODE_TH1 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SDMA_UCODE_TH1', 10)
AMDGPU_UCODE_ID_SDMA_RS64 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SDMA_RS64', 11)
AMDGPU_UCODE_ID_CP_CE = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_CE', 12)
AMDGPU_UCODE_ID_CP_PFP = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_PFP', 13)
AMDGPU_UCODE_ID_CP_ME = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_ME', 14)
AMDGPU_UCODE_ID_CP_RS64_PFP = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_RS64_PFP', 15)
AMDGPU_UCODE_ID_CP_RS64_ME = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_RS64_ME', 16)
AMDGPU_UCODE_ID_CP_RS64_MEC = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_RS64_MEC', 17)
AMDGPU_UCODE_ID_CP_RS64_PFP_P0_STACK = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_RS64_PFP_P0_STACK', 18)
AMDGPU_UCODE_ID_CP_RS64_PFP_P1_STACK = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_RS64_PFP_P1_STACK', 19)
AMDGPU_UCODE_ID_CP_RS64_ME_P0_STACK = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_RS64_ME_P0_STACK', 20)
AMDGPU_UCODE_ID_CP_RS64_ME_P1_STACK = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_RS64_ME_P1_STACK', 21)
AMDGPU_UCODE_ID_CP_RS64_MEC_P0_STACK = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_RS64_MEC_P0_STACK', 22)
AMDGPU_UCODE_ID_CP_RS64_MEC_P1_STACK = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_RS64_MEC_P1_STACK', 23)
AMDGPU_UCODE_ID_CP_RS64_MEC_P2_STACK = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_RS64_MEC_P2_STACK', 24)
AMDGPU_UCODE_ID_CP_RS64_MEC_P3_STACK = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_RS64_MEC_P3_STACK', 25)
AMDGPU_UCODE_ID_CP_MEC1 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_MEC1', 26)
AMDGPU_UCODE_ID_CP_MEC1_JT = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_MEC1_JT', 27)
AMDGPU_UCODE_ID_CP_MEC2 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_MEC2', 28)
AMDGPU_UCODE_ID_CP_MEC2_JT = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_MEC2_JT', 29)
AMDGPU_UCODE_ID_CP_MES = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_MES', 30)
AMDGPU_UCODE_ID_CP_MES_DATA = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_MES_DATA', 31)
AMDGPU_UCODE_ID_CP_MES1 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_MES1', 32)
AMDGPU_UCODE_ID_CP_MES1_DATA = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_CP_MES1_DATA', 33)
AMDGPU_UCODE_ID_IMU_I = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_IMU_I', 34)
AMDGPU_UCODE_ID_IMU_D = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_IMU_D', 35)
AMDGPU_UCODE_ID_GLOBAL_TAP_DELAYS = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_GLOBAL_TAP_DELAYS', 36)
AMDGPU_UCODE_ID_SE0_TAP_DELAYS = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SE0_TAP_DELAYS', 37)
AMDGPU_UCODE_ID_SE1_TAP_DELAYS = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SE1_TAP_DELAYS', 38)
AMDGPU_UCODE_ID_SE2_TAP_DELAYS = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SE2_TAP_DELAYS', 39)
AMDGPU_UCODE_ID_SE3_TAP_DELAYS = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SE3_TAP_DELAYS', 40)
AMDGPU_UCODE_ID_RLC_RESTORE_LIST_CNTL = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_RLC_RESTORE_LIST_CNTL', 41)
AMDGPU_UCODE_ID_RLC_RESTORE_LIST_GPM_MEM = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_RLC_RESTORE_LIST_GPM_MEM', 42)
AMDGPU_UCODE_ID_RLC_RESTORE_LIST_SRM_MEM = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_RLC_RESTORE_LIST_SRM_MEM', 43)
AMDGPU_UCODE_ID_RLC_IRAM = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_RLC_IRAM', 44)
AMDGPU_UCODE_ID_RLC_DRAM = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_RLC_DRAM', 45)
AMDGPU_UCODE_ID_RLC_P = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_RLC_P', 46)
AMDGPU_UCODE_ID_RLC_V = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_RLC_V', 47)
AMDGPU_UCODE_ID_RLC_G = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_RLC_G', 48)
AMDGPU_UCODE_ID_STORAGE = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_STORAGE', 49)
AMDGPU_UCODE_ID_SMC = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_SMC', 50)
AMDGPU_UCODE_ID_PPTABLE = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_PPTABLE', 51)
AMDGPU_UCODE_ID_UVD = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_UVD', 52)
AMDGPU_UCODE_ID_UVD1 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_UVD1', 53)
AMDGPU_UCODE_ID_VCE = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_VCE', 54)
AMDGPU_UCODE_ID_VCN = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_VCN', 55)
AMDGPU_UCODE_ID_VCN1 = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_VCN1', 56)
AMDGPU_UCODE_ID_DMCU_ERAM = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_DMCU_ERAM', 57)
AMDGPU_UCODE_ID_DMCU_INTV = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_DMCU_INTV', 58)
AMDGPU_UCODE_ID_VCN0_RAM = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_VCN0_RAM', 59)
AMDGPU_UCODE_ID_VCN1_RAM = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_VCN1_RAM', 60)
AMDGPU_UCODE_ID_DMCUB = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_DMCUB', 61)
AMDGPU_UCODE_ID_VPE_CTX = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_VPE_CTX', 62)
AMDGPU_UCODE_ID_VPE_CTL = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_VPE_CTL', 63)
AMDGPU_UCODE_ID_VPE = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_VPE', 64)
AMDGPU_UCODE_ID_UMSCH_MM_UCODE = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_UMSCH_MM_UCODE', 65)
AMDGPU_UCODE_ID_UMSCH_MM_DATA = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_UMSCH_MM_DATA', 66)
AMDGPU_UCODE_ID_UMSCH_MM_CMD_BUFFER = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_UMSCH_MM_CMD_BUFFER', 67)
AMDGPU_UCODE_ID_P2S_TABLE = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_P2S_TABLE', 68)
AMDGPU_UCODE_ID_JPEG_RAM = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_JPEG_RAM', 69)
AMDGPU_UCODE_ID_ISP = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_ISP', 70)
AMDGPU_UCODE_ID_MAXIMUM = enum_AMDGPU_UCODE_ID.define('AMDGPU_UCODE_ID_MAXIMUM', 71)

enum_AMDGPU_UCODE_STATUS = CEnum(ctypes.c_uint)
AMDGPU_UCODE_STATUS_INVALID = enum_AMDGPU_UCODE_STATUS.define('AMDGPU_UCODE_STATUS_INVALID', 0)
AMDGPU_UCODE_STATUS_NOT_LOADED = enum_AMDGPU_UCODE_STATUS.define('AMDGPU_UCODE_STATUS_NOT_LOADED', 1)
AMDGPU_UCODE_STATUS_LOADED = enum_AMDGPU_UCODE_STATUS.define('AMDGPU_UCODE_STATUS_LOADED', 2)

enum_amdgpu_firmware_load_type = CEnum(ctypes.c_uint)
AMDGPU_FW_LOAD_DIRECT = enum_amdgpu_firmware_load_type.define('AMDGPU_FW_LOAD_DIRECT', 0)
AMDGPU_FW_LOAD_PSP = enum_amdgpu_firmware_load_type.define('AMDGPU_FW_LOAD_PSP', 1)
AMDGPU_FW_LOAD_SMU = enum_amdgpu_firmware_load_type.define('AMDGPU_FW_LOAD_SMU', 2)
AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO = enum_amdgpu_firmware_load_type.define('AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO', 3)

class struct_amdgpu_firmware_info(Struct): pass
class struct_firmware(Struct): pass
struct_amdgpu_firmware_info._fields_ = [
  ('ucode_id', enum_AMDGPU_UCODE_ID),
  ('fw', ctypes.POINTER(struct_firmware)),
  ('mc_addr', ctypes.c_ulonglong),
  ('kaddr', ctypes.c_void_p),
  ('ucode_size', ctypes.c_uint),
  ('tmr_mc_addr_lo', ctypes.c_uint),
  ('tmr_mc_addr_hi', ctypes.c_uint),
]
enum_psp_gfx_crtl_cmd_id = CEnum(ctypes.c_uint)
GFX_CTRL_CMD_ID_INIT_RBI_RING = enum_psp_gfx_crtl_cmd_id.define('GFX_CTRL_CMD_ID_INIT_RBI_RING', 65536)
GFX_CTRL_CMD_ID_INIT_GPCOM_RING = enum_psp_gfx_crtl_cmd_id.define('GFX_CTRL_CMD_ID_INIT_GPCOM_RING', 131072)
GFX_CTRL_CMD_ID_DESTROY_RINGS = enum_psp_gfx_crtl_cmd_id.define('GFX_CTRL_CMD_ID_DESTROY_RINGS', 196608)
GFX_CTRL_CMD_ID_CAN_INIT_RINGS = enum_psp_gfx_crtl_cmd_id.define('GFX_CTRL_CMD_ID_CAN_INIT_RINGS', 262144)
GFX_CTRL_CMD_ID_ENABLE_INT = enum_psp_gfx_crtl_cmd_id.define('GFX_CTRL_CMD_ID_ENABLE_INT', 327680)
GFX_CTRL_CMD_ID_DISABLE_INT = enum_psp_gfx_crtl_cmd_id.define('GFX_CTRL_CMD_ID_DISABLE_INT', 393216)
GFX_CTRL_CMD_ID_MODE1_RST = enum_psp_gfx_crtl_cmd_id.define('GFX_CTRL_CMD_ID_MODE1_RST', 458752)
GFX_CTRL_CMD_ID_GBR_IH_SET = enum_psp_gfx_crtl_cmd_id.define('GFX_CTRL_CMD_ID_GBR_IH_SET', 524288)
GFX_CTRL_CMD_ID_CONSUME_CMD = enum_psp_gfx_crtl_cmd_id.define('GFX_CTRL_CMD_ID_CONSUME_CMD', 589824)
GFX_CTRL_CMD_ID_DESTROY_GPCOM_RING = enum_psp_gfx_crtl_cmd_id.define('GFX_CTRL_CMD_ID_DESTROY_GPCOM_RING', 786432)
GFX_CTRL_CMD_ID_MAX = enum_psp_gfx_crtl_cmd_id.define('GFX_CTRL_CMD_ID_MAX', 983040)

class struct_psp_gfx_ctrl(Struct): pass
struct_psp_gfx_ctrl._fields_ = [
  ('cmd_resp', ctypes.c_uint),
  ('rbi_wptr', ctypes.c_uint),
  ('rbi_rptr', ctypes.c_uint),
  ('gpcom_wptr', ctypes.c_uint),
  ('gpcom_rptr', ctypes.c_uint),
  ('ring_addr_lo', ctypes.c_uint),
  ('ring_addr_hi', ctypes.c_uint),
  ('ring_buf_size', ctypes.c_uint),
]
enum_psp_gfx_cmd_id = CEnum(ctypes.c_uint)
GFX_CMD_ID_LOAD_TA = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_LOAD_TA', 1)
GFX_CMD_ID_UNLOAD_TA = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_UNLOAD_TA', 2)
GFX_CMD_ID_INVOKE_CMD = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_INVOKE_CMD', 3)
GFX_CMD_ID_LOAD_ASD = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_LOAD_ASD', 4)
GFX_CMD_ID_SETUP_TMR = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_SETUP_TMR', 5)
GFX_CMD_ID_LOAD_IP_FW = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_LOAD_IP_FW', 6)
GFX_CMD_ID_DESTROY_TMR = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_DESTROY_TMR', 7)
GFX_CMD_ID_SAVE_RESTORE = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_SAVE_RESTORE', 8)
GFX_CMD_ID_SETUP_VMR = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_SETUP_VMR', 9)
GFX_CMD_ID_DESTROY_VMR = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_DESTROY_VMR', 10)
GFX_CMD_ID_PROG_REG = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_PROG_REG', 11)
GFX_CMD_ID_GET_FW_ATTESTATION = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_GET_FW_ATTESTATION', 15)
GFX_CMD_ID_LOAD_TOC = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_LOAD_TOC', 32)
GFX_CMD_ID_AUTOLOAD_RLC = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_AUTOLOAD_RLC', 33)
GFX_CMD_ID_BOOT_CFG = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_BOOT_CFG', 34)
GFX_CMD_ID_SRIOV_SPATIAL_PART = enum_psp_gfx_cmd_id.define('GFX_CMD_ID_SRIOV_SPATIAL_PART', 39)

enum_psp_gfx_boot_config_cmd = CEnum(ctypes.c_uint)
BOOTCFG_CMD_SET = enum_psp_gfx_boot_config_cmd.define('BOOTCFG_CMD_SET', 1)
BOOTCFG_CMD_GET = enum_psp_gfx_boot_config_cmd.define('BOOTCFG_CMD_GET', 2)
BOOTCFG_CMD_INVALIDATE = enum_psp_gfx_boot_config_cmd.define('BOOTCFG_CMD_INVALIDATE', 3)

enum_psp_gfx_boot_config = CEnum(ctypes.c_uint)
BOOT_CONFIG_GECC = enum_psp_gfx_boot_config.define('BOOT_CONFIG_GECC', 1)

class struct_psp_gfx_cmd_load_ta(Struct): pass
struct_psp_gfx_cmd_load_ta._fields_ = [
  ('app_phy_addr_lo', ctypes.c_uint),
  ('app_phy_addr_hi', ctypes.c_uint),
  ('app_len', ctypes.c_uint),
  ('cmd_buf_phy_addr_lo', ctypes.c_uint),
  ('cmd_buf_phy_addr_hi', ctypes.c_uint),
  ('cmd_buf_len', ctypes.c_uint),
]
class struct_psp_gfx_cmd_unload_ta(Struct): pass
struct_psp_gfx_cmd_unload_ta._fields_ = [
  ('session_id', ctypes.c_uint),
]
class struct_psp_gfx_buf_desc(Struct): pass
struct_psp_gfx_buf_desc._fields_ = [
  ('buf_phy_addr_lo', ctypes.c_uint),
  ('buf_phy_addr_hi', ctypes.c_uint),
  ('buf_size', ctypes.c_uint),
]
class struct_psp_gfx_buf_list(Struct): pass
struct_psp_gfx_buf_list._fields_ = [
  ('num_desc', ctypes.c_uint),
  ('total_size', ctypes.c_uint),
  ('buf_desc', (struct_psp_gfx_buf_desc * 64)),
]
class struct_psp_gfx_cmd_invoke_cmd(Struct): pass
struct_psp_gfx_cmd_invoke_cmd._fields_ = [
  ('session_id', ctypes.c_uint),
  ('ta_cmd_id', ctypes.c_uint),
  ('buf', struct_psp_gfx_buf_list),
]
class struct_psp_gfx_cmd_setup_tmr(Struct): pass
class struct_psp_gfx_cmd_setup_tmr_0(ctypes.Union): pass
class struct_psp_gfx_cmd_setup_tmr_0_bitfield(Struct): pass
struct_psp_gfx_cmd_setup_tmr_0_bitfield._fields_ = [
  ('sriov_enabled', ctypes.c_uint,1),
  ('virt_phy_addr', ctypes.c_uint,1),
  ('reserved', ctypes.c_uint,30),
]
struct_psp_gfx_cmd_setup_tmr_0._fields_ = [
  ('bitfield', struct_psp_gfx_cmd_setup_tmr_0_bitfield),
  ('tmr_flags', ctypes.c_uint),
]
struct_psp_gfx_cmd_setup_tmr._anonymous_ = ['_0']
struct_psp_gfx_cmd_setup_tmr._fields_ = [
  ('buf_phy_addr_lo', ctypes.c_uint),
  ('buf_phy_addr_hi', ctypes.c_uint),
  ('buf_size', ctypes.c_uint),
  ('_0', struct_psp_gfx_cmd_setup_tmr_0),
  ('system_phy_addr_lo', ctypes.c_uint),
  ('system_phy_addr_hi', ctypes.c_uint),
]
enum_psp_gfx_fw_type = CEnum(ctypes.c_uint)
GFX_FW_TYPE_NONE = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_NONE', 0)
GFX_FW_TYPE_CP_ME = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_CP_ME', 1)
GFX_FW_TYPE_CP_PFP = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_CP_PFP', 2)
GFX_FW_TYPE_CP_CE = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_CP_CE', 3)
GFX_FW_TYPE_CP_MEC = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_CP_MEC', 4)
GFX_FW_TYPE_CP_MEC_ME1 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_CP_MEC_ME1', 5)
GFX_FW_TYPE_CP_MEC_ME2 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_CP_MEC_ME2', 6)
GFX_FW_TYPE_RLC_V = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLC_V', 7)
GFX_FW_TYPE_RLC_G = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLC_G', 8)
GFX_FW_TYPE_SDMA0 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA0', 9)
GFX_FW_TYPE_SDMA1 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA1', 10)
GFX_FW_TYPE_DMCU_ERAM = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_DMCU_ERAM', 11)
GFX_FW_TYPE_DMCU_ISR = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_DMCU_ISR', 12)
GFX_FW_TYPE_VCN = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_VCN', 13)
GFX_FW_TYPE_UVD = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_UVD', 14)
GFX_FW_TYPE_VCE = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_VCE', 15)
GFX_FW_TYPE_ISP = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_ISP', 16)
GFX_FW_TYPE_ACP = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_ACP', 17)
GFX_FW_TYPE_SMU = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SMU', 18)
GFX_FW_TYPE_MMSCH = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_MMSCH', 19)
GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM', 20)
GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM', 21)
GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL', 22)
GFX_FW_TYPE_UVD1 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_UVD1', 23)
GFX_FW_TYPE_TOC = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_TOC', 24)
GFX_FW_TYPE_RLC_P = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLC_P', 25)
GFX_FW_TYPE_RLC_IRAM = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLC_IRAM', 26)
GFX_FW_TYPE_GLOBAL_TAP_DELAYS = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_GLOBAL_TAP_DELAYS', 27)
GFX_FW_TYPE_SE0_TAP_DELAYS = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SE0_TAP_DELAYS', 28)
GFX_FW_TYPE_SE1_TAP_DELAYS = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SE1_TAP_DELAYS', 29)
GFX_FW_TYPE_GLOBAL_SE0_SE1_SKEW_DELAYS = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_GLOBAL_SE0_SE1_SKEW_DELAYS', 30)
GFX_FW_TYPE_SDMA0_JT = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA0_JT', 31)
GFX_FW_TYPE_SDMA1_JT = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA1_JT', 32)
GFX_FW_TYPE_CP_MES = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_CP_MES', 33)
GFX_FW_TYPE_MES_STACK = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_MES_STACK', 34)
GFX_FW_TYPE_RLC_SRM_DRAM_SR = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLC_SRM_DRAM_SR', 35)
GFX_FW_TYPE_RLCG_SCRATCH_SR = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLCG_SCRATCH_SR', 36)
GFX_FW_TYPE_RLCP_SCRATCH_SR = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLCP_SCRATCH_SR', 37)
GFX_FW_TYPE_RLCV_SCRATCH_SR = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLCV_SCRATCH_SR', 38)
GFX_FW_TYPE_RLX6_DRAM_SR = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLX6_DRAM_SR', 39)
GFX_FW_TYPE_SDMA0_PG_CONTEXT = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA0_PG_CONTEXT', 40)
GFX_FW_TYPE_SDMA1_PG_CONTEXT = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA1_PG_CONTEXT', 41)
GFX_FW_TYPE_GLOBAL_MUX_SELECT_RAM = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_GLOBAL_MUX_SELECT_RAM', 42)
GFX_FW_TYPE_SE0_MUX_SELECT_RAM = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SE0_MUX_SELECT_RAM', 43)
GFX_FW_TYPE_SE1_MUX_SELECT_RAM = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SE1_MUX_SELECT_RAM', 44)
GFX_FW_TYPE_ACCUM_CTRL_RAM = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_ACCUM_CTRL_RAM', 45)
GFX_FW_TYPE_RLCP_CAM = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLCP_CAM', 46)
GFX_FW_TYPE_RLC_SPP_CAM_EXT = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLC_SPP_CAM_EXT', 47)
GFX_FW_TYPE_RLC_DRAM_BOOT = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RLC_DRAM_BOOT', 48)
GFX_FW_TYPE_VCN0_RAM = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_VCN0_RAM', 49)
GFX_FW_TYPE_VCN1_RAM = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_VCN1_RAM', 50)
GFX_FW_TYPE_DMUB = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_DMUB', 51)
GFX_FW_TYPE_SDMA2 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA2', 52)
GFX_FW_TYPE_SDMA3 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA3', 53)
GFX_FW_TYPE_SDMA4 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA4', 54)
GFX_FW_TYPE_SDMA5 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA5', 55)
GFX_FW_TYPE_SDMA6 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA6', 56)
GFX_FW_TYPE_SDMA7 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA7', 57)
GFX_FW_TYPE_VCN1 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_VCN1', 58)
GFX_FW_TYPE_CAP = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_CAP', 62)
GFX_FW_TYPE_SE2_TAP_DELAYS = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SE2_TAP_DELAYS', 65)
GFX_FW_TYPE_SE3_TAP_DELAYS = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SE3_TAP_DELAYS', 66)
GFX_FW_TYPE_REG_LIST = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_REG_LIST', 67)
GFX_FW_TYPE_IMU_I = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_IMU_I', 68)
GFX_FW_TYPE_IMU_D = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_IMU_D', 69)
GFX_FW_TYPE_LSDMA = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_LSDMA', 70)
GFX_FW_TYPE_SDMA_UCODE_TH0 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA_UCODE_TH0', 71)
GFX_FW_TYPE_SDMA_UCODE_TH1 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_SDMA_UCODE_TH1', 72)
GFX_FW_TYPE_PPTABLE = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_PPTABLE', 73)
GFX_FW_TYPE_DISCRETE_USB4 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_DISCRETE_USB4', 74)
GFX_FW_TYPE_TA = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_TA', 75)
GFX_FW_TYPE_RS64_MES = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_MES', 76)
GFX_FW_TYPE_RS64_MES_STACK = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_MES_STACK', 77)
GFX_FW_TYPE_RS64_KIQ = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_KIQ', 78)
GFX_FW_TYPE_RS64_KIQ_STACK = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_KIQ_STACK', 79)
GFX_FW_TYPE_ISP_DATA = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_ISP_DATA', 80)
GFX_FW_TYPE_CP_MES_KIQ = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_CP_MES_KIQ', 81)
GFX_FW_TYPE_MES_KIQ_STACK = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_MES_KIQ_STACK', 82)
GFX_FW_TYPE_UMSCH_DATA = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_UMSCH_DATA', 83)
GFX_FW_TYPE_UMSCH_UCODE = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_UMSCH_UCODE', 84)
GFX_FW_TYPE_UMSCH_CMD_BUFFER = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_UMSCH_CMD_BUFFER', 85)
GFX_FW_TYPE_USB_DP_COMBO_PHY = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_USB_DP_COMBO_PHY', 86)
GFX_FW_TYPE_RS64_PFP = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_PFP', 87)
GFX_FW_TYPE_RS64_ME = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_ME', 88)
GFX_FW_TYPE_RS64_MEC = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_MEC', 89)
GFX_FW_TYPE_RS64_PFP_P0_STACK = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_PFP_P0_STACK', 90)
GFX_FW_TYPE_RS64_PFP_P1_STACK = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_PFP_P1_STACK', 91)
GFX_FW_TYPE_RS64_ME_P0_STACK = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_ME_P0_STACK', 92)
GFX_FW_TYPE_RS64_ME_P1_STACK = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_ME_P1_STACK', 93)
GFX_FW_TYPE_RS64_MEC_P0_STACK = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_MEC_P0_STACK', 94)
GFX_FW_TYPE_RS64_MEC_P1_STACK = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_MEC_P1_STACK', 95)
GFX_FW_TYPE_RS64_MEC_P2_STACK = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_MEC_P2_STACK', 96)
GFX_FW_TYPE_RS64_MEC_P3_STACK = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_RS64_MEC_P3_STACK', 97)
GFX_FW_TYPE_VPEC_FW1 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_VPEC_FW1', 100)
GFX_FW_TYPE_VPEC_FW2 = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_VPEC_FW2', 101)
GFX_FW_TYPE_VPE = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_VPE', 102)
GFX_FW_TYPE_JPEG_RAM = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_JPEG_RAM', 128)
GFX_FW_TYPE_P2S_TABLE = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_P2S_TABLE', 129)
GFX_FW_TYPE_MAX = enum_psp_gfx_fw_type.define('GFX_FW_TYPE_MAX', 130)

class struct_psp_gfx_cmd_load_ip_fw(Struct): pass
struct_psp_gfx_cmd_load_ip_fw._fields_ = [
  ('fw_phy_addr_lo', ctypes.c_uint),
  ('fw_phy_addr_hi', ctypes.c_uint),
  ('fw_size', ctypes.c_uint),
  ('fw_type', enum_psp_gfx_fw_type),
]
class struct_psp_gfx_cmd_save_restore_ip_fw(Struct): pass
struct_psp_gfx_cmd_save_restore_ip_fw._fields_ = [
  ('save_fw', ctypes.c_uint),
  ('save_restore_addr_lo', ctypes.c_uint),
  ('save_restore_addr_hi', ctypes.c_uint),
  ('buf_size', ctypes.c_uint),
  ('fw_type', enum_psp_gfx_fw_type),
]
class struct_psp_gfx_cmd_reg_prog(Struct): pass
struct_psp_gfx_cmd_reg_prog._fields_ = [
  ('reg_value', ctypes.c_uint),
  ('reg_id', ctypes.c_uint),
]
class struct_psp_gfx_cmd_load_toc(Struct): pass
struct_psp_gfx_cmd_load_toc._fields_ = [
  ('toc_phy_addr_lo', ctypes.c_uint),
  ('toc_phy_addr_hi', ctypes.c_uint),
  ('toc_size', ctypes.c_uint),
]
class struct_psp_gfx_cmd_boot_cfg(Struct): pass
struct_psp_gfx_cmd_boot_cfg._fields_ = [
  ('timestamp', ctypes.c_uint),
  ('sub_cmd', enum_psp_gfx_boot_config_cmd),
  ('boot_config', ctypes.c_uint),
  ('boot_config_valid', ctypes.c_uint),
]
class struct_psp_gfx_cmd_sriov_spatial_part(Struct): pass
struct_psp_gfx_cmd_sriov_spatial_part._fields_ = [
  ('mode', ctypes.c_uint),
  ('override_ips', ctypes.c_uint),
  ('override_xcds_avail', ctypes.c_uint),
  ('override_this_aid', ctypes.c_uint),
]
class union_psp_gfx_commands(ctypes.Union): pass
union_psp_gfx_commands._fields_ = [
  ('cmd_load_ta', struct_psp_gfx_cmd_load_ta),
  ('cmd_unload_ta', struct_psp_gfx_cmd_unload_ta),
  ('cmd_invoke_cmd', struct_psp_gfx_cmd_invoke_cmd),
  ('cmd_setup_tmr', struct_psp_gfx_cmd_setup_tmr),
  ('cmd_load_ip_fw', struct_psp_gfx_cmd_load_ip_fw),
  ('cmd_save_restore_ip_fw', struct_psp_gfx_cmd_save_restore_ip_fw),
  ('cmd_setup_reg_prog', struct_psp_gfx_cmd_reg_prog),
  ('cmd_setup_vmr', struct_psp_gfx_cmd_setup_tmr),
  ('cmd_load_toc', struct_psp_gfx_cmd_load_toc),
  ('boot_cfg', struct_psp_gfx_cmd_boot_cfg),
  ('cmd_spatial_part', struct_psp_gfx_cmd_sriov_spatial_part),
]
class struct_psp_gfx_uresp_reserved(Struct): pass
struct_psp_gfx_uresp_reserved._fields_ = [
  ('reserved', (ctypes.c_uint * 8)),
]
class struct_psp_gfx_uresp_fwar_db_info(Struct): pass
struct_psp_gfx_uresp_fwar_db_info._fields_ = [
  ('fwar_db_addr_lo', ctypes.c_uint),
  ('fwar_db_addr_hi', ctypes.c_uint),
]
class struct_psp_gfx_uresp_bootcfg(Struct): pass
struct_psp_gfx_uresp_bootcfg._fields_ = [
  ('boot_cfg', ctypes.c_uint),
]
class union_psp_gfx_uresp(ctypes.Union): pass
union_psp_gfx_uresp._fields_ = [
  ('reserved', struct_psp_gfx_uresp_reserved),
  ('boot_cfg', struct_psp_gfx_uresp_bootcfg),
  ('fwar_db_info', struct_psp_gfx_uresp_fwar_db_info),
]
class struct_psp_gfx_resp(Struct): pass
struct_psp_gfx_resp._fields_ = [
  ('status', ctypes.c_uint),
  ('session_id', ctypes.c_uint),
  ('fw_addr_lo', ctypes.c_uint),
  ('fw_addr_hi', ctypes.c_uint),
  ('tmr_size', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 11)),
  ('uresp', union_psp_gfx_uresp),
]
class struct_psp_gfx_cmd_resp(Struct): pass
struct_psp_gfx_cmd_resp._fields_ = [
  ('buf_size', ctypes.c_uint),
  ('buf_version', ctypes.c_uint),
  ('cmd_id', ctypes.c_uint),
  ('resp_buf_addr_lo', ctypes.c_uint),
  ('resp_buf_addr_hi', ctypes.c_uint),
  ('resp_offset', ctypes.c_uint),
  ('resp_buf_size', ctypes.c_uint),
  ('cmd', union_psp_gfx_commands),
  ('reserved_1', (ctypes.c_ubyte * 52)),
  ('resp', struct_psp_gfx_resp),
  ('reserved_2', (ctypes.c_ubyte * 64)),
]
class struct_psp_gfx_rb_frame(Struct): pass
struct_psp_gfx_rb_frame._fields_ = [
  ('cmd_buf_addr_lo', ctypes.c_uint),
  ('cmd_buf_addr_hi', ctypes.c_uint),
  ('cmd_buf_size', ctypes.c_uint),
  ('fence_addr_lo', ctypes.c_uint),
  ('fence_addr_hi', ctypes.c_uint),
  ('fence_value', ctypes.c_uint),
  ('sid_lo', ctypes.c_uint),
  ('sid_hi', ctypes.c_uint),
  ('vmid', ctypes.c_ubyte),
  ('frame_type', ctypes.c_ubyte),
  ('reserved1', (ctypes.c_ubyte * 2)),
  ('reserved2', (ctypes.c_uint * 7)),
]
enum_tee_error_code = CEnum(ctypes.c_uint)
TEE_SUCCESS = enum_tee_error_code.define('TEE_SUCCESS', 0)
TEE_ERROR_NOT_SUPPORTED = enum_tee_error_code.define('TEE_ERROR_NOT_SUPPORTED', 4294901770)

enum_psp_shared_mem_size = CEnum(ctypes.c_uint)
PSP_ASD_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_ASD_SHARED_MEM_SIZE', 0)
PSP_XGMI_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_XGMI_SHARED_MEM_SIZE', 16384)
PSP_RAS_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_RAS_SHARED_MEM_SIZE', 16384)
PSP_HDCP_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_HDCP_SHARED_MEM_SIZE', 16384)
PSP_DTM_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_DTM_SHARED_MEM_SIZE', 16384)
PSP_RAP_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_RAP_SHARED_MEM_SIZE', 16384)
PSP_SECUREDISPLAY_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_SECUREDISPLAY_SHARED_MEM_SIZE', 16384)

enum_ta_type_id = CEnum(ctypes.c_uint)
TA_TYPE_XGMI = enum_ta_type_id.define('TA_TYPE_XGMI', 1)
TA_TYPE_RAS = enum_ta_type_id.define('TA_TYPE_RAS', 2)
TA_TYPE_HDCP = enum_ta_type_id.define('TA_TYPE_HDCP', 3)
TA_TYPE_DTM = enum_ta_type_id.define('TA_TYPE_DTM', 4)
TA_TYPE_RAP = enum_ta_type_id.define('TA_TYPE_RAP', 5)
TA_TYPE_SECUREDISPLAY = enum_ta_type_id.define('TA_TYPE_SECUREDISPLAY', 6)
TA_TYPE_MAX_INDEX = enum_ta_type_id.define('TA_TYPE_MAX_INDEX', 7)

class struct_psp_context(Struct): pass
class struct_psp_xgmi_node_info(Struct): pass
class struct_psp_xgmi_topology_info(Struct): pass
class struct_psp_bin_desc(Struct): pass
enum_psp_bootloader_cmd = CEnum(ctypes.c_uint)
PSP_BL__LOAD_SYSDRV = enum_psp_bootloader_cmd.define('PSP_BL__LOAD_SYSDRV', 65536)
PSP_BL__LOAD_SOSDRV = enum_psp_bootloader_cmd.define('PSP_BL__LOAD_SOSDRV', 131072)
PSP_BL__LOAD_KEY_DATABASE = enum_psp_bootloader_cmd.define('PSP_BL__LOAD_KEY_DATABASE', 524288)
PSP_BL__LOAD_SOCDRV = enum_psp_bootloader_cmd.define('PSP_BL__LOAD_SOCDRV', 720896)
PSP_BL__LOAD_DBGDRV = enum_psp_bootloader_cmd.define('PSP_BL__LOAD_DBGDRV', 786432)
PSP_BL__LOAD_HADDRV = enum_psp_bootloader_cmd.define('PSP_BL__LOAD_HADDRV', 786432)
PSP_BL__LOAD_INTFDRV = enum_psp_bootloader_cmd.define('PSP_BL__LOAD_INTFDRV', 851968)
PSP_BL__LOAD_RASDRV = enum_psp_bootloader_cmd.define('PSP_BL__LOAD_RASDRV', 917504)
PSP_BL__LOAD_IPKEYMGRDRV = enum_psp_bootloader_cmd.define('PSP_BL__LOAD_IPKEYMGRDRV', 983040)
PSP_BL__DRAM_LONG_TRAIN = enum_psp_bootloader_cmd.define('PSP_BL__DRAM_LONG_TRAIN', 1048576)
PSP_BL__DRAM_SHORT_TRAIN = enum_psp_bootloader_cmd.define('PSP_BL__DRAM_SHORT_TRAIN', 2097152)
PSP_BL__LOAD_TOS_SPL_TABLE = enum_psp_bootloader_cmd.define('PSP_BL__LOAD_TOS_SPL_TABLE', 268435456)

enum_psp_ring_type = CEnum(ctypes.c_uint)
PSP_RING_TYPE__INVALID = enum_psp_ring_type.define('PSP_RING_TYPE__INVALID', 0)
PSP_RING_TYPE__UM = enum_psp_ring_type.define('PSP_RING_TYPE__UM', 1)
PSP_RING_TYPE__KM = enum_psp_ring_type.define('PSP_RING_TYPE__KM', 2)

enum_psp_reg_prog_id = CEnum(ctypes.c_uint)
PSP_REG_IH_RB_CNTL = enum_psp_reg_prog_id.define('PSP_REG_IH_RB_CNTL', 0)
PSP_REG_IH_RB_CNTL_RING1 = enum_psp_reg_prog_id.define('PSP_REG_IH_RB_CNTL_RING1', 1)
PSP_REG_IH_RB_CNTL_RING2 = enum_psp_reg_prog_id.define('PSP_REG_IH_RB_CNTL_RING2', 2)
PSP_REG_LAST = enum_psp_reg_prog_id.define('PSP_REG_LAST', 3)

enum_psp_memory_training_init_flag = CEnum(ctypes.c_uint)
PSP_MEM_TRAIN_NOT_SUPPORT = enum_psp_memory_training_init_flag.define('PSP_MEM_TRAIN_NOT_SUPPORT', 0)
PSP_MEM_TRAIN_SUPPORT = enum_psp_memory_training_init_flag.define('PSP_MEM_TRAIN_SUPPORT', 1)
PSP_MEM_TRAIN_INIT_FAILED = enum_psp_memory_training_init_flag.define('PSP_MEM_TRAIN_INIT_FAILED', 2)
PSP_MEM_TRAIN_RESERVE_SUCCESS = enum_psp_memory_training_init_flag.define('PSP_MEM_TRAIN_RESERVE_SUCCESS', 4)
PSP_MEM_TRAIN_INIT_SUCCESS = enum_psp_memory_training_init_flag.define('PSP_MEM_TRAIN_INIT_SUCCESS', 8)

enum_psp_memory_training_ops = CEnum(ctypes.c_uint)
PSP_MEM_TRAIN_SEND_LONG_MSG = enum_psp_memory_training_ops.define('PSP_MEM_TRAIN_SEND_LONG_MSG', 1)
PSP_MEM_TRAIN_SAVE = enum_psp_memory_training_ops.define('PSP_MEM_TRAIN_SAVE', 2)
PSP_MEM_TRAIN_RESTORE = enum_psp_memory_training_ops.define('PSP_MEM_TRAIN_RESTORE', 4)
PSP_MEM_TRAIN_SEND_SHORT_MSG = enum_psp_memory_training_ops.define('PSP_MEM_TRAIN_SEND_SHORT_MSG', 8)
PSP_MEM_TRAIN_COLD_BOOT = enum_psp_memory_training_ops.define('PSP_MEM_TRAIN_COLD_BOOT', 1)
PSP_MEM_TRAIN_RESUME = enum_psp_memory_training_ops.define('PSP_MEM_TRAIN_RESUME', 8)

enum_psp_runtime_entry_type = CEnum(ctypes.c_uint)
PSP_RUNTIME_ENTRY_TYPE_INVALID = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_INVALID', 0)
PSP_RUNTIME_ENTRY_TYPE_TEST = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_TEST', 1)
PSP_RUNTIME_ENTRY_TYPE_MGPU_COMMON = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_MGPU_COMMON', 2)
PSP_RUNTIME_ENTRY_TYPE_MGPU_WAFL = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_MGPU_WAFL', 3)
PSP_RUNTIME_ENTRY_TYPE_MGPU_XGMI = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_MGPU_XGMI', 4)
PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG', 5)
PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS', 6)

enum_psp_runtime_boot_cfg_feature = CEnum(ctypes.c_uint)
BOOT_CFG_FEATURE_GECC = enum_psp_runtime_boot_cfg_feature.define('BOOT_CFG_FEATURE_GECC', 1)
BOOT_CFG_FEATURE_TWO_STAGE_DRAM_TRAINING = enum_psp_runtime_boot_cfg_feature.define('BOOT_CFG_FEATURE_TWO_STAGE_DRAM_TRAINING', 2)

enum_psp_runtime_scpm_authentication = CEnum(ctypes.c_uint)
SCPM_DISABLE = enum_psp_runtime_scpm_authentication.define('SCPM_DISABLE', 0)
SCPM_ENABLE = enum_psp_runtime_scpm_authentication.define('SCPM_ENABLE', 1)
SCPM_ENABLE_WITH_SCPM_ERR = enum_psp_runtime_scpm_authentication.define('SCPM_ENABLE_WITH_SCPM_ERR', 2)

class struct_amdgpu_device(Struct): pass
enum_amdgpu_interrupt_state = CEnum(ctypes.c_uint)
AMDGPU_IRQ_STATE_DISABLE = enum_amdgpu_interrupt_state.define('AMDGPU_IRQ_STATE_DISABLE', 0)
AMDGPU_IRQ_STATE_ENABLE = enum_amdgpu_interrupt_state.define('AMDGPU_IRQ_STATE_ENABLE', 1)

class struct_amdgpu_iv_entry(Struct): pass
struct_amdgpu_iv_entry._fields_ = [
  ('client_id', ctypes.c_uint),
  ('src_id', ctypes.c_uint),
  ('ring_id', ctypes.c_uint),
  ('vmid', ctypes.c_uint),
  ('vmid_src', ctypes.c_uint),
  ('timestamp', ctypes.c_ulonglong),
  ('timestamp_src', ctypes.c_uint),
  ('pasid', ctypes.c_uint),
  ('node_id', ctypes.c_uint),
  ('src_data', (ctypes.c_uint * 4)),
  ('iv_entry', ctypes.POINTER(ctypes.c_uint)),
]
enum_interrupt_node_id_per_aid = CEnum(ctypes.c_uint)
AID0_NODEID = enum_interrupt_node_id_per_aid.define('AID0_NODEID', 0)
XCD0_NODEID = enum_interrupt_node_id_per_aid.define('XCD0_NODEID', 1)
XCD1_NODEID = enum_interrupt_node_id_per_aid.define('XCD1_NODEID', 2)
AID1_NODEID = enum_interrupt_node_id_per_aid.define('AID1_NODEID', 4)
XCD2_NODEID = enum_interrupt_node_id_per_aid.define('XCD2_NODEID', 5)
XCD3_NODEID = enum_interrupt_node_id_per_aid.define('XCD3_NODEID', 6)
AID2_NODEID = enum_interrupt_node_id_per_aid.define('AID2_NODEID', 8)
XCD4_NODEID = enum_interrupt_node_id_per_aid.define('XCD4_NODEID', 9)
XCD5_NODEID = enum_interrupt_node_id_per_aid.define('XCD5_NODEID', 10)
AID3_NODEID = enum_interrupt_node_id_per_aid.define('AID3_NODEID', 12)
XCD6_NODEID = enum_interrupt_node_id_per_aid.define('XCD6_NODEID', 13)
XCD7_NODEID = enum_interrupt_node_id_per_aid.define('XCD7_NODEID', 14)
NODEID_MAX = enum_interrupt_node_id_per_aid.define('NODEID_MAX', 15)

enum_AMDGPU_DOORBELL_ASSIGNMENT = CEnum(ctypes.c_uint)
AMDGPU_DOORBELL_KIQ = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_KIQ', 0)
AMDGPU_DOORBELL_HIQ = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_HIQ', 1)
AMDGPU_DOORBELL_DIQ = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_DIQ', 2)
AMDGPU_DOORBELL_MEC_RING0 = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_MEC_RING0', 16)
AMDGPU_DOORBELL_MEC_RING1 = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_MEC_RING1', 17)
AMDGPU_DOORBELL_MEC_RING2 = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_MEC_RING2', 18)
AMDGPU_DOORBELL_MEC_RING3 = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_MEC_RING3', 19)
AMDGPU_DOORBELL_MEC_RING4 = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_MEC_RING4', 20)
AMDGPU_DOORBELL_MEC_RING5 = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_MEC_RING5', 21)
AMDGPU_DOORBELL_MEC_RING6 = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_MEC_RING6', 22)
AMDGPU_DOORBELL_MEC_RING7 = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_MEC_RING7', 23)
AMDGPU_DOORBELL_GFX_RING0 = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_GFX_RING0', 32)
AMDGPU_DOORBELL_sDMA_ENGINE0 = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_sDMA_ENGINE0', 480)
AMDGPU_DOORBELL_sDMA_ENGINE1 = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_sDMA_ENGINE1', 481)
AMDGPU_DOORBELL_IH = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_IH', 488)
AMDGPU_DOORBELL_MAX_ASSIGNMENT = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_MAX_ASSIGNMENT', 1023)
AMDGPU_DOORBELL_INVALID = enum_AMDGPU_DOORBELL_ASSIGNMENT.define('AMDGPU_DOORBELL_INVALID', 65535)

enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT = CEnum(ctypes.c_uint)
AMDGPU_VEGA20_DOORBELL_KIQ = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_KIQ', 0)
AMDGPU_VEGA20_DOORBELL_HIQ = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_HIQ', 1)
AMDGPU_VEGA20_DOORBELL_DIQ = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_DIQ', 2)
AMDGPU_VEGA20_DOORBELL_MEC_RING0 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_MEC_RING0', 3)
AMDGPU_VEGA20_DOORBELL_MEC_RING1 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_MEC_RING1', 4)
AMDGPU_VEGA20_DOORBELL_MEC_RING2 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_MEC_RING2', 5)
AMDGPU_VEGA20_DOORBELL_MEC_RING3 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_MEC_RING3', 6)
AMDGPU_VEGA20_DOORBELL_MEC_RING4 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_MEC_RING4', 7)
AMDGPU_VEGA20_DOORBELL_MEC_RING5 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_MEC_RING5', 8)
AMDGPU_VEGA20_DOORBELL_MEC_RING6 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_MEC_RING6', 9)
AMDGPU_VEGA20_DOORBELL_MEC_RING7 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_MEC_RING7', 10)
AMDGPU_VEGA20_DOORBELL_USERQUEUE_START = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_USERQUEUE_START', 11)
AMDGPU_VEGA20_DOORBELL_USERQUEUE_END = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_USERQUEUE_END', 138)
AMDGPU_VEGA20_DOORBELL_GFX_RING0 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_GFX_RING0', 139)
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE0 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE0', 256)
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE1 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE1', 266)
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE2 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE2', 276)
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE3 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE3', 286)
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE4 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE4', 296)
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE5 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE5', 306)
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE6 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE6', 316)
AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE7 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_sDMA_ENGINE7', 326)
AMDGPU_VEGA20_DOORBELL_IH = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_IH', 376)
AMDGPU_VEGA20_DOORBELL64_VCN0_1 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_VCN0_1', 392)
AMDGPU_VEGA20_DOORBELL64_VCN2_3 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_VCN2_3', 393)
AMDGPU_VEGA20_DOORBELL64_VCN4_5 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_VCN4_5', 394)
AMDGPU_VEGA20_DOORBELL64_VCN6_7 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_VCN6_7', 395)
AMDGPU_VEGA20_DOORBELL64_VCN8_9 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_VCN8_9', 396)
AMDGPU_VEGA20_DOORBELL64_VCNa_b = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_VCNa_b', 397)
AMDGPU_VEGA20_DOORBELL64_VCNc_d = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_VCNc_d', 398)
AMDGPU_VEGA20_DOORBELL64_VCNe_f = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_VCNe_f', 399)
AMDGPU_VEGA20_DOORBELL64_UVD_RING0_1 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_UVD_RING0_1', 392)
AMDGPU_VEGA20_DOORBELL64_UVD_RING2_3 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_UVD_RING2_3', 393)
AMDGPU_VEGA20_DOORBELL64_UVD_RING4_5 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_UVD_RING4_5', 394)
AMDGPU_VEGA20_DOORBELL64_UVD_RING6_7 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_UVD_RING6_7', 395)
AMDGPU_VEGA20_DOORBELL64_VCE_RING0_1 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_VCE_RING0_1', 396)
AMDGPU_VEGA20_DOORBELL64_VCE_RING2_3 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_VCE_RING2_3', 397)
AMDGPU_VEGA20_DOORBELL64_VCE_RING4_5 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_VCE_RING4_5', 398)
AMDGPU_VEGA20_DOORBELL64_VCE_RING6_7 = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_VCE_RING6_7', 399)
AMDGPU_VEGA20_DOORBELL64_FIRST_NON_CP = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_FIRST_NON_CP', 256)
AMDGPU_VEGA20_DOORBELL64_LAST_NON_CP = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL64_LAST_NON_CP', 399)
AMDGPU_VEGA20_DOORBELL_XCC1_KIQ_START = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_XCC1_KIQ_START', 400)
AMDGPU_VEGA20_DOORBELL_XCC1_MEC_RING0_START = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_XCC1_MEC_RING0_START', 407)
AMDGPU_VEGA20_DOORBELL_AID1_sDMA_START = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_AID1_sDMA_START', 464)
AMDGPU_VEGA20_DOORBELL_MAX_ASSIGNMENT = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_MAX_ASSIGNMENT', 503)
AMDGPU_VEGA20_DOORBELL_INVALID = enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT.define('AMDGPU_VEGA20_DOORBELL_INVALID', 65535)

enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT = CEnum(ctypes.c_uint)
AMDGPU_NAVI10_DOORBELL_KIQ = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_KIQ', 0)
AMDGPU_NAVI10_DOORBELL_HIQ = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_HIQ', 1)
AMDGPU_NAVI10_DOORBELL_DIQ = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_DIQ', 2)
AMDGPU_NAVI10_DOORBELL_MEC_RING0 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_MEC_RING0', 3)
AMDGPU_NAVI10_DOORBELL_MEC_RING1 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_MEC_RING1', 4)
AMDGPU_NAVI10_DOORBELL_MEC_RING2 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_MEC_RING2', 5)
AMDGPU_NAVI10_DOORBELL_MEC_RING3 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_MEC_RING3', 6)
AMDGPU_NAVI10_DOORBELL_MEC_RING4 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_MEC_RING4', 7)
AMDGPU_NAVI10_DOORBELL_MEC_RING5 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_MEC_RING5', 8)
AMDGPU_NAVI10_DOORBELL_MEC_RING6 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_MEC_RING6', 9)
AMDGPU_NAVI10_DOORBELL_MEC_RING7 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_MEC_RING7', 10)
AMDGPU_NAVI10_DOORBELL_MES_RING0 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_MES_RING0', 11)
AMDGPU_NAVI10_DOORBELL_MES_RING1 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_MES_RING1', 12)
AMDGPU_NAVI10_DOORBELL_USERQUEUE_START = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_USERQUEUE_START', 13)
AMDGPU_NAVI10_DOORBELL_USERQUEUE_END = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_USERQUEUE_END', 138)
AMDGPU_NAVI10_DOORBELL_GFX_RING0 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_GFX_RING0', 139)
AMDGPU_NAVI10_DOORBELL_GFX_RING1 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_GFX_RING1', 140)
AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_START = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_START', 141)
AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_END = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_GFX_USERQUEUE_END', 255)
AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0', 256)
AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE1 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE1', 266)
AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE2 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE2', 276)
AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE3 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE3', 286)
AMDGPU_NAVI10_DOORBELL_IH = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_IH', 376)
AMDGPU_NAVI10_DOORBELL64_VCN0_1 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL64_VCN0_1', 392)
AMDGPU_NAVI10_DOORBELL64_VCN2_3 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL64_VCN2_3', 393)
AMDGPU_NAVI10_DOORBELL64_VCN4_5 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL64_VCN4_5', 394)
AMDGPU_NAVI10_DOORBELL64_VCN6_7 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL64_VCN6_7', 395)
AMDGPU_NAVI10_DOORBELL64_VCN8_9 = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL64_VCN8_9', 396)
AMDGPU_NAVI10_DOORBELL64_VCNa_b = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL64_VCNa_b', 397)
AMDGPU_NAVI10_DOORBELL64_VCNc_d = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL64_VCNc_d', 398)
AMDGPU_NAVI10_DOORBELL64_VCNe_f = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL64_VCNe_f', 399)
AMDGPU_NAVI10_DOORBELL64_VPE = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL64_VPE', 400)
AMDGPU_NAVI10_DOORBELL64_FIRST_NON_CP = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL64_FIRST_NON_CP', 256)
AMDGPU_NAVI10_DOORBELL64_LAST_NON_CP = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL64_LAST_NON_CP', 400)
AMDGPU_NAVI10_DOORBELL_MAX_ASSIGNMENT = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_MAX_ASSIGNMENT', 400)
AMDGPU_NAVI10_DOORBELL_INVALID = enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT.define('AMDGPU_NAVI10_DOORBELL_INVALID', 65535)

enum_AMDGPU_DOORBELL64_ASSIGNMENT = CEnum(ctypes.c_uint)
AMDGPU_DOORBELL64_KIQ = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_KIQ', 0)
AMDGPU_DOORBELL64_HIQ = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_HIQ', 1)
AMDGPU_DOORBELL64_DIQ = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_DIQ', 2)
AMDGPU_DOORBELL64_MEC_RING0 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_MEC_RING0', 3)
AMDGPU_DOORBELL64_MEC_RING1 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_MEC_RING1', 4)
AMDGPU_DOORBELL64_MEC_RING2 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_MEC_RING2', 5)
AMDGPU_DOORBELL64_MEC_RING3 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_MEC_RING3', 6)
AMDGPU_DOORBELL64_MEC_RING4 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_MEC_RING4', 7)
AMDGPU_DOORBELL64_MEC_RING5 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_MEC_RING5', 8)
AMDGPU_DOORBELL64_MEC_RING6 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_MEC_RING6', 9)
AMDGPU_DOORBELL64_MEC_RING7 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_MEC_RING7', 10)
AMDGPU_DOORBELL64_USERQUEUE_START = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_USERQUEUE_START', 11)
AMDGPU_DOORBELL64_USERQUEUE_END = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_USERQUEUE_END', 138)
AMDGPU_DOORBELL64_GFX_RING0 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_GFX_RING0', 139)
AMDGPU_DOORBELL64_sDMA_ENGINE0 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_sDMA_ENGINE0', 240)
AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE0 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE0', 241)
AMDGPU_DOORBELL64_sDMA_ENGINE1 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_sDMA_ENGINE1', 242)
AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE1 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_sDMA_HI_PRI_ENGINE1', 243)
AMDGPU_DOORBELL64_IH = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_IH', 244)
AMDGPU_DOORBELL64_IH_RING1 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_IH_RING1', 245)
AMDGPU_DOORBELL64_IH_RING2 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_IH_RING2', 246)
AMDGPU_DOORBELL64_VCN0_1 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_VCN0_1', 248)
AMDGPU_DOORBELL64_VCN2_3 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_VCN2_3', 249)
AMDGPU_DOORBELL64_VCN4_5 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_VCN4_5', 250)
AMDGPU_DOORBELL64_VCN6_7 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_VCN6_7', 251)
AMDGPU_DOORBELL64_UVD_RING0_1 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_UVD_RING0_1', 248)
AMDGPU_DOORBELL64_UVD_RING2_3 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_UVD_RING2_3', 249)
AMDGPU_DOORBELL64_UVD_RING4_5 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_UVD_RING4_5', 250)
AMDGPU_DOORBELL64_UVD_RING6_7 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_UVD_RING6_7', 251)
AMDGPU_DOORBELL64_VCE_RING0_1 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_VCE_RING0_1', 252)
AMDGPU_DOORBELL64_VCE_RING2_3 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_VCE_RING2_3', 253)
AMDGPU_DOORBELL64_VCE_RING4_5 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_VCE_RING4_5', 254)
AMDGPU_DOORBELL64_VCE_RING6_7 = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_VCE_RING6_7', 255)
AMDGPU_DOORBELL64_FIRST_NON_CP = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_FIRST_NON_CP', 240)
AMDGPU_DOORBELL64_LAST_NON_CP = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_LAST_NON_CP', 255)
AMDGPU_DOORBELL64_MAX_ASSIGNMENT = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_MAX_ASSIGNMENT', 255)
AMDGPU_DOORBELL64_INVALID = enum_AMDGPU_DOORBELL64_ASSIGNMENT.define('AMDGPU_DOORBELL64_INVALID', 65535)

enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1 = CEnum(ctypes.c_uint)
AMDGPU_DOORBELL_LAYOUT1_KIQ_START = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_KIQ_START', 0)
AMDGPU_DOORBELL_LAYOUT1_HIQ = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_HIQ', 1)
AMDGPU_DOORBELL_LAYOUT1_DIQ = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_DIQ', 2)
AMDGPU_DOORBELL_LAYOUT1_MEC_RING_START = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_MEC_RING_START', 8)
AMDGPU_DOORBELL_LAYOUT1_MEC_RING_END = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_MEC_RING_END', 15)
AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_START = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_START', 16)
AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_END = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_USERQUEUE_END', 31)
AMDGPU_DOORBELL_LAYOUT1_XCC_RANGE = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_XCC_RANGE', 32)
AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_START = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_START', 256)
AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_END = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_sDMA_ENGINE_END', 415)
AMDGPU_DOORBELL_LAYOUT1_IH = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_IH', 416)
AMDGPU_DOORBELL_LAYOUT1_VCN_START = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_VCN_START', 432)
AMDGPU_DOORBELL_LAYOUT1_VCN_END = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_VCN_END', 488)
AMDGPU_DOORBELL_LAYOUT1_FIRST_NON_CP = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_FIRST_NON_CP', 256)
AMDGPU_DOORBELL_LAYOUT1_LAST_NON_CP = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_LAST_NON_CP', 488)
AMDGPU_DOORBELL_LAYOUT1_MAX_ASSIGNMENT = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_MAX_ASSIGNMENT', 488)
AMDGPU_DOORBELL_LAYOUT1_INVALID = enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1.define('AMDGPU_DOORBELL_LAYOUT1_INVALID', 65535)

enum_soc15_ih_clientid = CEnum(ctypes.c_uint)
SOC15_IH_CLIENTID_IH = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_IH', 0)
SOC15_IH_CLIENTID_ACP = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_ACP', 1)
SOC15_IH_CLIENTID_ATHUB = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_ATHUB', 2)
SOC15_IH_CLIENTID_BIF = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_BIF', 3)
SOC15_IH_CLIENTID_DCE = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_DCE', 4)
SOC15_IH_CLIENTID_ISP = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_ISP', 5)
SOC15_IH_CLIENTID_PCIE0 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_PCIE0', 6)
SOC15_IH_CLIENTID_RLC = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_RLC', 7)
SOC15_IH_CLIENTID_SDMA0 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SDMA0', 8)
SOC15_IH_CLIENTID_SDMA1 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SDMA1', 9)
SOC15_IH_CLIENTID_SE0SH = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SE0SH', 10)
SOC15_IH_CLIENTID_SE1SH = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SE1SH', 11)
SOC15_IH_CLIENTID_SE2SH = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SE2SH', 12)
SOC15_IH_CLIENTID_SE3SH = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SE3SH', 13)
SOC15_IH_CLIENTID_UVD1 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_UVD1', 14)
SOC15_IH_CLIENTID_THM = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_THM', 15)
SOC15_IH_CLIENTID_UVD = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_UVD', 16)
SOC15_IH_CLIENTID_VCE0 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_VCE0', 17)
SOC15_IH_CLIENTID_VMC = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_VMC', 18)
SOC15_IH_CLIENTID_XDMA = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_XDMA', 19)
SOC15_IH_CLIENTID_GRBM_CP = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_GRBM_CP', 20)
SOC15_IH_CLIENTID_ATS = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_ATS', 21)
SOC15_IH_CLIENTID_ROM_SMUIO = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_ROM_SMUIO', 22)
SOC15_IH_CLIENTID_DF = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_DF', 23)
SOC15_IH_CLIENTID_VCE1 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_VCE1', 24)
SOC15_IH_CLIENTID_PWR = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_PWR', 25)
SOC15_IH_CLIENTID_RESERVED = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_RESERVED', 26)
SOC15_IH_CLIENTID_UTCL2 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_UTCL2', 27)
SOC15_IH_CLIENTID_EA = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_EA', 28)
SOC15_IH_CLIENTID_UTCL2LOG = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_UTCL2LOG', 29)
SOC15_IH_CLIENTID_MP0 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_MP0', 30)
SOC15_IH_CLIENTID_MP1 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_MP1', 31)
SOC15_IH_CLIENTID_MAX = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_MAX', 32)
SOC15_IH_CLIENTID_VCN = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_VCN', 16)
SOC15_IH_CLIENTID_VCN1 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_VCN1', 14)
SOC15_IH_CLIENTID_SDMA2 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SDMA2', 1)
SOC15_IH_CLIENTID_SDMA3 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SDMA3', 4)
SOC15_IH_CLIENTID_SDMA3_Sienna_Cichlid = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SDMA3_Sienna_Cichlid', 5)
SOC15_IH_CLIENTID_SDMA4 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SDMA4', 5)
SOC15_IH_CLIENTID_SDMA5 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SDMA5', 17)
SOC15_IH_CLIENTID_SDMA6 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SDMA6', 19)
SOC15_IH_CLIENTID_SDMA7 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_SDMA7', 24)
SOC15_IH_CLIENTID_VMC1 = enum_soc15_ih_clientid.define('SOC15_IH_CLIENTID_VMC1', 6)

enum_soc21_ih_clientid = CEnum(ctypes.c_uint)
SOC21_IH_CLIENTID_IH = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_IH', 0)
SOC21_IH_CLIENTID_ATHUB = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_ATHUB', 2)
SOC21_IH_CLIENTID_BIF = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_BIF', 3)
SOC21_IH_CLIENTID_DCN = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_DCN', 4)
SOC21_IH_CLIENTID_ISP = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_ISP', 5)
SOC21_IH_CLIENTID_MP3 = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_MP3', 6)
SOC21_IH_CLIENTID_RLC = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_RLC', 7)
SOC21_IH_CLIENTID_GFX = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_GFX', 10)
SOC21_IH_CLIENTID_IMU = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_IMU', 11)
SOC21_IH_CLIENTID_VCN1 = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_VCN1', 14)
SOC21_IH_CLIENTID_THM = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_THM', 15)
SOC21_IH_CLIENTID_VCN = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_VCN', 16)
SOC21_IH_CLIENTID_VPE1 = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_VPE1', 17)
SOC21_IH_CLIENTID_VMC = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_VMC', 18)
SOC21_IH_CLIENTID_GRBM_CP = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_GRBM_CP', 20)
SOC21_IH_CLIENTID_ROM_SMUIO = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_ROM_SMUIO', 22)
SOC21_IH_CLIENTID_DF = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_DF', 23)
SOC21_IH_CLIENTID_VPE = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_VPE', 24)
SOC21_IH_CLIENTID_PWR = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_PWR', 25)
SOC21_IH_CLIENTID_LSDMA = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_LSDMA', 26)
SOC21_IH_CLIENTID_MP0 = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_MP0', 30)
SOC21_IH_CLIENTID_MP1 = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_MP1', 31)
SOC21_IH_CLIENTID_MAX = enum_soc21_ih_clientid.define('SOC21_IH_CLIENTID_MAX', 32)

AMDGPU_VM_MAX_UPDATE_SIZE = 0x3FFFF
AMDGPU_PTE_VALID = (1 << 0)
AMDGPU_PTE_SYSTEM = (1 << 1)
AMDGPU_PTE_SNOOPED = (1 << 2)
AMDGPU_PTE_TMZ = (1 << 3)
AMDGPU_PTE_EXECUTABLE = (1 << 4)
AMDGPU_PTE_READABLE = (1 << 5)
AMDGPU_PTE_WRITEABLE = (1 << 6)
AMDGPU_PTE_FRAG = lambda x: ((x & 0x1f) << 7)
AMDGPU_PTE_PRT = (1 << 51)
AMDGPU_PDE_PTE = (1 << 54)
AMDGPU_PTE_LOG = (1 << 55)
AMDGPU_PTE_TF = (1 << 56)
AMDGPU_PTE_NOALLOC = (1 << 58)
AMDGPU_PDE_BFS = lambda a: (a << 59)
AMDGPU_VM_NORETRY_FLAGS = (AMDGPU_PTE_EXECUTABLE | AMDGPU_PDE_PTE | AMDGPU_PTE_TF)
AMDGPU_VM_NORETRY_FLAGS_TF = (AMDGPU_PTE_VALID | AMDGPU_PTE_SYSTEM | AMDGPU_PTE_PRT)
AMDGPU_PTE_MTYPE_VG10_SHIFT = lambda mtype: ((mtype) << 57)
AMDGPU_PTE_MTYPE_VG10_MASK = AMDGPU_PTE_MTYPE_VG10_SHIFT(3)
AMDGPU_PTE_MTYPE_VG10 = lambda flags,mtype: (((flags) & (~AMDGPU_PTE_MTYPE_VG10_MASK)) | AMDGPU_PTE_MTYPE_VG10_SHIFT(mtype))
AMDGPU_MTYPE_NC = 0
AMDGPU_MTYPE_CC = 2
AMDGPU_PTE_MTYPE_NV10_SHIFT = lambda mtype: ((mtype) << 48)
AMDGPU_PTE_MTYPE_NV10_MASK = AMDGPU_PTE_MTYPE_NV10_SHIFT(7)
AMDGPU_PTE_MTYPE_NV10 = lambda flags,mtype: (((flags) & (~AMDGPU_PTE_MTYPE_NV10_MASK)) | AMDGPU_PTE_MTYPE_NV10_SHIFT(mtype))
AMDGPU_PTE_PRT_GFX12 = (1 << 56)
AMDGPU_PTE_MTYPE_GFX12_SHIFT = lambda mtype: ((mtype) << 54)
AMDGPU_PTE_MTYPE_GFX12_MASK = AMDGPU_PTE_MTYPE_GFX12_SHIFT(3)
AMDGPU_PTE_MTYPE_GFX12 = lambda flags,mtype: (((flags) & (~AMDGPU_PTE_MTYPE_GFX12_MASK)) | AMDGPU_PTE_MTYPE_GFX12_SHIFT(mtype))
AMDGPU_PTE_IS_PTE = (1 << 63)
AMDGPU_PDE_BFS_GFX12 = lambda a: (((a) & 0x1f) << 58)
AMDGPU_PDE_PTE_GFX12 = (1 << 63)
AMDGPU_VM_FAULT_STOP_NEVER = 0
AMDGPU_VM_FAULT_STOP_FIRST = 1
AMDGPU_VM_FAULT_STOP_ALWAYS = 2
AMDGPU_VM_RESERVED_VRAM = (8 << 20)
AMDGPU_MAX_VMHUBS = 13
AMDGPU_GFXHUB_START = 0
AMDGPU_MMHUB0_START = 8
AMDGPU_MMHUB1_START = 12
AMDGPU_GFXHUB = lambda x: (AMDGPU_GFXHUB_START + (x))
AMDGPU_MMHUB0 = lambda x: (AMDGPU_MMHUB0_START + (x))
AMDGPU_MMHUB1 = lambda x: (AMDGPU_MMHUB1_START + (x))
AMDGPU_IS_GFXHUB = lambda x: ((x) >= AMDGPU_GFXHUB_START and (x) < AMDGPU_MMHUB0_START)
AMDGPU_IS_MMHUB0 = lambda x: ((x) >= AMDGPU_MMHUB0_START and (x) < AMDGPU_MMHUB1_START)
AMDGPU_IS_MMHUB1 = lambda x: ((x) >= AMDGPU_MMHUB1_START and (x) < AMDGPU_MAX_VMHUBS)
AMDGPU_VA_RESERVED_CSA_SIZE = (2 << 20)
AMDGPU_VA_RESERVED_SEQ64_SIZE = (2 << 20)
AMDGPU_VA_RESERVED_SEQ64_START = lambda adev: (AMDGPU_VA_RESERVED_CSA_START(adev) - AMDGPU_VA_RESERVED_SEQ64_SIZE)
AMDGPU_VA_RESERVED_TRAP_SIZE = (2 << 12)
AMDGPU_VA_RESERVED_TRAP_START = lambda adev: (AMDGPU_VA_RESERVED_SEQ64_START(adev) - AMDGPU_VA_RESERVED_TRAP_SIZE)
AMDGPU_VA_RESERVED_BOTTOM = (1 << 16)
AMDGPU_VA_RESERVED_TOP = (AMDGPU_VA_RESERVED_TRAP_SIZE + AMDGPU_VA_RESERVED_SEQ64_SIZE + AMDGPU_VA_RESERVED_CSA_SIZE)
AMDGPU_VM_USE_CPU_FOR_GFX = (1 << 0)
AMDGPU_VM_USE_CPU_FOR_COMPUTE = (1 << 1)
PSP_HEADER_SIZE = 256
BINARY_SIGNATURE = 0x28211407
DISCOVERY_TABLE_SIGNATURE = 0x53445049
GC_TABLE_ID = 0x4347
HARVEST_TABLE_SIGNATURE = 0x56524148
VCN_INFO_TABLE_ID = 0x004E4356
MALL_INFO_TABLE_ID = 0x4C4C414D
NPS_INFO_TABLE_ID = 0x0053504E
VCN_INFO_TABLE_MAX_NUM_INSTANCES = 4
NPS_INFO_TABLE_MAX_NUM_INSTANCES = 12
HWIP_MAX_INSTANCE = 44
HW_ID_MAX = 300
MP1_HWID = 1
MP2_HWID = 2
THM_HWID = 3
SMUIO_HWID = 4
FUSE_HWID = 5
CLKA_HWID = 6
PWR_HWID = 10
GC_HWID = 11
UVD_HWID = 12
VCN_HWID = UVD_HWID
AUDIO_AZ_HWID = 13
ACP_HWID = 14
DCI_HWID = 15
DMU_HWID = 271
DCO_HWID = 16
DIO_HWID = 272
XDMA_HWID = 17
DCEAZ_HWID = 18
DAZ_HWID = 274
SDPMUX_HWID = 19
NTB_HWID = 20
VPE_HWID = 21
IOHC_HWID = 24
L2IMU_HWID = 28
VCE_HWID = 32
MMHUB_HWID = 34
ATHUB_HWID = 35
DBGU_NBIO_HWID = 36
DFX_HWID = 37
DBGU0_HWID = 38
DBGU1_HWID = 39
OSSSYS_HWID = 40
HDP_HWID = 41
SDMA0_HWID = 42
SDMA1_HWID = 43
ISP_HWID = 44
DBGU_IO_HWID = 45
DF_HWID = 46
CLKB_HWID = 47
FCH_HWID = 48
DFX_DAP_HWID = 49
L1IMU_PCIE_HWID = 50
L1IMU_NBIF_HWID = 51
L1IMU_IOAGR_HWID = 52
L1IMU3_HWID = 53
L1IMU4_HWID = 54
L1IMU5_HWID = 55
L1IMU6_HWID = 56
L1IMU7_HWID = 57
L1IMU8_HWID = 58
L1IMU9_HWID = 59
L1IMU10_HWID = 60
L1IMU11_HWID = 61
L1IMU12_HWID = 62
L1IMU13_HWID = 63
L1IMU14_HWID = 64
L1IMU15_HWID = 65
WAFLC_HWID = 66
FCH_USB_PD_HWID = 67
SDMA2_HWID = 68
SDMA3_HWID = 69
PCIE_HWID = 70
PCS_HWID = 80
DDCL_HWID = 89
SST_HWID = 90
LSDMA_HWID = 91
IOAGR_HWID = 100
NBIF_HWID = 108
IOAPIC_HWID = 124
SYSTEMHUB_HWID = 128
NTBCCP_HWID = 144
UMC_HWID = 150
SATA_HWID = 168
USB_HWID = 170
CCXSEC_HWID = 176
XGMI_HWID = 200
XGBE_HWID = 216
MP0_HWID = 255
hw_id_map = {GC_HWIP:GC_HWID,HDP_HWIP:HDP_HWID,SDMA0_HWIP:SDMA0_HWID,SDMA1_HWIP:SDMA1_HWID,SDMA2_HWIP:SDMA2_HWID,SDMA3_HWIP:SDMA3_HWID,LSDMA_HWIP:LSDMA_HWID,MMHUB_HWIP:MMHUB_HWID,ATHUB_HWIP:ATHUB_HWID,NBIO_HWIP:NBIF_HWID,MP0_HWIP:MP0_HWID,MP1_HWIP:MP1_HWID,UVD_HWIP:UVD_HWID,VCE_HWIP:VCE_HWID,DF_HWIP:DF_HWID,DCE_HWIP:DMU_HWID,OSSSYS_HWIP:OSSSYS_HWID,SMUIO_HWIP:SMUIO_HWID,PWR_HWIP:PWR_HWID,NBIF_HWIP:NBIF_HWID,THM_HWIP:THM_HWID,CLK_HWIP:CLKA_HWID,UMC_HWIP:UMC_HWID,XGMI_HWIP:XGMI_HWID,DCI_HWIP:DCI_HWID,PCIE_HWIP:PCIE_HWID,VPE_HWIP:VPE_HWID,ISP_HWIP:ISP_HWID}
int32_t = int
AMDGPU_SDMA0_UCODE_LOADED = 0x00000001
AMDGPU_SDMA1_UCODE_LOADED = 0x00000002
AMDGPU_CPCE_UCODE_LOADED = 0x00000004
AMDGPU_CPPFP_UCODE_LOADED = 0x00000008
AMDGPU_CPME_UCODE_LOADED = 0x00000010
AMDGPU_CPMEC1_UCODE_LOADED = 0x00000020
AMDGPU_CPMEC2_UCODE_LOADED = 0x00000040
AMDGPU_CPRLC_UCODE_LOADED = 0x00000100
PSP_GFX_CMD_BUF_VERSION = 0x00000001
GFX_CMD_STATUS_MASK = 0x0000FFFF
GFX_CMD_ID_MASK = 0x000F0000
GFX_CMD_RESERVED_MASK = 0x7FF00000
GFX_CMD_RESPONSE_MASK = 0x80000000
C2PMSG_CMD_GFX_USB_PD_FW_VER = 0x2000000
GFX_FLAG_RESPONSE = 0x80000000
GFX_BUF_MAX_DESC = 64
FRAME_TYPE_DESTROY = 1
PSP_ERR_UNKNOWN_COMMAND = 0x00000100
PSP_FENCE_BUFFER_SIZE = 0x1000
PSP_CMD_BUFFER_SIZE = 0x1000
PSP_1_MEG = 0x100000
PSP_TMR_ALIGNMENT = 0x100000
PSP_FW_NAME_LEN = 0x24
AMDGPU_XGMI_MAX_CONNECTED_NODES = 64
MEM_TRAIN_SYSTEM_SIGNATURE = 0x54534942
GDDR6_MEM_TRAINING_DATA_SIZE_IN_BYTES = 0x1000
GDDR6_MEM_TRAINING_OFFSET = 0x8000
BIST_MEM_TRAINING_ENCROACHED_SIZE = 0x2000000
PSP_RUNTIME_DB_SIZE_IN_BYTES = 0x10000
PSP_RUNTIME_DB_OFFSET = 0x100000
PSP_RUNTIME_DB_COOKIE_ID = 0x0ed5
PSP_RUNTIME_DB_VER_1 = 0x0100
PSP_RUNTIME_DB_DIAG_ENTRY_MAX_COUNT = 0x40
int32_t = int
AMDGPU_MAX_IRQ_SRC_ID = 0x100
AMDGPU_MAX_IRQ_CLIENT_ID = 0x100
AMDGPU_IRQ_CLIENTID_LEGACY = 0
AMDGPU_IRQ_CLIENTID_MAX = SOC15_IH_CLIENTID_MAX
AMDGPU_IRQ_SRC_DATA_MAX_SIZE_DW = 4
