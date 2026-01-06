# mypy: ignore-errors
from __future__ import annotations
import ctypes
from typing import Annotated
from tinygrad.runtime.support.c import DLL, record, CEnum, _IO, _IOW, _IOR, _IOWR, init_records
@record
class struct_v11_gfx_mqd:
  SIZE = 2048
  shadow_base_lo: Annotated[ctypes.c_uint32, 0]
  shadow_base_hi: Annotated[ctypes.c_uint32, 4]
  gds_bkup_base_lo: Annotated[ctypes.c_uint32, 8]
  gds_bkup_base_hi: Annotated[ctypes.c_uint32, 12]
  fw_work_area_base_lo: Annotated[ctypes.c_uint32, 16]
  fw_work_area_base_hi: Annotated[ctypes.c_uint32, 20]
  shadow_initialized: Annotated[ctypes.c_uint32, 24]
  ib_vmid: Annotated[ctypes.c_uint32, 28]
  reserved_8: Annotated[ctypes.c_uint32, 32]
  reserved_9: Annotated[ctypes.c_uint32, 36]
  reserved_10: Annotated[ctypes.c_uint32, 40]
  reserved_11: Annotated[ctypes.c_uint32, 44]
  reserved_12: Annotated[ctypes.c_uint32, 48]
  reserved_13: Annotated[ctypes.c_uint32, 52]
  reserved_14: Annotated[ctypes.c_uint32, 56]
  reserved_15: Annotated[ctypes.c_uint32, 60]
  reserved_16: Annotated[ctypes.c_uint32, 64]
  reserved_17: Annotated[ctypes.c_uint32, 68]
  reserved_18: Annotated[ctypes.c_uint32, 72]
  reserved_19: Annotated[ctypes.c_uint32, 76]
  reserved_20: Annotated[ctypes.c_uint32, 80]
  reserved_21: Annotated[ctypes.c_uint32, 84]
  reserved_22: Annotated[ctypes.c_uint32, 88]
  reserved_23: Annotated[ctypes.c_uint32, 92]
  reserved_24: Annotated[ctypes.c_uint32, 96]
  reserved_25: Annotated[ctypes.c_uint32, 100]
  reserved_26: Annotated[ctypes.c_uint32, 104]
  reserved_27: Annotated[ctypes.c_uint32, 108]
  reserved_28: Annotated[ctypes.c_uint32, 112]
  reserved_29: Annotated[ctypes.c_uint32, 116]
  reserved_30: Annotated[ctypes.c_uint32, 120]
  reserved_31: Annotated[ctypes.c_uint32, 124]
  reserved_32: Annotated[ctypes.c_uint32, 128]
  reserved_33: Annotated[ctypes.c_uint32, 132]
  reserved_34: Annotated[ctypes.c_uint32, 136]
  reserved_35: Annotated[ctypes.c_uint32, 140]
  reserved_36: Annotated[ctypes.c_uint32, 144]
  reserved_37: Annotated[ctypes.c_uint32, 148]
  reserved_38: Annotated[ctypes.c_uint32, 152]
  reserved_39: Annotated[ctypes.c_uint32, 156]
  reserved_40: Annotated[ctypes.c_uint32, 160]
  reserved_41: Annotated[ctypes.c_uint32, 164]
  reserved_42: Annotated[ctypes.c_uint32, 168]
  reserved_43: Annotated[ctypes.c_uint32, 172]
  reserved_44: Annotated[ctypes.c_uint32, 176]
  reserved_45: Annotated[ctypes.c_uint32, 180]
  reserved_46: Annotated[ctypes.c_uint32, 184]
  reserved_47: Annotated[ctypes.c_uint32, 188]
  reserved_48: Annotated[ctypes.c_uint32, 192]
  reserved_49: Annotated[ctypes.c_uint32, 196]
  reserved_50: Annotated[ctypes.c_uint32, 200]
  reserved_51: Annotated[ctypes.c_uint32, 204]
  reserved_52: Annotated[ctypes.c_uint32, 208]
  reserved_53: Annotated[ctypes.c_uint32, 212]
  reserved_54: Annotated[ctypes.c_uint32, 216]
  reserved_55: Annotated[ctypes.c_uint32, 220]
  reserved_56: Annotated[ctypes.c_uint32, 224]
  reserved_57: Annotated[ctypes.c_uint32, 228]
  reserved_58: Annotated[ctypes.c_uint32, 232]
  reserved_59: Annotated[ctypes.c_uint32, 236]
  reserved_60: Annotated[ctypes.c_uint32, 240]
  reserved_61: Annotated[ctypes.c_uint32, 244]
  reserved_62: Annotated[ctypes.c_uint32, 248]
  reserved_63: Annotated[ctypes.c_uint32, 252]
  reserved_64: Annotated[ctypes.c_uint32, 256]
  reserved_65: Annotated[ctypes.c_uint32, 260]
  reserved_66: Annotated[ctypes.c_uint32, 264]
  reserved_67: Annotated[ctypes.c_uint32, 268]
  reserved_68: Annotated[ctypes.c_uint32, 272]
  reserved_69: Annotated[ctypes.c_uint32, 276]
  reserved_70: Annotated[ctypes.c_uint32, 280]
  reserved_71: Annotated[ctypes.c_uint32, 284]
  reserved_72: Annotated[ctypes.c_uint32, 288]
  reserved_73: Annotated[ctypes.c_uint32, 292]
  reserved_74: Annotated[ctypes.c_uint32, 296]
  reserved_75: Annotated[ctypes.c_uint32, 300]
  reserved_76: Annotated[ctypes.c_uint32, 304]
  reserved_77: Annotated[ctypes.c_uint32, 308]
  reserved_78: Annotated[ctypes.c_uint32, 312]
  reserved_79: Annotated[ctypes.c_uint32, 316]
  reserved_80: Annotated[ctypes.c_uint32, 320]
  reserved_81: Annotated[ctypes.c_uint32, 324]
  reserved_82: Annotated[ctypes.c_uint32, 328]
  reserved_83: Annotated[ctypes.c_uint32, 332]
  checksum_lo: Annotated[ctypes.c_uint32, 336]
  checksum_hi: Annotated[ctypes.c_uint32, 340]
  cp_mqd_query_time_lo: Annotated[ctypes.c_uint32, 344]
  cp_mqd_query_time_hi: Annotated[ctypes.c_uint32, 348]
  reserved_88: Annotated[ctypes.c_uint32, 352]
  reserved_89: Annotated[ctypes.c_uint32, 356]
  reserved_90: Annotated[ctypes.c_uint32, 360]
  reserved_91: Annotated[ctypes.c_uint32, 364]
  cp_mqd_query_wave_count: Annotated[ctypes.c_uint32, 368]
  cp_mqd_query_gfx_hqd_rptr: Annotated[ctypes.c_uint32, 372]
  cp_mqd_query_gfx_hqd_wptr: Annotated[ctypes.c_uint32, 376]
  cp_mqd_query_gfx_hqd_offset: Annotated[ctypes.c_uint32, 380]
  reserved_96: Annotated[ctypes.c_uint32, 384]
  reserved_97: Annotated[ctypes.c_uint32, 388]
  reserved_98: Annotated[ctypes.c_uint32, 392]
  reserved_99: Annotated[ctypes.c_uint32, 396]
  reserved_100: Annotated[ctypes.c_uint32, 400]
  reserved_101: Annotated[ctypes.c_uint32, 404]
  reserved_102: Annotated[ctypes.c_uint32, 408]
  reserved_103: Annotated[ctypes.c_uint32, 412]
  control_buf_addr_lo: Annotated[ctypes.c_uint32, 416]
  control_buf_addr_hi: Annotated[ctypes.c_uint32, 420]
  disable_queue: Annotated[ctypes.c_uint32, 424]
  reserved_107: Annotated[ctypes.c_uint32, 428]
  reserved_108: Annotated[ctypes.c_uint32, 432]
  reserved_109: Annotated[ctypes.c_uint32, 436]
  reserved_110: Annotated[ctypes.c_uint32, 440]
  reserved_111: Annotated[ctypes.c_uint32, 444]
  reserved_112: Annotated[ctypes.c_uint32, 448]
  reserved_113: Annotated[ctypes.c_uint32, 452]
  reserved_114: Annotated[ctypes.c_uint32, 456]
  reserved_115: Annotated[ctypes.c_uint32, 460]
  reserved_116: Annotated[ctypes.c_uint32, 464]
  reserved_117: Annotated[ctypes.c_uint32, 468]
  reserved_118: Annotated[ctypes.c_uint32, 472]
  reserved_119: Annotated[ctypes.c_uint32, 476]
  reserved_120: Annotated[ctypes.c_uint32, 480]
  reserved_121: Annotated[ctypes.c_uint32, 484]
  reserved_122: Annotated[ctypes.c_uint32, 488]
  reserved_123: Annotated[ctypes.c_uint32, 492]
  reserved_124: Annotated[ctypes.c_uint32, 496]
  reserved_125: Annotated[ctypes.c_uint32, 500]
  reserved_126: Annotated[ctypes.c_uint32, 504]
  reserved_127: Annotated[ctypes.c_uint32, 508]
  cp_mqd_base_addr: Annotated[ctypes.c_uint32, 512]
  cp_mqd_base_addr_hi: Annotated[ctypes.c_uint32, 516]
  cp_gfx_hqd_active: Annotated[ctypes.c_uint32, 520]
  cp_gfx_hqd_vmid: Annotated[ctypes.c_uint32, 524]
  reserved_131: Annotated[ctypes.c_uint32, 528]
  reserved_132: Annotated[ctypes.c_uint32, 532]
  cp_gfx_hqd_queue_priority: Annotated[ctypes.c_uint32, 536]
  cp_gfx_hqd_quantum: Annotated[ctypes.c_uint32, 540]
  cp_gfx_hqd_base: Annotated[ctypes.c_uint32, 544]
  cp_gfx_hqd_base_hi: Annotated[ctypes.c_uint32, 548]
  cp_gfx_hqd_rptr: Annotated[ctypes.c_uint32, 552]
  cp_gfx_hqd_rptr_addr: Annotated[ctypes.c_uint32, 556]
  cp_gfx_hqd_rptr_addr_hi: Annotated[ctypes.c_uint32, 560]
  cp_rb_wptr_poll_addr_lo: Annotated[ctypes.c_uint32, 564]
  cp_rb_wptr_poll_addr_hi: Annotated[ctypes.c_uint32, 568]
  cp_rb_doorbell_control: Annotated[ctypes.c_uint32, 572]
  cp_gfx_hqd_offset: Annotated[ctypes.c_uint32, 576]
  cp_gfx_hqd_cntl: Annotated[ctypes.c_uint32, 580]
  reserved_146: Annotated[ctypes.c_uint32, 584]
  reserved_147: Annotated[ctypes.c_uint32, 588]
  cp_gfx_hqd_csmd_rptr: Annotated[ctypes.c_uint32, 592]
  cp_gfx_hqd_wptr: Annotated[ctypes.c_uint32, 596]
  cp_gfx_hqd_wptr_hi: Annotated[ctypes.c_uint32, 600]
  reserved_151: Annotated[ctypes.c_uint32, 604]
  reserved_152: Annotated[ctypes.c_uint32, 608]
  reserved_153: Annotated[ctypes.c_uint32, 612]
  reserved_154: Annotated[ctypes.c_uint32, 616]
  reserved_155: Annotated[ctypes.c_uint32, 620]
  cp_gfx_hqd_mapped: Annotated[ctypes.c_uint32, 624]
  cp_gfx_hqd_que_mgr_control: Annotated[ctypes.c_uint32, 628]
  reserved_158: Annotated[ctypes.c_uint32, 632]
  reserved_159: Annotated[ctypes.c_uint32, 636]
  cp_gfx_hqd_hq_status0: Annotated[ctypes.c_uint32, 640]
  cp_gfx_hqd_hq_control0: Annotated[ctypes.c_uint32, 644]
  cp_gfx_mqd_control: Annotated[ctypes.c_uint32, 648]
  reserved_163: Annotated[ctypes.c_uint32, 652]
  reserved_164: Annotated[ctypes.c_uint32, 656]
  reserved_165: Annotated[ctypes.c_uint32, 660]
  reserved_166: Annotated[ctypes.c_uint32, 664]
  reserved_167: Annotated[ctypes.c_uint32, 668]
  reserved_168: Annotated[ctypes.c_uint32, 672]
  reserved_169: Annotated[ctypes.c_uint32, 676]
  cp_num_prim_needed_count0_lo: Annotated[ctypes.c_uint32, 680]
  cp_num_prim_needed_count0_hi: Annotated[ctypes.c_uint32, 684]
  cp_num_prim_needed_count1_lo: Annotated[ctypes.c_uint32, 688]
  cp_num_prim_needed_count1_hi: Annotated[ctypes.c_uint32, 692]
  cp_num_prim_needed_count2_lo: Annotated[ctypes.c_uint32, 696]
  cp_num_prim_needed_count2_hi: Annotated[ctypes.c_uint32, 700]
  cp_num_prim_needed_count3_lo: Annotated[ctypes.c_uint32, 704]
  cp_num_prim_needed_count3_hi: Annotated[ctypes.c_uint32, 708]
  cp_num_prim_written_count0_lo: Annotated[ctypes.c_uint32, 712]
  cp_num_prim_written_count0_hi: Annotated[ctypes.c_uint32, 716]
  cp_num_prim_written_count1_lo: Annotated[ctypes.c_uint32, 720]
  cp_num_prim_written_count1_hi: Annotated[ctypes.c_uint32, 724]
  cp_num_prim_written_count2_lo: Annotated[ctypes.c_uint32, 728]
  cp_num_prim_written_count2_hi: Annotated[ctypes.c_uint32, 732]
  cp_num_prim_written_count3_lo: Annotated[ctypes.c_uint32, 736]
  cp_num_prim_written_count3_hi: Annotated[ctypes.c_uint32, 740]
  reserved_186: Annotated[ctypes.c_uint32, 744]
  reserved_187: Annotated[ctypes.c_uint32, 748]
  reserved_188: Annotated[ctypes.c_uint32, 752]
  reserved_189: Annotated[ctypes.c_uint32, 756]
  mp1_smn_fps_cnt: Annotated[ctypes.c_uint32, 760]
  sq_thread_trace_buf0_base: Annotated[ctypes.c_uint32, 764]
  sq_thread_trace_buf0_size: Annotated[ctypes.c_uint32, 768]
  sq_thread_trace_buf1_base: Annotated[ctypes.c_uint32, 772]
  sq_thread_trace_buf1_size: Annotated[ctypes.c_uint32, 776]
  sq_thread_trace_wptr: Annotated[ctypes.c_uint32, 780]
  sq_thread_trace_mask: Annotated[ctypes.c_uint32, 784]
  sq_thread_trace_token_mask: Annotated[ctypes.c_uint32, 788]
  sq_thread_trace_ctrl: Annotated[ctypes.c_uint32, 792]
  sq_thread_trace_status: Annotated[ctypes.c_uint32, 796]
  sq_thread_trace_dropped_cntr: Annotated[ctypes.c_uint32, 800]
  sq_thread_trace_finish_done_debug: Annotated[ctypes.c_uint32, 804]
  sq_thread_trace_gfx_draw_cntr: Annotated[ctypes.c_uint32, 808]
  sq_thread_trace_gfx_marker_cntr: Annotated[ctypes.c_uint32, 812]
  sq_thread_trace_hp3d_draw_cntr: Annotated[ctypes.c_uint32, 816]
  sq_thread_trace_hp3d_marker_cntr: Annotated[ctypes.c_uint32, 820]
  reserved_206: Annotated[ctypes.c_uint32, 824]
  reserved_207: Annotated[ctypes.c_uint32, 828]
  cp_sc_psinvoc_count0_lo: Annotated[ctypes.c_uint32, 832]
  cp_sc_psinvoc_count0_hi: Annotated[ctypes.c_uint32, 836]
  cp_pa_cprim_count_lo: Annotated[ctypes.c_uint32, 840]
  cp_pa_cprim_count_hi: Annotated[ctypes.c_uint32, 844]
  cp_pa_cinvoc_count_lo: Annotated[ctypes.c_uint32, 848]
  cp_pa_cinvoc_count_hi: Annotated[ctypes.c_uint32, 852]
  cp_vgt_vsinvoc_count_lo: Annotated[ctypes.c_uint32, 856]
  cp_vgt_vsinvoc_count_hi: Annotated[ctypes.c_uint32, 860]
  cp_vgt_gsinvoc_count_lo: Annotated[ctypes.c_uint32, 864]
  cp_vgt_gsinvoc_count_hi: Annotated[ctypes.c_uint32, 868]
  cp_vgt_gsprim_count_lo: Annotated[ctypes.c_uint32, 872]
  cp_vgt_gsprim_count_hi: Annotated[ctypes.c_uint32, 876]
  cp_vgt_iaprim_count_lo: Annotated[ctypes.c_uint32, 880]
  cp_vgt_iaprim_count_hi: Annotated[ctypes.c_uint32, 884]
  cp_vgt_iavert_count_lo: Annotated[ctypes.c_uint32, 888]
  cp_vgt_iavert_count_hi: Annotated[ctypes.c_uint32, 892]
  cp_vgt_hsinvoc_count_lo: Annotated[ctypes.c_uint32, 896]
  cp_vgt_hsinvoc_count_hi: Annotated[ctypes.c_uint32, 900]
  cp_vgt_dsinvoc_count_lo: Annotated[ctypes.c_uint32, 904]
  cp_vgt_dsinvoc_count_hi: Annotated[ctypes.c_uint32, 908]
  cp_vgt_csinvoc_count_lo: Annotated[ctypes.c_uint32, 912]
  cp_vgt_csinvoc_count_hi: Annotated[ctypes.c_uint32, 916]
  reserved_230: Annotated[ctypes.c_uint32, 920]
  reserved_231: Annotated[ctypes.c_uint32, 924]
  reserved_232: Annotated[ctypes.c_uint32, 928]
  reserved_233: Annotated[ctypes.c_uint32, 932]
  reserved_234: Annotated[ctypes.c_uint32, 936]
  reserved_235: Annotated[ctypes.c_uint32, 940]
  reserved_236: Annotated[ctypes.c_uint32, 944]
  reserved_237: Annotated[ctypes.c_uint32, 948]
  reserved_238: Annotated[ctypes.c_uint32, 952]
  reserved_239: Annotated[ctypes.c_uint32, 956]
  reserved_240: Annotated[ctypes.c_uint32, 960]
  reserved_241: Annotated[ctypes.c_uint32, 964]
  reserved_242: Annotated[ctypes.c_uint32, 968]
  reserved_243: Annotated[ctypes.c_uint32, 972]
  reserved_244: Annotated[ctypes.c_uint32, 976]
  reserved_245: Annotated[ctypes.c_uint32, 980]
  reserved_246: Annotated[ctypes.c_uint32, 984]
  reserved_247: Annotated[ctypes.c_uint32, 988]
  reserved_248: Annotated[ctypes.c_uint32, 992]
  reserved_249: Annotated[ctypes.c_uint32, 996]
  reserved_250: Annotated[ctypes.c_uint32, 1000]
  reserved_251: Annotated[ctypes.c_uint32, 1004]
  reserved_252: Annotated[ctypes.c_uint32, 1008]
  reserved_253: Annotated[ctypes.c_uint32, 1012]
  reserved_254: Annotated[ctypes.c_uint32, 1016]
  reserved_255: Annotated[ctypes.c_uint32, 1020]
  reserved_256: Annotated[ctypes.c_uint32, 1024]
  reserved_257: Annotated[ctypes.c_uint32, 1028]
  reserved_258: Annotated[ctypes.c_uint32, 1032]
  reserved_259: Annotated[ctypes.c_uint32, 1036]
  reserved_260: Annotated[ctypes.c_uint32, 1040]
  reserved_261: Annotated[ctypes.c_uint32, 1044]
  reserved_262: Annotated[ctypes.c_uint32, 1048]
  reserved_263: Annotated[ctypes.c_uint32, 1052]
  reserved_264: Annotated[ctypes.c_uint32, 1056]
  reserved_265: Annotated[ctypes.c_uint32, 1060]
  reserved_266: Annotated[ctypes.c_uint32, 1064]
  reserved_267: Annotated[ctypes.c_uint32, 1068]
  vgt_strmout_buffer_filled_size_0: Annotated[ctypes.c_uint32, 1072]
  vgt_strmout_buffer_filled_size_1: Annotated[ctypes.c_uint32, 1076]
  vgt_strmout_buffer_filled_size_2: Annotated[ctypes.c_uint32, 1080]
  vgt_strmout_buffer_filled_size_3: Annotated[ctypes.c_uint32, 1084]
  reserved_272: Annotated[ctypes.c_uint32, 1088]
  reserved_273: Annotated[ctypes.c_uint32, 1092]
  reserved_274: Annotated[ctypes.c_uint32, 1096]
  reserved_275: Annotated[ctypes.c_uint32, 1100]
  vgt_dma_max_size: Annotated[ctypes.c_uint32, 1104]
  vgt_dma_num_instances: Annotated[ctypes.c_uint32, 1108]
  reserved_278: Annotated[ctypes.c_uint32, 1112]
  reserved_279: Annotated[ctypes.c_uint32, 1116]
  reserved_280: Annotated[ctypes.c_uint32, 1120]
  reserved_281: Annotated[ctypes.c_uint32, 1124]
  reserved_282: Annotated[ctypes.c_uint32, 1128]
  reserved_283: Annotated[ctypes.c_uint32, 1132]
  reserved_284: Annotated[ctypes.c_uint32, 1136]
  reserved_285: Annotated[ctypes.c_uint32, 1140]
  reserved_286: Annotated[ctypes.c_uint32, 1144]
  reserved_287: Annotated[ctypes.c_uint32, 1148]
  it_set_base_ib_addr_lo: Annotated[ctypes.c_uint32, 1152]
  it_set_base_ib_addr_hi: Annotated[ctypes.c_uint32, 1156]
  reserved_290: Annotated[ctypes.c_uint32, 1160]
  reserved_291: Annotated[ctypes.c_uint32, 1164]
  reserved_292: Annotated[ctypes.c_uint32, 1168]
  reserved_293: Annotated[ctypes.c_uint32, 1172]
  reserved_294: Annotated[ctypes.c_uint32, 1176]
  reserved_295: Annotated[ctypes.c_uint32, 1180]
  reserved_296: Annotated[ctypes.c_uint32, 1184]
  reserved_297: Annotated[ctypes.c_uint32, 1188]
  reserved_298: Annotated[ctypes.c_uint32, 1192]
  reserved_299: Annotated[ctypes.c_uint32, 1196]
  reserved_300: Annotated[ctypes.c_uint32, 1200]
  reserved_301: Annotated[ctypes.c_uint32, 1204]
  reserved_302: Annotated[ctypes.c_uint32, 1208]
  reserved_303: Annotated[ctypes.c_uint32, 1212]
  reserved_304: Annotated[ctypes.c_uint32, 1216]
  reserved_305: Annotated[ctypes.c_uint32, 1220]
  reserved_306: Annotated[ctypes.c_uint32, 1224]
  reserved_307: Annotated[ctypes.c_uint32, 1228]
  reserved_308: Annotated[ctypes.c_uint32, 1232]
  reserved_309: Annotated[ctypes.c_uint32, 1236]
  reserved_310: Annotated[ctypes.c_uint32, 1240]
  reserved_311: Annotated[ctypes.c_uint32, 1244]
  reserved_312: Annotated[ctypes.c_uint32, 1248]
  reserved_313: Annotated[ctypes.c_uint32, 1252]
  reserved_314: Annotated[ctypes.c_uint32, 1256]
  reserved_315: Annotated[ctypes.c_uint32, 1260]
  reserved_316: Annotated[ctypes.c_uint32, 1264]
  reserved_317: Annotated[ctypes.c_uint32, 1268]
  reserved_318: Annotated[ctypes.c_uint32, 1272]
  reserved_319: Annotated[ctypes.c_uint32, 1276]
  reserved_320: Annotated[ctypes.c_uint32, 1280]
  reserved_321: Annotated[ctypes.c_uint32, 1284]
  reserved_322: Annotated[ctypes.c_uint32, 1288]
  reserved_323: Annotated[ctypes.c_uint32, 1292]
  reserved_324: Annotated[ctypes.c_uint32, 1296]
  reserved_325: Annotated[ctypes.c_uint32, 1300]
  reserved_326: Annotated[ctypes.c_uint32, 1304]
  reserved_327: Annotated[ctypes.c_uint32, 1308]
  reserved_328: Annotated[ctypes.c_uint32, 1312]
  reserved_329: Annotated[ctypes.c_uint32, 1316]
  reserved_330: Annotated[ctypes.c_uint32, 1320]
  reserved_331: Annotated[ctypes.c_uint32, 1324]
  reserved_332: Annotated[ctypes.c_uint32, 1328]
  reserved_333: Annotated[ctypes.c_uint32, 1332]
  reserved_334: Annotated[ctypes.c_uint32, 1336]
  reserved_335: Annotated[ctypes.c_uint32, 1340]
  reserved_336: Annotated[ctypes.c_uint32, 1344]
  reserved_337: Annotated[ctypes.c_uint32, 1348]
  reserved_338: Annotated[ctypes.c_uint32, 1352]
  reserved_339: Annotated[ctypes.c_uint32, 1356]
  reserved_340: Annotated[ctypes.c_uint32, 1360]
  reserved_341: Annotated[ctypes.c_uint32, 1364]
  reserved_342: Annotated[ctypes.c_uint32, 1368]
  reserved_343: Annotated[ctypes.c_uint32, 1372]
  reserved_344: Annotated[ctypes.c_uint32, 1376]
  reserved_345: Annotated[ctypes.c_uint32, 1380]
  reserved_346: Annotated[ctypes.c_uint32, 1384]
  reserved_347: Annotated[ctypes.c_uint32, 1388]
  reserved_348: Annotated[ctypes.c_uint32, 1392]
  reserved_349: Annotated[ctypes.c_uint32, 1396]
  reserved_350: Annotated[ctypes.c_uint32, 1400]
  reserved_351: Annotated[ctypes.c_uint32, 1404]
  reserved_352: Annotated[ctypes.c_uint32, 1408]
  reserved_353: Annotated[ctypes.c_uint32, 1412]
  reserved_354: Annotated[ctypes.c_uint32, 1416]
  reserved_355: Annotated[ctypes.c_uint32, 1420]
  spi_shader_pgm_rsrc3_ps: Annotated[ctypes.c_uint32, 1424]
  spi_shader_pgm_rsrc3_vs: Annotated[ctypes.c_uint32, 1428]
  spi_shader_pgm_rsrc3_gs: Annotated[ctypes.c_uint32, 1432]
  spi_shader_pgm_rsrc3_hs: Annotated[ctypes.c_uint32, 1436]
  spi_shader_pgm_rsrc4_ps: Annotated[ctypes.c_uint32, 1440]
  spi_shader_pgm_rsrc4_vs: Annotated[ctypes.c_uint32, 1444]
  spi_shader_pgm_rsrc4_gs: Annotated[ctypes.c_uint32, 1448]
  spi_shader_pgm_rsrc4_hs: Annotated[ctypes.c_uint32, 1452]
  db_occlusion_count0_low_00: Annotated[ctypes.c_uint32, 1456]
  db_occlusion_count0_hi_00: Annotated[ctypes.c_uint32, 1460]
  db_occlusion_count1_low_00: Annotated[ctypes.c_uint32, 1464]
  db_occlusion_count1_hi_00: Annotated[ctypes.c_uint32, 1468]
  db_occlusion_count2_low_00: Annotated[ctypes.c_uint32, 1472]
  db_occlusion_count2_hi_00: Annotated[ctypes.c_uint32, 1476]
  db_occlusion_count3_low_00: Annotated[ctypes.c_uint32, 1480]
  db_occlusion_count3_hi_00: Annotated[ctypes.c_uint32, 1484]
  db_occlusion_count0_low_01: Annotated[ctypes.c_uint32, 1488]
  db_occlusion_count0_hi_01: Annotated[ctypes.c_uint32, 1492]
  db_occlusion_count1_low_01: Annotated[ctypes.c_uint32, 1496]
  db_occlusion_count1_hi_01: Annotated[ctypes.c_uint32, 1500]
  db_occlusion_count2_low_01: Annotated[ctypes.c_uint32, 1504]
  db_occlusion_count2_hi_01: Annotated[ctypes.c_uint32, 1508]
  db_occlusion_count3_low_01: Annotated[ctypes.c_uint32, 1512]
  db_occlusion_count3_hi_01: Annotated[ctypes.c_uint32, 1516]
  db_occlusion_count0_low_02: Annotated[ctypes.c_uint32, 1520]
  db_occlusion_count0_hi_02: Annotated[ctypes.c_uint32, 1524]
  db_occlusion_count1_low_02: Annotated[ctypes.c_uint32, 1528]
  db_occlusion_count1_hi_02: Annotated[ctypes.c_uint32, 1532]
  db_occlusion_count2_low_02: Annotated[ctypes.c_uint32, 1536]
  db_occlusion_count2_hi_02: Annotated[ctypes.c_uint32, 1540]
  db_occlusion_count3_low_02: Annotated[ctypes.c_uint32, 1544]
  db_occlusion_count3_hi_02: Annotated[ctypes.c_uint32, 1548]
  db_occlusion_count0_low_03: Annotated[ctypes.c_uint32, 1552]
  db_occlusion_count0_hi_03: Annotated[ctypes.c_uint32, 1556]
  db_occlusion_count1_low_03: Annotated[ctypes.c_uint32, 1560]
  db_occlusion_count1_hi_03: Annotated[ctypes.c_uint32, 1564]
  db_occlusion_count2_low_03: Annotated[ctypes.c_uint32, 1568]
  db_occlusion_count2_hi_03: Annotated[ctypes.c_uint32, 1572]
  db_occlusion_count3_low_03: Annotated[ctypes.c_uint32, 1576]
  db_occlusion_count3_hi_03: Annotated[ctypes.c_uint32, 1580]
  db_occlusion_count0_low_04: Annotated[ctypes.c_uint32, 1584]
  db_occlusion_count0_hi_04: Annotated[ctypes.c_uint32, 1588]
  db_occlusion_count1_low_04: Annotated[ctypes.c_uint32, 1592]
  db_occlusion_count1_hi_04: Annotated[ctypes.c_uint32, 1596]
  db_occlusion_count2_low_04: Annotated[ctypes.c_uint32, 1600]
  db_occlusion_count2_hi_04: Annotated[ctypes.c_uint32, 1604]
  db_occlusion_count3_low_04: Annotated[ctypes.c_uint32, 1608]
  db_occlusion_count3_hi_04: Annotated[ctypes.c_uint32, 1612]
  db_occlusion_count0_low_05: Annotated[ctypes.c_uint32, 1616]
  db_occlusion_count0_hi_05: Annotated[ctypes.c_uint32, 1620]
  db_occlusion_count1_low_05: Annotated[ctypes.c_uint32, 1624]
  db_occlusion_count1_hi_05: Annotated[ctypes.c_uint32, 1628]
  db_occlusion_count2_low_05: Annotated[ctypes.c_uint32, 1632]
  db_occlusion_count2_hi_05: Annotated[ctypes.c_uint32, 1636]
  db_occlusion_count3_low_05: Annotated[ctypes.c_uint32, 1640]
  db_occlusion_count3_hi_05: Annotated[ctypes.c_uint32, 1644]
  db_occlusion_count0_low_06: Annotated[ctypes.c_uint32, 1648]
  db_occlusion_count0_hi_06: Annotated[ctypes.c_uint32, 1652]
  db_occlusion_count1_low_06: Annotated[ctypes.c_uint32, 1656]
  db_occlusion_count1_hi_06: Annotated[ctypes.c_uint32, 1660]
  db_occlusion_count2_low_06: Annotated[ctypes.c_uint32, 1664]
  db_occlusion_count2_hi_06: Annotated[ctypes.c_uint32, 1668]
  db_occlusion_count3_low_06: Annotated[ctypes.c_uint32, 1672]
  db_occlusion_count3_hi_06: Annotated[ctypes.c_uint32, 1676]
  db_occlusion_count0_low_07: Annotated[ctypes.c_uint32, 1680]
  db_occlusion_count0_hi_07: Annotated[ctypes.c_uint32, 1684]
  db_occlusion_count1_low_07: Annotated[ctypes.c_uint32, 1688]
  db_occlusion_count1_hi_07: Annotated[ctypes.c_uint32, 1692]
  db_occlusion_count2_low_07: Annotated[ctypes.c_uint32, 1696]
  db_occlusion_count2_hi_07: Annotated[ctypes.c_uint32, 1700]
  db_occlusion_count3_low_07: Annotated[ctypes.c_uint32, 1704]
  db_occlusion_count3_hi_07: Annotated[ctypes.c_uint32, 1708]
  db_occlusion_count0_low_10: Annotated[ctypes.c_uint32, 1712]
  db_occlusion_count0_hi_10: Annotated[ctypes.c_uint32, 1716]
  db_occlusion_count1_low_10: Annotated[ctypes.c_uint32, 1720]
  db_occlusion_count1_hi_10: Annotated[ctypes.c_uint32, 1724]
  db_occlusion_count2_low_10: Annotated[ctypes.c_uint32, 1728]
  db_occlusion_count2_hi_10: Annotated[ctypes.c_uint32, 1732]
  db_occlusion_count3_low_10: Annotated[ctypes.c_uint32, 1736]
  db_occlusion_count3_hi_10: Annotated[ctypes.c_uint32, 1740]
  db_occlusion_count0_low_11: Annotated[ctypes.c_uint32, 1744]
  db_occlusion_count0_hi_11: Annotated[ctypes.c_uint32, 1748]
  db_occlusion_count1_low_11: Annotated[ctypes.c_uint32, 1752]
  db_occlusion_count1_hi_11: Annotated[ctypes.c_uint32, 1756]
  db_occlusion_count2_low_11: Annotated[ctypes.c_uint32, 1760]
  db_occlusion_count2_hi_11: Annotated[ctypes.c_uint32, 1764]
  db_occlusion_count3_low_11: Annotated[ctypes.c_uint32, 1768]
  db_occlusion_count3_hi_11: Annotated[ctypes.c_uint32, 1772]
  db_occlusion_count0_low_12: Annotated[ctypes.c_uint32, 1776]
  db_occlusion_count0_hi_12: Annotated[ctypes.c_uint32, 1780]
  db_occlusion_count1_low_12: Annotated[ctypes.c_uint32, 1784]
  db_occlusion_count1_hi_12: Annotated[ctypes.c_uint32, 1788]
  db_occlusion_count2_low_12: Annotated[ctypes.c_uint32, 1792]
  db_occlusion_count2_hi_12: Annotated[ctypes.c_uint32, 1796]
  db_occlusion_count3_low_12: Annotated[ctypes.c_uint32, 1800]
  db_occlusion_count3_hi_12: Annotated[ctypes.c_uint32, 1804]
  db_occlusion_count0_low_13: Annotated[ctypes.c_uint32, 1808]
  db_occlusion_count0_hi_13: Annotated[ctypes.c_uint32, 1812]
  db_occlusion_count1_low_13: Annotated[ctypes.c_uint32, 1816]
  db_occlusion_count1_hi_13: Annotated[ctypes.c_uint32, 1820]
  db_occlusion_count2_low_13: Annotated[ctypes.c_uint32, 1824]
  db_occlusion_count2_hi_13: Annotated[ctypes.c_uint32, 1828]
  db_occlusion_count3_low_13: Annotated[ctypes.c_uint32, 1832]
  db_occlusion_count3_hi_13: Annotated[ctypes.c_uint32, 1836]
  db_occlusion_count0_low_14: Annotated[ctypes.c_uint32, 1840]
  db_occlusion_count0_hi_14: Annotated[ctypes.c_uint32, 1844]
  db_occlusion_count1_low_14: Annotated[ctypes.c_uint32, 1848]
  db_occlusion_count1_hi_14: Annotated[ctypes.c_uint32, 1852]
  db_occlusion_count2_low_14: Annotated[ctypes.c_uint32, 1856]
  db_occlusion_count2_hi_14: Annotated[ctypes.c_uint32, 1860]
  db_occlusion_count3_low_14: Annotated[ctypes.c_uint32, 1864]
  db_occlusion_count3_hi_14: Annotated[ctypes.c_uint32, 1868]
  db_occlusion_count0_low_15: Annotated[ctypes.c_uint32, 1872]
  db_occlusion_count0_hi_15: Annotated[ctypes.c_uint32, 1876]
  db_occlusion_count1_low_15: Annotated[ctypes.c_uint32, 1880]
  db_occlusion_count1_hi_15: Annotated[ctypes.c_uint32, 1884]
  db_occlusion_count2_low_15: Annotated[ctypes.c_uint32, 1888]
  db_occlusion_count2_hi_15: Annotated[ctypes.c_uint32, 1892]
  db_occlusion_count3_low_15: Annotated[ctypes.c_uint32, 1896]
  db_occlusion_count3_hi_15: Annotated[ctypes.c_uint32, 1900]
  db_occlusion_count0_low_16: Annotated[ctypes.c_uint32, 1904]
  db_occlusion_count0_hi_16: Annotated[ctypes.c_uint32, 1908]
  db_occlusion_count1_low_16: Annotated[ctypes.c_uint32, 1912]
  db_occlusion_count1_hi_16: Annotated[ctypes.c_uint32, 1916]
  db_occlusion_count2_low_16: Annotated[ctypes.c_uint32, 1920]
  db_occlusion_count2_hi_16: Annotated[ctypes.c_uint32, 1924]
  db_occlusion_count3_low_16: Annotated[ctypes.c_uint32, 1928]
  db_occlusion_count3_hi_16: Annotated[ctypes.c_uint32, 1932]
  db_occlusion_count0_low_17: Annotated[ctypes.c_uint32, 1936]
  db_occlusion_count0_hi_17: Annotated[ctypes.c_uint32, 1940]
  db_occlusion_count1_low_17: Annotated[ctypes.c_uint32, 1944]
  db_occlusion_count1_hi_17: Annotated[ctypes.c_uint32, 1948]
  db_occlusion_count2_low_17: Annotated[ctypes.c_uint32, 1952]
  db_occlusion_count2_hi_17: Annotated[ctypes.c_uint32, 1956]
  db_occlusion_count3_low_17: Annotated[ctypes.c_uint32, 1960]
  db_occlusion_count3_hi_17: Annotated[ctypes.c_uint32, 1964]
  reserved_492: Annotated[ctypes.c_uint32, 1968]
  reserved_493: Annotated[ctypes.c_uint32, 1972]
  reserved_494: Annotated[ctypes.c_uint32, 1976]
  reserved_495: Annotated[ctypes.c_uint32, 1980]
  reserved_496: Annotated[ctypes.c_uint32, 1984]
  reserved_497: Annotated[ctypes.c_uint32, 1988]
  reserved_498: Annotated[ctypes.c_uint32, 1992]
  reserved_499: Annotated[ctypes.c_uint32, 1996]
  reserved_500: Annotated[ctypes.c_uint32, 2000]
  reserved_501: Annotated[ctypes.c_uint32, 2004]
  reserved_502: Annotated[ctypes.c_uint32, 2008]
  reserved_503: Annotated[ctypes.c_uint32, 2012]
  reserved_504: Annotated[ctypes.c_uint32, 2016]
  reserved_505: Annotated[ctypes.c_uint32, 2020]
  reserved_506: Annotated[ctypes.c_uint32, 2024]
  reserved_507: Annotated[ctypes.c_uint32, 2028]
  reserved_508: Annotated[ctypes.c_uint32, 2032]
  reserved_509: Annotated[ctypes.c_uint32, 2036]
  reserved_510: Annotated[ctypes.c_uint32, 2040]
  reserved_511: Annotated[ctypes.c_uint32, 2044]
@record
class struct_v11_sdma_mqd:
  SIZE = 512
  sdmax_rlcx_rb_cntl: Annotated[ctypes.c_uint32, 0]
  sdmax_rlcx_rb_base: Annotated[ctypes.c_uint32, 4]
  sdmax_rlcx_rb_base_hi: Annotated[ctypes.c_uint32, 8]
  sdmax_rlcx_rb_rptr: Annotated[ctypes.c_uint32, 12]
  sdmax_rlcx_rb_rptr_hi: Annotated[ctypes.c_uint32, 16]
  sdmax_rlcx_rb_wptr: Annotated[ctypes.c_uint32, 20]
  sdmax_rlcx_rb_wptr_hi: Annotated[ctypes.c_uint32, 24]
  sdmax_rlcx_rb_rptr_addr_hi: Annotated[ctypes.c_uint32, 28]
  sdmax_rlcx_rb_rptr_addr_lo: Annotated[ctypes.c_uint32, 32]
  sdmax_rlcx_ib_cntl: Annotated[ctypes.c_uint32, 36]
  sdmax_rlcx_ib_rptr: Annotated[ctypes.c_uint32, 40]
  sdmax_rlcx_ib_offset: Annotated[ctypes.c_uint32, 44]
  sdmax_rlcx_ib_base_lo: Annotated[ctypes.c_uint32, 48]
  sdmax_rlcx_ib_base_hi: Annotated[ctypes.c_uint32, 52]
  sdmax_rlcx_ib_size: Annotated[ctypes.c_uint32, 56]
  sdmax_rlcx_skip_cntl: Annotated[ctypes.c_uint32, 60]
  sdmax_rlcx_context_status: Annotated[ctypes.c_uint32, 64]
  sdmax_rlcx_doorbell: Annotated[ctypes.c_uint32, 68]
  sdmax_rlcx_doorbell_log: Annotated[ctypes.c_uint32, 72]
  sdmax_rlcx_doorbell_offset: Annotated[ctypes.c_uint32, 76]
  sdmax_rlcx_csa_addr_lo: Annotated[ctypes.c_uint32, 80]
  sdmax_rlcx_csa_addr_hi: Annotated[ctypes.c_uint32, 84]
  sdmax_rlcx_sched_cntl: Annotated[ctypes.c_uint32, 88]
  sdmax_rlcx_ib_sub_remain: Annotated[ctypes.c_uint32, 92]
  sdmax_rlcx_preempt: Annotated[ctypes.c_uint32, 96]
  sdmax_rlcx_dummy_reg: Annotated[ctypes.c_uint32, 100]
  sdmax_rlcx_rb_wptr_poll_addr_hi: Annotated[ctypes.c_uint32, 104]
  sdmax_rlcx_rb_wptr_poll_addr_lo: Annotated[ctypes.c_uint32, 108]
  sdmax_rlcx_rb_aql_cntl: Annotated[ctypes.c_uint32, 112]
  sdmax_rlcx_minor_ptr_update: Annotated[ctypes.c_uint32, 116]
  sdmax_rlcx_rb_preempt: Annotated[ctypes.c_uint32, 120]
  sdmax_rlcx_midcmd_data0: Annotated[ctypes.c_uint32, 124]
  sdmax_rlcx_midcmd_data1: Annotated[ctypes.c_uint32, 128]
  sdmax_rlcx_midcmd_data2: Annotated[ctypes.c_uint32, 132]
  sdmax_rlcx_midcmd_data3: Annotated[ctypes.c_uint32, 136]
  sdmax_rlcx_midcmd_data4: Annotated[ctypes.c_uint32, 140]
  sdmax_rlcx_midcmd_data5: Annotated[ctypes.c_uint32, 144]
  sdmax_rlcx_midcmd_data6: Annotated[ctypes.c_uint32, 148]
  sdmax_rlcx_midcmd_data7: Annotated[ctypes.c_uint32, 152]
  sdmax_rlcx_midcmd_data8: Annotated[ctypes.c_uint32, 156]
  sdmax_rlcx_midcmd_data9: Annotated[ctypes.c_uint32, 160]
  sdmax_rlcx_midcmd_data10: Annotated[ctypes.c_uint32, 164]
  sdmax_rlcx_midcmd_cntl: Annotated[ctypes.c_uint32, 168]
  sdmax_rlcx_f32_dbg0: Annotated[ctypes.c_uint32, 172]
  sdmax_rlcx_f32_dbg1: Annotated[ctypes.c_uint32, 176]
  reserved_45: Annotated[ctypes.c_uint32, 180]
  reserved_46: Annotated[ctypes.c_uint32, 184]
  reserved_47: Annotated[ctypes.c_uint32, 188]
  reserved_48: Annotated[ctypes.c_uint32, 192]
  reserved_49: Annotated[ctypes.c_uint32, 196]
  reserved_50: Annotated[ctypes.c_uint32, 200]
  reserved_51: Annotated[ctypes.c_uint32, 204]
  reserved_52: Annotated[ctypes.c_uint32, 208]
  reserved_53: Annotated[ctypes.c_uint32, 212]
  reserved_54: Annotated[ctypes.c_uint32, 216]
  reserved_55: Annotated[ctypes.c_uint32, 220]
  reserved_56: Annotated[ctypes.c_uint32, 224]
  reserved_57: Annotated[ctypes.c_uint32, 228]
  reserved_58: Annotated[ctypes.c_uint32, 232]
  reserved_59: Annotated[ctypes.c_uint32, 236]
  reserved_60: Annotated[ctypes.c_uint32, 240]
  reserved_61: Annotated[ctypes.c_uint32, 244]
  reserved_62: Annotated[ctypes.c_uint32, 248]
  reserved_63: Annotated[ctypes.c_uint32, 252]
  reserved_64: Annotated[ctypes.c_uint32, 256]
  reserved_65: Annotated[ctypes.c_uint32, 260]
  reserved_66: Annotated[ctypes.c_uint32, 264]
  reserved_67: Annotated[ctypes.c_uint32, 268]
  reserved_68: Annotated[ctypes.c_uint32, 272]
  reserved_69: Annotated[ctypes.c_uint32, 276]
  reserved_70: Annotated[ctypes.c_uint32, 280]
  reserved_71: Annotated[ctypes.c_uint32, 284]
  reserved_72: Annotated[ctypes.c_uint32, 288]
  reserved_73: Annotated[ctypes.c_uint32, 292]
  reserved_74: Annotated[ctypes.c_uint32, 296]
  reserved_75: Annotated[ctypes.c_uint32, 300]
  reserved_76: Annotated[ctypes.c_uint32, 304]
  reserved_77: Annotated[ctypes.c_uint32, 308]
  reserved_78: Annotated[ctypes.c_uint32, 312]
  reserved_79: Annotated[ctypes.c_uint32, 316]
  reserved_80: Annotated[ctypes.c_uint32, 320]
  reserved_81: Annotated[ctypes.c_uint32, 324]
  reserved_82: Annotated[ctypes.c_uint32, 328]
  reserved_83: Annotated[ctypes.c_uint32, 332]
  reserved_84: Annotated[ctypes.c_uint32, 336]
  reserved_85: Annotated[ctypes.c_uint32, 340]
  reserved_86: Annotated[ctypes.c_uint32, 344]
  reserved_87: Annotated[ctypes.c_uint32, 348]
  reserved_88: Annotated[ctypes.c_uint32, 352]
  reserved_89: Annotated[ctypes.c_uint32, 356]
  reserved_90: Annotated[ctypes.c_uint32, 360]
  reserved_91: Annotated[ctypes.c_uint32, 364]
  reserved_92: Annotated[ctypes.c_uint32, 368]
  reserved_93: Annotated[ctypes.c_uint32, 372]
  reserved_94: Annotated[ctypes.c_uint32, 376]
  reserved_95: Annotated[ctypes.c_uint32, 380]
  reserved_96: Annotated[ctypes.c_uint32, 384]
  reserved_97: Annotated[ctypes.c_uint32, 388]
  reserved_98: Annotated[ctypes.c_uint32, 392]
  reserved_99: Annotated[ctypes.c_uint32, 396]
  reserved_100: Annotated[ctypes.c_uint32, 400]
  reserved_101: Annotated[ctypes.c_uint32, 404]
  reserved_102: Annotated[ctypes.c_uint32, 408]
  reserved_103: Annotated[ctypes.c_uint32, 412]
  reserved_104: Annotated[ctypes.c_uint32, 416]
  reserved_105: Annotated[ctypes.c_uint32, 420]
  reserved_106: Annotated[ctypes.c_uint32, 424]
  reserved_107: Annotated[ctypes.c_uint32, 428]
  reserved_108: Annotated[ctypes.c_uint32, 432]
  reserved_109: Annotated[ctypes.c_uint32, 436]
  reserved_110: Annotated[ctypes.c_uint32, 440]
  reserved_111: Annotated[ctypes.c_uint32, 444]
  reserved_112: Annotated[ctypes.c_uint32, 448]
  reserved_113: Annotated[ctypes.c_uint32, 452]
  reserved_114: Annotated[ctypes.c_uint32, 456]
  reserved_115: Annotated[ctypes.c_uint32, 460]
  reserved_116: Annotated[ctypes.c_uint32, 464]
  reserved_117: Annotated[ctypes.c_uint32, 468]
  reserved_118: Annotated[ctypes.c_uint32, 472]
  reserved_119: Annotated[ctypes.c_uint32, 476]
  reserved_120: Annotated[ctypes.c_uint32, 480]
  reserved_121: Annotated[ctypes.c_uint32, 484]
  reserved_122: Annotated[ctypes.c_uint32, 488]
  reserved_123: Annotated[ctypes.c_uint32, 492]
  reserved_124: Annotated[ctypes.c_uint32, 496]
  reserved_125: Annotated[ctypes.c_uint32, 500]
  sdma_engine_id: Annotated[ctypes.c_uint32, 504]
  sdma_queue_id: Annotated[ctypes.c_uint32, 508]
@record
class struct_v11_compute_mqd:
  SIZE = 2048
  header: Annotated[ctypes.c_uint32, 0]
  compute_dispatch_initiator: Annotated[ctypes.c_uint32, 4]
  compute_dim_x: Annotated[ctypes.c_uint32, 8]
  compute_dim_y: Annotated[ctypes.c_uint32, 12]
  compute_dim_z: Annotated[ctypes.c_uint32, 16]
  compute_start_x: Annotated[ctypes.c_uint32, 20]
  compute_start_y: Annotated[ctypes.c_uint32, 24]
  compute_start_z: Annotated[ctypes.c_uint32, 28]
  compute_num_thread_x: Annotated[ctypes.c_uint32, 32]
  compute_num_thread_y: Annotated[ctypes.c_uint32, 36]
  compute_num_thread_z: Annotated[ctypes.c_uint32, 40]
  compute_pipelinestat_enable: Annotated[ctypes.c_uint32, 44]
  compute_perfcount_enable: Annotated[ctypes.c_uint32, 48]
  compute_pgm_lo: Annotated[ctypes.c_uint32, 52]
  compute_pgm_hi: Annotated[ctypes.c_uint32, 56]
  compute_dispatch_pkt_addr_lo: Annotated[ctypes.c_uint32, 60]
  compute_dispatch_pkt_addr_hi: Annotated[ctypes.c_uint32, 64]
  compute_dispatch_scratch_base_lo: Annotated[ctypes.c_uint32, 68]
  compute_dispatch_scratch_base_hi: Annotated[ctypes.c_uint32, 72]
  compute_pgm_rsrc1: Annotated[ctypes.c_uint32, 76]
  compute_pgm_rsrc2: Annotated[ctypes.c_uint32, 80]
  compute_vmid: Annotated[ctypes.c_uint32, 84]
  compute_resource_limits: Annotated[ctypes.c_uint32, 88]
  compute_static_thread_mgmt_se0: Annotated[ctypes.c_uint32, 92]
  compute_static_thread_mgmt_se1: Annotated[ctypes.c_uint32, 96]
  compute_tmpring_size: Annotated[ctypes.c_uint32, 100]
  compute_static_thread_mgmt_se2: Annotated[ctypes.c_uint32, 104]
  compute_static_thread_mgmt_se3: Annotated[ctypes.c_uint32, 108]
  compute_restart_x: Annotated[ctypes.c_uint32, 112]
  compute_restart_y: Annotated[ctypes.c_uint32, 116]
  compute_restart_z: Annotated[ctypes.c_uint32, 120]
  compute_thread_trace_enable: Annotated[ctypes.c_uint32, 124]
  compute_misc_reserved: Annotated[ctypes.c_uint32, 128]
  compute_dispatch_id: Annotated[ctypes.c_uint32, 132]
  compute_threadgroup_id: Annotated[ctypes.c_uint32, 136]
  compute_req_ctrl: Annotated[ctypes.c_uint32, 140]
  reserved_36: Annotated[ctypes.c_uint32, 144]
  compute_user_accum_0: Annotated[ctypes.c_uint32, 148]
  compute_user_accum_1: Annotated[ctypes.c_uint32, 152]
  compute_user_accum_2: Annotated[ctypes.c_uint32, 156]
  compute_user_accum_3: Annotated[ctypes.c_uint32, 160]
  compute_pgm_rsrc3: Annotated[ctypes.c_uint32, 164]
  compute_ddid_index: Annotated[ctypes.c_uint32, 168]
  compute_shader_chksum: Annotated[ctypes.c_uint32, 172]
  compute_static_thread_mgmt_se4: Annotated[ctypes.c_uint32, 176]
  compute_static_thread_mgmt_se5: Annotated[ctypes.c_uint32, 180]
  compute_static_thread_mgmt_se6: Annotated[ctypes.c_uint32, 184]
  compute_static_thread_mgmt_se7: Annotated[ctypes.c_uint32, 188]
  compute_dispatch_interleave: Annotated[ctypes.c_uint32, 192]
  compute_relaunch: Annotated[ctypes.c_uint32, 196]
  compute_wave_restore_addr_lo: Annotated[ctypes.c_uint32, 200]
  compute_wave_restore_addr_hi: Annotated[ctypes.c_uint32, 204]
  compute_wave_restore_control: Annotated[ctypes.c_uint32, 208]
  reserved_53: Annotated[ctypes.c_uint32, 212]
  reserved_54: Annotated[ctypes.c_uint32, 216]
  reserved_55: Annotated[ctypes.c_uint32, 220]
  reserved_56: Annotated[ctypes.c_uint32, 224]
  reserved_57: Annotated[ctypes.c_uint32, 228]
  reserved_58: Annotated[ctypes.c_uint32, 232]
  reserved_59: Annotated[ctypes.c_uint32, 236]
  reserved_60: Annotated[ctypes.c_uint32, 240]
  reserved_61: Annotated[ctypes.c_uint32, 244]
  reserved_62: Annotated[ctypes.c_uint32, 248]
  reserved_63: Annotated[ctypes.c_uint32, 252]
  reserved_64: Annotated[ctypes.c_uint32, 256]
  compute_user_data_0: Annotated[ctypes.c_uint32, 260]
  compute_user_data_1: Annotated[ctypes.c_uint32, 264]
  compute_user_data_2: Annotated[ctypes.c_uint32, 268]
  compute_user_data_3: Annotated[ctypes.c_uint32, 272]
  compute_user_data_4: Annotated[ctypes.c_uint32, 276]
  compute_user_data_5: Annotated[ctypes.c_uint32, 280]
  compute_user_data_6: Annotated[ctypes.c_uint32, 284]
  compute_user_data_7: Annotated[ctypes.c_uint32, 288]
  compute_user_data_8: Annotated[ctypes.c_uint32, 292]
  compute_user_data_9: Annotated[ctypes.c_uint32, 296]
  compute_user_data_10: Annotated[ctypes.c_uint32, 300]
  compute_user_data_11: Annotated[ctypes.c_uint32, 304]
  compute_user_data_12: Annotated[ctypes.c_uint32, 308]
  compute_user_data_13: Annotated[ctypes.c_uint32, 312]
  compute_user_data_14: Annotated[ctypes.c_uint32, 316]
  compute_user_data_15: Annotated[ctypes.c_uint32, 320]
  cp_compute_csinvoc_count_lo: Annotated[ctypes.c_uint32, 324]
  cp_compute_csinvoc_count_hi: Annotated[ctypes.c_uint32, 328]
  reserved_83: Annotated[ctypes.c_uint32, 332]
  reserved_84: Annotated[ctypes.c_uint32, 336]
  reserved_85: Annotated[ctypes.c_uint32, 340]
  cp_mqd_query_time_lo: Annotated[ctypes.c_uint32, 344]
  cp_mqd_query_time_hi: Annotated[ctypes.c_uint32, 348]
  cp_mqd_connect_start_time_lo: Annotated[ctypes.c_uint32, 352]
  cp_mqd_connect_start_time_hi: Annotated[ctypes.c_uint32, 356]
  cp_mqd_connect_end_time_lo: Annotated[ctypes.c_uint32, 360]
  cp_mqd_connect_end_time_hi: Annotated[ctypes.c_uint32, 364]
  cp_mqd_connect_end_wf_count: Annotated[ctypes.c_uint32, 368]
  cp_mqd_connect_end_pq_rptr: Annotated[ctypes.c_uint32, 372]
  cp_mqd_connect_end_pq_wptr: Annotated[ctypes.c_uint32, 376]
  cp_mqd_connect_end_ib_rptr: Annotated[ctypes.c_uint32, 380]
  cp_mqd_readindex_lo: Annotated[ctypes.c_uint32, 384]
  cp_mqd_readindex_hi: Annotated[ctypes.c_uint32, 388]
  cp_mqd_save_start_time_lo: Annotated[ctypes.c_uint32, 392]
  cp_mqd_save_start_time_hi: Annotated[ctypes.c_uint32, 396]
  cp_mqd_save_end_time_lo: Annotated[ctypes.c_uint32, 400]
  cp_mqd_save_end_time_hi: Annotated[ctypes.c_uint32, 404]
  cp_mqd_restore_start_time_lo: Annotated[ctypes.c_uint32, 408]
  cp_mqd_restore_start_time_hi: Annotated[ctypes.c_uint32, 412]
  cp_mqd_restore_end_time_lo: Annotated[ctypes.c_uint32, 416]
  cp_mqd_restore_end_time_hi: Annotated[ctypes.c_uint32, 420]
  disable_queue: Annotated[ctypes.c_uint32, 424]
  reserved_107: Annotated[ctypes.c_uint32, 428]
  gds_cs_ctxsw_cnt0: Annotated[ctypes.c_uint32, 432]
  gds_cs_ctxsw_cnt1: Annotated[ctypes.c_uint32, 436]
  gds_cs_ctxsw_cnt2: Annotated[ctypes.c_uint32, 440]
  gds_cs_ctxsw_cnt3: Annotated[ctypes.c_uint32, 444]
  reserved_112: Annotated[ctypes.c_uint32, 448]
  reserved_113: Annotated[ctypes.c_uint32, 452]
  cp_pq_exe_status_lo: Annotated[ctypes.c_uint32, 456]
  cp_pq_exe_status_hi: Annotated[ctypes.c_uint32, 460]
  cp_packet_id_lo: Annotated[ctypes.c_uint32, 464]
  cp_packet_id_hi: Annotated[ctypes.c_uint32, 468]
  cp_packet_exe_status_lo: Annotated[ctypes.c_uint32, 472]
  cp_packet_exe_status_hi: Annotated[ctypes.c_uint32, 476]
  gds_save_base_addr_lo: Annotated[ctypes.c_uint32, 480]
  gds_save_base_addr_hi: Annotated[ctypes.c_uint32, 484]
  gds_save_mask_lo: Annotated[ctypes.c_uint32, 488]
  gds_save_mask_hi: Annotated[ctypes.c_uint32, 492]
  ctx_save_base_addr_lo: Annotated[ctypes.c_uint32, 496]
  ctx_save_base_addr_hi: Annotated[ctypes.c_uint32, 500]
  reserved_126: Annotated[ctypes.c_uint32, 504]
  reserved_127: Annotated[ctypes.c_uint32, 508]
  cp_mqd_base_addr_lo: Annotated[ctypes.c_uint32, 512]
  cp_mqd_base_addr_hi: Annotated[ctypes.c_uint32, 516]
  cp_hqd_active: Annotated[ctypes.c_uint32, 520]
  cp_hqd_vmid: Annotated[ctypes.c_uint32, 524]
  cp_hqd_persistent_state: Annotated[ctypes.c_uint32, 528]
  cp_hqd_pipe_priority: Annotated[ctypes.c_uint32, 532]
  cp_hqd_queue_priority: Annotated[ctypes.c_uint32, 536]
  cp_hqd_quantum: Annotated[ctypes.c_uint32, 540]
  cp_hqd_pq_base_lo: Annotated[ctypes.c_uint32, 544]
  cp_hqd_pq_base_hi: Annotated[ctypes.c_uint32, 548]
  cp_hqd_pq_rptr: Annotated[ctypes.c_uint32, 552]
  cp_hqd_pq_rptr_report_addr_lo: Annotated[ctypes.c_uint32, 556]
  cp_hqd_pq_rptr_report_addr_hi: Annotated[ctypes.c_uint32, 560]
  cp_hqd_pq_wptr_poll_addr_lo: Annotated[ctypes.c_uint32, 564]
  cp_hqd_pq_wptr_poll_addr_hi: Annotated[ctypes.c_uint32, 568]
  cp_hqd_pq_doorbell_control: Annotated[ctypes.c_uint32, 572]
  reserved_144: Annotated[ctypes.c_uint32, 576]
  cp_hqd_pq_control: Annotated[ctypes.c_uint32, 580]
  cp_hqd_ib_base_addr_lo: Annotated[ctypes.c_uint32, 584]
  cp_hqd_ib_base_addr_hi: Annotated[ctypes.c_uint32, 588]
  cp_hqd_ib_rptr: Annotated[ctypes.c_uint32, 592]
  cp_hqd_ib_control: Annotated[ctypes.c_uint32, 596]
  cp_hqd_iq_timer: Annotated[ctypes.c_uint32, 600]
  cp_hqd_iq_rptr: Annotated[ctypes.c_uint32, 604]
  cp_hqd_dequeue_request: Annotated[ctypes.c_uint32, 608]
  cp_hqd_dma_offload: Annotated[ctypes.c_uint32, 612]
  cp_hqd_sema_cmd: Annotated[ctypes.c_uint32, 616]
  cp_hqd_msg_type: Annotated[ctypes.c_uint32, 620]
  cp_hqd_atomic0_preop_lo: Annotated[ctypes.c_uint32, 624]
  cp_hqd_atomic0_preop_hi: Annotated[ctypes.c_uint32, 628]
  cp_hqd_atomic1_preop_lo: Annotated[ctypes.c_uint32, 632]
  cp_hqd_atomic1_preop_hi: Annotated[ctypes.c_uint32, 636]
  cp_hqd_hq_status0: Annotated[ctypes.c_uint32, 640]
  cp_hqd_hq_control0: Annotated[ctypes.c_uint32, 644]
  cp_mqd_control: Annotated[ctypes.c_uint32, 648]
  cp_hqd_hq_status1: Annotated[ctypes.c_uint32, 652]
  cp_hqd_hq_control1: Annotated[ctypes.c_uint32, 656]
  cp_hqd_eop_base_addr_lo: Annotated[ctypes.c_uint32, 660]
  cp_hqd_eop_base_addr_hi: Annotated[ctypes.c_uint32, 664]
  cp_hqd_eop_control: Annotated[ctypes.c_uint32, 668]
  cp_hqd_eop_rptr: Annotated[ctypes.c_uint32, 672]
  cp_hqd_eop_wptr: Annotated[ctypes.c_uint32, 676]
  cp_hqd_eop_done_events: Annotated[ctypes.c_uint32, 680]
  cp_hqd_ctx_save_base_addr_lo: Annotated[ctypes.c_uint32, 684]
  cp_hqd_ctx_save_base_addr_hi: Annotated[ctypes.c_uint32, 688]
  cp_hqd_ctx_save_control: Annotated[ctypes.c_uint32, 692]
  cp_hqd_cntl_stack_offset: Annotated[ctypes.c_uint32, 696]
  cp_hqd_cntl_stack_size: Annotated[ctypes.c_uint32, 700]
  cp_hqd_wg_state_offset: Annotated[ctypes.c_uint32, 704]
  cp_hqd_ctx_save_size: Annotated[ctypes.c_uint32, 708]
  cp_hqd_gds_resource_state: Annotated[ctypes.c_uint32, 712]
  cp_hqd_error: Annotated[ctypes.c_uint32, 716]
  cp_hqd_eop_wptr_mem: Annotated[ctypes.c_uint32, 720]
  cp_hqd_aql_control: Annotated[ctypes.c_uint32, 724]
  cp_hqd_pq_wptr_lo: Annotated[ctypes.c_uint32, 728]
  cp_hqd_pq_wptr_hi: Annotated[ctypes.c_uint32, 732]
  reserved_184: Annotated[ctypes.c_uint32, 736]
  reserved_185: Annotated[ctypes.c_uint32, 740]
  reserved_186: Annotated[ctypes.c_uint32, 744]
  reserved_187: Annotated[ctypes.c_uint32, 748]
  reserved_188: Annotated[ctypes.c_uint32, 752]
  reserved_189: Annotated[ctypes.c_uint32, 756]
  reserved_190: Annotated[ctypes.c_uint32, 760]
  reserved_191: Annotated[ctypes.c_uint32, 764]
  iqtimer_pkt_header: Annotated[ctypes.c_uint32, 768]
  iqtimer_pkt_dw0: Annotated[ctypes.c_uint32, 772]
  iqtimer_pkt_dw1: Annotated[ctypes.c_uint32, 776]
  iqtimer_pkt_dw2: Annotated[ctypes.c_uint32, 780]
  iqtimer_pkt_dw3: Annotated[ctypes.c_uint32, 784]
  iqtimer_pkt_dw4: Annotated[ctypes.c_uint32, 788]
  iqtimer_pkt_dw5: Annotated[ctypes.c_uint32, 792]
  iqtimer_pkt_dw6: Annotated[ctypes.c_uint32, 796]
  iqtimer_pkt_dw7: Annotated[ctypes.c_uint32, 800]
  iqtimer_pkt_dw8: Annotated[ctypes.c_uint32, 804]
  iqtimer_pkt_dw9: Annotated[ctypes.c_uint32, 808]
  iqtimer_pkt_dw10: Annotated[ctypes.c_uint32, 812]
  iqtimer_pkt_dw11: Annotated[ctypes.c_uint32, 816]
  iqtimer_pkt_dw12: Annotated[ctypes.c_uint32, 820]
  iqtimer_pkt_dw13: Annotated[ctypes.c_uint32, 824]
  iqtimer_pkt_dw14: Annotated[ctypes.c_uint32, 828]
  iqtimer_pkt_dw15: Annotated[ctypes.c_uint32, 832]
  iqtimer_pkt_dw16: Annotated[ctypes.c_uint32, 836]
  iqtimer_pkt_dw17: Annotated[ctypes.c_uint32, 840]
  iqtimer_pkt_dw18: Annotated[ctypes.c_uint32, 844]
  iqtimer_pkt_dw19: Annotated[ctypes.c_uint32, 848]
  iqtimer_pkt_dw20: Annotated[ctypes.c_uint32, 852]
  iqtimer_pkt_dw21: Annotated[ctypes.c_uint32, 856]
  iqtimer_pkt_dw22: Annotated[ctypes.c_uint32, 860]
  iqtimer_pkt_dw23: Annotated[ctypes.c_uint32, 864]
  iqtimer_pkt_dw24: Annotated[ctypes.c_uint32, 868]
  iqtimer_pkt_dw25: Annotated[ctypes.c_uint32, 872]
  iqtimer_pkt_dw26: Annotated[ctypes.c_uint32, 876]
  iqtimer_pkt_dw27: Annotated[ctypes.c_uint32, 880]
  iqtimer_pkt_dw28: Annotated[ctypes.c_uint32, 884]
  iqtimer_pkt_dw29: Annotated[ctypes.c_uint32, 888]
  iqtimer_pkt_dw30: Annotated[ctypes.c_uint32, 892]
  iqtimer_pkt_dw31: Annotated[ctypes.c_uint32, 896]
  reserved_225: Annotated[ctypes.c_uint32, 900]
  reserved_226: Annotated[ctypes.c_uint32, 904]
  reserved_227: Annotated[ctypes.c_uint32, 908]
  set_resources_header: Annotated[ctypes.c_uint32, 912]
  set_resources_dw1: Annotated[ctypes.c_uint32, 916]
  set_resources_dw2: Annotated[ctypes.c_uint32, 920]
  set_resources_dw3: Annotated[ctypes.c_uint32, 924]
  set_resources_dw4: Annotated[ctypes.c_uint32, 928]
  set_resources_dw5: Annotated[ctypes.c_uint32, 932]
  set_resources_dw6: Annotated[ctypes.c_uint32, 936]
  set_resources_dw7: Annotated[ctypes.c_uint32, 940]
  reserved_236: Annotated[ctypes.c_uint32, 944]
  reserved_237: Annotated[ctypes.c_uint32, 948]
  reserved_238: Annotated[ctypes.c_uint32, 952]
  reserved_239: Annotated[ctypes.c_uint32, 956]
  queue_doorbell_id0: Annotated[ctypes.c_uint32, 960]
  queue_doorbell_id1: Annotated[ctypes.c_uint32, 964]
  queue_doorbell_id2: Annotated[ctypes.c_uint32, 968]
  queue_doorbell_id3: Annotated[ctypes.c_uint32, 972]
  queue_doorbell_id4: Annotated[ctypes.c_uint32, 976]
  queue_doorbell_id5: Annotated[ctypes.c_uint32, 980]
  queue_doorbell_id6: Annotated[ctypes.c_uint32, 984]
  queue_doorbell_id7: Annotated[ctypes.c_uint32, 988]
  queue_doorbell_id8: Annotated[ctypes.c_uint32, 992]
  queue_doorbell_id9: Annotated[ctypes.c_uint32, 996]
  queue_doorbell_id10: Annotated[ctypes.c_uint32, 1000]
  queue_doorbell_id11: Annotated[ctypes.c_uint32, 1004]
  queue_doorbell_id12: Annotated[ctypes.c_uint32, 1008]
  queue_doorbell_id13: Annotated[ctypes.c_uint32, 1012]
  queue_doorbell_id14: Annotated[ctypes.c_uint32, 1016]
  queue_doorbell_id15: Annotated[ctypes.c_uint32, 1020]
  control_buf_addr_lo: Annotated[ctypes.c_uint32, 1024]
  control_buf_addr_hi: Annotated[ctypes.c_uint32, 1028]
  control_buf_wptr_lo: Annotated[ctypes.c_uint32, 1032]
  control_buf_wptr_hi: Annotated[ctypes.c_uint32, 1036]
  control_buf_dptr_lo: Annotated[ctypes.c_uint32, 1040]
  control_buf_dptr_hi: Annotated[ctypes.c_uint32, 1044]
  control_buf_num_entries: Annotated[ctypes.c_uint32, 1048]
  draw_ring_addr_lo: Annotated[ctypes.c_uint32, 1052]
  draw_ring_addr_hi: Annotated[ctypes.c_uint32, 1056]
  reserved_265: Annotated[ctypes.c_uint32, 1060]
  reserved_266: Annotated[ctypes.c_uint32, 1064]
  reserved_267: Annotated[ctypes.c_uint32, 1068]
  reserved_268: Annotated[ctypes.c_uint32, 1072]
  reserved_269: Annotated[ctypes.c_uint32, 1076]
  reserved_270: Annotated[ctypes.c_uint32, 1080]
  reserved_271: Annotated[ctypes.c_uint32, 1084]
  reserved_272: Annotated[ctypes.c_uint32, 1088]
  reserved_273: Annotated[ctypes.c_uint32, 1092]
  reserved_274: Annotated[ctypes.c_uint32, 1096]
  reserved_275: Annotated[ctypes.c_uint32, 1100]
  reserved_276: Annotated[ctypes.c_uint32, 1104]
  reserved_277: Annotated[ctypes.c_uint32, 1108]
  reserved_278: Annotated[ctypes.c_uint32, 1112]
  reserved_279: Annotated[ctypes.c_uint32, 1116]
  reserved_280: Annotated[ctypes.c_uint32, 1120]
  reserved_281: Annotated[ctypes.c_uint32, 1124]
  reserved_282: Annotated[ctypes.c_uint32, 1128]
  reserved_283: Annotated[ctypes.c_uint32, 1132]
  reserved_284: Annotated[ctypes.c_uint32, 1136]
  reserved_285: Annotated[ctypes.c_uint32, 1140]
  reserved_286: Annotated[ctypes.c_uint32, 1144]
  reserved_287: Annotated[ctypes.c_uint32, 1148]
  reserved_288: Annotated[ctypes.c_uint32, 1152]
  reserved_289: Annotated[ctypes.c_uint32, 1156]
  reserved_290: Annotated[ctypes.c_uint32, 1160]
  reserved_291: Annotated[ctypes.c_uint32, 1164]
  reserved_292: Annotated[ctypes.c_uint32, 1168]
  reserved_293: Annotated[ctypes.c_uint32, 1172]
  reserved_294: Annotated[ctypes.c_uint32, 1176]
  reserved_295: Annotated[ctypes.c_uint32, 1180]
  reserved_296: Annotated[ctypes.c_uint32, 1184]
  reserved_297: Annotated[ctypes.c_uint32, 1188]
  reserved_298: Annotated[ctypes.c_uint32, 1192]
  reserved_299: Annotated[ctypes.c_uint32, 1196]
  reserved_300: Annotated[ctypes.c_uint32, 1200]
  reserved_301: Annotated[ctypes.c_uint32, 1204]
  reserved_302: Annotated[ctypes.c_uint32, 1208]
  reserved_303: Annotated[ctypes.c_uint32, 1212]
  reserved_304: Annotated[ctypes.c_uint32, 1216]
  reserved_305: Annotated[ctypes.c_uint32, 1220]
  reserved_306: Annotated[ctypes.c_uint32, 1224]
  reserved_307: Annotated[ctypes.c_uint32, 1228]
  reserved_308: Annotated[ctypes.c_uint32, 1232]
  reserved_309: Annotated[ctypes.c_uint32, 1236]
  reserved_310: Annotated[ctypes.c_uint32, 1240]
  reserved_311: Annotated[ctypes.c_uint32, 1244]
  reserved_312: Annotated[ctypes.c_uint32, 1248]
  reserved_313: Annotated[ctypes.c_uint32, 1252]
  reserved_314: Annotated[ctypes.c_uint32, 1256]
  reserved_315: Annotated[ctypes.c_uint32, 1260]
  reserved_316: Annotated[ctypes.c_uint32, 1264]
  reserved_317: Annotated[ctypes.c_uint32, 1268]
  reserved_318: Annotated[ctypes.c_uint32, 1272]
  reserved_319: Annotated[ctypes.c_uint32, 1276]
  reserved_320: Annotated[ctypes.c_uint32, 1280]
  reserved_321: Annotated[ctypes.c_uint32, 1284]
  reserved_322: Annotated[ctypes.c_uint32, 1288]
  reserved_323: Annotated[ctypes.c_uint32, 1292]
  reserved_324: Annotated[ctypes.c_uint32, 1296]
  reserved_325: Annotated[ctypes.c_uint32, 1300]
  reserved_326: Annotated[ctypes.c_uint32, 1304]
  reserved_327: Annotated[ctypes.c_uint32, 1308]
  reserved_328: Annotated[ctypes.c_uint32, 1312]
  reserved_329: Annotated[ctypes.c_uint32, 1316]
  reserved_330: Annotated[ctypes.c_uint32, 1320]
  reserved_331: Annotated[ctypes.c_uint32, 1324]
  reserved_332: Annotated[ctypes.c_uint32, 1328]
  reserved_333: Annotated[ctypes.c_uint32, 1332]
  reserved_334: Annotated[ctypes.c_uint32, 1336]
  reserved_335: Annotated[ctypes.c_uint32, 1340]
  reserved_336: Annotated[ctypes.c_uint32, 1344]
  reserved_337: Annotated[ctypes.c_uint32, 1348]
  reserved_338: Annotated[ctypes.c_uint32, 1352]
  reserved_339: Annotated[ctypes.c_uint32, 1356]
  reserved_340: Annotated[ctypes.c_uint32, 1360]
  reserved_341: Annotated[ctypes.c_uint32, 1364]
  reserved_342: Annotated[ctypes.c_uint32, 1368]
  reserved_343: Annotated[ctypes.c_uint32, 1372]
  reserved_344: Annotated[ctypes.c_uint32, 1376]
  reserved_345: Annotated[ctypes.c_uint32, 1380]
  reserved_346: Annotated[ctypes.c_uint32, 1384]
  reserved_347: Annotated[ctypes.c_uint32, 1388]
  reserved_348: Annotated[ctypes.c_uint32, 1392]
  reserved_349: Annotated[ctypes.c_uint32, 1396]
  reserved_350: Annotated[ctypes.c_uint32, 1400]
  reserved_351: Annotated[ctypes.c_uint32, 1404]
  reserved_352: Annotated[ctypes.c_uint32, 1408]
  reserved_353: Annotated[ctypes.c_uint32, 1412]
  reserved_354: Annotated[ctypes.c_uint32, 1416]
  reserved_355: Annotated[ctypes.c_uint32, 1420]
  reserved_356: Annotated[ctypes.c_uint32, 1424]
  reserved_357: Annotated[ctypes.c_uint32, 1428]
  reserved_358: Annotated[ctypes.c_uint32, 1432]
  reserved_359: Annotated[ctypes.c_uint32, 1436]
  reserved_360: Annotated[ctypes.c_uint32, 1440]
  reserved_361: Annotated[ctypes.c_uint32, 1444]
  reserved_362: Annotated[ctypes.c_uint32, 1448]
  reserved_363: Annotated[ctypes.c_uint32, 1452]
  reserved_364: Annotated[ctypes.c_uint32, 1456]
  reserved_365: Annotated[ctypes.c_uint32, 1460]
  reserved_366: Annotated[ctypes.c_uint32, 1464]
  reserved_367: Annotated[ctypes.c_uint32, 1468]
  reserved_368: Annotated[ctypes.c_uint32, 1472]
  reserved_369: Annotated[ctypes.c_uint32, 1476]
  reserved_370: Annotated[ctypes.c_uint32, 1480]
  reserved_371: Annotated[ctypes.c_uint32, 1484]
  reserved_372: Annotated[ctypes.c_uint32, 1488]
  reserved_373: Annotated[ctypes.c_uint32, 1492]
  reserved_374: Annotated[ctypes.c_uint32, 1496]
  reserved_375: Annotated[ctypes.c_uint32, 1500]
  reserved_376: Annotated[ctypes.c_uint32, 1504]
  reserved_377: Annotated[ctypes.c_uint32, 1508]
  reserved_378: Annotated[ctypes.c_uint32, 1512]
  reserved_379: Annotated[ctypes.c_uint32, 1516]
  reserved_380: Annotated[ctypes.c_uint32, 1520]
  reserved_381: Annotated[ctypes.c_uint32, 1524]
  reserved_382: Annotated[ctypes.c_uint32, 1528]
  reserved_383: Annotated[ctypes.c_uint32, 1532]
  reserved_384: Annotated[ctypes.c_uint32, 1536]
  reserved_385: Annotated[ctypes.c_uint32, 1540]
  reserved_386: Annotated[ctypes.c_uint32, 1544]
  reserved_387: Annotated[ctypes.c_uint32, 1548]
  reserved_388: Annotated[ctypes.c_uint32, 1552]
  reserved_389: Annotated[ctypes.c_uint32, 1556]
  reserved_390: Annotated[ctypes.c_uint32, 1560]
  reserved_391: Annotated[ctypes.c_uint32, 1564]
  reserved_392: Annotated[ctypes.c_uint32, 1568]
  reserved_393: Annotated[ctypes.c_uint32, 1572]
  reserved_394: Annotated[ctypes.c_uint32, 1576]
  reserved_395: Annotated[ctypes.c_uint32, 1580]
  reserved_396: Annotated[ctypes.c_uint32, 1584]
  reserved_397: Annotated[ctypes.c_uint32, 1588]
  reserved_398: Annotated[ctypes.c_uint32, 1592]
  reserved_399: Annotated[ctypes.c_uint32, 1596]
  reserved_400: Annotated[ctypes.c_uint32, 1600]
  reserved_401: Annotated[ctypes.c_uint32, 1604]
  reserved_402: Annotated[ctypes.c_uint32, 1608]
  reserved_403: Annotated[ctypes.c_uint32, 1612]
  reserved_404: Annotated[ctypes.c_uint32, 1616]
  reserved_405: Annotated[ctypes.c_uint32, 1620]
  reserved_406: Annotated[ctypes.c_uint32, 1624]
  reserved_407: Annotated[ctypes.c_uint32, 1628]
  reserved_408: Annotated[ctypes.c_uint32, 1632]
  reserved_409: Annotated[ctypes.c_uint32, 1636]
  reserved_410: Annotated[ctypes.c_uint32, 1640]
  reserved_411: Annotated[ctypes.c_uint32, 1644]
  reserved_412: Annotated[ctypes.c_uint32, 1648]
  reserved_413: Annotated[ctypes.c_uint32, 1652]
  reserved_414: Annotated[ctypes.c_uint32, 1656]
  reserved_415: Annotated[ctypes.c_uint32, 1660]
  reserved_416: Annotated[ctypes.c_uint32, 1664]
  reserved_417: Annotated[ctypes.c_uint32, 1668]
  reserved_418: Annotated[ctypes.c_uint32, 1672]
  reserved_419: Annotated[ctypes.c_uint32, 1676]
  reserved_420: Annotated[ctypes.c_uint32, 1680]
  reserved_421: Annotated[ctypes.c_uint32, 1684]
  reserved_422: Annotated[ctypes.c_uint32, 1688]
  reserved_423: Annotated[ctypes.c_uint32, 1692]
  reserved_424: Annotated[ctypes.c_uint32, 1696]
  reserved_425: Annotated[ctypes.c_uint32, 1700]
  reserved_426: Annotated[ctypes.c_uint32, 1704]
  reserved_427: Annotated[ctypes.c_uint32, 1708]
  reserved_428: Annotated[ctypes.c_uint32, 1712]
  reserved_429: Annotated[ctypes.c_uint32, 1716]
  reserved_430: Annotated[ctypes.c_uint32, 1720]
  reserved_431: Annotated[ctypes.c_uint32, 1724]
  reserved_432: Annotated[ctypes.c_uint32, 1728]
  reserved_433: Annotated[ctypes.c_uint32, 1732]
  reserved_434: Annotated[ctypes.c_uint32, 1736]
  reserved_435: Annotated[ctypes.c_uint32, 1740]
  reserved_436: Annotated[ctypes.c_uint32, 1744]
  reserved_437: Annotated[ctypes.c_uint32, 1748]
  reserved_438: Annotated[ctypes.c_uint32, 1752]
  reserved_439: Annotated[ctypes.c_uint32, 1756]
  reserved_440: Annotated[ctypes.c_uint32, 1760]
  reserved_441: Annotated[ctypes.c_uint32, 1764]
  reserved_442: Annotated[ctypes.c_uint32, 1768]
  reserved_443: Annotated[ctypes.c_uint32, 1772]
  reserved_444: Annotated[ctypes.c_uint32, 1776]
  reserved_445: Annotated[ctypes.c_uint32, 1780]
  reserved_446: Annotated[ctypes.c_uint32, 1784]
  reserved_447: Annotated[ctypes.c_uint32, 1788]
  gws_0_val: Annotated[ctypes.c_uint32, 1792]
  gws_1_val: Annotated[ctypes.c_uint32, 1796]
  gws_2_val: Annotated[ctypes.c_uint32, 1800]
  gws_3_val: Annotated[ctypes.c_uint32, 1804]
  gws_4_val: Annotated[ctypes.c_uint32, 1808]
  gws_5_val: Annotated[ctypes.c_uint32, 1812]
  gws_6_val: Annotated[ctypes.c_uint32, 1816]
  gws_7_val: Annotated[ctypes.c_uint32, 1820]
  gws_8_val: Annotated[ctypes.c_uint32, 1824]
  gws_9_val: Annotated[ctypes.c_uint32, 1828]
  gws_10_val: Annotated[ctypes.c_uint32, 1832]
  gws_11_val: Annotated[ctypes.c_uint32, 1836]
  gws_12_val: Annotated[ctypes.c_uint32, 1840]
  gws_13_val: Annotated[ctypes.c_uint32, 1844]
  gws_14_val: Annotated[ctypes.c_uint32, 1848]
  gws_15_val: Annotated[ctypes.c_uint32, 1852]
  gws_16_val: Annotated[ctypes.c_uint32, 1856]
  gws_17_val: Annotated[ctypes.c_uint32, 1860]
  gws_18_val: Annotated[ctypes.c_uint32, 1864]
  gws_19_val: Annotated[ctypes.c_uint32, 1868]
  gws_20_val: Annotated[ctypes.c_uint32, 1872]
  gws_21_val: Annotated[ctypes.c_uint32, 1876]
  gws_22_val: Annotated[ctypes.c_uint32, 1880]
  gws_23_val: Annotated[ctypes.c_uint32, 1884]
  gws_24_val: Annotated[ctypes.c_uint32, 1888]
  gws_25_val: Annotated[ctypes.c_uint32, 1892]
  gws_26_val: Annotated[ctypes.c_uint32, 1896]
  gws_27_val: Annotated[ctypes.c_uint32, 1900]
  gws_28_val: Annotated[ctypes.c_uint32, 1904]
  gws_29_val: Annotated[ctypes.c_uint32, 1908]
  gws_30_val: Annotated[ctypes.c_uint32, 1912]
  gws_31_val: Annotated[ctypes.c_uint32, 1916]
  gws_32_val: Annotated[ctypes.c_uint32, 1920]
  gws_33_val: Annotated[ctypes.c_uint32, 1924]
  gws_34_val: Annotated[ctypes.c_uint32, 1928]
  gws_35_val: Annotated[ctypes.c_uint32, 1932]
  gws_36_val: Annotated[ctypes.c_uint32, 1936]
  gws_37_val: Annotated[ctypes.c_uint32, 1940]
  gws_38_val: Annotated[ctypes.c_uint32, 1944]
  gws_39_val: Annotated[ctypes.c_uint32, 1948]
  gws_40_val: Annotated[ctypes.c_uint32, 1952]
  gws_41_val: Annotated[ctypes.c_uint32, 1956]
  gws_42_val: Annotated[ctypes.c_uint32, 1960]
  gws_43_val: Annotated[ctypes.c_uint32, 1964]
  gws_44_val: Annotated[ctypes.c_uint32, 1968]
  gws_45_val: Annotated[ctypes.c_uint32, 1972]
  gws_46_val: Annotated[ctypes.c_uint32, 1976]
  gws_47_val: Annotated[ctypes.c_uint32, 1980]
  gws_48_val: Annotated[ctypes.c_uint32, 1984]
  gws_49_val: Annotated[ctypes.c_uint32, 1988]
  gws_50_val: Annotated[ctypes.c_uint32, 1992]
  gws_51_val: Annotated[ctypes.c_uint32, 1996]
  gws_52_val: Annotated[ctypes.c_uint32, 2000]
  gws_53_val: Annotated[ctypes.c_uint32, 2004]
  gws_54_val: Annotated[ctypes.c_uint32, 2008]
  gws_55_val: Annotated[ctypes.c_uint32, 2012]
  gws_56_val: Annotated[ctypes.c_uint32, 2016]
  gws_57_val: Annotated[ctypes.c_uint32, 2020]
  gws_58_val: Annotated[ctypes.c_uint32, 2024]
  gws_59_val: Annotated[ctypes.c_uint32, 2028]
  gws_60_val: Annotated[ctypes.c_uint32, 2032]
  gws_61_val: Annotated[ctypes.c_uint32, 2036]
  gws_62_val: Annotated[ctypes.c_uint32, 2040]
  gws_63_val: Annotated[ctypes.c_uint32, 2044]
@record
class struct_v12_gfx_mqd:
  SIZE = 2048
  shadow_base_lo: Annotated[uint32_t, 0]
  shadow_base_hi: Annotated[uint32_t, 4]
  reserved_2: Annotated[uint32_t, 8]
  reserved_3: Annotated[uint32_t, 12]
  fw_work_area_base_lo: Annotated[uint32_t, 16]
  fw_work_area_base_hi: Annotated[uint32_t, 20]
  shadow_initialized: Annotated[uint32_t, 24]
  ib_vmid: Annotated[uint32_t, 28]
  reserved_8: Annotated[uint32_t, 32]
  reserved_9: Annotated[uint32_t, 36]
  reserved_10: Annotated[uint32_t, 40]
  reserved_11: Annotated[uint32_t, 44]
  reserved_12: Annotated[uint32_t, 48]
  reserved_13: Annotated[uint32_t, 52]
  reserved_14: Annotated[uint32_t, 56]
  reserved_15: Annotated[uint32_t, 60]
  reserved_16: Annotated[uint32_t, 64]
  reserved_17: Annotated[uint32_t, 68]
  reserved_18: Annotated[uint32_t, 72]
  reserved_19: Annotated[uint32_t, 76]
  reserved_20: Annotated[uint32_t, 80]
  reserved_21: Annotated[uint32_t, 84]
  reserved_22: Annotated[uint32_t, 88]
  reserved_23: Annotated[uint32_t, 92]
  reserved_24: Annotated[uint32_t, 96]
  reserved_25: Annotated[uint32_t, 100]
  reserved_26: Annotated[uint32_t, 104]
  reserved_27: Annotated[uint32_t, 108]
  reserved_28: Annotated[uint32_t, 112]
  reserved_29: Annotated[uint32_t, 116]
  reserved_30: Annotated[uint32_t, 120]
  reserved_31: Annotated[uint32_t, 124]
  reserved_32: Annotated[uint32_t, 128]
  reserved_33: Annotated[uint32_t, 132]
  reserved_34: Annotated[uint32_t, 136]
  reserved_35: Annotated[uint32_t, 140]
  reserved_36: Annotated[uint32_t, 144]
  reserved_37: Annotated[uint32_t, 148]
  reserved_38: Annotated[uint32_t, 152]
  reserved_39: Annotated[uint32_t, 156]
  reserved_40: Annotated[uint32_t, 160]
  reserved_41: Annotated[uint32_t, 164]
  reserved_42: Annotated[uint32_t, 168]
  reserved_43: Annotated[uint32_t, 172]
  reserved_44: Annotated[uint32_t, 176]
  reserved_45: Annotated[uint32_t, 180]
  reserved_46: Annotated[uint32_t, 184]
  reserved_47: Annotated[uint32_t, 188]
  reserved_48: Annotated[uint32_t, 192]
  reserved_49: Annotated[uint32_t, 196]
  reserved_50: Annotated[uint32_t, 200]
  reserved_51: Annotated[uint32_t, 204]
  reserved_52: Annotated[uint32_t, 208]
  reserved_53: Annotated[uint32_t, 212]
  reserved_54: Annotated[uint32_t, 216]
  reserved_55: Annotated[uint32_t, 220]
  reserved_56: Annotated[uint32_t, 224]
  reserved_57: Annotated[uint32_t, 228]
  reserved_58: Annotated[uint32_t, 232]
  reserved_59: Annotated[uint32_t, 236]
  reserved_60: Annotated[uint32_t, 240]
  reserved_61: Annotated[uint32_t, 244]
  reserved_62: Annotated[uint32_t, 248]
  reserved_63: Annotated[uint32_t, 252]
  reserved_64: Annotated[uint32_t, 256]
  reserved_65: Annotated[uint32_t, 260]
  reserved_66: Annotated[uint32_t, 264]
  reserved_67: Annotated[uint32_t, 268]
  reserved_68: Annotated[uint32_t, 272]
  reserved_69: Annotated[uint32_t, 276]
  reserved_70: Annotated[uint32_t, 280]
  reserved_71: Annotated[uint32_t, 284]
  reserved_72: Annotated[uint32_t, 288]
  reserved_73: Annotated[uint32_t, 292]
  reserved_74: Annotated[uint32_t, 296]
  reserved_75: Annotated[uint32_t, 300]
  reserved_76: Annotated[uint32_t, 304]
  reserved_77: Annotated[uint32_t, 308]
  reserved_78: Annotated[uint32_t, 312]
  reserved_79: Annotated[uint32_t, 316]
  reserved_80: Annotated[uint32_t, 320]
  reserved_81: Annotated[uint32_t, 324]
  reserved_82: Annotated[uint32_t, 328]
  reserved_83: Annotated[uint32_t, 332]
  checksum_lo: Annotated[uint32_t, 336]
  checksum_hi: Annotated[uint32_t, 340]
  cp_mqd_query_time_lo: Annotated[uint32_t, 344]
  cp_mqd_query_time_hi: Annotated[uint32_t, 348]
  reserved_88: Annotated[uint32_t, 352]
  reserved_89: Annotated[uint32_t, 356]
  reserved_90: Annotated[uint32_t, 360]
  reserved_91: Annotated[uint32_t, 364]
  cp_mqd_query_wave_count: Annotated[uint32_t, 368]
  cp_mqd_query_gfx_hqd_rptr: Annotated[uint32_t, 372]
  cp_mqd_query_gfx_hqd_wptr: Annotated[uint32_t, 376]
  cp_mqd_query_gfx_hqd_offset: Annotated[uint32_t, 380]
  reserved_96: Annotated[uint32_t, 384]
  reserved_97: Annotated[uint32_t, 388]
  reserved_98: Annotated[uint32_t, 392]
  reserved_99: Annotated[uint32_t, 396]
  reserved_100: Annotated[uint32_t, 400]
  reserved_101: Annotated[uint32_t, 404]
  reserved_102: Annotated[uint32_t, 408]
  reserved_103: Annotated[uint32_t, 412]
  task_shader_control_buf_addr_lo: Annotated[uint32_t, 416]
  task_shader_control_buf_addr_hi: Annotated[uint32_t, 420]
  task_shader_read_rptr_lo: Annotated[uint32_t, 424]
  task_shader_read_rptr_hi: Annotated[uint32_t, 428]
  task_shader_num_entries: Annotated[uint32_t, 432]
  task_shader_num_entries_bits: Annotated[uint32_t, 436]
  task_shader_ring_buffer_addr_lo: Annotated[uint32_t, 440]
  task_shader_ring_buffer_addr_hi: Annotated[uint32_t, 444]
  reserved_112: Annotated[uint32_t, 448]
  reserved_113: Annotated[uint32_t, 452]
  reserved_114: Annotated[uint32_t, 456]
  reserved_115: Annotated[uint32_t, 460]
  reserved_116: Annotated[uint32_t, 464]
  reserved_117: Annotated[uint32_t, 468]
  reserved_118: Annotated[uint32_t, 472]
  reserved_119: Annotated[uint32_t, 476]
  reserved_120: Annotated[uint32_t, 480]
  reserved_121: Annotated[uint32_t, 484]
  reserved_122: Annotated[uint32_t, 488]
  reserved_123: Annotated[uint32_t, 492]
  reserved_124: Annotated[uint32_t, 496]
  reserved_125: Annotated[uint32_t, 500]
  reserved_126: Annotated[uint32_t, 504]
  reserved_127: Annotated[uint32_t, 508]
  cp_mqd_base_addr: Annotated[uint32_t, 512]
  cp_mqd_base_addr_hi: Annotated[uint32_t, 516]
  cp_gfx_hqd_active: Annotated[uint32_t, 520]
  cp_gfx_hqd_vmid: Annotated[uint32_t, 524]
  reserved_132: Annotated[uint32_t, 528]
  reserved_133: Annotated[uint32_t, 532]
  cp_gfx_hqd_queue_priority: Annotated[uint32_t, 536]
  cp_gfx_hqd_quantum: Annotated[uint32_t, 540]
  cp_gfx_hqd_base: Annotated[uint32_t, 544]
  cp_gfx_hqd_base_hi: Annotated[uint32_t, 548]
  cp_gfx_hqd_rptr: Annotated[uint32_t, 552]
  cp_gfx_hqd_rptr_addr: Annotated[uint32_t, 556]
  cp_gfx_hqd_rptr_addr_hi: Annotated[uint32_t, 560]
  cp_rb_wptr_poll_addr_lo: Annotated[uint32_t, 564]
  cp_rb_wptr_poll_addr_hi: Annotated[uint32_t, 568]
  cp_rb_doorbell_control: Annotated[uint32_t, 572]
  cp_gfx_hqd_offset: Annotated[uint32_t, 576]
  cp_gfx_hqd_cntl: Annotated[uint32_t, 580]
  reserved_146: Annotated[uint32_t, 584]
  reserved_147: Annotated[uint32_t, 588]
  cp_gfx_hqd_csmd_rptr: Annotated[uint32_t, 592]
  cp_gfx_hqd_wptr: Annotated[uint32_t, 596]
  cp_gfx_hqd_wptr_hi: Annotated[uint32_t, 600]
  reserved_151: Annotated[uint32_t, 604]
  reserved_152: Annotated[uint32_t, 608]
  reserved_153: Annotated[uint32_t, 612]
  reserved_154: Annotated[uint32_t, 616]
  reserved_155: Annotated[uint32_t, 620]
  cp_gfx_hqd_mapped: Annotated[uint32_t, 624]
  cp_gfx_hqd_que_mgr_control: Annotated[uint32_t, 628]
  reserved_158: Annotated[uint32_t, 632]
  reserved_159: Annotated[uint32_t, 636]
  cp_gfx_hqd_hq_status0: Annotated[uint32_t, 640]
  cp_gfx_hqd_hq_control0: Annotated[uint32_t, 644]
  cp_gfx_mqd_control: Annotated[uint32_t, 648]
  reserved_163: Annotated[uint32_t, 652]
  reserved_164: Annotated[uint32_t, 656]
  reserved_165: Annotated[uint32_t, 660]
  reserved_166: Annotated[uint32_t, 664]
  reserved_167: Annotated[uint32_t, 668]
  reserved_168: Annotated[uint32_t, 672]
  reserved_169: Annotated[uint32_t, 676]
  reserved_170: Annotated[uint32_t, 680]
  reserved_171: Annotated[uint32_t, 684]
  reserved_172: Annotated[uint32_t, 688]
  reserved_173: Annotated[uint32_t, 692]
  reserved_174: Annotated[uint32_t, 696]
  reserved_175: Annotated[uint32_t, 700]
  reserved_176: Annotated[uint32_t, 704]
  reserved_177: Annotated[uint32_t, 708]
  reserved_178: Annotated[uint32_t, 712]
  reserved_179: Annotated[uint32_t, 716]
  reserved_180: Annotated[uint32_t, 720]
  reserved_181: Annotated[uint32_t, 724]
  reserved_182: Annotated[uint32_t, 728]
  reserved_183: Annotated[uint32_t, 732]
  reserved_184: Annotated[uint32_t, 736]
  reserved_185: Annotated[uint32_t, 740]
  reserved_186: Annotated[uint32_t, 744]
  reserved_187: Annotated[uint32_t, 748]
  reserved_188: Annotated[uint32_t, 752]
  reserved_189: Annotated[uint32_t, 756]
  reserved_190: Annotated[uint32_t, 760]
  reserved_191: Annotated[uint32_t, 764]
  reserved_192: Annotated[uint32_t, 768]
  reserved_193: Annotated[uint32_t, 772]
  reserved_194: Annotated[uint32_t, 776]
  reserved_195: Annotated[uint32_t, 780]
  reserved_196: Annotated[uint32_t, 784]
  reserved_197: Annotated[uint32_t, 788]
  reserved_198: Annotated[uint32_t, 792]
  reserved_199: Annotated[uint32_t, 796]
  reserved_200: Annotated[uint32_t, 800]
  reserved_201: Annotated[uint32_t, 804]
  reserved_202: Annotated[uint32_t, 808]
  reserved_203: Annotated[uint32_t, 812]
  reserved_204: Annotated[uint32_t, 816]
  reserved_205: Annotated[uint32_t, 820]
  reserved_206: Annotated[uint32_t, 824]
  reserved_207: Annotated[uint32_t, 828]
  reserved_208: Annotated[uint32_t, 832]
  reserved_209: Annotated[uint32_t, 836]
  reserved_210: Annotated[uint32_t, 840]
  reserved_211: Annotated[uint32_t, 844]
  reserved_212: Annotated[uint32_t, 848]
  reserved_213: Annotated[uint32_t, 852]
  reserved_214: Annotated[uint32_t, 856]
  reserved_215: Annotated[uint32_t, 860]
  reserved_216: Annotated[uint32_t, 864]
  reserved_217: Annotated[uint32_t, 868]
  reserved_218: Annotated[uint32_t, 872]
  reserved_219: Annotated[uint32_t, 876]
  reserved_220: Annotated[uint32_t, 880]
  reserved_221: Annotated[uint32_t, 884]
  reserved_222: Annotated[uint32_t, 888]
  reserved_223: Annotated[uint32_t, 892]
  reserved_224: Annotated[uint32_t, 896]
  reserved_225: Annotated[uint32_t, 900]
  reserved_226: Annotated[uint32_t, 904]
  reserved_227: Annotated[uint32_t, 908]
  reserved_228: Annotated[uint32_t, 912]
  reserved_229: Annotated[uint32_t, 916]
  reserved_230: Annotated[uint32_t, 920]
  reserved_231: Annotated[uint32_t, 924]
  reserved_232: Annotated[uint32_t, 928]
  reserved_233: Annotated[uint32_t, 932]
  reserved_234: Annotated[uint32_t, 936]
  reserved_235: Annotated[uint32_t, 940]
  reserved_236: Annotated[uint32_t, 944]
  reserved_237: Annotated[uint32_t, 948]
  reserved_238: Annotated[uint32_t, 952]
  reserved_239: Annotated[uint32_t, 956]
  reserved_240: Annotated[uint32_t, 960]
  reserved_241: Annotated[uint32_t, 964]
  reserved_242: Annotated[uint32_t, 968]
  reserved_243: Annotated[uint32_t, 972]
  reserved_244: Annotated[uint32_t, 976]
  reserved_245: Annotated[uint32_t, 980]
  reserved_246: Annotated[uint32_t, 984]
  reserved_247: Annotated[uint32_t, 988]
  reserved_248: Annotated[uint32_t, 992]
  reserved_249: Annotated[uint32_t, 996]
  reserved_250: Annotated[uint32_t, 1000]
  reserved_251: Annotated[uint32_t, 1004]
  reserved_252: Annotated[uint32_t, 1008]
  reserved_253: Annotated[uint32_t, 1012]
  reserved_254: Annotated[uint32_t, 1016]
  reserved_255: Annotated[uint32_t, 1020]
  reserved_256: Annotated[uint32_t, 1024]
  reserved_257: Annotated[uint32_t, 1028]
  reserved_258: Annotated[uint32_t, 1032]
  reserved_259: Annotated[uint32_t, 1036]
  reserved_260: Annotated[uint32_t, 1040]
  reserved_261: Annotated[uint32_t, 1044]
  reserved_262: Annotated[uint32_t, 1048]
  reserved_263: Annotated[uint32_t, 1052]
  reserved_264: Annotated[uint32_t, 1056]
  reserved_265: Annotated[uint32_t, 1060]
  reserved_266: Annotated[uint32_t, 1064]
  reserved_267: Annotated[uint32_t, 1068]
  reserved_268: Annotated[uint32_t, 1072]
  reserved_269: Annotated[uint32_t, 1076]
  reserved_270: Annotated[uint32_t, 1080]
  reserved_271: Annotated[uint32_t, 1084]
  dfwx_flags: Annotated[uint32_t, 1088]
  dfwx_slot: Annotated[uint32_t, 1092]
  dfwx_client_data_addr_lo: Annotated[uint32_t, 1096]
  dfwx_client_data_addr_hi: Annotated[uint32_t, 1100]
  reserved_276: Annotated[uint32_t, 1104]
  reserved_277: Annotated[uint32_t, 1108]
  reserved_278: Annotated[uint32_t, 1112]
  reserved_279: Annotated[uint32_t, 1116]
  reserved_280: Annotated[uint32_t, 1120]
  reserved_281: Annotated[uint32_t, 1124]
  reserved_282: Annotated[uint32_t, 1128]
  reserved_283: Annotated[uint32_t, 1132]
  reserved_284: Annotated[uint32_t, 1136]
  reserved_285: Annotated[uint32_t, 1140]
  reserved_286: Annotated[uint32_t, 1144]
  reserved_287: Annotated[uint32_t, 1148]
  reserved_288: Annotated[uint32_t, 1152]
  reserved_289: Annotated[uint32_t, 1156]
  reserved_290: Annotated[uint32_t, 1160]
  reserved_291: Annotated[uint32_t, 1164]
  reserved_292: Annotated[uint32_t, 1168]
  reserved_293: Annotated[uint32_t, 1172]
  reserved_294: Annotated[uint32_t, 1176]
  reserved_295: Annotated[uint32_t, 1180]
  reserved_296: Annotated[uint32_t, 1184]
  reserved_297: Annotated[uint32_t, 1188]
  reserved_298: Annotated[uint32_t, 1192]
  reserved_299: Annotated[uint32_t, 1196]
  reserved_300: Annotated[uint32_t, 1200]
  reserved_301: Annotated[uint32_t, 1204]
  reserved_302: Annotated[uint32_t, 1208]
  reserved_303: Annotated[uint32_t, 1212]
  reserved_304: Annotated[uint32_t, 1216]
  reserved_305: Annotated[uint32_t, 1220]
  reserved_306: Annotated[uint32_t, 1224]
  reserved_307: Annotated[uint32_t, 1228]
  reserved_308: Annotated[uint32_t, 1232]
  reserved_309: Annotated[uint32_t, 1236]
  reserved_310: Annotated[uint32_t, 1240]
  reserved_311: Annotated[uint32_t, 1244]
  reserved_312: Annotated[uint32_t, 1248]
  reserved_313: Annotated[uint32_t, 1252]
  reserved_314: Annotated[uint32_t, 1256]
  reserved_315: Annotated[uint32_t, 1260]
  reserved_316: Annotated[uint32_t, 1264]
  reserved_317: Annotated[uint32_t, 1268]
  reserved_318: Annotated[uint32_t, 1272]
  reserved_319: Annotated[uint32_t, 1276]
  reserved_320: Annotated[uint32_t, 1280]
  reserved_321: Annotated[uint32_t, 1284]
  reserved_322: Annotated[uint32_t, 1288]
  reserved_323: Annotated[uint32_t, 1292]
  reserved_324: Annotated[uint32_t, 1296]
  reserved_325: Annotated[uint32_t, 1300]
  reserved_326: Annotated[uint32_t, 1304]
  reserved_327: Annotated[uint32_t, 1308]
  reserved_328: Annotated[uint32_t, 1312]
  reserved_329: Annotated[uint32_t, 1316]
  reserved_330: Annotated[uint32_t, 1320]
  reserved_331: Annotated[uint32_t, 1324]
  reserved_332: Annotated[uint32_t, 1328]
  reserved_333: Annotated[uint32_t, 1332]
  reserved_334: Annotated[uint32_t, 1336]
  reserved_335: Annotated[uint32_t, 1340]
  reserved_336: Annotated[uint32_t, 1344]
  reserved_337: Annotated[uint32_t, 1348]
  reserved_338: Annotated[uint32_t, 1352]
  reserved_339: Annotated[uint32_t, 1356]
  reserved_340: Annotated[uint32_t, 1360]
  reserved_341: Annotated[uint32_t, 1364]
  reserved_342: Annotated[uint32_t, 1368]
  reserved_343: Annotated[uint32_t, 1372]
  reserved_344: Annotated[uint32_t, 1376]
  reserved_345: Annotated[uint32_t, 1380]
  reserved_346: Annotated[uint32_t, 1384]
  reserved_347: Annotated[uint32_t, 1388]
  reserved_348: Annotated[uint32_t, 1392]
  reserved_349: Annotated[uint32_t, 1396]
  reserved_350: Annotated[uint32_t, 1400]
  reserved_351: Annotated[uint32_t, 1404]
  reserved_352: Annotated[uint32_t, 1408]
  reserved_353: Annotated[uint32_t, 1412]
  reserved_354: Annotated[uint32_t, 1416]
  reserved_355: Annotated[uint32_t, 1420]
  reserved_356: Annotated[uint32_t, 1424]
  reserved_357: Annotated[uint32_t, 1428]
  reserved_358: Annotated[uint32_t, 1432]
  reserved_359: Annotated[uint32_t, 1436]
  reserved_360: Annotated[uint32_t, 1440]
  reserved_361: Annotated[uint32_t, 1444]
  reserved_362: Annotated[uint32_t, 1448]
  reserved_363: Annotated[uint32_t, 1452]
  reserved_364: Annotated[uint32_t, 1456]
  reserved_365: Annotated[uint32_t, 1460]
  reserved_366: Annotated[uint32_t, 1464]
  reserved_367: Annotated[uint32_t, 1468]
  reserved_368: Annotated[uint32_t, 1472]
  reserved_369: Annotated[uint32_t, 1476]
  reserved_370: Annotated[uint32_t, 1480]
  reserved_371: Annotated[uint32_t, 1484]
  reserved_372: Annotated[uint32_t, 1488]
  reserved_373: Annotated[uint32_t, 1492]
  reserved_374: Annotated[uint32_t, 1496]
  reserved_375: Annotated[uint32_t, 1500]
  reserved_376: Annotated[uint32_t, 1504]
  reserved_377: Annotated[uint32_t, 1508]
  reserved_378: Annotated[uint32_t, 1512]
  reserved_379: Annotated[uint32_t, 1516]
  reserved_380: Annotated[uint32_t, 1520]
  reserved_381: Annotated[uint32_t, 1524]
  reserved_382: Annotated[uint32_t, 1528]
  reserved_383: Annotated[uint32_t, 1532]
  reserved_384: Annotated[uint32_t, 1536]
  reserved_385: Annotated[uint32_t, 1540]
  reserved_386: Annotated[uint32_t, 1544]
  reserved_387: Annotated[uint32_t, 1548]
  reserved_388: Annotated[uint32_t, 1552]
  reserved_389: Annotated[uint32_t, 1556]
  reserved_390: Annotated[uint32_t, 1560]
  reserved_391: Annotated[uint32_t, 1564]
  reserved_392: Annotated[uint32_t, 1568]
  reserved_393: Annotated[uint32_t, 1572]
  reserved_394: Annotated[uint32_t, 1576]
  reserved_395: Annotated[uint32_t, 1580]
  reserved_396: Annotated[uint32_t, 1584]
  reserved_397: Annotated[uint32_t, 1588]
  reserved_398: Annotated[uint32_t, 1592]
  reserved_399: Annotated[uint32_t, 1596]
  reserved_400: Annotated[uint32_t, 1600]
  reserved_401: Annotated[uint32_t, 1604]
  reserved_402: Annotated[uint32_t, 1608]
  reserved_403: Annotated[uint32_t, 1612]
  reserved_404: Annotated[uint32_t, 1616]
  reserved_405: Annotated[uint32_t, 1620]
  reserved_406: Annotated[uint32_t, 1624]
  reserved_407: Annotated[uint32_t, 1628]
  reserved_408: Annotated[uint32_t, 1632]
  reserved_409: Annotated[uint32_t, 1636]
  reserved_410: Annotated[uint32_t, 1640]
  reserved_411: Annotated[uint32_t, 1644]
  reserved_412: Annotated[uint32_t, 1648]
  reserved_413: Annotated[uint32_t, 1652]
  reserved_414: Annotated[uint32_t, 1656]
  reserved_415: Annotated[uint32_t, 1660]
  reserved_416: Annotated[uint32_t, 1664]
  reserved_417: Annotated[uint32_t, 1668]
  reserved_418: Annotated[uint32_t, 1672]
  reserved_419: Annotated[uint32_t, 1676]
  reserved_420: Annotated[uint32_t, 1680]
  reserved_421: Annotated[uint32_t, 1684]
  reserved_422: Annotated[uint32_t, 1688]
  reserved_423: Annotated[uint32_t, 1692]
  reserved_424: Annotated[uint32_t, 1696]
  reserved_425: Annotated[uint32_t, 1700]
  reserved_426: Annotated[uint32_t, 1704]
  reserved_427: Annotated[uint32_t, 1708]
  reserved_428: Annotated[uint32_t, 1712]
  reserved_429: Annotated[uint32_t, 1716]
  reserved_430: Annotated[uint32_t, 1720]
  reserved_431: Annotated[uint32_t, 1724]
  reserved_432: Annotated[uint32_t, 1728]
  reserved_433: Annotated[uint32_t, 1732]
  reserved_434: Annotated[uint32_t, 1736]
  reserved_435: Annotated[uint32_t, 1740]
  reserved_436: Annotated[uint32_t, 1744]
  reserved_437: Annotated[uint32_t, 1748]
  reserved_438: Annotated[uint32_t, 1752]
  reserved_439: Annotated[uint32_t, 1756]
  reserved_440: Annotated[uint32_t, 1760]
  reserved_441: Annotated[uint32_t, 1764]
  reserved_442: Annotated[uint32_t, 1768]
  reserved_443: Annotated[uint32_t, 1772]
  reserved_444: Annotated[uint32_t, 1776]
  reserved_445: Annotated[uint32_t, 1780]
  reserved_446: Annotated[uint32_t, 1784]
  reserved_447: Annotated[uint32_t, 1788]
  reserved_448: Annotated[uint32_t, 1792]
  reserved_449: Annotated[uint32_t, 1796]
  reserved_450: Annotated[uint32_t, 1800]
  reserved_451: Annotated[uint32_t, 1804]
  reserved_452: Annotated[uint32_t, 1808]
  reserved_453: Annotated[uint32_t, 1812]
  reserved_454: Annotated[uint32_t, 1816]
  reserved_455: Annotated[uint32_t, 1820]
  reserved_456: Annotated[uint32_t, 1824]
  reserved_457: Annotated[uint32_t, 1828]
  reserved_458: Annotated[uint32_t, 1832]
  reserved_459: Annotated[uint32_t, 1836]
  reserved_460: Annotated[uint32_t, 1840]
  reserved_461: Annotated[uint32_t, 1844]
  reserved_462: Annotated[uint32_t, 1848]
  reserved_463: Annotated[uint32_t, 1852]
  reserved_464: Annotated[uint32_t, 1856]
  reserved_465: Annotated[uint32_t, 1860]
  reserved_466: Annotated[uint32_t, 1864]
  reserved_467: Annotated[uint32_t, 1868]
  reserved_468: Annotated[uint32_t, 1872]
  reserved_469: Annotated[uint32_t, 1876]
  reserved_470: Annotated[uint32_t, 1880]
  reserved_471: Annotated[uint32_t, 1884]
  reserved_472: Annotated[uint32_t, 1888]
  reserved_473: Annotated[uint32_t, 1892]
  reserved_474: Annotated[uint32_t, 1896]
  reserved_475: Annotated[uint32_t, 1900]
  reserved_476: Annotated[uint32_t, 1904]
  reserved_477: Annotated[uint32_t, 1908]
  reserved_478: Annotated[uint32_t, 1912]
  reserved_479: Annotated[uint32_t, 1916]
  reserved_480: Annotated[uint32_t, 1920]
  reserved_481: Annotated[uint32_t, 1924]
  reserved_482: Annotated[uint32_t, 1928]
  reserved_483: Annotated[uint32_t, 1932]
  reserved_484: Annotated[uint32_t, 1936]
  reserved_485: Annotated[uint32_t, 1940]
  reserved_486: Annotated[uint32_t, 1944]
  reserved_487: Annotated[uint32_t, 1948]
  reserved_488: Annotated[uint32_t, 1952]
  reserved_489: Annotated[uint32_t, 1956]
  reserved_490: Annotated[uint32_t, 1960]
  reserved_491: Annotated[uint32_t, 1964]
  reserved_492: Annotated[uint32_t, 1968]
  reserved_493: Annotated[uint32_t, 1972]
  reserved_494: Annotated[uint32_t, 1976]
  reserved_495: Annotated[uint32_t, 1980]
  reserved_496: Annotated[uint32_t, 1984]
  reserved_497: Annotated[uint32_t, 1988]
  reserved_498: Annotated[uint32_t, 1992]
  reserved_499: Annotated[uint32_t, 1996]
  reserved_500: Annotated[uint32_t, 2000]
  reserved_501: Annotated[uint32_t, 2004]
  reserved_502: Annotated[uint32_t, 2008]
  reserved_503: Annotated[uint32_t, 2012]
  reserved_504: Annotated[uint32_t, 2016]
  reserved_505: Annotated[uint32_t, 2020]
  reserved_506: Annotated[uint32_t, 2024]
  reserved_507: Annotated[uint32_t, 2028]
  reserved_508: Annotated[uint32_t, 2032]
  reserved_509: Annotated[uint32_t, 2036]
  reserved_510: Annotated[uint32_t, 2040]
  reserved_511: Annotated[uint32_t, 2044]
uint32_t = ctypes.c_uint32
@record
class struct_v12_sdma_mqd:
  SIZE = 512
  sdmax_rlcx_rb_cntl: Annotated[uint32_t, 0]
  sdmax_rlcx_rb_base: Annotated[uint32_t, 4]
  sdmax_rlcx_rb_base_hi: Annotated[uint32_t, 8]
  sdmax_rlcx_rb_rptr: Annotated[uint32_t, 12]
  sdmax_rlcx_rb_rptr_hi: Annotated[uint32_t, 16]
  sdmax_rlcx_rb_wptr: Annotated[uint32_t, 20]
  sdmax_rlcx_rb_wptr_hi: Annotated[uint32_t, 24]
  sdmax_rlcx_rb_rptr_addr_lo: Annotated[uint32_t, 28]
  sdmax_rlcx_rb_rptr_addr_hi: Annotated[uint32_t, 32]
  sdmax_rlcx_ib_cntl: Annotated[uint32_t, 36]
  sdmax_rlcx_ib_rptr: Annotated[uint32_t, 40]
  sdmax_rlcx_ib_offset: Annotated[uint32_t, 44]
  sdmax_rlcx_ib_base_lo: Annotated[uint32_t, 48]
  sdmax_rlcx_ib_base_hi: Annotated[uint32_t, 52]
  sdmax_rlcx_ib_size: Annotated[uint32_t, 56]
  sdmax_rlcx_doorbell: Annotated[uint32_t, 60]
  sdmax_rlcx_doorbell_log: Annotated[uint32_t, 64]
  sdmax_rlcx_doorbell_offset: Annotated[uint32_t, 68]
  sdmax_rlcx_csa_addr_lo: Annotated[uint32_t, 72]
  sdmax_rlcx_csa_addr_hi: Annotated[uint32_t, 76]
  sdmax_rlcx_sched_cntl: Annotated[uint32_t, 80]
  sdmax_rlcx_ib_sub_remain: Annotated[uint32_t, 84]
  sdmax_rlcx_preempt: Annotated[uint32_t, 88]
  sdmax_rlcx_dummy_reg: Annotated[uint32_t, 92]
  sdmax_rlcx_rb_wptr_poll_addr_lo: Annotated[uint32_t, 96]
  sdmax_rlcx_rb_wptr_poll_addr_hi: Annotated[uint32_t, 100]
  sdmax_rlcx_rb_aql_cntl: Annotated[uint32_t, 104]
  sdmax_rlcx_minor_ptr_update: Annotated[uint32_t, 108]
  sdmax_rlcx_mcu_dbg0: Annotated[uint32_t, 112]
  sdmax_rlcx_mcu_dbg1: Annotated[uint32_t, 116]
  sdmax_rlcx_context_switch_status: Annotated[uint32_t, 120]
  sdmax_rlcx_midcmd_cntl: Annotated[uint32_t, 124]
  sdmax_rlcx_midcmd_data0: Annotated[uint32_t, 128]
  sdmax_rlcx_midcmd_data1: Annotated[uint32_t, 132]
  sdmax_rlcx_midcmd_data2: Annotated[uint32_t, 136]
  sdmax_rlcx_midcmd_data3: Annotated[uint32_t, 140]
  sdmax_rlcx_midcmd_data4: Annotated[uint32_t, 144]
  sdmax_rlcx_midcmd_data5: Annotated[uint32_t, 148]
  sdmax_rlcx_midcmd_data6: Annotated[uint32_t, 152]
  sdmax_rlcx_midcmd_data7: Annotated[uint32_t, 156]
  sdmax_rlcx_midcmd_data8: Annotated[uint32_t, 160]
  sdmax_rlcx_midcmd_data9: Annotated[uint32_t, 164]
  sdmax_rlcx_midcmd_data10: Annotated[uint32_t, 168]
  sdmax_rlcx_wait_unsatisfied_thd: Annotated[uint32_t, 172]
  sdmax_rlcx_mqd_base_addr_lo: Annotated[uint32_t, 176]
  sdmax_rlcx_mqd_base_addr_hi: Annotated[uint32_t, 180]
  sdmax_rlcx_mqd_control: Annotated[uint32_t, 184]
  reserved_47: Annotated[uint32_t, 188]
  reserved_48: Annotated[uint32_t, 192]
  reserved_49: Annotated[uint32_t, 196]
  reserved_50: Annotated[uint32_t, 200]
  reserved_51: Annotated[uint32_t, 204]
  reserved_52: Annotated[uint32_t, 208]
  reserved_53: Annotated[uint32_t, 212]
  reserved_54: Annotated[uint32_t, 216]
  reserved_55: Annotated[uint32_t, 220]
  reserved_56: Annotated[uint32_t, 224]
  reserved_57: Annotated[uint32_t, 228]
  reserved_58: Annotated[uint32_t, 232]
  reserved_59: Annotated[uint32_t, 236]
  reserved_60: Annotated[uint32_t, 240]
  reserved_61: Annotated[uint32_t, 244]
  reserved_62: Annotated[uint32_t, 248]
  reserved_63: Annotated[uint32_t, 252]
  reserved_64: Annotated[uint32_t, 256]
  reserved_65: Annotated[uint32_t, 260]
  reserved_66: Annotated[uint32_t, 264]
  reserved_67: Annotated[uint32_t, 268]
  reserved_68: Annotated[uint32_t, 272]
  reserved_69: Annotated[uint32_t, 276]
  reserved_70: Annotated[uint32_t, 280]
  reserved_71: Annotated[uint32_t, 284]
  reserved_72: Annotated[uint32_t, 288]
  reserved_73: Annotated[uint32_t, 292]
  reserved_74: Annotated[uint32_t, 296]
  reserved_75: Annotated[uint32_t, 300]
  reserved_76: Annotated[uint32_t, 304]
  reserved_77: Annotated[uint32_t, 308]
  reserved_78: Annotated[uint32_t, 312]
  reserved_79: Annotated[uint32_t, 316]
  reserved_80: Annotated[uint32_t, 320]
  reserved_81: Annotated[uint32_t, 324]
  reserved_82: Annotated[uint32_t, 328]
  reserved_83: Annotated[uint32_t, 332]
  reserved_84: Annotated[uint32_t, 336]
  reserved_85: Annotated[uint32_t, 340]
  reserved_86: Annotated[uint32_t, 344]
  reserved_87: Annotated[uint32_t, 348]
  reserved_88: Annotated[uint32_t, 352]
  reserved_89: Annotated[uint32_t, 356]
  reserved_90: Annotated[uint32_t, 360]
  reserved_91: Annotated[uint32_t, 364]
  reserved_92: Annotated[uint32_t, 368]
  reserved_93: Annotated[uint32_t, 372]
  reserved_94: Annotated[uint32_t, 376]
  reserved_95: Annotated[uint32_t, 380]
  reserved_96: Annotated[uint32_t, 384]
  reserved_97: Annotated[uint32_t, 388]
  reserved_98: Annotated[uint32_t, 392]
  reserved_99: Annotated[uint32_t, 396]
  reserved_100: Annotated[uint32_t, 400]
  reserved_101: Annotated[uint32_t, 404]
  reserved_102: Annotated[uint32_t, 408]
  reserved_103: Annotated[uint32_t, 412]
  reserved_104: Annotated[uint32_t, 416]
  reserved_105: Annotated[uint32_t, 420]
  reserved_106: Annotated[uint32_t, 424]
  reserved_107: Annotated[uint32_t, 428]
  reserved_108: Annotated[uint32_t, 432]
  reserved_109: Annotated[uint32_t, 436]
  reserved_110: Annotated[uint32_t, 440]
  reserved_111: Annotated[uint32_t, 444]
  reserved_112: Annotated[uint32_t, 448]
  reserved_113: Annotated[uint32_t, 452]
  reserved_114: Annotated[uint32_t, 456]
  reserved_115: Annotated[uint32_t, 460]
  reserved_116: Annotated[uint32_t, 464]
  reserved_117: Annotated[uint32_t, 468]
  reserved_118: Annotated[uint32_t, 472]
  reserved_119: Annotated[uint32_t, 476]
  reserved_120: Annotated[uint32_t, 480]
  reserved_121: Annotated[uint32_t, 484]
  reserved_122: Annotated[uint32_t, 488]
  reserved_123: Annotated[uint32_t, 492]
  reserved_124: Annotated[uint32_t, 496]
  reserved_125: Annotated[uint32_t, 500]
  sdma_engine_id: Annotated[uint32_t, 504]
  sdma_queue_id: Annotated[uint32_t, 508]
@record
class struct_v12_compute_mqd:
  SIZE = 2048
  header: Annotated[uint32_t, 0]
  compute_dispatch_initiator: Annotated[uint32_t, 4]
  compute_dim_x: Annotated[uint32_t, 8]
  compute_dim_y: Annotated[uint32_t, 12]
  compute_dim_z: Annotated[uint32_t, 16]
  compute_start_x: Annotated[uint32_t, 20]
  compute_start_y: Annotated[uint32_t, 24]
  compute_start_z: Annotated[uint32_t, 28]
  compute_num_thread_x: Annotated[uint32_t, 32]
  compute_num_thread_y: Annotated[uint32_t, 36]
  compute_num_thread_z: Annotated[uint32_t, 40]
  compute_pipelinestat_enable: Annotated[uint32_t, 44]
  compute_perfcount_enable: Annotated[uint32_t, 48]
  compute_pgm_lo: Annotated[uint32_t, 52]
  compute_pgm_hi: Annotated[uint32_t, 56]
  compute_dispatch_pkt_addr_lo: Annotated[uint32_t, 60]
  compute_dispatch_pkt_addr_hi: Annotated[uint32_t, 64]
  compute_dispatch_scratch_base_lo: Annotated[uint32_t, 68]
  compute_dispatch_scratch_base_hi: Annotated[uint32_t, 72]
  compute_pgm_rsrc1: Annotated[uint32_t, 76]
  compute_pgm_rsrc2: Annotated[uint32_t, 80]
  compute_vmid: Annotated[uint32_t, 84]
  compute_resource_limits: Annotated[uint32_t, 88]
  compute_static_thread_mgmt_se0: Annotated[uint32_t, 92]
  compute_static_thread_mgmt_se1: Annotated[uint32_t, 96]
  compute_tmpring_size: Annotated[uint32_t, 100]
  compute_static_thread_mgmt_se2: Annotated[uint32_t, 104]
  compute_static_thread_mgmt_se3: Annotated[uint32_t, 108]
  compute_restart_x: Annotated[uint32_t, 112]
  compute_restart_y: Annotated[uint32_t, 116]
  compute_restart_z: Annotated[uint32_t, 120]
  compute_thread_trace_enable: Annotated[uint32_t, 124]
  compute_misc_reserved: Annotated[uint32_t, 128]
  compute_dispatch_id: Annotated[uint32_t, 132]
  compute_threadgroup_id: Annotated[uint32_t, 136]
  compute_req_ctrl: Annotated[uint32_t, 140]
  reserved_36: Annotated[uint32_t, 144]
  compute_user_accum_0: Annotated[uint32_t, 148]
  compute_user_accum_1: Annotated[uint32_t, 152]
  compute_user_accum_2: Annotated[uint32_t, 156]
  compute_user_accum_3: Annotated[uint32_t, 160]
  compute_pgm_rsrc3: Annotated[uint32_t, 164]
  compute_ddid_index: Annotated[uint32_t, 168]
  compute_shader_chksum: Annotated[uint32_t, 172]
  compute_static_thread_mgmt_se4: Annotated[uint32_t, 176]
  compute_static_thread_mgmt_se5: Annotated[uint32_t, 180]
  compute_static_thread_mgmt_se6: Annotated[uint32_t, 184]
  compute_static_thread_mgmt_se7: Annotated[uint32_t, 188]
  compute_dispatch_interleave: Annotated[uint32_t, 192]
  compute_relaunch: Annotated[uint32_t, 196]
  compute_wave_restore_addr_lo: Annotated[uint32_t, 200]
  compute_wave_restore_addr_hi: Annotated[uint32_t, 204]
  compute_wave_restore_control: Annotated[uint32_t, 208]
  reserved_53: Annotated[uint32_t, 212]
  reserved_54: Annotated[uint32_t, 216]
  reserved_55: Annotated[uint32_t, 220]
  reserved_56: Annotated[uint32_t, 224]
  reserved_57: Annotated[uint32_t, 228]
  reserved_58: Annotated[uint32_t, 232]
  compute_static_thread_mgmt_se8: Annotated[uint32_t, 236]
  reserved_60: Annotated[uint32_t, 240]
  reserved_61: Annotated[uint32_t, 244]
  reserved_62: Annotated[uint32_t, 248]
  reserved_63: Annotated[uint32_t, 252]
  reserved_64: Annotated[uint32_t, 256]
  compute_user_data_0: Annotated[uint32_t, 260]
  compute_user_data_1: Annotated[uint32_t, 264]
  compute_user_data_2: Annotated[uint32_t, 268]
  compute_user_data_3: Annotated[uint32_t, 272]
  compute_user_data_4: Annotated[uint32_t, 276]
  compute_user_data_5: Annotated[uint32_t, 280]
  compute_user_data_6: Annotated[uint32_t, 284]
  compute_user_data_7: Annotated[uint32_t, 288]
  compute_user_data_8: Annotated[uint32_t, 292]
  compute_user_data_9: Annotated[uint32_t, 296]
  compute_user_data_10: Annotated[uint32_t, 300]
  compute_user_data_11: Annotated[uint32_t, 304]
  compute_user_data_12: Annotated[uint32_t, 308]
  compute_user_data_13: Annotated[uint32_t, 312]
  compute_user_data_14: Annotated[uint32_t, 316]
  compute_user_data_15: Annotated[uint32_t, 320]
  cp_compute_csinvoc_count_lo: Annotated[uint32_t, 324]
  cp_compute_csinvoc_count_hi: Annotated[uint32_t, 328]
  reserved_83: Annotated[uint32_t, 332]
  reserved_84: Annotated[uint32_t, 336]
  reserved_85: Annotated[uint32_t, 340]
  cp_mqd_query_time_lo: Annotated[uint32_t, 344]
  cp_mqd_query_time_hi: Annotated[uint32_t, 348]
  cp_mqd_connect_start_time_lo: Annotated[uint32_t, 352]
  cp_mqd_connect_start_time_hi: Annotated[uint32_t, 356]
  cp_mqd_connect_end_time_lo: Annotated[uint32_t, 360]
  cp_mqd_connect_end_time_hi: Annotated[uint32_t, 364]
  cp_mqd_connect_end_wf_count: Annotated[uint32_t, 368]
  cp_mqd_connect_end_pq_rptr: Annotated[uint32_t, 372]
  cp_mqd_connect_end_pq_wptr: Annotated[uint32_t, 376]
  cp_mqd_connect_end_ib_rptr: Annotated[uint32_t, 380]
  cp_mqd_readindex_lo: Annotated[uint32_t, 384]
  cp_mqd_readindex_hi: Annotated[uint32_t, 388]
  cp_mqd_save_start_time_lo: Annotated[uint32_t, 392]
  cp_mqd_save_start_time_hi: Annotated[uint32_t, 396]
  cp_mqd_save_end_time_lo: Annotated[uint32_t, 400]
  cp_mqd_save_end_time_hi: Annotated[uint32_t, 404]
  cp_mqd_restore_start_time_lo: Annotated[uint32_t, 408]
  cp_mqd_restore_start_time_hi: Annotated[uint32_t, 412]
  cp_mqd_restore_end_time_lo: Annotated[uint32_t, 416]
  cp_mqd_restore_end_time_hi: Annotated[uint32_t, 420]
  disable_queue: Annotated[uint32_t, 424]
  reserved_107: Annotated[uint32_t, 428]
  reserved_108: Annotated[uint32_t, 432]
  reserved_109: Annotated[uint32_t, 436]
  reserved_110: Annotated[uint32_t, 440]
  reserved_111: Annotated[uint32_t, 444]
  reserved_112: Annotated[uint32_t, 448]
  reserved_113: Annotated[uint32_t, 452]
  cp_pq_exe_status_lo: Annotated[uint32_t, 456]
  cp_pq_exe_status_hi: Annotated[uint32_t, 460]
  cp_packet_id_lo: Annotated[uint32_t, 464]
  cp_packet_id_hi: Annotated[uint32_t, 468]
  cp_packet_exe_status_lo: Annotated[uint32_t, 472]
  cp_packet_exe_status_hi: Annotated[uint32_t, 476]
  reserved_120: Annotated[uint32_t, 480]
  reserved_121: Annotated[uint32_t, 484]
  reserved_122: Annotated[uint32_t, 488]
  reserved_123: Annotated[uint32_t, 492]
  ctx_save_base_addr_lo: Annotated[uint32_t, 496]
  ctx_save_base_addr_hi: Annotated[uint32_t, 500]
  reserved_126: Annotated[uint32_t, 504]
  reserved_127: Annotated[uint32_t, 508]
  cp_mqd_base_addr_lo: Annotated[uint32_t, 512]
  cp_mqd_base_addr_hi: Annotated[uint32_t, 516]
  cp_hqd_active: Annotated[uint32_t, 520]
  cp_hqd_vmid: Annotated[uint32_t, 524]
  cp_hqd_persistent_state: Annotated[uint32_t, 528]
  cp_hqd_pipe_priority: Annotated[uint32_t, 532]
  cp_hqd_queue_priority: Annotated[uint32_t, 536]
  cp_hqd_quantum: Annotated[uint32_t, 540]
  cp_hqd_pq_base_lo: Annotated[uint32_t, 544]
  cp_hqd_pq_base_hi: Annotated[uint32_t, 548]
  cp_hqd_pq_rptr: Annotated[uint32_t, 552]
  cp_hqd_pq_rptr_report_addr_lo: Annotated[uint32_t, 556]
  cp_hqd_pq_rptr_report_addr_hi: Annotated[uint32_t, 560]
  cp_hqd_pq_wptr_poll_addr_lo: Annotated[uint32_t, 564]
  cp_hqd_pq_wptr_poll_addr_hi: Annotated[uint32_t, 568]
  cp_hqd_pq_doorbell_control: Annotated[uint32_t, 572]
  reserved_144: Annotated[uint32_t, 576]
  cp_hqd_pq_control: Annotated[uint32_t, 580]
  cp_hqd_ib_base_addr_lo: Annotated[uint32_t, 584]
  cp_hqd_ib_base_addr_hi: Annotated[uint32_t, 588]
  cp_hqd_ib_rptr: Annotated[uint32_t, 592]
  cp_hqd_ib_control: Annotated[uint32_t, 596]
  cp_hqd_iq_timer: Annotated[uint32_t, 600]
  cp_hqd_iq_rptr: Annotated[uint32_t, 604]
  cp_hqd_dequeue_request: Annotated[uint32_t, 608]
  cp_hqd_dma_offload: Annotated[uint32_t, 612]
  cp_hqd_sema_cmd: Annotated[uint32_t, 616]
  cp_hqd_msg_type: Annotated[uint32_t, 620]
  cp_hqd_atomic0_preop_lo: Annotated[uint32_t, 624]
  cp_hqd_atomic0_preop_hi: Annotated[uint32_t, 628]
  cp_hqd_atomic1_preop_lo: Annotated[uint32_t, 632]
  cp_hqd_atomic1_preop_hi: Annotated[uint32_t, 636]
  cp_hqd_hq_status0: Annotated[uint32_t, 640]
  cp_hqd_hq_control0: Annotated[uint32_t, 644]
  cp_mqd_control: Annotated[uint32_t, 648]
  cp_hqd_hq_status1: Annotated[uint32_t, 652]
  cp_hqd_hq_control1: Annotated[uint32_t, 656]
  cp_hqd_eop_base_addr_lo: Annotated[uint32_t, 660]
  cp_hqd_eop_base_addr_hi: Annotated[uint32_t, 664]
  cp_hqd_eop_control: Annotated[uint32_t, 668]
  cp_hqd_eop_rptr: Annotated[uint32_t, 672]
  cp_hqd_eop_wptr: Annotated[uint32_t, 676]
  cp_hqd_eop_done_events: Annotated[uint32_t, 680]
  cp_hqd_ctx_save_base_addr_lo: Annotated[uint32_t, 684]
  cp_hqd_ctx_save_base_addr_hi: Annotated[uint32_t, 688]
  cp_hqd_ctx_save_control: Annotated[uint32_t, 692]
  cp_hqd_cntl_stack_offset: Annotated[uint32_t, 696]
  cp_hqd_cntl_stack_size: Annotated[uint32_t, 700]
  cp_hqd_wg_state_offset: Annotated[uint32_t, 704]
  cp_hqd_ctx_save_size: Annotated[uint32_t, 708]
  reserved_178: Annotated[uint32_t, 712]
  cp_hqd_error: Annotated[uint32_t, 716]
  cp_hqd_eop_wptr_mem: Annotated[uint32_t, 720]
  cp_hqd_aql_control: Annotated[uint32_t, 724]
  cp_hqd_pq_wptr_lo: Annotated[uint32_t, 728]
  cp_hqd_pq_wptr_hi: Annotated[uint32_t, 732]
  reserved_184: Annotated[uint32_t, 736]
  reserved_185: Annotated[uint32_t, 740]
  reserved_186: Annotated[uint32_t, 744]
  reserved_187: Annotated[uint32_t, 748]
  reserved_188: Annotated[uint32_t, 752]
  reserved_189: Annotated[uint32_t, 756]
  reserved_190: Annotated[uint32_t, 760]
  reserved_191: Annotated[uint32_t, 764]
  iqtimer_pkt_header: Annotated[uint32_t, 768]
  iqtimer_pkt_dw0: Annotated[uint32_t, 772]
  iqtimer_pkt_dw1: Annotated[uint32_t, 776]
  iqtimer_pkt_dw2: Annotated[uint32_t, 780]
  iqtimer_pkt_dw3: Annotated[uint32_t, 784]
  iqtimer_pkt_dw4: Annotated[uint32_t, 788]
  iqtimer_pkt_dw5: Annotated[uint32_t, 792]
  iqtimer_pkt_dw6: Annotated[uint32_t, 796]
  iqtimer_pkt_dw7: Annotated[uint32_t, 800]
  iqtimer_pkt_dw8: Annotated[uint32_t, 804]
  iqtimer_pkt_dw9: Annotated[uint32_t, 808]
  iqtimer_pkt_dw10: Annotated[uint32_t, 812]
  iqtimer_pkt_dw11: Annotated[uint32_t, 816]
  iqtimer_pkt_dw12: Annotated[uint32_t, 820]
  iqtimer_pkt_dw13: Annotated[uint32_t, 824]
  iqtimer_pkt_dw14: Annotated[uint32_t, 828]
  iqtimer_pkt_dw15: Annotated[uint32_t, 832]
  iqtimer_pkt_dw16: Annotated[uint32_t, 836]
  iqtimer_pkt_dw17: Annotated[uint32_t, 840]
  iqtimer_pkt_dw18: Annotated[uint32_t, 844]
  iqtimer_pkt_dw19: Annotated[uint32_t, 848]
  iqtimer_pkt_dw20: Annotated[uint32_t, 852]
  iqtimer_pkt_dw21: Annotated[uint32_t, 856]
  iqtimer_pkt_dw22: Annotated[uint32_t, 860]
  iqtimer_pkt_dw23: Annotated[uint32_t, 864]
  iqtimer_pkt_dw24: Annotated[uint32_t, 868]
  iqtimer_pkt_dw25: Annotated[uint32_t, 872]
  iqtimer_pkt_dw26: Annotated[uint32_t, 876]
  iqtimer_pkt_dw27: Annotated[uint32_t, 880]
  iqtimer_pkt_dw28: Annotated[uint32_t, 884]
  iqtimer_pkt_dw29: Annotated[uint32_t, 888]
  iqtimer_pkt_dw30: Annotated[uint32_t, 892]
  iqtimer_pkt_dw31: Annotated[uint32_t, 896]
  reserved_225: Annotated[uint32_t, 900]
  reserved_226: Annotated[uint32_t, 904]
  reserved_227: Annotated[uint32_t, 908]
  set_resources_header: Annotated[uint32_t, 912]
  set_resources_dw1: Annotated[uint32_t, 916]
  set_resources_dw2: Annotated[uint32_t, 920]
  set_resources_dw3: Annotated[uint32_t, 924]
  set_resources_dw4: Annotated[uint32_t, 928]
  set_resources_dw5: Annotated[uint32_t, 932]
  set_resources_dw6: Annotated[uint32_t, 936]
  set_resources_dw7: Annotated[uint32_t, 940]
  reserved_236: Annotated[uint32_t, 944]
  reserved_237: Annotated[uint32_t, 948]
  reserved_238: Annotated[uint32_t, 952]
  reserved_239: Annotated[uint32_t, 956]
  queue_doorbell_id0: Annotated[uint32_t, 960]
  queue_doorbell_id1: Annotated[uint32_t, 964]
  queue_doorbell_id2: Annotated[uint32_t, 968]
  queue_doorbell_id3: Annotated[uint32_t, 972]
  queue_doorbell_id4: Annotated[uint32_t, 976]
  queue_doorbell_id5: Annotated[uint32_t, 980]
  queue_doorbell_id6: Annotated[uint32_t, 984]
  queue_doorbell_id7: Annotated[uint32_t, 988]
  queue_doorbell_id8: Annotated[uint32_t, 992]
  queue_doorbell_id9: Annotated[uint32_t, 996]
  queue_doorbell_id10: Annotated[uint32_t, 1000]
  queue_doorbell_id11: Annotated[uint32_t, 1004]
  queue_doorbell_id12: Annotated[uint32_t, 1008]
  queue_doorbell_id13: Annotated[uint32_t, 1012]
  queue_doorbell_id14: Annotated[uint32_t, 1016]
  queue_doorbell_id15: Annotated[uint32_t, 1020]
  control_buf_addr_lo: Annotated[uint32_t, 1024]
  control_buf_addr_hi: Annotated[uint32_t, 1028]
  control_buf_wptr_lo: Annotated[uint32_t, 1032]
  control_buf_wptr_hi: Annotated[uint32_t, 1036]
  control_buf_dptr_lo: Annotated[uint32_t, 1040]
  control_buf_dptr_hi: Annotated[uint32_t, 1044]
  control_buf_num_entries: Annotated[uint32_t, 1048]
  draw_ring_addr_lo: Annotated[uint32_t, 1052]
  draw_ring_addr_hi: Annotated[uint32_t, 1056]
  reserved_265: Annotated[uint32_t, 1060]
  reserved_266: Annotated[uint32_t, 1064]
  reserved_267: Annotated[uint32_t, 1068]
  reserved_268: Annotated[uint32_t, 1072]
  reserved_269: Annotated[uint32_t, 1076]
  reserved_270: Annotated[uint32_t, 1080]
  reserved_271: Annotated[uint32_t, 1084]
  dfwx_flags: Annotated[uint32_t, 1088]
  dfwx_slot: Annotated[uint32_t, 1092]
  dfwx_client_data_addr_lo: Annotated[uint32_t, 1096]
  dfwx_client_data_addr_hi: Annotated[uint32_t, 1100]
  reserved_276: Annotated[uint32_t, 1104]
  reserved_277: Annotated[uint32_t, 1108]
  reserved_278: Annotated[uint32_t, 1112]
  reserved_279: Annotated[uint32_t, 1116]
  reserved_280: Annotated[uint32_t, 1120]
  reserved_281: Annotated[uint32_t, 1124]
  reserved_282: Annotated[uint32_t, 1128]
  reserved_283: Annotated[uint32_t, 1132]
  reserved_284: Annotated[uint32_t, 1136]
  reserved_285: Annotated[uint32_t, 1140]
  reserved_286: Annotated[uint32_t, 1144]
  reserved_287: Annotated[uint32_t, 1148]
  reserved_288: Annotated[uint32_t, 1152]
  reserved_289: Annotated[uint32_t, 1156]
  reserved_290: Annotated[uint32_t, 1160]
  reserved_291: Annotated[uint32_t, 1164]
  reserved_292: Annotated[uint32_t, 1168]
  reserved_293: Annotated[uint32_t, 1172]
  reserved_294: Annotated[uint32_t, 1176]
  reserved_295: Annotated[uint32_t, 1180]
  reserved_296: Annotated[uint32_t, 1184]
  reserved_297: Annotated[uint32_t, 1188]
  reserved_298: Annotated[uint32_t, 1192]
  reserved_299: Annotated[uint32_t, 1196]
  reserved_300: Annotated[uint32_t, 1200]
  reserved_301: Annotated[uint32_t, 1204]
  reserved_302: Annotated[uint32_t, 1208]
  reserved_303: Annotated[uint32_t, 1212]
  reserved_304: Annotated[uint32_t, 1216]
  reserved_305: Annotated[uint32_t, 1220]
  reserved_306: Annotated[uint32_t, 1224]
  reserved_307: Annotated[uint32_t, 1228]
  reserved_308: Annotated[uint32_t, 1232]
  reserved_309: Annotated[uint32_t, 1236]
  reserved_310: Annotated[uint32_t, 1240]
  reserved_311: Annotated[uint32_t, 1244]
  reserved_312: Annotated[uint32_t, 1248]
  reserved_313: Annotated[uint32_t, 1252]
  reserved_314: Annotated[uint32_t, 1256]
  reserved_315: Annotated[uint32_t, 1260]
  reserved_316: Annotated[uint32_t, 1264]
  reserved_317: Annotated[uint32_t, 1268]
  reserved_318: Annotated[uint32_t, 1272]
  reserved_319: Annotated[uint32_t, 1276]
  reserved_320: Annotated[uint32_t, 1280]
  reserved_321: Annotated[uint32_t, 1284]
  reserved_322: Annotated[uint32_t, 1288]
  reserved_323: Annotated[uint32_t, 1292]
  reserved_324: Annotated[uint32_t, 1296]
  reserved_325: Annotated[uint32_t, 1300]
  reserved_326: Annotated[uint32_t, 1304]
  reserved_327: Annotated[uint32_t, 1308]
  reserved_328: Annotated[uint32_t, 1312]
  reserved_329: Annotated[uint32_t, 1316]
  reserved_330: Annotated[uint32_t, 1320]
  reserved_331: Annotated[uint32_t, 1324]
  reserved_332: Annotated[uint32_t, 1328]
  reserved_333: Annotated[uint32_t, 1332]
  reserved_334: Annotated[uint32_t, 1336]
  reserved_335: Annotated[uint32_t, 1340]
  reserved_336: Annotated[uint32_t, 1344]
  reserved_337: Annotated[uint32_t, 1348]
  reserved_338: Annotated[uint32_t, 1352]
  reserved_339: Annotated[uint32_t, 1356]
  reserved_340: Annotated[uint32_t, 1360]
  reserved_341: Annotated[uint32_t, 1364]
  reserved_342: Annotated[uint32_t, 1368]
  reserved_343: Annotated[uint32_t, 1372]
  reserved_344: Annotated[uint32_t, 1376]
  reserved_345: Annotated[uint32_t, 1380]
  reserved_346: Annotated[uint32_t, 1384]
  reserved_347: Annotated[uint32_t, 1388]
  reserved_348: Annotated[uint32_t, 1392]
  reserved_349: Annotated[uint32_t, 1396]
  reserved_350: Annotated[uint32_t, 1400]
  reserved_351: Annotated[uint32_t, 1404]
  reserved_352: Annotated[uint32_t, 1408]
  reserved_353: Annotated[uint32_t, 1412]
  reserved_354: Annotated[uint32_t, 1416]
  reserved_355: Annotated[uint32_t, 1420]
  reserved_356: Annotated[uint32_t, 1424]
  reserved_357: Annotated[uint32_t, 1428]
  reserved_358: Annotated[uint32_t, 1432]
  reserved_359: Annotated[uint32_t, 1436]
  reserved_360: Annotated[uint32_t, 1440]
  reserved_361: Annotated[uint32_t, 1444]
  reserved_362: Annotated[uint32_t, 1448]
  reserved_363: Annotated[uint32_t, 1452]
  reserved_364: Annotated[uint32_t, 1456]
  reserved_365: Annotated[uint32_t, 1460]
  reserved_366: Annotated[uint32_t, 1464]
  reserved_367: Annotated[uint32_t, 1468]
  reserved_368: Annotated[uint32_t, 1472]
  reserved_369: Annotated[uint32_t, 1476]
  reserved_370: Annotated[uint32_t, 1480]
  reserved_371: Annotated[uint32_t, 1484]
  reserved_372: Annotated[uint32_t, 1488]
  reserved_373: Annotated[uint32_t, 1492]
  reserved_374: Annotated[uint32_t, 1496]
  reserved_375: Annotated[uint32_t, 1500]
  reserved_376: Annotated[uint32_t, 1504]
  reserved_377: Annotated[uint32_t, 1508]
  reserved_378: Annotated[uint32_t, 1512]
  reserved_379: Annotated[uint32_t, 1516]
  reserved_380: Annotated[uint32_t, 1520]
  reserved_381: Annotated[uint32_t, 1524]
  reserved_382: Annotated[uint32_t, 1528]
  reserved_383: Annotated[uint32_t, 1532]
  reserved_384: Annotated[uint32_t, 1536]
  reserved_385: Annotated[uint32_t, 1540]
  reserved_386: Annotated[uint32_t, 1544]
  reserved_387: Annotated[uint32_t, 1548]
  reserved_388: Annotated[uint32_t, 1552]
  reserved_389: Annotated[uint32_t, 1556]
  reserved_390: Annotated[uint32_t, 1560]
  reserved_391: Annotated[uint32_t, 1564]
  reserved_392: Annotated[uint32_t, 1568]
  reserved_393: Annotated[uint32_t, 1572]
  reserved_394: Annotated[uint32_t, 1576]
  reserved_395: Annotated[uint32_t, 1580]
  reserved_396: Annotated[uint32_t, 1584]
  reserved_397: Annotated[uint32_t, 1588]
  reserved_398: Annotated[uint32_t, 1592]
  reserved_399: Annotated[uint32_t, 1596]
  reserved_400: Annotated[uint32_t, 1600]
  reserved_401: Annotated[uint32_t, 1604]
  reserved_402: Annotated[uint32_t, 1608]
  reserved_403: Annotated[uint32_t, 1612]
  reserved_404: Annotated[uint32_t, 1616]
  reserved_405: Annotated[uint32_t, 1620]
  reserved_406: Annotated[uint32_t, 1624]
  reserved_407: Annotated[uint32_t, 1628]
  reserved_408: Annotated[uint32_t, 1632]
  reserved_409: Annotated[uint32_t, 1636]
  reserved_410: Annotated[uint32_t, 1640]
  reserved_411: Annotated[uint32_t, 1644]
  reserved_412: Annotated[uint32_t, 1648]
  reserved_413: Annotated[uint32_t, 1652]
  reserved_414: Annotated[uint32_t, 1656]
  reserved_415: Annotated[uint32_t, 1660]
  reserved_416: Annotated[uint32_t, 1664]
  reserved_417: Annotated[uint32_t, 1668]
  reserved_418: Annotated[uint32_t, 1672]
  reserved_419: Annotated[uint32_t, 1676]
  reserved_420: Annotated[uint32_t, 1680]
  reserved_421: Annotated[uint32_t, 1684]
  reserved_422: Annotated[uint32_t, 1688]
  reserved_423: Annotated[uint32_t, 1692]
  reserved_424: Annotated[uint32_t, 1696]
  reserved_425: Annotated[uint32_t, 1700]
  reserved_426: Annotated[uint32_t, 1704]
  reserved_427: Annotated[uint32_t, 1708]
  reserved_428: Annotated[uint32_t, 1712]
  reserved_429: Annotated[uint32_t, 1716]
  reserved_430: Annotated[uint32_t, 1720]
  reserved_431: Annotated[uint32_t, 1724]
  reserved_432: Annotated[uint32_t, 1728]
  reserved_433: Annotated[uint32_t, 1732]
  reserved_434: Annotated[uint32_t, 1736]
  reserved_435: Annotated[uint32_t, 1740]
  reserved_436: Annotated[uint32_t, 1744]
  reserved_437: Annotated[uint32_t, 1748]
  reserved_438: Annotated[uint32_t, 1752]
  reserved_439: Annotated[uint32_t, 1756]
  reserved_440: Annotated[uint32_t, 1760]
  reserved_441: Annotated[uint32_t, 1764]
  reserved_442: Annotated[uint32_t, 1768]
  reserved_443: Annotated[uint32_t, 1772]
  reserved_444: Annotated[uint32_t, 1776]
  reserved_445: Annotated[uint32_t, 1780]
  reserved_446: Annotated[uint32_t, 1784]
  reserved_447: Annotated[uint32_t, 1788]
  gws_0_val: Annotated[uint32_t, 1792]
  gws_1_val: Annotated[uint32_t, 1796]
  gws_2_val: Annotated[uint32_t, 1800]
  gws_3_val: Annotated[uint32_t, 1804]
  gws_4_val: Annotated[uint32_t, 1808]
  gws_5_val: Annotated[uint32_t, 1812]
  gws_6_val: Annotated[uint32_t, 1816]
  gws_7_val: Annotated[uint32_t, 1820]
  gws_8_val: Annotated[uint32_t, 1824]
  gws_9_val: Annotated[uint32_t, 1828]
  gws_10_val: Annotated[uint32_t, 1832]
  gws_11_val: Annotated[uint32_t, 1836]
  gws_12_val: Annotated[uint32_t, 1840]
  gws_13_val: Annotated[uint32_t, 1844]
  gws_14_val: Annotated[uint32_t, 1848]
  gws_15_val: Annotated[uint32_t, 1852]
  gws_16_val: Annotated[uint32_t, 1856]
  gws_17_val: Annotated[uint32_t, 1860]
  gws_18_val: Annotated[uint32_t, 1864]
  gws_19_val: Annotated[uint32_t, 1868]
  gws_20_val: Annotated[uint32_t, 1872]
  gws_21_val: Annotated[uint32_t, 1876]
  gws_22_val: Annotated[uint32_t, 1880]
  gws_23_val: Annotated[uint32_t, 1884]
  gws_24_val: Annotated[uint32_t, 1888]
  gws_25_val: Annotated[uint32_t, 1892]
  gws_26_val: Annotated[uint32_t, 1896]
  gws_27_val: Annotated[uint32_t, 1900]
  gws_28_val: Annotated[uint32_t, 1904]
  gws_29_val: Annotated[uint32_t, 1908]
  gws_30_val: Annotated[uint32_t, 1912]
  gws_31_val: Annotated[uint32_t, 1916]
  gws_32_val: Annotated[uint32_t, 1920]
  gws_33_val: Annotated[uint32_t, 1924]
  gws_34_val: Annotated[uint32_t, 1928]
  gws_35_val: Annotated[uint32_t, 1932]
  gws_36_val: Annotated[uint32_t, 1936]
  gws_37_val: Annotated[uint32_t, 1940]
  gws_38_val: Annotated[uint32_t, 1944]
  gws_39_val: Annotated[uint32_t, 1948]
  gws_40_val: Annotated[uint32_t, 1952]
  gws_41_val: Annotated[uint32_t, 1956]
  gws_42_val: Annotated[uint32_t, 1960]
  gws_43_val: Annotated[uint32_t, 1964]
  gws_44_val: Annotated[uint32_t, 1968]
  gws_45_val: Annotated[uint32_t, 1972]
  gws_46_val: Annotated[uint32_t, 1976]
  gws_47_val: Annotated[uint32_t, 1980]
  gws_48_val: Annotated[uint32_t, 1984]
  gws_49_val: Annotated[uint32_t, 1988]
  gws_50_val: Annotated[uint32_t, 1992]
  gws_51_val: Annotated[uint32_t, 1996]
  gws_52_val: Annotated[uint32_t, 2000]
  gws_53_val: Annotated[uint32_t, 2004]
  gws_54_val: Annotated[uint32_t, 2008]
  gws_55_val: Annotated[uint32_t, 2012]
  gws_56_val: Annotated[uint32_t, 2016]
  gws_57_val: Annotated[uint32_t, 2020]
  gws_58_val: Annotated[uint32_t, 2024]
  gws_59_val: Annotated[uint32_t, 2028]
  gws_60_val: Annotated[uint32_t, 2032]
  gws_61_val: Annotated[uint32_t, 2036]
  gws_62_val: Annotated[uint32_t, 2040]
  gws_63_val: Annotated[uint32_t, 2044]
enum_amdgpu_vm_level = CEnum(ctypes.c_uint32)
AMDGPU_VM_PDB2 = enum_amdgpu_vm_level.define('AMDGPU_VM_PDB2', 0)
AMDGPU_VM_PDB1 = enum_amdgpu_vm_level.define('AMDGPU_VM_PDB1', 1)
AMDGPU_VM_PDB0 = enum_amdgpu_vm_level.define('AMDGPU_VM_PDB0', 2)
AMDGPU_VM_PTB = enum_amdgpu_vm_level.define('AMDGPU_VM_PTB', 3)

table = CEnum(ctypes.c_uint32)
IP_DISCOVERY = table.define('IP_DISCOVERY', 0)
GC = table.define('GC', 1)
HARVEST_INFO = table.define('HARVEST_INFO', 2)
VCN_INFO = table.define('VCN_INFO', 3)
MALL_INFO = table.define('MALL_INFO', 4)
NPS_INFO = table.define('NPS_INFO', 5)
TOTAL_TABLES = table.define('TOTAL_TABLES', 6)

@record
class struct_table_info:
  SIZE = 8
  offset: Annotated[uint16_t, 0]
  checksum: Annotated[uint16_t, 2]
  size: Annotated[uint16_t, 4]
  padding: Annotated[uint16_t, 6]
uint16_t = ctypes.c_uint16
table_info = struct_table_info
@record
class struct_binary_header:
  SIZE = 60
  binary_signature: Annotated[uint32_t, 0]
  version_major: Annotated[uint16_t, 4]
  version_minor: Annotated[uint16_t, 6]
  binary_checksum: Annotated[uint16_t, 8]
  binary_size: Annotated[uint16_t, 10]
  table_list: Annotated[(table_info* 6), 12]
binary_header = struct_binary_header
@record
class struct_die_info:
  SIZE = 4
  die_id: Annotated[uint16_t, 0]
  die_offset: Annotated[uint16_t, 2]
die_info = struct_die_info
@record
class struct_ip_discovery_header:
  SIZE = 80
  signature: Annotated[uint32_t, 0]
  version: Annotated[uint16_t, 4]
  size: Annotated[uint16_t, 6]
  id: Annotated[uint32_t, 8]
  num_dies: Annotated[uint16_t, 12]
  die_info: Annotated[(die_info* 16), 14]
  padding: Annotated[(uint16_t* 1), 78]
  base_addr_64_bit: Annotated[uint8_t, 0, 1, 0]
  reserved: Annotated[uint8_t, 0, 7, 1]
  reserved2: Annotated[uint8_t, 1]
uint8_t = ctypes.c_ubyte
ip_discovery_header = struct_ip_discovery_header
@record
class struct_ip:
  SIZE = 8
  hw_id: Annotated[uint16_t, 0]
  number_instance: Annotated[uint8_t, 2]
  num_base_address: Annotated[uint8_t, 3]
  major: Annotated[uint8_t, 4]
  minor: Annotated[uint8_t, 5]
  revision: Annotated[uint8_t, 6]
  harvest: Annotated[uint8_t, 7, 4, 0]
  reserved: Annotated[uint8_t, 7, 4, 4]
  base_address: Annotated[(uint32_t * 0), 8]
ip = struct_ip
@record
class struct_ip_v3:
  SIZE = 8
  hw_id: Annotated[uint16_t, 0]
  instance_number: Annotated[uint8_t, 2]
  num_base_address: Annotated[uint8_t, 3]
  major: Annotated[uint8_t, 4]
  minor: Annotated[uint8_t, 5]
  revision: Annotated[uint8_t, 6]
  sub_revision: Annotated[uint8_t, 7, 4, 0]
  variant: Annotated[uint8_t, 7, 4, 4]
  base_address: Annotated[(uint32_t * 0), 8]
ip_v3 = struct_ip_v3
@record
class struct_ip_v4:
  SIZE = 7
  hw_id: Annotated[uint16_t, 0]
  instance_number: Annotated[uint8_t, 2]
  num_base_address: Annotated[uint8_t, 3]
  major: Annotated[uint8_t, 4]
  minor: Annotated[uint8_t, 5]
  revision: Annotated[uint8_t, 6]
ip_v4 = struct_ip_v4
@record
class struct_die_header:
  SIZE = 4
  die_id: Annotated[uint16_t, 0]
  num_ips: Annotated[uint16_t, 2]
die_header = struct_die_header
@record
class struct_ip_structure:
  SIZE = 24
  header: Annotated[ctypes.POINTER(ip_discovery_header), 0]
  die: Annotated[struct_die, 8]
@record
class struct_die:
  SIZE = 16
  die_header: Annotated[ctypes.POINTER(die_header), 0]
  ip_list: Annotated[ctypes.POINTER(ip), 8]
  ip_v3_list: Annotated[ctypes.POINTER(ip_v3), 8]
  ip_v4_list: Annotated[ctypes.POINTER(ip_v4), 8]
ip_structure = struct_ip_structure
@record
class struct_gpu_info_header:
  SIZE = 12
  table_id: Annotated[uint32_t, 0]
  version_major: Annotated[uint16_t, 4]
  version_minor: Annotated[uint16_t, 6]
  size: Annotated[uint32_t, 8]
@record
class struct_gc_info_v1_0:
  SIZE = 88
  header: Annotated[struct_gpu_info_header, 0]
  gc_num_se: Annotated[uint32_t, 12]
  gc_num_wgp0_per_sa: Annotated[uint32_t, 16]
  gc_num_wgp1_per_sa: Annotated[uint32_t, 20]
  gc_num_rb_per_se: Annotated[uint32_t, 24]
  gc_num_gl2c: Annotated[uint32_t, 28]
  gc_num_gprs: Annotated[uint32_t, 32]
  gc_num_max_gs_thds: Annotated[uint32_t, 36]
  gc_gs_table_depth: Annotated[uint32_t, 40]
  gc_gsprim_buff_depth: Annotated[uint32_t, 44]
  gc_parameter_cache_depth: Annotated[uint32_t, 48]
  gc_double_offchip_lds_buffer: Annotated[uint32_t, 52]
  gc_wave_size: Annotated[uint32_t, 56]
  gc_max_waves_per_simd: Annotated[uint32_t, 60]
  gc_max_scratch_slots_per_cu: Annotated[uint32_t, 64]
  gc_lds_size: Annotated[uint32_t, 68]
  gc_num_sc_per_se: Annotated[uint32_t, 72]
  gc_num_sa_per_se: Annotated[uint32_t, 76]
  gc_num_packer_per_sc: Annotated[uint32_t, 80]
  gc_num_gl2a: Annotated[uint32_t, 84]
@record
class struct_gc_info_v1_1:
  SIZE = 100
  header: Annotated[struct_gpu_info_header, 0]
  gc_num_se: Annotated[uint32_t, 12]
  gc_num_wgp0_per_sa: Annotated[uint32_t, 16]
  gc_num_wgp1_per_sa: Annotated[uint32_t, 20]
  gc_num_rb_per_se: Annotated[uint32_t, 24]
  gc_num_gl2c: Annotated[uint32_t, 28]
  gc_num_gprs: Annotated[uint32_t, 32]
  gc_num_max_gs_thds: Annotated[uint32_t, 36]
  gc_gs_table_depth: Annotated[uint32_t, 40]
  gc_gsprim_buff_depth: Annotated[uint32_t, 44]
  gc_parameter_cache_depth: Annotated[uint32_t, 48]
  gc_double_offchip_lds_buffer: Annotated[uint32_t, 52]
  gc_wave_size: Annotated[uint32_t, 56]
  gc_max_waves_per_simd: Annotated[uint32_t, 60]
  gc_max_scratch_slots_per_cu: Annotated[uint32_t, 64]
  gc_lds_size: Annotated[uint32_t, 68]
  gc_num_sc_per_se: Annotated[uint32_t, 72]
  gc_num_sa_per_se: Annotated[uint32_t, 76]
  gc_num_packer_per_sc: Annotated[uint32_t, 80]
  gc_num_gl2a: Annotated[uint32_t, 84]
  gc_num_tcp_per_sa: Annotated[uint32_t, 88]
  gc_num_sdp_interface: Annotated[uint32_t, 92]
  gc_num_tcps: Annotated[uint32_t, 96]
@record
class struct_gc_info_v1_2:
  SIZE = 132
  header: Annotated[struct_gpu_info_header, 0]
  gc_num_se: Annotated[uint32_t, 12]
  gc_num_wgp0_per_sa: Annotated[uint32_t, 16]
  gc_num_wgp1_per_sa: Annotated[uint32_t, 20]
  gc_num_rb_per_se: Annotated[uint32_t, 24]
  gc_num_gl2c: Annotated[uint32_t, 28]
  gc_num_gprs: Annotated[uint32_t, 32]
  gc_num_max_gs_thds: Annotated[uint32_t, 36]
  gc_gs_table_depth: Annotated[uint32_t, 40]
  gc_gsprim_buff_depth: Annotated[uint32_t, 44]
  gc_parameter_cache_depth: Annotated[uint32_t, 48]
  gc_double_offchip_lds_buffer: Annotated[uint32_t, 52]
  gc_wave_size: Annotated[uint32_t, 56]
  gc_max_waves_per_simd: Annotated[uint32_t, 60]
  gc_max_scratch_slots_per_cu: Annotated[uint32_t, 64]
  gc_lds_size: Annotated[uint32_t, 68]
  gc_num_sc_per_se: Annotated[uint32_t, 72]
  gc_num_sa_per_se: Annotated[uint32_t, 76]
  gc_num_packer_per_sc: Annotated[uint32_t, 80]
  gc_num_gl2a: Annotated[uint32_t, 84]
  gc_num_tcp_per_sa: Annotated[uint32_t, 88]
  gc_num_sdp_interface: Annotated[uint32_t, 92]
  gc_num_tcps: Annotated[uint32_t, 96]
  gc_num_tcp_per_wpg: Annotated[uint32_t, 100]
  gc_tcp_l1_size: Annotated[uint32_t, 104]
  gc_num_sqc_per_wgp: Annotated[uint32_t, 108]
  gc_l1_instruction_cache_size_per_sqc: Annotated[uint32_t, 112]
  gc_l1_data_cache_size_per_sqc: Annotated[uint32_t, 116]
  gc_gl1c_per_sa: Annotated[uint32_t, 120]
  gc_gl1c_size_per_instance: Annotated[uint32_t, 124]
  gc_gl2c_per_gpu: Annotated[uint32_t, 128]
@record
class struct_gc_info_v1_3:
  SIZE = 164
  header: Annotated[struct_gpu_info_header, 0]
  gc_num_se: Annotated[uint32_t, 12]
  gc_num_wgp0_per_sa: Annotated[uint32_t, 16]
  gc_num_wgp1_per_sa: Annotated[uint32_t, 20]
  gc_num_rb_per_se: Annotated[uint32_t, 24]
  gc_num_gl2c: Annotated[uint32_t, 28]
  gc_num_gprs: Annotated[uint32_t, 32]
  gc_num_max_gs_thds: Annotated[uint32_t, 36]
  gc_gs_table_depth: Annotated[uint32_t, 40]
  gc_gsprim_buff_depth: Annotated[uint32_t, 44]
  gc_parameter_cache_depth: Annotated[uint32_t, 48]
  gc_double_offchip_lds_buffer: Annotated[uint32_t, 52]
  gc_wave_size: Annotated[uint32_t, 56]
  gc_max_waves_per_simd: Annotated[uint32_t, 60]
  gc_max_scratch_slots_per_cu: Annotated[uint32_t, 64]
  gc_lds_size: Annotated[uint32_t, 68]
  gc_num_sc_per_se: Annotated[uint32_t, 72]
  gc_num_sa_per_se: Annotated[uint32_t, 76]
  gc_num_packer_per_sc: Annotated[uint32_t, 80]
  gc_num_gl2a: Annotated[uint32_t, 84]
  gc_num_tcp_per_sa: Annotated[uint32_t, 88]
  gc_num_sdp_interface: Annotated[uint32_t, 92]
  gc_num_tcps: Annotated[uint32_t, 96]
  gc_num_tcp_per_wpg: Annotated[uint32_t, 100]
  gc_tcp_l1_size: Annotated[uint32_t, 104]
  gc_num_sqc_per_wgp: Annotated[uint32_t, 108]
  gc_l1_instruction_cache_size_per_sqc: Annotated[uint32_t, 112]
  gc_l1_data_cache_size_per_sqc: Annotated[uint32_t, 116]
  gc_gl1c_per_sa: Annotated[uint32_t, 120]
  gc_gl1c_size_per_instance: Annotated[uint32_t, 124]
  gc_gl2c_per_gpu: Annotated[uint32_t, 128]
  gc_tcp_size_per_cu: Annotated[uint32_t, 132]
  gc_tcp_cache_line_size: Annotated[uint32_t, 136]
  gc_instruction_cache_size_per_sqc: Annotated[uint32_t, 140]
  gc_instruction_cache_line_size: Annotated[uint32_t, 144]
  gc_scalar_data_cache_size_per_sqc: Annotated[uint32_t, 148]
  gc_scalar_data_cache_line_size: Annotated[uint32_t, 152]
  gc_tcc_size: Annotated[uint32_t, 156]
  gc_tcc_cache_line_size: Annotated[uint32_t, 160]
@record
class struct_gc_info_v2_0:
  SIZE = 80
  header: Annotated[struct_gpu_info_header, 0]
  gc_num_se: Annotated[uint32_t, 12]
  gc_num_cu_per_sh: Annotated[uint32_t, 16]
  gc_num_sh_per_se: Annotated[uint32_t, 20]
  gc_num_rb_per_se: Annotated[uint32_t, 24]
  gc_num_tccs: Annotated[uint32_t, 28]
  gc_num_gprs: Annotated[uint32_t, 32]
  gc_num_max_gs_thds: Annotated[uint32_t, 36]
  gc_gs_table_depth: Annotated[uint32_t, 40]
  gc_gsprim_buff_depth: Annotated[uint32_t, 44]
  gc_parameter_cache_depth: Annotated[uint32_t, 48]
  gc_double_offchip_lds_buffer: Annotated[uint32_t, 52]
  gc_wave_size: Annotated[uint32_t, 56]
  gc_max_waves_per_simd: Annotated[uint32_t, 60]
  gc_max_scratch_slots_per_cu: Annotated[uint32_t, 64]
  gc_lds_size: Annotated[uint32_t, 68]
  gc_num_sc_per_se: Annotated[uint32_t, 72]
  gc_num_packer_per_sc: Annotated[uint32_t, 76]
@record
class struct_gc_info_v2_1:
  SIZE = 108
  header: Annotated[struct_gpu_info_header, 0]
  gc_num_se: Annotated[uint32_t, 12]
  gc_num_cu_per_sh: Annotated[uint32_t, 16]
  gc_num_sh_per_se: Annotated[uint32_t, 20]
  gc_num_rb_per_se: Annotated[uint32_t, 24]
  gc_num_tccs: Annotated[uint32_t, 28]
  gc_num_gprs: Annotated[uint32_t, 32]
  gc_num_max_gs_thds: Annotated[uint32_t, 36]
  gc_gs_table_depth: Annotated[uint32_t, 40]
  gc_gsprim_buff_depth: Annotated[uint32_t, 44]
  gc_parameter_cache_depth: Annotated[uint32_t, 48]
  gc_double_offchip_lds_buffer: Annotated[uint32_t, 52]
  gc_wave_size: Annotated[uint32_t, 56]
  gc_max_waves_per_simd: Annotated[uint32_t, 60]
  gc_max_scratch_slots_per_cu: Annotated[uint32_t, 64]
  gc_lds_size: Annotated[uint32_t, 68]
  gc_num_sc_per_se: Annotated[uint32_t, 72]
  gc_num_packer_per_sc: Annotated[uint32_t, 76]
  gc_num_tcp_per_sh: Annotated[uint32_t, 80]
  gc_tcp_size_per_cu: Annotated[uint32_t, 84]
  gc_num_sdp_interface: Annotated[uint32_t, 88]
  gc_num_cu_per_sqc: Annotated[uint32_t, 92]
  gc_instruction_cache_size_per_sqc: Annotated[uint32_t, 96]
  gc_scalar_data_cache_size_per_sqc: Annotated[uint32_t, 100]
  gc_tcc_size: Annotated[uint32_t, 104]
@record
class struct_harvest_info_header:
  SIZE = 8
  signature: Annotated[uint32_t, 0]
  version: Annotated[uint32_t, 4]
harvest_info_header = struct_harvest_info_header
@record
class struct_harvest_info:
  SIZE = 4
  hw_id: Annotated[uint16_t, 0]
  number_instance: Annotated[uint8_t, 2]
  reserved: Annotated[uint8_t, 3]
harvest_info = struct_harvest_info
@record
class struct_harvest_table:
  SIZE = 136
  header: Annotated[harvest_info_header, 0]
  list: Annotated[(harvest_info* 32), 8]
harvest_table = struct_harvest_table
@record
class struct_mall_info_header:
  SIZE = 12
  table_id: Annotated[uint32_t, 0]
  version_major: Annotated[uint16_t, 4]
  version_minor: Annotated[uint16_t, 6]
  size_bytes: Annotated[uint32_t, 8]
@record
class struct_mall_info_v1_0:
  SIZE = 48
  header: Annotated[struct_mall_info_header, 0]
  mall_size_per_m: Annotated[uint32_t, 12]
  m_s_present: Annotated[uint32_t, 16]
  m_half_use: Annotated[uint32_t, 20]
  m_mall_config: Annotated[uint32_t, 24]
  reserved: Annotated[(uint32_t* 5), 28]
@record
class struct_mall_info_v2_0:
  SIZE = 48
  header: Annotated[struct_mall_info_header, 0]
  mall_size_per_umc: Annotated[uint32_t, 12]
  reserved: Annotated[(uint32_t* 8), 16]
@record
class struct_vcn_info_header:
  SIZE = 12
  table_id: Annotated[uint32_t, 0]
  version_major: Annotated[uint16_t, 4]
  version_minor: Annotated[uint16_t, 6]
  size_bytes: Annotated[uint32_t, 8]
@record
class struct_vcn_instance_info_v1_0:
  SIZE = 16
  instance_num: Annotated[uint32_t, 0]
  fuse_data: Annotated[union__fuse_data, 4]
  reserved: Annotated[(uint32_t* 2), 8]
@record
class union__fuse_data:
  SIZE = 4
  bits: Annotated[_anonstruct0, 0]
  all_bits: Annotated[uint32_t, 0]
@record
class _anonstruct0:
  SIZE = 4
  av1_disabled: Annotated[uint32_t, 0, 1, 0]
  vp9_disabled: Annotated[uint32_t, 0, 1, 1]
  hevc_disabled: Annotated[uint32_t, 0, 1, 2]
  h264_disabled: Annotated[uint32_t, 0, 1, 3]
  reserved: Annotated[uint32_t, 0, 28, 4]
@record
class struct_vcn_info_v1_0:
  SIZE = 96
  header: Annotated[struct_vcn_info_header, 0]
  num_of_instances: Annotated[uint32_t, 12]
  instance_info: Annotated[(struct_vcn_instance_info_v1_0* 4), 16]
  reserved: Annotated[(uint32_t* 4), 80]
@record
class struct_nps_info_header:
  SIZE = 12
  table_id: Annotated[uint32_t, 0]
  version_major: Annotated[uint16_t, 4]
  version_minor: Annotated[uint16_t, 6]
  size_bytes: Annotated[uint32_t, 8]
@record
class struct_nps_instance_info_v1_0:
  SIZE = 16
  base_address: Annotated[uint64_t, 0]
  limit_address: Annotated[uint64_t, 8]
uint64_t = ctypes.c_uint64
@record
class struct_nps_info_v1_0:
  SIZE = 212
  header: Annotated[struct_nps_info_header, 0]
  nps_type: Annotated[uint32_t, 12]
  count: Annotated[uint32_t, 16]
  instance_info: Annotated[(struct_nps_instance_info_v1_0* 12), 20]
enum_amd_hw_ip_block_type = CEnum(ctypes.c_uint32)
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

@record
class struct_common_firmware_header:
  SIZE = 32
  size_bytes: Annotated[ctypes.c_uint32, 0]
  header_size_bytes: Annotated[ctypes.c_uint32, 4]
  header_version_major: Annotated[ctypes.c_uint16, 8]
  header_version_minor: Annotated[ctypes.c_uint16, 10]
  ip_version_major: Annotated[ctypes.c_uint16, 12]
  ip_version_minor: Annotated[ctypes.c_uint16, 14]
  ucode_version: Annotated[ctypes.c_uint32, 16]
  ucode_size_bytes: Annotated[ctypes.c_uint32, 20]
  ucode_array_offset_bytes: Annotated[ctypes.c_uint32, 24]
  crc32: Annotated[ctypes.c_uint32, 28]
@record
class struct_mc_firmware_header_v1_0:
  SIZE = 40
  header: Annotated[struct_common_firmware_header, 0]
  io_debug_size_bytes: Annotated[ctypes.c_uint32, 32]
  io_debug_array_offset_bytes: Annotated[ctypes.c_uint32, 36]
@record
class struct_smc_firmware_header_v1_0:
  SIZE = 36
  header: Annotated[struct_common_firmware_header, 0]
  ucode_start_addr: Annotated[ctypes.c_uint32, 32]
@record
class struct_smc_firmware_header_v2_0:
  SIZE = 44
  v1_0: Annotated[struct_smc_firmware_header_v1_0, 0]
  ppt_offset_bytes: Annotated[ctypes.c_uint32, 36]
  ppt_size_bytes: Annotated[ctypes.c_uint32, 40]
@record
class struct_smc_soft_pptable_entry:
  SIZE = 12
  id: Annotated[ctypes.c_uint32, 0]
  ppt_offset_bytes: Annotated[ctypes.c_uint32, 4]
  ppt_size_bytes: Annotated[ctypes.c_uint32, 8]
@record
class struct_smc_firmware_header_v2_1:
  SIZE = 44
  v1_0: Annotated[struct_smc_firmware_header_v1_0, 0]
  pptable_count: Annotated[ctypes.c_uint32, 36]
  pptable_entry_offset: Annotated[ctypes.c_uint32, 40]
@record
class struct_psp_fw_legacy_bin_desc:
  SIZE = 12
  fw_version: Annotated[ctypes.c_uint32, 0]
  offset_bytes: Annotated[ctypes.c_uint32, 4]
  size_bytes: Annotated[ctypes.c_uint32, 8]
@record
class struct_psp_firmware_header_v1_0:
  SIZE = 44
  header: Annotated[struct_common_firmware_header, 0]
  sos: Annotated[struct_psp_fw_legacy_bin_desc, 32]
@record
class struct_psp_firmware_header_v1_1:
  SIZE = 68
  v1_0: Annotated[struct_psp_firmware_header_v1_0, 0]
  toc: Annotated[struct_psp_fw_legacy_bin_desc, 44]
  kdb: Annotated[struct_psp_fw_legacy_bin_desc, 56]
@record
class struct_psp_firmware_header_v1_2:
  SIZE = 68
  v1_0: Annotated[struct_psp_firmware_header_v1_0, 0]
  res: Annotated[struct_psp_fw_legacy_bin_desc, 44]
  kdb: Annotated[struct_psp_fw_legacy_bin_desc, 56]
@record
class struct_psp_firmware_header_v1_3:
  SIZE = 116
  v1_1: Annotated[struct_psp_firmware_header_v1_1, 0]
  spl: Annotated[struct_psp_fw_legacy_bin_desc, 68]
  rl: Annotated[struct_psp_fw_legacy_bin_desc, 80]
  sys_drv_aux: Annotated[struct_psp_fw_legacy_bin_desc, 92]
  sos_aux: Annotated[struct_psp_fw_legacy_bin_desc, 104]
@record
class struct_psp_fw_bin_desc:
  SIZE = 16
  fw_type: Annotated[ctypes.c_uint32, 0]
  fw_version: Annotated[ctypes.c_uint32, 4]
  offset_bytes: Annotated[ctypes.c_uint32, 8]
  size_bytes: Annotated[ctypes.c_uint32, 12]
enum_psp_fw_type = CEnum(ctypes.c_uint32)
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

@record
class struct_psp_firmware_header_v2_0:
  SIZE = 52
  header: Annotated[struct_common_firmware_header, 0]
  psp_fw_bin_count: Annotated[ctypes.c_uint32, 32]
  psp_fw_bin: Annotated[(struct_psp_fw_bin_desc* 1), 36]
@record
class struct_psp_firmware_header_v2_1:
  SIZE = 56
  header: Annotated[struct_common_firmware_header, 0]
  psp_fw_bin_count: Annotated[ctypes.c_uint32, 32]
  psp_aux_fw_bin_index: Annotated[ctypes.c_uint32, 36]
  psp_fw_bin: Annotated[(struct_psp_fw_bin_desc* 1), 40]
@record
class struct_ta_firmware_header_v1_0:
  SIZE = 92
  header: Annotated[struct_common_firmware_header, 0]
  xgmi: Annotated[struct_psp_fw_legacy_bin_desc, 32]
  ras: Annotated[struct_psp_fw_legacy_bin_desc, 44]
  hdcp: Annotated[struct_psp_fw_legacy_bin_desc, 56]
  dtm: Annotated[struct_psp_fw_legacy_bin_desc, 68]
  securedisplay: Annotated[struct_psp_fw_legacy_bin_desc, 80]
enum_ta_fw_type = CEnum(ctypes.c_uint32)
TA_FW_TYPE_UNKOWN = enum_ta_fw_type.define('TA_FW_TYPE_UNKOWN', 0)
TA_FW_TYPE_PSP_ASD = enum_ta_fw_type.define('TA_FW_TYPE_PSP_ASD', 1)
TA_FW_TYPE_PSP_XGMI = enum_ta_fw_type.define('TA_FW_TYPE_PSP_XGMI', 2)
TA_FW_TYPE_PSP_RAS = enum_ta_fw_type.define('TA_FW_TYPE_PSP_RAS', 3)
TA_FW_TYPE_PSP_HDCP = enum_ta_fw_type.define('TA_FW_TYPE_PSP_HDCP', 4)
TA_FW_TYPE_PSP_DTM = enum_ta_fw_type.define('TA_FW_TYPE_PSP_DTM', 5)
TA_FW_TYPE_PSP_RAP = enum_ta_fw_type.define('TA_FW_TYPE_PSP_RAP', 6)
TA_FW_TYPE_PSP_SECUREDISPLAY = enum_ta_fw_type.define('TA_FW_TYPE_PSP_SECUREDISPLAY', 7)
TA_FW_TYPE_MAX_INDEX = enum_ta_fw_type.define('TA_FW_TYPE_MAX_INDEX', 8)

@record
class struct_ta_firmware_header_v2_0:
  SIZE = 52
  header: Annotated[struct_common_firmware_header, 0]
  ta_fw_bin_count: Annotated[ctypes.c_uint32, 32]
  ta_fw_bin: Annotated[(struct_psp_fw_bin_desc* 1), 36]
@record
class struct_gfx_firmware_header_v1_0:
  SIZE = 44
  header: Annotated[struct_common_firmware_header, 0]
  ucode_feature_version: Annotated[ctypes.c_uint32, 32]
  jt_offset: Annotated[ctypes.c_uint32, 36]
  jt_size: Annotated[ctypes.c_uint32, 40]
@record
class struct_gfx_firmware_header_v2_0:
  SIZE = 60
  header: Annotated[struct_common_firmware_header, 0]
  ucode_feature_version: Annotated[ctypes.c_uint32, 32]
  ucode_size_bytes: Annotated[ctypes.c_uint32, 36]
  ucode_offset_bytes: Annotated[ctypes.c_uint32, 40]
  data_size_bytes: Annotated[ctypes.c_uint32, 44]
  data_offset_bytes: Annotated[ctypes.c_uint32, 48]
  ucode_start_addr_lo: Annotated[ctypes.c_uint32, 52]
  ucode_start_addr_hi: Annotated[ctypes.c_uint32, 56]
@record
class struct_mes_firmware_header_v1_0:
  SIZE = 72
  header: Annotated[struct_common_firmware_header, 0]
  mes_ucode_version: Annotated[ctypes.c_uint32, 32]
  mes_ucode_size_bytes: Annotated[ctypes.c_uint32, 36]
  mes_ucode_offset_bytes: Annotated[ctypes.c_uint32, 40]
  mes_ucode_data_version: Annotated[ctypes.c_uint32, 44]
  mes_ucode_data_size_bytes: Annotated[ctypes.c_uint32, 48]
  mes_ucode_data_offset_bytes: Annotated[ctypes.c_uint32, 52]
  mes_uc_start_addr_lo: Annotated[ctypes.c_uint32, 56]
  mes_uc_start_addr_hi: Annotated[ctypes.c_uint32, 60]
  mes_data_start_addr_lo: Annotated[ctypes.c_uint32, 64]
  mes_data_start_addr_hi: Annotated[ctypes.c_uint32, 68]
@record
class struct_rlc_firmware_header_v1_0:
  SIZE = 52
  header: Annotated[struct_common_firmware_header, 0]
  ucode_feature_version: Annotated[ctypes.c_uint32, 32]
  save_and_restore_offset: Annotated[ctypes.c_uint32, 36]
  clear_state_descriptor_offset: Annotated[ctypes.c_uint32, 40]
  avail_scratch_ram_locations: Annotated[ctypes.c_uint32, 44]
  master_pkt_description_offset: Annotated[ctypes.c_uint32, 48]
@record
class struct_rlc_firmware_header_v2_0:
  SIZE = 104
  header: Annotated[struct_common_firmware_header, 0]
  ucode_feature_version: Annotated[ctypes.c_uint32, 32]
  jt_offset: Annotated[ctypes.c_uint32, 36]
  jt_size: Annotated[ctypes.c_uint32, 40]
  save_and_restore_offset: Annotated[ctypes.c_uint32, 44]
  clear_state_descriptor_offset: Annotated[ctypes.c_uint32, 48]
  avail_scratch_ram_locations: Annotated[ctypes.c_uint32, 52]
  reg_restore_list_size: Annotated[ctypes.c_uint32, 56]
  reg_list_format_start: Annotated[ctypes.c_uint32, 60]
  reg_list_format_separate_start: Annotated[ctypes.c_uint32, 64]
  starting_offsets_start: Annotated[ctypes.c_uint32, 68]
  reg_list_format_size_bytes: Annotated[ctypes.c_uint32, 72]
  reg_list_format_array_offset_bytes: Annotated[ctypes.c_uint32, 76]
  reg_list_size_bytes: Annotated[ctypes.c_uint32, 80]
  reg_list_array_offset_bytes: Annotated[ctypes.c_uint32, 84]
  reg_list_format_separate_size_bytes: Annotated[ctypes.c_uint32, 88]
  reg_list_format_separate_array_offset_bytes: Annotated[ctypes.c_uint32, 92]
  reg_list_separate_size_bytes: Annotated[ctypes.c_uint32, 96]
  reg_list_separate_array_offset_bytes: Annotated[ctypes.c_uint32, 100]
@record
class struct_rlc_firmware_header_v2_1:
  SIZE = 156
  v2_0: Annotated[struct_rlc_firmware_header_v2_0, 0]
  reg_list_format_direct_reg_list_length: Annotated[ctypes.c_uint32, 104]
  save_restore_list_cntl_ucode_ver: Annotated[ctypes.c_uint32, 108]
  save_restore_list_cntl_feature_ver: Annotated[ctypes.c_uint32, 112]
  save_restore_list_cntl_size_bytes: Annotated[ctypes.c_uint32, 116]
  save_restore_list_cntl_offset_bytes: Annotated[ctypes.c_uint32, 120]
  save_restore_list_gpm_ucode_ver: Annotated[ctypes.c_uint32, 124]
  save_restore_list_gpm_feature_ver: Annotated[ctypes.c_uint32, 128]
  save_restore_list_gpm_size_bytes: Annotated[ctypes.c_uint32, 132]
  save_restore_list_gpm_offset_bytes: Annotated[ctypes.c_uint32, 136]
  save_restore_list_srm_ucode_ver: Annotated[ctypes.c_uint32, 140]
  save_restore_list_srm_feature_ver: Annotated[ctypes.c_uint32, 144]
  save_restore_list_srm_size_bytes: Annotated[ctypes.c_uint32, 148]
  save_restore_list_srm_offset_bytes: Annotated[ctypes.c_uint32, 152]
@record
class struct_rlc_firmware_header_v2_2:
  SIZE = 172
  v2_1: Annotated[struct_rlc_firmware_header_v2_1, 0]
  rlc_iram_ucode_size_bytes: Annotated[ctypes.c_uint32, 156]
  rlc_iram_ucode_offset_bytes: Annotated[ctypes.c_uint32, 160]
  rlc_dram_ucode_size_bytes: Annotated[ctypes.c_uint32, 164]
  rlc_dram_ucode_offset_bytes: Annotated[ctypes.c_uint32, 168]
@record
class struct_rlc_firmware_header_v2_3:
  SIZE = 204
  v2_2: Annotated[struct_rlc_firmware_header_v2_2, 0]
  rlcp_ucode_version: Annotated[ctypes.c_uint32, 172]
  rlcp_ucode_feature_version: Annotated[ctypes.c_uint32, 176]
  rlcp_ucode_size_bytes: Annotated[ctypes.c_uint32, 180]
  rlcp_ucode_offset_bytes: Annotated[ctypes.c_uint32, 184]
  rlcv_ucode_version: Annotated[ctypes.c_uint32, 188]
  rlcv_ucode_feature_version: Annotated[ctypes.c_uint32, 192]
  rlcv_ucode_size_bytes: Annotated[ctypes.c_uint32, 196]
  rlcv_ucode_offset_bytes: Annotated[ctypes.c_uint32, 200]
@record
class struct_rlc_firmware_header_v2_4:
  SIZE = 244
  v2_3: Annotated[struct_rlc_firmware_header_v2_3, 0]
  global_tap_delays_ucode_size_bytes: Annotated[ctypes.c_uint32, 204]
  global_tap_delays_ucode_offset_bytes: Annotated[ctypes.c_uint32, 208]
  se0_tap_delays_ucode_size_bytes: Annotated[ctypes.c_uint32, 212]
  se0_tap_delays_ucode_offset_bytes: Annotated[ctypes.c_uint32, 216]
  se1_tap_delays_ucode_size_bytes: Annotated[ctypes.c_uint32, 220]
  se1_tap_delays_ucode_offset_bytes: Annotated[ctypes.c_uint32, 224]
  se2_tap_delays_ucode_size_bytes: Annotated[ctypes.c_uint32, 228]
  se2_tap_delays_ucode_offset_bytes: Annotated[ctypes.c_uint32, 232]
  se3_tap_delays_ucode_size_bytes: Annotated[ctypes.c_uint32, 236]
  se3_tap_delays_ucode_offset_bytes: Annotated[ctypes.c_uint32, 240]
@record
class struct_sdma_firmware_header_v1_0:
  SIZE = 48
  header: Annotated[struct_common_firmware_header, 0]
  ucode_feature_version: Annotated[ctypes.c_uint32, 32]
  ucode_change_version: Annotated[ctypes.c_uint32, 36]
  jt_offset: Annotated[ctypes.c_uint32, 40]
  jt_size: Annotated[ctypes.c_uint32, 44]
@record
class struct_sdma_firmware_header_v1_1:
  SIZE = 52
  v1_0: Annotated[struct_sdma_firmware_header_v1_0, 0]
  digest_size: Annotated[ctypes.c_uint32, 48]
@record
class struct_sdma_firmware_header_v2_0:
  SIZE = 64
  header: Annotated[struct_common_firmware_header, 0]
  ucode_feature_version: Annotated[ctypes.c_uint32, 32]
  ctx_ucode_size_bytes: Annotated[ctypes.c_uint32, 36]
  ctx_jt_offset: Annotated[ctypes.c_uint32, 40]
  ctx_jt_size: Annotated[ctypes.c_uint32, 44]
  ctl_ucode_offset: Annotated[ctypes.c_uint32, 48]
  ctl_ucode_size_bytes: Annotated[ctypes.c_uint32, 52]
  ctl_jt_offset: Annotated[ctypes.c_uint32, 56]
  ctl_jt_size: Annotated[ctypes.c_uint32, 60]
@record
class struct_vpe_firmware_header_v1_0:
  SIZE = 64
  header: Annotated[struct_common_firmware_header, 0]
  ucode_feature_version: Annotated[ctypes.c_uint32, 32]
  ctx_ucode_size_bytes: Annotated[ctypes.c_uint32, 36]
  ctx_jt_offset: Annotated[ctypes.c_uint32, 40]
  ctx_jt_size: Annotated[ctypes.c_uint32, 44]
  ctl_ucode_offset: Annotated[ctypes.c_uint32, 48]
  ctl_ucode_size_bytes: Annotated[ctypes.c_uint32, 52]
  ctl_jt_offset: Annotated[ctypes.c_uint32, 56]
  ctl_jt_size: Annotated[ctypes.c_uint32, 60]
@record
class struct_umsch_mm_firmware_header_v1_0:
  SIZE = 80
  header: Annotated[struct_common_firmware_header, 0]
  umsch_mm_ucode_version: Annotated[ctypes.c_uint32, 32]
  umsch_mm_ucode_size_bytes: Annotated[ctypes.c_uint32, 36]
  umsch_mm_ucode_offset_bytes: Annotated[ctypes.c_uint32, 40]
  umsch_mm_ucode_data_version: Annotated[ctypes.c_uint32, 44]
  umsch_mm_ucode_data_size_bytes: Annotated[ctypes.c_uint32, 48]
  umsch_mm_ucode_data_offset_bytes: Annotated[ctypes.c_uint32, 52]
  umsch_mm_irq_start_addr_lo: Annotated[ctypes.c_uint32, 56]
  umsch_mm_irq_start_addr_hi: Annotated[ctypes.c_uint32, 60]
  umsch_mm_uc_start_addr_lo: Annotated[ctypes.c_uint32, 64]
  umsch_mm_uc_start_addr_hi: Annotated[ctypes.c_uint32, 68]
  umsch_mm_data_start_addr_lo: Annotated[ctypes.c_uint32, 72]
  umsch_mm_data_start_addr_hi: Annotated[ctypes.c_uint32, 76]
@record
class struct_sdma_firmware_header_v3_0:
  SIZE = 44
  header: Annotated[struct_common_firmware_header, 0]
  ucode_feature_version: Annotated[ctypes.c_uint32, 32]
  ucode_offset_bytes: Annotated[ctypes.c_uint32, 36]
  ucode_size_bytes: Annotated[ctypes.c_uint32, 40]
@record
class struct_gpu_info_firmware_v1_0:
  SIZE = 60
  gc_num_se: Annotated[ctypes.c_uint32, 0]
  gc_num_cu_per_sh: Annotated[ctypes.c_uint32, 4]
  gc_num_sh_per_se: Annotated[ctypes.c_uint32, 8]
  gc_num_rb_per_se: Annotated[ctypes.c_uint32, 12]
  gc_num_tccs: Annotated[ctypes.c_uint32, 16]
  gc_num_gprs: Annotated[ctypes.c_uint32, 20]
  gc_num_max_gs_thds: Annotated[ctypes.c_uint32, 24]
  gc_gs_table_depth: Annotated[ctypes.c_uint32, 28]
  gc_gsprim_buff_depth: Annotated[ctypes.c_uint32, 32]
  gc_parameter_cache_depth: Annotated[ctypes.c_uint32, 36]
  gc_double_offchip_lds_buffer: Annotated[ctypes.c_uint32, 40]
  gc_wave_size: Annotated[ctypes.c_uint32, 44]
  gc_max_waves_per_simd: Annotated[ctypes.c_uint32, 48]
  gc_max_scratch_slots_per_cu: Annotated[ctypes.c_uint32, 52]
  gc_lds_size: Annotated[ctypes.c_uint32, 56]
@record
class struct_gpu_info_firmware_v1_1:
  SIZE = 68
  v1_0: Annotated[struct_gpu_info_firmware_v1_0, 0]
  num_sc_per_sh: Annotated[ctypes.c_uint32, 60]
  num_packer_per_sc: Annotated[ctypes.c_uint32, 64]
@record
class struct_gpu_info_firmware_header_v1_0:
  SIZE = 36
  header: Annotated[struct_common_firmware_header, 0]
  version_major: Annotated[ctypes.c_uint16, 32]
  version_minor: Annotated[ctypes.c_uint16, 34]
@record
class struct_dmcu_firmware_header_v1_0:
  SIZE = 40
  header: Annotated[struct_common_firmware_header, 0]
  intv_offset_bytes: Annotated[ctypes.c_uint32, 32]
  intv_size_bytes: Annotated[ctypes.c_uint32, 36]
@record
class struct_dmcub_firmware_header_v1_0:
  SIZE = 40
  header: Annotated[struct_common_firmware_header, 0]
  inst_const_bytes: Annotated[ctypes.c_uint32, 32]
  bss_data_bytes: Annotated[ctypes.c_uint32, 36]
@record
class struct_imu_firmware_header_v1_0:
  SIZE = 48
  header: Annotated[struct_common_firmware_header, 0]
  imu_iram_ucode_size_bytes: Annotated[ctypes.c_uint32, 32]
  imu_iram_ucode_offset_bytes: Annotated[ctypes.c_uint32, 36]
  imu_dram_ucode_size_bytes: Annotated[ctypes.c_uint32, 40]
  imu_dram_ucode_offset_bytes: Annotated[ctypes.c_uint32, 44]
@record
class union_amdgpu_firmware_header:
  SIZE = 256
  common: Annotated[struct_common_firmware_header, 0]
  mc: Annotated[struct_mc_firmware_header_v1_0, 0]
  smc: Annotated[struct_smc_firmware_header_v1_0, 0]
  smc_v2_0: Annotated[struct_smc_firmware_header_v2_0, 0]
  psp: Annotated[struct_psp_firmware_header_v1_0, 0]
  psp_v1_1: Annotated[struct_psp_firmware_header_v1_1, 0]
  psp_v1_3: Annotated[struct_psp_firmware_header_v1_3, 0]
  psp_v2_0: Annotated[struct_psp_firmware_header_v2_0, 0]
  psp_v2_1: Annotated[struct_psp_firmware_header_v2_0, 0]
  ta: Annotated[struct_ta_firmware_header_v1_0, 0]
  ta_v2_0: Annotated[struct_ta_firmware_header_v2_0, 0]
  gfx: Annotated[struct_gfx_firmware_header_v1_0, 0]
  gfx_v2_0: Annotated[struct_gfx_firmware_header_v2_0, 0]
  rlc: Annotated[struct_rlc_firmware_header_v1_0, 0]
  rlc_v2_0: Annotated[struct_rlc_firmware_header_v2_0, 0]
  rlc_v2_1: Annotated[struct_rlc_firmware_header_v2_1, 0]
  rlc_v2_2: Annotated[struct_rlc_firmware_header_v2_2, 0]
  rlc_v2_3: Annotated[struct_rlc_firmware_header_v2_3, 0]
  rlc_v2_4: Annotated[struct_rlc_firmware_header_v2_4, 0]
  sdma: Annotated[struct_sdma_firmware_header_v1_0, 0]
  sdma_v1_1: Annotated[struct_sdma_firmware_header_v1_1, 0]
  sdma_v2_0: Annotated[struct_sdma_firmware_header_v2_0, 0]
  sdma_v3_0: Annotated[struct_sdma_firmware_header_v3_0, 0]
  gpu_info: Annotated[struct_gpu_info_firmware_header_v1_0, 0]
  dmcu: Annotated[struct_dmcu_firmware_header_v1_0, 0]
  dmcub: Annotated[struct_dmcub_firmware_header_v1_0, 0]
  imu: Annotated[struct_imu_firmware_header_v1_0, 0]
  raw: Annotated[(ctypes.c_ubyte* 256), 0]
enum_AMDGPU_UCODE_ID = CEnum(ctypes.c_uint32)
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

enum_AMDGPU_UCODE_STATUS = CEnum(ctypes.c_uint32)
AMDGPU_UCODE_STATUS_INVALID = enum_AMDGPU_UCODE_STATUS.define('AMDGPU_UCODE_STATUS_INVALID', 0)
AMDGPU_UCODE_STATUS_NOT_LOADED = enum_AMDGPU_UCODE_STATUS.define('AMDGPU_UCODE_STATUS_NOT_LOADED', 1)
AMDGPU_UCODE_STATUS_LOADED = enum_AMDGPU_UCODE_STATUS.define('AMDGPU_UCODE_STATUS_LOADED', 2)

enum_amdgpu_firmware_load_type = CEnum(ctypes.c_uint32)
AMDGPU_FW_LOAD_DIRECT = enum_amdgpu_firmware_load_type.define('AMDGPU_FW_LOAD_DIRECT', 0)
AMDGPU_FW_LOAD_PSP = enum_amdgpu_firmware_load_type.define('AMDGPU_FW_LOAD_PSP', 1)
AMDGPU_FW_LOAD_SMU = enum_amdgpu_firmware_load_type.define('AMDGPU_FW_LOAD_SMU', 2)
AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO = enum_amdgpu_firmware_load_type.define('AMDGPU_FW_LOAD_RLC_BACKDOOR_AUTO', 3)

@record
class struct_amdgpu_firmware_info:
  SIZE = 48
  ucode_id: Annotated[enum_AMDGPU_UCODE_ID, 0]
  fw: Annotated[ctypes.POINTER(struct_firmware), 8]
  mc_addr: Annotated[ctypes.c_uint64, 16]
  kaddr: Annotated[ctypes.POINTER(None), 24]
  ucode_size: Annotated[ctypes.c_uint32, 32]
  tmr_mc_addr_lo: Annotated[ctypes.c_uint32, 36]
  tmr_mc_addr_hi: Annotated[ctypes.c_uint32, 40]
class struct_firmware(ctypes.Structure): pass
enum_psp_gfx_crtl_cmd_id = CEnum(ctypes.c_uint32)
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

@record
class struct_psp_gfx_ctrl:
  SIZE = 32
  cmd_resp: Annotated[ctypes.c_uint32, 0]
  rbi_wptr: Annotated[ctypes.c_uint32, 4]
  rbi_rptr: Annotated[ctypes.c_uint32, 8]
  gpcom_wptr: Annotated[ctypes.c_uint32, 12]
  gpcom_rptr: Annotated[ctypes.c_uint32, 16]
  ring_addr_lo: Annotated[ctypes.c_uint32, 20]
  ring_addr_hi: Annotated[ctypes.c_uint32, 24]
  ring_buf_size: Annotated[ctypes.c_uint32, 28]
enum_psp_gfx_cmd_id = CEnum(ctypes.c_uint32)
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

enum_psp_gfx_boot_config_cmd = CEnum(ctypes.c_uint32)
BOOTCFG_CMD_SET = enum_psp_gfx_boot_config_cmd.define('BOOTCFG_CMD_SET', 1)
BOOTCFG_CMD_GET = enum_psp_gfx_boot_config_cmd.define('BOOTCFG_CMD_GET', 2)
BOOTCFG_CMD_INVALIDATE = enum_psp_gfx_boot_config_cmd.define('BOOTCFG_CMD_INVALIDATE', 3)

enum_psp_gfx_boot_config = CEnum(ctypes.c_uint32)
BOOT_CONFIG_GECC = enum_psp_gfx_boot_config.define('BOOT_CONFIG_GECC', 1)

@record
class struct_psp_gfx_cmd_load_ta:
  SIZE = 24
  app_phy_addr_lo: Annotated[ctypes.c_uint32, 0]
  app_phy_addr_hi: Annotated[ctypes.c_uint32, 4]
  app_len: Annotated[ctypes.c_uint32, 8]
  cmd_buf_phy_addr_lo: Annotated[ctypes.c_uint32, 12]
  cmd_buf_phy_addr_hi: Annotated[ctypes.c_uint32, 16]
  cmd_buf_len: Annotated[ctypes.c_uint32, 20]
@record
class struct_psp_gfx_cmd_unload_ta:
  SIZE = 4
  session_id: Annotated[ctypes.c_uint32, 0]
@record
class struct_psp_gfx_buf_desc:
  SIZE = 12
  buf_phy_addr_lo: Annotated[ctypes.c_uint32, 0]
  buf_phy_addr_hi: Annotated[ctypes.c_uint32, 4]
  buf_size: Annotated[ctypes.c_uint32, 8]
@record
class struct_psp_gfx_buf_list:
  SIZE = 776
  num_desc: Annotated[ctypes.c_uint32, 0]
  total_size: Annotated[ctypes.c_uint32, 4]
  buf_desc: Annotated[(struct_psp_gfx_buf_desc* 64), 8]
@record
class struct_psp_gfx_cmd_invoke_cmd:
  SIZE = 784
  session_id: Annotated[ctypes.c_uint32, 0]
  ta_cmd_id: Annotated[ctypes.c_uint32, 4]
  buf: Annotated[struct_psp_gfx_buf_list, 8]
@record
class struct_psp_gfx_cmd_setup_tmr:
  SIZE = 24
  buf_phy_addr_lo: Annotated[ctypes.c_uint32, 0]
  buf_phy_addr_hi: Annotated[ctypes.c_uint32, 4]
  buf_size: Annotated[ctypes.c_uint32, 8]
  bitfield: Annotated[_anonstruct1, 12]
  tmr_flags: Annotated[ctypes.c_uint32, 12]
  system_phy_addr_lo: Annotated[ctypes.c_uint32, 16]
  system_phy_addr_hi: Annotated[ctypes.c_uint32, 20]
@record
class _anonstruct1:
  SIZE = 4
  sriov_enabled: Annotated[ctypes.c_uint32, 0, 1, 0]
  virt_phy_addr: Annotated[ctypes.c_uint32, 0, 1, 1]
  reserved: Annotated[ctypes.c_uint32, 0, 30, 2]
enum_psp_gfx_fw_type = CEnum(ctypes.c_uint32)
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

@record
class struct_psp_gfx_cmd_load_ip_fw:
  SIZE = 16
  fw_phy_addr_lo: Annotated[ctypes.c_uint32, 0]
  fw_phy_addr_hi: Annotated[ctypes.c_uint32, 4]
  fw_size: Annotated[ctypes.c_uint32, 8]
  fw_type: Annotated[enum_psp_gfx_fw_type, 12]
@record
class struct_psp_gfx_cmd_save_restore_ip_fw:
  SIZE = 20
  save_fw: Annotated[ctypes.c_uint32, 0]
  save_restore_addr_lo: Annotated[ctypes.c_uint32, 4]
  save_restore_addr_hi: Annotated[ctypes.c_uint32, 8]
  buf_size: Annotated[ctypes.c_uint32, 12]
  fw_type: Annotated[enum_psp_gfx_fw_type, 16]
@record
class struct_psp_gfx_cmd_reg_prog:
  SIZE = 8
  reg_value: Annotated[ctypes.c_uint32, 0]
  reg_id: Annotated[ctypes.c_uint32, 4]
@record
class struct_psp_gfx_cmd_load_toc:
  SIZE = 12
  toc_phy_addr_lo: Annotated[ctypes.c_uint32, 0]
  toc_phy_addr_hi: Annotated[ctypes.c_uint32, 4]
  toc_size: Annotated[ctypes.c_uint32, 8]
@record
class struct_psp_gfx_cmd_boot_cfg:
  SIZE = 16
  timestamp: Annotated[ctypes.c_uint32, 0]
  sub_cmd: Annotated[enum_psp_gfx_boot_config_cmd, 4]
  boot_config: Annotated[ctypes.c_uint32, 8]
  boot_config_valid: Annotated[ctypes.c_uint32, 12]
@record
class struct_psp_gfx_cmd_sriov_spatial_part:
  SIZE = 16
  mode: Annotated[ctypes.c_uint32, 0]
  override_ips: Annotated[ctypes.c_uint32, 4]
  override_xcds_avail: Annotated[ctypes.c_uint32, 8]
  override_this_aid: Annotated[ctypes.c_uint32, 12]
@record
class union_psp_gfx_commands:
  SIZE = 784
  cmd_load_ta: Annotated[struct_psp_gfx_cmd_load_ta, 0]
  cmd_unload_ta: Annotated[struct_psp_gfx_cmd_unload_ta, 0]
  cmd_invoke_cmd: Annotated[struct_psp_gfx_cmd_invoke_cmd, 0]
  cmd_setup_tmr: Annotated[struct_psp_gfx_cmd_setup_tmr, 0]
  cmd_load_ip_fw: Annotated[struct_psp_gfx_cmd_load_ip_fw, 0]
  cmd_save_restore_ip_fw: Annotated[struct_psp_gfx_cmd_save_restore_ip_fw, 0]
  cmd_setup_reg_prog: Annotated[struct_psp_gfx_cmd_reg_prog, 0]
  cmd_setup_vmr: Annotated[struct_psp_gfx_cmd_setup_tmr, 0]
  cmd_load_toc: Annotated[struct_psp_gfx_cmd_load_toc, 0]
  boot_cfg: Annotated[struct_psp_gfx_cmd_boot_cfg, 0]
  cmd_spatial_part: Annotated[struct_psp_gfx_cmd_sriov_spatial_part, 0]
@record
class struct_psp_gfx_uresp_reserved:
  SIZE = 32
  reserved: Annotated[(ctypes.c_uint32* 8), 0]
@record
class struct_psp_gfx_uresp_fwar_db_info:
  SIZE = 8
  fwar_db_addr_lo: Annotated[ctypes.c_uint32, 0]
  fwar_db_addr_hi: Annotated[ctypes.c_uint32, 4]
@record
class struct_psp_gfx_uresp_bootcfg:
  SIZE = 4
  boot_cfg: Annotated[ctypes.c_uint32, 0]
@record
class union_psp_gfx_uresp:
  SIZE = 32
  reserved: Annotated[struct_psp_gfx_uresp_reserved, 0]
  boot_cfg: Annotated[struct_psp_gfx_uresp_bootcfg, 0]
  fwar_db_info: Annotated[struct_psp_gfx_uresp_fwar_db_info, 0]
@record
class struct_psp_gfx_resp:
  SIZE = 96
  status: Annotated[ctypes.c_uint32, 0]
  session_id: Annotated[ctypes.c_uint32, 4]
  fw_addr_lo: Annotated[ctypes.c_uint32, 8]
  fw_addr_hi: Annotated[ctypes.c_uint32, 12]
  tmr_size: Annotated[ctypes.c_uint32, 16]
  reserved: Annotated[(ctypes.c_uint32* 11), 20]
  uresp: Annotated[union_psp_gfx_uresp, 64]
@record
class struct_psp_gfx_cmd_resp:
  SIZE = 1024
  buf_size: Annotated[ctypes.c_uint32, 0]
  buf_version: Annotated[ctypes.c_uint32, 4]
  cmd_id: Annotated[ctypes.c_uint32, 8]
  resp_buf_addr_lo: Annotated[ctypes.c_uint32, 12]
  resp_buf_addr_hi: Annotated[ctypes.c_uint32, 16]
  resp_offset: Annotated[ctypes.c_uint32, 20]
  resp_buf_size: Annotated[ctypes.c_uint32, 24]
  cmd: Annotated[union_psp_gfx_commands, 28]
  reserved_1: Annotated[(ctypes.c_ubyte* 52), 812]
  resp: Annotated[struct_psp_gfx_resp, 864]
  reserved_2: Annotated[(ctypes.c_ubyte* 64), 960]
@record
class struct_psp_gfx_rb_frame:
  SIZE = 64
  cmd_buf_addr_lo: Annotated[ctypes.c_uint32, 0]
  cmd_buf_addr_hi: Annotated[ctypes.c_uint32, 4]
  cmd_buf_size: Annotated[ctypes.c_uint32, 8]
  fence_addr_lo: Annotated[ctypes.c_uint32, 12]
  fence_addr_hi: Annotated[ctypes.c_uint32, 16]
  fence_value: Annotated[ctypes.c_uint32, 20]
  sid_lo: Annotated[ctypes.c_uint32, 24]
  sid_hi: Annotated[ctypes.c_uint32, 28]
  vmid: Annotated[ctypes.c_ubyte, 32]
  frame_type: Annotated[ctypes.c_ubyte, 33]
  reserved1: Annotated[(ctypes.c_ubyte* 2), 34]
  reserved2: Annotated[(ctypes.c_uint32* 7), 36]
enum_tee_error_code = CEnum(ctypes.c_uint32)
TEE_SUCCESS = enum_tee_error_code.define('TEE_SUCCESS', 0)
TEE_ERROR_NOT_SUPPORTED = enum_tee_error_code.define('TEE_ERROR_NOT_SUPPORTED', 4294901770)

enum_psp_shared_mem_size = CEnum(ctypes.c_uint32)
PSP_ASD_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_ASD_SHARED_MEM_SIZE', 0)
PSP_XGMI_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_XGMI_SHARED_MEM_SIZE', 16384)
PSP_RAS_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_RAS_SHARED_MEM_SIZE', 16384)
PSP_HDCP_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_HDCP_SHARED_MEM_SIZE', 16384)
PSP_DTM_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_DTM_SHARED_MEM_SIZE', 16384)
PSP_RAP_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_RAP_SHARED_MEM_SIZE', 16384)
PSP_SECUREDISPLAY_SHARED_MEM_SIZE = enum_psp_shared_mem_size.define('PSP_SECUREDISPLAY_SHARED_MEM_SIZE', 16384)

enum_ta_type_id = CEnum(ctypes.c_uint32)
TA_TYPE_XGMI = enum_ta_type_id.define('TA_TYPE_XGMI', 1)
TA_TYPE_RAS = enum_ta_type_id.define('TA_TYPE_RAS', 2)
TA_TYPE_HDCP = enum_ta_type_id.define('TA_TYPE_HDCP', 3)
TA_TYPE_DTM = enum_ta_type_id.define('TA_TYPE_DTM', 4)
TA_TYPE_RAP = enum_ta_type_id.define('TA_TYPE_RAP', 5)
TA_TYPE_SECUREDISPLAY = enum_ta_type_id.define('TA_TYPE_SECUREDISPLAY', 6)
TA_TYPE_MAX_INDEX = enum_ta_type_id.define('TA_TYPE_MAX_INDEX', 7)

class struct_psp_context(ctypes.Structure): pass
class struct_psp_xgmi_node_info(ctypes.Structure): pass
class struct_psp_xgmi_topology_info(ctypes.Structure): pass
class struct_psp_bin_desc(ctypes.Structure): pass
enum_psp_bootloader_cmd = CEnum(ctypes.c_uint32)
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

enum_psp_ring_type = CEnum(ctypes.c_uint32)
PSP_RING_TYPE__INVALID = enum_psp_ring_type.define('PSP_RING_TYPE__INVALID', 0)
PSP_RING_TYPE__UM = enum_psp_ring_type.define('PSP_RING_TYPE__UM', 1)
PSP_RING_TYPE__KM = enum_psp_ring_type.define('PSP_RING_TYPE__KM', 2)

enum_psp_reg_prog_id = CEnum(ctypes.c_uint32)
PSP_REG_IH_RB_CNTL = enum_psp_reg_prog_id.define('PSP_REG_IH_RB_CNTL', 0)
PSP_REG_IH_RB_CNTL_RING1 = enum_psp_reg_prog_id.define('PSP_REG_IH_RB_CNTL_RING1', 1)
PSP_REG_IH_RB_CNTL_RING2 = enum_psp_reg_prog_id.define('PSP_REG_IH_RB_CNTL_RING2', 2)
PSP_REG_LAST = enum_psp_reg_prog_id.define('PSP_REG_LAST', 3)

enum_psp_memory_training_init_flag = CEnum(ctypes.c_uint32)
PSP_MEM_TRAIN_NOT_SUPPORT = enum_psp_memory_training_init_flag.define('PSP_MEM_TRAIN_NOT_SUPPORT', 0)
PSP_MEM_TRAIN_SUPPORT = enum_psp_memory_training_init_flag.define('PSP_MEM_TRAIN_SUPPORT', 1)
PSP_MEM_TRAIN_INIT_FAILED = enum_psp_memory_training_init_flag.define('PSP_MEM_TRAIN_INIT_FAILED', 2)
PSP_MEM_TRAIN_RESERVE_SUCCESS = enum_psp_memory_training_init_flag.define('PSP_MEM_TRAIN_RESERVE_SUCCESS', 4)
PSP_MEM_TRAIN_INIT_SUCCESS = enum_psp_memory_training_init_flag.define('PSP_MEM_TRAIN_INIT_SUCCESS', 8)

enum_psp_memory_training_ops = CEnum(ctypes.c_uint32)
PSP_MEM_TRAIN_SEND_LONG_MSG = enum_psp_memory_training_ops.define('PSP_MEM_TRAIN_SEND_LONG_MSG', 1)
PSP_MEM_TRAIN_SAVE = enum_psp_memory_training_ops.define('PSP_MEM_TRAIN_SAVE', 2)
PSP_MEM_TRAIN_RESTORE = enum_psp_memory_training_ops.define('PSP_MEM_TRAIN_RESTORE', 4)
PSP_MEM_TRAIN_SEND_SHORT_MSG = enum_psp_memory_training_ops.define('PSP_MEM_TRAIN_SEND_SHORT_MSG', 8)
PSP_MEM_TRAIN_COLD_BOOT = enum_psp_memory_training_ops.define('PSP_MEM_TRAIN_COLD_BOOT', 1)
PSP_MEM_TRAIN_RESUME = enum_psp_memory_training_ops.define('PSP_MEM_TRAIN_RESUME', 8)

enum_psp_runtime_entry_type = CEnum(ctypes.c_uint32)
PSP_RUNTIME_ENTRY_TYPE_INVALID = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_INVALID', 0)
PSP_RUNTIME_ENTRY_TYPE_TEST = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_TEST', 1)
PSP_RUNTIME_ENTRY_TYPE_MGPU_COMMON = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_MGPU_COMMON', 2)
PSP_RUNTIME_ENTRY_TYPE_MGPU_WAFL = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_MGPU_WAFL', 3)
PSP_RUNTIME_ENTRY_TYPE_MGPU_XGMI = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_MGPU_XGMI', 4)
PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG', 5)
PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS = enum_psp_runtime_entry_type.define('PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS', 6)

enum_psp_runtime_boot_cfg_feature = CEnum(ctypes.c_uint32)
BOOT_CFG_FEATURE_GECC = enum_psp_runtime_boot_cfg_feature.define('BOOT_CFG_FEATURE_GECC', 1)
BOOT_CFG_FEATURE_TWO_STAGE_DRAM_TRAINING = enum_psp_runtime_boot_cfg_feature.define('BOOT_CFG_FEATURE_TWO_STAGE_DRAM_TRAINING', 2)

enum_psp_runtime_scpm_authentication = CEnum(ctypes.c_uint32)
SCPM_DISABLE = enum_psp_runtime_scpm_authentication.define('SCPM_DISABLE', 0)
SCPM_ENABLE = enum_psp_runtime_scpm_authentication.define('SCPM_ENABLE', 1)
SCPM_ENABLE_WITH_SCPM_ERR = enum_psp_runtime_scpm_authentication.define('SCPM_ENABLE_WITH_SCPM_ERR', 2)

class struct_amdgpu_device(ctypes.Structure): pass
enum_amdgpu_interrupt_state = CEnum(ctypes.c_uint32)
AMDGPU_IRQ_STATE_DISABLE = enum_amdgpu_interrupt_state.define('AMDGPU_IRQ_STATE_DISABLE', 0)
AMDGPU_IRQ_STATE_ENABLE = enum_amdgpu_interrupt_state.define('AMDGPU_IRQ_STATE_ENABLE', 1)

@record
class struct_amdgpu_iv_entry:
  SIZE = 72
  client_id: Annotated[ctypes.c_uint32, 0]
  src_id: Annotated[ctypes.c_uint32, 4]
  ring_id: Annotated[ctypes.c_uint32, 8]
  vmid: Annotated[ctypes.c_uint32, 12]
  vmid_src: Annotated[ctypes.c_uint32, 16]
  timestamp: Annotated[ctypes.c_uint64, 24]
  timestamp_src: Annotated[ctypes.c_uint32, 32]
  pasid: Annotated[ctypes.c_uint32, 36]
  node_id: Annotated[ctypes.c_uint32, 40]
  src_data: Annotated[(ctypes.c_uint32* 4), 44]
  iv_entry: Annotated[ctypes.POINTER(ctypes.c_uint32), 64]
enum_interrupt_node_id_per_aid = CEnum(ctypes.c_uint32)
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

enum_AMDGPU_DOORBELL_ASSIGNMENT = CEnum(ctypes.c_uint32)
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

enum_AMDGPU_VEGA20_DOORBELL_ASSIGNMENT = CEnum(ctypes.c_uint32)
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

enum_AMDGPU_NAVI10_DOORBELL_ASSIGNMENT = CEnum(ctypes.c_uint32)
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

enum_AMDGPU_DOORBELL64_ASSIGNMENT = CEnum(ctypes.c_uint32)
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

enum_AMDGPU_DOORBELL_ASSIGNMENT_LAYOUT1 = CEnum(ctypes.c_uint32)
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

@record
class struct_v9_sdma_mqd:
  SIZE = 512
  sdmax_rlcx_rb_cntl: Annotated[uint32_t, 0]
  sdmax_rlcx_rb_base: Annotated[uint32_t, 4]
  sdmax_rlcx_rb_base_hi: Annotated[uint32_t, 8]
  sdmax_rlcx_rb_rptr: Annotated[uint32_t, 12]
  sdmax_rlcx_rb_rptr_hi: Annotated[uint32_t, 16]
  sdmax_rlcx_rb_wptr: Annotated[uint32_t, 20]
  sdmax_rlcx_rb_wptr_hi: Annotated[uint32_t, 24]
  sdmax_rlcx_rb_wptr_poll_cntl: Annotated[uint32_t, 28]
  sdmax_rlcx_rb_rptr_addr_hi: Annotated[uint32_t, 32]
  sdmax_rlcx_rb_rptr_addr_lo: Annotated[uint32_t, 36]
  sdmax_rlcx_ib_cntl: Annotated[uint32_t, 40]
  sdmax_rlcx_ib_rptr: Annotated[uint32_t, 44]
  sdmax_rlcx_ib_offset: Annotated[uint32_t, 48]
  sdmax_rlcx_ib_base_lo: Annotated[uint32_t, 52]
  sdmax_rlcx_ib_base_hi: Annotated[uint32_t, 56]
  sdmax_rlcx_ib_size: Annotated[uint32_t, 60]
  sdmax_rlcx_skip_cntl: Annotated[uint32_t, 64]
  sdmax_rlcx_context_status: Annotated[uint32_t, 68]
  sdmax_rlcx_doorbell: Annotated[uint32_t, 72]
  sdmax_rlcx_status: Annotated[uint32_t, 76]
  sdmax_rlcx_doorbell_log: Annotated[uint32_t, 80]
  sdmax_rlcx_watermark: Annotated[uint32_t, 84]
  sdmax_rlcx_doorbell_offset: Annotated[uint32_t, 88]
  sdmax_rlcx_csa_addr_lo: Annotated[uint32_t, 92]
  sdmax_rlcx_csa_addr_hi: Annotated[uint32_t, 96]
  sdmax_rlcx_ib_sub_remain: Annotated[uint32_t, 100]
  sdmax_rlcx_preempt: Annotated[uint32_t, 104]
  sdmax_rlcx_dummy_reg: Annotated[uint32_t, 108]
  sdmax_rlcx_rb_wptr_poll_addr_hi: Annotated[uint32_t, 112]
  sdmax_rlcx_rb_wptr_poll_addr_lo: Annotated[uint32_t, 116]
  sdmax_rlcx_rb_aql_cntl: Annotated[uint32_t, 120]
  sdmax_rlcx_minor_ptr_update: Annotated[uint32_t, 124]
  sdmax_rlcx_midcmd_data0: Annotated[uint32_t, 128]
  sdmax_rlcx_midcmd_data1: Annotated[uint32_t, 132]
  sdmax_rlcx_midcmd_data2: Annotated[uint32_t, 136]
  sdmax_rlcx_midcmd_data3: Annotated[uint32_t, 140]
  sdmax_rlcx_midcmd_data4: Annotated[uint32_t, 144]
  sdmax_rlcx_midcmd_data5: Annotated[uint32_t, 148]
  sdmax_rlcx_midcmd_data6: Annotated[uint32_t, 152]
  sdmax_rlcx_midcmd_data7: Annotated[uint32_t, 156]
  sdmax_rlcx_midcmd_data8: Annotated[uint32_t, 160]
  sdmax_rlcx_midcmd_cntl: Annotated[uint32_t, 164]
  reserved_42: Annotated[uint32_t, 168]
  reserved_43: Annotated[uint32_t, 172]
  reserved_44: Annotated[uint32_t, 176]
  reserved_45: Annotated[uint32_t, 180]
  reserved_46: Annotated[uint32_t, 184]
  reserved_47: Annotated[uint32_t, 188]
  reserved_48: Annotated[uint32_t, 192]
  reserved_49: Annotated[uint32_t, 196]
  reserved_50: Annotated[uint32_t, 200]
  reserved_51: Annotated[uint32_t, 204]
  reserved_52: Annotated[uint32_t, 208]
  reserved_53: Annotated[uint32_t, 212]
  reserved_54: Annotated[uint32_t, 216]
  reserved_55: Annotated[uint32_t, 220]
  reserved_56: Annotated[uint32_t, 224]
  reserved_57: Annotated[uint32_t, 228]
  reserved_58: Annotated[uint32_t, 232]
  reserved_59: Annotated[uint32_t, 236]
  reserved_60: Annotated[uint32_t, 240]
  reserved_61: Annotated[uint32_t, 244]
  reserved_62: Annotated[uint32_t, 248]
  reserved_63: Annotated[uint32_t, 252]
  reserved_64: Annotated[uint32_t, 256]
  reserved_65: Annotated[uint32_t, 260]
  reserved_66: Annotated[uint32_t, 264]
  reserved_67: Annotated[uint32_t, 268]
  reserved_68: Annotated[uint32_t, 272]
  reserved_69: Annotated[uint32_t, 276]
  reserved_70: Annotated[uint32_t, 280]
  reserved_71: Annotated[uint32_t, 284]
  reserved_72: Annotated[uint32_t, 288]
  reserved_73: Annotated[uint32_t, 292]
  reserved_74: Annotated[uint32_t, 296]
  reserved_75: Annotated[uint32_t, 300]
  reserved_76: Annotated[uint32_t, 304]
  reserved_77: Annotated[uint32_t, 308]
  reserved_78: Annotated[uint32_t, 312]
  reserved_79: Annotated[uint32_t, 316]
  reserved_80: Annotated[uint32_t, 320]
  reserved_81: Annotated[uint32_t, 324]
  reserved_82: Annotated[uint32_t, 328]
  reserved_83: Annotated[uint32_t, 332]
  reserved_84: Annotated[uint32_t, 336]
  reserved_85: Annotated[uint32_t, 340]
  reserved_86: Annotated[uint32_t, 344]
  reserved_87: Annotated[uint32_t, 348]
  reserved_88: Annotated[uint32_t, 352]
  reserved_89: Annotated[uint32_t, 356]
  reserved_90: Annotated[uint32_t, 360]
  reserved_91: Annotated[uint32_t, 364]
  reserved_92: Annotated[uint32_t, 368]
  reserved_93: Annotated[uint32_t, 372]
  reserved_94: Annotated[uint32_t, 376]
  reserved_95: Annotated[uint32_t, 380]
  reserved_96: Annotated[uint32_t, 384]
  reserved_97: Annotated[uint32_t, 388]
  reserved_98: Annotated[uint32_t, 392]
  reserved_99: Annotated[uint32_t, 396]
  reserved_100: Annotated[uint32_t, 400]
  reserved_101: Annotated[uint32_t, 404]
  reserved_102: Annotated[uint32_t, 408]
  reserved_103: Annotated[uint32_t, 412]
  reserved_104: Annotated[uint32_t, 416]
  reserved_105: Annotated[uint32_t, 420]
  reserved_106: Annotated[uint32_t, 424]
  reserved_107: Annotated[uint32_t, 428]
  reserved_108: Annotated[uint32_t, 432]
  reserved_109: Annotated[uint32_t, 436]
  reserved_110: Annotated[uint32_t, 440]
  reserved_111: Annotated[uint32_t, 444]
  reserved_112: Annotated[uint32_t, 448]
  reserved_113: Annotated[uint32_t, 452]
  reserved_114: Annotated[uint32_t, 456]
  reserved_115: Annotated[uint32_t, 460]
  reserved_116: Annotated[uint32_t, 464]
  reserved_117: Annotated[uint32_t, 468]
  reserved_118: Annotated[uint32_t, 472]
  reserved_119: Annotated[uint32_t, 476]
  reserved_120: Annotated[uint32_t, 480]
  reserved_121: Annotated[uint32_t, 484]
  reserved_122: Annotated[uint32_t, 488]
  reserved_123: Annotated[uint32_t, 492]
  reserved_124: Annotated[uint32_t, 496]
  reserved_125: Annotated[uint32_t, 500]
  sdma_engine_id: Annotated[uint32_t, 504]
  sdma_queue_id: Annotated[uint32_t, 508]
@record
class struct_v9_mqd:
  SIZE = 2048
  header: Annotated[uint32_t, 0]
  compute_dispatch_initiator: Annotated[uint32_t, 4]
  compute_dim_x: Annotated[uint32_t, 8]
  compute_dim_y: Annotated[uint32_t, 12]
  compute_dim_z: Annotated[uint32_t, 16]
  compute_start_x: Annotated[uint32_t, 20]
  compute_start_y: Annotated[uint32_t, 24]
  compute_start_z: Annotated[uint32_t, 28]
  compute_num_thread_x: Annotated[uint32_t, 32]
  compute_num_thread_y: Annotated[uint32_t, 36]
  compute_num_thread_z: Annotated[uint32_t, 40]
  compute_pipelinestat_enable: Annotated[uint32_t, 44]
  compute_perfcount_enable: Annotated[uint32_t, 48]
  compute_pgm_lo: Annotated[uint32_t, 52]
  compute_pgm_hi: Annotated[uint32_t, 56]
  compute_tba_lo: Annotated[uint32_t, 60]
  compute_tba_hi: Annotated[uint32_t, 64]
  compute_tma_lo: Annotated[uint32_t, 68]
  compute_tma_hi: Annotated[uint32_t, 72]
  compute_pgm_rsrc1: Annotated[uint32_t, 76]
  compute_pgm_rsrc2: Annotated[uint32_t, 80]
  compute_vmid: Annotated[uint32_t, 84]
  compute_resource_limits: Annotated[uint32_t, 88]
  compute_static_thread_mgmt_se0: Annotated[uint32_t, 92]
  compute_static_thread_mgmt_se1: Annotated[uint32_t, 96]
  compute_tmpring_size: Annotated[uint32_t, 100]
  compute_static_thread_mgmt_se2: Annotated[uint32_t, 104]
  compute_static_thread_mgmt_se3: Annotated[uint32_t, 108]
  compute_restart_x: Annotated[uint32_t, 112]
  compute_restart_y: Annotated[uint32_t, 116]
  compute_restart_z: Annotated[uint32_t, 120]
  compute_thread_trace_enable: Annotated[uint32_t, 124]
  compute_misc_reserved: Annotated[uint32_t, 128]
  compute_dispatch_id: Annotated[uint32_t, 132]
  compute_threadgroup_id: Annotated[uint32_t, 136]
  compute_relaunch: Annotated[uint32_t, 140]
  compute_wave_restore_addr_lo: Annotated[uint32_t, 144]
  compute_wave_restore_addr_hi: Annotated[uint32_t, 148]
  compute_wave_restore_control: Annotated[uint32_t, 152]
  compute_static_thread_mgmt_se4: Annotated[uint32_t, 0]
  compute_static_thread_mgmt_se5: Annotated[uint32_t, 4]
  compute_static_thread_mgmt_se6: Annotated[uint32_t, 8]
  compute_static_thread_mgmt_se7: Annotated[uint32_t, 12]
  compute_current_logic_xcc_id: Annotated[uint32_t, 0]
  compute_restart_cg_tg_id: Annotated[uint32_t, 4]
  compute_tg_chunk_size: Annotated[uint32_t, 8]
  compute_restore_tg_chunk_size: Annotated[uint32_t, 12]
  reserved_43: Annotated[uint32_t, 172]
  reserved_44: Annotated[uint32_t, 176]
  reserved_45: Annotated[uint32_t, 180]
  reserved_46: Annotated[uint32_t, 184]
  reserved_47: Annotated[uint32_t, 188]
  reserved_48: Annotated[uint32_t, 192]
  reserved_49: Annotated[uint32_t, 196]
  reserved_50: Annotated[uint32_t, 200]
  reserved_51: Annotated[uint32_t, 204]
  reserved_52: Annotated[uint32_t, 208]
  reserved_53: Annotated[uint32_t, 212]
  reserved_54: Annotated[uint32_t, 216]
  reserved_55: Annotated[uint32_t, 220]
  reserved_56: Annotated[uint32_t, 224]
  reserved_57: Annotated[uint32_t, 228]
  reserved_58: Annotated[uint32_t, 232]
  reserved_59: Annotated[uint32_t, 236]
  reserved_60: Annotated[uint32_t, 240]
  reserved_61: Annotated[uint32_t, 244]
  reserved_62: Annotated[uint32_t, 248]
  reserved_63: Annotated[uint32_t, 252]
  reserved_64: Annotated[uint32_t, 256]
  compute_user_data_0: Annotated[uint32_t, 260]
  compute_user_data_1: Annotated[uint32_t, 264]
  compute_user_data_2: Annotated[uint32_t, 268]
  compute_user_data_3: Annotated[uint32_t, 272]
  compute_user_data_4: Annotated[uint32_t, 276]
  compute_user_data_5: Annotated[uint32_t, 280]
  compute_user_data_6: Annotated[uint32_t, 284]
  compute_user_data_7: Annotated[uint32_t, 288]
  compute_user_data_8: Annotated[uint32_t, 292]
  compute_user_data_9: Annotated[uint32_t, 296]
  compute_user_data_10: Annotated[uint32_t, 300]
  compute_user_data_11: Annotated[uint32_t, 304]
  compute_user_data_12: Annotated[uint32_t, 308]
  compute_user_data_13: Annotated[uint32_t, 312]
  compute_user_data_14: Annotated[uint32_t, 316]
  compute_user_data_15: Annotated[uint32_t, 320]
  cp_compute_csinvoc_count_lo: Annotated[uint32_t, 324]
  cp_compute_csinvoc_count_hi: Annotated[uint32_t, 328]
  reserved_83: Annotated[uint32_t, 332]
  reserved_84: Annotated[uint32_t, 336]
  reserved_85: Annotated[uint32_t, 340]
  cp_mqd_query_time_lo: Annotated[uint32_t, 344]
  cp_mqd_query_time_hi: Annotated[uint32_t, 348]
  cp_mqd_connect_start_time_lo: Annotated[uint32_t, 352]
  cp_mqd_connect_start_time_hi: Annotated[uint32_t, 356]
  cp_mqd_connect_end_time_lo: Annotated[uint32_t, 360]
  cp_mqd_connect_end_time_hi: Annotated[uint32_t, 364]
  cp_mqd_connect_end_wf_count: Annotated[uint32_t, 368]
  cp_mqd_connect_end_pq_rptr: Annotated[uint32_t, 372]
  cp_mqd_connect_end_pq_wptr: Annotated[uint32_t, 376]
  cp_mqd_connect_end_ib_rptr: Annotated[uint32_t, 380]
  cp_mqd_readindex_lo: Annotated[uint32_t, 384]
  cp_mqd_readindex_hi: Annotated[uint32_t, 388]
  cp_mqd_save_start_time_lo: Annotated[uint32_t, 392]
  cp_mqd_save_start_time_hi: Annotated[uint32_t, 396]
  cp_mqd_save_end_time_lo: Annotated[uint32_t, 400]
  cp_mqd_save_end_time_hi: Annotated[uint32_t, 404]
  cp_mqd_restore_start_time_lo: Annotated[uint32_t, 408]
  cp_mqd_restore_start_time_hi: Annotated[uint32_t, 412]
  cp_mqd_restore_end_time_lo: Annotated[uint32_t, 416]
  cp_mqd_restore_end_time_hi: Annotated[uint32_t, 420]
  disable_queue: Annotated[uint32_t, 424]
  reserved_107: Annotated[uint32_t, 428]
  gds_cs_ctxsw_cnt0: Annotated[uint32_t, 432]
  gds_cs_ctxsw_cnt1: Annotated[uint32_t, 436]
  gds_cs_ctxsw_cnt2: Annotated[uint32_t, 440]
  gds_cs_ctxsw_cnt3: Annotated[uint32_t, 444]
  reserved_112: Annotated[uint32_t, 448]
  reserved_113: Annotated[uint32_t, 452]
  cp_pq_exe_status_lo: Annotated[uint32_t, 456]
  cp_pq_exe_status_hi: Annotated[uint32_t, 460]
  cp_packet_id_lo: Annotated[uint32_t, 464]
  cp_packet_id_hi: Annotated[uint32_t, 468]
  cp_packet_exe_status_lo: Annotated[uint32_t, 472]
  cp_packet_exe_status_hi: Annotated[uint32_t, 476]
  gds_save_base_addr_lo: Annotated[uint32_t, 480]
  gds_save_base_addr_hi: Annotated[uint32_t, 484]
  gds_save_mask_lo: Annotated[uint32_t, 488]
  gds_save_mask_hi: Annotated[uint32_t, 492]
  ctx_save_base_addr_lo: Annotated[uint32_t, 496]
  ctx_save_base_addr_hi: Annotated[uint32_t, 500]
  dynamic_cu_mask_addr_lo: Annotated[uint32_t, 504]
  dynamic_cu_mask_addr_hi: Annotated[uint32_t, 508]
  cp_mqd_base_addr_lo: Annotated[uint32_t, 512]
  cp_mqd_base_addr_hi: Annotated[uint32_t, 516]
  cp_hqd_active: Annotated[uint32_t, 520]
  cp_hqd_vmid: Annotated[uint32_t, 524]
  cp_hqd_persistent_state: Annotated[uint32_t, 528]
  cp_hqd_pipe_priority: Annotated[uint32_t, 532]
  cp_hqd_queue_priority: Annotated[uint32_t, 536]
  cp_hqd_quantum: Annotated[uint32_t, 540]
  cp_hqd_pq_base_lo: Annotated[uint32_t, 544]
  cp_hqd_pq_base_hi: Annotated[uint32_t, 548]
  cp_hqd_pq_rptr: Annotated[uint32_t, 552]
  cp_hqd_pq_rptr_report_addr_lo: Annotated[uint32_t, 556]
  cp_hqd_pq_rptr_report_addr_hi: Annotated[uint32_t, 560]
  cp_hqd_pq_wptr_poll_addr_lo: Annotated[uint32_t, 564]
  cp_hqd_pq_wptr_poll_addr_hi: Annotated[uint32_t, 568]
  cp_hqd_pq_doorbell_control: Annotated[uint32_t, 572]
  reserved_144: Annotated[uint32_t, 576]
  cp_hqd_pq_control: Annotated[uint32_t, 580]
  cp_hqd_ib_base_addr_lo: Annotated[uint32_t, 584]
  cp_hqd_ib_base_addr_hi: Annotated[uint32_t, 588]
  cp_hqd_ib_rptr: Annotated[uint32_t, 592]
  cp_hqd_ib_control: Annotated[uint32_t, 596]
  cp_hqd_iq_timer: Annotated[uint32_t, 600]
  cp_hqd_iq_rptr: Annotated[uint32_t, 604]
  cp_hqd_dequeue_request: Annotated[uint32_t, 608]
  cp_hqd_dma_offload: Annotated[uint32_t, 612]
  cp_hqd_sema_cmd: Annotated[uint32_t, 616]
  cp_hqd_msg_type: Annotated[uint32_t, 620]
  cp_hqd_atomic0_preop_lo: Annotated[uint32_t, 624]
  cp_hqd_atomic0_preop_hi: Annotated[uint32_t, 628]
  cp_hqd_atomic1_preop_lo: Annotated[uint32_t, 632]
  cp_hqd_atomic1_preop_hi: Annotated[uint32_t, 636]
  cp_hqd_hq_status0: Annotated[uint32_t, 640]
  cp_hqd_hq_control0: Annotated[uint32_t, 644]
  cp_mqd_control: Annotated[uint32_t, 648]
  cp_hqd_hq_status1: Annotated[uint32_t, 652]
  cp_hqd_hq_control1: Annotated[uint32_t, 656]
  cp_hqd_eop_base_addr_lo: Annotated[uint32_t, 660]
  cp_hqd_eop_base_addr_hi: Annotated[uint32_t, 664]
  cp_hqd_eop_control: Annotated[uint32_t, 668]
  cp_hqd_eop_rptr: Annotated[uint32_t, 672]
  cp_hqd_eop_wptr: Annotated[uint32_t, 676]
  cp_hqd_eop_done_events: Annotated[uint32_t, 680]
  cp_hqd_ctx_save_base_addr_lo: Annotated[uint32_t, 684]
  cp_hqd_ctx_save_base_addr_hi: Annotated[uint32_t, 688]
  cp_hqd_ctx_save_control: Annotated[uint32_t, 692]
  cp_hqd_cntl_stack_offset: Annotated[uint32_t, 696]
  cp_hqd_cntl_stack_size: Annotated[uint32_t, 700]
  cp_hqd_wg_state_offset: Annotated[uint32_t, 704]
  cp_hqd_ctx_save_size: Annotated[uint32_t, 708]
  cp_hqd_gds_resource_state: Annotated[uint32_t, 712]
  cp_hqd_error: Annotated[uint32_t, 716]
  cp_hqd_eop_wptr_mem: Annotated[uint32_t, 720]
  cp_hqd_aql_control: Annotated[uint32_t, 724]
  cp_hqd_pq_wptr_lo: Annotated[uint32_t, 728]
  cp_hqd_pq_wptr_hi: Annotated[uint32_t, 732]
  reserved_184: Annotated[uint32_t, 736]
  reserved_185: Annotated[uint32_t, 740]
  reserved_186: Annotated[uint32_t, 744]
  reserved_187: Annotated[uint32_t, 748]
  reserved_188: Annotated[uint32_t, 752]
  reserved_189: Annotated[uint32_t, 756]
  reserved_190: Annotated[uint32_t, 760]
  reserved_191: Annotated[uint32_t, 764]
  iqtimer_pkt_header: Annotated[uint32_t, 768]
  iqtimer_pkt_dw0: Annotated[uint32_t, 772]
  iqtimer_pkt_dw1: Annotated[uint32_t, 776]
  iqtimer_pkt_dw2: Annotated[uint32_t, 780]
  iqtimer_pkt_dw3: Annotated[uint32_t, 784]
  iqtimer_pkt_dw4: Annotated[uint32_t, 788]
  iqtimer_pkt_dw5: Annotated[uint32_t, 792]
  iqtimer_pkt_dw6: Annotated[uint32_t, 796]
  iqtimer_pkt_dw7: Annotated[uint32_t, 800]
  iqtimer_pkt_dw8: Annotated[uint32_t, 804]
  iqtimer_pkt_dw9: Annotated[uint32_t, 808]
  iqtimer_pkt_dw10: Annotated[uint32_t, 812]
  iqtimer_pkt_dw11: Annotated[uint32_t, 816]
  iqtimer_pkt_dw12: Annotated[uint32_t, 820]
  iqtimer_pkt_dw13: Annotated[uint32_t, 824]
  iqtimer_pkt_dw14: Annotated[uint32_t, 828]
  iqtimer_pkt_dw15: Annotated[uint32_t, 832]
  iqtimer_pkt_dw16: Annotated[uint32_t, 836]
  iqtimer_pkt_dw17: Annotated[uint32_t, 840]
  iqtimer_pkt_dw18: Annotated[uint32_t, 844]
  iqtimer_pkt_dw19: Annotated[uint32_t, 848]
  iqtimer_pkt_dw20: Annotated[uint32_t, 852]
  iqtimer_pkt_dw21: Annotated[uint32_t, 856]
  iqtimer_pkt_dw22: Annotated[uint32_t, 860]
  iqtimer_pkt_dw23: Annotated[uint32_t, 864]
  iqtimer_pkt_dw24: Annotated[uint32_t, 868]
  iqtimer_pkt_dw25: Annotated[uint32_t, 872]
  iqtimer_pkt_dw26: Annotated[uint32_t, 876]
  iqtimer_pkt_dw27: Annotated[uint32_t, 880]
  iqtimer_pkt_dw28: Annotated[uint32_t, 884]
  iqtimer_pkt_dw29: Annotated[uint32_t, 888]
  iqtimer_pkt_dw30: Annotated[uint32_t, 892]
  iqtimer_pkt_dw31: Annotated[uint32_t, 896]
  reserved_225: Annotated[uint32_t, 0]
  reserved_226: Annotated[uint32_t, 4]
  pm4_target_xcc_in_xcp: Annotated[uint32_t, 0]
  cp_mqd_stride_size: Annotated[uint32_t, 4]
  reserved_227: Annotated[uint32_t, 908]
  set_resources_header: Annotated[uint32_t, 912]
  set_resources_dw1: Annotated[uint32_t, 916]
  set_resources_dw2: Annotated[uint32_t, 920]
  set_resources_dw3: Annotated[uint32_t, 924]
  set_resources_dw4: Annotated[uint32_t, 928]
  set_resources_dw5: Annotated[uint32_t, 932]
  set_resources_dw6: Annotated[uint32_t, 936]
  set_resources_dw7: Annotated[uint32_t, 940]
  reserved_236: Annotated[uint32_t, 944]
  reserved_237: Annotated[uint32_t, 948]
  reserved_238: Annotated[uint32_t, 952]
  reserved_239: Annotated[uint32_t, 956]
  queue_doorbell_id0: Annotated[uint32_t, 960]
  queue_doorbell_id1: Annotated[uint32_t, 964]
  queue_doorbell_id2: Annotated[uint32_t, 968]
  queue_doorbell_id3: Annotated[uint32_t, 972]
  queue_doorbell_id4: Annotated[uint32_t, 976]
  queue_doorbell_id5: Annotated[uint32_t, 980]
  queue_doorbell_id6: Annotated[uint32_t, 984]
  queue_doorbell_id7: Annotated[uint32_t, 988]
  queue_doorbell_id8: Annotated[uint32_t, 992]
  queue_doorbell_id9: Annotated[uint32_t, 996]
  queue_doorbell_id10: Annotated[uint32_t, 1000]
  queue_doorbell_id11: Annotated[uint32_t, 1004]
  queue_doorbell_id12: Annotated[uint32_t, 1008]
  queue_doorbell_id13: Annotated[uint32_t, 1012]
  queue_doorbell_id14: Annotated[uint32_t, 1016]
  queue_doorbell_id15: Annotated[uint32_t, 1020]
  reserved_256: Annotated[uint32_t, 1024]
  reserved_257: Annotated[uint32_t, 1028]
  reserved_258: Annotated[uint32_t, 1032]
  reserved_259: Annotated[uint32_t, 1036]
  reserved_260: Annotated[uint32_t, 1040]
  reserved_261: Annotated[uint32_t, 1044]
  reserved_262: Annotated[uint32_t, 1048]
  reserved_263: Annotated[uint32_t, 1052]
  reserved_264: Annotated[uint32_t, 1056]
  reserved_265: Annotated[uint32_t, 1060]
  reserved_266: Annotated[uint32_t, 1064]
  reserved_267: Annotated[uint32_t, 1068]
  reserved_268: Annotated[uint32_t, 1072]
  reserved_269: Annotated[uint32_t, 1076]
  reserved_270: Annotated[uint32_t, 1080]
  reserved_271: Annotated[uint32_t, 1084]
  reserved_272: Annotated[uint32_t, 1088]
  reserved_273: Annotated[uint32_t, 1092]
  reserved_274: Annotated[uint32_t, 1096]
  reserved_275: Annotated[uint32_t, 1100]
  reserved_276: Annotated[uint32_t, 1104]
  reserved_277: Annotated[uint32_t, 1108]
  reserved_278: Annotated[uint32_t, 1112]
  reserved_279: Annotated[uint32_t, 1116]
  reserved_280: Annotated[uint32_t, 1120]
  reserved_281: Annotated[uint32_t, 1124]
  reserved_282: Annotated[uint32_t, 1128]
  reserved_283: Annotated[uint32_t, 1132]
  reserved_284: Annotated[uint32_t, 1136]
  reserved_285: Annotated[uint32_t, 1140]
  reserved_286: Annotated[uint32_t, 1144]
  reserved_287: Annotated[uint32_t, 1148]
  reserved_288: Annotated[uint32_t, 1152]
  reserved_289: Annotated[uint32_t, 1156]
  reserved_290: Annotated[uint32_t, 1160]
  reserved_291: Annotated[uint32_t, 1164]
  reserved_292: Annotated[uint32_t, 1168]
  reserved_293: Annotated[uint32_t, 1172]
  reserved_294: Annotated[uint32_t, 1176]
  reserved_295: Annotated[uint32_t, 1180]
  reserved_296: Annotated[uint32_t, 1184]
  reserved_297: Annotated[uint32_t, 1188]
  reserved_298: Annotated[uint32_t, 1192]
  reserved_299: Annotated[uint32_t, 1196]
  reserved_300: Annotated[uint32_t, 1200]
  reserved_301: Annotated[uint32_t, 1204]
  reserved_302: Annotated[uint32_t, 1208]
  reserved_303: Annotated[uint32_t, 1212]
  reserved_304: Annotated[uint32_t, 1216]
  reserved_305: Annotated[uint32_t, 1220]
  reserved_306: Annotated[uint32_t, 1224]
  reserved_307: Annotated[uint32_t, 1228]
  reserved_308: Annotated[uint32_t, 1232]
  reserved_309: Annotated[uint32_t, 1236]
  reserved_310: Annotated[uint32_t, 1240]
  reserved_311: Annotated[uint32_t, 1244]
  reserved_312: Annotated[uint32_t, 1248]
  reserved_313: Annotated[uint32_t, 1252]
  reserved_314: Annotated[uint32_t, 1256]
  reserved_315: Annotated[uint32_t, 1260]
  reserved_316: Annotated[uint32_t, 1264]
  reserved_317: Annotated[uint32_t, 1268]
  reserved_318: Annotated[uint32_t, 1272]
  reserved_319: Annotated[uint32_t, 1276]
  reserved_320: Annotated[uint32_t, 1280]
  reserved_321: Annotated[uint32_t, 1284]
  reserved_322: Annotated[uint32_t, 1288]
  reserved_323: Annotated[uint32_t, 1292]
  reserved_324: Annotated[uint32_t, 1296]
  reserved_325: Annotated[uint32_t, 1300]
  reserved_326: Annotated[uint32_t, 1304]
  reserved_327: Annotated[uint32_t, 1308]
  reserved_328: Annotated[uint32_t, 1312]
  reserved_329: Annotated[uint32_t, 1316]
  reserved_330: Annotated[uint32_t, 1320]
  reserved_331: Annotated[uint32_t, 1324]
  reserved_332: Annotated[uint32_t, 1328]
  reserved_333: Annotated[uint32_t, 1332]
  reserved_334: Annotated[uint32_t, 1336]
  reserved_335: Annotated[uint32_t, 1340]
  reserved_336: Annotated[uint32_t, 1344]
  reserved_337: Annotated[uint32_t, 1348]
  reserved_338: Annotated[uint32_t, 1352]
  reserved_339: Annotated[uint32_t, 1356]
  reserved_340: Annotated[uint32_t, 1360]
  reserved_341: Annotated[uint32_t, 1364]
  reserved_342: Annotated[uint32_t, 1368]
  reserved_343: Annotated[uint32_t, 1372]
  reserved_344: Annotated[uint32_t, 1376]
  reserved_345: Annotated[uint32_t, 1380]
  reserved_346: Annotated[uint32_t, 1384]
  reserved_347: Annotated[uint32_t, 1388]
  reserved_348: Annotated[uint32_t, 1392]
  reserved_349: Annotated[uint32_t, 1396]
  reserved_350: Annotated[uint32_t, 1400]
  reserved_351: Annotated[uint32_t, 1404]
  reserved_352: Annotated[uint32_t, 1408]
  reserved_353: Annotated[uint32_t, 1412]
  reserved_354: Annotated[uint32_t, 1416]
  reserved_355: Annotated[uint32_t, 1420]
  reserved_356: Annotated[uint32_t, 1424]
  reserved_357: Annotated[uint32_t, 1428]
  reserved_358: Annotated[uint32_t, 1432]
  reserved_359: Annotated[uint32_t, 1436]
  reserved_360: Annotated[uint32_t, 1440]
  reserved_361: Annotated[uint32_t, 1444]
  reserved_362: Annotated[uint32_t, 1448]
  reserved_363: Annotated[uint32_t, 1452]
  reserved_364: Annotated[uint32_t, 1456]
  reserved_365: Annotated[uint32_t, 1460]
  reserved_366: Annotated[uint32_t, 1464]
  reserved_367: Annotated[uint32_t, 1468]
  reserved_368: Annotated[uint32_t, 1472]
  reserved_369: Annotated[uint32_t, 1476]
  reserved_370: Annotated[uint32_t, 1480]
  reserved_371: Annotated[uint32_t, 1484]
  reserved_372: Annotated[uint32_t, 1488]
  reserved_373: Annotated[uint32_t, 1492]
  reserved_374: Annotated[uint32_t, 1496]
  reserved_375: Annotated[uint32_t, 1500]
  reserved_376: Annotated[uint32_t, 1504]
  reserved_377: Annotated[uint32_t, 1508]
  reserved_378: Annotated[uint32_t, 1512]
  reserved_379: Annotated[uint32_t, 1516]
  reserved_380: Annotated[uint32_t, 1520]
  reserved_381: Annotated[uint32_t, 1524]
  reserved_382: Annotated[uint32_t, 1528]
  reserved_383: Annotated[uint32_t, 1532]
  reserved_384: Annotated[uint32_t, 1536]
  reserved_385: Annotated[uint32_t, 1540]
  reserved_386: Annotated[uint32_t, 1544]
  reserved_387: Annotated[uint32_t, 1548]
  reserved_388: Annotated[uint32_t, 1552]
  reserved_389: Annotated[uint32_t, 1556]
  reserved_390: Annotated[uint32_t, 1560]
  reserved_391: Annotated[uint32_t, 1564]
  reserved_392: Annotated[uint32_t, 1568]
  reserved_393: Annotated[uint32_t, 1572]
  reserved_394: Annotated[uint32_t, 1576]
  reserved_395: Annotated[uint32_t, 1580]
  reserved_396: Annotated[uint32_t, 1584]
  reserved_397: Annotated[uint32_t, 1588]
  reserved_398: Annotated[uint32_t, 1592]
  reserved_399: Annotated[uint32_t, 1596]
  reserved_400: Annotated[uint32_t, 1600]
  reserved_401: Annotated[uint32_t, 1604]
  reserved_402: Annotated[uint32_t, 1608]
  reserved_403: Annotated[uint32_t, 1612]
  reserved_404: Annotated[uint32_t, 1616]
  reserved_405: Annotated[uint32_t, 1620]
  reserved_406: Annotated[uint32_t, 1624]
  reserved_407: Annotated[uint32_t, 1628]
  reserved_408: Annotated[uint32_t, 1632]
  reserved_409: Annotated[uint32_t, 1636]
  reserved_410: Annotated[uint32_t, 1640]
  reserved_411: Annotated[uint32_t, 1644]
  reserved_412: Annotated[uint32_t, 1648]
  reserved_413: Annotated[uint32_t, 1652]
  reserved_414: Annotated[uint32_t, 1656]
  reserved_415: Annotated[uint32_t, 1660]
  reserved_416: Annotated[uint32_t, 1664]
  reserved_417: Annotated[uint32_t, 1668]
  reserved_418: Annotated[uint32_t, 1672]
  reserved_419: Annotated[uint32_t, 1676]
  reserved_420: Annotated[uint32_t, 1680]
  reserved_421: Annotated[uint32_t, 1684]
  reserved_422: Annotated[uint32_t, 1688]
  reserved_423: Annotated[uint32_t, 1692]
  reserved_424: Annotated[uint32_t, 1696]
  reserved_425: Annotated[uint32_t, 1700]
  reserved_426: Annotated[uint32_t, 1704]
  reserved_427: Annotated[uint32_t, 1708]
  reserved_428: Annotated[uint32_t, 1712]
  reserved_429: Annotated[uint32_t, 1716]
  reserved_430: Annotated[uint32_t, 1720]
  reserved_431: Annotated[uint32_t, 1724]
  reserved_432: Annotated[uint32_t, 1728]
  reserved_433: Annotated[uint32_t, 1732]
  reserved_434: Annotated[uint32_t, 1736]
  reserved_435: Annotated[uint32_t, 1740]
  reserved_436: Annotated[uint32_t, 1744]
  reserved_437: Annotated[uint32_t, 1748]
  reserved_438: Annotated[uint32_t, 1752]
  reserved_439: Annotated[uint32_t, 1756]
  reserved_440: Annotated[uint32_t, 1760]
  reserved_441: Annotated[uint32_t, 1764]
  reserved_442: Annotated[uint32_t, 1768]
  reserved_443: Annotated[uint32_t, 1772]
  reserved_444: Annotated[uint32_t, 1776]
  reserved_445: Annotated[uint32_t, 1780]
  reserved_446: Annotated[uint32_t, 1784]
  reserved_447: Annotated[uint32_t, 1788]
  reserved_448: Annotated[uint32_t, 1792]
  reserved_449: Annotated[uint32_t, 1796]
  reserved_450: Annotated[uint32_t, 1800]
  reserved_451: Annotated[uint32_t, 1804]
  reserved_452: Annotated[uint32_t, 1808]
  reserved_453: Annotated[uint32_t, 1812]
  reserved_454: Annotated[uint32_t, 1816]
  reserved_455: Annotated[uint32_t, 1820]
  reserved_456: Annotated[uint32_t, 1824]
  reserved_457: Annotated[uint32_t, 1828]
  reserved_458: Annotated[uint32_t, 1832]
  reserved_459: Annotated[uint32_t, 1836]
  reserved_460: Annotated[uint32_t, 1840]
  reserved_461: Annotated[uint32_t, 1844]
  reserved_462: Annotated[uint32_t, 1848]
  reserved_463: Annotated[uint32_t, 1852]
  reserved_464: Annotated[uint32_t, 1856]
  reserved_465: Annotated[uint32_t, 1860]
  reserved_466: Annotated[uint32_t, 1864]
  reserved_467: Annotated[uint32_t, 1868]
  reserved_468: Annotated[uint32_t, 1872]
  reserved_469: Annotated[uint32_t, 1876]
  reserved_470: Annotated[uint32_t, 1880]
  reserved_471: Annotated[uint32_t, 1884]
  reserved_472: Annotated[uint32_t, 1888]
  reserved_473: Annotated[uint32_t, 1892]
  reserved_474: Annotated[uint32_t, 1896]
  reserved_475: Annotated[uint32_t, 1900]
  reserved_476: Annotated[uint32_t, 1904]
  reserved_477: Annotated[uint32_t, 1908]
  reserved_478: Annotated[uint32_t, 1912]
  reserved_479: Annotated[uint32_t, 1916]
  reserved_480: Annotated[uint32_t, 1920]
  reserved_481: Annotated[uint32_t, 1924]
  reserved_482: Annotated[uint32_t, 1928]
  reserved_483: Annotated[uint32_t, 1932]
  reserved_484: Annotated[uint32_t, 1936]
  reserved_485: Annotated[uint32_t, 1940]
  reserved_486: Annotated[uint32_t, 1944]
  reserved_487: Annotated[uint32_t, 1948]
  reserved_488: Annotated[uint32_t, 1952]
  reserved_489: Annotated[uint32_t, 1956]
  reserved_490: Annotated[uint32_t, 1960]
  reserved_491: Annotated[uint32_t, 1964]
  reserved_492: Annotated[uint32_t, 1968]
  reserved_493: Annotated[uint32_t, 1972]
  reserved_494: Annotated[uint32_t, 1976]
  reserved_495: Annotated[uint32_t, 1980]
  reserved_496: Annotated[uint32_t, 1984]
  reserved_497: Annotated[uint32_t, 1988]
  reserved_498: Annotated[uint32_t, 1992]
  reserved_499: Annotated[uint32_t, 1996]
  reserved_500: Annotated[uint32_t, 2000]
  reserved_501: Annotated[uint32_t, 2004]
  reserved_502: Annotated[uint32_t, 2008]
  reserved_503: Annotated[uint32_t, 2012]
  reserved_504: Annotated[uint32_t, 2016]
  reserved_505: Annotated[uint32_t, 2020]
  reserved_506: Annotated[uint32_t, 2024]
  reserved_507: Annotated[uint32_t, 2028]
  reserved_508: Annotated[uint32_t, 2032]
  reserved_509: Annotated[uint32_t, 2036]
  reserved_510: Annotated[uint32_t, 2040]
  reserved_511: Annotated[uint32_t, 2044]
@record
class struct_v9_mqd_allocation:
  SIZE = 2064
  mqd: Annotated[struct_v9_mqd, 0]
  wptr_poll_mem: Annotated[uint32_t, 2048]
  rptr_report_mem: Annotated[uint32_t, 2052]
  dynamic_cu_mask: Annotated[uint32_t, 2056]
  dynamic_rb_mask: Annotated[uint32_t, 2060]
@record
class struct_v9_ce_ib_state:
  SIZE = 40
  ce_ib_completion_status: Annotated[uint32_t, 0]
  ce_constegnine_count: Annotated[uint32_t, 4]
  ce_ibOffset_ib1: Annotated[uint32_t, 8]
  ce_ibOffset_ib2: Annotated[uint32_t, 12]
  ce_chainib_addrlo_ib1: Annotated[uint32_t, 16]
  ce_chainib_addrlo_ib2: Annotated[uint32_t, 20]
  ce_chainib_addrhi_ib1: Annotated[uint32_t, 24]
  ce_chainib_addrhi_ib2: Annotated[uint32_t, 28]
  ce_chainib_size_ib1: Annotated[uint32_t, 32]
  ce_chainib_size_ib2: Annotated[uint32_t, 36]
@record
class struct_v9_de_ib_state:
  SIZE = 108
  ib_completion_status: Annotated[uint32_t, 0]
  de_constEngine_count: Annotated[uint32_t, 4]
  ib_offset_ib1: Annotated[uint32_t, 8]
  ib_offset_ib2: Annotated[uint32_t, 12]
  chain_ib_addrlo_ib1: Annotated[uint32_t, 16]
  chain_ib_addrlo_ib2: Annotated[uint32_t, 20]
  chain_ib_addrhi_ib1: Annotated[uint32_t, 24]
  chain_ib_addrhi_ib2: Annotated[uint32_t, 28]
  chain_ib_size_ib1: Annotated[uint32_t, 32]
  chain_ib_size_ib2: Annotated[uint32_t, 36]
  preamble_begin_ib1: Annotated[uint32_t, 40]
  preamble_begin_ib2: Annotated[uint32_t, 44]
  preamble_end_ib1: Annotated[uint32_t, 48]
  preamble_end_ib2: Annotated[uint32_t, 52]
  chain_ib_pream_addrlo_ib1: Annotated[uint32_t, 56]
  chain_ib_pream_addrlo_ib2: Annotated[uint32_t, 60]
  chain_ib_pream_addrhi_ib1: Annotated[uint32_t, 64]
  chain_ib_pream_addrhi_ib2: Annotated[uint32_t, 68]
  draw_indirect_baseLo: Annotated[uint32_t, 72]
  draw_indirect_baseHi: Annotated[uint32_t, 76]
  disp_indirect_baseLo: Annotated[uint32_t, 80]
  disp_indirect_baseHi: Annotated[uint32_t, 84]
  gds_backup_addrlo: Annotated[uint32_t, 88]
  gds_backup_addrhi: Annotated[uint32_t, 92]
  index_base_addrlo: Annotated[uint32_t, 96]
  index_base_addrhi: Annotated[uint32_t, 100]
  sample_cntl: Annotated[uint32_t, 104]
@record
class struct_v9_gfx_meta_data:
  SIZE = 4096
  ce_payload: Annotated[struct_v9_ce_ib_state, 0]
  reserved1: Annotated[(uint32_t* 54), 40]
  de_payload: Annotated[struct_v9_de_ib_state, 256]
  DeIbBaseAddrLo: Annotated[uint32_t, 364]
  DeIbBaseAddrHi: Annotated[uint32_t, 368]
  reserved2: Annotated[(uint32_t* 931), 372]
enum_soc15_ih_clientid = CEnum(ctypes.c_uint32)
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

enum_soc21_ih_clientid = CEnum(ctypes.c_uint32)
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

init_records()
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