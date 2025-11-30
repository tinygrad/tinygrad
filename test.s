.text
.globl kernel
.p2align 8 // TODO: need more?
.type kernel,@function

kernel:

	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v10, 0             // 000000001608: CA100080 190A0080
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v0, 0               // 000000001610: CA100080 09000080
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v2, 0              // 000000001618: CA100080 11020080
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v26, 0             // 000000001620: CA100080 131A0080
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v8, 0              // 000000001628: CA100080 0D080080
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v14, 0              // 000000001630: CA100080 050E0080
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v12, 0              // 000000001638: CA100080 070C0080
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v4, 0              // 000000001640: CA100080 0B040080
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v22, 0              // 000000001648: CA100080 01160080
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v16, 0             // 000000001650: CA100080 17100080
	s_waitcnt lgkmcnt(0)                                       // 000000001658: BF89FC07
	s_add_u32 s4, s2, 0xfffffe80                               // 00000000165C: 8004FF02 FFFFFE80
	v_writelane_b32 v37, s0, 0                                 // 000000001664: D7610025 00010000
	s_addc_u32 s5, s3, -1                                      // 00000000166C: 8205C103
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v20, 0             // 000000001670: CA100080 0F140080
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v18, 0             // 000000001678: CA100080 15120080
	v_writelane_b32 v37, s1, 1                                 // 000000001680: D7610025 00010201
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v6, 0               // 000000001688: CA100080 03060080
	v_mov_b32_e32 v24, 0                                       // 000000001690: 7E300280
	s_mov_b32 s33, 0                                           // 000000001694: BEA10080
	v_writelane_b32 v37, s2, 2                                 // 000000001698: D7610025 00010402
	v_writelane_b32 v37, s3, 3                                 // 0000000016A0: D7610025 00010603
	s_mov_b64 s[0:1], 16                                       // 0000000016A8: BE800190
	v_writelane_b32 v37, s4, 4                                 // 0000000016AC: D7610025 00010804
	v_writelane_b32 v37, s5, 5                                 // 0000000016B4: D7610025 00010A05
	v_writelane_b32 v37, s0, 6                                 // 0000000016BC: D7610025 00010C00
	v_writelane_b32 v37, s1, 7                                 // 0000000016C4: D7610025 00010E01
	s_branch 30                                                // 0000000016CC: BFA0001E <r_3_3_3_8_8_8+0x148>
	v_readlane_b32 s33, v37, 9                                 // 0000000016D0: D7600021 00011325
	v_readlane_b32 s0, v37, 6                                  // 0000000016D8: D7600000 00010D25
	v_readlane_b32 s1, v37, 7                                  // 0000000016E0: D7600001 00010F25
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000016E8: BF870113
	s_add_i32 s33, s33, 1                                      // 0000000016EC: 81218121
	s_add_u32 s0, s0, 0x100                                    // 0000000016F0: 8000FF00 00000100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000016F8: BF8700B1
	s_addc_u32 s1, s1, 0                                       // 0000000016FC: 82018001
	v_writelane_b32 v37, s0, 6                                 // 000000001700: D7610025 00010C00
	v_writelane_b32 v37, s1, 7                                 // 000000001708: D7610025 00010E01
	v_readlane_b32 s0, v37, 4                                  // 000000001710: D7600000 00010925
	v_readlane_b32 s1, v37, 5                                  // 000000001718: D7600001 00010B25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001720: BF870092
	s_add_u32 s0, s0, 0x400                                    // 000000001724: 8000FF00 00000400
	s_addc_u32 s1, s1, 0                                       // 00000000172C: 82018001
	v_writelane_b32 v37, s0, 4                                 // 000000001730: D7610025 00010800
	s_cmp_eq_u32 s33, 8                                        // 000000001738: BF068821
	v_writelane_b32 v37, s1, 5                                 // 00000000173C: D7610025 00010A01
	s_cbranch_scc1 948                                         // 000000001744: BFA203B4 <r_3_3_3_8_8_8+0x1018>
	s_cmp_eq_u32 s33, 0                                        // 000000001748: BF068021
	s_mov_b32 s101, -1                                         // 00000000174C: BEE500C1
	s_cselect_b32 s0, -1, 0                                    // 000000001750: 980080C1
	s_lshl_b32 s100, s33, 8                                    // 000000001754: 84648821
	s_cmp_lg_u32 s33, 0                                        // 000000001758: BF078021
	v_writelane_b32 v37, s0, 8                                 // 00000000175C: D7610025 00011000
	s_cbranch_scc1 3                                           // 000000001764: BFA20003 <r_3_3_3_8_8_8+0x174>
	s_mov_b32 s101, 0                                          // 000000001768: BEE50080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000176C: BF870009
	s_mov_b64 s[34:35], s[100:101]                             // 000000001770: BEA20164
	s_mov_b32 s13, s101                                        // 000000001774: BE8D0065
	s_mov_b32 s14, s101                                        // 000000001778: BE8E0065
	s_mov_b32 s15, s101                                        // 00000000177C: BE8F0065
	s_mov_b32 s4, s101                                         // 000000001780: BE840065
	s_mov_b32 s5, s101                                         // 000000001784: BE850065
	s_mov_b32 s6, s101                                         // 000000001788: BE860065
	s_mov_b32 s7, s101                                         // 00000000178C: BE870065
	s_mov_b32 s0, s101                                         // 000000001790: BE800065
	s_and_not1_b32 vcc_lo, exec_lo, s101                       // 000000001794: 916A657E
	s_mov_b32 s1, s101                                         // 000000001798: BE810065
	s_mov_b32 s2, s101                                         // 00000000179C: BE820065
	s_mov_b32 s3, s101                                         // 0000000017A0: BE830065
	s_mov_b32 s8, s101                                         // 0000000017A4: BE880065
	s_mov_b32 s9, s101                                         // 0000000017A8: BE890065
	s_mov_b32 s10, s101                                        // 0000000017AC: BE8A0065
	s_mov_b32 s11, s101                                        // 0000000017B0: BE8B0065
	s_mov_b32 s24, s101                                        // 0000000017B4: BE980065
	s_mov_b32 s25, s101                                        // 0000000017B8: BE990065
	s_mov_b32 s26, s101                                        // 0000000017BC: BE9A0065
	s_mov_b32 s27, s101                                        // 0000000017C0: BE9B0065
	s_mov_b32 s16, s101                                        // 0000000017C4: BE900065
	s_mov_b32 s17, s101                                        // 0000000017C8: BE910065
	s_mov_b32 s18, s101                                        // 0000000017CC: BE920065
	s_mov_b32 s19, s101                                        // 0000000017D0: BE930065
	s_mov_b32 s20, s101                                        // 0000000017D4: BE940065
	s_mov_b32 s21, s101                                        // 0000000017D8: BE950065
	s_mov_b32 s22, s101                                        // 0000000017DC: BE960065
	s_mov_b32 s23, s101                                        // 0000000017E0: BE970065
	s_mov_b32 s28, s101                                        // 0000000017E4: BE9C0065
	s_mov_b32 s29, s101                                        // 0000000017E8: BE9D0065
	s_mov_b32 s30, s101                                        // 0000000017EC: BE9E0065
	s_mov_b32 s31, s101                                        // 0000000017F0: BE9F0065
	s_cbranch_vccnz 25                                         // 0000000017F4: BFA40019 <r_3_3_3_8_8_8+0x25c>
	v_readlane_b32 s4, v37, 0                                  // 0000000017F8: D7600004 00010125
	v_readlane_b32 s6, v37, 2                                  // 000000001800: D7600006 00010525
	s_mov_b32 s101, 0                                          // 000000001808: BEE50080
	v_readlane_b32 s7, v37, 3                                  // 00000000180C: D7600007 00010725
	s_lshl_b64 s[0:1], s[100:101], 2                           // 000000001814: 84808264
	v_readlane_b32 s5, v37, 1                                  // 000000001818: D7600005 00010325
	s_add_u32 s2, s6, s0                                       // 000000001820: 80020006
	s_mov_b64 s[34:35], s[100:101]                             // 000000001824: BEA20164
	s_addc_u32 s17, s7, s1                                     // 000000001828: 82110107
	s_add_u32 s0, s2, 0xfffffd00                               // 00000000182C: 8000FF02 FFFFFD00
	s_addc_u32 s1, s17, -1                                     // 000000001834: 8201C111
	s_add_u32 s16, s2, 0xfffffe40                              // 000000001838: 8010FF02 FFFFFE40
	s_load_b512 s[0:15], s[0:1], null                          // 000000001840: F4100000 F8000000
	s_addc_u32 s17, s17, -1                                    // 000000001848: 8211C111
	s_load_b512 s[16:31], s[16:17], null                       // 00000000184C: F4100408 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001854: BF89FC07
	s_mov_b32 s101, s12                                        // 000000001858: BEE5000C
	v_readlane_b32 s36, v37, 0                                 // 00000000185C: D7600024 00010125
	v_readlane_b32 s38, v37, 2                                 // 000000001864: D7600026 00010525
	s_cmp_lg_u32 s33, 7                                        // 00000000186C: BF078721
	v_readlane_b32 s39, v37, 3                                 // 000000001870: D7600027 00010725
	s_cselect_b32 s104, -1, 0                                  // 000000001878: 986880C1
	s_lshl_b64 s[34:35], s[34:35], 2                           // 00000000187C: 84A28222
	v_readlane_b32 s37, v37, 1                                 // 000000001880: D7600025 00010325
	s_add_u32 s34, s38, s34                                    // 000000001888: 80222226
	s_addc_u32 s35, s39, s35                                   // 00000000188C: 82232327
	s_clause 0x1                                               // 000000001890: BF850001
	s_load_b512 s[36:51], s[34:35], 0x1100                     // 000000001894: F4100911 F8001100
	s_load_b512 s[52:67], s[34:35], 0x1240                     // 00000000189C: F4100D11 F8001240
	s_cmp_eq_u32 s33, 7                                        // 0000000018A4: BF068721
	s_mov_b32 s80, 0                                           // 0000000018A8: BED00080
	s_mov_b32 s81, 0                                           // 0000000018AC: BED10080
	s_mov_b32 s82, 0                                           // 0000000018B0: BED20080
	s_mov_b32 s83, 0                                           // 0000000018B4: BED30080
	s_mov_b32 s72, 0                                           // 0000000018B8: BEC80080
	s_mov_b32 s73, 0                                           // 0000000018BC: BEC90080
	s_mov_b32 s74, 0                                           // 0000000018C0: BECA0080
	s_mov_b32 s75, 0                                           // 0000000018C4: BECB0080
	s_mov_b32 s68, 0                                           // 0000000018C8: BEC40080
	s_mov_b32 s69, 0                                           // 0000000018CC: BEC50080
	s_mov_b32 s70, 0                                           // 0000000018D0: BEC60080
	s_mov_b32 s71, 0                                           // 0000000018D4: BEC70080
	s_mov_b32 s76, 0                                           // 0000000018D8: BECC0080
	s_mov_b32 s77, 0                                           // 0000000018DC: BECD0080
	s_mov_b32 s78, 0                                           // 0000000018E0: BECE0080
	s_mov_b32 s79, 0                                           // 0000000018E4: BECF0080
	s_mov_b32 s92, 0                                           // 0000000018E8: BEDC0080
	s_mov_b32 s93, 0                                           // 0000000018EC: BEDD0080
	s_mov_b32 s94, 0                                           // 0000000018F0: BEDE0080
	s_mov_b32 s95, 0                                           // 0000000018F4: BEDF0080
	s_mov_b32 s84, 0                                           // 0000000018F8: BED40080
	s_mov_b32 s85, 0                                           // 0000000018FC: BED50080
	s_mov_b32 s86, 0                                           // 000000001900: BED60080
	s_mov_b32 s87, 0                                           // 000000001904: BED70080
	s_mov_b32 s88, 0                                           // 000000001908: BED80080
	s_mov_b32 s89, 0                                           // 00000000190C: BED90080
	s_mov_b32 s90, 0                                           // 000000001910: BEDA0080
	s_mov_b32 s91, 0                                           // 000000001914: BEDB0080
	s_mov_b32 s96, 0                                           // 000000001918: BEE00080
	s_mov_b32 s97, 0                                           // 00000000191C: BEE10080
	s_mov_b32 s98, 0                                           // 000000001920: BEE20080
	s_mov_b32 s99, 0                                           // 000000001924: BEE30080
	v_writelane_b32 v37, s33, 9                                // 000000001928: D7610025 00011221
	s_cbranch_scc1 5                                           // 000000001930: BFA20005 <r_3_3_3_8_8_8+0x348>
	s_clause 0x1                                               // 000000001934: BF850001
	s_load_b512 s[68:83], s[34:35], 0x2500                     // 000000001938: F4101111 F8002500
	s_load_b512 s[84:99], s[34:35], 0x2640                     // 000000001940: F4101511 F8002640
	s_waitcnt lgkmcnt(0)                                       // 000000001948: BF89FC07
	v_dual_add_f32 v26, s88, v26 :: v_dual_add_f32 v25, s93, v25// 00000000194C: C9083458 1A18325D
	v_add_f32_e32 v23, s84, v23                                // 000000001954: 062E2E54
	v_dual_add_f32 v7, s72, v7 :: v_dual_add_f32 v16, s68, v16 // 000000001958: C9080E48 07102044
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 000000001960: BF8701B3
	v_dual_add_f32 v26, s89, v26 :: v_dual_add_f32 v17, s25, v17// 000000001964: C9083459 1A102219
	v_dual_add_f32 v11, s40, v11 :: v_dual_add_f32 v20, s36, v20// 00000000196C: C9081628 0B142824
	v_dual_add_f32 v9, s61, v9 :: v_dual_add_f32 v10, s77, v10 // 000000001974: C908123D 090A144D
	v_dual_add_f32 v26, s90, v26 :: v_dual_add_f32 v19, s45, v19// 00000000197C: C908345A 1A12262D
	v_dual_add_f32 v7, s73, v7 :: v_dual_add_f32 v18, s0, v18  // 000000001984: C9080E49 07122400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000198C: BF870194
	v_dual_add_f32 v11, s41, v11 :: v_dual_add_f32 v16, s69, v16// 000000001990: C9081629 0B102045
	v_dual_add_f32 v26, s91, v26 :: v_dual_add_f32 v25, s94, v25// 000000001998: C908345B 1A18325E
	v_dual_add_f32 v10, s78, v10 :: v_dual_add_f32 v23, s85, v23// 0000000019A0: C908144E 0A162E55
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000019A8: BF870194
	v_dual_add_f32 v7, s74, v7 :: v_dual_add_f32 v20, s37, v20 // 0000000019AC: C9080E4A 07142825
	v_dual_add_f32 v26, s92, v26 :: v_dual_add_f32 v11, s42, v11// 0000000019B4: C908345C 1A0A162A
	v_dual_add_f32 v18, s1, v18 :: v_dual_add_f32 v13, s9, v13 // 0000000019BC: C9082401 120C1A09
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000019C4: BF870194
	v_dual_add_f32 v25, s95, v25 :: v_dual_add_f32 v10, s79, v10// 0000000019C8: C908325F 190A144F
	v_dual_add_f32 v26, s93, v26 :: v_dual_add_f32 v7, s75, v7 // 0000000019D0: C908345D 1A060E4B
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000019D8: BF8701A4
	v_dual_add_f32 v16, s70, v16 :: v_dual_add_f32 v11, s43, v11// 0000000019DC: C9082046 100A162B
	v_dual_add_f32 v20, s38, v20 :: v_dual_add_f32 v9, s62, v9 // 0000000019E4: C9082826 1408123E
	v_dual_add_f32 v26, s94, v26 :: v_dual_add_f32 v23, s86, v23// 0000000019EC: C908345E 1A162E56
	v_dual_add_f32 v17, s26, v17 :: v_dual_add_f32 v10, s80, v10// 0000000019F4: C908221A 110A1450
	v_dual_add_f32 v7, s76, v7 :: v_dual_add_f32 v18, s2, v18  // 0000000019FC: C9080E4C 07122402
	v_dual_add_f32 v11, s44, v11 :: v_dual_add_f32 v16, s71, v16// 000000001A04: C908162C 0B102047
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000001A0C: BF870223
	v_dual_add_f32 v25, s96, v25 :: v_dual_add_f32 v10, s81, v10// 000000001A10: C9083260 190A1451
	v_dual_add_f32 v26, s95, v26 :: v_dual_add_f32 v23, s87, v23// 000000001A18: C908345F 1A162E57
	v_dual_add_f32 v7, s77, v7 :: v_dual_add_f32 v20, s39, v20 // 000000001A20: C9080E4D 07142827
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001A28: BF870214
	v_dual_add_f32 v11, s45, v11 :: v_dual_add_f32 v18, s3, v18// 000000001A2C: C908162D 0B122403
	v_dual_add_f32 v25, s97, v25 :: v_dual_add_f32 v2, 0, v2   // 000000001A34: C9083261 19020480
	v_dual_add_f32 v9, s63, v9 :: v_dual_add_f32 v14, s20, v14 // 000000001A3C: C908123F 090E1C14
	v_dual_add_f32 v17, s27, v17 :: v_dual_add_f32 v12, s4, v12// 000000001A44: C908221B 110C1804
	v_dual_add_f32 v10, s82, v10 :: v_dual_add_f32 v19, s46, v19// 000000001A4C: C9081452 0A12262E
	v_dual_add_f32 v23, s88, v23 :: v_dual_add_f32 v8, s56, v8 // 000000001A54: C9082E58 17081038
	v_dual_add_f32 v7, s78, v7 :: v_dual_add_f32 v16, s72, v16 // 000000001A5C: C9080E4E 07102048
	v_dual_add_f32 v11, s46, v11 :: v_dual_add_f32 v20, s40, v20// 000000001A64: C908162E 0B142828
	v_dual_add_f32 v13, s10, v13 :: v_dual_add_f32 v0, 0, v0   // 000000001A6C: C9081A0A 0D000080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001A74: BF870004
	v_dual_add_f32 v25, s98, v25 :: v_dual_add_f32 v8, s57, v8 // 000000001A78: C9083262 19081039
	v_dual_add_f32 v23, s89, v23 :: v_dual_add_f32 v14, s21, v14// 000000001A80: C9082E59 170E1C15
	v_dual_add_f32 v10, s83, v10 :: v_dual_add_f32 v19, s47, v19// 000000001A88: C9081453 0A12262F
	v_dual_add_f32 v12, s5, v12 :: v_dual_add_f32 v7, s79, v7  // 000000001A90: C9081805 0C060E4F
	v_dual_add_f32 v18, s4, v18 :: v_dual_add_f32 v11, s47, v11// 000000001A98: C9082404 120A162F
	v_dual_add_f32 v16, s73, v16 :: v_dual_add_f32 v15, s52, v15// 000000001AA0: C9082049 100E1E34
	v_add_f32_e32 v21, s16, v21                                // 000000001AA8: 062A2A10
	v_dual_add_f32 v25, s99, v25 :: v_dual_add_f32 v8, s58, v8 // 000000001AAC: C9083263 1908103A
	v_dual_add_f32 v23, s90, v23 :: v_dual_add_f32 v14, s22, v14// 000000001AB4: C9082E5A 170E1C16
	v_dual_add_f32 v13, s11, v13 :: v_dual_add_f32 v12, s6, v12// 000000001ABC: C9081A0B 0D0C1806
	v_dual_add_f32 v1, 0, v1 :: v_dual_add_f32 v20, s41, v20   // 000000001AC4: C9080280 01142829
	v_dual_add_f32 v18, s5, v18 :: v_dual_add_f32 v15, s53, v15// 000000001ACC: C9082405 120E1E35
	v_dual_add_f32 v16, s74, v16 :: v_dual_add_f32 v21, s17, v21// 000000001AD4: C908204A 10142A11
	v_dual_add_f32 v9, s64, v9 :: v_dual_add_f32 v8, s59, v8   // 000000001ADC: C9081240 0908103B
	v_dual_add_f32 v17, s28, v17 :: v_dual_add_f32 v14, s23, v14// 000000001AE4: C908221C 110E1C17
	v_dual_add_f32 v19, s48, v19 :: v_dual_add_f32 v12, s7, v12// 000000001AEC: C9082630 130C1807
	v_dual_add_f32 v20, s42, v20 :: v_dual_add_f32 v15, s54, v15// 000000001AF4: C908282A 140E1E36
	v_add_f32_e32 v6, 0, v6                                    // 000000001AFC: 060C0C80
	v_dual_add_f32 v18, s6, v18 :: v_dual_add_f32 v21, s18, v21// 000000001B00: C9082406 12142A12
	v_add_f32_e32 v24, 0, v24                                  // 000000001B08: 06303080
	v_writelane_b32 v37, s100, 10                              // 000000001B0C: D7610025 00011464
	v_dual_add_f32 v13, s101, v13 :: v_dual_add_f32 v8, s60, v8// 000000001B14: C9081A65 0D08103C
	v_dual_add_f32 v9, s65, v9 :: v_dual_add_f32 v14, s24, v14 // 000000001B1C: C9081241 090E1C18
	v_dual_add_f32 v17, s29, v17 :: v_dual_add_f32 v12, s8, v12// 000000001B24: C908221D 110C1808
	v_add_f32_e32 v15, s55, v15                                // 000000001B2C: 061E1E37
	v_add_f32_e32 v21, s19, v21                                // 000000001B30: 062A2A13
	v_dual_add_f32 v19, s49, v19 :: v_dual_add_f32 v8, s61, v8 // 000000001B34: C9082631 1308103D
	v_dual_add_f32 v13, s13, v13 :: v_dual_add_f32 v14, s25, v14// 000000001B3C: C9081A0D 0D0E1C19
	v_dual_add_f32 v9, s66, v9 :: v_dual_add_f32 v12, s9, v12  // 000000001B44: C9081242 090C1809
	v_add_f32_e32 v15, s56, v15                                // 000000001B4C: 061E1E38
	v_add_f32_e32 v21, s20, v21                                // 000000001B50: 062A2A14
	v_writelane_b32 v37, s101, 11                              // 000000001B54: D7610025 00011665
	v_dual_add_f32 v17, s30, v17 :: v_dual_add_f32 v8, s62, v8 // 000000001B5C: C908221E 1108103E
	v_dual_add_f32 v19, s50, v19 :: v_dual_add_f32 v14, s26, v14// 000000001B64: C9082632 130E1C1A
	v_dual_add_f32 v13, s14, v13 :: v_dual_add_f32 v12, s10, v12// 000000001B6C: C9081A0E 0D0C180A
	v_add_f32_e32 v15, s57, v15                                // 000000001B74: 061E1E39
	v_add_f32_e32 v21, s21, v21                                // 000000001B78: 062A2A15
	v_readlane_b32 s34, v37, 4                                 // 000000001B7C: D7600022 00010925
	v_dual_add_f32 v9, s67, v9 :: v_dual_add_f32 v8, s63, v8   // 000000001B84: C9081243 0908103F
	v_dual_add_f32 v17, s31, v17 :: v_dual_add_f32 v14, s27, v14// 000000001B8C: C908221F 110E1C1B
	v_dual_add_f32 v19, s51, v19 :: v_dual_add_f32 v12, s11, v12// 000000001B94: C9082633 130C180B
	v_dual_add_f32 v13, s15, v13 :: v_dual_add_f32 v4, 0, v4   // 000000001B9C: C9081A0F 0D040880
	v_dual_add_f32 v5, 0, v5 :: v_dual_add_f32 v22, 0, v22     // 000000001BA4: C9080A80 05162C80
	v_add_f32_e32 v15, s58, v15                                // 000000001BAC: 061E1E3A
	v_add_f32_e32 v21, s22, v21                                // 000000001BB0: 062A2A16
	v_add_f32_e32 v3, 0, v3                                    // 000000001BB4: 06060680
	v_readlane_b32 s35, v37, 5                                 // 000000001BB8: D7600023 00010B25
	s_mov_b32 vcc_hi, 1                                        // 000000001BC0: BEEB0081
	s_mov_b64 s[102:103], 0                                    // 000000001BC4: BEE60180
	s_branch 357                                               // 000000001BC8: BFA00165 <r_3_3_3_8_8_8+0xb60>
	s_clause 0x1                                               // 000000001BCC: BF850001
	s_load_b512 s[16:31], s[100:101], 0x23c0                   // 000000001BD0: F4100432 F80023C0
	s_load_b512 s[84:99], s[100:101], 0x2500                   // 000000001BD8: F4101532 F8002500
	v_dual_add_f32 v22, s72, v22 :: v_dual_add_f32 v5, s77, v5 // 000000001BE0: C9082C48 16040A4D
	v_add_f32_e32 v24, s68, v24                                // 000000001BE8: 06303044
	s_waitcnt lgkmcnt(0)                                       // 000000001BEC: BF89FC07
	v_add_f32_e32 v10, s93, v10                                // 000000001BF0: 0614145D
	s_add_i32 vcc_hi, vcc_hi, 1                                // 000000001BF4: 816B816B
	s_add_u32 s102, s102, 16                                   // 000000001BF8: 80669066
	v_dual_add_f32 v5, s78, v5 :: v_dual_add_f32 v22, s73, v22 // 000000001BFC: C9080A4E 05162C49
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001C04: BF870132
	v_add_f32_e32 v10, s94, v10                                // 000000001C08: 0614145E
	s_addc_u32 s103, s103, 0                                   // 000000001C0C: 82678067
	s_add_u32 s34, s34, 64                                     // 000000001C10: 8022C022
	v_dual_add_f32 v5, s79, v5 :: v_dual_add_f32 v24, s69, v24 // 000000001C14: C9080A4F 05183045
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001C1C: BF870132
	v_add_f32_e32 v10, s95, v10                                // 000000001C20: 0614145F
	s_addc_u32 s35, s35, 0                                     // 000000001C24: 82238023
	s_cmpk_eq_i32 s102, 0x70                                   // 000000001C28: B1E60070
	v_dual_add_f32 v5, s80, v5 :: v_dual_add_f32 v22, s74, v22 // 000000001C2C: C9080A50 05162C4A
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000001C34: BF8701A2
	v_dual_add_f32 v10, s96, v10 :: v_dual_add_f32 v1, s20, v1 // 000000001C38: C9081460 0A000214
	v_add_f32_e32 v3, s16, v3                                  // 000000001C40: 06060610
	v_dual_add_f32 v5, s81, v5 :: v_dual_add_f32 v24, s70, v24 // 000000001C44: C9080A51 05183046
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001C4C: BF870193
	v_dual_add_f32 v10, s97, v10 :: v_dual_add_f32 v1, s21, v1 // 000000001C50: C9081461 0A000215
	v_add_f32_e32 v3, s17, v3                                  // 000000001C58: 06060611
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001C5C: BF870193
	v_dual_add_f32 v5, s82, v5 :: v_dual_add_f32 v22, s75, v22 // 000000001C60: C9080A52 05162C4B
	v_dual_add_f32 v24, s71, v24 :: v_dual_add_f32 v1, s22, v1 // 000000001C68: C9083047 18000216
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001C70: BF870192
	v_dual_add_f32 v10, s98, v10 :: v_dual_add_f32 v5, s83, v5 // 000000001C74: C9081462 0A040A53
	v_add_f32_e32 v22, s76, v22                                // 000000001C7C: 062C2C4C
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001C80: BF870193
	v_dual_add_f32 v24, s72, v24 :: v_dual_add_f32 v3, s18, v3 // 000000001C84: C9083048 18020612
	v_dual_add_f32 v1, s23, v1 :: v_dual_add_f32 v10, s99, v10 // 000000001C8C: C9080217 010A1463
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001C94: BF870193
	v_add_f32_e32 v22, s77, v22                                // 000000001C98: 062C2C4D
	v_dual_add_f32 v24, s73, v24 :: v_dual_add_f32 v3, s19, v3 // 000000001C9C: C9083049 18020613
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001CA4: BF870193
	v_add_f32_e32 v1, s24, v1                                  // 000000001CA8: 06020218
	v_dual_add_f32 v13, s45, v13 :: v_dual_add_f32 v22, s78, v22// 000000001CAC: C9081A2D 0D162C4E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001CB4: BF870193
	v_dual_add_f32 v24, s74, v24 :: v_dual_add_f32 v3, s20, v3 // 000000001CB8: C908304A 18020614
	v_add_f32_e32 v1, s25, v1                                  // 000000001CC0: 06020219
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001CC4: BF870003
	v_dual_add_f32 v13, s46, v13 :: v_dual_add_f32 v22, s79, v22// 000000001CC8: C9081A2E 0D162C4F
	v_readlane_b32 s68, v37, 28                                // 000000001CD0: D7600044 00013925
	v_readlane_b32 s72, v36, 0                                 // 000000001CD8: D7600048 00010124
	v_readlane_b32 s69, v37, 29                                // 000000001CE0: D7600045 00013B25
	v_readlane_b32 s73, v36, 1                                 // 000000001CE8: D7600049 00010324
	v_readlane_b32 s70, v37, 30                                // 000000001CF0: D7600046 00013D25
	v_add_f32_e32 v6, s68, v6                                  // 000000001CF8: 060C0C44
	v_add_f32_e32 v4, s72, v4                                  // 000000001CFC: 06080848
	v_readlane_b32 s74, v36, 2                                 // 000000001D00: D760004A 00010524
	v_readlane_b32 s77, v36, 5                                 // 000000001D08: D760004D 00010B24
	v_readlane_b32 s75, v36, 3                                 // 000000001D10: D760004B 00010724
	v_dual_add_f32 v6, s69, v6 :: v_dual_add_f32 v1, s26, v1   // 000000001D18: C9080C45 0600021A
	v_add_f32_e32 v4, s73, v4                                  // 000000001D20: 06080849
	v_readlane_b32 s78, v36, 6                                 // 000000001D24: D760004E 00010D24
	v_dual_add_f32 v2, s77, v2 :: v_dual_add_f32 v3, s21, v3   // 000000001D2C: C908044D 02020615
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001D34: BF870214
	v_add_f32_e32 v6, s70, v6                                  // 000000001D38: 060C0C46
	v_add_f32_e32 v4, s74, v4                                  // 000000001D3C: 0608084A
	v_readlane_b32 s79, v36, 7                                 // 000000001D40: D760004F 00010F24
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001D48: BF870194
	v_dual_add_f32 v2, s78, v2 :: v_dual_add_f32 v1, s27, v1   // 000000001D4C: C908044E 0200021B
	v_dual_add_f32 v3, s22, v3 :: v_dual_add_f32 v4, s75, v4   // 000000001D54: C9080616 0304084B
	v_readlane_b32 s71, v37, 31                                // 000000001D5C: D7600047 00013F25
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 000000001D64: BF870233
	v_dual_add_f32 v2, s79, v2 :: v_dual_add_f32 v13, s47, v13 // 000000001D68: C908044F 020C1A2F
	v_readlane_b32 s76, v36, 4                                 // 000000001D70: D760004C 00010924
	v_readlane_b32 s80, v36, 8                                 // 000000001D78: D7600050 00011124
	v_add_f32_e32 v6, s71, v6                                  // 000000001D80: 060C0C47
	v_readlane_b32 s81, v36, 9                                 // 000000001D84: D7600051 00011324
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000001D8C: BF870244
	v_dual_add_f32 v13, s48, v13 :: v_dual_add_f32 v4, s76, v4 // 000000001D90: C9081A30 0D04084C
	v_readlane_b32 s82, v36, 10                                // 000000001D98: D7600052 00011524
	v_readlane_b32 s83, v36, 11                                // 000000001DA0: D7600053 00011724
	v_add_f32_e32 v15, s52, v15                                // 000000001DA8: 061E1E34
	v_dual_add_f32 v13, s49, v13 :: v_dual_add_f32 v2, s80, v2 // 000000001DAC: C9081A31 0D020450
	v_add_f32_e32 v23, s0, v23                                 // 000000001DB4: 062E2E00
	v_add_f32_e32 v7, s88, v7                                  // 000000001DB8: 060E0E58
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001DBC: BF870214
	v_add_f32_e32 v15, s53, v15                                // 000000001DC0: 061E1E35
	v_dual_add_f32 v13, s50, v13 :: v_dual_add_f32 v6, s72, v6 // 000000001DC4: C9081A32 0D060C48
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001DCC: BF870114
	v_dual_add_f32 v2, s81, v2 :: v_dual_add_f32 v23, s1, v23  // 000000001DD0: C9080451 02162E01
	v_dual_add_f32 v0, s25, v0 :: v_dual_add_f32 v13, s51, v13 // 000000001DD8: C9080019 000C1A33
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001DE0: BF870193
	v_add_f32_e32 v6, s73, v6                                  // 000000001DE4: 060C0C49
	v_dual_add_f32 v2, s82, v2 :: v_dual_add_f32 v23, s2, v23  // 000000001DE8: C9080452 02162E02
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001DF0: BF870193
	v_dual_add_f32 v0, s26, v0 :: v_dual_add_f32 v15, s54, v15 // 000000001DF4: C908001A 000E1E36
	v_add_f32_e32 v6, s74, v6                                  // 000000001DFC: 060C0C4A
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001E00: BF870193
	v_dual_add_f32 v2, s83, v2 :: v_dual_add_f32 v23, s3, v23  // 000000001E04: C9080453 02162E03
	v_dual_add_f32 v0, s27, v0 :: v_dual_add_f32 v15, s55, v15 // 000000001E0C: C908001B 000E1E37
	v_add_f32_e32 v7, s89, v7                                  // 000000001E14: 060E0E59
	v_add_f32_e32 v25, s9, v25                                 // 000000001E18: 06323209
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001E1C: BF870213
	v_dual_add_f32 v23, s4, v23 :: v_dual_add_f32 v0, s28, v0  // 000000001E20: C9082E04 1700001C
	v_dual_add_f32 v15, s56, v15 :: v_dual_add_f32 v18, s36, v18// 000000001E28: C9081E38 0F122424
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001E30: BF870194
	v_add_f32_e32 v7, s90, v7                                  // 000000001E34: 060E0E5A
	v_dual_add_f32 v23, s5, v23 :: v_dual_add_f32 v0, s29, v0  // 000000001E38: C9082E05 1700001D
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000001E40: BF870223
	v_add_f32_e32 v15, s57, v15                                // 000000001E44: 061E1E39
	v_add_f32_e32 v25, s10, v25                                // 000000001E48: 0632320A
	v_add_f32_e32 v7, s91, v7                                  // 000000001E4C: 060E0E5B
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001E50: BF870214
	v_dual_add_f32 v23, s6, v23 :: v_dual_add_f32 v0, s30, v0  // 000000001E54: C9082E06 1700001E
	v_add_f32_e32 v15, s58, v15                                // 000000001E5C: 061E1E3A
	v_add_f32_e32 v9, s61, v9                                  // 000000001E60: 0612123D
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001E64: BF870214
	v_add_f32_e32 v7, s92, v7                                  // 000000001E68: 060E0E5C
	v_dual_add_f32 v25, s11, v25 :: v_dual_add_f32 v0, s31, v0 // 000000001E6C: C908320B 1900001F
	v_readlane_b32 s16, v37, 12                                // 000000001E74: D7600010 00011925
	v_add_f32_e32 v4, s77, v4                                  // 000000001E7C: 0608084D
	v_readlane_b32 s17, v37, 13                                // 000000001E80: D7600011 00011B25
	v_readlane_b32 s18, v37, 14                                // 000000001E88: D7600012 00011D25
	v_readlane_b32 s19, v37, 15                                // 000000001E90: D7600013 00011F25
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000001E98: BF870244
	v_dual_add_f32 v21, s16, v21 :: v_dual_add_f32 v4, s78, v4 // 000000001E9C: C9082A10 1504084E
	v_readlane_b32 s20, v37, 16                                // 000000001EA4: D7600014 00012125
	v_readlane_b32 s21, v37, 17                                // 000000001EAC: D7600015 00012325
	v_readlane_b32 s22, v37, 18                                // 000000001EB4: D7600016 00012525
	v_dual_add_f32 v21, s17, v21 :: v_dual_add_f32 v4, s79, v4 // 000000001EBC: C9082A11 1504084F
	v_add_f32_e32 v7, s93, v7                                  // 000000001EC4: 060E0E5D
	v_readlane_b32 s25, v37, 21                                // 000000001EC8: D7600019 00012B25
	v_readlane_b32 s26, v37, 22                                // 000000001ED0: D760001A 00012D25
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000001ED8: BF870244
	v_add_f32_e32 v21, s18, v21                                // 000000001EDC: 062A2A12
	v_readlane_b32 s27, v37, 23                                // 000000001EE0: D760001B 00012F25
	v_add_f32_e32 v7, s94, v7                                  // 000000001EE8: 060E0E5E
	v_dual_add_f32 v17, s25, v17 :: v_dual_add_f32 v14, s20, v14// 000000001EEC: C9082219 110E1C14
	v_add_f32_e32 v21, s19, v21                                // 000000001EF4: 062A2A13
	v_add_f32_e32 v9, s62, v9                                  // 000000001EF8: 0612123E
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001EFC: BF870214
	v_add_f32_e32 v7, s95, v7                                  // 000000001F00: 060E0E5F
	v_dual_add_f32 v17, s26, v17 :: v_dual_add_f32 v26, s4, v26// 000000001F04: C908221A 111A3404
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001F0C: BF870214
	v_dual_add_f32 v21, s20, v21 :: v_dual_add_f32 v12, s40, v12// 000000001F10: C9082A14 150C1828
	v_dual_add_f32 v14, s21, v14 :: v_dual_add_f32 v9, s63, v9 // 000000001F18: C9081C15 0E08123F
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001F20: BF870193
	v_add_f32_e32 v17, s27, v17                                // 000000001F24: 0622221B
	v_dual_add_f32 v21, s21, v21 :: v_dual_add_f32 v12, s41, v12// 000000001F28: C9082A15 150C1829
	v_add_f32_e32 v18, s37, v18                                // 000000001F30: 06242425
	v_readlane_b32 s23, v37, 19                                // 000000001F34: D7600017 00012725
	v_readlane_b32 s28, v37, 24                                // 000000001F3C: D760001C 00013125
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001F44: BF870214
	v_dual_add_f32 v21, s22, v21 :: v_dual_add_f32 v12, s42, v12// 000000001F48: C9082A16 150C182A
	v_add_f32_e32 v18, s38, v18                                // 000000001F50: 06242426
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000001F54: BF870223
	v_dual_add_f32 v14, s22, v14 :: v_dual_add_f32 v17, s28, v17// 000000001F58: C9081C16 0E10221C
	v_readlane_b32 s24, v37, 20                                // 000000001F60: D7600018 00012925
	v_add_f32_e32 v12, s43, v12                                // 000000001F68: 0618182B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001F6C: BF870004
	v_add_f32_e32 v18, s39, v18                                // 000000001F70: 06242427
	v_readlane_b32 s29, v37, 25                                // 000000001F74: D760001D 00013325
	v_add_f32_e32 v25, s12, v25                                // 000000001F7C: 0632320C
	v_readlane_b32 s30, v37, 26                                // 000000001F80: D760001E 00013525
	v_add_f32_e32 v12, s44, v12                                // 000000001F88: 0618182C
	v_add_f32_e32 v18, s40, v18                                // 000000001F8C: 06242428
	v_readlane_b32 s31, v37, 27                                // 000000001F90: D760001F 00013725
	v_add_f32_e32 v9, s64, v9                                  // 000000001F98: 06121240
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001F9C: BF870214
	v_dual_add_f32 v17, s29, v17 :: v_dual_add_f32 v12, s45, v12// 000000001FA0: C908221D 110C182D
	v_dual_add_f32 v18, s41, v18 :: v_dual_add_f32 v25, s13, v25// 000000001FA8: C9082429 1218320D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001FB0: BF870193
	v_add_f32_e32 v9, s65, v9                                  // 000000001FB4: 06121241
	v_dual_add_f32 v17, s30, v17 :: v_dual_add_f32 v12, s46, v12// 000000001FB8: C908221E 110C182E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001FC0: BF870193
	v_dual_add_f32 v18, s42, v18 :: v_dual_add_f32 v25, s14, v25// 000000001FC4: C908242A 1218320E
	v_add_f32_e32 v9, s66, v9                                  // 000000001FCC: 06121242
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001FD0: BF870003
	v_dual_add_f32 v17, s31, v17 :: v_dual_add_f32 v12, s47, v12// 000000001FD4: C908221F 110C182F
	v_readlane_b32 s36, v36, 12                                // 000000001FDC: D7600024 00011924
	v_readlane_b32 s40, v36, 16                                // 000000001FE4: D7600028 00012124
	v_readlane_b32 s41, v36, 17                                // 000000001FEC: D7600029 00012324
	v_readlane_b32 s42, v36, 18                                // 000000001FF4: D760002A 00012524
	v_readlane_b32 s43, v36, 19                                // 000000001FFC: D760002B 00012724
	v_readlane_b32 s37, v36, 13                                // 000000002004: D7600025 00011B24
	v_add_f32_e32 v11, s40, v11                                // 00000000200C: 06161628
	v_readlane_b32 s44, v36, 20                                // 000000002010: D760002C 00012924
	v_add_f32_e32 v20, s36, v20                                // 000000002018: 06282824
	v_readlane_b32 s38, v36, 14                                // 00000000201C: D7600026 00011D24
	v_readlane_b32 s45, v36, 21                                // 000000002024: D760002D 00012B24
	v_add_f32_e32 v11, s41, v11                                // 00000000202C: 06161629
	v_readlane_b32 s39, v36, 15                                // 000000002030: D7600027 00011F24
	v_add_f32_e32 v20, s37, v20                                // 000000002038: 06282825
	v_readlane_b32 s46, v36, 22                                // 00000000203C: D760002E 00012D24
	v_readlane_b32 s47, v36, 23                                // 000000002044: D760002F 00012F24
	v_add_f32_e32 v11, s42, v11                                // 00000000204C: 0616162A
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 000000002050: BF870234
	v_dual_add_f32 v19, s45, v19 :: v_dual_add_f32 v20, s38, v20// 000000002054: C908262D 13142826
	v_readlane_b32 s48, v36, 24                                // 00000000205C: D7600030 00013124
	v_readlane_b32 s49, v36, 25                                // 000000002064: D7600031 00013324
	v_add_f32_e32 v11, s43, v11                                // 00000000206C: 0616162B
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 000000002070: BF870234
	v_dual_add_f32 v19, s46, v19 :: v_dual_add_f32 v20, s39, v20// 000000002074: C908262E 13142827
	v_add_f32_e32 v8, s56, v8                                  // 00000000207C: 06101038
	v_readlane_b32 s50, v36, 26                                // 000000002080: D7600032 00013524
	v_add_f32_e32 v11, s44, v11                                // 000000002088: 0616162C
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 00000000208C: BF8701A4
	v_dual_add_f32 v19, s47, v19 :: v_dual_add_f32 v16, s84, v16// 000000002090: C908262F 13102054
	v_add_f32_e32 v20, s40, v20                                // 000000002098: 06282828
	v_dual_add_f32 v8, s57, v8 :: v_dual_add_f32 v11, s45, v11 // 00000000209C: C9081039 080A162D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000020A4: BF870213
	v_dual_add_f32 v26, s5, v26 :: v_dual_add_f32 v19, s48, v19// 0000000020A8: C9083405 1A122630
	v_add_f32_e32 v16, s85, v16                                // 0000000020B0: 06202055
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000020B4: BF870213
	v_dual_add_f32 v20, s41, v20 :: v_dual_add_f32 v11, s46, v11// 0000000020B8: C9082829 140A162E
	v_add_f32_e32 v8, s58, v8                                  // 0000000020C0: 0610103A
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000020C4: BF870214
	v_add_f32_e32 v26, s6, v26                                 // 0000000020C8: 06343406
	v_add_f32_e32 v16, s86, v16                                // 0000000020CC: 06202056
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000020D0: BF8701A4
	v_dual_add_f32 v20, s42, v20 :: v_dual_add_f32 v11, s47, v11// 0000000020D4: C908282A 140A162F
	v_dual_add_f32 v14, s23, v14 :: v_dual_add_f32 v19, s49, v19// 0000000020DC: C9081C17 0E122631
	v_add_f32_e32 v16, s87, v16                                // 0000000020E4: 06202057
	v_add_f32_e32 v8, s59, v8                                  // 0000000020E8: 0610103B
	v_add_f32_e32 v26, s7, v26                                 // 0000000020EC: 06343407
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000020F0: BF870004
	v_add_f32_e32 v14, s24, v14                                // 0000000020F4: 061C1C18
	v_readlane_b32 s51, v36, 27                                // 0000000020F8: D7600033 00013724
	v_add_f32_e32 v16, s88, v16                                // 000000002100: 06202058
	v_dual_add_f32 v8, s60, v8 :: v_dual_add_f32 v19, s50, v19 // 000000002104: C908103C 08122632
	v_add_f32_e32 v26, s8, v26                                 // 00000000210C: 06343408
	v_add_f32_e32 v14, s25, v14                                // 000000002110: 061C1C19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000002114: BF870214
	v_add_f32_e32 v16, s89, v16                                // 000000002118: 06202059
	v_dual_add_f32 v8, s61, v8 :: v_dual_add_f32 v19, s51, v19 // 00000000211C: C908103D 08122633
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000002124: BF870214
	v_add_f32_e32 v26, s9, v26                                 // 000000002128: 06343409
	v_add_f32_e32 v14, s26, v14                                // 00000000212C: 061C1C1A
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000002130: BF870214
	v_add_f32_e32 v16, s90, v16                                // 000000002134: 0620205A
	v_dual_add_f32 v8, s62, v8 :: v_dual_add_f32 v9, s67, v9   // 000000002138: C908103E 08081243
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000002140: BF870214
	v_dual_add_f32 v26, s10, v26 :: v_dual_add_f32 v25, s15, v25// 000000002144: C908340A 1A18320F
	v_add_f32_e32 v14, s27, v14                                // 00000000214C: 061C1C1B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000002150: BF870193
	v_add_f32_e32 v8, s63, v8                                  // 000000002154: 0610103F
	v_add_f32_e32 v26, s11, v26                                // 000000002158: 0634340B
	s_cbranch_scc1 64860                                       // 00000000215C: BFA2FD5C <r_3_3_3_8_8_8+0xd0>
	v_readlane_b32 s0, v37, 8                                  // 000000002160: D7600000 00011125
	s_mov_b32 s2, -1                                           // 000000002168: BE8200C1
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000216C: BF870001
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 000000002170: 916A007E
	s_cbranch_vccnz 10                                         // 000000002174: BFA4000A <r_3_3_3_8_8_8+0xba0>
	v_readlane_b32 s0, v37, 6                                  // 000000002178: D7600000 00010D25
	v_readlane_b32 s1, v37, 7                                  // 000000002180: D7600001 00010F25
	s_cmpk_lg_i32 s102, 0x60                                   // 000000002188: B2660060
	s_mov_b32 s2, 0                                            // 00000000218C: BE820080
	s_cselect_b32 s6, -1, 0                                    // 000000002190: 980680C1
	s_add_u32 s0, s0, s102                                     // 000000002194: 80006600
	s_addc_u32 s1, s1, s103                                    // 000000002198: 82016701
	s_mov_b32 s48, 0                                           // 00000000219C: BEB00080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000021A0: BF870009
	s_mov_b32 s49, s48                                         // 0000000021A4: BEB10030
	s_mov_b32 s50, s48                                         // 0000000021A8: BEB20030
	s_mov_b32 s51, s48                                         // 0000000021AC: BEB30030
	s_mov_b32 s40, s48                                         // 0000000021B0: BEA80030
	s_mov_b32 s41, s48                                         // 0000000021B4: BEA90030
	s_mov_b32 s42, s48                                         // 0000000021B8: BEAA0030
	s_mov_b32 s43, s48                                         // 0000000021BC: BEAB0030
	s_mov_b32 s36, s48                                         // 0000000021C0: BEA40030
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 0000000021C4: 916A027E
	s_mov_b32 s37, s48                                         // 0000000021C8: BEA50030
	s_mov_b32 s38, s48                                         // 0000000021CC: BEA60030
	s_mov_b32 s39, s48                                         // 0000000021D0: BEA70030
	s_mov_b32 s44, s48                                         // 0000000021D4: BEAC0030
	s_mov_b32 s45, s48                                         // 0000000021D8: BEAD0030
	s_mov_b32 s46, s48                                         // 0000000021DC: BEAE0030
	s_mov_b32 s47, s48                                         // 0000000021E0: BEAF0030
	s_mov_b32 s76, s48                                         // 0000000021E4: BECC0030
	s_mov_b32 s77, s48                                         // 0000000021E8: BECD0030
	s_mov_b32 s78, s48                                         // 0000000021EC: BECE0030
	s_mov_b32 s79, s48                                         // 0000000021F0: BECF0030
	s_mov_b32 s68, s48                                         // 0000000021F4: BEC40030
	s_mov_b32 s69, s48                                         // 0000000021F8: BEC50030
	s_mov_b32 s70, s48                                         // 0000000021FC: BEC60030
	s_mov_b32 s71, s48                                         // 000000002200: BEC70030
	s_mov_b32 s72, s48                                         // 000000002204: BEC80030
	s_mov_b32 s73, s48                                         // 000000002208: BEC90030
	s_mov_b32 s74, s48                                         // 00000000220C: BECA0030
	s_mov_b32 s75, s48                                         // 000000002210: BECB0030
	s_mov_b32 s80, s48                                         // 000000002214: BED00030
	s_mov_b32 s81, s48                                         // 000000002218: BED10030
	s_mov_b32 s82, s48                                         // 00000000221C: BED20030
	s_mov_b32 s83, s48                                         // 000000002220: BED30030
	s_mov_b32 s16, s48                                         // 000000002224: BE900030
	s_mov_b32 s17, s48                                         // 000000002228: BE910030
	s_mov_b32 s18, s48                                         // 00000000222C: BE920030
	s_mov_b32 s19, s48                                         // 000000002230: BE930030
	s_mov_b32 s8, s48                                          // 000000002234: BE880030
	s_mov_b32 s9, s48                                          // 000000002238: BE890030
	s_mov_b32 s10, s48                                         // 00000000223C: BE8A0030
	s_mov_b32 s11, s48                                         // 000000002240: BE8B0030
	s_mov_b32 s12, s48                                         // 000000002244: BE8C0030
	s_mov_b32 s13, s48                                         // 000000002248: BE8D0030
	s_mov_b32 s14, s48                                         // 00000000224C: BE8E0030
	s_mov_b32 s15, s48                                         // 000000002250: BE8F0030
	s_mov_b32 s20, s48                                         // 000000002254: BE940030
	s_mov_b32 s21, s48                                         // 000000002258: BE950030
	s_mov_b32 s22, s48                                         // 00000000225C: BE960030
	s_mov_b32 s23, s48                                         // 000000002260: BE970030
	s_cbranch_vccnz 41                                         // 000000002264: BFA40029 <r_3_3_3_8_8_8+0xd0c>
	s_add_u32 s4, s34, 0xfffffd80                              // 000000002268: 8004FF22 FFFFFD80
	s_addc_u32 s5, s35, -1                                     // 000000002270: 8205C123
	s_add_u32 s2, s34, 0xfffffec0                              // 000000002274: 8002FF22 FFFFFEC0
	s_addc_u32 s3, s35, -1                                     // 00000000227C: 8203C123
	s_mov_b32 s1, 0                                            // 000000002280: BE810080
	s_cmpk_eq_i32 s102, 0x60                                   // 000000002284: B1E60060
	s_mov_b32 s23, 0                                           // 000000002288: BE970080
	s_mov_b32 s22, 0                                           // 00000000228C: BE960080
	s_mov_b32 s21, 0                                           // 000000002290: BE950080
	s_mov_b32 s20, 0                                           // 000000002294: BE940080
	s_mov_b32 s15, 0                                           // 000000002298: BE8F0080
	s_mov_b32 s14, 0                                           // 00000000229C: BE8E0080
	s_mov_b32 s13, 0                                           // 0000000022A0: BE8D0080
	s_mov_b32 s12, 0                                           // 0000000022A4: BE8C0080
	s_mov_b32 s11, 0                                           // 0000000022A8: BE8B0080
	s_mov_b32 s10, 0                                           // 0000000022AC: BE8A0080
	s_mov_b32 s9, 0                                            // 0000000022B0: BE890080
	s_mov_b32 s8, 0                                            // 0000000022B4: BE880080
	s_mov_b32 s19, 0                                           // 0000000022B8: BE930080
	s_mov_b32 s18, 0                                           // 0000000022BC: BE920080
	s_mov_b32 s17, 0                                           // 0000000022C0: BE910080
	s_mov_b32 s16, 0                                           // 0000000022C4: BE900080
	s_mov_b32 s6, 0                                            // 0000000022C8: BE860080
	s_cbranch_scc1 3                                           // 0000000022CC: BFA20003 <r_3_3_3_8_8_8+0xcdc>
	s_load_b512 s[8:23], s[34:35], null                        // 0000000022D0: F4100211 F8000000
	s_mov_b32 s6, -1                                           // 0000000022D8: BE8600C1
	s_clause 0x1                                               // 0000000022DC: BF850001
	s_load_b512 s[68:83], s[4:5], null                         // 0000000022E0: F4101102 F8000000
	s_load_b512 s[36:51], s[2:3], null                         // 0000000022E8: F4100901 F8000000
	v_readlane_b32 s2, v37, 10                                 // 0000000022F0: D7600002 00011525
	s_lshl_b32 s0, vcc_hi, 4                                   // 0000000022F8: 8400846B
	v_readlane_b32 s3, v37, 11                                 // 0000000022FC: D7600003 00011725
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002304: BF870002
	s_add_i32 s0, s0, s2                                       // 000000002308: 81000200
	s_waitcnt lgkmcnt(0)                                       // 00000000230C: BF89FC07
	v_writelane_b32 v37, s8, 12                                // 000000002310: D7610025 00011808
	v_cndmask_b32_e64 v27, 0, 1, s6                            // 000000002318: D501001B 00190280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000002320: 84808200
	s_mov_b32 s96, 0                                           // 000000002324: BEE00080
	s_mov_b32 s60, 0                                           // 000000002328: BEBC0080
	v_writelane_b32 v37, s9, 13                                // 00000000232C: D7610025 00011A09
	v_cmp_ne_u32_e64 s33, 1, v27                               // 000000002334: D44D0021 00023681
	s_mov_b32 s61, 0                                           // 00000000233C: BEBD0080
	s_mov_b32 s62, 0                                           // 000000002340: BEBE0080
	s_mov_b32 s63, 0                                           // 000000002344: BEBF0080
	v_writelane_b32 v37, s10, 14                               // 000000002348: D7610025 00011C0A
	s_mov_b32 s52, 0                                           // 000000002350: BEB40080
	s_mov_b32 s53, 0                                           // 000000002354: BEB50080
	s_mov_b32 s54, 0                                           // 000000002358: BEB60080
	s_mov_b32 s55, 0                                           // 00000000235C: BEB70080
	v_writelane_b32 v37, s11, 15                               // 000000002360: D7610025 00011E0B
	s_mov_b32 s56, 0                                           // 000000002368: BEB80080
	s_mov_b32 s57, 0                                           // 00000000236C: BEB90080
	s_mov_b32 s58, 0                                           // 000000002370: BEBA0080
	s_mov_b32 s59, 0                                           // 000000002374: BEBB0080
	v_writelane_b32 v37, s12, 16                               // 000000002378: D7610025 0001200C
	s_mov_b32 s64, 0                                           // 000000002380: BEC00080
	s_mov_b32 s65, 0                                           // 000000002384: BEC10080
	s_mov_b32 s66, 0                                           // 000000002388: BEC20080
	s_mov_b32 s67, 0                                           // 00000000238C: BEC30080
	v_writelane_b32 v37, s13, 17                               // 000000002390: D7610025 0001220D
	v_writelane_b32 v37, s14, 18                               // 000000002398: D7610025 0001240E
	v_writelane_b32 v37, s15, 19                               // 0000000023A0: D7610025 0001260F
	v_writelane_b32 v37, s16, 20                               // 0000000023A8: D7610025 00012810
	v_writelane_b32 v37, s17, 21                               // 0000000023B0: D7610025 00012A11
	v_writelane_b32 v37, s18, 22                               // 0000000023B8: D7610025 00012C12
	v_writelane_b32 v37, s19, 23                               // 0000000023C0: D7610025 00012E13
	v_writelane_b32 v37, s20, 24                               // 0000000023C8: D7610025 00013014
	v_writelane_b32 v37, s21, 25                               // 0000000023D0: D7610025 00013215
	v_writelane_b32 v37, s22, 26                               // 0000000023D8: D7610025 00013416
	v_writelane_b32 v37, s23, 27                               // 0000000023E0: D7610025 00013617
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 0000000023E8: BF8701C1
	v_readlane_b32 s8, v37, 0                                  // 0000000023EC: D7600008 00010125
	v_readlane_b32 s10, v37, 2                                 // 0000000023F4: D760000A 00010525
	v_readlane_b32 s11, v37, 3                                 // 0000000023FC: D760000B 00010725
	v_readlane_b32 s9, v37, 1                                  // 000000002404: D7600009 00010325
	s_add_u32 s100, s10, s0                                    // 00000000240C: 8064000A
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002410: BF870002
	s_addc_u32 s101, s11, s1                                   // 000000002414: 8265010B
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 000000002418: 916A067E
	s_cbranch_vccnz 2                                          // 00000000241C: BFA40002 <r_3_3_3_8_8_8+0xe28>
	s_load_b512 s[52:67], s[100:101], 0x1240                   // 000000002420: F4100D32 F8001240
	s_load_b512 s[0:15], s[100:101], 0xfc0                     // 000000002428: F4100032 F8000FC0
	s_mov_b32 s97, 0                                           // 000000002430: BEE10080
	s_mov_b32 s98, 0                                           // 000000002434: BEE20080
	s_and_not1_b32 vcc_lo, exec_lo, s104                       // 000000002438: 916A687E
	s_mov_b32 s99, 0                                           // 00000000243C: BEE30080
	s_mov_b32 s88, 0                                           // 000000002440: BED80080
	s_mov_b32 s89, 0                                           // 000000002444: BED90080
	s_mov_b32 s90, 0                                           // 000000002448: BEDA0080
	s_mov_b32 s91, 0                                           // 00000000244C: BEDB0080
	s_mov_b32 s84, 0                                           // 000000002450: BED40080
	s_mov_b32 s85, 0                                           // 000000002454: BED50080
	s_mov_b32 s86, 0                                           // 000000002458: BED60080
	s_mov_b32 s87, 0                                           // 00000000245C: BED70080
	s_mov_b32 s92, 0                                           // 000000002460: BEDC0080
	s_mov_b32 s93, 0                                           // 000000002464: BEDD0080
	s_mov_b32 s94, 0                                           // 000000002468: BEDE0080
	s_mov_b32 s95, 0                                           // 00000000246C: BEDF0080
	s_mov_b32 s24, 0                                           // 000000002470: BE980080
	s_mov_b32 s25, 0                                           // 000000002474: BE990080
	s_mov_b32 s26, 0                                           // 000000002478: BE9A0080
	s_waitcnt lgkmcnt(0)                                       // 00000000247C: BF89FC07
	v_writelane_b32 v37, s0, 28                                // 000000002480: D7610025 00013800
	v_writelane_b32 v36, s4, 0                                 // 000000002488: D7610024 00010004
	s_mov_b32 s27, 0                                           // 000000002490: BE9B0080
	s_mov_b32 s16, 0                                           // 000000002494: BE900080
	s_mov_b32 s17, 0                                           // 000000002498: BE910080
	v_writelane_b32 v37, s1, 29                                // 00000000249C: D7610025 00013A01
	v_writelane_b32 v36, s5, 1                                 // 0000000024A4: D7610024 00010205
	s_mov_b32 s18, 0                                           // 0000000024AC: BE920080
	s_mov_b32 s19, 0                                           // 0000000024B0: BE930080
	s_mov_b32 s20, 0                                           // 0000000024B4: BE940080
	v_writelane_b32 v37, s2, 30                                // 0000000024B8: D7610025 00013C02
	v_writelane_b32 v36, s6, 2                                 // 0000000024C0: D7610024 00010406
	s_mov_b32 s21, 0                                           // 0000000024C8: BE950080
	s_mov_b32 s22, 0                                           // 0000000024CC: BE960080
	s_mov_b32 s23, 0                                           // 0000000024D0: BE970080
	v_writelane_b32 v37, s3, 31                                // 0000000024D4: D7610025 00013E03
	v_writelane_b32 v36, s7, 3                                 // 0000000024DC: D7610024 00010607
	s_mov_b32 s28, 0                                           // 0000000024E4: BE9C0080
	s_mov_b32 s29, 0                                           // 0000000024E8: BE9D0080
	s_mov_b32 s30, 0                                           // 0000000024EC: BE9E0080
	s_mov_b32 s31, 0                                           // 0000000024F0: BE9F0080
	v_writelane_b32 v36, s8, 4                                 // 0000000024F4: D7610024 00010808
	v_writelane_b32 v36, s9, 5                                 // 0000000024FC: D7610024 00010A09
	v_writelane_b32 v36, s10, 6                                // 000000002504: D7610024 00010C0A
	v_writelane_b32 v36, s11, 7                                // 00000000250C: D7610024 00010E0B
	v_writelane_b32 v36, s12, 8                                // 000000002514: D7610024 0001100C
	v_writelane_b32 v36, s13, 9                                // 00000000251C: D7610024 0001120D
	v_writelane_b32 v36, s14, 10                               // 000000002524: D7610024 0001140E
	v_writelane_b32 v36, s15, 11                               // 00000000252C: D7610024 0001160F
	s_load_b512 s[0:15], s[100:101], 0x1100                    // 000000002534: F4100032 F8001100
	s_waitcnt lgkmcnt(0)                                       // 00000000253C: BF89FC07
	v_writelane_b32 v36, s0, 12                                // 000000002540: D7610024 00011800
	v_writelane_b32 v36, s1, 13                                // 000000002548: D7610024 00011A01
	v_writelane_b32 v36, s2, 14                                // 000000002550: D7610024 00011C02
	v_writelane_b32 v36, s3, 15                                // 000000002558: D7610024 00011E03
	v_writelane_b32 v36, s4, 16                                // 000000002560: D7610024 00012004
	v_writelane_b32 v36, s5, 17                                // 000000002568: D7610024 00012205
	v_writelane_b32 v36, s6, 18                                // 000000002570: D7610024 00012406
	v_writelane_b32 v36, s7, 19                                // 000000002578: D7610024 00012607
	v_writelane_b32 v36, s8, 20                                // 000000002580: D7610024 00012808
	v_writelane_b32 v36, s9, 21                                // 000000002588: D7610024 00012A09
	v_writelane_b32 v36, s10, 22                               // 000000002590: D7610024 00012C0A
	v_writelane_b32 v36, s11, 23                               // 000000002598: D7610024 00012E0B
	v_writelane_b32 v36, s12, 24                               // 0000000025A0: D7610024 0001300C
	v_writelane_b32 v36, s13, 25                               // 0000000025A8: D7610024 0001320D
	v_writelane_b32 v36, s14, 26                               // 0000000025B0: D7610024 0001340E
	v_writelane_b32 v36, s15, 27                               // 0000000025B8: D7610024 0001360F
	s_mov_b32 s8, 0                                            // 0000000025C0: BE880080
	s_mov_b32 s9, 0                                            // 0000000025C4: BE890080
	s_mov_b32 s10, 0                                           // 0000000025C8: BE8A0080
	s_mov_b32 s11, 0                                           // 0000000025CC: BE8B0080
	s_mov_b32 s0, 0                                            // 0000000025D0: BE800080
	s_mov_b32 s1, 0                                            // 0000000025D4: BE810080
	s_mov_b32 s2, 0                                            // 0000000025D8: BE820080
	s_mov_b32 s3, 0                                            // 0000000025DC: BE830080
	s_mov_b32 s4, 0                                            // 0000000025E0: BE840080
	s_mov_b32 s5, 0                                            // 0000000025E4: BE850080
	s_mov_b32 s6, 0                                            // 0000000025E8: BE860080
	s_mov_b32 s7, 0                                            // 0000000025EC: BE870080
	s_mov_b32 s12, 0                                           // 0000000025F0: BE8C0080
	s_mov_b32 s13, 0                                           // 0000000025F4: BE8D0080
	s_mov_b32 s14, 0                                           // 0000000025F8: BE8E0080
	s_mov_b32 s15, 0                                           // 0000000025FC: BE8F0080
	s_cbranch_vccnz 64887                                      // 000000002600: BFA4FD77 <r_3_3_3_8_8_8+0x5e0>
	s_and_b32 vcc_lo, exec_lo, s33                             // 000000002604: 8B6A217E
	s_cbranch_vccnz 64880                                      // 000000002608: BFA4FD70 <r_3_3_3_8_8_8+0x5cc>
	s_load_b512 s[0:15], s[100:101], 0x2640                    // 00000000260C: F4100032 F8002640
	s_branch 64877                                             // 000000002614: BFA0FD6D <r_3_3_3_8_8_8+0x5cc>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002618: BF870001
	v_readlane_b32 s0, v37, 0                                  // 00000000261C: D7600000 00010125
	v_dual_mov_b32 v35, 0 :: v_dual_mul_f32 v26, 0x3b272f05, v26// 000000002624: CA060080 231A34FF 3B272F05
	v_dual_mul_f32 v27, 0x3b3f112b, v25 :: v_dual_mul_f32 v0, 0x3b3f112b, v0// 000000002630: C8C632FF 1B0000FF 3B3F112B
	v_mul_f32_e32 v25, 0x3b3f112b, v23                         // 00000000263C: 10322EFF 3B3F112B
	v_readlane_b32 s1, v37, 1                                  // 000000002644: D7600001 00010325
	v_mul_f32_e32 v23, 0x3b3f112b, v24                         // 00000000264C: 102E30FF 3B3F112B
	v_dual_mul_f32 v24, 0x3b272f05, v22 :: v_dual_mul_f32 v15, 0x3b272f05, v15// 000000002654: C8C62CFF 180E1EFF 3B272F05
	v_mul_f32_e32 v28, 0x3b272f05, v13                         // 000000002660: 10381AFF 3B272F05
	global_store_b96 v35, v[25:27], s[0:1] offset:96           // 000000002668: DC720060 00001923
	v_mul_f32_e32 v25, 0x3b3f112b, v5                          // 000000002670: 10320AFF 3B3F112B
	v_mul_f32_e32 v26, 0x3b272f05, v18                         // 000000002678: 103424FF 3B272F05
	v_mul_f32_e32 v27, 0x3b124925, v12                         // 000000002680: 103618FF 3B124925
	v_mul_f32_e32 v29, 0x3b3f112b, v21                         // 000000002688: 103A2AFF 3B3F112B
	v_mul_f32_e32 v30, 0x3b272f05, v14                         // 000000002690: 103C1CFF 3B272F05
	v_mul_f32_e32 v14, 0x3b124925, v19                         // 000000002698: 101C26FF 3B124925
	v_mul_f32_e32 v31, 0x3b3f112b, v17                         // 0000000026A0: 103E22FF 3B3F112B
	v_mul_f32_e32 v32, 0x3b272f05, v6                          // 0000000026A8: 10400CFF 3B272F05
	v_mul_f32_e32 v33, 0x3b124925, v4                          // 0000000026B0: 104208FF 3B124925
	v_mul_f32_e32 v34, 0x3b272f05, v2                          // 0000000026B8: 104404FF 3B272F05
	v_mul_f32_e32 v2, 0x3b124925, v7                           // 0000000026C0: 10040EFF 3B124925
	v_mul_f32_e32 v12, 0x3b124925, v20                         // 0000000026C8: 101828FF 3B124925
	v_mul_f32_e32 v13, 0x3b000000, v11                         // 0000000026D0: 101A16FF 3B000000
	v_mul_f32_e32 v17, 0x3b124925, v8                          // 0000000026D8: 102210FF 3B124925
	v_mul_f32_e32 v18, 0x3b272f05, v9                          // 0000000026E0: 102412FF 3B272F05
	v_mul_f32_e32 v19, 0x3b3f112b, v3                          // 0000000026E8: 102606FF 3B3F112B
	v_mul_f32_e32 v20, 0x3b272f05, v1                          // 0000000026F0: 102802FF 3B272F05
	v_readlane_b32 s2, v37, 2                                  // 0000000026F8: D7600002 00010525
	v_readlane_b32 s3, v37, 3                                  // 000000002700: D7600003 00010725
	v_mul_f32_e32 v1, 0x3b272f05, v16                          // 000000002708: 100220FF 3B272F05
	v_mul_f32_e32 v3, 0x3b272f05, v10                          // 000000002710: 100614FF 3B272F05
	s_clause 0x5                                               // 000000002718: BF850005
	global_store_b128 v35, v[23:26], s[0:1]                    // 00000000271C: DC760000 00001723
	global_store_b128 v35, v[27:30], s[0:1] offset:16          // 000000002724: DC760010 00001B23
	global_store_b128 v35, v[31:34], s[0:1] offset:32          // 00000000272C: DC760020 00001F23
	global_store_b128 v35, v[12:15], s[0:1] offset:48          // 000000002734: DC760030 00000C23
	global_store_b128 v35, v[17:20], s[0:1] offset:64          // 00000000273C: DC760040 00001123
	global_store_b128 v35, v[0:3], s[0:1] offset:80            // 000000002744: DC760050 00000023
	s_nop 0                                                    // 00000000274C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000002750: BFB60003
	s_endpgm                                                   // 000000002754: BFB00000

.rodata
.amdhsa_kernel kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel
