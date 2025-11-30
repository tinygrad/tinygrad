.text
.globl kernel
.p2align 8 // TODO: need more?
.type kernel,@function

kernel:

	s_load_b128 s[4:7], s[0:1], 0x0                            // 000000001600: F4004100 F8000000
	s_mov_b64 s[2:3], 16                                       // 000000001608: BE820190
	s_movk_i32 s0, 0xfe80                                      // 00000000160C: B000FE80
	v_writelane_b32 v28, s2, 0                                 // 000000001610: D761001C 00010002
	s_mov_b32 s1, -1                                           // 000000001618: BE8100C1
	s_mov_b32 s85, 0                                           // 00000000161C: BED50080
	s_mov_b32 s8, 0                                            // 000000001620: BE880080
	s_mov_b32 s25, 0                                           // 000000001624: BE990080
	v_writelane_b32 v28, s3, 1                                 // 000000001628: D761001C 00010203
	s_mov_b32 s19, 0                                           // 000000001630: BE930080
	s_mov_b32 s18, 0                                           // 000000001634: BE920080
	s_mov_b32 s22, 0                                           // 000000001638: BE960080
	s_mov_b32 s100, 0                                          // 00000000163C: BEE40080
	s_mov_b32 s69, 0                                           // 000000001640: BEC50080
	s_mov_b32 s72, 0                                           // 000000001644: BEC80080
	s_mov_b32 s68, 0                                           // 000000001648: BEC40080
	s_mov_b32 s27, 0                                           // 00000000164C: BE9B0080
	s_mov_b32 s13, 0                                           // 000000001650: BE8D0080
	s_mov_b32 s21, 0                                           // 000000001654: BE950080
	s_mov_b32 s23, 0                                           // 000000001658: BE970080
	s_wait_kmcnt 0x0                                           // 00000000165C: BFC70000
	v_writelane_b32 v28, s4, 2                                 // 000000001660: D761001C 00010404
	s_add_nc_u64 s[0:1], s[6:7], s[0:1]                        // 000000001668: A9800006
	s_mov_b32 s101, 0                                          // 00000000166C: BEE50080
	s_mov_b32 s70, 0                                           // 000000001670: BEC60080
	s_mov_b32 s74, 0                                           // 000000001674: BECA0080
	v_writelane_b32 v28, s5, 3                                 // 000000001678: D761001C 00010605
	s_mov_b32 s84, 0                                           // 000000001680: BED40080
	s_mov_b32 s73, 0                                           // 000000001684: BEC90080
	s_mov_b32 s26, 0                                           // 000000001688: BE9A0080
	s_mov_b32 s9, 0                                            // 00000000168C: BE890080
	v_writelane_b32 v28, s6, 4                                 // 000000001690: D761001C 00010806
	s_mov_b32 s20, 0                                           // 000000001698: BE940080
	s_mov_b32 s24, 0                                           // 00000000169C: BE980080
	s_mov_b32 s104, 0                                          // 0000000016A0: BEE80080
	s_mov_b32 s71, 0                                           // 0000000016A4: BEC70080
	v_writelane_b32 v28, s7, 5                                 // 0000000016A8: D761001C 00010A07
	s_mov_b32 s75, 0                                           // 0000000016B0: BECB0080
	s_mov_b32 s33, 0                                           // 0000000016B4: BEA10080
	v_writelane_b32 v28, s0, 6                                 // 0000000016B8: D761001C 00010C00
	v_writelane_b32 v28, s1, 7                                 // 0000000016C0: D761001C 00010E01
	s_mov_b32 s0, 0                                            // 0000000016C8: BE800080
	s_wait_alu 0xfffe                                          // 0000000016CC: BF88FFFE
	v_writelane_b32 v28, s0, 8                                 // 0000000016D0: D761001C 00011000
	s_branch 30                                                // 0000000016D8: BFA0001E <r_3_3_3_8_8_8+0x154>
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000016DC: BF870131
	v_readlane_b32 s0, v28, 0                                  // 0000000016E0: D7600000 0001011C
	v_readlane_b32 s1, v28, 1                                  // 0000000016E8: D7600001 0001031C
	v_readlane_b32 s33, v27, 2                                 // 0000000016F0: D7600021 0001051B
	s_add_nc_u64 s[0:1], s[0:1], 0x100                         // 0000000016F8: A980FF00 00000100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 000000001700: BF8700D1
	s_add_co_i32 s33, s33, 1                                   // 000000001704: 81218121
	s_wait_alu 0xfffe                                          // 000000001708: BF88FFFE
	v_writelane_b32 v28, s0, 0                                 // 00000000170C: D761001C 00010000
	s_cmp_eq_u32 s33, 8                                        // 000000001714: BF068821
	v_writelane_b32 v28, s1, 1                                 // 000000001718: D761001C 00010201
	v_readlane_b32 s0, v28, 6                                  // 000000001720: D7600000 00010D1C
	v_readlane_b32 s1, v28, 7                                  // 000000001728: D7600001 00010F1C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001730: BF870001
	s_add_nc_u64 s[0:1], s[0:1], 0x400                         // 000000001734: A980FF00 00000400
	s_wait_alu 0xfffe                                          // 00000000173C: BF88FFFE
	v_writelane_b32 v28, s0, 6                                 // 000000001740: D761001C 00010C00
	v_writelane_b32 v28, s1, 7                                 // 000000001748: D761001C 00010E01
	s_cbranch_scc1 1307                                        // 000000001750: BFA2051B <r_3_3_3_8_8_8+0x15c0>
	v_writelane_b32 v28, s26, 9                                // 000000001754: D761001C 0001121A
	s_cmp_eq_u32 s33, 0                                        // 00000000175C: BF068021
	s_mov_b32 s37, -1                                          // 000000001760: BEA500C1
	s_cselect_b32 s0, -1, 0                                    // 000000001764: 980080C1
	s_lshl_b32 s36, s33, 8                                     // 000000001768: 84248821
	v_writelane_b32 v28, s25, 10                               // 00000000176C: D761001C 00011419
	s_cmp_lg_u32 s33, 0                                        // 000000001774: BF078021
	v_writelane_b32 v28, s18, 11                               // 000000001778: D761001C 00011612
	s_wait_alu 0xfffe                                          // 000000001780: BF88FFFE
	v_writelane_b32 v28, s0, 12                                // 000000001784: D761001C 00011800
	s_cbranch_scc1 3                                           // 00000000178C: BFA20003 <r_3_3_3_8_8_8+0x19c>
	s_mov_b32 s37, 0                                           // 000000001790: BEA50080
	s_wait_alu 0xfffe                                          // 000000001794: BF88FFFE
	s_mov_b64 s[34:35], s[36:37]                               // 000000001798: BEA20124
	v_writelane_b32 v28, s27, 13                               // 00000000179C: D761001C 00011A1B
	s_mov_b32 s103, s85                                        // 0000000017A4: BEE70055
	s_mov_b32 s102, s84                                        // 0000000017A8: BEE60054
	s_mov_b32 s14, s37                                         // 0000000017AC: BE8E0025
	s_mov_b32 s15, s37                                         // 0000000017B0: BE8F0025
	v_writelane_b32 v28, s75, 14                               // 0000000017B4: D761001C 00011C4B
	s_mov_b32 s4, s37                                          // 0000000017BC: BE840025
	s_mov_b32 s5, s37                                          // 0000000017C0: BE850025
	s_mov_b32 s6, s37                                          // 0000000017C4: BE860025
	s_mov_b32 s7, s37                                          // 0000000017C8: BE870025
	v_writelane_b32 v28, s74, 15                               // 0000000017CC: D761001C 00011E4A
	s_mov_b32 s0, s37                                          // 0000000017D4: BE800025
	s_and_not1_b32 vcc_lo, exec_lo, s37                        // 0000000017D8: 916A257E
	s_mov_b32 s1, s37                                          // 0000000017DC: BE810025
	s_mov_b32 s2, s37                                          // 0000000017E0: BE820025
	v_writelane_b32 v28, s24, 16                               // 0000000017E4: D761001C 00012018
	s_mov_b32 s3, s37                                          // 0000000017EC: BE830025
	s_mov_b32 s10, s37                                         // 0000000017F0: BE8A0025
	s_mov_b32 s11, s37                                         // 0000000017F4: BE8B0025
	s_mov_b32 s24, s37                                         // 0000000017F8: BE980025
	v_writelane_b32 v28, s9, 17                                // 0000000017FC: D761001C 00012209
	s_mov_b32 s9, s37                                          // 000000001804: BE890025
	s_mov_b32 s25, s37                                         // 000000001808: BE990025
	s_mov_b32 s26, s37                                         // 00000000180C: BE9A0025
	s_mov_b32 s27, s37                                         // 000000001810: BE9B0025
	v_writelane_b32 v28, s13, 18                               // 000000001814: D761001C 0001240D
	s_mov_b32 s13, s37                                         // 00000000181C: BE8D0025
	s_mov_b32 s16, s37                                         // 000000001820: BE900025
	s_mov_b32 s17, s37                                         // 000000001824: BE910025
	s_mov_b32 s18, s37                                         // 000000001828: BE920025
	v_writelane_b32 v28, s23, 19                               // 00000000182C: D761001C 00012617
	s_mov_b32 s23, s37                                         // 000000001834: BE970025
	s_mov_b32 s28, s37                                         // 000000001838: BE9C0025
	s_mov_b32 s29, s37                                         // 00000000183C: BE9D0025
	s_mov_b32 s30, s37                                         // 000000001840: BE9E0025
	v_writelane_b32 v28, s22, 20                               // 000000001844: D761001C 00012816
	s_mov_b32 s22, s37                                         // 00000000184C: BE960025
	s_mov_b32 s31, s37                                         // 000000001850: BE9F0025
	v_writelane_b32 v28, s21, 21                               // 000000001854: D761001C 00012A15
	s_mov_b32 s21, s37                                         // 00000000185C: BE950025
	v_writelane_b32 v28, s20, 22                               // 000000001860: D761001C 00012C14
	s_mov_b32 s20, s37                                         // 000000001868: BE940025
	v_writelane_b32 v28, s73, 23                               // 00000000186C: D761001C 00012E49
	v_writelane_b32 v28, s72, 24                               // 000000001874: D761001C 00013048
	v_writelane_b32 v28, s68, 25                               // 00000000187C: D761001C 00013244
	v_writelane_b32 v28, s71, 26                               // 000000001884: D761001C 00013447
	v_writelane_b32 v28, s19, 27                               // 00000000188C: D761001C 00013613
	s_mov_b32 s19, s37                                         // 000000001894: BE930025
	v_writelane_b32 v28, s70, 28                               // 000000001898: D761001C 00013846
	v_writelane_b32 v28, s69, 29                               // 0000000018A0: D761001C 00013A45
	v_writelane_b32 v28, s8, 30                                // 0000000018A8: D761001C 00013C08
	s_mov_b32 s8, s37                                          // 0000000018B0: BE880025
	s_cbranch_vccnz 29                                         // 0000000018B4: BFA4001D <r_3_3_3_8_8_8+0x32c>
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018B8: BF870001
	v_readlane_b32 s4, v28, 2                                  // 0000000018BC: D7600004 0001051C
	v_readlane_b32 s6, v28, 4                                  // 0000000018C4: D7600006 0001091C
	v_readlane_b32 s7, v28, 5                                  // 0000000018CC: D7600007 00010B1C
	s_mov_b32 s37, 0                                           // 0000000018D4: BEA50080
	s_movk_i32 s18, 0xfe40                                     // 0000000018D8: B012FE40
	s_wait_alu 0xfffe                                          // 0000000018DC: BF88FFFE
	s_lshl_b64 s[0:1], s[36:37], 2                             // 0000000018E0: 84808224
	s_mov_b32 s19, -1                                          // 0000000018E4: BE9300C1
	s_wait_alu 0xfffe                                          // 0000000018E8: BF88FFFE
	s_add_nc_u64 s[16:17], s[6:7], s[0:1]                      // 0000000018EC: A9900006
	s_movk_i32 s0, 0xfd00                                      // 0000000018F0: B000FD00
	s_mov_b32 s1, -1                                           // 0000000018F4: BE8100C1
	v_readlane_b32 s5, v28, 3                                  // 0000000018F8: D7600005 0001071C
	s_wait_alu 0xfffe                                          // 000000001900: BF88FFFE
	s_add_nc_u64 s[0:1], s[16:17], s[0:1]                      // 000000001904: A9800010
	s_add_nc_u64 s[16:17], s[16:17], s[18:19]                  // 000000001908: A9901210
	s_clause 0x1                                               // 00000000190C: BF850001
	s_load_b512 s[0:15], s[0:1], 0x0                           // 000000001910: F4008000 F8000000
	s_load_b512 s[16:31], s[16:17], 0x0                        // 000000001918: F4008408 F8000000
	s_mov_b64 s[34:35], s[36:37]                               // 000000001920: BEA20124
	s_wait_kmcnt 0x0                                           // 000000001924: BFC70000
	s_mov_b32 s37, s12                                         // 000000001928: BEA5000C
	v_writelane_b32 v28, s36, 31                               // 00000000192C: D761001C 00013E24
	s_cmp_lg_u32 s33, 7                                        // 000000001934: BF078721
	s_wait_alu 0xfffe                                          // 000000001938: BF88FFFE
	v_writelane_b32 v27, s37, 0                                // 00000000193C: D761001B 00010025
	s_cselect_b32 s12, -1, 0                                   // 000000001944: 980C80C1
	s_lshl_b64 s[34:35], s[34:35], 2                           // 000000001948: 84A28222
	v_readlane_b32 s36, v28, 2                                 // 00000000194C: D7600024 0001051C
	v_readlane_b32 s38, v28, 4                                 // 000000001954: D7600026 0001091C
	v_readlane_b32 s39, v28, 5                                 // 00000000195C: D7600027 00010B1C
	v_readlane_b32 s37, v28, 3                                 // 000000001964: D7600025 0001071C
	s_wait_alu 0xfffe                                          // 00000000196C: BF88FFFE
	v_writelane_b32 v27, s12, 1                                // 000000001970: D761001B 0001020C
	s_cmp_eq_u32 s33, 7                                        // 000000001978: BF068721
	s_mov_b32 s64, 0                                           // 00000000197C: BEC00080
	s_add_nc_u64 s[34:35], s[38:39], s[34:35]                  // 000000001980: A9A22226
	s_clause 0x1                                               // 000000001984: BF850001
	s_load_b512 s[36:51], s[34:35], 0x1100                     // 000000001988: F4008911 F8001100
	s_load_b512 s[68:83], s[34:35], 0x1240                     // 000000001990: F4009111 F8001240
	s_mov_b32 s65, 0                                           // 000000001998: BEC10080
	s_mov_b32 s66, 0                                           // 00000000199C: BEC20080
	s_mov_b32 s67, 0                                           // 0000000019A0: BEC30080
	s_mov_b32 s56, 0                                           // 0000000019A4: BEB80080
	s_mov_b32 s57, 0                                           // 0000000019A8: BEB90080
	s_mov_b32 s58, 0                                           // 0000000019AC: BEBA0080
	s_mov_b32 s59, 0                                           // 0000000019B0: BEBB0080
	s_mov_b32 s52, 0                                           // 0000000019B4: BEB40080
	s_mov_b32 s53, 0                                           // 0000000019B8: BEB50080
	s_mov_b32 s54, 0                                           // 0000000019BC: BEB60080
	s_mov_b32 s55, 0                                           // 0000000019C0: BEB70080
	s_mov_b32 s60, 0                                           // 0000000019C4: BEBC0080
	s_mov_b32 s61, 0                                           // 0000000019C8: BEBD0080
	s_mov_b32 s62, 0                                           // 0000000019CC: BEBE0080
	s_mov_b32 s63, 0                                           // 0000000019D0: BEBF0080
	s_mov_b32 s92, 0                                           // 0000000019D4: BEDC0080
	s_mov_b32 s93, 0                                           // 0000000019D8: BEDD0080
	s_mov_b32 s94, 0                                           // 0000000019DC: BEDE0080
	s_mov_b32 s95, 0                                           // 0000000019E0: BEDF0080
	s_mov_b32 s84, 0                                           // 0000000019E4: BED40080
	s_mov_b32 s85, 0                                           // 0000000019E8: BED50080
	s_mov_b32 s86, 0                                           // 0000000019EC: BED60080
	s_mov_b32 s87, 0                                           // 0000000019F0: BED70080
	s_mov_b32 s88, 0                                           // 0000000019F4: BED80080
	s_mov_b32 s89, 0                                           // 0000000019F8: BED90080
	s_mov_b32 s90, 0                                           // 0000000019FC: BEDA0080
	s_mov_b32 s91, 0                                           // 000000001A00: BEDB0080
	s_mov_b32 s96, 0                                           // 000000001A04: BEE00080
	s_mov_b32 s97, 0                                           // 000000001A08: BEE10080
	s_mov_b32 s98, 0                                           // 000000001A0C: BEE20080
	s_mov_b32 s99, 0                                           // 000000001A10: BEE30080
	v_writelane_b32 v27, s33, 2                                // 000000001A14: D761001B 00010421
	s_cbranch_scc1 5                                           // 000000001A1C: BFA20005 <r_3_3_3_8_8_8+0x434>
	s_clause 0x1                                               // 000000001A20: BF850001
	s_load_b512 s[52:67], s[34:35], 0x2500                     // 000000001A24: F4008D11 F8002500
	s_load_b512 s[84:99], s[34:35], 0x2640                     // 000000001A2C: F4009511 F8002640
	v_readlane_b32 s12, v28, 8                                 // 000000001A34: D760000C 0001111C
	s_wait_kmcnt 0x0                                           // 000000001A3C: BFC70000
	s_add_f32 s33, s103, s93                                   // 000000001A40: A0215D67
	s_wait_alu 0xfffe                                          // 000000001A44: BF88FFFE
	s_add_f32 s34, s102, s84                                   // 000000001A48: A0225466
	s_add_f32 s100, s100, 0                                    // 000000001A4C: A0648064
	s_add_f32 s101, s101, 0                                    // 000000001A50: A0658065
	s_add_f32 s12, s12, s88                                    // 000000001A54: A00C580C
	s_add_f32 s33, s94, s33                                    // 000000001A58: A021215E
	s_add_f32 s34, s85, s34                                    // 000000001A5C: A0222255
	s_add_f32 s104, s104, 0                                    // 000000001A60: A0688068
	s_wait_alu 0xfffe                                          // 000000001A64: BF88FFFE
	s_add_f32 s12, s89, s12                                    // 000000001A68: A00C0C59
	s_add_f32 s33, s95, s33                                    // 000000001A6C: A021215F
	s_add_f32 s34, s86, s34                                    // 000000001A70: A0222256
	s_mov_b32 vcc_hi, 1                                        // 000000001A74: BEEB0081
	s_wait_alu 0xfffe                                          // 000000001A78: BF88FFFE
	s_add_f32 s12, s90, s12                                    // 000000001A7C: A00C0C5A
	s_add_f32 s33, s33, s96                                    // 000000001A80: A0216021
	s_add_f32 s34, s87, s34                                    // 000000001A84: A0222257
	s_mov_b64 s[102:103], 0                                    // 000000001A88: BEE60180
	s_wait_alu 0xfffe                                          // 000000001A8C: BF88FFFE
	s_add_f32 s12, s91, s12                                    // 000000001A90: A00C0C5B
	s_add_f32 s33, s97, s33                                    // 000000001A94: A0212161
	s_add_f32 s34, s88, s34                                    // 000000001A98: A0222258
	s_wait_alu 0xfffe                                          // 000000001A9C: BF88FFFE
	s_add_f32 s12, s92, s12                                    // 000000001AA0: A00C0C5C
	s_add_f32 s33, s98, s33                                    // 000000001AA4: A0212162
	s_add_f32 s34, s89, s34                                    // 000000001AA8: A0222259
	s_wait_alu 0xfffe                                          // 000000001AAC: BF88FFFE
	s_add_f32 s12, s93, s12                                    // 000000001AB0: A00C0C5D
	s_add_f32 s85, s99, s33                                    // 000000001AB4: A0552163
	s_add_f32 s84, s90, s34                                    // 000000001AB8: A054225A
	s_wait_alu 0xfffe                                          // 000000001ABC: BF88FFFE
	s_add_f32 s12, s94, s12                                    // 000000001AC0: A00C0C5E
	s_wait_alu 0xfffe                                          // 000000001AC4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 000000001AC8: BF87052A
	s_add_f32 s12, s95, s12                                    // 000000001ACC: A00C0C5F
	s_wait_alu 0xfffe                                          // 000000001AD0: BF88FFFE
	v_writelane_b32 v28, s12, 8                                // 000000001AD4: D761001C 0001100C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 000000001ADC: BF8701B1
	v_readlane_b32 s33, v28, 30                                // 000000001AE0: D7600021 00013D1C
	v_readlane_b32 s12, v28, 25                                // 000000001AE8: D760000C 0001331C
	v_readlane_b32 s34, v28, 23                                // 000000001AF0: D7600022 00012F1C
	s_add_f32 s33, s33, s77                                    // 000000001AF8: A0214D21
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001AFC: BF870092
	s_add_f32 s12, s12, s72                                    // 000000001B00: A00C480C
	s_add_f32 s34, s34, s68                                    // 000000001B04: A0224422
	s_wait_alu 0xfffe                                          // 000000001B08: BF88FFFE
	s_add_f32 s33, s78, s33                                    // 000000001B0C: A021214E
	s_add_f32 s12, s73, s12                                    // 000000001B10: A00C0C49
	s_add_f32 s34, s69, s34                                    // 000000001B14: A0222245
	s_wait_alu 0xfffe                                          // 000000001B18: BF88FFFE
	s_add_f32 s33, s79, s33                                    // 000000001B1C: A021214F
	s_add_f32 s12, s74, s12                                    // 000000001B20: A00C0C4A
	s_add_f32 s34, s70, s34                                    // 000000001B24: A0222246
	s_wait_alu 0xfffe                                          // 000000001B28: BF88FFFE
	s_add_f32 s33, s33, s80                                    // 000000001B2C: A0215021
	s_add_f32 s12, s75, s12                                    // 000000001B30: A00C0C4B
	s_add_f32 s34, s71, s34                                    // 000000001B34: A0222247
	s_wait_alu 0xfffe                                          // 000000001B38: BF88FFFE
	s_add_f32 s33, s81, s33                                    // 000000001B3C: A0212151
	s_add_f32 s12, s76, s12                                    // 000000001B40: A00C0C4C
	s_add_f32 s34, s34, s72                                    // 000000001B44: A0224822
	s_wait_alu 0xfffe                                          // 000000001B48: BF88FFFE
	s_add_f32 s33, s82, s33                                    // 000000001B4C: A0212152
	s_add_f32 s12, s77, s12                                    // 000000001B50: A00C0C4D
	s_add_f32 s34, s73, s34                                    // 000000001B54: A0222249
	s_wait_alu 0xfffe                                          // 000000001B58: BF88FFFE
	s_add_f32 s33, s83, s33                                    // 000000001B5C: A0212153
	s_add_f32 s12, s78, s12                                    // 000000001B60: A00C0C4E
	s_add_f32 s73, s74, s34                                    // 000000001B64: A049224A
	s_wait_alu 0xfffe                                          // 000000001B68: BF88FFFE
	v_writelane_b32 v28, s33, 30                               // 000000001B6C: D761001C 00013C21
	s_add_f32 s68, s79, s12                                    // 000000001B74: A0440C4F
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001B78: BF870001
	v_readlane_b32 s33, v28, 10                                // 000000001B7C: D7600021 0001151C
	v_readlane_b32 s12, v28, 13                                // 000000001B84: D760000C 00011B1C
	v_readlane_b32 s34, v28, 9                                 // 000000001B8C: D7600022 0001131C
	v_readlane_b32 s69, v28, 29                                // 000000001B94: D7600045 00013B1C
	v_readlane_b32 s72, v28, 24                                // 000000001B9C: D7600048 0001311C
	s_add_f32 s33, s33, s25                                    // 000000001BA4: A0211921
	s_add_f32 s12, s12, s20                                    // 000000001BA8: A00C140C
	s_add_f32 s16, s34, s16                                    // 000000001BAC: A0101022
	v_readlane_b32 s70, v28, 28                                // 000000001BB0: D7600046 0001391C
	s_wait_alu 0xfffe                                          // 000000001BB8: BF88FFFE
	s_add_f32 s33, s26, s33                                    // 000000001BBC: A021211A
	s_add_f32 s12, s21, s12                                    // 000000001BC0: A00C0C15
	s_add_f32 s16, s17, s16                                    // 000000001BC4: A0101011
	v_readlane_b32 s74, v28, 15                                // 000000001BC8: D760004A 00011F1C
	s_wait_alu 0xfffe                                          // 000000001BD0: BF88FFFE
	s_add_f32 s17, s27, s33                                    // 000000001BD4: A011211B
	s_add_f32 s12, s22, s12                                    // 000000001BD8: A00C0C16
	s_add_f32 s16, s18, s16                                    // 000000001BDC: A0101012
	v_readlane_b32 s71, v28, 26                                // 000000001BE0: D7600047 0001351C
	s_wait_alu 0xfffe                                          // 000000001BE8: BF88FFFE
	s_add_f32 s17, s17, s28                                    // 000000001BEC: A0111C11
	s_add_f32 s12, s23, s12                                    // 000000001BF0: A00C0C17
	s_add_f32 s16, s19, s16                                    // 000000001BF4: A0101013
	v_readlane_b32 s18, v28, 31                                // 000000001BF8: D7600012 00013F1C
	s_wait_alu 0xfffe                                          // 000000001C00: BF88FFFE
	s_add_f32 s17, s29, s17                                    // 000000001C04: A011111D
	s_add_f32 s12, s24, s12                                    // 000000001C08: A00C0C18
	s_add_f32 s16, s20, s16                                    // 000000001C0C: A0101014
	v_readlane_b32 s19, v27, 0                                 // 000000001C10: D7600013 0001011B
	s_wait_alu 0xfffe                                          // 000000001C18: BF88FFFE
	s_add_f32 s17, s30, s17                                    // 000000001C1C: A011111E
	s_add_f32 s12, s25, s12                                    // 000000001C20: A00C0C19
	s_add_f32 s16, s21, s16                                    // 000000001C24: A0101015
	v_readlane_b32 s18, v28, 19                                // 000000001C28: D7600012 0001271C
	s_wait_alu 0xfffe                                          // 000000001C30: BF88FFFE
	s_add_f32 s25, s31, s17                                    // 000000001C34: A019111F
	v_readlane_b32 s17, v28, 20                                // 000000001C38: D7600011 0001291C
	s_add_f32 s12, s26, s12                                    // 000000001C40: A00C0C1A
	s_add_f32 s26, s22, s16                                    // 000000001C44: A01A1016
	v_readlane_b32 s16, v28, 11                                // 000000001C48: D7600010 0001171C
	s_add_f32 s18, s18, s4                                     // 000000001C50: A0120412
	s_wait_alu 0xfffe                                          // 000000001C54: BF88FFFE
	s_add_f32 s27, s27, s12                                    // 000000001C58: A01B0C1B
	v_readlane_b32 s12, v28, 27                                // 000000001C5C: D760000C 0001371C
	s_add_f32 s17, s17, s9                                     // 000000001C64: A0110911
	s_add_f32 s16, s16, s45                                    // 000000001C68: A0102D10
	v_readlane_b32 s75, v28, 14                                // 000000001C6C: D760004B 00011D1C
	v_readlane_b32 s34, v28, 6                                 // 000000001C74: D7600022 00010D1C
	s_wait_alu 0xfffe                                          // 000000001C7C: BF88FFFE
	s_add_f32 s17, s10, s17                                    // 000000001C80: A011110A
	s_add_f32 s12, s12, s61                                    // 000000001C84: A00C3D0C
	s_add_f32 s16, s46, s16                                    // 000000001C88: A010102E
	v_readlane_b32 s35, v28, 7                                 // 000000001C8C: D7600023 00010F1C
	s_wait_alu 0xfffe                                          // 000000001C94: BF88FFFE
	s_add_f32 s17, s11, s17                                    // 000000001C98: A011110B
	s_add_f32 s12, s62, s12                                    // 000000001C9C: A00C0C3E
	s_add_f32 s16, s47, s16                                    // 000000001CA0: A010102F
	s_add_f32 s69, s69, 0                                      // 000000001CA4: A0458045
	s_wait_alu 0xfffe                                          // 000000001CA8: BF88FFFE
	s_add_f32 s17, s17, s19                                    // 000000001CAC: A0111311
	s_add_f32 s12, s63, s12                                    // 000000001CB0: A00C0C3F
	s_add_f32 s16, s48, s16                                    // 000000001CB4: A0101030
	s_add_f32 s72, s72, 0                                      // 000000001CB8: A0488048
	s_wait_alu 0xfffe                                          // 000000001CBC: BF88FFFE
	s_add_f32 s13, s13, s17                                    // 000000001CC0: A00D110D
	s_add_f32 s12, s12, s64                                    // 000000001CC4: A00C400C
	v_readlane_b32 s17, v28, 21                                // 000000001CC8: D7600011 00012B1C
	s_add_f32 s16, s49, s16                                    // 000000001CD0: A0101031
	s_wait_alu 0xfffe                                          // 000000001CD4: BF88FFFE
	s_add_f32 s13, s14, s13                                    // 000000001CD8: A00D0D0E
	v_readlane_b32 s14, v28, 18                                // 000000001CDC: D760000E 0001251C
	s_add_f32 s12, s65, s12                                    // 000000001CE4: A00C0C41
	s_add_f32 s17, s17, s40                                    // 000000001CE8: A0112811
	s_add_f32 s16, s50, s16                                    // 000000001CEC: A0101032
	s_wait_alu 0xfffe                                          // 000000001CF0: BF88FFFE
	s_add_f32 s22, s15, s13                                    // 000000001CF4: A0160D0F
	s_add_f32 s12, s66, s12                                    // 000000001CF8: A00C0C42
	s_add_f32 s14, s14, s56                                    // 000000001CFC: A00E380E
	s_add_f32 s70, s70, 0                                      // 000000001D00: A0468046
	s_add_f32 s74, s74, 0                                      // 000000001D04: A04A804A
	s_wait_alu 0xfffe                                          // 000000001D08: BF88FFFE
	s_add_f32 s19, s67, s12                                    // 000000001D0C: A0130C43
	s_add_f32 s12, s57, s14                                    // 000000001D10: A00C0E39
	s_add_f32 s14, s41, s17                                    // 000000001D14: A00E1129
	s_add_f32 s17, s5, s18                                     // 000000001D18: A0111205
	s_add_f32 s18, s51, s16                                    // 000000001D1C: A0121033
	s_wait_alu 0xfffe                                          // 000000001D20: BF88FFFE
	s_add_f32 s12, s58, s12                                    // 000000001D24: A00C0C3A
	s_add_f32 s14, s42, s14                                    // 000000001D28: A00E0E2A
	s_add_f32 s16, s6, s17                                     // 000000001D2C: A0101106
	s_add_f32 s71, s71, 0                                      // 000000001D30: A0478047
	s_wait_alu 0xfffe                                          // 000000001D34: BF88FFFE
	s_add_f32 s12, s59, s12                                    // 000000001D38: A00C0C3B
	s_add_f32 s13, s43, s14                                    // 000000001D3C: A00D0E2B
	s_add_f32 s7, s7, s16                                      // 000000001D40: A0071007
	s_add_f32 s75, s75, 0                                      // 000000001D44: A04B804B
	s_wait_alu 0xfffe                                          // 000000001D48: BF88FFFE
	s_add_f32 s12, s60, s12                                    // 000000001D4C: A00C0C3C
	s_add_f32 s13, s44, s13                                    // 000000001D50: A00D0D2C
	s_add_f32 s7, s8, s7                                       // 000000001D54: A0070708
	s_wait_alu 0xfffe                                          // 000000001D58: BF88FFFE
	s_add_f32 s8, s61, s12                                     // 000000001D5C: A0080C3D
	s_add_f32 s12, s45, s13                                    // 000000001D60: A00C0D2D
	s_add_f32 s7, s9, s7                                       // 000000001D64: A0070709
	v_readlane_b32 s13, v28, 16                                // 000000001D68: D760000D 0001211C
	s_wait_alu 0xfffe                                          // 000000001D70: BF88FFFE
	s_add_f32 s8, s62, s8                                      // 000000001D74: A008083E
	s_add_f32 s9, s46, s12                                     // 000000001D78: A0090C2E
	s_add_f32 s7, s10, s7                                      // 000000001D7C: A007070A
	v_readlane_b32 s10, v28, 17                                // 000000001D80: D760000A 0001231C
	v_readlane_b32 s12, v28, 22                                // 000000001D88: D760000C 00012D1C
	s_add_f32 s0, s13, s0                                      // 000000001D90: A000000D
	s_wait_alu 0xfffe                                          // 000000001D94: BF88FFFE
	s_add_f32 s13, s63, s8                                     // 000000001D98: A00D083F
	s_add_f32 s21, s47, s9                                     // 000000001D9C: A015092F
	s_add_f32 s10, s10, s52                                    // 000000001DA0: A00A340A
	s_add_f32 s12, s12, s36                                    // 000000001DA4: A00C240C
	s_add_f32 s0, s1, s0                                       // 000000001DA8: A0000001
	s_add_f32 s23, s11, s7                                     // 000000001DAC: A017070B
	s_add_f32 s8, s53, s10                                     // 000000001DB0: A0080A35
	s_wait_alu 0xfffe                                          // 000000001DB4: BF88FFFE
	s_add_f32 s10, s37, s12                                    // 000000001DB8: A00A0C25
	s_add_f32 s0, s2, s0                                       // 000000001DBC: A0000002
	s_add_f32 s1, s54, s8                                      // 000000001DC0: A0010836
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001DC4: BF870009
	s_add_f32 s8, s38, s10                                     // 000000001DC8: A0080A26
	s_wait_alu 0xfffe                                          // 000000001DCC: BF88FFFE
	s_add_f32 s0, s3, s0                                       // 000000001DD0: A0000003
	s_add_f32 s1, s55, s1                                      // 000000001DD4: A0010137
	s_add_f32 s2, s39, s8                                      // 000000001DD8: A0020827
	s_wait_alu 0xfffe                                          // 000000001DDC: BF88FFFE
	s_add_f32 s0, s4, s0                                       // 000000001DE0: A0000004
	s_add_f32 s1, s56, s1                                      // 000000001DE4: A0010138
	s_add_f32 s2, s40, s2                                      // 000000001DE8: A0020228
	s_wait_alu 0xfffe                                          // 000000001DEC: BF88FFFE
	s_add_f32 s0, s5, s0                                       // 000000001DF0: A0000005
	s_add_f32 s1, s57, s1                                      // 000000001DF4: A0010139
	s_add_f32 s2, s41, s2                                      // 000000001DF8: A0020229
	s_wait_alu 0xfffe                                          // 000000001DFC: BF88FFFE
	s_add_f32 s24, s6, s0                                      // 000000001E00: A0180006
	s_add_f32 s9, s58, s1                                      // 000000001E04: A009013A
	s_add_f32 s20, s42, s2                                     // 000000001E08: A014022A
	v_writelane_b32 v28, s27, 13                               // 000000001E0C: D761001C 00011A1B
	s_mov_b32 s3, -1                                           // 000000001E14: BE8300C1
	v_writelane_b32 v28, s75, 14                               // 000000001E18: D761001C 00011C4B
	v_writelane_b32 v28, s74, 15                               // 000000001E20: D761001C 00011E4A
	s_wait_alu 0xfffe                                          // 000000001E28: BF88FFFE
	v_writelane_b32 v28, s24, 16                               // 000000001E2C: D761001C 00012018
	v_writelane_b32 v28, s9, 17                                // 000000001E34: D761001C 00012209
	v_writelane_b32 v28, s13, 18                               // 000000001E3C: D761001C 0001240D
	v_writelane_b32 v28, s23, 19                               // 000000001E44: D761001C 00012617
	v_writelane_b32 v28, s22, 20                               // 000000001E4C: D761001C 00012816
	v_writelane_b32 v28, s21, 21                               // 000000001E54: D761001C 00012A15
	v_writelane_b32 v28, s20, 22                               // 000000001E5C: D761001C 00012C14
	v_writelane_b32 v28, s73, 23                               // 000000001E64: D761001C 00012E49
	v_writelane_b32 v28, s72, 24                               // 000000001E6C: D761001C 00013048
	v_writelane_b32 v28, s68, 25                               // 000000001E74: D761001C 00013244
	v_writelane_b32 v28, s71, 26                               // 000000001E7C: D761001C 00013447
	v_writelane_b32 v28, s19, 27                               // 000000001E84: D761001C 00013613
	v_writelane_b32 v28, s70, 28                               // 000000001E8C: D761001C 00013846
	v_writelane_b32 v28, s69, 29                               // 000000001E94: D761001C 00013A45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001E9C: BF870091
	v_readlane_b32 s0, v28, 12                                 // 000000001EA0: D7600000 0001191C
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 000000001EA8: 916A007E
	s_cbranch_vccnz 10                                         // 000000001EAC: BFA4000A <r_3_3_3_8_8_8+0x8d8>
	v_readlane_b32 s0, v28, 0                                  // 000000001EB0: D7600000 0001011C
	v_readlane_b32 s1, v28, 1                                  // 000000001EB8: D7600001 0001031C
	s_cmp_lg_u32 s102, 0x60                                    // 000000001EC0: BF07FF66 00000060
	s_mov_b32 s3, 0                                            // 000000001EC8: BE830080
	s_cselect_b32 s2, -1, 0                                    // 000000001ECC: 980280C1
	s_mov_b32 s64, 0                                           // 000000001ED0: BEC00080
	s_add_nc_u64 s[0:1], s[0:1], s[102:103]                    // 000000001ED4: A9806600
	s_mov_b32 s36, s64                                         // 000000001ED8: BEA40040
	s_mov_b32 s44, s64                                         // 000000001EDC: BEAC0040
	s_mov_b32 s45, s64                                         // 000000001EE0: BEAD0040
	s_mov_b32 s46, s64                                         // 000000001EE4: BEAE0040
	s_mov_b32 s47, s64                                         // 000000001EE8: BEAF0040
	s_mov_b32 s37, s64                                         // 000000001EEC: BEA50040
	s_mov_b32 s38, s64                                         // 000000001EF0: BEA60040
	s_mov_b32 s39, s64                                         // 000000001EF4: BEA70040
	s_mov_b32 s40, s64                                         // 000000001EF8: BEA80040
	s_mov_b32 s41, s64                                         // 000000001EFC: BEA90040
	s_mov_b32 s42, s64                                         // 000000001F00: BEAA0040
	s_mov_b32 s43, s64                                         // 000000001F04: BEAB0040
	s_mov_b32 s48, s64                                         // 000000001F08: BEB00040
	s_mov_b32 s49, s64                                         // 000000001F0C: BEB10040
	s_mov_b32 s50, s64                                         // 000000001F10: BEB20040
	s_mov_b32 s51, s64                                         // 000000001F14: BEB30040
	s_wait_alu 0xfffe                                          // 000000001F18: BF88FFFE
	v_writelane_b32 v27, s36, 3                                // 000000001F1C: D761001B 00010624
	s_mov_b32 s65, s64                                         // 000000001F24: BEC10040
	s_mov_b32 s66, s64                                         // 000000001F28: BEC20040
	s_mov_b32 s67, s64                                         // 000000001F2C: BEC30040
	s_mov_b32 s56, s64                                         // 000000001F30: BEB80040
	v_writelane_b32 v27, s37, 4                                // 000000001F34: D761001B 00010825
	s_mov_b32 s57, s64                                         // 000000001F3C: BEB90040
	s_mov_b32 s58, s64                                         // 000000001F40: BEBA0040
	s_mov_b32 s59, s64                                         // 000000001F44: BEBB0040
	s_mov_b32 s52, s64                                         // 000000001F48: BEB40040
	v_writelane_b32 v27, s38, 5                                // 000000001F4C: D761001B 00010A26
	s_and_not1_b32 vcc_lo, exec_lo, s3                         // 000000001F54: 916A037E
	s_mov_b32 s53, s64                                         // 000000001F58: BEB50040
	s_mov_b32 s54, s64                                         // 000000001F5C: BEB60040
	s_mov_b32 s55, s64                                         // 000000001F60: BEB70040
	v_writelane_b32 v27, s39, 6                                // 000000001F64: D761001B 00010C27
	s_mov_b32 s60, s64                                         // 000000001F6C: BEBC0040
	s_mov_b32 s61, s64                                         // 000000001F70: BEBD0040
	s_mov_b32 s62, s64                                         // 000000001F74: BEBE0040
	s_mov_b32 s63, s64                                         // 000000001F78: BEBF0040
	v_writelane_b32 v27, s40, 7                                // 000000001F7C: D761001B 00010E28
	s_mov_b32 s76, s64                                         // 000000001F84: BECC0040
	s_mov_b32 s77, s64                                         // 000000001F88: BECD0040
	s_mov_b32 s78, s64                                         // 000000001F8C: BECE0040
	s_mov_b32 s79, s64                                         // 000000001F90: BECF0040
	v_writelane_b32 v27, s41, 8                                // 000000001F94: D761001B 00011029
	s_mov_b32 s68, s64                                         // 000000001F9C: BEC40040
	s_mov_b32 s69, s64                                         // 000000001FA0: BEC50040
	s_mov_b32 s70, s64                                         // 000000001FA4: BEC60040
	s_mov_b32 s71, s64                                         // 000000001FA8: BEC70040
	v_writelane_b32 v27, s42, 9                                // 000000001FAC: D761001B 0001122A
	s_mov_b32 s72, s64                                         // 000000001FB4: BEC80040
	s_mov_b32 s73, s64                                         // 000000001FB8: BEC90040
	s_mov_b32 s74, s64                                         // 000000001FBC: BECA0040
	s_mov_b32 s75, s64                                         // 000000001FC0: BECB0040
	v_writelane_b32 v27, s43, 10                               // 000000001FC4: D761001B 0001142B
	s_mov_b32 s80, s64                                         // 000000001FCC: BED00040
	s_mov_b32 s81, s64                                         // 000000001FD0: BED10040
	s_mov_b32 s82, s64                                         // 000000001FD4: BED20040
	s_mov_b32 s83, s64                                         // 000000001FD8: BED30040
	v_writelane_b32 v27, s44, 11                               // 000000001FDC: D761001B 0001162C
	v_writelane_b32 v27, s45, 12                               // 000000001FE4: D761001B 0001182D
	v_writelane_b32 v27, s46, 13                               // 000000001FEC: D761001B 00011A2E
	v_writelane_b32 v27, s47, 14                               // 000000001FF4: D761001B 00011C2F
	v_writelane_b32 v27, s48, 15                               // 000000001FFC: D761001B 00011E30
	v_writelane_b32 v27, s49, 16                               // 000000002004: D761001B 00012031
	v_writelane_b32 v27, s50, 17                               // 00000000200C: D761001B 00012232
	v_writelane_b32 v27, s51, 18                               // 000000002014: D761001B 00012433
	s_cbranch_vccnz 79                                         // 00000000201C: BFA4004F <r_3_3_3_8_8_8+0xb5c>
	s_mov_b32 s1, 0                                            // 000000002020: BE810080
	s_cmp_eq_u32 s102, 0x60                                    // 000000002024: BF06FF66 00000060
	s_mov_b32 s51, 0                                           // 00000000202C: BEB30080
	s_mov_b32 s50, 0                                           // 000000002030: BEB20080
	s_mov_b32 s49, 0                                           // 000000002034: BEB10080
	s_mov_b32 s48, 0                                           // 000000002038: BEB00080
	s_mov_b32 s43, 0                                           // 00000000203C: BEAB0080
	s_mov_b32 s42, 0                                           // 000000002040: BEAA0080
	s_mov_b32 s41, 0                                           // 000000002044: BEA90080
	s_mov_b32 s40, 0                                           // 000000002048: BEA80080
	s_mov_b32 s39, 0                                           // 00000000204C: BEA70080
	s_mov_b32 s38, 0                                           // 000000002050: BEA60080
	s_mov_b32 s37, 0                                           // 000000002054: BEA50080
	s_mov_b32 s36, 0                                           // 000000002058: BEA40080
	s_mov_b32 s47, 0                                           // 00000000205C: BEAF0080
	s_mov_b32 s46, 0                                           // 000000002060: BEAE0080
	s_mov_b32 s45, 0                                           // 000000002064: BEAD0080
	s_mov_b32 s44, 0                                           // 000000002068: BEAC0080
	s_mov_b32 s2, 0                                            // 00000000206C: BE820080
	s_cbranch_scc1 3                                           // 000000002070: BFA20003 <r_3_3_3_8_8_8+0xa80>
	s_load_b512 s[36:51], s[34:35], 0x0                        // 000000002074: F4008911 F8000000
	s_mov_b32 s2, -1                                           // 00000000207C: BE8200C1
	s_wait_kmcnt 0x0                                           // 000000002080: BFC70000
	s_wait_alu 0xfffe                                          // 000000002084: BF88FFFE
	v_writelane_b32 v27, s36, 3                                // 000000002088: D761001B 00010624
	s_movk_i32 s4, 0xfd80                                      // 000000002090: B004FD80
	s_mov_b32 s5, -1                                           // 000000002094: BE8500C1
	s_movk_i32 s6, 0xfec0                                      // 000000002098: B006FEC0
	s_mov_b32 s7, -1                                           // 00000000209C: BE8700C1
	v_writelane_b32 v27, s37, 4                                // 0000000020A0: D761001B 00010825
	s_wait_alu 0xfffe                                          // 0000000020A8: BF88FFFE
	s_add_nc_u64 s[4:5], s[34:35], s[4:5]                      // 0000000020AC: A9840422
	s_add_nc_u64 s[6:7], s[34:35], s[6:7]                      // 0000000020B0: A9860622
	s_clause 0x1                                               // 0000000020B4: BF850001
	s_load_b512 s[68:83], s[4:5], 0x0                          // 0000000020B8: F4009102 F8000000
	s_load_b512 s[52:67], s[6:7], 0x0                          // 0000000020C0: F4008D03 F8000000
	v_readlane_b32 s4, v28, 31                                 // 0000000020C8: D7600004 00013F1C
	v_writelane_b32 v27, s38, 5                                // 0000000020D0: D761001B 00010A26
	s_lshl_b32 s0, vcc_hi, 4                                   // 0000000020D8: 8400846B
	s_wait_alu 0xfffe                                          // 0000000020DC: BF88FFFE
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000020E0: BF870002
	s_add_co_i32 s0, s0, s4                                    // 0000000020E4: 81000400
	v_writelane_b32 v27, s39, 6                                // 0000000020E8: D761001B 00010C27
	v_writelane_b32 v27, s40, 7                                // 0000000020F0: D761001B 00010E28
	v_writelane_b32 v27, s41, 8                                // 0000000020F8: D761001B 00011029
	v_writelane_b32 v27, s42, 9                                // 000000002100: D761001B 0001122A
	v_writelane_b32 v27, s43, 10                               // 000000002108: D761001B 0001142B
	v_writelane_b32 v27, s44, 11                               // 000000002110: D761001B 0001162C
	v_writelane_b32 v27, s45, 12                               // 000000002118: D761001B 0001182D
	v_writelane_b32 v27, s46, 13                               // 000000002120: D761001B 00011A2E
	v_writelane_b32 v27, s47, 14                               // 000000002128: D761001B 00011C2F
	v_writelane_b32 v27, s48, 15                               // 000000002130: D761001B 00011E30
	v_writelane_b32 v27, s49, 16                               // 000000002138: D761001B 00012031
	v_writelane_b32 v27, s50, 17                               // 000000002140: D761001B 00012232
	v_writelane_b32 v27, s51, 18                               // 000000002148: D761001B 00012433
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002150: BF870001
	v_readlane_b32 s5, v27, 0                                  // 000000002154: D7600005 0001011B
	v_writelane_b32 v28, s26, 9                                // 00000000215C: D761001C 0001121A
	v_writelane_b32 v27, s85, 19                               // 000000002164: D761001B 00012655
	v_cndmask_b32_e64 v0, 0, 1, s2                             // 00000000216C: D5010000 00090280
	s_wait_alu 0xfffe                                          // 000000002174: BF88FFFE
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000002178: 84808200
	s_mov_b32 s12, 0                                           // 00000000217C: BE8C0080
	v_writelane_b32 v28, s25, 10                               // 000000002180: D761001C 00011419
	v_writelane_b32 v27, s84, 20                               // 000000002188: D761001B 00012854
	v_cmp_ne_u32_e64 s33, 1, v0                                // 000000002190: D44D0021 00020081
	s_mov_b32 s24, 0                                           // 000000002198: BE980080
	s_mov_b32 s25, 0                                           // 00000000219C: BE990080
	v_writelane_b32 v28, s18, 11                               // 0000000021A0: D761001C 00011612
	v_writelane_b32 v27, s104, 21                              // 0000000021A8: D761001B 00012A68
	s_mov_b32 s104, s101                                       // 0000000021B0: BEE80065
	s_mov_b32 s26, 0                                           // 0000000021B4: BE9A0080
	s_mov_b32 s27, 0                                           // 0000000021B8: BE9B0080
	v_readlane_b32 s4, v28, 2                                  // 0000000021BC: D7600004 0001051C
	v_readlane_b32 s6, v28, 4                                  // 0000000021C4: D7600006 0001091C
	v_readlane_b32 s7, v28, 5                                  // 0000000021CC: D7600007 00010B1C
	v_writelane_b32 v27, s100, 22                              // 0000000021D4: D761001B 00012C64
	s_mov_b32 s16, 0                                           // 0000000021DC: BE900080
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 0000000021E0: 916A027E
	s_mov_b32 s17, 0                                           // 0000000021E4: BE910080
	s_wait_alu 0xfffe                                          // 0000000021E8: BF88FFFE
	s_add_nc_u64 s[100:101], s[6:7], s[0:1]                    // 0000000021EC: A9E40006
	s_mov_b32 s18, 0                                           // 0000000021F0: BE920080
	s_mov_b32 s19, 0                                           // 0000000021F4: BE930080
	s_mov_b32 s20, 0                                           // 0000000021F8: BE940080
	s_mov_b32 s21, 0                                           // 0000000021FC: BE950080
	s_mov_b32 s22, 0                                           // 000000002200: BE960080
	s_mov_b32 s23, 0                                           // 000000002204: BE970080
	s_mov_b32 s28, 0                                           // 000000002208: BE9C0080
	s_mov_b32 s29, 0                                           // 00000000220C: BE9D0080
	s_mov_b32 s30, 0                                           // 000000002210: BE9E0080
	s_mov_b32 s31, 0                                           // 000000002214: BE9F0080
	v_readlane_b32 s5, v28, 3                                  // 000000002218: D7600005 0001071C
	s_cbranch_vccnz 2                                          // 000000002220: BFA40002 <r_3_3_3_8_8_8+0xc2c>
	s_load_b512 s[16:31], s[100:101], 0x1240                   // 000000002224: F4008432 F8001240
	s_wait_kmcnt 0x0                                           // 00000000222C: BFC70000
	v_writelane_b32 v27, s16, 23                               // 000000002230: D761001B 00012E10
	s_mov_b32 s13, 0                                           // 000000002238: BE8D0080
	v_writelane_b32 v26, s25, 0                                // 00000000223C: D761001A 00010019
	s_mov_b32 s14, 0                                           // 000000002244: BE8E0080
	s_mov_b32 s15, 0                                           // 000000002248: BE8F0080
	v_writelane_b32 v27, s17, 24                               // 00000000224C: D761001B 00013011
	s_mov_b32 s4, 0                                            // 000000002254: BE840080
	v_writelane_b32 v26, s26, 1                                // 000000002258: D761001A 0001021A
	s_mov_b32 s5, 0                                            // 000000002260: BE850080
	s_mov_b32 s6, 0                                            // 000000002264: BE860080
	s_wait_alu 0xfffe                                          // 000000002268: BF88FFFE
	v_writelane_b32 v27, s18, 25                               // 00000000226C: D761001B 00013212
	s_mov_b32 s7, 0                                            // 000000002274: BE870080
	v_writelane_b32 v26, s27, 2                                // 000000002278: D761001A 0001041B
	s_mov_b32 s1, 0                                            // 000000002280: BE810080
	s_mov_b32 s2, 0                                            // 000000002284: BE820080
	v_writelane_b32 v27, s19, 26                               // 000000002288: D761001B 00013413
	s_mov_b32 s3, 0                                            // 000000002290: BE830080
	v_writelane_b32 v26, s28, 3                                // 000000002294: D761001A 0001061C
	s_mov_b32 s8, 0                                            // 00000000229C: BE880080
	s_mov_b32 s9, 0                                            // 0000000022A0: BE890080
	v_writelane_b32 v27, s20, 27                               // 0000000022A4: D761001B 00013614
	s_mov_b32 s10, 0                                           // 0000000022AC: BE8A0080
	v_writelane_b32 v26, s29, 4                                // 0000000022B0: D761001A 0001081D
	s_mov_b32 s11, 0                                           // 0000000022B8: BE8B0080
	s_mov_b32 s92, 0                                           // 0000000022BC: BEDC0080
	v_writelane_b32 v27, s21, 28                               // 0000000022C0: D761001B 00013815
	s_mov_b32 s93, 0                                           // 0000000022C8: BEDD0080
	v_writelane_b32 v26, s30, 5                                // 0000000022CC: D761001A 00010A1E
	s_mov_b32 s94, 0                                           // 0000000022D4: BEDE0080
	s_mov_b32 s95, 0                                           // 0000000022D8: BEDF0080
	v_writelane_b32 v27, s22, 29                               // 0000000022DC: D761001B 00013A16
	s_mov_b32 s84, 0                                           // 0000000022E4: BED40080
	v_writelane_b32 v26, s31, 6                                // 0000000022E8: D761001A 00010C1F
	s_mov_b32 s85, 0                                           // 0000000022F0: BED50080
	s_mov_b32 s86, 0                                           // 0000000022F4: BED60080
	v_writelane_b32 v27, s23, 30                               // 0000000022F8: D761001B 00013C17
	s_mov_b32 s87, 0                                           // 000000002300: BED70080
	s_mov_b32 s88, 0                                           // 000000002304: BED80080
	s_mov_b32 s89, 0                                           // 000000002308: BED90080
	s_mov_b32 s90, 0                                           // 00000000230C: BEDA0080
	v_writelane_b32 v27, s24, 31                               // 000000002310: D761001B 00013E18
	s_load_b512 s[16:31], s[100:101], 0xfc0                    // 000000002318: F4008432 F8000FC0
	s_mov_b32 s91, 0                                           // 000000002320: BEDB0080
	s_mov_b32 s96, 0                                           // 000000002324: BEE00080
	s_mov_b32 s97, 0                                           // 000000002328: BEE10080
	v_readlane_b32 s0, v27, 1                                  // 00000000232C: D7600000 0001031B
	s_mov_b32 s98, 0                                           // 000000002334: BEE20080
	s_mov_b32 s99, 0                                           // 000000002338: BEE30080
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000233C: BF870001
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 000000002340: 916A007E
	s_mov_b32 s0, 0                                            // 000000002344: BE800080
	s_wait_kmcnt 0x0                                           // 000000002348: BFC70000
	v_writelane_b32 v26, s16, 7                                // 00000000234C: D761001A 00010E10
	v_writelane_b32 v26, s17, 8                                // 000000002354: D761001A 00011011
	v_writelane_b32 v26, s18, 9                                // 00000000235C: D761001A 00011212
	v_writelane_b32 v26, s19, 10                               // 000000002364: D761001A 00011413
	v_writelane_b32 v26, s20, 11                               // 00000000236C: D761001A 00011614
	v_writelane_b32 v26, s21, 12                               // 000000002374: D761001A 00011815
	v_writelane_b32 v26, s22, 13                               // 00000000237C: D761001A 00011A16
	v_writelane_b32 v26, s23, 14                               // 000000002384: D761001A 00011C17
	v_writelane_b32 v26, s24, 15                               // 00000000238C: D761001A 00011E18
	v_writelane_b32 v26, s25, 16                               // 000000002394: D761001A 00012019
	v_writelane_b32 v26, s26, 17                               // 00000000239C: D761001A 0001221A
	v_writelane_b32 v26, s27, 18                               // 0000000023A4: D761001A 0001241B
	v_writelane_b32 v26, s28, 19                               // 0000000023AC: D761001A 0001261C
	v_writelane_b32 v26, s29, 20                               // 0000000023B4: D761001A 0001281D
	v_writelane_b32 v26, s30, 21                               // 0000000023BC: D761001A 00012A1E
	v_writelane_b32 v26, s31, 22                               // 0000000023C4: D761001A 00012C1F
	s_load_b512 s[16:31], s[100:101], 0x1100                   // 0000000023CC: F4008432 F8001100
	s_wait_kmcnt 0x0                                           // 0000000023D4: BFC70000
	v_writelane_b32 v26, s16, 23                               // 0000000023D8: D761001A 00012E10
	v_writelane_b32 v25, s25, 0                                // 0000000023E0: D7610019 00010019
	v_writelane_b32 v26, s17, 24                               // 0000000023E8: D761001A 00013011
	v_writelane_b32 v25, s26, 1                                // 0000000023F0: D7610019 0001021A
	v_writelane_b32 v26, s18, 25                               // 0000000023F8: D761001A 00013212
	v_writelane_b32 v25, s27, 2                                // 000000002400: D7610019 0001041B
	v_writelane_b32 v26, s19, 26                               // 000000002408: D761001A 00013413
	v_writelane_b32 v25, s28, 3                                // 000000002410: D7610019 0001061C
	v_writelane_b32 v26, s20, 27                               // 000000002418: D761001A 00013614
	v_writelane_b32 v25, s29, 4                                // 000000002420: D7610019 0001081D
	v_writelane_b32 v26, s21, 28                               // 000000002428: D761001A 00013815
	v_writelane_b32 v25, s30, 5                                // 000000002430: D7610019 00010A1E
	v_writelane_b32 v26, s22, 29                               // 000000002438: D761001A 00013A16
	v_writelane_b32 v25, s31, 6                                // 000000002440: D7610019 00010C1F
	s_mov_b32 s25, 0                                           // 000000002448: BE990080
	s_mov_b32 s26, 0                                           // 00000000244C: BE9A0080
	s_mov_b32 s27, 0                                           // 000000002450: BE9B0080
	v_writelane_b32 v26, s23, 30                               // 000000002454: D761001A 00013C17
	s_mov_b32 s16, 0                                           // 00000000245C: BE900080
	s_mov_b32 s17, 0                                           // 000000002460: BE910080
	s_mov_b32 s18, 0                                           // 000000002464: BE920080
	s_mov_b32 s19, 0                                           // 000000002468: BE930080
	v_writelane_b32 v26, s24, 31                               // 00000000246C: D761001A 00013E18
	s_mov_b32 s24, 0                                           // 000000002474: BE980080
	s_mov_b32 s20, 0                                           // 000000002478: BE940080
	s_mov_b32 s21, 0                                           // 00000000247C: BE950080
	s_mov_b32 s22, 0                                           // 000000002480: BE960080
	s_mov_b32 s23, 0                                           // 000000002484: BE970080
	s_mov_b32 s28, 0                                           // 000000002488: BE9C0080
	s_mov_b32 s29, 0                                           // 00000000248C: BE9D0080
	s_mov_b32 s30, 0                                           // 000000002490: BE9E0080
	s_mov_b32 s31, 0                                           // 000000002494: BE9F0080
	s_cbranch_vccnz 9                                          // 000000002498: BFA40009 <r_3_3_3_8_8_8+0xec0>
	s_and_b32 vcc_lo, exec_lo, s33                             // 00000000249C: 8B6A217E
	s_cbranch_vccnz 2                                          // 0000000024A0: BFA40002 <r_3_3_3_8_8_8+0xeac>
	s_load_b512 s[84:99], s[100:101], 0x2640                   // 0000000024A4: F4009532 F8002640
	s_clause 0x1                                               // 0000000024AC: BF850001
	s_load_b512 s[16:31], s[100:101], 0x23c0                   // 0000000024B0: F4008432 F80023C0
	s_load_b512 s[0:15], s[100:101], 0x2500                    // 0000000024B8: F4008032 F8002500
	v_readlane_b32 s33, v28, 15                                // 0000000024C0: D7600021 00011F1C
	v_readlane_b32 s100, v28, 14                               // 0000000024C8: D7600064 00011D1C
	v_readlane_b32 s36, v26, 7                                 // 0000000024D0: D7600024 00010F1A
	v_readlane_b32 s40, v26, 11                                // 0000000024D8: D7600028 0001171A
	v_readlane_b32 s37, v26, 8                                 // 0000000024E0: D7600025 0001111A
	s_add_f32 s33, s33, s72                                    // 0000000024E8: A0214821
	s_add_f32 s68, s100, s68                                   // 0000000024EC: A0444464
	v_readlane_b32 s41, v26, 12                                // 0000000024F0: D7600029 0001191A
	v_readlane_b32 s38, v26, 9                                 // 0000000024F8: D7600026 0001131A
	s_wait_alu 0xfffe                                          // 000000002500: BF88FFFE
	s_add_f32 s33, s73, s33                                    // 000000002504: A0212149
	s_add_f32 s68, s69, s68                                    // 000000002508: A0444445
	v_readlane_b32 s42, v26, 13                                // 00000000250C: D760002A 00011B1A
	v_readlane_b32 s39, v26, 10                                // 000000002514: D7600027 0001151A
	s_wait_alu 0xfffe                                          // 00000000251C: BF88FFFE
	s_add_f32 s33, s74, s33                                    // 000000002520: A021214A
	s_add_f32 s68, s70, s68                                    // 000000002524: A0444446
	v_readlane_b32 s43, v26, 14                                // 000000002528: D760002B 00011D1A
	v_readlane_b32 s44, v26, 15                                // 000000002530: D760002C 00011F1A
	s_wait_alu 0xfffe                                          // 000000002538: BF88FFFE
	s_add_f32 s33, s75, s33                                    // 00000000253C: A021214B
	s_add_f32 s68, s71, s68                                    // 000000002540: A0444447
	v_readlane_b32 s45, v26, 16                                // 000000002544: D760002D 0001211A
	v_readlane_b32 s46, v26, 17                                // 00000000254C: D760002E 0001231A
	s_wait_alu 0xfffe                                          // 000000002554: BF88FFFE
	s_add_f32 s33, s76, s33                                    // 000000002558: A021214C
	s_add_f32 s68, s72, s68                                    // 00000000255C: A0444448
	v_readlane_b32 s47, v26, 18                                // 000000002560: D760002F 0001251A
	v_readlane_b32 s100, v28, 24                               // 000000002568: D7600064 0001311C
	s_wait_alu 0xfffe                                          // 000000002570: BF88FFFE
	s_add_f32 s33, s77, s33                                    // 000000002574: A021214D
	s_add_f32 s68, s73, s68                                    // 000000002578: A0444449
	v_readlane_b32 s48, v26, 19                                // 00000000257C: D7600030 0001271A
	v_readlane_b32 s49, v26, 20                                // 000000002584: D7600031 0001291A
	s_wait_alu 0xfffe                                          // 00000000258C: BF88FFFE
	s_add_f32 s33, s78, s33                                    // 000000002590: A021214E
	s_add_f32 s75, s74, s68                                    // 000000002594: A04B444A
	v_readlane_b32 s68, v28, 26                                // 000000002598: D7600044 0001351C
	s_add_f32 s100, s100, s77                                  // 0000000025A0: A0644D64
	s_wait_alu 0xfffe                                          // 0000000025A4: BF88FFFE
	s_add_f32 s74, s79, s33                                    // 0000000025A8: A04A214F
	v_readlane_b32 s33, v28, 28                                // 0000000025AC: D7600021 0001391C
	v_readlane_b32 s50, v26, 21                                // 0000000025B4: D7600032 00012B1A
	s_add_f32 s68, s68, s36                                    // 0000000025BC: A0442444
	s_add_f32 s69, s78, s100                                   // 0000000025C0: A045644E
	v_readlane_b32 s51, v26, 22                                // 0000000025C4: D7600033 00012D1A
	s_add_f32 s33, s33, s40                                    // 0000000025CC: A0212821
	s_wait_alu 0xfffe                                          // 0000000025D0: BF88FFFE
	s_add_f32 s68, s37, s68                                    // 0000000025D4: A0444425
	s_add_f32 s69, s79, s69                                    // 0000000025D8: A045454F
	s_add_nc_u64 s[102:103], s[102:103], 16                    // 0000000025DC: A9E69066
	s_add_f32 s33, s41, s33                                    // 0000000025E0: A0212129
	s_wait_alu 0xfffe                                          // 0000000025E4: BF88FFFE
	s_add_f32 s68, s38, s68                                    // 0000000025E8: A0444426
	s_add_f32 s69, s69, s80                                    // 0000000025EC: A0455045
	s_add_co_i32 vcc_hi, vcc_hi, 1                             // 0000000025F0: 816B816B
	s_add_f32 s33, s42, s33                                    // 0000000025F4: A021212A
	s_wait_alu 0xfffe                                          // 0000000025F8: BF88FFFE
	s_add_f32 s68, s39, s68                                    // 0000000025FC: A0444427
	s_add_f32 s69, s81, s69                                    // 000000002600: A0454551
	s_cmp_eq_u32 s102, 0x70                                    // 000000002604: BF06FF66 00000070
	s_add_f32 s33, s43, s33                                    // 00000000260C: A021212B
	s_wait_alu 0xfffe                                          // 000000002610: BF88FFFE
	s_add_f32 s68, s40, s68                                    // 000000002614: A0444428
	s_add_f32 s69, s82, s69                                    // 000000002618: A0454552
	s_add_nc_u64 s[34:35], s[34:35], 64                        // 00000000261C: A9A2C022
	s_add_f32 s33, s44, s33                                    // 000000002620: A021212C
	s_wait_alu 0xfffe                                          // 000000002624: BF88FFFE
	s_add_f32 s68, s41, s68                                    // 000000002628: A0444429
	s_add_f32 s72, s83, s69                                    // 00000000262C: A0484553
	v_readlane_b32 s69, v28, 29                                // 000000002630: D7600045 00013B1C
	s_add_f32 s33, s45, s33                                    // 000000002638: A021212D
	s_wait_alu 0xfffe                                          // 00000000263C: BF88FFFE
	s_add_f32 s71, s42, s68                                    // 000000002640: A047442A
	v_readlane_b32 s68, v27, 21                                // 000000002644: D7600044 00012B1B
	s_add_f32 s33, s46, s33                                    // 00000000264C: A021212E
	s_add_f32 s69, s69, s45                                    // 000000002650: A0452D45
	s_wait_kmcnt 0x0                                           // 000000002654: BFC70000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002658: BF870001
	s_add_f32 s16, s68, s16                                    // 00000000265C: A0101044
	v_readlane_b32 s68, v27, 22                                // 000000002660: D7600044 00012D1B
	s_wait_alu 0xfffe                                          // 000000002668: BF88FFFE
	s_add_f32 s70, s47, s33                                    // 00000000266C: A046212F
	s_add_f32 s33, s104, s20                                   // 000000002670: A0211468
	s_add_f32 s16, s17, s16                                    // 000000002674: A0101011
	s_add_f32 s69, s46, s69                                    // 000000002678: A045452E
	s_add_f32 s68, s68, s25                                    // 00000000267C: A0441944
	s_wait_alu 0xfffe                                          // 000000002680: BF88FFFE
	s_add_f32 s33, s21, s33                                    // 000000002684: A0212115
	s_add_f32 s16, s18, s16                                    // 000000002688: A0101012
	s_add_f32 s69, s47, s69                                    // 00000000268C: A045452F
	s_wait_alu 0xfffe                                          // 000000002690: BF88FFFE
	s_add_f32 s17, s22, s33                                    // 000000002694: A0112116
	s_add_f32 s33, s26, s68                                    // 000000002698: A021441A
	s_add_f32 s16, s19, s16                                    // 00000000269C: A0101013
	s_add_f32 s69, s69, s48                                    // 0000000026A0: A0453045
	s_wait_alu 0xfffe                                          // 0000000026A4: BF88FFFE
	s_add_f32 s17, s23, s17                                    // 0000000026A8: A0111117
	s_add_f32 s18, s27, s33                                    // 0000000026AC: A012211B
	s_add_f32 s16, s20, s16                                    // 0000000026B0: A0101014
	s_add_f32 s69, s49, s69                                    // 0000000026B4: A0454531
	s_wait_alu 0xfffe                                          // 0000000026B8: BF88FFFE
	s_add_f32 s17, s24, s17                                    // 0000000026BC: A0111118
	s_add_f32 s18, s18, s28                                    // 0000000026C0: A0121C12
	s_add_f32 s16, s21, s16                                    // 0000000026C4: A0101015
	s_add_f32 s69, s50, s69                                    // 0000000026C8: A0454532
	s_wait_alu 0xfffe                                          // 0000000026CC: BF88FFFE
	s_add_f32 s17, s25, s17                                    // 0000000026D0: A0111119
	s_add_f32 s18, s29, s18                                    // 0000000026D4: A012121D
	s_add_f32 s104, s22, s16                                   // 0000000026D8: A0681016
	v_readlane_b32 s16, v28, 19                                // 0000000026DC: D7600010 0001271C
	s_wait_alu 0xfffe                                          // 0000000026E4: BF88FFFE
	s_add_f32 s17, s26, s17                                    // 0000000026E8: A011111A
	s_add_f32 s18, s30, s18                                    // 0000000026EC: A012121E
	s_add_f32 s69, s51, s69                                    // 0000000026F0: A0454533
	v_readlane_b32 s36, v27, 23                                // 0000000026F4: D7600024 00012F1B
	s_wait_alu 0xfffe                                          // 0000000026FC: BF88FFFE
	s_add_f32 s101, s27, s17                                   // 000000002700: A065111B
	s_add_f32 s100, s31, s18                                   // 000000002704: A064121F
	v_readlane_b32 s17, v28, 16                                // 000000002708: D7600011 0001211C
	v_readlane_b32 s18, v28, 20                                // 000000002710: D7600012 0001291C
	s_add_f32 s16, s16, s56                                    // 000000002718: A0103810
	v_readlane_b32 s37, v27, 24                                // 00000000271C: D7600025 0001311B
	v_readlane_b32 s38, v27, 25                                // 000000002724: D7600026 0001331B
	s_add_f32 s17, s17, s52                                    // 00000000272C: A0113411
	s_wait_alu 0xfffe                                          // 000000002730: BF88FFFE
	s_add_f32 s16, s57, s16                                    // 000000002734: A0101039
	s_add_f32 s18, s18, s61                                    // 000000002738: A0123D12
	v_readlane_b32 s40, v27, 27                                // 00000000273C: D7600028 0001371B
	s_add_f32 s17, s53, s17                                    // 000000002744: A0111135
	s_wait_alu 0xfffe                                          // 000000002748: BF88FFFE
	s_add_f32 s16, s58, s16                                    // 00000000274C: A010103A
	s_add_f32 s18, s62, s18                                    // 000000002750: A012123E
	v_readlane_b32 s39, v27, 26                                // 000000002754: D7600027 0001351B
	s_add_f32 s17, s54, s17                                    // 00000000275C: A0111136
	s_wait_alu 0xfffe                                          // 000000002760: BF88FFFE
	s_add_f32 s16, s59, s16                                    // 000000002764: A010103B
	s_add_f32 s18, s63, s18                                    // 000000002768: A012123F
	v_readlane_b32 s41, v27, 28                                // 00000000276C: D7600029 0001391B
	s_add_f32 s17, s55, s17                                    // 000000002774: A0111137
	s_wait_alu 0xfffe                                          // 000000002778: BF88FFFE
	s_add_f32 s16, s60, s16                                    // 00000000277C: A010103C
	s_add_f32 s18, s18, s64                                    // 000000002780: A0124012
	v_readlane_b32 s42, v27, 29                                // 000000002784: D760002A 00013B1B
	s_add_f32 s17, s56, s17                                    // 00000000278C: A0111138
	s_wait_alu 0xfffe                                          // 000000002790: BF88FFFE
	s_add_f32 s16, s61, s16                                    // 000000002794: A010103D
	s_add_f32 s18, s65, s18                                    // 000000002798: A0121241
	v_readlane_b32 s43, v27, 30                                // 00000000279C: D760002B 00013D1B
	s_add_f32 s17, s57, s17                                    // 0000000027A4: A0111139
	s_wait_alu 0xfffe                                          // 0000000027A8: BF88FFFE
	s_add_f32 s16, s62, s16                                    // 0000000027AC: A010103E
	s_add_f32 s18, s66, s18                                    // 0000000027B0: A0121242
	v_readlane_b32 s44, v27, 31                                // 0000000027B4: D760002C 00013F1B
	s_add_f32 s24, s58, s17                                    // 0000000027BC: A018113A
	s_wait_alu 0xfffe                                          // 0000000027C0: BF88FFFE
	s_add_f32 s23, s63, s16                                    // 0000000027C4: A017103F
	s_add_f32 s22, s67, s18                                    // 0000000027C8: A0161243
	v_readlane_b32 s52, v26, 23                                // 0000000027CC: D7600034 00012F1A
	v_readlane_b32 s16, v28, 21                                // 0000000027D4: D7600010 00012B1C
	v_readlane_b32 s56, v26, 27                                // 0000000027DC: D7600038 0001371A
	v_readlane_b32 s57, v26, 28                                // 0000000027E4: D7600039 0001391A
	v_readlane_b32 s17, v28, 22                                // 0000000027EC: D7600011 00012D1C
	v_readlane_b32 s53, v26, 24                                // 0000000027F4: D7600035 0001311A
	v_readlane_b32 s58, v26, 29                                // 0000000027FC: D760003A 00013B1A
	s_add_f32 s16, s16, s56                                    // 000000002804: A0103810
	v_readlane_b32 s54, v26, 25                                // 000000002808: D7600036 0001331A
	s_add_f32 s17, s17, s52                                    // 000000002810: A0113411
	v_readlane_b32 s59, v26, 30                                // 000000002814: D760003B 00013D1A
	s_wait_alu 0xfffe                                          // 00000000281C: BF88FFFE
	s_add_f32 s16, s57, s16                                    // 000000002820: A0101039
	v_readlane_b32 s55, v26, 26                                // 000000002824: D7600037 0001351A
	s_add_f32 s17, s53, s17                                    // 00000000282C: A0111135
	v_readlane_b32 s60, v26, 31                                // 000000002830: D760003C 00013F1A
	s_wait_alu 0xfffe                                          // 000000002838: BF88FFFE
	s_add_f32 s16, s58, s16                                    // 00000000283C: A010103A
	v_readlane_b32 s61, v25, 0                                 // 000000002840: D760003D 00010119
	s_add_f32 s17, s54, s17                                    // 000000002848: A0111136
	v_readlane_b32 s62, v25, 1                                 // 00000000284C: D760003E 00010319
	s_wait_alu 0xfffe                                          // 000000002854: BF88FFFE
	s_add_f32 s16, s59, s16                                    // 000000002858: A010103B
	v_readlane_b32 s63, v25, 2                                 // 00000000285C: D760003F 00010519
	s_add_f32 s17, s55, s17                                    // 000000002864: A0111137
	v_readlane_b32 s18, v28, 11                                // 000000002868: D7600012 0001171C
	s_wait_alu 0xfffe                                          // 000000002870: BF88FFFE
	s_add_f32 s16, s60, s16                                    // 000000002874: A010103C
	v_readlane_b32 s64, v25, 3                                 // 000000002878: D7600040 00010719
	s_add_f32 s17, s56, s17                                    // 000000002880: A0111138
	v_readlane_b32 s65, v25, 4                                 // 000000002884: D7600041 00010919
	s_wait_alu 0xfffe                                          // 00000000288C: BF88FFFE
	s_add_f32 s16, s61, s16                                    // 000000002890: A010103D
	s_add_f32 s18, s18, s61                                    // 000000002894: A0123D12
	s_add_f32 s17, s57, s17                                    // 000000002898: A0111139
	v_readlane_b32 s66, v25, 5                                 // 00000000289C: D7600042 00010B19
	s_wait_alu 0xfffe                                          // 0000000028A4: BF88FFFE
	s_add_f32 s16, s62, s16                                    // 0000000028A8: A010103E
	s_add_f32 s18, s62, s18                                    // 0000000028AC: A012123E
	s_add_f32 s20, s58, s17                                    // 0000000028B0: A014113A
	v_readlane_b32 s17, v28, 17                                // 0000000028B4: D7600011 0001231C
	s_wait_alu 0xfffe                                          // 0000000028BC: BF88FFFE
	s_add_f32 s21, s63, s16                                    // 0000000028C0: A015103F
	v_readlane_b32 s16, v28, 18                                // 0000000028C4: D7600010 0001251C
	s_add_f32 s18, s63, s18                                    // 0000000028CC: A012123F
	v_readlane_b32 s67, v25, 6                                 // 0000000028D0: D7600043 00010D19
	s_add_f32 s0, s17, s0                                      // 0000000028D8: A0000011
	v_readlane_b32 s17, v28, 27                                // 0000000028DC: D7600011 0001371C
	s_add_f32 s16, s16, s4                                     // 0000000028E4: A0100410
	s_wait_alu 0xfffe                                          // 0000000028E8: BF88FFFE
	s_add_f32 s18, s64, s18                                    // 0000000028EC: A0121240
	s_add_f32 s0, s1, s0                                       // 0000000028F0: A0000001
	v_readlane_b32 s45, v26, 0                                 // 0000000028F4: D760002D 0001011A
	s_add_f32 s16, s5, s16                                     // 0000000028FC: A0101005
	s_add_f32 s17, s17, s9                                     // 000000002900: A0110911
	s_wait_alu 0xfffe                                          // 000000002904: BF88FFFE
	s_add_f32 s0, s2, s0                                       // 000000002908: A0000002
	s_add_f32 s18, s65, s18                                    // 00000000290C: A0121241
	s_add_f32 s1, s6, s16                                      // 000000002910: A0011006
	s_add_f32 s16, s10, s17                                    // 000000002914: A010110A
	s_wait_alu 0xfffe                                          // 000000002918: BF88FFFE
	s_add_f32 s0, s3, s0                                       // 00000000291C: A0000003
	s_add_f32 s18, s66, s18                                    // 000000002920: A0121242
	s_add_f32 s1, s7, s1                                       // 000000002924: A0010107
	s_add_f32 s2, s11, s16                                     // 000000002928: A002100B
	s_wait_alu 0xfffe                                          // 00000000292C: BF88FFFE
	s_add_f32 s0, s4, s0                                       // 000000002930: A0000004
	s_add_f32 s18, s67, s18                                    // 000000002934: A0121243
	s_add_f32 s1, s8, s1                                       // 000000002938: A0010108
	s_add_f32 s2, s2, s12                                      // 00000000293C: A0020C02
	s_wait_alu 0xfffe                                          // 000000002940: BF88FFFE
	s_add_f32 s0, s5, s0                                       // 000000002944: A0000005
	v_readlane_b32 s52, v27, 3                                 // 000000002948: D7600034 0001071B
	s_add_f32 s1, s9, s1                                       // 000000002950: A0010109
	s_add_f32 s2, s13, s2                                      // 000000002954: A002020D
	s_wait_alu 0xfffe                                          // 000000002958: BF88FFFE
	s_add_f32 s9, s6, s0                                       // 00000000295C: A0090006
	v_readlane_b32 s0, v28, 9                                  // 000000002960: D7600000 0001131C
	s_add_f32 s1, s10, s1                                      // 000000002968: A001010A
	s_add_f32 s2, s14, s2                                      // 00000000296C: A002020E
	v_readlane_b32 s53, v27, 4                                 // 000000002970: D7600035 0001091B
	v_readlane_b32 s54, v27, 5                                 // 000000002978: D7600036 00010B1B
	s_wait_alu 0xfffe                                          // 000000002980: BF88FFFE
	s_add_f32 s13, s11, s1                                     // 000000002984: A00D010B
	v_readlane_b32 s1, v28, 23                                 // 000000002988: D7600001 00012F1C
	s_add_f32 s0, s0, s52                                      // 000000002990: A0003400
	s_add_f32 s19, s15, s2                                     // 000000002994: A013020F
	v_readlane_b32 s2, v27, 20                                 // 000000002998: D7600002 0001291B
	v_readlane_b32 s55, v27, 6                                 // 0000000029A0: D7600037 00010D1B
	s_add_f32 s1, s1, s36                                      // 0000000029A8: A0012401
	s_wait_alu 0xfffe                                          // 0000000029AC: BF88FFFE
	s_add_f32 s0, s53, s0                                      // 0000000029B0: A0000035
	v_readlane_b32 s56, v27, 7                                 // 0000000029B4: D7600038 00010F1B
	v_readlane_b32 s3, v28, 13                                 // 0000000029BC: D7600003 00011B1C
	v_readlane_b32 s4, v28, 25                                 // 0000000029C4: D7600004 0001331C
	v_readlane_b32 s5, v28, 8                                  // 0000000029CC: D7600005 0001111C
	s_add_f32 s2, s2, s84                                      // 0000000029D4: A0025402
	s_add_f32 s1, s37, s1                                      // 0000000029D8: A0010125
	s_wait_alu 0xfffe                                          // 0000000029DC: BF88FFFE
	s_add_f32 s0, s54, s0                                      // 0000000029E0: A0000036
	v_readlane_b32 s57, v27, 8                                 // 0000000029E4: D7600039 0001111B
	s_add_f32 s2, s85, s2                                      // 0000000029EC: A0020255
	s_add_f32 s1, s38, s1                                      // 0000000029F0: A0010126
	s_wait_alu 0xfffe                                          // 0000000029F4: BF88FFFE
	s_add_f32 s0, s55, s0                                      // 0000000029F8: A0000037
	s_add_f32 s3, s3, s56                                      // 0000000029FC: A0033803
	s_add_f32 s4, s4, s40                                      // 000000002A00: A0042804
	s_add_f32 s5, s5, s88                                      // 000000002A04: A0055805
	v_readlane_b32 s58, v27, 9                                 // 000000002A08: D760003A 0001131B
	s_add_f32 s2, s86, s2                                      // 000000002A10: A0020256
	s_add_f32 s1, s39, s1                                      // 000000002A14: A0010127
	s_wait_alu 0xfffe                                          // 000000002A18: BF88FFFE
	s_add_f32 s0, s56, s0                                      // 000000002A1C: A0000038
	s_add_f32 s3, s57, s3                                      // 000000002A20: A0030339
	s_add_f32 s4, s41, s4                                      // 000000002A24: A0040429
	s_add_f32 s5, s89, s5                                      // 000000002A28: A0050559
	v_readlane_b32 s59, v27, 10                                // 000000002A2C: D760003B 0001151B
	s_add_f32 s2, s87, s2                                      // 000000002A34: A0020257
	s_add_f32 s1, s1, s40                                      // 000000002A38: A0012801
	s_wait_alu 0xfffe                                          // 000000002A3C: BF88FFFE
	s_add_f32 s0, s57, s0                                      // 000000002A40: A0000039
	s_add_f32 s3, s58, s3                                      // 000000002A44: A003033A
	s_add_f32 s4, s42, s4                                      // 000000002A48: A004042A
	s_add_f32 s5, s90, s5                                      // 000000002A4C: A005055A
	s_add_f32 s2, s88, s2                                      // 000000002A50: A0020258
	s_add_f32 s1, s41, s1                                      // 000000002A54: A0010129
	s_wait_alu 0xfffe                                          // 000000002A58: BF88FFFE
	s_add_f32 s26, s58, s0                                     // 000000002A5C: A01A003A
	s_add_f32 s0, s59, s3                                      // 000000002A60: A000033B
	s_add_f32 s3, s43, s4                                      // 000000002A64: A003042B
	s_add_f32 s4, s91, s5                                      // 000000002A68: A004055B
	s_add_f32 s2, s89, s2                                      // 000000002A6C: A0020259
	s_add_f32 s73, s42, s1                                     // 000000002A70: A049012A
	s_wait_alu 0xfffe                                          // 000000002A74: BF88FFFE
	s_add_f32 s1, s44, s3                                      // 000000002A78: A001032C
	s_add_f32 s3, s92, s4                                      // 000000002A7C: A003045C
	v_readlane_b32 s61, v27, 12                                // 000000002A80: D760003D 0001191B
	s_add_f32 s84, s90, s2                                     // 000000002A88: A054025A
	v_readlane_b32 s4, v28, 30                                 // 000000002A8C: D7600004 00013D1C
	s_wait_alu 0xfffe                                          // 000000002A94: BF88FFFE
	s_add_f32 s2, s93, s3                                      // 000000002A98: A002035D
	v_readlane_b32 s3, v28, 10                                 // 000000002A9C: D7600003 0001151C
	v_readlane_b32 s5, v27, 19                                 // 000000002AA4: D7600005 0001271B
	v_readlane_b32 s60, v27, 11                                // 000000002AAC: D760003C 0001171B
	v_readlane_b32 s62, v27, 13                                // 000000002AB4: D760003E 00011B1B
	v_readlane_b32 s46, v26, 1                                 // 000000002ABC: D760002E 0001031A
	s_add_f32 s3, s3, s61                                      // 000000002AC4: A0033D03
	s_add_f32 s4, s4, s45                                      // 000000002AC8: A0042D04
	s_add_f32 s5, s5, s93                                      // 000000002ACC: A0055D05
	v_readlane_b32 s63, v27, 14                                // 000000002AD0: D760003F 00011D1B
	v_readlane_b32 s47, v26, 2                                 // 000000002AD8: D760002F 0001051A
	s_add_f32 s0, s60, s0                                      // 000000002AE0: A000003C
	s_wait_alu 0xfffe                                          // 000000002AE4: BF88FFFE
	s_add_f32 s3, s62, s3                                      // 000000002AE8: A003033E
	s_add_f32 s4, s46, s4                                      // 000000002AEC: A004042E
	s_add_f32 s5, s94, s5                                      // 000000002AF0: A005055E
	v_readlane_b32 s64, v27, 15                                // 000000002AF4: D7600040 00011F1B
	v_readlane_b32 s48, v26, 3                                 // 000000002AFC: D7600030 0001071A
	s_add_f32 s0, s61, s0                                      // 000000002B04: A000003D
	s_wait_alu 0xfffe                                          // 000000002B08: BF88FFFE
	s_add_f32 s3, s63, s3                                      // 000000002B0C: A003033F
	s_add_f32 s4, s47, s4                                      // 000000002B10: A004042F
	s_add_f32 s5, s95, s5                                      // 000000002B14: A005055F
	v_readlane_b32 s65, v27, 16                                // 000000002B18: D7600041 0001211B
	v_readlane_b32 s49, v26, 4                                 // 000000002B20: D7600031 0001091A
	s_add_f32 s1, s45, s1                                      // 000000002B28: A001012D
	s_add_f32 s0, s62, s0                                      // 000000002B2C: A000003E
	s_wait_alu 0xfffe                                          // 000000002B30: BF88FFFE
	s_add_f32 s3, s3, s64                                      // 000000002B34: A0034003
	s_add_f32 s4, s4, s48                                      // 000000002B38: A0043004
	s_add_f32 s5, s5, s96                                      // 000000002B3C: A0056005
	v_readlane_b32 s66, v27, 17                                // 000000002B40: D7600042 0001231B
	v_readlane_b32 s50, v26, 5                                 // 000000002B48: D7600032 00010B1A
	s_add_f32 s1, s46, s1                                      // 000000002B50: A001012E
	s_add_f32 s27, s63, s0                                     // 000000002B54: A01B003F
	s_wait_alu 0xfffe                                          // 000000002B58: BF88FFFE
	s_add_f32 s0, s65, s3                                      // 000000002B5C: A0000341
	s_add_f32 s3, s49, s4                                      // 000000002B60: A0030431
	s_add_f32 s4, s97, s5                                      // 000000002B64: A0040561
	v_readlane_b32 s67, v27, 18                                // 000000002B68: D7600043 0001251B
	v_readlane_b32 s51, v26, 6                                 // 000000002B70: D7600033 00010D1A
	s_add_f32 s2, s94, s2                                      // 000000002B78: A002025E
	s_add_f32 s68, s47, s1                                     // 000000002B7C: A044012F
	s_wait_alu 0xfffe                                          // 000000002B80: BF88FFFE
	s_add_f32 s0, s66, s0                                      // 000000002B84: A0000042
	s_add_f32 s1, s50, s3                                      // 000000002B88: A0010332
	s_add_f32 s3, s98, s4                                      // 000000002B8C: A0030462
	s_add_f32 s2, s95, s2                                      // 000000002B90: A002025F
	s_wait_alu 0xfffe                                          // 000000002B94: BF88FFFE
	s_add_f32 s25, s67, s0                                     // 000000002B98: A0190043
	s_add_f32 s8, s51, s1                                      // 000000002B9C: A0080133
	s_add_f32 s85, s99, s3                                     // 000000002BA0: A0550363
	v_writelane_b32 v28, s2, 8                                 // 000000002BA4: D761001C 00011002
	s_cbranch_scc1 64203                                       // 000000002BAC: BFA2FACB <r_3_3_3_8_8_8+0xdc>
	s_wait_alu 0xfffe                                          // 000000002BB0: BF88FFFE
	v_writelane_b32 v28, s8, 30                                // 000000002BB4: D761001C 00013C08
	s_branch 64659                                             // 000000002BBC: BFA0FC93 <r_3_3_3_8_8_8+0x80c>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002BC0: BF870001
	v_readlane_b32 s1, v28, 8                                  // 000000002BC4: D7600001 0001111C
	s_mul_f32 s0, s84, 0x3b3f112b                              // 000000002BCC: A200FF54 3B3F112B
	s_mul_f32 s2, s85, 0x3b3f112b                              // 000000002BD4: A202FF55 3B3F112B
	v_readlane_b32 s4, v28, 2                                  // 000000002BDC: D7600004 0001051C
	v_readlane_b32 s5, v28, 3                                  // 000000002BE4: D7600005 0001071C
	s_mul_f32 s1, s1, 0x3b272f05                               // 000000002BEC: A201FF01 3B272F05
	v_mov_b32_e32 v24, 0                                       // 000000002BF4: 7E300280
	s_mul_f32 s3, s24, 0x3b272f05                              // 000000002BF8: A203FF18 3B272F05
	s_wait_alu 0xfffe                                          // 000000002C00: BF88FFFE
	v_dual_mov_b32 v2, s2 :: v_dual_mov_b32 v1, s1             // 000000002C04: CA100002 02000001
	s_mul_f32 s1, s74, 0x3b272f05                              // 000000002C0C: A201FF4A 3B272F05
	v_dual_mov_b32 v3, s3 :: v_dual_mov_b32 v0, s0             // 000000002C14: CA100003 03000000
	s_mul_f32 s0, s75, 0x3b3f112b                              // 000000002C1C: A200FF4B 3B3F112B
	s_mul_f32 s2, s72, 0x3b3f112b                              // 000000002C24: A202FF48 3B3F112B
	s_mul_f32 s3, s27, 0x3b272f05                              // 000000002C2C: A203FF1B 3B272F05
	v_readlane_b32 s6, v28, 4                                  // 000000002C34: D7600006 0001091C
	global_store_b96 v24, v[0:2], s[4:5] offset:96             // 000000002C3C: EE070004 00000000 00006018
	s_wait_alu 0xfffe                                          // 000000002C48: BF88FFFE
	v_dual_mov_b32 v1, s1 :: v_dual_mov_b32 v2, s2             // 000000002C4C: CA100001 01020002
	v_mov_b32_e32 v7, s3                                       // 000000002C54: 7E0E0203
	s_mul_f32 s1, s22, 0x3b272f05                              // 000000002C58: A201FF16 3B272F05
	v_mov_b32_e32 v0, s0                                       // 000000002C60: 7E000200
	s_mul_f32 s0, s23, 0x3b124925                              // 000000002C64: A200FF17 3B124925
	s_mul_f32 s2, s26, 0x3b3f112b                              // 000000002C6C: A202FF1A 3B3F112B
	s_wait_alu 0xfffe                                          // 000000002C74: BF88FFFE
	v_mov_b32_e32 v5, s1                                       // 000000002C78: 7E0A0201
	s_mul_f32 s1, s71, 0x3b272f05                              // 000000002C7C: A201FF47 3B272F05
	v_mov_b32_e32 v4, s0                                       // 000000002C84: 7E080200
	s_mul_f32 s0, s25, 0x3b3f112b                              // 000000002C88: A200FF19 3B3F112B
	s_mul_f32 s3, s69, 0x3b272f05                              // 000000002C90: A203FF45 3B272F05
	s_wait_alu 0xfffe                                          // 000000002C98: BF88FFFE
	v_dual_mov_b32 v9, s1 :: v_dual_mov_b32 v6, s2             // 000000002C9C: CA100001 09060002
	s_mul_f32 s2, s70, 0x3b124925                              // 000000002CA4: A202FF46 3B124925
	v_mov_b32_e32 v11, s3                                      // 000000002CAC: 7E160203
	s_mul_f32 s1, s21, 0x3b000000                              // 000000002CB0: A201FF15 3B000000
	v_mov_b32_e32 v8, s0                                       // 000000002CB8: 7E100200
	s_mul_f32 s0, s20, 0x3b124925                              // 000000002CBC: A200FF14 3B124925
	s_mul_f32 s3, s73, 0x3b272f05                              // 000000002CC4: A203FF49 3B272F05
	s_wait_alu 0xfffe                                          // 000000002CCC: BF88FFFE
	v_dual_mov_b32 v13, s1 :: v_dual_mov_b32 v10, s2           // 000000002CD0: CA100001 0D0A0002
	s_mul_f32 s2, s18, 0x3b124925                              // 000000002CD8: A202FF12 3B124925
	v_mov_b32_e32 v15, s3                                      // 000000002CE0: 7E1E0203
	s_mul_f32 s1, s8, 0x3b272f05                               // 000000002CE4: A201FF08 3B272F05
	v_mov_b32_e32 v12, s0                                      // 000000002CEC: 7E180200
	s_mul_f32 s0, s68, 0x3b124925                              // 000000002CF0: A200FF44 3B124925
	s_mul_f32 s3, s101, 0x3b272f05                             // 000000002CF8: A203FF65 3B272F05
	s_wait_alu 0xfffe                                          // 000000002D00: BF88FFFE
	v_dual_mov_b32 v17, s1 :: v_dual_mov_b32 v14, s2           // 000000002D04: CA100001 110E0002
	s_mul_f32 s2, s104, 0x3b3f112b                             // 000000002D0C: A202FF68 3B3F112B
	v_mov_b32_e32 v19, s3                                      // 000000002D14: 7E260203
	s_mul_f32 s1, s9, 0x3b272f05                               // 000000002D18: A201FF09 3B272F05
	v_mov_b32_e32 v16, s0                                      // 000000002D20: 7E200200
	s_mul_f32 s0, s100, 0x3b3f112b                             // 000000002D24: A200FF64 3B3F112B
	s_mul_f32 s3, s19, 0x3b272f05                              // 000000002D2C: A203FF13 3B272F05
	s_wait_alu 0xfffe                                          // 000000002D34: BF88FFFE
	v_dual_mov_b32 v21, s1 :: v_dual_mov_b32 v18, s2           // 000000002D38: CA100001 15120002
	s_mul_f32 s2, s13, 0x3b124925                              // 000000002D40: A202FF0D 3B124925
	v_mov_b32_e32 v23, s3                                      // 000000002D48: 7E2E0203
	v_readlane_b32 s7, v28, 5                                  // 000000002D4C: D7600007 00010B1C
	v_mov_b32_e32 v20, s0                                      // 000000002D54: 7E280200
	s_wait_alu 0xfffe                                          // 000000002D58: BF88FFFE
	v_mov_b32_e32 v22, s2                                      // 000000002D5C: 7E2C0202
	s_clause 0x5                                               // 000000002D60: BF850005
	global_store_b128 v24, v[0:3], s[4:5]                      // 000000002D64: EE074004 00000000 00000018
	global_store_b128 v24, v[4:7], s[4:5] offset:16            // 000000002D70: EE074004 02000000 00001018
	global_store_b128 v24, v[8:11], s[4:5] offset:32           // 000000002D7C: EE074004 04000000 00002018
	global_store_b128 v24, v[12:15], s[4:5] offset:48          // 000000002D88: EE074004 06000000 00003018
	global_store_b128 v24, v[16:19], s[4:5] offset:64          // 000000002D94: EE074004 08000000 00004018
	global_store_b128 v24, v[20:23], s[4:5] offset:80          // 000000002DA0: EE074004 0A000000 00005018
	s_nop 0                                                    // 000000002DAC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000002DB0: BFB60003
	s_endpgm                                                   // 000000002DB4: BFB00000

.rodata
.amdhsa_kernel kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel
