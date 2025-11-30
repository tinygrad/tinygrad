.text
.globl kernel
.p2align 8 // TODO: need more?
.type kernel,@function

kernel:
	s_load_b128 s[12:15], s[0:1], 0x0                          // 000000001600: F4004300 F8000000
	s_movk_i32 s10, 0xfd00                                     // 000000001608: B00AFD00
	s_mov_b32 s11, -1                                          // 00000000160C: BE8B00C1
	s_movk_i32 s0, 0xfc00                                      // 000000001610: B000FC00
	s_mov_b32 s1, -1                                           // 000000001614: BE8100C1
	s_movk_i32 s2, 0xfd40                                      // 000000001618: B002FD40
	s_mov_b32 s3, -1                                           // 00000000161C: BE8300C1
	s_movk_i32 s4, 0xfc40                                      // 000000001620: B004FC40
	s_mov_b32 s5, -1                                           // 000000001624: BE8500C1
	s_movk_i32 s6, 0xfd80                                      // 000000001628: B006FD80
	s_mov_b32 s7, -1                                           // 00000000162C: BE8700C1
	s_movk_i32 s8, 0xfc80                                      // 000000001630: B008FC80
	s_mov_b32 s9, -1                                           // 000000001634: BE8900C1
	s_wait_kmcnt 0x0                                           // 000000001638: BFC70000
	s_add_nc_u64 s[10:11], s[14:15], s[10:11]                  // 00000000163C: A98A0A0E
	s_add_nc_u64 s[0:1], s[14:15], s[0:1]                      // 000000001640: A980000E
	v_writelane_b32 v29, s10, 0                                // 000000001644: D761001D 0001000A
	v_writelane_b32 v29, s11, 1                                // 00000000164C: D761001D 0001020B
	v_writelane_b32 v29, s0, 2                                 // 000000001654: D761001D 00010400
	v_writelane_b32 v29, s1, 3                                 // 00000000165C: D761001D 00010601
	s_add_nc_u64 s[0:1], s[14:15], 0x1000                      // 000000001664: A980FF0E 00001000
	s_wait_alu 0xfffe                                          // 00000000166C: BF88FFFE
	v_writelane_b32 v29, s0, 4                                 // 000000001670: D761001D 00010800
	v_writelane_b32 v29, s1, 5                                 // 000000001678: D761001D 00010A01
	s_add_nc_u64 s[0:1], s[14:15], 0x2400                      // 000000001680: A980FF0E 00002400
	s_wait_alu 0xfffe                                          // 000000001688: BF88FFFE
	v_writelane_b32 v29, s0, 6                                 // 00000000168C: D761001D 00010C00
	v_writelane_b32 v29, s1, 7                                 // 000000001694: D761001D 00010E01
	s_add_nc_u64 s[0:1], s[14:15], s[2:3]                      // 00000000169C: A980020E
	s_movk_i32 s2, 0xfcc0                                      // 0000000016A0: B002FCC0
	s_mov_b32 s3, -1                                           // 0000000016A4: BE8300C1
	s_wait_alu 0xfffe                                          // 0000000016A8: BF88FFFE
	v_writelane_b32 v29, s0, 8                                 // 0000000016AC: D761001D 00011000
	v_writelane_b32 v29, s1, 9                                 // 0000000016B4: D761001D 00011201
	s_add_nc_u64 s[0:1], s[14:15], s[4:5]                      // 0000000016BC: A980040E
	s_wait_alu 0xfffe                                          // 0000000016C0: BF88FFFE
	v_writelane_b32 v29, s0, 10                                // 0000000016C4: D761001D 00011400
	v_writelane_b32 v29, s1, 11                                // 0000000016CC: D761001D 00011601
	s_add_nc_u64 s[0:1], s[14:15], 0x1040                      // 0000000016D4: A980FF0E 00001040
	s_wait_alu 0xfffe                                          // 0000000016DC: BF88FFFE
	v_writelane_b32 v29, s0, 12                                // 0000000016E0: D761001D 00011800
	v_writelane_b32 v29, s1, 13                                // 0000000016E8: D761001D 00011A01
	s_add_nc_u64 s[0:1], s[14:15], 0x2440                      // 0000000016F0: A980FF0E 00002440
	s_wait_alu 0xfffe                                          // 0000000016F8: BF88FFFE
	v_writelane_b32 v29, s0, 14                                // 0000000016FC: D761001D 00011C00
	v_writelane_b32 v29, s1, 15                                // 000000001704: D761001D 00011E01
	s_add_nc_u64 s[0:1], s[14:15], s[6:7]                      // 00000000170C: A980060E
	s_wait_alu 0xfffe                                          // 000000001710: BF88FFFE
	v_writelane_b32 v29, s0, 16                                // 000000001714: D761001D 00012000
	v_writelane_b32 v29, s1, 17                                // 00000000171C: D761001D 00012201
	s_add_nc_u64 s[0:1], s[14:15], s[8:9]                      // 000000001724: A980080E
	s_wait_alu 0xfffe                                          // 000000001728: BF88FFFE
	v_writelane_b32 v29, s0, 18                                // 00000000172C: D761001D 00012400
	v_writelane_b32 v29, s1, 19                                // 000000001734: D761001D 00012601
	s_add_nc_u64 s[0:1], s[14:15], 0x1080                      // 00000000173C: A980FF0E 00001080
	s_wait_alu 0xfffe                                          // 000000001744: BF88FFFE
	v_writelane_b32 v29, s0, 20                                // 000000001748: D761001D 00012800
	v_writelane_b32 v29, s1, 21                                // 000000001750: D761001D 00012A01
	s_add_nc_u64 s[0:1], s[14:15], 0x2480                      // 000000001758: A980FF0E 00002480
	s_wait_alu 0xfffe                                          // 000000001760: BF88FFFE
	v_writelane_b32 v29, s0, 22                                // 000000001764: D761001D 00012C00
	v_writelane_b32 v29, s1, 23                                // 00000000176C: D761001D 00012E01
	s_movk_i32 s0, 0xfdc0                                      // 000000001774: B000FDC0
	s_mov_b32 s1, -1                                           // 000000001778: BE8100C1
	s_wait_alu 0xfffe                                          // 00000000177C: BF88FFFE
	s_add_nc_u64 s[0:1], s[14:15], s[0:1]                      // 000000001780: A980000E
	s_wait_alu 0xfffe                                          // 000000001784: BF88FFFE
	v_writelane_b32 v29, s0, 24                                // 000000001788: D761001D 00013000
	v_writelane_b32 v29, s1, 25                                // 000000001790: D761001D 00013201
	s_add_nc_u64 s[0:1], s[14:15], s[2:3]                      // 000000001798: A980020E
	s_add_nc_u64 s[2:3], s[14:15], 0x24c0                      // 00000000179C: A982FF0E 000024C0
	s_wait_alu 0xfffe                                          // 0000000017A4: BF88FFFE
	v_writelane_b32 v29, s0, 26                                // 0000000017A8: D761001D 00013400
	v_writelane_b32 v29, s1, 27                                // 0000000017B0: D761001D 00013601
	s_add_nc_u64 s[0:1], s[14:15], 0x10c0                      // 0000000017B8: A980FF0E 000010C0
	s_wait_alu 0xfffe                                          // 0000000017C0: BF88FFFE
	v_writelane_b32 v29, s0, 28                                // 0000000017C4: D761001D 00013800
	v_writelane_b32 v29, s1, 29                                // 0000000017CC: D761001D 00013A01
	s_movk_i32 s0, 0xfe00                                      // 0000000017D4: B000FE00
	s_mov_b32 s1, -1                                           // 0000000017D8: BE8100C1
	v_writelane_b32 v29, s2, 30                                // 0000000017DC: D761001D 00013C02
	v_writelane_b32 v29, s3, 31                                // 0000000017E4: D761001D 00013E03
	s_or_saveexec_b32 s105, -1                                 // 0000000017EC: BEE922C1
	scratch_store_b32 off, v29, off offset:4                   // 0000000017F0: ED06807C 0E800000 00000400
	s_mov_b32 exec_lo, s105                                    // 0000000017FC: BEFE0069
	s_wait_alu 0xfffe                                          // 000000001800: BF88FFFE
	s_add_nc_u64 s[0:1], s[14:15], s[0:1]                      // 000000001804: A980000E
	s_add_nc_u64 s[2:3], s[14:15], 0x2500                      // 000000001808: A982FF0E 00002500
	s_wait_alu 0xfffe                                          // 000000001810: BF88FFFE
	v_writelane_b32 v23, s0, 0                                 // 000000001814: D7610017 00010000
	s_mov_b32 s9, 0                                            // 00000000181C: BE890080
	s_mov_b32 s19, 0                                           // 000000001820: BE930080
	s_mov_b32 s34, 0                                           // 000000001824: BEA20080
	s_mov_b32 s33, 0                                           // 000000001828: BEA10080
	v_writelane_b32 v23, s1, 1                                 // 00000000182C: D7610017 00010201
	s_add_nc_u64 s[0:1], s[14:15], 0x1100                      // 000000001834: A980FF0E 00001100
	s_mov_b32 s4, 0                                            // 00000000183C: BE840080
	s_mov_b32 s8, 0                                            // 000000001840: BE880080
	s_mov_b32 s17, 0                                           // 000000001844: BE910080
	s_wait_alu 0xfffe                                          // 000000001848: BF88FFFE
	v_writelane_b32 v23, s0, 2                                 // 00000000184C: D7610017 00010400
	s_mov_b32 s21, 0                                           // 000000001854: BE950080
	s_mov_b32 s18, 0                                           // 000000001858: BE920080
	s_mov_b32 s23, 0                                           // 00000000185C: BE970080
	s_mov_b32 s22, 0                                           // 000000001860: BE960080
	v_writelane_b32 v23, s1, 3                                 // 000000001864: D7610017 00010601
	s_movk_i32 s0, 0xfe40                                      // 00000000186C: B000FE40
	s_mov_b32 s1, -1                                           // 000000001870: BE8100C1
	s_mov_b32 s24, 0                                           // 000000001874: BE980080
	s_wait_alu 0xfffe                                          // 000000001878: BF88FFFE
	s_add_nc_u64 s[0:1], s[14:15], s[0:1]                      // 00000000187C: A980000E
	v_writelane_b32 v23, s2, 4                                 // 000000001880: D7610017 00010802
	s_mov_b32 s39, 0                                           // 000000001888: BEA70080
	s_mov_b32 s25, 0                                           // 00000000188C: BE990080
	s_mov_b32 s29, 0                                           // 000000001890: BE9D0080
	s_mov_b32 s27, 0                                           // 000000001894: BE9B0080
	v_writelane_b32 v23, s3, 5                                 // 000000001898: D7610017 00010A03
	s_add_nc_u64 s[2:3], s[14:15], 0x2540                      // 0000000018A0: A982FF0E 00002540
	s_mov_b32 s41, 0                                           // 0000000018A8: BEA90080
	s_mov_b32 s44, 0                                           // 0000000018AC: BEAC0080
	s_mov_b32 s28, 0                                           // 0000000018B0: BE9C0080
	s_wait_alu 0xfffe                                          // 0000000018B4: BF88FFFE
	v_writelane_b32 v23, s0, 6                                 // 0000000018B8: D7610017 00010C00
	s_mov_b32 s45, 0                                           // 0000000018C0: BEAD0080
	s_mov_b32 s30, 0                                           // 0000000018C4: BE9E0080
	s_mov_b32 s42, 0                                           // 0000000018C8: BEAA0080
	s_mov_b32 s31, 0                                           // 0000000018CC: BE9F0080
	v_writelane_b32 v23, s1, 7                                 // 0000000018D0: D7610017 00010E01
	s_add_nc_u64 s[0:1], s[14:15], 0x1140                      // 0000000018D8: A980FF0E 00001140
	s_mov_b32 vcc_hi, 0                                        // 0000000018E0: BEEB0080
	s_mov_b32 s11, 0                                           // 0000000018E4: BE8B0080
	s_mov_b32 s40, 0                                           // 0000000018E8: BEA80080
	s_wait_alu 0xfffe                                          // 0000000018EC: BF88FFFE
	v_writelane_b32 v23, s0, 8                                 // 0000000018F0: D7610017 00011000
	v_writelane_b32 v23, s1, 9                                 // 0000000018F8: D7610017 00011201
	s_movk_i32 s0, 0xfe80                                      // 000000001900: B000FE80
	s_mov_b32 s1, -1                                           // 000000001904: BE8100C1
	s_wait_alu 0xfffe                                          // 000000001908: BF88FFFE
	s_add_nc_u64 s[0:1], s[14:15], s[0:1]                      // 00000000190C: A980000E
	v_writelane_b32 v23, s2, 10                                // 000000001910: D7610017 00011402
	v_writelane_b32 v23, s3, 11                                // 000000001918: D7610017 00011603
	s_mov_b32 s2, 0                                            // 000000001920: BE820080
	s_wait_alu 0xfffe                                          // 000000001924: BF88FFFE
	v_writelane_b32 v23, s0, 12                                // 000000001928: D7610017 00011800
	v_writelane_b32 v23, s1, 13                                // 000000001930: D7610017 00011A01
	s_add_nc_u64 s[0:1], s[14:15], 0x1180                      // 000000001938: A980FF0E 00001180
	s_wait_alu 0xfffe                                          // 000000001940: BF88FFFE
	v_writelane_b32 v23, s0, 14                                // 000000001944: D7610017 00011C00
	v_writelane_b32 v23, s1, 15                                // 00000000194C: D7610017 00011E01
	s_movk_i32 s0, 0xfec0                                      // 000000001954: B000FEC0
	s_mov_b32 s1, -1                                           // 000000001958: BE8100C1
	s_wait_alu 0xfffe                                          // 00000000195C: BF88FFFE
	s_add_nc_u64 s[0:1], s[14:15], s[0:1]                      // 000000001960: A980000E
	v_writelane_b32 v23, s2, 16                                // 000000001964: D7610017 00012002
	v_writelane_b32 v23, s2, 17                                // 00000000196C: D7610017 00012202
	v_writelane_b32 v23, s2, 18                                // 000000001974: D7610017 00012402
	v_writelane_b32 v23, s2, 19                                // 00000000197C: D7610017 00012602
	s_add_nc_u64 s[2:3], s[14:15], 0x2580                      // 000000001984: A982FF0E 00002580
	s_wait_alu 0xfffe                                          // 00000000198C: BF88FFFE
	v_writelane_b32 v23, s2, 20                                // 000000001990: D7610017 00012802
	v_writelane_b32 v23, s3, 21                                // 000000001998: D7610017 00012A03
	v_writelane_b32 v23, s0, 22                                // 0000000019A0: D7610017 00012C00
	v_writelane_b32 v23, s1, 23                                // 0000000019A8: D7610017 00012E01
	v_writelane_b32 v23, s12, 24                               // 0000000019B0: D7610017 0001300C
	v_writelane_b32 v23, s13, 25                               // 0000000019B8: D7610017 0001320D
	v_writelane_b32 v23, s14, 26                               // 0000000019C0: D7610017 0001340E
	v_writelane_b32 v23, s15, 27                               // 0000000019C8: D7610017 0001360F
	v_writelane_b32 v23, s42, 28                               // 0000000019D0: D7610017 0001382A
	v_writelane_b32 v23, s31, 29                               // 0000000019D8: D7610017 00013A1F
	v_writelane_b32 v23, s24, 30                               // 0000000019E0: D7610017 00013C18
	v_writelane_b32 v23, s29, 31                               // 0000000019E8: D7610017 00013E1D
	s_or_saveexec_b32 s105, -1                                 // 0000000019F0: BEE922C1
	scratch_store_b32 off, v23, off offset:8                   // 0000000019F4: ED06807C 0B800000 00000800
	s_wait_alu 0xfffe                                          // 000000001A00: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000001A04: BEFE0069
	s_cmp_lg_u32 vcc_hi, 0                                     // 000000001A08: BF07806B
	v_writelane_b32 v29, s21, 0                                // 000000001A0C: D761001D 00010015
	s_cselect_b32 s0, -1, 0                                    // 000000001A14: 980080C1
	s_cmp_lg_u32 vcc_hi, 7                                     // 000000001A18: BF07876B
	s_cselect_b32 s104, -1, 0                                  // 000000001A1C: 986880C1
	v_writelane_b32 v29, s25, 1                                // 000000001A20: D761001D 00010219
	s_lshl_b32 s10, vcc_hi, 8                                  // 000000001A28: 840A886B
	s_cmp_eq_u32 vcc_hi, 0                                     // 000000001A2C: BF06806B
	v_writelane_b32 v29, s27, 2                                // 000000001A30: D761001D 0001041B
	v_writelane_b32 v29, s30, 3                                // 000000001A38: D761001D 0001061E
	v_writelane_b32 v29, s4, 4                                 // 000000001A40: D761001D 00010804
	v_writelane_b32 v29, s28, 5                                // 000000001A48: D761001D 00010A1C
	v_writelane_b32 v29, s39, 6                                // 000000001A50: D761001D 00010C27
	v_writelane_b32 v29, s45, 7                                // 000000001A58: D761001D 00010E2D
	v_writelane_b32 v29, s23, 8                                // 000000001A60: D761001D 00011017
	v_writelane_b32 v29, s22, 9                                // 000000001A68: D761001D 00011216
	v_writelane_b32 v29, s44, 10                               // 000000001A70: D761001D 0001142C
	v_writelane_b32 v29, s18, 11                               // 000000001A78: D761001D 00011612
	v_writelane_b32 v29, s41, 12                               // 000000001A80: D761001D 00011829
	s_wait_alu 0xfffe                                          // 000000001A88: BF88FFFE
	v_writelane_b32 v29, s10, 13                               // 000000001A8C: D761001D 00011A0A
	v_writelane_b32 v29, s11, 14                               // 000000001A94: D761001D 00011C0B
	s_cbranch_scc1 69                                          // 000000001A9C: BFA20045 <r_3_3_3_8_8_8+0x5b4>
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001AA0: 8482820A
	s_movk_i32 s4, 0xfd00                                      // 000000001AA4: B004FD00
	s_wait_alu 0xfffe                                          // 000000001AA8: BF88FFFE
	s_add_nc_u64 s[2:3], s[14:15], s[2:3]                      // 000000001AAC: A982020E
	s_mov_b32 s5, -1                                           // 000000001AB0: BE8500C1
	s_load_b256 s[20:27], s[2:3], 0x1100                       // 000000001AB4: F4006501 F8001100
	s_add_nc_u64 s[4:5], s[2:3], s[4:5]                        // 000000001ABC: A9840402
	s_movk_i32 s6, 0xfe40                                      // 000000001AC0: B006FE40
	s_load_b512 s[44:59], s[4:5], 0x0                          // 000000001AC4: F4008B02 F8000000
	s_mov_b32 s7, -1                                           // 000000001ACC: BE8700C1
	s_mov_b32 s16, 0                                           // 000000001AD0: BE900080
	s_wait_alu 0xfffe                                          // 000000001AD4: BF88FFFE
	s_add_nc_u64 s[2:3], s[2:3], s[6:7]                        // 000000001AD8: A9820602
	s_mov_b32 s1, s104                                         // 000000001ADC: BE810068
	s_load_b512 s[60:75], s[2:3], 0x0                          // 000000001AE0: F4008F01 F8000000
	s_mov_b32 s2, 0                                            // 000000001AE8: BE820080
	s_wait_kmcnt 0x0                                           // 000000001AEC: BFC70000
	v_writelane_b32 v29, s20, 31                               // 000000001AF0: D761001D 00013E14
	v_writelane_b32 v27, s21, 0                                // 000000001AF8: D761001B 00010015
	v_writelane_b32 v29, s44, 15                               // 000000001B00: D761001D 00011E2C
	v_writelane_b32 v27, s22, 1                                // 000000001B08: D761001B 00010216
	v_writelane_b32 v29, s45, 16                               // 000000001B10: D761001D 0001202D
	v_writelane_b32 v27, s23, 2                                // 000000001B18: D761001B 00010417
	v_writelane_b32 v29, s46, 17                               // 000000001B20: D761001D 0001222E
	v_writelane_b32 v27, s24, 3                                // 000000001B28: D761001B 00010618
	v_writelane_b32 v29, s47, 18                               // 000000001B30: D761001D 0001242F
	v_writelane_b32 v27, s25, 4                                // 000000001B38: D761001B 00010819
	v_writelane_b32 v29, s48, 19                               // 000000001B40: D761001D 00012630
	v_writelane_b32 v27, s26, 5                                // 000000001B48: D761001B 00010A1A
	v_writelane_b32 v29, s49, 20                               // 000000001B50: D761001D 00012831
	v_writelane_b32 v27, s27, 6                                // 000000001B58: D761001B 00010C1B
	v_writelane_b32 v29, s50, 21                               // 000000001B60: D761001D 00012A32
	v_writelane_b32 v29, s51, 22                               // 000000001B68: D761001D 00012C33
	v_writelane_b32 v29, s52, 23                               // 000000001B70: D761001D 00012E34
	v_writelane_b32 v29, s53, 24                               // 000000001B78: D761001D 00013035
	v_writelane_b32 v29, s54, 25                               // 000000001B80: D761001D 00013236
	v_writelane_b32 v29, s55, 26                               // 000000001B88: D761001D 00013437
	v_writelane_b32 v29, s56, 27                               // 000000001B90: D761001D 00013638
	v_writelane_b32 v29, s57, 28                               // 000000001B98: D761001D 00013839
	v_writelane_b32 v29, s58, 29                               // 000000001BA0: D761001D 00013A3A
	v_writelane_b32 v29, s59, 30                               // 000000001BA8: D761001D 00013C3B
	s_branch 51                                                // 000000001BB0: BFA00033 <r_3_3_3_8_8_8+0x680>
	s_mov_b32 s2, -1                                           // 000000001BB4: BE8200C1
	v_writelane_b32 v29, s36, 15                               // 000000001BB8: D761001D 00011E24
	s_mov_b32 s1, s11                                          // 000000001BC0: BE81000B
	v_writelane_b32 v29, s37, 16                               // 000000001BC4: D761001D 00012025
	v_writelane_b32 v29, s38, 17                               // 000000001BCC: D761001D 00012226
	v_writelane_b32 v29, s39, 18                               // 000000001BD4: D761001D 00012427
	v_writelane_b32 v29, s40, 19                               // 000000001BDC: D761001D 00012628
	v_writelane_b32 v29, s41, 20                               // 000000001BE4: D761001D 00012829
	v_writelane_b32 v29, s42, 21                               // 000000001BEC: D761001D 00012A2A
	v_writelane_b32 v29, s43, 22                               // 000000001BF4: D761001D 00012C2B
	v_writelane_b32 v29, s44, 23                               // 000000001BFC: D761001D 00012E2C
	v_writelane_b32 v29, s45, 24                               // 000000001C04: D761001D 0001302D
	v_writelane_b32 v29, s46, 25                               // 000000001C0C: D761001D 0001322E
	v_writelane_b32 v29, s47, 26                               // 000000001C14: D761001D 0001342F
	v_writelane_b32 v29, s48, 27                               // 000000001C1C: D761001D 00013630
	v_writelane_b32 v29, s49, 28                               // 000000001C24: D761001D 00013831
	v_writelane_b32 v29, s50, 29                               // 000000001C2C: D761001D 00013A32
	v_writelane_b32 v29, s51, 30                               // 000000001C34: D761001D 00013C33
	v_writelane_b32 v29, s0, 31                                // 000000001C3C: D761001D 00013E00
	s_wait_alu 0xfffe                                          // 000000001C44: BF88FFFE
	v_writelane_b32 v27, s1, 0                                 // 000000001C48: D761001B 00010001
	v_writelane_b32 v27, s2, 1                                 // 000000001C50: D761001B 00010202
	v_writelane_b32 v27, s3, 2                                 // 000000001C58: D761001B 00010403
	v_writelane_b32 v27, s4, 3                                 // 000000001C60: D761001B 00010604
	v_writelane_b32 v27, s5, 4                                 // 000000001C68: D761001B 00010805
	v_writelane_b32 v27, s6, 5                                 // 000000001C70: D761001B 00010A06
	v_writelane_b32 v27, s7, 6                                 // 000000001C78: D761001B 00010C07
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001C80: BF870001
	v_readlane_b32 s4, v29, 13                                 // 000000001C84: D7600004 00011B1D
	s_wait_alu 0xfffe                                          // 000000001C8C: BF88FFFE
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 000000001C90: 916A027E
	s_or_saveexec_b32 s105, -1                                 // 000000001C94: BEE922C1
	scratch_store_b32 off, v29, off offset:12                  // 000000001C98: ED06807C 0E800000 00000C00
	s_wait_alu 0xfffe                                          // 000000001CA4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000001CA8: BEFE0069
	v_readlane_b32 s5, v29, 14                                 // 000000001CAC: D7600005 00011D1D
	s_cbranch_vccnz 90                                         // 000000001CB4: BFA4005A <r_3_3_3_8_8_8+0x820>
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001CB8: BF870001
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001CBC: 84828204
	s_wait_alu 0xfffe                                          // 000000001CC0: BF88FFFE
	s_add_nc_u64 s[2:3], s[14:15], s[2:3]                      // 000000001CC4: A982020E
	s_load_b256 s[20:27], s[2:3], 0x1100                       // 000000001CC8: F4006501 F8001100
	s_or_saveexec_b32 s105, -1                                 // 000000001CD0: BEE922C1
	scratch_load_b32 v29, off, off offset:12 th:TH_LOAD_LU     // 000000001CD4: ED05007C 0030001D 00000C00
	s_wait_alu 0xfffe                                          // 000000001CE0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000001CE4: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000001CE8: BFC00000
	s_wait_kmcnt 0x0                                           // 000000001CEC: BFC70000
	v_writelane_b32 v29, s20, 31                               // 000000001CF0: D761001D 00013E14
	s_mov_b32 s42, s40                                         // 000000001CF8: BEAA0028
	s_mov_b32 s43, s40                                         // 000000001CFC: BEAB0028
	s_mov_b32 s41, s40                                         // 000000001D00: BEA90028
	s_wait_alu 0xfffe                                          // 000000001D04: BF88FFFE
	s_mov_b64 s[46:47], s[42:43]                               // 000000001D08: BEAE012A
	s_mov_b64 s[58:59], s[42:43]                               // 000000001D0C: BEBA012A
	s_mov_b64 s[50:51], s[42:43]                               // 000000001D10: BEB2012A
	s_mov_b64 s[44:45], s[40:41]                               // 000000001D14: BEAC0128
	s_mov_b64 s[54:55], s[42:43]                               // 000000001D18: BEB6012A
	s_mov_b64 s[56:57], s[40:41]                               // 000000001D1C: BEB80128
	s_mov_b64 s[48:49], s[40:41]                               // 000000001D20: BEB00128
	s_mov_b64 s[52:53], s[40:41]                               // 000000001D24: BEB40128
	s_wait_alu 0xfffe                                          // 000000001D28: BF88FFFE
	v_writelane_b32 v29, s44, 15                               // 000000001D2C: D761001D 00011E2C
	v_writelane_b32 v27, s21, 0                                // 000000001D34: D761001B 00010015
	s_mov_b64 s[70:71], s[42:43]                               // 000000001D3C: BEC6012A
	s_mov_b64 s[62:63], s[42:43]                               // 000000001D40: BEBE012A
	s_mov_b64 s[66:67], s[42:43]                               // 000000001D44: BEC2012A
	v_writelane_b32 v29, s45, 16                               // 000000001D48: D761001D 0001202D
	v_writelane_b32 v27, s22, 1                                // 000000001D50: D761001B 00010216
	s_mov_b32 s1, -1                                           // 000000001D58: BE8100C1
	s_mov_b64 s[68:69], s[40:41]                               // 000000001D5C: BEC40128
	s_mov_b64 s[60:61], s[40:41]                               // 000000001D60: BEBC0128
	v_writelane_b32 v29, s46, 17                               // 000000001D64: D761001D 0001222E
	v_writelane_b32 v27, s23, 2                                // 000000001D6C: D761001B 00010417
	s_mov_b64 s[64:65], s[40:41]                               // 000000001D74: BEC00128
	v_writelane_b32 v29, s47, 18                               // 000000001D78: D761001D 0001242F
	v_writelane_b32 v27, s24, 3                                // 000000001D80: D761001B 00010618
	v_writelane_b32 v29, s48, 19                               // 000000001D88: D761001D 00012630
	v_writelane_b32 v27, s25, 4                                // 000000001D90: D761001B 00010819
	v_writelane_b32 v29, s49, 20                               // 000000001D98: D761001D 00012831
	v_writelane_b32 v27, s26, 5                                // 000000001DA0: D761001B 00010A1A
	v_writelane_b32 v29, s50, 21                               // 000000001DA8: D761001D 00012A32
	v_writelane_b32 v27, s27, 6                                // 000000001DB0: D761001B 00010C1B
	v_writelane_b32 v29, s51, 22                               // 000000001DB8: D761001D 00012C33
	v_writelane_b32 v29, s52, 23                               // 000000001DC0: D761001D 00012E34
	v_writelane_b32 v29, s53, 24                               // 000000001DC8: D761001D 00013035
	v_writelane_b32 v29, s54, 25                               // 000000001DD0: D761001D 00013236
	v_writelane_b32 v29, s55, 26                               // 000000001DD8: D761001D 00013437
	v_writelane_b32 v29, s56, 27                               // 000000001DE0: D761001D 00013638
	v_writelane_b32 v29, s57, 28                               // 000000001DE8: D761001D 00013839
	v_writelane_b32 v29, s58, 29                               // 000000001DF0: D761001D 00013A3A
	v_writelane_b32 v29, s59, 30                               // 000000001DF8: D761001D 00013C3B
	s_or_saveexec_b32 s105, -1                                 // 000000001E00: BEE922C1
	scratch_store_b32 off, v29, off offset:12                  // 000000001E04: ED06807C 0E800000 00000C00
	s_wait_alu 0xfffe                                          // 000000001E10: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000001E14: BEFE0069
	s_mov_b64 s[74:75], s[42:43]                               // 000000001E18: BECA012A
	s_mov_b64 s[72:73], s[40:41]                               // 000000001E1C: BEC80128
	v_writelane_b32 v27, s60, 7                                // 000000001E20: D761001B 00010E3C
	s_mov_b32 s44, s16                                         // 000000001E28: BEAC0010
	s_mov_b32 s52, s16                                         // 000000001E2C: BEB40010
	s_mov_b32 s53, s16                                         // 000000001E30: BEB50010
	s_mov_b32 s54, s16                                         // 000000001E34: BEB60010
	v_writelane_b32 v27, s61, 8                                // 000000001E38: D761001B 0001103D
	s_mov_b32 s55, s16                                         // 000000001E40: BEB70010
	s_mov_b32 s45, s16                                         // 000000001E44: BEAD0010
	s_mov_b32 s46, s16                                         // 000000001E48: BEAE0010
	s_mov_b32 s47, s16                                         // 000000001E4C: BEAF0010
	v_writelane_b32 v27, s62, 9                                // 000000001E50: D761001B 0001123E
	s_mov_b32 s48, s16                                         // 000000001E58: BEB00010
	s_mov_b32 s49, s16                                         // 000000001E5C: BEB10010
	s_mov_b32 s50, s16                                         // 000000001E60: BEB20010
	s_mov_b32 s51, s16                                         // 000000001E64: BEB30010
	v_writelane_b32 v27, s63, 10                               // 000000001E68: D761001B 0001143F
	s_mov_b32 s56, s16                                         // 000000001E70: BEB80010
	s_mov_b32 s57, s16                                         // 000000001E74: BEB90010
	s_mov_b32 s58, s16                                         // 000000001E78: BEBA0010
	s_mov_b32 s59, s16                                         // 000000001E7C: BEBB0010
	v_writelane_b32 v27, s64, 11                               // 000000001E80: D761001B 00011640
	s_wait_alu 0xfffe                                          // 000000001E88: BF88FFFE
	s_and_not1_b32 vcc_lo, exec_lo, s1                         // 000000001E8C: 916A017E
	v_writelane_b32 v27, s65, 12                               // 000000001E90: D761001B 00011841
	v_writelane_b32 v27, s66, 13                               // 000000001E98: D761001B 00011A42
	v_writelane_b32 v27, s67, 14                               // 000000001EA0: D761001B 00011C43
	v_writelane_b32 v27, s68, 15                               // 000000001EA8: D761001B 00011E44
	v_writelane_b32 v27, s69, 16                               // 000000001EB0: D761001B 00012045
	v_writelane_b32 v27, s70, 17                               // 000000001EB8: D761001B 00012246
	v_writelane_b32 v27, s71, 18                               // 000000001EC0: D761001B 00012447
	v_writelane_b32 v27, s72, 19                               // 000000001EC8: D761001B 00012648
	v_writelane_b32 v27, s73, 20                               // 000000001ED0: D761001B 00012849
	v_writelane_b32 v27, s74, 21                               // 000000001ED8: D761001B 00012A4A
	v_writelane_b32 v27, s75, 22                               // 000000001EE0: D761001B 00012C4B
	s_mov_b32 s68, s16                                         // 000000001EE8: BEC40010
	s_mov_b32 s69, s16                                         // 000000001EEC: BEC50010
	s_mov_b32 s70, s16                                         // 000000001EF0: BEC60010
	s_mov_b32 s71, s16                                         // 000000001EF4: BEC70010
	v_writelane_b32 v27, s34, 23                               // 000000001EF8: D761001B 00012E22
	s_mov_b32 s60, s16                                         // 000000001F00: BEBC0010
	s_mov_b32 s61, s16                                         // 000000001F04: BEBD0010
	s_mov_b32 s62, s16                                         // 000000001F08: BEBE0010
	s_mov_b32 s63, s16                                         // 000000001F0C: BEBF0010
	v_writelane_b32 v27, s33, 24                               // 000000001F10: D761001B 00013021
	s_mov_b32 s64, s16                                         // 000000001F18: BEC00010
	s_mov_b32 s65, s16                                         // 000000001F1C: BEC10010
	s_mov_b32 s66, s16                                         // 000000001F20: BEC20010
	s_mov_b32 s67, s16                                         // 000000001F24: BEC30010
	v_writelane_b32 v27, s19, 25                               // 000000001F28: D761001B 00013213
	s_mov_b32 s72, s16                                         // 000000001F30: BEC80010
	s_mov_b32 s73, s16                                         // 000000001F34: BEC90010
	s_mov_b32 s74, s16                                         // 000000001F38: BECA0010
	s_mov_b32 s75, s16                                         // 000000001F3C: BECB0010
	v_writelane_b32 v27, s9, 26                                // 000000001F40: D761001B 00013409
	v_writelane_b32 v27, s17, 27                               // 000000001F48: D761001B 00013611
	v_writelane_b32 v27, s8, 28                                // 000000001F50: D761001B 00013808
	v_writelane_b32 v27, s44, 29                               // 000000001F58: D761001B 00013A2C
	v_writelane_b32 v28, s47, 0                                // 000000001F60: D761001C 0001002F
	v_writelane_b32 v27, s45, 30                               // 000000001F68: D761001B 00013C2D
	v_writelane_b32 v28, s48, 1                                // 000000001F70: D761001C 00010230
	v_writelane_b32 v27, s46, 31                               // 000000001F78: D761001B 00013E2E
	v_writelane_b32 v28, s49, 2                                // 000000001F80: D761001C 00010431
	v_writelane_b32 v28, s50, 3                                // 000000001F88: D761001C 00010632
	v_writelane_b32 v28, s51, 4                                // 000000001F90: D761001C 00010833
	v_writelane_b32 v28, s52, 5                                // 000000001F98: D761001C 00010A34
	v_writelane_b32 v28, s53, 6                                // 000000001FA0: D761001C 00010C35
	v_writelane_b32 v28, s54, 7                                // 000000001FA8: D761001C 00010E36
	v_writelane_b32 v28, s55, 8                                // 000000001FB0: D761001C 00011037
	v_writelane_b32 v28, s56, 9                                // 000000001FB8: D761001C 00011238
	v_writelane_b32 v28, s57, 10                               // 000000001FC0: D761001C 00011439
	v_writelane_b32 v28, s58, 11                               // 000000001FC8: D761001C 0001163A
	v_writelane_b32 v28, s59, 12                               // 000000001FD0: D761001C 0001183B
	s_cbranch_vccnz 41                                         // 000000001FD8: BFA40029 <r_3_3_3_8_8_8+0xa80>
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001FDC: 84828204
	s_wait_alu 0xfffe                                          // 000000001FE0: BF88FFFE
	s_add_nc_u64 s[2:3], s[14:15], s[2:3]                      // 000000001FE4: A982020E
	s_clause 0x1                                               // 000000001FE8: BF850001
	s_load_b512 s[44:59], s[2:3], 0x2500                       // 000000001FEC: F4008B01 F8002500
	s_load_b512 s[60:75], s[2:3], 0x2640                       // 000000001FF4: F4008F01 F8002640
	s_wait_kmcnt 0x0                                           // 000000001FFC: BFC70000
	v_writelane_b32 v27, s44, 29                               // 000000002000: D761001B 00013A2C
	v_writelane_b32 v28, s47, 0                                // 000000002008: D761001C 0001002F
	v_writelane_b32 v27, s45, 30                               // 000000002010: D761001B 00013C2D
	v_writelane_b32 v28, s48, 1                                // 000000002018: D761001C 00010230
	v_writelane_b32 v27, s46, 31                               // 000000002020: D761001B 00013E2E
	v_writelane_b32 v28, s49, 2                                // 000000002028: D761001C 00010431
	v_writelane_b32 v28, s50, 3                                // 000000002030: D761001C 00010632
	v_writelane_b32 v28, s51, 4                                // 000000002038: D761001C 00010833
	v_writelane_b32 v28, s52, 5                                // 000000002040: D761001C 00010A34
	v_writelane_b32 v28, s53, 6                                // 000000002048: D761001C 00010C35
	v_writelane_b32 v28, s54, 7                                // 000000002050: D761001C 00010E36
	v_writelane_b32 v28, s55, 8                                // 000000002058: D761001C 00011037
	v_writelane_b32 v28, s56, 9                                // 000000002060: D761001C 00011238
	v_writelane_b32 v28, s57, 10                               // 000000002068: D761001C 00011439
	v_writelane_b32 v28, s58, 11                               // 000000002070: D761001C 0001163A
	v_writelane_b32 v28, s59, 12                               // 000000002078: D761001C 0001183B
	s_wait_alu 0xfffe                                          // 000000002080: BF88FFFE
	v_writelane_b32 v28, s60, 13                               // 000000002084: D761001C 00011A3C
	v_cndmask_b32_e64 v0, 0, 1, s0                             // 00000000208C: D5010000 00010280
	s_lshl_b64 s[6:7], s[4:5], 2                               // 000000002094: 84868204
	v_cndmask_b32_e64 v1, 0, 1, s104                           // 000000002098: D5010001 01A10280
	s_wait_alu 0xfffe                                          // 0000000020A0: BF88FFFE
	s_add_nc_u64 s[34:35], s[14:15], s[6:7]                    // 0000000020A4: A9A2060E
	v_writelane_b32 v28, s61, 14                               // 0000000020A8: D761001C 00011C3D
	v_cmp_ne_u32_e64 s33, 1, v0                                // 0000000020B0: D44D0021 00020081
	v_writelane_b32 v28, s62, 15                               // 0000000020B8: D761001C 00011E3E
	v_writelane_b32 v28, s63, 16                               // 0000000020C0: D761001C 0001203F
	v_writelane_b32 v28, s64, 17                               // 0000000020C8: D761001C 00012240
	v_writelane_b32 v28, s65, 18                               // 0000000020D0: D761001C 00012441
	v_writelane_b32 v28, s66, 19                               // 0000000020D8: D761001C 00012642
	v_writelane_b32 v28, s67, 20                               // 0000000020E0: D761001C 00012843
	v_writelane_b32 v28, s68, 21                               // 0000000020E8: D761001C 00012A44
	v_writelane_b32 v28, s69, 22                               // 0000000020F0: D761001C 00012C45
	v_writelane_b32 v28, s70, 23                               // 0000000020F8: D761001C 00012E46
	v_writelane_b32 v28, s71, 24                               // 000000002100: D761001C 00013047
	v_writelane_b32 v28, s72, 25                               // 000000002108: D761001C 00013248
	v_writelane_b32 v28, s73, 26                               // 000000002110: D761001C 00013449
	v_writelane_b32 v28, s74, 27                               // 000000002118: D761001C 0001364A
	v_writelane_b32 v28, s75, 28                               // 000000002120: D761001C 0001384B
	s_or_saveexec_b32 s105, -1                                 // 000000002128: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 00000000212C: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000002138: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000213C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000002140: BFC00000
	v_readlane_b32 s2, v29, 4                                  // 000000002144: D7600002 0001091D
	v_readlane_b32 s3, v29, 5                                  // 00000000214C: D7600003 00010B1D
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 000000002154: 916A007E
	v_readlane_b32 s0, v29, 8                                  // 000000002158: D7600000 0001111D
	v_readlane_b32 s1, v29, 9                                  // 000000002160: D7600001 0001131D
	v_cmp_ne_u32_e64 s8, 1, v1                                 // 000000002168: D44D0008 00020281
	s_add_nc_u64 s[4:5], s[2:3], s[6:7]                        // 000000002170: A9840602
	v_writelane_b32 v28, s6, 29                                // 000000002174: D761001C 00013A06
	s_add_nc_u64 s[2:3], s[34:35], 64                          // 00000000217C: A982C022
	v_writelane_b32 v28, s7, 30                                // 000000002180: D761001C 00013C07
	s_add_nc_u64 s[6:7], s[0:1], s[6:7]                        // 000000002188: A9860600
	s_wait_alu 0xfffe                                          // 00000000218C: BF88FFFE
	v_writelane_b32 v28, s6, 31                                // 000000002190: D761001C 00013E06
	s_or_saveexec_b32 s105, -1                                 // 000000002198: BEE922C1
	scratch_store_b32 off, v28, off                            // 00000000219C: ED06807C 0E000000 00000000
	s_wait_alu 0xfffe                                          // 0000000021A8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000021AC: BEFE0069
	v_writelane_b32 v23, s7, 0                                 // 0000000021B0: D7610017 00010007
	v_writelane_b32 v23, s8, 1                                 // 0000000021B8: D7610017 00010208
	s_cbranch_vccnz 4174                                       // 0000000021C0: BFA4104E <r_3_3_3_8_8_8+0x4cfc>
	s_mov_b32 s41, s40                                         // 0000000021C4: BEA90028
	s_mov_b32 s42, s40                                         // 0000000021C8: BEAA0028
	s_mov_b32 s43, s40                                         // 0000000021CC: BEAB0028
	s_and_b32 vcc_lo, exec_lo, s8                              // 0000000021D0: 8B6A087E
	s_wait_alu 0xfffe                                          // 0000000021D4: BF88FFFE
	s_mov_b64 s[12:13], s[40:41]                               // 0000000021D8: BE8C0128
	s_mov_b64 s[8:9], s[40:41]                                 // 0000000021DC: BE880128
	s_mov_b64 s[14:15], s[42:43]                               // 0000000021E0: BE8E012A
	s_mov_b64 s[10:11], s[42:43]                               // 0000000021E4: BE8A012A
	s_wait_alu 0xfffe                                          // 0000000021E8: BF88FFFE
	v_writelane_b32 v25, s8, 18                                // 0000000021EC: D7610019 00012408
	v_writelane_b32 v25, s9, 19                                // 0000000021F4: D7610019 00012609
	v_writelane_b32 v25, s10, 20                               // 0000000021FC: D7610019 0001280A
	v_writelane_b32 v25, s11, 21                               // 000000002204: D7610019 00012A0B
	v_writelane_b32 v25, s12, 22                               // 00000000220C: D7610019 00012C0C
	v_writelane_b32 v25, s13, 23                               // 000000002214: D7610019 00012E0D
	v_writelane_b32 v25, s14, 24                               // 00000000221C: D7610019 0001300E
	v_writelane_b32 v25, s15, 25                               // 000000002224: D7610019 0001320F
	s_cbranch_vccnz 43                                         // 00000000222C: BFA4002B <r_3_3_3_8_8_8+0xcdc>
	s_or_saveexec_b32 s105, -1                                 // 000000002230: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000002234: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000002240: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002244: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000002248: BFC00000
	v_readlane_b32 s0, v29, 6                                  // 00000000224C: D7600000 00010D1D
	v_readlane_b32 s1, v29, 7                                  // 000000002254: D7600001 00010F1D
	s_or_saveexec_b32 s105, -1                                 // 00000000225C: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000002260: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 00000000226C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002270: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000002274: BFC00000
	v_readlane_b32 s8, v29, 29                                 // 000000002278: D7600008 00013B1D
	v_readlane_b32 s9, v29, 30                                 // 000000002280: D7600009 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002288: BF870001
	s_add_nc_u64 s[0:1], s[0:1], s[8:9]                        // 00000000228C: A9800800
	s_load_b256 s[8:15], s[0:1], 0x0                           // 000000002290: F4006200 F8000000
	s_wait_kmcnt 0x0                                           // 000000002298: BFC70000
	v_writelane_b32 v25, s8, 18                                // 00000000229C: D7610019 00012408
	v_writelane_b32 v25, s9, 19                                // 0000000022A4: D7610019 00012609
	v_writelane_b32 v25, s10, 20                               // 0000000022AC: D7610019 0001280A
	v_writelane_b32 v25, s11, 21                               // 0000000022B4: D7610019 00012A0B
	v_writelane_b32 v25, s12, 22                               // 0000000022BC: D7610019 00012C0C
	v_writelane_b32 v25, s13, 23                               // 0000000022C4: D7610019 00012E0D
	v_writelane_b32 v25, s14, 24                               // 0000000022CC: D7610019 0001300E
	v_writelane_b32 v25, s15, 25                               // 0000000022D4: D7610019 0001320F
	s_or_saveexec_b32 s105, -1                                 // 0000000022DC: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 0000000022E0: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 0000000022EC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000022F0: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000022F4: BFC00000
	v_readlane_b32 s0, v29, 2                                  // 0000000022F8: D7600000 0001051D
	v_readlane_b32 s1, v29, 3                                  // 000000002300: D7600001 0001071D
	s_or_saveexec_b32 s105, -1                                 // 000000002308: BEE922C1
	scratch_load_b32 v29, off, off                             // 00000000230C: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 000000002318: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000231C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000002320: BFC00000
	v_readlane_b32 s8, v29, 29                                 // 000000002324: D7600008 00013B1D
	v_readlane_b32 s9, v29, 30                                 // 00000000232C: D7600009 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002334: BF870001
	s_add_nc_u64 s[0:1], s[0:1], s[8:9]                        // 000000002338: A9800800
	s_clause 0x2                                               // 00000000233C: BF850002
	s_load_b256 s[44:51], s[4:5], 0x0                          // 000000002340: F4006B02 F8000000
	s_load_b256 s[8:15], s[0:1], 0x0                           // 000000002348: F4006200 F8000000
	s_load_b512 s[52:67], s[6:7], 0x0                          // 000000002350: F4008D03 F8000000
	s_movk_i32 s0, 0xfe80                                      // 000000002358: B000FE80
	s_mov_b32 s1, -1                                           // 00000000235C: BE8100C1
	s_movk_i32 s6, 0xfc20                                      // 000000002360: B006FC20
	s_wait_alu 0xfffe                                          // 000000002364: BF88FFFE
	s_add_nc_u64 s[0:1], s[34:35], s[0:1]                      // 000000002368: A9800022
	s_mov_b32 s7, -1                                           // 00000000236C: BE8700C1
	s_load_b512 s[16:31], s[0:1], 0x0                          // 000000002370: F4008400 F8000000
	s_add_nc_u64 s[6:7], s[34:35], s[6:7]                      // 000000002378: A9860622
	s_clause 0x1                                               // 00000000237C: BF850001
	s_load_b256 s[72:79], s[34:35], 0x1140                     // 000000002380: F4007211 F8001140
	s_load_b256 s[96:103], s[6:7], 0x0                         // 000000002388: F4007803 F8000000
	s_add_nc_u64 s[0:1], s[34:35], 64                          // 000000002390: A980C022
	s_mov_b32 s6, s104                                         // 000000002394: BE860068
	s_wait_kmcnt 0x0                                           // 000000002398: BFC70000
	v_writelane_b32 v25, s8, 10                                // 00000000239C: D7610019 00011408
	v_writelane_b32 v23, s52, 2                                // 0000000023A4: D7610017 00010434
	v_writelane_b32 v25, s9, 11                                // 0000000023AC: D7610019 00011609
	v_writelane_b32 v23, s53, 3                                // 0000000023B4: D7610017 00010635
	v_writelane_b32 v25, s10, 12                               // 0000000023BC: D7610019 0001180A
	v_writelane_b32 v23, s54, 4                                // 0000000023C4: D7610017 00010836
	v_writelane_b32 v25, s11, 13                               // 0000000023CC: D7610019 00011A0B
	v_writelane_b32 v23, s55, 5                                // 0000000023D4: D7610017 00010A37
	v_writelane_b32 v25, s12, 14                               // 0000000023DC: D7610019 00011C0C
	v_writelane_b32 v23, s56, 6                                // 0000000023E4: D7610017 00010C38
	v_writelane_b32 v25, s13, 15                               // 0000000023EC: D7610019 00011E0D
	v_writelane_b32 v23, s57, 7                                // 0000000023F4: D7610017 00010E39
	v_writelane_b32 v25, s14, 16                               // 0000000023FC: D7610019 0001200E
	v_writelane_b32 v23, s58, 8                                // 000000002404: D7610017 0001103A
	v_writelane_b32 v25, s15, 17                               // 00000000240C: D7610019 0001220F
	v_writelane_b32 v23, s59, 9                                // 000000002414: D7610017 0001123B
	s_mov_b32 s8, 0                                            // 00000000241C: BE880080
	v_writelane_b32 v23, s60, 10                               // 000000002420: D7610017 0001143C
	v_writelane_b32 v23, s61, 11                               // 000000002428: D7610017 0001163D
	v_writelane_b32 v23, s62, 12                               // 000000002430: D7610017 0001183E
	v_writelane_b32 v23, s63, 13                               // 000000002438: D7610017 00011A3F
	v_writelane_b32 v23, s64, 14                               // 000000002440: D7610017 00011C40
	v_writelane_b32 v23, s65, 15                               // 000000002448: D7610017 00011E41
	v_writelane_b32 v23, s66, 16                               // 000000002450: D7610017 00012042
	v_writelane_b32 v23, s67, 17                               // 000000002458: D7610017 00012243
	v_writelane_b32 v23, s16, 26                               // 000000002460: D7610017 00013410
	v_writelane_b32 v25, s22, 0                                // 000000002468: D7610019 00010016
	v_writelane_b32 v23, s17, 27                               // 000000002470: D7610017 00013611
	v_writelane_b32 v25, s23, 1                                // 000000002478: D7610019 00010217
	v_writelane_b32 v23, s18, 28                               // 000000002480: D7610017 00013812
	v_writelane_b32 v25, s24, 2                                // 000000002488: D7610019 00010418
	v_writelane_b32 v23, s19, 29                               // 000000002490: D7610017 00013A13
	v_writelane_b32 v25, s25, 3                                // 000000002498: D7610019 00010619
	v_writelane_b32 v23, s20, 30                               // 0000000024A0: D7610017 00013C14
	v_writelane_b32 v25, s26, 4                                // 0000000024A8: D7610019 0001081A
	v_writelane_b32 v23, s21, 31                               // 0000000024B0: D7610017 00013E15
	v_writelane_b32 v25, s27, 5                                // 0000000024B8: D7610019 00010A1B
	v_writelane_b32 v23, s96, 18                               // 0000000024C0: D7610017 00012460
	v_writelane_b32 v25, s28, 6                                // 0000000024C8: D7610019 00010C1C
	v_writelane_b32 v23, s97, 19                               // 0000000024D0: D7610017 00012661
	v_writelane_b32 v25, s29, 7                                // 0000000024D8: D7610019 00010E1D
	v_writelane_b32 v23, s98, 20                               // 0000000024E0: D7610017 00012862
	v_writelane_b32 v25, s30, 8                                // 0000000024E8: D7610019 0001101E
	v_writelane_b32 v23, s99, 21                               // 0000000024F0: D7610017 00012A63
	v_writelane_b32 v25, s31, 9                                // 0000000024F8: D7610019 0001121F
	v_writelane_b32 v23, s100, 22                              // 000000002500: D7610017 00012C64
	v_writelane_b32 v23, s101, 23                              // 000000002508: D7610017 00012E65
	v_writelane_b32 v23, s102, 24                              // 000000002510: D7610017 00013066
	v_writelane_b32 v23, s103, 25                              // 000000002518: D7610017 00013267
	s_branch 175                                               // 000000002520: BFA000AF <r_3_3_3_8_8_8+0x11e0>
	s_or_saveexec_b32 s105, -1                                 // 000000002524: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000002528: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000002534: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002538: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000253C: BFC00000
	v_readlane_b32 s0, v29, 6                                  // 000000002540: D7600000 00010D1D
	v_readlane_b32 s1, v29, 7                                  // 000000002548: D7600001 00010F1D
	s_or_saveexec_b32 s105, -1                                 // 000000002550: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000002554: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 000000002560: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002564: BEFE0069
	s_mov_b32 s42, s40                                         // 000000002568: BEAA0028
	s_mov_b32 s43, s40                                         // 00000000256C: BEAB0028
	s_mov_b32 s41, s40                                         // 000000002570: BEA90028
	s_wait_alu 0xfffe                                          // 000000002574: BF88FFFE
	s_mov_b64 s[98:99], s[42:43]                               // 000000002578: BEE2012A
	s_mov_b64 s[102:103], s[42:43]                             // 00000000257C: BEE6012A
	s_mov_b64 s[96:97], s[40:41]                               // 000000002580: BEE00128
	s_mov_b64 s[100:101], s[40:41]                             // 000000002584: BEE40128
	s_wait_alu 0xfffe                                          // 000000002588: BF88FFFE
	v_writelane_b32 v23, s96, 18                               // 00000000258C: D7610017 00012460
	s_wait_loadcnt 0x0                                         // 000000002594: BFC00000
	v_readlane_b32 s6, v29, 29                                 // 000000002598: D7600006 00013B1D
	v_readlane_b32 s7, v29, 30                                 // 0000000025A0: D7600007 00013D1D
	s_mov_b64 s[54:55], s[42:43]                               // 0000000025A8: BEB6012A
	s_mov_b64 s[66:67], s[42:43]                               // 0000000025AC: BEC2012A
	v_writelane_b32 v23, s97, 19                               // 0000000025B0: D7610017 00012661
	s_mov_b64 s[58:59], s[42:43]                               // 0000000025B8: BEBA012A
	s_add_nc_u64 s[0:1], s[0:1], s[6:7]                        // 0000000025BC: A9800600
	s_clause 0x1                                               // 0000000025C0: BF850001
	s_load_b256 s[44:51], s[4:5], 0x0                          // 0000000025C4: F4006B02 F8000000
	s_load_b256 s[4:11], s[0:1], 0x0                           // 0000000025CC: F4006100 F8000000
	s_mov_b64 s[52:53], s[40:41]                               // 0000000025D4: BEB40128
	v_writelane_b32 v23, s98, 20                               // 0000000025D8: D7610017 00012862
	s_mov_b64 s[62:63], s[42:43]                               // 0000000025E0: BEBE012A
	s_mov_b64 s[64:65], s[40:41]                               // 0000000025E4: BEC00128
	s_mov_b64 s[56:57], s[40:41]                               // 0000000025E8: BEB80128
	s_mov_b64 s[60:61], s[40:41]                               // 0000000025EC: BEBC0128
	v_writelane_b32 v23, s99, 21                               // 0000000025F0: D7610017 00012A63
	s_mov_b64 s[12:13], s[40:41]                               // 0000000025F8: BE8C0128
	s_mov_b64 s[14:15], s[42:43]                               // 0000000025FC: BE8E012A
	s_mov_b64 s[24:25], s[40:41]                               // 000000002600: BE980128
	s_mov_b64 s[16:17], s[40:41]                               // 000000002604: BE900128
	v_writelane_b32 v23, s100, 22                              // 000000002608: D7610017 00012C64
	s_mov_b64 s[20:21], s[40:41]                               // 000000002610: BE940128
	s_mov_b64 s[28:29], s[40:41]                               // 000000002614: BE9C0128
	s_mov_b64 s[26:27], s[42:43]                               // 000000002618: BE9A012A
	s_mov_b64 s[18:19], s[42:43]                               // 00000000261C: BE92012A
	v_writelane_b32 v23, s101, 23                              // 000000002620: D7610017 00012E65
	s_mov_b64 s[22:23], s[42:43]                               // 000000002628: BE96012A
	s_mov_b64 s[30:31], s[42:43]                               // 00000000262C: BE9E012A
	s_wait_kmcnt 0x0                                           // 000000002630: BFC70000
	v_writelane_b32 v25, s4, 18                                // 000000002634: D7610019 00012404
	s_load_b256 s[72:79], s[34:35], 0x1140                     // 00000000263C: F4007211 F8001140
	v_writelane_b32 v23, s102, 24                              // 000000002644: D7610017 00013066
	s_mov_b64 s[0:1], s[2:3]                                   // 00000000264C: BE800102
	v_writelane_b32 v25, s5, 19                                // 000000002650: D7610019 00012605
	v_writelane_b32 v23, s103, 25                              // 000000002658: D7610017 00013267
	v_writelane_b32 v25, s6, 20                                // 000000002660: D7610019 00012806
	v_writelane_b32 v23, s52, 2                                // 000000002668: D7610017 00010434
	v_writelane_b32 v25, s7, 21                                // 000000002670: D7610019 00012A07
	v_writelane_b32 v23, s53, 3                                // 000000002678: D7610017 00010635
	v_writelane_b32 v25, s8, 22                                // 000000002680: D7610019 00012C08
	v_writelane_b32 v23, s54, 4                                // 000000002688: D7610017 00010836
	v_writelane_b32 v25, s9, 23                                // 000000002690: D7610019 00012E09
	v_writelane_b32 v23, s55, 5                                // 000000002698: D7610017 00010A37
	v_writelane_b32 v25, s10, 24                               // 0000000026A0: D7610019 0001300A
	v_writelane_b32 v23, s56, 6                                // 0000000026A8: D7610017 00010C38
	v_writelane_b32 v25, s11, 25                               // 0000000026B0: D7610019 0001320B
	s_mov_b64 s[8:9], s[40:41]                                 // 0000000026B8: BE880128
	v_writelane_b32 v23, s57, 7                                // 0000000026BC: D7610017 00010E39
	s_mov_b64 s[10:11], s[42:43]                               // 0000000026C4: BE8A012A
	s_mov_b32 s6, -1                                           // 0000000026C8: BE8600C1
	s_wait_alu 0xfffe                                          // 0000000026CC: BF88FFFE
	v_writelane_b32 v25, s8, 10                                // 0000000026D0: D7610019 00011408
	v_writelane_b32 v23, s58, 8                                // 0000000026D8: D7610017 0001103A
	v_writelane_b32 v25, s9, 11                                // 0000000026E0: D7610019 00011609
	v_writelane_b32 v23, s59, 9                                // 0000000026E8: D7610017 0001123B
	v_writelane_b32 v25, s10, 12                               // 0000000026F0: D7610019 0001180A
	v_writelane_b32 v23, s60, 10                               // 0000000026F8: D7610017 0001143C
	v_writelane_b32 v25, s11, 13                               // 000000002700: D7610019 00011A0B
	v_writelane_b32 v23, s61, 11                               // 000000002708: D7610017 0001163D
	v_writelane_b32 v25, s12, 14                               // 000000002710: D7610019 00011C0C
	v_writelane_b32 v23, s62, 12                               // 000000002718: D7610017 0001183E
	v_writelane_b32 v25, s13, 15                               // 000000002720: D7610019 00011E0D
	v_writelane_b32 v23, s63, 13                               // 000000002728: D7610017 00011A3F
	v_writelane_b32 v25, s14, 16                               // 000000002730: D7610019 0001200E
	v_writelane_b32 v23, s64, 14                               // 000000002738: D7610017 00011C40
	v_writelane_b32 v25, s15, 17                               // 000000002740: D7610019 0001220F
	v_writelane_b32 v23, s65, 15                               // 000000002748: D7610017 00011E41
	v_writelane_b32 v23, s66, 16                               // 000000002750: D7610017 00012042
	v_writelane_b32 v23, s67, 17                               // 000000002758: D7610017 00012243
	v_writelane_b32 v23, s16, 26                               // 000000002760: D7610017 00013410
	v_writelane_b32 v25, s22, 0                                // 000000002768: D7610019 00010016
	v_writelane_b32 v23, s17, 27                               // 000000002770: D7610017 00013611
	v_writelane_b32 v25, s23, 1                                // 000000002778: D7610019 00010217
	v_writelane_b32 v23, s18, 28                               // 000000002780: D7610017 00013812
	v_writelane_b32 v25, s24, 2                                // 000000002788: D7610019 00010418
	v_writelane_b32 v23, s19, 29                               // 000000002790: D7610017 00013A13
	v_writelane_b32 v25, s25, 3                                // 000000002798: D7610019 00010619
	v_writelane_b32 v23, s20, 30                               // 0000000027A0: D7610017 00013C14
	v_writelane_b32 v25, s26, 4                                // 0000000027A8: D7610019 0001081A
	v_writelane_b32 v23, s21, 31                               // 0000000027B0: D7610017 00013E15
	v_writelane_b32 v25, s27, 5                                // 0000000027B8: D7610019 00010A1B
	v_writelane_b32 v25, s28, 6                                // 0000000027C0: D7610019 00010C1C
	v_writelane_b32 v25, s29, 7                                // 0000000027C8: D7610019 00010E1D
	v_writelane_b32 v25, s30, 8                                // 0000000027D0: D7610019 0001101E
	v_writelane_b32 v25, s31, 9                                // 0000000027D8: D7610019 0001121F
	s_wait_kmcnt 0x0                                           // 0000000027E0: BFC70000
	v_writelane_b32 v25, s72, 26                               // 0000000027E4: D7610019 00013448
	v_writelane_b32 v29, s78, 0                                // 0000000027EC: D761001D 0001004E
	s_wait_alu 0xfffe                                          // 0000000027F4: BF88FFFE
	s_mov_b32 s52, s8                                          // 0000000027F8: BEB40008
	s_mov_b32 s60, s8                                          // 0000000027FC: BEBC0008
	s_mov_b32 s61, s8                                          // 000000002800: BEBD0008
	s_mov_b32 s62, s8                                          // 000000002804: BEBE0008
	v_writelane_b32 v29, s79, 1                                // 000000002808: D761001D 0001024F
	s_mov_b32 s63, s8                                          // 000000002810: BEBF0008
	s_mov_b32 s53, s8                                          // 000000002814: BEB50008
	s_mov_b32 s54, s8                                          // 000000002818: BEB60008
	s_mov_b32 s55, s8                                          // 00000000281C: BEB70008
	v_writelane_b32 v29, s44, 2                                // 000000002820: D761001D 0001042C
	s_mov_b32 s56, s8                                          // 000000002828: BEB80008
	s_mov_b32 s57, s8                                          // 00000000282C: BEB90008
	s_mov_b32 s58, s8                                          // 000000002830: BEBA0008
	s_mov_b32 s59, s8                                          // 000000002834: BEBB0008
	v_writelane_b32 v29, s45, 3                                // 000000002838: D761001D 0001062D
	s_mov_b32 s64, s8                                          // 000000002840: BEC00008
	s_mov_b32 s65, s8                                          // 000000002844: BEC10008
	s_mov_b32 s66, s8                                          // 000000002848: BEC20008
	s_mov_b32 s67, s8                                          // 00000000284C: BEC30008
	v_writelane_b32 v29, s46, 4                                // 000000002850: D761001D 0001082E
	v_writelane_b32 v25, s73, 27                               // 000000002858: D7610019 00013649
	s_mov_b32 s24, s8                                          // 000000002860: BE980008
	s_mov_b32 s25, s8                                          // 000000002864: BE990008
	s_mov_b32 s26, s8                                          // 000000002868: BE9A0008
	v_writelane_b32 v29, s47, 5                                // 00000000286C: D761001D 00010A2F
	v_writelane_b32 v25, s74, 28                               // 000000002874: D7610019 0001384A
	s_mov_b32 s27, s8                                          // 00000000287C: BE9B0008
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 000000002880: 916A067E
	s_mov_b32 s16, s8                                          // 000000002884: BE900008
	v_writelane_b32 v29, s48, 6                                // 000000002888: D761001D 00010C30
	v_writelane_b32 v25, s75, 29                               // 000000002890: D7610019 00013A4B
	s_mov_b32 s17, s8                                          // 000000002898: BE910008
	s_mov_b32 s18, s8                                          // 00000000289C: BE920008
	s_mov_b32 s19, s8                                          // 0000000028A0: BE930008
	v_writelane_b32 v29, s49, 7                                // 0000000028A4: D761001D 00010E31
	v_writelane_b32 v25, s76, 30                               // 0000000028AC: D7610019 00013C4C
	s_mov_b32 s20, s8                                          // 0000000028B4: BE940008
	s_mov_b32 s21, s8                                          // 0000000028B8: BE950008
	s_mov_b32 s22, s8                                          // 0000000028BC: BE960008
	v_writelane_b32 v29, s50, 8                                // 0000000028C0: D761001D 00011032
	s_mov_b32 s23, s8                                          // 0000000028C8: BE970008
	s_mov_b32 s28, s8                                          // 0000000028CC: BE9C0008
	s_mov_b32 s29, s8                                          // 0000000028D0: BE9D0008
	s_mov_b32 s30, s8                                          // 0000000028D4: BE9E0008
	v_writelane_b32 v29, s51, 9                                // 0000000028D8: D761001D 00011233
	s_mov_b32 s48, s8                                          // 0000000028E0: BEB00008
	s_mov_b32 s49, s8                                          // 0000000028E4: BEB10008
	s_mov_b32 s50, s8                                          // 0000000028E8: BEB20008
	s_mov_b32 s51, s8                                          // 0000000028EC: BEB30008
	s_wait_alu 0xfffe                                          // 0000000028F0: BF88FFFE
	v_writelane_b32 v29, s52, 10                               // 0000000028F4: D761001D 00011434
	s_mov_b32 s44, s8                                          // 0000000028FC: BEAC0008
	s_mov_b32 s45, s8                                          // 000000002900: BEAD0008
	s_mov_b32 s46, s8                                          // 000000002904: BEAE0008
	s_mov_b32 s47, s8                                          // 000000002908: BEAF0008
	v_writelane_b32 v29, s53, 11                               // 00000000290C: D761001D 00011635
	s_mov_b32 s31, s8                                          // 000000002914: BE9F0008
	v_writelane_b32 v25, s77, 31                               // 000000002918: D7610019 00013E4D
	v_writelane_b32 v29, s54, 12                               // 000000002920: D761001D 00011836
	v_writelane_b32 v29, s55, 13                               // 000000002928: D761001D 00011A37
	v_writelane_b32 v29, s56, 14                               // 000000002930: D761001D 00011C38
	v_writelane_b32 v29, s57, 15                               // 000000002938: D761001D 00011E39
	v_writelane_b32 v29, s58, 16                               // 000000002940: D761001D 0001203A
	v_writelane_b32 v29, s59, 17                               // 000000002948: D761001D 0001223B
	v_writelane_b32 v29, s60, 18                               // 000000002950: D761001D 0001243C
	v_writelane_b32 v29, s61, 19                               // 000000002958: D761001D 0001263D
	v_writelane_b32 v29, s62, 20                               // 000000002960: D761001D 0001283E
	v_writelane_b32 v29, s63, 21                               // 000000002968: D761001D 00012A3F
	v_writelane_b32 v29, s64, 22                               // 000000002970: D761001D 00012C40
	v_writelane_b32 v29, s65, 23                               // 000000002978: D761001D 00012E41
	v_writelane_b32 v29, s66, 24                               // 000000002980: D761001D 00013042
	v_writelane_b32 v29, s67, 25                               // 000000002988: D761001D 00013243
	s_cbranch_vccnz 40                                         // 000000002990: BFA40028 <r_3_3_3_8_8_8+0x1434>
	s_load_b512 s[16:31], s[0:1], 0x2500                       // 000000002994: F4008400 F8002500
	s_wait_kmcnt 0x0                                           // 00000000299C: BFC70000
	v_writelane_b32 v29, s16, 10                               // 0000000029A0: D761001D 00011410
	v_writelane_b32 v29, s17, 11                               // 0000000029A8: D761001D 00011611
	v_writelane_b32 v29, s18, 12                               // 0000000029B0: D761001D 00011812
	v_writelane_b32 v29, s19, 13                               // 0000000029B8: D761001D 00011A13
	v_writelane_b32 v29, s20, 14                               // 0000000029C0: D761001D 00011C14
	v_writelane_b32 v29, s21, 15                               // 0000000029C8: D761001D 00011E15
	v_writelane_b32 v29, s22, 16                               // 0000000029D0: D761001D 00012016
	v_writelane_b32 v29, s23, 17                               // 0000000029D8: D761001D 00012217
	v_writelane_b32 v29, s24, 18                               // 0000000029E0: D761001D 00012418
	v_writelane_b32 v29, s25, 19                               // 0000000029E8: D761001D 00012619
	v_writelane_b32 v29, s26, 20                               // 0000000029F0: D761001D 0001281A
	v_writelane_b32 v29, s27, 21                               // 0000000029F8: D761001D 00012A1B
	v_writelane_b32 v29, s28, 22                               // 000000002A00: D761001D 00012C1C
	v_writelane_b32 v29, s29, 23                               // 000000002A08: D761001D 00012E1D
	v_writelane_b32 v29, s30, 24                               // 000000002A10: D761001D 0001301E
	v_writelane_b32 v29, s31, 25                               // 000000002A18: D761001D 0001321F
	s_clause 0x1                                               // 000000002A20: BF850001
	s_load_b256 s[44:51], s[0:1], 0x23e0                       // 000000002A24: F4006B00 F80023E0
	s_load_b512 s[16:31], s[0:1], 0x2640                       // 000000002A2C: F4008400 F8002640
	s_wait_kmcnt 0x0                                           // 000000002A34: BFC70000
	s_wait_alu 0xfffe                                          // 000000002A38: BF88FFFE
	v_writelane_b32 v29, s44, 26                               // 000000002A3C: D761001D 0001342C
	v_writelane_b32 v29, s45, 27                               // 000000002A44: D761001D 0001362D
	v_writelane_b32 v29, s46, 28                               // 000000002A4C: D761001D 0001382E
	v_writelane_b32 v29, s47, 29                               // 000000002A54: D761001D 00013A2F
	v_writelane_b32 v29, s48, 30                               // 000000002A5C: D761001D 00013C30
	v_writelane_b32 v29, s49, 31                               // 000000002A64: D761001D 00013E31
	s_or_saveexec_b32 s105, -1                                 // 000000002A6C: BEE922C1
	scratch_store_b32 off, v29, off offset:52                  // 000000002A70: ED06807C 0E800000 00003400
	s_wait_alu 0xfffe                                          // 000000002A7C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002A80: BEFE0069
	v_writelane_b32 v28, s50, 0                                // 000000002A84: D761001C 00010032
	s_and_b32 vcc_lo, exec_lo, s33                             // 000000002A8C: 8B6A217E
	v_writelane_b32 v28, s51, 1                                // 000000002A90: D761001C 00010233
	s_or_saveexec_b32 s105, -1                                 // 000000002A98: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000002A9C: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000002AA8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002AAC: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000002AB0: BFC00000
	v_readlane_b32 s2, v29, 12                                 // 000000002AB4: D7600002 0001191D
	v_readlane_b32 s3, v29, 13                                 // 000000002ABC: D7600003 00011B1D
	s_or_saveexec_b32 s105, -1                                 // 000000002AC4: BEE922C1
	scratch_load_b32 v26, off, off                             // 000000002AC8: ED05007C 0000001A 00000000
	s_wait_alu 0xfffe                                          // 000000002AD4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002AD8: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000002ADC: BFC00000
	v_readlane_b32 s4, v26, 29                                 // 000000002AE0: D7600004 00013B1A
	v_readlane_b32 s5, v26, 30                                 // 000000002AE8: D7600005 00013D1A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000002AF0: BF8700B1
	s_add_nc_u64 s[6:7], s[2:3], s[4:5]                        // 000000002AF4: A9860402
	v_readlane_b32 s2, v29, 16                                 // 000000002AF8: D7600002 0001211D
	v_readlane_b32 s3, v29, 17                                 // 000000002B00: D7600003 0001231D
	s_add_nc_u64 s[8:9], s[2:3], s[4:5]                        // 000000002B08: A9880402
	s_add_nc_u64 s[4:5], s[34:35], 0x80                        // 000000002B0C: A984FF22 00000080
	s_wait_alu 0xfffe                                          // 000000002B14: BF88FFFE
	v_writelane_b32 v28, s8, 2                                 // 000000002B18: D761001C 00010408
	v_writelane_b32 v28, s9, 3                                 // 000000002B20: D761001C 00010609
	s_cbranch_vccnz 3687                                       // 000000002B28: BFA40E67 <r_3_3_3_8_8_8+0x4ec8>
	s_or_saveexec_b32 s105, -1                                 // 000000002B2C: BEE922C1
	s_wait_alu 0xfffe                                          // 000000002B30: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002B34: BEFE0069
	s_mov_b32 s42, s40                                         // 000000002B38: BEAA0028
	s_mov_b32 s43, s40                                         // 000000002B3C: BEAB0028
	s_mov_b32 s41, s40                                         // 000000002B40: BEA90028
	s_wait_alu 0xfffe                                          // 000000002B44: BF88FFFE
	s_mov_b64 s[46:47], s[42:43]                               // 000000002B48: BEAE012A
	s_mov_b64 s[50:51], s[42:43]                               // 000000002B4C: BEB2012A
	s_mov_b64 s[44:45], s[40:41]                               // 000000002B50: BEAC0128
	s_mov_b64 s[48:49], s[40:41]                               // 000000002B54: BEB00128
	s_wait_alu 0xfffe                                          // 000000002B58: BF88FFFE
	v_writelane_b32 v26, s44, 4                                // 000000002B5C: D761001A 0001082C
	v_readlane_b32 s2, v23, 1                                  // 000000002B64: D7600002 00010317
	v_writelane_b32 v26, s45, 5                                // 000000002B6C: D761001A 00010A2D
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002B74: BF870002
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000002B78: 8B6A027E
	v_writelane_b32 v26, s46, 6                                // 000000002B7C: D761001A 00010C2E
	v_writelane_b32 v26, s47, 7                                // 000000002B84: D761001A 00010E2F
	v_writelane_b32 v26, s48, 8                                // 000000002B8C: D761001A 00011030
	v_writelane_b32 v26, s49, 9                                // 000000002B94: D761001A 00011231
	v_writelane_b32 v26, s50, 10                               // 000000002B9C: D761001A 00011432
	v_writelane_b32 v26, s51, 11                               // 000000002BA4: D761001A 00011633
	s_cbranch_vccnz 43                                         // 000000002BAC: BFA4002B <r_3_3_3_8_8_8+0x165c>
	s_or_saveexec_b32 s105, -1                                 // 000000002BB0: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000002BB4: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000002BC0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002BC4: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000002BC8: BFC00000
	v_readlane_b32 s2, v29, 14                                 // 000000002BCC: D7600002 00011D1D
	v_readlane_b32 s3, v29, 15                                 // 000000002BD4: D7600003 00011F1D
	s_or_saveexec_b32 s105, -1                                 // 000000002BDC: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000002BE0: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 000000002BEC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002BF0: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000002BF4: BFC00000
	v_readlane_b32 s10, v29, 29                                // 000000002BF8: D760000A 00013B1D
	v_readlane_b32 s11, v29, 30                                // 000000002C00: D760000B 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002C08: BF870001
	s_add_nc_u64 s[2:3], s[2:3], s[10:11]                      // 000000002C0C: A9820A02
	s_load_b256 s[44:51], s[2:3], 0x0                          // 000000002C10: F4006B01 F8000000
	s_wait_kmcnt 0x0                                           // 000000002C18: BFC70000
	v_writelane_b32 v26, s44, 4                                // 000000002C1C: D761001A 0001082C
	v_writelane_b32 v26, s45, 5                                // 000000002C24: D761001A 00010A2D
	v_writelane_b32 v26, s46, 6                                // 000000002C2C: D761001A 00010C2E
	v_writelane_b32 v26, s47, 7                                // 000000002C34: D761001A 00010E2F
	v_writelane_b32 v26, s48, 8                                // 000000002C3C: D761001A 00011030
	v_writelane_b32 v26, s49, 9                                // 000000002C44: D761001A 00011231
	v_writelane_b32 v26, s50, 10                               // 000000002C4C: D761001A 00011432
	v_writelane_b32 v26, s51, 11                               // 000000002C54: D761001A 00011633
	s_or_saveexec_b32 s105, -1                                 // 000000002C5C: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000002C60: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000002C6C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002C70: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000002C74: BFC00000
	v_readlane_b32 s2, v29, 10                                 // 000000002C78: D7600002 0001151D
	v_readlane_b32 s3, v29, 11                                 // 000000002C80: D7600003 0001171D
	s_or_saveexec_b32 s105, -1                                 // 000000002C88: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000002C8C: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 000000002C98: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002C9C: BEFE0069
	s_load_b256 s[44:51], s[6:7], 0x0                          // 000000002CA0: F4006B03 F8000000
	s_wait_loadcnt 0x0                                         // 000000002CA8: BFC00000
	v_readlane_b32 s10, v29, 29                                // 000000002CAC: D760000A 00013B1D
	v_readlane_b32 s11, v29, 30                                // 000000002CB4: D760000B 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002CBC: BF870001
	s_add_nc_u64 s[2:3], s[2:3], s[10:11]                      // 000000002CC0: A9820A02
	s_wait_kmcnt 0x0                                           // 000000002CC4: BFC70000
	v_writelane_b32 v21, s44, 20                               // 000000002CC8: D7610015 0001282C
	v_writelane_b32 v21, s45, 21                               // 000000002CD0: D7610015 00012A2D
	v_writelane_b32 v21, s46, 22                               // 000000002CD8: D7610015 00012C2E
	v_writelane_b32 v21, s47, 23                               // 000000002CE0: D7610015 00012E2F
	v_writelane_b32 v21, s48, 24                               // 000000002CE8: D7610015 00013030
	v_writelane_b32 v21, s49, 25                               // 000000002CF0: D7610015 00013231
	v_writelane_b32 v21, s50, 26                               // 000000002CF8: D7610015 00013432
	v_writelane_b32 v21, s51, 27                               // 000000002D00: D7610015 00013633
	s_load_b256 s[44:51], s[2:3], 0x0                          // 000000002D08: F4006B01 F8000000
	s_wait_kmcnt 0x0                                           // 000000002D10: BFC70000
	v_writelane_b32 v21, s44, 28                               // 000000002D14: D7610015 0001382C
	v_writelane_b32 v26, s48, 0                                // 000000002D1C: D761001A 00010030
	v_writelane_b32 v21, s45, 29                               // 000000002D24: D7610015 00013A2D
	v_writelane_b32 v26, s49, 1                                // 000000002D2C: D761001A 00010231
	v_writelane_b32 v21, s46, 30                               // 000000002D34: D7610015 00013C2E
	v_writelane_b32 v26, s50, 2                                // 000000002D3C: D761001A 00010432
	v_writelane_b32 v21, s47, 31                               // 000000002D44: D7610015 00013E2F
	v_writelane_b32 v26, s51, 3                                // 000000002D4C: D761001A 00010633
	s_or_saveexec_b32 s105, -1                                 // 000000002D54: BEE922C1
	s_wait_alu 0xfffe                                          // 000000002D58: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002D5C: BEFE0069
	s_load_b512 s[44:59], s[8:9], 0x0                          // 000000002D60: F4008B04 F8000000
	s_or_saveexec_b32 s105, -1                                 // 000000002D68: BEE922C1
	v_mov_b32_e32 v29, v28                                     // 000000002D6C: 7E3A031C
	s_wait_alu 0xfffe                                          // 000000002D70: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002D74: BEFE0069
	s_wait_kmcnt 0x0                                           // 000000002D78: BFC70000
	v_writelane_b32 v29, s44, 4                                // 000000002D7C: D761001D 0001082C
	s_movk_i32 s2, 0xfec0                                      // 000000002D84: B002FEC0
	s_mov_b32 s3, -1                                           // 000000002D88: BE8300C1
	s_movk_i32 s8, 0xfc60                                      // 000000002D8C: B008FC60
	s_wait_alu 0xfffe                                          // 000000002D90: BF88FFFE
	s_add_nc_u64 s[2:3], s[34:35], s[2:3]                      // 000000002D94: A9820222
	v_writelane_b32 v29, s45, 5                                // 000000002D98: D761001D 00010A2D
	s_mov_b32 s9, -1                                           // 000000002DA0: BE8900C1
	s_wait_alu 0xfffe                                          // 000000002DA4: BF88FFFE
	s_add_nc_u64 s[8:9], s[34:35], s[8:9]                      // 000000002DA8: A9880822
	v_writelane_b32 v29, s46, 6                                // 000000002DAC: D761001D 00010C2E
	v_writelane_b32 v29, s47, 7                                // 000000002DB4: D761001D 00010E2F
	v_writelane_b32 v29, s48, 8                                // 000000002DBC: D761001D 00011030
	v_writelane_b32 v29, s49, 9                                // 000000002DC4: D761001D 00011231
	v_writelane_b32 v29, s50, 10                               // 000000002DCC: D761001D 00011432
	v_writelane_b32 v29, s51, 11                               // 000000002DD4: D761001D 00011633
	v_writelane_b32 v29, s52, 12                               // 000000002DDC: D761001D 00011834
	v_writelane_b32 v29, s53, 13                               // 000000002DE4: D761001D 00011A35
	v_writelane_b32 v29, s54, 14                               // 000000002DEC: D761001D 00011C36
	v_writelane_b32 v29, s55, 15                               // 000000002DF4: D761001D 00011E37
	v_writelane_b32 v29, s56, 16                               // 000000002DFC: D761001D 00012038
	v_writelane_b32 v29, s57, 17                               // 000000002E04: D761001D 00012239
	v_writelane_b32 v29, s58, 18                               // 000000002E0C: D761001D 0001243A
	v_writelane_b32 v29, s59, 19                               // 000000002E14: D761001D 0001263B
	s_load_b256 s[44:51], s[34:35], 0x1180                     // 000000002E1C: F4006B11 F8001180
	s_wait_kmcnt 0x0                                           // 000000002E24: BFC70000
	v_writelane_b32 v21, s44, 12                               // 000000002E28: D7610015 0001182C
	v_writelane_b32 v21, s45, 13                               // 000000002E30: D7610015 00011A2D
	v_writelane_b32 v21, s46, 14                               // 000000002E38: D7610015 00011C2E
	v_writelane_b32 v21, s47, 15                               // 000000002E40: D7610015 00011E2F
	v_writelane_b32 v21, s48, 16                               // 000000002E48: D7610015 00012030
	v_writelane_b32 v21, s49, 17                               // 000000002E50: D7610015 00012231
	v_writelane_b32 v21, s50, 18                               // 000000002E58: D7610015 00012432
	v_writelane_b32 v21, s51, 19                               // 000000002E60: D7610015 00012633
	s_load_b512 s[44:59], s[2:3], 0x0                          // 000000002E68: F4008B01 F8000000
	s_wait_kmcnt 0x0                                           // 000000002E70: BFC70000
	v_writelane_b32 v29, s44, 28                               // 000000002E74: D761001D 0001382C
	v_writelane_b32 v21, s48, 0                                // 000000002E7C: D7610015 00010030
	v_writelane_b32 v29, s45, 29                               // 000000002E84: D761001D 00013A2D
	v_writelane_b32 v21, s49, 1                                // 000000002E8C: D7610015 00010231
	v_writelane_b32 v29, s46, 30                               // 000000002E94: D761001D 00013C2E
	v_writelane_b32 v21, s50, 2                                // 000000002E9C: D7610015 00010432
	v_writelane_b32 v29, s47, 31                               // 000000002EA4: D761001D 00013E2F
	v_writelane_b32 v21, s51, 3                                // 000000002EAC: D7610015 00010633
	v_writelane_b32 v21, s52, 4                                // 000000002EB4: D7610015 00010834
	v_writelane_b32 v21, s53, 5                                // 000000002EBC: D7610015 00010A35
	v_writelane_b32 v21, s54, 6                                // 000000002EC4: D7610015 00010C36
	v_writelane_b32 v21, s55, 7                                // 000000002ECC: D7610015 00010E37
	v_writelane_b32 v21, s56, 8                                // 000000002ED4: D7610015 00011038
	v_writelane_b32 v21, s57, 9                                // 000000002EDC: D7610015 00011239
	v_writelane_b32 v21, s58, 10                               // 000000002EE4: D7610015 0001143A
	v_writelane_b32 v21, s59, 11                               // 000000002EEC: D7610015 0001163B
	s_load_b256 s[44:51], s[8:9], 0x0                          // 000000002EF4: F4006B04 F8000000
	s_wait_kmcnt 0x0                                           // 000000002EFC: BFC70000
	v_writelane_b32 v29, s44, 20                               // 000000002F00: D761001D 0001282C
	v_writelane_b32 v29, s45, 21                               // 000000002F08: D761001D 00012A2D
	v_writelane_b32 v29, s46, 22                               // 000000002F10: D761001D 00012C2E
	v_writelane_b32 v29, s47, 23                               // 000000002F18: D761001D 00012E2F
	v_writelane_b32 v29, s48, 24                               // 000000002F20: D761001D 00013030
	v_writelane_b32 v29, s49, 25                               // 000000002F28: D761001D 00013231
	v_writelane_b32 v29, s50, 26                               // 000000002F30: D761001D 00013432
	v_writelane_b32 v29, s51, 27                               // 000000002F38: D761001D 00013633
	s_or_saveexec_b32 s105, -1                                 // 000000002F40: BEE922C1
	scratch_store_b32 off, v29, off offset:16                  // 000000002F44: ED06807C 0E800000 00001000
	s_wait_alu 0xfffe                                          // 000000002F50: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002F54: BEFE0069
	s_add_nc_u64 s[2:3], s[34:35], 0x80                        // 000000002F58: A982FF22 00000080
	s_mov_b32 s12, 0                                           // 000000002F60: BE8C0080
	s_mov_b32 s8, s104                                         // 000000002F64: BE880068
	s_branch 228                                               // 000000002F68: BFA000E4 <r_3_3_3_8_8_8+0x1cfc>
	s_or_saveexec_b32 s105, -1                                 // 000000002F6C: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000002F70: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000002F7C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002F80: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000002F84: BFC00000
	v_readlane_b32 s2, v29, 14                                 // 000000002F88: D7600002 00011D1D
	v_readlane_b32 s3, v29, 15                                 // 000000002F90: D7600003 00011F1D
	s_or_saveexec_b32 s105, -1                                 // 000000002F98: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000002F9C: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 000000002FA8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000002FAC: BEFE0069
	s_load_b256 s[44:51], s[6:7], 0x0                          // 000000002FB0: F4006B03 F8000000
	s_wait_loadcnt 0x0                                         // 000000002FB8: BFC00000
	v_readlane_b32 s8, v29, 29                                 // 000000002FBC: D7600008 00013B1D
	v_readlane_b32 s9, v29, 30                                 // 000000002FC4: D7600009 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002FCC: BF870001
	s_add_nc_u64 s[2:3], s[2:3], s[8:9]                        // 000000002FD0: A9820802
	s_wait_kmcnt 0x0                                           // 000000002FD4: BFC70000
	v_writelane_b32 v21, s44, 20                               // 000000002FD8: D7610015 0001282C
	v_writelane_b32 v21, s45, 21                               // 000000002FE0: D7610015 00012A2D
	v_writelane_b32 v21, s46, 22                               // 000000002FE8: D7610015 00012C2E
	v_writelane_b32 v21, s47, 23                               // 000000002FF0: D7610015 00012E2F
	v_writelane_b32 v21, s48, 24                               // 000000002FF8: D7610015 00013030
	v_writelane_b32 v21, s49, 25                               // 000000003000: D7610015 00013231
	v_writelane_b32 v21, s50, 26                               // 000000003008: D7610015 00013432
	v_writelane_b32 v21, s51, 27                               // 000000003010: D7610015 00013633
	s_load_b256 s[44:51], s[2:3], 0x0                          // 000000003018: F4006B01 F8000000
	s_or_saveexec_b32 s105, -1                                 // 000000003020: BEE922C1
	s_wait_alu 0xfffe                                          // 000000003024: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003028: BEFE0069
	s_wait_kmcnt 0x0                                           // 00000000302C: BFC70000
	v_writelane_b32 v26, s44, 4                                // 000000003030: D761001A 0001082C
	s_mov_b32 s42, s40                                         // 000000003038: BEAA0028
	s_mov_b32 s43, s40                                         // 00000000303C: BEAB0028
	s_mov_b32 s41, s40                                         // 000000003040: BEA90028
	s_mov_b32 s8, -1                                           // 000000003044: BE8800C1
	v_writelane_b32 v26, s45, 5                                // 000000003048: D761001A 00010A2D
	s_wait_alu 0xfffe                                          // 000000003050: BF88FFFE
	s_mov_b64 s[66:67], s[42:43]                               // 000000003054: BEC2012A
	s_mov_b64 s[102:103], s[42:43]                             // 000000003058: BEE6012A
	s_mov_b64 s[82:83], s[42:43]                               // 00000000305C: BED2012A
	s_mov_b64 s[58:59], s[42:43]                               // 000000003060: BEBA012A
	v_writelane_b32 v26, s46, 6                                // 000000003064: D761001A 00010C2E
	s_mov_b64 s[54:55], s[42:43]                               // 00000000306C: BEB6012A
	s_mov_b64 s[64:65], s[40:41]                               // 000000003070: BEC00128
	s_mov_b64 s[100:101], s[40:41]                             // 000000003074: BEE40128
	s_mov_b64 s[80:81], s[40:41]                               // 000000003078: BED00128
	v_writelane_b32 v26, s47, 7                                // 00000000307C: D761001A 00010E2F
	s_mov_b64 s[56:57], s[40:41]                               // 000000003084: BEB80128
	s_mov_b64 s[52:53], s[40:41]                               // 000000003088: BEB40128
	v_writelane_b32 v26, s48, 8                                // 00000000308C: D761001A 00011030
	v_writelane_b32 v26, s49, 9                                // 000000003094: D761001A 00011231
	v_writelane_b32 v26, s50, 10                               // 00000000309C: D761001A 00011432
	v_writelane_b32 v26, s51, 11                               // 0000000030A4: D761001A 00011633
	s_load_b256 s[44:51], s[34:35], 0x1180                     // 0000000030AC: F4006B11 F8001180
	s_wait_kmcnt 0x0                                           // 0000000030B4: BFC70000
	v_writelane_b32 v21, s44, 12                               // 0000000030B8: D7610015 0001182C
	v_writelane_b32 v21, s45, 13                               // 0000000030C0: D7610015 00011A2D
	v_writelane_b32 v21, s46, 14                               // 0000000030C8: D7610015 00011C2E
	v_writelane_b32 v21, s47, 15                               // 0000000030D0: D7610015 00011E2F
	v_writelane_b32 v21, s48, 16                               // 0000000030D8: D7610015 00012030
	v_writelane_b32 v21, s49, 17                               // 0000000030E0: D7610015 00012231
	v_writelane_b32 v21, s50, 18                               // 0000000030E8: D7610015 00012432
	v_writelane_b32 v21, s51, 19                               // 0000000030F0: D7610015 00012633
	s_mov_b64 s[46:47], s[42:43]                               // 0000000030F8: BEAE012A
	s_mov_b64 s[50:51], s[42:43]                               // 0000000030FC: BEB2012A
	s_mov_b64 s[44:45], s[40:41]                               // 000000003100: BEAC0128
	s_mov_b64 s[48:49], s[40:41]                               // 000000003104: BEB00128
	s_wait_alu 0xfffe                                          // 000000003108: BF88FFFE
	v_writelane_b32 v21, s44, 28                               // 00000000310C: D7610015 0001382C
	v_writelane_b32 v26, s48, 0                                // 000000003114: D761001A 00010030
	v_writelane_b32 v21, s45, 29                               // 00000000311C: D7610015 00013A2D
	v_writelane_b32 v26, s49, 1                                // 000000003124: D761001A 00010231
	v_writelane_b32 v21, s46, 30                               // 00000000312C: D7610015 00013C2E
	v_writelane_b32 v26, s50, 2                                // 000000003134: D761001A 00010432
	v_writelane_b32 v21, s47, 31                               // 00000000313C: D7610015 00013E2F
	v_writelane_b32 v26, s51, 3                                // 000000003144: D761001A 00010633
	s_or_saveexec_b32 s105, -1                                 // 00000000314C: BEE922C1
	s_wait_alu 0xfffe                                          // 000000003150: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003154: BEFE0069
	s_mov_b64 s[74:75], s[42:43]                               // 000000003158: BECA012A
	s_mov_b64 s[78:79], s[42:43]                               // 00000000315C: BECE012A
	s_mov_b64 s[98:99], s[42:43]                               // 000000003160: BEE2012A
	s_mov_b64 s[72:73], s[40:41]                               // 000000003164: BEC80128
	s_mov_b64 s[76:77], s[40:41]                               // 000000003168: BECC0128
	s_mov_b64 s[96:97], s[40:41]                               // 00000000316C: BEE00128
	s_or_saveexec_b32 s105, -1                                 // 000000003170: BEE922C1
	scratch_load_b32 v29, off, off offset:16 th:TH_LOAD_LU     // 000000003174: ED05007C 0030001D 00001000
	s_wait_alu 0xfffe                                          // 000000003180: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003184: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000003188: BFC00000
	v_writelane_b32 v29, s96, 20                               // 00000000318C: D761001D 00012860
	s_mov_b64 s[62:63], s[42:43]                               // 000000003194: BEBE012A
	s_mov_b64 s[60:61], s[40:41]                               // 000000003198: BEBC0128
	s_mov_b64 s[86:87], s[42:43]                               // 00000000319C: BED6012A
	s_mov_b64 s[84:85], s[40:41]                               // 0000000031A0: BED40128
	v_writelane_b32 v29, s97, 21                               // 0000000031A4: D761001D 00012A61
	v_writelane_b32 v29, s98, 22                               // 0000000031AC: D761001D 00012C62
	v_writelane_b32 v29, s99, 23                               // 0000000031B4: D761001D 00012E63
	v_writelane_b32 v29, s100, 24                              // 0000000031BC: D761001D 00013064
	v_writelane_b32 v29, s101, 25                              // 0000000031C4: D761001D 00013265
	v_writelane_b32 v29, s102, 26                              // 0000000031CC: D761001D 00013466
	v_writelane_b32 v29, s103, 27                              // 0000000031D4: D761001D 00013667
	v_writelane_b32 v29, s52, 4                                // 0000000031DC: D761001D 00010834
	v_writelane_b32 v29, s53, 5                                // 0000000031E4: D761001D 00010A35
	v_writelane_b32 v29, s54, 6                                // 0000000031EC: D761001D 00010C36
	v_writelane_b32 v29, s55, 7                                // 0000000031F4: D761001D 00010E37
	v_writelane_b32 v29, s56, 8                                // 0000000031FC: D761001D 00011038
	v_writelane_b32 v29, s57, 9                                // 000000003204: D761001D 00011239
	v_writelane_b32 v29, s58, 10                               // 00000000320C: D761001D 0001143A
	v_writelane_b32 v29, s59, 11                               // 000000003214: D761001D 0001163B
	s_wait_alu 0xfffe                                          // 00000000321C: BF88FFFE
	v_writelane_b32 v29, s60, 12                               // 000000003220: D761001D 0001183C
	v_writelane_b32 v29, s61, 13                               // 000000003228: D761001D 00011A3D
	v_writelane_b32 v29, s62, 14                               // 000000003230: D761001D 00011C3E
	v_writelane_b32 v29, s63, 15                               // 000000003238: D761001D 00011E3F
	v_writelane_b32 v29, s64, 16                               // 000000003240: D761001D 00012040
	v_writelane_b32 v29, s65, 17                               // 000000003248: D761001D 00012241
	v_writelane_b32 v29, s66, 18                               // 000000003250: D761001D 00012442
	v_writelane_b32 v29, s67, 19                               // 000000003258: D761001D 00012643
	v_writelane_b32 v29, s72, 28                               // 000000003260: D761001D 00013848
	v_writelane_b32 v29, s73, 29                               // 000000003268: D761001D 00013A49
	v_writelane_b32 v29, s74, 30                               // 000000003270: D761001D 00013C4A
	v_writelane_b32 v29, s75, 31                               // 000000003278: D761001D 00013E4B
	s_or_saveexec_b32 s105, -1                                 // 000000003280: BEE922C1
	scratch_store_b32 off, v29, off offset:16                  // 000000003284: ED06807C 0E800000 00001000
	s_wait_alu 0xfffe                                          // 000000003290: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003294: BEFE0069
	v_writelane_b32 v21, s76, 0                                // 000000003298: D7610015 0001004C
	s_mov_b64 s[2:3], s[4:5]                                   // 0000000032A0: BE820104
	v_writelane_b32 v21, s77, 1                                // 0000000032A4: D7610015 0001024D
	v_writelane_b32 v21, s78, 2                                // 0000000032AC: D7610015 0001044E
	v_writelane_b32 v21, s79, 3                                // 0000000032B4: D7610015 0001064F
	v_writelane_b32 v21, s80, 4                                // 0000000032BC: D7610015 00010850
	v_writelane_b32 v21, s81, 5                                // 0000000032C4: D7610015 00010A51
	v_writelane_b32 v21, s82, 6                                // 0000000032CC: D7610015 00010C52
	v_writelane_b32 v21, s83, 7                                // 0000000032D4: D7610015 00010E53
	v_writelane_b32 v21, s84, 8                                // 0000000032DC: D7610015 00011054
	v_writelane_b32 v21, s85, 9                                // 0000000032E4: D7610015 00011255
	v_writelane_b32 v21, s86, 10                               // 0000000032EC: D7610015 00011456
	v_writelane_b32 v21, s87, 11                               // 0000000032F4: D7610015 00011657
	s_load_b256 s[44:51], s[34:35], 0x1120                     // 0000000032FC: F4006B11 F8001120
	s_or_saveexec_b32 s105, -1                                 // 000000003304: BEE922C1
	s_wait_alu 0xfffe                                          // 000000003308: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000330C: BEFE0069
	s_wait_kmcnt 0x0                                           // 000000003310: BFC70000
	v_writelane_b32 v26, s44, 12                               // 000000003314: D761001A 0001182C
	v_writelane_b32 v26, s45, 13                               // 00000000331C: D761001A 00011A2D
	v_writelane_b32 v26, s46, 14                               // 000000003324: D761001A 00011C2E
	v_writelane_b32 v26, s47, 15                               // 00000000332C: D761001A 00011E2F
	v_writelane_b32 v26, s48, 16                               // 000000003334: D761001A 00012030
	v_writelane_b32 v26, s49, 17                               // 00000000333C: D761001A 00012231
	v_writelane_b32 v26, s50, 18                               // 000000003344: D761001A 00012432
	v_writelane_b32 v26, s51, 19                               // 00000000334C: D761001A 00012633
	s_load_b512 s[44:59], s[34:35], 0x1240                     // 000000003354: F4008B11 F8001240
	s_wait_kmcnt 0x0                                           // 00000000335C: BFC70000
	v_writelane_b32 v26, s44, 20                               // 000000003360: D761001A 0001282C
	v_writelane_b32 v26, s45, 21                               // 000000003368: D761001A 00012A2D
	v_writelane_b32 v26, s46, 22                               // 000000003370: D761001A 00012C2E
	v_writelane_b32 v26, s47, 23                               // 000000003378: D761001A 00012E2F
	v_writelane_b32 v26, s48, 24                               // 000000003380: D761001A 00013030
	v_writelane_b32 v26, s49, 25                               // 000000003388: D761001A 00013231
	v_writelane_b32 v26, s50, 26                               // 000000003390: D761001A 00013432
	v_writelane_b32 v26, s51, 27                               // 000000003398: D761001A 00013633
	v_writelane_b32 v26, s52, 28                               // 0000000033A0: D761001A 00013834
	v_writelane_b32 v26, s53, 29                               // 0000000033A8: D761001A 00013A35
	v_writelane_b32 v26, s54, 30                               // 0000000033B0: D761001A 00013C36
	v_writelane_b32 v26, s55, 31                               // 0000000033B8: D761001A 00013E37
	s_or_saveexec_b32 s105, -1                                 // 0000000033C0: BEE922C1
	scratch_store_b32 off, v26, off offset:60                  // 0000000033C4: ED06807C 0D000000 00003C00
	s_wait_alu 0xfffe                                          // 0000000033D0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000033D4: BEFE0069
	v_writelane_b32 v28, s56, 0                                // 0000000033D8: D761001C 00010038
	s_mov_b32 s72, s12                                         // 0000000033E0: BEC8000C
	s_mov_b32 s80, s12                                         // 0000000033E4: BED0000C
	s_mov_b32 s81, s12                                         // 0000000033E8: BED1000C
	s_mov_b32 s82, s12                                         // 0000000033EC: BED2000C
	v_writelane_b32 v28, s57, 1                                // 0000000033F0: D761001C 00010239
	s_mov_b32 s83, s12                                         // 0000000033F8: BED3000C
	s_mov_b32 s73, s12                                         // 0000000033FC: BEC9000C
	s_mov_b32 s74, s12                                         // 000000003400: BECA000C
	s_mov_b32 s75, s12                                         // 000000003404: BECB000C
	v_writelane_b32 v28, s58, 2                                // 000000003408: D761001C 0001043A
	s_mov_b32 s76, s12                                         // 000000003410: BECC000C
	s_mov_b32 s77, s12                                         // 000000003414: BECD000C
	s_mov_b32 s78, s12                                         // 000000003418: BECE000C
	s_mov_b32 s79, s12                                         // 00000000341C: BECF000C
	v_writelane_b32 v28, s59, 3                                // 000000003420: D761001C 0001063B
	s_load_b256 s[44:51], s[0:1], 0x1120                       // 000000003428: F4006B00 F8001120
	s_mov_b32 s84, s12                                         // 000000003430: BED4000C
	s_mov_b32 s85, s12                                         // 000000003434: BED5000C
	s_mov_b32 s86, s12                                         // 000000003438: BED6000C
	s_mov_b32 s87, s12                                         // 00000000343C: BED7000C
	s_mov_b32 s60, s12                                         // 000000003440: BEBC000C
	s_and_not1_b32 vcc_lo, exec_lo, s8                         // 000000003444: 916A087E
	s_mov_b32 s61, s12                                         // 000000003448: BEBD000C
	s_mov_b32 s62, s12                                         // 00000000344C: BEBE000C
	s_mov_b32 s63, s12                                         // 000000003450: BEBF000C
	s_mov_b32 s52, s12                                         // 000000003454: BEB4000C
	s_mov_b32 s53, s12                                         // 000000003458: BEB5000C
	s_mov_b32 s54, s12                                         // 00000000345C: BEB6000C
	s_mov_b32 s55, s12                                         // 000000003460: BEB7000C
	s_mov_b32 s56, s12                                         // 000000003464: BEB8000C
	s_mov_b32 s57, s12                                         // 000000003468: BEB9000C
	s_mov_b32 s58, s12                                         // 00000000346C: BEBA000C
	s_mov_b32 s59, s12                                         // 000000003470: BEBB000C
	s_wait_kmcnt 0x0                                           // 000000003474: BFC70000
	v_writelane_b32 v28, s44, 4                                // 000000003478: D761001C 0001082C
	v_writelane_b32 v28, s45, 5                                // 000000003480: D761001C 00010A2D
	v_writelane_b32 v28, s46, 6                                // 000000003488: D761001C 00010C2E
	v_writelane_b32 v28, s47, 7                                // 000000003490: D761001C 00010E2F
	v_writelane_b32 v28, s48, 8                                // 000000003498: D761001C 00011030
	v_writelane_b32 v28, s49, 9                                // 0000000034A0: D761001C 00011231
	v_writelane_b32 v28, s50, 10                               // 0000000034A8: D761001C 00011432
	v_writelane_b32 v28, s51, 11                               // 0000000034B0: D761001C 00011633
	s_load_b256 s[44:51], s[0:1], 0xfe0                        // 0000000034B8: F4006B00 F8000FE0
	s_wait_kmcnt 0x0                                           // 0000000034C0: BFC70000
	v_writelane_b32 v28, s44, 12                               // 0000000034C4: D761001C 0001182C
	v_writelane_b32 v28, s45, 13                               // 0000000034CC: D761001C 00011A2D
	v_writelane_b32 v28, s46, 14                               // 0000000034D4: D761001C 00011C2E
	v_writelane_b32 v28, s47, 15                               // 0000000034DC: D761001C 00011E2F
	v_writelane_b32 v28, s48, 16                               // 0000000034E4: D761001C 00012030
	v_writelane_b32 v28, s49, 17                               // 0000000034EC: D761001C 00012231
	v_writelane_b32 v28, s50, 18                               // 0000000034F4: D761001C 00012432
	v_writelane_b32 v28, s51, 19                               // 0000000034FC: D761001C 00012633
	s_mov_b32 s48, s12                                         // 000000003504: BEB0000C
	s_mov_b32 s49, s12                                         // 000000003508: BEB1000C
	s_mov_b32 s50, s12                                         // 00000000350C: BEB2000C
	s_mov_b32 s51, s12                                         // 000000003510: BEB3000C
	v_writelane_b32 v28, s72, 20                               // 000000003514: D761001C 00012848
	s_mov_b32 s44, s12                                         // 00000000351C: BEAC000C
	s_mov_b32 s45, s12                                         // 000000003520: BEAD000C
	s_mov_b32 s46, s12                                         // 000000003524: BEAE000C
	s_mov_b32 s47, s12                                         // 000000003528: BEAF000C
	v_writelane_b32 v28, s73, 21                               // 00000000352C: D761001C 00012A49
	v_writelane_b32 v28, s74, 22                               // 000000003534: D761001C 00012C4A
	v_writelane_b32 v28, s75, 23                               // 00000000353C: D761001C 00012E4B
	v_writelane_b32 v28, s76, 24                               // 000000003544: D761001C 0001304C
	v_writelane_b32 v28, s77, 25                               // 00000000354C: D761001C 0001324D
	v_writelane_b32 v28, s78, 26                               // 000000003554: D761001C 0001344E
	v_writelane_b32 v28, s79, 27                               // 00000000355C: D761001C 0001364F
	v_writelane_b32 v28, s80, 28                               // 000000003564: D761001C 00013850
	v_writelane_b32 v28, s81, 29                               // 00000000356C: D761001C 00013A51
	v_writelane_b32 v28, s82, 30                               // 000000003574: D761001C 00013C52
	v_writelane_b32 v28, s83, 31                               // 00000000357C: D761001C 00013E53
	s_or_saveexec_b32 s105, -1                                 // 000000003584: BEE922C1
	scratch_store_b32 off, v28, off offset:20                  // 000000003588: ED06807C 0E000000 00001400
	s_wait_alu 0xfffe                                          // 000000003594: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003598: BEFE0069
	v_writelane_b32 v29, s84, 0                                // 00000000359C: D761001D 00010054
	s_mov_b32 s64, s12                                         // 0000000035A4: BEC0000C
	s_mov_b32 s65, s12                                         // 0000000035A8: BEC1000C
	s_mov_b32 s66, s12                                         // 0000000035AC: BEC2000C
	v_writelane_b32 v29, s85, 1                                // 0000000035B0: D761001D 00010255
	v_writelane_b32 v29, s86, 2                                // 0000000035B8: D761001D 00010456
	v_writelane_b32 v29, s87, 3                                // 0000000035C0: D761001D 00010657
	v_writelane_b32 v29, s44, 4                                // 0000000035C8: D761001D 0001082C
	s_mov_b32 s67, s48                                         // 0000000035D0: BEC30030
	v_writelane_b32 v29, s45, 5                                // 0000000035D4: D761001D 00010A2D
	v_writelane_b32 v29, s46, 6                                // 0000000035DC: D761001D 00010C2E
	v_writelane_b32 v29, s47, 7                                // 0000000035E4: D761001D 00010E2F
	v_writelane_b32 v29, s48, 8                                // 0000000035EC: D761001D 00011030
	v_writelane_b32 v29, s49, 9                                // 0000000035F4: D761001D 00011231
	v_writelane_b32 v29, s50, 10                               // 0000000035FC: D761001D 00011432
	v_writelane_b32 v29, s51, 11                               // 000000003604: D761001D 00011633
	v_writelane_b32 v29, s52, 12                               // 00000000360C: D761001D 00011834
	v_writelane_b32 v29, s53, 13                               // 000000003614: D761001D 00011A35
	v_writelane_b32 v29, s54, 14                               // 00000000361C: D761001D 00011C36
	v_writelane_b32 v29, s55, 15                               // 000000003624: D761001D 00011E37
	v_writelane_b32 v29, s56, 16                               // 00000000362C: D761001D 00012038
	v_writelane_b32 v29, s57, 17                               // 000000003634: D761001D 00012239
	v_writelane_b32 v29, s58, 18                               // 00000000363C: D761001D 0001243A
	v_writelane_b32 v29, s59, 19                               // 000000003644: D761001D 0001263B
	v_writelane_b32 v29, s60, 20                               // 00000000364C: D761001D 0001283C
	v_writelane_b32 v29, s61, 21                               // 000000003654: D761001D 00012A3D
	v_writelane_b32 v29, s62, 22                               // 00000000365C: D761001D 00012C3E
	v_writelane_b32 v29, s63, 23                               // 000000003664: D761001D 00012E3F
	s_wait_alu 0xfffe                                          // 00000000366C: BF88FFFE
	v_writelane_b32 v29, s64, 24                               // 000000003670: D761001D 00013040
	v_writelane_b32 v29, s65, 25                               // 000000003678: D761001D 00013241
	v_writelane_b32 v29, s66, 26                               // 000000003680: D761001D 00013442
	v_writelane_b32 v29, s67, 27                               // 000000003688: D761001D 00013643
	s_load_b512 s[44:59], s[0:1], 0x1240                       // 000000003690: F4008B00 F8001240
	s_wait_kmcnt 0x0                                           // 000000003698: BFC70000
	v_writelane_b32 v29, s44, 28                               // 00000000369C: D761001D 0001382C
	v_writelane_b32 v28, s48, 0                                // 0000000036A4: D761001C 00010030
	v_writelane_b32 v29, s45, 29                               // 0000000036AC: D761001D 00013A2D
	v_writelane_b32 v28, s49, 1                                // 0000000036B4: D761001C 00010231
	v_writelane_b32 v29, s46, 30                               // 0000000036BC: D761001D 00013C2E
	v_writelane_b32 v28, s50, 2                                // 0000000036C4: D761001C 00010432
	v_writelane_b32 v29, s47, 31                               // 0000000036CC: D761001D 00013E2F
	v_writelane_b32 v28, s51, 3                                // 0000000036D4: D761001C 00010633
	v_writelane_b32 v28, s52, 4                                // 0000000036DC: D761001C 00010834
	v_writelane_b32 v28, s53, 5                                // 0000000036E4: D761001C 00010A35
	v_writelane_b32 v28, s54, 6                                // 0000000036EC: D761001C 00010C36
	v_writelane_b32 v28, s55, 7                                // 0000000036F4: D761001C 00010E37
	v_writelane_b32 v28, s56, 8                                // 0000000036FC: D761001C 00011038
	v_writelane_b32 v28, s57, 9                                // 000000003704: D761001C 00011239
	v_writelane_b32 v28, s58, 10                               // 00000000370C: D761001C 0001143A
	v_writelane_b32 v28, s59, 11                               // 000000003714: D761001C 0001163B
	s_or_saveexec_b32 s105, -1                                 // 00000000371C: BEE922C1
	s_wait_alu 0xfffe                                          // 000000003720: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003724: BEFE0069
	s_cbranch_vccnz 102                                        // 000000003728: BFA40066 <r_3_3_3_8_8_8+0x22c4>
	s_load_b512 s[44:59], s[2:3], 0x2500                       // 00000000372C: F4008B01 F8002500
	s_or_saveexec_b32 s105, -1                                 // 000000003734: BEE922C1
	scratch_load_b32 v26, off, off offset:20 th:TH_LOAD_LU     // 000000003738: ED05007C 0030001A 00001400
	s_wait_alu 0xfffe                                          // 000000003744: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003748: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000374C: BFC00000
	s_wait_kmcnt 0x0                                           // 000000003750: BFC70000
	v_writelane_b32 v26, s44, 20                               // 000000003754: D761001A 0001282C
	v_writelane_b32 v26, s45, 21                               // 00000000375C: D761001A 00012A2D
	v_writelane_b32 v26, s46, 22                               // 000000003764: D761001A 00012C2E
	v_writelane_b32 v26, s47, 23                               // 00000000376C: D761001A 00012E2F
	v_writelane_b32 v26, s48, 24                               // 000000003774: D761001A 00013030
	v_writelane_b32 v26, s49, 25                               // 00000000377C: D761001A 00013231
	v_writelane_b32 v26, s50, 26                               // 000000003784: D761001A 00013432
	v_writelane_b32 v26, s51, 27                               // 00000000378C: D761001A 00013633
	v_writelane_b32 v26, s52, 28                               // 000000003794: D761001A 00013834
	v_writelane_b32 v26, s53, 29                               // 00000000379C: D761001A 00013A35
	v_writelane_b32 v26, s54, 30                               // 0000000037A4: D761001A 00013C36
	v_writelane_b32 v26, s55, 31                               // 0000000037AC: D761001A 00013E37
	s_or_saveexec_b32 s105, -1                                 // 0000000037B4: BEE922C1
	scratch_store_b32 off, v26, off offset:20                  // 0000000037B8: ED06807C 0D000000 00001400
	s_wait_alu 0xfffe                                          // 0000000037C4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000037C8: BEFE0069
	v_writelane_b32 v29, s56, 0                                // 0000000037CC: D761001D 00010038
	v_writelane_b32 v29, s57, 1                                // 0000000037D4: D761001D 00010239
	v_writelane_b32 v29, s58, 2                                // 0000000037DC: D761001D 0001043A
	v_writelane_b32 v29, s59, 3                                // 0000000037E4: D761001D 0001063B
	s_load_b256 s[44:51], s[2:3], 0x23e0                       // 0000000037EC: F4006B01 F80023E0
	s_wait_kmcnt 0x0                                           // 0000000037F4: BFC70000
	v_writelane_b32 v29, s44, 4                                // 0000000037F8: D761001D 0001082C
	v_writelane_b32 v29, s45, 5                                // 000000003800: D761001D 00010A2D
	v_writelane_b32 v29, s46, 6                                // 000000003808: D761001D 00010C2E
	v_writelane_b32 v29, s47, 7                                // 000000003810: D761001D 00010E2F
	v_writelane_b32 v29, s48, 8                                // 000000003818: D761001D 00011030
	v_writelane_b32 v29, s49, 9                                // 000000003820: D761001D 00011231
	v_writelane_b32 v29, s50, 10                               // 000000003828: D761001D 00011432
	v_writelane_b32 v29, s51, 11                               // 000000003830: D761001D 00011633
	s_load_b512 s[44:59], s[2:3], 0x2640                       // 000000003838: F4008B01 F8002640
	s_wait_kmcnt 0x0                                           // 000000003840: BFC70000
	v_writelane_b32 v29, s44, 12                               // 000000003844: D761001D 0001182C
	v_writelane_b32 v29, s45, 13                               // 00000000384C: D761001D 00011A2D
	v_writelane_b32 v29, s46, 14                               // 000000003854: D761001D 00011C2E
	v_writelane_b32 v29, s47, 15                               // 00000000385C: D761001D 00011E2F
	v_writelane_b32 v29, s48, 16                               // 000000003864: D761001D 00012030
	v_writelane_b32 v29, s49, 17                               // 00000000386C: D761001D 00012231
	v_writelane_b32 v29, s50, 18                               // 000000003874: D761001D 00012432
	v_writelane_b32 v29, s51, 19                               // 00000000387C: D761001D 00012633
	v_writelane_b32 v29, s52, 20                               // 000000003884: D761001D 00012834
	v_writelane_b32 v29, s53, 21                               // 00000000388C: D761001D 00012A35
	v_writelane_b32 v29, s54, 22                               // 000000003894: D761001D 00012C36
	v_writelane_b32 v29, s55, 23                               // 00000000389C: D761001D 00012E37
	v_writelane_b32 v29, s56, 24                               // 0000000038A4: D761001D 00013038
	v_writelane_b32 v29, s57, 25                               // 0000000038AC: D761001D 00013239
	v_writelane_b32 v29, s58, 26                               // 0000000038B4: D761001D 0001343A
	v_writelane_b32 v29, s59, 27                               // 0000000038BC: D761001D 0001363B
	s_or_saveexec_b32 s105, -1                                 // 0000000038C4: BEE922C1
	scratch_store_b32 off, v29, off offset:64                  // 0000000038C8: ED06807C 0E800000 00004000
	s_wait_alu 0xfffe                                          // 0000000038D4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000038D8: BEFE0069
	s_load_b256 s[44:51], s[2:3], 0x1120                       // 0000000038DC: F4006B01 F8001120
	s_or_saveexec_b32 s105, -1                                 // 0000000038E4: BEE922C1
	s_wait_alu 0xfffe                                          // 0000000038E8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000038EC: BEFE0069
	s_wait_kmcnt 0x0                                           // 0000000038F0: BFC70000
	v_writelane_b32 v28, s44, 12                               // 0000000038F4: D761001C 0001182C
	v_writelane_b32 v28, s45, 13                               // 0000000038FC: D761001C 00011A2D
	v_writelane_b32 v28, s46, 14                               // 000000003904: D761001C 00011C2E
	v_writelane_b32 v28, s47, 15                               // 00000000390C: D761001C 00011E2F
	v_writelane_b32 v28, s48, 16                               // 000000003914: D761001C 00012030
	v_writelane_b32 v28, s49, 17                               // 00000000391C: D761001C 00012231
	v_writelane_b32 v28, s50, 18                               // 000000003924: D761001C 00012432
	v_writelane_b32 v28, s51, 19                               // 00000000392C: D761001C 00012633
	s_load_b256 s[44:51], s[2:3], 0xfe0                        // 000000003934: F4006B01 F8000FE0
	s_wait_kmcnt 0x0                                           // 00000000393C: BFC70000
	v_writelane_b32 v28, s44, 20                               // 000000003940: D761001C 0001282C
	v_writelane_b32 v28, s45, 21                               // 000000003948: D761001C 00012A2D
	v_writelane_b32 v28, s46, 22                               // 000000003950: D761001C 00012C2E
	v_writelane_b32 v28, s47, 23                               // 000000003958: D761001C 00012E2F
	v_writelane_b32 v28, s48, 24                               // 000000003960: D761001C 00013030
	v_writelane_b32 v28, s49, 25                               // 000000003968: D761001C 00013231
	v_writelane_b32 v28, s50, 26                               // 000000003970: D761001C 00013432
	v_writelane_b32 v28, s51, 27                               // 000000003978: D761001C 00013633
	s_load_b512 s[44:59], s[2:3], 0x1240                       // 000000003980: F4008B01 F8001240
	s_wait_kmcnt 0x0                                           // 000000003988: BFC70000
	v_writelane_b32 v28, s44, 28                               // 00000000398C: D761001C 0001382C
	v_writelane_b32 v28, s45, 29                               // 000000003994: D761001C 00013A2D
	v_writelane_b32 v28, s46, 30                               // 00000000399C: D761001C 00013C2E
	v_writelane_b32 v28, s47, 31                               // 0000000039A4: D761001C 00013E2F
	s_or_saveexec_b32 s105, -1                                 // 0000000039AC: BEE922C1
	scratch_store_b32 off, v28, off offset:68                  // 0000000039B0: ED06807C 0E000000 00004400
	s_wait_alu 0xfffe                                          // 0000000039BC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000039C0: BEFE0069
	v_writelane_b32 v26, s48, 0                                // 0000000039C4: D761001A 00010030
	s_and_b32 vcc_lo, exec_lo, s33                             // 0000000039CC: 8B6A217E
	v_writelane_b32 v26, s49, 1                                // 0000000039D0: D761001A 00010231
	v_writelane_b32 v26, s50, 2                                // 0000000039D8: D761001A 00010432
	v_writelane_b32 v26, s51, 3                                // 0000000039E0: D761001A 00010633
	v_writelane_b32 v26, s52, 4                                // 0000000039E8: D761001A 00010834
	v_writelane_b32 v26, s53, 5                                // 0000000039F0: D761001A 00010A35
	v_writelane_b32 v26, s54, 6                                // 0000000039F8: D761001A 00010C36
	v_writelane_b32 v26, s55, 7                                // 000000003A00: D761001A 00010E37
	v_writelane_b32 v26, s56, 8                                // 000000003A08: D761001A 00011038
	v_writelane_b32 v26, s57, 9                                // 000000003A10: D761001A 00011239
	v_writelane_b32 v26, s58, 10                               // 000000003A18: D761001A 0001143A
	v_writelane_b32 v26, s59, 11                               // 000000003A20: D761001A 0001163B
	s_or_saveexec_b32 s105, -1                                 // 000000003A28: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000003A2C: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000003A38: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003A3C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000003A40: BFC00000
	v_readlane_b32 s0, v29, 20                                 // 000000003A44: D7600000 0001291D
	v_readlane_b32 s1, v29, 21                                 // 000000003A4C: D7600001 00012B1D
	s_or_saveexec_b32 s105, -1                                 // 000000003A54: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000003A58: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 000000003A64: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003A68: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000003A6C: BFC00000
	v_readlane_b32 s2, v29, 29                                 // 000000003A70: D7600002 00013B1D
	v_readlane_b32 s3, v29, 30                                 // 000000003A78: D7600003 00013D1D
	s_add_nc_u64 s[36:37], s[34:35], 0xc0                      // 000000003A80: A9A4FF22 000000C0
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003A88: BF870001
	s_add_nc_u64 s[4:5], s[0:1], s[2:3]                        // 000000003A8C: A9840200
	s_cbranch_vccnz 2857                                       // 000000003A90: BFA40B29 <r_3_3_3_8_8_8+0x5138>
	s_or_saveexec_b32 s105, -1                                 // 000000003A94: BEE922C1
	v_mov_b32_e32 v28, v26                                     // 000000003A98: 7E38031A
	s_wait_alu 0xfffe                                          // 000000003A9C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003AA0: BEFE0069
	s_mov_b32 s41, s40                                         // 000000003AA4: BEA90028
	s_mov_b32 s42, s40                                         // 000000003AA8: BEAA0028
	s_mov_b32 s43, s40                                         // 000000003AAC: BEAB0028
	s_wait_alu 0xfffe                                          // 000000003AB0: BF88FFFE
	s_mov_b64 s[12:13], s[40:41]                               // 000000003AB4: BE8C0128
	s_mov_b64 s[8:9], s[40:41]                                 // 000000003AB8: BE880128
	s_mov_b64 s[14:15], s[42:43]                               // 000000003ABC: BE8E012A
	s_mov_b64 s[10:11], s[42:43]                               // 000000003AC0: BE8A012A
	s_wait_alu 0xfffe                                          // 000000003AC4: BF88FFFE
	v_writelane_b32 v24, s8, 12                                // 000000003AC8: D7610018 00011808
	v_readlane_b32 s0, v23, 1                                  // 000000003AD0: D7600000 00010317
	v_writelane_b32 v24, s9, 13                                // 000000003AD8: D7610018 00011A09
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003AE0: BF870002
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000003AE4: 8B6A007E
	v_writelane_b32 v24, s10, 14                               // 000000003AE8: D7610018 00011C0A
	v_writelane_b32 v24, s11, 15                               // 000000003AF0: D7610018 00011E0B
	v_writelane_b32 v24, s12, 16                               // 000000003AF8: D7610018 0001200C
	v_writelane_b32 v24, s13, 17                               // 000000003B00: D7610018 0001220D
	v_writelane_b32 v24, s14, 18                               // 000000003B08: D7610018 0001240E
	v_writelane_b32 v24, s15, 19                               // 000000003B10: D7610018 0001260F
	s_cbranch_vccnz 43                                         // 000000003B18: BFA4002B <r_3_3_3_8_8_8+0x25c8>
	s_or_saveexec_b32 s105, -1                                 // 000000003B1C: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000003B20: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000003B2C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003B30: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000003B34: BFC00000
	v_readlane_b32 s0, v29, 22                                 // 000000003B38: D7600000 00012D1D
	v_readlane_b32 s1, v29, 23                                 // 000000003B40: D7600001 00012F1D
	s_or_saveexec_b32 s105, -1                                 // 000000003B48: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000003B4C: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 000000003B58: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003B5C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000003B60: BFC00000
	v_readlane_b32 s2, v29, 29                                 // 000000003B64: D7600002 00013B1D
	v_readlane_b32 s3, v29, 30                                 // 000000003B6C: D7600003 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003B74: BF870001
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]                        // 000000003B78: A9800200
	s_load_b256 s[8:15], s[0:1], 0x0                           // 000000003B7C: F4006200 F8000000
	s_wait_kmcnt 0x0                                           // 000000003B84: BFC70000
	v_writelane_b32 v24, s8, 12                                // 000000003B88: D7610018 00011808
	v_writelane_b32 v24, s9, 13                                // 000000003B90: D7610018 00011A09
	v_writelane_b32 v24, s10, 14                               // 000000003B98: D7610018 00011C0A
	v_writelane_b32 v24, s11, 15                               // 000000003BA0: D7610018 00011E0B
	v_writelane_b32 v24, s12, 16                               // 000000003BA8: D7610018 0001200C
	v_writelane_b32 v24, s13, 17                               // 000000003BB0: D7610018 0001220D
	v_writelane_b32 v24, s14, 18                               // 000000003BB8: D7610018 0001240E
	v_writelane_b32 v24, s15, 19                               // 000000003BC0: D7610018 0001260F
	s_or_saveexec_b32 s105, -1                                 // 000000003BC8: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000003BCC: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000003BD8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003BDC: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000003BE0: BFC00000
	v_readlane_b32 s0, v29, 18                                 // 000000003BE4: D7600000 0001251D
	v_readlane_b32 s1, v29, 19                                 // 000000003BEC: D7600001 0001271D
	s_or_saveexec_b32 s105, -1                                 // 000000003BF4: BEE922C1
	scratch_load_b32 v26, off, off                             // 000000003BF8: ED05007C 0000001A 00000000
	s_wait_alu 0xfffe                                          // 000000003C04: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003C08: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000003C0C: BFC00000
	v_readlane_b32 s2, v26, 29                                 // 000000003C10: D7600002 00013B1A
	v_readlane_b32 s3, v26, 30                                 // 000000003C18: D7600003 00013D1A
	v_readlane_b32 s6, v29, 24                                 // 000000003C20: D7600006 0001311D
	v_readlane_b32 s7, v29, 25                                 // 000000003C28: D7600007 0001331D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003C30: BF870093
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]                        // 000000003C34: A9800200
	s_add_nc_u64 s[6:7], s[6:7], s[2:3]                        // 000000003C38: A9860206
	s_load_b256 s[44:51], s[0:1], 0x0                          // 000000003C3C: F4006B00 F8000000
	s_wait_kmcnt 0x0                                           // 000000003C44: BFC70000
	v_writelane_b32 v24, s44, 4                                // 000000003C48: D7610018 0001082C
	v_writelane_b32 v24, s45, 5                                // 000000003C50: D7610018 00010A2D
	v_writelane_b32 v24, s46, 6                                // 000000003C58: D7610018 00010C2E
	v_writelane_b32 v24, s47, 7                                // 000000003C60: D7610018 00010E2F
	v_writelane_b32 v24, s48, 8                                // 000000003C68: D7610018 00011030
	v_writelane_b32 v24, s49, 9                                // 000000003C70: D7610018 00011231
	v_writelane_b32 v24, s50, 10                               // 000000003C78: D7610018 00011432
	v_writelane_b32 v24, s51, 11                               // 000000003C80: D7610018 00011633
	s_load_b256 s[44:51], s[4:5], 0x0                          // 000000003C88: F4006B02 F8000000
	s_wait_kmcnt 0x0                                           // 000000003C90: BFC70000
	v_writelane_b32 v26, s44, 28                               // 000000003C94: D761001A 0001382C
	v_writelane_b32 v24, s48, 0                                // 000000003C9C: D7610018 00010030
	v_writelane_b32 v26, s45, 29                               // 000000003CA4: D761001A 00013A2D
	v_writelane_b32 v24, s49, 1                                // 000000003CAC: D7610018 00010231
	v_writelane_b32 v26, s46, 30                               // 000000003CB4: D761001A 00013C2E
	v_writelane_b32 v24, s50, 2                                // 000000003CBC: D7610018 00010432
	v_writelane_b32 v26, s47, 31                               // 000000003CC4: D761001A 00013E2F
	v_writelane_b32 v24, s51, 3                                // 000000003CCC: D7610018 00010633
	s_or_saveexec_b32 s105, -1                                 // 000000003CD4: BEE922C1
	s_wait_alu 0xfffe                                          // 000000003CD8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003CDC: BEFE0069
	s_load_b512 s[44:59], s[6:7], 0x0                          // 000000003CE0: F4008B03 F8000000
	s_or_saveexec_b32 s105, -1                                 // 000000003CE8: BEE922C1
	v_mov_b32_e32 v29, v28                                     // 000000003CEC: 7E3A031C
	s_wait_alu 0xfffe                                          // 000000003CF0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003CF4: BEFE0069
	s_wait_kmcnt 0x0                                           // 000000003CF8: BFC70000
	v_writelane_b32 v29, s44, 12                               // 000000003CFC: D761001D 0001182C
	s_movk_i32 s0, 0xff00                                      // 000000003D04: B000FF00
	s_mov_b32 s1, -1                                           // 000000003D08: BE8100C1
	s_movk_i32 s6, 0xfca0                                      // 000000003D0C: B006FCA0
	s_wait_alu 0xfffe                                          // 000000003D10: BF88FFFE
	s_add_nc_u64 s[0:1], s[34:35], s[0:1]                      // 000000003D14: A9800022
	v_writelane_b32 v29, s45, 13                               // 000000003D18: D761001D 00011A2D
	s_mov_b32 s7, -1                                           // 000000003D20: BE8700C1
	s_wait_alu 0xfffe                                          // 000000003D24: BF88FFFE
	s_add_nc_u64 s[6:7], s[34:35], s[6:7]                      // 000000003D28: A9860622
	v_writelane_b32 v29, s46, 14                               // 000000003D2C: D761001D 00011C2E
	v_writelane_b32 v29, s47, 15                               // 000000003D34: D761001D 00011E2F
	v_writelane_b32 v29, s48, 16                               // 000000003D3C: D761001D 00012030
	v_writelane_b32 v29, s49, 17                               // 000000003D44: D761001D 00012231
	v_writelane_b32 v29, s50, 18                               // 000000003D4C: D761001D 00012432
	v_writelane_b32 v29, s51, 19                               // 000000003D54: D761001D 00012633
	v_writelane_b32 v29, s52, 20                               // 000000003D5C: D761001D 00012834
	v_writelane_b32 v29, s53, 21                               // 000000003D64: D761001D 00012A35
	v_writelane_b32 v29, s54, 22                               // 000000003D6C: D761001D 00012C36
	v_writelane_b32 v29, s55, 23                               // 000000003D74: D761001D 00012E37
	v_writelane_b32 v29, s56, 24                               // 000000003D7C: D761001D 00013038
	v_writelane_b32 v29, s57, 25                               // 000000003D84: D761001D 00013239
	v_writelane_b32 v29, s58, 26                               // 000000003D8C: D761001D 0001343A
	v_writelane_b32 v29, s59, 27                               // 000000003D94: D761001D 0001363B
	s_load_b256 s[44:51], s[34:35], 0x11c0                     // 000000003D9C: F4006B11 F80011C0
	s_wait_kmcnt 0x0                                           // 000000003DA4: BFC70000
	v_writelane_b32 v26, s44, 20                               // 000000003DA8: D761001A 0001282C
	v_writelane_b32 v26, s45, 21                               // 000000003DB0: D761001A 00012A2D
	v_writelane_b32 v26, s46, 22                               // 000000003DB8: D761001A 00012C2E
	v_writelane_b32 v26, s47, 23                               // 000000003DC0: D761001A 00012E2F
	v_writelane_b32 v26, s48, 24                               // 000000003DC8: D761001A 00013030
	v_writelane_b32 v26, s49, 25                               // 000000003DD0: D761001A 00013231
	v_writelane_b32 v26, s50, 26                               // 000000003DD8: D761001A 00013432
	v_writelane_b32 v26, s51, 27                               // 000000003DE0: D761001A 00013633
	s_load_b512 s[44:59], s[0:1], 0x0                          // 000000003DE8: F4008B00 F8000000
	s_wait_kmcnt 0x0                                           // 000000003DF0: BFC70000
	v_writelane_b32 v26, s44, 4                                // 000000003DF4: D761001A 0001082C
	v_writelane_b32 v26, s45, 5                                // 000000003DFC: D761001A 00010A2D
	v_writelane_b32 v26, s46, 6                                // 000000003E04: D761001A 00010C2E
	v_writelane_b32 v26, s47, 7                                // 000000003E0C: D761001A 00010E2F
	v_writelane_b32 v26, s48, 8                                // 000000003E14: D761001A 00011030
	v_writelane_b32 v26, s49, 9                                // 000000003E1C: D761001A 00011231
	v_writelane_b32 v26, s50, 10                               // 000000003E24: D761001A 00011432
	v_writelane_b32 v26, s51, 11                               // 000000003E2C: D761001A 00011633
	v_writelane_b32 v26, s52, 12                               // 000000003E34: D761001A 00011834
	v_writelane_b32 v26, s53, 13                               // 000000003E3C: D761001A 00011A35
	v_writelane_b32 v26, s54, 14                               // 000000003E44: D761001A 00011C36
	v_writelane_b32 v26, s55, 15                               // 000000003E4C: D761001A 00011E37
	v_writelane_b32 v26, s56, 16                               // 000000003E54: D761001A 00012038
	v_writelane_b32 v26, s57, 17                               // 000000003E5C: D761001A 00012239
	v_writelane_b32 v26, s58, 18                               // 000000003E64: D761001A 0001243A
	v_writelane_b32 v26, s59, 19                               // 000000003E6C: D761001A 0001263B
	s_load_b256 s[44:51], s[6:7], 0x0                          // 000000003E74: F4006B03 F8000000
	s_wait_kmcnt 0x0                                           // 000000003E7C: BFC70000
	v_writelane_b32 v29, s44, 28                               // 000000003E80: D761001D 0001382C
	v_writelane_b32 v29, s45, 29                               // 000000003E88: D761001D 00013A2D
	v_writelane_b32 v29, s46, 30                               // 000000003E90: D761001D 00013C2E
	v_writelane_b32 v29, s47, 31                               // 000000003E98: D761001D 00013E2F
	s_or_saveexec_b32 s105, -1                                 // 000000003EA0: BEE922C1
	scratch_store_b32 off, v29, off offset:24                  // 000000003EA4: ED06807C 0E800000 00001800
	s_wait_alu 0xfffe                                          // 000000003EB0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003EB4: BEFE0069
	v_writelane_b32 v26, s48, 0                                // 000000003EB8: D761001A 00010030
	v_writelane_b32 v26, s49, 1                                // 000000003EC0: D761001A 00010231
	v_writelane_b32 v26, s50, 2                                // 000000003EC8: D761001A 00010432
	v_writelane_b32 v26, s51, 3                                // 000000003ED0: D761001A 00010633
	s_or_saveexec_b32 s105, -1                                 // 000000003ED8: BEE922C1
	scratch_store_b32 off, v26, off offset:28                  // 000000003EDC: ED06807C 0D000000 00001C00
	s_wait_alu 0xfffe                                          // 000000003EE8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003EEC: BEFE0069
	s_add_nc_u64 s[0:1], s[34:35], 0xc0                        // 000000003EF0: A980FF22 000000C0
	s_mov_b32 s56, 0                                           // 000000003EF8: BEB80080
	s_mov_b32 s38, s104                                        // 000000003EFC: BEA60068
	s_branch 241                                               // 000000003F00: BFA000F1 <r_3_3_3_8_8_8+0x2cc8>
	s_or_saveexec_b32 s105, -1                                 // 000000003F04: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000003F08: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000003F14: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003F18: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000003F1C: BFC00000
	v_readlane_b32 s0, v29, 22                                 // 000000003F20: D7600000 00012D1D
	v_readlane_b32 s1, v29, 23                                 // 000000003F28: D7600001 00012F1D
	s_or_saveexec_b32 s105, -1                                 // 000000003F30: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000003F34: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 000000003F40: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003F44: BEFE0069
	s_load_b256 s[44:51], s[4:5], 0x0                          // 000000003F48: F4006B02 F8000000
	s_wait_loadcnt 0x0                                         // 000000003F50: BFC00000
	v_readlane_b32 s2, v29, 29                                 // 000000003F54: D7600002 00013B1D
	v_readlane_b32 s3, v29, 30                                 // 000000003F5C: D7600003 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003F64: BF870001
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]                        // 000000003F68: A9800200
	s_or_saveexec_b32 s105, -1                                 // 000000003F6C: BEE922C1
	scratch_load_b32 v28, off, off offset:28 th:TH_LOAD_LU     // 000000003F70: ED05007C 0030001C 00001C00
	s_wait_alu 0xfffe                                          // 000000003F7C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003F80: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000003F84: BFC00000
	s_wait_kmcnt 0x0                                           // 000000003F88: BFC70000
	v_writelane_b32 v28, s44, 28                               // 000000003F8C: D761001C 0001382C
	v_writelane_b32 v28, s45, 29                               // 000000003F94: D761001C 00013A2D
	v_writelane_b32 v28, s46, 30                               // 000000003F9C: D761001C 00013C2E
	v_writelane_b32 v28, s47, 31                               // 000000003FA4: D761001C 00013E2F
	s_or_saveexec_b32 s105, -1                                 // 000000003FAC: BEE922C1
	s_wait_alu 0xfffe                                          // 000000003FB0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000003FB4: BEFE0069
	v_writelane_b32 v24, s48, 0                                // 000000003FB8: D7610018 00010030
	s_load_b256 s[0:7], s[0:1], 0x0                            // 000000003FC0: F4006000 F8000000
	s_mov_b32 s42, s40                                         // 000000003FC8: BEAA0028
	s_mov_b32 s43, s40                                         // 000000003FCC: BEAB0028
	s_mov_b32 s41, s40                                         // 000000003FD0: BEA90028
	v_writelane_b32 v24, s49, 1                                // 000000003FD4: D7610018 00010231
	s_wait_alu 0xfffe                                          // 000000003FDC: BF88FFFE
	s_mov_b64 s[58:59], s[42:43]                               // 000000003FE0: BEBA012A
	s_mov_b32 s38, -1                                          // 000000003FE4: BEA600C1
	s_mov_b64 s[56:57], s[40:41]                               // 000000003FE8: BEB80128
	s_mov_b64 s[86:87], s[42:43]                               // 000000003FEC: BED6012A
	v_writelane_b32 v24, s50, 2                                // 000000003FF0: D7610018 00010432
	s_mov_b64 s[8:9], s[40:41]                                 // 000000003FF8: BE880128
	s_mov_b64 s[78:79], s[42:43]                               // 000000003FFC: BECE012A
	s_mov_b64 s[74:75], s[42:43]                               // 000000004000: BECA012A
	s_mov_b64 s[84:85], s[40:41]                               // 000000004004: BED40128
	v_writelane_b32 v24, s51, 3                                // 000000004008: D7610018 00010633
	s_load_b256 s[44:51], s[34:35], 0x11c0                     // 000000004010: F4006B11 F80011C0
	s_mov_b64 s[10:11], s[42:43]                               // 000000004018: BE8A012A
	s_mov_b64 s[76:77], s[40:41]                               // 00000000401C: BECC0128
	s_mov_b64 s[72:73], s[40:41]                               // 000000004020: BEC80128
	s_wait_kmcnt 0x0                                           // 000000004024: BFC70000
	v_writelane_b32 v24, s0, 12                                // 000000004028: D7610018 00011800
	v_writelane_b32 v24, s1, 13                                // 000000004030: D7610018 00011A01
	v_writelane_b32 v24, s2, 14                                // 000000004038: D7610018 00011C02
	v_writelane_b32 v28, s44, 20                               // 000000004040: D761001C 0001282C
	v_writelane_b32 v24, s3, 15                                // 000000004048: D7610018 00011E03
	v_writelane_b32 v28, s45, 21                               // 000000004050: D761001C 00012A2D
	v_writelane_b32 v24, s4, 16                                // 000000004058: D7610018 00012004
	v_writelane_b32 v28, s46, 22                               // 000000004060: D761001C 00012C2E
	v_writelane_b32 v24, s5, 17                                // 000000004068: D7610018 00012205
	v_writelane_b32 v28, s47, 23                               // 000000004070: D761001C 00012E2F
	v_writelane_b32 v24, s6, 18                                // 000000004078: D7610018 00012406
	v_writelane_b32 v28, s48, 24                               // 000000004080: D761001C 00013030
	v_writelane_b32 v24, s7, 19                                // 000000004088: D7610018 00012607
	v_writelane_b32 v28, s49, 25                               // 000000004090: D761001C 00013231
	v_writelane_b32 v28, s50, 26                               // 000000004098: D761001C 00013432
	v_writelane_b32 v28, s51, 27                               // 0000000040A0: D761001C 00013633
	s_mov_b64 s[46:47], s[42:43]                               // 0000000040A8: BEAE012A
	s_mov_b64 s[50:51], s[42:43]                               // 0000000040AC: BEB2012A
	s_mov_b64 s[44:45], s[40:41]                               // 0000000040B0: BEAC0128
	s_mov_b64 s[48:49], s[40:41]                               // 0000000040B4: BEB00128
	s_wait_alu 0xfffe                                          // 0000000040B8: BF88FFFE
	v_writelane_b32 v24, s44, 4                                // 0000000040BC: D7610018 0001082C
	v_writelane_b32 v24, s45, 5                                // 0000000040C4: D7610018 00010A2D
	v_writelane_b32 v24, s46, 6                                // 0000000040CC: D7610018 00010C2E
	v_writelane_b32 v24, s47, 7                                // 0000000040D4: D7610018 00010E2F
	v_writelane_b32 v24, s48, 8                                // 0000000040DC: D7610018 00011030
	v_writelane_b32 v24, s49, 9                                // 0000000040E4: D7610018 00011231
	v_writelane_b32 v24, s50, 10                               // 0000000040EC: D7610018 00011432
	v_writelane_b32 v24, s51, 11                               // 0000000040F4: D7610018 00011633
	s_or_saveexec_b32 s105, -1                                 // 0000000040FC: BEE922C1
	s_wait_alu 0xfffe                                          // 000000004100: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004104: BEFE0069
	s_mov_b64 s[0:1], s[40:41]                                 // 000000004108: BE800128
	s_mov_b64 s[4:5], s[40:41]                                 // 00000000410C: BE840128
	s_mov_b64 s[54:55], s[42:43]                               // 000000004110: BEB6012A
	s_mov_b64 s[2:3], s[42:43]                                 // 000000004114: BE82012A
	s_mov_b64 s[6:7], s[42:43]                                 // 000000004118: BE86012A
	s_mov_b64 s[52:53], s[40:41]                               // 00000000411C: BEB40128
	s_or_saveexec_b32 s105, -1                                 // 000000004120: BEE922C1
	scratch_load_b32 v29, off, off offset:24 th:TH_LOAD_LU     // 000000004124: ED05007C 0030001D 00001800
	s_wait_alu 0xfffe                                          // 000000004130: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004134: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000004138: BFC00000
	v_writelane_b32 v29, s52, 28                               // 00000000413C: D761001D 00013834
	s_mov_b64 s[82:83], s[42:43]                               // 000000004144: BED2012A
	s_mov_b64 s[80:81], s[40:41]                               // 000000004148: BED00128
	v_writelane_b32 v28, s56, 0                                // 00000000414C: D761001C 00010038
	v_writelane_b32 v29, s53, 29                               // 000000004154: D761001D 00013A35
	v_writelane_b32 v28, s57, 1                                // 00000000415C: D761001C 00010239
	v_writelane_b32 v29, s54, 30                               // 000000004164: D761001D 00013C36
	v_writelane_b32 v28, s58, 2                                // 00000000416C: D761001C 0001043A
	v_writelane_b32 v29, s55, 31                               // 000000004174: D761001D 00013E37
	v_writelane_b32 v28, s59, 3                                // 00000000417C: D761001C 0001063B
	v_writelane_b32 v29, s72, 12                               // 000000004184: D761001D 00011848
	v_writelane_b32 v29, s73, 13                               // 00000000418C: D761001D 00011A49
	v_writelane_b32 v29, s74, 14                               // 000000004194: D761001D 00011C4A
	v_writelane_b32 v29, s75, 15                               // 00000000419C: D761001D 00011E4B
	v_writelane_b32 v29, s76, 16                               // 0000000041A4: D761001D 0001204C
	v_writelane_b32 v29, s77, 17                               // 0000000041AC: D761001D 0001224D
	v_writelane_b32 v29, s78, 18                               // 0000000041B4: D761001D 0001244E
	v_writelane_b32 v29, s79, 19                               // 0000000041BC: D761001D 0001264F
	s_wait_alu 0xfffe                                          // 0000000041C4: BF88FFFE
	v_writelane_b32 v29, s80, 20                               // 0000000041C8: D761001D 00012850
	v_writelane_b32 v29, s81, 21                               // 0000000041D0: D761001D 00012A51
	v_writelane_b32 v29, s82, 22                               // 0000000041D8: D761001D 00012C52
	v_writelane_b32 v29, s83, 23                               // 0000000041E0: D761001D 00012E53
	v_writelane_b32 v29, s84, 24                               // 0000000041E8: D761001D 00013054
	v_writelane_b32 v29, s85, 25                               // 0000000041F0: D761001D 00013255
	v_writelane_b32 v29, s86, 26                               // 0000000041F8: D761001D 00013456
	v_writelane_b32 v29, s87, 27                               // 000000004200: D761001D 00013657
	s_or_saveexec_b32 s105, -1                                 // 000000004208: BEE922C1
	scratch_store_b32 off, v29, off offset:24                  // 00000000420C: ED06807C 0E800000 00001800
	s_wait_alu 0xfffe                                          // 000000004218: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000421C: BEFE0069
	s_mov_b64 s[12:13], s[40:41]                               // 000000004220: BE8C0128
	s_mov_b64 s[14:15], s[42:43]                               // 000000004224: BE8E012A
	v_writelane_b32 v28, s0, 4                                 // 000000004228: D761001C 00010800
	v_writelane_b32 v28, s1, 5                                 // 000000004230: D761001C 00010A01
	v_writelane_b32 v28, s2, 6                                 // 000000004238: D761001C 00010C02
	v_writelane_b32 v28, s3, 7                                 // 000000004240: D761001C 00010E03
	v_writelane_b32 v28, s4, 8                                 // 000000004248: D761001C 00011004
	v_writelane_b32 v28, s5, 9                                 // 000000004250: D761001C 00011205
	v_writelane_b32 v28, s6, 10                                // 000000004258: D761001C 00011406
	v_writelane_b32 v28, s7, 11                                // 000000004260: D761001C 00011607
	v_writelane_b32 v28, s8, 12                                // 000000004268: D761001C 00011808
	v_writelane_b32 v28, s9, 13                                // 000000004270: D761001C 00011A09
	v_writelane_b32 v28, s10, 14                               // 000000004278: D761001C 00011C0A
	v_writelane_b32 v28, s11, 15                               // 000000004280: D761001C 00011E0B
	s_wait_alu 0xfffe                                          // 000000004288: BF88FFFE
	v_writelane_b32 v28, s12, 16                               // 00000000428C: D761001C 0001200C
	v_writelane_b32 v28, s13, 17                               // 000000004294: D761001C 0001220D
	v_writelane_b32 v28, s14, 18                               // 00000000429C: D761001C 0001240E
	v_writelane_b32 v28, s15, 19                               // 0000000042A4: D761001C 0001260F
	s_or_saveexec_b32 s105, -1                                 // 0000000042AC: BEE922C1
	scratch_store_b32 off, v28, off offset:28                  // 0000000042B0: ED06807C 0E000000 00001C00
	s_wait_alu 0xfffe                                          // 0000000042BC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000042C0: BEFE0069
	s_mov_b64 s[0:1], s[36:37]                                 // 0000000042C4: BE800124
	s_or_saveexec_b32 s105, -1                                 // 0000000042C8: BEE922C1
	s_wait_alu 0xfffe                                          // 0000000042CC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000042D0: BEFE0069
	v_writelane_b32 v24, s16, 20                               // 0000000042D4: D7610018 00012810
	v_writelane_b32 v24, s17, 21                               // 0000000042DC: D7610018 00012A11
	v_writelane_b32 v24, s18, 22                               // 0000000042E4: D7610018 00012C12
	v_writelane_b32 v24, s19, 23                               // 0000000042EC: D7610018 00012E13
	v_writelane_b32 v24, s20, 24                               // 0000000042F4: D7610018 00013014
	v_writelane_b32 v24, s21, 25                               // 0000000042FC: D7610018 00013215
	v_writelane_b32 v24, s22, 26                               // 000000004304: D7610018 00013416
	v_writelane_b32 v24, s23, 27                               // 00000000430C: D7610018 00013617
	v_writelane_b32 v24, s24, 28                               // 000000004314: D7610018 00013818
	v_writelane_b32 v24, s25, 29                               // 00000000431C: D7610018 00013A19
	v_writelane_b32 v24, s26, 30                               // 000000004324: D7610018 00013C1A
	v_writelane_b32 v24, s27, 31                               // 00000000432C: D7610018 00013E1B
	s_or_saveexec_b32 s105, -1                                 // 000000004334: BEE922C1
	scratch_store_b32 off, v24, off offset:96                  // 000000004338: ED06807C 0C000000 00006000
	s_wait_alu 0xfffe                                          // 000000004344: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004348: BEFE0069
	v_writelane_b32 v26, s28, 0                                // 00000000434C: D761001A 0001001C
	s_mov_b32 s57, s56                                         // 000000004354: BEB90038
	s_mov_b32 s58, s56                                         // 000000004358: BEBA0038
	s_mov_b32 s59, s56                                         // 00000000435C: BEBB0038
	s_mov_b32 s12, s56                                         // 000000004360: BE8C0038
	v_writelane_b32 v26, s29, 1                                // 000000004364: D761001A 0001021D
	s_mov_b32 s13, s56                                         // 00000000436C: BE8D0038
	s_mov_b32 s14, s56                                         // 000000004370: BE8E0038
	s_mov_b32 s15, s56                                         // 000000004374: BE8F0038
	s_and_not1_b32 vcc_lo, exec_lo, s38                        // 000000004378: 916A267E
	v_writelane_b32 v26, s30, 2                                // 00000000437C: D761001A 0001041E
	s_mov_b32 s4, s56                                          // 000000004384: BE840038
	s_mov_b32 s5, s56                                          // 000000004388: BE850038
	s_mov_b32 s6, s56                                          // 00000000438C: BE860038
	s_mov_b32 s7, s56                                          // 000000004390: BE870038
	v_writelane_b32 v26, s31, 3                                // 000000004394: D761001A 0001061F
	s_mov_b32 s16, s56                                         // 00000000439C: BE900038
	s_mov_b32 s24, s56                                         // 0000000043A0: BE980038
	s_mov_b32 s25, s56                                         // 0000000043A4: BE990038
	s_mov_b32 s26, s56                                         // 0000000043A8: BE9A0038
	s_mov_b32 s27, s56                                         // 0000000043AC: BE9B0038
	s_mov_b32 s17, s56                                         // 0000000043B0: BE910038
	s_mov_b32 s18, s56                                         // 0000000043B4: BE920038
	s_mov_b32 s19, s56                                         // 0000000043B8: BE930038
	s_mov_b32 s20, s56                                         // 0000000043BC: BE940038
	s_mov_b32 s21, s56                                         // 0000000043C0: BE950038
	s_mov_b32 s22, s56                                         // 0000000043C4: BE960038
	s_mov_b32 s23, s56                                         // 0000000043C8: BE970038
	s_mov_b32 s28, s56                                         // 0000000043CC: BE9C0038
	s_mov_b32 s29, s56                                         // 0000000043D0: BE9D0038
	s_mov_b32 s30, s56                                         // 0000000043D4: BE9E0038
	s_mov_b32 s31, s56                                         // 0000000043D8: BE9F0038
	s_wait_alu 0xfffe                                          // 0000000043DC: BF88FFFE
	v_writelane_b32 v26, s16, 4                                // 0000000043E0: D761001A 00010810
	s_mov_b32 s8, s56                                          // 0000000043E8: BE880038
	s_mov_b32 s9, s56                                          // 0000000043EC: BE890038
	s_mov_b32 s10, s56                                         // 0000000043F0: BE8A0038
	s_mov_b32 s11, s56                                         // 0000000043F4: BE8B0038
	v_writelane_b32 v26, s17, 5                                // 0000000043F8: D761001A 00010A11
	s_mov_b32 s52, s56                                         // 000000004400: BEB40038
	s_mov_b32 s53, s56                                         // 000000004404: BEB50038
	s_mov_b32 s54, s56                                         // 000000004408: BEB60038
	s_mov_b32 s55, s56                                         // 00000000440C: BEB70038
	v_writelane_b32 v26, s18, 6                                // 000000004410: D761001A 00010C12
	v_writelane_b32 v26, s19, 7                                // 000000004418: D761001A 00010E13
	v_writelane_b32 v26, s20, 8                                // 000000004420: D761001A 00011014
	v_writelane_b32 v26, s21, 9                                // 000000004428: D761001A 00011215
	v_writelane_b32 v26, s22, 10                               // 000000004430: D761001A 00011416
	v_writelane_b32 v26, s23, 11                               // 000000004438: D761001A 00011617
	v_writelane_b32 v26, s24, 12                               // 000000004440: D761001A 00011818
	v_writelane_b32 v26, s25, 13                               // 000000004448: D761001A 00011A19
	v_writelane_b32 v26, s26, 14                               // 000000004450: D761001A 00011C1A
	v_writelane_b32 v26, s27, 15                               // 000000004458: D761001A 00011E1B
	v_writelane_b32 v26, s28, 16                               // 000000004460: D761001A 0001201C
	v_writelane_b32 v26, s29, 17                               // 000000004468: D761001A 0001221D
	v_writelane_b32 v26, s30, 18                               // 000000004470: D761001A 0001241E
	v_writelane_b32 v26, s31, 19                               // 000000004478: D761001A 0001261F
	s_mov_b32 s16, s56                                         // 000000004480: BE900038
	s_mov_b32 s17, s56                                         // 000000004484: BE910038
	s_mov_b32 s18, s56                                         // 000000004488: BE920038
	s_mov_b32 s19, s56                                         // 00000000448C: BE930038
	s_cbranch_vccnz 40                                         // 000000004490: BFA40028 <r_3_3_3_8_8_8+0x2f34>
	s_load_b512 s[4:19], s[0:1], 0x2500                        // 000000004494: F4008100 F8002500
	s_wait_kmcnt 0x0                                           // 00000000449C: BFC70000
	v_writelane_b32 v26, s4, 4                                 // 0000000044A0: D761001A 00010804
	v_writelane_b32 v26, s5, 5                                 // 0000000044A8: D761001A 00010A05
	v_writelane_b32 v26, s6, 6                                 // 0000000044B0: D761001A 00010C06
	v_writelane_b32 v26, s7, 7                                 // 0000000044B8: D761001A 00010E07
	v_writelane_b32 v26, s8, 8                                 // 0000000044C0: D761001A 00011008
	v_writelane_b32 v26, s9, 9                                 // 0000000044C8: D761001A 00011209
	v_writelane_b32 v26, s10, 10                               // 0000000044D0: D761001A 0001140A
	v_writelane_b32 v26, s11, 11                               // 0000000044D8: D761001A 0001160B
	v_writelane_b32 v26, s12, 12                               // 0000000044E0: D761001A 0001180C
	v_writelane_b32 v26, s13, 13                               // 0000000044E8: D761001A 00011A0D
	v_writelane_b32 v26, s14, 14                               // 0000000044F0: D761001A 00011C0E
	v_writelane_b32 v26, s15, 15                               // 0000000044F8: D761001A 00011E0F
	v_writelane_b32 v26, s16, 16                               // 000000004500: D761001A 00012010
	v_writelane_b32 v26, s17, 17                               // 000000004508: D761001A 00012211
	v_writelane_b32 v26, s18, 18                               // 000000004510: D761001A 00012412
	v_writelane_b32 v26, s19, 19                               // 000000004518: D761001A 00012613
	s_clause 0x1                                               // 000000004520: BF850001
	s_load_b256 s[52:59], s[0:1], 0x23e0                       // 000000004524: F4006D00 F80023E0
	s_load_b512 s[4:19], s[0:1], 0x2640                        // 00000000452C: F4008100 F8002640
	s_wait_kmcnt 0x0                                           // 000000004534: BFC70000
	v_writelane_b32 v26, s4, 20                                // 000000004538: D761001A 00012804
	v_writelane_b32 v26, s5, 21                                // 000000004540: D761001A 00012A05
	v_writelane_b32 v26, s6, 22                                // 000000004548: D761001A 00012C06
	v_writelane_b32 v26, s7, 23                                // 000000004550: D761001A 00012E07
	s_wait_alu 0xfffe                                          // 000000004558: BF88FFFE
	v_writelane_b32 v26, s8, 24                                // 00000000455C: D761001A 00013008
	v_writelane_b32 v26, s9, 25                                // 000000004564: D761001A 00013209
	v_writelane_b32 v26, s10, 26                               // 00000000456C: D761001A 0001340A
	v_writelane_b32 v26, s11, 27                               // 000000004574: D761001A 0001360B
	v_writelane_b32 v26, s12, 28                               // 00000000457C: D761001A 0001380C
	v_writelane_b32 v26, s13, 29                               // 000000004584: D761001A 00013A0D
	v_writelane_b32 v26, s14, 30                               // 00000000458C: D761001A 00013C0E
	v_writelane_b32 v26, s15, 31                               // 000000004594: D761001A 00013E0F
	s_or_saveexec_b32 s105, -1                                 // 00000000459C: BEE922C1
	scratch_store_b32 off, v26, off offset:100                 // 0000000045A0: ED06807C 0D000000 00006400
	s_wait_alu 0xfffe                                          // 0000000045AC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000045B0: BEFE0069
	v_writelane_b32 v29, s16, 0                                // 0000000045B4: D761001D 00010010
	v_writelane_b32 v29, s17, 1                                // 0000000045BC: D761001D 00010211
	v_writelane_b32 v29, s18, 2                                // 0000000045C4: D761001D 00010412
	v_writelane_b32 v29, s19, 3                                // 0000000045CC: D761001D 00010613
	s_load_b256 s[16:23], s[0:1], 0x1120                       // 0000000045D4: F4006400 F8001120
	s_wait_kmcnt 0x0                                           // 0000000045DC: BFC70000
	v_writelane_b32 v29, s16, 4                                // 0000000045E0: D761001D 00010810
	v_writelane_b32 v29, s17, 5                                // 0000000045E8: D761001D 00010A11
	v_writelane_b32 v29, s18, 6                                // 0000000045F0: D761001D 00010C12
	v_writelane_b32 v29, s19, 7                                // 0000000045F8: D761001D 00010E13
	v_writelane_b32 v29, s20, 8                                // 000000004600: D761001D 00011014
	v_writelane_b32 v29, s21, 9                                // 000000004608: D761001D 00011215
	v_writelane_b32 v29, s22, 10                               // 000000004610: D761001D 00011416
	v_writelane_b32 v29, s23, 11                               // 000000004618: D761001D 00011617
	s_clause 0x1                                               // 000000004620: BF850001
	s_load_b256 s[16:23], s[0:1], 0xfe0                        // 000000004624: F4006400 F8000FE0
	s_load_b512 s[0:15], s[0:1], 0x1240                        // 00000000462C: F4008000 F8001240
	s_wait_kmcnt 0x0                                           // 000000004634: BFC70000
	v_writelane_b32 v29, s16, 12                               // 000000004638: D761001D 00011810
	v_writelane_b32 v29, s17, 13                               // 000000004640: D761001D 00011A11
	v_writelane_b32 v29, s18, 14                               // 000000004648: D761001D 00011C12
	v_writelane_b32 v29, s19, 15                               // 000000004650: D761001D 00011E13
	v_writelane_b32 v29, s20, 16                               // 000000004658: D761001D 00012014
	v_writelane_b32 v29, s21, 17                               // 000000004660: D761001D 00012215
	v_writelane_b32 v29, s22, 18                               // 000000004668: D761001D 00012416
	v_writelane_b32 v29, s23, 19                               // 000000004670: D761001D 00012617
	v_writelane_b32 v29, s0, 20                                // 000000004678: D761001D 00012800
	v_writelane_b32 v29, s1, 21                                // 000000004680: D761001D 00012A01
	v_writelane_b32 v29, s2, 22                                // 000000004688: D761001D 00012C02
	v_writelane_b32 v29, s3, 23                                // 000000004690: D761001D 00012E03
	v_writelane_b32 v29, s4, 24                                // 000000004698: D761001D 00013004
	v_writelane_b32 v29, s5, 25                                // 0000000046A0: D761001D 00013205
	v_writelane_b32 v29, s6, 26                                // 0000000046A8: D761001D 00013406
	v_writelane_b32 v29, s7, 27                                // 0000000046B0: D761001D 00013607
	v_writelane_b32 v29, s8, 28                                // 0000000046B8: D761001D 00013808
	v_writelane_b32 v29, s9, 29                                // 0000000046C0: D761001D 00013A09
	v_writelane_b32 v29, s10, 30                               // 0000000046C8: D761001D 00013C0A
	v_writelane_b32 v29, s11, 31                               // 0000000046D0: D761001D 00013E0B
	s_or_saveexec_b32 s105, -1                                 // 0000000046D8: BEE922C1
	scratch_store_b32 off, v29, off offset:76                  // 0000000046DC: ED06807C 0E800000 00004C00
	s_wait_alu 0xfffe                                          // 0000000046E8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000046EC: BEFE0069
	v_writelane_b32 v24, s12, 0                                // 0000000046F0: D7610018 0001000C
	s_and_b32 vcc_lo, exec_lo, s33                             // 0000000046F8: 8B6A217E
	v_writelane_b32 v24, s13, 1                                // 0000000046FC: D7610018 0001020D
	v_writelane_b32 v24, s14, 2                                // 000000004704: D7610018 0001040E
	v_writelane_b32 v24, s15, 3                                // 00000000470C: D7610018 0001060F
	s_or_saveexec_b32 s105, -1                                 // 000000004714: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000004718: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000004724: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004728: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000472C: BFC00000
	v_readlane_b32 s0, v29, 28                                 // 000000004730: D7600000 0001391D
	v_readlane_b32 s1, v29, 29                                 // 000000004738: D7600001 00013B1D
	s_or_saveexec_b32 s105, -1                                 // 000000004740: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000004744: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 000000004750: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004754: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000004758: BFC00000
	v_readlane_b32 s2, v29, 29                                 // 00000000475C: D7600002 00013B1D
	v_readlane_b32 s3, v29, 30                                 // 000000004764: D7600003 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000476C: BF870001
	s_add_nc_u64 s[4:5], s[0:1], s[2:3]                        // 000000004770: A9840200
	s_add_nc_u64 s[2:3], s[34:35], 0x100                       // 000000004774: A982FF22 00000100
	s_cbranch_vccnz 2196                                       // 00000000477C: BFA40894 <r_3_3_3_8_8_8+0x53d0>
	s_or_saveexec_b32 s105, -1                                 // 000000004780: BEE922C1
	s_wait_alu 0xfffe                                          // 000000004784: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004788: BEFE0069
	v_readlane_b32 s0, v23, 1                                  // 00000000478C: D7600000 00010317
	s_mov_b32 s41, s40                                         // 000000004794: BEA90028
	s_mov_b32 s42, s40                                         // 000000004798: BEAA0028
	s_mov_b32 s43, s40                                         // 00000000479C: BEAB0028
	s_wait_alu 0xfffe                                          // 0000000047A0: BF88FFFE
	s_mov_b64 s[28:29], s[40:41]                               // 0000000047A4: BE9C0128
	s_mov_b64 s[24:25], s[40:41]                               // 0000000047A8: BE980128
	s_and_b32 vcc_lo, exec_lo, s0                              // 0000000047AC: 8B6A007E
	s_mov_b64 s[30:31], s[42:43]                               // 0000000047B0: BE9E012A
	s_mov_b64 s[26:27], s[42:43]                               // 0000000047B4: BE9A012A
	s_cbranch_vccnz 26                                         // 0000000047B8: BFA4001A <r_3_3_3_8_8_8+0x3224>
	s_or_saveexec_b32 s105, -1                                 // 0000000047BC: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 0000000047C0: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 0000000047CC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000047D0: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000047D4: BFC00000
	v_readlane_b32 s0, v29, 30                                 // 0000000047D8: D7600000 00013D1D
	v_readlane_b32 s1, v29, 31                                 // 0000000047E0: D7600001 00013F1D
	s_or_saveexec_b32 s105, -1                                 // 0000000047E8: BEE922C1
	scratch_load_b32 v29, off, off                             // 0000000047EC: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 0000000047F8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000047FC: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000004800: BFC00000
	v_readlane_b32 s6, v29, 29                                 // 000000004804: D7600006 00013B1D
	v_readlane_b32 s7, v29, 30                                 // 00000000480C: D7600007 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004814: BF870001
	s_add_nc_u64 s[0:1], s[0:1], s[6:7]                        // 000000004818: A9800600
	s_load_b256 s[24:31], s[0:1], 0x0                          // 00000000481C: F4006600 F8000000
	s_or_saveexec_b32 s105, -1                                 // 000000004824: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000004828: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000004834: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004838: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000483C: BFC00000
	v_readlane_b32 s0, v29, 26                                 // 000000004840: D7600000 0001351D
	v_readlane_b32 s1, v29, 27                                 // 000000004848: D7600001 0001371D
	s_or_saveexec_b32 s105, -1                                 // 000000004850: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000004854: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 000000004860: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004864: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000004868: BFC00000
	v_readlane_b32 s8, v29, 29                                 // 00000000486C: D7600008 00013B1D
	v_readlane_b32 s9, v29, 30                                 // 000000004874: D7600009 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000487C: BF870001
	s_add_nc_u64 s[0:1], s[0:1], s[8:9]                        // 000000004880: A9800800
	s_or_saveexec_b32 s105, -1                                 // 000000004884: BEE922C1
	scratch_load_b32 v29, off, off offset:8                    // 000000004888: ED05007C 0000001D 00000800
	s_wait_alu 0xfffe                                          // 000000004894: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004898: BEFE0069
	s_load_b256 s[44:51], s[0:1], 0x0                          // 00000000489C: F4006B00 F8000000
	s_wait_loadcnt 0x0                                         // 0000000048A4: BFC00000
	v_readlane_b32 s6, v29, 0                                  // 0000000048A8: D7600006 0001011D
	v_readlane_b32 s7, v29, 1                                  // 0000000048B0: D7600007 0001031D
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000048B8: BF870001
	s_add_nc_u64 s[6:7], s[6:7], s[8:9]                        // 0000000048BC: A9860806
	s_load_b256 s[8:15], s[4:5], 0x0                           // 0000000048C0: F4006202 F8000000
	s_wait_kmcnt 0x0                                           // 0000000048C8: BFC70000
	v_writelane_b32 v20, s44, 28                               // 0000000048CC: D7610014 0001382C
	v_writelane_b32 v26, s48, 0                                // 0000000048D4: D761001A 00010030
	v_writelane_b32 v20, s45, 29                               // 0000000048DC: D7610014 00013A2D
	v_writelane_b32 v26, s49, 1                                // 0000000048E4: D761001A 00010231
	v_writelane_b32 v20, s46, 30                               // 0000000048EC: D7610014 00013C2E
	v_writelane_b32 v26, s50, 2                                // 0000000048F4: D761001A 00010432
	v_writelane_b32 v20, s47, 31                               // 0000000048FC: D7610014 00013E2F
	v_writelane_b32 v26, s51, 3                                // 000000004904: D761001A 00010633
	v_writelane_b32 v20, s8, 20                                // 00000000490C: D7610014 00012808
	v_writelane_b32 v20, s9, 21                                // 000000004914: D7610014 00012A09
	v_writelane_b32 v20, s10, 22                               // 00000000491C: D7610014 00012C0A
	v_writelane_b32 v20, s11, 23                               // 000000004924: D7610014 00012E0B
	v_writelane_b32 v20, s12, 24                               // 00000000492C: D7610014 0001300C
	v_writelane_b32 v20, s13, 25                               // 000000004934: D7610014 0001320D
	v_writelane_b32 v20, s14, 26                               // 00000000493C: D7610014 0001340E
	v_writelane_b32 v20, s15, 27                               // 000000004944: D7610014 0001360F
	s_load_b512 s[8:23], s[6:7], 0x0                           // 00000000494C: F4008203 F8000000
	s_or_saveexec_b32 s105, -1                                 // 000000004954: BEE922C1
	v_mov_b32_e32 v29, v24                                     // 000000004958: 7E3A0318
	s_wait_alu 0xfffe                                          // 00000000495C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004960: BEFE0069
	s_wait_kmcnt 0x0                                           // 000000004964: BFC70000
	v_writelane_b32 v29, s8, 4                                 // 000000004968: D761001D 00010808
	s_load_b256 s[44:51], s[34:35], 0x1200                     // 000000004970: F4006B11 F8001200
	s_movk_i32 s0, 0xff40                                      // 000000004978: B000FF40
	s_mov_b32 s1, -1                                           // 00000000497C: BE8100C1
	s_movk_i32 s6, 0xfce0                                      // 000000004980: B006FCE0
	v_writelane_b32 v29, s9, 5                                 // 000000004984: D761001D 00010A09
	s_wait_alu 0xfffe                                          // 00000000498C: BF88FFFE
	s_add_nc_u64 s[0:1], s[34:35], s[0:1]                      // 000000004990: A9800022
	s_mov_b32 s7, -1                                           // 000000004994: BE8700C1
	s_wait_alu 0xfffe                                          // 000000004998: BF88FFFE
	s_add_nc_u64 s[6:7], s[34:35], s[6:7]                      // 00000000499C: A9860622
	v_writelane_b32 v29, s10, 6                                // 0000000049A0: D761001D 00010C0A
	v_writelane_b32 v29, s11, 7                                // 0000000049A8: D761001D 00010E0B
	v_writelane_b32 v29, s12, 8                                // 0000000049B0: D761001D 0001100C
	s_wait_kmcnt 0x0                                           // 0000000049B8: BFC70000
	v_writelane_b32 v20, s44, 12                               // 0000000049BC: D7610014 0001182C
	v_writelane_b32 v29, s13, 9                                // 0000000049C4: D761001D 0001120D
	v_writelane_b32 v20, s45, 13                               // 0000000049CC: D7610014 00011A2D
	v_writelane_b32 v29, s14, 10                               // 0000000049D4: D761001D 0001140E
	v_writelane_b32 v20, s46, 14                               // 0000000049DC: D7610014 00011C2E
	v_writelane_b32 v29, s15, 11                               // 0000000049E4: D761001D 0001160F
	v_writelane_b32 v20, s47, 15                               // 0000000049EC: D7610014 00011E2F
	v_writelane_b32 v29, s16, 12                               // 0000000049F4: D761001D 00011810
	v_writelane_b32 v20, s48, 16                               // 0000000049FC: D7610014 00012030
	v_writelane_b32 v29, s17, 13                               // 000000004A04: D761001D 00011A11
	v_writelane_b32 v20, s49, 17                               // 000000004A0C: D7610014 00012231
	v_writelane_b32 v29, s18, 14                               // 000000004A14: D761001D 00011C12
	v_writelane_b32 v20, s50, 18                               // 000000004A1C: D7610014 00012432
	v_writelane_b32 v29, s19, 15                               // 000000004A24: D761001D 00011E13
	v_writelane_b32 v20, s51, 19                               // 000000004A2C: D7610014 00012633
	v_writelane_b32 v29, s20, 16                               // 000000004A34: D761001D 00012014
	v_writelane_b32 v29, s21, 17                               // 000000004A3C: D761001D 00012215
	v_writelane_b32 v29, s22, 18                               // 000000004A44: D761001D 00012416
	v_writelane_b32 v29, s23, 19                               // 000000004A4C: D761001D 00012617
	s_load_b512 s[8:23], s[0:1], 0x0                           // 000000004A54: F4008200 F8000000
	s_wait_kmcnt 0x0                                           // 000000004A5C: BFC70000
	v_writelane_b32 v29, s8, 28                                // 000000004A60: D761001D 00013808
	v_writelane_b32 v20, s12, 0                                // 000000004A68: D7610014 0001000C
	v_writelane_b32 v29, s9, 29                                // 000000004A70: D761001D 00013A09
	v_writelane_b32 v20, s13, 1                                // 000000004A78: D7610014 0001020D
	v_writelane_b32 v29, s10, 30                               // 000000004A80: D761001D 00013C0A
	v_writelane_b32 v20, s14, 2                                // 000000004A88: D7610014 0001040E
	v_writelane_b32 v29, s11, 31                               // 000000004A90: D761001D 00013E0B
	v_writelane_b32 v20, s15, 3                                // 000000004A98: D7610014 0001060F
	v_writelane_b32 v20, s16, 4                                // 000000004AA0: D7610014 00010810
	v_writelane_b32 v20, s17, 5                                // 000000004AA8: D7610014 00010A11
	v_writelane_b32 v20, s18, 6                                // 000000004AB0: D7610014 00010C12
	v_writelane_b32 v20, s19, 7                                // 000000004AB8: D7610014 00010E13
	v_writelane_b32 v20, s20, 8                                // 000000004AC0: D7610014 00011014
	v_writelane_b32 v20, s21, 9                                // 000000004AC8: D7610014 00011215
	v_writelane_b32 v20, s22, 10                               // 000000004AD0: D7610014 00011416
	v_writelane_b32 v20, s23, 11                               // 000000004AD8: D7610014 00011617
	s_or_saveexec_b32 s105, -1                                 // 000000004AE0: BEE922C1
	scratch_store_b32 off, v20, off offset:36                  // 000000004AE4: ED06807C 0A000000 00002400
	s_wait_alu 0xfffe                                          // 000000004AF0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004AF4: BEFE0069
	s_load_b256 s[44:51], s[6:7], 0x0                          // 000000004AF8: F4006B03 F8000000
	s_wait_kmcnt 0x0                                           // 000000004B00: BFC70000
	v_writelane_b32 v29, s44, 20                               // 000000004B04: D761001D 0001282C
	v_writelane_b32 v29, s45, 21                               // 000000004B0C: D761001D 00012A2D
	v_writelane_b32 v29, s46, 22                               // 000000004B14: D761001D 00012C2E
	v_writelane_b32 v29, s47, 23                               // 000000004B1C: D761001D 00012E2F
	v_writelane_b32 v29, s48, 24                               // 000000004B24: D761001D 00013030
	v_writelane_b32 v29, s49, 25                               // 000000004B2C: D761001D 00013231
	v_writelane_b32 v29, s50, 26                               // 000000004B34: D761001D 00013432
	v_writelane_b32 v29, s51, 27                               // 000000004B3C: D761001D 00013633
	s_or_saveexec_b32 s105, -1                                 // 000000004B44: BEE922C1
	scratch_store_b32 off, v29, off offset:32                  // 000000004B48: ED06807C 0E800000 00002000
	s_wait_alu 0xfffe                                          // 000000004B54: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004B58: BEFE0069
	s_add_nc_u64 s[0:1], s[34:35], 0x100                       // 000000004B5C: A980FF22 00000100
	s_mov_b32 s8, 0                                            // 000000004B64: BE880080
	s_mov_b32 s6, s104                                         // 000000004B68: BE860068
	s_branch 219                                               // 000000004B6C: BFA000DB <r_3_3_3_8_8_8+0x38dc>
	s_or_saveexec_b32 s105, -1                                 // 000000004B70: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 000000004B74: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000004B80: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004B84: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000004B88: BFC00000
	v_readlane_b32 s0, v29, 30                                 // 000000004B8C: D7600000 00013D1D
	v_readlane_b32 s1, v29, 31                                 // 000000004B94: D7600001 00013F1D
	s_or_saveexec_b32 s105, -1                                 // 000000004B9C: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000004BA0: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 000000004BAC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004BB0: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000004BB4: BFC00000
	v_readlane_b32 s6, v29, 29                                 // 000000004BB8: D7600006 00013B1D
	v_readlane_b32 s7, v29, 30                                 // 000000004BC0: D7600007 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004BC8: BF870001
	s_add_nc_u64 s[0:1], s[0:1], s[6:7]                        // 000000004BCC: A9800600
	s_load_b256 s[4:11], s[4:5], 0x0                           // 000000004BD0: F4006102 F8000000
	s_or_saveexec_b32 s105, -1                                 // 000000004BD8: BEE922C1
	scratch_load_b32 v24, off, off offset:36 th:TH_LOAD_LU     // 000000004BDC: ED05007C 00300018 00002400
	s_wait_alu 0xfffe                                          // 000000004BE8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004BEC: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000004BF0: BFC00000
	s_wait_kmcnt 0x0                                           // 000000004BF4: BFC70000
	v_writelane_b32 v24, s4, 20                                // 000000004BF8: D7610018 00012804
	s_clause 0x1                                               // 000000004C00: BF850001
	s_load_b256 s[24:31], s[0:1], 0x0                          // 000000004C04: F4006600 F8000000
	s_load_b256 s[44:51], s[34:35], 0x1200                     // 000000004C0C: F4006B11 F8001200
	s_mov_b32 s42, s40                                         // 000000004C14: BEAA0028
	s_mov_b32 s43, s40                                         // 000000004C18: BEAB0028
	s_mov_b32 s41, s40                                         // 000000004C1C: BEA90028
	v_writelane_b32 v24, s5, 21                                // 000000004C20: D7610018 00012A05
	s_wait_alu 0xfffe                                          // 000000004C28: BF88FFFE
	s_mov_b64 s[20:21], s[40:41]                               // 000000004C2C: BE940128
	s_mov_b64 s[78:79], s[42:43]                               // 000000004C30: BECE012A
	s_mov_b64 s[90:91], s[42:43]                               // 000000004C34: BEDA012A
	s_mov_b64 s[12:13], s[40:41]                               // 000000004C38: BE8C0128
	v_writelane_b32 v24, s6, 22                                // 000000004C3C: D7610018 00012C06
	s_mov_b64 s[82:83], s[42:43]                               // 000000004C44: BED2012A
	s_mov_b64 s[86:87], s[42:43]                               // 000000004C48: BED6012A
	s_mov_b64 s[74:75], s[42:43]                               // 000000004C4C: BECA012A
	s_mov_b64 s[22:23], s[42:43]                               // 000000004C50: BE96012A
	v_writelane_b32 v24, s7, 23                                // 000000004C54: D7610018 00012E07
	s_mov_b64 s[76:77], s[40:41]                               // 000000004C5C: BECC0128
	s_mov_b64 s[88:89], s[40:41]                               // 000000004C60: BED80128
	s_mov_b64 s[14:15], s[42:43]                               // 000000004C64: BE8E012A
	s_mov_b64 s[80:81], s[40:41]                               // 000000004C68: BED00128
	v_writelane_b32 v24, s8, 24                                // 000000004C6C: D7610018 00013008
	s_mov_b64 s[84:85], s[40:41]                               // 000000004C74: BED40128
	s_mov_b64 s[72:73], s[40:41]                               // 000000004C78: BEC80128
	v_writelane_b32 v24, s9, 25                                // 000000004C7C: D7610018 00013209
	v_writelane_b32 v24, s10, 26                               // 000000004C84: D7610018 0001340A
	v_writelane_b32 v24, s11, 27                               // 000000004C8C: D7610018 0001360B
	s_mov_b32 s6, -1                                           // 000000004C94: BE8600C1
	s_mov_b64 s[8:9], s[40:41]                                 // 000000004C98: BE880128
	s_mov_b64 s[10:11], s[42:43]                               // 000000004C9C: BE8A012A
	s_wait_kmcnt 0x0                                           // 000000004CA0: BFC70000
	v_writelane_b32 v24, s44, 12                               // 000000004CA4: D7610018 0001182C
	v_writelane_b32 v24, s45, 13                               // 000000004CAC: D7610018 00011A2D
	v_writelane_b32 v24, s46, 14                               // 000000004CB4: D7610018 00011C2E
	v_writelane_b32 v24, s47, 15                               // 000000004CBC: D7610018 00011E2F
	v_writelane_b32 v24, s48, 16                               // 000000004CC4: D7610018 00012030
	v_writelane_b32 v24, s49, 17                               // 000000004CCC: D7610018 00012231
	v_writelane_b32 v24, s50, 18                               // 000000004CD4: D7610018 00012432
	v_writelane_b32 v24, s51, 19                               // 000000004CDC: D7610018 00012633
	s_mov_b64 s[46:47], s[42:43]                               // 000000004CE4: BEAE012A
	s_mov_b64 s[50:51], s[42:43]                               // 000000004CE8: BEB2012A
	s_mov_b64 s[44:45], s[40:41]                               // 000000004CEC: BEAC0128
	s_mov_b64 s[48:49], s[40:41]                               // 000000004CF0: BEB00128
	s_wait_alu 0xfffe                                          // 000000004CF4: BF88FFFE
	v_writelane_b32 v24, s44, 28                               // 000000004CF8: D7610018 0001382C
	v_writelane_b32 v26, s48, 0                                // 000000004D00: D761001A 00010030
	v_writelane_b32 v24, s45, 29                               // 000000004D08: D7610018 00013A2D
	v_writelane_b32 v26, s49, 1                                // 000000004D10: D761001A 00010231
	v_writelane_b32 v24, s46, 30                               // 000000004D18: D7610018 00013C2E
	v_writelane_b32 v26, s50, 2                                // 000000004D20: D761001A 00010432
	v_writelane_b32 v24, s47, 31                               // 000000004D28: D7610018 00013E2F
	v_writelane_b32 v26, s51, 3                                // 000000004D30: D761001A 00010633
	s_or_saveexec_b32 s105, -1                                 // 000000004D38: BEE922C1
	scratch_load_b32 v29, off, off offset:32 th:TH_LOAD_LU     // 000000004D3C: ED05007C 0030001D 00002000
	s_wait_alu 0xfffe                                          // 000000004D48: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004D4C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000004D50: BFC00000
	v_writelane_b32 v29, s72, 20                               // 000000004D54: D761001D 00012848
	s_mov_b64 s[16:17], s[40:41]                               // 000000004D5C: BE900128
	s_mov_b64 s[18:19], s[42:43]                               // 000000004D60: BE92012A
	s_mov_b64 s[94:95], s[42:43]                               // 000000004D64: BEDE012A
	s_mov_b64 s[92:93], s[40:41]                               // 000000004D68: BEDC0128
	v_writelane_b32 v29, s73, 21                               // 000000004D6C: D761001D 00012A49
	v_writelane_b32 v29, s74, 22                               // 000000004D74: D761001D 00012C4A
	v_writelane_b32 v29, s75, 23                               // 000000004D7C: D761001D 00012E4B
	v_writelane_b32 v29, s76, 24                               // 000000004D84: D761001D 0001304C
	v_writelane_b32 v29, s77, 25                               // 000000004D8C: D761001D 0001324D
	v_writelane_b32 v29, s78, 26                               // 000000004D94: D761001D 0001344E
	v_writelane_b32 v29, s79, 27                               // 000000004D9C: D761001D 0001364F
	v_writelane_b32 v29, s8, 4                                 // 000000004DA4: D761001D 00010808
	v_writelane_b32 v29, s9, 5                                 // 000000004DAC: D761001D 00010A09
	v_writelane_b32 v29, s10, 6                                // 000000004DB4: D761001D 00010C0A
	v_writelane_b32 v29, s11, 7                                // 000000004DBC: D761001D 00010E0B
	v_writelane_b32 v29, s12, 8                                // 000000004DC4: D761001D 0001100C
	v_writelane_b32 v29, s13, 9                                // 000000004DCC: D761001D 0001120D
	v_writelane_b32 v29, s14, 10                               // 000000004DD4: D761001D 0001140E
	v_writelane_b32 v29, s15, 11                               // 000000004DDC: D761001D 0001160F
	s_wait_alu 0xfffe                                          // 000000004DE4: BF88FFFE
	v_writelane_b32 v29, s16, 12                               // 000000004DE8: D761001D 00011810
	v_writelane_b32 v29, s17, 13                               // 000000004DF0: D761001D 00011A11
	v_writelane_b32 v29, s18, 14                               // 000000004DF8: D761001D 00011C12
	v_writelane_b32 v29, s19, 15                               // 000000004E00: D761001D 00011E13
	v_writelane_b32 v29, s20, 16                               // 000000004E08: D761001D 00012014
	v_writelane_b32 v29, s21, 17                               // 000000004E10: D761001D 00012215
	v_writelane_b32 v29, s22, 18                               // 000000004E18: D761001D 00012416
	v_writelane_b32 v29, s23, 19                               // 000000004E20: D761001D 00012617
	v_writelane_b32 v29, s80, 28                               // 000000004E28: D761001D 00013850
	v_writelane_b32 v29, s81, 29                               // 000000004E30: D761001D 00013A51
	v_writelane_b32 v29, s82, 30                               // 000000004E38: D761001D 00013C52
	v_writelane_b32 v29, s83, 31                               // 000000004E40: D761001D 00013E53
	s_or_saveexec_b32 s105, -1                                 // 000000004E48: BEE922C1
	scratch_store_b32 off, v29, off offset:32                  // 000000004E4C: ED06807C 0E800000 00002000
	s_wait_alu 0xfffe                                          // 000000004E58: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004E5C: BEFE0069
	v_writelane_b32 v24, s84, 0                                // 000000004E60: D7610018 00010054
	v_writelane_b32 v24, s85, 1                                // 000000004E68: D7610018 00010255
	v_writelane_b32 v24, s86, 2                                // 000000004E70: D7610018 00010456
	v_writelane_b32 v24, s87, 3                                // 000000004E78: D7610018 00010657
	v_writelane_b32 v24, s88, 4                                // 000000004E80: D7610018 00010858
	v_writelane_b32 v24, s89, 5                                // 000000004E88: D7610018 00010A59
	v_writelane_b32 v24, s90, 6                                // 000000004E90: D7610018 00010C5A
	v_writelane_b32 v24, s91, 7                                // 000000004E98: D7610018 00010E5B
	v_writelane_b32 v24, s92, 8                                // 000000004EA0: D7610018 0001105C
	v_writelane_b32 v24, s93, 9                                // 000000004EA8: D7610018 0001125D
	v_writelane_b32 v24, s94, 10                               // 000000004EB0: D7610018 0001145E
	v_writelane_b32 v24, s95, 11                               // 000000004EB8: D7610018 0001165F
	s_or_saveexec_b32 s105, -1                                 // 000000004EC0: BEE922C1
	scratch_store_b32 off, v24, off offset:36                  // 000000004EC4: ED06807C 0C000000 00002400
	s_wait_alu 0xfffe                                          // 000000004ED0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000004ED4: BEFE0069
	s_mov_b64 s[0:1], s[2:3]                                   // 000000004ED8: BE800102
	s_wait_alu 0xfffe                                          // 000000004EDC: BF88FFFE
	s_mov_b32 s44, s8                                          // 000000004EE0: BEAC0008
	s_mov_b32 s48, s8                                          // 000000004EE4: BEB00008
	s_mov_b32 s49, s8                                          // 000000004EE8: BEB10008
	s_mov_b32 s50, s8                                          // 000000004EEC: BEB20008
	s_mov_b32 s51, s8                                          // 000000004EF0: BEB30008
	s_mov_b32 s45, s8                                          // 000000004EF4: BEAD0008
	s_mov_b32 s46, s8                                          // 000000004EF8: BEAE0008
	s_mov_b32 s47, s8                                          // 000000004EFC: BEAF0008
	s_wait_alu 0xfffe                                          // 000000004F00: BF88FFFE
	v_writelane_b32 v26, s44, 4                                // 000000004F04: D761001A 0001082C
	s_mov_b32 s12, s8                                          // 000000004F0C: BE8C0008
	s_mov_b32 s13, s8                                          // 000000004F10: BE8D0008
	s_mov_b32 s14, s8                                          // 000000004F14: BE8E0008
	s_mov_b32 s15, s8                                          // 000000004F18: BE8F0008
	v_writelane_b32 v26, s45, 5                                // 000000004F1C: D761001A 00010A2D
	s_mov_b32 s80, s8                                          // 000000004F24: BED00008
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 000000004F28: 916A067E
	s_mov_b32 s81, s8                                          // 000000004F2C: BED10008
	s_mov_b32 s82, s8                                          // 000000004F30: BED20008
	v_writelane_b32 v26, s46, 6                                // 000000004F34: D761001A 00010C2E
	s_mov_b32 s83, s8                                          // 000000004F3C: BED30008
	s_mov_b32 s72, s8                                          // 000000004F40: BEC80008
	s_mov_b32 s73, s8                                          // 000000004F44: BEC90008
	s_mov_b32 s74, s8                                          // 000000004F48: BECA0008
	v_writelane_b32 v26, s47, 7                                // 000000004F4C: D761001A 00010E2F
	s_mov_b32 s75, s8                                          // 000000004F54: BECB0008
	s_mov_b32 s76, s8                                          // 000000004F58: BECC0008
	s_mov_b32 s77, s8                                          // 000000004F5C: BECD0008
	s_mov_b32 s78, s8                                          // 000000004F60: BECE0008
	v_writelane_b32 v26, s48, 8                                // 000000004F64: D761001A 00011030
	s_mov_b32 s79, s8                                          // 000000004F6C: BECF0008
	s_mov_b32 s4, s8                                           // 000000004F70: BE840008
	s_mov_b32 s5, s8                                           // 000000004F74: BE850008
	s_mov_b32 s6, s8                                           // 000000004F78: BE860008
	v_writelane_b32 v26, s49, 9                                // 000000004F7C: D761001A 00011231
	s_mov_b32 s7, s8                                           // 000000004F84: BE870008
	s_mov_b32 s9, s8                                           // 000000004F88: BE890008
	s_mov_b32 s10, s8                                          // 000000004F8C: BE8A0008
	s_mov_b32 s11, s8                                          // 000000004F90: BE8B0008
	v_writelane_b32 v26, s50, 10                               // 000000004F94: D761001A 00011432
	s_mov_b32 s84, s8                                          // 000000004F9C: BED40008
	s_mov_b32 s85, s8                                          // 000000004FA0: BED50008
	s_mov_b32 s86, s8                                          // 000000004FA4: BED60008
	s_mov_b32 s87, s8                                          // 000000004FA8: BED70008
	s_mov_b32 s16, s8                                          // 000000004FAC: BE900008
	s_mov_b32 s17, s8                                          // 000000004FB0: BE910008
	s_mov_b32 s18, s8                                          // 000000004FB4: BE920008
	s_mov_b32 s19, s48                                         // 000000004FB8: BE930030
	v_writelane_b32 v26, s51, 11                               // 000000004FBC: D761001A 00011633
	s_cbranch_vccnz 24                                         // 000000004FC4: BFA40018 <r_3_3_3_8_8_8+0x3a28>
	s_clause 0x2                                               // 000000004FC8: BF850002
	s_load_b512 s[72:87], s[0:1], 0x2500                       // 000000004FCC: F4009200 F8002500
	s_load_b256 s[44:51], s[0:1], 0x23e0                       // 000000004FD4: F4006B00 F80023E0
	s_load_b512 s[4:19], s[0:1], 0x2640                        // 000000004FDC: F4008100 F8002640
	s_wait_kmcnt 0x0                                           // 000000004FE4: BFC70000
	v_writelane_b32 v26, s44, 4                                // 000000004FE8: D761001A 0001082C
	v_writelane_b32 v26, s45, 5                                // 000000004FF0: D761001A 00010A2D
	v_writelane_b32 v26, s46, 6                                // 000000004FF8: D761001A 00010C2E
	v_writelane_b32 v26, s47, 7                                // 000000005000: D761001A 00010E2F
	v_writelane_b32 v26, s48, 8                                // 000000005008: D761001A 00011030
	v_writelane_b32 v26, s49, 9                                // 000000005010: D761001A 00011231
	v_writelane_b32 v26, s50, 10                               // 000000005018: D761001A 00011432
	v_writelane_b32 v26, s51, 11                               // 000000005020: D761001A 00011633
	s_wait_alu 0xfffe                                          // 000000005028: BF88FFFE
	v_writelane_b32 v26, s72, 12                               // 00000000502C: D761001A 00011848
	v_writelane_b32 v26, s73, 13                               // 000000005034: D761001A 00011A49
	v_writelane_b32 v26, s74, 14                               // 00000000503C: D761001A 00011C4A
	v_writelane_b32 v26, s75, 15                               // 000000005044: D761001A 00011E4B
	v_writelane_b32 v26, s76, 16                               // 00000000504C: D761001A 0001204C
	v_writelane_b32 v26, s77, 17                               // 000000005054: D761001A 0001224D
	v_writelane_b32 v26, s78, 18                               // 00000000505C: D761001A 0001244E
	v_writelane_b32 v26, s79, 19                               // 000000005064: D761001A 0001264F
	v_writelane_b32 v26, s80, 20                               // 00000000506C: D761001A 00012850
	v_writelane_b32 v26, s81, 21                               // 000000005074: D761001A 00012A51
	v_writelane_b32 v26, s82, 22                               // 00000000507C: D761001A 00012C52
	v_writelane_b32 v26, s83, 23                               // 000000005084: D761001A 00012E53
	v_writelane_b32 v26, s84, 24                               // 00000000508C: D761001A 00013054
	v_writelane_b32 v26, s85, 25                               // 000000005094: D761001A 00013255
	v_writelane_b32 v26, s86, 26                               // 00000000509C: D761001A 00013456
	v_writelane_b32 v26, s87, 27                               // 0000000050A4: D761001A 00013657
	v_writelane_b32 v26, s4, 28                                // 0000000050AC: D761001A 00013804
	v_writelane_b32 v26, s5, 29                                // 0000000050B4: D761001A 00013A05
	v_writelane_b32 v26, s6, 30                                // 0000000050BC: D761001A 00013C06
	v_writelane_b32 v26, s7, 31                                // 0000000050C4: D761001A 00013E07
	s_or_saveexec_b32 s105, -1                                 // 0000000050CC: BEE922C1
	scratch_store_b32 off, v26, off offset:72                  // 0000000050D0: ED06807C 0D000000 00004800
	s_wait_alu 0xfffe                                          // 0000000050DC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000050E0: BEFE0069
	v_writelane_b32 v29, s8, 0                                 // 0000000050E4: D761001D 00010008
	s_load_b256 s[44:51], s[0:1], 0x1120                       // 0000000050EC: F4006B00 F8001120
	v_writelane_b32 v29, s9, 1                                 // 0000000050F4: D761001D 00010209
	v_writelane_b32 v29, s10, 2                                // 0000000050FC: D761001D 0001040A
	v_writelane_b32 v29, s11, 3                                // 000000005104: D761001D 0001060B
	v_writelane_b32 v29, s12, 4                                // 00000000510C: D761001D 0001080C
	v_writelane_b32 v29, s13, 5                                // 000000005114: D761001D 00010A0D
	v_writelane_b32 v29, s14, 6                                // 00000000511C: D761001D 00010C0E
	v_writelane_b32 v29, s15, 7                                // 000000005124: D761001D 00010E0F
	v_writelane_b32 v29, s16, 8                                // 00000000512C: D761001D 00011010
	v_writelane_b32 v29, s17, 9                                // 000000005134: D761001D 00011211
	v_writelane_b32 v29, s18, 10                               // 00000000513C: D761001D 00011412
	v_writelane_b32 v29, s19, 11                               // 000000005144: D761001D 00011613
	s_wait_kmcnt 0x0                                           // 00000000514C: BFC70000
	v_writelane_b32 v29, s44, 12                               // 000000005150: D761001D 0001182C
	v_writelane_b32 v29, s45, 13                               // 000000005158: D761001D 00011A2D
	v_writelane_b32 v29, s46, 14                               // 000000005160: D761001D 00011C2E
	v_writelane_b32 v29, s47, 15                               // 000000005168: D761001D 00011E2F
	v_writelane_b32 v29, s48, 16                               // 000000005170: D761001D 00012030
	v_writelane_b32 v29, s49, 17                               // 000000005178: D761001D 00012231
	v_writelane_b32 v29, s50, 18                               // 000000005180: D761001D 00012432
	v_writelane_b32 v29, s51, 19                               // 000000005188: D761001D 00012633
	s_clause 0x1                                               // 000000005190: BF850001
	s_load_b256 s[44:51], s[0:1], 0xfe0                        // 000000005194: F4006B00 F8000FE0
	s_load_b512 s[0:15], s[0:1], 0x1240                        // 00000000519C: F4008000 F8001240
	s_wait_kmcnt 0x0                                           // 0000000051A4: BFC70000
	v_writelane_b32 v29, s44, 20                               // 0000000051A8: D761001D 0001282C
	v_writelane_b32 v29, s45, 21                               // 0000000051B0: D761001D 00012A2D
	v_writelane_b32 v29, s46, 22                               // 0000000051B8: D761001D 00012C2E
	v_writelane_b32 v29, s47, 23                               // 0000000051C0: D761001D 00012E2F
	v_writelane_b32 v29, s48, 24                               // 0000000051C8: D761001D 00013030
	v_writelane_b32 v29, s49, 25                               // 0000000051D0: D761001D 00013231
	v_writelane_b32 v29, s50, 26                               // 0000000051D8: D761001D 00013432
	v_writelane_b32 v29, s51, 27                               // 0000000051E0: D761001D 00013633
	v_writelane_b32 v29, s0, 28                                // 0000000051E8: D761001D 00013800
	v_writelane_b32 v29, s1, 29                                // 0000000051F0: D761001D 00013A01
	v_writelane_b32 v29, s2, 30                                // 0000000051F8: D761001D 00013C02
	v_writelane_b32 v29, s3, 31                                // 000000005200: D761001D 00013E03
	s_or_saveexec_b32 s105, -1                                 // 000000005208: BEE922C1
	scratch_store_b32 off, v29, off offset:80                  // 00000000520C: ED06807C 0E800000 00005000
	s_wait_alu 0xfffe                                          // 000000005218: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000521C: BEFE0069
	v_writelane_b32 v26, s4, 0                                 // 000000005220: D761001A 00010004
	s_and_b32 vcc_lo, exec_lo, s33                             // 000000005228: 8B6A217E
	v_writelane_b32 v26, s5, 1                                 // 00000000522C: D761001A 00010205
	v_writelane_b32 v26, s6, 2                                 // 000000005234: D761001A 00010406
	v_writelane_b32 v26, s7, 3                                 // 00000000523C: D761001A 00010607
	v_writelane_b32 v26, s8, 4                                 // 000000005244: D761001A 00010808
	v_writelane_b32 v26, s9, 5                                 // 00000000524C: D761001A 00010A09
	v_writelane_b32 v26, s10, 6                                // 000000005254: D761001A 00010C0A
	v_writelane_b32 v26, s11, 7                                // 00000000525C: D761001A 00010E0B
	v_writelane_b32 v26, s12, 8                                // 000000005264: D761001A 0001100C
	v_writelane_b32 v26, s13, 9                                // 00000000526C: D761001A 0001120D
	v_writelane_b32 v26, s14, 10                               // 000000005274: D761001A 0001140E
	v_writelane_b32 v26, s15, 11                               // 00000000527C: D761001A 0001160F
	s_or_saveexec_b32 s105, -1                                 // 000000005284: BEE922C1
	scratch_load_b32 v29, off, off offset:8                    // 000000005288: ED05007C 0000001D 00000800
	s_wait_alu 0xfffe                                          // 000000005294: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005298: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000529C: BFC00000
	v_readlane_b32 s0, v29, 2                                  // 0000000052A0: D7600000 0001051D
	v_readlane_b32 s1, v29, 3                                  // 0000000052A8: D7600001 0001071D
	s_or_saveexec_b32 s105, -1                                 // 0000000052B0: BEE922C1
	scratch_load_b32 v29, off, off                             // 0000000052B4: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 0000000052C0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000052C4: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000052C8: BFC00000
	v_readlane_b32 s2, v29, 29                                 // 0000000052CC: D7600002 00013B1D
	v_readlane_b32 s3, v29, 30                                 // 0000000052D4: D7600003 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000052DC: BF870001
	s_add_nc_u64 s[2:3], s[0:1], s[2:3]                        // 0000000052E0: A9820200
	s_add_nc_u64 s[0:1], s[34:35], 0x140                       // 0000000052E4: A980FF22 00000140
	s_cbranch_vccnz 1610                                       // 0000000052EC: BFA4064A <r_3_3_3_8_8_8+0x5618>
	v_readlane_b32 s4, v23, 1                                  // 0000000052F0: D7600004 00010317
	s_mov_b32 s42, s40                                         // 0000000052F8: BEAA0028
	s_mov_b32 s43, s40                                         // 0000000052FC: BEAB0028
	s_mov_b32 s41, s40                                         // 000000005300: BEA90028
	s_wait_alu 0xfffe                                          // 000000005304: BF88FFFE
	s_mov_b64 s[82:83], s[42:43]                               // 000000005308: BED2012A
	s_mov_b64 s[78:79], s[42:43]                               // 00000000530C: BECE012A
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000005310: 8B6A047E
	s_mov_b64 s[80:81], s[40:41]                               // 000000005314: BED00128
	s_mov_b64 s[76:77], s[40:41]                               // 000000005318: BECC0128
	s_cbranch_vccnz 26                                         // 00000000531C: BFA4001A <r_3_3_3_8_8_8+0x3d88>
	s_or_saveexec_b32 s105, -1                                 // 000000005320: BEE922C1
	scratch_load_b32 v29, off, off offset:8                    // 000000005324: ED05007C 0000001D 00000800
	s_wait_alu 0xfffe                                          // 000000005330: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005334: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000005338: BFC00000
	v_readlane_b32 s4, v29, 4                                  // 00000000533C: D7600004 0001091D
	v_readlane_b32 s5, v29, 5                                  // 000000005344: D7600005 00010B1D
	s_or_saveexec_b32 s105, -1                                 // 00000000534C: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000005350: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 00000000535C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005360: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000005364: BFC00000
	v_readlane_b32 s6, v29, 29                                 // 000000005368: D7600006 00013B1D
	v_readlane_b32 s7, v29, 30                                 // 000000005370: D7600007 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005378: BF870001
	s_add_nc_u64 s[4:5], s[4:5], s[6:7]                        // 00000000537C: A9840604
	s_load_b256 s[76:83], s[4:5], 0x0                          // 000000005380: F4007302 F8000000
	s_or_saveexec_b32 s105, -1                                 // 000000005388: BEE922C1
	scratch_load_b32 v29, off, off offset:4                    // 00000000538C: ED05007C 0000001D 00000400
	s_wait_alu 0xfffe                                          // 000000005398: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000539C: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000053A0: BFC00000
	v_readlane_b32 s4, v29, 0                                  // 0000000053A4: D7600004 0001011D
	v_readlane_b32 s5, v29, 1                                  // 0000000053AC: D7600005 0001031D
	s_or_saveexec_b32 s105, -1                                 // 0000000053B4: BEE922C1
	scratch_load_b32 v29, off, off                             // 0000000053B8: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 0000000053C4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000053C8: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000053CC: BFC00000
	v_readlane_b32 s8, v29, 29                                 // 0000000053D0: D7600008 00013B1D
	v_readlane_b32 s9, v29, 30                                 // 0000000053D8: D7600009 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000053E0: BF870001
	s_add_nc_u64 s[4:5], s[4:5], s[8:9]                        // 0000000053E4: A9840804
	s_or_saveexec_b32 s105, -1                                 // 0000000053E8: BEE922C1
	scratch_load_b32 v29, off, off offset:8                    // 0000000053EC: ED05007C 0000001D 00000800
	s_wait_alu 0xfffe                                          // 0000000053F8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000053FC: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000005400: BFC00000
	v_readlane_b32 s6, v29, 6                                  // 000000005404: D7600006 00010D1D
	v_readlane_b32 s7, v29, 7                                  // 00000000540C: D7600007 00010F1D
	s_load_b256 s[44:51], s[2:3], 0x0                          // 000000005414: F4006B01 F8000000
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000541C: BF870001
	s_add_nc_u64 s[6:7], s[6:7], s[8:9]                        // 000000005420: A9860806
	s_load_b256 s[8:15], s[4:5], 0x0                           // 000000005424: F4006202 F8000000
	s_wait_kmcnt 0x0                                           // 00000000542C: BFC70000
	v_writelane_b32 v29, s44, 28                               // 000000005430: D761001D 0001382C
	v_writelane_b32 v29, s45, 29                               // 000000005438: D761001D 00013A2D
	v_writelane_b32 v22, s8, 4                                 // 000000005440: D7610016 00010808
	v_writelane_b32 v29, s46, 30                               // 000000005448: D761001D 00013C2E
	v_writelane_b32 v22, s9, 5                                 // 000000005450: D7610016 00010A09
	v_writelane_b32 v29, s47, 31                               // 000000005458: D761001D 00013E2F
	v_writelane_b32 v22, s10, 6                                // 000000005460: D7610016 00010C0A
	v_writelane_b32 v22, s11, 7                                // 000000005468: D7610016 00010E0B
	v_writelane_b32 v22, s12, 8                                // 000000005470: D7610016 0001100C
	v_writelane_b32 v22, s13, 9                                // 000000005478: D7610016 0001120D
	v_writelane_b32 v22, s14, 10                               // 000000005480: D7610016 0001140E
	v_writelane_b32 v22, s15, 11                               // 000000005488: D7610016 0001160F
	s_load_b512 s[4:19], s[6:7], 0x0                           // 000000005490: F4008103 F8000000
	v_writelane_b32 v22, s48, 0                                // 000000005498: D7610016 00010030
	v_writelane_b32 v22, s49, 1                                // 0000000054A0: D7610016 00010231
	v_writelane_b32 v22, s50, 2                                // 0000000054A8: D7610016 00010432
	v_writelane_b32 v22, s51, 3                                // 0000000054B0: D7610016 00010633
	s_wait_kmcnt 0x0                                           // 0000000054B8: BFC70000
	v_writelane_b32 v26, s4, 12                                // 0000000054BC: D761001A 00011804
	v_writelane_b32 v26, s5, 13                                // 0000000054C4: D761001A 00011A05
	v_writelane_b32 v26, s6, 14                                // 0000000054CC: D761001A 00011C06
	v_writelane_b32 v26, s7, 15                                // 0000000054D4: D761001A 00011E07
	v_writelane_b32 v26, s8, 16                                // 0000000054DC: D761001A 00012008
	v_writelane_b32 v26, s9, 17                                // 0000000054E4: D761001A 00012209
	v_writelane_b32 v26, s10, 18                               // 0000000054EC: D761001A 0001240A
	v_writelane_b32 v26, s11, 19                               // 0000000054F4: D761001A 0001260B
	v_writelane_b32 v26, s12, 20                               // 0000000054FC: D761001A 0001280C
	v_writelane_b32 v26, s13, 21                               // 000000005504: D761001A 00012A0D
	v_writelane_b32 v26, s14, 22                               // 00000000550C: D761001A 00012C0E
	v_writelane_b32 v26, s15, 23                               // 000000005514: D761001A 00012E0F
	v_writelane_b32 v26, s16, 24                               // 00000000551C: D761001A 00013010
	v_writelane_b32 v26, s17, 25                               // 000000005524: D761001A 00013211
	v_writelane_b32 v26, s18, 26                               // 00000000552C: D761001A 00013412
	v_writelane_b32 v26, s19, 27                               // 000000005534: D761001A 00013613
	s_load_b256 s[4:11], s[34:35], 0x1240                      // 00000000553C: F4006111 F8001240
	s_wait_kmcnt 0x0                                           // 000000005544: BFC70000
	v_writelane_b32 v29, s4, 20                                // 000000005548: D761001D 00012804
	v_writelane_b32 v29, s5, 21                                // 000000005550: D761001D 00012A05
	v_writelane_b32 v29, s6, 22                                // 000000005558: D761001D 00012C06
	v_writelane_b32 v29, s7, 23                                // 000000005560: D761001D 00012E07
	v_writelane_b32 v29, s8, 24                                // 000000005568: D761001D 00013008
	v_writelane_b32 v29, s9, 25                                // 000000005570: D761001D 00013209
	v_writelane_b32 v29, s10, 26                               // 000000005578: D761001D 0001340A
	v_writelane_b32 v29, s11, 27                               // 000000005580: D761001D 0001360B
	s_movk_i32 s4, 0xff80                                      // 000000005588: B004FF80
	s_mov_b32 s5, -1                                           // 00000000558C: BE8500C1
	s_movk_i32 s6, 0xfd20                                      // 000000005590: B006FD20
	s_wait_alu 0xfffe                                          // 000000005594: BF88FFFE
	s_add_nc_u64 s[4:5], s[34:35], s[4:5]                      // 000000005598: A9840422
	s_mov_b32 s7, -1                                           // 00000000559C: BE8700C1
	s_load_b512 s[8:23], s[4:5], 0x0                           // 0000000055A0: F4008202 F8000000
	s_add_nc_u64 s[6:7], s[34:35], s[6:7]                      // 0000000055A8: A9860622
	s_wait_kmcnt 0x0                                           // 0000000055AC: BFC70000
	v_writelane_b32 v29, s8, 4                                 // 0000000055B0: D761001D 00010808
	v_writelane_b32 v29, s9, 5                                 // 0000000055B8: D761001D 00010A09
	v_writelane_b32 v29, s10, 6                                // 0000000055C0: D761001D 00010C0A
	v_writelane_b32 v29, s11, 7                                // 0000000055C8: D761001D 00010E0B
	v_writelane_b32 v29, s12, 8                                // 0000000055D0: D761001D 0001100C
	v_writelane_b32 v29, s13, 9                                // 0000000055D8: D761001D 0001120D
	v_writelane_b32 v29, s14, 10                               // 0000000055E0: D761001D 0001140E
	v_writelane_b32 v29, s15, 11                               // 0000000055E8: D761001D 0001160F
	v_writelane_b32 v29, s16, 12                               // 0000000055F0: D761001D 00011810
	v_writelane_b32 v29, s17, 13                               // 0000000055F8: D761001D 00011A11
	v_writelane_b32 v29, s18, 14                               // 000000005600: D761001D 00011C12
	v_writelane_b32 v29, s19, 15                               // 000000005608: D761001D 00011E13
	v_writelane_b32 v29, s20, 16                               // 000000005610: D761001D 00012014
	v_writelane_b32 v29, s21, 17                               // 000000005618: D761001D 00012215
	v_writelane_b32 v29, s22, 18                               // 000000005620: D761001D 00012416
	v_writelane_b32 v29, s23, 19                               // 000000005628: D761001D 00012617
	s_load_b256 s[4:11], s[6:7], 0x0                           // 000000005630: F4006103 F8000000
	s_wait_kmcnt 0x0                                           // 000000005638: BFC70000
	v_writelane_b32 v26, s4, 28                                // 00000000563C: D761001A 00013804
	v_writelane_b32 v29, s8, 0                                 // 000000005644: D761001D 00010008
	v_writelane_b32 v26, s5, 29                                // 00000000564C: D761001A 00013A05
	v_writelane_b32 v29, s9, 1                                 // 000000005654: D761001D 00010209
	v_writelane_b32 v26, s6, 30                                // 00000000565C: D761001A 00013C06
	v_writelane_b32 v29, s10, 2                                // 000000005664: D761001D 0001040A
	v_writelane_b32 v26, s7, 31                                // 00000000566C: D761001A 00013E07
	v_writelane_b32 v29, s11, 3                                // 000000005674: D761001D 0001060B
	s_or_saveexec_b32 s105, -1                                 // 00000000567C: BEE922C1
	scratch_store_b32 off, v29, off offset:88                  // 000000005680: ED06807C 0E800000 00005800
	s_wait_alu 0xfffe                                          // 00000000568C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005690: BEFE0069
	s_add_nc_u64 s[18:19], s[34:35], 0x140                     // 000000005694: A992FF22 00000140
	s_mov_b32 s12, 0                                           // 00000000569C: BE8C0080
	s_mov_b32 s4, s104                                         // 0000000056A0: BE840068
	s_branch 206                                               // 0000000056A4: BFA000CE <r_3_3_3_8_8_8+0x43e0>
	s_or_saveexec_b32 s105, -1                                 // 0000000056A8: BEE922C1
	scratch_load_b32 v29, off, off offset:8                    // 0000000056AC: ED05007C 0000001D 00000800
	s_wait_alu 0xfffe                                          // 0000000056B8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000056BC: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000056C0: BFC00000
	v_readlane_b32 s4, v29, 4                                  // 0000000056C4: D7600004 0001091D
	v_readlane_b32 s5, v29, 5                                  // 0000000056CC: D7600005 00010B1D
	s_or_saveexec_b32 s105, -1                                 // 0000000056D4: BEE922C1
	scratch_load_b32 v29, off, off                             // 0000000056D8: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 0000000056E4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000056E8: BEFE0069
	s_load_b256 s[8:15], s[2:3], 0x0                           // 0000000056EC: F4006201 F8000000
	s_wait_loadcnt 0x0                                         // 0000000056F4: BFC00000
	v_readlane_b32 s6, v29, 29                                 // 0000000056F8: D7600006 00013B1D
	v_readlane_b32 s7, v29, 30                                 // 000000005700: D7600007 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005708: BF870001
	s_add_nc_u64 s[4:5], s[4:5], s[6:7]                        // 00000000570C: A9840604
	s_or_saveexec_b32 s105, -1                                 // 000000005710: BEE922C1
	scratch_load_b32 v29, off, off offset:88 th:TH_LOAD_LU     // 000000005714: ED05007C 0030001D 00005800
	s_wait_alu 0xfffe                                          // 000000005720: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005724: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000005728: BFC00000
	s_wait_kmcnt 0x0                                           // 00000000572C: BFC70000
	v_writelane_b32 v29, s8, 28                                // 000000005730: D761001D 00013808
	v_writelane_b32 v22, s12, 0                                // 000000005738: D7610016 0001000C
	s_mov_b32 s41, s40                                         // 000000005740: BEA90028
	s_mov_b32 s42, s40                                         // 000000005744: BEAA0028
	s_mov_b32 s43, s40                                         // 000000005748: BEAB0028
	v_writelane_b32 v29, s9, 29                                // 00000000574C: D761001D 00013A09
	v_writelane_b32 v22, s13, 1                                // 000000005754: D7610016 0001020D
	s_wait_alu 0xfffe                                          // 00000000575C: BF88FFFE
	s_mov_b64 s[16:17], s[40:41]                               // 000000005760: BE900128
	s_mov_b64 s[18:19], s[42:43]                               // 000000005764: BE92012A
	s_mov_b64 s[86:87], s[42:43]                               // 000000005768: BED6012A
	v_writelane_b32 v29, s10, 30                               // 00000000576C: D761001D 00013C0A
	v_writelane_b32 v22, s14, 2                                // 000000005774: D7610016 0001040E
	s_mov_b64 s[62:63], s[42:43]                               // 00000000577C: BEBE012A
	s_mov_b64 s[98:99], s[42:43]                               // 000000005780: BEE2012A
	s_mov_b64 s[70:71], s[42:43]                               // 000000005784: BEC6012A
	v_writelane_b32 v29, s11, 31                               // 000000005788: D761001D 00013E0B
	v_writelane_b32 v22, s15, 3                                // 000000005790: D7610016 0001060F
	s_clause 0x1                                               // 000000005798: BF850001
	s_load_b256 s[76:83], s[4:5], 0x0                          // 00000000579C: F4007302 F8000000
	s_load_b256 s[4:11], s[34:35], 0x1240                      // 0000000057A4: F4006111 F8001240
	s_mov_b64 s[12:13], s[40:41]                               // 0000000057AC: BE8C0128
	s_mov_b64 s[14:15], s[42:43]                               // 0000000057B0: BE8E012A
	s_mov_b64 s[90:91], s[42:43]                               // 0000000057B4: BEDA012A
	s_mov_b64 s[84:85], s[40:41]                               // 0000000057B8: BED40128
	s_mov_b64 s[60:61], s[40:41]                               // 0000000057BC: BEBC0128
	s_mov_b64 s[66:67], s[42:43]                               // 0000000057C0: BEC2012A
	s_mov_b64 s[94:95], s[42:43]                               // 0000000057C4: BEDE012A
	s_mov_b64 s[74:75], s[42:43]                               // 0000000057C8: BECA012A
	s_mov_b64 s[96:97], s[40:41]                               // 0000000057CC: BEE00128
	s_mov_b64 s[68:69], s[40:41]                               // 0000000057D0: BEC40128
	s_mov_b64 s[88:89], s[40:41]                               // 0000000057D4: BED80128
	s_mov_b64 s[64:65], s[40:41]                               // 0000000057D8: BEC00128
	s_mov_b64 s[92:93], s[40:41]                               // 0000000057DC: BEDC0128
	s_mov_b64 s[72:73], s[40:41]                               // 0000000057E0: BEC80128
	s_wait_kmcnt 0x0                                           // 0000000057E4: BFC70000
	v_writelane_b32 v29, s4, 20                                // 0000000057E8: D761001D 00012804
	v_writelane_b32 v29, s5, 21                                // 0000000057F0: D761001D 00012A05
	v_writelane_b32 v29, s6, 22                                // 0000000057F8: D761001D 00012C06
	v_writelane_b32 v29, s7, 23                                // 000000005800: D761001D 00012E07
	v_writelane_b32 v29, s8, 24                                // 000000005808: D761001D 00013008
	v_writelane_b32 v29, s9, 25                                // 000000005810: D761001D 00013209
	v_writelane_b32 v29, s10, 26                               // 000000005818: D761001D 0001340A
	v_writelane_b32 v29, s11, 27                               // 000000005820: D761001D 0001360B
	s_mov_b64 s[8:9], s[40:41]                                 // 000000005828: BE880128
	s_mov_b64 s[10:11], s[42:43]                               // 00000000582C: BE8A012A
	s_wait_alu 0xfffe                                          // 000000005830: BF88FFFE
	v_writelane_b32 v22, s8, 4                                 // 000000005834: D7610016 00010808
	s_mov_b32 s4, -1                                           // 00000000583C: BE8400C1
	v_writelane_b32 v22, s9, 5                                 // 000000005840: D7610016 00010A09
	v_writelane_b32 v22, s10, 6                                // 000000005848: D7610016 00010C0A
	v_writelane_b32 v22, s11, 7                                // 000000005850: D7610016 00010E0B
	v_writelane_b32 v22, s12, 8                                // 000000005858: D7610016 0001100C
	v_writelane_b32 v22, s13, 9                                // 000000005860: D7610016 0001120D
	v_writelane_b32 v22, s14, 10                               // 000000005868: D7610016 0001140E
	v_writelane_b32 v22, s15, 11                               // 000000005870: D7610016 0001160F
	s_mov_b64 s[12:13], s[40:41]                               // 000000005878: BE8C0128
	s_mov_b64 s[14:15], s[42:43]                               // 00000000587C: BE8E012A
	s_wait_alu 0xfffe                                          // 000000005880: BF88FFFE
	v_writelane_b32 v26, s12, 28                               // 000000005884: D761001A 0001380C
	v_writelane_b32 v29, s16, 0                                // 00000000588C: D761001D 00010010
	v_writelane_b32 v26, s13, 29                               // 000000005894: D761001A 00013A0D
	v_writelane_b32 v29, s17, 1                                // 00000000589C: D761001D 00010211
	v_writelane_b32 v26, s14, 30                               // 0000000058A4: D761001A 00013C0E
	v_writelane_b32 v29, s18, 2                                // 0000000058AC: D761001D 00010412
	v_writelane_b32 v26, s15, 31                               // 0000000058B4: D761001A 00013E0F
	v_writelane_b32 v29, s19, 3                                // 0000000058BC: D761001D 00010613
	v_writelane_b32 v26, s84, 12                               // 0000000058C4: D761001A 00011854
	v_writelane_b32 v29, s60, 4                                // 0000000058CC: D761001D 0001083C
	v_writelane_b32 v26, s85, 13                               // 0000000058D4: D761001A 00011A55
	v_writelane_b32 v29, s61, 5                                // 0000000058DC: D761001D 00010A3D
	v_writelane_b32 v26, s86, 14                               // 0000000058E4: D761001A 00011C56
	v_writelane_b32 v29, s62, 6                                // 0000000058EC: D761001D 00010C3E
	v_writelane_b32 v26, s87, 15                               // 0000000058F4: D761001A 00011E57
	v_writelane_b32 v29, s63, 7                                // 0000000058FC: D761001D 00010E3F
	v_writelane_b32 v26, s88, 16                               // 000000005904: D761001A 00012058
	v_writelane_b32 v29, s64, 8                                // 00000000590C: D761001D 00011040
	v_writelane_b32 v26, s89, 17                               // 000000005914: D761001A 00012259
	v_writelane_b32 v29, s65, 9                                // 00000000591C: D761001D 00011241
	v_writelane_b32 v26, s90, 18                               // 000000005924: D761001A 0001245A
	v_writelane_b32 v29, s66, 10                               // 00000000592C: D761001D 00011442
	v_writelane_b32 v26, s91, 19                               // 000000005934: D761001A 0001265B
	v_writelane_b32 v29, s67, 11                               // 00000000593C: D761001D 00011643
	v_writelane_b32 v26, s92, 20                               // 000000005944: D761001A 0001285C
	v_writelane_b32 v29, s68, 12                               // 00000000594C: D761001D 00011844
	v_writelane_b32 v26, s93, 21                               // 000000005954: D761001A 00012A5D
	v_writelane_b32 v29, s69, 13                               // 00000000595C: D761001D 00011A45
	v_writelane_b32 v26, s94, 22                               // 000000005964: D761001A 00012C5E
	v_writelane_b32 v29, s70, 14                               // 00000000596C: D761001D 00011C46
	v_writelane_b32 v26, s95, 23                               // 000000005974: D761001A 00012E5F
	v_writelane_b32 v29, s71, 15                               // 00000000597C: D761001D 00011E47
	v_writelane_b32 v26, s96, 24                               // 000000005984: D761001A 00013060
	v_writelane_b32 v29, s72, 16                               // 00000000598C: D761001D 00012048
	v_writelane_b32 v26, s97, 25                               // 000000005994: D761001A 00013261
	v_writelane_b32 v29, s73, 17                               // 00000000599C: D761001D 00012249
	v_writelane_b32 v26, s98, 26                               // 0000000059A4: D761001A 00013462
	v_writelane_b32 v29, s74, 18                               // 0000000059AC: D761001D 0001244A
	v_writelane_b32 v26, s99, 27                               // 0000000059B4: D761001A 00013663
	v_writelane_b32 v29, s75, 19                               // 0000000059BC: D761001D 0001264B
	s_or_saveexec_b32 s105, -1                                 // 0000000059C4: BEE922C1
	scratch_store_b32 off, v29, off offset:88                  // 0000000059C8: ED06807C 0E800000 00005800
	s_wait_alu 0xfffe                                          // 0000000059D4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000059D8: BEFE0069
	s_mov_b64 s[18:19], s[0:1]                                 // 0000000059DC: BE920100
	v_writelane_b32 v22, s24, 12                               // 0000000059E0: D7610016 00011818
	v_writelane_b32 v22, s25, 13                               // 0000000059E8: D7610016 00011A19
	v_writelane_b32 v22, s26, 14                               // 0000000059F0: D7610016 00011C1A
	v_writelane_b32 v22, s27, 15                               // 0000000059F8: D7610016 00011E1B
	v_writelane_b32 v22, s28, 16                               // 000000005A00: D7610016 0001201C
	v_writelane_b32 v22, s29, 17                               // 000000005A08: D7610016 0001221D
	v_writelane_b32 v22, s30, 18                               // 000000005A10: D7610016 0001241E
	v_writelane_b32 v22, s31, 19                               // 000000005A18: D7610016 0001261F
	s_or_saveexec_b32 s105, -1                                 // 000000005A20: BEE922C1
	scratch_store_b32 off, v27, off offset:40                  // 000000005A24: ED06807C 0D800000 00002800
	s_wait_alu 0xfffe                                          // 000000005A30: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005A34: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000005A38: BEE922C1
	scratch_store_b32 off, v25, off offset:48                  // 000000005A3C: ED06807C 0C800000 00003000
	s_wait_alu 0xfffe                                          // 000000005A48: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005A4C: BEFE0069
	s_mov_b32 s13, s12                                         // 000000005A50: BE8D000C
	s_mov_b32 s14, s12                                         // 000000005A54: BE8E000C
	s_mov_b32 s15, s12                                         // 000000005A58: BE8F000C
	s_wait_alu 0xfffe                                          // 000000005A5C: BF88FFFE
	s_mov_b64 s[8:9], s[12:13]                                 // 000000005A60: BE88010C
	s_and_not1_b32 vcc_lo, exec_lo, s4                         // 000000005A64: 916A047E
	s_wait_alu 0xfffe                                          // 000000005A68: BF88FFFE
	s_mov_b32 s60, s8                                          // 000000005A6C: BEBC0008
	s_mov_b32 s68, s8                                          // 000000005A70: BEC40008
	s_mov_b32 s69, s8                                          // 000000005A74: BEC50008
	s_mov_b32 s70, s8                                          // 000000005A78: BEC60008
	s_mov_b32 s71, s8                                          // 000000005A7C: BEC70008
	s_mov_b32 s61, s8                                          // 000000005A80: BEBD0008
	s_mov_b32 s62, s8                                          // 000000005A84: BEBE0008
	s_mov_b32 s63, s8                                          // 000000005A88: BEBF0008
	s_mov_b32 s64, s8                                          // 000000005A8C: BEC00008
	s_mov_b32 s65, s8                                          // 000000005A90: BEC10008
	s_mov_b32 s66, s8                                          // 000000005A94: BEC20008
	s_mov_b32 s67, s8                                          // 000000005A98: BEC30008
	s_mov_b32 s72, s8                                          // 000000005A9C: BEC80008
	s_mov_b32 s73, s8                                          // 000000005AA0: BEC90008
	s_mov_b32 s74, s8                                          // 000000005AA4: BECA0008
	s_mov_b32 s75, s8                                          // 000000005AA8: BECB0008
	s_wait_alu 0xfffe                                          // 000000005AAC: BF88FFFE
	v_writelane_b32 v22, s60, 20                               // 000000005AB0: D7610016 0001283C
	v_writelane_b32 v29, s72, 0                                // 000000005AB8: D761001D 00010048
	s_mov_b32 s4, s8                                           // 000000005AC0: BE840008
	s_mov_b64 s[10:11], s[14:15]                               // 000000005AC4: BE8A010E
	s_mov_b32 s5, s8                                           // 000000005AC8: BE850008
	s_mov_b32 s6, s8                                           // 000000005ACC: BE860008
	v_writelane_b32 v29, s73, 1                                // 000000005AD0: D761001D 00010249
	s_mov_b32 s7, s8                                           // 000000005AD8: BE870008
	s_mov_b32 s84, s8                                          // 000000005ADC: BED40008
	s_mov_b32 s85, s8                                          // 000000005AE0: BED50008
	s_mov_b32 s86, s8                                          // 000000005AE4: BED60008
	v_writelane_b32 v29, s74, 2                                // 000000005AE8: D761001D 0001044A
	s_mov_b32 s87, s8                                          // 000000005AF0: BED70008
	s_mov_b32 s88, s8                                          // 000000005AF4: BED80008
	s_mov_b32 s89, s8                                          // 000000005AF8: BED90008
	s_mov_b32 s90, s8                                          // 000000005AFC: BEDA0008
	v_writelane_b32 v29, s75, 3                                // 000000005B00: D761001D 0001064B
	s_mov_b32 s91, s8                                          // 000000005B08: BEDB0008
	s_mov_b32 s96, s8                                          // 000000005B0C: BEE00008
	s_mov_b32 s97, s8                                          // 000000005B10: BEE10008
	s_mov_b32 s98, s8                                          // 000000005B14: BEE20008
	s_wait_alu 0xfffe                                          // 000000005B18: BF88FFFE
	v_writelane_b32 v29, s4, 4                                 // 000000005B1C: D761001D 00010804
	s_mov_b32 s92, s12                                         // 000000005B24: BEDC000C
	s_mov_b32 s93, s12                                         // 000000005B28: BEDD000C
	s_mov_b32 s94, s12                                         // 000000005B2C: BEDE000C
	s_mov_b32 s95, s12                                         // 000000005B30: BEDF000C
	v_writelane_b32 v29, s5, 5                                 // 000000005B34: D761001D 00010A05
	s_mov_b32 s99, s8                                          // 000000005B3C: BEE30008
	v_writelane_b32 v22, s61, 21                               // 000000005B40: D7610016 00012A3D
	v_writelane_b32 v29, s6, 6                                 // 000000005B48: D761001D 00010C06
	v_writelane_b32 v22, s62, 22                               // 000000005B50: D7610016 00012C3E
	v_writelane_b32 v29, s7, 7                                 // 000000005B58: D761001D 00010E07
	v_writelane_b32 v22, s63, 23                               // 000000005B60: D7610016 00012E3F
	v_writelane_b32 v29, s8, 8                                 // 000000005B68: D761001D 00011008
	v_writelane_b32 v22, s64, 24                               // 000000005B70: D7610016 00013040
	v_writelane_b32 v29, s9, 9                                 // 000000005B78: D761001D 00011209
	v_writelane_b32 v22, s65, 25                               // 000000005B80: D7610016 00013241
	v_writelane_b32 v29, s10, 10                               // 000000005B88: D761001D 0001140A
	v_writelane_b32 v22, s66, 26                               // 000000005B90: D7610016 00013442
	v_writelane_b32 v29, s11, 11                               // 000000005B98: D761001D 0001160B
	v_writelane_b32 v22, s67, 27                               // 000000005BA0: D7610016 00013643
	v_writelane_b32 v29, s84, 12                               // 000000005BA8: D761001D 00011854
	v_writelane_b32 v22, s68, 28                               // 000000005BB0: D7610016 00013844
	v_writelane_b32 v29, s85, 13                               // 000000005BB8: D761001D 00011A55
	v_writelane_b32 v22, s69, 29                               // 000000005BC0: D7610016 00013A45
	v_writelane_b32 v29, s86, 14                               // 000000005BC8: D761001D 00011C56
	v_writelane_b32 v22, s70, 30                               // 000000005BD0: D7610016 00013C46
	v_writelane_b32 v29, s87, 15                               // 000000005BD8: D761001D 00011E57
	v_writelane_b32 v22, s71, 31                               // 000000005BE0: D7610016 00013E47
	v_writelane_b32 v29, s88, 16                               // 000000005BE8: D761001D 00012058
	v_writelane_b32 v29, s89, 17                               // 000000005BF0: D761001D 00012259
	v_writelane_b32 v29, s90, 18                               // 000000005BF8: D761001D 0001245A
	v_writelane_b32 v29, s91, 19                               // 000000005C00: D761001D 0001265B
	s_wait_alu 0xfffe                                          // 000000005C08: BF88FFFE
	v_writelane_b32 v29, s92, 20                               // 000000005C0C: D761001D 0001285C
	v_writelane_b32 v29, s93, 21                               // 000000005C14: D761001D 00012A5D
	v_writelane_b32 v29, s94, 22                               // 000000005C1C: D761001D 00012C5E
	v_writelane_b32 v29, s95, 23                               // 000000005C24: D761001D 00012E5F
	v_writelane_b32 v29, s96, 24                               // 000000005C2C: D761001D 00013060
	v_writelane_b32 v29, s97, 25                               // 000000005C34: D761001D 00013261
	v_writelane_b32 v29, s98, 26                               // 000000005C3C: D761001D 00013462
	v_writelane_b32 v29, s99, 27                               // 000000005C44: D761001D 00013663
	s_cbranch_vccnz 89                                         // 000000005C4C: BFA40059 <r_3_3_3_8_8_8+0x47b4>
	s_load_b512 s[0:15], s[18:19], 0x2500                      // 000000005C50: F4008009 F8002500
	s_wait_kmcnt 0x0                                           // 000000005C58: BFC70000
	v_writelane_b32 v22, s0, 20                                // 000000005C5C: D7610016 00012800
	v_writelane_b32 v29, s12, 0                                // 000000005C64: D761001D 0001000C
	v_writelane_b32 v22, s1, 21                                // 000000005C6C: D7610016 00012A01
	v_writelane_b32 v29, s13, 1                                // 000000005C74: D761001D 0001020D
	v_writelane_b32 v22, s2, 22                                // 000000005C7C: D7610016 00012C02
	v_writelane_b32 v29, s14, 2                                // 000000005C84: D761001D 0001040E
	v_writelane_b32 v22, s3, 23                                // 000000005C8C: D7610016 00012E03
	v_writelane_b32 v29, s15, 3                                // 000000005C94: D761001D 0001060F
	v_writelane_b32 v22, s4, 24                                // 000000005C9C: D7610016 00013004
	v_writelane_b32 v22, s5, 25                                // 000000005CA4: D7610016 00013205
	v_writelane_b32 v22, s6, 26                                // 000000005CAC: D7610016 00013406
	v_writelane_b32 v22, s7, 27                                // 000000005CB4: D7610016 00013607
	s_load_b256 s[0:7], s[18:19], 0x23e0                       // 000000005CBC: F4006009 F80023E0
	v_writelane_b32 v22, s8, 28                                // 000000005CC4: D7610016 00013808
	v_writelane_b32 v22, s9, 29                                // 000000005CCC: D7610016 00013A09
	v_writelane_b32 v22, s10, 30                               // 000000005CD4: D7610016 00013C0A
	v_writelane_b32 v22, s11, 31                               // 000000005CDC: D7610016 00013E0B
	s_wait_kmcnt 0x0                                           // 000000005CE4: BFC70000
	v_writelane_b32 v29, s0, 4                                 // 000000005CE8: D761001D 00010800
	v_writelane_b32 v29, s1, 5                                 // 000000005CF0: D761001D 00010A01
	v_writelane_b32 v29, s2, 6                                 // 000000005CF8: D761001D 00010C02
	v_writelane_b32 v29, s3, 7                                 // 000000005D00: D761001D 00010E03
	v_writelane_b32 v29, s4, 8                                 // 000000005D08: D761001D 00011004
	v_writelane_b32 v29, s5, 9                                 // 000000005D10: D761001D 00011205
	v_writelane_b32 v29, s6, 10                                // 000000005D18: D761001D 00011406
	v_writelane_b32 v29, s7, 11                                // 000000005D20: D761001D 00011607
	s_load_b512 s[0:15], s[18:19], 0x2640                      // 000000005D28: F4008009 F8002640
	s_wait_kmcnt 0x0                                           // 000000005D30: BFC70000
	v_writelane_b32 v29, s0, 12                                // 000000005D34: D761001D 00011800
	v_writelane_b32 v29, s1, 13                                // 000000005D3C: D761001D 00011A01
	v_writelane_b32 v29, s2, 14                                // 000000005D44: D761001D 00011C02
	v_writelane_b32 v29, s3, 15                                // 000000005D4C: D761001D 00011E03
	v_writelane_b32 v29, s4, 16                                // 000000005D54: D761001D 00012004
	v_writelane_b32 v29, s5, 17                                // 000000005D5C: D761001D 00012205
	v_writelane_b32 v29, s6, 18                                // 000000005D64: D761001D 00012406
	v_writelane_b32 v29, s7, 19                                // 000000005D6C: D761001D 00012607
	v_writelane_b32 v29, s8, 20                                // 000000005D74: D761001D 00012808
	v_writelane_b32 v29, s9, 21                                // 000000005D7C: D761001D 00012A09
	v_writelane_b32 v29, s10, 22                               // 000000005D84: D761001D 00012C0A
	v_writelane_b32 v29, s11, 23                               // 000000005D8C: D761001D 00012E0B
	v_writelane_b32 v29, s12, 24                               // 000000005D94: D761001D 0001300C
	v_writelane_b32 v29, s13, 25                               // 000000005D9C: D761001D 0001320D
	v_writelane_b32 v29, s14, 26                               // 000000005DA4: D761001D 0001340E
	v_writelane_b32 v29, s15, 27                               // 000000005DAC: D761001D 0001360F
	s_load_b256 s[0:7], s[18:19], 0x1120                       // 000000005DB4: F4006009 F8001120
	s_wait_kmcnt 0x0                                           // 000000005DBC: BFC70000
	v_writelane_b32 v29, s0, 28                                // 000000005DC0: D761001D 00013800
	v_writelane_b32 v29, s1, 29                                // 000000005DC8: D761001D 00013A01
	v_writelane_b32 v29, s2, 30                                // 000000005DD0: D761001D 00013C02
	v_writelane_b32 v29, s3, 31                                // 000000005DD8: D761001D 00013E03
	s_or_saveexec_b32 s105, -1                                 // 000000005DE0: BEE922C1
	scratch_store_b32 off, v29, off offset:92                  // 000000005DE4: ED06807C 0E800000 00005C00
	s_wait_alu 0xfffe                                          // 000000005DF0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005DF4: BEFE0069
	v_writelane_b32 v20, s4, 0                                 // 000000005DF8: D7610014 00010004
	s_and_b32 vcc_lo, exec_lo, s33                             // 000000005E00: 8B6A217E
	v_writelane_b32 v20, s5, 1                                 // 000000005E04: D7610014 00010205
	v_writelane_b32 v20, s6, 2                                 // 000000005E0C: D7610014 00010406
	v_writelane_b32 v20, s7, 3                                 // 000000005E14: D7610014 00010607
	s_load_b256 s[0:7], s[18:19], 0xfe0                        // 000000005E1C: F4006009 F8000FE0
	s_wait_kmcnt 0x0                                           // 000000005E24: BFC70000
	v_writelane_b32 v20, s0, 4                                 // 000000005E28: D7610014 00010800
	v_writelane_b32 v20, s1, 5                                 // 000000005E30: D7610014 00010A01
	v_writelane_b32 v20, s2, 6                                 // 000000005E38: D7610014 00010C02
	v_writelane_b32 v20, s3, 7                                 // 000000005E40: D7610014 00010E03
	v_writelane_b32 v20, s4, 8                                 // 000000005E48: D7610014 00011004
	v_writelane_b32 v20, s5, 9                                 // 000000005E50: D7610014 00011205
	v_writelane_b32 v20, s6, 10                                // 000000005E58: D7610014 00011406
	v_writelane_b32 v20, s7, 11                                // 000000005E60: D7610014 00011607
	s_load_b512 s[0:15], s[18:19], 0x1240                      // 000000005E68: F4008009 F8001240
	s_wait_kmcnt 0x0                                           // 000000005E70: BFC70000
	v_writelane_b32 v20, s0, 12                                // 000000005E74: D7610014 00011800
	v_writelane_b32 v20, s1, 13                                // 000000005E7C: D7610014 00011A01
	v_writelane_b32 v20, s2, 14                                // 000000005E84: D7610014 00011C02
	v_writelane_b32 v20, s3, 15                                // 000000005E8C: D7610014 00011E03
	v_writelane_b32 v20, s4, 16                                // 000000005E94: D7610014 00012004
	v_writelane_b32 v20, s5, 17                                // 000000005E9C: D7610014 00012205
	v_writelane_b32 v20, s6, 18                                // 000000005EA4: D7610014 00012406
	v_writelane_b32 v20, s7, 19                                // 000000005EAC: D7610014 00012607
	v_writelane_b32 v20, s8, 20                                // 000000005EB4: D7610014 00012808
	v_writelane_b32 v20, s9, 21                                // 000000005EBC: D7610014 00012A09
	v_writelane_b32 v20, s10, 22                               // 000000005EC4: D7610014 00012C0A
	v_writelane_b32 v20, s11, 23                               // 000000005ECC: D7610014 00012E0B
	v_writelane_b32 v20, s12, 24                               // 000000005ED4: D7610014 0001300C
	v_writelane_b32 v20, s13, 25                               // 000000005EDC: D7610014 0001320D
	v_writelane_b32 v20, s14, 26                               // 000000005EE4: D7610014 0001340E
	v_writelane_b32 v20, s15, 27                               // 000000005EEC: D7610014 0001360F
	s_or_saveexec_b32 s105, -1                                 // 000000005EF4: BEE922C1
	scratch_load_b32 v29, off, off offset:8                    // 000000005EF8: ED05007C 0000001D 00000800
	s_wait_alu 0xfffe                                          // 000000005F04: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005F08: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000005F0C: BFC00000
	v_readlane_b32 s18, v29, 8                                 // 000000005F10: D7600012 0001111D
	v_readlane_b32 s19, v29, 9                                 // 000000005F18: D7600013 0001131D
	s_or_saveexec_b32 s105, -1                                 // 000000005F20: BEE922C1
	scratch_load_b32 v29, off, off                             // 000000005F24: ED05007C 0000001D 00000000
	s_wait_alu 0xfffe                                          // 000000005F30: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005F34: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000005F38: BFC00000
	v_readlane_b32 s0, v29, 29                                 // 000000005F3C: D7600000 00013B1D
	v_readlane_b32 s1, v29, 30                                 // 000000005F44: D7600001 00013D1D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005F4C: BF870001
	s_add_nc_u64 s[20:21], s[18:19], s[0:1]                    // 000000005F50: A9940012
	s_add_nc_u64 s[18:19], s[34:35], 0x180                     // 000000005F54: A992FF22 00000180
	s_cbranch_vccnz 954                                        // 000000005F5C: BFA403BA <r_3_3_3_8_8_8+0x5848>
	s_or_saveexec_b32 s105, -1                                 // 000000005F60: BEE922C1
	v_mov_b32_e32 v29, v20                                     // 000000005F64: 7E3A0314
	s_wait_alu 0xfffe                                          // 000000005F68: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005F6C: BEFE0069
	v_readlane_b32 s0, v23, 1                                  // 000000005F70: D7600000 00010317
	s_mov_b32 s42, s40                                         // 000000005F78: BEAA0028
	s_mov_b32 s43, s40                                         // 000000005F7C: BEAB0028
	s_mov_b32 s41, s40                                         // 000000005F80: BEA90028
	s_wait_alu 0xfffe                                          // 000000005F84: BF88FFFE
	s_mov_b64 s[98:99], s[42:43]                               // 000000005F88: BEE2012A
	s_mov_b64 s[94:95], s[42:43]                               // 000000005F8C: BEDE012A
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000005F90: 8B6A007E
	s_mov_b64 s[96:97], s[40:41]                               // 000000005F94: BEE00128
	s_mov_b64 s[92:93], s[40:41]                               // 000000005F98: BEDC0128
	s_cbranch_vccnz 26                                         // 000000005F9C: BFA4001A <r_3_3_3_8_8_8+0x4a08>
	s_or_saveexec_b32 s105, -1                                 // 000000005FA0: BEE922C1
	scratch_load_b32 v27, off, off offset:8                    // 000000005FA4: ED05007C 0000001B 00000800
	s_wait_alu 0xfffe                                          // 000000005FB0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005FB4: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000005FB8: BFC00000
	v_readlane_b32 s22, v27, 10                                // 000000005FBC: D7600016 0001151B
	v_readlane_b32 s23, v27, 11                                // 000000005FC4: D7600017 0001171B
	s_or_saveexec_b32 s105, -1                                 // 000000005FCC: BEE922C1
	scratch_load_b32 v27, off, off                             // 000000005FD0: ED05007C 0000001B 00000000
	s_wait_alu 0xfffe                                          // 000000005FDC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000005FE0: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000005FE4: BFC00000
	v_readlane_b32 s0, v27, 29                                 // 000000005FE8: D7600000 00013B1B
	v_readlane_b32 s1, v27, 30                                 // 000000005FF0: D7600001 00013D1B
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005FF8: BF870001
	s_add_nc_u64 s[22:23], s[22:23], s[0:1]                    // 000000005FFC: A9960016
	s_load_b256 s[92:99], s[22:23], 0x0                        // 000000006000: F400770B F8000000
	s_or_saveexec_b32 s105, -1                                 // 000000006008: BEE922C1
	scratch_load_b32 v20, off, off                             // 00000000600C: ED05007C 00000014 00000000
	s_wait_alu 0xfffe                                          // 000000006018: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000601C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000006020: BFC00000
	v_readlane_b32 s0, v20, 31                                 // 000000006024: D7600000 00013F14
	v_readlane_b32 s1, v23, 0                                  // 00000000602C: D7600001 00010117
	s_load_b256 s[24:31], s[0:1], 0x0                          // 000000006034: F4006600 F8000000
	s_wait_kmcnt 0x0                                           // 00000000603C: BFC70000
	v_writelane_b32 v28, s24, 20                               // 000000006040: D761001C 00012818
	v_writelane_b32 v28, s25, 21                               // 000000006048: D761001C 00012A19
	v_writelane_b32 v28, s26, 22                               // 000000006050: D761001C 00012C1A
	v_writelane_b32 v28, s27, 23                               // 000000006058: D761001C 00012E1B
	v_writelane_b32 v28, s28, 24                               // 000000006060: D761001C 0001301C
	v_writelane_b32 v28, s29, 25                               // 000000006068: D761001C 0001321D
	v_writelane_b32 v28, s30, 26                               // 000000006070: D761001C 0001341E
	v_writelane_b32 v28, s31, 27                               // 000000006078: D761001C 0001361F
	s_or_saveexec_b32 s105, -1                                 // 000000006080: BEE922C1
	scratch_load_b32 v27, off, off offset:8                    // 000000006084: ED05007C 0000001B 00000800
	s_wait_alu 0xfffe                                          // 000000006090: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000006094: BEFE0069
	s_load_b256 s[24:31], s[20:21], 0x0                        // 000000006098: F400660A F8000000
	s_wait_loadcnt 0x0                                         // 0000000060A0: BFC00000
	v_readlane_b32 s16, v27, 12                                // 0000000060A4: D7600010 0001191B
	v_readlane_b32 s0, v20, 29                                 // 0000000060AC: D7600000 00013B14
	v_readlane_b32 s17, v27, 13                                // 0000000060B4: D7600011 00011B1B
	v_readlane_b32 s1, v20, 30                                 // 0000000060BC: D7600001 00013D14
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000060C4: BF870001
	s_add_nc_u64 s[16:17], s[16:17], s[0:1]                    // 0000000060C8: A9900010
	s_load_b512 s[0:15], s[16:17], 0x0                         // 0000000060CC: F4008008 F8000000
	s_wait_kmcnt 0x0                                           // 0000000060D4: BFC70000
	v_writelane_b32 v28, s24, 12                               // 0000000060D8: D761001C 00011818
	v_writelane_b32 v28, s25, 13                               // 0000000060E0: D761001C 00011A19
	v_writelane_b32 v28, s26, 14                               // 0000000060E8: D761001C 00011C1A
	v_writelane_b32 v28, s27, 15                               // 0000000060F0: D761001C 00011E1B
	v_writelane_b32 v28, s28, 16                               // 0000000060F8: D761001C 0001201C
	v_writelane_b32 v28, s29, 17                               // 000000006100: D761001C 0001221D
	v_writelane_b32 v28, s30, 18                               // 000000006108: D761001C 0001241E
	v_writelane_b32 v28, s31, 19                               // 000000006110: D761001C 0001261F
	s_or_saveexec_b32 s105, -1                                 // 000000006118: BEE922C1
	v_mov_b32_e32 v20, v29                                     // 00000000611C: 7E28031D
	s_wait_alu 0xfffe                                          // 000000006120: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000006124: BEFE0069
	v_writelane_b32 v20, s0, 28                                // 000000006128: D7610014 00013800
	v_writelane_b32 v25, s4, 0                                 // 000000006130: D7610019 00010004
	s_movk_i32 s16, 0xffc0                                     // 000000006138: B010FFC0
	s_mov_b32 s17, -1                                          // 00000000613C: BE9100C1
	s_load_b256 s[24:31], s[34:35], 0x1280                     // 000000006140: F4006611 F8001280
	v_writelane_b32 v20, s1, 29                                // 000000006148: D7610014 00013A01
	v_writelane_b32 v25, s5, 1                                 // 000000006150: D7610019 00010205
	s_add_nc_u64 s[16:17], s[34:35], s[16:17]                  // 000000006158: A9901022
	s_movk_i32 s22, 0xfd60                                     // 00000000615C: B016FD60
	s_mov_b32 s23, -1                                          // 000000006160: BE9700C1
	v_writelane_b32 v20, s2, 30                                // 000000006164: D7610014 00013C02
	v_writelane_b32 v25, s6, 2                                 // 00000000616C: D7610019 00010406
	s_wait_alu 0xfffe                                          // 000000006174: BF88FFFE
	s_add_nc_u64 s[22:23], s[34:35], s[22:23]                  // 000000006178: A9961622
	s_add_nc_u64 s[74:75], s[34:35], 0x180                     // 00000000617C: A9CAFF22 00000180
	s_mov_b32 s88, 0                                           // 000000006184: BED80080
	v_writelane_b32 v20, s3, 31                                // 000000006188: D7610014 00013E03
	v_writelane_b32 v25, s7, 3                                 // 000000006190: D7610019 00010607
	v_writelane_b32 v25, s8, 4                                 // 000000006198: D7610019 00010808
	s_wait_kmcnt 0x0                                           // 0000000061A0: BFC70000
	v_writelane_b32 v28, s24, 4                                // 0000000061A4: D761001C 00010818
	v_writelane_b32 v25, s9, 5                                 // 0000000061AC: D7610019 00010A09
	v_writelane_b32 v28, s25, 5                                // 0000000061B4: D761001C 00010A19
	v_writelane_b32 v25, s10, 6                                // 0000000061BC: D7610019 00010C0A
	v_writelane_b32 v28, s26, 6                                // 0000000061C4: D761001C 00010C1A
	v_writelane_b32 v25, s11, 7                                // 0000000061CC: D7610019 00010E0B
	v_writelane_b32 v28, s27, 7                                // 0000000061D4: D761001C 00010E1B
	v_writelane_b32 v25, s12, 8                                // 0000000061DC: D7610019 0001100C
	v_writelane_b32 v28, s28, 8                                // 0000000061E4: D761001C 0001101C
	v_writelane_b32 v25, s13, 9                                // 0000000061EC: D7610019 0001120D
	v_writelane_b32 v28, s29, 9                                // 0000000061F4: D761001C 0001121D
	v_writelane_b32 v25, s14, 10                               // 0000000061FC: D7610019 0001140E
	v_writelane_b32 v28, s30, 10                               // 000000006204: D761001C 0001141E
	v_writelane_b32 v25, s15, 11                               // 00000000620C: D7610019 0001160F
	s_load_b512 s[0:15], s[16:17], 0x0                         // 000000006214: F4008008 F8000000
	s_mov_b32 s17, 0                                           // 00000000621C: BE910080
	v_writelane_b32 v28, s31, 11                               // 000000006220: D761001C 0001161F
	s_load_b256 s[24:31], s[22:23], 0x0                        // 000000006228: F400660B F8000000
	s_mov_b32 s16, s104                                        // 000000006230: BE900068
	s_wait_kmcnt 0x0                                           // 000000006234: BFC70000
	v_writelane_b32 v25, s0, 20                                // 000000006238: D7610019 00012800
	v_writelane_b32 v28, s12, 0                                // 000000006240: D761001C 0001000C
	v_writelane_b32 v25, s1, 21                                // 000000006248: D7610019 00012A01
	v_writelane_b32 v28, s13, 1                                // 000000006250: D761001C 0001020D
	v_writelane_b32 v25, s2, 22                                // 000000006258: D7610019 00012C02
	v_writelane_b32 v28, s14, 2                                // 000000006260: D761001C 0001040E
	v_writelane_b32 v25, s3, 23                                // 000000006268: D7610019 00012E03
	v_writelane_b32 v28, s15, 3                                // 000000006270: D761001C 0001060F
	v_writelane_b32 v25, s4, 24                                // 000000006278: D7610019 00013004
	v_writelane_b32 v25, s5, 25                                // 000000006280: D7610019 00013205
	v_writelane_b32 v25, s6, 26                                // 000000006288: D7610019 00013406
	v_writelane_b32 v25, s7, 27                                // 000000006290: D7610019 00013607
	v_writelane_b32 v25, s8, 28                                // 000000006298: D7610019 00013808
	v_writelane_b32 v25, s9, 29                                // 0000000062A0: D7610019 00013A09
	v_writelane_b32 v25, s10, 30                               // 0000000062A8: D7610019 00013C0A
	v_writelane_b32 v25, s11, 31                               // 0000000062B0: D7610019 00013E0B
	v_writelane_b32 v25, s24, 12                               // 0000000062B8: D7610019 00011818
	v_writelane_b32 v25, s25, 13                               // 0000000062C0: D7610019 00011A19
	v_writelane_b32 v25, s26, 14                               // 0000000062C8: D7610019 00011C1A
	v_writelane_b32 v25, s27, 15                               // 0000000062D0: D7610019 00011E1B
	v_writelane_b32 v25, s28, 16                               // 0000000062D8: D7610019 0001201C
	v_writelane_b32 v25, s29, 17                               // 0000000062E0: D7610019 0001221D
	v_writelane_b32 v25, s30, 18                               // 0000000062E8: D7610019 0001241E
	v_writelane_b32 v25, s31, 19                               // 0000000062F0: D7610019 0001261F
	s_branch 853                                               // 0000000062F8: BFA00355 <r_3_3_3_8_8_8+0x5a50>
	v_writelane_b32 v23, s36, 2                                // 0000000062FC: D7610017 00010424
	s_mov_b32 s6, 0                                            // 000000006304: BE860080
	v_writelane_b32 v23, s37, 3                                // 000000006308: D7610017 00010625
	v_writelane_b32 v23, s38, 4                                // 000000006310: D7610017 00010826
	v_writelane_b32 v23, s39, 5                                // 000000006318: D7610017 00010A27
	v_writelane_b32 v23, s40, 6                                // 000000006320: D7610017 00010C28
	v_writelane_b32 v23, s41, 7                                // 000000006328: D7610017 00010E29
	v_writelane_b32 v23, s42, 8                                // 000000006330: D7610017 0001102A
	v_writelane_b32 v23, s43, 9                                // 000000006338: D7610017 0001122B
	v_writelane_b32 v23, s44, 10                               // 000000006340: D7610017 0001142C
	v_writelane_b32 v23, s45, 11                               // 000000006348: D7610017 0001162D
	v_writelane_b32 v23, s46, 12                               // 000000006350: D7610017 0001182E
	v_writelane_b32 v23, s47, 13                               // 000000006358: D7610017 00011A2F
	v_writelane_b32 v23, s48, 14                               // 000000006360: D7610017 00011C30
	v_writelane_b32 v23, s49, 15                               // 000000006368: D7610017 00011E31
	v_writelane_b32 v23, s50, 16                               // 000000006370: D7610017 00012032
	v_writelane_b32 v23, s51, 17                               // 000000006378: D7610017 00012233
	v_writelane_b32 v23, s40, 18                               // 000000006380: D7610017 00012428
	v_writelane_b32 v23, s41, 19                               // 000000006388: D7610017 00012629
	v_writelane_b32 v23, s42, 20                               // 000000006390: D7610017 0001282A
	v_writelane_b32 v23, s43, 21                               // 000000006398: D7610017 00012A2B
	v_writelane_b32 v23, s44, 22                               // 0000000063A0: D7610017 00012C2C
	v_writelane_b32 v23, s45, 23                               // 0000000063A8: D7610017 00012E2D
	v_writelane_b32 v23, s46, 24                               // 0000000063B0: D7610017 0001302E
	v_writelane_b32 v23, s47, 25                               // 0000000063B8: D7610017 0001322F
	v_writelane_b32 v23, s8, 26                                // 0000000063C0: D7610017 00013408
	v_writelane_b32 v25, s14, 0                                // 0000000063C8: D7610019 0001000E
	v_writelane_b32 v23, s9, 27                                // 0000000063D0: D7610017 00013609
	v_writelane_b32 v25, s15, 1                                // 0000000063D8: D7610019 0001020F
	v_writelane_b32 v23, s10, 28                               // 0000000063E0: D7610017 0001380A
	v_writelane_b32 v25, s16, 2                                // 0000000063E8: D7610019 00010410
	v_writelane_b32 v23, s11, 29                               // 0000000063F0: D7610017 00013A0B
	v_writelane_b32 v25, s17, 3                                // 0000000063F8: D7610019 00010611
	v_writelane_b32 v23, s12, 30                               // 000000006400: D7610017 00013C0C
	v_writelane_b32 v25, s18, 4                                // 000000006408: D7610019 00010812
	v_writelane_b32 v23, s13, 31                               // 000000006410: D7610017 00013E0D
	v_writelane_b32 v25, s19, 5                                // 000000006418: D7610019 00010A13
	v_writelane_b32 v25, s20, 6                                // 000000006420: D7610019 00010C14
	v_writelane_b32 v25, s21, 7                                // 000000006428: D7610019 00010E15
	v_writelane_b32 v25, s22, 8                                // 000000006430: D7610019 00011016
	v_writelane_b32 v25, s23, 9                                // 000000006438: D7610019 00011217
	v_writelane_b32 v25, s8, 10                                // 000000006440: D7610019 00011408
	v_writelane_b32 v25, s9, 11                                // 000000006448: D7610019 00011609
	v_writelane_b32 v25, s10, 12                               // 000000006450: D7610019 0001180A
	v_writelane_b32 v25, s11, 13                               // 000000006458: D7610019 00011A0B
	v_writelane_b32 v25, s12, 14                               // 000000006460: D7610019 00011C0C
	v_writelane_b32 v25, s13, 15                               // 000000006468: D7610019 00011E0D
	v_writelane_b32 v25, s14, 16                               // 000000006470: D7610019 0001200E
	v_writelane_b32 v25, s15, 17                               // 000000006478: D7610019 0001220F
	v_writelane_b32 v25, s12, 18                               // 000000006480: D7610019 0001240C
	v_writelane_b32 v25, s13, 19                               // 000000006488: D7610019 0001260D
	v_writelane_b32 v25, s14, 20                               // 000000006490: D7610019 0001280E
	v_writelane_b32 v25, s15, 21                               // 000000006498: D7610019 00012A0F
	v_writelane_b32 v25, s16, 22                               // 0000000064A0: D7610019 00012C10
	v_writelane_b32 v25, s17, 23                               // 0000000064A8: D7610019 00012E11
	v_writelane_b32 v25, s18, 24                               // 0000000064B0: D7610019 00013012
	v_writelane_b32 v25, s19, 25                               // 0000000064B8: D7610019 00013213
	s_cbranch_execnz 61464                                     // 0000000064C0: BFA6F018 <r_3_3_3_8_8_8+0xf24>
	s_branch 61638                                             // 0000000064C4: BFA0F0C6 <r_3_3_3_8_8_8+0x11e0>
	v_writelane_b32 v28, s36, 4                                // 0000000064C8: D761001C 00010824
	s_mov_b32 s8, 0                                            // 0000000064D0: BE880080
	v_writelane_b32 v28, s37, 5                                // 0000000064D4: D761001C 00010A25
	v_writelane_b32 v28, s38, 6                                // 0000000064DC: D761001C 00010C26
	v_writelane_b32 v28, s39, 7                                // 0000000064E4: D761001C 00010E27
	v_writelane_b32 v28, s40, 8                                // 0000000064EC: D761001C 00011028
	v_writelane_b32 v28, s41, 9                                // 0000000064F4: D761001C 00011229
	v_writelane_b32 v28, s42, 10                               // 0000000064FC: D761001C 0001142A
	v_writelane_b32 v28, s43, 11                               // 000000006504: D761001C 0001162B
	v_writelane_b32 v28, s44, 12                               // 00000000650C: D761001C 0001182C
	v_writelane_b32 v28, s45, 13                               // 000000006514: D761001C 00011A2D
	v_writelane_b32 v28, s46, 14                               // 00000000651C: D761001C 00011C2E
	v_writelane_b32 v28, s47, 15                               // 000000006524: D761001C 00011E2F
	v_writelane_b32 v28, s48, 16                               // 00000000652C: D761001C 00012030
	v_writelane_b32 v28, s49, 17                               // 000000006534: D761001C 00012231
	v_writelane_b32 v28, s50, 18                               // 00000000653C: D761001C 00012432
	v_writelane_b32 v28, s51, 19                               // 000000006544: D761001C 00012633
	v_writelane_b32 v28, s40, 20                               // 00000000654C: D761001C 00012828
	v_writelane_b32 v28, s41, 21                               // 000000006554: D761001C 00012A29
	v_writelane_b32 v28, s42, 22                               // 00000000655C: D761001C 00012C2A
	v_writelane_b32 v28, s43, 23                               // 000000006564: D761001C 00012E2B
	v_writelane_b32 v28, s44, 24                               // 00000000656C: D761001C 0001302C
	v_writelane_b32 v28, s45, 25                               // 000000006574: D761001C 0001322D
	v_writelane_b32 v28, s46, 26                               // 00000000657C: D761001C 0001342E
	v_writelane_b32 v28, s47, 27                               // 000000006584: D761001C 0001362F
	v_writelane_b32 v28, s36, 28                               // 00000000658C: D761001C 00013824
	v_writelane_b32 v28, s37, 29                               // 000000006594: D761001C 00013A25
	v_writelane_b32 v28, s38, 30                               // 00000000659C: D761001C 00013C26
	v_writelane_b32 v28, s39, 31                               // 0000000065A4: D761001C 00013E27
	s_or_saveexec_b32 s105, -1                                 // 0000000065AC: BEE922C1
	scratch_store_b32 off, v28, off offset:16                  // 0000000065B0: ED06807C 0E000000 00001000
	s_wait_alu 0xfffe                                          // 0000000065BC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000065C0: BEFE0069
	v_writelane_b32 v21, s40, 0                                // 0000000065C4: D7610015 00010028
	v_writelane_b32 v21, s41, 1                                // 0000000065CC: D7610015 00010229
	v_writelane_b32 v21, s42, 2                                // 0000000065D4: D7610015 0001042A
	v_writelane_b32 v21, s43, 3                                // 0000000065DC: D7610015 0001062B
	v_writelane_b32 v21, s44, 4                                // 0000000065E4: D7610015 0001082C
	v_writelane_b32 v21, s45, 5                                // 0000000065EC: D7610015 00010A2D
	v_writelane_b32 v21, s46, 6                                // 0000000065F4: D7610015 00010C2E
	v_writelane_b32 v21, s47, 7                                // 0000000065FC: D7610015 00010E2F
	v_writelane_b32 v21, s48, 8                                // 000000006604: D7610015 00011030
	v_writelane_b32 v21, s49, 9                                // 00000000660C: D7610015 00011231
	v_writelane_b32 v21, s50, 10                               // 000000006614: D7610015 00011432
	v_writelane_b32 v21, s51, 11                               // 00000000661C: D7610015 00011633
	v_writelane_b32 v21, s40, 12                               // 000000006624: D7610015 00011828
	v_writelane_b32 v21, s41, 13                               // 00000000662C: D7610015 00011A29
	v_writelane_b32 v21, s42, 14                               // 000000006634: D7610015 00011C2A
	v_writelane_b32 v21, s43, 15                               // 00000000663C: D7610015 00011E2B
	v_writelane_b32 v21, s44, 16                               // 000000006644: D7610015 0001202C
	v_writelane_b32 v21, s45, 17                               // 00000000664C: D7610015 0001222D
	v_writelane_b32 v21, s46, 18                               // 000000006654: D7610015 0001242E
	v_writelane_b32 v21, s47, 19                               // 00000000665C: D7610015 0001262F
	v_writelane_b32 v21, s40, 20                               // 000000006664: D7610015 00012828
	v_writelane_b32 v21, s41, 21                               // 00000000666C: D7610015 00012A29
	v_writelane_b32 v21, s42, 22                               // 000000006674: D7610015 00012C2A
	v_writelane_b32 v21, s43, 23                               // 00000000667C: D7610015 00012E2B
	v_writelane_b32 v21, s44, 24                               // 000000006684: D7610015 0001302C
	v_writelane_b32 v21, s45, 25                               // 00000000668C: D7610015 0001322D
	v_writelane_b32 v21, s46, 26                               // 000000006694: D7610015 0001342E
	v_writelane_b32 v21, s47, 27                               // 00000000669C: D7610015 0001362F
	v_writelane_b32 v21, s40, 28                               // 0000000066A4: D7610015 00013828
	v_writelane_b32 v26, s44, 0                                // 0000000066AC: D761001A 0001002C
	v_writelane_b32 v21, s41, 29                               // 0000000066B4: D7610015 00013A29
	v_writelane_b32 v26, s45, 1                                // 0000000066BC: D761001A 0001022D
	v_writelane_b32 v21, s42, 30                               // 0000000066C4: D7610015 00013C2A
	v_writelane_b32 v26, s46, 2                                // 0000000066CC: D761001A 0001042E
	v_writelane_b32 v21, s43, 31                               // 0000000066D4: D7610015 00013E2B
	v_writelane_b32 v26, s47, 3                                // 0000000066DC: D761001A 0001062F
	v_writelane_b32 v26, s36, 4                                // 0000000066E4: D761001A 00010824
	v_writelane_b32 v26, s37, 5                                // 0000000066EC: D761001A 00010A25
	v_writelane_b32 v26, s38, 6                                // 0000000066F4: D761001A 00010C26
	v_writelane_b32 v26, s39, 7                                // 0000000066FC: D761001A 00010E27
	v_writelane_b32 v26, s40, 8                                // 000000006704: D761001A 00011028
	v_writelane_b32 v26, s41, 9                                // 00000000670C: D761001A 00011229
	v_writelane_b32 v26, s42, 10                               // 000000006714: D761001A 0001142A
	v_writelane_b32 v26, s43, 11                               // 00000000671C: D761001A 0001162B
	s_or_saveexec_b32 s105, -1                                 // 000000006724: BEE922C1
	s_wait_alu 0xfffe                                          // 000000006728: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000672C: BEFE0069
	s_cbranch_execnz 61966                                     // 000000006730: BFA6F20E <r_3_3_3_8_8_8+0x196c>
	s_branch 62193                                             // 000000006734: BFA0F2F1 <r_3_3_3_8_8_8+0x1cfc>
	s_mov_b32 s38, 0                                           // 000000006738: BEA60080
	s_mov_b32 s7, -1                                           // 00000000673C: BE8700C1
	s_wait_alu 0xfffe                                          // 000000006740: BF88FFFE
	v_writelane_b32 v26, s36, 12                               // 000000006744: D761001A 00011824
	v_writelane_b32 v26, s37, 13                               // 00000000674C: D761001A 00011A25
	v_writelane_b32 v26, s38, 14                               // 000000006754: D761001A 00011C26
	v_writelane_b32 v26, s39, 15                               // 00000000675C: D761001A 00011E27
	v_writelane_b32 v26, s40, 16                               // 000000006764: D761001A 00012028
	v_writelane_b32 v26, s41, 17                               // 00000000676C: D761001A 00012229
	v_writelane_b32 v26, s42, 18                               // 000000006774: D761001A 0001242A
	v_writelane_b32 v26, s43, 19                               // 00000000677C: D761001A 0001262B
	v_writelane_b32 v26, s44, 20                               // 000000006784: D761001A 0001282C
	v_writelane_b32 v26, s45, 21                               // 00000000678C: D761001A 00012A2D
	v_writelane_b32 v26, s46, 22                               // 000000006794: D761001A 00012C2E
	v_writelane_b32 v26, s47, 23                               // 00000000679C: D761001A 00012E2F
	v_writelane_b32 v26, s48, 24                               // 0000000067A4: D761001A 00013030
	v_writelane_b32 v26, s49, 25                               // 0000000067AC: D761001A 00013231
	v_writelane_b32 v26, s50, 26                               // 0000000067B4: D761001A 00013432
	v_writelane_b32 v26, s51, 27                               // 0000000067BC: D761001A 00013633
	v_writelane_b32 v26, s40, 28                               // 0000000067C4: D761001A 00013828
	v_writelane_b32 v26, s41, 29                               // 0000000067CC: D761001A 00013A29
	v_writelane_b32 v26, s42, 30                               // 0000000067D4: D761001A 00013C2A
	v_writelane_b32 v26, s43, 31                               // 0000000067DC: D761001A 00013E2B
	s_or_saveexec_b32 s105, -1                                 // 0000000067E4: BEE922C1
	scratch_store_b32 off, v26, off offset:24                  // 0000000067E8: ED06807C 0D000000 00001800
	s_wait_alu 0xfffe                                          // 0000000067F4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000067F8: BEFE0069
	v_writelane_b32 v29, s44, 0                                // 0000000067FC: D761001D 0001002C
	v_writelane_b32 v29, s45, 1                                // 000000006804: D761001D 0001022D
	v_writelane_b32 v29, s46, 2                                // 00000000680C: D761001D 0001042E
	v_writelane_b32 v29, s47, 3                                // 000000006814: D761001D 0001062F
	v_writelane_b32 v29, s0, 4                                 // 00000000681C: D761001D 00010800
	v_writelane_b32 v29, s1, 5                                 // 000000006824: D761001D 00010A01
	v_writelane_b32 v29, s2, 6                                 // 00000000682C: D761001D 00010C02
	v_writelane_b32 v29, s3, 7                                 // 000000006834: D761001D 00010E03
	v_writelane_b32 v29, s4, 8                                 // 00000000683C: D761001D 00011004
	v_writelane_b32 v29, s5, 9                                 // 000000006844: D761001D 00011205
	v_writelane_b32 v29, s6, 10                                // 00000000684C: D761001D 00011406
	v_writelane_b32 v29, s7, 11                                // 000000006854: D761001D 00011607
	v_writelane_b32 v29, s8, 12                                // 00000000685C: D761001D 00011808
	v_writelane_b32 v29, s9, 13                                // 000000006864: D761001D 00011A09
	v_writelane_b32 v29, s10, 14                               // 00000000686C: D761001D 00011C0A
	v_writelane_b32 v29, s11, 15                               // 000000006874: D761001D 00011E0B
	v_writelane_b32 v29, s12, 16                               // 00000000687C: D761001D 0001200C
	v_writelane_b32 v29, s13, 17                               // 000000006884: D761001D 0001220D
	v_writelane_b32 v29, s14, 18                               // 00000000688C: D761001D 0001240E
	v_writelane_b32 v29, s15, 19                               // 000000006894: D761001D 0001260F
	v_writelane_b32 v29, s40, 20                               // 00000000689C: D761001D 00012828
	v_writelane_b32 v29, s41, 21                               // 0000000068A4: D761001D 00012A29
	v_writelane_b32 v29, s42, 22                               // 0000000068AC: D761001D 00012C2A
	v_writelane_b32 v29, s43, 23                               // 0000000068B4: D761001D 00012E2B
	v_writelane_b32 v29, s44, 24                               // 0000000068BC: D761001D 0001302C
	v_writelane_b32 v29, s45, 25                               // 0000000068C4: D761001D 0001322D
	v_writelane_b32 v29, s46, 26                               // 0000000068CC: D761001D 0001342E
	v_writelane_b32 v29, s47, 27                               // 0000000068D4: D761001D 0001362F
	v_writelane_b32 v29, s40, 28                               // 0000000068DC: D761001D 00013828
	v_writelane_b32 v29, s41, 29                               // 0000000068E4: D761001D 00013A29
	v_writelane_b32 v29, s42, 30                               // 0000000068EC: D761001D 00013C2A
	v_writelane_b32 v29, s43, 31                               // 0000000068F4: D761001D 00013E2B
	s_or_saveexec_b32 s105, -1                                 // 0000000068FC: BEE922C1
	scratch_store_b32 off, v29, off offset:28                  // 000000006900: ED06807C 0E800000 00001C00
	s_wait_alu 0xfffe                                          // 00000000690C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000006910: BEFE0069
	v_writelane_b32 v24, s44, 0                                // 000000006914: D7610018 0001002C
	v_writelane_b32 v24, s45, 1                                // 00000000691C: D7610018 0001022D
	v_writelane_b32 v24, s46, 2                                // 000000006924: D7610018 0001042E
	v_writelane_b32 v24, s47, 3                                // 00000000692C: D7610018 0001062F
	v_writelane_b32 v24, s40, 4                                // 000000006934: D7610018 00010828
	v_writelane_b32 v24, s41, 5                                // 00000000693C: D7610018 00010A29
	v_writelane_b32 v24, s42, 6                                // 000000006944: D7610018 00010C2A
	v_writelane_b32 v24, s43, 7                                // 00000000694C: D7610018 00010E2B
	v_writelane_b32 v24, s44, 8                                // 000000006954: D7610018 0001102C
	v_writelane_b32 v24, s45, 9                                // 00000000695C: D7610018 0001122D
	v_writelane_b32 v24, s46, 10                               // 000000006964: D7610018 0001142E
	v_writelane_b32 v24, s47, 11                               // 00000000696C: D7610018 0001162F
	v_writelane_b32 v24, s0, 12                                // 000000006974: D7610018 00011800
	v_writelane_b32 v24, s1, 13                                // 00000000697C: D7610018 00011A01
	v_writelane_b32 v24, s2, 14                                // 000000006984: D7610018 00011C02
	v_writelane_b32 v24, s3, 15                                // 00000000698C: D7610018 00011E03
	v_writelane_b32 v24, s4, 16                                // 000000006994: D7610018 00012004
	v_writelane_b32 v24, s5, 17                                // 00000000699C: D7610018 00012205
	v_writelane_b32 v24, s6, 18                                // 0000000069A4: D7610018 00012406
	v_writelane_b32 v24, s7, 19                                // 0000000069AC: D7610018 00012607
	s_or_saveexec_b32 s105, -1                                 // 0000000069B4: BEE922C1
	s_wait_alu 0xfffe                                          // 0000000069B8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000069BC: BEFE0069
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000069C0: BF870009
	s_and_b32 vcc_lo, exec_lo, s7                              // 0000000069C4: 8B6A077E
	s_cbranch_vccnz 62798                                      // 0000000069C8: BFA4F54E <r_3_3_3_8_8_8+0x2904>
	s_branch 63038                                             // 0000000069CC: BFA0F63E <r_3_3_3_8_8_8+0x2cc8>
	s_mov_b32 s7, -1                                           // 0000000069D0: BE8700C1
	s_mov_b32 s6, 0                                            // 0000000069D4: BE860080
	v_writelane_b32 v24, s0, 4                                 // 0000000069D8: D7610018 00010800
	v_writelane_b32 v24, s1, 5                                 // 0000000069E0: D7610018 00010A01
	s_wait_alu 0xfffe                                          // 0000000069E8: BF88FFFE
	v_writelane_b32 v24, s2, 6                                 // 0000000069EC: D7610018 00010C02
	v_writelane_b32 v24, s3, 7                                 // 0000000069F4: D7610018 00010E03
	v_writelane_b32 v24, s4, 8                                 // 0000000069FC: D7610018 00011004
	v_writelane_b32 v24, s5, 9                                 // 000000006A04: D7610018 00011205
	v_writelane_b32 v24, s6, 10                                // 000000006A0C: D7610018 00011406
	v_writelane_b32 v24, s7, 11                                // 000000006A14: D7610018 00011607
	v_writelane_b32 v24, s8, 12                                // 000000006A1C: D7610018 00011808
	v_writelane_b32 v24, s9, 13                                // 000000006A24: D7610018 00011A09
	v_writelane_b32 v24, s10, 14                               // 000000006A2C: D7610018 00011C0A
	v_writelane_b32 v24, s11, 15                               // 000000006A34: D7610018 00011E0B
	v_writelane_b32 v24, s12, 16                               // 000000006A3C: D7610018 0001200C
	v_writelane_b32 v24, s13, 17                               // 000000006A44: D7610018 0001220D
	v_writelane_b32 v24, s14, 18                               // 000000006A4C: D7610018 0001240E
	v_writelane_b32 v24, s15, 19                               // 000000006A54: D7610018 0001260F
	v_writelane_b32 v24, s12, 20                               // 000000006A5C: D7610018 0001280C
	v_writelane_b32 v24, s13, 21                               // 000000006A64: D7610018 00012A0D
	v_writelane_b32 v24, s14, 22                               // 000000006A6C: D7610018 00012C0E
	v_writelane_b32 v24, s15, 23                               // 000000006A74: D7610018 00012E0F
	v_writelane_b32 v24, s16, 24                               // 000000006A7C: D7610018 00013010
	v_writelane_b32 v24, s17, 25                               // 000000006A84: D7610018 00013211
	v_writelane_b32 v24, s18, 26                               // 000000006A8C: D7610018 00013412
	v_writelane_b32 v24, s19, 27                               // 000000006A94: D7610018 00013613
	v_writelane_b32 v24, s4, 28                                // 000000006A9C: D7610018 00013804
	v_writelane_b32 v24, s5, 29                                // 000000006AA4: D7610018 00013A05
	v_writelane_b32 v24, s6, 30                                // 000000006AAC: D7610018 00013C06
	v_writelane_b32 v24, s7, 31                                // 000000006AB4: D7610018 00013E07
	s_or_saveexec_b32 s105, -1                                 // 000000006ABC: BEE922C1
	scratch_store_b32 off, v24, off offset:32                  // 000000006AC0: ED06807C 0C000000 00002000
	s_wait_alu 0xfffe                                          // 000000006ACC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000006AD0: BEFE0069
	v_writelane_b32 v29, s8, 0                                 // 000000006AD4: D761001D 00010008
	v_writelane_b32 v29, s9, 1                                 // 000000006ADC: D761001D 00010209
	v_writelane_b32 v29, s10, 2                                // 000000006AE4: D761001D 0001040A
	v_writelane_b32 v29, s11, 3                                // 000000006AEC: D761001D 0001060B
	v_writelane_b32 v29, s12, 4                                // 000000006AF4: D761001D 0001080C
	v_writelane_b32 v29, s13, 5                                // 000000006AFC: D761001D 00010A0D
	v_writelane_b32 v29, s14, 6                                // 000000006B04: D761001D 00010C0E
	v_writelane_b32 v29, s15, 7                                // 000000006B0C: D761001D 00010E0F
	v_writelane_b32 v29, s16, 8                                // 000000006B14: D761001D 00011010
	v_writelane_b32 v29, s17, 9                                // 000000006B1C: D761001D 00011211
	v_writelane_b32 v29, s18, 10                               // 000000006B24: D761001D 00011412
	v_writelane_b32 v29, s19, 11                               // 000000006B2C: D761001D 00011613
	v_writelane_b32 v29, s12, 12                               // 000000006B34: D761001D 0001180C
	v_writelane_b32 v29, s13, 13                               // 000000006B3C: D761001D 00011A0D
	v_writelane_b32 v29, s14, 14                               // 000000006B44: D761001D 00011C0E
	v_writelane_b32 v29, s15, 15                               // 000000006B4C: D761001D 00011E0F
	v_writelane_b32 v29, s16, 16                               // 000000006B54: D761001D 00012010
	v_writelane_b32 v29, s17, 17                               // 000000006B5C: D761001D 00012211
	v_writelane_b32 v29, s18, 18                               // 000000006B64: D761001D 00012412
	v_writelane_b32 v29, s19, 19                               // 000000006B6C: D761001D 00012613
	v_writelane_b32 v29, s8, 20                                // 000000006B74: D761001D 00012808
	v_writelane_b32 v29, s9, 21                                // 000000006B7C: D761001D 00012A09
	v_writelane_b32 v29, s10, 22                               // 000000006B84: D761001D 00012C0A
	v_writelane_b32 v29, s11, 23                               // 000000006B8C: D761001D 00012E0B
	v_writelane_b32 v29, s12, 24                               // 000000006B94: D761001D 0001300C
	v_writelane_b32 v29, s13, 25                               // 000000006B9C: D761001D 0001320D
	v_writelane_b32 v29, s14, 26                               // 000000006BA4: D761001D 0001340E
	v_writelane_b32 v29, s15, 27                               // 000000006BAC: D761001D 0001360F
	v_writelane_b32 v29, s12, 28                               // 000000006BB4: D761001D 0001380C
	v_writelane_b32 v29, s13, 29                               // 000000006BBC: D761001D 00013A0D
	v_writelane_b32 v29, s14, 30                               // 000000006BC4: D761001D 00013C0E
	v_writelane_b32 v29, s15, 31                               // 000000006BCC: D761001D 00013E0F
	s_or_saveexec_b32 s105, -1                                 // 000000006BD4: BEE922C1
	scratch_store_b32 off, v29, off offset:36                  // 000000006BD8: ED06807C 0E800000 00002400
	s_wait_alu 0xfffe                                          // 000000006BE4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000006BE8: BEFE0069
	v_writelane_b32 v26, s16, 0                                // 000000006BEC: D761001A 00010010
	v_writelane_b32 v26, s17, 1                                // 000000006BF4: D761001A 00010211
	v_writelane_b32 v26, s18, 2                                // 000000006BFC: D761001A 00010412
	v_writelane_b32 v26, s19, 3                                // 000000006C04: D761001A 00010613
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000006C0C: 8B6A077E
	s_cbranch_vccnz 63447                                      // 000000006C10: BFA4F7D7 <r_3_3_3_8_8_8+0x3570>
	s_branch 63665                                             // 000000006C14: BFA0F8B1 <r_3_3_3_8_8_8+0x38dc>
	s_mov_b32 s4, 0                                            // 000000006C18: BE840080
	s_mov_b32 s5, -1                                           // 000000006C1C: BE8500C1
	s_wait_alu 0xfffe                                          // 000000006C20: BF88FFFE
	v_writelane_b32 v26, s4, 12                                // 000000006C24: D761001A 00011804
	v_writelane_b32 v26, s5, 13                                // 000000006C2C: D761001A 00011A05
	v_writelane_b32 v26, s6, 14                                // 000000006C34: D761001A 00011C06
	v_writelane_b32 v26, s7, 15                                // 000000006C3C: D761001A 00011E07
	v_writelane_b32 v26, s8, 16                                // 000000006C44: D761001A 00012008
	v_writelane_b32 v26, s9, 17                                // 000000006C4C: D761001A 00012209
	v_writelane_b32 v26, s10, 18                               // 000000006C54: D761001A 0001240A
	v_writelane_b32 v26, s11, 19                               // 000000006C5C: D761001A 0001260B
	v_writelane_b32 v26, s12, 20                               // 000000006C64: D761001A 0001280C
	v_writelane_b32 v26, s13, 21                               // 000000006C6C: D761001A 00012A0D
	v_writelane_b32 v26, s14, 22                               // 000000006C74: D761001A 00012C0E
	v_writelane_b32 v26, s15, 23                               // 000000006C7C: D761001A 00012E0F
	v_writelane_b32 v26, s16, 24                               // 000000006C84: D761001A 00013010
	v_writelane_b32 v26, s17, 25                               // 000000006C8C: D761001A 00013211
	v_writelane_b32 v26, s18, 26                               // 000000006C94: D761001A 00013412
	v_writelane_b32 v26, s19, 27                               // 000000006C9C: D761001A 00013613
	v_writelane_b32 v26, s4, 28                                // 000000006CA4: D761001A 00013804
	v_writelane_b32 v29, s8, 0                                 // 000000006CAC: D761001D 00010008
	v_writelane_b32 v26, s5, 29                                // 000000006CB4: D761001A 00013A05
	v_writelane_b32 v29, s9, 1                                 // 000000006CBC: D761001D 00010209
	v_writelane_b32 v26, s6, 30                                // 000000006CC4: D761001A 00013C06
	v_writelane_b32 v29, s10, 2                                // 000000006CCC: D761001D 0001040A
	v_writelane_b32 v26, s7, 31                                // 000000006CD4: D761001A 00013E07
	v_writelane_b32 v29, s11, 3                                // 000000006CDC: D761001D 0001060B
	v_writelane_b32 v29, s0, 4                                 // 000000006CE4: D761001D 00010800
	v_writelane_b32 v29, s1, 5                                 // 000000006CEC: D761001D 00010A01
	v_writelane_b32 v29, s2, 6                                 // 000000006CF4: D761001D 00010C02
	v_writelane_b32 v29, s3, 7                                 // 000000006CFC: D761001D 00010E03
	v_writelane_b32 v29, s4, 8                                 // 000000006D04: D761001D 00011004
	v_writelane_b32 v29, s5, 9                                 // 000000006D0C: D761001D 00011205
	v_writelane_b32 v29, s6, 10                                // 000000006D14: D761001D 00011406
	v_writelane_b32 v29, s7, 11                                // 000000006D1C: D761001D 00011607
	v_writelane_b32 v29, s8, 12                                // 000000006D24: D761001D 00011808
	v_writelane_b32 v29, s9, 13                                // 000000006D2C: D761001D 00011A09
	v_writelane_b32 v29, s10, 14                               // 000000006D34: D761001D 00011C0A
	v_writelane_b32 v29, s11, 15                               // 000000006D3C: D761001D 00011E0B
	v_writelane_b32 v29, s12, 16                               // 000000006D44: D761001D 0001200C
	v_writelane_b32 v29, s13, 17                               // 000000006D4C: D761001D 0001220D
	v_writelane_b32 v29, s14, 18                               // 000000006D54: D761001D 0001240E
	v_writelane_b32 v29, s15, 19                               // 000000006D5C: D761001D 0001260F
	v_writelane_b32 v29, s4, 20                                // 000000006D64: D761001D 00012804
	v_writelane_b32 v29, s5, 21                                // 000000006D6C: D761001D 00012A05
	v_writelane_b32 v29, s6, 22                                // 000000006D74: D761001D 00012C06
	v_writelane_b32 v29, s7, 23                                // 000000006D7C: D761001D 00012E07
	v_writelane_b32 v29, s8, 24                                // 000000006D84: D761001D 00013008
	v_writelane_b32 v29, s9, 25                                // 000000006D8C: D761001D 00013209
	v_writelane_b32 v29, s10, 26                               // 000000006D94: D761001D 0001340A
	v_writelane_b32 v29, s11, 27                               // 000000006D9C: D761001D 0001360B
	v_writelane_b32 v29, s12, 28                               // 000000006DA4: D761001D 0001380C
	v_writelane_b32 v29, s13, 29                               // 000000006DAC: D761001D 00013A0D
	v_writelane_b32 v29, s14, 30                               // 000000006DB4: D761001D 00013C0E
	v_writelane_b32 v29, s15, 31                               // 000000006DBC: D761001D 00013E0F
	s_or_saveexec_b32 s105, -1                                 // 000000006DC4: BEE922C1
	scratch_store_b32 off, v29, off offset:88                  // 000000006DC8: ED06807C 0E800000 00005800
	s_wait_alu 0xfffe                                          // 000000006DD4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000006DD8: BEFE0069
	v_writelane_b32 v22, s16, 0                                // 000000006DDC: D7610016 00010010
	v_writelane_b32 v22, s17, 1                                // 000000006DE4: D7610016 00010211
	v_writelane_b32 v22, s18, 2                                // 000000006DEC: D7610016 00010412
	v_writelane_b32 v22, s19, 3                                // 000000006DF4: D7610016 00010613
	v_writelane_b32 v22, s4, 4                                 // 000000006DFC: D7610016 00010804
	v_writelane_b32 v22, s5, 5                                 // 000000006E04: D7610016 00010A05
	v_writelane_b32 v22, s6, 6                                 // 000000006E0C: D7610016 00010C06
	v_writelane_b32 v22, s7, 7                                 // 000000006E14: D7610016 00010E07
	v_writelane_b32 v22, s8, 8                                 // 000000006E1C: D7610016 00011008
	v_writelane_b32 v22, s9, 9                                 // 000000006E24: D7610016 00011209
	v_writelane_b32 v22, s10, 10                               // 000000006E2C: D7610016 0001140A
	v_writelane_b32 v22, s11, 11                               // 000000006E34: D7610016 0001160B
	s_and_b32 vcc_lo, exec_lo, s5                              // 000000006E3C: 8B6A057E
	s_cbranch_vccnz 64025                                      // 000000006E40: BFA4FA19 <r_3_3_3_8_8_8+0x40a8>
	s_branch 64230                                             // 000000006E44: BFA0FAE6 <r_3_3_3_8_8_8+0x43e0>
	s_mov_b32 s17, -1                                          // 000000006E48: BE9100C1
	v_writelane_b32 v20, s0, 28                                // 000000006E4C: D7610014 00013800
	v_writelane_b32 v25, s4, 0                                 // 000000006E54: D7610019 00010004
	s_mov_b32 s16, 0                                           // 000000006E5C: BE900080
	v_writelane_b32 v20, s1, 29                                // 000000006E60: D7610014 00013A01
	v_writelane_b32 v25, s5, 1                                 // 000000006E68: D7610019 00010205
	v_writelane_b32 v20, s2, 30                                // 000000006E70: D7610014 00013C02
	v_writelane_b32 v25, s6, 2                                 // 000000006E78: D7610019 00010406
	v_writelane_b32 v20, s3, 31                                // 000000006E80: D7610014 00013E03
	v_writelane_b32 v25, s7, 3                                 // 000000006E88: D7610019 00010607
	v_writelane_b32 v25, s8, 4                                 // 000000006E90: D7610019 00010808
	v_writelane_b32 v25, s9, 5                                 // 000000006E98: D7610019 00010A09
	v_writelane_b32 v25, s10, 6                                // 000000006EA0: D7610019 00010C0A
	v_writelane_b32 v25, s11, 7                                // 000000006EA8: D7610019 00010E0B
	v_writelane_b32 v25, s12, 8                                // 000000006EB0: D7610019 0001100C
	v_writelane_b32 v25, s13, 9                                // 000000006EB8: D7610019 0001120D
	v_writelane_b32 v25, s14, 10                               // 000000006EC0: D7610019 0001140E
	v_writelane_b32 v25, s15, 11                               // 000000006EC8: D7610019 0001160F
	v_writelane_b32 v25, s0, 12                                // 000000006ED0: D7610019 00011800
	v_writelane_b32 v25, s1, 13                                // 000000006ED8: D7610019 00011A01
	v_writelane_b32 v25, s2, 14                                // 000000006EE0: D7610019 00011C02
	v_writelane_b32 v25, s3, 15                                // 000000006EE8: D7610019 00011E03
	v_writelane_b32 v25, s4, 16                                // 000000006EF0: D7610019 00012004
	v_writelane_b32 v25, s5, 17                                // 000000006EF8: D7610019 00012205
	v_writelane_b32 v25, s6, 18                                // 000000006F00: D7610019 00012406
	v_writelane_b32 v25, s7, 19                                // 000000006F08: D7610019 00012607
	v_writelane_b32 v25, s0, 20                                // 000000006F10: D7610019 00012800
	v_writelane_b32 v28, s12, 0                                // 000000006F18: D761001C 0001000C
	v_writelane_b32 v25, s1, 21                                // 000000006F20: D7610019 00012A01
	v_writelane_b32 v28, s13, 1                                // 000000006F28: D761001C 0001020D
	v_writelane_b32 v25, s2, 22                                // 000000006F30: D7610019 00012C02
	v_writelane_b32 v28, s14, 2                                // 000000006F38: D761001C 0001040E
	v_writelane_b32 v25, s3, 23                                // 000000006F40: D7610019 00012E03
	v_writelane_b32 v28, s15, 3                                // 000000006F48: D761001C 0001060F
	v_writelane_b32 v25, s4, 24                                // 000000006F50: D7610019 00013004
	v_writelane_b32 v25, s5, 25                                // 000000006F58: D7610019 00013205
	v_writelane_b32 v25, s6, 26                                // 000000006F60: D7610019 00013406
	v_writelane_b32 v25, s7, 27                                // 000000006F68: D7610019 00013607
	v_writelane_b32 v28, s0, 4                                 // 000000006F70: D761001C 00010800
	v_writelane_b32 v25, s8, 28                                // 000000006F78: D7610019 00013808
	v_writelane_b32 v28, s1, 5                                 // 000000006F80: D761001C 00010A01
	v_writelane_b32 v25, s9, 29                                // 000000006F88: D7610019 00013A09
	v_writelane_b32 v28, s2, 6                                 // 000000006F90: D761001C 00010C02
	v_writelane_b32 v25, s10, 30                               // 000000006F98: D7610019 00013C0A
	v_writelane_b32 v28, s3, 7                                 // 000000006FA0: D761001C 00010E03
	v_writelane_b32 v25, s11, 31                               // 000000006FA8: D7610019 00013E0B
	v_writelane_b32 v28, s4, 8                                 // 000000006FB0: D761001C 00011004
	v_writelane_b32 v28, s5, 9                                 // 000000006FB8: D761001C 00011205
	v_writelane_b32 v28, s6, 10                                // 000000006FC0: D761001C 00011406
	v_writelane_b32 v28, s7, 11                                // 000000006FC8: D761001C 00011607
	v_writelane_b32 v28, s0, 12                                // 000000006FD0: D761001C 00011800
	v_writelane_b32 v28, s1, 13                                // 000000006FD8: D761001C 00011A01
	v_writelane_b32 v28, s2, 14                                // 000000006FE0: D761001C 00011C02
	v_writelane_b32 v28, s3, 15                                // 000000006FE8: D761001C 00011E03
	v_writelane_b32 v28, s4, 16                                // 000000006FF0: D761001C 00012004
	v_writelane_b32 v28, s5, 17                                // 000000006FF8: D761001C 00012205
	v_writelane_b32 v28, s6, 18                                // 000000007000: D761001C 00012406
	v_writelane_b32 v28, s7, 19                                // 000000007008: D761001C 00012607
	v_writelane_b32 v28, s0, 20                                // 000000007010: D761001C 00012800
	v_writelane_b32 v28, s1, 21                                // 000000007018: D761001C 00012A01
	v_writelane_b32 v28, s2, 22                                // 000000007020: D761001C 00012C02
	v_writelane_b32 v28, s3, 23                                // 000000007028: D761001C 00012E03
	v_writelane_b32 v28, s4, 24                                // 000000007030: D761001C 00013004
	v_writelane_b32 v28, s5, 25                                // 000000007038: D761001C 00013205
	v_writelane_b32 v28, s6, 26                                // 000000007040: D761001C 00013406
	v_writelane_b32 v28, s7, 27                                // 000000007048: D761001C 00013607
	v_writelane_b32 v28, s52, 28                               // 000000007050: D761001C 00013834
	v_writelane_b32 v29, s56, 0                                // 000000007058: D761001D 00010038
	v_writelane_b32 v28, s53, 29                               // 000000007060: D761001C 00013A35
	v_writelane_b32 v29, s57, 1                                // 000000007068: D761001D 00010239
	v_writelane_b32 v28, s54, 30                               // 000000007070: D761001C 00013C36
	v_writelane_b32 v29, s58, 2                                // 000000007078: D761001D 0001043A
	v_writelane_b32 v28, s55, 31                               // 000000007080: D761001C 00013E37
	v_writelane_b32 v29, s59, 3                                // 000000007088: D761001D 0001063B
	s_or_saveexec_b32 s105, -1                                 // 000000007090: BEE922C1
	scratch_load_b32 v27, off, off offset:12                   // 000000007094: ED05007C 0000001B 00000C00
	s_wait_alu 0xfffe                                          // 0000000070A0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000070A4: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000070A8: BFC00000
	v_readlane_b32 s37, v27, 0                                 // 0000000070AC: D7600025 0001011B
	v_readlane_b32 s36, v27, 2                                 // 0000000070B4: D7600024 0001051B
	s_and_b32 vcc_lo, exec_lo, s17                             // 0000000070BC: 8B6A117E
	s_cbranch_vccz 192                                         // 0000000070C0: BFA300C0 <r_3_3_3_8_8_8+0x5dc4>
	s_or_saveexec_b32 s105, -1                                 // 0000000070C4: BEE922C1
	scratch_load_b32 v27, off, off offset:8                    // 0000000070C8: ED05007C 0000001B 00000800
	s_wait_alu 0xfffe                                          // 0000000070D4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000070D8: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000070DC: BFC00000
	v_readlane_b32 s16, v27, 10                                // 0000000070E0: D7600010 0001151B
	v_readlane_b32 s17, v27, 11                                // 0000000070E8: D7600011 0001171B
	s_or_saveexec_b32 s105, -1                                 // 0000000070F0: BEE922C1
	scratch_load_b32 v27, off, off                             // 0000000070F4: ED05007C 0000001B 00000000
	s_wait_alu 0xfffe                                          // 000000007100: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007104: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000007108: BFC00000
	v_readlane_b32 s0, v27, 29                                 // 00000000710C: D7600000 00013B1B
	v_readlane_b32 s1, v27, 30                                 // 000000007114: D7600001 00013D1B
	s_mov_b32 s41, s40                                         // 00000000711C: BEA90028
	s_mov_b32 s42, s40                                         // 000000007120: BEAA0028
	s_mov_b32 s43, s40                                         // 000000007124: BEAB0028
	s_wait_alu 0xfffe                                          // 000000007128: BF88FFFE
	s_mov_b64 s[28:29], s[40:41]                               // 00000000712C: BE9C0128
	s_add_nc_u64 s[16:17], s[16:17], s[0:1]                    // 000000007130: A9900010
	s_clause 0x2                                               // 000000007134: BF850002
	s_load_b256 s[0:7], s[20:21], 0x0                          // 000000007138: F400600A F8000000
	s_load_b256 s[92:99], s[16:17], 0x0                        // 000000007140: F4007708 F8000000
	s_load_b256 s[20:27], s[34:35], 0x1280                     // 000000007148: F4006511 F8001280
	s_mov_b64 s[30:31], s[42:43]                               // 000000007150: BE9E012A
	s_mov_b64 s[12:13], s[40:41]                               // 000000007154: BE8C0128
	s_mov_b64 s[8:9], s[40:41]                                 // 000000007158: BE880128
	s_mov_b64 s[14:15], s[42:43]                               // 00000000715C: BE8E012A
	s_mov_b64 s[10:11], s[42:43]                               // 000000007160: BE8A012A
	s_mov_b64 s[46:47], s[42:43]                               // 000000007164: BEAE012A
	s_mov_b64 s[54:55], s[42:43]                               // 000000007168: BEB6012A
	s_mov_b64 s[44:45], s[40:41]                               // 00000000716C: BEAC0128
	s_mov_b64 s[50:51], s[42:43]                               // 000000007170: BEB2012A
	s_mov_b64 s[58:59], s[42:43]                               // 000000007174: BEBA012A
	s_mov_b64 s[52:53], s[40:41]                               // 000000007178: BEB40128
	s_mov_b64 s[48:49], s[40:41]                               // 00000000717C: BEB00128
	s_mov_b64 s[56:57], s[40:41]                               // 000000007180: BEB80128
	s_mov_b32 s16, -1                                          // 000000007184: BE9000C1
	s_mov_b64 s[74:75], s[18:19]                               // 000000007188: BECA0112
	s_wait_kmcnt 0x0                                           // 00000000718C: BFC70000
	v_writelane_b32 v28, s0, 12                                // 000000007190: D761001C 00011800
	v_writelane_b32 v28, s1, 13                                // 000000007198: D761001C 00011A01
	v_writelane_b32 v28, s2, 14                                // 0000000071A0: D761001C 00011C02
	v_writelane_b32 v28, s3, 15                                // 0000000071A8: D761001C 00011E03
	v_writelane_b32 v28, s4, 16                                // 0000000071B0: D761001C 00012004
	v_writelane_b32 v28, s5, 17                                // 0000000071B8: D761001C 00012205
	v_writelane_b32 v28, s6, 18                                // 0000000071C0: D761001C 00012406
	v_writelane_b32 v28, s7, 19                                // 0000000071C8: D761001C 00012607
	s_mov_b64 s[4:5], s[40:41]                                 // 0000000071D0: BE840128
	s_mov_b64 s[0:1], s[40:41]                                 // 0000000071D4: BE800128
	s_mov_b64 s[6:7], s[42:43]                                 // 0000000071D8: BE86012A
	s_mov_b64 s[2:3], s[42:43]                                 // 0000000071DC: BE82012A
	v_writelane_b32 v28, s20, 4                                // 0000000071E0: D761001C 00010814
	s_wait_alu 0xfffe                                          // 0000000071E8: BF88FFFE
	v_writelane_b32 v20, s0, 28                                // 0000000071EC: D7610014 00013800
	v_writelane_b32 v28, s21, 5                                // 0000000071F4: D761001C 00010A15
	v_writelane_b32 v20, s1, 29                                // 0000000071FC: D7610014 00013A01
	v_writelane_b32 v28, s22, 6                                // 000000007204: D761001C 00010C16
	v_writelane_b32 v20, s2, 30                                // 00000000720C: D7610014 00013C02
	v_writelane_b32 v28, s23, 7                                // 000000007214: D761001C 00010E17
	v_writelane_b32 v20, s3, 31                                // 00000000721C: D7610014 00013E03
	v_writelane_b32 v28, s24, 8                                // 000000007224: D761001C 00011018
	v_writelane_b32 v28, s25, 9                                // 00000000722C: D761001C 00011219
	v_writelane_b32 v28, s26, 10                               // 000000007234: D761001C 0001141A
	v_writelane_b32 v28, s27, 11                               // 00000000723C: D761001C 0001161B
	s_mov_b64 s[24:25], s[40:41]                               // 000000007244: BE980128
	s_mov_b64 s[20:21], s[40:41]                               // 000000007248: BE940128
	s_mov_b64 s[26:27], s[42:43]                               // 00000000724C: BE9A012A
	s_mov_b64 s[22:23], s[42:43]                               // 000000007250: BE96012A
	s_wait_alu 0xfffe                                          // 000000007254: BF88FFFE
	v_writelane_b32 v28, s20, 20                               // 000000007258: D761001C 00012814
	v_writelane_b32 v28, s21, 21                               // 000000007260: D761001C 00012A15
	v_writelane_b32 v28, s22, 22                               // 000000007268: D761001C 00012C16
	v_writelane_b32 v28, s23, 23                               // 000000007270: D761001C 00012E17
	v_writelane_b32 v28, s24, 24                               // 000000007278: D761001C 00013018
	v_writelane_b32 v28, s25, 25                               // 000000007280: D761001C 00013219
	v_writelane_b32 v28, s26, 26                               // 000000007288: D761001C 0001341A
	v_writelane_b32 v28, s27, 27                               // 000000007290: D761001C 0001361B
	s_mov_b64 s[24:25], s[40:41]                               // 000000007298: BE980128
	s_mov_b64 s[26:27], s[42:43]                               // 00000000729C: BE9A012A
	s_wait_alu 0xfffe                                          // 0000000072A0: BF88FFFE
	v_writelane_b32 v25, s24, 12                               // 0000000072A4: D7610019 00011818
	v_writelane_b32 v25, s25, 13                               // 0000000072AC: D7610019 00011A19
	v_writelane_b32 v25, s26, 14                               // 0000000072B4: D7610019 00011C1A
	v_writelane_b32 v25, s27, 15                               // 0000000072BC: D7610019 00011E1B
	v_writelane_b32 v25, s28, 16                               // 0000000072C4: D7610019 0001201C
	v_writelane_b32 v25, s29, 17                               // 0000000072CC: D7610019 0001221D
	v_writelane_b32 v25, s30, 18                               // 0000000072D4: D7610019 0001241E
	v_writelane_b32 v25, s31, 19                               // 0000000072DC: D7610019 0001261F
	v_writelane_b32 v25, s4, 0                                 // 0000000072E4: D7610019 00010004
	v_writelane_b32 v25, s5, 1                                 // 0000000072EC: D7610019 00010205
	v_writelane_b32 v25, s6, 2                                 // 0000000072F4: D7610019 00010406
	v_writelane_b32 v25, s7, 3                                 // 0000000072FC: D7610019 00010607
	v_writelane_b32 v25, s8, 4                                 // 000000007304: D7610019 00010808
	v_writelane_b32 v25, s9, 5                                 // 00000000730C: D7610019 00010A09
	v_writelane_b32 v25, s10, 6                                // 000000007314: D7610019 00010C0A
	v_writelane_b32 v25, s11, 7                                // 00000000731C: D7610019 00010E0B
	v_writelane_b32 v25, s12, 8                                // 000000007324: D7610019 0001100C
	v_writelane_b32 v25, s13, 9                                // 00000000732C: D7610019 0001120D
	v_writelane_b32 v25, s14, 10                               // 000000007334: D7610019 0001140E
	v_writelane_b32 v25, s15, 11                               // 00000000733C: D7610019 0001160F
	v_writelane_b32 v25, s44, 20                               // 000000007344: D7610019 0001282C
	v_writelane_b32 v28, s56, 0                                // 00000000734C: D761001C 00010038
	v_writelane_b32 v25, s45, 21                               // 000000007354: D7610019 00012A2D
	v_writelane_b32 v28, s57, 1                                // 00000000735C: D761001C 00010239
	v_writelane_b32 v25, s46, 22                               // 000000007364: D7610019 00012C2E
	v_writelane_b32 v28, s58, 2                                // 00000000736C: D761001C 0001043A
	v_writelane_b32 v25, s47, 23                               // 000000007374: D7610019 00012E2F
	v_writelane_b32 v28, s59, 3                                // 00000000737C: D761001C 0001063B
	v_writelane_b32 v25, s48, 24                               // 000000007384: D7610019 00013030
	v_writelane_b32 v25, s49, 25                               // 00000000738C: D7610019 00013231
	v_writelane_b32 v25, s50, 26                               // 000000007394: D7610019 00013432
	v_writelane_b32 v25, s51, 27                               // 00000000739C: D7610019 00013633
	v_writelane_b32 v25, s52, 28                               // 0000000073A4: D7610019 00013834
	v_writelane_b32 v25, s53, 29                               // 0000000073AC: D7610019 00013A35
	v_writelane_b32 v25, s54, 30                               // 0000000073B4: D7610019 00013C36
	v_writelane_b32 v25, s55, 31                               // 0000000073BC: D7610019 00013E37
	v_writelane_b32 v29, s76, 4                                // 0000000073C4: D761001D 0001084C
	v_writelane_b32 v29, s77, 5                                // 0000000073CC: D761001D 00010A4D
	v_writelane_b32 v29, s78, 6                                // 0000000073D4: D761001D 00010C4E
	v_writelane_b32 v29, s79, 7                                // 0000000073DC: D761001D 00010E4F
	v_writelane_b32 v29, s80, 8                                // 0000000073E4: D761001D 00011050
	v_writelane_b32 v29, s81, 9                                // 0000000073EC: D761001D 00011251
	v_writelane_b32 v29, s82, 10                               // 0000000073F4: D761001D 00011452
	v_writelane_b32 v29, s83, 11                               // 0000000073FC: D761001D 00011653
	s_or_saveexec_b32 s105, -1                                 // 000000007404: BEE922C1
	scratch_store_b32 off, v23, off offset:44                  // 000000007408: ED06807C 0B800000 00002C00
	s_wait_alu 0xfffe                                          // 000000007414: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007418: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 00000000741C: BEE922C1
	scratch_store_b32 off, v21, off offset:56                  // 000000007420: ED06807C 0A800000 00003800
	s_wait_alu 0xfffe                                          // 00000000742C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007430: BEFE0069
	s_mov_b32 s89, s88                                         // 000000007434: BED90058
	s_mov_b32 s90, s88                                         // 000000007438: BEDA0058
	s_mov_b32 s91, s88                                         // 00000000743C: BEDB0058
	s_mov_b32 s24, s88                                         // 000000007440: BE980058
	s_mov_b32 s25, s88                                         // 000000007444: BE990058
	s_mov_b32 s26, s88                                         // 000000007448: BE9A0058
	s_mov_b32 s27, s88                                         // 00000000744C: BE9B0058
	s_mov_b32 s60, s88                                         // 000000007450: BEBC0058
	s_and_not1_b32 vcc_lo, exec_lo, s16                        // 000000007454: 916A107E
	s_mov_b32 s61, s88                                         // 000000007458: BEBD0058
	s_mov_b32 s62, s88                                         // 00000000745C: BEBE0058
	s_mov_b32 s63, s88                                         // 000000007460: BEBF0058
	s_mov_b32 s52, s88                                         // 000000007464: BEB40058
	s_mov_b32 s53, s88                                         // 000000007468: BEB50058
	s_mov_b32 s54, s88                                         // 00000000746C: BEB60058
	s_mov_b32 s55, s88                                         // 000000007470: BEB70058
	s_mov_b32 s56, s88                                         // 000000007474: BEB80058
	s_mov_b32 s57, s88                                         // 000000007478: BEB90058
	s_mov_b32 s58, s88                                         // 00000000747C: BEBA0058
	s_mov_b32 s59, s88                                         // 000000007480: BEBB0058
	s_mov_b32 s16, s88                                         // 000000007484: BE900058
	s_mov_b32 s17, s88                                         // 000000007488: BE910058
	s_mov_b32 s18, s88                                         // 00000000748C: BE920058
	s_mov_b32 s19, s88                                         // 000000007490: BE930058
	s_mov_b32 s20, s88                                         // 000000007494: BE940058
	s_mov_b32 s21, s88                                         // 000000007498: BE950058
	s_mov_b32 s22, s88                                         // 00000000749C: BE960058
	s_mov_b32 s23, s88                                         // 0000000074A0: BE970058
	s_mov_b32 s84, s88                                         // 0000000074A4: BED40058
	s_mov_b32 s85, s88                                         // 0000000074A8: BED50058
	s_mov_b32 s86, s88                                         // 0000000074AC: BED60058
	s_mov_b32 s87, s88                                         // 0000000074B0: BED70058
	s_mov_b32 s64, s88                                         // 0000000074B4: BEC00058
	s_mov_b32 s65, s88                                         // 0000000074B8: BEC10058
	s_mov_b32 s66, s88                                         // 0000000074BC: BEC20058
	s_mov_b32 s67, s88                                         // 0000000074C0: BEC30058
	s_mov_b32 s28, s88                                         // 0000000074C4: BE9C0058
	s_mov_b32 s29, s88                                         // 0000000074C8: BE9D0058
	s_mov_b32 s30, s88                                         // 0000000074CC: BE9E0058
	s_mov_b32 s31, s88                                         // 0000000074D0: BE9F0058
	s_cbranch_vccnz 7                                          // 0000000074D4: BFA40007 <r_3_3_3_8_8_8+0x5ef4>
	s_clause 0x2                                               // 0000000074D8: BF850002
	s_load_b512 s[52:67], s[74:75], 0x2500                     // 0000000074DC: F4008D25 F8002500
	s_load_b256 s[84:91], s[74:75], 0x23e0                     // 0000000074E4: F4007525 F80023E0
	s_load_b512 s[16:31], s[74:75], 0x2640                     // 0000000074EC: F4008425 F8002640
	s_load_b256 s[0:7], s[74:75], 0x1120                       // 0000000074F4: F4006025 F8001120
	s_and_b32 vcc_lo, exec_lo, s33                             // 0000000074FC: 8B6A217E
	s_wait_kmcnt 0x0                                           // 000000007500: BFC70000
	v_writelane_b32 v29, s0, 12                                // 000000007504: D761001D 00011800
	v_writelane_b32 v29, s1, 13                                // 00000000750C: D761001D 00011A01
	v_writelane_b32 v29, s2, 14                                // 000000007514: D761001D 00011C02
	v_writelane_b32 v29, s3, 15                                // 00000000751C: D761001D 00011E03
	v_writelane_b32 v29, s4, 16                                // 000000007524: D761001D 00012004
	v_writelane_b32 v29, s5, 17                                // 00000000752C: D761001D 00012205
	v_writelane_b32 v29, s6, 18                                // 000000007534: D761001D 00012406
	v_writelane_b32 v29, s7, 19                                // 00000000753C: D761001D 00012607
	s_load_b256 s[0:7], s[74:75], 0xfe0                        // 000000007544: F4006025 F8000FE0
	s_wait_kmcnt 0x0                                           // 00000000754C: BFC70000
	v_writelane_b32 v29, s0, 20                                // 000000007550: D761001D 00012800
	v_writelane_b32 v29, s1, 21                                // 000000007558: D761001D 00012A01
	v_writelane_b32 v29, s2, 22                                // 000000007560: D761001D 00012C02
	v_writelane_b32 v29, s3, 23                                // 000000007568: D761001D 00012E03
	v_writelane_b32 v29, s4, 24                                // 000000007570: D761001D 00013004
	v_writelane_b32 v29, s5, 25                                // 000000007578: D761001D 00013205
	v_writelane_b32 v29, s6, 26                                // 000000007580: D761001D 00013406
	v_writelane_b32 v29, s7, 27                                // 000000007588: D761001D 00013607
	s_load_b512 s[0:15], s[74:75], 0x1240                      // 000000007590: F4008025 F8001240
	s_wait_kmcnt 0x0                                           // 000000007598: BFC70000
	v_writelane_b32 v29, s0, 28                                // 00000000759C: D761001D 00013800
	v_writelane_b32 v21, s4, 0                                 // 0000000075A4: D7610015 00010004
	v_writelane_b32 v29, s1, 29                                // 0000000075AC: D761001D 00013A01
	v_writelane_b32 v21, s5, 1                                 // 0000000075B4: D7610015 00010205
	v_writelane_b32 v29, s2, 30                                // 0000000075BC: D761001D 00013C02
	v_writelane_b32 v21, s6, 2                                 // 0000000075C4: D7610015 00010406
	v_writelane_b32 v29, s3, 31                                // 0000000075CC: D761001D 00013E03
	v_writelane_b32 v21, s7, 3                                 // 0000000075D4: D7610015 00010607
	v_writelane_b32 v21, s8, 4                                 // 0000000075DC: D7610015 00010808
	v_writelane_b32 v21, s9, 5                                 // 0000000075E4: D7610015 00010A09
	v_writelane_b32 v21, s10, 6                                // 0000000075EC: D7610015 00010C0A
	v_writelane_b32 v21, s11, 7                                // 0000000075F4: D7610015 00010E0B
	v_writelane_b32 v21, s12, 8                                // 0000000075FC: D7610015 0001100C
	v_writelane_b32 v21, s13, 9                                // 000000007604: D7610015 0001120D
	v_writelane_b32 v21, s14, 10                               // 00000000760C: D7610015 0001140E
	v_writelane_b32 v21, s15, 11                               // 000000007614: D7610015 0001160F
	s_or_saveexec_b32 s105, -1                                 // 00000000761C: BEE922C1
	scratch_load_b32 v27, off, off offset:8                    // 000000007620: ED05007C 0000001B 00000800
	s_wait_alu 0xfffe                                          // 00000000762C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007630: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000007634: BFC00000
	v_readlane_b32 s74, v27, 14                                // 000000007638: D760004A 00011D1B
	v_readlane_b32 s75, v27, 15                                // 000000007640: D760004B 00011F1B
	s_or_saveexec_b32 s105, -1                                 // 000000007648: BEE922C1
	scratch_load_b32 v27, off, off                             // 00000000764C: ED05007C 0000001B 00000000
	s_wait_alu 0xfffe                                          // 000000007658: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000765C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000007660: BFC00000
	v_readlane_b32 s0, v27, 29                                 // 000000007664: D7600000 00013B1B
	v_readlane_b32 s1, v27, 30                                 // 00000000766C: D7600001 00013D1B
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007674: BF870001
	s_add_nc_u64 s[76:77], s[74:75], s[0:1]                    // 000000007678: A9CC004A
	s_add_nc_u64 s[74:75], s[34:35], 0x1c0                     // 00000000767C: A9CAFF22 000001C0
	s_cbranch_vccnz 5435                                       // 000000007684: BFA4153B <r_3_3_3_8_8_8+0xb574>
	s_mov_b32 s41, s40                                         // 000000007688: BEA90028
	s_mov_b32 s42, s40                                         // 00000000768C: BEAA0028
	s_mov_b32 s43, s40                                         // 000000007690: BEAB0028
	s_or_saveexec_b32 s105, -1                                 // 000000007694: BEE922C1
	scratch_load_b32 v23, off, off offset:44                   // 000000007698: ED05007C 00000017 00002C00
	s_wait_alu 0xfffe                                          // 0000000076A4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000076A8: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000076AC: BFC00000
	v_readlane_b32 s0, v23, 1                                  // 0000000076B0: D7600000 00010317
	s_mov_b64 s[4:5], s[40:41]                                 // 0000000076B8: BE840128
	s_mov_b64 s[6:7], s[42:43]                                 // 0000000076BC: BE86012A
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000076C0: BF870001
	s_and_b32 vcc_lo, exec_lo, s0                              // 0000000076C4: 8B6A007E
	s_mov_b64 s[0:1], s[40:41]                                 // 0000000076C8: BE800128
	s_mov_b64 s[2:3], s[42:43]                                 // 0000000076CC: BE82012A
	s_wait_alu 0xfffe                                          // 0000000076D0: BF88FFFE
	v_writelane_b32 v24, s0, 28                                // 0000000076D4: D7610018 00013800
	v_writelane_b32 v27, s4, 0                                 // 0000000076DC: D761001B 00010004
	v_writelane_b32 v24, s1, 29                                // 0000000076E4: D7610018 00013A01
	v_writelane_b32 v27, s5, 1                                 // 0000000076EC: D761001B 00010205
	v_writelane_b32 v24, s2, 30                                // 0000000076F4: D7610018 00013C02
	v_writelane_b32 v27, s6, 2                                 // 0000000076FC: D761001B 00010406
	v_writelane_b32 v24, s3, 31                                // 000000007704: D7610018 00013E03
	v_writelane_b32 v27, s7, 3                                 // 00000000770C: D761001B 00010607
	s_cbranch_vccnz 43                                         // 000000007714: BFA4002B <r_3_3_3_8_8_8+0x61c4>
	s_or_saveexec_b32 s105, -1                                 // 000000007718: BEE922C1
	scratch_load_b32 v23, off, off offset:8                    // 00000000771C: ED05007C 00000017 00000800
	s_wait_alu 0xfffe                                          // 000000007728: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000772C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000007730: BFC00000
	v_readlane_b32 s78, v23, 20                                // 000000007734: D760004E 00012917
	v_readlane_b32 s79, v23, 21                                // 00000000773C: D760004F 00012B17
	s_or_saveexec_b32 s105, -1                                 // 000000007744: BEE922C1
	scratch_load_b32 v23, off, off                             // 000000007748: ED05007C 00000017 00000000
	s_wait_alu 0xfffe                                          // 000000007754: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007758: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000775C: BFC00000
	v_readlane_b32 s0, v23, 29                                 // 000000007760: D7600000 00013B17
	v_readlane_b32 s1, v23, 30                                 // 000000007768: D7600001 00013D17
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007770: BF870001
	s_add_nc_u64 s[78:79], s[78:79], s[0:1]                    // 000000007774: A9CE004E
	s_load_b256 s[0:7], s[78:79], 0x0                          // 000000007778: F4006027 F8000000
	s_wait_kmcnt 0x0                                           // 000000007780: BFC70000
	v_writelane_b32 v24, s0, 28                                // 000000007784: D7610018 00013800
	v_writelane_b32 v27, s4, 0                                 // 00000000778C: D761001B 00010004
	v_writelane_b32 v24, s1, 29                                // 000000007794: D7610018 00013A01
	v_writelane_b32 v27, s5, 1                                 // 00000000779C: D761001B 00010205
	v_writelane_b32 v24, s2, 30                                // 0000000077A4: D7610018 00013C02
	v_writelane_b32 v27, s6, 2                                 // 0000000077AC: D761001B 00010406
	v_writelane_b32 v24, s3, 31                                // 0000000077B4: D7610018 00013E03
	v_writelane_b32 v27, s7, 3                                 // 0000000077BC: D761001B 00010607
	s_or_saveexec_b32 s105, -1                                 // 0000000077C4: BEE922C1
	scratch_load_b32 v23, off, off offset:16                   // 0000000077C8: ED05007C 00000017 00001000
	s_wait_alu 0xfffe                                          // 0000000077D4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000077D8: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000077DC: BFC00000
	v_readlane_b32 s0, v23, 2                                  // 0000000077E0: D7600000 00010517
	v_readlane_b32 s1, v23, 3                                  // 0000000077E8: D7600001 00010717
	s_load_b256 s[0:7], s[0:1], 0x0                            // 0000000077F0: F4006000 F8000000
	s_wait_kmcnt 0x0                                           // 0000000077F8: BFC70000
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000077FC: BF870112
	v_writelane_b32 v24, s0, 20                                // 000000007800: D7610018 00012800
	v_writelane_b32 v24, s1, 21                                // 000000007808: D7610018 00012A01
	v_writelane_b32 v24, s2, 22                                // 000000007810: D7610018 00012C02
	v_writelane_b32 v24, s3, 23                                // 000000007818: D7610018 00012E03
	v_writelane_b32 v24, s4, 24                                // 000000007820: D7610018 00013004
	v_writelane_b32 v24, s5, 25                                // 000000007828: D7610018 00013205
	v_writelane_b32 v24, s6, 26                                // 000000007830: D7610018 00013406
	v_writelane_b32 v24, s7, 27                                // 000000007838: D7610018 00013607
	s_or_saveexec_b32 s105, -1                                 // 000000007840: BEE922C1
	scratch_load_b32 v23, off, off offset:8                    // 000000007844: ED05007C 00000017 00000800
	s_wait_alu 0xfffe                                          // 000000007850: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007854: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000007858: BFC00000
	v_readlane_b32 s72, v23, 22                                // 00000000785C: D7600048 00012D17
	v_readlane_b32 s73, v23, 23                                // 000000007864: D7600049 00012F17
	s_or_saveexec_b32 s105, -1                                 // 00000000786C: BEE922C1
	scratch_load_b32 v23, off, off                             // 000000007870: ED05007C 00000017 00000000
	s_wait_alu 0xfffe                                          // 00000000787C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007880: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000007884: BFC00000
	v_readlane_b32 s0, v23, 29                                 // 000000007888: D7600000 00013B17
	v_readlane_b32 s1, v23, 30                                 // 000000007890: D7600001 00013D17
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007898: BF870001
	s_add_nc_u64 s[72:73], s[72:73], s[0:1]                    // 00000000789C: A9C80048
	s_load_b256 s[0:7], s[76:77], 0x0                          // 0000000078A0: F4006026 F8000000
	s_wait_kmcnt 0x0                                           // 0000000078A8: BFC70000
	v_writelane_b32 v24, s0, 12                                // 0000000078AC: D7610018 00011800
	v_writelane_b32 v24, s1, 13                                // 0000000078B4: D7610018 00011A01
	v_writelane_b32 v24, s2, 14                                // 0000000078BC: D7610018 00011C02
	v_writelane_b32 v24, s3, 15                                // 0000000078C4: D7610018 00011E03
	v_writelane_b32 v24, s4, 16                                // 0000000078CC: D7610018 00012004
	v_writelane_b32 v24, s5, 17                                // 0000000078D4: D7610018 00012205
	v_writelane_b32 v24, s6, 18                                // 0000000078DC: D7610018 00012406
	v_writelane_b32 v24, s7, 19                                // 0000000078E4: D7610018 00012607
	s_load_b512 s[0:15], s[72:73], 0x0                         // 0000000078EC: F4008024 F8000000
	s_movk_i32 s72, 0xfda0                                     // 0000000078F4: B048FDA0
	s_mov_b32 s73, -1                                          // 0000000078F8: BEC900C1
	s_wait_alu 0xfffe                                          // 0000000078FC: BF88FFFE
	s_add_nc_u64 s[72:73], s[34:35], s[72:73]                  // 000000007900: A9C84822
	s_wait_kmcnt 0x0                                           // 000000007904: BFC70000
	v_writelane_b32 v21, s0, 12                                // 000000007908: D7610015 00011800
	v_writelane_b32 v21, s1, 13                                // 000000007910: D7610015 00011A01
	v_writelane_b32 v21, s2, 14                                // 000000007918: D7610015 00011C02
	v_writelane_b32 v21, s3, 15                                // 000000007920: D7610015 00011E03
	v_writelane_b32 v21, s4, 16                                // 000000007928: D7610015 00012004
	v_writelane_b32 v21, s5, 17                                // 000000007930: D7610015 00012205
	v_writelane_b32 v21, s6, 18                                // 000000007938: D7610015 00012406
	v_writelane_b32 v21, s7, 19                                // 000000007940: D7610015 00012607
	v_writelane_b32 v21, s8, 20                                // 000000007948: D7610015 00012808
	v_writelane_b32 v21, s9, 21                                // 000000007950: D7610015 00012A09
	v_writelane_b32 v21, s10, 22                               // 000000007958: D7610015 00012C0A
	v_writelane_b32 v21, s11, 23                               // 000000007960: D7610015 00012E0B
	v_writelane_b32 v21, s12, 24                               // 000000007968: D7610015 0001300C
	v_writelane_b32 v21, s13, 25                               // 000000007970: D7610015 0001320D
	v_writelane_b32 v21, s14, 26                               // 000000007978: D7610015 0001340E
	v_writelane_b32 v21, s15, 27                               // 000000007980: D7610015 0001360F
	s_load_b256 s[0:7], s[34:35], 0x12c0                       // 000000007988: F4006011 F80012C0
	s_add_nc_u64 s[8:9], s[34:35], 0x1c0                       // 000000007990: A988FF22 000001C0
	s_wait_kmcnt 0x0                                           // 000000007998: BFC70000
	v_writelane_b32 v24, s0, 4                                 // 00000000799C: D7610018 00010800
	v_writelane_b32 v24, s1, 5                                 // 0000000079A4: D7610018 00010A01
	v_writelane_b32 v24, s2, 6                                 // 0000000079AC: D7610018 00010C02
	v_writelane_b32 v24, s3, 7                                 // 0000000079B4: D7610018 00010E03
	v_writelane_b32 v24, s4, 8                                 // 0000000079BC: D7610018 00011004
	v_writelane_b32 v24, s5, 9                                 // 0000000079C4: D7610018 00011205
	v_writelane_b32 v24, s6, 10                                // 0000000079CC: D7610018 00011406
	v_writelane_b32 v24, s7, 11                                // 0000000079D4: D7610018 00011607
	s_load_b256 s[0:7], s[72:73], 0x0                          // 0000000079DC: F4006024 F8000000
	s_wait_kmcnt 0x0                                           // 0000000079E4: BFC70000
	v_writelane_b32 v21, s0, 28                                // 0000000079E8: D7610015 00013800
	v_writelane_b32 v24, s4, 0                                 // 0000000079F0: D7610018 00010004
	v_writelane_b32 v21, s1, 29                                // 0000000079F8: D7610015 00013A01
	v_writelane_b32 v24, s5, 1                                 // 000000007A00: D7610018 00010205
	v_writelane_b32 v21, s2, 30                                // 000000007A08: D7610015 00013C02
	v_writelane_b32 v24, s6, 2                                 // 000000007A10: D7610018 00010406
	v_writelane_b32 v21, s3, 31                                // 000000007A18: D7610015 00013E03
	v_writelane_b32 v24, s7, 3                                 // 000000007A20: D7610018 00010607
	s_mov_b32 s4, 0                                            // 000000007A28: BE840080
	s_branch 168                                               // 000000007A2C: BFA000A8 <r_3_3_3_8_8_8+0x66d0>
	s_or_saveexec_b32 s105, -1                                 // 000000007A30: BEE922C1
	scratch_load_b32 v23, off, off offset:8                    // 000000007A34: ED05007C 00000017 00000800
	s_wait_alu 0xfffe                                          // 000000007A40: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007A44: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000007A48: BFC00000
	v_readlane_b32 s72, v23, 20                                // 000000007A4C: D7600048 00012917
	v_readlane_b32 s73, v23, 21                                // 000000007A54: D7600049 00012B17
	s_or_saveexec_b32 s105, -1                                 // 000000007A5C: BEE922C1
	scratch_load_b32 v23, off, off                             // 000000007A60: ED05007C 00000017 00000000
	s_wait_alu 0xfffe                                          // 000000007A6C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007A70: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000007A74: BFC00000
	v_readlane_b32 s0, v23, 29                                 // 000000007A78: D7600000 00013B17
	v_readlane_b32 s1, v23, 30                                 // 000000007A80: D7600001 00013D17
	s_mov_b32 s42, s40                                         // 000000007A88: BEAA0028
	s_mov_b32 s43, s40                                         // 000000007A8C: BEAB0028
	s_mov_b32 s41, s40                                         // 000000007A90: BEA90028
	s_wait_alu 0xfffe                                          // 000000007A94: BF88FFFE
	s_mov_b64 s[82:83], s[42:43]                               // 000000007A98: BED2012A
	s_add_nc_u64 s[70:71], s[72:73], s[0:1]                    // 000000007A9C: A9C60048
	s_load_b256 s[0:7], s[76:77], 0x0                          // 000000007AA0: F4006026 F8000000
	s_mov_b64 s[78:79], s[42:43]                               // 000000007AA8: BECE012A
	s_mov_b64 s[76:77], s[40:41]                               // 000000007AAC: BECC0128
	s_mov_b64 s[80:81], s[40:41]                               // 000000007AB0: BED00128
	s_wait_alu 0xfffe                                          // 000000007AB4: BF88FFFE
	v_writelane_b32 v21, s76, 28                               // 000000007AB8: D7610015 0001384C
	s_mov_b64 s[12:13], s[40:41]                               // 000000007AC0: BE8C0128
	s_mov_b64 s[8:9], s[40:41]                                 // 000000007AC4: BE880128
	s_mov_b64 s[14:15], s[42:43]                               // 000000007AC8: BE8E012A
	s_mov_b64 s[10:11], s[42:43]                               // 000000007ACC: BE8A012A
	v_writelane_b32 v21, s77, 29                               // 000000007AD0: D7610015 00013A4D
	s_mov_b64 s[46:47], s[42:43]                               // 000000007AD8: BEAE012A
	s_mov_b64 s[50:51], s[42:43]                               // 000000007ADC: BEB2012A
	s_mov_b64 s[44:45], s[40:41]                               // 000000007AE0: BEAC0128
	s_mov_b64 s[48:49], s[40:41]                               // 000000007AE4: BEB00128
	v_writelane_b32 v21, s78, 30                               // 000000007AE8: D7610015 00013C4E
	s_mov_b32 s104, -1                                         // 000000007AF0: BEE800C1
	v_writelane_b32 v21, s79, 31                               // 000000007AF4: D7610015 00013E4F
	s_wait_kmcnt 0x0                                           // 000000007AFC: BFC70000
	v_writelane_b32 v24, s0, 12                                // 000000007B00: D7610018 00011800
	v_writelane_b32 v24, s1, 13                                // 000000007B08: D7610018 00011A01
	v_writelane_b32 v24, s2, 14                                // 000000007B10: D7610018 00011C02
	v_writelane_b32 v24, s3, 15                                // 000000007B18: D7610018 00011E03
	v_writelane_b32 v24, s4, 16                                // 000000007B20: D7610018 00012004
	v_writelane_b32 v24, s5, 17                                // 000000007B28: D7610018 00012205
	v_writelane_b32 v24, s6, 18                                // 000000007B30: D7610018 00012406
	v_writelane_b32 v24, s7, 19                                // 000000007B38: D7610018 00012607
	s_load_b256 s[0:7], s[70:71], 0x0                          // 000000007B40: F4006023 F8000000
	s_wait_kmcnt 0x0                                           // 000000007B48: BFC70000
	v_writelane_b32 v24, s0, 28                                // 000000007B4C: D7610018 00013800
	v_writelane_b32 v27, s4, 0                                 // 000000007B54: D761001B 00010004
	v_writelane_b32 v24, s1, 29                                // 000000007B5C: D7610018 00013A01
	v_writelane_b32 v27, s5, 1                                 // 000000007B64: D761001B 00010205
	v_writelane_b32 v24, s2, 30                                // 000000007B6C: D7610018 00013C02
	v_writelane_b32 v27, s6, 2                                 // 000000007B74: D761001B 00010406
	v_writelane_b32 v24, s3, 31                                // 000000007B7C: D7610018 00013E03
	v_writelane_b32 v27, s7, 3                                 // 000000007B84: D761001B 00010607
	s_load_b256 s[0:7], s[34:35], 0x12c0                       // 000000007B8C: F4006011 F80012C0
	s_wait_kmcnt 0x0                                           // 000000007B94: BFC70000
	v_writelane_b32 v24, s0, 4                                 // 000000007B98: D7610018 00010800
	v_writelane_b32 v24, s1, 5                                 // 000000007BA0: D7610018 00010A01
	v_writelane_b32 v24, s2, 6                                 // 000000007BA8: D7610018 00010C02
	v_writelane_b32 v24, s3, 7                                 // 000000007BB0: D7610018 00010E03
	v_writelane_b32 v24, s4, 8                                 // 000000007BB8: D7610018 00011004
	v_writelane_b32 v24, s5, 9                                 // 000000007BC0: D7610018 00011205
	v_writelane_b32 v24, s6, 10                                // 000000007BC8: D7610018 00011406
	v_writelane_b32 v24, s7, 11                                // 000000007BD0: D7610018 00011607
	s_mov_b64 s[4:5], s[40:41]                                 // 000000007BD8: BE840128
	s_mov_b64 s[0:1], s[40:41]                                 // 000000007BDC: BE800128
	s_mov_b64 s[6:7], s[42:43]                                 // 000000007BE0: BE86012A
	s_mov_b64 s[2:3], s[42:43]                                 // 000000007BE4: BE82012A
	s_wait_alu 0xfffe                                          // 000000007BE8: BF88FFFE
	v_writelane_b32 v21, s0, 12                                // 000000007BEC: D7610015 00011800
	v_writelane_b32 v24, s44, 20                               // 000000007BF4: D7610018 0001282C
	v_writelane_b32 v21, s1, 13                                // 000000007BFC: D7610015 00011A01
	v_writelane_b32 v24, s45, 21                               // 000000007C04: D7610018 00012A2D
	v_writelane_b32 v21, s2, 14                                // 000000007C0C: D7610015 00011C02
	v_writelane_b32 v24, s46, 22                               // 000000007C14: D7610018 00012C2E
	v_writelane_b32 v21, s3, 15                                // 000000007C1C: D7610015 00011E03
	v_writelane_b32 v24, s47, 23                               // 000000007C24: D7610018 00012E2F
	v_writelane_b32 v21, s4, 16                                // 000000007C2C: D7610015 00012004
	v_writelane_b32 v24, s48, 24                               // 000000007C34: D7610018 00013030
	v_writelane_b32 v21, s5, 17                                // 000000007C3C: D7610015 00012205
	v_writelane_b32 v24, s49, 25                               // 000000007C44: D7610018 00013231
	v_writelane_b32 v21, s6, 18                                // 000000007C4C: D7610015 00012406
	v_writelane_b32 v24, s50, 26                               // 000000007C54: D7610018 00013432
	v_writelane_b32 v21, s7, 19                                // 000000007C5C: D7610015 00012607
	v_writelane_b32 v24, s51, 27                               // 000000007C64: D7610018 00013633
	v_writelane_b32 v21, s8, 20                                // 000000007C6C: D7610015 00012808
	v_writelane_b32 v24, s80, 0                                // 000000007C74: D7610018 00010050
	v_writelane_b32 v21, s9, 21                                // 000000007C7C: D7610015 00012A09
	v_writelane_b32 v24, s81, 1                                // 000000007C84: D7610018 00010251
	v_writelane_b32 v21, s10, 22                               // 000000007C8C: D7610015 00012C0A
	v_writelane_b32 v24, s82, 2                                // 000000007C94: D7610018 00010452
	v_writelane_b32 v21, s11, 23                               // 000000007C9C: D7610015 00012E0B
	v_writelane_b32 v24, s83, 3                                // 000000007CA4: D7610018 00010653
	v_writelane_b32 v21, s12, 24                               // 000000007CAC: D7610015 0001300C
	v_writelane_b32 v21, s13, 25                               // 000000007CB4: D7610015 0001320D
	v_writelane_b32 v21, s14, 26                               // 000000007CBC: D7610015 0001340E
	v_writelane_b32 v21, s15, 27                               // 000000007CC4: D7610015 0001360F
	s_mov_b64 s[8:9], s[74:75]                                 // 000000007CCC: BE88014A
	v_writelane_b32 v27, s92, 4                                // 000000007CD0: D761001B 0001085C
	v_writelane_b32 v27, s93, 5                                // 000000007CD8: D761001B 00010A5D
	v_writelane_b32 v27, s94, 6                                // 000000007CE0: D761001B 00010C5E
	v_writelane_b32 v27, s95, 7                                // 000000007CE8: D761001B 00010E5F
	v_writelane_b32 v27, s96, 8                                // 000000007CF0: D761001B 00011060
	v_writelane_b32 v27, s97, 9                                // 000000007CF8: D761001B 00011261
	v_writelane_b32 v27, s98, 10                               // 000000007D00: D761001B 00011462
	v_writelane_b32 v27, s99, 11                               // 000000007D08: D761001B 00011663
	v_writelane_b32 v27, s40, 12                               // 000000007D10: D761001B 00011828
	v_writelane_b32 v27, s41, 13                               // 000000007D18: D761001B 00011A29
	v_writelane_b32 v27, s42, 14                               // 000000007D20: D761001B 00011C2A
	v_writelane_b32 v27, s43, 15                               // 000000007D28: D761001B 00011E2B
	s_or_saveexec_b32 s105, -1                                 // 000000007D30: BEE922C1
	scratch_store_b32 off, v26, off offset:84                  // 000000007D34: ED06807C 0D000000 00005400
	s_wait_alu 0xfffe                                          // 000000007D40: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007D44: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000007D48: BEE922C1
	scratch_store_b32 off, v20, off offset:104                 // 000000007D4C: ED06807C 0A000000 00006800
	s_wait_alu 0xfffe                                          // 000000007D58: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007D5C: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000007D60: BEE922C1
	scratch_store_b32 off, v28, off offset:108                 // 000000007D64: ED06807C 0E000000 00006C00
	s_wait_alu 0xfffe                                          // 000000007D70: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007D74: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000007D78: BEE922C1
	scratch_store_b32 off, v29, off offset:112                 // 000000007D7C: ED06807C 0E800000 00007000
	s_wait_alu 0xfffe                                          // 000000007D88: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007D8C: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000007D90: BEE922C1
	scratch_store_b32 off, v24, off offset:116                 // 000000007D94: ED06807C 0C000000 00007400
	s_wait_alu 0xfffe                                          // 000000007DA0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007DA4: BEFE0069
	s_mov_b32 s5, s4                                           // 000000007DA8: BE850004
	s_mov_b32 s6, s4                                           // 000000007DAC: BE860004
	s_mov_b32 s7, s4                                           // 000000007DB0: BE870004
	s_mov_b32 s76, s4                                          // 000000007DB4: BECC0004
	s_mov_b32 s77, s4                                          // 000000007DB8: BECD0004
	s_mov_b32 s78, s4                                          // 000000007DBC: BECE0004
	s_mov_b32 s79, s4                                          // 000000007DC0: BECF0004
	s_mov_b32 s68, s4                                          // 000000007DC4: BEC40004
	s_and_not1_b32 vcc_lo, exec_lo, s104                       // 000000007DC8: 916A687E
	s_mov_b32 s69, s4                                          // 000000007DCC: BEC50004
	s_mov_b32 s70, s4                                          // 000000007DD0: BEC60004
	s_mov_b32 s71, s4                                          // 000000007DD4: BEC70004
	s_mov_b32 s72, s4                                          // 000000007DD8: BEC80004
	s_mov_b32 s73, s4                                          // 000000007DDC: BEC90004
	s_mov_b32 s74, s4                                          // 000000007DE0: BECA0004
	s_mov_b32 s75, s4                                          // 000000007DE4: BECB0004
	s_mov_b32 s0, s4                                           // 000000007DE8: BE800004
	s_mov_b32 s1, s4                                           // 000000007DEC: BE810004
	s_mov_b32 s2, s4                                           // 000000007DF0: BE820004
	s_mov_b32 s3, s4                                           // 000000007DF4: BE830004
	s_mov_b32 s80, s4                                          // 000000007DF8: BED00004
	s_mov_b32 s81, s4                                          // 000000007DFC: BED10004
	s_mov_b32 s82, s4                                          // 000000007E00: BED20004
	s_mov_b32 s83, s4                                          // 000000007E04: BED30004
	s_cbranch_vccnz 5                                          // 000000007E08: BFA40005 <r_3_3_3_8_8_8+0x6820>
	s_clause 0x1                                               // 000000007E0C: BF850001
	s_load_b256 s[0:7], s[8:9], 0x23e0                         // 000000007E10: F4006004 F80023E0
	s_load_b512 s[68:83], s[8:9], 0x2500                       // 000000007E18: F4009104 F8002500
	s_wait_kmcnt 0x0                                           // 000000007E20: BFC70000
	s_wait_alu 0xfffe                                          // 000000007E24: BF88FFFE
	v_writelane_b32 v27, s0, 16                                // 000000007E28: D761001B 00012000
	v_writelane_b32 v27, s1, 17                                // 000000007E30: D761001B 00012201
	v_writelane_b32 v27, s2, 18                                // 000000007E38: D761001B 00012402
	v_writelane_b32 v27, s3, 19                                // 000000007E40: D761001B 00012603
	v_writelane_b32 v27, s4, 20                                // 000000007E48: D761001B 00012804
	v_writelane_b32 v27, s5, 21                                // 000000007E50: D761001B 00012A05
	v_writelane_b32 v27, s6, 22                                // 000000007E58: D761001B 00012C06
	v_writelane_b32 v27, s7, 23                                // 000000007E60: D761001B 00012E07
	s_or_saveexec_b32 s105, -1                                 // 000000007E68: BEE922C1
	scratch_load_b32 v29, off, off offset:12                   // 000000007E6C: ED05007C 0000001D 00000C00
	s_wait_alu 0xfffe                                          // 000000007E78: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007E7C: BEFE0069
	s_mov_b64 s[98:99], s[8:9]                                 // 000000007E80: BEE20108
	s_wait_loadcnt 0x0                                         // 000000007E84: BFC00000
	v_readlane_b32 s0, v29, 15                                 // 000000007E88: D7600000 00011F1D
	v_readlane_b32 s33, v29, 6                                 // 000000007E90: D7600021 00010D1D
	v_readlane_b32 s4, v29, 19                                 // 000000007E98: D7600004 0001271D
	v_readlane_b32 s34, v29, 7                                 // 000000007EA0: D7600022 00010F1D
	v_readlane_b32 s8, v29, 23                                 // 000000007EA8: D7600008 00012F1D
	v_readlane_b32 s9, v29, 24                                 // 000000007EB0: D7600009 0001311D
	v_readlane_b32 s1, v29, 16                                 // 000000007EB8: D7600001 0001211D
	s_add_f32 s33, s33, s4                                     // 000000007EC0: A0210421
	s_add_f32 s34, s34, s0                                     // 000000007EC4: A0220022
	v_readlane_b32 s2, v29, 17                                 // 000000007EC8: D7600002 0001231D
	v_readlane_b32 s3, v29, 18                                 // 000000007ED0: D7600003 0001251D
	v_readlane_b32 s5, v29, 20                                 // 000000007ED8: D7600005 0001291D
	v_readlane_b32 s6, v29, 21                                 // 000000007EE0: D7600006 00012B1D
	v_readlane_b32 s7, v29, 22                                 // 000000007EE8: D7600007 00012D1D
	v_readlane_b32 s10, v29, 25                                // 000000007EF0: D760000A 0001331D
	v_readlane_b32 s11, v29, 26                                // 000000007EF8: D760000B 0001351D
	v_readlane_b32 s12, v29, 27                                // 000000007F00: D760000C 0001371D
	v_readlane_b32 s13, v29, 28                                // 000000007F08: D760000D 0001391D
	v_readlane_b32 s14, v29, 29                                // 000000007F10: D760000E 00013B1D
	v_readlane_b32 s15, v29, 30                                // 000000007F18: D760000F 00013D1D
	v_readlane_b32 s0, v29, 4                                  // 000000007F20: D7600000 0001091D
	s_or_saveexec_b32 s105, -1                                 // 000000007F28: BEE922C1
	s_wait_alu 0xfffe                                          // 000000007F2C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007F30: BEFE0069
	s_add_f32 s33, s5, s33                                     // 000000007F34: A0212105
	s_add_f32 s35, s0, s9                                      // 000000007F38: A0230900
	s_add_f32 s34, s1, s34                                     // 000000007F3C: A0222201
	s_wait_alu 0xfffe                                          // 000000007F40: BF88FFFE
	s_add_f32 s33, s6, s33                                     // 000000007F44: A0212106
	s_add_f32 s35, s10, s35                                    // 000000007F48: A023230A
	s_add_f32 s34, s2, s34                                     // 000000007F4C: A0222202
	s_wait_alu 0xfffe                                          // 000000007F50: BF88FFFE
	s_add_f32 s33, s7, s33                                     // 000000007F54: A0212107
	s_add_f32 s35, s11, s35                                    // 000000007F58: A023230B
	s_add_f32 s34, s3, s34                                     // 000000007F5C: A0222203
	s_wait_alu 0xfffe                                          // 000000007F60: BF88FFFE
	s_add_f32 s33, s33, s8                                     // 000000007F64: A0210821
	s_add_f32 s35, s35, s12                                    // 000000007F68: A0230C23
	s_add_f32 s34, s4, s34                                     // 000000007F6C: A0222204
	s_wait_alu 0xfffe                                          // 000000007F70: BF88FFFE
	s_add_f32 s33, s9, s33                                     // 000000007F74: A0212109
	s_add_f32 s35, s13, s35                                    // 000000007F78: A023230D
	s_add_f32 s34, s5, s34                                     // 000000007F7C: A0222205
	s_wait_alu 0xfffe                                          // 000000007F80: BF88FFFE
	s_add_f32 s33, s10, s33                                    // 000000007F84: A021210A
	s_add_f32 s35, s14, s35                                    // 000000007F88: A023230E
	s_add_f32 s34, s6, s34                                     // 000000007F8C: A0222206
	s_wait_alu 0xfffe                                          // 000000007F90: BF88FFFE
	s_add_f32 s33, s11, s33                                    // 000000007F94: A021210B
	s_add_f32 s38, s15, s35                                    // 000000007F98: A026230F
	s_or_saveexec_b32 s105, -1                                 // 000000007F9C: BEE922C1
	s_wait_alu 0xfffe                                          // 000000007FA0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007FA4: BEFE0069
	v_readlane_b32 s35, v29, 9                                 // 000000007FA8: D7600023 0001131D
	s_or_saveexec_b32 s105, -1                                 // 000000007FB0: BEE922C1
	scratch_load_b32 v28, off, off offset:40                   // 000000007FB4: ED05007C 0000001C 00002800
	s_wait_alu 0xfffe                                          // 000000007FC0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007FC4: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000007FC8: BFC00000
	v_readlane_b32 s0, v28, 29                                 // 000000007FCC: D7600000 00013B1C
	v_readlane_b32 s1, v28, 30                                 // 000000007FD4: D7600001 00013D1C
	v_readlane_b32 s2, v28, 31                                 // 000000007FDC: D7600002 00013F1C
	s_or_saveexec_b32 s105, -1                                 // 000000007FE4: BEE922C1
	scratch_load_b32 v23, off, off th:TH_LOAD_LU               // 000000007FE8: ED05007C 00300017 00000000
	s_wait_alu 0xfffe                                          // 000000007FF4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000007FF8: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000007FFC: BFC00000
	v_readlane_b32 s4, v23, 1                                  // 000000008000: D7600004 00010317
	v_readlane_b32 s8, v23, 5                                  // 000000008008: D7600008 00010B17
	v_readlane_b32 s9, v23, 6                                  // 000000008010: D7600009 00010D17
	v_readlane_b32 s3, v23, 0                                  // 000000008018: D7600003 00010117
	v_readlane_b32 s5, v23, 2                                  // 000000008020: D7600005 00010517
	v_readlane_b32 s6, v23, 3                                  // 000000008028: D7600006 00010717
	v_readlane_b32 s7, v23, 4                                  // 000000008030: D7600007 00010917
	v_readlane_b32 s10, v23, 7                                 // 000000008038: D760000A 00010F17
	v_readlane_b32 s11, v23, 8                                 // 000000008040: D760000B 00011117
	v_readlane_b32 s12, v23, 9                                 // 000000008048: D760000C 00011317
	v_readlane_b32 s13, v23, 10                                // 000000008050: D760000D 00011517
	v_readlane_b32 s14, v23, 11                                // 000000008058: D760000E 00011717
	v_readlane_b32 s15, v23, 12                                // 000000008060: D760000F 00011917
	s_add_f32 s35, s35, s4                                     // 000000008068: A0230423
	v_readlane_b32 s92, v29, 5                                 // 00000000806C: D760005C 00010B1D
	s_or_saveexec_b32 s105, -1                                 // 000000008074: BEE922C1
	s_wait_alu 0xfffe                                          // 000000008078: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000807C: BEFE0069
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 000000008080: BF8700D1
	s_add_f32 s101, s92, s0                                    // 000000008084: A065005C
	v_readlane_b32 s92, v28, 24                                // 000000008088: D760005C 0001311C
	s_add_f32 s35, s5, s35                                     // 000000008090: A0232305
	s_wait_alu 0xfffe                                          // 000000008094: BF88FFFE
	s_add_f32 s101, s1, s101                                   // 000000008098: A0656501
	s_add_f32 s102, s92, s9                                    // 00000000809C: A066095C
	s_add_f32 s35, s6, s35                                     // 0000000080A0: A0232306
	s_wait_alu 0xfffe                                          // 0000000080A4: BF88FFFE
	s_add_f32 s101, s2, s101                                   // 0000000080A8: A0656502
	v_readlane_b32 s92, v29, 8                                 // 0000000080AC: D760005C 0001111D
	s_add_f32 s102, s10, s102                                  // 0000000080B4: A066660A
	s_add_f32 s35, s7, s35                                     // 0000000080B8: A0232307
	s_wait_alu 0xfffe                                          // 0000000080BC: BF88FFFE
	s_add_f32 s101, s3, s101                                   // 0000000080C0: A0656503
	s_add_f32 s102, s11, s102                                  // 0000000080C4: A066660B
	s_add_f32 s35, s35, s8                                     // 0000000080C8: A0230823
	s_wait_alu 0xfffe                                          // 0000000080CC: BF88FFFE
	s_add_f32 s101, s101, s4                                   // 0000000080D0: A0650465
	s_add_f32 s102, s102, s12                                  // 0000000080D4: A0660C66
	s_add_f32 s35, s9, s35                                     // 0000000080D8: A0232309
	s_wait_alu 0xfffe                                          // 0000000080DC: BF88FFFE
	s_add_f32 s101, s5, s101                                   // 0000000080E0: A0656505
	s_add_f32 s102, s13, s102                                  // 0000000080E4: A066660D
	s_add_f32 s35, s10, s35                                    // 0000000080E8: A023230A
	s_wait_alu 0xfffe                                          // 0000000080EC: BF88FFFE
	s_add_f32 s101, s6, s101                                   // 0000000080F0: A0656506
	s_add_f32 s102, s14, s102                                  // 0000000080F4: A066660E
	s_add_f32 s35, s11, s35                                    // 0000000080F8: A023230B
	s_wait_alu 0xfffe                                          // 0000000080FC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000008100: BF870009
	s_add_f32 s41, s15, s102                                   // 000000008104: A029660F
	v_readlane_b32 s0, v28, 7                                  // 000000008108: D7600000 00010F1C
	v_readlane_b32 s4, v28, 11                                 // 000000008110: D7600004 0001171C
	v_readlane_b32 s8, v28, 15                                 // 000000008118: D7600008 00011F1C
	v_readlane_b32 s9, v28, 16                                 // 000000008120: D7600009 0001211C
	v_readlane_b32 s1, v28, 8                                  // 000000008128: D7600001 0001111C
	v_readlane_b32 s2, v28, 9                                  // 000000008130: D7600002 0001131C
	v_readlane_b32 s3, v28, 10                                 // 000000008138: D7600003 0001151C
	v_readlane_b32 s5, v28, 12                                 // 000000008140: D7600005 0001191C
	v_readlane_b32 s6, v28, 13                                 // 000000008148: D7600006 00011B1C
	v_readlane_b32 s7, v28, 14                                 // 000000008150: D7600007 00011D1C
	v_readlane_b32 s10, v28, 17                                // 000000008158: D760000A 0001231C
	v_readlane_b32 s11, v28, 18                                // 000000008160: D760000B 0001251C
	v_readlane_b32 s12, v28, 19                                // 000000008168: D760000C 0001271C
	v_readlane_b32 s13, v28, 20                                // 000000008170: D760000D 0001291C
	v_readlane_b32 s14, v28, 21                                // 000000008178: D760000E 00012B1C
	v_readlane_b32 s15, v28, 22                                // 000000008180: D760000F 00012D1C
	s_add_f32 s102, s92, s4                                    // 000000008188: A066045C
	v_readlane_b32 s92, v29, 10                                // 00000000818C: D760005C 0001151D
	s_or_saveexec_b32 s105, -1                                 // 000000008194: BEE922C1
	s_wait_alu 0xfffe                                          // 000000008198: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000819C: BEFE0069
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000081A0: BF870001
	s_add_f32 s103, s92, s0                                    // 0000000081A4: A067005C
	s_add_f32 s102, s5, s102                                   // 0000000081A8: A0666605
	v_readlane_b32 s92, v28, 23                                // 0000000081AC: D760005C 00012F1C
	s_or_saveexec_b32 s105, -1                                 // 0000000081B4: BEE922C1
	v_mov_b32_e32 v20, v28                                     // 0000000081B8: 7E28031C
	s_wait_alu 0xfffe                                          // 0000000081BC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000081C0: BEFE0069
	s_add_f32 s104, s92, s9                                    // 0000000081C4: A068095C
	s_add_f32 s103, s1, s103                                   // 0000000081C8: A0676701
	s_add_f32 s102, s6, s102                                   // 0000000081CC: A0666606
	v_readlane_b32 s92, v29, 11                                // 0000000081D0: D760005C 0001171D
	s_wait_alu 0xfffe                                          // 0000000081D8: BF88FFFE
	s_add_f32 s104, s10, s104                                  // 0000000081DC: A068680A
	s_add_f32 s103, s2, s103                                   // 0000000081E0: A0676702
	s_add_f32 s102, s7, s102                                   // 0000000081E4: A0666607
	s_wait_alu 0xfffe                                          // 0000000081E8: BF88FFFE
	s_add_f32 s104, s11, s104                                  // 0000000081EC: A068680B
	s_add_f32 s103, s3, s103                                   // 0000000081F0: A0676703
	s_add_f32 s102, s102, s8                                   // 0000000081F4: A0660866
	s_wait_alu 0xfffe                                          // 0000000081F8: BF88FFFE
	s_add_f32 s104, s104, s12                                  // 0000000081FC: A0680C68
	s_add_f32 s103, s4, s103                                   // 000000008200: A0676704
	s_add_f32 s102, s9, s102                                   // 000000008204: A0666609
	s_wait_alu 0xfffe                                          // 000000008208: BF88FFFE
	s_add_f32 s104, s13, s104                                  // 00000000820C: A068680D
	s_add_f32 s103, s5, s103                                   // 000000008210: A0676705
	s_add_f32 s102, s10, s102                                  // 000000008214: A066660A
	s_wait_alu 0xfffe                                          // 000000008218: BF88FFFE
	s_add_f32 s100, s14, s104                                  // 00000000821C: A064680E
	s_add_f32 s39, s6, s103                                    // 000000008220: A0276706
	s_add_f32 s104, s11, s102                                  // 000000008224: A068660B
	s_wait_alu 0xfffe                                          // 000000008228: BF88FFFE
	s_add_f32 s42, s15, s100                                   // 00000000822C: A02A640F
	s_or_saveexec_b32 s105, -1                                 // 000000008230: BEE922C1
	v_mov_b32_e32 v26, v29                                     // 000000008234: 7E34031D
	s_wait_alu 0xfffe                                          // 000000008238: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000823C: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000008240: BEE922C1
	scratch_store_b32 off, v26, off offset:12                  // 000000008244: ED06807C 0D000000 00000C00
	s_wait_alu 0xfffe                                          // 000000008250: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008254: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000008258: BEE922C1
	scratch_load_b32 v29, off, off offset:60                   // 00000000825C: ED05007C 0000001D 00003C00
	s_wait_alu 0xfffe                                          // 000000008268: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000826C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000008270: BFC00000
	v_readlane_b32 s0, v29, 20                                 // 000000008274: D7600000 0001291D
	v_readlane_b32 s4, v29, 24                                 // 00000000827C: D7600004 0001311D
	v_readlane_b32 s8, v29, 28                                 // 000000008284: D7600008 0001391D
	v_readlane_b32 s9, v29, 29                                 // 00000000828C: D7600009 00013B1D
	v_readlane_b32 s1, v29, 21                                 // 000000008294: D7600001 00012B1D
	v_readlane_b32 s2, v29, 22                                 // 00000000829C: D7600002 00012D1D
	v_readlane_b32 s3, v29, 23                                 // 0000000082A4: D7600003 00012F1D
	v_readlane_b32 s5, v29, 25                                 // 0000000082AC: D7600005 0001331D
	v_readlane_b32 s6, v29, 26                                 // 0000000082B4: D7600006 0001351D
	v_readlane_b32 s7, v29, 27                                 // 0000000082BC: D7600007 0001371D
	v_readlane_b32 s10, v29, 30                                // 0000000082C4: D760000A 00013D1D
	v_readlane_b32 s11, v29, 31                                // 0000000082CC: D760000B 00013F1D
	s_or_saveexec_b32 s105, -1                                 // 0000000082D4: BEE922C1
	scratch_load_b32 v28, off, off offset:20                   // 0000000082D8: ED05007C 0000001C 00001400
	s_wait_alu 0xfffe                                          // 0000000082E4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000082E8: BEFE0069
	s_add_f32 s100, s92, s4                                    // 0000000082EC: A064045C
	v_readlane_b32 s92, v26, 12                                // 0000000082F0: D760005C 0001191A
	s_wait_loadcnt 0x0                                         // 0000000082F8: BFC00000
	v_readlane_b32 s12, v28, 0                                 // 0000000082FC: D760000C 0001011C
	v_readlane_b32 s13, v28, 1                                 // 000000008304: D760000D 0001031C
	v_readlane_b32 s14, v28, 2                                 // 00000000830C: D760000E 0001051C
	v_readlane_b32 s15, v28, 3                                 // 000000008314: D760000F 0001071C
	s_add_f32 s102, s92, s0                                    // 00000000831C: A066005C
	s_wait_alu 0xfffe                                          // 000000008320: BF88FFFE
	s_add_f32 s100, s5, s100                                   // 000000008324: A0646405
	s_or_saveexec_b32 s105, -1                                 // 000000008328: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000832C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008330: BEFE0069
	v_readlane_b32 s92, v20, 25                                // 000000008334: D760005C 00013314
	s_add_f32 s102, s1, s102                                   // 00000000833C: A0666601
	s_add_f32 s100, s6, s100                                   // 000000008340: A0646406
	s_wait_alu 0xfffe                                          // 000000008344: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000008348: BF870009
	s_add_f32 s102, s2, s102                                   // 00000000834C: A0666602
	s_add_f32 s103, s92, s9                                    // 000000008350: A067095C
	s_add_f32 s100, s7, s100                                   // 000000008354: A0646407
	v_readlane_b32 s92, v20, 26                                // 000000008358: D760005C 00013514
	s_wait_alu 0xfffe                                          // 000000008360: BF88FFFE
	s_add_f32 s102, s3, s102                                   // 000000008364: A0666603
	s_add_f32 s103, s10, s103                                  // 000000008368: A067670A
	s_add_f32 s100, s100, s8                                   // 00000000836C: A0640864
	s_wait_alu 0xfffe                                          // 000000008370: BF88FFFE
	s_add_f32 s102, s4, s102                                   // 000000008374: A0666604
	s_add_f32 s103, s11, s103                                  // 000000008378: A067670B
	s_add_f32 s100, s9, s100                                   // 00000000837C: A0646409
	s_wait_alu 0xfffe                                          // 000000008380: BF88FFFE
	s_add_f32 s102, s5, s102                                   // 000000008384: A0666605
	s_add_f32 s103, s103, s12                                  // 000000008388: A0670C67
	s_add_f32 s100, s10, s100                                  // 00000000838C: A064640A
	s_wait_alu 0xfffe                                          // 000000008390: BF88FFFE
	s_add_f32 s40, s6, s102                                    // 000000008394: A0286606
	s_add_f32 s103, s13, s103                                  // 000000008398: A067670D
	s_add_f32 s102, s11, s100                                  // 00000000839C: A066640B
	s_wait_alu 0xfffe                                          // 0000000083A0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 0000000083A4: BF870529
	s_add_f32 s103, s14, s103                                  // 0000000083A8: A067670E
	s_wait_alu 0xfffe                                          // 0000000083AC: BF88FFFE
	s_add_f32 s43, s15, s103                                   // 0000000083B0: A02B670F
	v_readlane_b32 s0, v23, 13                                 // 0000000083B4: D7600000 00011B17
	v_readlane_b32 s4, v23, 17                                 // 0000000083BC: D7600004 00012317
	v_readlane_b32 s5, v23, 18                                 // 0000000083C4: D7600005 00012517
	v_readlane_b32 s9, v23, 22                                 // 0000000083CC: D7600009 00012D17
	v_readlane_b32 s1, v23, 14                                 // 0000000083D4: D7600001 00011D17
	v_readlane_b32 s6, v23, 19                                 // 0000000083DC: D7600006 00012717
	s_add_f32 s100, s37, s4                                    // 0000000083E4: A0640425
	v_readlane_b32 s10, v23, 23                                // 0000000083E8: D760000A 00012F17
	s_add_f32 s103, s36, s0                                    // 0000000083F0: A0670024
	s_add_f32 s92, s92, s9                                     // 0000000083F4: A05C095C
	s_wait_alu 0xfffe                                          // 0000000083F8: BF88FFFE
	s_add_f32 s100, s5, s100                                   // 0000000083FC: A0646405
	v_readlane_b32 s2, v23, 15                                 // 000000008400: D7600002 00011F17
	v_readlane_b32 s7, v23, 20                                 // 000000008408: D7600007 00012917
	v_readlane_b32 s11, v23, 24                                // 000000008410: D760000B 00013117
	s_add_f32 s93, s1, s103                                    // 000000008418: A05D6701
	s_wait_alu 0xfffe                                          // 00000000841C: BF88FFFE
	s_add_f32 s94, s6, s100                                    // 000000008420: A05E6406
	s_add_f32 s92, s10, s92                                    // 000000008424: A05C5C0A
	v_readlane_b32 s3, v23, 16                                 // 000000008428: D7600003 00012117
	v_readlane_b32 s8, v23, 21                                 // 000000008430: D7600008 00012B17
	v_readlane_b32 s12, v23, 25                                // 000000008438: D760000C 00013317
	s_add_f32 s93, s2, s93                                     // 000000008440: A05D5D02
	s_wait_alu 0xfffe                                          // 000000008444: BF88FFFE
	s_add_f32 s94, s7, s94                                     // 000000008448: A05E5E07
	s_add_f32 s92, s11, s92                                    // 00000000844C: A05C5C0B
	v_readlane_b32 s13, v23, 26                                // 000000008450: D760000D 00013517
	s_add_f32 s93, s3, s93                                     // 000000008458: A05D5D03
	s_wait_alu 0xfffe                                          // 00000000845C: BF88FFFE
	s_add_f32 s94, s94, s8                                     // 000000008460: A05E085E
	s_add_f32 s92, s92, s12                                    // 000000008464: A05C0C5C
	v_readlane_b32 s14, v23, 27                                // 000000008468: D760000E 00013717
	s_add_f32 s93, s4, s93                                     // 000000008470: A05D5D04
	s_wait_alu 0xfffe                                          // 000000008474: BF88FFFE
	s_add_f32 s94, s9, s94                                     // 000000008478: A05E5E09
	s_add_f32 s92, s13, s92                                    // 00000000847C: A05C5C0D
	v_readlane_b32 s15, v23, 28                                // 000000008480: D760000F 00013917
	s_add_f32 s93, s5, s93                                     // 000000008488: A05D5D05
	s_wait_alu 0xfffe                                          // 00000000848C: BF88FFFE
	s_add_f32 s94, s10, s94                                    // 000000008490: A05E5E0A
	s_add_f32 s92, s14, s92                                    // 000000008494: A05C5C0E
	s_add_f32 s36, s6, s93                                     // 000000008498: A0245D06
	s_wait_alu 0xfffe                                          // 00000000849C: BF88FFFE
	s_add_f32 s103, s11, s94                                   // 0000000084A0: A0675E0B
	s_add_f32 s37, s15, s92                                    // 0000000084A4: A0255C0F
	s_or_saveexec_b32 s105, -1                                 // 0000000084A8: BEE922C1
	scratch_load_b32 v29, off, off offset:44                   // 0000000084AC: ED05007C 0000001D 00002C00
	s_wait_alu 0xfffe                                          // 0000000084B8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000084BC: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000084C0: BFC00000
	v_readlane_b32 s0, v29, 2                                  // 0000000084C4: D7600000 0001051D
	v_readlane_b32 s4, v29, 6                                  // 0000000084CC: D7600004 00010D1D
	v_readlane_b32 s8, v29, 10                                 // 0000000084D4: D7600008 0001151D
	v_readlane_b32 s9, v29, 11                                 // 0000000084DC: D7600009 0001171D
	v_readlane_b32 s1, v29, 3                                  // 0000000084E4: D7600001 0001071D
	v_readlane_b32 s2, v29, 4                                  // 0000000084EC: D7600002 0001091D
	v_readlane_b32 s3, v29, 5                                  // 0000000084F4: D7600003 00010B1D
	v_readlane_b32 s5, v29, 7                                  // 0000000084FC: D7600005 00010F1D
	v_readlane_b32 s6, v29, 8                                  // 000000008504: D7600006 0001111D
	v_readlane_b32 s7, v29, 9                                  // 00000000850C: D7600007 0001131D
	v_readlane_b32 s10, v29, 12                                // 000000008514: D760000A 0001191D
	v_readlane_b32 s11, v29, 13                                // 00000000851C: D760000B 00011B1D
	v_readlane_b32 s12, v29, 14                                // 000000008524: D760000C 00011D1D
	v_readlane_b32 s13, v29, 15                                // 00000000852C: D760000D 00011F1D
	v_readlane_b32 s14, v29, 16                                // 000000008534: D760000E 0001211D
	s_or_saveexec_b32 s105, -1                                 // 00000000853C: BEE922C1
	s_wait_alu 0xfffe                                          // 000000008540: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008544: BEFE0069
	s_add_f32 s33, s33, s4                                     // 000000008548: A0210421
	s_add_f32 s92, s38, s9                                     // 00000000854C: A05C0926
	s_add_f32 s34, s34, s0                                     // 000000008550: A0220022
	v_readlane_b32 s15, v29, 17                                // 000000008554: D760000F 0001231D
	s_wait_alu 0xfffe                                          // 00000000855C: BF88FFFE
	s_add_f32 s33, s5, s33                                     // 000000008560: A0212105
	s_add_f32 s92, s10, s92                                    // 000000008564: A05C5C0A
	s_add_f32 s34, s1, s34                                     // 000000008568: A0222201
	s_wait_alu 0xfffe                                          // 00000000856C: BF88FFFE
	s_add_f32 s33, s6, s33                                     // 000000008570: A0212106
	s_add_f32 s92, s11, s92                                    // 000000008574: A05C5C0B
	s_add_f32 s34, s2, s34                                     // 000000008578: A0222202
	s_wait_alu 0xfffe                                          // 00000000857C: BF88FFFE
	s_add_f32 s33, s7, s33                                     // 000000008580: A0212107
	s_add_f32 s92, s92, s12                                    // 000000008584: A05C0C5C
	s_add_f32 s34, s3, s34                                     // 000000008588: A0222203
	s_wait_alu 0xfffe                                          // 00000000858C: BF88FFFE
	s_add_f32 s33, s33, s8                                     // 000000008590: A0210821
	s_add_f32 s92, s13, s92                                    // 000000008594: A05C5C0D
	s_add_f32 s34, s4, s34                                     // 000000008598: A0222204
	s_wait_alu 0xfffe                                          // 00000000859C: BF88FFFE
	s_add_f32 s33, s9, s33                                     // 0000000085A0: A0212109
	s_add_f32 s92, s14, s92                                    // 0000000085A4: A05C5C0E
	s_add_f32 s34, s5, s34                                     // 0000000085A8: A0222205
	s_wait_alu 0xfffe                                          // 0000000085AC: BF88FFFE
	s_add_f32 s33, s10, s33                                    // 0000000085B0: A021210A
	s_add_f32 s44, s15, s92                                    // 0000000085B4: A02C5C0F
	s_add_f32 s93, s6, s34                                     // 0000000085B8: A05D2206
	s_wait_alu 0xfffe                                          // 0000000085BC: BF88FFFE
	s_add_f32 s94, s11, s33                                    // 0000000085C0: A05E210B
	s_or_saveexec_b32 s105, -1                                 // 0000000085C4: BEE922C1
	scratch_load_b32 v26, off, off offset:52                   // 0000000085C8: ED05007C 0000001A 00003400
	s_wait_alu 0xfffe                                          // 0000000085D4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000085D8: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000085DC: BFC00000
	v_readlane_b32 s0, v26, 10                                 // 0000000085E0: D7600000 0001151A
	v_readlane_b32 s4, v26, 14                                 // 0000000085E8: D7600004 00011D1A
	v_readlane_b32 s5, v26, 15                                 // 0000000085F0: D7600005 00011F1A
	v_readlane_b32 s9, v26, 19                                 // 0000000085F8: D7600009 0001271A
	v_readlane_b32 s1, v26, 11                                 // 000000008600: D7600001 0001171A
	v_readlane_b32 s6, v26, 16                                 // 000000008608: D7600006 0001211A
	s_add_f32 s33, s35, s4                                     // 000000008610: A0210423
	v_readlane_b32 s10, v26, 20                                // 000000008614: D760000A 0001291A
	s_add_f32 s34, s101, s0                                    // 00000000861C: A0220065
	s_add_f32 s35, s41, s9                                     // 000000008620: A0230929
	s_wait_alu 0xfffe                                          // 000000008624: BF88FFFE
	s_add_f32 s33, s5, s33                                     // 000000008628: A0212105
	v_readlane_b32 s2, v26, 12                                 // 00000000862C: D7600002 0001191A
	v_readlane_b32 s7, v26, 17                                 // 000000008634: D7600007 0001231A
	v_readlane_b32 s11, v26, 21                                // 00000000863C: D760000B 00012B1A
	s_add_f32 s34, s1, s34                                     // 000000008644: A0222201
	s_wait_alu 0xfffe                                          // 000000008648: BF88FFFE
	s_add_f32 s33, s6, s33                                     // 00000000864C: A0212106
	s_add_f32 s35, s10, s35                                    // 000000008650: A023230A
	v_readlane_b32 s3, v26, 13                                 // 000000008654: D7600003 00011B1A
	v_readlane_b32 s8, v26, 18                                 // 00000000865C: D7600008 0001251A
	v_readlane_b32 s12, v26, 22                                // 000000008664: D760000C 00012D1A
	s_add_f32 s34, s2, s34                                     // 00000000866C: A0222202
	s_wait_alu 0xfffe                                          // 000000008670: BF88FFFE
	s_add_f32 s33, s7, s33                                     // 000000008674: A0212107
	s_add_f32 s35, s11, s35                                    // 000000008678: A023230B
	v_readlane_b32 s13, v26, 23                                // 00000000867C: D760000D 00012F1A
	s_add_f32 s34, s3, s34                                     // 000000008684: A0222203
	s_wait_alu 0xfffe                                          // 000000008688: BF88FFFE
	s_add_f32 s33, s33, s8                                     // 00000000868C: A0210821
	s_add_f32 s35, s35, s12                                    // 000000008690: A0230C23
	v_readlane_b32 s14, v26, 24                                // 000000008694: D760000E 0001311A
	s_add_f32 s34, s34, s4                                     // 00000000869C: A0220422
	s_wait_alu 0xfffe                                          // 0000000086A0: BF88FFFE
	s_add_f32 s33, s9, s33                                     // 0000000086A4: A0212109
	s_add_f32 s35, s13, s35                                    // 0000000086A8: A023230D
	v_readlane_b32 s15, v26, 25                                // 0000000086AC: D760000F 0001331A
	s_add_f32 s34, s5, s34                                     // 0000000086B4: A0222205
	s_wait_alu 0xfffe                                          // 0000000086B8: BF88FFFE
	s_add_f32 s33, s10, s33                                    // 0000000086BC: A021210A
	s_add_f32 s35, s14, s35                                    // 0000000086C0: A023230E
	s_add_f32 s34, s6, s34                                     // 0000000086C4: A0222206
	s_wait_alu 0xfffe                                          // 0000000086C8: BF88FFFE
	s_add_f32 s33, s11, s33                                    // 0000000086CC: A021210B
	s_add_f32 s41, s15, s35                                    // 0000000086D0: A029230F
	s_or_saveexec_b32 s105, -1                                 // 0000000086D4: BEE922C1
	s_wait_alu 0xfffe                                          // 0000000086D8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000086DC: BEFE0069
	v_readlane_b32 s0, v29, 26                                 // 0000000086E0: D7600000 0001351D
	v_readlane_b32 s4, v29, 30                                 // 0000000086E8: D7600004 00013D1D
	v_readlane_b32 s1, v29, 27                                 // 0000000086F0: D7600001 0001371D
	v_readlane_b32 s2, v29, 28                                 // 0000000086F8: D7600002 0001391D
	v_readlane_b32 s3, v29, 29                                 // 000000008700: D7600003 00013B1D
	v_readlane_b32 s5, v29, 31                                 // 000000008708: D7600005 00013F1D
	s_or_saveexec_b32 s105, -1                                 // 000000008710: BEE922C1
	scratch_load_b32 v29, off, off offset:48                   // 000000008714: ED05007C 0000001D 00003000
	s_wait_alu 0xfffe                                          // 000000008720: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008724: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000008728: BFC00000
	v_readlane_b32 s9, v29, 3                                  // 00000000872C: D7600009 0001071D
	v_readlane_b32 s10, v29, 4                                 // 000000008734: D760000A 0001091D
	s_add_f32 s35, s104, s4                                    // 00000000873C: A0230468
	v_readlane_b32 s6, v29, 0                                  // 000000008740: D7600006 0001011D
	v_readlane_b32 s11, v29, 5                                 // 000000008748: D760000B 00010B1D
	s_add_f32 s95, s42, s9                                     // 000000008750: A05F092A
	s_add_f32 s92, s39, s0                                     // 000000008754: A05C0027
	s_wait_alu 0xfffe                                          // 000000008758: BF88FFFE
	s_add_f32 s35, s5, s35                                     // 00000000875C: A0232305
	v_readlane_b32 s7, v29, 1                                  // 000000008760: D7600007 0001031D
	s_add_f32 s95, s10, s95                                    // 000000008768: A05F5F0A
	v_readlane_b32 s12, v29, 6                                 // 00000000876C: D760000C 00010D1D
	s_add_f32 s92, s1, s92                                     // 000000008774: A05C5C01
	s_wait_alu 0xfffe                                          // 000000008778: BF88FFFE
	s_add_f32 s35, s6, s35                                     // 00000000877C: A0232306
	s_add_f32 s95, s11, s95                                    // 000000008780: A05F5F0B
	v_readlane_b32 s8, v29, 2                                  // 000000008784: D7600008 0001051D
	v_readlane_b32 s13, v29, 7                                 // 00000000878C: D760000D 00010F1D
	s_add_f32 s92, s2, s92                                     // 000000008794: A05C5C02
	s_wait_alu 0xfffe                                          // 000000008798: BF88FFFE
	s_add_f32 s35, s7, s35                                     // 00000000879C: A0232307
	s_add_f32 s95, s95, s12                                    // 0000000087A0: A05F0C5F
	v_readlane_b32 s14, v29, 8                                 // 0000000087A4: D760000E 0001111D
	s_add_f32 s92, s3, s92                                     // 0000000087AC: A05C5C03
	s_wait_alu 0xfffe                                          // 0000000087B0: BF88FFFE
	s_add_f32 s35, s35, s8                                     // 0000000087B4: A0230823
	s_add_f32 s95, s13, s95                                    // 0000000087B8: A05F5F0D
	v_readlane_b32 s15, v29, 9                                 // 0000000087BC: D760000F 0001131D
	s_add_f32 s92, s4, s92                                     // 0000000087C4: A05C5C04
	s_wait_alu 0xfffe                                          // 0000000087C8: BF88FFFE
	s_add_f32 s35, s9, s35                                     // 0000000087CC: A0232309
	s_add_f32 s95, s14, s95                                    // 0000000087D0: A05F5F0E
	s_add_f32 s92, s5, s92                                     // 0000000087D4: A05C5C05
	s_wait_alu 0xfffe                                          // 0000000087D8: BF88FFFE
	s_add_f32 s35, s10, s35                                    // 0000000087DC: A023230A
	s_add_f32 s42, s15, s95                                    // 0000000087E0: A02A5F0F
	s_add_f32 s38, s6, s92                                     // 0000000087E4: A0265C06
	s_wait_alu 0xfffe                                          // 0000000087E8: BF88FFFE
	s_add_f32 s101, s11, s35                                   // 0000000087EC: A065230B
	s_or_saveexec_b32 s105, -1                                 // 0000000087F0: BEE922C1
	scratch_load_b32 v29, off, off offset:64                   // 0000000087F4: ED05007C 0000001D 00004000
	s_wait_alu 0xfffe                                          // 000000008800: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008804: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000008808: BFC00000
	v_readlane_b32 s0, v29, 28                                 // 00000000880C: D7600000 0001391D
	v_readlane_b32 s1, v29, 29                                 // 000000008814: D7600001 00013B1D
	v_readlane_b32 s2, v29, 30                                 // 00000000881C: D7600002 00013D1D
	v_readlane_b32 s3, v29, 31                                 // 000000008824: D7600003 00013F1D
	s_or_saveexec_b32 s105, -1                                 // 00000000882C: BEE922C1
	v_mov_b32_e32 v23, v29                                     // 000000008830: 7E2E031D
	s_wait_alu 0xfffe                                          // 000000008834: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008838: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 00000000883C: BEE922C1
	scratch_load_b32 v29, off, off offset:68                   // 000000008840: ED05007C 0000001D 00004400
	s_wait_alu 0xfffe                                          // 00000000884C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008850: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000008854: BFC00000
	v_readlane_b32 s4, v29, 0                                  // 000000008858: D7600004 0001011D
	v_readlane_b32 s8, v29, 4                                  // 000000008860: D7600008 0001091D
	v_readlane_b32 s9, v29, 5                                  // 000000008868: D7600009 00010B1D
	v_readlane_b32 s5, v29, 1                                  // 000000008870: D7600005 0001031D
	v_readlane_b32 s6, v29, 2                                  // 000000008878: D7600006 0001051D
	v_readlane_b32 s7, v29, 3                                  // 000000008880: D7600007 0001071D
	v_readlane_b32 s10, v29, 6                                 // 000000008888: D760000A 00010D1D
	v_readlane_b32 s11, v29, 7                                 // 000000008890: D760000B 00010F1D
	v_readlane_b32 s12, v29, 8                                 // 000000008898: D760000C 0001111D
	v_readlane_b32 s13, v29, 9                                 // 0000000088A0: D760000D 0001131D
	v_readlane_b32 s14, v29, 10                                // 0000000088A8: D760000E 0001151D
	v_readlane_b32 s15, v29, 11                                // 0000000088B0: D760000F 0001171D
	s_or_saveexec_b32 s105, -1                                 // 0000000088B8: BEE922C1
	v_mov_b32_e32 v20, v29                                     // 0000000088BC: 7E28031D
	s_wait_alu 0xfffe                                          // 0000000088C0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000088C4: BEFE0069
	s_add_f32 s35, s102, s4                                    // 0000000088C8: A0230466
	s_add_f32 s92, s40, s0                                     // 0000000088CC: A05C0028
	s_add_f32 s95, s43, s9                                     // 0000000088D0: A05F092B
	s_wait_alu 0xfffe                                          // 0000000088D4: BF88FFFE
	s_add_f32 s35, s5, s35                                     // 0000000088D8: A0232305
	s_add_f32 s92, s1, s92                                     // 0000000088DC: A05C5C01
	s_add_f32 s95, s10, s95                                    // 0000000088E0: A05F5F0A
	s_wait_alu 0xfffe                                          // 0000000088E4: BF88FFFE
	s_add_f32 s35, s6, s35                                     // 0000000088E8: A0232306
	s_add_f32 s92, s2, s92                                     // 0000000088EC: A05C5C02
	s_add_f32 s95, s11, s95                                    // 0000000088F0: A05F5F0B
	s_wait_alu 0xfffe                                          // 0000000088F4: BF88FFFE
	s_add_f32 s35, s7, s35                                     // 0000000088F8: A0232307
	s_add_f32 s92, s3, s92                                     // 0000000088FC: A05C5C03
	s_add_f32 s95, s95, s12                                    // 000000008900: A05F0C5F
	s_wait_alu 0xfffe                                          // 000000008904: BF88FFFE
	s_add_f32 s35, s35, s8                                     // 000000008908: A0230823
	s_add_f32 s92, s4, s92                                     // 00000000890C: A05C5C04
	s_add_f32 s95, s13, s95                                    // 000000008910: A05F5F0D
	s_wait_alu 0xfffe                                          // 000000008914: BF88FFFE
	s_add_f32 s35, s9, s35                                     // 000000008918: A0232309
	s_add_f32 s92, s5, s92                                     // 00000000891C: A05C5C05
	s_add_f32 s95, s14, s95                                    // 000000008920: A05F5F0E
	s_wait_alu 0xfffe                                          // 000000008924: BF88FFFE
	s_add_f32 s35, s10, s35                                    // 000000008928: A023230A
	s_add_f32 s39, s6, s92                                     // 00000000892C: A0275C06
	s_add_f32 s40, s15, s95                                    // 000000008930: A0285F0F
	s_wait_alu 0xfffe                                          // 000000008934: BF88FFFE
	s_add_f32 s102, s11, s35                                   // 000000008938: A066230B
	s_or_saveexec_b32 s105, -1                                 // 00000000893C: BEE922C1
	scratch_load_b32 v29, off, off offset:96                   // 000000008940: ED05007C 0000001D 00006000
	s_wait_alu 0xfffe                                          // 00000000894C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008950: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000008954: BFC00000
	v_readlane_b32 s0, v29, 20                                 // 000000008958: D7600000 0001291D
	v_readlane_b32 s4, v29, 24                                 // 000000008960: D7600004 0001311D
	v_readlane_b32 s8, v29, 28                                 // 000000008968: D7600008 0001391D
	v_readlane_b32 s9, v29, 29                                 // 000000008970: D7600009 00013B1D
	v_readlane_b32 s1, v29, 21                                 // 000000008978: D7600001 00012B1D
	v_readlane_b32 s2, v29, 22                                 // 000000008980: D7600002 00012D1D
	v_readlane_b32 s3, v29, 23                                 // 000000008988: D7600003 00012F1D
	v_readlane_b32 s5, v29, 25                                 // 000000008990: D7600005 0001331D
	v_readlane_b32 s6, v29, 26                                 // 000000008998: D7600006 0001351D
	v_readlane_b32 s7, v29, 27                                 // 0000000089A0: D7600007 0001371D
	v_readlane_b32 s10, v29, 30                                // 0000000089A8: D760000A 00013D1D
	v_readlane_b32 s11, v29, 31                                // 0000000089B0: D760000B 00013F1D
	s_or_saveexec_b32 s105, -1                                 // 0000000089B8: BEE922C1
	scratch_load_b32 v26, off, off offset:100 th:TH_LOAD_LU    // 0000000089BC: ED05007C 0030001A 00006400
	s_wait_alu 0xfffe                                          // 0000000089C8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000089CC: BEFE0069
	s_add_f32 s35, s103, s4                                    // 0000000089D0: A0230467
	s_add_f32 s92, s36, s0                                     // 0000000089D4: A05C0024
	s_add_f32 s95, s37, s9                                     // 0000000089D8: A05F0925
	s_wait_loadcnt 0x0                                         // 0000000089DC: BFC00000
	v_readlane_b32 s12, v26, 0                                 // 0000000089E0: D760000C 0001011A
	s_wait_alu 0xfffe                                          // 0000000089E8: BF88FFFE
	s_add_f32 s35, s5, s35                                     // 0000000089EC: A0232305
	s_add_f32 s92, s1, s92                                     // 0000000089F0: A05C5C01
	s_add_f32 s95, s10, s95                                    // 0000000089F4: A05F5F0A
	v_readlane_b32 s13, v26, 1                                 // 0000000089F8: D760000D 0001031A
	s_wait_alu 0xfffe                                          // 000000008A00: BF88FFFE
	s_add_f32 s35, s6, s35                                     // 000000008A04: A0232306
	s_add_f32 s92, s2, s92                                     // 000000008A08: A05C5C02
	s_add_f32 s95, s11, s95                                    // 000000008A0C: A05F5F0B
	v_readlane_b32 s14, v26, 2                                 // 000000008A10: D760000E 0001051A
	s_wait_alu 0xfffe                                          // 000000008A18: BF88FFFE
	s_add_f32 s35, s7, s35                                     // 000000008A1C: A0232307
	s_add_f32 s92, s3, s92                                     // 000000008A20: A05C5C03
	s_add_f32 s95, s95, s12                                    // 000000008A24: A05F0C5F
	v_readlane_b32 s15, v26, 3                                 // 000000008A28: D760000F 0001071A
	s_wait_alu 0xfffe                                          // 000000008A30: BF88FFFE
	s_add_f32 s35, s35, s8                                     // 000000008A34: A0230823
	s_add_f32 s92, s4, s92                                     // 000000008A38: A05C5C04
	s_add_f32 s95, s13, s95                                    // 000000008A3C: A05F5F0D
	s_wait_alu 0xfffe                                          // 000000008A40: BF88FFFE
	s_add_f32 s35, s9, s35                                     // 000000008A44: A0232309
	s_add_f32 s92, s5, s92                                     // 000000008A48: A05C5C05
	s_add_f32 s95, s14, s95                                    // 000000008A4C: A05F5F0E
	s_wait_alu 0xfffe                                          // 000000008A50: BF88FFFE
	s_add_f32 s35, s10, s35                                    // 000000008A54: A023230A
	s_add_f32 s36, s6, s92                                     // 000000008A58: A0245C06
	s_add_f32 s37, s15, s95                                    // 000000008A5C: A0255F0F
	s_wait_alu 0xfffe                                          // 000000008A60: BF88FFFE
	s_add_f32 s35, s11, s35                                    // 000000008A64: A023230B
	s_or_saveexec_b32 s105, -1                                 // 000000008A68: BEE922C1
	v_mov_b32_e32 v24, v27                                     // 000000008A6C: 7E30031B
	s_wait_alu 0xfffe                                          // 000000008A70: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008A74: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000008A78: BEE922C1
	scratch_load_b32 v27, off, off offset:16                   // 000000008A7C: ED05007C 0000001B 00001000
	s_wait_alu 0xfffe                                          // 000000008A88: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008A8C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000008A90: BFC00000
	v_readlane_b32 s0, v27, 4                                  // 000000008A94: D7600000 0001091B
	v_readlane_b32 s4, v27, 8                                  // 000000008A9C: D7600004 0001111B
	v_readlane_b32 s8, v27, 12                                 // 000000008AA4: D7600008 0001191B
	v_readlane_b32 s9, v27, 13                                 // 000000008AAC: D7600009 00011B1B
	v_readlane_b32 s1, v27, 5                                  // 000000008AB4: D7600001 00010B1B
	v_readlane_b32 s2, v27, 6                                  // 000000008ABC: D7600002 00010D1B
	v_readlane_b32 s3, v27, 7                                  // 000000008AC4: D7600003 00010F1B
	v_readlane_b32 s5, v27, 9                                  // 000000008ACC: D7600005 0001131B
	v_readlane_b32 s6, v27, 10                                 // 000000008AD4: D7600006 0001151B
	v_readlane_b32 s7, v27, 11                                 // 000000008ADC: D7600007 0001171B
	v_readlane_b32 s10, v27, 14                                // 000000008AE4: D760000A 00011D1B
	v_readlane_b32 s11, v27, 15                                // 000000008AEC: D760000B 00011F1B
	v_readlane_b32 s12, v27, 16                                // 000000008AF4: D760000C 0001211B
	v_readlane_b32 s13, v27, 17                                // 000000008AFC: D760000D 0001231B
	v_readlane_b32 s14, v27, 18                                // 000000008B04: D760000E 0001251B
	s_or_saveexec_b32 s105, -1                                 // 000000008B0C: BEE922C1
	s_wait_alu 0xfffe                                          // 000000008B10: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008B14: BEFE0069
	s_add_f32 s92, s94, s4                                     // 000000008B18: A05C045E
	s_add_f32 s93, s93, s0                                     // 000000008B1C: A05D005D
	s_add_f32 s94, s44, s9                                     // 000000008B20: A05E092C
	v_readlane_b32 s15, v27, 19                                // 000000008B24: D760000F 0001271B
	s_wait_alu 0xfffe                                          // 000000008B2C: BF88FFFE
	s_add_f32 s92, s5, s92                                     // 000000008B30: A05C5C05
	s_add_f32 s93, s1, s93                                     // 000000008B34: A05D5D01
	s_add_f32 s94, s10, s94                                    // 000000008B38: A05E5E0A
	s_wait_alu 0xfffe                                          // 000000008B3C: BF88FFFE
	s_add_f32 s92, s6, s92                                     // 000000008B40: A05C5C06
	s_add_f32 s93, s2, s93                                     // 000000008B44: A05D5D02
	s_add_f32 s94, s11, s94                                    // 000000008B48: A05E5E0B
	s_wait_alu 0xfffe                                          // 000000008B4C: BF88FFFE
	s_add_f32 s92, s7, s92                                     // 000000008B50: A05C5C07
	s_add_f32 s93, s3, s93                                     // 000000008B54: A05D5D03
	s_add_f32 s94, s94, s12                                    // 000000008B58: A05E0C5E
	s_wait_alu 0xfffe                                          // 000000008B5C: BF88FFFE
	s_add_f32 s92, s92, s8                                     // 000000008B60: A05C085C
	s_add_f32 s93, s4, s93                                     // 000000008B64: A05D5D04
	s_add_f32 s94, s13, s94                                    // 000000008B68: A05E5E0D
	s_wait_alu 0xfffe                                          // 000000008B6C: BF88FFFE
	s_add_f32 s92, s9, s92                                     // 000000008B70: A05C5C09
	s_add_f32 s93, s5, s93                                     // 000000008B74: A05D5D05
	s_add_f32 s94, s14, s94                                    // 000000008B78: A05E5E0E
	s_wait_alu 0xfffe                                          // 000000008B7C: BF88FFFE
	s_add_f32 s92, s10, s92                                    // 000000008B80: A05C5C0A
	s_add_f32 s93, s6, s93                                     // 000000008B84: A05D5D06
	s_add_f32 s43, s15, s94                                    // 000000008B88: A02B5E0F
	s_wait_alu 0xfffe                                          // 000000008B8C: BF88FFFE
	s_add_f32 s92, s11, s92                                    // 000000008B90: A05C5C0B
	s_or_saveexec_b32 s105, -1                                 // 000000008B94: BEE922C1
	s_wait_alu 0xfffe                                          // 000000008B98: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008B9C: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000008BA0: BEE922C1
	scratch_store_b32 off, v28, off offset:20                  // 000000008BA4: ED06807C 0E000000 00001400
	s_wait_alu 0xfffe                                          // 000000008BB0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008BB4: BEFE0069
	v_readlane_b32 s0, v28, 20                                 // 000000008BB8: D7600000 0001291C
	v_readlane_b32 s4, v28, 24                                 // 000000008BC0: D7600004 0001311C
	v_readlane_b32 s8, v28, 28                                 // 000000008BC8: D7600008 0001391C
	v_readlane_b32 s9, v28, 29                                 // 000000008BD0: D7600009 00013B1C
	v_readlane_b32 s1, v28, 21                                 // 000000008BD8: D7600001 00012B1C
	v_readlane_b32 s2, v28, 22                                 // 000000008BE0: D7600002 00012D1C
	v_readlane_b32 s3, v28, 23                                 // 000000008BE8: D7600003 00012F1C
	v_readlane_b32 s5, v28, 25                                 // 000000008BF0: D7600005 0001331C
	v_readlane_b32 s6, v28, 26                                 // 000000008BF8: D7600006 0001351C
	v_readlane_b32 s7, v28, 27                                 // 000000008C00: D7600007 0001371C
	v_readlane_b32 s10, v28, 30                                // 000000008C08: D760000A 00013D1C
	v_readlane_b32 s11, v28, 31                                // 000000008C10: D760000B 00013F1C
	s_or_saveexec_b32 s105, -1                                 // 000000008C18: BEE922C1
	s_wait_alu 0xfffe                                          // 000000008C1C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008C20: BEFE0069
	v_readlane_b32 s12, v23, 0                                 // 000000008C24: D760000C 00010117
	v_readlane_b32 s13, v23, 1                                 // 000000008C2C: D760000D 00010317
	v_readlane_b32 s14, v23, 2                                 // 000000008C34: D760000E 00010517
	v_readlane_b32 s15, v23, 3                                 // 000000008C3C: D760000F 00010717
	s_or_saveexec_b32 s105, -1                                 // 000000008C44: BEE922C1
	s_wait_alu 0xfffe                                          // 000000008C48: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008C4C: BEFE0069
	s_add_f32 s33, s33, s4                                     // 000000008C50: A0210421
	s_add_f32 s34, s34, s0                                     // 000000008C54: A0220022
	s_add_f32 s94, s41, s9                                     // 000000008C58: A05E0929
	s_wait_alu 0xfffe                                          // 000000008C5C: BF88FFFE
	s_add_f32 s33, s5, s33                                     // 000000008C60: A0212105
	s_add_f32 s34, s1, s34                                     // 000000008C64: A0222201
	s_add_f32 s94, s10, s94                                    // 000000008C68: A05E5E0A
	s_wait_alu 0xfffe                                          // 000000008C6C: BF88FFFE
	s_add_f32 s33, s6, s33                                     // 000000008C70: A0212106
	s_add_f32 s34, s2, s34                                     // 000000008C74: A0222202
	s_add_f32 s94, s11, s94                                    // 000000008C78: A05E5E0B
	s_wait_alu 0xfffe                                          // 000000008C7C: BF88FFFE
	s_add_f32 s33, s7, s33                                     // 000000008C80: A0212107
	s_add_f32 s34, s3, s34                                     // 000000008C84: A0222203
	s_add_f32 s94, s94, s12                                    // 000000008C88: A05E0C5E
	s_wait_alu 0xfffe                                          // 000000008C8C: BF88FFFE
	s_add_f32 s33, s33, s8                                     // 000000008C90: A0210821
	s_add_f32 s34, s34, s4                                     // 000000008C94: A0220422
	s_add_f32 s94, s13, s94                                    // 000000008C98: A05E5E0D
	s_wait_alu 0xfffe                                          // 000000008C9C: BF88FFFE
	s_add_f32 s33, s9, s33                                     // 000000008CA0: A0212109
	s_add_f32 s34, s5, s34                                     // 000000008CA4: A0222205
	s_add_f32 s94, s14, s94                                    // 000000008CA8: A05E5E0E
	s_wait_alu 0xfffe                                          // 000000008CAC: BF88FFFE
	s_add_f32 s33, s10, s33                                    // 000000008CB0: A021210A
	s_add_f32 s34, s6, s34                                     // 000000008CB4: A0222206
	s_add_f32 s104, s15, s94                                   // 000000008CB8: A0685E0F
	s_wait_alu 0xfffe                                          // 000000008CBC: BF88FFFE
	s_add_f32 s33, s11, s33                                    // 000000008CC0: A021210B
	s_or_saveexec_b32 s105, -1                                 // 000000008CC4: BEE922C1
	s_wait_alu 0xfffe                                          // 000000008CC8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008CCC: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000008CD0: BEE922C1
	scratch_store_b32 off, v27, off offset:16                  // 000000008CD4: ED06807C 0D800000 00001000
	s_wait_alu 0xfffe                                          // 000000008CE0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008CE4: BEFE0069
	v_readlane_b32 s0, v27, 28                                 // 000000008CE8: D7600000 0001391B
	v_readlane_b32 s1, v27, 29                                 // 000000008CF0: D7600001 00013B1B
	v_readlane_b32 s2, v27, 30                                 // 000000008CF8: D7600002 00013D1B
	v_readlane_b32 s3, v27, 31                                 // 000000008D00: D7600003 00013F1B
	s_or_saveexec_b32 s105, -1                                 // 000000008D08: BEE922C1
	scratch_load_b32 v28, off, off offset:56                   // 000000008D0C: ED05007C 0000001C 00003800
	s_wait_alu 0xfffe                                          // 000000008D18: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008D1C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000008D20: BFC00000
	v_readlane_b32 s4, v28, 0                                  // 000000008D24: D7600004 0001011C
	v_readlane_b32 s9, v28, 5                                  // 000000008D2C: D7600009 00010B1C
	v_readlane_b32 s5, v28, 1                                  // 000000008D34: D7600005 0001031C
	v_readlane_b32 s10, v28, 6                                 // 000000008D3C: D760000A 00010D1C
	v_readlane_b32 s6, v28, 2                                  // 000000008D44: D7600006 0001051C
	s_add_f32 s94, s101, s4                                    // 000000008D4C: A05E0465
	s_add_f32 s96, s42, s9                                     // 000000008D50: A060092A
	v_readlane_b32 s11, v28, 7                                 // 000000008D54: D760000B 00010F1C
	s_add_f32 s95, s38, s0                                     // 000000008D5C: A05F0026
	s_wait_alu 0xfffe                                          // 000000008D60: BF88FFFE
	s_add_f32 s94, s5, s94                                     // 000000008D64: A05E5E05
	s_add_f32 s96, s10, s96                                    // 000000008D68: A060600A
	v_readlane_b32 s7, v28, 3                                  // 000000008D6C: D7600007 0001071C
	v_readlane_b32 s12, v28, 8                                 // 000000008D74: D760000C 0001111C
	s_add_f32 s95, s1, s95                                     // 000000008D7C: A05F5F01
	s_wait_alu 0xfffe                                          // 000000008D80: BF88FFFE
	s_add_f32 s94, s6, s94                                     // 000000008D84: A05E5E06
	s_add_f32 s96, s11, s96                                    // 000000008D88: A060600B
	v_readlane_b32 s8, v28, 4                                  // 000000008D8C: D7600008 0001091C
	v_readlane_b32 s13, v28, 9                                 // 000000008D94: D760000D 0001131C
	s_add_f32 s95, s2, s95                                     // 000000008D9C: A05F5F02
	s_wait_alu 0xfffe                                          // 000000008DA0: BF88FFFE
	s_add_f32 s94, s7, s94                                     // 000000008DA4: A05E5E07
	s_add_f32 s96, s96, s12                                    // 000000008DA8: A0600C60
	v_readlane_b32 s14, v28, 10                                // 000000008DAC: D760000E 0001151C
	s_add_f32 s95, s3, s95                                     // 000000008DB4: A05F5F03
	s_wait_alu 0xfffe                                          // 000000008DB8: BF88FFFE
	s_add_f32 s94, s94, s8                                     // 000000008DBC: A05E085E
	s_add_f32 s96, s13, s96                                    // 000000008DC0: A060600D
	v_readlane_b32 s15, v28, 11                                // 000000008DC4: D760000F 0001171C
	s_add_f32 s95, s4, s95                                     // 000000008DCC: A05F5F04
	s_wait_alu 0xfffe                                          // 000000008DD0: BF88FFFE
	s_add_f32 s94, s9, s94                                     // 000000008DD4: A05E5E09
	s_add_f32 s96, s14, s96                                    // 000000008DD8: A060600E
	s_add_f32 s95, s5, s95                                     // 000000008DDC: A05F5F05
	s_wait_alu 0xfffe                                          // 000000008DE0: BF88FFFE
	s_add_f32 s94, s10, s94                                    // 000000008DE4: A05E5E0A
	s_add_f32 s41, s15, s96                                    // 000000008DE8: A029600F
	s_add_f32 s38, s6, s95                                     // 000000008DEC: A0265F06
	s_wait_alu 0xfffe                                          // 000000008DF0: BF88FFFE
	s_add_f32 s101, s11, s94                                   // 000000008DF4: A0655E0B
	s_or_saveexec_b32 s105, -1                                 // 000000008DF8: BEE922C1
	s_wait_alu 0xfffe                                          // 000000008DFC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008E00: BEFE0069
	v_readlane_b32 s0, v20, 28                                 // 000000008E04: D7600000 00013914
	v_readlane_b32 s1, v20, 29                                 // 000000008E0C: D7600001 00013B14
	v_readlane_b32 s2, v20, 30                                 // 000000008E14: D7600002 00013D14
	v_readlane_b32 s3, v20, 31                                 // 000000008E1C: D7600003 00013F14
	s_or_saveexec_b32 s105, -1                                 // 000000008E24: BEE922C1
	scratch_load_b32 v28, off, off offset:24                   // 000000008E28: ED05007C 0000001C 00001800
	s_wait_alu 0xfffe                                          // 000000008E34: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008E38: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000008E3C: BFC00000
	v_readlane_b32 s4, v28, 0                                  // 000000008E40: D7600004 0001011C
	v_readlane_b32 s8, v28, 4                                  // 000000008E48: D7600008 0001091C
	v_readlane_b32 s9, v28, 5                                  // 000000008E50: D7600009 00010B1C
	v_readlane_b32 s5, v28, 1                                  // 000000008E58: D7600005 0001031C
	v_readlane_b32 s6, v28, 2                                  // 000000008E60: D7600006 0001051C
	v_readlane_b32 s7, v28, 3                                  // 000000008E68: D7600007 0001071C
	v_readlane_b32 s10, v28, 6                                 // 000000008E70: D760000A 00010D1C
	v_readlane_b32 s11, v28, 7                                 // 000000008E78: D760000B 00010F1C
	v_readlane_b32 s12, v28, 8                                 // 000000008E80: D760000C 0001111C
	v_readlane_b32 s13, v28, 9                                 // 000000008E88: D760000D 0001131C
	v_readlane_b32 s14, v28, 10                                // 000000008E90: D760000E 0001151C
	v_readlane_b32 s15, v28, 11                                // 000000008E98: D760000F 0001171C
	s_or_saveexec_b32 s105, -1                                 // 000000008EA0: BEE922C1
	s_wait_alu 0xfffe                                          // 000000008EA4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008EA8: BEFE0069
	s_add_f32 s94, s102, s4                                    // 000000008EAC: A05E0466
	s_add_f32 s95, s39, s0                                     // 000000008EB0: A05F0027
	s_add_f32 s96, s40, s9                                     // 000000008EB4: A0600928
	s_wait_alu 0xfffe                                          // 000000008EB8: BF88FFFE
	s_add_f32 s94, s5, s94                                     // 000000008EBC: A05E5E05
	s_add_f32 s95, s1, s95                                     // 000000008EC0: A05F5F01
	s_add_f32 s96, s10, s96                                    // 000000008EC4: A060600A
	s_wait_alu 0xfffe                                          // 000000008EC8: BF88FFFE
	s_add_f32 s94, s6, s94                                     // 000000008ECC: A05E5E06
	s_add_f32 s95, s2, s95                                     // 000000008ED0: A05F5F02
	s_add_f32 s96, s11, s96                                    // 000000008ED4: A060600B
	s_wait_alu 0xfffe                                          // 000000008ED8: BF88FFFE
	s_add_f32 s94, s7, s94                                     // 000000008EDC: A05E5E07
	s_add_f32 s95, s3, s95                                     // 000000008EE0: A05F5F03
	s_add_f32 s96, s96, s12                                    // 000000008EE4: A0600C60
	s_wait_alu 0xfffe                                          // 000000008EE8: BF88FFFE
	s_add_f32 s94, s94, s8                                     // 000000008EEC: A05E085E
	s_add_f32 s95, s4, s95                                     // 000000008EF0: A05F5F04
	s_add_f32 s96, s13, s96                                    // 000000008EF4: A060600D
	s_wait_alu 0xfffe                                          // 000000008EF8: BF88FFFE
	s_add_f32 s94, s9, s94                                     // 000000008EFC: A05E5E09
	s_add_f32 s95, s5, s95                                     // 000000008F00: A05F5F05
	s_add_f32 s96, s14, s96                                    // 000000008F04: A060600E
	s_wait_alu 0xfffe                                          // 000000008F08: BF88FFFE
	s_add_f32 s94, s10, s94                                    // 000000008F0C: A05E5E0A
	s_add_f32 s39, s6, s95                                     // 000000008F10: A0275F06
	s_add_f32 s103, s15, s96                                   // 000000008F14: A067600F
	s_wait_alu 0xfffe                                          // 000000008F18: BF88FFFE
	s_add_f32 s102, s11, s94                                   // 000000008F1C: A0665E0B
	s_or_saveexec_b32 s105, -1                                 // 000000008F20: BEE922C1
	s_wait_alu 0xfffe                                          // 000000008F24: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000008F28: BEFE0069
	v_readlane_b32 s0, v23, 12                                 // 000000008F2C: D7600000 00011917
	v_readlane_b32 s4, v23, 16                                 // 000000008F34: D7600004 00012117
	v_readlane_b32 s5, v23, 17                                 // 000000008F3C: D7600005 00012317
	v_readlane_b32 s9, v23, 21                                 // 000000008F44: D7600009 00012B17
	v_readlane_b32 s1, v23, 13                                 // 000000008F4C: D7600001 00011B17
	v_readlane_b32 s6, v23, 18                                 // 000000008F54: D7600006 00012517
	s_add_f32 s35, s35, s4                                     // 000000008F5C: A0230423
	v_readlane_b32 s10, v23, 22                                // 000000008F60: D760000A 00012D17
	s_add_f32 s94, s36, s0                                     // 000000008F68: A05E0024
	s_add_f32 s95, s37, s9                                     // 000000008F6C: A05F0925
	s_wait_alu 0xfffe                                          // 000000008F70: BF88FFFE
	s_add_f32 s35, s5, s35                                     // 000000008F74: A0232305
	v_readlane_b32 s2, v23, 14                                 // 000000008F78: D7600002 00011D17
	v_readlane_b32 s7, v23, 19                                 // 000000008F80: D7600007 00012717
	v_readlane_b32 s11, v23, 23                                // 000000008F88: D760000B 00012F17
	s_add_f32 s94, s1, s94                                     // 000000008F90: A05E5E01
	s_wait_alu 0xfffe                                          // 000000008F94: BF88FFFE
	s_add_f32 s35, s6, s35                                     // 000000008F98: A0232306
	s_add_f32 s95, s10, s95                                    // 000000008F9C: A05F5F0A
	v_readlane_b32 s3, v23, 15                                 // 000000008FA0: D7600003 00011F17
	v_readlane_b32 s8, v23, 20                                 // 000000008FA8: D7600008 00012917
	v_readlane_b32 s12, v23, 24                                // 000000008FB0: D760000C 00013117
	s_add_f32 s94, s2, s94                                     // 000000008FB8: A05E5E02
	s_wait_alu 0xfffe                                          // 000000008FBC: BF88FFFE
	s_add_f32 s35, s7, s35                                     // 000000008FC0: A0232307
	s_add_f32 s95, s11, s95                                    // 000000008FC4: A05F5F0B
	v_readlane_b32 s13, v23, 25                                // 000000008FC8: D760000D 00013317
	s_add_f32 s94, s3, s94                                     // 000000008FD0: A05E5E03
	s_wait_alu 0xfffe                                          // 000000008FD4: BF88FFFE
	s_add_f32 s35, s35, s8                                     // 000000008FD8: A0230823
	s_add_f32 s95, s95, s12                                    // 000000008FDC: A05F0C5F
	v_readlane_b32 s14, v23, 26                                // 000000008FE0: D760000E 00013517
	s_add_f32 s94, s4, s94                                     // 000000008FE8: A05E5E04
	s_wait_alu 0xfffe                                          // 000000008FEC: BF88FFFE
	s_add_f32 s35, s9, s35                                     // 000000008FF0: A0232309
	s_add_f32 s95, s13, s95                                    // 000000008FF4: A05F5F0D
	v_readlane_b32 s15, v23, 27                                // 000000008FF8: D760000F 00013717
	s_add_f32 s94, s5, s94                                     // 000000009000: A05E5E05
	s_wait_alu 0xfffe                                          // 000000009004: BF88FFFE
	s_add_f32 s35, s10, s35                                    // 000000009008: A023230A
	s_add_f32 s95, s14, s95                                    // 00000000900C: A05F5F0E
	s_add_f32 s36, s6, s94                                     // 000000009010: A0245E06
	s_wait_alu 0xfffe                                          // 000000009014: BF88FFFE
	s_add_f32 s35, s11, s35                                    // 000000009018: A023230B
	s_add_f32 s37, s15, s95                                    // 00000000901C: A0255F0F
	s_or_saveexec_b32 s105, -1                                 // 000000009020: BEE922C1
	s_wait_alu 0xfffe                                          // 000000009024: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009028: BEFE0069
	v_readlane_b32 s0, v28, 12                                 // 00000000902C: D7600000 0001191C
	v_readlane_b32 s4, v28, 16                                 // 000000009034: D7600004 0001211C
	v_readlane_b32 s8, v28, 20                                 // 00000000903C: D7600008 0001291C
	v_readlane_b32 s9, v28, 21                                 // 000000009044: D7600009 00012B1C
	v_readlane_b32 s1, v28, 13                                 // 00000000904C: D7600001 00011B1C
	v_readlane_b32 s2, v28, 14                                 // 000000009054: D7600002 00011D1C
	v_readlane_b32 s3, v28, 15                                 // 00000000905C: D7600003 00011F1C
	v_readlane_b32 s5, v28, 17                                 // 000000009064: D7600005 0001231C
	v_readlane_b32 s6, v28, 18                                 // 00000000906C: D7600006 0001251C
	v_readlane_b32 s7, v28, 19                                 // 000000009074: D7600007 0001271C
	v_readlane_b32 s10, v28, 22                                // 00000000907C: D760000A 00012D1C
	v_readlane_b32 s11, v28, 23                                // 000000009084: D760000B 00012F1C
	v_readlane_b32 s12, v28, 24                                // 00000000908C: D760000C 0001311C
	v_readlane_b32 s13, v28, 25                                // 000000009094: D760000D 0001331C
	v_readlane_b32 s14, v28, 26                                // 00000000909C: D760000E 0001351C
	s_or_saveexec_b32 s105, -1                                 // 0000000090A4: BEE922C1
	s_wait_alu 0xfffe                                          // 0000000090A8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000090AC: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 0000000090B0: BEE922C1
	scratch_store_b32 off, v28, off offset:24                  // 0000000090B4: ED06807C 0E000000 00001800
	s_wait_alu 0xfffe                                          // 0000000090C0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000090C4: BEFE0069
	s_add_f32 s92, s92, s4                                     // 0000000090C8: A05C045C
	s_add_f32 s93, s93, s0                                     // 0000000090CC: A05D005D
	s_add_f32 s94, s43, s9                                     // 0000000090D0: A05E092B
	v_readlane_b32 s15, v28, 27                                // 0000000090D4: D760000F 0001371C
	s_wait_alu 0xfffe                                          // 0000000090DC: BF88FFFE
	s_add_f32 s92, s5, s92                                     // 0000000090E0: A05C5C05
	s_add_f32 s93, s1, s93                                     // 0000000090E4: A05D5D01
	s_add_f32 s94, s10, s94                                    // 0000000090E8: A05E5E0A
	s_wait_alu 0xfffe                                          // 0000000090EC: BF88FFFE
	s_add_f32 s92, s6, s92                                     // 0000000090F0: A05C5C06
	s_add_f32 s93, s2, s93                                     // 0000000090F4: A05D5D02
	s_add_f32 s94, s11, s94                                    // 0000000090F8: A05E5E0B
	s_wait_alu 0xfffe                                          // 0000000090FC: BF88FFFE
	s_add_f32 s92, s7, s92                                     // 000000009100: A05C5C07
	s_add_f32 s93, s3, s93                                     // 000000009104: A05D5D03
	s_add_f32 s94, s94, s12                                    // 000000009108: A05E0C5E
	s_wait_alu 0xfffe                                          // 00000000910C: BF88FFFE
	s_add_f32 s92, s92, s8                                     // 000000009110: A05C085C
	s_add_f32 s93, s4, s93                                     // 000000009114: A05D5D04
	s_add_f32 s94, s13, s94                                    // 000000009118: A05E5E0D
	s_wait_alu 0xfffe                                          // 00000000911C: BF88FFFE
	s_add_f32 s92, s9, s92                                     // 000000009120: A05C5C09
	s_add_f32 s93, s5, s93                                     // 000000009124: A05D5D05
	s_add_f32 s94, s14, s94                                    // 000000009128: A05E5E0E
	s_wait_alu 0xfffe                                          // 00000000912C: BF88FFFE
	s_add_f32 s92, s10, s92                                    // 000000009130: A05C5C0A
	s_add_f32 s93, s6, s93                                     // 000000009134: A05D5D06
	s_add_f32 s40, s15, s94                                    // 000000009138: A0285E0F
	s_wait_alu 0xfffe                                          // 00000000913C: BF88FFFE
	s_add_f32 s92, s11, s92                                    // 000000009140: A05C5C0B
	v_readlane_b32 s0, v26, 4                                  // 000000009144: D7600000 0001091A
	v_readlane_b32 s4, v26, 8                                  // 00000000914C: D7600004 0001111A
	v_readlane_b32 s5, v26, 9                                  // 000000009154: D7600005 0001131A
	v_readlane_b32 s1, v26, 5                                  // 00000000915C: D7600001 00010B1A
	v_readlane_b32 s6, v26, 10                                 // 000000009164: D7600006 0001151A
	v_readlane_b32 s9, v26, 13                                 // 00000000916C: D7600009 00011B1A
	s_add_f32 s33, s33, s4                                     // 000000009174: A0210421
	s_add_f32 s34, s34, s0                                     // 000000009178: A0220022
	v_readlane_b32 s2, v26, 6                                  // 00000000917C: D7600002 00010D1A
	v_readlane_b32 s7, v26, 11                                 // 000000009184: D7600007 0001171A
	s_wait_alu 0xfffe                                          // 00000000918C: BF88FFFE
	s_add_f32 s33, s5, s33                                     // 000000009190: A0212105
	v_readlane_b32 s10, v26, 14                                // 000000009194: D760000A 00011D1A
	s_add_f32 s94, s104, s9                                    // 00000000919C: A05E0968
	s_add_f32 s34, s1, s34                                     // 0000000091A0: A0222201
	s_wait_alu 0xfffe                                          // 0000000091A4: BF88FFFE
	s_add_f32 s33, s6, s33                                     // 0000000091A8: A0212106
	v_readlane_b32 s3, v26, 7                                  // 0000000091AC: D7600003 00010F1A
	v_readlane_b32 s8, v26, 12                                 // 0000000091B4: D7600008 0001191A
	v_readlane_b32 s11, v26, 15                                // 0000000091BC: D760000B 00011F1A
	s_add_f32 s94, s10, s94                                    // 0000000091C4: A05E5E0A
	s_add_f32 s34, s2, s34                                     // 0000000091C8: A0222202
	s_wait_alu 0xfffe                                          // 0000000091CC: BF88FFFE
	s_add_f32 s33, s7, s33                                     // 0000000091D0: A0212107
	v_readlane_b32 s12, v26, 16                                // 0000000091D4: D760000C 0001211A
	s_add_f32 s94, s11, s94                                    // 0000000091DC: A05E5E0B
	s_add_f32 s34, s3, s34                                     // 0000000091E0: A0222203
	s_wait_alu 0xfffe                                          // 0000000091E4: BF88FFFE
	s_add_f32 s33, s33, s8                                     // 0000000091E8: A0210821
	v_readlane_b32 s13, v26, 17                                // 0000000091EC: D760000D 0001231A
	s_add_f32 s94, s94, s12                                    // 0000000091F4: A05E0C5E
	s_add_f32 s34, s34, s4                                     // 0000000091F8: A0220422
	s_wait_alu 0xfffe                                          // 0000000091FC: BF88FFFE
	s_add_f32 s33, s9, s33                                     // 000000009200: A0212109
	v_readlane_b32 s14, v26, 18                                // 000000009204: D760000E 0001251A
	s_add_f32 s94, s13, s94                                    // 00000000920C: A05E5E0D
	s_add_f32 s34, s5, s34                                     // 000000009210: A0222205
	s_wait_alu 0xfffe                                          // 000000009214: BF88FFFE
	s_add_f32 s33, s10, s33                                    // 000000009218: A021210A
	v_readlane_b32 s15, v26, 19                                // 00000000921C: D760000F 0001271A
	s_add_f32 s94, s14, s94                                    // 000000009224: A05E5E0E
	s_add_f32 s34, s6, s34                                     // 000000009228: A0222206
	s_wait_alu 0xfffe                                          // 00000000922C: BF88FFFE
	s_add_f32 s33, s11, s33                                    // 000000009230: A021210B
	s_add_f32 s104, s15, s94                                   // 000000009234: A0685E0F
	s_or_saveexec_b32 s105, -1                                 // 000000009238: BEE922C1
	scratch_load_b32 v29, off, off offset:28                   // 00000000923C: ED05007C 0000001D 00001C00
	s_wait_alu 0xfffe                                          // 000000009248: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000924C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000009250: BFC00000
	v_readlane_b32 s0, v29, 4                                  // 000000009254: D7600000 0001091D
	v_readlane_b32 s4, v29, 8                                  // 00000000925C: D7600004 0001111D
	v_readlane_b32 s9, v29, 13                                 // 000000009264: D7600009 00011B1D
	v_readlane_b32 s5, v29, 9                                  // 00000000926C: D7600005 0001131D
	v_readlane_b32 s10, v29, 14                                // 000000009274: D760000A 00011D1D
	v_readlane_b32 s1, v29, 5                                  // 00000000927C: D7600001 00010B1D
	s_add_f32 s94, s101, s4                                    // 000000009284: A05E0465
	s_add_f32 s96, s41, s9                                     // 000000009288: A0600929
	v_readlane_b32 s6, v29, 10                                 // 00000000928C: D7600006 0001151D
	v_readlane_b32 s11, v29, 15                                // 000000009294: D760000B 00011F1D
	s_add_f32 s95, s38, s0                                     // 00000000929C: A05F0026
	s_wait_alu 0xfffe                                          // 0000000092A0: BF88FFFE
	s_add_f32 s94, s5, s94                                     // 0000000092A4: A05E5E05
	s_add_f32 s96, s10, s96                                    // 0000000092A8: A060600A
	v_readlane_b32 s2, v29, 6                                  // 0000000092AC: D7600002 00010D1D
	v_readlane_b32 s7, v29, 11                                 // 0000000092B4: D7600007 0001171D
	v_readlane_b32 s12, v29, 16                                // 0000000092BC: D760000C 0001211D
	s_add_f32 s95, s1, s95                                     // 0000000092C4: A05F5F01
	s_wait_alu 0xfffe                                          // 0000000092C8: BF88FFFE
	s_add_f32 s94, s6, s94                                     // 0000000092CC: A05E5E06
	s_add_f32 s96, s11, s96                                    // 0000000092D0: A060600B
	v_readlane_b32 s3, v29, 7                                  // 0000000092D4: D7600003 00010F1D
	v_readlane_b32 s8, v29, 12                                 // 0000000092DC: D7600008 0001191D
	v_readlane_b32 s13, v29, 17                                // 0000000092E4: D760000D 0001231D
	s_add_f32 s95, s2, s95                                     // 0000000092EC: A05F5F02
	s_wait_alu 0xfffe                                          // 0000000092F0: BF88FFFE
	s_add_f32 s94, s7, s94                                     // 0000000092F4: A05E5E07
	s_add_f32 s96, s96, s12                                    // 0000000092F8: A0600C60
	v_readlane_b32 s14, v29, 18                                // 0000000092FC: D760000E 0001251D
	s_add_f32 s95, s3, s95                                     // 000000009304: A05F5F03
	s_wait_alu 0xfffe                                          // 000000009308: BF88FFFE
	s_add_f32 s94, s94, s8                                     // 00000000930C: A05E085E
	s_add_f32 s96, s13, s96                                    // 000000009310: A060600D
	v_readlane_b32 s15, v29, 19                                // 000000009314: D760000F 0001271D
	s_add_f32 s95, s4, s95                                     // 00000000931C: A05F5F04
	s_wait_alu 0xfffe                                          // 000000009320: BF88FFFE
	s_add_f32 s94, s9, s94                                     // 000000009324: A05E5E09
	s_add_f32 s96, s14, s96                                    // 000000009328: A060600E
	s_add_f32 s95, s5, s95                                     // 00000000932C: A05F5F05
	s_wait_alu 0xfffe                                          // 000000009330: BF88FFFE
	s_add_f32 s94, s10, s94                                    // 000000009334: A05E5E0A
	s_add_f32 s41, s15, s96                                    // 000000009338: A029600F
	s_add_f32 s38, s6, s95                                     // 00000000933C: A0265F06
	s_wait_alu 0xfffe                                          // 000000009340: BF88FFFE
	s_add_f32 s101, s11, s94                                   // 000000009344: A0655E0B
	s_or_saveexec_b32 s105, -1                                 // 000000009348: BEE922C1
	scratch_load_b32 v29, off, off offset:76                   // 00000000934C: ED05007C 0000001D 00004C00
	s_wait_alu 0xfffe                                          // 000000009358: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000935C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000009360: BFC00000
	v_readlane_b32 s0, v29, 20                                 // 000000009364: D7600000 0001291D
	v_readlane_b32 s4, v29, 24                                 // 00000000936C: D7600004 0001311D
	v_readlane_b32 s8, v29, 28                                 // 000000009374: D7600008 0001391D
	v_readlane_b32 s9, v29, 29                                 // 00000000937C: D7600009 00013B1D
	v_readlane_b32 s1, v29, 21                                 // 000000009384: D7600001 00012B1D
	v_readlane_b32 s2, v29, 22                                 // 00000000938C: D7600002 00012D1D
	v_readlane_b32 s3, v29, 23                                 // 000000009394: D7600003 00012F1D
	v_readlane_b32 s5, v29, 25                                 // 00000000939C: D7600005 0001331D
	v_readlane_b32 s6, v29, 26                                 // 0000000093A4: D7600006 0001351D
	v_readlane_b32 s7, v29, 27                                 // 0000000093AC: D7600007 0001371D
	v_readlane_b32 s10, v29, 30                                // 0000000093B4: D760000A 00013D1D
	v_readlane_b32 s11, v29, 31                                // 0000000093BC: D760000B 00013F1D
	s_or_saveexec_b32 s105, -1                                 // 0000000093C4: BEE922C1
	v_mov_b32_e32 v28, v29                                     // 0000000093C8: 7E38031D
	s_wait_alu 0xfffe                                          // 0000000093CC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000093D0: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 0000000093D4: BEE922C1
	scratch_load_b32 v29, off, off offset:32                   // 0000000093D8: ED05007C 0000001D 00002000
	s_wait_alu 0xfffe                                          // 0000000093E4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000093E8: BEFE0069
	s_add_f32 s94, s102, s4                                    // 0000000093EC: A05E0466
	s_add_f32 s95, s39, s0                                     // 0000000093F0: A05F0027
	s_add_f32 s96, s103, s9                                    // 0000000093F4: A0600967
	s_wait_loadcnt 0x0                                         // 0000000093F8: BFC00000
	v_readlane_b32 s12, v29, 0                                 // 0000000093FC: D760000C 0001011D
	s_wait_alu 0xfffe                                          // 000000009404: BF88FFFE
	s_add_f32 s94, s5, s94                                     // 000000009408: A05E5E05
	s_add_f32 s95, s1, s95                                     // 00000000940C: A05F5F01
	s_add_f32 s96, s10, s96                                    // 000000009410: A060600A
	v_readlane_b32 s13, v29, 1                                 // 000000009414: D760000D 0001031D
	s_wait_alu 0xfffe                                          // 00000000941C: BF88FFFE
	s_add_f32 s94, s6, s94                                     // 000000009420: A05E5E06
	s_add_f32 s95, s2, s95                                     // 000000009424: A05F5F02
	s_add_f32 s96, s11, s96                                    // 000000009428: A060600B
	v_readlane_b32 s14, v29, 2                                 // 00000000942C: D760000E 0001051D
	s_wait_alu 0xfffe                                          // 000000009434: BF88FFFE
	s_add_f32 s94, s7, s94                                     // 000000009438: A05E5E07
	s_add_f32 s95, s3, s95                                     // 00000000943C: A05F5F03
	s_add_f32 s96, s96, s12                                    // 000000009440: A0600C60
	v_readlane_b32 s15, v29, 3                                 // 000000009444: D760000F 0001071D
	s_wait_alu 0xfffe                                          // 00000000944C: BF88FFFE
	s_add_f32 s94, s94, s8                                     // 000000009450: A05E085E
	s_add_f32 s95, s4, s95                                     // 000000009454: A05F5F04
	s_add_f32 s96, s13, s96                                    // 000000009458: A060600D
	s_wait_alu 0xfffe                                          // 00000000945C: BF88FFFE
	s_add_f32 s94, s9, s94                                     // 000000009460: A05E5E09
	s_add_f32 s95, s5, s95                                     // 000000009464: A05F5F05
	s_add_f32 s96, s14, s96                                    // 000000009468: A060600E
	s_wait_alu 0xfffe                                          // 00000000946C: BF88FFFE
	s_add_f32 s94, s10, s94                                    // 000000009470: A05E5E0A
	s_add_f32 s39, s6, s95                                     // 000000009474: A0275F06
	s_add_f32 s103, s15, s96                                   // 000000009478: A067600F
	s_wait_alu 0xfffe                                          // 00000000947C: BF88FFFE
	s_add_f32 s102, s11, s94                                   // 000000009480: A0665E0B
	v_readlane_b32 s0, v26, 20                                 // 000000009484: D7600000 0001291A
	v_readlane_b32 s4, v26, 24                                 // 00000000948C: D7600004 0001311A
	v_readlane_b32 s8, v26, 28                                 // 000000009494: D7600008 0001391A
	v_readlane_b32 s9, v26, 29                                 // 00000000949C: D7600009 00013B1A
	v_readlane_b32 s1, v26, 21                                 // 0000000094A4: D7600001 00012B1A
	v_readlane_b32 s2, v26, 22                                 // 0000000094AC: D7600002 00012D1A
	v_readlane_b32 s3, v26, 23                                 // 0000000094B4: D7600003 00012F1A
	v_readlane_b32 s5, v26, 25                                 // 0000000094BC: D7600005 0001331A
	v_readlane_b32 s6, v26, 26                                 // 0000000094C4: D7600006 0001351A
	v_readlane_b32 s7, v26, 27                                 // 0000000094CC: D7600007 0001371A
	v_readlane_b32 s10, v26, 30                                // 0000000094D4: D760000A 00013D1A
	v_readlane_b32 s11, v26, 31                                // 0000000094DC: D760000B 00013F1A
	s_or_saveexec_b32 s105, -1                                 // 0000000094E4: BEE922C1
	s_wait_alu 0xfffe                                          // 0000000094E8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000094EC: BEFE0069
	s_add_f32 s35, s35, s4                                     // 0000000094F0: A0230423
	s_add_f32 s94, s36, s0                                     // 0000000094F4: A05E0024
	s_add_f32 s95, s37, s9                                     // 0000000094F8: A05F0925
	v_readlane_b32 s12, v28, 0                                 // 0000000094FC: D760000C 0001011C
	s_wait_alu 0xfffe                                          // 000000009504: BF88FFFE
	s_add_f32 s35, s5, s35                                     // 000000009508: A0232305
	s_add_f32 s94, s1, s94                                     // 00000000950C: A05E5E01
	s_add_f32 s95, s10, s95                                    // 000000009510: A05F5F0A
	v_readlane_b32 s13, v28, 1                                 // 000000009514: D760000D 0001031C
	s_wait_alu 0xfffe                                          // 00000000951C: BF88FFFE
	s_add_f32 s35, s6, s35                                     // 000000009520: A0232306
	s_add_f32 s94, s2, s94                                     // 000000009524: A05E5E02
	s_add_f32 s95, s11, s95                                    // 000000009528: A05F5F0B
	v_readlane_b32 s14, v28, 2                                 // 00000000952C: D760000E 0001051C
	s_wait_alu 0xfffe                                          // 000000009534: BF88FFFE
	s_add_f32 s35, s7, s35                                     // 000000009538: A0232307
	s_add_f32 s94, s3, s94                                     // 00000000953C: A05E5E03
	s_add_f32 s95, s95, s12                                    // 000000009540: A05F0C5F
	v_readlane_b32 s15, v28, 3                                 // 000000009544: D760000F 0001071C
	s_wait_alu 0xfffe                                          // 00000000954C: BF88FFFE
	s_add_f32 s35, s35, s8                                     // 000000009550: A0230823
	s_add_f32 s94, s4, s94                                     // 000000009554: A05E5E04
	s_add_f32 s95, s13, s95                                    // 000000009558: A05F5F0D
	s_wait_alu 0xfffe                                          // 00000000955C: BF88FFFE
	s_add_f32 s35, s9, s35                                     // 000000009560: A0232309
	s_add_f32 s94, s5, s94                                     // 000000009564: A05E5E05
	s_add_f32 s95, s14, s95                                    // 000000009568: A05F5F0E
	s_wait_alu 0xfffe                                          // 00000000956C: BF88FFFE
	s_add_f32 s35, s10, s35                                    // 000000009570: A023230A
	s_add_f32 s36, s6, s94                                     // 000000009574: A0245E06
	s_add_f32 s37, s15, s95                                    // 000000009578: A0255F0F
	s_wait_alu 0xfffe                                          // 00000000957C: BF88FFFE
	s_add_f32 s35, s11, s35                                    // 000000009580: A023230B
	v_readlane_b32 s0, v29, 4                                  // 000000009584: D7600000 0001091D
	v_readlane_b32 s4, v29, 8                                  // 00000000958C: D7600004 0001111D
	v_readlane_b32 s8, v29, 12                                 // 000000009594: D7600008 0001191D
	v_readlane_b32 s9, v29, 13                                 // 00000000959C: D7600009 00011B1D
	v_readlane_b32 s1, v29, 5                                  // 0000000095A4: D7600001 00010B1D
	v_readlane_b32 s2, v29, 6                                  // 0000000095AC: D7600002 00010D1D
	v_readlane_b32 s3, v29, 7                                  // 0000000095B4: D7600003 00010F1D
	v_readlane_b32 s5, v29, 9                                  // 0000000095BC: D7600005 0001131D
	v_readlane_b32 s6, v29, 10                                 // 0000000095C4: D7600006 0001151D
	v_readlane_b32 s7, v29, 11                                 // 0000000095CC: D7600007 0001171D
	v_readlane_b32 s10, v29, 14                                // 0000000095D4: D760000A 00011D1D
	v_readlane_b32 s11, v29, 15                                // 0000000095DC: D760000B 00011F1D
	v_readlane_b32 s12, v29, 16                                // 0000000095E4: D760000C 0001211D
	v_readlane_b32 s13, v29, 17                                // 0000000095EC: D760000D 0001231D
	v_readlane_b32 s14, v29, 18                                // 0000000095F4: D760000E 0001251D
	s_or_saveexec_b32 s105, -1                                 // 0000000095FC: BEE922C1
	s_wait_alu 0xfffe                                          // 000000009600: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009604: BEFE0069
	s_add_f32 s92, s92, s4                                     // 000000009608: A05C045C
	s_add_f32 s93, s93, s0                                     // 00000000960C: A05D005D
	s_add_f32 s94, s40, s9                                     // 000000009610: A05E0928
	v_readlane_b32 s15, v29, 19                                // 000000009614: D760000F 0001271D
	s_wait_alu 0xfffe                                          // 00000000961C: BF88FFFE
	s_add_f32 s92, s5, s92                                     // 000000009620: A05C5C05
	s_add_f32 s93, s1, s93                                     // 000000009624: A05D5D01
	s_add_f32 s94, s10, s94                                    // 000000009628: A05E5E0A
	s_wait_alu 0xfffe                                          // 00000000962C: BF88FFFE
	s_add_f32 s92, s6, s92                                     // 000000009630: A05C5C06
	s_add_f32 s93, s2, s93                                     // 000000009634: A05D5D02
	s_add_f32 s94, s11, s94                                    // 000000009638: A05E5E0B
	s_wait_alu 0xfffe                                          // 00000000963C: BF88FFFE
	s_add_f32 s92, s7, s92                                     // 000000009640: A05C5C07
	s_add_f32 s93, s3, s93                                     // 000000009644: A05D5D03
	s_add_f32 s94, s94, s12                                    // 000000009648: A05E0C5E
	s_wait_alu 0xfffe                                          // 00000000964C: BF88FFFE
	s_add_f32 s92, s92, s8                                     // 000000009650: A05C085C
	s_add_f32 s93, s4, s93                                     // 000000009654: A05D5D04
	s_add_f32 s94, s13, s94                                    // 000000009658: A05E5E0D
	s_wait_alu 0xfffe                                          // 00000000965C: BF88FFFE
	s_add_f32 s92, s9, s92                                     // 000000009660: A05C5C09
	s_add_f32 s93, s5, s93                                     // 000000009664: A05D5D05
	s_add_f32 s94, s14, s94                                    // 000000009668: A05E5E0E
	s_wait_alu 0xfffe                                          // 00000000966C: BF88FFFE
	s_add_f32 s92, s10, s92                                    // 000000009670: A05C5C0A
	s_add_f32 s93, s6, s93                                     // 000000009674: A05D5D06
	s_add_f32 s40, s15, s94                                    // 000000009678: A0285E0F
	s_wait_alu 0xfffe                                          // 00000000967C: BF88FFFE
	s_add_f32 s92, s11, s92                                    // 000000009680: A05C5C0B
	s_or_saveexec_b32 s105, -1                                 // 000000009684: BEE922C1
	scratch_load_b32 v28, off, off offset:72                   // 000000009688: ED05007C 0000001C 00004800
	s_wait_alu 0xfffe                                          // 000000009694: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009698: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000969C: BFC00000
	v_readlane_b32 s0, v28, 12                                 // 0000000096A0: D7600000 0001191C
	v_readlane_b32 s4, v28, 16                                 // 0000000096A8: D7600004 0001211C
	v_readlane_b32 s8, v28, 20                                 // 0000000096B0: D7600008 0001291C
	v_readlane_b32 s9, v28, 21                                 // 0000000096B8: D7600009 00012B1C
	v_readlane_b32 s1, v28, 13                                 // 0000000096C0: D7600001 00011B1C
	v_readlane_b32 s2, v28, 14                                 // 0000000096C8: D7600002 00011D1C
	v_readlane_b32 s3, v28, 15                                 // 0000000096D0: D7600003 00011F1C
	v_readlane_b32 s5, v28, 17                                 // 0000000096D8: D7600005 0001231C
	v_readlane_b32 s6, v28, 18                                 // 0000000096E0: D7600006 0001251C
	v_readlane_b32 s7, v28, 19                                 // 0000000096E8: D7600007 0001271C
	v_readlane_b32 s10, v28, 22                                // 0000000096F0: D760000A 00012D1C
	v_readlane_b32 s11, v28, 23                                // 0000000096F8: D760000B 00012F1C
	v_readlane_b32 s12, v28, 24                                // 000000009700: D760000C 0001311C
	v_readlane_b32 s13, v28, 25                                // 000000009708: D760000D 0001331C
	v_readlane_b32 s14, v28, 26                                // 000000009710: D760000E 0001351C
	v_readlane_b32 s15, v28, 27                                // 000000009718: D760000F 0001371C
	s_or_saveexec_b32 s105, -1                                 // 000000009720: BEE922C1
	v_mov_b32_e32 v27, v28                                     // 000000009724: 7E36031C
	s_wait_alu 0xfffe                                          // 000000009728: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000972C: BEFE0069
	s_add_f32 s33, s33, s4                                     // 000000009730: A0210421
	s_add_f32 s34, s34, s0                                     // 000000009734: A0220022
	s_add_f32 s94, s104, s9                                    // 000000009738: A05E0968
	s_wait_alu 0xfffe                                          // 00000000973C: BF88FFFE
	s_add_f32 s33, s5, s33                                     // 000000009740: A0212105
	s_add_f32 s34, s1, s34                                     // 000000009744: A0222201
	s_add_f32 s94, s10, s94                                    // 000000009748: A05E5E0A
	s_wait_alu 0xfffe                                          // 00000000974C: BF88FFFE
	s_add_f32 s33, s6, s33                                     // 000000009750: A0212106
	s_add_f32 s34, s2, s34                                     // 000000009754: A0222202
	s_add_f32 s94, s11, s94                                    // 000000009758: A05E5E0B
	s_wait_alu 0xfffe                                          // 00000000975C: BF88FFFE
	s_add_f32 s33, s7, s33                                     // 000000009760: A0212107
	s_add_f32 s34, s3, s34                                     // 000000009764: A0222203
	s_add_f32 s94, s94, s12                                    // 000000009768: A05E0C5E
	s_wait_alu 0xfffe                                          // 00000000976C: BF88FFFE
	s_add_f32 s33, s33, s8                                     // 000000009770: A0210821
	s_add_f32 s34, s34, s4                                     // 000000009774: A0220422
	s_add_f32 s94, s13, s94                                    // 000000009778: A05E5E0D
	s_wait_alu 0xfffe                                          // 00000000977C: BF88FFFE
	s_add_f32 s33, s9, s33                                     // 000000009780: A0212109
	s_add_f32 s34, s5, s34                                     // 000000009784: A0222205
	s_add_f32 s94, s14, s94                                    // 000000009788: A05E5E0E
	s_wait_alu 0xfffe                                          // 00000000978C: BF88FFFE
	s_add_f32 s95, s10, s33                                    // 000000009790: A05F210A
	s_add_f32 s33, s6, s34                                     // 000000009794: A0212206
	s_add_f32 s104, s15, s94                                   // 000000009798: A0685E0F
	s_wait_alu 0xfffe                                          // 00000000979C: BF88FFFE
	s_add_f32 s34, s11, s95                                    // 0000000097A0: A0225F0B
	s_or_saveexec_b32 s105, -1                                 // 0000000097A4: BEE922C1
	s_wait_alu 0xfffe                                          // 0000000097A8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000097AC: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 0000000097B0: BEE922C1
	scratch_store_b32 off, v29, off offset:32                  // 0000000097B4: ED06807C 0E800000 00002000
	s_wait_alu 0xfffe                                          // 0000000097C0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000097C4: BEFE0069
	v_readlane_b32 s0, v29, 28                                 // 0000000097C8: D7600000 0001391D
	v_readlane_b32 s1, v29, 29                                 // 0000000097D0: D7600001 00013B1D
	v_readlane_b32 s2, v29, 30                                 // 0000000097D8: D7600002 00013D1D
	v_readlane_b32 s3, v29, 31                                 // 0000000097E0: D7600003 00013F1D
	s_or_saveexec_b32 s105, -1                                 // 0000000097E8: BEE922C1
	scratch_load_b32 v29, off, off offset:36                   // 0000000097EC: ED05007C 0000001D 00002400
	s_wait_alu 0xfffe                                          // 0000000097F8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000097FC: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000009800: BFC00000
	v_readlane_b32 s4, v29, 0                                  // 000000009804: D7600004 0001011D
	v_readlane_b32 s9, v29, 5                                  // 00000000980C: D7600009 00010B1D
	v_readlane_b32 s5, v29, 1                                  // 000000009814: D7600005 0001031D
	v_readlane_b32 s10, v29, 6                                 // 00000000981C: D760000A 00010D1D
	v_readlane_b32 s6, v29, 2                                  // 000000009824: D7600006 0001051D
	s_add_f32 s94, s101, s4                                    // 00000000982C: A05E0465
	s_add_f32 s96, s41, s9                                     // 000000009830: A0600929
	v_readlane_b32 s11, v29, 7                                 // 000000009834: D760000B 00010F1D
	s_add_f32 s95, s38, s0                                     // 00000000983C: A05F0026
	s_wait_alu 0xfffe                                          // 000000009840: BF88FFFE
	s_add_f32 s94, s5, s94                                     // 000000009844: A05E5E05
	s_add_f32 s96, s10, s96                                    // 000000009848: A060600A
	v_readlane_b32 s7, v29, 3                                  // 00000000984C: D7600007 0001071D
	v_readlane_b32 s12, v29, 8                                 // 000000009854: D760000C 0001111D
	s_add_f32 s95, s1, s95                                     // 00000000985C: A05F5F01
	s_wait_alu 0xfffe                                          // 000000009860: BF88FFFE
	s_add_f32 s94, s6, s94                                     // 000000009864: A05E5E06
	s_add_f32 s96, s11, s96                                    // 000000009868: A060600B
	v_readlane_b32 s8, v29, 4                                  // 00000000986C: D7600008 0001091D
	v_readlane_b32 s13, v29, 9                                 // 000000009874: D760000D 0001131D
	s_add_f32 s95, s2, s95                                     // 00000000987C: A05F5F02
	s_wait_alu 0xfffe                                          // 000000009880: BF88FFFE
	s_add_f32 s94, s7, s94                                     // 000000009884: A05E5E07
	s_add_f32 s96, s96, s12                                    // 000000009888: A0600C60
	v_readlane_b32 s14, v29, 10                                // 00000000988C: D760000E 0001151D
	s_add_f32 s95, s3, s95                                     // 000000009894: A05F5F03
	s_wait_alu 0xfffe                                          // 000000009898: BF88FFFE
	s_add_f32 s94, s94, s8                                     // 00000000989C: A05E085E
	s_add_f32 s96, s13, s96                                    // 0000000098A0: A060600D
	v_readlane_b32 s15, v29, 11                                // 0000000098A4: D760000F 0001171D
	s_add_f32 s95, s4, s95                                     // 0000000098AC: A05F5F04
	s_wait_alu 0xfffe                                          // 0000000098B0: BF88FFFE
	s_add_f32 s94, s9, s94                                     // 0000000098B4: A05E5E09
	s_add_f32 s96, s14, s96                                    // 0000000098B8: A060600E
	s_add_f32 s95, s5, s95                                     // 0000000098BC: A05F5F05
	s_wait_alu 0xfffe                                          // 0000000098C0: BF88FFFE
	s_add_f32 s94, s10, s94                                    // 0000000098C4: A05E5E0A
	s_add_f32 s41, s15, s96                                    // 0000000098C8: A029600F
	s_add_f32 s38, s6, s95                                     // 0000000098CC: A0265F06
	s_wait_alu 0xfffe                                          // 0000000098D0: BF88FFFE
	s_add_f32 s101, s11, s94                                   // 0000000098D4: A0655E0B
	s_or_saveexec_b32 s105, -1                                 // 0000000098D8: BEE922C1
	scratch_load_b32 v28, off, off offset:80                   // 0000000098DC: ED05007C 0000001C 00005000
	s_wait_alu 0xfffe                                          // 0000000098E8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 0000000098EC: BEFE0069
	s_wait_loadcnt 0x0                                         // 0000000098F0: BFC00000
	v_readlane_b32 s0, v28, 28                                 // 0000000098F4: D7600000 0001391C
	v_readlane_b32 s1, v28, 29                                 // 0000000098FC: D7600001 00013B1C
	v_readlane_b32 s2, v28, 30                                 // 000000009904: D7600002 00013D1C
	v_readlane_b32 s3, v28, 31                                 // 00000000990C: D7600003 00013F1C
	s_or_saveexec_b32 s105, -1                                 // 000000009914: BEE922C1
	scratch_load_b32 v29, off, off offset:84                   // 000000009918: ED05007C 0000001D 00005400
	s_wait_alu 0xfffe                                          // 000000009924: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009928: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000992C: BFC00000
	v_readlane_b32 s4, v29, 0                                  // 000000009930: D7600004 0001011D
	v_readlane_b32 s8, v29, 4                                  // 000000009938: D7600008 0001091D
	v_readlane_b32 s9, v29, 5                                  // 000000009940: D7600009 00010B1D
	v_readlane_b32 s5, v29, 1                                  // 000000009948: D7600005 0001031D
	v_readlane_b32 s6, v29, 2                                  // 000000009950: D7600006 0001051D
	v_readlane_b32 s7, v29, 3                                  // 000000009958: D7600007 0001071D
	v_readlane_b32 s10, v29, 6                                 // 000000009960: D760000A 00010D1D
	v_readlane_b32 s11, v29, 7                                 // 000000009968: D760000B 00010F1D
	v_readlane_b32 s12, v29, 8                                 // 000000009970: D760000C 0001111D
	v_readlane_b32 s13, v29, 9                                 // 000000009978: D760000D 0001131D
	v_readlane_b32 s14, v29, 10                                // 000000009980: D760000E 0001151D
	v_readlane_b32 s15, v29, 11                                // 000000009988: D760000F 0001171D
	s_or_saveexec_b32 s105, -1                                 // 000000009990: BEE922C1
	s_wait_alu 0xfffe                                          // 000000009994: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009998: BEFE0069
	s_add_f32 s94, s102, s4                                    // 00000000999C: A05E0466
	s_add_f32 s95, s39, s0                                     // 0000000099A0: A05F0027
	s_add_f32 s96, s103, s9                                    // 0000000099A4: A0600967
	s_wait_alu 0xfffe                                          // 0000000099A8: BF88FFFE
	s_add_f32 s94, s5, s94                                     // 0000000099AC: A05E5E05
	s_add_f32 s95, s1, s95                                     // 0000000099B0: A05F5F01
	s_add_f32 s96, s10, s96                                    // 0000000099B4: A060600A
	s_wait_alu 0xfffe                                          // 0000000099B8: BF88FFFE
	s_add_f32 s94, s6, s94                                     // 0000000099BC: A05E5E06
	s_add_f32 s95, s2, s95                                     // 0000000099C0: A05F5F02
	s_add_f32 s96, s11, s96                                    // 0000000099C4: A060600B
	s_wait_alu 0xfffe                                          // 0000000099C8: BF88FFFE
	s_add_f32 s94, s7, s94                                     // 0000000099CC: A05E5E07
	s_add_f32 s95, s3, s95                                     // 0000000099D0: A05F5F03
	s_add_f32 s96, s96, s12                                    // 0000000099D4: A0600C60
	s_wait_alu 0xfffe                                          // 0000000099D8: BF88FFFE
	s_add_f32 s94, s94, s8                                     // 0000000099DC: A05E085E
	s_add_f32 s95, s4, s95                                     // 0000000099E0: A05F5F04
	s_add_f32 s96, s13, s96                                    // 0000000099E4: A060600D
	s_wait_alu 0xfffe                                          // 0000000099E8: BF88FFFE
	s_add_f32 s94, s9, s94                                     // 0000000099EC: A05E5E09
	s_add_f32 s95, s5, s95                                     // 0000000099F0: A05F5F05
	s_add_f32 s96, s14, s96                                    // 0000000099F4: A060600E
	s_wait_alu 0xfffe                                          // 0000000099F8: BF88FFFE
	s_add_f32 s94, s10, s94                                    // 0000000099FC: A05E5E0A
	s_add_f32 s39, s6, s95                                     // 000000009A00: A0275F06
	s_add_f32 s103, s15, s96                                   // 000000009A04: A067600F
	s_wait_alu 0xfffe                                          // 000000009A08: BF88FFFE
	s_add_f32 s102, s11, s94                                   // 000000009A0C: A0665E0B
	s_or_saveexec_b32 s105, -1                                 // 000000009A10: BEE922C1
	s_wait_alu 0xfffe                                          // 000000009A14: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009A18: BEFE0069
	v_readlane_b32 s0, v27, 28                                 // 000000009A1C: D7600000 0001391B
	v_readlane_b32 s4, v28, 0                                  // 000000009A24: D7600004 0001011C
	v_readlane_b32 s5, v28, 1                                  // 000000009A2C: D7600005 0001031C
	v_readlane_b32 s9, v28, 5                                  // 000000009A34: D7600009 00010B1C
	v_readlane_b32 s1, v27, 29                                 // 000000009A3C: D7600001 00013B1B
	v_readlane_b32 s6, v28, 2                                  // 000000009A44: D7600006 0001051C
	s_add_f32 s35, s35, s4                                     // 000000009A4C: A0230423
	v_readlane_b32 s10, v28, 6                                 // 000000009A50: D760000A 00010D1C
	s_add_f32 s94, s36, s0                                     // 000000009A58: A05E0024
	s_add_f32 s95, s37, s9                                     // 000000009A5C: A05F0925
	s_wait_alu 0xfffe                                          // 000000009A60: BF88FFFE
	s_add_f32 s35, s5, s35                                     // 000000009A64: A0232305
	v_readlane_b32 s2, v27, 30                                 // 000000009A68: D7600002 00013D1B
	v_readlane_b32 s7, v28, 3                                  // 000000009A70: D7600007 0001071C
	v_readlane_b32 s11, v28, 7                                 // 000000009A78: D760000B 00010F1C
	s_add_f32 s94, s1, s94                                     // 000000009A80: A05E5E01
	s_wait_alu 0xfffe                                          // 000000009A84: BF88FFFE
	s_add_f32 s35, s6, s35                                     // 000000009A88: A0232306
	s_add_f32 s95, s10, s95                                    // 000000009A8C: A05F5F0A
	v_readlane_b32 s3, v27, 31                                 // 000000009A90: D7600003 00013F1B
	v_readlane_b32 s8, v28, 4                                  // 000000009A98: D7600008 0001091C
	v_readlane_b32 s12, v28, 8                                 // 000000009AA0: D760000C 0001111C
	s_add_f32 s94, s2, s94                                     // 000000009AA8: A05E5E02
	s_wait_alu 0xfffe                                          // 000000009AAC: BF88FFFE
	s_add_f32 s35, s7, s35                                     // 000000009AB0: A0232307
	s_add_f32 s95, s11, s95                                    // 000000009AB4: A05F5F0B
	v_readlane_b32 s13, v28, 9                                 // 000000009AB8: D760000D 0001131C
	s_add_f32 s94, s3, s94                                     // 000000009AC0: A05E5E03
	s_wait_alu 0xfffe                                          // 000000009AC4: BF88FFFE
	s_add_f32 s35, s35, s8                                     // 000000009AC8: A0230823
	s_add_f32 s95, s95, s12                                    // 000000009ACC: A05F0C5F
	v_readlane_b32 s14, v28, 10                                // 000000009AD0: D760000E 0001151C
	s_add_f32 s94, s4, s94                                     // 000000009AD8: A05E5E04
	s_wait_alu 0xfffe                                          // 000000009ADC: BF88FFFE
	s_add_f32 s35, s9, s35                                     // 000000009AE0: A0232309
	s_add_f32 s95, s13, s95                                    // 000000009AE4: A05F5F0D
	v_readlane_b32 s15, v28, 11                                // 000000009AE8: D760000F 0001171C
	s_add_f32 s94, s5, s94                                     // 000000009AF0: A05E5E05
	s_wait_alu 0xfffe                                          // 000000009AF4: BF88FFFE
	s_add_f32 s35, s10, s35                                    // 000000009AF8: A023230A
	s_add_f32 s95, s14, s95                                    // 000000009AFC: A05F5F0E
	s_add_f32 s37, s6, s94                                     // 000000009B00: A0255E06
	s_wait_alu 0xfffe                                          // 000000009B04: BF88FFFE
	s_add_f32 s35, s11, s35                                    // 000000009B08: A023230B
	s_add_f32 s42, s15, s95                                    // 000000009B0C: A02A5F0F
	v_readlane_b32 s0, v29, 12                                 // 000000009B10: D7600000 0001191D
	v_readlane_b32 s4, v29, 16                                 // 000000009B18: D7600004 0001211D
	v_readlane_b32 s8, v29, 20                                 // 000000009B20: D7600008 0001291D
	v_readlane_b32 s9, v29, 21                                 // 000000009B28: D7600009 00012B1D
	v_readlane_b32 s1, v29, 13                                 // 000000009B30: D7600001 00011B1D
	v_readlane_b32 s2, v29, 14                                 // 000000009B38: D7600002 00011D1D
	v_readlane_b32 s3, v29, 15                                 // 000000009B40: D7600003 00011F1D
	v_readlane_b32 s5, v29, 17                                 // 000000009B48: D7600005 0001231D
	v_readlane_b32 s6, v29, 18                                 // 000000009B50: D7600006 0001251D
	v_readlane_b32 s7, v29, 19                                 // 000000009B58: D7600007 0001271D
	v_readlane_b32 s10, v29, 22                                // 000000009B60: D760000A 00012D1D
	v_readlane_b32 s11, v29, 23                                // 000000009B68: D760000B 00012F1D
	v_readlane_b32 s12, v29, 24                                // 000000009B70: D760000C 0001311D
	v_readlane_b32 s13, v29, 25                                // 000000009B78: D760000D 0001331D
	v_readlane_b32 s14, v29, 26                                // 000000009B80: D760000E 0001351D
	s_or_saveexec_b32 s105, -1                                 // 000000009B88: BEE922C1
	s_wait_alu 0xfffe                                          // 000000009B8C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009B90: BEFE0069
	s_add_f32 s92, s92, s4                                     // 000000009B94: A05C045C
	s_add_f32 s93, s93, s0                                     // 000000009B98: A05D005D
	s_add_f32 s94, s40, s9                                     // 000000009B9C: A05E0928
	v_readlane_b32 s15, v29, 27                                // 000000009BA0: D760000F 0001371D
	s_wait_alu 0xfffe                                          // 000000009BA8: BF88FFFE
	s_add_f32 s92, s5, s92                                     // 000000009BAC: A05C5C05
	s_add_f32 s93, s1, s93                                     // 000000009BB0: A05D5D01
	s_add_f32 s94, s10, s94                                    // 000000009BB4: A05E5E0A
	s_wait_alu 0xfffe                                          // 000000009BB8: BF88FFFE
	s_add_f32 s92, s6, s92                                     // 000000009BBC: A05C5C06
	s_add_f32 s93, s2, s93                                     // 000000009BC0: A05D5D02
	s_add_f32 s94, s11, s94                                    // 000000009BC4: A05E5E0B
	s_wait_alu 0xfffe                                          // 000000009BC8: BF88FFFE
	s_add_f32 s92, s7, s92                                     // 000000009BCC: A05C5C07
	s_add_f32 s93, s3, s93                                     // 000000009BD0: A05D5D03
	s_add_f32 s94, s94, s12                                    // 000000009BD4: A05E0C5E
	s_wait_alu 0xfffe                                          // 000000009BD8: BF88FFFE
	s_add_f32 s92, s92, s8                                     // 000000009BDC: A05C085C
	s_add_f32 s93, s4, s93                                     // 000000009BE0: A05D5D04
	s_add_f32 s94, s13, s94                                    // 000000009BE4: A05E5E0D
	s_wait_alu 0xfffe                                          // 000000009BE8: BF88FFFE
	s_add_f32 s92, s9, s92                                     // 000000009BEC: A05C5C09
	s_add_f32 s93, s5, s93                                     // 000000009BF0: A05D5D05
	s_add_f32 s94, s14, s94                                    // 000000009BF4: A05E5E0E
	s_wait_alu 0xfffe                                          // 000000009BF8: BF88FFFE
	s_add_f32 s92, s10, s92                                    // 000000009BFC: A05C5C0A
	s_add_f32 s93, s6, s93                                     // 000000009C00: A05D5D06
	s_add_f32 s0, s15, s94                                     // 000000009C04: A0005E0F
	s_wait_alu 0xfffe                                          // 000000009C08: BF88FFFE
	s_add_f32 s92, s11, s92                                    // 000000009C0C: A05C5C0B
	s_or_saveexec_b32 s105, -1                                 // 000000009C10: BEE922C1
	s_wait_alu 0xfffe                                          // 000000009C14: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009C18: BEFE0069
	v_writelane_b32 v24, s0, 24                                // 000000009C1C: D7610018 00013000
	v_readlane_b32 s0, v22, 20                                 // 000000009C24: D7600000 00012916
	v_readlane_b32 s4, v22, 24                                 // 000000009C2C: D7600004 00013116
	v_readlane_b32 s8, v22, 28                                 // 000000009C34: D7600008 00013916
	v_readlane_b32 s9, v22, 29                                 // 000000009C3C: D7600009 00013B16
	v_readlane_b32 s1, v22, 21                                 // 000000009C44: D7600001 00012B16
	v_readlane_b32 s2, v22, 22                                 // 000000009C4C: D7600002 00012D16
	v_readlane_b32 s3, v22, 23                                 // 000000009C54: D7600003 00012F16
	v_readlane_b32 s5, v22, 25                                 // 000000009C5C: D7600005 00013316
	v_readlane_b32 s6, v22, 26                                 // 000000009C64: D7600006 00013516
	v_readlane_b32 s7, v22, 27                                 // 000000009C6C: D7600007 00013716
	v_readlane_b32 s10, v22, 30                                // 000000009C74: D760000A 00013D16
	v_readlane_b32 s11, v22, 31                                // 000000009C7C: D760000B 00013F16
	s_or_saveexec_b32 s105, -1                                 // 000000009C84: BEE922C1
	scratch_load_b32 v29, off, off offset:92                   // 000000009C88: ED05007C 0000001D 00005C00
	s_wait_alu 0xfffe                                          // 000000009C94: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009C98: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000009C9C: BFC00000
	v_readlane_b32 s12, v29, 0                                 // 000000009CA0: D760000C 0001011D
	v_readlane_b32 s13, v29, 1                                 // 000000009CA8: D760000D 0001031D
	v_readlane_b32 s14, v29, 2                                 // 000000009CB0: D760000E 0001051D
	v_readlane_b32 s15, v29, 3                                 // 000000009CB8: D760000F 0001071D
	s_or_saveexec_b32 s105, -1                                 // 000000009CC0: BEE922C1
	s_wait_alu 0xfffe                                          // 000000009CC4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009CC8: BEFE0069
	s_add_f32 s34, s34, s4                                     // 000000009CCC: A0220422
	s_add_f32 s33, s33, s0                                     // 000000009CD0: A0210021
	s_add_f32 s36, s104, s9                                    // 000000009CD4: A0240968
	s_wait_alu 0xfffe                                          // 000000009CD8: BF88FFFE
	s_add_f32 s34, s5, s34                                     // 000000009CDC: A0222205
	s_add_f32 s33, s1, s33                                     // 000000009CE0: A0212101
	s_add_f32 s36, s10, s36                                    // 000000009CE4: A024240A
	s_wait_alu 0xfffe                                          // 000000009CE8: BF88FFFE
	s_add_f32 s34, s6, s34                                     // 000000009CEC: A0222206
	s_add_f32 s33, s2, s33                                     // 000000009CF0: A0212102
	s_add_f32 s36, s11, s36                                    // 000000009CF4: A024240B
	s_wait_alu 0xfffe                                          // 000000009CF8: BF88FFFE
	s_add_f32 s34, s7, s34                                     // 000000009CFC: A0222207
	s_add_f32 s33, s3, s33                                     // 000000009D00: A0212103
	s_add_f32 s36, s36, s12                                    // 000000009D04: A0240C24
	s_wait_alu 0xfffe                                          // 000000009D08: BF88FFFE
	s_add_f32 s34, s34, s8                                     // 000000009D0C: A0220822
	s_add_f32 s33, s33, s4                                     // 000000009D10: A0210421
	s_add_f32 s36, s13, s36                                    // 000000009D14: A024240D
	s_wait_alu 0xfffe                                          // 000000009D18: BF88FFFE
	s_add_f32 s34, s9, s34                                     // 000000009D1C: A0222209
	s_add_f32 s33, s5, s33                                     // 000000009D20: A0212105
	s_add_f32 s36, s14, s36                                    // 000000009D24: A024240E
	s_wait_alu 0xfffe                                          // 000000009D28: BF88FFFE
	s_add_f32 s34, s10, s34                                    // 000000009D2C: A022220A
	s_add_f32 s104, s6, s33                                    // 000000009D30: A0682106
	s_add_f32 s96, s15, s36                                    // 000000009D34: A060240F
	s_wait_alu 0xfffe                                          // 000000009D38: BF88FFFE
	s_add_f32 s34, s11, s34                                    // 000000009D3C: A022220B
	s_or_saveexec_b32 s105, -1                                 // 000000009D40: BEE922C1
	scratch_store_b32 off, v22, off offset:100                 // 000000009D44: ED06807C 0B000000 00006400
	s_wait_alu 0xfffe                                          // 000000009D50: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009D54: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000009D58: BEE922C1
	scratch_load_b32 v26, off, off offset:88 th:TH_LOAD_LU     // 000000009D5C: ED05007C 0030001A 00005800
	s_wait_alu 0xfffe                                          // 000000009D68: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009D6C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000009D70: BFC00000
	v_readlane_b32 s0, v26, 4                                  // 000000009D74: D7600000 0001091A
	v_readlane_b32 s4, v26, 8                                  // 000000009D7C: D7600004 0001111A
	v_readlane_b32 s5, v26, 9                                  // 000000009D84: D7600005 0001131A
	v_readlane_b32 s9, v26, 13                                 // 000000009D8C: D7600009 00011B1A
	v_readlane_b32 s1, v26, 5                                  // 000000009D94: D7600001 00010B1A
	v_readlane_b32 s6, v26, 10                                 // 000000009D9C: D7600006 0001151A
	s_add_f32 s33, s101, s4                                    // 000000009DA4: A0210465
	v_readlane_b32 s10, v26, 14                                // 000000009DA8: D760000A 00011D1A
	s_add_f32 s36, s38, s0                                     // 000000009DB0: A0240026
	s_add_f32 s38, s41, s9                                     // 000000009DB4: A0260929
	s_wait_alu 0xfffe                                          // 000000009DB8: BF88FFFE
	s_add_f32 s33, s5, s33                                     // 000000009DBC: A0212105
	v_readlane_b32 s2, v26, 6                                  // 000000009DC0: D7600002 00010D1A
	v_readlane_b32 s7, v26, 11                                 // 000000009DC8: D7600007 0001171A
	v_readlane_b32 s11, v26, 15                                // 000000009DD0: D760000B 00011F1A
	s_add_f32 s36, s1, s36                                     // 000000009DD8: A0242401
	s_wait_alu 0xfffe                                          // 000000009DDC: BF88FFFE
	s_add_f32 s33, s6, s33                                     // 000000009DE0: A0212106
	s_add_f32 s38, s10, s38                                    // 000000009DE4: A026260A
	v_readlane_b32 s3, v26, 7                                  // 000000009DE8: D7600003 00010F1A
	v_readlane_b32 s8, v26, 12                                 // 000000009DF0: D7600008 0001191A
	v_readlane_b32 s12, v26, 16                                // 000000009DF8: D760000C 0001211A
	s_add_f32 s36, s2, s36                                     // 000000009E00: A0242402
	s_wait_alu 0xfffe                                          // 000000009E04: BF88FFFE
	s_add_f32 s33, s7, s33                                     // 000000009E08: A0212107
	s_add_f32 s38, s11, s38                                    // 000000009E0C: A026260B
	v_readlane_b32 s13, v26, 17                                // 000000009E10: D760000D 0001231A
	s_add_f32 s36, s3, s36                                     // 000000009E18: A0242403
	s_wait_alu 0xfffe                                          // 000000009E1C: BF88FFFE
	s_add_f32 s33, s33, s8                                     // 000000009E20: A0210821
	s_add_f32 s38, s38, s12                                    // 000000009E24: A0260C26
	v_readlane_b32 s14, v26, 18                                // 000000009E28: D760000E 0001251A
	s_add_f32 s36, s4, s36                                     // 000000009E30: A0242404
	s_wait_alu 0xfffe                                          // 000000009E34: BF88FFFE
	s_add_f32 s33, s9, s33                                     // 000000009E38: A0212109
	s_add_f32 s38, s13, s38                                    // 000000009E3C: A026260D
	v_readlane_b32 s15, v26, 19                                // 000000009E40: D760000F 0001271A
	s_add_f32 s36, s5, s36                                     // 000000009E48: A0242405
	s_wait_alu 0xfffe                                          // 000000009E4C: BF88FFFE
	s_add_f32 s33, s10, s33                                    // 000000009E50: A021210A
	s_add_f32 s38, s14, s38                                    // 000000009E54: A026260E
	s_add_f32 s94, s6, s36                                     // 000000009E58: A05E2406
	s_wait_alu 0xfffe                                          // 000000009E5C: BF88FFFE
	s_add_f32 s101, s11, s33                                   // 000000009E60: A065210B
	s_add_f32 s97, s15, s38                                    // 000000009E64: A061260F
	s_or_saveexec_b32 s105, -1                                 // 000000009E68: BEE922C1
	scratch_load_b32 v20, off, off offset:104 th:TH_LOAD_LU    // 000000009E6C: ED05007C 00300014 00006800
	s_wait_alu 0xfffe                                          // 000000009E78: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009E7C: BEFE0069
	s_wait_loadcnt 0x0                                         // 000000009E80: BFC00000
	v_readlane_b32 s0, v20, 12                                 // 000000009E84: D7600000 00011914
	v_readlane_b32 s4, v20, 16                                 // 000000009E8C: D7600004 00012114
	v_readlane_b32 s5, v20, 17                                 // 000000009E94: D7600005 00012314
	v_readlane_b32 s9, v20, 21                                 // 000000009E9C: D7600009 00012B14
	v_readlane_b32 s1, v20, 13                                 // 000000009EA4: D7600001 00011B14
	v_readlane_b32 s6, v20, 18                                 // 000000009EAC: D7600006 00012514
	s_add_f32 s33, s102, s4                                    // 000000009EB4: A0210466
	v_readlane_b32 s10, v20, 22                                // 000000009EB8: D760000A 00012D14
	s_add_f32 s36, s39, s0                                     // 000000009EC0: A0240027
	s_add_f32 s41, s103, s9                                    // 000000009EC4: A0290967
	s_wait_alu 0xfffe                                          // 000000009EC8: BF88FFFE
	s_add_f32 s33, s5, s33                                     // 000000009ECC: A0212105
	v_readlane_b32 s2, v20, 14                                 // 000000009ED0: D7600002 00011D14
	v_readlane_b32 s7, v20, 19                                 // 000000009ED8: D7600007 00012714
	v_readlane_b32 s11, v20, 23                                // 000000009EE0: D760000B 00012F14
	s_add_f32 s36, s1, s36                                     // 000000009EE8: A0242401
	s_wait_alu 0xfffe                                          // 000000009EEC: BF88FFFE
	s_add_f32 s33, s6, s33                                     // 000000009EF0: A0212106
	s_add_f32 s41, s10, s41                                    // 000000009EF4: A029290A
	v_readlane_b32 s3, v20, 15                                 // 000000009EF8: D7600003 00011F14
	v_readlane_b32 s8, v20, 20                                 // 000000009F00: D7600008 00012914
	v_readlane_b32 s12, v20, 24                                // 000000009F08: D760000C 00013114
	s_add_f32 s36, s2, s36                                     // 000000009F10: A0242402
	s_wait_alu 0xfffe                                          // 000000009F14: BF88FFFE
	s_add_f32 s33, s7, s33                                     // 000000009F18: A0212107
	s_add_f32 s41, s11, s41                                    // 000000009F1C: A029290B
	v_readlane_b32 s13, v20, 25                                // 000000009F20: D760000D 00013314
	s_add_f32 s36, s3, s36                                     // 000000009F28: A0242403
	s_wait_alu 0xfffe                                          // 000000009F2C: BF88FFFE
	s_add_f32 s33, s33, s8                                     // 000000009F30: A0210821
	s_add_f32 s41, s41, s12                                    // 000000009F34: A0290C29
	v_readlane_b32 s14, v20, 26                                // 000000009F38: D760000E 00013514
	s_add_f32 s36, s4, s36                                     // 000000009F40: A0242404
	s_wait_alu 0xfffe                                          // 000000009F44: BF88FFFE
	s_add_f32 s33, s9, s33                                     // 000000009F48: A0212109
	s_add_f32 s41, s13, s41                                    // 000000009F4C: A029290D
	v_readlane_b32 s15, v20, 27                                // 000000009F50: D760000F 00013714
	s_add_f32 s36, s5, s36                                     // 000000009F58: A0242405
	s_wait_alu 0xfffe                                          // 000000009F5C: BF88FFFE
	s_add_f32 s33, s10, s33                                    // 000000009F60: A021210A
	s_add_f32 s41, s14, s41                                    // 000000009F64: A029290E
	s_add_f32 s95, s6, s36                                     // 000000009F68: A05F2406
	s_wait_alu 0xfffe                                          // 000000009F6C: BF88FFFE
	s_add_f32 s102, s11, s33                                   // 000000009F70: A066210B
	s_add_f32 s100, s15, s41                                   // 000000009F74: A064290F
	s_or_saveexec_b32 s105, -1                                 // 000000009F78: BEE922C1
	s_wait_alu 0xfffe                                          // 000000009F7C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 000000009F80: BEFE0069
	v_readlane_b32 s0, v29, 12                                 // 000000009F84: D7600000 0001191D
	v_readlane_b32 s4, v29, 16                                 // 000000009F8C: D7600004 0001211D
	v_readlane_b32 s5, v29, 17                                 // 000000009F94: D7600005 0001231D
	v_readlane_b32 s9, v29, 21                                 // 000000009F9C: D7600009 00012B1D
	v_readlane_b32 s1, v29, 13                                 // 000000009FA4: D7600001 00011B1D
	v_readlane_b32 s6, v29, 18                                 // 000000009FAC: D7600006 0001251D
	s_add_f32 s33, s35, s4                                     // 000000009FB4: A0210423
	v_readlane_b32 s10, v29, 22                                // 000000009FB8: D760000A 00012D1D
	s_add_f32 s0, s37, s0                                      // 000000009FC0: A0000025
	s_add_f32 s35, s42, s9                                     // 000000009FC4: A023092A
	s_wait_alu 0xfffe                                          // 000000009FC8: BF88FFFE
	s_add_f32 s33, s5, s33                                     // 000000009FCC: A0212105
	v_readlane_b32 s2, v29, 14                                 // 000000009FD0: D7600002 00011D1D
	v_readlane_b32 s11, v29, 23                                // 000000009FD8: D760000B 00012F1D
	s_add_f32 s0, s1, s0                                       // 000000009FE0: A0000001
	s_wait_alu 0xfffe                                          // 000000009FE4: BF88FFFE
	s_add_f32 s1, s6, s33                                      // 000000009FE8: A0012106
	s_add_f32 s33, s10, s35                                    // 000000009FEC: A021230A
	v_readlane_b32 s3, v29, 15                                 // 000000009FF0: D7600003 00011F1D
	v_readlane_b32 s7, v29, 19                                 // 000000009FF8: D7600007 0001271D
	v_readlane_b32 s12, v29, 24                                // 00000000A000: D760000C 0001311D
	s_add_f32 s0, s2, s0                                       // 00000000A008: A0000002
	s_wait_alu 0xfffe                                          // 00000000A00C: BF88FFFE
	s_add_f32 s2, s11, s33                                     // 00000000A010: A002210B
	v_readlane_b32 s8, v29, 20                                 // 00000000A014: D7600008 0001291D
	v_readlane_b32 s13, v29, 25                                // 00000000A01C: D760000D 0001331D
	s_add_f32 s1, s7, s1                                       // 00000000A024: A0010107
	s_add_f32 s0, s3, s0                                       // 00000000A028: A0000003
	s_wait_alu 0xfffe                                          // 00000000A02C: BF88FFFE
	s_add_f32 s2, s2, s12                                      // 00000000A030: A0020C02
	v_readlane_b32 s14, v29, 26                                // 00000000A034: D760000E 0001351D
	s_add_f32 s1, s1, s8                                       // 00000000A03C: A0010801
	s_add_f32 s0, s4, s0                                       // 00000000A040: A0000004
	s_wait_alu 0xfffe                                          // 00000000A044: BF88FFFE
	s_add_f32 s2, s13, s2                                      // 00000000A048: A002020D
	v_readlane_b32 s36, v20, 28                                // 00000000A04C: D7600024 00013914
	v_readlane_b32 s15, v29, 27                                // 00000000A054: D760000F 0001371D
	v_readlane_b32 s40, v25, 0                                 // 00000000A05C: D7600028 00010119
	s_add_f32 s1, s9, s1                                       // 00000000A064: A0010109
	s_add_f32 s0, s5, s0                                       // 00000000A068: A0000005
	s_wait_alu 0xfffe                                          // 00000000A06C: BF88FFFE
	s_add_f32 s2, s14, s2                                      // 00000000A070: A002020E
	v_readlane_b32 s39, v20, 31                                // 00000000A074: D7600027 00013F14
	v_readlane_b32 s41, v25, 1                                 // 00000000A07C: D7600029 00010319
	v_readlane_b32 s42, v25, 2                                 // 00000000A084: D760002A 00010519
	v_readlane_b32 s44, v25, 4                                 // 00000000A08C: D760002C 00010919
	v_readlane_b32 s45, v25, 5                                 // 00000000A094: D760002D 00010B19
	s_add_f32 s1, s10, s1                                      // 00000000A09C: A001010A
	s_add_f32 s3, s6, s0                                       // 00000000A0A0: A0030006
	s_wait_alu 0xfffe                                          // 00000000A0A4: BF88FFFE
	s_add_f32 s4, s15, s2                                      // 00000000A0A8: A004020F
	s_add_f32 s0, s92, s40                                     // 00000000A0AC: A000285C
	s_add_f32 s5, s11, s1                                      // 00000000A0B0: A005010B
	v_readlane_b32 s37, v20, 29                                // 00000000A0B4: D7600025 00013B14
	v_readlane_b32 s38, v20, 30                                // 00000000A0BC: D7600026 00013D14
	v_readlane_b32 s43, v25, 3                                 // 00000000A0C4: D760002B 00010719
	v_readlane_b32 s46, v25, 6                                 // 00000000A0CC: D760002E 00010D19
	v_readlane_b32 s47, v25, 7                                 // 00000000A0D4: D760002F 00010F19
	v_readlane_b32 s48, v25, 8                                 // 00000000A0DC: D7600030 00011119
	v_readlane_b32 s49, v25, 9                                 // 00000000A0E4: D7600031 00011319
	v_readlane_b32 s50, v25, 10                                // 00000000A0EC: D7600032 00011519
	v_readlane_b32 s51, v25, 11                                // 00000000A0F4: D7600033 00011719
	s_add_f32 s1, s93, s36                                     // 00000000A0FC: A001245D
	s_wait_alu 0xfffe                                          // 00000000A100: BF88FFFE
	s_add_f32 s0, s41, s0                                      // 00000000A104: A0000029
	v_readlane_b32 s2, v24, 24                                 // 00000000A108: D7600002 00013118
	s_or_saveexec_b32 s105, -1                                 // 00000000A110: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000A114: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A118: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 00000000A11C: BEE922C1
	scratch_store_b32 off, v24, off offset:124                 // 00000000A120: ED06807C 0C000000 00007C00
	s_wait_alu 0xfffe                                          // 00000000A12C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A130: BEFE0069
	s_add_f32 s2, s2, s45                                      // 00000000A134: A0022D02
	s_add_f32 s1, s37, s1                                      // 00000000A138: A0010125
	s_add_f32 s0, s42, s0                                      // 00000000A13C: A000002A
	s_add_f32 s6, s104, s52                                    // 00000000A140: A0063468
	s_wait_alu 0xfffe                                          // 00000000A144: BF88FFFE
	s_add_f32 s2, s46, s2                                      // 00000000A148: A002022E
	s_add_f32 s7, s96, s61                                     // 00000000A14C: A0073D60
	s_add_f32 s1, s38, s1                                      // 00000000A150: A0010126
	s_add_f32 s0, s43, s0                                      // 00000000A154: A000002B
	s_wait_alu 0xfffe                                          // 00000000A158: BF88FFFE
	s_add_f32 s2, s47, s2                                      // 00000000A15C: A002022F
	s_add_f32 s6, s53, s6                                      // 00000000A160: A0060635
	s_add_f32 s7, s62, s7                                      // 00000000A164: A007073E
	s_add_f32 s1, s39, s1                                      // 00000000A168: A0010127
	s_wait_alu 0xfffe                                          // 00000000A16C: BF88FFFE
	s_add_f32 s2, s2, s48                                      // 00000000A170: A0023002
	s_add_f32 s0, s0, s44                                      // 00000000A174: A0002C00
	s_add_f32 s6, s54, s6                                      // 00000000A178: A0060636
	s_add_f32 s7, s63, s7                                      // 00000000A17C: A007073F
	s_wait_alu 0xfffe                                          // 00000000A180: BF88FFFE
	s_add_f32 s2, s49, s2                                      // 00000000A184: A0020231
	s_add_f32 s1, s40, s1                                      // 00000000A188: A0010128
	s_add_f32 s0, s45, s0                                      // 00000000A18C: A000002D
	s_add_f32 s6, s55, s6                                      // 00000000A190: A0060637
	s_wait_alu 0xfffe                                          // 00000000A194: BF88FFFE
	s_add_f32 s2, s50, s2                                      // 00000000A198: A0020232
	s_add_f32 s7, s7, s64                                      // 00000000A19C: A0074007
	s_add_f32 s1, s41, s1                                      // 00000000A1A0: A0010129
	s_add_f32 s0, s46, s0                                      // 00000000A1A4: A000002E
	s_wait_alu 0xfffe                                          // 00000000A1A8: BF88FFFE
	s_add_f32 s33, s51, s2                                     // 00000000A1AC: A0210233
	s_add_f32 s2, s34, s56                                     // 00000000A1B0: A0023822
	s_add_f32 s6, s6, s56                                      // 00000000A1B4: A0063806
	s_add_f32 s7, s65, s7                                      // 00000000A1B8: A0070741
	s_add_f32 s1, s42, s1                                      // 00000000A1BC: A001012A
	s_wait_alu 0xfffe                                          // 00000000A1C0: BF88FFFE
	s_add_f32 s2, s57, s2                                      // 00000000A1C4: A0020239
	s_add_f32 s0, s47, s0                                      // 00000000A1C8: A000002F
	v_readlane_b32 s40, v25, 20                                // 00000000A1CC: D7600028 00012919
	v_readlane_b32 s41, v25, 21                                // 00000000A1D4: D7600029 00012B19
	s_wait_alu 0xfffe                                          // 00000000A1DC: BF88FFFE
	s_add_f32 s2, s58, s2                                      // 00000000A1E0: A002023A
	v_readlane_b32 s42, v25, 22                                // 00000000A1E4: D760002A 00012D19
	v_readlane_b32 s44, v25, 24                                // 00000000A1EC: D760002C 00013119
	v_readlane_b32 s45, v25, 25                                // 00000000A1F4: D760002D 00013319
	s_wait_alu 0xfffe                                          // 00000000A1FC: BF88FFFE
	s_add_f32 s2, s59, s2                                      // 00000000A200: A002023B
	s_add_f32 s6, s57, s6                                      // 00000000A204: A0060639
	s_add_f32 s7, s66, s7                                      // 00000000A208: A0070742
	v_readlane_b32 s43, v25, 23                                // 00000000A20C: D760002B 00012F19
	s_wait_alu 0xfffe                                          // 00000000A214: BF88FFFE
	s_add_f32 s2, s2, s60                                      // 00000000A218: A0023C02
	v_readlane_b32 s46, v25, 26                                // 00000000A21C: D760002E 00013519
	s_add_f32 s35, s67, s7                                     // 00000000A224: A0230743
	v_readlane_b32 s47, v25, 27                                // 00000000A228: D760002F 00013719
	s_wait_alu 0xfffe                                          // 00000000A230: BF88FFFE
	s_add_f32 s2, s61, s2                                      // 00000000A234: A002023D
	v_readlane_b32 s48, v25, 28                                // 00000000A238: D7600030 00013919
	v_readlane_b32 s49, v25, 29                                // 00000000A240: D7600031 00013B19
	v_readlane_b32 s50, v25, 30                                // 00000000A248: D7600032 00013D19
	s_wait_alu 0xfffe                                          // 00000000A250: BF88FFFE
	s_add_f32 s8, s62, s2                                      // 00000000A254: A008023E
	s_add_f32 s2, s58, s6                                      // 00000000A258: A002063A
	v_readlane_b32 s51, v25, 31                                // 00000000A25C: D7600033 00013F19
	s_wait_alu 0xfffe                                          // 00000000A264: BF88FFFE
	s_add_f32 s39, s63, s8                                     // 00000000A268: A027083F
	s_or_saveexec_b32 s105, -1                                 // 00000000A26C: BEE922C1
	scratch_store_b32 off, v25, off offset:120                 // 00000000A270: ED06807C 0C800000 00007800
	s_wait_alu 0xfffe                                          // 00000000A27C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A280: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 00000000A284: BEE922C1
	scratch_load_b32 v25, off, off offset:108 th:TH_LOAD_LU    // 00000000A288: ED05007C 00300019 00006C00
	s_wait_alu 0xfffe                                          // 00000000A294: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A298: BEFE0069
	s_add_f32 s6, s101, s44                                    // 00000000A29C: A0062C65
	s_add_f32 s8, s97, s49                                     // 00000000A2A0: A0083161
	s_add_f32 s7, s94, s40                                     // 00000000A2A4: A007285E
	s_wait_loadcnt 0x0                                         // 00000000A2A8: BFC00000
	v_readlane_b32 s52, v25, 0                                 // 00000000A2AC: D7600034 00010119
	s_wait_alu 0xfffe                                          // 00000000A2B4: BF88FFFE
	s_add_f32 s6, s45, s6                                      // 00000000A2B8: A006062D
	s_add_f32 s8, s50, s8                                      // 00000000A2BC: A0080832
	s_add_f32 s7, s41, s7                                      // 00000000A2C0: A0070729
	v_readlane_b32 s53, v25, 1                                 // 00000000A2C4: D7600035 00010319
	s_wait_alu 0xfffe                                          // 00000000A2CC: BF88FFFE
	s_add_f32 s6, s46, s6                                      // 00000000A2D0: A006062E
	s_add_f32 s8, s51, s8                                      // 00000000A2D4: A0080833
	s_add_f32 s7, s42, s7                                      // 00000000A2D8: A007072A
	v_readlane_b32 s54, v25, 2                                 // 00000000A2DC: D7600036 00010519
	s_wait_alu 0xfffe                                          // 00000000A2E4: BF88FFFE
	s_add_f32 s6, s47, s6                                      // 00000000A2E8: A006062F
	s_add_f32 s8, s8, s52                                      // 00000000A2EC: A0083408
	s_add_f32 s7, s43, s7                                      // 00000000A2F0: A007072B
	v_readlane_b32 s55, v25, 3                                 // 00000000A2F4: D7600037 00010719
	s_wait_alu 0xfffe                                          // 00000000A2FC: BF88FFFE
	s_add_f32 s6, s6, s48                                      // 00000000A300: A0063006
	s_add_f32 s8, s53, s8                                      // 00000000A304: A0080835
	s_add_f32 s7, s44, s7                                      // 00000000A308: A007072C
	s_wait_alu 0xfffe                                          // 00000000A30C: BF88FFFE
	s_add_f32 s6, s49, s6                                      // 00000000A310: A0060631
	s_add_f32 s8, s54, s8                                      // 00000000A314: A0080836
	s_add_f32 s7, s45, s7                                      // 00000000A318: A007072D
	s_wait_alu 0xfffe                                          // 00000000A31C: BF88FFFE
	s_add_f32 s6, s50, s6                                      // 00000000A320: A0060632
	s_add_f32 s34, s55, s8                                     // 00000000A324: A0220837
	s_add_f32 s40, s46, s7                                     // 00000000A328: A028072E
	s_wait_alu 0xfffe                                          // 00000000A32C: BF88FFFE
	s_add_f32 s37, s51, s6                                     // 00000000A330: A0250633
	s_or_saveexec_b32 s105, -1                                 // 00000000A334: BEE922C1
	scratch_load_b32 v29, off, off offset:112 th:TH_LOAD_LU    // 00000000A338: ED05007C 0030001D 00007000
	s_wait_alu 0xfffe                                          // 00000000A344: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A348: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000A34C: BFC00000
	v_readlane_b32 s44, v29, 28                                // 00000000A350: D760002C 0001391D
	v_readlane_b32 s48, v21, 0                                 // 00000000A358: D7600030 00010115
	s_add_f32 s5, s5, s20                                      // 00000000A360: A0051405
	v_readlane_b32 s45, v29, 29                                // 00000000A364: D760002D 00013B1D
	v_readlane_b32 s49, v21, 1                                 // 00000000A36C: D7600031 00010315
	v_readlane_b32 s53, v21, 5                                 // 00000000A374: D7600035 00010B15
	s_add_f32 s6, s102, s48                                    // 00000000A37C: A0063066
	s_add_f32 s7, s95, s44                                     // 00000000A380: A0072C5F
	s_add_f32 s3, s3, s16                                      // 00000000A384: A0031003
	s_wait_alu 0xfffe                                          // 00000000A388: BF88FFFE
	s_add_f32 s5, s21, s5                                      // 00000000A38C: A0050515
	v_readlane_b32 s46, v29, 30                                // 00000000A390: D760002E 00013D1D
	v_readlane_b32 s50, v21, 2                                 // 00000000A398: D7600032 00010515
	v_readlane_b32 s54, v21, 6                                 // 00000000A3A0: D7600036 00010D15
	s_add_f32 s6, s49, s6                                      // 00000000A3A8: A0060631
	s_add_f32 s8, s100, s53                                    // 00000000A3AC: A0083564
	s_add_f32 s7, s45, s7                                      // 00000000A3B0: A007072D
	s_add_f32 s4, s4, s25                                      // 00000000A3B4: A0041904
	s_add_f32 s3, s17, s3                                      // 00000000A3B8: A0030311
	s_wait_alu 0xfffe                                          // 00000000A3BC: BF88FFFE
	s_add_f32 s5, s22, s5                                      // 00000000A3C0: A0050516
	v_readlane_b32 s47, v29, 31                                // 00000000A3C4: D760002F 00013F1D
	v_readlane_b32 s51, v21, 3                                 // 00000000A3CC: D7600033 00010715
	v_readlane_b32 s55, v21, 7                                 // 00000000A3D4: D7600037 00010F15
	s_add_f32 s6, s50, s6                                      // 00000000A3DC: A0060632
	s_add_f32 s8, s54, s8                                      // 00000000A3E0: A0080836
	s_add_f32 s7, s46, s7                                      // 00000000A3E4: A007072E
	s_add_f32 s4, s26, s4                                      // 00000000A3E8: A004041A
	s_add_f32 s3, s18, s3                                      // 00000000A3EC: A0030312
	s_wait_alu 0xfffe                                          // 00000000A3F0: BF88FFFE
	s_add_f32 s5, s23, s5                                      // 00000000A3F4: A0050517
	v_readlane_b32 s52, v21, 4                                 // 00000000A3F8: D7600034 00010915
	v_readlane_b32 s56, v21, 8                                 // 00000000A400: D7600038 00011115
	s_add_f32 s6, s51, s6                                      // 00000000A408: A0060633
	s_add_f32 s8, s55, s8                                      // 00000000A40C: A0080837
	s_add_f32 s7, s47, s7                                      // 00000000A410: A007072F
	s_add_f32 s4, s27, s4                                      // 00000000A414: A004041B
	s_add_f32 s3, s19, s3                                      // 00000000A418: A0030313
	s_wait_alu 0xfffe                                          // 00000000A41C: BF88FFFE
	s_add_f32 s5, s5, s24                                      // 00000000A420: A0051805
	v_readlane_b32 s57, v21, 9                                 // 00000000A424: D7600039 00011315
	s_add_f32 s6, s6, s52                                      // 00000000A42C: A0063406
	s_add_f32 s8, s8, s56                                      // 00000000A430: A0083808
	s_add_f32 s7, s48, s7                                      // 00000000A434: A0070730
	s_add_f32 s4, s4, s28                                      // 00000000A438: A0041C04
	s_add_f32 s3, s20, s3                                      // 00000000A43C: A0030314
	s_wait_alu 0xfffe                                          // 00000000A440: BF88FFFE
	s_add_f32 s5, s25, s5                                      // 00000000A444: A0050519
	v_readlane_b32 s58, v21, 10                                // 00000000A448: D760003A 00011515
	s_add_f32 s6, s53, s6                                      // 00000000A450: A0060635
	s_add_f32 s8, s57, s8                                      // 00000000A454: A0080839
	s_add_f32 s7, s49, s7                                      // 00000000A458: A0070731
	s_add_f32 s4, s29, s4                                      // 00000000A45C: A004041D
	s_add_f32 s3, s21, s3                                      // 00000000A460: A0030315
	s_wait_alu 0xfffe                                          // 00000000A464: BF88FFFE
	s_add_f32 s5, s26, s5                                      // 00000000A468: A005051A
	v_readlane_b32 s59, v21, 11                                // 00000000A46C: D760003B 00011715
	s_add_f32 s6, s54, s6                                      // 00000000A474: A0060636
	s_add_f32 s8, s58, s8                                      // 00000000A478: A008083A
	s_add_f32 s41, s50, s7                                     // 00000000A47C: A0290732
	s_add_f32 s4, s30, s4                                      // 00000000A480: A004041E
	s_add_f32 s18, s22, s3                                     // 00000000A484: A0120316
	s_wait_alu 0xfffe                                          // 00000000A488: BF88FFFE
	s_add_f32 s17, s27, s5                                     // 00000000A48C: A011051B
	s_add_f32 s38, s55, s6                                     // 00000000A490: A0260637
	s_add_f32 s36, s59, s8                                     // 00000000A494: A024083B
	s_add_f32 s16, s31, s4                                     // 00000000A498: A010041F
	s_or_saveexec_b32 s105, -1                                 // 00000000A49C: BEE922C1
	scratch_load_b32 v23, off, off offset:8 th:TH_LOAD_LU      // 00000000A4A0: ED05007C 00300017 00000800
	s_wait_alu 0xfffe                                          // 00000000A4AC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A4B0: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000A4B4: BFC00000
	v_readlane_b32 s3, v23, 19                                 // 00000000A4B8: D7600003 00012717
	s_or_saveexec_b32 s105, -1                                 // 00000000A4C0: BEE922C1
	scratch_load_b32 v28, off, off offset:12                   // 00000000A4C4: ED05007C 0000001C 00000C00
	s_wait_alu 0xfffe                                          // 00000000A4D0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A4D4: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000A4D8: BFC00000
	v_readlane_b32 s4, v28, 31                                 // 00000000A4DC: D7600004 00013F1C
	s_or_saveexec_b32 s105, -1                                 // 00000000A4E4: BEE922C1
	scratch_store_b32 off, v21, off                            // 00000000A4E8: ED06807C 0A800000 00000000
	s_wait_alu 0xfffe                                          // 00000000A4F4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A4F8: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 00000000A4FC: BEE922C1
	scratch_load_b32 v27, off, off offset:40                   // 00000000A500: ED05007C 0000001B 00002800
	s_wait_alu 0xfffe                                          // 00000000A50C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A510: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000A514: BFC00000
	v_readlane_b32 s5, v27, 0                                  // 00000000A518: D7600005 0001011B
	s_add_f32 s3, s3, s4                                       // 00000000A520: A0030403
	v_readlane_b32 s6, v27, 1                                  // 00000000A524: D7600006 0001031B
	v_readlane_b32 s8, v27, 3                                  // 00000000A52C: D7600008 0001071B
	v_readlane_b32 s4, v23, 30                                 // 00000000A534: D7600004 00013D17
	s_wait_alu 0xfffe                                          // 00000000A53C: BF88FFFE
	s_add_f32 s3, s5, s3                                       // 00000000A540: A0030305
	v_readlane_b32 s7, v27, 2                                  // 00000000A544: D7600007 0001051B
	v_readlane_b32 s9, v27, 4                                  // 00000000A54C: D7600009 0001091B
	v_readlane_b32 s10, v27, 5                                 // 00000000A554: D760000A 00010B1B
	s_wait_alu 0xfffe                                          // 00000000A55C: BF88FFFE
	s_add_f32 s3, s6, s3                                       // 00000000A560: A0030306
	s_add_f32 s4, s4, s8                                       // 00000000A564: A0040804
	v_readlane_b32 s11, v27, 6                                 // 00000000A568: D760000B 00010D1B
	v_readlane_b32 s5, v23, 16                                 // 00000000A570: D7600005 00012117
	s_wait_alu 0xfffe                                          // 00000000A578: BF88FFFE
	s_add_f32 s3, s7, s3                                       // 00000000A57C: A0030307
	s_add_f32 s4, s9, s4                                       // 00000000A580: A0040409
	s_wait_alu 0xfffe                                          // 00000000A584: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000A588: BF870499
	s_add_f32 s3, s8, s3                                       // 00000000A58C: A0030308
	s_add_f32 s4, s10, s4                                      // 00000000A590: A004040A
	s_wait_alu 0xfffe                                          // 00000000A594: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000A598: BF870499
	s_add_f32 s3, s9, s3                                       // 00000000A59C: A0030309
	s_add_f32 s4, s11, s4                                      // 00000000A5A0: A004040B
	s_wait_alu 0xfffe                                          // 00000000A5A4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000A5A8: BF870009
	s_add_f32 s3, s10, s3                                      // 00000000A5AC: A003030A
	s_or_saveexec_b32 s105, -1                                 // 00000000A5B0: BEE922C1
	scratch_load_b32 v22, off, off offset:60 th:TH_LOAD_LU     // 00000000A5B4: ED05007C 00300016 00003C00
	s_wait_alu 0xfffe                                          // 00000000A5C0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A5C4: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000A5C8: BFC00000
	v_readlane_b32 s8, v22, 12                                 // 00000000A5CC: D7600008 00011916
	v_readlane_b32 s9, v22, 13                                 // 00000000A5D4: D7600009 00011B16
	v_readlane_b32 s10, v22, 14                                // 00000000A5DC: D760000A 00011D16
	v_readlane_b32 s11, v22, 15                                // 00000000A5E4: D760000B 00011F16
	v_readlane_b32 s12, v22, 16                                // 00000000A5EC: D760000C 00012116
	v_readlane_b32 s13, v22, 17                                // 00000000A5F4: D760000D 00012316
	v_readlane_b32 s14, v22, 18                                // 00000000A5FC: D760000E 00012516
	v_readlane_b32 s15, v22, 19                                // 00000000A604: D760000F 00012716
	s_or_saveexec_b32 s105, -1                                 // 00000000A60C: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000A610: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A614: BEFE0069
	s_add_f32 s5, s5, s9                                       // 00000000A618: A0050905
	s_add_f32 s4, s4, s8                                       // 00000000A61C: A0040804
	v_readlane_b32 s6, v23, 29                                 // 00000000A620: D7600006 00013B17
	s_wait_alu 0xfffe                                          // 00000000A628: BF88FFFE
	s_add_f32 s5, s10, s5                                      // 00000000A62C: A005050A
	s_add_f32 s4, s9, s4                                       // 00000000A630: A0040409
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 00000000A634: BF8704D1
	s_add_f32 s6, s6, 0                                        // 00000000A638: A0068006
	s_wait_alu 0xfffe                                          // 00000000A63C: BF88FFFE
	s_add_f32 s5, s11, s5                                      // 00000000A640: A005050B
	s_add_f32 s4, s10, s4                                      // 00000000A644: A004040A
	s_wait_alu 0xfffe                                          // 00000000A648: BF88FFFE
	s_add_f32 s5, s5, s12                                      // 00000000A64C: A0050C05
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000A650: BF8704A9
	s_add_f32 s4, s11, s4                                      // 00000000A654: A004040B
	s_wait_alu 0xfffe                                          // 00000000A658: BF88FFFE
	s_add_f32 s5, s13, s5                                      // 00000000A65C: A005050D
	s_wait_alu 0xfffe                                          // 00000000A660: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000A664: BF87052A
	s_add_f32 s5, s14, s5                                      // 00000000A668: A005050E
	s_wait_alu 0xfffe                                          // 00000000A66C: BF88FFFE
	s_add_f32 s5, s15, s5                                      // 00000000A670: A005050F
	s_or_saveexec_b32 s105, -1                                 // 00000000A674: BEE922C1
	scratch_load_b32 v24, off, off offset:48 th:TH_LOAD_LU     // 00000000A678: ED05007C 00300018 00003000
	s_wait_alu 0xfffe                                          // 00000000A684: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A688: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000A68C: BFC00000
	v_readlane_b32 s8, v24, 10                                 // 00000000A690: D7600008 00011518
	v_readlane_b32 s9, v24, 11                                 // 00000000A698: D7600009 00011718
	v_readlane_b32 s10, v24, 12                                // 00000000A6A0: D760000A 00011918
	v_readlane_b32 s11, v24, 13                                // 00000000A6A8: D760000B 00011B18
	v_readlane_b32 s12, v24, 14                                // 00000000A6B0: D760000C 00011D18
	v_readlane_b32 s13, v24, 15                                // 00000000A6B8: D760000D 00011F18
	v_readlane_b32 s14, v24, 16                                // 00000000A6C0: D760000E 00012118
	v_readlane_b32 s15, v24, 17                                // 00000000A6C8: D760000F 00012318
	s_or_saveexec_b32 s105, -1                                 // 00000000A6D0: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000A6D4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A6D8: BEFE0069
	s_add_f32 s6, s6, s8                                       // 00000000A6DC: A0060806
	v_readlane_b32 s7, v23, 18                                 // 00000000A6E0: D7600007 00012517
	v_readlane_b32 s8, v23, 28                                 // 00000000A6E8: D7600008 00013917
	s_wait_alu 0xfffe                                          // 00000000A6F0: BF88FFFE
	s_add_f32 s6, s9, s6                                       // 00000000A6F4: A0060609
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A6F8: BF870092
	s_add_f32 s7, s7, 0                                        // 00000000A6FC: A0078007
	s_add_f32 s8, s8, 0                                        // 00000000A700: A0088008
	s_wait_alu 0xfffe                                          // 00000000A704: BF88FFFE
	s_add_f32 s6, s10, s6                                      // 00000000A708: A006060A
	s_add_f32 s7, s7, s12                                      // 00000000A70C: A0070C07
	s_wait_alu 0xfffe                                          // 00000000A710: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000A714: BF870499
	s_add_f32 s6, s11, s6                                      // 00000000A718: A006060B
	s_add_f32 s7, s13, s7                                      // 00000000A71C: A007070D
	s_wait_alu 0xfffe                                          // 00000000A720: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000A724: BF870499
	s_add_f32 s6, s12, s6                                      // 00000000A728: A006060C
	s_add_f32 s7, s14, s7                                      // 00000000A72C: A007070E
	s_wait_alu 0xfffe                                          // 00000000A730: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000A734: BF870499
	s_add_f32 s6, s13, s6                                      // 00000000A738: A006060D
	s_add_f32 s7, s15, s7                                      // 00000000A73C: A007070F
	s_wait_alu 0xfffe                                          // 00000000A740: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000A744: BF870009
	s_add_f32 s6, s14, s6                                      // 00000000A748: A006060E
	s_or_saveexec_b32 s105, -1                                 // 00000000A74C: BEE922C1
	scratch_load_b32 v27, off, off offset:52 th:TH_LOAD_LU     // 00000000A750: ED05007C 0030001B 00003400
	s_wait_alu 0xfffe                                          // 00000000A75C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A760: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000A764: BFC00000
	v_readlane_b32 s20, v27, 2                                 // 00000000A768: D7600014 0001051B
	v_readlane_b32 s21, v27, 3                                 // 00000000A770: D7600015 0001071B
	v_readlane_b32 s22, v27, 4                                 // 00000000A778: D7600016 0001091B
	v_readlane_b32 s23, v27, 5                                 // 00000000A780: D7600017 00010B1B
	v_readlane_b32 s24, v27, 6                                 // 00000000A788: D7600018 00010D1B
	v_readlane_b32 s25, v27, 7                                 // 00000000A790: D7600019 00010F1B
	v_readlane_b32 s27, v27, 9                                 // 00000000A798: D760001B 0001131B
	v_readlane_b32 s26, v27, 8                                 // 00000000A7A0: D760001A 0001111B
	s_or_saveexec_b32 s105, -1                                 // 00000000A7A8: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000A7AC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A7B0: BEFE0069
	s_add_f32 s8, s8, s20                                      // 00000000A7B4: A0081408
	v_readlane_b32 s9, v23, 31                                 // 00000000A7B8: D7600009 00013F17
	s_wait_alu 0xfffe                                          // 00000000A7C0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A7C4: BF870099
	s_add_f32 s8, s21, s8                                      // 00000000A7C8: A0080815
	s_add_f32 s9, s9, 0                                        // 00000000A7CC: A0098009
	s_wait_alu 0xfffe                                          // 00000000A7D0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000A7D4: BF870499
	s_add_f32 s8, s22, s8                                      // 00000000A7D8: A0080816
	s_add_f32 s9, s9, s24                                      // 00000000A7DC: A0091809
	s_wait_alu 0xfffe                                          // 00000000A7E0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000A7E4: BF870499
	s_add_f32 s8, s23, s8                                      // 00000000A7E8: A0080817
	s_add_f32 s9, s25, s9                                      // 00000000A7EC: A0090919
	s_wait_alu 0xfffe                                          // 00000000A7F0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000A7F4: BF870499
	s_add_f32 s8, s24, s8                                      // 00000000A7F8: A0080818
	s_add_f32 s9, s26, s9                                      // 00000000A7FC: A009091A
	s_wait_alu 0xfffe                                          // 00000000A800: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000A804: BF870499
	s_add_f32 s8, s25, s8                                      // 00000000A808: A0080819
	s_add_f32 s9, s27, s9                                      // 00000000A80C: A009091B
	s_wait_alu 0xfffe                                          // 00000000A810: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000A814: BF870009
	s_add_f32 s8, s26, s8                                      // 00000000A818: A008081A
	s_or_saveexec_b32 s105, -1                                 // 00000000A81C: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000A820: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A824: BEFE0069
	v_readlane_b32 s10, v28, 3                                 // 00000000A828: D760000A 0001071C
	v_readlane_b32 s20, v24, 18                                // 00000000A830: D7600014 00012518
	v_readlane_b32 s21, v24, 19                                // 00000000A838: D7600015 00012718
	v_readlane_b32 s22, v24, 20                                // 00000000A840: D7600016 00012918
	v_readlane_b32 s23, v24, 21                                // 00000000A848: D7600017 00012B18
	s_add_f32 s10, s10, 0                                      // 00000000A850: A00A800A
	v_readlane_b32 s24, v24, 22                                // 00000000A854: D7600018 00012D18
	v_readlane_b32 s25, v24, 23                                // 00000000A85C: D7600019 00012F18
	v_readlane_b32 s27, v24, 25                                // 00000000A864: D760001B 00013318
	s_wait_alu 0xfffe                                          // 00000000A86C: BF88FFFE
	s_add_f32 s10, s10, s20                                    // 00000000A870: A00A140A
	v_readlane_b32 s26, v24, 24                                // 00000000A874: D760001A 00013118
	v_readlane_b32 s11, v28, 1                                 // 00000000A87C: D760000B 0001031C
	s_wait_alu 0xfffe                                          // 00000000A884: BF88FFFE
	s_add_f32 s10, s21, s10                                    // 00000000A888: A00A0A15
	s_wait_alu 0xfffe                                          // 00000000A88C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000A890: BF87000A
	s_add_f32 s10, s22, s10                                    // 00000000A894: A00A0A16
	s_or_saveexec_b32 s105, -1                                 // 00000000A898: BEE922C1
	scratch_load_b32 v21, off, off offset:124                  // 00000000A89C: ED05007C 00000015 00007C00
	s_wait_alu 0xfffe                                          // 00000000A8A8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A8AC: BEFE0069
	s_add_f32 s11, s11, 0                                      // 00000000A8B0: A00B800B
	s_add_f32 s10, s23, s10                                    // 00000000A8B4: A00A0A17
	v_readlane_b32 s12, v23, 17                                // 00000000A8B8: D760000C 00012317
	s_wait_alu 0xfffe                                          // 00000000A8C0: BF88FFFE
	s_add_f32 s11, s11, s24                                    // 00000000A8C4: A00B180B
	s_add_f32 s10, s24, s10                                    // 00000000A8C8: A00A0A18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 00000000A8CC: BF8704D1
	s_add_f32 s12, s12, 0                                      // 00000000A8D0: A00C800C
	s_wait_alu 0xfffe                                          // 00000000A8D4: BF88FFFE
	s_add_f32 s11, s25, s11                                    // 00000000A8D8: A00B0B19
	s_add_f32 s10, s25, s10                                    // 00000000A8DC: A00A0A19
	s_wait_alu 0xfffe                                          // 00000000A8E0: BF88FFFE
	s_add_f32 s11, s26, s11                                    // 00000000A8E4: A00B0B1A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000A8E8: BF8704A9
	s_add_f32 s10, s26, s10                                    // 00000000A8EC: A00A0A1A
	s_wait_alu 0xfffe                                          // 00000000A8F0: BF88FFFE
	s_add_f32 s11, s27, s11                                    // 00000000A8F4: A00B0B1B
	v_readlane_b32 s20, v24, 26                                // 00000000A8F8: D7600014 00013518
	v_readlane_b32 s21, v24, 27                                // 00000000A900: D7600015 00013718
	v_readlane_b32 s22, v24, 28                                // 00000000A908: D7600016 00013918
	v_readlane_b32 s24, v24, 30                                // 00000000A910: D7600018 00013D18
	v_readlane_b32 s23, v24, 29                                // 00000000A918: D7600017 00013B18
	s_add_f32 s3, s3, s20                                      // 00000000A920: A0031403
	v_readlane_b32 s25, v24, 31                                // 00000000A924: D7600019 00013F18
	v_readlane_b32 s26, v27, 0                                 // 00000000A92C: D760001A 0001011B
	s_add_f32 s4, s4, s24                                      // 00000000A934: A0041804
	s_wait_alu 0xfffe                                          // 00000000A938: BF88FFFE
	s_add_f32 s3, s21, s3                                      // 00000000A93C: A0030315
	v_readlane_b32 s27, v27, 1                                 // 00000000A940: D760001B 0001031B
	s_add_f32 s4, s25, s4                                      // 00000000A948: A0040419
	s_wait_alu 0xfffe                                          // 00000000A94C: BF88FFFE
	s_add_f32 s3, s22, s3                                      // 00000000A950: A0030316
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000A954: BF8704A9
	s_add_f32 s4, s26, s4                                      // 00000000A958: A004041A
	s_wait_alu 0xfffe                                          // 00000000A95C: BF88FFFE
	s_add_f32 s3, s23, s3                                      // 00000000A960: A0030317
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000A964: BF8704A9
	s_add_f32 s4, s27, s4                                      // 00000000A968: A004041B
	s_wait_alu 0xfffe                                          // 00000000A96C: BF88FFFE
	s_add_f32 s3, s24, s3                                      // 00000000A970: A0030318
	s_wait_alu 0xfffe                                          // 00000000A974: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000A978: BF87052A
	s_add_f32 s3, s25, s3                                      // 00000000A97C: A0030319
	s_wait_alu 0xfffe                                          // 00000000A980: BF88FFFE
	s_add_f32 s3, s26, s3                                      // 00000000A984: A003031A
	s_or_saveexec_b32 s105, -1                                 // 00000000A988: BEE922C1
	scratch_load_b32 v28, off, off offset:44 th:TH_LOAD_LU     // 00000000A98C: ED05007C 0030001C 00002C00
	s_wait_alu 0xfffe                                          // 00000000A998: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000A99C: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000A9A0: BFC00000
	v_readlane_b32 s20, v28, 18                                // 00000000A9A4: D7600014 0001251C
	v_readlane_b32 s21, v28, 19                                // 00000000A9AC: D7600015 0001271C
	v_readlane_b32 s22, v28, 20                                // 00000000A9B4: D7600016 0001291C
	v_readlane_b32 s23, v28, 21                                // 00000000A9BC: D7600017 00012B1C
	v_readlane_b32 s24, v28, 22                                // 00000000A9C4: D7600018 00012D1C
	v_readlane_b32 s25, v28, 23                                // 00000000A9CC: D7600019 00012F1C
	s_add_f32 s12, s12, s21                                    // 00000000A9D4: A00C150C
	s_add_f32 s7, s7, s20                                      // 00000000A9D8: A0071407
	v_readlane_b32 s26, v28, 24                                // 00000000A9DC: D760001A 0001311C
	v_readlane_b32 s27, v28, 25                                // 00000000A9E4: D760001B 0001331C
	s_wait_alu 0xfffe                                          // 00000000A9EC: BF88FFFE
	s_add_f32 s12, s22, s12                                    // 00000000A9F0: A00C0C16
	s_add_f32 s7, s21, s7                                      // 00000000A9F4: A0070715
	s_wait_alu 0xfffe                                          // 00000000A9F8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000A9FC: BF870499
	s_add_f32 s12, s23, s12                                    // 00000000AA00: A00C0C17
	s_add_f32 s7, s22, s7                                      // 00000000AA04: A0070716
	s_wait_alu 0xfffe                                          // 00000000AA08: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000AA0C: BF870499
	s_add_f32 s12, s12, s24                                    // 00000000AA10: A00C180C
	s_add_f32 s7, s23, s7                                      // 00000000AA14: A0070717
	s_wait_alu 0xfffe                                          // 00000000AA18: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000AA1C: BF870529
	s_add_f32 s12, s25, s12                                    // 00000000AA20: A00C0C19
	s_wait_alu 0xfffe                                          // 00000000AA24: BF88FFFE
	s_add_f32 s12, s26, s12                                    // 00000000AA28: A00C0C1A
	s_wait_alu 0xfffe                                          // 00000000AA2C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000AA30: BF87000A
	s_add_f32 s12, s27, s12                                    // 00000000AA34: A00C0C1B
	s_or_saveexec_b32 s105, -1                                 // 00000000AA38: BEE922C1
	scratch_load_b32 v28, off, off offset:40 th:TH_LOAD_LU     // 00000000AA3C: ED05007C 0030001C 00002800
	s_wait_alu 0xfffe                                          // 00000000AA48: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000AA4C: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000AA50: BFC00000
	v_readlane_b32 s13, v28, 27                                // 00000000AA54: D760000D 0001371C
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000AA5C: BF870001
	s_add_f32 s13, s13, 0                                      // 00000000AA60: A00D800D
	s_or_saveexec_b32 s105, -1                                 // 00000000AA64: BEE922C1
	scratch_load_b32 v24, off, off offset:20 th:TH_LOAD_LU     // 00000000AA68: ED05007C 00300018 00001400
	s_wait_alu 0xfffe                                          // 00000000AA74: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000AA78: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000AA7C: BFC00000
	v_readlane_b32 s20, v24, 12                                // 00000000AA80: D7600014 00011918
	v_readlane_b32 s21, v24, 13                                // 00000000AA88: D7600015 00011B18
	v_readlane_b32 s22, v24, 14                                // 00000000AA90: D7600016 00011D18
	v_readlane_b32 s23, v24, 15                                // 00000000AA98: D7600017 00011F18
	v_readlane_b32 s24, v24, 16                                // 00000000AAA0: D7600018 00012118
	v_readlane_b32 s25, v24, 17                                // 00000000AAA8: D7600019 00012318
	v_readlane_b32 s27, v24, 19                                // 00000000AAB0: D760001B 00012718
	v_readlane_b32 s26, v24, 18                                // 00000000AAB8: D760001A 00012518
	s_or_saveexec_b32 s105, -1                                 // 00000000AAC0: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000AAC4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000AAC8: BEFE0069
	s_add_f32 s13, s13, s21                                    // 00000000AACC: A00D150D
	s_add_f32 s9, s9, s20                                      // 00000000AAD0: A0091409
	v_readlane_b32 s14, v28, 28                                // 00000000AAD4: D760000E 0001391C
	s_wait_alu 0xfffe                                          // 00000000AADC: BF88FFFE
	s_add_f32 s13, s22, s13                                    // 00000000AAE0: A00D0D16
	s_add_f32 s9, s21, s9                                      // 00000000AAE4: A0090915
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 00000000AAE8: BF8704D1
	s_add_f32 s14, s14, 0                                      // 00000000AAEC: A00E800E
	s_wait_alu 0xfffe                                          // 00000000AAF0: BF88FFFE
	s_add_f32 s13, s23, s13                                    // 00000000AAF4: A00D0D17
	s_add_f32 s9, s22, s9                                      // 00000000AAF8: A0090916
	s_wait_alu 0xfffe                                          // 00000000AAFC: BF88FFFE
	s_add_f32 s13, s13, s24                                    // 00000000AB00: A00D180D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000AB04: BF8704A9
	s_add_f32 s9, s23, s9                                      // 00000000AB08: A0090917
	s_wait_alu 0xfffe                                          // 00000000AB0C: BF88FFFE
	s_add_f32 s13, s25, s13                                    // 00000000AB10: A00D0D19
	s_wait_alu 0xfffe                                          // 00000000AB14: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000AB18: BF87052A
	s_add_f32 s13, s26, s13                                    // 00000000AB1C: A00D0D1A
	s_wait_alu 0xfffe                                          // 00000000AB20: BF88FFFE
	s_add_f32 s13, s27, s13                                    // 00000000AB24: A00D0D1B
	v_readlane_b32 s20, v27, 26                                // 00000000AB28: D7600014 0001351B
	v_readlane_b32 s21, v27, 27                                // 00000000AB30: D7600015 0001371B
	v_readlane_b32 s22, v27, 28                                // 00000000AB38: D7600016 0001391B
	v_readlane_b32 s23, v27, 29                                // 00000000AB40: D7600017 00013B1B
	v_readlane_b32 s24, v27, 30                                // 00000000AB48: D7600018 00013D1B
	v_readlane_b32 s25, v27, 31                                // 00000000AB50: D7600019 00013F1B
	s_or_saveexec_b32 s105, -1                                 // 00000000AB58: BEE922C1
	scratch_load_b32 v28, off, off offset:16 th:TH_LOAD_LU     // 00000000AB5C: ED05007C 0030001C 00001000
	s_wait_alu 0xfffe                                          // 00000000AB68: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000AB6C: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000AB70: BFC00000
	v_readlane_b32 s27, v28, 1                                 // 00000000AB74: D760001B 0001031C
	v_readlane_b32 s26, v28, 0                                 // 00000000AB7C: D760001A 0001011C
	s_or_saveexec_b32 s105, -1                                 // 00000000AB84: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000AB88: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000AB8C: BEFE0069
	s_add_f32 s14, s14, s21                                    // 00000000AB90: A00E150E
	s_add_f32 s11, s11, s20                                    // 00000000AB94: A00B140B
	s_wait_alu 0xfffe                                          // 00000000AB98: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000AB9C: BF870499
	s_add_f32 s14, s22, s14                                    // 00000000ABA0: A00E0E16
	s_add_f32 s11, s21, s11                                    // 00000000ABA4: A00B0B15
	s_wait_alu 0xfffe                                          // 00000000ABA8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000ABAC: BF870499
	s_add_f32 s14, s23, s14                                    // 00000000ABB0: A00E0E17
	s_add_f32 s11, s22, s11                                    // 00000000ABB4: A00B0B16
	s_wait_alu 0xfffe                                          // 00000000ABB8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000ABBC: BF870499
	s_add_f32 s14, s14, s24                                    // 00000000ABC0: A00E180E
	s_add_f32 s11, s23, s11                                    // 00000000ABC4: A00B0B17
	s_wait_alu 0xfffe                                          // 00000000ABC8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000ABCC: BF870529
	s_add_f32 s14, s25, s14                                    // 00000000ABD0: A00E0E19
	s_wait_alu 0xfffe                                          // 00000000ABD4: BF88FFFE
	s_add_f32 s14, s26, s14                                    // 00000000ABD8: A00E0E1A
	s_wait_alu 0xfffe                                          // 00000000ABDC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000ABE0: BF87000A
	s_add_f32 s14, s27, s14                                    // 00000000ABE4: A00E0E1B
	v_readlane_b32 s20, v24, 4                                 // 00000000ABE8: D7600014 00010918
	v_readlane_b32 s21, v24, 5                                 // 00000000ABF0: D7600015 00010B18
	v_readlane_b32 s22, v24, 6                                 // 00000000ABF8: D7600016 00010D18
	v_readlane_b32 s23, v24, 7                                 // 00000000AC00: D7600017 00010F18
	v_readlane_b32 s24, v24, 8                                 // 00000000AC08: D7600018 00011118
	s_add_f32 s4, s4, s20                                      // 00000000AC10: A0041404
	s_add_f32 s5, s5, s21                                      // 00000000AC14: A0051505
	v_readlane_b32 s25, v24, 9                                 // 00000000AC18: D7600019 00011318
	v_readlane_b32 s26, v24, 10                                // 00000000AC20: D760001A 00011518
	s_wait_alu 0xfffe                                          // 00000000AC28: BF88FFFE
	s_add_f32 s4, s21, s4                                      // 00000000AC2C: A0040415
	s_add_f32 s5, s22, s5                                      // 00000000AC30: A0050516
	v_readlane_b32 s27, v24, 11                                // 00000000AC34: D760001B 00011718
	s_wait_alu 0xfffe                                          // 00000000AC3C: BF88FFFE
	s_add_f32 s4, s22, s4                                      // 00000000AC40: A0040416
	s_add_f32 s5, s23, s5                                      // 00000000AC44: A0050517
	s_wait_alu 0xfffe                                          // 00000000AC48: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000AC4C: BF870499
	s_add_f32 s4, s23, s4                                      // 00000000AC50: A0040417
	s_add_f32 s5, s5, s24                                      // 00000000AC54: A0051805
	s_wait_alu 0xfffe                                          // 00000000AC58: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000AC5C: BF87052A
	s_add_f32 s5, s25, s5                                      // 00000000AC60: A0050519
	s_wait_alu 0xfffe                                          // 00000000AC64: BF88FFFE
	s_add_f32 s5, s26, s5                                      // 00000000AC68: A005051A
	s_wait_alu 0xfffe                                          // 00000000AC6C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000AC70: BF87000A
	s_add_f32 s5, s27, s5                                      // 00000000AC74: A005051B
	s_or_saveexec_b32 s105, -1                                 // 00000000AC78: BEE922C1
	scratch_load_b32 v27, off, off offset:56 th:TH_LOAD_LU     // 00000000AC7C: ED05007C 0030001B 00003800
	s_wait_alu 0xfffe                                          // 00000000AC88: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000AC8C: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000AC90: BFC00000
	v_readlane_b32 s20, v27, 28                                // 00000000AC94: D7600014 0001391B
	v_readlane_b32 s21, v27, 29                                // 00000000AC9C: D7600015 00013B1B
	v_readlane_b32 s22, v27, 30                                // 00000000ACA4: D7600016 00013D1B
	v_readlane_b32 s23, v27, 31                                // 00000000ACAC: D7600017 00013F1B
	v_readlane_b32 s24, v22, 0                                 // 00000000ACB4: D7600018 00010116
	s_add_f32 s6, s6, s20                                      // 00000000ACBC: A0061406
	v_readlane_b32 s25, v22, 1                                 // 00000000ACC0: D7600019 00010316
	v_readlane_b32 s26, v22, 2                                 // 00000000ACC8: D760001A 00010516
	v_readlane_b32 s27, v22, 3                                 // 00000000ACD0: D760001B 00010716
	s_wait_alu 0xfffe                                          // 00000000ACD8: BF88FFFE
	s_add_f32 s6, s21, s6                                      // 00000000ACDC: A0060615
	s_add_f32 s7, s7, s24                                      // 00000000ACE0: A0071807
	s_wait_alu 0xfffe                                          // 00000000ACE4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000ACE8: BF870499
	s_add_f32 s6, s22, s6                                      // 00000000ACEC: A0060616
	s_add_f32 s7, s25, s7                                      // 00000000ACF0: A0070719
	s_wait_alu 0xfffe                                          // 00000000ACF4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000ACF8: BF870499
	s_add_f32 s6, s23, s6                                      // 00000000ACFC: A0060617
	s_add_f32 s7, s26, s7                                      // 00000000AD00: A007071A
	s_wait_alu 0xfffe                                          // 00000000AD04: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000AD08: BF870499
	s_add_f32 s6, s24, s6                                      // 00000000AD0C: A0060618
	s_add_f32 s7, s27, s7                                      // 00000000AD10: A007071B
	s_wait_alu 0xfffe                                          // 00000000AD14: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000AD18: BF870529
	s_add_f32 s6, s25, s6                                      // 00000000AD1C: A0060619
	s_wait_alu 0xfffe                                          // 00000000AD20: BF88FFFE
	s_add_f32 s6, s26, s6                                      // 00000000AD24: A006061A
	v_readlane_b32 s20, v27, 20                                // 00000000AD28: D7600014 0001291B
	v_readlane_b32 s21, v27, 21                                // 00000000AD30: D7600015 00012B1B
	v_readlane_b32 s22, v27, 22                                // 00000000AD38: D7600016 00012D1B
	v_readlane_b32 s23, v27, 23                                // 00000000AD40: D7600017 00012F1B
	v_readlane_b32 s24, v27, 24                                // 00000000AD48: D7600018 0001311B
	v_readlane_b32 s25, v27, 25                                // 00000000AD50: D7600019 0001331B
	v_readlane_b32 s27, v27, 27                                // 00000000AD58: D760001B 0001371B
	v_readlane_b32 s26, v27, 26                                // 00000000AD60: D760001A 0001351B
	s_or_saveexec_b32 s105, -1                                 // 00000000AD68: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000AD6C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000AD70: BEFE0069
	s_add_f32 s8, s8, s20                                      // 00000000AD74: A0081408
	s_add_f32 s9, s9, s24                                      // 00000000AD78: A0091809
	s_wait_alu 0xfffe                                          // 00000000AD7C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000AD80: BF870499
	s_add_f32 s8, s21, s8                                      // 00000000AD84: A0080815
	s_add_f32 s9, s25, s9                                      // 00000000AD88: A0090919
	s_wait_alu 0xfffe                                          // 00000000AD8C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000AD90: BF870499
	s_add_f32 s8, s22, s8                                      // 00000000AD94: A0080816
	s_add_f32 s9, s26, s9                                      // 00000000AD98: A009091A
	s_wait_alu 0xfffe                                          // 00000000AD9C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000ADA0: BF870499
	s_add_f32 s8, s23, s8                                      // 00000000ADA4: A0080817
	s_add_f32 s9, s27, s9                                      // 00000000ADA8: A009091B
	s_wait_alu 0xfffe                                          // 00000000ADAC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000ADB0: BF870529
	s_add_f32 s8, s24, s8                                      // 00000000ADB4: A0080818
	s_wait_alu 0xfffe                                          // 00000000ADB8: BF88FFFE
	s_add_f32 s8, s25, s8                                      // 00000000ADBC: A0080819
	s_wait_alu 0xfffe                                          // 00000000ADC0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000ADC4: BF87000A
	s_add_f32 s8, s26, s8                                      // 00000000ADC8: A008081A
	v_readlane_b32 s20, v22, 4                                 // 00000000ADCC: D7600014 00010916
	v_readlane_b32 s21, v22, 5                                 // 00000000ADD4: D7600015 00010B16
	v_readlane_b32 s22, v22, 6                                 // 00000000ADDC: D7600016 00010D16
	v_readlane_b32 s23, v22, 7                                 // 00000000ADE4: D7600017 00010F16
	v_readlane_b32 s24, v22, 8                                 // 00000000ADEC: D7600018 00011116
	s_add_f32 s10, s10, s20                                    // 00000000ADF4: A00A140A
	v_readlane_b32 s25, v22, 9                                 // 00000000ADF8: D7600019 00011316
	v_readlane_b32 s26, v22, 10                                // 00000000AE00: D760001A 00011516
	v_readlane_b32 s27, v22, 11                                // 00000000AE08: D760001B 00011716
	s_wait_alu 0xfffe                                          // 00000000AE10: BF88FFFE
	s_add_f32 s10, s21, s10                                    // 00000000AE14: A00A0A15
	s_add_f32 s11, s11, s24                                    // 00000000AE18: A00B180B
	s_wait_alu 0xfffe                                          // 00000000AE1C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000AE20: BF870499
	s_add_f32 s10, s22, s10                                    // 00000000AE24: A00A0A16
	s_add_f32 s11, s25, s11                                    // 00000000AE28: A00B0B19
	s_wait_alu 0xfffe                                          // 00000000AE2C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000AE30: BF870499
	s_add_f32 s10, s23, s10                                    // 00000000AE34: A00A0A17
	s_add_f32 s11, s26, s11                                    // 00000000AE38: A00B0B1A
	s_wait_alu 0xfffe                                          // 00000000AE3C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000AE40: BF870499
	s_add_f32 s10, s24, s10                                    // 00000000AE44: A00A0A18
	s_add_f32 s11, s27, s11                                    // 00000000AE48: A00B0B1B
	s_wait_alu 0xfffe                                          // 00000000AE4C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000AE50: BF870529
	s_add_f32 s10, s25, s10                                    // 00000000AE54: A00A0A19
	s_wait_alu 0xfffe                                          // 00000000AE58: BF88FFFE
	s_add_f32 s10, s26, s10                                    // 00000000AE5C: A00A0A1A
	v_readlane_b32 s20, v27, 12                                // 00000000AE60: D7600014 0001191B
	v_readlane_b32 s21, v27, 13                                // 00000000AE68: D7600015 00011B1B
	v_readlane_b32 s22, v27, 14                                // 00000000AE70: D7600016 00011D1B
	v_readlane_b32 s24, v27, 16                                // 00000000AE78: D7600018 0001211B
	v_readlane_b32 s23, v27, 15                                // 00000000AE80: D7600017 00011F1B
	s_add_f32 s3, s3, s20                                      // 00000000AE88: A0031403
	v_readlane_b32 s25, v27, 17                                // 00000000AE8C: D7600019 0001231B
	v_readlane_b32 s26, v27, 18                                // 00000000AE94: D760001A 0001251B
	s_add_f32 s4, s4, s24                                      // 00000000AE9C: A0041804
	s_wait_alu 0xfffe                                          // 00000000AEA0: BF88FFFE
	s_add_f32 s3, s21, s3                                      // 00000000AEA4: A0030315
	v_readlane_b32 s27, v27, 19                                // 00000000AEA8: D760001B 0001271B
	s_add_f32 s4, s25, s4                                      // 00000000AEB0: A0040419
	s_wait_alu 0xfffe                                          // 00000000AEB4: BF88FFFE
	s_add_f32 s3, s22, s3                                      // 00000000AEB8: A0030316
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000AEBC: BF8704A9
	s_add_f32 s4, s26, s4                                      // 00000000AEC0: A004041A
	s_wait_alu 0xfffe                                          // 00000000AEC4: BF88FFFE
	s_add_f32 s3, s23, s3                                      // 00000000AEC8: A0030317
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000AECC: BF8704A9
	s_add_f32 s4, s27, s4                                      // 00000000AED0: A004041B
	s_wait_alu 0xfffe                                          // 00000000AED4: BF88FFFE
	s_add_f32 s3, s24, s3                                      // 00000000AED8: A0030318
	s_wait_alu 0xfffe                                          // 00000000AEDC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000AEE0: BF87052A
	s_add_f32 s3, s25, s3                                      // 00000000AEE4: A0030319
	s_wait_alu 0xfffe                                          // 00000000AEE8: BF88FFFE
	s_add_f32 s3, s26, s3                                      // 00000000AEEC: A003031A
	s_or_saveexec_b32 s105, -1                                 // 00000000AEF0: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000AEF4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000AEF8: BEFE0069
	v_readlane_b32 s20, v28, 20                                // 00000000AEFC: D7600014 0001291C
	v_readlane_b32 s21, v28, 21                                // 00000000AF04: D7600015 00012B1C
	v_readlane_b32 s22, v28, 22                                // 00000000AF0C: D7600016 00012D1C
	v_readlane_b32 s23, v28, 23                                // 00000000AF14: D7600017 00012F1C
	v_readlane_b32 s24, v28, 24                                // 00000000AF1C: D7600018 0001311C
	v_readlane_b32 s25, v28, 25                                // 00000000AF24: D7600019 0001331C
	s_add_f32 s12, s12, s21                                    // 00000000AF2C: A00C150C
	s_add_f32 s7, s7, s20                                      // 00000000AF30: A0071407
	v_readlane_b32 s26, v28, 26                                // 00000000AF34: D760001A 0001351C
	v_readlane_b32 s27, v28, 27                                // 00000000AF3C: D760001B 0001371C
	s_wait_alu 0xfffe                                          // 00000000AF44: BF88FFFE
	s_add_f32 s12, s22, s12                                    // 00000000AF48: A00C0C16
	s_add_f32 s7, s21, s7                                      // 00000000AF4C: A0070715
	s_wait_alu 0xfffe                                          // 00000000AF50: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000AF54: BF870499
	s_add_f32 s12, s23, s12                                    // 00000000AF58: A00C0C17
	s_add_f32 s7, s22, s7                                      // 00000000AF5C: A0070716
	s_wait_alu 0xfffe                                          // 00000000AF60: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000AF64: BF870499
	s_add_f32 s12, s12, s24                                    // 00000000AF68: A00C180C
	s_add_f32 s7, s23, s7                                      // 00000000AF6C: A0070717
	s_wait_alu 0xfffe                                          // 00000000AF70: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000AF74: BF870529
	s_add_f32 s12, s25, s12                                    // 00000000AF78: A00C0C19
	s_wait_alu 0xfffe                                          // 00000000AF7C: BF88FFFE
	s_add_f32 s12, s26, s12                                    // 00000000AF80: A00C0C1A
	s_wait_alu 0xfffe                                          // 00000000AF84: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000AF88: BF87000A
	s_add_f32 s12, s27, s12                                    // 00000000AF8C: A00C0C1B
	s_or_saveexec_b32 s105, -1                                 // 00000000AF90: BEE922C1
	scratch_load_b32 v28, off, off offset:68 th:TH_LOAD_LU     // 00000000AF94: ED05007C 0030001C 00004400
	s_wait_alu 0xfffe                                          // 00000000AFA0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000AFA4: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000AFA8: BFC00000
	v_readlane_b32 s20, v28, 20                                // 00000000AFAC: D7600014 0001291C
	v_readlane_b32 s21, v28, 21                                // 00000000AFB4: D7600015 00012B1C
	v_readlane_b32 s22, v28, 22                                // 00000000AFBC: D7600016 00012D1C
	v_readlane_b32 s23, v28, 23                                // 00000000AFC4: D7600017 00012F1C
	v_readlane_b32 s24, v28, 24                                // 00000000AFCC: D7600018 0001311C
	v_readlane_b32 s25, v28, 25                                // 00000000AFD4: D7600019 0001331C
	v_readlane_b32 s27, v28, 27                                // 00000000AFDC: D760001B 0001371C
	v_readlane_b32 s26, v28, 26                                // 00000000AFE4: D760001A 0001351C
	s_or_saveexec_b32 s105, -1                                 // 00000000AFEC: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000AFF0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000AFF4: BEFE0069
	s_add_f32 s13, s13, s21                                    // 00000000AFF8: A00D150D
	s_add_f32 s9, s9, s20                                      // 00000000AFFC: A0091409
	s_wait_alu 0xfffe                                          // 00000000B000: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B004: BF870499
	s_add_f32 s13, s22, s13                                    // 00000000B008: A00D0D16
	s_add_f32 s9, s21, s9                                      // 00000000B00C: A0090915
	s_wait_alu 0xfffe                                          // 00000000B010: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B014: BF870499
	s_add_f32 s13, s23, s13                                    // 00000000B018: A00D0D17
	s_add_f32 s9, s22, s9                                      // 00000000B01C: A0090916
	s_wait_alu 0xfffe                                          // 00000000B020: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B024: BF870499
	s_add_f32 s13, s13, s24                                    // 00000000B028: A00D180D
	s_add_f32 s9, s23, s9                                      // 00000000B02C: A0090917
	s_wait_alu 0xfffe                                          // 00000000B030: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B034: BF870529
	s_add_f32 s13, s25, s13                                    // 00000000B038: A00D0D19
	s_wait_alu 0xfffe                                          // 00000000B03C: BF88FFFE
	s_add_f32 s13, s26, s13                                    // 00000000B040: A00D0D1A
	s_wait_alu 0xfffe                                          // 00000000B044: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000B048: BF87000A
	s_add_f32 s13, s27, s13                                    // 00000000B04C: A00D0D1B
	s_or_saveexec_b32 s105, -1                                 // 00000000B050: BEE922C1
	scratch_load_b32 v22, off, off offset:64 th:TH_LOAD_LU     // 00000000B054: ED05007C 00300016 00004000
	s_wait_alu 0xfffe                                          // 00000000B060: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B064: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000B068: BFC00000
	v_readlane_b32 s20, v22, 4                                 // 00000000B06C: D7600014 00010916
	v_readlane_b32 s21, v22, 5                                 // 00000000B074: D7600015 00010B16
	v_readlane_b32 s22, v22, 6                                 // 00000000B07C: D7600016 00010D16
	v_readlane_b32 s23, v22, 7                                 // 00000000B084: D7600017 00010F16
	v_readlane_b32 s24, v22, 8                                 // 00000000B08C: D7600018 00011116
	v_readlane_b32 s25, v22, 9                                 // 00000000B094: D7600019 00011316
	s_add_f32 s14, s14, s21                                    // 00000000B09C: A00E150E
	s_add_f32 s11, s11, s20                                    // 00000000B0A0: A00B140B
	v_readlane_b32 s26, v22, 10                                // 00000000B0A4: D760001A 00011516
	v_readlane_b32 s27, v22, 11                                // 00000000B0AC: D760001B 00011716
	s_wait_alu 0xfffe                                          // 00000000B0B4: BF88FFFE
	s_add_f32 s14, s22, s14                                    // 00000000B0B8: A00E0E16
	s_add_f32 s11, s21, s11                                    // 00000000B0BC: A00B0B15
	s_wait_alu 0xfffe                                          // 00000000B0C0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B0C4: BF870499
	s_add_f32 s14, s23, s14                                    // 00000000B0C8: A00E0E17
	s_add_f32 s11, s22, s11                                    // 00000000B0CC: A00B0B16
	s_wait_alu 0xfffe                                          // 00000000B0D0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B0D4: BF870499
	s_add_f32 s14, s14, s24                                    // 00000000B0D8: A00E180E
	s_add_f32 s11, s23, s11                                    // 00000000B0DC: A00B0B17
	s_wait_alu 0xfffe                                          // 00000000B0E0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B0E4: BF870529
	s_add_f32 s14, s25, s14                                    // 00000000B0E8: A00E0E19
	s_wait_alu 0xfffe                                          // 00000000B0EC: BF88FFFE
	s_add_f32 s14, s26, s14                                    // 00000000B0F0: A00E0E1A
	s_wait_alu 0xfffe                                          // 00000000B0F4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000B0F8: BF87000A
	s_add_f32 s14, s27, s14                                    // 00000000B0FC: A00E0E1B
	v_readlane_b32 s20, v28, 12                                // 00000000B100: D7600014 0001191C
	v_readlane_b32 s21, v28, 13                                // 00000000B108: D7600015 00011B1C
	v_readlane_b32 s22, v28, 14                                // 00000000B110: D7600016 00011D1C
	v_readlane_b32 s23, v28, 15                                // 00000000B118: D7600017 00011F1C
	v_readlane_b32 s24, v28, 16                                // 00000000B120: D7600018 0001211C
	s_add_f32 s4, s4, s20                                      // 00000000B128: A0041404
	s_add_f32 s5, s5, s21                                      // 00000000B12C: A0051505
	v_readlane_b32 s25, v28, 17                                // 00000000B130: D7600019 0001231C
	v_readlane_b32 s26, v28, 18                                // 00000000B138: D760001A 0001251C
	s_wait_alu 0xfffe                                          // 00000000B140: BF88FFFE
	s_add_f32 s4, s21, s4                                      // 00000000B144: A0040415
	s_add_f32 s5, s22, s5                                      // 00000000B148: A0050516
	v_readlane_b32 s27, v28, 19                                // 00000000B14C: D760001B 0001271C
	s_wait_alu 0xfffe                                          // 00000000B154: BF88FFFE
	s_add_f32 s4, s22, s4                                      // 00000000B158: A0040416
	s_add_f32 s5, s23, s5                                      // 00000000B15C: A0050517
	s_wait_alu 0xfffe                                          // 00000000B160: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B164: BF870499
	s_add_f32 s4, s23, s4                                      // 00000000B168: A0040417
	s_add_f32 s5, s5, s24                                      // 00000000B16C: A0051805
	s_wait_alu 0xfffe                                          // 00000000B170: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B174: BF87052A
	s_add_f32 s5, s25, s5                                      // 00000000B178: A0050519
	s_wait_alu 0xfffe                                          // 00000000B17C: BF88FFFE
	s_add_f32 s5, s26, s5                                      // 00000000B180: A005051A
	s_wait_alu 0xfffe                                          // 00000000B184: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000B188: BF87000A
	s_add_f32 s5, s27, s5                                      // 00000000B18C: A005051B
	s_or_saveexec_b32 s105, -1                                 // 00000000B190: BEE922C1
	scratch_load_b32 v28, off, off offset:96 th:TH_LOAD_LU     // 00000000B194: ED05007C 0030001C 00006000
	s_wait_alu 0xfffe                                          // 00000000B1A0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B1A4: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000B1A8: BFC00000
	v_readlane_b32 s20, v28, 4                                 // 00000000B1AC: D7600014 0001091C
	v_readlane_b32 s21, v28, 5                                 // 00000000B1B4: D7600015 00010B1C
	v_readlane_b32 s22, v28, 6                                 // 00000000B1BC: D7600016 00010D1C
	v_readlane_b32 s23, v28, 7                                 // 00000000B1C4: D7600017 00010F1C
	v_readlane_b32 s24, v28, 8                                 // 00000000B1CC: D7600018 0001111C
	s_add_f32 s6, s6, s20                                      // 00000000B1D4: A0061406
	v_readlane_b32 s25, v28, 9                                 // 00000000B1D8: D7600019 0001131C
	v_readlane_b32 s26, v28, 10                                // 00000000B1E0: D760001A 0001151C
	v_readlane_b32 s27, v28, 11                                // 00000000B1E8: D760001B 0001171C
	s_wait_alu 0xfffe                                          // 00000000B1F0: BF88FFFE
	s_add_f32 s6, s21, s6                                      // 00000000B1F4: A0060615
	s_add_f32 s7, s7, s24                                      // 00000000B1F8: A0071807
	s_wait_alu 0xfffe                                          // 00000000B1FC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B200: BF870499
	s_add_f32 s6, s22, s6                                      // 00000000B204: A0060616
	s_add_f32 s7, s25, s7                                      // 00000000B208: A0070719
	s_wait_alu 0xfffe                                          // 00000000B20C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B210: BF870499
	s_add_f32 s6, s23, s6                                      // 00000000B214: A0060617
	s_add_f32 s7, s26, s7                                      // 00000000B218: A007071A
	s_wait_alu 0xfffe                                          // 00000000B21C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B220: BF870499
	s_add_f32 s6, s24, s6                                      // 00000000B224: A0060618
	s_add_f32 s7, s27, s7                                      // 00000000B228: A007071B
	s_wait_alu 0xfffe                                          // 00000000B22C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B230: BF870529
	s_add_f32 s6, s25, s6                                      // 00000000B234: A0060619
	s_wait_alu 0xfffe                                          // 00000000B238: BF88FFFE
	s_add_f32 s6, s26, s6                                      // 00000000B23C: A006061A
	s_or_saveexec_b32 s105, -1                                 // 00000000B240: BEE922C1
	scratch_load_b32 v27, off, off offset:28 th:TH_LOAD_LU     // 00000000B244: ED05007C 0030001B 00001C00
	s_wait_alu 0xfffe                                          // 00000000B250: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B254: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000B258: BFC00000
	v_readlane_b32 s20, v27, 28                                // 00000000B25C: D7600014 0001391B
	v_readlane_b32 s21, v27, 29                                // 00000000B264: D7600015 00013B1B
	v_readlane_b32 s22, v27, 30                                // 00000000B26C: D7600016 00013D1B
	v_readlane_b32 s23, v27, 31                                // 00000000B274: D7600017 00013F1B
	s_or_saveexec_b32 s105, -1                                 // 00000000B27C: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000B280: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B284: BEFE0069
	s_add_f32 s8, s8, s20                                      // 00000000B288: A0081408
	v_readlane_b32 s24, v28, 0                                 // 00000000B28C: D7600018 0001011C
	v_readlane_b32 s25, v28, 1                                 // 00000000B294: D7600019 0001031C
	v_readlane_b32 s26, v28, 2                                 // 00000000B29C: D760001A 0001051C
	s_wait_alu 0xfffe                                          // 00000000B2A4: BF88FFFE
	s_add_f32 s8, s21, s8                                      // 00000000B2A8: A0080815
	v_readlane_b32 s27, v28, 3                                 // 00000000B2AC: D760001B 0001071C
	s_add_f32 s9, s9, s24                                      // 00000000B2B4: A0091809
	s_wait_alu 0xfffe                                          // 00000000B2B8: BF88FFFE
	s_add_f32 s8, s22, s8                                      // 00000000B2BC: A0080816
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000B2C0: BF8704A9
	s_add_f32 s9, s25, s9                                      // 00000000B2C4: A0090919
	s_wait_alu 0xfffe                                          // 00000000B2C8: BF88FFFE
	s_add_f32 s8, s23, s8                                      // 00000000B2CC: A0080817
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000B2D0: BF8704A9
	s_add_f32 s9, s26, s9                                      // 00000000B2D4: A009091A
	s_wait_alu 0xfffe                                          // 00000000B2D8: BF88FFFE
	s_add_f32 s8, s24, s8                                      // 00000000B2DC: A0080818
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000B2E0: BF8704A9
	s_add_f32 s9, s27, s9                                      // 00000000B2E4: A009091B
	s_wait_alu 0xfffe                                          // 00000000B2E8: BF88FFFE
	s_add_f32 s8, s25, s8                                      // 00000000B2EC: A0080819
	s_wait_alu 0xfffe                                          // 00000000B2F0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000B2F4: BF87000A
	s_add_f32 s8, s26, s8                                      // 00000000B2F8: A008081A
	v_readlane_b32 s20, v28, 12                                // 00000000B2FC: D7600014 0001191C
	v_readlane_b32 s21, v28, 13                                // 00000000B304: D7600015 00011B1C
	v_readlane_b32 s22, v28, 14                                // 00000000B30C: D7600016 00011D1C
	v_readlane_b32 s23, v28, 15                                // 00000000B314: D7600017 00011F1C
	v_readlane_b32 s24, v28, 16                                // 00000000B31C: D7600018 0001211C
	s_add_f32 s10, s10, s20                                    // 00000000B324: A00A140A
	v_readlane_b32 s25, v28, 17                                // 00000000B328: D7600019 0001231C
	v_readlane_b32 s26, v28, 18                                // 00000000B330: D760001A 0001251C
	v_readlane_b32 s27, v28, 19                                // 00000000B338: D760001B 0001271C
	s_wait_alu 0xfffe                                          // 00000000B340: BF88FFFE
	s_add_f32 s10, s21, s10                                    // 00000000B344: A00A0A15
	s_add_f32 s11, s11, s24                                    // 00000000B348: A00B180B
	s_wait_alu 0xfffe                                          // 00000000B34C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B350: BF870499
	s_add_f32 s10, s22, s10                                    // 00000000B354: A00A0A16
	s_add_f32 s11, s25, s11                                    // 00000000B358: A00B0B19
	s_wait_alu 0xfffe                                          // 00000000B35C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B360: BF870499
	s_add_f32 s10, s23, s10                                    // 00000000B364: A00A0A17
	s_add_f32 s11, s26, s11                                    // 00000000B368: A00B0B1A
	s_wait_alu 0xfffe                                          // 00000000B36C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B370: BF870499
	s_add_f32 s10, s24, s10                                    // 00000000B374: A00A0A18
	s_add_f32 s11, s27, s11                                    // 00000000B378: A00B0B1B
	s_wait_alu 0xfffe                                          // 00000000B37C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B380: BF870529
	s_add_f32 s10, s25, s10                                    // 00000000B384: A00A0A19
	s_wait_alu 0xfffe                                          // 00000000B388: BF88FFFE
	s_add_f32 s10, s26, s10                                    // 00000000B38C: A00A0A1A
	v_readlane_b32 s20, v27, 20                                // 00000000B390: D7600014 0001291B
	v_readlane_b32 s21, v27, 21                                // 00000000B398: D7600015 00012B1B
	v_readlane_b32 s22, v27, 22                                // 00000000B3A0: D7600016 00012D1B
	v_readlane_b32 s24, v27, 24                                // 00000000B3A8: D7600018 0001311B
	v_readlane_b32 s23, v27, 23                                // 00000000B3B0: D7600017 00012F1B
	s_add_f32 s3, s3, s20                                      // 00000000B3B8: A0031403
	v_readlane_b32 s25, v27, 25                                // 00000000B3BC: D7600019 0001331B
	v_readlane_b32 s26, v27, 26                                // 00000000B3C4: D760001A 0001351B
	s_add_f32 s4, s4, s24                                      // 00000000B3CC: A0041804
	s_wait_alu 0xfffe                                          // 00000000B3D0: BF88FFFE
	s_add_f32 s3, s21, s3                                      // 00000000B3D4: A0030315
	v_readlane_b32 s27, v27, 27                                // 00000000B3D8: D760001B 0001371B
	s_add_f32 s4, s25, s4                                      // 00000000B3E0: A0040419
	s_wait_alu 0xfffe                                          // 00000000B3E4: BF88FFFE
	s_add_f32 s3, s22, s3                                      // 00000000B3E8: A0030316
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000B3EC: BF8704A9
	s_add_f32 s4, s26, s4                                      // 00000000B3F0: A004041A
	s_wait_alu 0xfffe                                          // 00000000B3F4: BF88FFFE
	s_add_f32 s3, s23, s3                                      // 00000000B3F8: A0030317
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000B3FC: BF8704A9
	s_add_f32 s4, s27, s4                                      // 00000000B400: A004041B
	s_wait_alu 0xfffe                                          // 00000000B404: BF88FFFE
	s_add_f32 s3, s24, s3                                      // 00000000B408: A0030318
	s_wait_alu 0xfffe                                          // 00000000B40C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B410: BF87052A
	s_add_f32 s3, s25, s3                                      // 00000000B414: A0030319
	s_wait_alu 0xfffe                                          // 00000000B418: BF88FFFE
	s_add_f32 s3, s26, s3                                      // 00000000B41C: A003031A
	s_or_saveexec_b32 s105, -1                                 // 00000000B420: BEE922C1
	scratch_load_b32 v28, off, off offset:24 th:TH_LOAD_LU     // 00000000B424: ED05007C 0030001C 00001800
	s_wait_alu 0xfffe                                          // 00000000B430: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B434: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000B438: BFC00000
	v_readlane_b32 s20, v28, 28                                // 00000000B43C: D7600014 0001391C
	v_readlane_b32 s21, v28, 29                                // 00000000B444: D7600015 00013B1C
	v_readlane_b32 s22, v28, 30                                // 00000000B44C: D7600016 00013D1C
	v_readlane_b32 s23, v28, 31                                // 00000000B454: D7600017 00013F1C
	v_readlane_b32 s24, v27, 0                                 // 00000000B45C: D7600018 0001011B
	v_readlane_b32 s25, v27, 1                                 // 00000000B464: D7600019 0001031B
	s_add_f32 s12, s12, s21                                    // 00000000B46C: A00C150C
	s_add_f32 s7, s7, s20                                      // 00000000B470: A0071407
	v_readlane_b32 s26, v27, 2                                 // 00000000B474: D760001A 0001051B
	v_readlane_b32 s27, v27, 3                                 // 00000000B47C: D760001B 0001071B
	s_wait_alu 0xfffe                                          // 00000000B484: BF88FFFE
	s_add_f32 s12, s22, s12                                    // 00000000B488: A00C0C16
	s_add_f32 s7, s21, s7                                      // 00000000B48C: A0070715
	s_wait_alu 0xfffe                                          // 00000000B490: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B494: BF870499
	s_add_f32 s12, s23, s12                                    // 00000000B498: A00C0C17
	s_add_f32 s7, s22, s7                                      // 00000000B49C: A0070716
	s_wait_alu 0xfffe                                          // 00000000B4A0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B4A4: BF870499
	s_add_f32 s12, s12, s24                                    // 00000000B4A8: A00C180C
	s_add_f32 s7, s23, s7                                      // 00000000B4AC: A0070717
	s_wait_alu 0xfffe                                          // 00000000B4B0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B4B4: BF870529
	s_add_f32 s12, s25, s12                                    // 00000000B4B8: A00C0C19
	s_wait_alu 0xfffe                                          // 00000000B4BC: BF88FFFE
	s_add_f32 s12, s26, s12                                    // 00000000B4C0: A00C0C1A
	s_wait_alu 0xfffe                                          // 00000000B4C4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000B4C8: BF87000A
	s_add_f32 s12, s27, s12                                    // 00000000B4CC: A00C0C1B
	s_or_saveexec_b32 s105, -1                                 // 00000000B4D0: BEE922C1
	scratch_load_b32 v28, off, off offset:76 th:TH_LOAD_LU     // 00000000B4D4: ED05007C 0030001C 00004C00
	s_wait_alu 0xfffe                                          // 00000000B4E0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B4E4: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000B4E8: BFC00000
	v_readlane_b32 s20, v28, 12                                // 00000000B4EC: D7600014 0001191C
	v_readlane_b32 s21, v28, 13                                // 00000000B4F4: D7600015 00011B1C
	v_readlane_b32 s22, v28, 14                                // 00000000B4FC: D7600016 00011D1C
	v_readlane_b32 s23, v28, 15                                // 00000000B504: D7600017 00011F1C
	v_readlane_b32 s24, v28, 16                                // 00000000B50C: D7600018 0001211C
	v_readlane_b32 s25, v28, 17                                // 00000000B514: D7600019 0001231C
	v_readlane_b32 s27, v28, 19                                // 00000000B51C: D760001B 0001271C
	v_readlane_b32 s26, v28, 18                                // 00000000B524: D760001A 0001251C
	s_or_saveexec_b32 s105, -1                                 // 00000000B52C: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000B530: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B534: BEFE0069
	s_add_f32 s13, s13, s21                                    // 00000000B538: A00D150D
	s_add_f32 s9, s9, s20                                      // 00000000B53C: A0091409
	s_wait_alu 0xfffe                                          // 00000000B540: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B544: BF870499
	s_add_f32 s13, s22, s13                                    // 00000000B548: A00D0D16
	s_add_f32 s9, s21, s9                                      // 00000000B54C: A0090915
	s_wait_alu 0xfffe                                          // 00000000B550: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B554: BF870499
	s_add_f32 s13, s23, s13                                    // 00000000B558: A00D0D17
	s_add_f32 s9, s22, s9                                      // 00000000B55C: A0090916
	s_wait_alu 0xfffe                                          // 00000000B560: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B564: BF870499
	s_add_f32 s13, s13, s24                                    // 00000000B568: A00D180D
	s_add_f32 s9, s23, s9                                      // 00000000B56C: A0090917
	s_wait_alu 0xfffe                                          // 00000000B570: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B574: BF870529
	s_add_f32 s13, s25, s13                                    // 00000000B578: A00D0D19
	s_wait_alu 0xfffe                                          // 00000000B57C: BF88FFFE
	s_add_f32 s13, s26, s13                                    // 00000000B580: A00D0D1A
	s_wait_alu 0xfffe                                          // 00000000B584: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000B588: BF87000A
	s_add_f32 s13, s27, s13                                    // 00000000B58C: A00D0D1B
	v_readlane_b32 s20, v25, 28                                // 00000000B590: D7600014 00013919
	v_readlane_b32 s21, v25, 29                                // 00000000B598: D7600015 00013B19
	v_readlane_b32 s22, v25, 30                                // 00000000B5A0: D7600016 00013D19
	v_readlane_b32 s23, v25, 31                                // 00000000B5A8: D7600017 00013F19
	v_readlane_b32 s24, v29, 0                                 // 00000000B5B0: D7600018 0001011D
	v_readlane_b32 s25, v29, 1                                 // 00000000B5B8: D7600019 0001031D
	s_add_f32 s14, s14, s21                                    // 00000000B5C0: A00E150E
	s_add_f32 s11, s11, s20                                    // 00000000B5C4: A00B140B
	v_readlane_b32 s26, v29, 2                                 // 00000000B5C8: D760001A 0001051D
	v_readlane_b32 s27, v29, 3                                 // 00000000B5D0: D760001B 0001071D
	s_wait_alu 0xfffe                                          // 00000000B5D8: BF88FFFE
	s_add_f32 s14, s22, s14                                    // 00000000B5DC: A00E0E16
	s_add_f32 s11, s21, s11                                    // 00000000B5E0: A00B0B15
	s_wait_alu 0xfffe                                          // 00000000B5E4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B5E8: BF870499
	s_add_f32 s14, s23, s14                                    // 00000000B5EC: A00E0E17
	s_add_f32 s11, s22, s11                                    // 00000000B5F0: A00B0B16
	s_wait_alu 0xfffe                                          // 00000000B5F4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B5F8: BF870499
	s_add_f32 s14, s14, s24                                    // 00000000B5FC: A00E180E
	s_add_f32 s11, s23, s11                                    // 00000000B600: A00B0B17
	s_wait_alu 0xfffe                                          // 00000000B604: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B608: BF870529
	s_add_f32 s14, s25, s14                                    // 00000000B60C: A00E0E19
	s_wait_alu 0xfffe                                          // 00000000B610: BF88FFFE
	s_add_f32 s14, s26, s14                                    // 00000000B614: A00E0E1A
	s_wait_alu 0xfffe                                          // 00000000B618: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000B61C: BF87000A
	s_add_f32 s14, s27, s14                                    // 00000000B620: A00E0E1B
	v_readlane_b32 s20, v28, 4                                 // 00000000B624: D7600014 0001091C
	v_readlane_b32 s21, v28, 5                                 // 00000000B62C: D7600015 00010B1C
	v_readlane_b32 s22, v28, 6                                 // 00000000B634: D7600016 00010D1C
	v_readlane_b32 s23, v28, 7                                 // 00000000B63C: D7600017 00010F1C
	v_readlane_b32 s24, v28, 8                                 // 00000000B644: D7600018 0001111C
	s_add_f32 s4, s4, s20                                      // 00000000B64C: A0041404
	s_add_f32 s5, s5, s21                                      // 00000000B650: A0051505
	v_readlane_b32 s25, v28, 9                                 // 00000000B654: D7600019 0001131C
	v_readlane_b32 s26, v28, 10                                // 00000000B65C: D760001A 0001151C
	s_wait_alu 0xfffe                                          // 00000000B664: BF88FFFE
	s_add_f32 s4, s21, s4                                      // 00000000B668: A0040415
	s_add_f32 s5, s22, s5                                      // 00000000B66C: A0050516
	v_readlane_b32 s27, v28, 11                                // 00000000B670: D760001B 0001171C
	s_wait_alu 0xfffe                                          // 00000000B678: BF88FFFE
	s_add_f32 s4, s22, s4                                      // 00000000B67C: A0040416
	s_add_f32 s5, s23, s5                                      // 00000000B680: A0050517
	s_wait_alu 0xfffe                                          // 00000000B684: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B688: BF870499
	s_add_f32 s4, s23, s4                                      // 00000000B68C: A0040417
	s_add_f32 s5, s5, s24                                      // 00000000B690: A0051805
	s_wait_alu 0xfffe                                          // 00000000B694: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B698: BF87052A
	s_add_f32 s5, s25, s5                                      // 00000000B69C: A0050519
	s_wait_alu 0xfffe                                          // 00000000B6A0: BF88FFFE
	s_add_f32 s5, s26, s5                                      // 00000000B6A4: A005051A
	s_wait_alu 0xfffe                                          // 00000000B6A8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000B6AC: BF87000A
	s_add_f32 s5, s27, s5                                      // 00000000B6B0: A005051B
	s_or_saveexec_b32 s105, -1                                 // 00000000B6B4: BEE922C1
	scratch_load_b32 v28, off, off offset:36 th:TH_LOAD_LU     // 00000000B6B8: ED05007C 0030001C 00002400
	s_wait_alu 0xfffe                                          // 00000000B6C4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B6C8: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000B6CC: BFC00000
	v_readlane_b32 s20, v28, 28                                // 00000000B6D0: D7600014 0001391C
	v_readlane_b32 s21, v28, 29                                // 00000000B6D8: D7600015 00013B1C
	v_readlane_b32 s22, v28, 30                                // 00000000B6E0: D7600016 00013D1C
	v_readlane_b32 s23, v28, 31                                // 00000000B6E8: D7600017 00013F1C
	s_or_saveexec_b32 s105, -1                                 // 00000000B6F0: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000B6F4: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B6F8: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 00000000B6FC: BEE922C1
	scratch_load_b32 v27, off, off offset:72 th:TH_LOAD_LU     // 00000000B700: ED05007C 0030001B 00004800
	s_wait_alu 0xfffe                                          // 00000000B70C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B710: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000B714: BFC00000
	v_readlane_b32 s24, v27, 0                                 // 00000000B718: D7600018 0001011B
	v_readlane_b32 s25, v27, 1                                 // 00000000B720: D7600019 0001031B
	v_readlane_b32 s27, v27, 3                                 // 00000000B728: D760001B 0001071B
	v_readlane_b32 s26, v27, 2                                 // 00000000B730: D760001A 0001051B
	s_or_saveexec_b32 s105, -1                                 // 00000000B738: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000B73C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B740: BEFE0069
	s_add_f32 s6, s6, s20                                      // 00000000B744: A0061406
	s_add_f32 s7, s7, s24                                      // 00000000B748: A0071807
	s_wait_alu 0xfffe                                          // 00000000B74C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B750: BF870499
	s_add_f32 s6, s21, s6                                      // 00000000B754: A0060615
	s_add_f32 s7, s25, s7                                      // 00000000B758: A0070719
	s_wait_alu 0xfffe                                          // 00000000B75C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B760: BF870499
	s_add_f32 s6, s22, s6                                      // 00000000B764: A0060616
	s_add_f32 s7, s26, s7                                      // 00000000B768: A007071A
	s_wait_alu 0xfffe                                          // 00000000B76C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B770: BF870499
	s_add_f32 s6, s23, s6                                      // 00000000B774: A0060617
	s_add_f32 s7, s27, s7                                      // 00000000B778: A007071B
	s_wait_alu 0xfffe                                          // 00000000B77C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B780: BF870529
	s_add_f32 s6, s24, s6                                      // 00000000B784: A0060618
	s_wait_alu 0xfffe                                          // 00000000B788: BF88FFFE
	s_add_f32 s6, s25, s6                                      // 00000000B78C: A0060619
	s_wait_alu 0xfffe                                          // 00000000B790: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000B794: BF87000A
	s_add_f32 s6, s26, s6                                      // 00000000B798: A006061A
	s_or_saveexec_b32 s105, -1                                 // 00000000B79C: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000B7A0: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B7A4: BEFE0069
	v_readlane_b32 s20, v28, 20                                // 00000000B7A8: D7600014 0001291C
	v_readlane_b32 s21, v28, 21                                // 00000000B7B0: D7600015 00012B1C
	v_readlane_b32 s22, v28, 22                                // 00000000B7B8: D7600016 00012D1C
	v_readlane_b32 s23, v28, 23                                // 00000000B7C0: D7600017 00012F1C
	v_readlane_b32 s24, v28, 24                                // 00000000B7C8: D7600018 0001311C
	v_readlane_b32 s25, v28, 25                                // 00000000B7D0: D7600019 0001331C
	v_readlane_b32 s27, v28, 27                                // 00000000B7D8: D760001B 0001371C
	v_readlane_b32 s26, v28, 26                                // 00000000B7E0: D760001A 0001351C
	s_or_saveexec_b32 s105, -1                                 // 00000000B7E8: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000B7EC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B7F0: BEFE0069
	s_add_f32 s8, s8, s20                                      // 00000000B7F4: A0081408
	s_add_f32 s9, s9, s24                                      // 00000000B7F8: A0091809
	s_wait_alu 0xfffe                                          // 00000000B7FC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B800: BF870499
	s_add_f32 s8, s21, s8                                      // 00000000B804: A0080815
	s_add_f32 s9, s25, s9                                      // 00000000B808: A0090919
	s_wait_alu 0xfffe                                          // 00000000B80C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B810: BF870499
	s_add_f32 s8, s22, s8                                      // 00000000B814: A0080816
	s_add_f32 s9, s26, s9                                      // 00000000B818: A009091A
	s_wait_alu 0xfffe                                          // 00000000B81C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B820: BF870499
	s_add_f32 s8, s23, s8                                      // 00000000B824: A0080817
	s_add_f32 s9, s27, s9                                      // 00000000B828: A009091B
	s_wait_alu 0xfffe                                          // 00000000B82C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B830: BF870529
	s_add_f32 s8, s24, s8                                      // 00000000B834: A0080818
	s_wait_alu 0xfffe                                          // 00000000B838: BF88FFFE
	s_add_f32 s8, s25, s8                                      // 00000000B83C: A0080819
	s_wait_alu 0xfffe                                          // 00000000B840: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000B844: BF87000A
	s_add_f32 s8, s26, s8                                      // 00000000B848: A008081A
	s_or_saveexec_b32 s105, -1                                 // 00000000B84C: BEE922C1
	scratch_load_b32 v22, off, off offset:100 th:TH_LOAD_LU    // 00000000B850: ED05007C 00300016 00006400
	s_wait_alu 0xfffe                                          // 00000000B85C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B860: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000B864: BFC00000
	v_readlane_b32 s20, v22, 12                                // 00000000B868: D7600014 00011916
	v_readlane_b32 s21, v22, 13                                // 00000000B870: D7600015 00011B16
	v_readlane_b32 s22, v22, 14                                // 00000000B878: D7600016 00011D16
	v_readlane_b32 s23, v22, 15                                // 00000000B880: D7600017 00011F16
	v_readlane_b32 s24, v22, 16                                // 00000000B888: D7600018 00012116
	v_readlane_b32 s25, v22, 17                                // 00000000B890: D7600019 00012316
	v_readlane_b32 s27, v22, 19                                // 00000000B898: D760001B 00012716
	v_readlane_b32 s26, v22, 18                                // 00000000B8A0: D760001A 00012516
	s_or_saveexec_b32 s105, -1                                 // 00000000B8A8: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000B8AC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B8B0: BEFE0069
	s_add_f32 s10, s10, s20                                    // 00000000B8B4: A00A140A
	s_add_f32 s11, s11, s24                                    // 00000000B8B8: A00B180B
	s_wait_alu 0xfffe                                          // 00000000B8BC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B8C0: BF870499
	s_add_f32 s10, s21, s10                                    // 00000000B8C4: A00A0A15
	s_add_f32 s11, s25, s11                                    // 00000000B8C8: A00B0B19
	s_wait_alu 0xfffe                                          // 00000000B8CC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B8D0: BF870499
	s_add_f32 s10, s22, s10                                    // 00000000B8D4: A00A0A16
	s_add_f32 s11, s26, s11                                    // 00000000B8D8: A00B0B1A
	s_wait_alu 0xfffe                                          // 00000000B8DC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000B8E0: BF870499
	s_add_f32 s10, s23, s10                                    // 00000000B8E4: A00A0A17
	s_add_f32 s11, s27, s11                                    // 00000000B8E8: A00B0B1B
	s_wait_alu 0xfffe                                          // 00000000B8EC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B8F0: BF870529
	s_add_f32 s10, s24, s10                                    // 00000000B8F4: A00A0A18
	s_wait_alu 0xfffe                                          // 00000000B8F8: BF88FFFE
	s_add_f32 s10, s25, s10                                    // 00000000B8FC: A00A0A19
	s_wait_alu 0xfffe                                          // 00000000B900: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000B904: BF87000A
	s_add_f32 s10, s26, s10                                    // 00000000B908: A00A0A1A
	v_readlane_b32 s20, v28, 12                                // 00000000B90C: D7600014 0001191C
	v_readlane_b32 s21, v28, 13                                // 00000000B914: D7600015 00011B1C
	v_readlane_b32 s22, v28, 14                                // 00000000B91C: D7600016 00011D1C
	v_readlane_b32 s24, v28, 16                                // 00000000B924: D7600018 0001211C
	v_readlane_b32 s23, v28, 15                                // 00000000B92C: D7600017 00011F1C
	s_add_f32 s3, s3, s20                                      // 00000000B934: A0031403
	v_readlane_b32 s25, v28, 17                                // 00000000B938: D7600019 0001231C
	v_readlane_b32 s26, v28, 18                                // 00000000B940: D760001A 0001251C
	s_add_f32 s4, s4, s24                                      // 00000000B948: A0041804
	s_wait_alu 0xfffe                                          // 00000000B94C: BF88FFFE
	s_add_f32 s3, s21, s3                                      // 00000000B950: A0030315
	v_readlane_b32 s27, v28, 19                                // 00000000B954: D760001B 0001271C
	s_add_f32 s4, s25, s4                                      // 00000000B95C: A0040419
	s_wait_alu 0xfffe                                          // 00000000B960: BF88FFFE
	s_add_f32 s3, s22, s3                                      // 00000000B964: A0030316
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000B968: BF8704A9
	s_add_f32 s4, s26, s4                                      // 00000000B96C: A004041A
	s_wait_alu 0xfffe                                          // 00000000B970: BF88FFFE
	s_add_f32 s3, s23, s3                                      // 00000000B974: A0030317
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000B978: BF8704A9
	s_add_f32 s4, s27, s4                                      // 00000000B97C: A004041B
	s_wait_alu 0xfffe                                          // 00000000B980: BF88FFFE
	s_add_f32 s3, s24, s3                                      // 00000000B984: A0030318
	s_wait_alu 0xfffe                                          // 00000000B988: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000B98C: BF87052A
	s_add_f32 s3, s25, s3                                      // 00000000B990: A0030319
	s_wait_alu 0xfffe                                          // 00000000B994: BF88FFFE
	s_add_f32 s3, s26, s3                                      // 00000000B998: A003031A
	s_or_saveexec_b32 s105, -1                                 // 00000000B99C: BEE922C1
	scratch_load_b32 v28, off, off offset:32 th:TH_LOAD_LU     // 00000000B9A0: ED05007C 0030001C 00002000
	s_wait_alu 0xfffe                                          // 00000000B9AC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000B9B0: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000B9B4: BFC00000
	v_readlane_b32 s20, v28, 20                                // 00000000B9B8: D7600014 0001291C
	v_readlane_b32 s21, v28, 21                                // 00000000B9C0: D7600015 00012B1C
	v_readlane_b32 s22, v28, 22                                // 00000000B9C8: D7600016 00012D1C
	v_readlane_b32 s23, v28, 23                                // 00000000B9D0: D7600017 00012F1C
	v_readlane_b32 s24, v28, 24                                // 00000000B9D8: D7600018 0001311C
	v_readlane_b32 s25, v28, 25                                // 00000000B9E0: D7600019 0001331C
	s_add_f32 s12, s12, s21                                    // 00000000B9E8: A00C150C
	s_add_f32 s7, s7, s20                                      // 00000000B9EC: A0071407
	v_readlane_b32 s26, v28, 26                                // 00000000B9F0: D760001A 0001351C
	v_readlane_b32 s27, v28, 27                                // 00000000B9F8: D760001B 0001371C
	s_wait_alu 0xfffe                                          // 00000000BA00: BF88FFFE
	s_add_f32 s12, s22, s12                                    // 00000000BA04: A00C0C16
	s_add_f32 s7, s21, s7                                      // 00000000BA08: A0070715
	s_wait_alu 0xfffe                                          // 00000000BA0C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BA10: BF870499
	s_add_f32 s12, s23, s12                                    // 00000000BA14: A00C0C17
	s_add_f32 s7, s22, s7                                      // 00000000BA18: A0070716
	s_wait_alu 0xfffe                                          // 00000000BA1C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BA20: BF870499
	s_add_f32 s12, s12, s24                                    // 00000000BA24: A00C180C
	s_add_f32 s7, s23, s7                                      // 00000000BA28: A0070717
	s_wait_alu 0xfffe                                          // 00000000BA2C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000BA30: BF870529
	s_add_f32 s12, s25, s12                                    // 00000000BA34: A00C0C19
	s_wait_alu 0xfffe                                          // 00000000BA38: BF88FFFE
	s_add_f32 s12, s26, s12                                    // 00000000BA3C: A00C0C1A
	s_wait_alu 0xfffe                                          // 00000000BA40: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000BA44: BF87000A
	s_add_f32 s12, s27, s12                                    // 00000000BA48: A00C0C1B
	s_or_saveexec_b32 s105, -1                                 // 00000000BA4C: BEE922C1
	scratch_load_b32 v24, off, off offset:80 th:TH_LOAD_LU     // 00000000BA50: ED05007C 00300018 00005000
	s_wait_alu 0xfffe                                          // 00000000BA5C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000BA60: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000BA64: BFC00000
	v_readlane_b32 s20, v24, 20                                // 00000000BA68: D7600014 00012918
	v_readlane_b32 s21, v24, 21                                // 00000000BA70: D7600015 00012B18
	v_readlane_b32 s22, v24, 22                                // 00000000BA78: D7600016 00012D18
	v_readlane_b32 s23, v24, 23                                // 00000000BA80: D7600017 00012F18
	v_readlane_b32 s24, v24, 24                                // 00000000BA88: D7600018 00013118
	v_readlane_b32 s25, v24, 25                                // 00000000BA90: D7600019 00013318
	v_readlane_b32 s27, v24, 27                                // 00000000BA98: D760001B 00013718
	v_readlane_b32 s26, v24, 26                                // 00000000BAA0: D760001A 00013518
	s_or_saveexec_b32 s105, -1                                 // 00000000BAA8: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000BAAC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000BAB0: BEFE0069
	s_add_f32 s13, s13, s21                                    // 00000000BAB4: A00D150D
	s_add_f32 s9, s9, s20                                      // 00000000BAB8: A0091409
	s_wait_alu 0xfffe                                          // 00000000BABC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BAC0: BF870499
	s_add_f32 s13, s22, s13                                    // 00000000BAC4: A00D0D16
	s_add_f32 s9, s21, s9                                      // 00000000BAC8: A0090915
	s_wait_alu 0xfffe                                          // 00000000BACC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BAD0: BF870499
	s_add_f32 s13, s23, s13                                    // 00000000BAD4: A00D0D17
	s_add_f32 s9, s22, s9                                      // 00000000BAD8: A0090916
	s_wait_alu 0xfffe                                          // 00000000BADC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BAE0: BF870499
	s_add_f32 s13, s13, s24                                    // 00000000BAE4: A00D180D
	s_add_f32 s9, s23, s9                                      // 00000000BAE8: A0090917
	s_wait_alu 0xfffe                                          // 00000000BAEC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000BAF0: BF870529
	s_add_f32 s13, s25, s13                                    // 00000000BAF4: A00D0D19
	s_wait_alu 0xfffe                                          // 00000000BAF8: BF88FFFE
	s_add_f32 s13, s26, s13                                    // 00000000BAFC: A00D0D1A
	s_wait_alu 0xfffe                                          // 00000000BB00: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 00000000BB04: BF8700DA
	s_add_f32 s13, s27, s13                                    // 00000000BB08: A00D0D1B
	s_or_saveexec_b32 s105, -1                                 // 00000000BB0C: BEE922C1
	v_mov_b32_e32 v28, v27                                     // 00000000BB10: 7E38031B
	s_wait_alu 0xfffe                                          // 00000000BB14: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000BB18: BEFE0069
	v_readlane_b32 s20, v28, 4                                 // 00000000BB1C: D7600014 0001091C
	v_readlane_b32 s21, v28, 5                                 // 00000000BB24: D7600015 00010B1C
	v_readlane_b32 s22, v28, 6                                 // 00000000BB2C: D7600016 00010D1C
	v_readlane_b32 s23, v28, 7                                 // 00000000BB34: D7600017 00010F1C
	v_readlane_b32 s24, v28, 8                                 // 00000000BB3C: D7600018 0001111C
	v_readlane_b32 s25, v28, 9                                 // 00000000BB44: D7600019 0001131C
	s_add_f32 s14, s14, s21                                    // 00000000BB4C: A00E150E
	s_add_f32 s11, s11, s20                                    // 00000000BB50: A00B140B
	v_readlane_b32 s26, v28, 10                                // 00000000BB54: D760001A 0001151C
	v_readlane_b32 s27, v28, 11                                // 00000000BB5C: D760001B 0001171C
	s_wait_alu 0xfffe                                          // 00000000BB64: BF88FFFE
	s_add_f32 s14, s22, s14                                    // 00000000BB68: A00E0E16
	s_add_f32 s11, s21, s11                                    // 00000000BB6C: A00B0B15
	s_wait_alu 0xfffe                                          // 00000000BB70: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BB74: BF870499
	s_add_f32 s14, s23, s14                                    // 00000000BB78: A00E0E17
	s_add_f32 s11, s22, s11                                    // 00000000BB7C: A00B0B16
	s_wait_alu 0xfffe                                          // 00000000BB80: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BB84: BF870499
	s_add_f32 s14, s14, s24                                    // 00000000BB88: A00E180E
	s_add_f32 s11, s23, s11                                    // 00000000BB8C: A00B0B17
	s_wait_alu 0xfffe                                          // 00000000BB90: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000BB94: BF870529
	s_add_f32 s14, s25, s14                                    // 00000000BB98: A00E0E19
	s_wait_alu 0xfffe                                          // 00000000BB9C: BF88FFFE
	s_add_f32 s14, s26, s14                                    // 00000000BBA0: A00E0E1A
	s_wait_alu 0xfffe                                          // 00000000BBA4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000BBA8: BF87000A
	s_add_f32 s14, s27, s14                                    // 00000000BBAC: A00E0E1B
	v_readlane_b32 s20, v24, 12                                // 00000000BBB0: D7600014 00011918
	v_readlane_b32 s21, v24, 13                                // 00000000BBB8: D7600015 00011B18
	v_readlane_b32 s22, v24, 14                                // 00000000BBC0: D7600016 00011D18
	v_readlane_b32 s23, v24, 15                                // 00000000BBC8: D7600017 00011F18
	v_readlane_b32 s24, v24, 16                                // 00000000BBD0: D7600018 00012118
	s_add_f32 s4, s4, s20                                      // 00000000BBD8: A0041404
	s_add_f32 s5, s5, s21                                      // 00000000BBDC: A0051505
	v_readlane_b32 s25, v24, 17                                // 00000000BBE0: D7600019 00012318
	v_readlane_b32 s26, v24, 18                                // 00000000BBE8: D760001A 00012518
	s_wait_alu 0xfffe                                          // 00000000BBF0: BF88FFFE
	s_add_f32 s4, s21, s4                                      // 00000000BBF4: A0040415
	s_add_f32 s5, s22, s5                                      // 00000000BBF8: A0050516
	v_readlane_b32 s27, v24, 19                                // 00000000BBFC: D760001B 00012718
	s_wait_alu 0xfffe                                          // 00000000BC04: BF88FFFE
	s_add_f32 s4, s22, s4                                      // 00000000BC08: A0040416
	s_add_f32 s5, s23, s5                                      // 00000000BC0C: A0050517
	s_wait_alu 0xfffe                                          // 00000000BC10: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BC14: BF870499
	s_add_f32 s4, s23, s4                                      // 00000000BC18: A0040417
	s_add_f32 s5, s5, s24                                      // 00000000BC1C: A0051805
	s_wait_alu 0xfffe                                          // 00000000BC20: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000BC24: BF87052A
	s_add_f32 s5, s25, s5                                      // 00000000BC28: A0050519
	s_wait_alu 0xfffe                                          // 00000000BC2C: BF88FFFE
	s_add_f32 s5, s26, s5                                      // 00000000BC30: A005051A
	s_wait_alu 0xfffe                                          // 00000000BC34: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 00000000BC38: BF8700DA
	s_add_f32 s5, s27, s5                                      // 00000000BC3C: A005051B
	s_or_saveexec_b32 s105, -1                                 // 00000000BC40: BEE922C1
	v_mov_b32_e32 v28, v22                                     // 00000000BC44: 7E380316
	s_wait_alu 0xfffe                                          // 00000000BC48: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000BC4C: BEFE0069
	v_readlane_b32 s20, v28, 4                                 // 00000000BC50: D7600014 0001091C
	v_readlane_b32 s21, v28, 5                                 // 00000000BC58: D7600015 00010B1C
	v_readlane_b32 s22, v28, 6                                 // 00000000BC60: D7600016 00010D1C
	v_readlane_b32 s23, v28, 7                                 // 00000000BC68: D7600017 00010F1C
	v_readlane_b32 s24, v28, 8                                 // 00000000BC70: D7600018 0001111C
	v_readlane_b32 s25, v28, 9                                 // 00000000BC78: D7600019 0001131C
	v_readlane_b32 s27, v28, 11                                // 00000000BC80: D760001B 0001171C
	v_readlane_b32 s26, v28, 10                                // 00000000BC88: D760001A 0001151C
	s_or_saveexec_b32 s105, -1                                 // 00000000BC90: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000BC94: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000BC98: BEFE0069
	s_add_f32 s6, s6, s20                                      // 00000000BC9C: A0061406
	s_add_f32 s7, s7, s24                                      // 00000000BCA0: A0071807
	s_wait_alu 0xfffe                                          // 00000000BCA4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BCA8: BF870499
	s_add_f32 s6, s21, s6                                      // 00000000BCAC: A0060615
	s_add_f32 s7, s25, s7                                      // 00000000BCB0: A0070719
	s_wait_alu 0xfffe                                          // 00000000BCB4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BCB8: BF870499
	s_add_f32 s6, s22, s6                                      // 00000000BCBC: A0060616
	s_add_f32 s7, s26, s7                                      // 00000000BCC0: A007071A
	s_wait_alu 0xfffe                                          // 00000000BCC4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BCC8: BF870499
	s_add_f32 s6, s23, s6                                      // 00000000BCCC: A0060617
	s_add_f32 s7, s27, s7                                      // 00000000BCD0: A007071B
	s_wait_alu 0xfffe                                          // 00000000BCD4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000BCD8: BF870529
	s_add_f32 s6, s24, s6                                      // 00000000BCDC: A0060618
	s_wait_alu 0xfffe                                          // 00000000BCE0: BF88FFFE
	s_add_f32 s6, s25, s6                                      // 00000000BCE4: A0060619
	s_wait_alu 0xfffe                                          // 00000000BCE8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000BCEC: BF87000A
	s_add_f32 s6, s26, s6                                      // 00000000BCF0: A006061A
	v_readlane_b32 s20, v26, 28                                // 00000000BCF4: D7600014 0001391A
	v_readlane_b32 s21, v26, 29                                // 00000000BCFC: D7600015 00013B1A
	v_readlane_b32 s22, v26, 30                                // 00000000BD04: D7600016 00013D1A
	v_readlane_b32 s23, v26, 31                                // 00000000BD0C: D7600017 00013F1A
	v_readlane_b32 s24, v28, 0                                 // 00000000BD14: D7600018 0001011C
	s_add_f32 s8, s8, s20                                      // 00000000BD1C: A0081408
	v_readlane_b32 s25, v28, 1                                 // 00000000BD20: D7600019 0001031C
	v_readlane_b32 s26, v28, 2                                 // 00000000BD28: D760001A 0001051C
	v_readlane_b32 s27, v28, 3                                 // 00000000BD30: D760001B 0001071C
	s_wait_alu 0xfffe                                          // 00000000BD38: BF88FFFE
	s_add_f32 s8, s21, s8                                      // 00000000BD3C: A0080815
	s_add_f32 s9, s9, s24                                      // 00000000BD40: A0091809
	s_wait_alu 0xfffe                                          // 00000000BD44: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BD48: BF870499
	s_add_f32 s8, s22, s8                                      // 00000000BD4C: A0080816
	s_add_f32 s9, s25, s9                                      // 00000000BD50: A0090919
	s_wait_alu 0xfffe                                          // 00000000BD54: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BD58: BF870499
	s_add_f32 s8, s23, s8                                      // 00000000BD5C: A0080817
	s_add_f32 s9, s26, s9                                      // 00000000BD60: A009091A
	s_wait_alu 0xfffe                                          // 00000000BD64: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BD68: BF870499
	s_add_f32 s8, s24, s8                                      // 00000000BD6C: A0080818
	s_add_f32 s9, s27, s9                                      // 00000000BD70: A009091B
	s_wait_alu 0xfffe                                          // 00000000BD74: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000BD78: BF870529
	s_add_f32 s8, s25, s8                                      // 00000000BD7C: A0080819
	s_wait_alu 0xfffe                                          // 00000000BD80: BF88FFFE
	s_add_f32 s8, s26, s8                                      // 00000000BD84: A008081A
	v_readlane_b32 s20, v29, 4                                 // 00000000BD88: D7600014 0001091D
	v_readlane_b32 s21, v29, 5                                 // 00000000BD90: D7600015 00010B1D
	v_readlane_b32 s22, v29, 6                                 // 00000000BD98: D7600016 00010D1D
	v_readlane_b32 s23, v29, 7                                 // 00000000BDA0: D7600017 00010F1D
	v_readlane_b32 s24, v29, 8                                 // 00000000BDA8: D7600018 0001111D
	s_add_f32 s10, s10, s20                                    // 00000000BDB0: A00A140A
	v_readlane_b32 s25, v29, 9                                 // 00000000BDB4: D7600019 0001131D
	v_readlane_b32 s26, v29, 10                                // 00000000BDBC: D760001A 0001151D
	v_readlane_b32 s27, v29, 11                                // 00000000BDC4: D760001B 0001171D
	s_wait_alu 0xfffe                                          // 00000000BDCC: BF88FFFE
	s_add_f32 s10, s21, s10                                    // 00000000BDD0: A00A0A15
	s_add_f32 s11, s11, s24                                    // 00000000BDD4: A00B180B
	s_wait_alu 0xfffe                                          // 00000000BDD8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BDDC: BF870499
	s_add_f32 s10, s22, s10                                    // 00000000BDE0: A00A0A16
	s_add_f32 s11, s25, s11                                    // 00000000BDE4: A00B0B19
	s_wait_alu 0xfffe                                          // 00000000BDE8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BDEC: BF870499
	s_add_f32 s10, s23, s10                                    // 00000000BDF0: A00A0A17
	s_add_f32 s11, s26, s11                                    // 00000000BDF4: A00B0B1A
	s_wait_alu 0xfffe                                          // 00000000BDF8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BDFC: BF870499
	s_add_f32 s10, s24, s10                                    // 00000000BE00: A00A0A18
	s_add_f32 s11, s27, s11                                    // 00000000BE04: A00B0B1B
	s_wait_alu 0xfffe                                          // 00000000BE08: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000BE0C: BF870529
	s_add_f32 s10, s25, s10                                    // 00000000BE10: A00A0A19
	s_wait_alu 0xfffe                                          // 00000000BE14: BF88FFFE
	s_add_f32 s10, s26, s10                                    // 00000000BE18: A00A0A1A
	v_readlane_b32 s20, v26, 20                                // 00000000BE1C: D7600014 0001291A
	v_readlane_b32 s21, v26, 21                                // 00000000BE24: D7600015 00012B1A
	v_readlane_b32 s22, v26, 22                                // 00000000BE2C: D7600016 00012D1A
	v_readlane_b32 s24, v26, 24                                // 00000000BE34: D7600018 0001311A
	v_readlane_b32 s23, v26, 23                                // 00000000BE3C: D7600017 00012F1A
	s_add_f32 s3, s3, s20                                      // 00000000BE44: A0031403
	v_readlane_b32 s25, v26, 25                                // 00000000BE48: D7600019 0001331A
	v_readlane_b32 s26, v26, 26                                // 00000000BE50: D760001A 0001351A
	s_add_f32 s4, s4, s24                                      // 00000000BE58: A0041804
	s_wait_alu 0xfffe                                          // 00000000BE5C: BF88FFFE
	s_add_f32 s3, s21, s3                                      // 00000000BE60: A0030315
	v_readlane_b32 s27, v26, 27                                // 00000000BE64: D760001B 0001371A
	s_add_f32 s4, s25, s4                                      // 00000000BE6C: A0040419
	s_wait_alu 0xfffe                                          // 00000000BE70: BF88FFFE
	s_add_f32 s3, s22, s3                                      // 00000000BE74: A0030316
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000BE78: BF8704A9
	s_add_f32 s4, s26, s4                                      // 00000000BE7C: A004041A
	s_wait_alu 0xfffe                                          // 00000000BE80: BF88FFFE
	s_add_f32 s3, s23, s3                                      // 00000000BE84: A0030317
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000BE88: BF8704A9
	s_add_f32 s4, s27, s4                                      // 00000000BE8C: A004041B
	s_wait_alu 0xfffe                                          // 00000000BE90: BF88FFFE
	s_add_f32 s3, s24, s3                                      // 00000000BE94: A0030318
	s_wait_alu 0xfffe                                          // 00000000BE98: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000BE9C: BF87052A
	s_add_f32 s3, s25, s3                                      // 00000000BEA0: A0030319
	s_wait_alu 0xfffe                                          // 00000000BEA4: BF88FFFE
	s_add_f32 s3, s26, s3                                      // 00000000BEA8: A003031A
	s_or_saveexec_b32 s105, -1                                 // 00000000BEAC: BEE922C1
	scratch_load_b32 v28, off, off offset:84 th:TH_LOAD_LU     // 00000000BEB0: ED05007C 0030001C 00005400
	s_wait_alu 0xfffe                                          // 00000000BEBC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000BEC0: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000BEC4: BFC00000
	v_readlane_b32 s20, v28, 28                                // 00000000BEC8: D7600014 0001391C
	v_readlane_b32 s21, v28, 29                                // 00000000BED0: D7600015 00013B1C
	v_readlane_b32 s22, v28, 30                                // 00000000BED8: D7600016 00013D1C
	v_readlane_b32 s23, v28, 31                                // 00000000BEE0: D7600017 00013F1C
	v_readlane_b32 s24, v26, 0                                 // 00000000BEE8: D7600018 0001011A
	v_readlane_b32 s25, v26, 1                                 // 00000000BEF0: D7600019 0001031A
	s_add_f32 s12, s12, s21                                    // 00000000BEF8: A00C150C
	s_add_f32 s7, s7, s20                                      // 00000000BEFC: A0071407
	v_readlane_b32 s26, v26, 2                                 // 00000000BF00: D760001A 0001051A
	v_readlane_b32 s27, v26, 3                                 // 00000000BF08: D760001B 0001071A
	s_wait_alu 0xfffe                                          // 00000000BF10: BF88FFFE
	s_add_f32 s12, s22, s12                                    // 00000000BF14: A00C0C16
	s_add_f32 s7, s21, s7                                      // 00000000BF18: A0070715
	s_wait_alu 0xfffe                                          // 00000000BF1C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BF20: BF870499
	s_add_f32 s12, s23, s12                                    // 00000000BF24: A00C0C17
	s_add_f32 s7, s22, s7                                      // 00000000BF28: A0070716
	s_wait_alu 0xfffe                                          // 00000000BF2C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BF30: BF870499
	s_add_f32 s12, s12, s24                                    // 00000000BF34: A00C180C
	s_add_f32 s7, s23, s7                                      // 00000000BF38: A0070717
	s_wait_alu 0xfffe                                          // 00000000BF3C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000BF40: BF870529
	s_add_f32 s12, s25, s12                                    // 00000000BF44: A00C0C19
	s_wait_alu 0xfffe                                          // 00000000BF48: BF88FFFE
	s_add_f32 s12, s26, s12                                    // 00000000BF4C: A00C0C1A
	s_wait_alu 0xfffe                                          // 00000000BF50: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000BF54: BF87000A
	s_add_f32 s12, s27, s12                                    // 00000000BF58: A00C0C1B
	v_readlane_b32 s20, v20, 4                                 // 00000000BF5C: D7600014 00010914
	v_readlane_b32 s21, v20, 5                                 // 00000000BF64: D7600015 00010B14
	v_readlane_b32 s22, v20, 6                                 // 00000000BF6C: D7600016 00010D14
	v_readlane_b32 s23, v20, 7                                 // 00000000BF74: D7600017 00010F14
	v_readlane_b32 s24, v20, 8                                 // 00000000BF7C: D7600018 00011114
	s_add_f32 s9, s9, s20                                      // 00000000BF84: A0091409
	s_add_f32 s13, s13, s21                                    // 00000000BF88: A00D150D
	v_readlane_b32 s25, v20, 9                                 // 00000000BF8C: D7600019 00011314
	v_readlane_b32 s26, v20, 10                                // 00000000BF94: D760001A 00011514
	s_wait_alu 0xfffe                                          // 00000000BF9C: BF88FFFE
	s_add_f32 s9, s21, s9                                      // 00000000BFA0: A0090915
	s_add_f32 s13, s22, s13                                    // 00000000BFA4: A00D0D16
	v_readlane_b32 s27, v20, 11                                // 00000000BFA8: D760001B 00011714
	s_wait_alu 0xfffe                                          // 00000000BFB0: BF88FFFE
	s_add_f32 s9, s22, s9                                      // 00000000BFB4: A0090916
	s_add_f32 s13, s23, s13                                    // 00000000BFB8: A00D0D17
	s_wait_alu 0xfffe                                          // 00000000BFBC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000BFC0: BF870499
	s_add_f32 s9, s23, s9                                      // 00000000BFC4: A0090917
	s_add_f32 s13, s13, s24                                    // 00000000BFC8: A00D180D
	s_wait_alu 0xfffe                                          // 00000000BFCC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000BFD0: BF87052A
	s_add_f32 s13, s25, s13                                    // 00000000BFD4: A00D0D19
	s_wait_alu 0xfffe                                          // 00000000BFD8: BF88FFFE
	s_add_f32 s13, s26, s13                                    // 00000000BFDC: A00D0D1A
	s_wait_alu 0xfffe                                          // 00000000BFE0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000BFE4: BF87000A
	s_add_f32 s13, s27, s13                                    // 00000000BFE8: A00D0D1B
	s_or_saveexec_b32 s105, -1                                 // 00000000BFEC: BEE922C1
	scratch_load_b32 v28, off, off offset:92 th:TH_LOAD_LU     // 00000000BFF0: ED05007C 0030001C 00005C00
	s_wait_alu 0xfffe                                          // 00000000BFFC: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000C000: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000C004: BFC00000
	v_readlane_b32 s20, v28, 4                                 // 00000000C008: D7600014 0001091C
	v_readlane_b32 s21, v28, 5                                 // 00000000C010: D7600015 00010B1C
	v_readlane_b32 s22, v28, 6                                 // 00000000C018: D7600016 00010D1C
	v_readlane_b32 s23, v28, 7                                 // 00000000C020: D7600017 00010F1C
	v_readlane_b32 s24, v28, 8                                 // 00000000C028: D7600018 0001111C
	v_readlane_b32 s25, v28, 9                                 // 00000000C030: D7600019 0001131C
	v_readlane_b32 s27, v28, 11                                // 00000000C038: D760001B 0001171C
	v_readlane_b32 s26, v28, 10                                // 00000000C040: D760001A 0001151C
	s_or_saveexec_b32 s105, -1                                 // 00000000C048: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000C04C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000C050: BEFE0069
	s_add_f32 s14, s14, s21                                    // 00000000C054: A00E150E
	s_add_f32 s11, s11, s20                                    // 00000000C058: A00B140B
	s_wait_alu 0xfffe                                          // 00000000C05C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C060: BF870499
	s_add_f32 s14, s22, s14                                    // 00000000C064: A00E0E16
	s_add_f32 s11, s21, s11                                    // 00000000C068: A00B0B15
	s_wait_alu 0xfffe                                          // 00000000C06C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C070: BF870499
	s_add_f32 s14, s23, s14                                    // 00000000C074: A00E0E17
	s_add_f32 s11, s22, s11                                    // 00000000C078: A00B0B16
	s_wait_alu 0xfffe                                          // 00000000C07C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C080: BF870499
	s_add_f32 s14, s14, s24                                    // 00000000C084: A00E180E
	s_add_f32 s11, s23, s11                                    // 00000000C088: A00B0B17
	s_wait_alu 0xfffe                                          // 00000000C08C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000C090: BF870529
	s_add_f32 s14, s25, s14                                    // 00000000C094: A00E0E19
	s_wait_alu 0xfffe                                          // 00000000C098: BF88FFFE
	s_add_f32 s14, s26, s14                                    // 00000000C09C: A00E0E1A
	s_wait_alu 0xfffe                                          // 00000000C0A0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000C0A4: BF87000A
	s_add_f32 s14, s27, s14                                    // 00000000C0A8: A00E0E1B
	v_readlane_b32 s20, v28, 28                                // 00000000C0AC: D7600014 0001391C
	v_readlane_b32 s21, v28, 29                                // 00000000C0B4: D7600015 00013B1C
	v_readlane_b32 s22, v28, 30                                // 00000000C0BC: D7600016 00013D1C
	v_readlane_b32 s23, v28, 31                                // 00000000C0C4: D7600017 00013F1C
	v_readlane_b32 s24, v20, 0                                 // 00000000C0CC: D7600018 00010114
	v_readlane_b32 s25, v20, 1                                 // 00000000C0D4: D7600019 00010314
	s_add_f32 s5, s5, s21                                      // 00000000C0DC: A0051505
	s_add_f32 s4, s4, s20                                      // 00000000C0E0: A0041404
	v_readlane_b32 s26, v20, 2                                 // 00000000C0E4: D760001A 00010514
	v_readlane_b32 s27, v20, 3                                 // 00000000C0EC: D760001B 00010714
	s_wait_alu 0xfffe                                          // 00000000C0F4: BF88FFFE
	s_add_f32 s5, s22, s5                                      // 00000000C0F8: A0050516
	s_add_f32 s4, s21, s4                                      // 00000000C0FC: A0040415
	s_wait_alu 0xfffe                                          // 00000000C100: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C104: BF870499
	s_add_f32 s5, s23, s5                                      // 00000000C108: A0050517
	s_add_f32 s4, s22, s4                                      // 00000000C10C: A0040416
	s_wait_alu 0xfffe                                          // 00000000C110: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C114: BF870499
	s_add_f32 s5, s5, s24                                      // 00000000C118: A0051805
	s_add_f32 s4, s23, s4                                      // 00000000C11C: A0040417
	s_wait_alu 0xfffe                                          // 00000000C120: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000C124: BF870529
	s_add_f32 s5, s25, s5                                      // 00000000C128: A0050519
	s_wait_alu 0xfffe                                          // 00000000C12C: BF88FFFE
	s_add_f32 s5, s26, s5                                      // 00000000C130: A005051A
	s_wait_alu 0xfffe                                          // 00000000C134: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000C138: BF87000A
	s_add_f32 s5, s27, s5                                      // 00000000C13C: A005051B
	v_readlane_b32 s20, v25, 20                                // 00000000C140: D7600014 00012919
	v_readlane_b32 s21, v25, 21                                // 00000000C148: D7600015 00012B19
	v_readlane_b32 s22, v25, 22                                // 00000000C150: D7600016 00012D19
	v_readlane_b32 s23, v25, 23                                // 00000000C158: D7600017 00012F19
	v_readlane_b32 s24, v25, 24                                // 00000000C160: D7600018 00013119
	s_add_f32 s6, s6, s20                                      // 00000000C168: A0061406
	v_readlane_b32 s25, v25, 25                                // 00000000C16C: D7600019 00013319
	v_readlane_b32 s26, v25, 26                                // 00000000C174: D760001A 00013519
	v_readlane_b32 s27, v25, 27                                // 00000000C17C: D760001B 00013719
	s_wait_alu 0xfffe                                          // 00000000C184: BF88FFFE
	s_add_f32 s6, s21, s6                                      // 00000000C188: A0060615
	s_add_f32 s7, s7, s24                                      // 00000000C18C: A0071807
	s_wait_alu 0xfffe                                          // 00000000C190: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C194: BF870499
	s_add_f32 s6, s22, s6                                      // 00000000C198: A0060616
	s_add_f32 s7, s25, s7                                      // 00000000C19C: A0070719
	s_wait_alu 0xfffe                                          // 00000000C1A0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C1A4: BF870499
	s_add_f32 s6, s23, s6                                      // 00000000C1A8: A0060617
	s_add_f32 s7, s26, s7                                      // 00000000C1AC: A007071A
	s_wait_alu 0xfffe                                          // 00000000C1B0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C1B4: BF870499
	s_add_f32 s6, s24, s6                                      // 00000000C1B8: A0060618
	s_add_f32 s7, s27, s7                                      // 00000000C1BC: A007071B
	s_wait_alu 0xfffe                                          // 00000000C1C0: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000C1C4: BF870529
	s_add_f32 s6, s25, s6                                      // 00000000C1C8: A0060619
	s_wait_alu 0xfffe                                          // 00000000C1CC: BF88FFFE
	s_add_f32 s6, s26, s6                                      // 00000000C1D0: A006061A
	v_readlane_b32 s20, v25, 12                                // 00000000C1D4: D7600014 00011919
	v_readlane_b32 s21, v25, 13                                // 00000000C1DC: D7600015 00011B19
	v_readlane_b32 s22, v25, 14                                // 00000000C1E4: D7600016 00011D19
	v_readlane_b32 s23, v25, 15                                // 00000000C1EC: D7600017 00011F19
	v_readlane_b32 s24, v25, 16                                // 00000000C1F4: D7600018 00012119
	s_add_f32 s8, s8, s20                                      // 00000000C1FC: A0081408
	v_readlane_b32 s25, v25, 17                                // 00000000C200: D7600019 00012319
	v_readlane_b32 s26, v25, 18                                // 00000000C208: D760001A 00012519
	v_readlane_b32 s27, v25, 19                                // 00000000C210: D760001B 00012719
	s_wait_alu 0xfffe                                          // 00000000C218: BF88FFFE
	s_add_f32 s8, s21, s8                                      // 00000000C21C: A0080815
	s_add_f32 s9, s9, s24                                      // 00000000C220: A0091809
	s_wait_alu 0xfffe                                          // 00000000C224: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C228: BF870499
	s_add_f32 s8, s22, s8                                      // 00000000C22C: A0080816
	s_add_f32 s9, s25, s9                                      // 00000000C230: A0090919
	s_wait_alu 0xfffe                                          // 00000000C234: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C238: BF870499
	s_add_f32 s8, s23, s8                                      // 00000000C23C: A0080817
	s_add_f32 s9, s26, s9                                      // 00000000C240: A009091A
	s_wait_alu 0xfffe                                          // 00000000C244: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C248: BF870499
	s_add_f32 s8, s24, s8                                      // 00000000C24C: A0080818
	s_add_f32 s9, s27, s9                                      // 00000000C250: A009091B
	s_wait_alu 0xfffe                                          // 00000000C254: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000C258: BF870529
	s_add_f32 s8, s25, s8                                      // 00000000C25C: A0080819
	s_wait_alu 0xfffe                                          // 00000000C260: BF88FFFE
	s_add_f32 s8, s26, s8                                      // 00000000C264: A008081A
	v_readlane_b32 s20, v21, 4                                 // 00000000C268: D7600014 00010915
	v_readlane_b32 s21, v21, 5                                 // 00000000C270: D7600015 00010B15
	v_readlane_b32 s22, v21, 6                                 // 00000000C278: D7600016 00010D15
	v_readlane_b32 s23, v21, 7                                 // 00000000C280: D7600017 00010F15
	v_readlane_b32 s24, v21, 8                                 // 00000000C288: D7600018 00011115
	s_add_f32 s10, s10, s20                                    // 00000000C290: A00A140A
	v_readlane_b32 s25, v21, 9                                 // 00000000C294: D7600019 00011315
	v_readlane_b32 s26, v21, 10                                // 00000000C29C: D760001A 00011515
	v_readlane_b32 s27, v21, 11                                // 00000000C2A4: D760001B 00011715
	s_wait_alu 0xfffe                                          // 00000000C2AC: BF88FFFE
	s_add_f32 s10, s21, s10                                    // 00000000C2B0: A00A0A15
	s_add_f32 s11, s11, s24                                    // 00000000C2B4: A00B180B
	s_wait_alu 0xfffe                                          // 00000000C2B8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C2BC: BF870499
	s_add_f32 s10, s22, s10                                    // 00000000C2C0: A00A0A16
	s_add_f32 s11, s25, s11                                    // 00000000C2C4: A00B0B19
	s_wait_alu 0xfffe                                          // 00000000C2C8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C2CC: BF870499
	s_add_f32 s10, s23, s10                                    // 00000000C2D0: A00A0A17
	s_add_f32 s11, s26, s11                                    // 00000000C2D4: A00B0B1A
	s_wait_alu 0xfffe                                          // 00000000C2D8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C2DC: BF870499
	s_add_f32 s10, s24, s10                                    // 00000000C2E0: A00A0A18
	s_add_f32 s11, s27, s11                                    // 00000000C2E4: A00B0B1B
	s_wait_alu 0xfffe                                          // 00000000C2E8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000C2EC: BF870529
	s_add_f32 s10, s25, s10                                    // 00000000C2F0: A00A0A19
	s_wait_alu 0xfffe                                          // 00000000C2F4: BF88FFFE
	s_add_f32 s10, s26, s10                                    // 00000000C2F8: A00A0A1A
	v_readlane_b32 s20, v25, 4                                 // 00000000C2FC: D7600014 00010919
	v_readlane_b32 s21, v25, 5                                 // 00000000C304: D7600015 00010B19
	v_readlane_b32 s22, v25, 6                                 // 00000000C30C: D7600016 00010D19
	v_readlane_b32 s24, v25, 8                                 // 00000000C314: D7600018 00011119
	v_readlane_b32 s23, v25, 7                                 // 00000000C31C: D7600017 00010F19
	s_add_f32 s3, s3, s20                                      // 00000000C324: A0031403
	v_readlane_b32 s25, v25, 9                                 // 00000000C328: D7600019 00011319
	v_readlane_b32 s26, v25, 10                                // 00000000C330: D760001A 00011519
	s_add_f32 s4, s4, s24                                      // 00000000C338: A0041804
	s_wait_alu 0xfffe                                          // 00000000C33C: BF88FFFE
	s_add_f32 s3, s21, s3                                      // 00000000C340: A0030315
	v_readlane_b32 s27, v25, 11                                // 00000000C344: D760001B 00011719
	s_add_f32 s4, s25, s4                                      // 00000000C34C: A0040419
	s_wait_alu 0xfffe                                          // 00000000C350: BF88FFFE
	s_add_f32 s3, s22, s3                                      // 00000000C354: A0030316
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000C358: BF8704A9
	s_add_f32 s4, s26, s4                                      // 00000000C35C: A004041A
	s_wait_alu 0xfffe                                          // 00000000C360: BF88FFFE
	s_add_f32 s3, s23, s3                                      // 00000000C364: A0030317
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000C368: BF8704A9
	s_add_f32 s4, s27, s4                                      // 00000000C36C: A004041B
	s_wait_alu 0xfffe                                          // 00000000C370: BF88FFFE
	s_add_f32 s3, s24, s3                                      // 00000000C374: A0030318
	s_wait_alu 0xfffe                                          // 00000000C378: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000C37C: BF87052A
	s_add_f32 s3, s25, s3                                      // 00000000C380: A0030319
	s_wait_alu 0xfffe                                          // 00000000C384: BF88FFFE
	s_add_f32 s3, s26, s3                                      // 00000000C388: A003031A
	s_or_saveexec_b32 s105, -1                                 // 00000000C38C: BEE922C1
	scratch_load_b32 v28, off, off offset:120 th:TH_LOAD_LU    // 00000000C390: ED05007C 0030001C 00007800
	s_wait_alu 0xfffe                                          // 00000000C39C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000C3A0: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000C3A4: BFC00000
	v_readlane_b32 s20, v28, 12                                // 00000000C3A8: D7600014 0001191C
	v_readlane_b32 s21, v28, 13                                // 00000000C3B0: D7600015 00011B1C
	v_readlane_b32 s22, v28, 14                                // 00000000C3B8: D7600016 00011D1C
	v_readlane_b32 s23, v28, 15                                // 00000000C3C0: D7600017 00011F1C
	v_readlane_b32 s24, v28, 16                                // 00000000C3C8: D7600018 0001211C
	v_readlane_b32 s25, v28, 17                                // 00000000C3D0: D7600019 0001231C
	s_add_f32 s12, s12, s21                                    // 00000000C3D8: A00C150C
	v_readlane_b32 s26, v28, 18                                // 00000000C3DC: D760001A 0001251C
	v_readlane_b32 s27, v28, 19                                // 00000000C3E4: D760001B 0001271C
	s_add_f32 s7, s7, s20                                      // 00000000C3EC: A0071407
	s_wait_alu 0xfffe                                          // 00000000C3F0: BF88FFFE
	s_add_f32 s12, s22, s12                                    // 00000000C3F4: A00C0C16
	s_add_f32 s11, s11, s84                                    // 00000000C3F8: A00B540B
	s_add_f32 s7, s21, s7                                      // 00000000C3FC: A0070715
	s_wait_alu 0xfffe                                          // 00000000C400: BF88FFFE
	s_add_f32 s12, s23, s12                                    // 00000000C404: A00C0C17
	s_add_f32 s11, s85, s11                                    // 00000000C408: A00B0B55
	s_add_f32 s7, s22, s7                                      // 00000000C40C: A0070716
	s_wait_alu 0xfffe                                          // 00000000C410: BF88FFFE
	s_add_f32 s12, s12, s24                                    // 00000000C414: A00C180C
	s_add_f32 s11, s86, s11                                    // 00000000C418: A00B0B56
	s_add_f32 s7, s23, s7                                      // 00000000C41C: A0070717
	s_wait_alu 0xfffe                                          // 00000000C420: BF88FFFE
	s_add_f32 s12, s25, s12                                    // 00000000C424: A00C0C19
	s_add_f32 s11, s87, s11                                    // 00000000C428: A00B0B57
	s_wait_alu 0xfffe                                          // 00000000C42C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000C430: BF870529
	s_add_f32 s12, s26, s12                                    // 00000000C434: A00C0C1A
	s_wait_alu 0xfffe                                          // 00000000C438: BF88FFFE
	s_add_f32 s21, s27, s12                                    // 00000000C43C: A0150C1B
	v_readlane_b32 s24, v29, 20                                // 00000000C440: D7600018 0001291D
	v_readlane_b32 s25, v29, 21                                // 00000000C448: D7600019 00012B1D
	v_readlane_b32 s26, v29, 22                                // 00000000C450: D760001A 00012D1D
	v_readlane_b32 s27, v29, 23                                // 00000000C458: D760001B 00012F1D
	v_readlane_b32 s28, v29, 24                                // 00000000C460: D760001C 0001311D
	v_readlane_b32 s29, v29, 25                                // 00000000C468: D760001D 0001331D
	s_add_f32 s12, s13, s25                                    // 00000000C470: A00C190D
	s_add_f32 s9, s9, s24                                      // 00000000C474: A0091809
	v_readlane_b32 s30, v29, 26                                // 00000000C478: D760001E 0001351D
	v_readlane_b32 s31, v29, 27                                // 00000000C480: D760001F 0001371D
	s_wait_alu 0xfffe                                          // 00000000C488: BF88FFFE
	s_add_f32 s12, s26, s12                                    // 00000000C48C: A00C0C1A
	s_add_f32 s9, s25, s9                                      // 00000000C490: A0090919
	s_wait_alu 0xfffe                                          // 00000000C494: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C498: BF870499
	s_add_f32 s12, s27, s12                                    // 00000000C49C: A00C0C1B
	s_add_f32 s9, s26, s9                                      // 00000000C4A0: A009091A
	s_wait_alu 0xfffe                                          // 00000000C4A4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C4A8: BF870499
	s_add_f32 s12, s12, s28                                    // 00000000C4AC: A00C1C0C
	s_add_f32 s9, s27, s9                                      // 00000000C4B0: A009091B
	s_wait_alu 0xfffe                                          // 00000000C4B4: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000C4B8: BF870529
	s_add_f32 s12, s29, s12                                    // 00000000C4BC: A00C0C1D
	s_wait_alu 0xfffe                                          // 00000000C4C0: BF88FFFE
	s_add_f32 s12, s30, s12                                    // 00000000C4C4: A00C0C1E
	s_wait_alu 0xfffe                                          // 00000000C4C8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_2)                          // 00000000C4CC: BF87000A
	s_add_f32 s22, s31, s12                                    // 00000000C4D0: A0160C1F
	v_readlane_b32 s24, v29, 12                                // 00000000C4D4: D7600018 0001191D
	v_readlane_b32 s25, v29, 13                                // 00000000C4DC: D7600019 00011B1D
	v_readlane_b32 s26, v29, 14                                // 00000000C4E4: D760001A 00011D1D
	v_readlane_b32 s27, v29, 15                                // 00000000C4EC: D760001B 00011F1D
	s_add_f32 s12, s14, s85                                    // 00000000C4F4: A00C550E
	v_readlane_b32 s28, v29, 16                                // 00000000C4F8: D760001C 0001211D
	s_add_f32 s5, s5, s25                                      // 00000000C500: A0051905
	v_readlane_b32 s29, v29, 17                                // 00000000C504: D760001D 0001231D
	s_wait_alu 0xfffe                                          // 00000000C50C: BF88FFFE
	s_add_f32 s12, s86, s12                                    // 00000000C510: A00C0C56
	s_add_f32 s4, s4, s24                                      // 00000000C514: A0041804
	s_add_f32 s5, s26, s5                                      // 00000000C518: A005051A
	v_readlane_b32 s30, v29, 18                                // 00000000C51C: D760001E 0001251D
	s_wait_alu 0xfffe                                          // 00000000C524: BF88FFFE
	s_add_f32 s12, s87, s12                                    // 00000000C528: A00C0C57
	s_add_f32 s4, s25, s4                                      // 00000000C52C: A0040419
	s_add_f32 s5, s27, s5                                      // 00000000C530: A005051B
	v_readlane_b32 s31, v29, 19                                // 00000000C534: D760001F 0001271D
	s_wait_alu 0xfffe                                          // 00000000C53C: BF88FFFE
	s_add_f32 s12, s12, s88                                    // 00000000C540: A00C580C
	s_add_f32 s4, s26, s4                                      // 00000000C544: A004041A
	s_add_f32 s5, s5, s28                                      // 00000000C548: A0051C05
	s_wait_alu 0xfffe                                          // 00000000C54C: BF88FFFE
	s_add_f32 s12, s89, s12                                    // 00000000C550: A00C0C59
	s_add_f32 s4, s27, s4                                      // 00000000C554: A004041B
	s_add_f32 s5, s29, s5                                      // 00000000C558: A005051D
	s_wait_alu 0xfffe                                          // 00000000C55C: BF88FFFE
	s_add_f32 s12, s90, s12                                    // 00000000C560: A00C0C5A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000C564: BF8704A9
	s_add_f32 s5, s30, s5                                      // 00000000C568: A005051E
	s_wait_alu 0xfffe                                          // 00000000C56C: BF88FFFE
	s_add_f32 s20, s91, s12                                    // 00000000C570: A0140C5B
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000C574: BF870009
	s_add_f32 s19, s31, s5                                     // 00000000C578: A013051F
	s_or_saveexec_b32 s105, -1                                 // 00000000C57C: BEE922C1
	scratch_load_b32 v29, off, off offset:116 th:TH_LOAD_LU    // 00000000C580: ED05007C 0030001D 00007400
	s_wait_alu 0xfffe                                          // 00000000C58C: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000C590: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000C594: BFC00000
	v_readlane_b32 s44, v29, 20                                // 00000000C598: D760002C 0001291D
	v_readlane_b32 s45, v29, 21                                // 00000000C5A0: D760002D 00012B1D
	v_readlane_b32 s46, v29, 22                                // 00000000C5A8: D760002E 00012D1D
	v_readlane_b32 s47, v29, 23                                // 00000000C5B0: D760002F 00012F1D
	v_readlane_b32 s48, v29, 24                                // 00000000C5B8: D7600030 0001311D
	s_add_f32 s5, s6, s44                                      // 00000000C5C0: A0052C06
	v_readlane_b32 s49, v29, 25                                // 00000000C5C4: D7600031 0001331D
	v_readlane_b32 s50, v29, 26                                // 00000000C5CC: D7600032 0001351D
	v_readlane_b32 s51, v29, 27                                // 00000000C5D4: D7600033 0001371D
	s_wait_alu 0xfffe                                          // 00000000C5DC: BF88FFFE
	s_add_f32 s5, s45, s5                                      // 00000000C5E0: A005052D
	s_add_f32 s6, s7, s48                                      // 00000000C5E4: A0063007
	s_wait_alu 0xfffe                                          // 00000000C5E8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C5EC: BF870499
	s_add_f32 s5, s46, s5                                      // 00000000C5F0: A005052E
	s_add_f32 s6, s49, s6                                      // 00000000C5F4: A0060631
	s_wait_alu 0xfffe                                          // 00000000C5F8: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C5FC: BF870499
	s_add_f32 s5, s47, s5                                      // 00000000C600: A005052F
	s_add_f32 s6, s50, s6                                      // 00000000C604: A0060632
	s_wait_alu 0xfffe                                          // 00000000C608: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C60C: BF870499
	s_add_f32 s5, s48, s5                                      // 00000000C610: A0050530
	s_add_f32 s23, s51, s6                                     // 00000000C614: A0170633
	s_wait_alu 0xfffe                                          // 00000000C618: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000C61C: BF870529
	s_add_f32 s5, s49, s5                                      // 00000000C620: A0050531
	s_wait_alu 0xfffe                                          // 00000000C624: BF88FFFE
	s_add_f32 s31, s50, s5                                     // 00000000C628: A01F0532
	v_readlane_b32 s44, v29, 12                                // 00000000C62C: D760002C 0001191D
	v_readlane_b32 s45, v29, 13                                // 00000000C634: D760002D 00011B1D
	v_readlane_b32 s46, v29, 14                                // 00000000C63C: D760002E 00011D1D
	v_readlane_b32 s47, v29, 15                                // 00000000C644: D760002F 00011F1D
	v_readlane_b32 s48, v29, 16                                // 00000000C64C: D7600030 0001211D
	s_add_f32 s5, s8, s44                                      // 00000000C654: A0052C08
	v_readlane_b32 s49, v29, 17                                // 00000000C658: D7600031 0001231D
	v_readlane_b32 s50, v29, 18                                // 00000000C660: D7600032 0001251D
	v_readlane_b32 s51, v29, 19                                // 00000000C668: D7600033 0001271D
	s_wait_alu 0xfffe                                          // 00000000C670: BF88FFFE
	s_add_f32 s5, s45, s5                                      // 00000000C674: A005052D
	s_add_f32 s6, s9, s48                                      // 00000000C678: A0063009
	s_wait_alu 0xfffe                                          // 00000000C67C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C680: BF870499
	s_add_f32 s5, s46, s5                                      // 00000000C684: A005052E
	s_add_f32 s6, s49, s6                                      // 00000000C688: A0060631
	s_wait_alu 0xfffe                                          // 00000000C68C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C690: BF870499
	s_add_f32 s5, s47, s5                                      // 00000000C694: A005052F
	s_add_f32 s6, s50, s6                                      // 00000000C698: A0060632
	s_wait_alu 0xfffe                                          // 00000000C69C: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000C6A0: BF870499
	s_add_f32 s5, s48, s5                                      // 00000000C6A4: A0050530
	s_add_f32 s24, s51, s6                                     // 00000000C6A8: A0180633
	s_wait_alu 0xfffe                                          // 00000000C6AC: BF88FFFE
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 00000000C6B0: BF870529
	s_add_f32 s5, s49, s5                                      // 00000000C6B4: A0050531
	s_wait_alu 0xfffe                                          // 00000000C6B8: BF88FFFE
	s_add_f32 s42, s50, s5                                     // 00000000C6BC: A02A0532
	v_readlane_b32 s44, v29, 28                                // 00000000C6C0: D760002C 0001391D
	v_readlane_b32 s45, v29, 29                                // 00000000C6C8: D760002D 00013B1D
	v_readlane_b32 s46, v29, 30                                // 00000000C6D0: D760002E 00013D1D
	v_readlane_b32 s48, v21, 0                                 // 00000000C6D8: D7600030 00010115
	v_readlane_b32 s47, v29, 31                                // 00000000C6E0: D760002F 00013F1D
	s_add_f32 s5, s10, s44                                     // 00000000C6E8: A0052C0A
	v_readlane_b32 s49, v21, 1                                 // 00000000C6EC: D7600031 00010315
	v_readlane_b32 s50, v21, 2                                 // 00000000C6F4: D7600032 00010515
	s_add_f32 s6, s11, s48                                     // 00000000C6FC: A006300B
	s_wait_alu 0xfffe                                          // 00000000C700: BF88FFFE
	s_add_f32 s5, s45, s5                                      // 00000000C704: A005052D
	v_readlane_b32 s8, v29, 4                                  // 00000000C708: D7600008 0001091D
	v_readlane_b32 s9, v29, 5                                  // 00000000C710: D7600009 00010B1D
	s_add_f32 s6, s49, s6                                      // 00000000C718: A0060631
	s_wait_alu 0xfffe                                          // 00000000C71C: BF88FFFE
	s_add_f32 s5, s46, s5                                      // 00000000C720: A005052E
	v_readlane_b32 s51, v21, 3                                 // 00000000C724: D7600033 00010715
	s_add_f32 s3, s3, s8                                       // 00000000C72C: A0030803
	v_readlane_b32 s10, v29, 6                                 // 00000000C730: D760000A 00010D1D
	s_wait_alu 0xfffe                                          // 00000000C738: BF88FFFE
	s_add_f32 s5, s47, s5                                      // 00000000C73C: A005052F
	v_readlane_b32 s12, v29, 8                                 // 00000000C740: D760000C 0001111D
	s_add_f32 s6, s50, s6                                      // 00000000C748: A0060632
	s_add_f32 s3, s9, s3                                       // 00000000C74C: A0030309
	s_wait_alu 0xfffe                                          // 00000000C750: BF88FFFE
	s_add_f32 s5, s48, s5                                      // 00000000C754: A0050530
	v_readlane_b32 s11, v29, 7                                 // 00000000C758: D760000B 00010F1D
	s_add_f32 s25, s51, s6                                     // 00000000C760: A0190633
	s_add_f32 s3, s10, s3                                      // 00000000C764: A003030A
	s_wait_alu 0xfffe                                          // 00000000C768: BF88FFFE
	s_add_f32 s5, s49, s5                                      // 00000000C76C: A0050531
	s_add_f32 s4, s4, s12                                      // 00000000C770: A0040C04
	v_readlane_b32 s13, v29, 9                                 // 00000000C774: D760000D 0001131D
	v_readlane_b32 s14, v29, 10                                // 00000000C77C: D760000E 0001151D
	s_wait_alu 0xfffe                                          // 00000000C784: BF88FFFE
	s_add_f32 s30, s50, s5                                     // 00000000C788: A01E0532
	v_readlane_b32 s15, v29, 11                                // 00000000C78C: D760000F 0001171D
	s_add_f32 s3, s11, s3                                      // 00000000C794: A003030B
	s_or_saveexec_b32 s105, -1                                 // 00000000C798: BEE922C1
	scratch_load_b32 v28, off, off th:TH_LOAD_LU               // 00000000C79C: ED05007C 0030001C 00000000
	s_wait_alu 0xfffe                                          // 00000000C7A8: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000C7AC: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000C7B0: BFC00000
	v_readlane_b32 s44, v28, 12                                // 00000000C7B4: D760002C 0001191C
	v_readlane_b32 s45, v28, 13                                // 00000000C7BC: D760002D 00011B1C
	v_readlane_b32 s46, v28, 14                                // 00000000C7C4: D760002E 00011D1C
	v_readlane_b32 s47, v28, 15                                // 00000000C7CC: D760002F 00011F1C
	v_readlane_b32 s48, v28, 16                                // 00000000C7D4: D7600030 0001211C
	v_readlane_b32 s49, v28, 17                                // 00000000C7DC: D7600031 0001231C
	v_readlane_b32 s50, v28, 18                                // 00000000C7E4: D7600032 0001251C
	v_readlane_b32 s51, v28, 19                                // 00000000C7EC: D7600033 0001271C
	v_readlane_b32 s52, v28, 20                                // 00000000C7F4: D7600034 0001291C
	v_readlane_b32 s53, v28, 21                                // 00000000C7FC: D7600035 00012B1C
	v_readlane_b32 s54, v28, 22                                // 00000000C804: D7600036 00012D1C
	v_readlane_b32 s55, v28, 23                                // 00000000C80C: D7600037 00012F1C
	v_readlane_b32 s56, v28, 24                                // 00000000C814: D7600038 0001311C
	v_readlane_b32 s57, v28, 25                                // 00000000C81C: D7600039 0001331C
	v_readlane_b32 s58, v28, 26                                // 00000000C824: D760003A 0001351C
	v_readlane_b32 s59, v28, 27                                // 00000000C82C: D760003B 0001371C
	s_or_saveexec_b32 s105, -1                                 // 00000000C834: BEE922C1
	s_wait_alu 0xfffe                                          // 00000000C838: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000C83C: BEFE0069
	s_add_f32 s1, s1, s44                                      // 00000000C840: A0012C01
	s_add_f32 s3, s12, s3                                      // 00000000C844: A003030C
	s_add_f32 s4, s13, s4                                      // 00000000C848: A004040D
	s_add_f32 s2, s2, s68                                      // 00000000C84C: A0024402
	s_wait_alu 0xfffe                                          // 00000000C850: BF88FFFE
	s_add_f32 s1, s45, s1                                      // 00000000C854: A001012D
	s_add_f32 s3, s13, s3                                      // 00000000C858: A003030D
	s_add_f32 s4, s14, s4                                      // 00000000C85C: A004040E
	s_add_f32 s2, s69, s2                                      // 00000000C860: A0020245
	s_wait_alu 0xfffe                                          // 00000000C864: BF88FFFE
	s_add_f32 s1, s46, s1                                      // 00000000C868: A001012E
	s_add_f32 s3, s14, s3                                      // 00000000C86C: A003030E
	s_add_f32 s26, s15, s4                                     // 00000000C870: A01A040F
	s_load_b256 s[8:15], s[98:99], 0xfe0                       // 00000000C874: F4006231 F8000FE0
	s_add_f32 s1, s47, s1                                      // 00000000C87C: A001012F
	s_add_f32 s2, s70, s2                                      // 00000000C880: A0020246
	s_add_f32 s29, s0, s48                                     // 00000000C884: A01D3000
	v_readlane_b32 s60, v28, 28                                // 00000000C888: D760003C 0001391C
	s_wait_alu 0xfffe                                          // 00000000C890: BF88FFFE
	s_add_f32 s1, s48, s1                                      // 00000000C894: A0010130
	s_add_f32 s2, s71, s2                                      // 00000000C898: A0020247
	v_readlane_b32 s61, v28, 29                                // 00000000C89C: D760003D 00013B1C
	v_writelane_b32 v23, s3, 19                                // 00000000C8A4: D7610017 00012603
	s_wait_alu 0xfffe                                          // 00000000C8AC: BF88FFFE
	s_add_f32 s27, s49, s1                                     // 00000000C8B0: A01B0131
	s_add_f32 s2, s2, s72                                      // 00000000C8B4: A0024802
	s_add_f32 s23, s23, s60                                    // 00000000C8B8: A0173C17
	v_readlane_b32 s62, v28, 30                                // 00000000C8BC: D760003E 00013D1C
	s_wait_alu 0xfffe                                          // 00000000C8C4: BF88FFFE
	s_add_f32 s45, s50, s27                                    // 00000000C8C8: A02D1B32
	s_add_f32 s27, s49, s29                                    // 00000000C8CC: A01B1D31
	s_add_f32 s28, s73, s2                                     // 00000000C8D0: A01C0249
	s_load_b256 s[0:7], s[98:99], 0x1120                       // 00000000C8D4: F4006031 F8001120
	v_readlane_b32 s84, v21, 16                                // 00000000C8DC: D7600054 00012115
	s_add_f32 s23, s61, s23                                    // 00000000C8E4: A017173D
	v_readlane_b32 s63, v28, 31                                // 00000000C8E8: D760003F 00013F1C
	s_wait_kmcnt 0x0                                           // 00000000C8F0: BFC70000
	s_add_f32 s8, s24, s8                                      // 00000000C8F4: A0080818
	s_add_f32 s24, s50, s27                                    // 00000000C8F8: A0181B32
	s_add_f32 s27, s18, 0                                      // 00000000C8FC: A01B8012
	v_readlane_b32 s85, v21, 17                                // 00000000C900: D7600055 00012315
	s_wait_alu 0xfffe                                          // 00000000C908: BF88FFFE
	s_add_f32 s23, s62, s23                                    // 00000000C90C: A017173E
	s_add_f32 s24, s51, s24                                    // 00000000C910: A0181833
	s_add_f32 s25, s25, s84                                    // 00000000C914: A0195419
	v_readlane_b32 s86, v21, 18                                // 00000000C918: D7600056 00012515
	s_add_f32 s8, s9, s8                                       // 00000000C920: A0080809
	s_wait_alu 0xfffe                                          // 00000000C924: BF88FFFE
	s_add_f32 s18, s24, s52                                    // 00000000C928: A0123418
	s_add_f32 s24, s85, s25                                    // 00000000C92C: A0181955
	s_add_f32 s25, s39, s72                                    // 00000000C930: A0194827
	s_add_f32 s23, s63, s23                                    // 00000000C934: A017173F
	s_wait_alu 0xfffe                                          // 00000000C938: BF88FFFE
	s_add_f32 s18, s53, s18                                    // 00000000C93C: A0121235
	s_add_f32 s8, s10, s8                                      // 00000000C940: A008080A
	s_add_f32 s9, s22, s9                                      // 00000000C944: A0090916
	v_writelane_b32 v23, s23, 18                               // 00000000C948: D7610017 00012417
	s_wait_alu 0xfffe                                          // 00000000C950: BF88FFFE
	s_add_f32 s18, s54, s18                                    // 00000000C954: A0121236
	s_add_f32 s23, s86, s24                                    // 00000000C958: A0171856
	s_add_f32 s24, s73, s25                                    // 00000000C95C: A0181949
	s_add_f32 s29, s11, s8                                     // 00000000C960: A01D080B
	s_wait_alu 0xfffe                                          // 00000000C964: BF88FFFE
	s_add_f32 s39, s55, s18                                    // 00000000C968: A0271237
	s_add_f32 s18, s21, s61                                    // 00000000C96C: A0123D15
	s_add_f32 s8, s74, s24                                     // 00000000C970: A008184A
	s_add_f32 s0, s26, s0                                      // 00000000C974: A000001A
	v_readlane_b32 s64, v29, 0                                 // 00000000C978: D7600040 0001011D
	s_wait_alu 0xfffe                                          // 00000000C980: BF88FFFE
	s_add_f32 s18, s62, s18                                    // 00000000C984: A012123E
	s_add_f32 s8, s75, s8                                      // 00000000C988: A008084B
	s_add_f32 s9, s10, s9                                      // 00000000C98C: A009090A
	s_add_f32 s0, s1, s0                                       // 00000000C990: A0000001
	s_wait_alu 0xfffe                                          // 00000000C994: BF88FFFE
	s_add_f32 s10, s63, s18                                    // 00000000C998: A00A123F
	v_readlane_b32 s65, v29, 1                                 // 00000000C99C: D7600041 0001031D
	s_add_f32 s8, s8, s76                                      // 00000000C9A4: A0084C08
	s_add_f32 s9, s11, s9                                      // 00000000C9A8: A009090B
	s_add_f32 s0, s2, s0                                       // 00000000C9AC: A0000002
	s_wait_alu 0xfffe                                          // 00000000C9B0: BF88FFFE
	s_add_f32 s10, s10, s64                                    // 00000000C9B4: A00A400A
	v_readlane_b32 s66, v29, 2                                 // 00000000C9B8: D7600042 0001051D
	s_add_f32 s8, s77, s8                                      // 00000000C9C0: A008084D
	s_add_f32 s9, s9, s12                                      // 00000000C9C4: A0090C09
	s_add_f32 s24, s3, s0                                      // 00000000C9C8: A0180003
	s_wait_alu 0xfffe                                          // 00000000C9CC: BF88FFFE
	s_add_f32 s0, s65, s10                                     // 00000000C9D0: A0000A41
	v_readlane_b32 s67, v29, 3                                 // 00000000C9D4: D7600043 0001071D
	s_add_f32 s8, s78, s8                                      // 00000000C9DC: A008084E
	s_add_f32 s9, s13, s9                                      // 00000000C9E0: A009090D
	s_wait_alu 0xfffe                                          // 00000000C9E4: BF88FFFE
	s_add_f32 s0, s66, s0                                      // 00000000C9E8: A0000042
	s_add_f32 s21, s17, 0                                      // 00000000C9EC: A0158011
	s_add_f32 s22, s79, s8                                     // 00000000C9F0: A016084F
	s_add_f32 s8, s14, s9                                      // 00000000C9F4: A008090E
	s_wait_alu 0xfffe                                          // 00000000C9F8: BF88FFFE
	s_add_f32 s0, s67, s0                                      // 00000000C9FC: A0000043
	s_add_f32 s1, s19, s1                                      // 00000000CA00: A0010113
	s_add_f32 s9, s35, s77                                     // 00000000CA04: A0094D23
	s_add_f32 s17, s15, s8                                     // 00000000CA08: A011080F
	s_wait_alu 0xfffe                                          // 00000000CA0C: BF88FFFE
	v_writelane_b32 v23, s0, 17                                // 00000000CA10: D7610017 00012200
	s_add_f32 s0, s20, s85                                     // 00000000CA18: A0005514
	s_add_f32 s8, s33, s53                                     // 00000000CA1C: A0083521
	v_readlane_b32 s87, v21, 19                                // 00000000CA20: D7600057 00012715
	s_add_f32 s1, s2, s1                                       // 00000000CA28: A0010102
	s_wait_alu 0xfffe                                          // 00000000CA2C: BF88FFFE
	s_add_f32 s0, s86, s0                                      // 00000000CA30: A0000056
	s_add_f32 s8, s54, s8                                      // 00000000CA34: A0080836
	s_add_f32 s2, s78, s9                                      // 00000000CA38: A002094E
	v_readlane_b32 s88, v21, 20                                // 00000000CA3C: D7600058 00012915
	s_wait_alu 0xfffe                                          // 00000000CA44: BF88FFFE
	s_add_f32 s0, s87, s0                                      // 00000000CA48: A0000057
	s_add_f32 s8, s55, s8                                      // 00000000CA4C: A0080837
	s_add_f32 s1, s3, s1                                       // 00000000CA50: A0010103
	s_add_f32 s2, s79, s2                                      // 00000000CA54: A002024F
	v_readlane_b32 s89, v21, 21                                // 00000000CA58: D7600059 00012B15
	s_wait_alu 0xfffe                                          // 00000000CA60: BF88FFFE
	s_add_f32 s0, s0, s88                                      // 00000000CA64: A0005800
	s_add_f32 s3, s8, s56                                      // 00000000CA68: A0033808
	s_add_f32 s1, s1, s4                                       // 00000000CA6C: A0010401
	s_add_f32 s2, s2, s80                                      // 00000000CA70: A0025002
	v_readlane_b32 s90, v21, 22                                // 00000000CA74: D760005A 00012D15
	s_wait_alu 0xfffe                                          // 00000000CA7C: BF88FFFE
	s_add_f32 s0, s89, s0                                      // 00000000CA80: A0000059
	s_add_f32 s3, s57, s3                                      // 00000000CA84: A0030339
	s_add_f32 s1, s5, s1                                       // 00000000CA88: A0010105
	s_add_f32 s2, s81, s2                                      // 00000000CA8C: A0020251
	v_readlane_b32 s91, v21, 23                                // 00000000CA90: D760005B 00012F15
	s_wait_alu 0xfffe                                          // 00000000CA98: BF88FFFE
	s_add_f32 s0, s90, s0                                      // 00000000CA9C: A000005A
	s_add_f32 s3, s58, s3                                      // 00000000CAA0: A003033A
	s_add_f32 s1, s6, s1                                       // 00000000CAA4: A0010106
	s_add_f32 s2, s82, s2                                      // 00000000CAA8: A0020252
	s_add_f32 s28, s74, s28                                    // 00000000CAAC: A01C1C4A
	s_add_f32 s44, s40, 0                                      // 00000000CAB0: A02C8028
	s_add_f32 s41, s41, 0                                      // 00000000CAB4: A0298029
	s_add_f32 s25, s87, s23                                    // 00000000CAB8: A0191757
	s_add_f32 s23, s37, 0                                      // 00000000CABC: A0178025
	s_add_f32 s18, s38, 0                                      // 00000000CAC0: A0128026
	s_wait_alu 0xfffe                                          // 00000000CAC4: BF88FFFE
	s_add_f32 s8, s91, s0                                      // 00000000CAC8: A008005B
	s_add_f32 s4, s59, s3                                      // 00000000CACC: A004033B
	s_add_f32 s0, s7, s1                                       // 00000000CAD0: A0000107
	s_add_f32 s33, s83, s2                                     // 00000000CAD4: A0210253
	s_add_f32 s34, s34, 0                                      // 00000000CAD8: A0228022
	s_add_f32 s19, s36, 0                                      // 00000000CADC: A0138024
	s_add_co_i32 vcc_hi, vcc_hi, 1                             // 00000000CAE0: 816B816B
	s_add_f32 s9, s16, 0                                       // 00000000CAE4: A0098010
	s_cmp_eq_u32 vcc_hi, 8                                     // 00000000CAE8: BF06886B
	s_wait_alu 0xfffe                                          // 00000000CAEC: BF88FFFE
	v_writelane_b32 v23, s0, 16                                // 00000000CAF0: D7610017 00012000
	s_cbranch_scc1 145                                         // 00000000CAF8: BFA20091 <r_3_3_3_8_8_8+0xb740>
	v_readlane_b32 s92, v21, 12                                // 00000000CAFC: D760005C 00011915
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000CB04: BF870002
	v_readlane_b32 s12, v23, 24                                // 00000000CB08: D760000C 00013117
	v_readlane_b32 s14, v23, 26                                // 00000000CB10: D760000E 00013517
	v_readlane_b32 s15, v23, 27                                // 00000000CB18: D760000F 00013717
	v_readlane_b32 s13, v23, 25                                // 00000000CB20: D760000D 00013317
	s_mov_b32 s40, s92                                         // 00000000CB28: BEA8005C
	v_readlane_b32 s93, v21, 13                                // 00000000CB2C: D760005D 00011B15
	v_readlane_b32 s94, v21, 14                                // 00000000CB34: D760005E 00011D15
	v_readlane_b32 s95, v21, 15                                // 00000000CB3C: D760005F 00011F15
	s_or_saveexec_b32 s105, -1                                 // 00000000CB44: BEE922C1
	scratch_load_b32 v29, off, off offset:12 th:TH_LOAD_LU     // 00000000CB48: ED05007C 0030001D 00000C00
	s_wait_alu 0xfffe                                          // 00000000CB54: BF88FFFE
	s_mov_b32 exec_lo, s105                                    // 00000000CB58: BEFE0069
	s_wait_loadcnt 0x0                                         // 00000000CB5C: BFC00000
	v_readlane_b32 s10, v29, 13                                // 00000000CB60: D760000A 00011B1D
	v_readlane_b32 s11, v29, 14                                // 00000000CB68: D760000B 00011D1D
	s_branch 54167                                             // 00000000CB70: BFA0D397 <r_3_3_3_8_8_8+0x3d0>
	v_writelane_b32 v21, s0, 12                                // 00000000CB74: D7610015 00011800
	s_mov_b32 s104, 0                                          // 00000000CB7C: BEE80080
	v_writelane_b32 v21, s1, 13                                // 00000000CB80: D7610015 00011A01
	v_writelane_b32 v21, s2, 14                                // 00000000CB88: D7610015 00011C02
	v_writelane_b32 v21, s3, 15                                // 00000000CB90: D7610015 00011E03
	v_writelane_b32 v21, s4, 16                                // 00000000CB98: D7610015 00012004
	v_writelane_b32 v21, s5, 17                                // 00000000CBA0: D7610015 00012205
	v_writelane_b32 v21, s6, 18                                // 00000000CBA8: D7610015 00012406
	v_writelane_b32 v21, s7, 19                                // 00000000CBB0: D7610015 00012607
	v_writelane_b32 v21, s8, 20                                // 00000000CBB8: D7610015 00012808
	v_writelane_b32 v21, s9, 21                                // 00000000CBC0: D7610015 00012A09
	v_writelane_b32 v21, s10, 22                               // 00000000CBC8: D7610015 00012C0A
	v_writelane_b32 v21, s11, 23                               // 00000000CBD0: D7610015 00012E0B
	v_writelane_b32 v21, s12, 24                               // 00000000CBD8: D7610015 0001300C
	v_writelane_b32 v21, s13, 25                               // 00000000CBE0: D7610015 0001320D
	v_writelane_b32 v21, s14, 26                               // 00000000CBE8: D7610015 0001340E
	v_writelane_b32 v21, s15, 27                               // 00000000CBF0: D7610015 0001360F
	v_writelane_b32 v21, s4, 28                                // 00000000CBF8: D7610015 00013804
	v_writelane_b32 v24, s8, 0                                 // 00000000CC00: D7610018 00010008
	v_writelane_b32 v21, s5, 29                                // 00000000CC08: D7610015 00013A05
	v_writelane_b32 v24, s9, 1                                 // 00000000CC10: D7610018 00010209
	v_writelane_b32 v21, s6, 30                                // 00000000CC18: D7610015 00013C06
	v_writelane_b32 v24, s10, 2                                // 00000000CC20: D7610018 0001040A
	v_writelane_b32 v21, s7, 31                                // 00000000CC28: D7610015 00013E07
	v_writelane_b32 v24, s11, 3                                // 00000000CC30: D7610018 0001060B
	v_writelane_b32 v24, s4, 4                                 // 00000000CC38: D7610018 00010804
	v_writelane_b32 v24, s5, 5                                 // 00000000CC40: D7610018 00010A05
	v_writelane_b32 v24, s6, 6                                 // 00000000CC48: D7610018 00010C06
	v_writelane_b32 v24, s7, 7                                 // 00000000CC50: D7610018 00010E07
	v_writelane_b32 v24, s8, 8                                 // 00000000CC58: D7610018 00011008
	v_writelane_b32 v24, s9, 9                                 // 00000000CC60: D7610018 00011209
	v_writelane_b32 v24, s10, 10                               // 00000000CC68: D7610018 0001140A
	v_writelane_b32 v24, s11, 11                               // 00000000CC70: D7610018 0001160B
	v_writelane_b32 v24, s4, 12                                // 00000000CC78: D7610018 00011804
	v_writelane_b32 v24, s5, 13                                // 00000000CC80: D7610018 00011A05
	v_writelane_b32 v24, s6, 14                                // 00000000CC88: D7610018 00011C06
	v_writelane_b32 v24, s7, 15                                // 00000000CC90: D7610018 00011E07
	v_writelane_b32 v24, s8, 16                                // 00000000CC98: D7610018 00012008
	v_writelane_b32 v24, s9, 17                                // 00000000CCA0: D7610018 00012209
	v_writelane_b32 v24, s10, 18                               // 00000000CCA8: D7610018 0001240A
	v_writelane_b32 v24, s11, 19                               // 00000000CCB0: D7610018 0001260B
	v_writelane_b32 v24, s4, 20                                // 00000000CCB8: D7610018 00012804
	v_writelane_b32 v24, s5, 21                                // 00000000CCC0: D7610018 00012A05
	v_writelane_b32 v24, s6, 22                                // 00000000CCC8: D7610018 00012C06
	v_writelane_b32 v24, s7, 23                                // 00000000CCD0: D7610018 00012E07
	v_writelane_b32 v24, s8, 24                                // 00000000CCD8: D7610018 00013008
	v_writelane_b32 v24, s9, 25                                // 00000000CCE0: D7610018 00013209
	v_writelane_b32 v24, s10, 26                               // 00000000CCE8: D7610018 0001340A
	v_writelane_b32 v24, s11, 27                               // 00000000CCF0: D7610018 0001360B
	v_writelane_b32 v24, s0, 28                                // 00000000CCF8: D7610018 00013800
	v_writelane_b32 v27, s4, 0                                 // 00000000CD00: D761001B 00010004
	v_writelane_b32 v24, s1, 29                                // 00000000CD08: D7610018 00013A01
	v_writelane_b32 v27, s5, 1                                 // 00000000CD10: D761001B 00010205
	v_writelane_b32 v24, s2, 30                                // 00000000CD18: D7610018 00013C02
	v_writelane_b32 v27, s6, 2                                 // 00000000CD20: D761001B 00010406
	v_writelane_b32 v24, s3, 31                                // 00000000CD28: D7610018 00013E03
	v_writelane_b32 v27, s7, 3                                 // 00000000CD30: D761001B 00010607
	s_cbranch_execnz 60221                                     // 00000000CD38: BFA6EB3D <r_3_3_3_8_8_8+0x6430>
	s_branch 60388                                             // 00000000CD3C: BFA0EBE4 <r_3_3_3_8_8_8+0x66d0>
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000CD40: BF870001
	v_readlane_b32 s1, v23, 18                                 // 00000000CD44: D7600001 00012517
	v_readlane_b32 s2, v23, 17                                 // 00000000CD4C: D7600002 00012317
	s_mul_f32 s0, s31, 0x3b3f112b                              // 00000000CD54: A200FF1F 3B3F112B
	s_mul_f32 s3, s45, 0x3b272f05                              // 00000000CD5C: A203FF2D 3B272F05
	v_mov_b32_e32 v19, 0                                       // 00000000CD64: 7E260280
	s_mul_f32 s1, s1, 0x3b272f05                               // 00000000CD68: A201FF01 3B272F05
	s_wait_alu 0xfffe                                          // 00000000CD70: BF88FFFE
	v_mov_b32_e32 v0, s0                                       // 00000000CD74: 7E000200
	s_mul_f32 s2, s2, 0x3b3f112b                               // 00000000CD78: A202FF02 3B3F112B
	s_mul_f32 s0, s39, 0x3b124925                              // 00000000CD80: A200FF27 3B124925
	v_mov_b32_e32 v1, s1                                       // 00000000CD88: 7E020201
	s_wait_alu 0xfffe                                          // 00000000CD8C: BF88FFFE
	v_dual_mov_b32 v3, s3 :: v_dual_mov_b32 v2, s2             // 00000000CD90: CA100003 03020002
	v_mov_b32_e32 v4, s0                                       // 00000000CD98: 7E080200
	s_mul_f32 s1, s4, 0x3b272f05                               // 00000000CD9C: A201FF04 3B272F05
	s_mul_f32 s2, s44, 0x3b3f112b                              // 00000000CDA4: A202FF2C 3B3F112B
	s_mul_f32 s3, s23, 0x3b272f05                              // 00000000CDAC: A203FF17 3B272F05
	v_readlane_b32 s4, v23, 24                                 // 00000000CDB4: D7600004 00013117
	v_readlane_b32 s5, v23, 25                                 // 00000000CDBC: D7600005 00013317
	s_wait_alu 0xfffe                                          // 00000000CDC4: BF88FFFE
	v_dual_mov_b32 v5, s1 :: v_dual_mov_b32 v6, s2             // 00000000CDC8: CA100001 05060002
	s_mul_f32 s0, s34, 0x3b3f112b                              // 00000000CDD0: A200FF22 3B3F112B
	v_mov_b32_e32 v7, s3                                       // 00000000CDD8: 7E0E0203
	s_mul_f32 s1, s42, 0x3b272f05                              // 00000000CDDC: A201FF2A 3B272F05
	s_mul_f32 s2, s29, 0x3b124925                              // 00000000CDE4: A202FF1D 3B124925
	s_clause 0x1                                               // 00000000CDEC: BF850001
	global_store_b128 v19, v[0:3], s[4:5]                      // 00000000CDF0: EE074004 00000000 00000013
	global_store_b128 v19, v[4:7], s[4:5] offset:16            // 00000000CDFC: EE074004 02000000 00001013
	s_wait_alu 0xfffe                                          // 00000000CE08: BF88FFFE
	v_dual_mov_b32 v0, s0 :: v_dual_mov_b32 v1, s1             // 00000000CE0C: CA100000 00000001
	v_mov_b32_e32 v2, s2                                       // 00000000CE14: 7E040202
	v_readlane_b32 s0, v23, 19                                 // 00000000CE18: D7600000 00012717
	v_readlane_b32 s2, v23, 16                                 // 00000000CE20: D7600002 00012117
	s_mul_f32 s3, s17, 0x3b272f05                              // 00000000CE28: A203FF11 3B272F05
	s_mul_f32 s1, s24, 0x3b000000                              // 00000000CE30: A201FF18 3B000000
	v_readlane_b32 s6, v23, 26                                 // 00000000CE38: D7600006 00013517
	s_mul_f32 s0, s0, 0x3b124925                               // 00000000CE40: A200FF00 3B124925
	s_mul_f32 s2, s2, 0x3b124925                               // 00000000CE48: A202FF02 3B124925
	s_wait_alu 0xfffe                                          // 00000000CE50: BF88FFFE
	v_mov_b32_e32 v3, s3                                       // 00000000CE54: 7E060203
	s_mul_f32 s3, s41, 0x3b272f05                              // 00000000CE58: A203FF29 3B272F05
	v_dual_mov_b32 v4, s0 :: v_dual_mov_b32 v5, s1             // 00000000CE60: CA100000 04040001
	v_mov_b32_e32 v6, s2                                       // 00000000CE68: 7E0C0202
	s_mul_f32 s0, s18, 0x3b124925                              // 00000000CE6C: A200FF12 3B124925
	s_mul_f32 s1, s19, 0x3b272f05                              // 00000000CE74: A201FF13 3B272F05
	s_mul_f32 s2, s30, 0x3b3f112b                              // 00000000CE7C: A202FF1E 3B3F112B
	s_wait_alu 0xfffe                                          // 00000000CE84: BF88FFFE
	v_dual_mov_b32 v7, s3 :: v_dual_mov_b32 v8, s0             // 00000000CE88: CA100003 07080000
	s_mul_f32 s3, s25, 0x3b272f05                              // 00000000CE90: A203FF19 3B272F05
	v_dual_mov_b32 v9, s1 :: v_dual_mov_b32 v10, s2            // 00000000CE98: CA100001 090A0002
	s_mul_f32 s0, s8, 0x3b3f112b                               // 00000000CEA0: A200FF08 3B3F112B
	s_mul_f32 s1, s28, 0x3b272f05                              // 00000000CEA8: A201FF1C 3B272F05
	s_mul_f32 s2, s22, 0x3b124925                              // 00000000CEB0: A202FF16 3B124925
	s_wait_alu 0xfffe                                          // 00000000CEB8: BF88FFFE
	v_dual_mov_b32 v11, s3 :: v_dual_mov_b32 v12, s0           // 00000000CEBC: CA100003 0B0C0000
	s_mul_f32 s3, s33, 0x3b272f05                              // 00000000CEC4: A203FF21 3B272F05
	v_dual_mov_b32 v13, s1 :: v_dual_mov_b32 v14, s2           // 00000000CECC: CA100001 0D0E0002
	s_mul_f32 s0, s27, 0x3b3f112b                              // 00000000CED4: A200FF1B 3B3F112B
	s_mul_f32 s1, s21, 0x3b272f05                              // 00000000CEDC: A201FF15 3B272F05
	s_mul_f32 s2, s9, 0x3b3f112b                               // 00000000CEE4: A202FF09 3B3F112B
	s_wait_alu 0xfffe                                          // 00000000CEEC: BF88FFFE
	v_dual_mov_b32 v15, s3 :: v_dual_mov_b32 v16, s0           // 00000000CEF0: CA100003 0F100000
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000CEF8: BF870009
	v_dual_mov_b32 v17, s1 :: v_dual_mov_b32 v18, s2           // 00000000CEFC: CA100001 11120002
	v_readlane_b32 s7, v23, 27                                 // 00000000CF04: D7600007 00013717
	s_clause 0x4                                               // 00000000CF0C: BF850004
	global_store_b128 v19, v[0:3], s[4:5] offset:32            // 00000000CF10: EE074004 00000000 00002013
	global_store_b128 v19, v[4:7], s[4:5] offset:48            // 00000000CF1C: EE074004 02000000 00003013
	global_store_b128 v19, v[8:11], s[4:5] offset:64           // 00000000CF28: EE074004 04000000 00004013
	global_store_b128 v19, v[12:15], s[4:5] offset:80          // 00000000CF34: EE074004 06000000 00005013
	global_store_b96 v19, v[16:18], s[4:5] offset:96           // 00000000CF40: EE070004 08000000 00006013
	s_endpgm                                                   // 00000000CF4C: BFB00000

.rodata
.amdhsa_kernel kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_private_segment_fixed_size 132
  .amdhsa_kernarg_size 16
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel
