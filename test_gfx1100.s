.text
.globl kernel
.p2align 8 // TODO: need more?
.type kernel,@function

kernel:

	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v20, 0             // 000000001608: CA100080 0F140080
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v10, 0             // 000000001610: CA100080 110A0080
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v0, 0              // 000000001618: CA100080 15000080
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v4, 0              // 000000001620: CA100080 19040080
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v14, 0              // 000000001628: CA100080 030E0080
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v19, 0             // 000000001630: CA100080 10120080
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v9, 0              // 000000001638: CA100080 12080080
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v1, 0              // 000000001640: CA100080 18000080
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v5, 0               // 000000001648: CA100080 02040080
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0             // 000000001650: CA100080 0C0C0080
	s_waitcnt lgkmcnt(0)                                       // 000000001658: BF89FC07
	s_add_u32 s4, s2, 0xfffffd00                               // 00000000165C: 8004FF02 FFFFFD00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001664: BF8704A9
	v_writelane_b32 v45, s4, 0                                 // 000000001668: D761002D 00010004
	s_addc_u32 s4, s3, -1                                      // 000000001670: 8204C103
	v_writelane_b32 v45, s4, 1                                 // 000000001674: D761002D 00010204
	s_add_u32 s4, s2, 0x2640                                   // 00000000167C: 8004FF02 00002640
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001684: BF8704A9
	v_writelane_b32 v45, s4, 2                                 // 000000001688: D761002D 00010404
	s_addc_u32 s4, s3, 0                                       // 000000001690: 82048003
	v_writelane_b32 v45, s4, 3                                 // 000000001694: D761002D 00010604
	s_add_u32 s4, s2, 0x2500                                   // 00000000169C: 8004FF02 00002500
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000016A4: BF8704A9
	v_writelane_b32 v45, s4, 4                                 // 0000000016A8: D761002D 00010804
	s_addc_u32 s4, s3, 0                                       // 0000000016B0: 82048003
	v_writelane_b32 v45, s4, 5                                 // 0000000016B4: D761002D 00010A04
	s_add_u32 s4, s2, 0x1240                                   // 0000000016BC: 8004FF02 00001240
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000016C4: BF8704A9
	v_writelane_b32 v45, s4, 6                                 // 0000000016C8: D761002D 00010C04
	s_addc_u32 s4, s3, 0                                       // 0000000016D0: 82048003
	v_writelane_b32 v45, s4, 7                                 // 0000000016D4: D761002D 00010E04
	s_add_u32 s4, s2, 0x1120                                   // 0000000016DC: 8004FF02 00001120
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000016E4: BF8704A9
	v_writelane_b32 v45, s4, 8                                 // 0000000016E8: D761002D 00011004
	s_addc_u32 s4, s3, 0                                       // 0000000016F0: 82048003
	v_writelane_b32 v45, s4, 9                                 // 0000000016F4: D761002D 00011204
	s_add_u32 s4, s2, 0xfffffc00                               // 0000000016FC: 8004FF02 FFFFFC00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001704: BF8704A9
	v_writelane_b32 v45, s4, 10                                // 000000001708: D761002D 00011404
	s_addc_u32 s4, s3, -1                                      // 000000001710: 8204C103
	v_writelane_b32 v45, s4, 11                                // 000000001714: D761002D 00011604
	s_add_u32 s4, s2, 0x1000                                   // 00000000171C: 8004FF02 00001000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001724: BF8704A9
	v_writelane_b32 v45, s4, 12                                // 000000001728: D761002D 00011804
	s_addc_u32 s4, s3, 0                                       // 000000001730: 82048003
	v_writelane_b32 v45, s4, 13                                // 000000001734: D761002D 00011A04
	s_add_u32 s4, s2, 0x2400                                   // 00000000173C: 8004FF02 00002400
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001744: BF8704A9
	v_writelane_b32 v45, s4, 14                                // 000000001748: D761002D 00011C04
	s_addc_u32 s4, s3, 0                                       // 000000001750: 82048003
	v_writelane_b32 v45, s4, 15                                // 000000001754: D761002D 00011E04
	s_add_u32 s4, s2, 0xfffffd40                               // 00000000175C: 8004FF02 FFFFFD40
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001764: BF8704A9
	v_writelane_b32 v45, s4, 16                                // 000000001768: D761002D 00012004
	s_addc_u32 s4, s3, -1                                      // 000000001770: 8204C103
	v_writelane_b32 v45, s4, 17                                // 000000001774: D761002D 00012204
	s_add_u32 s4, s2, 64                                       // 00000000177C: 8004C002
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001780: BF8704A9
	v_writelane_b32 v45, s4, 18                                // 000000001784: D761002D 00012404
	s_addc_u32 s4, s3, 0                                       // 00000000178C: 82048003
	v_writelane_b32 v45, s4, 19                                // 000000001790: D761002D 00012604
	s_add_u32 s4, s2, 0xfffffc40                               // 000000001798: 8004FF02 FFFFFC40
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017A0: BF8704A9
	v_writelane_b32 v45, s4, 20                                // 0000000017A4: D761002D 00012804
	s_addc_u32 s4, s3, -1                                      // 0000000017AC: 8204C103
	v_writelane_b32 v45, s4, 21                                // 0000000017B0: D761002D 00012A04
	s_add_u32 s4, s2, 0x1040                                   // 0000000017B8: 8004FF02 00001040
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017C0: BF8704A9
	v_writelane_b32 v45, s4, 22                                // 0000000017C4: D761002D 00012C04
	s_addc_u32 s4, s3, 0                                       // 0000000017CC: 82048003
	v_writelane_b32 v45, s4, 23                                // 0000000017D0: D761002D 00012E04
	s_add_u32 s4, s2, 0x2440                                   // 0000000017D8: 8004FF02 00002440
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017E0: BF8704A9
	v_writelane_b32 v45, s4, 24                                // 0000000017E4: D761002D 00013004
	s_addc_u32 s4, s3, 0                                       // 0000000017EC: 82048003
	v_writelane_b32 v45, s4, 25                                // 0000000017F0: D761002D 00013204
	s_add_u32 s4, s2, 0xfffffd80                               // 0000000017F8: 8004FF02 FFFFFD80
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001800: BF8704A9
	v_writelane_b32 v45, s4, 26                                // 000000001804: D761002D 00013404
	s_addc_u32 s4, s3, -1                                      // 00000000180C: 8204C103
	v_writelane_b32 v45, s4, 27                                // 000000001810: D761002D 00013604
	s_add_u32 s4, s2, 0x80                                     // 000000001818: 8004FF02 00000080
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001820: BF8704A9
	v_writelane_b32 v45, s4, 28                                // 000000001824: D761002D 00013804
	s_addc_u32 s4, s3, 0                                       // 00000000182C: 82048003
	v_writelane_b32 v45, s4, 29                                // 000000001830: D761002D 00013A04
	s_add_u32 s4, s2, 0xfffffc80                               // 000000001838: 8004FF02 FFFFFC80
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001840: BF8704A9
	v_writelane_b32 v45, s4, 30                                // 000000001844: D761002D 00013C04
	s_addc_u32 s4, s3, -1                                      // 00000000184C: 8204C103
	v_writelane_b32 v45, s4, 31                                // 000000001850: D761002D 00013E04
	s_or_saveexec_b32 s105, -1                                 // 000000001858: BEE922C1
	scratch_store_b32 off, v45, off                            // 00000000185C: DC690000 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000001864: BEFE0069
	s_add_u32 s4, s2, 0x1080                                   // 000000001868: 8004FF02 00001080
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001870: BF8704A9
	v_writelane_b32 v45, s4, 0                                 // 000000001874: D761002D 00010004
	s_addc_u32 s4, s3, 0                                       // 00000000187C: 82048003
	v_writelane_b32 v45, s4, 1                                 // 000000001880: D761002D 00010204
	s_add_u32 s4, s2, 0x2480                                   // 000000001888: 8004FF02 00002480
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001890: BF8704A9
	v_writelane_b32 v45, s4, 2                                 // 000000001894: D761002D 00010404
	s_addc_u32 s4, s3, 0                                       // 00000000189C: 82048003
	v_writelane_b32 v45, s4, 3                                 // 0000000018A0: D761002D 00010604
	s_add_u32 s4, s2, 0xfffffdc0                               // 0000000018A8: 8004FF02 FFFFFDC0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000018B0: BF8704A9
	v_writelane_b32 v45, s4, 4                                 // 0000000018B4: D761002D 00010804
	s_addc_u32 s4, s3, -1                                      // 0000000018BC: 8204C103
	v_writelane_b32 v45, s4, 5                                 // 0000000018C0: D761002D 00010A04
	s_add_u32 s4, s2, 0xc0                                     // 0000000018C8: 8004FF02 000000C0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000018D0: BF8704A9
	v_writelane_b32 v45, s4, 6                                 // 0000000018D4: D761002D 00010C04
	s_addc_u32 s4, s3, 0                                       // 0000000018DC: 82048003
	v_writelane_b32 v45, s4, 7                                 // 0000000018E0: D761002D 00010E04
	s_add_u32 s4, s2, 0xfffffcc0                               // 0000000018E8: 8004FF02 FFFFFCC0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000018F0: BF8704A9
	v_writelane_b32 v45, s4, 8                                 // 0000000018F4: D761002D 00011004
	s_addc_u32 s4, s3, -1                                      // 0000000018FC: 8204C103
	v_writelane_b32 v45, s4, 9                                 // 000000001900: D761002D 00011204
	s_add_u32 s4, s2, 0x10c0                                   // 000000001908: 8004FF02 000010C0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001910: BF8704A9
	v_writelane_b32 v45, s4, 10                                // 000000001914: D761002D 00011404
	s_addc_u32 s4, s3, 0                                       // 00000000191C: 82048003
	v_writelane_b32 v45, s4, 11                                // 000000001920: D761002D 00011604
	s_add_u32 s4, s2, 0x24c0                                   // 000000001928: 8004FF02 000024C0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001930: BF8704A9
	v_writelane_b32 v45, s4, 12                                // 000000001934: D761002D 00011804
	s_addc_u32 s4, s3, 0                                       // 00000000193C: 82048003
	v_writelane_b32 v45, s4, 13                                // 000000001940: D761002D 00011A04
	s_add_u32 s4, s2, 0xfffffe00                               // 000000001948: 8004FF02 FFFFFE00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001950: BF8704A9
	v_writelane_b32 v45, s4, 14                                // 000000001954: D761002D 00011C04
	s_addc_u32 s4, s3, -1                                      // 00000000195C: 8204C103
	v_writelane_b32 v45, s4, 15                                // 000000001960: D761002D 00011E04
	s_add_u32 s4, s2, 0x100                                    // 000000001968: 8004FF02 00000100
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001970: BF8704A9
	v_writelane_b32 v45, s4, 16                                // 000000001974: D761002D 00012004
	s_addc_u32 s4, s3, 0                                       // 00000000197C: 82048003
	v_writelane_b32 v45, s4, 17                                // 000000001980: D761002D 00012204
	s_add_u32 s4, s2, 0x1100                                   // 000000001988: 8004FF02 00001100
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001990: BF8704A9
	v_writelane_b32 v45, s4, 18                                // 000000001994: D761002D 00012404
	s_addc_u32 s4, s3, 0                                       // 00000000199C: 82048003
	v_writelane_b32 v45, s4, 19                                // 0000000019A0: D761002D 00012604
	s_add_u32 s4, s2, 0xfffffe40                               // 0000000019A8: 8004FF02 FFFFFE40
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000019B0: BF8704A9
	v_writelane_b32 v45, s4, 20                                // 0000000019B4: D761002D 00012804
	s_addc_u32 s4, s3, -1                                      // 0000000019BC: 8204C103
	v_writelane_b32 v45, s4, 21                                // 0000000019C0: D761002D 00012A04
	s_add_u32 s4, s2, 0x140                                    // 0000000019C8: 8004FF02 00000140
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000019D0: BF8704A9
	v_writelane_b32 v45, s4, 22                                // 0000000019D4: D761002D 00012C04
	s_addc_u32 s4, s3, 0                                       // 0000000019DC: 82048003
	v_writelane_b32 v45, s4, 23                                // 0000000019E0: D761002D 00012E04
	s_add_u32 s4, s2, 0x1140                                   // 0000000019E8: 8004FF02 00001140
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000019F0: BF8704A9
	v_writelane_b32 v45, s4, 24                                // 0000000019F4: D761002D 00013004
	s_addc_u32 s4, s3, 0                                       // 0000000019FC: 82048003
	v_writelane_b32 v45, s4, 25                                // 000000001A00: D761002D 00013204
	s_add_u32 s4, s2, 0x2540                                   // 000000001A08: 8004FF02 00002540
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001A10: BF8704A9
	v_writelane_b32 v45, s4, 26                                // 000000001A14: D761002D 00013404
	s_addc_u32 s4, s3, 0                                       // 000000001A1C: 82048003
	v_writelane_b32 v45, s4, 27                                // 000000001A20: D761002D 00013604
	s_add_u32 s4, s2, 0xfffffe80                               // 000000001A28: 8004FF02 FFFFFE80
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001A30: BF8704A9
	v_writelane_b32 v45, s4, 28                                // 000000001A34: D761002D 00013804
	s_addc_u32 s4, s3, -1                                      // 000000001A3C: 8204C103
	v_writelane_b32 v45, s4, 29                                // 000000001A40: D761002D 00013A04
	s_add_u32 s4, s2, 0x180                                    // 000000001A48: 8004FF02 00000180
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001A50: BF8704A9
	v_writelane_b32 v45, s4, 30                                // 000000001A54: D761002D 00013C04
	s_addc_u32 s4, s3, 0                                       // 000000001A5C: 82048003
	v_writelane_b32 v45, s4, 31                                // 000000001A60: D761002D 00013E04
	s_or_saveexec_b32 s105, -1                                 // 000000001A68: BEE922C1
	scratch_store_b32 off, v45, off offset:12                  // 000000001A6C: DC69000C 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000001A74: BEFE0069
	s_add_u32 s4, s2, 0x1180                                   // 000000001A78: 8004FF02 00001180
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001A80: BF8704A9
	v_writelane_b32 v45, s4, 0                                 // 000000001A84: D761002D 00010004
	s_addc_u32 s4, s3, 0                                       // 000000001A8C: 82048003
	v_writelane_b32 v45, s4, 1                                 // 000000001A90: D761002D 00010204
	s_add_u32 s4, s2, 0x2580                                   // 000000001A98: 8004FF02 00002580
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001AA0: BF8704A9
	v_writelane_b32 v45, s4, 2                                 // 000000001AA4: D761002D 00010404
	s_addc_u32 s4, s3, 0                                       // 000000001AAC: 82048003
	v_writelane_b32 v45, s4, 3                                 // 000000001AB0: D761002D 00010604
	s_add_u32 s4, s2, 0xfffffec0                               // 000000001AB8: 8004FF02 FFFFFEC0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001AC0: BF8704A9
	v_writelane_b32 v45, s4, 4                                 // 000000001AC4: D761002D 00010804
	s_addc_u32 s4, s3, -1                                      // 000000001ACC: 8204C103
	v_writelane_b32 v45, s4, 5                                 // 000000001AD0: D761002D 00010A04
	s_add_u32 s4, s2, 0x1c0                                    // 000000001AD8: 8004FF02 000001C0
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001AE0: BF870009
	v_writelane_b32 v45, s4, 6                                 // 000000001AE4: D761002D 00010C04
	v_writelane_b32 v45, s0, 7                                 // 000000001AEC: D761002D 00010E00
	v_writelane_b32 v45, s1, 8                                 // 000000001AF4: D761002D 00011001
	v_writelane_b32 v45, s2, 9                                 // 000000001AFC: D761002D 00011202
	v_writelane_b32 v45, s3, 10                                // 000000001B04: D761002D 00011403
	s_addc_u32 s0, s3, 0                                       // 000000001B0C: 82008003
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B10: BF870009
	v_writelane_b32 v45, s0, 11                                // 000000001B14: D761002D 00011600
	s_or_saveexec_b32 s105, -1                                 // 000000001B1C: BEE922C1
	scratch_store_b32 off, v45, off offset:8                   // 000000001B20: DC690008 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000001B28: BEFE0069
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0             // 000000001B2C: CA100080 16160080
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v26, 0             // 000000001B34: CA100080 0B1A0080
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v6, 0               // 000000001B3C: CA100080 07060080
	v_mov_b32_e32 v8, 0                                        // 000000001B44: 7E100280
	s_mov_b32 s100, 0                                          // 000000001B48: BEE40080
	s_mov_b32 s31, 0                                           // 000000001B4C: BE9F0080
	s_mov_b32 s104, 0                                          // 000000001B50: BEE80080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001B54: BF870009
	s_cmp_lg_u32 s104, 0                                       // 000000001B58: BF078068
	s_cselect_b32 s4, -1, 0                                    // 000000001B5C: 980480C1
	s_cmp_lg_u32 s104, 7                                       // 000000001B60: BF078768
	s_cselect_b32 vcc_hi, -1, 0                                // 000000001B64: 986B80C1
	s_lshl_b32 s30, s104, 8                                    // 000000001B68: 841E8868
	s_cmp_eq_u32 s104, 0                                       // 000000001B6C: BF068068
	s_cbranch_scc1 3511                                        // 000000001B70: BFA20DB7 <r_3_3_3_8_8_8+0x3c50>
	s_lshl_b64 s[0:1], s[30:31], 2                             // 000000001B74: 8480821E
	s_or_saveexec_b32 s105, -1                                 // 000000001B78: BEE922C1
	scratch_load_b32 v45, off, off offset:8                    // 000000001B7C: DC510008 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000001B84: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000001B88: BF8903F7
	v_readlane_b32 s8, v45, 7                                  // 000000001B8C: D7600008 00010F2D
	v_readlane_b32 s10, v45, 9                                 // 000000001B94: D760000A 0001132D
	v_readlane_b32 s11, v45, 10                                // 000000001B9C: D760000B 0001152D
	v_readlane_b32 s9, v45, 8                                  // 000000001BA4: D7600009 0001112D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001BAC: BF870113
	s_add_u32 s0, s10, s0                                      // 000000001BB0: 8000000A
	s_addc_u32 s1, s11, s1                                     // 000000001BB4: 8201010B
	s_add_u32 s2, s0, 0xfffffd00                               // 000000001BB8: 8002FF00 FFFFFD00
	s_load_b256 s[8:15], s[0:1], 0x1100                        // 000000001BC0: F40C0200 F8001100
	s_addc_u32 s3, s1, -1                                      // 000000001BC8: 8203C101
	s_add_u32 s6, s0, 0xfffffe40                               // 000000001BCC: 8006FF00 FFFFFE40
	s_load_b512 s[52:67], s[2:3], null                         // 000000001BD4: F4100D01 F8000000
	s_addc_u32 s7, s1, -1                                      // 000000001BDC: 8207C101
	s_waitcnt lgkmcnt(0)                                       // 000000001BE0: BF89FC07
	v_writelane_b32 v45, s8, 12                                // 000000001BE4: D761002D 00011808
	v_writelane_b32 v45, s9, 13                                // 000000001BEC: D761002D 00011A09
	v_writelane_b32 v45, s10, 14                               // 000000001BF4: D761002D 00011C0A
	v_writelane_b32 v45, s11, 15                               // 000000001BFC: D761002D 00011E0B
	v_writelane_b32 v45, s12, 16                               // 000000001C04: D761002D 0001200C
	v_writelane_b32 v45, s13, 17                               // 000000001C0C: D761002D 0001220D
	v_writelane_b32 v45, s14, 18                               // 000000001C14: D761002D 0001240E
	v_writelane_b32 v45, s15, 19                               // 000000001C1C: D761002D 0001260F
	s_or_saveexec_b32 s105, -1                                 // 000000001C24: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001C28: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000001C2C: BEFE0069
	s_load_b512 s[36:51], s[6:7], null                         // 000000001C30: F4100903 F8000000
	s_mov_b32 s8, 0                                            // 000000001C38: BE880080
	s_mov_b32 s0, vcc_hi                                       // 000000001C3C: BE80006B
	s_cbranch_execnz 57                                        // 000000001C40: BFA60039 <r_3_3_3_8_8_8+0x728>
	s_lshl_b64 s[0:1], s[30:31], 2                             // 000000001C44: 8480821E
	s_or_saveexec_b32 s105, -1                                 // 000000001C48: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)// 000000001C4C: BF8701D9
	s_mov_b32 exec_lo, s105                                    // 000000001C50: BEFE0069
	v_readlane_b32 s8, v45, 7                                  // 000000001C54: D7600008 00010F2D
	v_readlane_b32 s10, v45, 9                                 // 000000001C5C: D760000A 0001132D
	v_readlane_b32 s11, v45, 10                                // 000000001C64: D760000B 0001152D
	v_readlane_b32 s9, v45, 8                                  // 000000001C6C: D7600009 0001112D
	s_add_u32 s0, s10, s0                                      // 000000001C74: 8000000A
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001C78: BF870002
	s_addc_u32 s1, s11, s1                                     // 000000001C7C: 8201010B
	s_load_b256 s[8:15], s[0:1], 0x1100                        // 000000001C80: F40C0200 F8001100
	s_waitcnt lgkmcnt(0)                                       // 000000001C88: BF89FC07
	v_writelane_b32 v45, s8, 12                                // 000000001C8C: D761002D 00011808
	v_writelane_b32 v45, s9, 13                                // 000000001C94: D761002D 00011A09
	v_writelane_b32 v45, s10, 14                               // 000000001C9C: D761002D 00011C0A
	v_writelane_b32 v45, s11, 15                               // 000000001CA4: D761002D 00011E0B
	v_writelane_b32 v45, s12, 16                               // 000000001CAC: D761002D 0001200C
	v_writelane_b32 v45, s13, 17                               // 000000001CB4: D761002D 0001220D
	v_writelane_b32 v45, s14, 18                               // 000000001CBC: D761002D 0001240E
	v_writelane_b32 v45, s15, 19                               // 000000001CC4: D761002D 0001260F
	s_or_saveexec_b32 s105, -1                                 // 000000001CCC: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001CD0: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000001CD4: BEFE0069
	s_mov_b32 s101, s100                                       // 000000001CD8: BEE50064
	s_mov_b32 s102, s100                                       // 000000001CDC: BEE60064
	s_mov_b32 s103, s100                                       // 000000001CE0: BEE70064
	s_mov_b64 s[64:65], s[100:101]                             // 000000001CE4: BEC00164
	s_mov_b64 s[44:45], s[100:101]                             // 000000001CE8: BEAC0164
	s_mov_b64 s[56:57], s[100:101]                             // 000000001CEC: BEB80164
	s_mov_b64 s[52:53], s[100:101]                             // 000000001CF0: BEB40164
	s_mov_b64 s[36:37], s[100:101]                             // 000000001CF4: BEA40164
	s_mov_b64 s[40:41], s[100:101]                             // 000000001CF8: BEA80164
	s_mov_b64 s[60:61], s[100:101]                             // 000000001CFC: BEBC0164
	s_mov_b64 s[48:49], s[100:101]                             // 000000001D00: BEB00164
	s_mov_b32 s0, -1                                           // 000000001D04: BE8000C1
	s_mov_b64 s[66:67], s[102:103]                             // 000000001D08: BEC20166
	s_mov_b64 s[46:47], s[102:103]                             // 000000001D0C: BEAE0166
	s_mov_b64 s[58:59], s[102:103]                             // 000000001D10: BEBA0166
	s_mov_b64 s[54:55], s[102:103]                             // 000000001D14: BEB60166
	s_mov_b64 s[38:39], s[102:103]                             // 000000001D18: BEA60166
	s_mov_b64 s[42:43], s[102:103]                             // 000000001D1C: BEAA0166
	s_mov_b64 s[62:63], s[102:103]                             // 000000001D20: BEBE0166
	s_mov_b64 s[50:51], s[102:103]                             // 000000001D24: BEB20166
	s_or_saveexec_b32 s105, -1                                 // 000000001D28: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001D2C: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000001D30: BEFE0069
	v_writelane_b32 v45, s52, 20                               // 000000001D34: D761002D 00012834
	v_writelane_b32 v45, s53, 21                               // 000000001D3C: D761002D 00012A35
	v_writelane_b32 v45, s54, 22                               // 000000001D44: D761002D 00012C36
	v_writelane_b32 v45, s55, 23                               // 000000001D4C: D761002D 00012E37
	v_writelane_b32 v45, s56, 24                               // 000000001D54: D761002D 00013038
	v_writelane_b32 v45, s57, 25                               // 000000001D5C: D761002D 00013239
	v_writelane_b32 v45, s58, 26                               // 000000001D64: D761002D 0001343A
	v_writelane_b32 v45, s59, 27                               // 000000001D6C: D761002D 0001363B
	v_writelane_b32 v45, s60, 28                               // 000000001D74: D761002D 0001383C
	v_writelane_b32 v45, s61, 29                               // 000000001D7C: D761002D 00013A3D
	v_writelane_b32 v45, s62, 30                               // 000000001D84: D761002D 00013C3E
	v_writelane_b32 v45, s63, 31                               // 000000001D8C: D761002D 00013E3F
	s_or_saveexec_b32 s105, -1                                 // 000000001D94: BEE922C1
	scratch_store_b32 off, v45, off offset:8                   // 000000001D98: DC690008 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000001DA0: BEFE0069
	v_writelane_b32 v44, s64, 0                                // 000000001DA4: D761002C 00010040
	s_mov_b32 s72, s8                                          // 000000001DAC: BEC80008
	s_mov_b32 s73, s8                                          // 000000001DB0: BEC90008
	s_mov_b32 s74, s8                                          // 000000001DB4: BECA0008
	s_mov_b32 s75, s8                                          // 000000001DB8: BECB0008
	v_writelane_b32 v44, s65, 1                                // 000000001DBC: D761002C 00010241
	s_mov_b32 s68, s8                                          // 000000001DC4: BEC40008
	s_mov_b32 s69, s8                                          // 000000001DC8: BEC50008
	s_mov_b32 s70, s8                                          // 000000001DCC: BEC60008
	s_mov_b32 s71, s8                                          // 000000001DD0: BEC70008
	v_writelane_b32 v44, s66, 2                                // 000000001DD4: D761002C 00010442
	s_mov_b32 s76, s8                                          // 000000001DDC: BECC0008
	s_mov_b32 s77, s8                                          // 000000001DE0: BECD0008
	s_mov_b32 s78, s8                                          // 000000001DE4: BECE0008
	s_mov_b32 s79, s8                                          // 000000001DE8: BECF0008
	v_writelane_b32 v44, s67, 3                                // 000000001DEC: D761002C 00010643
	s_mov_b32 s64, s8                                          // 000000001DF4: BEC00008
	s_mov_b32 s65, s8                                          // 000000001DF8: BEC10008
	s_mov_b32 s66, s8                                          // 000000001DFC: BEC20008
	s_mov_b32 s67, s8                                          // 000000001E00: BEC30008
	v_writelane_b32 v44, s64, 4                                // 000000001E04: D761002C 00010840
	s_mov_b32 s52, s8                                          // 000000001E0C: BEB40008
	s_mov_b32 s60, s8                                          // 000000001E10: BEBC0008
	s_mov_b32 s61, s8                                          // 000000001E14: BEBD0008
	s_mov_b32 s62, s8                                          // 000000001E18: BEBE0008
	v_writelane_b32 v44, s65, 5                                // 000000001E1C: D761002C 00010A41
	s_mov_b32 s63, s8                                          // 000000001E24: BEBF0008
	s_mov_b32 s53, s8                                          // 000000001E28: BEB50008
	s_mov_b32 s54, s8                                          // 000000001E2C: BEB60008
	s_mov_b32 s55, s8                                          // 000000001E30: BEB70008
	v_writelane_b32 v44, s66, 6                                // 000000001E34: D761002C 00010C42
	s_mov_b32 s56, s8                                          // 000000001E3C: BEB80008
	s_mov_b32 s57, s8                                          // 000000001E40: BEB90008
	s_mov_b32 s58, s8                                          // 000000001E44: BEBA0008
	s_mov_b32 s59, s8                                          // 000000001E48: BEBB0008
	v_writelane_b32 v44, s67, 7                                // 000000001E4C: D761002C 00010E43
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 000000001E54: 916A007E
	v_writelane_b32 v44, s68, 8                                // 000000001E58: D761002C 00011044
	v_writelane_b32 v44, s69, 9                                // 000000001E60: D761002C 00011245
	v_writelane_b32 v44, s70, 10                               // 000000001E68: D761002C 00011446
	v_writelane_b32 v44, s71, 11                               // 000000001E70: D761002C 00011647
	v_writelane_b32 v44, s72, 12                               // 000000001E78: D761002C 00011848
	v_writelane_b32 v44, s73, 13                               // 000000001E80: D761002C 00011A49
	v_writelane_b32 v44, s74, 14                               // 000000001E88: D761002C 00011C4A
	v_writelane_b32 v44, s75, 15                               // 000000001E90: D761002C 00011E4B
	v_writelane_b32 v44, s76, 16                               // 000000001E98: D761002C 0001204C
	v_writelane_b32 v44, s77, 17                               // 000000001EA0: D761002C 0001224D
	v_writelane_b32 v44, s78, 18                               // 000000001EA8: D761002C 0001244E
	v_writelane_b32 v44, s79, 19                               // 000000001EB0: D761002C 0001264F
	s_mov_b32 s64, s8                                          // 000000001EB8: BEC00008
	s_mov_b32 s65, s8                                          // 000000001EBC: BEC10008
	s_mov_b32 s66, s8                                          // 000000001EC0: BEC20008
	s_mov_b32 s67, s8                                          // 000000001EC4: BEC30008
	v_writelane_b32 v44, s52, 20                               // 000000001EC8: D761002C 00012834
	v_writelane_b32 v44, s53, 21                               // 000000001ED0: D761002C 00012A35
	v_writelane_b32 v44, s54, 22                               // 000000001ED8: D761002C 00012C36
	v_writelane_b32 v44, s55, 23                               // 000000001EE0: D761002C 00012E37
	v_writelane_b32 v44, s56, 24                               // 000000001EE8: D761002C 00013038
	v_writelane_b32 v44, s57, 25                               // 000000001EF0: D761002C 00013239
	v_writelane_b32 v44, s58, 26                               // 000000001EF8: D761002C 0001343A
	v_writelane_b32 v44, s59, 27                               // 000000001F00: D761002C 0001363B
	v_writelane_b32 v44, s60, 28                               // 000000001F08: D761002C 0001383C
	v_writelane_b32 v44, s61, 29                               // 000000001F10: D761002C 00013A3D
	v_writelane_b32 v44, s62, 30                               // 000000001F18: D761002C 00013C3E
	v_writelane_b32 v44, s63, 31                               // 000000001F20: D761002C 00013E3F
	s_or_saveexec_b32 s105, -1                                 // 000000001F28: BEE922C1
	scratch_store_b32 off, v44, off offset:72                  // 000000001F2C: DC690048 007C2C00
	s_mov_b32 exec_lo, s105                                    // 000000001F34: BEFE0069
	v_writelane_b32 v41, s64, 0                                // 000000001F38: D7610029 00010040
	v_writelane_b32 v41, s65, 1                                // 000000001F40: D7610029 00010241
	v_writelane_b32 v41, s66, 2                                // 000000001F48: D7610029 00010442
	v_writelane_b32 v41, s67, 3                                // 000000001F50: D7610029 00010643
	s_cbranch_vccnz 98                                         // 000000001F58: BFA40062 <r_3_3_3_8_8_8+0xae4>
	s_lshl_b64 s[0:1], s[30:31], 2                             // 000000001F5C: 8480821E
	s_or_saveexec_b32 s105, -1                                 // 000000001F60: BEE922C1
	scratch_load_b32 v45, off, off                             // 000000001F64: DC510000 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000001F6C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000001F70: BF8903F7
	v_readlane_b32 s2, v45, 4                                  // 000000001F74: D7600002 0001092D
	v_readlane_b32 s3, v45, 5                                  // 000000001F7C: D7600003 00010B2D
	v_readlane_b32 s5, v45, 2                                  // 000000001F84: D7600005 0001052D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001F8C: BF870113
	s_add_u32 s2, s2, s0                                       // 000000001F90: 80020002
	s_addc_u32 s3, s3, s1                                      // 000000001F94: 82030103
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001F98: BF8700B1
	s_add_u32 s0, s5, s0                                       // 000000001F9C: 80000005
	s_load_b512 s[52:67], s[2:3], null                         // 000000001FA0: F4100D01 F8000000
	v_readlane_b32 s5, v45, 3                                  // 000000001FA8: D7600005 0001072D
	s_addc_u32 s1, s5, s1                                      // 000000001FB0: 82010105
	s_or_saveexec_b32 s105, -1                                 // 000000001FB4: BEE922C1
	scratch_load_b32 v45, off, off offset:72                   // 000000001FB8: DC510048 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000001FC0: BEFE0069
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000001FC4: BF890007
	v_writelane_b32 v45, s52, 4                                // 000000001FC8: D761002D 00010834
	v_writelane_b32 v45, s53, 5                                // 000000001FD0: D761002D 00010A35
	v_writelane_b32 v45, s54, 6                                // 000000001FD8: D761002D 00010C36
	v_writelane_b32 v45, s55, 7                                // 000000001FE0: D761002D 00010E37
	v_writelane_b32 v45, s56, 8                                // 000000001FE8: D761002D 00011038
	v_writelane_b32 v45, s57, 9                                // 000000001FF0: D761002D 00011239
	v_writelane_b32 v45, s58, 10                               // 000000001FF8: D761002D 0001143A
	v_writelane_b32 v45, s59, 11                               // 000000002000: D761002D 0001163B
	v_writelane_b32 v45, s60, 12                               // 000000002008: D761002D 0001183C
	v_writelane_b32 v45, s61, 13                               // 000000002010: D761002D 00011A3D
	v_writelane_b32 v45, s62, 14                               // 000000002018: D761002D 00011C3E
	v_writelane_b32 v45, s63, 15                               // 000000002020: D761002D 00011E3F
	v_writelane_b32 v45, s64, 16                               // 000000002028: D761002D 00012040
	v_writelane_b32 v45, s65, 17                               // 000000002030: D761002D 00012241
	v_writelane_b32 v45, s66, 18                               // 000000002038: D761002D 00012442
	v_writelane_b32 v45, s67, 19                               // 000000002040: D761002D 00012643
	s_load_b512 s[52:67], s[0:1], null                         // 000000002048: F4100D00 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000002050: BF89FC07
	v_writelane_b32 v45, s52, 20                               // 000000002054: D761002D 00012834
	v_writelane_b32 v45, s53, 21                               // 00000000205C: D761002D 00012A35
	v_writelane_b32 v45, s54, 22                               // 000000002064: D761002D 00012C36
	v_writelane_b32 v45, s55, 23                               // 00000000206C: D761002D 00012E37
	v_writelane_b32 v45, s56, 24                               // 000000002074: D761002D 00013038
	v_writelane_b32 v45, s57, 25                               // 00000000207C: D761002D 00013239
	v_writelane_b32 v45, s58, 26                               // 000000002084: D761002D 0001343A
	v_writelane_b32 v45, s59, 27                               // 00000000208C: D761002D 0001363B
	v_writelane_b32 v45, s60, 28                               // 000000002094: D761002D 0001383C
	v_writelane_b32 v45, s61, 29                               // 00000000209C: D761002D 00013A3D
	v_writelane_b32 v45, s62, 30                               // 0000000020A4: D761002D 00013C3E
	v_writelane_b32 v45, s63, 31                               // 0000000020AC: D761002D 00013E3F
	s_or_saveexec_b32 s105, -1                                 // 0000000020B4: BEE922C1
	scratch_store_b32 off, v45, off offset:72                  // 0000000020B8: DC690048 007C2D00
	s_mov_b32 exec_lo, s105                                    // 0000000020C0: BEFE0069
	v_writelane_b32 v41, s64, 0                                // 0000000020C4: D7610029 00010040
	v_writelane_b32 v41, s65, 1                                // 0000000020CC: D7610029 00010241
	v_writelane_b32 v41, s66, 2                                // 0000000020D4: D7610029 00010442
	v_writelane_b32 v41, s67, 3                                // 0000000020DC: D7610029 00010643
	s_lshl_b64 s[28:29], s[30:31], 2                           // 0000000020E4: 849C821E
	v_cndmask_b32_e64 v27, 0, 1, s4                            // 0000000020E8: D501001B 00110280
	v_cndmask_b32_e64 v28, 0, 1, vcc_hi                        // 0000000020F0: D501001C 01AD0280
	s_or_saveexec_b32 s105, -1                                 // 0000000020F8: BEE922C1
	scratch_load_b32 v45, off, off                             // 0000000020FC: DC510000 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000002104: BEFE0069
	s_waitcnt lgkmcnt(0)                                       // 000000002108: BF89FC07
	v_writelane_b32 v41, s36, 4                                // 00000000210C: D7610029 00010824
	s_waitcnt vmcnt(0)                                         // 000000002114: BF8903F7
	v_readlane_b32 s0, v45, 8                                  // 000000002118: D7600000 0001112D
	v_readlane_b32 s1, v45, 9                                  // 000000002120: D7600001 0001132D
	v_readlane_b32 s2, v45, 6                                  // 000000002128: D7600002 00010D2D
	v_readlane_b32 s3, v45, 7                                  // 000000002130: D7600003 00010F2D
	v_writelane_b32 v41, s37, 5                                // 000000002138: D7610029 00010A25
	s_add_u32 s0, s0, s28                                      // 000000002140: 80001C00
	v_cmp_ne_u32_e64 s34, 1, v27                               // 000000002144: D44D0022 00023681
	v_cmp_ne_u32_e64 s33, 1, v28                               // 00000000214C: D44D0021 00023881
	s_addc_u32 s1, s1, s29                                     // 000000002154: 82011D01
	v_writelane_b32 v41, s38, 6                                // 000000002158: D7610029 00010C26
	s_add_u32 s2, s2, s28                                      // 000000002160: 80021C02
	s_addc_u32 s3, s3, s29                                     // 000000002164: 82031D03
	s_and_not1_b32 vcc_lo, exec_lo, s4                         // 000000002168: 916A047E
	v_writelane_b32 v41, s39, 7                                // 00000000216C: D7610029 00010E27
	v_writelane_b32 v41, s40, 8                                // 000000002174: D7610029 00011028
	v_writelane_b32 v41, s41, 9                                // 00000000217C: D7610029 00011229
	v_writelane_b32 v41, s42, 10                               // 000000002184: D7610029 0001142A
	v_writelane_b32 v41, s43, 11                               // 00000000218C: D7610029 0001162B
	v_writelane_b32 v41, s44, 12                               // 000000002194: D7610029 0001182C
	v_writelane_b32 v41, s45, 13                               // 00000000219C: D7610029 00011A2D
	v_writelane_b32 v41, s46, 14                               // 0000000021A4: D7610029 00011C2E
	v_writelane_b32 v41, s47, 15                               // 0000000021AC: D7610029 00011E2F
	v_writelane_b32 v41, s48, 16                               // 0000000021B4: D7610029 00012030
	v_writelane_b32 v41, s49, 17                               // 0000000021BC: D7610029 00012231
	v_writelane_b32 v41, s50, 18                               // 0000000021C4: D7610029 00012432
	v_writelane_b32 v41, s51, 19                               // 0000000021CC: D7610029 00012633
	s_cbranch_vccnz 3128                                       // 0000000021D4: BFA40C38 <r_3_3_3_8_8_8+0x3cb8>
	v_readlane_b32 s4, v45, 10                                 // 0000000021D8: D7600004 0001152D
	v_readlane_b32 s5, v45, 13                                 // 0000000021E0: D7600005 00011B2D
	s_mov_b32 s101, s100                                       // 0000000021E8: BEE50064
	s_mov_b32 s102, s100                                       // 0000000021EC: BEE60064
	s_mov_b32 s103, s100                                       // 0000000021F0: BEE70064
	s_add_u32 s6, s4, s28                                      // 0000000021F4: 80061C04
	v_readlane_b32 s4, v45, 11                                 // 0000000021F8: D7600004 0001172D
	s_mov_b64 s[12:13], s[100:101]                             // 000000002200: BE8C0164
	s_mov_b64 s[8:9], s[100:101]                               // 000000002204: BE880164
	s_mov_b64 s[14:15], s[102:103]                             // 000000002208: BE8E0166
	s_mov_b64 s[10:11], s[102:103]                             // 00000000220C: BE8A0166
	s_addc_u32 s7, s4, s29                                     // 000000002210: 82071D04
	v_readlane_b32 s4, v45, 12                                 // 000000002214: D7600004 0001192D
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000221C: BF870001
	s_add_u32 s4, s4, s28                                      // 000000002220: 80041C04
	s_addc_u32 s5, s5, s29                                     // 000000002224: 82051D05
	s_and_b32 vcc_lo, exec_lo, s33                             // 000000002228: 8B6A217E
	s_cbranch_vccnz 9                                          // 00000000222C: BFA40009 <r_3_3_3_8_8_8+0xc54>
	v_readlane_b32 s8, v45, 14                                 // 000000002230: D7600008 00011D2D
	v_readlane_b32 s9, v45, 15                                 // 000000002238: D7600009 00011F2D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002240: BF870092
	s_add_u32 s8, s8, s28                                      // 000000002244: 80081C08
	s_addc_u32 s9, s9, s29                                     // 000000002248: 82091D09
	s_load_b256 s[8:15], s[8:9], null                          // 00000000224C: F40C0204 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000002254: BF89FC07
	v_writelane_b32 v42, s8, 4                                 // 000000002258: D761002A 00010808
	s_load_b256 s[16:23], s[4:5], null                         // 000000002260: F40C0402 F8000000
	v_readlane_b32 s4, v45, 16                                 // 000000002268: D7600004 0001212D
	v_readlane_b32 s5, v45, 19                                 // 000000002270: D7600005 0001272D
	v_writelane_b32 v42, s9, 5                                 // 000000002278: D761002A 00010A09
	v_writelane_b32 v42, s10, 6                                // 000000002280: D761002A 00010C0A
	v_writelane_b32 v42, s11, 7                                // 000000002288: D761002A 00010E0B
	v_writelane_b32 v42, s12, 8                                // 000000002290: D761002A 0001100C
	v_writelane_b32 v42, s13, 9                                // 000000002298: D761002A 0001120D
	v_writelane_b32 v42, s14, 10                               // 0000000022A0: D761002A 0001140E
	v_writelane_b32 v42, s15, 11                               // 0000000022A8: D761002A 0001160F
	s_load_b256 s[8:15], s[6:7], null                          // 0000000022B0: F40C0203 F8000000
	s_add_u32 s6, s4, s28                                      // 0000000022B8: 80061C04
	v_readlane_b32 s4, v45, 17                                 // 0000000022BC: D7600004 0001232D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000022C4: BF8700B1
	s_addc_u32 s7, s4, s29                                     // 0000000022C8: 82071D04
	v_readlane_b32 s4, v45, 18                                 // 0000000022CC: D7600004 0001252D
	s_load_b512 s[60:75], s[6:7], null                         // 0000000022D4: F4100F03 F8000000
	s_add_u32 s4, s4, s28                                      // 0000000022DC: 80041C04
	s_addc_u32 s5, s5, s29                                     // 0000000022E0: 82051D05
	s_add_u32 s6, s4, 0xfffffe40                               // 0000000022E4: 8006FF04 FFFFFE40
	s_addc_u32 s7, s5, -1                                      // 0000000022EC: 8207C105
	s_load_b512 s[44:59], s[6:7], null                         // 0000000022F0: F4100B03 F8000000
	s_waitcnt lgkmcnt(0)                                       // 0000000022F8: BF89FC07
	v_writelane_b32 v43, s8, 28                                // 0000000022FC: D761002B 00013808
	v_writelane_b32 v42, s12, 0                                // 000000002304: D761002A 0001000C
	s_mov_b32 s6, vcc_hi                                       // 00000000230C: BE86006B
	v_writelane_b32 v43, s9, 29                                // 000000002310: D761002B 00013A09
	v_writelane_b32 v42, s13, 1                                // 000000002318: D761002A 0001020D
	v_writelane_b32 v41, s60, 20                               // 000000002320: D7610029 0001283C
	v_writelane_b32 v43, s10, 30                               // 000000002328: D761002B 00013C0A
	v_writelane_b32 v42, s14, 2                                // 000000002330: D761002A 0001040E
	v_writelane_b32 v41, s61, 21                               // 000000002338: D7610029 00012A3D
	v_writelane_b32 v43, s11, 31                               // 000000002340: D761002B 00013E0B
	v_writelane_b32 v42, s15, 3                                // 000000002348: D761002A 0001060F
	s_load_b256 s[8:15], s[4:5], 0x1100                        // 000000002350: F40C0202 F8001100
	v_writelane_b32 v41, s62, 22                               // 000000002358: D7610029 00012C3E
	v_writelane_b32 v43, s72, 0                                // 000000002360: D761002B 00010048
	v_writelane_b32 v41, s63, 23                               // 000000002368: D7610029 00012E3F
	v_writelane_b32 v43, s73, 1                                // 000000002370: D761002B 00010249
	v_writelane_b32 v41, s64, 24                               // 000000002378: D7610029 00013040
	v_writelane_b32 v43, s74, 2                                // 000000002380: D761002B 0001044A
	v_writelane_b32 v41, s65, 25                               // 000000002388: D7610029 00013241
	v_writelane_b32 v43, s75, 3                                // 000000002390: D761002B 0001064B
	v_writelane_b32 v41, s66, 26                               // 000000002398: D7610029 00013442
	s_waitcnt lgkmcnt(0)                                       // 0000000023A0: BF89FC07
	v_writelane_b32 v43, s8, 20                                // 0000000023A4: D761002B 00012808
	v_writelane_b32 v41, s67, 27                               // 0000000023AC: D7610029 00013643
	v_writelane_b32 v43, s9, 21                                // 0000000023B4: D761002B 00012A09
	v_writelane_b32 v41, s68, 28                               // 0000000023BC: D7610029 00013844
	v_writelane_b32 v43, s10, 22                               // 0000000023C4: D761002B 00012C0A
	v_writelane_b32 v41, s69, 29                               // 0000000023CC: D7610029 00013A45
	v_writelane_b32 v43, s11, 23                               // 0000000023D4: D761002B 00012E0B
	v_writelane_b32 v41, s70, 30                               // 0000000023DC: D7610029 00013C46
	v_writelane_b32 v43, s12, 24                               // 0000000023E4: D761002B 0001300C
	v_writelane_b32 v41, s71, 31                               // 0000000023EC: D7610029 00013E47
	v_writelane_b32 v43, s13, 25                               // 0000000023F4: D761002B 0001320D
	v_writelane_b32 v43, s14, 26                               // 0000000023FC: D761002B 0001340E
	v_writelane_b32 v43, s15, 27                               // 000000002404: D761002B 0001360F
	s_add_u32 s8, s4, 0xfffffbe0                               // 00000000240C: 8008FF04 FFFFFBE0
	s_addc_u32 s9, s5, -1                                      // 000000002414: 8209C105
	s_mov_b32 s12, 0                                           // 000000002418: BE8C0080
	s_load_b256 s[36:43], s[8:9], null                         // 00000000241C: F40C0904 F8000000
	v_writelane_b32 v43, s44, 4                                // 000000002424: D761002B 0001082C
	v_writelane_b32 v43, s45, 5                                // 00000000242C: D761002B 00010A2D
	v_writelane_b32 v43, s46, 6                                // 000000002434: D761002B 00010C2E
	v_writelane_b32 v43, s47, 7                                // 00000000243C: D761002B 00010E2F
	v_writelane_b32 v43, s48, 8                                // 000000002444: D761002B 00011030
	v_writelane_b32 v43, s49, 9                                // 00000000244C: D761002B 00011231
	v_writelane_b32 v43, s50, 10                               // 000000002454: D761002B 00011432
	v_writelane_b32 v43, s51, 11                               // 00000000245C: D761002B 00011633
	v_writelane_b32 v43, s52, 12                               // 000000002464: D761002B 00011834
	v_writelane_b32 v43, s53, 13                               // 00000000246C: D761002B 00011A35
	v_writelane_b32 v43, s54, 14                               // 000000002474: D761002B 00011C36
	v_writelane_b32 v43, s55, 15                               // 00000000247C: D761002B 00011E37
	v_writelane_b32 v43, s56, 16                               // 000000002484: D761002B 00012038
	v_writelane_b32 v43, s57, 17                               // 00000000248C: D761002B 00012239
	v_writelane_b32 v43, s58, 18                               // 000000002494: D761002B 0001243A
	v_writelane_b32 v43, s59, 19                               // 00000000249C: D761002B 0001263B
	s_branch 171                                               // 0000000024A4: BFA000AB <r_3_3_3_8_8_8+0x1154>
	s_or_saveexec_b32 s105, -1                                 // 0000000024A8: BEE922C1
	scratch_load_b32 v45, off, off                             // 0000000024AC: DC510000 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 0000000024B4: BEFE0069
	s_waitcnt vmcnt(0)                                         // 0000000024B8: BF8903F7
	v_readlane_b32 s4, v45, 12                                 // 0000000024BC: D7600004 0001192D
	v_readlane_b32 s5, v45, 13                                 // 0000000024C4: D7600005 00011B2D
	v_readlane_b32 s6, v45, 14                                 // 0000000024CC: D7600006 00011D2D
	s_mov_b32 s101, s100                                       // 0000000024D4: BEE50064
	s_mov_b32 s102, s100                                       // 0000000024D8: BEE60064
	s_add_u32 s4, s4, s28                                      // 0000000024DC: 80041C04
	s_addc_u32 s5, s5, s29                                     // 0000000024E0: 82051D05
	s_add_u32 s6, s6, s28                                      // 0000000024E4: 80061C06
	s_load_b256 s[16:23], s[4:5], null                         // 0000000024E8: F40C0402 F8000000
	v_readlane_b32 s4, v45, 15                                 // 0000000024F0: D7600004 00011F2D
	v_readlane_b32 s5, v45, 19                                 // 0000000024F8: D7600005 0001272D
	s_mov_b32 s103, s100                                       // 000000002500: BEE70064
	s_mov_b64 s[88:89], s[100:101]                             // 000000002504: BED80164
	s_mov_b64 s[80:81], s[100:101]                             // 000000002508: BED00164
	s_addc_u32 s7, s4, s29                                     // 00000000250C: 82071D04
	v_readlane_b32 s4, v45, 18                                 // 000000002510: D7600004 0001252D
	s_load_b256 s[8:15], s[6:7], null                          // 000000002518: F40C0203 F8000000
	s_mov_b64 s[76:77], s[100:101]                             // 000000002520: BECC0164
	s_mov_b64 s[84:85], s[100:101]                             // 000000002524: BED40164
	s_mov_b64 s[90:91], s[102:103]                             // 000000002528: BEDA0166
	s_add_u32 s4, s4, s28                                      // 00000000252C: 80041C04
	s_addc_u32 s5, s5, s29                                     // 000000002530: 82051D05
	s_mov_b64 s[82:83], s[102:103]                             // 000000002534: BED20166
	s_mov_b64 s[78:79], s[102:103]                             // 000000002538: BECE0166
	s_mov_b64 s[86:87], s[102:103]                             // 00000000253C: BED60166
	v_writelane_b32 v41, s76, 20                               // 000000002540: D7610029 0001284C
	s_mov_b64 s[52:53], s[100:101]                             // 000000002548: BEB40164
	s_mov_b64 s[44:45], s[100:101]                             // 00000000254C: BEAC0164
	s_mov_b64 s[48:49], s[100:101]                             // 000000002550: BEB00164
	s_mov_b64 s[56:57], s[100:101]                             // 000000002554: BEB80164
	s_mov_b64 s[54:55], s[102:103]                             // 000000002558: BEB60166
	s_mov_b64 s[46:47], s[102:103]                             // 00000000255C: BEAE0166
	s_mov_b64 s[50:51], s[102:103]                             // 000000002560: BEB20166
	s_mov_b64 s[58:59], s[102:103]                             // 000000002564: BEBA0166
	v_writelane_b32 v41, s77, 21                               // 000000002568: D7610029 00012A4D
	s_waitcnt lgkmcnt(0)                                       // 000000002570: BF89FC07
	s_mov_b64 s[40:41], s[100:101]                             // 000000002574: BEA80164
	v_writelane_b32 v42, s8, 4                                 // 000000002578: D761002A 00010808
	s_mov_b64 s[36:37], s[100:101]                             // 000000002580: BEA40164
	s_mov_b32 s6, -1                                           // 000000002584: BE8600C1
	v_writelane_b32 v41, s78, 22                               // 000000002588: D7610029 00012C4E
	s_mov_b64 s[42:43], s[102:103]                             // 000000002590: BEAA0166
	v_writelane_b32 v42, s9, 5                                 // 000000002594: D761002A 00010A09
	s_mov_b64 s[38:39], s[102:103]                             // 00000000259C: BEA60166
	v_writelane_b32 v41, s79, 23                               // 0000000025A0: D7610029 00012E4F
	v_writelane_b32 v42, s10, 6                                // 0000000025A8: D761002A 00010C0A
	v_writelane_b32 v41, s80, 24                               // 0000000025B0: D7610029 00013050
	v_writelane_b32 v42, s11, 7                                // 0000000025B8: D761002A 00010E0B
	v_writelane_b32 v41, s81, 25                               // 0000000025C0: D7610029 00013251
	v_writelane_b32 v42, s12, 8                                // 0000000025C8: D761002A 0001100C
	v_writelane_b32 v41, s82, 26                               // 0000000025D0: D7610029 00013452
	v_writelane_b32 v42, s13, 9                                // 0000000025D8: D761002A 0001120D
	v_writelane_b32 v41, s83, 27                               // 0000000025E0: D7610029 00013653
	v_writelane_b32 v42, s14, 10                               // 0000000025E8: D761002A 0001140E
	v_writelane_b32 v41, s84, 28                               // 0000000025F0: D7610029 00013854
	v_writelane_b32 v42, s15, 11                               // 0000000025F8: D761002A 0001160F
	s_load_b256 s[8:15], s[4:5], 0x1100                        // 000000002600: F40C0202 F8001100
	v_writelane_b32 v41, s85, 29                               // 000000002608: D7610029 00013A55
	v_writelane_b32 v41, s86, 30                               // 000000002610: D7610029 00013C56
	v_writelane_b32 v41, s87, 31                               // 000000002618: D7610029 00013E57
	s_waitcnt lgkmcnt(0)                                       // 000000002620: BF89FC07
	v_writelane_b32 v43, s8, 20                                // 000000002624: D761002B 00012808
	v_writelane_b32 v43, s9, 21                                // 00000000262C: D761002B 00012A09
	v_writelane_b32 v43, s10, 22                               // 000000002634: D761002B 00012C0A
	v_writelane_b32 v43, s11, 23                               // 00000000263C: D761002B 00012E0B
	v_writelane_b32 v43, s12, 24                               // 000000002644: D761002B 0001300C
	v_writelane_b32 v43, s13, 25                               // 00000000264C: D761002B 0001320D
	v_writelane_b32 v43, s14, 26                               // 000000002654: D761002B 0001340E
	v_writelane_b32 v43, s15, 27                               // 00000000265C: D761002B 0001360F
	s_mov_b64 s[12:13], s[100:101]                             // 000000002664: BE8C0164
	s_mov_b64 s[8:9], s[100:101]                               // 000000002668: BE880164
	s_mov_b64 s[14:15], s[102:103]                             // 00000000266C: BE8E0166
	s_mov_b64 s[10:11], s[102:103]                             // 000000002670: BE8A0166
	v_writelane_b32 v43, s8, 28                                // 000000002674: D761002B 00013808
	v_writelane_b32 v42, s12, 0                                // 00000000267C: D761002A 0001000C
	v_writelane_b32 v43, s9, 29                                // 000000002684: D761002B 00013A09
	v_writelane_b32 v42, s13, 1                                // 00000000268C: D761002A 0001020D
	v_writelane_b32 v43, s10, 30                               // 000000002694: D761002B 00013C0A
	v_writelane_b32 v42, s14, 2                                // 00000000269C: D761002A 0001040E
	v_writelane_b32 v43, s11, 31                               // 0000000026A4: D761002B 00013E0B
	v_writelane_b32 v42, s15, 3                                // 0000000026AC: D761002A 0001060F
	v_writelane_b32 v43, s88, 0                                // 0000000026B4: D761002B 00010058
	v_writelane_b32 v43, s89, 1                                // 0000000026BC: D761002B 00010259
	v_writelane_b32 v43, s90, 2                                // 0000000026C4: D761002B 0001045A
	v_writelane_b32 v43, s91, 3                                // 0000000026CC: D761002B 0001065B
	v_writelane_b32 v43, s44, 4                                // 0000000026D4: D761002B 0001082C
	v_writelane_b32 v43, s45, 5                                // 0000000026DC: D761002B 00010A2D
	v_writelane_b32 v43, s46, 6                                // 0000000026E4: D761002B 00010C2E
	v_writelane_b32 v43, s47, 7                                // 0000000026EC: D761002B 00010E2F
	v_writelane_b32 v43, s48, 8                                // 0000000026F4: D761002B 00011030
	v_writelane_b32 v43, s49, 9                                // 0000000026FC: D761002B 00011231
	v_writelane_b32 v43, s50, 10                               // 000000002704: D761002B 00011432
	v_writelane_b32 v43, s51, 11                               // 00000000270C: D761002B 00011633
	v_writelane_b32 v43, s52, 12                               // 000000002714: D761002B 00011834
	v_writelane_b32 v43, s53, 13                               // 00000000271C: D761002B 00011A35
	v_writelane_b32 v43, s54, 14                               // 000000002724: D761002B 00011C36
	v_writelane_b32 v43, s55, 15                               // 00000000272C: D761002B 00011E37
	v_writelane_b32 v43, s56, 16                               // 000000002734: D761002B 00012038
	v_writelane_b32 v43, s57, 17                               // 00000000273C: D761002B 00012239
	v_writelane_b32 v43, s58, 18                               // 000000002744: D761002B 0001243A
	v_writelane_b32 v43, s59, 19                               // 00000000274C: D761002B 0001263B
	v_writelane_b32 v42, s16, 12                               // 000000002754: D761002A 00011810
	v_writelane_b32 v42, s17, 13                               // 00000000275C: D761002A 00011A11
	v_writelane_b32 v42, s18, 14                               // 000000002764: D761002A 00011C12
	v_writelane_b32 v42, s19, 15                               // 00000000276C: D761002A 00011E13
	v_writelane_b32 v42, s20, 16                               // 000000002774: D761002A 00012014
	v_writelane_b32 v42, s21, 17                               // 00000000277C: D761002A 00012215
	v_writelane_b32 v42, s22, 18                               // 000000002784: D761002A 00012416
	v_writelane_b32 v42, s23, 19                               // 00000000278C: D761002A 00012617
	s_or_saveexec_b32 s105, -1                                 // 000000002794: BEE922C1
	scratch_store_b32 off, v41, off offset:88                  // 000000002798: DC690058 007C2900
	s_mov_b32 exec_lo, s105                                    // 0000000027A0: BEFE0069
	s_mov_b32 s84, s12                                         // 0000000027A4: BED4000C
	s_mov_b32 s92, s12                                         // 0000000027A8: BEDC000C
	s_mov_b32 s93, s12                                         // 0000000027AC: BEDD000C
	s_mov_b32 s94, s12                                         // 0000000027B0: BEDE000C
	s_mov_b32 s95, s12                                         // 0000000027B4: BEDF000C
	s_mov_b32 s85, s12                                         // 0000000027B8: BED5000C
	s_mov_b32 s86, s12                                         // 0000000027BC: BED6000C
	s_mov_b32 s87, s12                                         // 0000000027C0: BED7000C
	s_mov_b32 s88, s12                                         // 0000000027C4: BED8000C
	s_mov_b32 s89, s12                                         // 0000000027C8: BED9000C
	s_mov_b32 s90, s12                                         // 0000000027CC: BEDA000C
	s_mov_b32 s91, s12                                         // 0000000027D0: BEDB000C
	s_mov_b32 s96, s12                                         // 0000000027D4: BEE0000C
	s_mov_b32 s97, s12                                         // 0000000027D8: BEE1000C
	s_mov_b32 s98, s12                                         // 0000000027DC: BEE2000C
	s_mov_b32 s99, s12                                         // 0000000027E0: BEE3000C
	v_writelane_b32 v42, s84, 20                               // 0000000027E4: D761002A 00012854
	v_writelane_b32 v45, s96, 0                                // 0000000027EC: D761002D 00010060
	s_mov_b32 s8, s12                                          // 0000000027F4: BE88000C
	s_mov_b32 s13, s12                                         // 0000000027F8: BE8D000C
	s_mov_b32 s14, s12                                         // 0000000027FC: BE8E000C
	s_mov_b32 s15, s12                                         // 000000002800: BE8F000C
	v_writelane_b32 v45, s97, 1                                // 000000002804: D761002D 00010261
	s_mov_b32 s9, s12                                          // 00000000280C: BE89000C
	s_mov_b32 s10, s12                                         // 000000002810: BE8A000C
	s_mov_b32 s11, s12                                         // 000000002814: BE8B000C
	s_mov_b32 s52, s12                                         // 000000002818: BEB4000C
	v_writelane_b32 v45, s98, 2                                // 00000000281C: D761002D 00010462
	s_mov_b32 s53, s12                                         // 000000002824: BEB5000C
	s_mov_b32 s54, s12                                         // 000000002828: BEB6000C
	s_mov_b32 s55, s12                                         // 00000000282C: BEB7000C
	s_mov_b32 s44, s12                                         // 000000002830: BEAC000C
	v_writelane_b32 v45, s99, 3                                // 000000002834: D761002D 00010663
	s_mov_b32 s45, s12                                         // 00000000283C: BEAD000C
	s_mov_b32 s46, s12                                         // 000000002840: BEAE000C
	s_mov_b32 s47, s12                                         // 000000002844: BEAF000C
	s_mov_b32 s48, s12                                         // 000000002848: BEB0000C
	s_mov_b32 s49, s12                                         // 00000000284C: BEB1000C
	s_mov_b32 s50, s12                                         // 000000002850: BEB2000C
	s_mov_b32 s51, s12                                         // 000000002854: BEB3000C
	s_mov_b32 s56, s12                                         // 000000002858: BEB8000C
	s_mov_b32 s57, s12                                         // 00000000285C: BEB9000C
	s_mov_b32 s58, s12                                         // 000000002860: BEBA000C
	v_writelane_b32 v45, s8, 4                                 // 000000002864: D761002D 00010808
	s_mov_b32 s59, s12                                         // 00000000286C: BEBB000C
	v_writelane_b32 v42, s85, 21                               // 000000002870: D761002A 00012A55
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 000000002878: 916A067E
	v_writelane_b32 v45, s9, 5                                 // 00000000287C: D761002D 00010A09
	v_writelane_b32 v42, s86, 22                               // 000000002884: D761002A 00012C56
	v_writelane_b32 v45, s10, 6                                // 00000000288C: D761002D 00010C0A
	v_writelane_b32 v42, s87, 23                               // 000000002894: D761002A 00012E57
	v_writelane_b32 v45, s11, 7                                // 00000000289C: D761002D 00010E0B
	v_writelane_b32 v42, s88, 24                               // 0000000028A4: D761002A 00013058
	v_writelane_b32 v45, s12, 8                                // 0000000028AC: D761002D 0001100C
	v_writelane_b32 v42, s89, 25                               // 0000000028B4: D761002A 00013259
	v_writelane_b32 v45, s13, 9                                // 0000000028BC: D761002D 0001120D
	v_writelane_b32 v42, s90, 26                               // 0000000028C4: D761002A 0001345A
	v_writelane_b32 v45, s14, 10                               // 0000000028CC: D761002D 0001140E
	v_writelane_b32 v42, s91, 27                               // 0000000028D4: D761002A 0001365B
	v_writelane_b32 v45, s15, 11                               // 0000000028DC: D761002D 0001160F
	v_writelane_b32 v42, s92, 28                               // 0000000028E4: D761002A 0001385C
	v_writelane_b32 v45, s44, 12                               // 0000000028EC: D761002D 0001182C
	v_writelane_b32 v42, s93, 29                               // 0000000028F4: D761002A 00013A5D
	v_writelane_b32 v45, s45, 13                               // 0000000028FC: D761002D 00011A2D
	v_writelane_b32 v42, s94, 30                               // 000000002904: D761002A 00013C5E
	v_writelane_b32 v45, s46, 14                               // 00000000290C: D761002D 00011C2E
	v_writelane_b32 v42, s95, 31                               // 000000002914: D761002A 00013E5F
	v_writelane_b32 v45, s47, 15                               // 00000000291C: D761002D 00011E2F
	v_writelane_b32 v45, s48, 16                               // 000000002924: D761002D 00012030
	v_writelane_b32 v45, s49, 17                               // 00000000292C: D761002D 00012231
	v_writelane_b32 v45, s50, 18                               // 000000002934: D761002D 00012432
	v_writelane_b32 v45, s51, 19                               // 00000000293C: D761002D 00012633
	v_writelane_b32 v45, s52, 20                               // 000000002944: D761002D 00012834
	v_writelane_b32 v45, s53, 21                               // 00000000294C: D761002D 00012A35
	v_writelane_b32 v45, s54, 22                               // 000000002954: D761002D 00012C36
	v_writelane_b32 v45, s55, 23                               // 00000000295C: D761002D 00012E37
	v_writelane_b32 v45, s56, 24                               // 000000002964: D761002D 00013038
	v_writelane_b32 v45, s57, 25                               // 00000000296C: D761002D 00013239
	v_writelane_b32 v45, s58, 26                               // 000000002974: D761002D 0001343A
	v_writelane_b32 v45, s59, 27                               // 00000000297C: D761002D 0001363B
	s_cbranch_vccnz 89                                         // 000000002984: BFA40059 <r_3_3_3_8_8_8+0x14ec>
	s_clause 0x1                                               // 000000002988: BF850001
	s_load_b512 s[44:59], s[4:5], 0x2500                       // 00000000298C: F4100B02 F8002500
	s_load_b256 s[8:15], s[4:5], 0x23e0                        // 000000002994: F40C0202 F80023E0
	s_waitcnt lgkmcnt(0)                                       // 00000000299C: BF89FC07
	v_writelane_b32 v42, s44, 20                               // 0000000029A0: D761002A 0001282C
	v_writelane_b32 v45, s56, 0                                // 0000000029A8: D761002D 00010038
	v_writelane_b32 v42, s45, 21                               // 0000000029B0: D761002A 00012A2D
	v_writelane_b32 v45, s57, 1                                // 0000000029B8: D761002D 00010239
	v_writelane_b32 v42, s46, 22                               // 0000000029C0: D761002A 00012C2E
	v_writelane_b32 v45, s58, 2                                // 0000000029C8: D761002D 0001043A
	v_writelane_b32 v42, s47, 23                               // 0000000029D0: D761002A 00012E2F
	v_writelane_b32 v45, s59, 3                                // 0000000029D8: D761002D 0001063B
	v_writelane_b32 v42, s48, 24                               // 0000000029E0: D761002A 00013030
	v_writelane_b32 v45, s8, 4                                 // 0000000029E8: D761002D 00010808
	v_writelane_b32 v42, s49, 25                               // 0000000029F0: D761002A 00013231
	v_writelane_b32 v45, s9, 5                                 // 0000000029F8: D761002D 00010A09
	v_writelane_b32 v42, s50, 26                               // 000000002A00: D761002A 00013432
	v_writelane_b32 v45, s10, 6                                // 000000002A08: D761002D 00010C0A
	v_writelane_b32 v42, s51, 27                               // 000000002A10: D761002A 00013633
	v_writelane_b32 v45, s11, 7                                // 000000002A18: D761002D 00010E0B
	v_writelane_b32 v42, s52, 28                               // 000000002A20: D761002A 00013834
	v_writelane_b32 v45, s12, 8                                // 000000002A28: D761002D 0001100C
	v_writelane_b32 v42, s53, 29                               // 000000002A30: D761002A 00013A35
	v_writelane_b32 v45, s13, 9                                // 000000002A38: D761002D 0001120D
	v_writelane_b32 v42, s54, 30                               // 000000002A40: D761002A 00013C36
	v_writelane_b32 v45, s14, 10                               // 000000002A48: D761002D 0001140E
	v_writelane_b32 v42, s55, 31                               // 000000002A50: D761002A 00013E37
	v_writelane_b32 v45, s15, 11                               // 000000002A58: D761002D 0001160F
	s_load_b512 s[8:23], s[4:5], 0x2640                        // 000000002A60: F4100202 F8002640
	s_waitcnt lgkmcnt(0)                                       // 000000002A68: BF89FC07
	v_writelane_b32 v45, s8, 12                                // 000000002A6C: D761002D 00011808
	v_writelane_b32 v45, s9, 13                                // 000000002A74: D761002D 00011A09
	v_writelane_b32 v45, s10, 14                               // 000000002A7C: D761002D 00011C0A
	v_writelane_b32 v45, s11, 15                               // 000000002A84: D761002D 00011E0B
	v_writelane_b32 v45, s12, 16                               // 000000002A8C: D761002D 0001200C
	v_writelane_b32 v45, s13, 17                               // 000000002A94: D761002D 0001220D
	v_writelane_b32 v45, s14, 18                               // 000000002A9C: D761002D 0001240E
	v_writelane_b32 v45, s15, 19                               // 000000002AA4: D761002D 0001260F
	v_writelane_b32 v45, s16, 20                               // 000000002AAC: D761002D 00012810
	v_writelane_b32 v45, s17, 21                               // 000000002AB4: D761002D 00012A11
	v_writelane_b32 v45, s18, 22                               // 000000002ABC: D761002D 00012C12
	v_writelane_b32 v45, s19, 23                               // 000000002AC4: D761002D 00012E13
	v_writelane_b32 v45, s20, 24                               // 000000002ACC: D761002D 00013014
	v_writelane_b32 v45, s21, 25                               // 000000002AD4: D761002D 00013215
	v_writelane_b32 v45, s22, 26                               // 000000002ADC: D761002D 00013416
	v_writelane_b32 v45, s23, 27                               // 000000002AE4: D761002D 00013617
	s_waitcnt lgkmcnt(0)                                       // 000000002AEC: BF89FC07
	v_writelane_b32 v45, s36, 28                               // 000000002AF0: D761002D 00013824
	s_and_b32 vcc_lo, exec_lo, s34                             // 000000002AF8: 8B6A227E
	v_writelane_b32 v45, s37, 29                               // 000000002AFC: D761002D 00013A25
	v_writelane_b32 v45, s38, 30                               // 000000002B04: D761002D 00013C26
	v_writelane_b32 v45, s39, 31                               // 000000002B0C: D761002D 00013E27
	s_or_saveexec_b32 s105, -1                                 // 000000002B14: BEE922C1
	scratch_store_b32 off, v45, off offset:40                  // 000000002B18: DC690028 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000002B20: BEFE0069
	v_writelane_b32 v44, s40, 0                                // 000000002B24: D761002C 00010028
	v_writelane_b32 v44, s41, 1                                // 000000002B2C: D761002C 00010229
	v_writelane_b32 v44, s42, 2                                // 000000002B34: D761002C 0001042A
	v_writelane_b32 v44, s43, 3                                // 000000002B3C: D761002C 0001062B
	s_cbranch_vccnz 2641                                       // 000000002B44: BFA40A51 <r_3_3_3_8_8_8+0x3e8c>
	s_or_saveexec_b32 s105, -1                                 // 000000002B48: BEE922C1
	scratch_load_b32 v45, off, off                             // 000000002B4C: DC510000 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000002B54: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000002B58: BF8903F7
	v_readlane_b32 s6, v45, 20                                 // 000000002B5C: D7600006 0001292D
	v_readlane_b32 s7, v45, 23                                 // 000000002B64: D7600007 00012F2D
	s_mov_b32 s101, s100                                       // 000000002B6C: BEE50064
	s_mov_b32 s102, s100                                       // 000000002B70: BEE60064
	s_mov_b32 s103, s100                                       // 000000002B74: BEE70064
	s_add_u32 s8, s6, s28                                      // 000000002B78: 80081C06
	v_readlane_b32 s6, v45, 21                                 // 000000002B7C: D7600006 00012B2D
	s_mov_b64 s[16:17], s[100:101]                             // 000000002B84: BE900164
	s_mov_b64 s[12:13], s[100:101]                             // 000000002B88: BE8C0164
	s_mov_b64 s[18:19], s[102:103]                             // 000000002B8C: BE920166
	s_mov_b64 s[14:15], s[102:103]                             // 000000002B90: BE8E0166
	s_addc_u32 s9, s6, s29                                     // 000000002B94: 82091D06
	v_readlane_b32 s6, v45, 22                                 // 000000002B98: D7600006 00012D2D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002BA0: BF870001
	s_add_u32 s6, s6, s28                                      // 000000002BA4: 80061C06
	s_addc_u32 s7, s7, s29                                     // 000000002BA8: 82071D07
	s_and_b32 vcc_lo, exec_lo, s33                             // 000000002BAC: 8B6A217E
	s_cbranch_vccnz 9                                          // 000000002BB0: BFA40009 <r_3_3_3_8_8_8+0x15d8>
	v_readlane_b32 s10, v45, 24                                // 000000002BB4: D760000A 0001312D
	v_readlane_b32 s11, v45, 25                                // 000000002BBC: D760000B 0001332D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002BC4: BF870092
	s_add_u32 s10, s10, s28                                    // 000000002BC8: 800A1C0A
	s_addc_u32 s11, s11, s29                                   // 000000002BCC: 820B1D0B
	s_load_b256 s[12:19], s[10:11], null                       // 000000002BD0: F40C0305 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000002BD8: BF89FC07
	v_writelane_b32 v38, s12, 4                                // 000000002BDC: D7610026 0001080C
	s_load_b256 s[92:99], s[8:9], null                         // 000000002BE4: F40C1704 F8000000
	s_mov_b32 s56, 0                                           // 000000002BEC: BEB80080
	s_mov_b32 s24, vcc_hi                                      // 000000002BF0: BE98006B
	v_writelane_b32 v38, s13, 5                                // 000000002BF4: D7610026 00010A0D
	v_writelane_b32 v38, s14, 6                                // 000000002BFC: D7610026 00010C0E
	v_writelane_b32 v38, s15, 7                                // 000000002C04: D7610026 00010E0F
	v_writelane_b32 v38, s16, 8                                // 000000002C0C: D7610026 00011010
	s_waitcnt lgkmcnt(0)                                       // 000000002C14: BF89FC07
	v_writelane_b32 v39, s92, 28                               // 000000002C18: D7610027 0001385C
	v_writelane_b32 v38, s17, 9                                // 000000002C20: D7610026 00011211
	v_writelane_b32 v39, s93, 29                               // 000000002C28: D7610027 00013A5D
	v_writelane_b32 v38, s18, 10                               // 000000002C30: D7610026 00011412
	v_writelane_b32 v39, s94, 30                               // 000000002C38: D7610027 00013C5E
	v_writelane_b32 v38, s19, 11                               // 000000002C40: D7610026 00011613
	s_load_b256 s[8:15], s[6:7], null                          // 000000002C48: F40C0203 F8000000
	v_writelane_b32 v39, s95, 31                               // 000000002C50: D7610027 00013E5F
	v_readlane_b32 s6, v45, 26                                 // 000000002C58: D7600006 0001352D
	v_readlane_b32 s7, v45, 29                                 // 000000002C60: D7600007 00013B2D
	v_writelane_b32 v38, s96, 0                                // 000000002C68: D7610026 00010060
	v_writelane_b32 v38, s97, 1                                // 000000002C70: D7610026 00010261
	v_writelane_b32 v38, s98, 2                                // 000000002C78: D7610026 00010462
	v_writelane_b32 v38, s99, 3                                // 000000002C80: D7610026 00010663
	s_waitcnt lgkmcnt(0)                                       // 000000002C88: BF89FC07
	v_writelane_b32 v39, s8, 20                                // 000000002C8C: D7610027 00012808
	v_writelane_b32 v39, s9, 21                                // 000000002C94: D7610027 00012A09
	v_writelane_b32 v39, s10, 22                               // 000000002C9C: D7610027 00012C0A
	v_writelane_b32 v39, s11, 23                               // 000000002CA4: D7610027 00012E0B
	v_writelane_b32 v39, s12, 24                               // 000000002CAC: D7610027 0001300C
	v_writelane_b32 v39, s13, 25                               // 000000002CB4: D7610027 0001320D
	v_writelane_b32 v39, s14, 26                               // 000000002CBC: D7610027 0001340E
	v_writelane_b32 v39, s15, 27                               // 000000002CC4: D7610027 0001360F
	s_add_u32 s8, s6, s28                                      // 000000002CCC: 80081C06
	v_readlane_b32 s6, v45, 27                                 // 000000002CD0: D7600006 0001372D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000002CD8: BF8700B1
	s_addc_u32 s9, s6, s29                                     // 000000002CDC: 82091D06
	v_readlane_b32 s6, v45, 28                                 // 000000002CE0: D7600006 0001392D
	s_load_b512 s[36:51], s[8:9], null                         // 000000002CE8: F4100904 F8000000
	s_add_u32 s6, s6, s28                                      // 000000002CF0: 80061C06
	s_addc_u32 s7, s7, s29                                     // 000000002CF4: 82071D07
	s_add_u32 s8, s6, 0xfffffe40                               // 000000002CF8: 8008FF06 FFFFFE40
	s_load_b256 s[60:67], s[6:7], 0x1100                       // 000000002D00: F40C0F03 F8001100
	s_addc_u32 s9, s7, -1                                      // 000000002D08: 8209C107
	s_add_u32 s10, s6, 0xfffffbe0                              // 000000002D0C: 800AFF06 FFFFFBE0
	s_addc_u32 s11, s7, -1                                     // 000000002D14: 820BC107
	s_waitcnt lgkmcnt(0)                                       // 000000002D18: BF89FC07
	v_writelane_b32 v44, s36, 4                                // 000000002D1C: D761002C 00010824
	v_writelane_b32 v44, s37, 5                                // 000000002D24: D761002C 00010A25
	v_writelane_b32 v39, s60, 12                               // 000000002D2C: D7610027 0001183C
	v_writelane_b32 v44, s38, 6                                // 000000002D34: D761002C 00010C26
	v_writelane_b32 v39, s61, 13                               // 000000002D3C: D7610027 00011A3D
	v_writelane_b32 v44, s39, 7                                // 000000002D44: D761002C 00010E27
	v_writelane_b32 v39, s62, 14                               // 000000002D4C: D7610027 00011C3E
	v_writelane_b32 v44, s40, 8                                // 000000002D54: D761002C 00011028
	v_writelane_b32 v39, s63, 15                               // 000000002D5C: D7610027 00011E3F
	v_writelane_b32 v44, s41, 9                                // 000000002D64: D761002C 00011229
	v_writelane_b32 v39, s64, 16                               // 000000002D6C: D7610027 00012040
	v_writelane_b32 v44, s42, 10                               // 000000002D74: D761002C 0001142A
	v_writelane_b32 v39, s65, 17                               // 000000002D7C: D7610027 00012241
	v_writelane_b32 v44, s43, 11                               // 000000002D84: D761002C 0001162B
	v_writelane_b32 v39, s66, 18                               // 000000002D8C: D7610027 00012442
	v_writelane_b32 v44, s44, 12                               // 000000002D94: D761002C 0001182C
	v_writelane_b32 v39, s67, 19                               // 000000002D9C: D7610027 00012643
	s_load_b256 s[60:67], s[10:11], null                       // 000000002DA4: F40C0F05 F8000000
	v_writelane_b32 v44, s45, 13                               // 000000002DAC: D761002C 00011A2D
	v_writelane_b32 v44, s46, 14                               // 000000002DB4: D761002C 00011C2E
	v_writelane_b32 v44, s47, 15                               // 000000002DBC: D761002C 00011E2F
	v_writelane_b32 v44, s48, 16                               // 000000002DC4: D761002C 00012030
	v_writelane_b32 v44, s49, 17                               // 000000002DCC: D761002C 00012231
	v_writelane_b32 v44, s50, 18                               // 000000002DD4: D761002C 00012432
	v_writelane_b32 v44, s51, 19                               // 000000002DDC: D761002C 00012633
	s_load_b512 s[36:51], s[8:9], null                         // 000000002DE4: F4100904 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000002DEC: BF89FC07
	v_writelane_b32 v44, s36, 28                               // 000000002DF0: D761002C 00013824
	v_writelane_b32 v39, s40, 0                                // 000000002DF8: D7610027 00010028
	v_writelane_b32 v44, s37, 29                               // 000000002E00: D761002C 00013A25
	v_writelane_b32 v39, s41, 1                                // 000000002E08: D7610027 00010229
	v_writelane_b32 v44, s38, 30                               // 000000002E10: D761002C 00013C26
	v_writelane_b32 v39, s42, 2                                // 000000002E18: D7610027 0001042A
	v_writelane_b32 v44, s39, 31                               // 000000002E20: D761002C 00013E27
	v_writelane_b32 v39, s43, 3                                // 000000002E28: D7610027 0001062B
	v_writelane_b32 v44, s60, 20                               // 000000002E30: D761002C 0001283C
	v_writelane_b32 v39, s44, 4                                // 000000002E38: D7610027 0001082C
	v_writelane_b32 v44, s61, 21                               // 000000002E40: D761002C 00012A3D
	v_writelane_b32 v39, s45, 5                                // 000000002E48: D7610027 00010A2D
	v_writelane_b32 v44, s62, 22                               // 000000002E50: D761002C 00012C3E
	v_writelane_b32 v39, s46, 6                                // 000000002E58: D7610027 00010C2E
	v_writelane_b32 v44, s63, 23                               // 000000002E60: D761002C 00012E3F
	v_writelane_b32 v39, s47, 7                                // 000000002E68: D7610027 00010E2F
	v_writelane_b32 v44, s64, 24                               // 000000002E70: D761002C 00013040
	v_writelane_b32 v39, s48, 8                                // 000000002E78: D7610027 00011030
	v_writelane_b32 v44, s65, 25                               // 000000002E80: D761002C 00013241
	v_writelane_b32 v39, s49, 9                                // 000000002E88: D7610027 00011231
	v_writelane_b32 v44, s66, 26                               // 000000002E90: D761002C 00013442
	v_writelane_b32 v39, s50, 10                               // 000000002E98: D7610027 00011432
	v_writelane_b32 v44, s67, 27                               // 000000002EA0: D761002C 00013643
	v_writelane_b32 v39, s51, 11                               // 000000002EA8: D7610027 00011633
	s_branch 203                                               // 000000002EB0: BFA000CB <r_3_3_3_8_8_8+0x1be0>
	s_or_saveexec_b32 s105, -1                                 // 000000002EB4: BEE922C1
	scratch_load_b32 v45, off, off                             // 000000002EB8: DC510000 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000002EC0: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000002EC4: BF8903F7
	v_readlane_b32 s6, v45, 22                                 // 000000002EC8: D7600006 00012D2D
	v_readlane_b32 s7, v45, 23                                 // 000000002ED0: D7600007 00012F2D
	s_mov_b32 s101, s100                                       // 000000002ED8: BEE50064
	s_mov_b32 s102, s100                                       // 000000002EDC: BEE60064
	s_mov_b32 s103, s100                                       // 000000002EE0: BEE70064
	s_add_u32 s6, s6, s28                                      // 000000002EE4: 80061C06
	s_addc_u32 s7, s7, s29                                     // 000000002EE8: 82071D07
	s_mov_b64 s[96:97], s[100:101]                             // 000000002EEC: BEE00164
	s_mov_b64 s[92:93], s[100:101]                             // 000000002EF0: BEDC0164
	s_mov_b64 s[98:99], s[102:103]                             // 000000002EF4: BEE20166
	s_mov_b64 s[94:95], s[102:103]                             // 000000002EF8: BEDE0166
	s_load_b256 s[12:19], s[6:7], null                         // 000000002EFC: F40C0303 F8000000
	v_writelane_b32 v44, s92, 20                               // 000000002F04: D761002C 0001285C
	v_readlane_b32 s8, v45, 24                                 // 000000002F0C: D7600008 0001312D
	v_readlane_b32 s6, v45, 25                                 // 000000002F14: D7600006 0001332D
	v_readlane_b32 s7, v45, 29                                 // 000000002F1C: D7600007 00013B2D
	s_mov_b64 s[48:49], s[100:101]                             // 000000002F24: BEB00164
	v_writelane_b32 v44, s93, 21                               // 000000002F28: D761002C 00012A5D
	s_add_u32 s8, s8, s28                                      // 000000002F30: 80081C08
	s_addc_u32 s9, s6, s29                                     // 000000002F34: 82091D06
	v_readlane_b32 s6, v45, 28                                 // 000000002F38: D7600006 0001392D
	s_mov_b64 s[40:41], s[100:101]                             // 000000002F40: BEA80164
	v_writelane_b32 v44, s94, 22                               // 000000002F44: D761002C 00012C5E
	s_mov_b64 s[36:37], s[100:101]                             // 000000002F4C: BEA40164
	s_mov_b64 s[44:45], s[100:101]                             // 000000002F50: BEAC0164
	s_add_u32 s6, s6, s28                                      // 000000002F54: 80061C06
	s_addc_u32 s7, s7, s29                                     // 000000002F58: 82071D07
	v_writelane_b32 v44, s95, 23                               // 000000002F5C: D761002C 00012E5F
	s_mov_b64 s[50:51], s[102:103]                             // 000000002F64: BEB20166
	s_mov_b64 s[42:43], s[102:103]                             // 000000002F68: BEAA0166
	s_mov_b64 s[38:39], s[102:103]                             // 000000002F6C: BEA60166
	s_waitcnt lgkmcnt(0)                                       // 000000002F70: BF89FC07
	v_writelane_b32 v39, s12, 20                               // 000000002F74: D7610027 0001280C
	v_writelane_b32 v44, s96, 24                               // 000000002F7C: D761002C 00013060
	s_mov_b64 s[46:47], s[102:103]                             // 000000002F84: BEAE0166
	s_load_b256 s[60:67], s[6:7], 0x1100                       // 000000002F88: F40C0F03 F8001100
	s_mov_b64 s[20:21], s[100:101]                             // 000000002F90: BE940164
	v_writelane_b32 v39, s13, 21                               // 000000002F94: D7610027 00012A0D
	v_writelane_b32 v44, s97, 25                               // 000000002F9C: D761002C 00013261
	s_mov_b64 s[22:23], s[102:103]                             // 000000002FA4: BE960166
	s_mov_b32 s24, -1                                          // 000000002FA8: BE9800C1
	v_writelane_b32 v39, s14, 22                               // 000000002FAC: D7610027 00012C0E
	v_writelane_b32 v44, s98, 26                               // 000000002FB4: D761002C 00013462
	v_writelane_b32 v39, s15, 23                               // 000000002FBC: D7610027 00012E0F
	v_writelane_b32 v44, s99, 27                               // 000000002FC4: D761002C 00013663
	v_writelane_b32 v39, s16, 24                               // 000000002FCC: D7610027 00013010
	v_writelane_b32 v44, s36, 4                                // 000000002FD4: D761002C 00010824
	v_writelane_b32 v39, s17, 25                               // 000000002FDC: D7610027 00013211
	v_writelane_b32 v44, s37, 5                                // 000000002FE4: D761002C 00010A25
	v_writelane_b32 v39, s18, 26                               // 000000002FEC: D7610027 00013412
	v_writelane_b32 v44, s38, 6                                // 000000002FF4: D761002C 00010C26
	v_writelane_b32 v39, s19, 27                               // 000000002FFC: D7610027 00013613
	v_writelane_b32 v44, s39, 7                                // 000000003004: D761002C 00010E27
	s_load_b256 s[8:15], s[8:9], null                          // 00000000300C: F40C0204 F8000000
	s_mov_b64 s[16:17], s[100:101]                             // 000000003014: BE900164
	s_mov_b64 s[18:19], s[102:103]                             // 000000003018: BE920166
	s_waitcnt lgkmcnt(0)                                       // 00000000301C: BF89FC07
	v_writelane_b32 v39, s60, 12                               // 000000003020: D7610027 0001183C
	v_writelane_b32 v44, s40, 8                                // 000000003028: D761002C 00011028
	v_writelane_b32 v39, s61, 13                               // 000000003030: D7610027 00011A3D
	v_writelane_b32 v44, s41, 9                                // 000000003038: D761002C 00011229
	v_writelane_b32 v39, s62, 14                               // 000000003040: D7610027 00011C3E
	v_writelane_b32 v44, s42, 10                               // 000000003048: D761002C 0001142A
	v_writelane_b32 v39, s63, 15                               // 000000003050: D7610027 00011E3F
	v_writelane_b32 v44, s43, 11                               // 000000003058: D761002C 0001162B
	v_writelane_b32 v38, s8, 4                                 // 000000003060: D7610026 00010808
	v_writelane_b32 v39, s64, 16                               // 000000003068: D7610027 00012040
	v_writelane_b32 v44, s44, 12                               // 000000003070: D761002C 0001182C
	v_writelane_b32 v38, s9, 5                                 // 000000003078: D7610026 00010A09
	v_writelane_b32 v39, s65, 17                               // 000000003080: D7610027 00012241
	v_writelane_b32 v44, s45, 13                               // 000000003088: D761002C 00011A2D
	v_writelane_b32 v38, s10, 6                                // 000000003090: D7610026 00010C0A
	v_writelane_b32 v39, s66, 18                               // 000000003098: D7610027 00012442
	v_writelane_b32 v44, s46, 14                               // 0000000030A0: D761002C 00011C2E
	v_writelane_b32 v38, s11, 7                                // 0000000030A8: D7610026 00010E0B
	v_writelane_b32 v39, s67, 19                               // 0000000030B0: D7610027 00012643
	v_writelane_b32 v44, s47, 15                               // 0000000030B8: D761002C 00011E2F
	s_mov_b64 s[64:65], s[100:101]                             // 0000000030C0: BEC00164
	s_mov_b64 s[60:61], s[100:101]                             // 0000000030C4: BEBC0164
	s_mov_b64 s[66:67], s[102:103]                             // 0000000030C8: BEC20166
	s_mov_b64 s[62:63], s[102:103]                             // 0000000030CC: BEBE0166
	v_writelane_b32 v39, s60, 28                               // 0000000030D0: D7610027 0001383C
	v_writelane_b32 v44, s48, 16                               // 0000000030D8: D761002C 00012030
	v_writelane_b32 v38, s12, 8                                // 0000000030E0: D7610026 0001100C
	v_writelane_b32 v39, s61, 29                               // 0000000030E8: D7610027 00013A3D
	v_writelane_b32 v44, s49, 17                               // 0000000030F0: D761002C 00012231
	v_writelane_b32 v38, s13, 9                                // 0000000030F8: D7610026 0001120D
	v_writelane_b32 v39, s62, 30                               // 000000003100: D7610027 00013C3E
	v_writelane_b32 v44, s50, 18                               // 000000003108: D761002C 00012432
	v_writelane_b32 v38, s14, 10                               // 000000003110: D7610026 0001140E
	v_writelane_b32 v39, s63, 31                               // 000000003118: D7610027 00013E3F
	v_writelane_b32 v44, s51, 19                               // 000000003120: D761002C 00012633
	v_writelane_b32 v38, s15, 11                               // 000000003128: D7610026 0001160F
	s_mov_b64 s[8:9], s[100:101]                               // 000000003130: BE880164
	s_mov_b64 s[12:13], s[100:101]                             // 000000003134: BE8C0164
	s_mov_b64 s[10:11], s[102:103]                             // 000000003138: BE8A0166
	s_mov_b64 s[14:15], s[102:103]                             // 00000000313C: BE8E0166
	v_writelane_b32 v44, s8, 28                                // 000000003140: D761002C 00013808
	v_writelane_b32 v39, s12, 0                                // 000000003148: D7610027 0001000C
	v_writelane_b32 v38, s64, 0                                // 000000003150: D7610026 00010040
	v_writelane_b32 v44, s9, 29                                // 000000003158: D761002C 00013A09
	v_writelane_b32 v39, s13, 1                                // 000000003160: D7610027 0001020D
	v_writelane_b32 v38, s65, 1                                // 000000003168: D7610026 00010241
	v_writelane_b32 v44, s10, 30                               // 000000003170: D761002C 00013C0A
	v_writelane_b32 v39, s14, 2                                // 000000003178: D7610027 0001040E
	v_writelane_b32 v38, s66, 2                                // 000000003180: D7610026 00010442
	v_writelane_b32 v44, s11, 31                               // 000000003188: D761002C 00013E0B
	v_writelane_b32 v39, s15, 3                                // 000000003190: D7610027 0001060F
	v_writelane_b32 v38, s67, 3                                // 000000003198: D7610026 00010643
	v_writelane_b32 v39, s16, 4                                // 0000000031A0: D7610027 00010810
	v_writelane_b32 v39, s17, 5                                // 0000000031A8: D7610027 00010A11
	v_writelane_b32 v39, s18, 6                                // 0000000031B0: D7610027 00010C12
	v_writelane_b32 v39, s19, 7                                // 0000000031B8: D7610027 00010E13
	v_writelane_b32 v39, s20, 8                                // 0000000031C0: D7610027 00011014
	v_writelane_b32 v39, s21, 9                                // 0000000031C8: D7610027 00011215
	v_writelane_b32 v39, s22, 10                               // 0000000031D0: D7610027 00011416
	v_writelane_b32 v39, s23, 11                               // 0000000031D8: D7610027 00011617
	s_clause 0x1                                               // 0000000031E0: BF850001
	s_load_b256 s[92:99], s[0:1], null                         // 0000000031E4: F40C1700 F8000000
	s_load_b512 s[8:23], s[2:3], null                          // 0000000031EC: F4100201 F8000000
	s_mov_b32 s57, s56                                         // 0000000031F4: BEB90038
	s_mov_b32 s58, s56                                         // 0000000031F8: BEBA0038
	s_mov_b32 s59, s56                                         // 0000000031FC: BEBB0038
	s_mov_b32 s44, s56                                         // 000000003200: BEAC0038
	s_mov_b32 s45, s56                                         // 000000003204: BEAD0038
	s_mov_b32 s46, s56                                         // 000000003208: BEAE0038
	s_mov_b32 s47, s56                                         // 00000000320C: BEAF0038
	s_mov_b32 s36, s56                                         // 000000003210: BEA40038
	s_mov_b32 s37, s56                                         // 000000003214: BEA50038
	s_mov_b32 s38, s56                                         // 000000003218: BEA60038
	s_mov_b32 s39, s56                                         // 00000000321C: BEA70038
	s_mov_b32 s40, s56                                         // 000000003220: BEA80038
	s_mov_b32 s41, s56                                         // 000000003224: BEA90038
	s_mov_b32 s42, s56                                         // 000000003228: BEAA0038
	s_mov_b32 s43, s56                                         // 00000000322C: BEAB0038
	s_mov_b32 s52, s56                                         // 000000003230: BEB40038
	s_mov_b32 s53, s56                                         // 000000003234: BEB50038
	s_mov_b32 s54, s56                                         // 000000003238: BEB60038
	s_waitcnt lgkmcnt(0)                                       // 00000000323C: BF89FC07
	v_writelane_b32 v38, s92, 12                               // 000000003240: D7610026 0001185C
	s_mov_b32 s55, s56                                         // 000000003248: BEB70038
	s_and_not1_b32 vcc_lo, exec_lo, s24                        // 00000000324C: 916A187E
	v_writelane_b32 v38, s93, 13                               // 000000003250: D7610026 00011A5D
	v_writelane_b32 v38, s94, 14                               // 000000003258: D7610026 00011C5E
	v_writelane_b32 v38, s95, 15                               // 000000003260: D7610026 00011E5F
	v_writelane_b32 v38, s96, 16                               // 000000003268: D7610026 00012060
	v_writelane_b32 v38, s97, 17                               // 000000003270: D7610026 00012261
	v_writelane_b32 v38, s98, 18                               // 000000003278: D7610026 00012462
	v_writelane_b32 v38, s99, 19                               // 000000003280: D7610026 00012663
	s_load_b256 s[92:99], s[4:5], 0x1120                       // 000000003288: F40C1702 F8001120
	v_writelane_b32 v38, s8, 20                                // 000000003290: D7610026 00012808
	v_writelane_b32 v45, s20, 0                                // 000000003298: D761002D 00010014
	v_writelane_b32 v38, s9, 21                                // 0000000032A0: D7610026 00012A09
	v_writelane_b32 v45, s21, 1                                // 0000000032A8: D761002D 00010215
	v_writelane_b32 v38, s10, 22                               // 0000000032B0: D7610026 00012C0A
	v_writelane_b32 v45, s22, 2                                // 0000000032B8: D761002D 00010416
	v_writelane_b32 v38, s11, 23                               // 0000000032C0: D7610026 00012E0B
	v_writelane_b32 v45, s23, 3                                // 0000000032C8: D761002D 00010617
	s_mov_b32 s8, s56                                          // 0000000032D0: BE880038
	s_mov_b32 s9, s56                                          // 0000000032D4: BE890038
	s_mov_b32 s10, s56                                         // 0000000032D8: BE8A0038
	v_writelane_b32 v38, s12, 24                               // 0000000032DC: D7610026 0001300C
	s_waitcnt lgkmcnt(0)                                       // 0000000032E4: BF89FC07
	v_writelane_b32 v45, s92, 4                                // 0000000032E8: D761002D 0001085C
	s_mov_b32 s11, s56                                         // 0000000032F0: BE8B0038
	s_mov_b32 s12, s56                                         // 0000000032F4: BE8C0038
	s_mov_b32 s20, s56                                         // 0000000032F8: BE940038
	v_writelane_b32 v38, s13, 25                               // 0000000032FC: D7610026 0001320D
	v_writelane_b32 v45, s93, 5                                // 000000003304: D761002D 00010A5D
	s_mov_b32 s13, s56                                         // 00000000330C: BE8D0038
	s_mov_b32 s21, s56                                         // 000000003310: BE950038
	s_mov_b32 s22, s56                                         // 000000003314: BE960038
	v_writelane_b32 v38, s14, 26                               // 000000003318: D7610026 0001340E
	v_writelane_b32 v45, s94, 6                                // 000000003320: D761002D 00010C5E
	s_mov_b32 s14, s56                                         // 000000003328: BE8E0038
	s_mov_b32 s23, s56                                         // 00000000332C: BE970038
	v_writelane_b32 v38, s15, 27                               // 000000003330: D7610026 0001360F
	v_writelane_b32 v45, s95, 7                                // 000000003338: D761002D 00010E5F
	s_mov_b32 s15, s56                                         // 000000003340: BE8F0038
	v_writelane_b32 v38, s16, 28                               // 000000003344: D7610026 00013810
	v_writelane_b32 v45, s96, 8                                // 00000000334C: D761002D 00011060
	s_mov_b32 s16, s56                                         // 000000003354: BE900038
	v_writelane_b32 v38, s17, 29                               // 000000003358: D7610026 00013A11
	v_writelane_b32 v45, s97, 9                                // 000000003360: D761002D 00011261
	s_mov_b32 s17, s56                                         // 000000003368: BE910038
	v_writelane_b32 v38, s18, 30                               // 00000000336C: D7610026 00013C12
	v_writelane_b32 v45, s98, 10                               // 000000003374: D761002D 00011462
	s_mov_b32 s18, s56                                         // 00000000337C: BE920038
	v_writelane_b32 v38, s19, 31                               // 000000003380: D7610026 00013E13
	v_writelane_b32 v45, s99, 11                               // 000000003388: D761002D 00011663
	s_load_b256 s[92:99], s[4:5], 0xfe0                        // 000000003390: F40C1702 F8000FE0
	s_mov_b32 s19, s56                                         // 000000003398: BE930038
	s_waitcnt lgkmcnt(0)                                       // 00000000339C: BF89FC07
	v_writelane_b32 v45, s92, 12                               // 0000000033A0: D761002D 0001185C
	v_writelane_b32 v45, s93, 13                               // 0000000033A8: D761002D 00011A5D
	v_writelane_b32 v45, s94, 14                               // 0000000033B0: D761002D 00011C5E
	v_writelane_b32 v45, s95, 15                               // 0000000033B8: D761002D 00011E5F
	v_writelane_b32 v45, s96, 16                               // 0000000033C0: D761002D 00012060
	v_writelane_b32 v45, s97, 17                               // 0000000033C8: D761002D 00012261
	v_writelane_b32 v45, s98, 18                               // 0000000033D0: D761002D 00012462
	v_writelane_b32 v45, s99, 19                               // 0000000033D8: D761002D 00012663
	v_writelane_b32 v45, s8, 20                                // 0000000033E0: D761002D 00012808
	v_writelane_b32 v45, s9, 21                                // 0000000033E8: D761002D 00012A09
	v_writelane_b32 v45, s10, 22                               // 0000000033F0: D761002D 00012C0A
	v_writelane_b32 v45, s11, 23                               // 0000000033F8: D761002D 00012E0B
	v_writelane_b32 v45, s12, 24                               // 000000003400: D761002D 0001300C
	v_writelane_b32 v45, s13, 25                               // 000000003408: D761002D 0001320D
	v_writelane_b32 v45, s14, 26                               // 000000003410: D761002D 0001340E
	v_writelane_b32 v45, s15, 27                               // 000000003418: D761002D 0001360F
	v_writelane_b32 v45, s16, 28                               // 000000003420: D761002D 00013810
	v_writelane_b32 v45, s17, 29                               // 000000003428: D761002D 00013A11
	v_writelane_b32 v45, s18, 30                               // 000000003430: D761002D 00013C12
	v_writelane_b32 v45, s19, 31                               // 000000003438: D761002D 00013E13
	s_or_saveexec_b32 s105, -1                                 // 000000003440: BEE922C1
	scratch_store_b32 off, v45, off offset:16                  // 000000003444: DC690010 007C2D00
	s_mov_b32 exec_lo, s105                                    // 00000000344C: BEFE0069
	v_writelane_b32 v37, s20, 0                                // 000000003450: D7610025 00010014
	s_mov_b32 s48, s56                                         // 000000003458: BEB00038
	s_mov_b32 s49, s56                                         // 00000000345C: BEB10038
	s_mov_b32 s50, s56                                         // 000000003460: BEB20038
	s_mov_b32 s51, s56                                         // 000000003464: BEB30038
	v_writelane_b32 v37, s21, 1                                // 000000003468: D7610025 00010215
	v_writelane_b32 v37, s22, 2                                // 000000003470: D7610025 00010416
	v_writelane_b32 v37, s23, 3                                // 000000003478: D7610025 00010617
	s_load_b512 s[8:23], s[4:5], 0x1240                        // 000000003480: F4100202 F8001240
	s_waitcnt lgkmcnt(0)                                       // 000000003488: BF89FC07
	v_writelane_b32 v37, s8, 4                                 // 00000000348C: D7610025 00010808
	v_writelane_b32 v37, s9, 5                                 // 000000003494: D7610025 00010A09
	v_writelane_b32 v37, s10, 6                                // 00000000349C: D7610025 00010C0A
	v_writelane_b32 v37, s11, 7                                // 0000000034A4: D7610025 00010E0B
	v_writelane_b32 v37, s12, 8                                // 0000000034AC: D7610025 0001100C
	v_writelane_b32 v37, s13, 9                                // 0000000034B4: D7610025 0001120D
	v_writelane_b32 v37, s14, 10                               // 0000000034BC: D7610025 0001140E
	v_writelane_b32 v37, s15, 11                               // 0000000034C4: D7610025 0001160F
	v_writelane_b32 v37, s16, 12                               // 0000000034CC: D7610025 00011810
	v_writelane_b32 v37, s17, 13                               // 0000000034D4: D7610025 00011A11
	v_writelane_b32 v37, s18, 14                               // 0000000034DC: D7610025 00011C12
	v_writelane_b32 v37, s19, 15                               // 0000000034E4: D7610025 00011E13
	v_writelane_b32 v37, s20, 16                               // 0000000034EC: D7610025 00012014
	v_writelane_b32 v37, s21, 17                               // 0000000034F4: D7610025 00012215
	v_writelane_b32 v37, s22, 18                               // 0000000034FC: D7610025 00012416
	v_writelane_b32 v37, s23, 19                               // 000000003504: D7610025 00012617
	s_cbranch_vccnz 48                                         // 00000000350C: BFA40030 <r_3_3_3_8_8_8+0x1fd0>
	s_load_b512 s[8:23], s[6:7], 0x2500                        // 000000003510: F4100203 F8002500
	s_or_saveexec_b32 s105, -1                                 // 000000003518: BEE922C1
	scratch_load_b32 v45, off, off offset:16                   // 00000000351C: DC510010 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000003524: BEFE0069
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000003528: BF890007
	v_writelane_b32 v45, s8, 20                                // 00000000352C: D761002D 00012808
	v_writelane_b32 v45, s9, 21                                // 000000003534: D761002D 00012A09
	v_writelane_b32 v45, s10, 22                               // 00000000353C: D761002D 00012C0A
	v_writelane_b32 v45, s11, 23                               // 000000003544: D761002D 00012E0B
	v_writelane_b32 v45, s12, 24                               // 00000000354C: D761002D 0001300C
	v_writelane_b32 v45, s13, 25                               // 000000003554: D761002D 0001320D
	v_writelane_b32 v45, s14, 26                               // 00000000355C: D761002D 0001340E
	v_writelane_b32 v45, s15, 27                               // 000000003564: D761002D 0001360F
	v_writelane_b32 v45, s16, 28                               // 00000000356C: D761002D 00013810
	v_writelane_b32 v45, s17, 29                               // 000000003574: D761002D 00013A11
	v_writelane_b32 v45, s18, 30                               // 00000000357C: D761002D 00013C12
	v_writelane_b32 v45, s19, 31                               // 000000003584: D761002D 00013E13
	s_or_saveexec_b32 s105, -1                                 // 00000000358C: BEE922C1
	scratch_store_b32 off, v45, off offset:16                  // 000000003590: DC690010 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000003598: BEFE0069
	s_clause 0x1                                               // 00000000359C: BF850001
	s_load_b256 s[52:59], s[6:7], 0x23e0                       // 0000000035A0: F40C0D03 F80023E0
	s_load_b512 s[36:51], s[6:7], 0x2640                       // 0000000035A8: F4100903 F8002640
	v_writelane_b32 v37, s20, 0                                // 0000000035B0: D7610025 00010014
	v_writelane_b32 v37, s21, 1                                // 0000000035B8: D7610025 00010215
	v_writelane_b32 v37, s22, 2                                // 0000000035C0: D7610025 00010416
	v_writelane_b32 v37, s23, 3                                // 0000000035C8: D7610025 00010617
	s_waitcnt lgkmcnt(0)                                       // 0000000035D0: BF89FC07
	v_writelane_b32 v37, s52, 20                               // 0000000035D4: D7610025 00012834
	s_load_b256 s[92:99], s[6:7], 0x1120                       // 0000000035DC: F40C1703 F8001120
	s_and_b32 vcc_lo, exec_lo, s34                             // 0000000035E4: 8B6A227E
	v_writelane_b32 v37, s53, 21                               // 0000000035E8: D7610025 00012A35
	v_writelane_b32 v37, s54, 22                               // 0000000035F0: D7610025 00012C36
	v_writelane_b32 v37, s55, 23                               // 0000000035F8: D7610025 00012E37
	v_writelane_b32 v37, s56, 24                               // 000000003600: D7610025 00013038
	v_writelane_b32 v37, s57, 25                               // 000000003608: D7610025 00013239
	v_writelane_b32 v37, s58, 26                               // 000000003610: D7610025 0001343A
	v_writelane_b32 v37, s59, 27                               // 000000003618: D7610025 0001363B
	s_waitcnt lgkmcnt(0)                                       // 000000003620: BF89FC07
	v_writelane_b32 v37, s92, 28                               // 000000003624: D7610025 0001385C
	v_writelane_b32 v41, s96, 0                                // 00000000362C: D7610029 00010060
	v_writelane_b32 v37, s93, 29                               // 000000003634: D7610025 00013A5D
	v_writelane_b32 v41, s97, 1                                // 00000000363C: D7610029 00010261
	v_writelane_b32 v37, s94, 30                               // 000000003644: D7610025 00013C5E
	v_writelane_b32 v41, s98, 2                                // 00000000364C: D7610029 00010462
	v_writelane_b32 v37, s95, 31                               // 000000003654: D7610025 00013E5F
	v_writelane_b32 v41, s99, 3                                // 00000000365C: D7610029 00010663
	s_clause 0x1                                               // 000000003664: BF850001
	s_load_b256 s[92:99], s[6:7], 0xfe0                        // 000000003668: F40C1703 F8000FE0
	s_load_b512 s[0:15], s[6:7], 0x1240                        // 000000003670: F4100003 F8001240
	s_waitcnt lgkmcnt(0)                                       // 000000003678: BF89FC07
	v_writelane_b32 v41, s92, 4                                // 00000000367C: D7610029 0001085C
	v_writelane_b32 v41, s93, 5                                // 000000003684: D7610029 00010A5D
	v_writelane_b32 v41, s94, 6                                // 00000000368C: D7610029 00010C5E
	v_writelane_b32 v41, s95, 7                                // 000000003694: D7610029 00010E5F
	v_writelane_b32 v41, s96, 8                                // 00000000369C: D7610029 00011060
	v_writelane_b32 v41, s97, 9                                // 0000000036A4: D7610029 00011261
	v_writelane_b32 v41, s98, 10                               // 0000000036AC: D7610029 00011462
	v_writelane_b32 v41, s99, 11                               // 0000000036B4: D7610029 00011663
	v_writelane_b32 v41, s0, 12                                // 0000000036BC: D7610029 00011800
	v_writelane_b32 v41, s1, 13                                // 0000000036C4: D7610029 00011A01
	v_writelane_b32 v41, s2, 14                                // 0000000036CC: D7610029 00011C02
	v_writelane_b32 v41, s3, 15                                // 0000000036D4: D7610029 00011E03
	v_writelane_b32 v41, s4, 16                                // 0000000036DC: D7610029 00012004
	v_writelane_b32 v41, s5, 17                                // 0000000036E4: D7610029 00012205
	v_writelane_b32 v41, s6, 18                                // 0000000036EC: D7610029 00012406
	v_writelane_b32 v41, s7, 19                                // 0000000036F4: D7610029 00012607
	v_writelane_b32 v41, s8, 20                                // 0000000036FC: D7610029 00012808
	v_writelane_b32 v41, s9, 21                                // 000000003704: D7610029 00012A09
	v_writelane_b32 v41, s10, 22                               // 00000000370C: D7610029 00012C0A
	v_writelane_b32 v41, s11, 23                               // 000000003714: D7610029 00012E0B
	v_writelane_b32 v41, s12, 24                               // 00000000371C: D7610029 0001300C
	v_writelane_b32 v41, s13, 25                               // 000000003724: D7610029 0001320D
	v_writelane_b32 v41, s14, 26                               // 00000000372C: D7610029 0001340E
	v_writelane_b32 v41, s15, 27                               // 000000003734: D7610029 0001360F
	s_cbranch_vccnz 2024                                       // 00000000373C: BFA407E8 <r_3_3_3_8_8_8+0x40e0>
	s_or_saveexec_b32 s105, -1                                 // 000000003740: BEE922C1
	scratch_load_b32 v45, off, off                             // 000000003744: DC510000 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000374C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000003750: BF8903F7
	v_readlane_b32 s0, v45, 30                                 // 000000003754: D7600000 00013D2D
	s_or_saveexec_b32 s105, -1                                 // 00000000375C: BEE922C1
	scratch_store_b32 off, v41, off offset:4                   // 000000003760: DC690004 007C2900
	s_mov_b32 exec_lo, s105                                    // 000000003768: BEFE0069
	s_add_u32 s2, s0, s28                                      // 00000000376C: 80021C00
	v_readlane_b32 s0, v45, 31                                 // 000000003770: D7600000 00013F2D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003778: BF870001
	s_addc_u32 s3, s0, s29                                     // 00000000377C: 82031D00
	s_or_saveexec_b32 s105, -1                                 // 000000003780: BEE922C1
	scratch_load_b32 v45, off, off offset:12                   // 000000003784: DC51000C 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000378C: BEFE0069
	s_mov_b32 s101, s100                                       // 000000003790: BEE50064
	s_mov_b32 s102, s100                                       // 000000003794: BEE60064
	s_mov_b32 s103, s100                                       // 000000003798: BEE70064
	s_mov_b64 s[8:9], s[100:101]                               // 00000000379C: BE880164
	s_mov_b64 s[4:5], s[100:101]                               // 0000000037A0: BE840164
	s_mov_b64 s[10:11], s[102:103]                             // 0000000037A4: BE8A0166
	s_mov_b64 s[6:7], s[102:103]                               // 0000000037A8: BE860166
	v_writelane_b32 v36, s4, 28                                // 0000000037AC: D7610024 00013804
	v_writelane_b32 v40, s8, 0                                 // 0000000037B4: D7610028 00010008
	s_waitcnt vmcnt(0)                                         // 0000000037BC: BF8903F7
	v_readlane_b32 s0, v45, 0                                  // 0000000037C0: D7600000 0001012D
	v_readlane_b32 s1, v45, 1                                  // 0000000037C8: D7600001 0001032D
	v_writelane_b32 v36, s5, 29                                // 0000000037D0: D7610024 00013A05
	v_writelane_b32 v40, s9, 1                                 // 0000000037D8: D7610028 00010209
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000037E0: BF870194
	s_add_u32 s0, s0, s28                                      // 0000000037E4: 80001C00
	s_addc_u32 s1, s1, s29                                     // 0000000037E8: 82011D01
	s_and_b32 vcc_lo, exec_lo, s33                             // 0000000037EC: 8B6A217E
	v_writelane_b32 v36, s6, 30                                // 0000000037F0: D7610024 00013C06
	v_writelane_b32 v40, s10, 2                                // 0000000037F8: D7610028 0001040A
	v_writelane_b32 v36, s7, 31                                // 000000003800: D7610024 00013E07
	v_writelane_b32 v40, s11, 3                                // 000000003808: D7610028 0001060B
	s_or_saveexec_b32 s105, -1                                 // 000000003810: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003814: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000003818: BEFE0069
	s_cbranch_vccnz 32                                         // 00000000381C: BFA40020 <r_3_3_3_8_8_8+0x22a0>
	v_readlane_b32 s4, v45, 2                                  // 000000003820: D7600004 0001052D
	v_readlane_b32 s5, v45, 3                                  // 000000003828: D7600005 0001072D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003830: BF870092
	s_add_u32 s4, s4, s28                                      // 000000003834: 80041C04
	s_addc_u32 s5, s5, s29                                     // 000000003838: 82051D05
	s_load_b256 s[4:11], s[4:5], null                          // 00000000383C: F40C0102 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000003844: BF89FC07
	v_writelane_b32 v36, s4, 28                                // 000000003848: D7610024 00013804
	v_writelane_b32 v36, s5, 29                                // 000000003850: D7610024 00013A05
	v_writelane_b32 v36, s6, 30                                // 000000003858: D7610024 00013C06
	v_writelane_b32 v36, s7, 31                                // 000000003860: D7610024 00013E07
	s_or_saveexec_b32 s105, -1                                 // 000000003868: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000386C: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000003870: BEFE0069
	v_writelane_b32 v40, s8, 0                                 // 000000003874: D7610028 00010008
	v_writelane_b32 v40, s9, 1                                 // 00000000387C: D7610028 00010209
	v_writelane_b32 v40, s10, 2                                // 000000003884: D7610028 0001040A
	v_writelane_b32 v40, s11, 3                                // 00000000388C: D7610028 0001060B
	s_or_saveexec_b32 s105, -1                                 // 000000003894: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003898: BF870009
	s_mov_b32 exec_lo, s105                                    // 00000000389C: BEFE0069
	s_load_b256 s[4:11], s[2:3], null                          // 0000000038A0: F40C0101 F8000000
	s_waitcnt lgkmcnt(0)                                       // 0000000038A8: BF89FC07
	v_writelane_b32 v36, s4, 20                                // 0000000038AC: D7610024 00012804
	v_writelane_b32 v36, s5, 21                                // 0000000038B4: D7610024 00012A05
	v_writelane_b32 v36, s6, 22                                // 0000000038BC: D7610024 00012C06
	v_writelane_b32 v36, s7, 23                                // 0000000038C4: D7610024 00012E07
	v_writelane_b32 v36, s8, 24                                // 0000000038CC: D7610024 00013008
	v_writelane_b32 v36, s9, 25                                // 0000000038D4: D7610024 00013209
	v_writelane_b32 v36, s10, 26                               // 0000000038DC: D7610024 0001340A
	v_writelane_b32 v36, s11, 27                               // 0000000038E4: D7610024 0001360B
	s_load_b256 s[0:7], s[0:1], null                           // 0000000038EC: F40C0000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 0000000038F4: BF89FC07
	v_writelane_b32 v36, s0, 12                                // 0000000038F8: D7610024 00011800
	v_writelane_b32 v36, s1, 13                                // 000000003900: D7610024 00011A01
	v_writelane_b32 v36, s2, 14                                // 000000003908: D7610024 00011C02
	v_writelane_b32 v36, s3, 15                                // 000000003910: D7610024 00011E03
	v_writelane_b32 v36, s4, 16                                // 000000003918: D7610024 00012004
	v_writelane_b32 v36, s5, 17                                // 000000003920: D7610024 00012205
	v_writelane_b32 v36, s6, 18                                // 000000003928: D7610024 00012406
	v_writelane_b32 v36, s7, 19                                // 000000003930: D7610024 00012607
	v_readlane_b32 s0, v45, 4                                  // 000000003938: D7600000 0001092D
	v_readlane_b32 s1, v45, 7                                  // 000000003940: D7600001 00010F2D
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000003948: BF8700A2
	s_add_u32 s2, s0, s28                                      // 00000000394C: 80021C00
	v_readlane_b32 s0, v45, 5                                  // 000000003950: D7600000 00010B2D
	s_addc_u32 s3, s0, s29                                     // 000000003958: 82031D00
	v_readlane_b32 s0, v45, 6                                  // 00000000395C: D7600000 00010D2D
	s_load_b512 s[4:19], s[2:3], null                          // 000000003964: F4100101 F8000000
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000396C: BF870001
	s_add_u32 s0, s0, s28                                      // 000000003970: 80001C00
	s_addc_u32 s1, s1, s29                                     // 000000003974: 82011D01
	s_or_saveexec_b32 s105, -1                                 // 000000003978: BEE922C1
	scratch_load_b32 v45, off, off offset:4                    // 00000000397C: DC510004 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000003984: BEFE0069
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000003988: BF890007
	v_writelane_b32 v45, s4, 28                                // 00000000398C: D761002D 00013804
	v_writelane_b32 v45, s5, 29                                // 000000003994: D761002D 00013A05
	v_writelane_b32 v45, s6, 30                                // 00000000399C: D761002D 00013C06
	v_writelane_b32 v45, s7, 31                                // 0000000039A4: D761002D 00013E07
	s_or_saveexec_b32 s105, -1                                 // 0000000039AC: BEE922C1
	scratch_store_b32 off, v45, off offset:4                   // 0000000039B0: DC690004 007C2D00
	s_mov_b32 exec_lo, s105                                    // 0000000039B8: BEFE0069
	s_add_u32 s2, s0, 0xfffffe40                               // 0000000039BC: 8002FF00 FFFFFE40
	v_writelane_b32 v45, s8, 0                                 // 0000000039C4: D761002D 00010008
	s_addc_u32 s3, s1, -1                                      // 0000000039CC: 8203C101
	v_writelane_b32 v45, s9, 1                                 // 0000000039D0: D761002D 00010209
	v_writelane_b32 v45, s10, 2                                // 0000000039D8: D761002D 0001040A
	v_writelane_b32 v45, s11, 3                                // 0000000039E0: D761002D 0001060B
	v_writelane_b32 v45, s12, 4                                // 0000000039E8: D761002D 0001080C
	v_writelane_b32 v45, s13, 5                                // 0000000039F0: D761002D 00010A0D
	v_writelane_b32 v45, s14, 6                                // 0000000039F8: D761002D 00010C0E
	v_writelane_b32 v45, s15, 7                                // 000000003A00: D761002D 00010E0F
	v_writelane_b32 v45, s16, 8                                // 000000003A08: D761002D 00011010
	v_writelane_b32 v45, s17, 9                                // 000000003A10: D761002D 00011211
	v_writelane_b32 v45, s18, 10                               // 000000003A18: D761002D 00011412
	v_writelane_b32 v45, s19, 11                               // 000000003A20: D761002D 00011613
	s_load_b256 s[4:11], s[0:1], 0x1100                        // 000000003A28: F40C0100 F8001100
	s_waitcnt lgkmcnt(0)                                       // 000000003A30: BF89FC07
	v_writelane_b32 v36, s4, 4                                 // 000000003A34: D7610024 00010804
	v_writelane_b32 v36, s5, 5                                 // 000000003A3C: D7610024 00010A05
	v_writelane_b32 v36, s6, 6                                 // 000000003A44: D7610024 00010C06
	v_writelane_b32 v36, s7, 7                                 // 000000003A4C: D7610024 00010E07
	v_writelane_b32 v36, s8, 8                                 // 000000003A54: D7610024 00011008
	v_writelane_b32 v36, s9, 9                                 // 000000003A5C: D7610024 00011209
	v_writelane_b32 v36, s10, 10                               // 000000003A64: D7610024 0001140A
	v_writelane_b32 v36, s11, 11                               // 000000003A6C: D7610024 0001160B
	s_load_b512 s[8:23], s[2:3], null                          // 000000003A74: F4100201 F8000000
	s_add_u32 s4, s0, 0xfffffbe0                               // 000000003A7C: 8004FF00 FFFFFBE0
	s_addc_u32 s5, s1, -1                                      // 000000003A84: 8205C101
	s_waitcnt lgkmcnt(0)                                       // 000000003A88: BF89FC07
	v_writelane_b32 v45, s8, 20                                // 000000003A8C: D761002D 00012808
	v_writelane_b32 v36, s20, 0                                // 000000003A94: D7610024 00010014
	v_writelane_b32 v45, s9, 21                                // 000000003A9C: D761002D 00012A09
	v_writelane_b32 v36, s21, 1                                // 000000003AA4: D7610024 00010215
	v_writelane_b32 v45, s10, 22                               // 000000003AAC: D761002D 00012C0A
	v_writelane_b32 v36, s22, 2                                // 000000003AB4: D7610024 00010416
	v_writelane_b32 v45, s11, 23                               // 000000003ABC: D761002D 00012E0B
	v_writelane_b32 v36, s23, 3                                // 000000003AC4: D7610024 00010617
	v_writelane_b32 v45, s12, 24                               // 000000003ACC: D761002D 0001300C
	v_writelane_b32 v45, s13, 25                               // 000000003AD4: D761002D 0001320D
	v_writelane_b32 v45, s14, 26                               // 000000003ADC: D761002D 0001340E
	v_writelane_b32 v45, s15, 27                               // 000000003AE4: D761002D 0001360F
	v_writelane_b32 v45, s16, 28                               // 000000003AEC: D761002D 00013810
	v_writelane_b32 v45, s17, 29                               // 000000003AF4: D761002D 00013A11
	v_writelane_b32 v45, s18, 30                               // 000000003AFC: D761002D 00013C12
	v_writelane_b32 v45, s19, 31                               // 000000003B04: D761002D 00013E13
	s_or_saveexec_b32 s105, -1                                 // 000000003B0C: BEE922C1
	scratch_store_b32 off, v36, off offset:24                  // 000000003B10: DC690018 007C2400
	s_mov_b32 exec_lo, s105                                    // 000000003B18: BEFE0069
	s_load_b256 s[4:11], s[4:5], null                          // 000000003B1C: F40C0102 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000003B24: BF89FC07
	v_writelane_b32 v45, s4, 12                                // 000000003B28: D761002D 00011804
	v_writelane_b32 v45, s5, 13                                // 000000003B30: D761002D 00011A05
	v_writelane_b32 v45, s6, 14                                // 000000003B38: D761002D 00011C06
	v_writelane_b32 v45, s7, 15                                // 000000003B40: D761002D 00011E07
	v_writelane_b32 v45, s8, 16                                // 000000003B48: D761002D 00012008
	v_writelane_b32 v45, s9, 17                                // 000000003B50: D761002D 00012209
	v_writelane_b32 v45, s10, 18                               // 000000003B58: D761002D 0001240A
	v_writelane_b32 v45, s11, 19                               // 000000003B60: D761002D 0001260B
	s_or_saveexec_b32 s105, -1                                 // 000000003B68: BEE922C1
	scratch_store_b32 off, v45, off offset:20                  // 000000003B6C: DC690014 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000003B74: BEFE0069
	s_mov_b32 s8, 0                                            // 000000003B78: BE880080
	s_mov_b32 s2, vcc_hi                                       // 000000003B7C: BE82006B
	s_branch 240                                               // 000000003B80: BFA000F0 <r_3_3_3_8_8_8+0x2944>
	s_or_saveexec_b32 s105, -1                                 // 000000003B84: BEE922C1
	scratch_load_b32 v45, off, off offset:12                   // 000000003B88: DC51000C 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000003B90: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000003B94: BF8903F7
	v_readlane_b32 s0, v45, 0                                  // 000000003B98: D7600000 0001012D
	v_readlane_b32 s1, v45, 1                                  // 000000003BA0: D7600001 0001032D
	v_readlane_b32 s2, v45, 2                                  // 000000003BA8: D7600002 0001052D
	s_mov_b32 s105, exec_lo                                    // 000000003BB0: BEE9007E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000003BB4: BF870113
	s_add_u32 s0, s0, s28                                      // 000000003BB8: 80001C00
	s_addc_u32 s1, s1, s29                                     // 000000003BBC: 82011D01
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003BC0: BF870001
	s_add_u32 s2, s2, s28                                      // 000000003BC4: 80021C02
	s_load_b256 s[4:11], s[0:1], null                          // 000000003BC8: F40C0100 F8000000
	s_mov_b32 exec_lo, -1                                      // 000000003BD0: BEFE00C1
	scratch_load_b32 v36, off, off offset:24                   // 000000003BD4: DC510018 247C0000
	s_mov_b32 exec_lo, s105                                    // 000000003BDC: BEFE0069
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000003BE0: BF890007
	v_writelane_b32 v36, s4, 12                                // 000000003BE4: D7610024 00011804
	v_readlane_b32 s0, v45, 3                                  // 000000003BEC: D7600000 0001072D
	v_readlane_b32 s1, v45, 7                                  // 000000003BF4: D7600001 00010F2D
	v_writelane_b32 v36, s5, 13                                // 000000003BFC: D7610024 00011A05
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000003C04: BF870133
	s_addc_u32 s3, s0, s29                                     // 000000003C08: 82031D00
	v_readlane_b32 s0, v45, 6                                  // 000000003C0C: D7600000 00010D2D
	v_writelane_b32 v36, s6, 14                                // 000000003C14: D7610024 00011C06
	s_add_u32 s0, s0, s28                                      // 000000003C1C: 80001C00
	s_addc_u32 s1, s1, s29                                     // 000000003C20: 82011D01
	v_writelane_b32 v36, s7, 15                                // 000000003C24: D7610024 00011E07
	v_writelane_b32 v36, s8, 16                                // 000000003C2C: D7610024 00012008
	v_writelane_b32 v36, s9, 17                                // 000000003C34: D7610024 00012209
	v_writelane_b32 v36, s10, 18                               // 000000003C3C: D7610024 0001240A
	v_writelane_b32 v36, s11, 19                               // 000000003C44: D7610024 0001260B
	s_load_b256 s[4:11], s[2:3], null                          // 000000003C4C: F40C0101 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000003C54: BF89FC07
	v_writelane_b32 v36, s4, 28                                // 000000003C58: D7610024 00013804
	v_writelane_b32 v36, s5, 29                                // 000000003C60: D7610024 00013A05
	v_writelane_b32 v36, s6, 30                                // 000000003C68: D7610024 00013C06
	v_writelane_b32 v36, s7, 31                                // 000000003C70: D7610024 00013E07
	s_or_saveexec_b32 s105, -1                                 // 000000003C78: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003C7C: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000003C80: BEFE0069
	v_writelane_b32 v40, s8, 0                                 // 000000003C84: D7610028 00010008
	v_writelane_b32 v40, s9, 1                                 // 000000003C8C: D7610028 00010209
	v_writelane_b32 v40, s10, 2                                // 000000003C94: D7610028 0001040A
	v_writelane_b32 v40, s11, 3                                // 000000003C9C: D7610028 0001060B
	s_or_saveexec_b32 s105, -1                                 // 000000003CA4: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003CA8: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000003CAC: BEFE0069
	s_load_b256 s[4:11], s[0:1], 0x1100                        // 000000003CB0: F40C0100 F8001100
	s_mov_b32 s101, s100                                       // 000000003CB8: BEE50064
	s_mov_b32 s102, s100                                       // 000000003CBC: BEE60064
	s_mov_b32 s103, s100                                       // 000000003CC0: BEE70064
	s_mov_b32 s2, -1                                           // 000000003CC4: BE8200C1
	s_mov_b64 s[24:25], s[100:101]                             // 000000003CC8: BE980164
	s_mov_b64 s[96:97], s[100:101]                             // 000000003CCC: BEE00164
	s_mov_b64 s[76:77], s[100:101]                             // 000000003CD0: BECC0164
	s_mov_b64 s[16:17], s[100:101]                             // 000000003CD4: BE900164
	s_mov_b64 s[12:13], s[100:101]                             // 000000003CD8: BE8C0164
	s_mov_b64 s[68:69], s[100:101]                             // 000000003CDC: BEC40164
	s_mov_b64 s[72:73], s[100:101]                             // 000000003CE0: BEC80164
	s_mov_b64 s[92:93], s[100:101]                             // 000000003CE4: BEDC0164
	s_mov_b64 s[26:27], s[102:103]                             // 000000003CE8: BE9A0166
	s_mov_b64 s[98:99], s[102:103]                             // 000000003CEC: BEE20166
	s_mov_b64 s[78:79], s[102:103]                             // 000000003CF0: BECE0166
	s_mov_b64 s[18:19], s[102:103]                             // 000000003CF4: BE920166
	s_mov_b64 s[14:15], s[102:103]                             // 000000003CF8: BE8E0166
	s_mov_b64 s[70:71], s[102:103]                             // 000000003CFC: BEC60166
	s_mov_b64 s[74:75], s[102:103]                             // 000000003D00: BECA0166
	s_waitcnt lgkmcnt(0)                                       // 000000003D04: BF89FC07
	v_writelane_b32 v36, s4, 4                                 // 000000003D08: D7610024 00010804
	s_mov_b64 s[94:95], s[102:103]                             // 000000003D10: BEDE0166
	v_writelane_b32 v36, s5, 5                                 // 000000003D14: D7610024 00010A05
	v_writelane_b32 v36, s6, 6                                 // 000000003D1C: D7610024 00010C06
	v_writelane_b32 v36, s7, 7                                 // 000000003D24: D7610024 00010E07
	v_writelane_b32 v36, s8, 8                                 // 000000003D2C: D7610024 00011008
	v_writelane_b32 v36, s9, 9                                 // 000000003D34: D7610024 00011209
	v_writelane_b32 v36, s10, 10                               // 000000003D3C: D7610024 0001140A
	v_writelane_b32 v36, s11, 11                               // 000000003D44: D7610024 0001160B
	s_mov_b64 s[8:9], s[100:101]                               // 000000003D4C: BE880164
	s_mov_b64 s[4:5], s[100:101]                               // 000000003D50: BE840164
	s_mov_b64 s[10:11], s[102:103]                             // 000000003D54: BE8A0166
	s_mov_b64 s[6:7], s[102:103]                               // 000000003D58: BE860166
	v_writelane_b32 v36, s4, 20                                // 000000003D5C: D7610024 00012804
	v_writelane_b32 v36, s5, 21                                // 000000003D64: D7610024 00012A05
	v_writelane_b32 v36, s6, 22                                // 000000003D6C: D7610024 00012C06
	v_writelane_b32 v36, s7, 23                                // 000000003D74: D7610024 00012E07
	v_writelane_b32 v36, s8, 24                                // 000000003D7C: D7610024 00013008
	v_writelane_b32 v36, s9, 25                                // 000000003D84: D7610024 00013209
	v_writelane_b32 v36, s10, 26                               // 000000003D8C: D7610024 0001340A
	v_writelane_b32 v36, s11, 27                               // 000000003D94: D7610024 0001360B
	s_or_saveexec_b32 s105, -1                                 // 000000003D9C: BEE922C1
	scratch_load_b32 v45, off, off offset:20                   // 000000003DA0: DC510014 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000003DA8: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000003DAC: BF8903F7
	v_writelane_b32 v45, s92, 12                               // 000000003DB0: D761002D 0001185C
	s_mov_b64 s[20:21], s[100:101]                             // 000000003DB8: BE940164
	s_mov_b64 s[22:23], s[102:103]                             // 000000003DBC: BE960166
	v_writelane_b32 v45, s93, 13                               // 000000003DC0: D761002D 00011A5D
	v_writelane_b32 v45, s94, 14                               // 000000003DC8: D761002D 00011C5E
	v_writelane_b32 v45, s95, 15                               // 000000003DD0: D761002D 00011E5F
	v_writelane_b32 v45, s96, 16                               // 000000003DD8: D761002D 00012060
	v_writelane_b32 v45, s97, 17                               // 000000003DE0: D761002D 00012261
	v_writelane_b32 v45, s98, 18                               // 000000003DE8: D761002D 00012462
	v_writelane_b32 v45, s99, 19                               // 000000003DF0: D761002D 00012663
	s_or_saveexec_b32 s105, -1                                 // 000000003DF8: BEE922C1
	scratch_load_b32 v41, off, off offset:4                    // 000000003DFC: DC510004 297C0000
	s_mov_b32 exec_lo, s105                                    // 000000003E04: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000003E08: BF8903F7
	v_writelane_b32 v41, s12, 28                               // 000000003E0C: D7610029 0001380C
	v_writelane_b32 v41, s13, 29                               // 000000003E14: D7610029 00013A0D
	v_writelane_b32 v41, s14, 30                               // 000000003E1C: D7610029 00013C0E
	v_writelane_b32 v41, s15, 31                               // 000000003E24: D7610029 00013E0F
	s_or_saveexec_b32 s105, -1                                 // 000000003E2C: BEE922C1
	scratch_store_b32 off, v41, off offset:4                   // 000000003E30: DC690004 007C2900
	s_mov_b32 exec_lo, s105                                    // 000000003E38: BEFE0069
	v_writelane_b32 v45, s16, 0                                // 000000003E3C: D761002D 00010010
	s_mov_b64 s[80:81], s[100:101]                             // 000000003E44: BED00164
	s_mov_b64 s[82:83], s[102:103]                             // 000000003E48: BED20166
	v_writelane_b32 v45, s17, 1                                // 000000003E4C: D761002D 00010211
	v_writelane_b32 v45, s18, 2                                // 000000003E54: D761002D 00010412
	v_writelane_b32 v45, s19, 3                                // 000000003E5C: D761002D 00010613
	v_writelane_b32 v45, s20, 4                                // 000000003E64: D761002D 00010814
	v_writelane_b32 v45, s21, 5                                // 000000003E6C: D761002D 00010A15
	v_writelane_b32 v45, s22, 6                                // 000000003E74: D761002D 00010C16
	v_writelane_b32 v45, s23, 7                                // 000000003E7C: D761002D 00010E17
	v_writelane_b32 v45, s24, 8                                // 000000003E84: D761002D 00011018
	v_writelane_b32 v45, s25, 9                                // 000000003E8C: D761002D 00011219
	v_writelane_b32 v45, s26, 10                               // 000000003E94: D761002D 0001141A
	v_writelane_b32 v45, s27, 11                               // 000000003E9C: D761002D 0001161B
	v_writelane_b32 v45, s68, 20                               // 000000003EA4: D761002D 00012844
	v_writelane_b32 v45, s69, 21                               // 000000003EAC: D761002D 00012A45
	v_writelane_b32 v45, s70, 22                               // 000000003EB4: D761002D 00012C46
	v_writelane_b32 v45, s71, 23                               // 000000003EBC: D761002D 00012E47
	v_writelane_b32 v45, s72, 24                               // 000000003EC4: D761002D 00013048
	v_writelane_b32 v45, s73, 25                               // 000000003ECC: D761002D 00013249
	v_writelane_b32 v45, s74, 26                               // 000000003ED4: D761002D 0001344A
	v_writelane_b32 v45, s75, 27                               // 000000003EDC: D761002D 0001364B
	v_writelane_b32 v45, s76, 28                               // 000000003EE4: D761002D 0001384C
	v_writelane_b32 v45, s77, 29                               // 000000003EEC: D761002D 00013A4D
	v_writelane_b32 v45, s78, 30                               // 000000003EF4: D761002D 00013C4E
	v_writelane_b32 v45, s79, 31                               // 000000003EFC: D761002D 00013E4F
	s_or_saveexec_b32 s105, -1                                 // 000000003F04: BEE922C1
	scratch_store_b32 off, v45, off offset:20                  // 000000003F08: DC690014 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000003F10: BEFE0069
	v_writelane_b32 v36, s80, 0                                // 000000003F14: D7610024 00010050
	v_writelane_b32 v36, s81, 1                                // 000000003F1C: D7610024 00010251
	v_writelane_b32 v36, s82, 2                                // 000000003F24: D7610024 00010452
	v_writelane_b32 v36, s83, 3                                // 000000003F2C: D7610024 00010653
	s_or_saveexec_b32 s105, -1                                 // 000000003F34: BEE922C1
	scratch_store_b32 off, v36, off offset:24                  // 000000003F38: DC690018 007C2400
	s_mov_b32 exec_lo, s105                                    // 000000003F40: BEFE0069
	s_mov_b32 s20, s8                                          // 000000003F44: BE940008
	s_mov_b32 s21, s8                                          // 000000003F48: BE950008
	s_mov_b32 s22, s8                                          // 000000003F4C: BE960008
	s_mov_b32 s23, s8                                          // 000000003F50: BE970008
	s_mov_b32 s76, s8                                          // 000000003F54: BECC0008
	s_mov_b32 s77, s8                                          // 000000003F58: BECD0008
	s_mov_b32 s78, s8                                          // 000000003F5C: BECE0008
	s_mov_b32 s79, s8                                          // 000000003F60: BECF0008
	s_mov_b32 s68, s8                                          // 000000003F64: BEC40008
	s_mov_b32 s69, s8                                          // 000000003F68: BEC50008
	s_mov_b32 s70, s8                                          // 000000003F6C: BEC60008
	s_mov_b32 s71, s8                                          // 000000003F70: BEC70008
	s_mov_b32 s72, s8                                          // 000000003F74: BEC80008
	s_mov_b32 s73, s8                                          // 000000003F78: BEC90008
	s_mov_b32 s74, s8                                          // 000000003F7C: BECA0008
	s_mov_b32 s75, s8                                          // 000000003F80: BECB0008
	s_mov_b32 s12, s8                                          // 000000003F84: BE8C0008
	s_mov_b32 s13, s8                                          // 000000003F88: BE8D0008
	s_mov_b32 s14, s8                                          // 000000003F8C: BE8E0008
	s_mov_b32 s15, s8                                          // 000000003F90: BE8F0008
	s_mov_b32 s16, s8                                          // 000000003F94: BE900008
	s_mov_b32 s17, s8                                          // 000000003F98: BE910008
	s_mov_b32 s18, s8                                          // 000000003F9C: BE920008
	s_mov_b32 s19, s8                                          // 000000003FA0: BE930008
	s_mov_b32 s80, s8                                          // 000000003FA4: BED00008
	s_mov_b32 s81, s8                                          // 000000003FA8: BED10008
	s_mov_b32 s82, s8                                          // 000000003FAC: BED20008
	s_mov_b32 s83, s8                                          // 000000003FB0: BED30008
	s_mov_b32 s24, s8                                          // 000000003FB4: BE980008
	s_mov_b32 s25, s8                                          // 000000003FB8: BE990008
	s_mov_b32 s26, s8                                          // 000000003FBC: BE9A0008
	s_mov_b32 s9, s8                                           // 000000003FC0: BE890008
	s_mov_b32 s10, s8                                          // 000000003FC4: BE8A0008
	s_mov_b32 s11, s8                                          // 000000003FC8: BE8B0008
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 000000003FCC: 916A027E
	s_mov_b32 s4, s8                                           // 000000003FD0: BE840008
	s_mov_b32 s5, s8                                           // 000000003FD4: BE850008
	s_mov_b32 s6, s8                                           // 000000003FD8: BE860008
	s_mov_b32 s7, s8                                           // 000000003FDC: BE870008
	s_or_saveexec_b32 s105, -1                                 // 000000003FE0: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000003FE4: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000003FE8: BEFE0069
	v_writelane_b32 v40, s4, 4                                 // 000000003FEC: D7610028 00010804
	s_mov_b32 s27, s8                                          // 000000003FF4: BE9B0008
	v_writelane_b32 v40, s5, 5                                 // 000000003FF8: D7610028 00010A05
	v_writelane_b32 v40, s6, 6                                 // 000000004000: D7610028 00010C06
	v_writelane_b32 v40, s7, 7                                 // 000000004008: D7610028 00010E07
	v_writelane_b32 v40, s8, 8                                 // 000000004010: D7610028 00011008
	v_writelane_b32 v40, s9, 9                                 // 000000004018: D7610028 00011209
	v_writelane_b32 v40, s10, 10                               // 000000004020: D7610028 0001140A
	v_writelane_b32 v40, s11, 11                               // 000000004028: D7610028 0001160B
	s_cbranch_vccnz 24                                         // 000000004030: BFA40018 <r_3_3_3_8_8_8+0x2a94>
	s_clause 0x2                                               // 000000004034: BF850002
	s_load_b512 s[68:83], s[0:1], 0x2500                       // 000000004038: F4101100 F8002500
	s_load_b256 s[4:11], s[0:1], 0x23e0                        // 000000004040: F40C0100 F80023E0
	s_load_b512 s[12:27], s[0:1], 0x2640                       // 000000004048: F4100300 F8002640
	s_waitcnt lgkmcnt(0)                                       // 000000004050: BF89FC07
	v_writelane_b32 v40, s4, 4                                 // 000000004054: D7610028 00010804
	v_writelane_b32 v40, s5, 5                                 // 00000000405C: D7610028 00010A05
	v_writelane_b32 v40, s6, 6                                 // 000000004064: D7610028 00010C06
	v_writelane_b32 v40, s7, 7                                 // 00000000406C: D7610028 00010E07
	v_writelane_b32 v40, s8, 8                                 // 000000004074: D7610028 00011008
	v_writelane_b32 v40, s9, 9                                 // 00000000407C: D7610028 00011209
	v_writelane_b32 v40, s10, 10                               // 000000004084: D7610028 0001140A
	v_writelane_b32 v40, s11, 11                               // 00000000408C: D7610028 0001160B
	s_clause 0x1                                               // 000000004094: BF850001
	s_load_b256 s[4:11], s[0:1], 0x1120                        // 000000004098: F40C0100 F8001120
	s_load_b512 s[84:99], s[0:1], 0x1240                       // 0000000040A0: F4101500 F8001240
	s_waitcnt lgkmcnt(0)                                       // 0000000040A8: BF89FC07
	v_writelane_b32 v40, s4, 12                                // 0000000040AC: D7610028 00011804
	v_writelane_b32 v40, s5, 13                                // 0000000040B4: D7610028 00011A05
	v_writelane_b32 v40, s6, 14                                // 0000000040BC: D7610028 00011C06
	v_writelane_b32 v40, s7, 15                                // 0000000040C4: D7610028 00011E07
	v_writelane_b32 v40, s8, 16                                // 0000000040CC: D7610028 00012008
	v_writelane_b32 v40, s9, 17                                // 0000000040D4: D7610028 00012209
	v_writelane_b32 v40, s10, 18                               // 0000000040DC: D7610028 0001240A
	v_writelane_b32 v40, s11, 19                               // 0000000040E4: D7610028 0001260B
	s_load_b256 s[4:11], s[0:1], 0xfe0                         // 0000000040EC: F40C0100 F8000FE0
	s_waitcnt lgkmcnt(0)                                       // 0000000040F4: BF89FC07
	v_writelane_b32 v40, s4, 20                                // 0000000040F8: D7610028 00012804
	v_writelane_b32 v40, s5, 21                                // 000000004100: D7610028 00012A05
	v_writelane_b32 v40, s6, 22                                // 000000004108: D7610028 00012C06
	v_writelane_b32 v40, s7, 23                                // 000000004110: D7610028 00012E07
	v_writelane_b32 v40, s8, 24                                // 000000004118: D7610028 00013008
	v_writelane_b32 v40, s9, 25                                // 000000004120: D7610028 00013209
	v_writelane_b32 v40, s10, 26                               // 000000004128: D7610028 0001340A
	v_writelane_b32 v40, s11, 27                               // 000000004130: D7610028 0001360B
	v_writelane_b32 v40, s84, 28                               // 000000004138: D7610028 00013854
	v_writelane_b32 v40, s85, 29                               // 000000004140: D7610028 00013A55
	v_writelane_b32 v40, s86, 30                               // 000000004148: D7610028 00013C56
	v_writelane_b32 v40, s87, 31                               // 000000004150: D7610028 00013E57
	s_or_saveexec_b32 s105, -1                                 // 000000004158: BEE922C1
	scratch_store_b32 off, v40, off offset:68                  // 00000000415C: DC690044 007C2800
	s_mov_b32 exec_lo, s105                                    // 000000004164: BEFE0069
	v_writelane_b32 v36, s88, 0                                // 000000004168: D7610024 00010058
	s_and_b32 vcc_lo, exec_lo, s34                             // 000000004170: 8B6A227E
	v_writelane_b32 v36, s89, 1                                // 000000004174: D7610024 00010259
	v_writelane_b32 v36, s90, 2                                // 00000000417C: D7610024 0001045A
	v_writelane_b32 v36, s91, 3                                // 000000004184: D7610024 0001065B
	v_writelane_b32 v36, s92, 4                                // 00000000418C: D7610024 0001085C
	v_writelane_b32 v36, s93, 5                                // 000000004194: D7610024 00010A5D
	v_writelane_b32 v36, s94, 6                                // 00000000419C: D7610024 00010C5E
	v_writelane_b32 v36, s95, 7                                // 0000000041A4: D7610024 00010E5F
	v_writelane_b32 v36, s96, 8                                // 0000000041AC: D7610024 00011060
	v_writelane_b32 v36, s97, 9                                // 0000000041B4: D7610024 00011261
	v_writelane_b32 v36, s98, 10                               // 0000000041BC: D7610024 00011462
	v_writelane_b32 v36, s99, 11                               // 0000000041C4: D7610024 00011663
	v_writelane_b32 v36, s12, 12                               // 0000000041CC: D7610024 0001180C
	v_writelane_b32 v36, s13, 13                               // 0000000041D4: D7610024 00011A0D
	v_writelane_b32 v36, s14, 14                               // 0000000041DC: D7610024 00011C0E
	v_writelane_b32 v36, s15, 15                               // 0000000041E4: D7610024 00011E0F
	v_writelane_b32 v36, s16, 16                               // 0000000041EC: D7610024 00012010
	v_writelane_b32 v36, s17, 17                               // 0000000041F4: D7610024 00012211
	v_writelane_b32 v36, s18, 18                               // 0000000041FC: D7610024 00012412
	v_writelane_b32 v36, s19, 19                               // 000000004204: D7610024 00012613
	v_writelane_b32 v36, s20, 20                               // 00000000420C: D7610024 00012814
	v_writelane_b32 v36, s21, 21                               // 000000004214: D7610024 00012A15
	v_writelane_b32 v36, s22, 22                               // 00000000421C: D7610024 00012C16
	v_writelane_b32 v36, s23, 23                               // 000000004224: D7610024 00012E17
	v_writelane_b32 v36, s24, 24                               // 00000000422C: D7610024 00013018
	v_writelane_b32 v36, s25, 25                               // 000000004234: D7610024 00013219
	v_writelane_b32 v36, s26, 26                               // 00000000423C: D7610024 0001341A
	v_writelane_b32 v36, s27, 27                               // 000000004244: D7610024 0001361B
	s_cbranch_vccnz 1480                                       // 00000000424C: BFA405C8 <r_3_3_3_8_8_8+0x4370>
	s_or_saveexec_b32 s105, -1                                 // 000000004250: BEE922C1
	scratch_load_b32 v45, off, off offset:12                   // 000000004254: DC51000C 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000425C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000004260: BF8903F7
	v_readlane_b32 s0, v45, 8                                  // 000000004264: D7600000 0001112D
	s_or_saveexec_b32 s105, -1                                 // 00000000426C: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004270: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000004274: BEFE0069
	s_mov_b32 s101, s100                                       // 000000004278: BEE50064
	s_mov_b32 s102, s100                                       // 00000000427C: BEE60064
	s_mov_b32 s103, s100                                       // 000000004280: BEE70064
	s_mov_b64 s[8:9], s[100:101]                               // 000000004284: BE880164
	s_mov_b64 s[4:5], s[100:101]                               // 000000004288: BE840164
	s_mov_b64 s[10:11], s[102:103]                             // 00000000428C: BE8A0166
	s_mov_b64 s[6:7], s[102:103]                               // 000000004290: BE860166
	v_writelane_b32 v41, s4, 20                                // 000000004294: D7610029 00012804
	s_add_u32 s2, s0, s28                                      // 00000000429C: 80021C00
	v_readlane_b32 s0, v45, 9                                  // 0000000042A0: D7600000 0001132D
	v_readlane_b32 s1, v45, 11                                 // 0000000042A8: D7600001 0001172D
	v_writelane_b32 v41, s5, 21                                // 0000000042B0: D7610029 00012A05
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000042B8: BF870133
	s_addc_u32 s3, s0, s29                                     // 0000000042BC: 82031D00
	v_readlane_b32 s0, v45, 10                                 // 0000000042C0: D7600000 0001152D
	v_writelane_b32 v41, s6, 22                                // 0000000042C8: D7610029 00012C06
	s_add_u32 s0, s0, s28                                      // 0000000042D0: 80001C00
	s_addc_u32 s1, s1, s29                                     // 0000000042D4: 82011D01
	v_writelane_b32 v41, s7, 23                                // 0000000042D8: D7610029 00012E07
	s_and_b32 vcc_lo, exec_lo, s33                             // 0000000042E0: 8B6A217E
	v_writelane_b32 v41, s8, 24                                // 0000000042E4: D7610029 00013008
	v_writelane_b32 v41, s9, 25                                // 0000000042EC: D7610029 00013209
	v_writelane_b32 v41, s10, 26                               // 0000000042F4: D7610029 0001340A
	v_writelane_b32 v41, s11, 27                               // 0000000042FC: D7610029 0001360B
	s_cbranch_vccnz 26                                         // 000000004304: BFA4001A <r_3_3_3_8_8_8+0x2d70>
	v_readlane_b32 s4, v45, 12                                 // 000000004308: D7600004 0001192D
	v_readlane_b32 s5, v45, 13                                 // 000000004310: D7600005 00011B2D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004318: BF870092
	s_add_u32 s4, s4, s28                                      // 00000000431C: 80041C04
	s_addc_u32 s5, s5, s29                                     // 000000004320: 82051D05
	s_load_b256 s[4:11], s[4:5], null                          // 000000004324: F40C0102 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000432C: BF89FC07
	v_writelane_b32 v41, s4, 20                                // 000000004330: D7610029 00012804
	v_writelane_b32 v41, s5, 21                                // 000000004338: D7610029 00012A05
	v_writelane_b32 v41, s6, 22                                // 000000004340: D7610029 00012C06
	v_writelane_b32 v41, s7, 23                                // 000000004348: D7610029 00012E07
	v_writelane_b32 v41, s8, 24                                // 000000004350: D7610029 00013008
	v_writelane_b32 v41, s9, 25                                // 000000004358: D7610029 00013209
	v_writelane_b32 v41, s10, 26                               // 000000004360: D7610029 0001340A
	v_writelane_b32 v41, s11, 27                               // 000000004368: D7610029 0001360B
	s_load_b256 s[4:11], s[2:3], null                          // 000000004370: F40C0101 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000004378: BF89FC07
	v_writelane_b32 v41, s4, 12                                // 00000000437C: D7610029 00011804
	v_writelane_b32 v41, s5, 13                                // 000000004384: D7610029 00011A05
	v_writelane_b32 v41, s6, 14                                // 00000000438C: D7610029 00011C06
	v_writelane_b32 v41, s7, 15                                // 000000004394: D7610029 00011E07
	v_writelane_b32 v41, s8, 16                                // 00000000439C: D7610029 00012008
	v_writelane_b32 v41, s9, 17                                // 0000000043A4: D7610029 00012209
	v_writelane_b32 v41, s10, 18                               // 0000000043AC: D7610029 0001240A
	v_writelane_b32 v41, s11, 19                               // 0000000043B4: D7610029 0001260B
	s_load_b256 s[0:7], s[0:1], null                           // 0000000043BC: F40C0000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 0000000043C4: BF89FC07
	v_writelane_b32 v41, s0, 4                                 // 0000000043C8: D7610029 00010800
	v_writelane_b32 v41, s1, 5                                 // 0000000043D0: D7610029 00010A01
	v_writelane_b32 v41, s2, 6                                 // 0000000043D8: D7610029 00010C02
	v_writelane_b32 v41, s3, 7                                 // 0000000043E0: D7610029 00010E03
	v_writelane_b32 v41, s4, 8                                 // 0000000043E8: D7610029 00011004
	v_writelane_b32 v41, s5, 9                                 // 0000000043F0: D7610029 00011205
	v_writelane_b32 v41, s6, 10                                // 0000000043F8: D7610029 00011406
	v_writelane_b32 v41, s7, 11                                // 000000004400: D7610029 00011607
	v_readlane_b32 s0, v45, 14                                 // 000000004408: D7600000 00011D2D
	v_readlane_b32 s1, v45, 17                                 // 000000004410: D7600001 0001232D
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000004418: BF8700A2
	s_add_u32 s2, s0, s28                                      // 00000000441C: 80021C00
	v_readlane_b32 s0, v45, 15                                 // 000000004420: D7600000 00011F2D
	s_addc_u32 s3, s0, s29                                     // 000000004428: 82031D00
	v_readlane_b32 s0, v45, 16                                 // 00000000442C: D7600000 0001212D
	s_load_b512 s[4:19], s[2:3], null                          // 000000004434: F4100101 F8000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000443C: BF8704B1
	s_add_u32 s0, s0, s28                                      // 000000004440: 80001C00
	s_addc_u32 s1, s1, s29                                     // 000000004444: 82011D01
	s_or_saveexec_b32 s105, -1                                 // 000000004448: BEE922C1
	s_mov_b32 exec_lo, s105                                    // 00000000444C: BEFE0069
	s_waitcnt lgkmcnt(0)                                       // 000000004450: BF89FC07
	v_writelane_b32 v36, s4, 28                                // 000000004454: D7610024 00013804
	v_writelane_b32 v36, s5, 29                                // 00000000445C: D7610024 00013A05
	v_writelane_b32 v36, s6, 30                                // 000000004464: D7610024 00013C06
	v_writelane_b32 v36, s7, 31                                // 00000000446C: D7610024 00013E07
	s_or_saveexec_b32 s105, -1                                 // 000000004474: BEE922C1
	scratch_store_b32 off, v36, off offset:36                  // 000000004478: DC690024 007C2400
	s_mov_b32 exec_lo, s105                                    // 000000004480: BEFE0069
	v_writelane_b32 v40, s8, 0                                 // 000000004484: D7610028 00010008
	s_add_u32 s2, s0, 0xfffffe40                               // 00000000448C: 8002FF00 FFFFFE40
	s_addc_u32 s3, s1, -1                                      // 000000004494: 8203C101
	v_writelane_b32 v40, s9, 1                                 // 000000004498: D7610028 00010209
	v_writelane_b32 v40, s10, 2                                // 0000000044A0: D7610028 0001040A
	v_writelane_b32 v40, s11, 3                                // 0000000044A8: D7610028 0001060B
	v_writelane_b32 v40, s12, 4                                // 0000000044B0: D7610028 0001080C
	v_writelane_b32 v40, s13, 5                                // 0000000044B8: D7610028 00010A0D
	v_writelane_b32 v40, s14, 6                                // 0000000044C0: D7610028 00010C0E
	v_writelane_b32 v40, s15, 7                                // 0000000044C8: D7610028 00010E0F
	v_writelane_b32 v40, s16, 8                                // 0000000044D0: D7610028 00011010
	v_writelane_b32 v40, s17, 9                                // 0000000044D8: D7610028 00011211
	v_writelane_b32 v40, s18, 10                               // 0000000044E0: D7610028 00011412
	v_writelane_b32 v40, s19, 11                               // 0000000044E8: D7610028 00011613
	s_load_b256 s[4:11], s[0:1], 0x1100                        // 0000000044F0: F40C0100 F8001100
	s_waitcnt lgkmcnt(0)                                       // 0000000044F8: BF89FC07
	v_writelane_b32 v40, s4, 28                                // 0000000044FC: D7610028 00013804
	v_writelane_b32 v41, s8, 0                                 // 000000004504: D7610029 00010008
	v_writelane_b32 v40, s5, 29                                // 00000000450C: D7610028 00013A05
	v_writelane_b32 v41, s9, 1                                 // 000000004514: D7610029 00010209
	v_writelane_b32 v40, s6, 30                                // 00000000451C: D7610028 00013C06
	v_writelane_b32 v41, s10, 2                                // 000000004524: D7610029 0001040A
	v_writelane_b32 v40, s7, 31                                // 00000000452C: D7610028 00013E07
	v_writelane_b32 v41, s11, 3                                // 000000004534: D7610029 0001060B
	s_load_b512 s[8:23], s[2:3], null                          // 00000000453C: F4100201 F8000000
	s_add_u32 s4, s0, 0xfffffbe0                               // 000000004544: 8004FF00 FFFFFBE0
	s_addc_u32 s5, s1, -1                                      // 00000000454C: 8205C101
	s_load_b256 s[52:59], s[4:5], null                         // 000000004550: F40C0D02 F8000000
	s_mov_b32 s2, vcc_hi                                       // 000000004558: BE82006B
	s_waitcnt lgkmcnt(0)                                       // 00000000455C: BF89FC07
	v_writelane_b32 v40, s8, 12                                // 000000004560: D7610028 00011808
	v_writelane_b32 v40, s9, 13                                // 000000004568: D7610028 00011A09
	v_writelane_b32 v40, s10, 14                               // 000000004570: D7610028 00011C0A
	v_writelane_b32 v40, s11, 15                               // 000000004578: D7610028 00011E0B
	v_writelane_b32 v40, s12, 16                               // 000000004580: D7610028 0001200C
	v_writelane_b32 v40, s13, 17                               // 000000004588: D7610028 0001220D
	v_writelane_b32 v40, s14, 18                               // 000000004590: D7610028 0001240E
	v_writelane_b32 v40, s15, 19                               // 000000004598: D7610028 0001260F
	v_writelane_b32 v40, s16, 20                               // 0000000045A0: D7610028 00012810
	v_writelane_b32 v40, s17, 21                               // 0000000045A8: D7610028 00012A11
	v_writelane_b32 v40, s18, 22                               // 0000000045B0: D7610028 00012C12
	v_writelane_b32 v40, s19, 23                               // 0000000045B8: D7610028 00012E13
	v_writelane_b32 v40, s20, 24                               // 0000000045C0: D7610028 00013014
	v_writelane_b32 v40, s21, 25                               // 0000000045C8: D7610028 00013215
	v_writelane_b32 v40, s22, 26                               // 0000000045D0: D7610028 00013416
	v_writelane_b32 v40, s23, 27                               // 0000000045D8: D7610028 00013617
	s_mov_b32 s20, 0                                           // 0000000045E0: BE940080
	s_branch 197                                               // 0000000045E4: BFA000C5 <r_3_3_3_8_8_8+0x32fc>
	s_or_saveexec_b32 s105, -1                                 // 0000000045E8: BEE922C1
	scratch_load_b32 v45, off, off offset:12                   // 0000000045EC: DC51000C 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 0000000045F4: BEFE0069
	s_waitcnt vmcnt(0)                                         // 0000000045F8: BF8903F7
	v_readlane_b32 s0, v45, 10                                 // 0000000045FC: D7600000 0001152D
	v_readlane_b32 s1, v45, 11                                 // 000000004604: D7600001 0001172D
	v_readlane_b32 s2, v45, 12                                 // 00000000460C: D7600002 0001192D
	s_mov_b32 s101, s100                                       // 000000004614: BEE50064
	s_mov_b32 s102, s100                                       // 000000004618: BEE60064
	s_add_u32 s0, s0, s28                                      // 00000000461C: 80001C00
	s_addc_u32 s1, s1, s29                                     // 000000004620: 82011D01
	s_add_u32 s2, s2, s28                                      // 000000004624: 80021C02
	s_load_b256 s[4:11], s[0:1], null                          // 000000004628: F40C0100 F8000000
	v_readlane_b32 s0, v45, 13                                 // 000000004630: D7600000 00011B2D
	v_readlane_b32 s1, v45, 17                                 // 000000004638: D7600001 0001232D
	s_mov_b32 s103, s100                                       // 000000004640: BEE70064
	s_mov_b64 s[56:57], s[100:101]                             // 000000004644: BEB80164
	s_mov_b64 s[52:53], s[100:101]                             // 000000004648: BEB40164
	s_addc_u32 s3, s0, s29                                     // 00000000464C: 82031D00
	v_readlane_b32 s0, v45, 16                                 // 000000004650: D7600000 0001212D
	s_mov_b64 s[58:59], s[102:103]                             // 000000004658: BEBA0166
	s_mov_b64 s[20:21], s[100:101]                             // 00000000465C: BE940164
	s_mov_b64 s[54:55], s[102:103]                             // 000000004660: BEB60166
	s_mov_b64 s[96:97], s[100:101]                             // 000000004664: BEE00164
	s_add_u32 s0, s0, s28                                      // 000000004668: 80001C00
	s_addc_u32 s1, s1, s29                                     // 00000000466C: 82011D01
	s_mov_b64 s[88:89], s[100:101]                             // 000000004670: BED80164
	s_mov_b64 s[84:85], s[100:101]                             // 000000004674: BED40164
	s_mov_b64 s[12:13], s[100:101]                             // 000000004678: BE8C0164
	s_mov_b64 s[16:17], s[100:101]                             // 00000000467C: BE900164
	s_mov_b64 s[92:93], s[100:101]                             // 000000004680: BEDC0164
	s_mov_b64 s[98:99], s[102:103]                             // 000000004684: BEE20166
	s_waitcnt lgkmcnt(0)                                       // 000000004688: BF89FC07
	v_writelane_b32 v41, s4, 4                                 // 00000000468C: D7610029 00010804
	s_mov_b64 s[22:23], s[102:103]                             // 000000004694: BE960166
	s_mov_b64 s[90:91], s[102:103]                             // 000000004698: BEDA0166
	s_mov_b64 s[86:87], s[102:103]                             // 00000000469C: BED60166
	s_mov_b64 s[14:15], s[102:103]                             // 0000000046A0: BE8E0166
	v_writelane_b32 v41, s5, 5                                 // 0000000046A4: D7610029 00010A05
	s_mov_b64 s[18:19], s[102:103]                             // 0000000046AC: BE920166
	s_mov_b64 s[94:95], s[102:103]                             // 0000000046B0: BEDE0166
	v_writelane_b32 v41, s6, 6                                 // 0000000046B4: D7610029 00010C06
	v_writelane_b32 v41, s7, 7                                 // 0000000046BC: D7610029 00010E07
	v_writelane_b32 v41, s8, 8                                 // 0000000046C4: D7610029 00011008
	v_writelane_b32 v41, s9, 9                                 // 0000000046CC: D7610029 00011209
	v_writelane_b32 v41, s10, 10                               // 0000000046D4: D7610029 0001140A
	v_writelane_b32 v41, s11, 11                               // 0000000046DC: D7610029 0001160B
	s_load_b256 s[4:11], s[2:3], null                          // 0000000046E4: F40C0101 F8000000
	s_mov_b32 s2, -1                                           // 0000000046EC: BE8200C1
	s_waitcnt lgkmcnt(0)                                       // 0000000046F0: BF89FC07
	v_writelane_b32 v41, s4, 20                                // 0000000046F4: D7610029 00012804
	v_writelane_b32 v41, s5, 21                                // 0000000046FC: D7610029 00012A05
	v_writelane_b32 v41, s6, 22                                // 000000004704: D7610029 00012C06
	v_writelane_b32 v41, s7, 23                                // 00000000470C: D7610029 00012E07
	v_writelane_b32 v41, s8, 24                                // 000000004714: D7610029 00013008
	v_writelane_b32 v41, s9, 25                                // 00000000471C: D7610029 00013209
	v_writelane_b32 v41, s10, 26                               // 000000004724: D7610029 0001340A
	v_writelane_b32 v41, s11, 27                               // 00000000472C: D7610029 0001360B
	s_load_b256 s[4:11], s[0:1], 0x1100                        // 000000004734: F40C0100 F8001100
	s_waitcnt lgkmcnt(0)                                       // 00000000473C: BF89FC07
	v_writelane_b32 v40, s4, 28                                // 000000004740: D7610028 00013804
	v_writelane_b32 v41, s8, 0                                 // 000000004748: D7610029 00010008
	v_writelane_b32 v40, s5, 29                                // 000000004750: D7610028 00013A05
	v_writelane_b32 v41, s9, 1                                 // 000000004758: D7610029 00010209
	v_writelane_b32 v40, s6, 30                                // 000000004760: D7610028 00013C06
	v_writelane_b32 v41, s10, 2                                // 000000004768: D7610029 0001040A
	v_writelane_b32 v40, s7, 31                                // 000000004770: D7610028 00013E07
	v_writelane_b32 v41, s11, 3                                // 000000004778: D7610029 0001060B
	s_mov_b64 s[8:9], s[100:101]                               // 000000004780: BE880164
	s_mov_b64 s[4:5], s[100:101]                               // 000000004784: BE840164
	s_mov_b64 s[10:11], s[102:103]                             // 000000004788: BE8A0166
	s_mov_b64 s[6:7], s[102:103]                               // 00000000478C: BE860166
	v_writelane_b32 v41, s4, 12                                // 000000004790: D7610029 00011804
	v_writelane_b32 v41, s5, 13                                // 000000004798: D7610029 00011A05
	v_writelane_b32 v41, s6, 14                                // 0000000047A0: D7610029 00011C06
	v_writelane_b32 v41, s7, 15                                // 0000000047A8: D7610029 00011E07
	v_writelane_b32 v41, s8, 16                                // 0000000047B0: D7610029 00012008
	v_writelane_b32 v41, s9, 17                                // 0000000047B8: D7610029 00012209
	v_writelane_b32 v41, s10, 18                               // 0000000047C0: D7610029 0001240A
	v_writelane_b32 v41, s11, 19                               // 0000000047C8: D7610029 0001260B
	s_or_saveexec_b32 s105, -1                                 // 0000000047D0: BEE922C1
	scratch_load_b32 v45, off, off offset:36                   // 0000000047D4: DC510024 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 0000000047DC: BEFE0069
	s_waitcnt vmcnt(0)                                         // 0000000047E0: BF8903F7
	v_writelane_b32 v45, s84, 28                               // 0000000047E4: D761002D 00013854
	v_writelane_b32 v45, s85, 29                               // 0000000047EC: D761002D 00013A55
	v_writelane_b32 v45, s86, 30                               // 0000000047F4: D761002D 00013C56
	v_writelane_b32 v45, s87, 31                               // 0000000047FC: D761002D 00013E57
	s_or_saveexec_b32 s105, -1                                 // 000000004804: BEE922C1
	scratch_store_b32 off, v45, off offset:36                  // 000000004808: DC690024 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000004810: BEFE0069
	v_writelane_b32 v40, s88, 0                                // 000000004814: D7610028 00010058
	s_mov_b64 s[24:25], s[100:101]                             // 00000000481C: BE980164
	s_mov_b64 s[26:27], s[102:103]                             // 000000004820: BE9A0166
	v_writelane_b32 v40, s89, 1                                // 000000004824: D7610028 00010259
	v_writelane_b32 v40, s90, 2                                // 00000000482C: D7610028 0001045A
	v_writelane_b32 v40, s91, 3                                // 000000004834: D7610028 0001065B
	v_writelane_b32 v40, s92, 4                                // 00000000483C: D7610028 0001085C
	v_writelane_b32 v40, s93, 5                                // 000000004844: D7610028 00010A5D
	v_writelane_b32 v40, s94, 6                                // 00000000484C: D7610028 00010C5E
	v_writelane_b32 v40, s95, 7                                // 000000004854: D7610028 00010E5F
	v_writelane_b32 v40, s96, 8                                // 00000000485C: D7610028 00011060
	v_writelane_b32 v40, s97, 9                                // 000000004864: D7610028 00011261
	v_writelane_b32 v40, s98, 10                               // 00000000486C: D7610028 00011462
	v_writelane_b32 v40, s99, 11                               // 000000004874: D7610028 00011663
	v_writelane_b32 v40, s12, 12                               // 00000000487C: D7610028 0001180C
	v_writelane_b32 v40, s13, 13                               // 000000004884: D7610028 00011A0D
	v_writelane_b32 v40, s14, 14                               // 00000000488C: D7610028 00011C0E
	v_writelane_b32 v40, s15, 15                               // 000000004894: D7610028 00011E0F
	v_writelane_b32 v40, s16, 16                               // 00000000489C: D7610028 00012010
	v_writelane_b32 v40, s17, 17                               // 0000000048A4: D7610028 00012211
	v_writelane_b32 v40, s18, 18                               // 0000000048AC: D7610028 00012412
	v_writelane_b32 v40, s19, 19                               // 0000000048B4: D7610028 00012613
	v_writelane_b32 v40, s20, 20                               // 0000000048BC: D7610028 00012814
	v_writelane_b32 v40, s21, 21                               // 0000000048C4: D7610028 00012A15
	v_writelane_b32 v40, s22, 22                               // 0000000048CC: D7610028 00012C16
	v_writelane_b32 v40, s23, 23                               // 0000000048D4: D7610028 00012E17
	v_writelane_b32 v40, s24, 24                               // 0000000048DC: D7610028 00013018
	v_writelane_b32 v40, s25, 25                               // 0000000048E4: D7610028 00013219
	v_writelane_b32 v40, s26, 26                               // 0000000048EC: D7610028 0001341A
	v_writelane_b32 v40, s27, 27                               // 0000000048F4: D7610028 0001361B
	v_writelane_b32 v41, s52, 28                               // 0000000048FC: D7610029 00013834
	v_writelane_b32 v45, s56, 0                                // 000000004904: D761002D 00010038
	v_writelane_b32 v41, s53, 29                               // 00000000490C: D7610029 00013A35
	v_writelane_b32 v45, s57, 1                                // 000000004914: D761002D 00010239
	v_writelane_b32 v41, s54, 30                               // 00000000491C: D7610029 00013C36
	v_writelane_b32 v45, s58, 2                                // 000000004924: D761002D 0001043A
	v_writelane_b32 v41, s55, 31                               // 00000000492C: D7610029 00013E37
	v_writelane_b32 v45, s59, 3                                // 000000004934: D761002D 0001063B
	s_or_saveexec_b32 s105, -1                                 // 00000000493C: BEE922C1
	scratch_store_b32 off, v39, off offset:56                  // 000000004940: DC690038 007C2700
	s_mov_b32 exec_lo, s105                                    // 000000004948: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 00000000494C: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004950: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000004954: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000004958: BEE922C1
	scratch_store_b32 off, v38, off offset:60                  // 00000000495C: DC69003C 007C2600
	s_mov_b32 exec_lo, s105                                    // 000000004964: BEFE0069
	s_mov_b32 s52, s20                                         // 000000004968: BEB40014
	s_mov_b32 s60, s20                                         // 00000000496C: BEBC0014
	s_mov_b32 s61, s20                                         // 000000004970: BEBD0014
	s_mov_b32 s62, s20                                         // 000000004974: BEBE0014
	s_mov_b32 s63, s20                                         // 000000004978: BEBF0014
	s_mov_b32 s53, s20                                         // 00000000497C: BEB50014
	s_mov_b32 s54, s20                                         // 000000004980: BEB60014
	s_mov_b32 s55, s20                                         // 000000004984: BEB70014
	s_mov_b32 s56, s20                                         // 000000004988: BEB80014
	s_mov_b32 s57, s20                                         // 00000000498C: BEB90014
	s_mov_b32 s58, s20                                         // 000000004990: BEBA0014
	s_mov_b32 s59, s20                                         // 000000004994: BEBB0014
	s_mov_b32 s64, s20                                         // 000000004998: BEC00014
	s_mov_b32 s65, s20                                         // 00000000499C: BEC10014
	s_mov_b32 s66, s20                                         // 0000000049A0: BEC20014
	s_mov_b32 s67, s20                                         // 0000000049A4: BEC30014
	v_writelane_b32 v45, s52, 4                                // 0000000049A8: D761002D 00010834
	s_mov_b32 s84, s20                                         // 0000000049B0: BED40014
	s_mov_b32 s92, s20                                         // 0000000049B4: BEDC0014
	s_mov_b32 s93, s20                                         // 0000000049B8: BEDD0014
	s_mov_b32 s94, s20                                         // 0000000049BC: BEDE0014
	v_writelane_b32 v45, s53, 5                                // 0000000049C0: D761002D 00010A35
	s_mov_b32 s95, s20                                         // 0000000049C8: BEDF0014
	s_mov_b32 s85, s20                                         // 0000000049CC: BED50014
	s_mov_b32 s86, s20                                         // 0000000049D0: BED60014
	s_mov_b32 s87, s20                                         // 0000000049D4: BED70014
	v_writelane_b32 v45, s54, 6                                // 0000000049D8: D761002D 00010C36
	s_mov_b32 s88, s20                                         // 0000000049E0: BED80014
	s_mov_b32 s89, s20                                         // 0000000049E4: BED90014
	s_mov_b32 s90, s20                                         // 0000000049E8: BEDA0014
	s_mov_b32 s91, s20                                         // 0000000049EC: BEDB0014
	v_writelane_b32 v45, s55, 7                                // 0000000049F0: D761002D 00010E37
	s_mov_b32 s96, s20                                         // 0000000049F8: BEE00014
	s_mov_b32 s97, s20                                         // 0000000049FC: BEE10014
	s_mov_b32 s98, s20                                         // 000000004A00: BEE20014
	s_mov_b32 s99, s20                                         // 000000004A04: BEE30014
	v_writelane_b32 v45, s56, 8                                // 000000004A08: D761002D 00011038
	s_mov_b32 s21, s20                                         // 000000004A10: BE950014
	s_mov_b32 s22, s20                                         // 000000004A14: BE960014
	s_mov_b32 s23, s20                                         // 000000004A18: BE970014
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 000000004A1C: 916A027E
	v_writelane_b32 v45, s57, 9                                // 000000004A20: D761002D 00011239
	s_mov_b32 s16, s20                                         // 000000004A28: BE900014
	s_mov_b32 s17, s20                                         // 000000004A2C: BE910014
	s_mov_b32 s18, s20                                         // 000000004A30: BE920014
	s_mov_b32 s19, s20                                         // 000000004A34: BE930014
	v_writelane_b32 v45, s58, 10                               // 000000004A38: D761002D 0001143A
	v_writelane_b32 v45, s59, 11                               // 000000004A40: D761002D 0001163B
	v_writelane_b32 v45, s60, 12                               // 000000004A48: D761002D 0001183C
	v_writelane_b32 v45, s61, 13                               // 000000004A50: D761002D 00011A3D
	v_writelane_b32 v45, s62, 14                               // 000000004A58: D761002D 00011C3E
	v_writelane_b32 v45, s63, 15                               // 000000004A60: D761002D 00011E3F
	v_writelane_b32 v45, s64, 16                               // 000000004A68: D761002D 00012040
	v_writelane_b32 v45, s65, 17                               // 000000004A70: D761002D 00012241
	v_writelane_b32 v45, s66, 18                               // 000000004A78: D761002D 00012442
	v_writelane_b32 v45, s67, 19                               // 000000004A80: D761002D 00012643
	v_writelane_b32 v45, s84, 20                               // 000000004A88: D761002D 00012854
	v_writelane_b32 v39, s96, 0                                // 000000004A90: D7610027 00010060
	v_writelane_b32 v45, s85, 21                               // 000000004A98: D761002D 00012A55
	v_writelane_b32 v39, s97, 1                                // 000000004AA0: D7610027 00010261
	v_writelane_b32 v45, s86, 22                               // 000000004AA8: D761002D 00012C56
	v_writelane_b32 v39, s98, 2                                // 000000004AB0: D7610027 00010462
	v_writelane_b32 v45, s87, 23                               // 000000004AB8: D761002D 00012E57
	v_writelane_b32 v39, s99, 3                                // 000000004AC0: D7610027 00010663
	v_writelane_b32 v45, s88, 24                               // 000000004AC8: D761002D 00013058
	v_writelane_b32 v45, s89, 25                               // 000000004AD0: D761002D 00013259
	v_writelane_b32 v45, s90, 26                               // 000000004AD8: D761002D 0001345A
	v_writelane_b32 v45, s91, 27                               // 000000004AE0: D761002D 0001365B
	v_writelane_b32 v45, s92, 28                               // 000000004AE8: D761002D 0001385C
	v_writelane_b32 v45, s93, 29                               // 000000004AF0: D761002D 00013A5D
	v_writelane_b32 v45, s94, 30                               // 000000004AF8: D761002D 00013C5E
	v_writelane_b32 v45, s95, 31                               // 000000004B00: D761002D 00013E5F
	s_cbranch_vccnz 73                                         // 000000004B08: BFA40049 <r_3_3_3_8_8_8+0x3630>
	s_load_b512 s[4:19], s[0:1], 0x2500                        // 000000004B0C: F4100100 F8002500
	s_waitcnt lgkmcnt(0)                                       // 000000004B14: BF89FC07
	v_writelane_b32 v45, s4, 4                                 // 000000004B18: D761002D 00010804
	v_writelane_b32 v45, s5, 5                                 // 000000004B20: D761002D 00010A05
	v_writelane_b32 v45, s6, 6                                 // 000000004B28: D761002D 00010C06
	v_writelane_b32 v45, s7, 7                                 // 000000004B30: D761002D 00010E07
	v_writelane_b32 v45, s8, 8                                 // 000000004B38: D761002D 00011008
	v_writelane_b32 v45, s9, 9                                 // 000000004B40: D761002D 00011209
	v_writelane_b32 v45, s10, 10                               // 000000004B48: D761002D 0001140A
	v_writelane_b32 v45, s11, 11                               // 000000004B50: D761002D 0001160B
	v_writelane_b32 v45, s12, 12                               // 000000004B58: D761002D 0001180C
	v_writelane_b32 v45, s13, 13                               // 000000004B60: D761002D 00011A0D
	v_writelane_b32 v45, s14, 14                               // 000000004B68: D761002D 00011C0E
	v_writelane_b32 v45, s15, 15                               // 000000004B70: D761002D 00011E0F
	v_writelane_b32 v45, s16, 16                               // 000000004B78: D761002D 00012010
	v_writelane_b32 v45, s17, 17                               // 000000004B80: D761002D 00012211
	v_writelane_b32 v45, s18, 18                               // 000000004B88: D761002D 00012412
	v_writelane_b32 v45, s19, 19                               // 000000004B90: D761002D 00012613
	s_clause 0x1                                               // 000000004B98: BF850001
	s_load_b256 s[16:23], s[0:1], 0x23e0                       // 000000004B9C: F40C0400 F80023E0
	s_load_b512 s[84:99], s[0:1], 0x2640                       // 000000004BA4: F4101500 F8002640
	s_waitcnt lgkmcnt(0)                                       // 000000004BAC: BF89FC07
	v_writelane_b32 v45, s84, 20                               // 000000004BB0: D761002D 00012854
	v_writelane_b32 v39, s96, 0                                // 000000004BB8: D7610027 00010060
	v_writelane_b32 v45, s85, 21                               // 000000004BC0: D761002D 00012A55
	v_writelane_b32 v39, s97, 1                                // 000000004BC8: D7610027 00010261
	v_writelane_b32 v45, s86, 22                               // 000000004BD0: D761002D 00012C56
	v_writelane_b32 v39, s98, 2                                // 000000004BD8: D7610027 00010462
	v_writelane_b32 v45, s87, 23                               // 000000004BE0: D761002D 00012E57
	v_writelane_b32 v39, s99, 3                                // 000000004BE8: D7610027 00010663
	v_writelane_b32 v45, s88, 24                               // 000000004BF0: D761002D 00013058
	v_writelane_b32 v45, s89, 25                               // 000000004BF8: D761002D 00013259
	v_writelane_b32 v45, s90, 26                               // 000000004C00: D761002D 0001345A
	v_writelane_b32 v45, s91, 27                               // 000000004C08: D761002D 0001365B
	v_writelane_b32 v45, s92, 28                               // 000000004C10: D761002D 0001385C
	v_writelane_b32 v45, s93, 29                               // 000000004C18: D761002D 00013A5D
	v_writelane_b32 v45, s94, 30                               // 000000004C20: D761002D 00013C5E
	v_writelane_b32 v45, s95, 31                               // 000000004C28: D761002D 00013E5F
	s_load_b256 s[52:59], s[0:1], 0x1120                       // 000000004C30: F40C0D00 F8001120
	s_waitcnt lgkmcnt(0)                                       // 000000004C38: BF89FC07
	v_writelane_b32 v39, s52, 4                                // 000000004C3C: D7610027 00010834
	v_writelane_b32 v39, s53, 5                                // 000000004C44: D7610027 00010A35
	v_writelane_b32 v39, s54, 6                                // 000000004C4C: D7610027 00010C36
	v_writelane_b32 v39, s55, 7                                // 000000004C54: D7610027 00010E37
	v_writelane_b32 v39, s56, 8                                // 000000004C5C: D7610027 00011038
	v_writelane_b32 v39, s57, 9                                // 000000004C64: D7610027 00011239
	v_writelane_b32 v39, s58, 10                               // 000000004C6C: D7610027 0001143A
	v_writelane_b32 v39, s59, 11                               // 000000004C74: D7610027 0001163B
	s_clause 0x1                                               // 000000004C7C: BF850001
	s_load_b256 s[52:59], s[0:1], 0xfe0                        // 000000004C80: F40C0D00 F8000FE0
	s_load_b512 s[0:15], s[0:1], 0x1240                        // 000000004C88: F4100000 F8001240
	s_waitcnt lgkmcnt(0)                                       // 000000004C90: BF89FC07
	v_writelane_b32 v39, s52, 12                               // 000000004C94: D7610027 00011834
	v_writelane_b32 v39, s53, 13                               // 000000004C9C: D7610027 00011A35
	v_writelane_b32 v39, s54, 14                               // 000000004CA4: D7610027 00011C36
	v_writelane_b32 v39, s55, 15                               // 000000004CAC: D7610027 00011E37
	v_writelane_b32 v39, s56, 16                               // 000000004CB4: D7610027 00012038
	v_writelane_b32 v39, s57, 17                               // 000000004CBC: D7610027 00012239
	v_writelane_b32 v39, s58, 18                               // 000000004CC4: D7610027 0001243A
	v_writelane_b32 v39, s59, 19                               // 000000004CCC: D7610027 0001263B
	v_writelane_b32 v39, s0, 20                                // 000000004CD4: D7610027 00012800
	v_writelane_b32 v39, s1, 21                                // 000000004CDC: D7610027 00012A01
	v_writelane_b32 v39, s2, 22                                // 000000004CE4: D7610027 00012C02
	v_writelane_b32 v39, s3, 23                                // 000000004CEC: D7610027 00012E03
	v_writelane_b32 v39, s4, 24                                // 000000004CF4: D7610027 00013004
	v_writelane_b32 v39, s5, 25                                // 000000004CFC: D7610027 00013205
	v_writelane_b32 v39, s6, 26                                // 000000004D04: D7610027 00013406
	v_writelane_b32 v39, s7, 27                                // 000000004D0C: D7610027 00013607
	v_writelane_b32 v39, s8, 28                                // 000000004D14: D7610027 00013808
	v_writelane_b32 v39, s9, 29                                // 000000004D1C: D7610027 00013A09
	v_writelane_b32 v39, s10, 30                               // 000000004D24: D7610027 00013C0A
	v_writelane_b32 v39, s11, 31                               // 000000004D2C: D7610027 00013E0B
	s_or_saveexec_b32 s105, -1                                 // 000000004D34: BEE922C1
	scratch_store_b32 off, v39, off offset:100                 // 000000004D38: DC690064 007C2700
	s_mov_b32 exec_lo, s105                                    // 000000004D40: BEFE0069
	v_writelane_b32 v38, s12, 0                                // 000000004D44: D7610026 0001000C
	s_and_b32 vcc_lo, exec_lo, s34                             // 000000004D4C: 8B6A227E
	v_writelane_b32 v38, s13, 1                                // 000000004D50: D7610026 0001020D
	v_writelane_b32 v38, s14, 2                                // 000000004D58: D7610026 0001040E
	v_writelane_b32 v38, s15, 3                                // 000000004D60: D7610026 0001060F
	s_or_saveexec_b32 s105, -1                                 // 000000004D68: BEE922C1
	scratch_load_b32 v39, off, off offset:40                   // 000000004D6C: DC510028 277C0000
	s_mov_b32 exec_lo, s105                                    // 000000004D74: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000004D78: BF8903F7
	v_readlane_b32 s0, v39, 28                                 // 000000004D7C: D7600000 00013927
	v_readlane_b32 s1, v39, 29                                 // 000000004D84: D7600001 00013B27
	v_readlane_b32 s2, v39, 30                                 // 000000004D8C: D7600002 00013D27
	v_readlane_b32 s3, v39, 31                                 // 000000004D94: D7600003 00013F27
	v_readlane_b32 s4, v44, 0                                  // 000000004D9C: D7600004 0001012C
	v_readlane_b32 s5, v44, 1                                  // 000000004DA4: D7600005 0001032C
	v_readlane_b32 s6, v44, 2                                  // 000000004DAC: D7600006 0001052C
	s_or_saveexec_b32 s105, -1                                 // 000000004DB4: BEE922C1
	scratch_store_b32 off, v44, off offset:52                  // 000000004DB8: DC690034 007C2C00
	s_mov_b32 exec_lo, s105                                    // 000000004DC0: BEFE0069
	v_readlane_b32 s7, v44, 3                                  // 000000004DC4: D7600007 0001072C
	v_readlane_b32 s0, v41, 28                                 // 000000004DCC: D7600000 00013929
	v_readlane_b32 s1, v41, 29                                 // 000000004DD4: D7600001 00013B29
	v_readlane_b32 s2, v41, 30                                 // 000000004DDC: D7600002 00013D29
	v_readlane_b32 s3, v41, 31                                 // 000000004DE4: D7600003 00013F29
	v_readlane_b32 s4, v45, 0                                  // 000000004DEC: D7600004 0001012D
	v_readlane_b32 s5, v45, 1                                  // 000000004DF4: D7600005 0001032D
	v_readlane_b32 s6, v45, 2                                  // 000000004DFC: D7600006 0001052D
	s_or_saveexec_b32 s105, -1                                 // 000000004E04: BEE922C1
	scratch_store_b32 off, v45, off offset:96                  // 000000004E08: DC690060 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000004E10: BEFE0069
	v_writelane_b32 v38, s68, 4                                // 000000004E14: D7610026 00010844
	v_readlane_b32 s7, v45, 3                                  // 000000004E1C: D7600007 0001072D
	v_writelane_b32 v38, s69, 5                                // 000000004E24: D7610026 00010A45
	v_writelane_b32 v38, s70, 6                                // 000000004E2C: D7610026 00010C46
	v_writelane_b32 v38, s71, 7                                // 000000004E34: D7610026 00010E47
	v_writelane_b32 v38, s72, 8                                // 000000004E3C: D7610026 00011048
	v_writelane_b32 v38, s73, 9                                // 000000004E44: D7610026 00011249
	v_writelane_b32 v38, s74, 10                               // 000000004E4C: D7610026 0001144A
	v_writelane_b32 v38, s75, 11                               // 000000004E54: D7610026 0001164B
	v_writelane_b32 v38, s76, 12                               // 000000004E5C: D7610026 0001184C
	v_writelane_b32 v38, s77, 13                               // 000000004E64: D7610026 00011A4D
	v_writelane_b32 v38, s78, 14                               // 000000004E6C: D7610026 00011C4E
	v_writelane_b32 v38, s79, 15                               // 000000004E74: D7610026 00011E4F
	v_writelane_b32 v38, s80, 16                               // 000000004E7C: D7610026 00012050
	v_writelane_b32 v38, s81, 17                               // 000000004E84: D7610026 00012251
	v_writelane_b32 v38, s82, 18                               // 000000004E8C: D7610026 00012452
	v_writelane_b32 v38, s83, 19                               // 000000004E94: D7610026 00012653
	s_cbranch_vccnz 829                                        // 000000004E9C: BFA4033D <r_3_3_3_8_8_8+0x4594>
	s_or_saveexec_b32 s105, -1                                 // 000000004EA0: BEE922C1
	scratch_load_b32 v45, off, off                             // 000000004EA4: DC510000 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000004EAC: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000004EB0: BF8903F7
	v_readlane_b32 s0, v45, 0                                  // 000000004EB4: D7600000 0001012D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000004EBC: BF8700A1
	s_add_u32 s2, s0, s28                                      // 000000004EC0: 80021C00
	v_readlane_b32 s0, v45, 1                                  // 000000004EC4: D7600000 0001032D
	s_addc_u32 s3, s0, s29                                     // 000000004ECC: 82031D00
	s_or_saveexec_b32 s105, -1                                 // 000000004ED0: BEE922C1
	scratch_load_b32 v45, off, off offset:12                   // 000000004ED4: DC51000C 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000004EDC: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000004EE0: BF8903F7
	v_readlane_b32 s0, v45, 18                                 // 000000004EE4: D7600000 0001252D
	v_readlane_b32 s1, v45, 19                                 // 000000004EEC: D7600001 0001272D
	v_readlane_b32 s8, v42, 12                                 // 000000004EF4: D7600008 0001192A
	s_mov_b32 s101, s100                                       // 000000004EFC: BEE50064
	v_readlane_b32 s9, v42, 13                                 // 000000004F00: D7600009 00011B2A
	v_readlane_b32 s10, v42, 14                                // 000000004F08: D760000A 00011D2A
	v_readlane_b32 s11, v42, 15                                // 000000004F10: D760000B 00011F2A
	v_readlane_b32 s12, v42, 16                                // 000000004F18: D760000C 0001212A
	v_readlane_b32 s13, v42, 17                                // 000000004F20: D760000D 0001232A
	v_readlane_b32 s14, v42, 18                                // 000000004F28: D760000E 0001252A
	v_readlane_b32 s15, v42, 19                                // 000000004F30: D760000F 0001272A
	s_mov_b32 s102, s100                                       // 000000004F38: BEE60064
	s_mov_b32 s103, s100                                       // 000000004F3C: BEE70064
	s_add_u32 s0, s0, s28                                      // 000000004F40: 80001C00
	s_mov_b64 s[56:57], s[100:101]                             // 000000004F44: BEB80164
	s_mov_b64 s[52:53], s[100:101]                             // 000000004F48: BEB40164
	s_addc_u32 s1, s1, s29                                     // 000000004F4C: 82011D01
	s_and_b32 vcc_lo, exec_lo, s33                             // 000000004F50: 8B6A217E
	s_mov_b64 s[58:59], s[102:103]                             // 000000004F54: BEBA0166
	s_mov_b64 s[54:55], s[102:103]                             // 000000004F58: BEB60166
	v_readlane_b32 s92, v37, 20                                // 000000004F5C: D760005C 00012925
	v_readlane_b32 s93, v37, 21                                // 000000004F64: D760005D 00012B25
	v_readlane_b32 s94, v37, 22                                // 000000004F6C: D760005E 00012D25
	v_readlane_b32 s95, v37, 23                                // 000000004F74: D760005F 00012F25
	v_readlane_b32 s96, v37, 24                                // 000000004F7C: D7600060 00013125
	v_readlane_b32 s97, v37, 25                                // 000000004F84: D7600061 00013325
	v_readlane_b32 s98, v37, 26                                // 000000004F8C: D7600062 00013525
	v_readlane_b32 s99, v37, 27                                // 000000004F94: D7600063 00013725
	s_cbranch_vccnz 14                                         // 000000004F9C: BFA4000E <r_3_3_3_8_8_8+0x39d8>
	s_or_saveexec_b32 s105, -1                                 // 000000004FA0: BEE922C1
	scratch_load_b32 v45, off, off                             // 000000004FA4: DC510000 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000004FAC: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000004FB0: BF8903F7
	v_readlane_b32 s4, v45, 4                                  // 000000004FB4: D7600004 0001092D
	v_readlane_b32 s5, v45, 5                                  // 000000004FBC: D7600005 00010B2D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004FC4: BF870092
	s_add_u32 s4, s4, s28                                      // 000000004FC8: 80041C04
	s_addc_u32 s5, s5, s29                                     // 000000004FCC: 82051D05
	s_load_b256 s[52:59], s[4:5], null                         // 000000004FD0: F40C0D02 F8000000
	s_load_b256 s[68:75], s[2:3], null                         // 000000004FD8: F40C1101 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000004FE0: BF89FC07
	v_writelane_b32 v44, s68, 4                                // 000000004FE4: D761002C 00010844
	v_writelane_b32 v44, s69, 5                                // 000000004FEC: D761002C 00010A45
	v_writelane_b32 v44, s70, 6                                // 000000004FF4: D761002C 00010C46
	v_writelane_b32 v44, s71, 7                                // 000000004FFC: D761002C 00010E47
	v_writelane_b32 v44, s72, 8                                // 000000005004: D761002C 00011048
	v_writelane_b32 v44, s73, 9                                // 00000000500C: D761002C 00011249
	v_writelane_b32 v44, s74, 10                               // 000000005014: D761002C 0001144A
	v_writelane_b32 v44, s75, 11                               // 00000000501C: D761002C 0001164B
	s_load_b256 s[68:75], s[0:1], null                         // 000000005024: F40C1100 F8000000
	s_or_saveexec_b32 s105, -1                                 // 00000000502C: BEE922C1
	scratch_load_b32 v45, off, off offset:12                   // 000000005030: DC51000C 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000005038: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000503C: BF8903F7
	v_readlane_b32 s0, v45, 20                                 // 000000005040: D7600000 0001292D
	v_readlane_b32 s1, v45, 23                                 // 000000005048: D7600001 00012F2D
	s_mov_b32 s24, 0                                           // 000000005050: BE980080
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000005054: BF8700A2
	s_add_u32 s2, s0, s28                                      // 000000005058: 80021C00
	v_readlane_b32 s0, v45, 21                                 // 00000000505C: D7600000 00012B2D
	s_addc_u32 s3, s0, s29                                     // 000000005064: 82031D00
	v_readlane_b32 s0, v45, 22                                 // 000000005068: D7600000 00012D2D
	s_load_b512 s[76:91], s[2:3], null                         // 000000005070: F4101301 F8000000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005078: BF870001
	s_add_u32 s0, s0, s28                                      // 00000000507C: 80001C00
	s_addc_u32 s1, s1, s29                                     // 000000005080: 82011D01
	s_add_u32 s2, s0, 0xfffffe40                               // 000000005084: 8002FF00 FFFFFE40
	s_addc_u32 s3, s1, -1                                      // 00000000508C: 8203C101
	s_add_u32 s4, s0, 0xfffffbe0                               // 000000005090: 8004FF00 FFFFFBE0
	s_addc_u32 s5, s1, -1                                      // 000000005098: 8205C101
	s_waitcnt lgkmcnt(0)                                       // 00000000509C: BF89FC07
	v_writelane_b32 v38, s76, 20                               // 0000000050A0: D7610026 0001284C
	v_writelane_b32 v39, s88, 0                                // 0000000050A8: D7610027 00010058
	v_writelane_b32 v38, s77, 21                               // 0000000050B0: D7610026 00012A4D
	v_writelane_b32 v39, s89, 1                                // 0000000050B8: D7610027 00010259
	v_writelane_b32 v38, s78, 22                               // 0000000050C0: D7610026 00012C4E
	v_writelane_b32 v39, s90, 2                                // 0000000050C8: D7610027 0001045A
	v_writelane_b32 v38, s79, 23                               // 0000000050D0: D7610026 00012E4F
	v_writelane_b32 v39, s91, 3                                // 0000000050D8: D7610027 0001065B
	v_writelane_b32 v38, s80, 24                               // 0000000050E0: D7610026 00013050
	v_writelane_b32 v38, s81, 25                               // 0000000050E8: D7610026 00013251
	v_writelane_b32 v38, s82, 26                               // 0000000050F0: D7610026 00013452
	v_writelane_b32 v38, s83, 27                               // 0000000050F8: D7610026 00013653
	s_load_b256 s[76:83], s[0:1], 0x1100                       // 000000005100: F40C1300 F8001100
	v_writelane_b32 v38, s84, 28                               // 000000005108: D7610026 00013854
	v_writelane_b32 v38, s85, 29                               // 000000005110: D7610026 00013A55
	v_writelane_b32 v38, s86, 30                               // 000000005118: D7610026 00013C56
	v_writelane_b32 v38, s87, 31                               // 000000005120: D7610026 00013E57
	s_waitcnt lgkmcnt(0)                                       // 000000005128: BF89FC07
	v_writelane_b32 v39, s76, 28                               // 00000000512C: D7610027 0001384C
	v_writelane_b32 v44, s80, 0                                // 000000005134: D761002C 00010050
	v_writelane_b32 v39, s77, 29                               // 00000000513C: D7610027 00013A4D
	v_writelane_b32 v44, s81, 1                                // 000000005144: D761002C 00010251
	v_writelane_b32 v39, s78, 30                               // 00000000514C: D7610027 00013C4E
	v_writelane_b32 v44, s82, 2                                // 000000005154: D761002C 00010452
	v_writelane_b32 v39, s79, 31                               // 00000000515C: D7610027 00013E4F
	v_writelane_b32 v44, s83, 3                                // 000000005164: D761002C 00010653
	s_load_b512 s[76:91], s[2:3], null                         // 00000000516C: F4101301 F8000000
	s_mov_b32 s3, 0                                            // 000000005174: BE830080
	s_mov_b32 s2, vcc_hi                                       // 000000005178: BE82006B
	s_waitcnt lgkmcnt(0)                                       // 00000000517C: BF89FC07
	v_writelane_b32 v39, s76, 12                               // 000000005180: D7610027 0001184C
	v_writelane_b32 v39, s77, 13                               // 000000005188: D7610027 00011A4D
	v_writelane_b32 v39, s78, 14                               // 000000005190: D7610027 00011C4E
	v_writelane_b32 v39, s79, 15                               // 000000005198: D7610027 00011E4F
	v_writelane_b32 v39, s80, 16                               // 0000000051A0: D7610027 00012050
	v_writelane_b32 v39, s81, 17                               // 0000000051A8: D7610027 00012251
	v_writelane_b32 v39, s82, 18                               // 0000000051B0: D7610027 00012452
	v_writelane_b32 v39, s83, 19                               // 0000000051B8: D7610027 00012653
	v_writelane_b32 v39, s84, 20                               // 0000000051C0: D7610027 00012854
	v_writelane_b32 v39, s85, 21                               // 0000000051C8: D7610027 00012A55
	v_writelane_b32 v39, s86, 22                               // 0000000051D0: D7610027 00012C56
	v_writelane_b32 v39, s87, 23                               // 0000000051D8: D7610027 00012E57
	v_writelane_b32 v39, s88, 24                               // 0000000051E0: D7610027 00013058
	v_writelane_b32 v39, s89, 25                               // 0000000051E8: D7610027 00013259
	v_writelane_b32 v39, s90, 26                               // 0000000051F0: D7610027 0001345A
	v_writelane_b32 v39, s91, 27                               // 0000000051F8: D7610027 0001365B
	s_load_b256 s[76:83], s[4:5], null                         // 000000005200: F40C1302 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000005208: BF89FC07
	v_writelane_b32 v39, s76, 4                                // 00000000520C: D7610027 0001084C
	v_writelane_b32 v39, s77, 5                                // 000000005214: D7610027 00010A4D
	v_writelane_b32 v39, s78, 6                                // 00000000521C: D7610027 00010C4E
	v_writelane_b32 v39, s79, 7                                // 000000005224: D7610027 00010E4F
	v_writelane_b32 v39, s80, 8                                // 00000000522C: D7610027 00011050
	v_writelane_b32 v39, s81, 9                                // 000000005234: D7610027 00011251
	v_writelane_b32 v39, s82, 10                               // 00000000523C: D7610027 00011452
	v_writelane_b32 v39, s83, 11                               // 000000005244: D7610027 00011653
	s_branch 739                                               // 00000000524C: BFA002E3 <r_3_3_3_8_8_8+0x47dc>
	s_mov_b32 s0, s31                                          // 000000005250: BE80001F
	s_or_saveexec_b32 s105, -1                                 // 000000005254: BEE922C1
	scratch_load_b32 v45, off, off offset:8                    // 000000005258: DC510008 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000005260: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000005264: BF8903F7
	v_writelane_b32 v45, s8, 12                                // 000000005268: D761002D 00011808
	v_writelane_b32 v45, s9, 13                                // 000000005270: D761002D 00011A09
	v_writelane_b32 v45, s10, 14                               // 000000005278: D761002D 00011C0A
	v_writelane_b32 v45, s11, 15                               // 000000005280: D761002D 00011E0B
	v_writelane_b32 v45, s12, 16                               // 000000005288: D761002D 0001200C
	v_writelane_b32 v45, s13, 17                               // 000000005290: D761002D 0001220D
	v_writelane_b32 v45, s14, 18                               // 000000005298: D761002D 0001240E
	v_writelane_b32 v45, s15, 19                               // 0000000052A0: D761002D 0001260F
	s_or_saveexec_b32 s105, -1                                 // 0000000052A8: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000052AC: BF870009
	s_mov_b32 exec_lo, s105                                    // 0000000052B0: BEFE0069
	s_branch 62051                                             // 0000000052B4: BFA0F263 <r_3_3_3_8_8_8+0x644>
	s_mov_b32 s7, -1                                           // 0000000052B8: BE8700C1
	v_writelane_b32 v41, s36, 20                               // 0000000052BC: D7610029 00012824
	v_writelane_b32 v43, s48, 0                                // 0000000052C4: D761002B 00010030
	s_mov_b32 s6, 0                                            // 0000000052CC: BE860080
	v_writelane_b32 v41, s37, 21                               // 0000000052D0: D7610029 00012A25
	v_writelane_b32 v43, s49, 1                                // 0000000052D8: D761002B 00010231
	v_writelane_b32 v41, s38, 22                               // 0000000052E0: D7610029 00012C26
	v_writelane_b32 v43, s50, 2                                // 0000000052E8: D761002B 00010432
	v_writelane_b32 v41, s39, 23                               // 0000000052F0: D7610029 00012E27
	v_writelane_b32 v43, s51, 3                                // 0000000052F8: D761002B 00010633
	v_writelane_b32 v41, s40, 24                               // 000000005300: D7610029 00013028
	v_writelane_b32 v43, s0, 4                                 // 000000005308: D761002B 00010800
	v_writelane_b32 v41, s41, 25                               // 000000005310: D7610029 00013229
	v_writelane_b32 v43, s1, 5                                 // 000000005318: D761002B 00010A01
	v_writelane_b32 v41, s42, 26                               // 000000005320: D7610029 0001342A
	v_writelane_b32 v43, s2, 6                                 // 000000005328: D761002B 00010C02
	v_writelane_b32 v41, s43, 27                               // 000000005330: D7610029 0001362B
	v_writelane_b32 v43, s3, 7                                 // 000000005338: D761002B 00010E03
	v_writelane_b32 v41, s44, 28                               // 000000005340: D7610029 0001382C
	v_writelane_b32 v43, s4, 8                                 // 000000005348: D761002B 00011004
	v_writelane_b32 v41, s45, 29                               // 000000005350: D7610029 00013A2D
	v_writelane_b32 v43, s5, 9                                 // 000000005358: D761002B 00011205
	v_writelane_b32 v41, s46, 30                               // 000000005360: D7610029 00013C2E
	v_writelane_b32 v43, s6, 10                                // 000000005368: D761002B 00011406
	v_writelane_b32 v41, s47, 31                               // 000000005370: D7610029 00013E2F
	v_writelane_b32 v43, s7, 11                                // 000000005378: D761002B 00011607
	v_writelane_b32 v43, s8, 12                                // 000000005380: D761002B 00011808
	v_writelane_b32 v43, s9, 13                                // 000000005388: D761002B 00011A09
	v_writelane_b32 v43, s10, 14                               // 000000005390: D761002B 00011C0A
	v_writelane_b32 v43, s11, 15                               // 000000005398: D761002B 00011E0B
	v_writelane_b32 v43, s12, 16                               // 0000000053A0: D761002B 0001200C
	v_writelane_b32 v43, s13, 17                               // 0000000053A8: D761002B 0001220D
	v_writelane_b32 v43, s14, 18                               // 0000000053B0: D761002B 0001240E
	v_writelane_b32 v43, s15, 19                               // 0000000053B8: D761002B 0001260F
	v_writelane_b32 v43, s4, 20                                // 0000000053C0: D761002B 00012804
	v_writelane_b32 v43, s5, 21                                // 0000000053C8: D761002B 00012A05
	v_writelane_b32 v43, s6, 22                                // 0000000053D0: D761002B 00012C06
	v_writelane_b32 v43, s7, 23                                // 0000000053D8: D761002B 00012E07
	v_writelane_b32 v43, s8, 24                                // 0000000053E0: D761002B 00013008
	v_writelane_b32 v43, s9, 25                                // 0000000053E8: D761002B 00013209
	v_writelane_b32 v43, s10, 26                               // 0000000053F0: D761002B 0001340A
	v_writelane_b32 v43, s11, 27                               // 0000000053F8: D761002B 0001360B
	v_writelane_b32 v43, s4, 28                                // 000000005400: D761002B 00013804
	v_writelane_b32 v42, s8, 0                                 // 000000005408: D761002A 00010008
	v_writelane_b32 v43, s5, 29                                // 000000005410: D761002B 00013A05
	v_writelane_b32 v42, s9, 1                                 // 000000005418: D761002A 00010209
	v_writelane_b32 v43, s6, 30                                // 000000005420: D761002B 00013C06
	v_writelane_b32 v42, s10, 2                                // 000000005428: D761002A 0001040A
	v_writelane_b32 v43, s7, 31                                // 000000005430: D761002B 00013E07
	v_writelane_b32 v42, s11, 3                                // 000000005438: D761002A 0001060B
	v_writelane_b32 v42, s8, 4                                 // 000000005440: D761002A 00010808
	v_writelane_b32 v42, s9, 5                                 // 000000005448: D761002A 00010A09
	v_writelane_b32 v42, s10, 6                                // 000000005450: D761002A 00010C0A
	v_writelane_b32 v42, s11, 7                                // 000000005458: D761002A 00010E0B
	v_writelane_b32 v42, s12, 8                                // 000000005460: D761002A 0001100C
	v_writelane_b32 v42, s13, 9                                // 000000005468: D761002A 0001120D
	v_writelane_b32 v42, s14, 10                               // 000000005470: D761002A 0001140E
	v_writelane_b32 v42, s15, 11                               // 000000005478: D761002A 0001160F
	s_and_b32 vcc_lo, exec_lo, s7                              // 000000005480: 8B6A077E
	s_cbranch_vccnz 62472                                      // 000000005484: BFA4F408 <r_3_3_3_8_8_8+0xea8>
	s_branch 62642                                             // 000000005488: BFA0F4B2 <r_3_3_3_8_8_8+0x1154>
	s_mov_b32 s9, -1                                           // 00000000548C: BE8900C1
	v_writelane_b32 v44, s36, 4                                // 000000005490: D761002C 00010824
	s_mov_b32 s24, 0                                           // 000000005498: BE980080
	v_writelane_b32 v44, s37, 5                                // 00000000549C: D761002C 00010A25
	v_writelane_b32 v44, s38, 6                                // 0000000054A4: D761002C 00010C26
	v_writelane_b32 v44, s39, 7                                // 0000000054AC: D761002C 00010E27
	v_writelane_b32 v44, s40, 8                                // 0000000054B4: D761002C 00011028
	v_writelane_b32 v44, s41, 9                                // 0000000054BC: D761002C 00011229
	v_writelane_b32 v44, s42, 10                               // 0000000054C4: D761002C 0001142A
	v_writelane_b32 v44, s43, 11                               // 0000000054CC: D761002C 0001162B
	v_writelane_b32 v44, s44, 12                               // 0000000054D4: D761002C 0001182C
	v_writelane_b32 v44, s45, 13                               // 0000000054DC: D761002C 00011A2D
	v_writelane_b32 v44, s46, 14                               // 0000000054E4: D761002C 00011C2E
	v_writelane_b32 v44, s47, 15                               // 0000000054EC: D761002C 00011E2F
	v_writelane_b32 v44, s48, 16                               // 0000000054F4: D761002C 00012030
	v_writelane_b32 v44, s49, 17                               // 0000000054FC: D761002C 00012231
	v_writelane_b32 v44, s50, 18                               // 000000005504: D761002C 00012432
	v_writelane_b32 v44, s51, 19                               // 00000000550C: D761002C 00012633
	v_writelane_b32 v44, s56, 20                               // 000000005514: D761002C 00012838
	v_writelane_b32 v44, s57, 21                               // 00000000551C: D761002C 00012A39
	v_writelane_b32 v44, s58, 22                               // 000000005524: D761002C 00012C3A
	v_writelane_b32 v44, s59, 23                               // 00000000552C: D761002C 00012E3B
	v_writelane_b32 v44, s60, 24                               // 000000005534: D761002C 0001303C
	v_writelane_b32 v44, s61, 25                               // 00000000553C: D761002C 0001323D
	v_writelane_b32 v44, s62, 26                               // 000000005544: D761002C 0001343E
	v_writelane_b32 v44, s63, 27                               // 00000000554C: D761002C 0001363F
	v_writelane_b32 v44, s4, 28                                // 000000005554: D761002C 00013804
	v_writelane_b32 v39, s8, 0                                 // 00000000555C: D7610027 00010008
	v_writelane_b32 v44, s5, 29                                // 000000005564: D761002C 00013A05
	v_writelane_b32 v39, s9, 1                                 // 00000000556C: D7610027 00010209
	v_writelane_b32 v44, s6, 30                                // 000000005574: D761002C 00013C06
	v_writelane_b32 v39, s10, 2                                // 00000000557C: D7610027 0001040A
	v_writelane_b32 v44, s7, 31                                // 000000005584: D761002C 00013E07
	v_writelane_b32 v39, s11, 3                                // 00000000558C: D7610027 0001060B
	v_writelane_b32 v39, s12, 4                                // 000000005594: D7610027 0001080C
	v_writelane_b32 v39, s13, 5                                // 00000000559C: D7610027 00010A0D
	v_writelane_b32 v39, s14, 6                                // 0000000055A4: D7610027 00010C0E
	v_writelane_b32 v39, s15, 7                                // 0000000055AC: D7610027 00010E0F
	v_writelane_b32 v39, s16, 8                                // 0000000055B4: D7610027 00011010
	v_writelane_b32 v39, s17, 9                                // 0000000055BC: D7610027 00011211
	v_writelane_b32 v39, s18, 10                               // 0000000055C4: D7610027 00011412
	v_writelane_b32 v39, s19, 11                               // 0000000055CC: D7610027 00011613
	v_writelane_b32 v39, s56, 12                               // 0000000055D4: D7610027 00011838
	v_writelane_b32 v39, s57, 13                               // 0000000055DC: D7610027 00011A39
	v_writelane_b32 v39, s58, 14                               // 0000000055E4: D7610027 00011C3A
	v_writelane_b32 v39, s59, 15                               // 0000000055EC: D7610027 00011E3B
	v_writelane_b32 v39, s60, 16                               // 0000000055F4: D7610027 0001203C
	v_writelane_b32 v39, s61, 17                               // 0000000055FC: D7610027 0001223D
	v_writelane_b32 v39, s62, 18                               // 000000005604: D7610027 0001243E
	v_writelane_b32 v39, s63, 19                               // 00000000560C: D7610027 0001263F
	v_writelane_b32 v39, s8, 20                                // 000000005614: D7610027 00012808
	v_writelane_b32 v39, s9, 21                                // 00000000561C: D7610027 00012A09
	v_writelane_b32 v39, s10, 22                               // 000000005624: D7610027 00012C0A
	v_writelane_b32 v39, s11, 23                               // 00000000562C: D7610027 00012E0B
	v_writelane_b32 v39, s12, 24                               // 000000005634: D7610027 0001300C
	v_writelane_b32 v39, s13, 25                               // 00000000563C: D7610027 0001320D
	v_writelane_b32 v39, s14, 26                               // 000000005644: D7610027 0001340E
	v_writelane_b32 v39, s15, 27                               // 00000000564C: D7610027 0001360F
	v_writelane_b32 v39, s80, 28                               // 000000005654: D7610027 00013850
	v_writelane_b32 v38, s84, 0                                // 00000000565C: D7610026 00010054
	v_writelane_b32 v39, s81, 29                               // 000000005664: D7610027 00013A51
	v_writelane_b32 v38, s85, 1                                // 00000000566C: D7610026 00010255
	v_writelane_b32 v39, s82, 30                               // 000000005674: D7610027 00013C52
	v_writelane_b32 v38, s86, 2                                // 00000000567C: D7610026 00010456
	v_writelane_b32 v39, s83, 31                               // 000000005684: D7610027 00013E53
	v_writelane_b32 v38, s87, 3                                // 00000000568C: D7610026 00010657
	v_writelane_b32 v38, s12, 4                                // 000000005694: D7610026 0001080C
	v_writelane_b32 v38, s13, 5                                // 00000000569C: D7610026 00010A0D
	v_writelane_b32 v38, s14, 6                                // 0000000056A4: D7610026 00010C0E
	v_writelane_b32 v38, s15, 7                                // 0000000056AC: D7610026 00010E0F
	v_writelane_b32 v38, s16, 8                                // 0000000056B4: D7610026 00011010
	v_writelane_b32 v38, s17, 9                                // 0000000056BC: D7610026 00011211
	v_writelane_b32 v38, s18, 10                               // 0000000056C4: D7610026 00011412
	v_writelane_b32 v38, s19, 11                               // 0000000056CC: D7610026 00011613
	s_and_b32 vcc_lo, exec_lo, s9                              // 0000000056D4: 8B6A097E
	s_cbranch_vccnz 62966                                      // 0000000056D8: BFA4F5F6 <r_3_3_3_8_8_8+0x18b4>
	s_branch 63168                                             // 0000000056DC: BFA0F6C0 <r_3_3_3_8_8_8+0x1be0>
	s_mov_b32 s3, -1                                           // 0000000056E0: BE8300C1
	s_mov_b32 s2, 0                                            // 0000000056E4: BE820080
	v_writelane_b32 v41, s0, 28                                // 0000000056E8: D7610029 00013800
	v_writelane_b32 v41, s1, 29                                // 0000000056F0: D7610029 00013A01
	v_writelane_b32 v41, s2, 30                                // 0000000056F8: D7610029 00013C02
	v_writelane_b32 v41, s3, 31                                // 000000005700: D7610029 00013E03
	s_or_saveexec_b32 s105, -1                                 // 000000005708: BEE922C1
	scratch_store_b32 off, v41, off offset:4                   // 00000000570C: DC690004 007C2900
	s_mov_b32 exec_lo, s105                                    // 000000005714: BEFE0069
	v_writelane_b32 v45, s4, 0                                 // 000000005718: D761002D 00010004
	v_writelane_b32 v45, s5, 1                                 // 000000005720: D761002D 00010205
	v_writelane_b32 v45, s6, 2                                 // 000000005728: D761002D 00010406
	v_writelane_b32 v45, s7, 3                                 // 000000005730: D761002D 00010607
	v_writelane_b32 v45, s8, 4                                 // 000000005738: D761002D 00010808
	v_writelane_b32 v45, s9, 5                                 // 000000005740: D761002D 00010A09
	v_writelane_b32 v45, s10, 6                                // 000000005748: D761002D 00010C0A
	v_writelane_b32 v45, s11, 7                                // 000000005750: D761002D 00010E0B
	v_writelane_b32 v45, s12, 8                                // 000000005758: D761002D 0001100C
	v_writelane_b32 v45, s13, 9                                // 000000005760: D761002D 0001120D
	v_writelane_b32 v45, s14, 10                               // 000000005768: D761002D 0001140E
	v_writelane_b32 v45, s15, 11                               // 000000005770: D761002D 0001160F
	v_writelane_b32 v45, s0, 12                                // 000000005778: D761002D 00011800
	v_writelane_b32 v45, s1, 13                                // 000000005780: D761002D 00011A01
	v_writelane_b32 v45, s2, 14                                // 000000005788: D761002D 00011C02
	v_writelane_b32 v45, s3, 15                                // 000000005790: D761002D 00011E03
	v_writelane_b32 v45, s4, 16                                // 000000005798: D761002D 00012004
	v_writelane_b32 v45, s5, 17                                // 0000000057A0: D761002D 00012205
	v_writelane_b32 v45, s6, 18                                // 0000000057A8: D761002D 00012406
	v_writelane_b32 v45, s7, 19                                // 0000000057B0: D761002D 00012607
	v_writelane_b32 v45, s4, 20                                // 0000000057B8: D761002D 00012804
	v_writelane_b32 v45, s5, 21                                // 0000000057C0: D761002D 00012A05
	v_writelane_b32 v45, s6, 22                                // 0000000057C8: D761002D 00012C06
	v_writelane_b32 v45, s7, 23                                // 0000000057D0: D761002D 00012E07
	v_writelane_b32 v45, s8, 24                                // 0000000057D8: D761002D 00013008
	v_writelane_b32 v45, s9, 25                                // 0000000057E0: D761002D 00013209
	v_writelane_b32 v45, s10, 26                               // 0000000057E8: D761002D 0001340A
	v_writelane_b32 v45, s11, 27                               // 0000000057F0: D761002D 0001360B
	v_writelane_b32 v45, s12, 28                               // 0000000057F8: D761002D 0001380C
	v_writelane_b32 v45, s13, 29                               // 000000005800: D761002D 00013A0D
	v_writelane_b32 v45, s14, 30                               // 000000005808: D761002D 00013C0E
	v_writelane_b32 v45, s15, 31                               // 000000005810: D761002D 00013E0F
	s_or_saveexec_b32 s105, -1                                 // 000000005818: BEE922C1
	scratch_store_b32 off, v45, off offset:20                  // 00000000581C: DC690014 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000005824: BEFE0069
	v_writelane_b32 v36, s16, 0                                // 000000005828: D7610024 00010010
	v_writelane_b32 v36, s17, 1                                // 000000005830: D7610024 00010211
	v_writelane_b32 v36, s18, 2                                // 000000005838: D7610024 00010412
	v_writelane_b32 v36, s19, 3                                // 000000005840: D7610024 00010613
	v_writelane_b32 v36, s0, 4                                 // 000000005848: D7610024 00010800
	v_writelane_b32 v36, s1, 5                                 // 000000005850: D7610024 00010A01
	v_writelane_b32 v36, s2, 6                                 // 000000005858: D7610024 00010C02
	v_writelane_b32 v36, s3, 7                                 // 000000005860: D7610024 00010E03
	v_writelane_b32 v36, s4, 8                                 // 000000005868: D7610024 00011004
	v_writelane_b32 v36, s5, 9                                 // 000000005870: D7610024 00011205
	v_writelane_b32 v36, s6, 10                                // 000000005878: D7610024 00011406
	v_writelane_b32 v36, s7, 11                                // 000000005880: D7610024 00011607
	v_writelane_b32 v36, s0, 12                                // 000000005888: D7610024 00011800
	v_writelane_b32 v36, s1, 13                                // 000000005890: D7610024 00011A01
	v_writelane_b32 v36, s2, 14                                // 000000005898: D7610024 00011C02
	v_writelane_b32 v36, s3, 15                                // 0000000058A0: D7610024 00011E03
	v_writelane_b32 v36, s4, 16                                // 0000000058A8: D7610024 00012004
	v_writelane_b32 v36, s5, 17                                // 0000000058B0: D7610024 00012205
	v_writelane_b32 v36, s6, 18                                // 0000000058B8: D7610024 00012406
	v_writelane_b32 v36, s7, 19                                // 0000000058C0: D7610024 00012607
	v_writelane_b32 v36, s0, 20                                // 0000000058C8: D7610024 00012800
	v_writelane_b32 v36, s1, 21                                // 0000000058D0: D7610024 00012A01
	v_writelane_b32 v36, s2, 22                                // 0000000058D8: D7610024 00012C02
	v_writelane_b32 v36, s3, 23                                // 0000000058E0: D7610024 00012E03
	v_writelane_b32 v36, s4, 24                                // 0000000058E8: D7610024 00013004
	v_writelane_b32 v36, s5, 25                                // 0000000058F0: D7610024 00013205
	v_writelane_b32 v36, s6, 26                                // 0000000058F8: D7610024 00013406
	v_writelane_b32 v36, s7, 27                                // 000000005900: D7610024 00013607
	v_writelane_b32 v36, s4, 28                                // 000000005908: D7610024 00013804
	v_writelane_b32 v36, s5, 29                                // 000000005910: D7610024 00013A05
	v_writelane_b32 v36, s6, 30                                // 000000005918: D7610024 00013C06
	v_writelane_b32 v36, s7, 31                                // 000000005920: D7610024 00013E07
	s_or_saveexec_b32 s105, -1                                 // 000000005928: BEE922C1
	scratch_store_b32 off, v36, off offset:24                  // 00000000592C: DC690018 007C2400
	s_mov_b32 exec_lo, s105                                    // 000000005934: BEFE0069
	v_writelane_b32 v40, s8, 0                                 // 000000005938: D7610028 00010008
	v_writelane_b32 v40, s9, 1                                 // 000000005940: D7610028 00010209
	v_writelane_b32 v40, s10, 2                                // 000000005948: D7610028 0001040A
	v_writelane_b32 v40, s11, 3                                // 000000005950: D7610028 0001060B
	s_or_saveexec_b32 s105, -1                                 // 000000005958: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000595C: BF870499
	s_mov_b32 exec_lo, s105                                    // 000000005960: BEFE0069
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000005964: 8B6A037E
	s_cbranch_vccnz 63622                                      // 000000005968: BFA4F886 <r_3_3_3_8_8_8+0x2584>
	s_branch 63861                                             // 00000000596C: BFA0F975 <r_3_3_3_8_8_8+0x2944>
	s_mov_b32 s3, -1                                           // 000000005970: BE8300C1
	s_mov_b32 s2, 0                                            // 000000005974: BE820080
	v_writelane_b32 v36, s0, 28                                // 000000005978: D7610024 00013800
	v_writelane_b32 v36, s1, 29                                // 000000005980: D7610024 00013A01
	v_writelane_b32 v36, s2, 30                                // 000000005988: D7610024 00013C02
	v_writelane_b32 v36, s3, 31                                // 000000005990: D7610024 00013E03
	s_or_saveexec_b32 s105, -1                                 // 000000005998: BEE922C1
	scratch_store_b32 off, v36, off offset:36                  // 00000000599C: DC690024 007C2400
	s_mov_b32 exec_lo, s105                                    // 0000000059A4: BEFE0069
	v_writelane_b32 v40, s4, 0                                 // 0000000059A8: D7610028 00010004
	v_writelane_b32 v40, s5, 1                                 // 0000000059B0: D7610028 00010205
	v_writelane_b32 v40, s6, 2                                 // 0000000059B8: D7610028 00010406
	v_writelane_b32 v40, s7, 3                                 // 0000000059C0: D7610028 00010607
	v_writelane_b32 v40, s8, 4                                 // 0000000059C8: D7610028 00010808
	v_writelane_b32 v40, s9, 5                                 // 0000000059D0: D7610028 00010A09
	v_writelane_b32 v40, s10, 6                                // 0000000059D8: D7610028 00010C0A
	v_writelane_b32 v40, s11, 7                                // 0000000059E0: D7610028 00010E0B
	v_writelane_b32 v40, s12, 8                                // 0000000059E8: D7610028 0001100C
	v_writelane_b32 v40, s13, 9                                // 0000000059F0: D7610028 0001120D
	v_writelane_b32 v40, s14, 10                               // 0000000059F8: D7610028 0001140E
	v_writelane_b32 v40, s15, 11                               // 000000005A00: D7610028 0001160F
	v_writelane_b32 v40, s0, 12                                // 000000005A08: D7610028 00011800
	v_writelane_b32 v40, s1, 13                                // 000000005A10: D7610028 00011A01
	v_writelane_b32 v40, s2, 14                                // 000000005A18: D7610028 00011C02
	v_writelane_b32 v40, s3, 15                                // 000000005A20: D7610028 00011E03
	v_writelane_b32 v40, s4, 16                                // 000000005A28: D7610028 00012004
	v_writelane_b32 v40, s5, 17                                // 000000005A30: D7610028 00012205
	v_writelane_b32 v40, s6, 18                                // 000000005A38: D7610028 00012406
	v_writelane_b32 v40, s7, 19                                // 000000005A40: D7610028 00012607
	v_writelane_b32 v40, s8, 20                                // 000000005A48: D7610028 00012808
	v_writelane_b32 v40, s9, 21                                // 000000005A50: D7610028 00012A09
	v_writelane_b32 v40, s10, 22                               // 000000005A58: D7610028 00012C0A
	v_writelane_b32 v40, s11, 23                               // 000000005A60: D7610028 00012E0B
	v_writelane_b32 v40, s12, 24                               // 000000005A68: D7610028 0001300C
	v_writelane_b32 v40, s13, 25                               // 000000005A70: D7610028 0001320D
	v_writelane_b32 v40, s14, 26                               // 000000005A78: D7610028 0001340E
	v_writelane_b32 v40, s15, 27                               // 000000005A80: D7610028 0001360F
	v_writelane_b32 v40, s0, 28                                // 000000005A88: D7610028 00013800
	v_writelane_b32 v41, s4, 0                                 // 000000005A90: D7610029 00010004
	v_writelane_b32 v40, s1, 29                                // 000000005A98: D7610028 00013A01
	v_writelane_b32 v41, s5, 1                                 // 000000005AA0: D7610029 00010205
	v_writelane_b32 v40, s2, 30                                // 000000005AA8: D7610028 00013C02
	v_writelane_b32 v41, s6, 2                                 // 000000005AB0: D7610029 00010406
	v_writelane_b32 v40, s3, 31                                // 000000005AB8: D7610028 00013E03
	v_writelane_b32 v41, s7, 3                                 // 000000005AC0: D7610029 00010607
	v_writelane_b32 v41, s0, 4                                 // 000000005AC8: D7610029 00010800
	v_writelane_b32 v41, s1, 5                                 // 000000005AD0: D7610029 00010A01
	v_writelane_b32 v41, s2, 6                                 // 000000005AD8: D7610029 00010C02
	v_writelane_b32 v41, s3, 7                                 // 000000005AE0: D7610029 00010E03
	v_writelane_b32 v41, s4, 8                                 // 000000005AE8: D7610029 00011004
	v_writelane_b32 v41, s5, 9                                 // 000000005AF0: D7610029 00011205
	v_writelane_b32 v41, s6, 10                                // 000000005AF8: D7610029 00011406
	v_writelane_b32 v41, s7, 11                                // 000000005B00: D7610029 00011607
	v_writelane_b32 v41, s0, 12                                // 000000005B08: D7610029 00011800
	v_writelane_b32 v41, s1, 13                                // 000000005B10: D7610029 00011A01
	v_writelane_b32 v41, s2, 14                                // 000000005B18: D7610029 00011C02
	v_writelane_b32 v41, s3, 15                                // 000000005B20: D7610029 00011E03
	v_writelane_b32 v41, s4, 16                                // 000000005B28: D7610029 00012004
	v_writelane_b32 v41, s5, 17                                // 000000005B30: D7610029 00012205
	v_writelane_b32 v41, s6, 18                                // 000000005B38: D7610029 00012406
	v_writelane_b32 v41, s7, 19                                // 000000005B40: D7610029 00012607
	v_writelane_b32 v41, s4, 20                                // 000000005B48: D7610029 00012804
	v_writelane_b32 v41, s5, 21                                // 000000005B50: D7610029 00012A05
	v_writelane_b32 v41, s6, 22                                // 000000005B58: D7610029 00012C06
	v_writelane_b32 v41, s7, 23                                // 000000005B60: D7610029 00012E07
	v_writelane_b32 v41, s8, 24                                // 000000005B68: D7610029 00013008
	v_writelane_b32 v41, s9, 25                                // 000000005B70: D7610029 00013209
	v_writelane_b32 v41, s10, 26                               // 000000005B78: D7610029 0001340A
	v_writelane_b32 v41, s11, 27                               // 000000005B80: D7610029 0001360B
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000005B88: 8B6A037E
	s_cbranch_vccnz 64150                                      // 000000005B8C: BFA4FA96 <r_3_3_3_8_8_8+0x2fe8>
	s_branch 64346                                             // 000000005B90: BFA0FB5A <r_3_3_3_8_8_8+0x32fc>
	s_mov_b32 s3, -1                                           // 000000005B94: BE8300C1
	s_mov_b32 s2, 0                                            // 000000005B98: BE820080
	v_readlane_b32 s92, v37, 20                                // 000000005B9C: D760005C 00012925
	v_writelane_b32 v38, s0, 20                                // 000000005BA4: D7610026 00012800
	v_writelane_b32 v39, s12, 0                                // 000000005BAC: D7610027 0001000C
	v_readlane_b32 s93, v37, 21                                // 000000005BB4: D760005D 00012B25
	v_readlane_b32 s94, v37, 22                                // 000000005BBC: D760005E 00012D25
	v_readlane_b32 s95, v37, 23                                // 000000005BC4: D760005F 00012F25
	v_writelane_b32 v38, s1, 21                                // 000000005BCC: D7610026 00012A01
	v_writelane_b32 v39, s13, 1                                // 000000005BD4: D7610027 0001020D
	v_readlane_b32 s96, v37, 24                                // 000000005BDC: D7600060 00013125
	v_readlane_b32 s97, v37, 25                                // 000000005BE4: D7600061 00013325
	v_readlane_b32 s98, v37, 26                                // 000000005BEC: D7600062 00013525
	v_writelane_b32 v38, s2, 22                                // 000000005BF4: D7610026 00012C02
	v_writelane_b32 v39, s14, 2                                // 000000005BFC: D7610027 0001040E
	v_readlane_b32 s99, v37, 27                                // 000000005C04: D7600063 00013725
	v_writelane_b32 v38, s3, 23                                // 000000005C0C: D7610026 00012E03
	v_writelane_b32 v39, s15, 3                                // 000000005C14: D7610027 0001060F
	v_writelane_b32 v38, s4, 24                                // 000000005C1C: D7610026 00013004
	v_writelane_b32 v38, s5, 25                                // 000000005C24: D7610026 00013205
	v_writelane_b32 v38, s6, 26                                // 000000005C2C: D7610026 00013406
	v_writelane_b32 v38, s7, 27                                // 000000005C34: D7610026 00013607
	v_writelane_b32 v39, s0, 4                                 // 000000005C3C: D7610027 00010800
	v_writelane_b32 v38, s8, 28                                // 000000005C44: D7610026 00013808
	v_writelane_b32 v39, s1, 5                                 // 000000005C4C: D7610027 00010A01
	v_writelane_b32 v38, s9, 29                                // 000000005C54: D7610026 00013A09
	v_writelane_b32 v39, s2, 6                                 // 000000005C5C: D7610027 00010C02
	v_writelane_b32 v38, s10, 30                               // 000000005C64: D7610026 00013C0A
	v_writelane_b32 v39, s3, 7                                 // 000000005C6C: D7610027 00010E03
	v_writelane_b32 v38, s11, 31                               // 000000005C74: D7610026 00013E0B
	v_writelane_b32 v39, s4, 8                                 // 000000005C7C: D7610027 00011004
	v_writelane_b32 v39, s5, 9                                 // 000000005C84: D7610027 00011205
	v_writelane_b32 v39, s6, 10                                // 000000005C8C: D7610027 00011406
	v_writelane_b32 v39, s7, 11                                // 000000005C94: D7610027 00011607
	v_writelane_b32 v39, s0, 12                                // 000000005C9C: D7610027 00011800
	v_writelane_b32 v39, s1, 13                                // 000000005CA4: D7610027 00011A01
	v_writelane_b32 v39, s2, 14                                // 000000005CAC: D7610027 00011C02
	v_writelane_b32 v39, s3, 15                                // 000000005CB4: D7610027 00011E03
	v_writelane_b32 v39, s4, 16                                // 000000005CBC: D7610027 00012004
	v_writelane_b32 v39, s5, 17                                // 000000005CC4: D7610027 00012205
	v_writelane_b32 v39, s6, 18                                // 000000005CCC: D7610027 00012406
	v_writelane_b32 v39, s7, 19                                // 000000005CD4: D7610027 00012607
	v_writelane_b32 v39, s8, 20                                // 000000005CDC: D7610027 00012808
	v_writelane_b32 v39, s9, 21                                // 000000005CE4: D7610027 00012A09
	v_writelane_b32 v39, s10, 22                               // 000000005CEC: D7610027 00012C0A
	v_writelane_b32 v39, s11, 23                               // 000000005CF4: D7610027 00012E0B
	v_writelane_b32 v39, s12, 24                               // 000000005CFC: D7610027 0001300C
	v_writelane_b32 v39, s13, 25                               // 000000005D04: D7610027 0001320D
	v_writelane_b32 v39, s14, 26                               // 000000005D0C: D7610027 0001340E
	v_writelane_b32 v39, s15, 27                               // 000000005D14: D7610027 0001360F
	v_readlane_b32 s8, v42, 12                                 // 000000005D1C: D7600008 0001192A
	v_readlane_b32 s9, v42, 13                                 // 000000005D24: D7600009 00011B2A
	v_readlane_b32 s10, v42, 14                                // 000000005D2C: D760000A 00011D2A
	v_readlane_b32 s11, v42, 15                                // 000000005D34: D760000B 00011F2A
	v_writelane_b32 v39, s0, 28                                // 000000005D3C: D7610027 00013800
	v_writelane_b32 v44, s4, 0                                 // 000000005D44: D761002C 00010004
	v_readlane_b32 s12, v42, 16                                // 000000005D4C: D760000C 0001212A
	v_readlane_b32 s13, v42, 17                                // 000000005D54: D760000D 0001232A
	v_readlane_b32 s14, v42, 18                                // 000000005D5C: D760000E 0001252A
	v_writelane_b32 v39, s1, 29                                // 000000005D64: D7610027 00013A01
	v_writelane_b32 v44, s5, 1                                 // 000000005D6C: D761002C 00010205
	v_readlane_b32 s15, v42, 19                                // 000000005D74: D760000F 0001272A
	v_writelane_b32 v39, s2, 30                                // 000000005D7C: D7610027 00013C02
	v_writelane_b32 v44, s6, 2                                 // 000000005D84: D761002C 00010406
	v_writelane_b32 v39, s3, 31                                // 000000005D8C: D7610027 00013E03
	v_writelane_b32 v44, s7, 3                                 // 000000005D94: D761002C 00010607
	v_writelane_b32 v44, s0, 4                                 // 000000005D9C: D761002C 00010800
	v_writelane_b32 v44, s1, 5                                 // 000000005DA4: D761002C 00010A01
	v_writelane_b32 v44, s2, 6                                 // 000000005DAC: D761002C 00010C02
	v_writelane_b32 v44, s3, 7                                 // 000000005DB4: D761002C 00010E03
	v_writelane_b32 v44, s4, 8                                 // 000000005DBC: D761002C 00011004
	v_writelane_b32 v44, s5, 9                                 // 000000005DC4: D761002C 00011205
	v_writelane_b32 v44, s6, 10                                // 000000005DCC: D761002C 00011406
	v_writelane_b32 v44, s7, 11                                // 000000005DD4: D761002C 00011607
	v_writelane_b32 v44, s36, 12                               // 000000005DDC: D761002C 00011824
	v_writelane_b32 v44, s37, 13                               // 000000005DE4: D761002C 00011A25
	v_writelane_b32 v44, s38, 14                               // 000000005DEC: D761002C 00011C26
	v_writelane_b32 v44, s39, 15                               // 000000005DF4: D761002C 00011E27
	v_writelane_b32 v44, s40, 16                               // 000000005DFC: D761002C 00012028
	v_writelane_b32 v44, s41, 17                               // 000000005E04: D761002C 00012229
	v_writelane_b32 v44, s42, 18                               // 000000005E0C: D761002C 0001242A
	v_writelane_b32 v44, s43, 19                               // 000000005E14: D761002C 0001262B
	v_writelane_b32 v44, s44, 20                               // 000000005E1C: D761002C 0001282C
	v_writelane_b32 v44, s45, 21                               // 000000005E24: D761002C 00012A2D
	v_writelane_b32 v44, s46, 22                               // 000000005E2C: D761002C 00012C2E
	v_writelane_b32 v44, s47, 23                               // 000000005E34: D761002C 00012E2F
	v_writelane_b32 v44, s48, 24                               // 000000005E3C: D761002C 00013030
	v_writelane_b32 v44, s49, 25                               // 000000005E44: D761002C 00013231
	v_writelane_b32 v44, s50, 26                               // 000000005E4C: D761002C 00013432
	v_writelane_b32 v44, s51, 27                               // 000000005E54: D761002C 00013633
	v_writelane_b32 v44, s16, 28                               // 000000005E5C: D761002C 00013810
	v_writelane_b32 v44, s17, 29                               // 000000005E64: D761002C 00013A11
	v_writelane_b32 v44, s18, 30                               // 000000005E6C: D761002C 00013C12
	v_writelane_b32 v44, s19, 31                               // 000000005E74: D761002C 00013E13
	s_or_saveexec_b32 s105, -1                                 // 000000005E7C: BEE922C1
	scratch_store_b32 off, v44, off offset:28                  // 000000005E80: DC69001C 007C2C00
	s_mov_b32 exec_lo, s105                                    // 000000005E88: BEFE0069
	v_writelane_b32 v36, s20, 0                                // 000000005E8C: D7610024 00010014
	v_writelane_b32 v36, s21, 1                                // 000000005E94: D7610024 00010215
	v_writelane_b32 v36, s22, 2                                // 000000005E9C: D7610024 00010416
	v_writelane_b32 v36, s23, 3                                // 000000005EA4: D7610024 00010617
	s_or_saveexec_b32 s105, -1                                 // 000000005EAC: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005EB0: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000005EB4: BEFE0069
	v_writelane_b32 v42, s8, 12                                // 000000005EB8: D761002A 00011808
	s_and_b32 vcc_lo, exec_lo, s3                              // 000000005EC0: 8B6A037E
	v_writelane_b32 v42, s9, 13                                // 000000005EC4: D761002A 00011A09
	v_writelane_b32 v42, s10, 14                               // 000000005ECC: D761002A 00011C0A
	v_writelane_b32 v42, s11, 15                               // 000000005ED4: D761002A 00011E0B
	v_writelane_b32 v42, s12, 16                               // 000000005EDC: D761002A 0001200C
	v_writelane_b32 v42, s13, 17                               // 000000005EE4: D761002A 0001220D
	v_writelane_b32 v42, s14, 18                               // 000000005EEC: D761002A 0001240E
	v_writelane_b32 v42, s15, 19                               // 000000005EF4: D761002A 0001260F
	s_cbranch_vccz 188                                         // 000000005EFC: BFA300BC <r_3_3_3_8_8_8+0x4bf0>
	s_or_saveexec_b32 s105, -1                                 // 000000005F00: BEE922C1
	scratch_load_b32 v44, off, off offset:12                   // 000000005F04: DC51000C 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 000000005F0C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000005F10: BF8903F7
	v_readlane_b32 s0, v44, 18                                 // 000000005F14: D7600000 0001252C
	v_readlane_b32 s1, v44, 19                                 // 000000005F1C: D7600001 0001272C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005F24: BF870092
	s_add_u32 s0, s0, s28                                      // 000000005F28: 80001C00
	s_addc_u32 s1, s1, s29                                     // 000000005F2C: 82011D01
	s_or_saveexec_b32 s105, -1                                 // 000000005F30: BEE922C1
	scratch_load_b32 v45, off, off                             // 000000005F34: DC510000 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000005F3C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000005F40: BF8903F7
	v_readlane_b32 s2, v45, 4                                  // 000000005F44: D7600002 0001092D
	s_load_b256 s[68:75], s[0:1], null                         // 000000005F4C: F40C1100 F8000000
	v_readlane_b32 s0, v45, 5                                  // 000000005F54: D7600000 00010B2D
	v_readlane_b32 s1, v44, 23                                 // 000000005F5C: D7600001 00012F2C
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000005F64: BF870113
	s_add_u32 s2, s2, s28                                      // 000000005F68: 80021C02
	s_addc_u32 s3, s0, s29                                     // 000000005F6C: 82031D00
	v_readlane_b32 s0, v44, 22                                 // 000000005F70: D7600000 00012D2C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005F78: BF870001
	s_add_u32 s0, s0, s28                                      // 000000005F7C: 80001C00
	s_addc_u32 s1, s1, s29                                     // 000000005F80: 82011D01
	s_clause 0x1                                               // 000000005F84: BF850001
	s_load_b256 s[52:59], s[2:3], null                         // 000000005F88: F40C0D01 F8000000
	s_load_b256 s[4:11], s[0:1], 0x1100                        // 000000005F90: F40C0100 F8001100
	s_waitcnt lgkmcnt(0)                                       // 000000005F98: BF89FC07
	v_writelane_b32 v39, s4, 28                                // 000000005F9C: D7610027 00013804
	v_writelane_b32 v39, s5, 29                                // 000000005FA4: D7610027 00013A05
	v_writelane_b32 v39, s6, 30                                // 000000005FAC: D7610027 00013C06
	v_writelane_b32 v39, s7, 31                                // 000000005FB4: D7610027 00013E07
	s_or_saveexec_b32 s105, -1                                 // 000000005FBC: BEE922C1
	scratch_load_b32 v45, off, off offset:28                   // 000000005FC0: DC51001C 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000005FC8: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000005FCC: BF8903F7
	v_writelane_b32 v45, s8, 0                                 // 000000005FD0: D761002D 00010008
	s_mov_b32 s101, s100                                       // 000000005FD8: BEE50064
	s_mov_b32 s102, s100                                       // 000000005FDC: BEE60064
	s_mov_b32 s103, s100                                       // 000000005FE0: BEE70064
	s_mov_b32 s2, -1                                           // 000000005FE4: BE8200C1
	v_writelane_b32 v45, s9, 1                                 // 000000005FE8: D761002D 00010209
	s_mov_b64 s[48:49], s[100:101]                             // 000000005FF0: BEB00164
	s_mov_b64 s[12:13], s[100:101]                             // 000000005FF4: BE8C0164
	s_mov_b64 s[84:85], s[100:101]                             // 000000005FF8: BED40164
	s_mov_b64 s[40:41], s[100:101]                             // 000000005FFC: BEA80164
	v_writelane_b32 v45, s10, 2                                // 000000006000: D761002D 0001040A
	s_mov_b64 s[36:37], s[100:101]                             // 000000006008: BEA40164
	s_mov_b64 s[50:51], s[102:103]                             // 00000000600C: BEB20166
	s_mov_b64 s[14:15], s[102:103]                             // 000000006010: BE8E0166
	s_mov_b64 s[86:87], s[102:103]                             // 000000006014: BED60166
	v_writelane_b32 v45, s11, 3                                // 000000006018: D761002D 0001060B
	s_mov_b64 s[8:9], s[100:101]                               // 000000006020: BE880164
	s_mov_b64 s[4:5], s[100:101]                               // 000000006024: BE840164
	s_mov_b64 s[10:11], s[102:103]                             // 000000006028: BE8A0166
	s_mov_b64 s[6:7], s[102:103]                               // 00000000602C: BE860166
	v_writelane_b32 v45, s4, 4                                 // 000000006030: D761002D 00010804
	s_mov_b64 s[42:43], s[102:103]                             // 000000006038: BEAA0166
	s_mov_b64 s[38:39], s[102:103]                             // 00000000603C: BEA60166
	v_writelane_b32 v45, s5, 5                                 // 000000006040: D761002D 00010A05
	v_writelane_b32 v45, s6, 6                                 // 000000006048: D761002D 00010C06
	v_writelane_b32 v45, s7, 7                                 // 000000006050: D761002D 00010E07
	v_writelane_b32 v45, s8, 8                                 // 000000006058: D761002D 00011008
	v_writelane_b32 v45, s9, 9                                 // 000000006060: D761002D 00011209
	v_writelane_b32 v45, s10, 10                               // 000000006068: D761002D 0001140A
	v_writelane_b32 v45, s11, 11                               // 000000006070: D761002D 0001160B
	s_or_saveexec_b32 s105, -1                                 // 000000006078: BEE922C1
	scratch_store_b32 off, v45, off offset:28                  // 00000000607C: DC69001C 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000006084: BEFE0069
	s_mov_b64 s[8:9], s[100:101]                               // 000000006088: BE880164
	s_mov_b64 s[10:11], s[102:103]                             // 00000000608C: BE8A0166
	v_writelane_b32 v39, s8, 4                                 // 000000006090: D7610027 00010808
	s_mov_b64 s[44:45], s[100:101]                             // 000000006098: BEAC0164
	s_mov_b64 s[46:47], s[102:103]                             // 00000000609C: BEAE0166
	v_writelane_b32 v38, s36, 20                               // 0000000060A0: D7610026 00012824
	s_mov_b64 s[76:77], s[100:101]                             // 0000000060A8: BECC0164
	v_writelane_b32 v39, s9, 5                                 // 0000000060AC: D7610027 00010A09
	s_mov_b64 s[80:81], s[100:101]                             // 0000000060B4: BED00164
	s_mov_b64 s[88:89], s[100:101]                             // 0000000060B8: BED80164
	s_mov_b64 s[78:79], s[102:103]                             // 0000000060BC: BECE0166
	s_mov_b64 s[82:83], s[102:103]                             // 0000000060C0: BED20166
	v_writelane_b32 v39, s10, 6                                // 0000000060C4: D7610027 00010C0A
	s_mov_b64 s[90:91], s[102:103]                             // 0000000060CC: BEDA0166
	v_writelane_b32 v38, s37, 21                               // 0000000060D0: D7610026 00012A25
	v_writelane_b32 v39, s11, 7                                // 0000000060D8: D7610027 00010E0B
	v_writelane_b32 v38, s38, 22                               // 0000000060E0: D7610026 00012C26
	v_writelane_b32 v39, s12, 8                                // 0000000060E8: D7610027 0001100C
	v_writelane_b32 v38, s39, 23                               // 0000000060F0: D7610026 00012E27
	v_writelane_b32 v39, s13, 9                                // 0000000060F8: D7610027 0001120D
	v_writelane_b32 v38, s40, 24                               // 000000006100: D7610026 00013028
	v_writelane_b32 v39, s14, 10                               // 000000006108: D7610027 0001140E
	v_writelane_b32 v38, s41, 25                               // 000000006110: D7610026 00013229
	v_writelane_b32 v39, s15, 11                               // 000000006118: D7610027 0001160F
	v_writelane_b32 v38, s42, 26                               // 000000006120: D7610026 0001342A
	v_writelane_b32 v39, s48, 0                                // 000000006128: D7610027 00010030
	v_writelane_b32 v38, s43, 27                               // 000000006130: D7610026 0001362B
	v_writelane_b32 v39, s49, 1                                // 000000006138: D7610027 00010231
	v_writelane_b32 v38, s44, 28                               // 000000006140: D7610026 0001382C
	v_writelane_b32 v39, s50, 2                                // 000000006148: D7610027 00010432
	v_writelane_b32 v38, s45, 29                               // 000000006150: D7610026 00013A2D
	v_writelane_b32 v39, s51, 3                                // 000000006158: D7610027 00010633
	v_writelane_b32 v38, s46, 30                               // 000000006160: D7610026 00013C2E
	v_writelane_b32 v39, s76, 12                               // 000000006168: D7610027 0001184C
	v_writelane_b32 v38, s47, 31                               // 000000006170: D7610026 00013E2F
	v_writelane_b32 v39, s77, 13                               // 000000006178: D7610027 00011A4D
	v_writelane_b32 v39, s78, 14                               // 000000006180: D7610027 00011C4E
	v_writelane_b32 v39, s79, 15                               // 000000006188: D7610027 00011E4F
	v_writelane_b32 v39, s80, 16                               // 000000006190: D7610027 00012050
	v_writelane_b32 v39, s81, 17                               // 000000006198: D7610027 00012251
	v_writelane_b32 v39, s82, 18                               // 0000000061A0: D7610027 00012452
	v_writelane_b32 v39, s83, 19                               // 0000000061A8: D7610027 00012653
	v_writelane_b32 v39, s84, 20                               // 0000000061B0: D7610027 00012854
	v_writelane_b32 v39, s85, 21                               // 0000000061B8: D7610027 00012A55
	v_writelane_b32 v39, s86, 22                               // 0000000061C0: D7610027 00012C56
	v_writelane_b32 v39, s87, 23                               // 0000000061C8: D7610027 00012E57
	v_writelane_b32 v39, s88, 24                               // 0000000061D0: D7610027 00013058
	v_writelane_b32 v39, s89, 25                               // 0000000061D8: D7610027 00013259
	v_writelane_b32 v39, s90, 26                               // 0000000061E0: D7610027 0001345A
	v_writelane_b32 v39, s91, 27                               // 0000000061E8: D7610027 0001365B
	s_or_saveexec_b32 s105, -1                                 // 0000000061F0: BEE922C1
	v_mov_b32_e32 v45, v36                                     // 0000000061F4: 7E5A0324
	s_mov_b32 exec_lo, s105                                    // 0000000061F8: BEFE0069
	v_writelane_b32 v45, s68, 4                                // 0000000061FC: D761002D 00010844
	v_writelane_b32 v45, s69, 5                                // 000000006204: D761002D 00010A45
	v_writelane_b32 v45, s70, 6                                // 00000000620C: D761002D 00010C46
	v_writelane_b32 v45, s71, 7                                // 000000006214: D761002D 00010E47
	v_writelane_b32 v45, s72, 8                                // 00000000621C: D761002D 00011048
	v_writelane_b32 v45, s73, 9                                // 000000006224: D761002D 00011249
	v_writelane_b32 v45, s74, 10                               // 00000000622C: D761002D 0001144A
	v_writelane_b32 v45, s75, 11                               // 000000006234: D761002D 0001164B
	s_or_saveexec_b32 s105, -1                                 // 00000000623C: BEE922C1
	scratch_store_b32 off, v43, off offset:44                  // 000000006240: DC69002C 007C2B00
	s_mov_b32 exec_lo, s105                                    // 000000006248: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 00000000624C: BEE922C1
	scratch_store_b32 off, v42, off offset:48                  // 000000006250: DC690030 007C2A00
	s_mov_b32 exec_lo, s105                                    // 000000006258: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 00000000625C: BEE922C1
	scratch_store_b32 off, v37, off offset:64                  // 000000006260: DC690040 007C2500
	s_mov_b32 exec_lo, s105                                    // 000000006268: BEFE0069
	s_mov_b32 s25, s24                                         // 00000000626C: BE990018
	s_mov_b32 s26, s24                                         // 000000006270: BE9A0018
	s_mov_b32 s27, s24                                         // 000000006274: BE9B0018
	s_mov_b64 s[8:9], s[24:25]                                 // 000000006278: BE880118
	s_mov_b64 s[10:11], s[26:27]                               // 00000000627C: BE8A011A
	s_mov_b32 s4, s8                                           // 000000006280: BE840008
	s_mov_b32 s5, s8                                           // 000000006284: BE850008
	s_mov_b32 s6, s8                                           // 000000006288: BE860008
	s_mov_b32 s7, s8                                           // 00000000628C: BE870008
	s_mov_b32 s76, s8                                          // 000000006290: BECC0008
	s_mov_b32 s77, s8                                          // 000000006294: BECD0008
	s_mov_b32 s78, s8                                          // 000000006298: BECE0008
	s_mov_b32 s79, s8                                          // 00000000629C: BECF0008
	s_mov_b32 s68, s8                                          // 0000000062A0: BEC40008
	s_mov_b32 s69, s8                                          // 0000000062A4: BEC50008
	s_mov_b32 s70, s8                                          // 0000000062A8: BEC60008
	s_mov_b32 s71, s8                                          // 0000000062AC: BEC70008
	s_mov_b32 s72, s8                                          // 0000000062B0: BEC80008
	s_mov_b32 s73, s8                                          // 0000000062B4: BEC90008
	s_mov_b32 s74, s8                                          // 0000000062B8: BECA0008
	s_mov_b32 s75, s8                                          // 0000000062BC: BECB0008
	s_mov_b32 s36, s8                                          // 0000000062C0: BEA40008
	s_mov_b32 s37, s8                                          // 0000000062C4: BEA50008
	s_mov_b32 s38, s8                                          // 0000000062C8: BEA60008
	s_mov_b32 s39, s8                                          // 0000000062CC: BEA70008
	s_mov_b32 s40, s8                                          // 0000000062D0: BEA80008
	s_mov_b32 s41, s8                                          // 0000000062D4: BEA90008
	s_mov_b32 s42, s8                                          // 0000000062D8: BEAA0008
	s_mov_b32 s43, s8                                          // 0000000062DC: BEAB0008
	s_mov_b32 s80, s8                                          // 0000000062E0: BED00008
	s_mov_b32 s81, s8                                          // 0000000062E4: BED10008
	s_mov_b32 s82, s8                                          // 0000000062E8: BED20008
	s_mov_b32 s83, s8                                          // 0000000062EC: BED30008
	s_mov_b32 s48, s8                                          // 0000000062F0: BEB00008
	s_mov_b32 s49, s8                                          // 0000000062F4: BEB10008
	s_mov_b32 s50, s8                                          // 0000000062F8: BEB20008
	v_writelane_b32 v45, s4, 12                                // 0000000062FC: D761002D 00011804
	s_mov_b32 s44, s24                                         // 000000006304: BEAC0018
	s_mov_b32 s45, s24                                         // 000000006308: BEAD0018
	s_mov_b32 s46, s24                                         // 00000000630C: BEAE0018
	s_mov_b32 s47, s24                                         // 000000006310: BEAF0018
	v_writelane_b32 v45, s5, 13                                // 000000006314: D761002D 00011A05
	s_and_not1_b32 vcc_lo, exec_lo, s2                         // 00000000631C: 916A027E
	s_mov_b32 s51, s8                                          // 000000006320: BEB30008
	v_writelane_b32 v45, s6, 14                                // 000000006324: D761002D 00011C06
	v_writelane_b32 v45, s7, 15                                // 00000000632C: D761002D 00011E07
	v_writelane_b32 v45, s8, 16                                // 000000006334: D761002D 00012008
	v_writelane_b32 v45, s9, 17                                // 00000000633C: D761002D 00012209
	v_writelane_b32 v45, s10, 18                               // 000000006344: D761002D 0001240A
	v_writelane_b32 v45, s11, 19                               // 00000000634C: D761002D 0001260B
	s_cbranch_vccnz 24                                         // 000000006354: BFA40018 <r_3_3_3_8_8_8+0x4db8>
	s_clause 0x2                                               // 000000006358: BF850002
	s_load_b512 s[68:83], s[0:1], 0x2500                       // 00000000635C: F4101100 F8002500
	s_load_b256 s[4:11], s[0:1], 0x23e0                        // 000000006364: F40C0100 F80023E0
	s_load_b512 s[36:51], s[0:1], 0x2640                       // 00000000636C: F4100900 F8002640
	s_waitcnt lgkmcnt(0)                                       // 000000006374: BF89FC07
	v_writelane_b32 v45, s4, 12                                // 000000006378: D761002D 00011804
	v_writelane_b32 v45, s5, 13                                // 000000006380: D761002D 00011A05
	v_writelane_b32 v45, s6, 14                                // 000000006388: D761002D 00011C06
	v_writelane_b32 v45, s7, 15                                // 000000006390: D761002D 00011E07
	v_writelane_b32 v45, s8, 16                                // 000000006398: D761002D 00012008
	v_writelane_b32 v45, s9, 17                                // 0000000063A0: D761002D 00012209
	v_writelane_b32 v45, s10, 18                               // 0000000063A8: D761002D 0001240A
	v_writelane_b32 v45, s11, 19                               // 0000000063B0: D761002D 0001260B
	s_load_b256 s[4:11], s[0:1], 0x1120                        // 0000000063B8: F40C0100 F8001120
	s_waitcnt lgkmcnt(0)                                       // 0000000063C0: BF89FC07
	v_writelane_b32 v45, s4, 20                                // 0000000063C4: D761002D 00012804
	v_writelane_b32 v45, s5, 21                                // 0000000063CC: D761002D 00012A05
	v_writelane_b32 v45, s6, 22                                // 0000000063D4: D761002D 00012C06
	v_writelane_b32 v45, s7, 23                                // 0000000063DC: D761002D 00012E07
	v_writelane_b32 v45, s8, 24                                // 0000000063E4: D761002D 00013008
	v_writelane_b32 v45, s9, 25                                // 0000000063EC: D761002D 00013209
	v_writelane_b32 v45, s10, 26                               // 0000000063F4: D761002D 0001340A
	v_writelane_b32 v45, s11, 27                               // 0000000063FC: D761002D 0001360B
	s_load_b256 s[4:11], s[0:1], 0xfe0                         // 000000006404: F40C0100 F8000FE0
	s_waitcnt lgkmcnt(0)                                       // 00000000640C: BF89FC07
	v_writelane_b32 v45, s4, 28                                // 000000006410: D761002D 00013804
	v_writelane_b32 v45, s5, 29                                // 000000006418: D761002D 00013A05
	v_writelane_b32 v45, s6, 30                                // 000000006420: D761002D 00013C06
	v_writelane_b32 v45, s7, 31                                // 000000006428: D761002D 00013E07
	s_or_saveexec_b32 s105, -1                                 // 000000006430: BEE922C1
	scratch_store_b32 off, v45, off offset:108                 // 000000006434: DC69006C 007C2D00
	s_mov_b32 exec_lo, s105                                    // 00000000643C: BEFE0069
	v_writelane_b32 v44, s8, 0                                 // 000000006440: D761002C 00010008
	s_and_b32 vcc_lo, exec_lo, s34                             // 000000006448: 8B6A227E
	v_writelane_b32 v44, s9, 1                                 // 00000000644C: D761002C 00010209
	v_writelane_b32 v44, s10, 2                                // 000000006454: D761002C 0001040A
	v_writelane_b32 v44, s11, 3                                // 00000000645C: D761002C 0001060B
	s_load_b512 s[0:15], s[0:1], 0x1240                        // 000000006464: F4100000 F8001240
	s_waitcnt lgkmcnt(0)                                       // 00000000646C: BF89FC07
	v_writelane_b32 v44, s0, 4                                 // 000000006470: D761002C 00010800
	v_writelane_b32 v44, s1, 5                                 // 000000006478: D761002C 00010A01
	v_writelane_b32 v44, s2, 6                                 // 000000006480: D761002C 00010C02
	v_writelane_b32 v44, s3, 7                                 // 000000006488: D761002C 00010E03
	v_writelane_b32 v44, s4, 8                                 // 000000006490: D761002C 00011004
	v_writelane_b32 v44, s5, 9                                 // 000000006498: D761002C 00011205
	v_writelane_b32 v44, s6, 10                                // 0000000064A0: D761002C 00011406
	v_writelane_b32 v44, s7, 11                                // 0000000064A8: D761002C 00011607
	v_writelane_b32 v44, s8, 12                                // 0000000064B0: D761002C 00011808
	v_writelane_b32 v44, s9, 13                                // 0000000064B8: D761002C 00011A09
	v_writelane_b32 v44, s10, 14                               // 0000000064C0: D761002C 00011C0A
	v_writelane_b32 v44, s11, 15                               // 0000000064C8: D761002C 00011E0B
	v_writelane_b32 v44, s12, 16                               // 0000000064D0: D761002C 0001200C
	v_writelane_b32 v44, s13, 17                               // 0000000064D8: D761002C 0001220D
	v_writelane_b32 v44, s14, 18                               // 0000000064E0: D761002C 0001240E
	v_writelane_b32 v44, s15, 19                               // 0000000064E8: D761002C 0001260F
	v_writelane_b32 v44, s52, 20                               // 0000000064F0: D761002C 00012834
	v_writelane_b32 v44, s53, 21                               // 0000000064F8: D761002C 00012A35
	v_writelane_b32 v44, s54, 22                               // 000000006500: D761002C 00012C36
	v_writelane_b32 v44, s55, 23                               // 000000006508: D761002C 00012E37
	v_writelane_b32 v44, s56, 24                               // 000000006510: D761002C 00013038
	v_writelane_b32 v44, s57, 25                               // 000000006518: D761002C 00013239
	v_writelane_b32 v44, s58, 26                               // 000000006520: D761002C 0001343A
	v_writelane_b32 v44, s59, 27                               // 000000006528: D761002C 0001363B
	s_or_saveexec_b32 s105, -1                                 // 000000006530: BEE922C1
	scratch_load_b32 v42, off, off offset:36                   // 000000006534: DC510024 2A7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000653C: BEFE0069
	s_cbranch_vccnz 239                                        // 000000006540: BFA400EF <r_3_3_3_8_8_8+0x5300>
	s_or_saveexec_b32 s105, -1                                 // 000000006544: BEE922C1
	scratch_load_b32 v45, off, off                             // 000000006548: DC510000 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000006550: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000006554: BF8903F7
	v_readlane_b32 s0, v45, 16                                 // 000000006558: D7600000 0001212D
	s_or_saveexec_b32 s105, -1                                 // 000000006560: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006564: BF870099
	s_mov_b32 exec_lo, s105                                    // 000000006568: BEFE0069
	s_add_u32 s2, s0, s28                                      // 00000000656C: 80021C00
	v_readlane_b32 s0, v45, 17                                 // 000000006570: D7600000 0001232D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006578: BF870001
	s_addc_u32 s3, s0, s29                                     // 00000000657C: 82031D00
	s_or_saveexec_b32 s105, -1                                 // 000000006580: BEE922C1
	scratch_load_b32 v45, off, off offset:12                   // 000000006584: DC51000C 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000658C: BEFE0069
	s_mov_b32 s101, s100                                       // 000000006590: BEE50064
	s_mov_b32 s102, s100                                       // 000000006594: BEE60064
	s_mov_b32 s103, s100                                       // 000000006598: BEE70064
	s_mov_b64 s[8:9], s[100:101]                               // 00000000659C: BE880164
	s_mov_b64 s[4:5], s[100:101]                               // 0000000065A0: BE840164
	s_mov_b64 s[10:11], s[102:103]                             // 0000000065A4: BE8A0166
	s_mov_b64 s[6:7], s[102:103]                               // 0000000065A8: BE860166
	v_writelane_b32 v37, s4, 12                                // 0000000065AC: D7610025 00011804
	s_waitcnt vmcnt(0)                                         // 0000000065B4: BF8903F7
	v_readlane_b32 s0, v45, 24                                 // 0000000065B8: D7600000 0001312D
	v_readlane_b32 s1, v45, 25                                 // 0000000065C0: D7600001 0001332D
	v_writelane_b32 v37, s5, 13                                // 0000000065C8: D7610025 00011A05
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000065D0: BF870113
	s_add_u32 s0, s0, s28                                      // 0000000065D4: 80001C00
	s_addc_u32 s1, s1, s29                                     // 0000000065D8: 82011D01
	s_and_b32 vcc_lo, exec_lo, s33                             // 0000000065DC: 8B6A217E
	v_writelane_b32 v37, s6, 14                                // 0000000065E0: D7610025 00011C06
	v_writelane_b32 v37, s7, 15                                // 0000000065E8: D7610025 00011E07
	v_writelane_b32 v37, s8, 16                                // 0000000065F0: D7610025 00012008
	v_writelane_b32 v37, s9, 17                                // 0000000065F8: D7610025 00012209
	v_writelane_b32 v37, s10, 18                               // 000000006600: D7610025 0001240A
	v_writelane_b32 v37, s11, 19                               // 000000006608: D7610025 0001260B
	s_cbranch_vccnz 26                                         // 000000006610: BFA4001A <r_3_3_3_8_8_8+0x507c>
	v_readlane_b32 s4, v45, 26                                 // 000000006614: D7600004 0001352D
	v_readlane_b32 s5, v45, 27                                 // 00000000661C: D7600005 0001372D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006624: BF870092
	s_add_u32 s4, s4, s28                                      // 000000006628: 80041C04
	s_addc_u32 s5, s5, s29                                     // 00000000662C: 82051D05
	s_load_b256 s[4:11], s[4:5], null                          // 000000006630: F40C0102 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000006638: BF89FC07
	v_writelane_b32 v37, s4, 12                                // 00000000663C: D7610025 00011804
	v_writelane_b32 v37, s5, 13                                // 000000006644: D7610025 00011A05
	v_writelane_b32 v37, s6, 14                                // 00000000664C: D7610025 00011C06
	v_writelane_b32 v37, s7, 15                                // 000000006654: D7610025 00011E07
	v_writelane_b32 v37, s8, 16                                // 00000000665C: D7610025 00012008
	v_writelane_b32 v37, s9, 17                                // 000000006664: D7610025 00012209
	v_writelane_b32 v37, s10, 18                               // 00000000666C: D7610025 0001240A
	v_writelane_b32 v37, s11, 19                               // 000000006674: D7610025 0001260B
	s_clause 0x1                                               // 00000000667C: BF850001
	s_load_b256 s[60:67], s[2:3], null                         // 000000006680: F40C0F01 F8000000
	s_load_b256 s[20:27], s[0:1], null                         // 000000006688: F40C0500 F8000000
	v_readlane_b32 s0, v45, 28                                 // 000000006690: D7600000 0001392D
	v_readlane_b32 s1, v45, 29                                 // 000000006698: D7600001 00013B2D
	v_readlane_b32 s2, v45, 30                                 // 0000000066A0: D7600002 00013D2D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000066A8: BF870113
	s_add_u32 s0, s0, s28                                      // 0000000066AC: 80001C00
	s_addc_u32 s1, s1, s29                                     // 0000000066B0: 82011D01
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000066B4: BF8700A1
	s_add_u32 s18, s2, s28                                     // 0000000066B8: 80121C02
	v_readlane_b32 s2, v45, 31                                 // 0000000066BC: D7600002 00013F2D
	s_addc_u32 s19, s2, s29                                    // 0000000066C4: 82131D02
	s_load_b512 s[0:15], s[0:1], null                          // 0000000066C8: F4100000 F8000000
	s_or_saveexec_b32 s105, -1                                 // 0000000066D0: BEE922C1
	v_mov_b32_e32 v45, v44                                     // 0000000066D4: 7E5A032C
	s_mov_b32 exec_lo, s105                                    // 0000000066D8: BEFE0069
	s_waitcnt lgkmcnt(0)                                       // 0000000066DC: BF89FC07
	v_writelane_b32 v45, s0, 28                                // 0000000066E0: D761002D 00013800
	v_writelane_b32 v43, s4, 0                                 // 0000000066E8: D761002B 00010004
	s_load_b256 s[84:91], s[18:19], 0x1100                     // 0000000066F0: F40C1509 F8001100
	s_mov_b32 s16, vcc_hi                                      // 0000000066F8: BE90006B
	v_writelane_b32 v45, s1, 29                                // 0000000066FC: D761002D 00013A01
	v_writelane_b32 v43, s5, 1                                 // 000000006704: D761002B 00010205
	v_writelane_b32 v45, s2, 30                                // 00000000670C: D761002D 00013C02
	v_writelane_b32 v43, s6, 2                                 // 000000006714: D761002B 00010406
	v_writelane_b32 v45, s3, 31                                // 00000000671C: D761002D 00013E03
	v_writelane_b32 v43, s7, 3                                 // 000000006724: D761002B 00010607
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000672C: BF870002
	v_readlane_b32 s52, v45, 20                                // 000000006730: D7600034 0001292D
	v_writelane_b32 v43, s8, 4                                 // 000000006738: D761002B 00010808
	s_waitcnt lgkmcnt(0)                                       // 000000006740: BF89FC07
	v_writelane_b32 v37, s84, 4                                // 000000006744: D7610025 00010854
	v_readlane_b32 s53, v45, 21                                // 00000000674C: D7600035 00012B2D
	v_readlane_b32 s54, v45, 22                                // 000000006754: D7600036 00012D2D
	v_readlane_b32 s55, v45, 23                                // 00000000675C: D7600037 00012F2D
	v_writelane_b32 v43, s9, 5                                 // 000000006764: D761002B 00010A09
	v_writelane_b32 v37, s85, 5                                // 00000000676C: D7610025 00010A55
	v_readlane_b32 s56, v45, 24                                // 000000006774: D7600038 0001312D
	v_readlane_b32 s57, v45, 25                                // 00000000677C: D7600039 0001332D
	v_readlane_b32 s58, v45, 26                                // 000000006784: D760003A 0001352D
	v_writelane_b32 v43, s10, 6                                // 00000000678C: D761002B 00010C0A
	v_writelane_b32 v37, s86, 6                                // 000000006794: D7610025 00010C56
	v_writelane_b32 v43, s11, 7                                // 00000000679C: D761002B 00010E0B
	v_writelane_b32 v37, s87, 7                                // 0000000067A4: D7610025 00010E57
	v_writelane_b32 v43, s12, 8                                // 0000000067AC: D761002B 0001100C
	v_writelane_b32 v37, s88, 8                                // 0000000067B4: D7610025 00011058
	v_writelane_b32 v43, s13, 9                                // 0000000067BC: D761002B 0001120D
	v_writelane_b32 v37, s89, 9                                // 0000000067C4: D7610025 00011259
	v_writelane_b32 v43, s14, 10                               // 0000000067CC: D761002B 0001140E
	v_writelane_b32 v37, s90, 10                               // 0000000067D4: D7610025 0001145A
	v_writelane_b32 v43, s15, 11                               // 0000000067DC: D761002B 0001160F
	s_add_u32 s0, s18, 0xfffffe40                              // 0000000067E4: 8000FF12 FFFFFE40
	s_addc_u32 s1, s19, -1                                     // 0000000067EC: 8201C113
	v_writelane_b32 v37, s91, 11                               // 0000000067F0: D7610025 0001165B
	s_load_b512 s[84:99], s[0:1], null                         // 0000000067F8: F4101500 F8000000
	s_add_u32 s2, s18, 0xfffffbe0                              // 000000006800: 8002FF12 FFFFFBE0
	s_addc_u32 s3, s19, -1                                     // 000000006808: 8203C113
	s_mov_b32 s1, 0                                            // 00000000680C: BE810080
	s_waitcnt lgkmcnt(0)                                       // 000000006810: BF89FC07
	v_writelane_b32 v43, s84, 20                               // 000000006814: D761002B 00012854
	v_writelane_b32 v37, s96, 0                                // 00000000681C: D7610025 00010060
	v_writelane_b32 v43, s85, 21                               // 000000006824: D761002B 00012A55
	v_writelane_b32 v37, s97, 1                                // 00000000682C: D7610025 00010261
	v_writelane_b32 v43, s86, 22                               // 000000006834: D761002B 00012C56
	v_writelane_b32 v37, s98, 2                                // 00000000683C: D7610025 00010462
	v_writelane_b32 v43, s87, 23                               // 000000006844: D761002B 00012E57
	v_writelane_b32 v37, s99, 3                                // 00000000684C: D7610025 00010663
	v_writelane_b32 v43, s88, 24                               // 000000006854: D761002B 00013058
	v_writelane_b32 v43, s89, 25                               // 00000000685C: D761002B 00013259
	v_writelane_b32 v43, s90, 26                               // 000000006864: D761002B 0001345A
	v_writelane_b32 v43, s91, 27                               // 00000000686C: D761002B 0001365B
	s_load_b256 s[84:91], s[2:3], null                         // 000000006874: F40C1501 F8000000
	v_writelane_b32 v43, s92, 28                               // 00000000687C: D761002B 0001385C
	v_writelane_b32 v43, s93, 29                               // 000000006884: D761002B 00013A5D
	v_writelane_b32 v43, s94, 30                               // 00000000688C: D761002B 00013C5E
	v_writelane_b32 v43, s95, 31                               // 000000006894: D761002B 00013E5F
	s_waitcnt lgkmcnt(0)                                       // 00000000689C: BF89FC07
	v_writelane_b32 v43, s84, 12                               // 0000000068A0: D761002B 00011854
	v_writelane_b32 v43, s85, 13                               // 0000000068A8: D761002B 00011A55
	v_writelane_b32 v43, s86, 14                               // 0000000068B0: D761002B 00011C56
	v_writelane_b32 v43, s87, 15                               // 0000000068B8: D761002B 00011E57
	v_writelane_b32 v43, s88, 16                               // 0000000068C0: D761002B 00012058
	v_writelane_b32 v43, s89, 17                               // 0000000068C8: D761002B 00012259
	v_writelane_b32 v43, s90, 18                               // 0000000068D0: D761002B 0001245A
	v_writelane_b32 v43, s91, 19                               // 0000000068D8: D761002B 0001265B
	s_mov_b32 s88, 0                                           // 0000000068E0: BED80080
	s_or_saveexec_b32 s105, -1                                 // 0000000068E4: BEE922C1
	scratch_store_b32 off, v45, off offset:80                  // 0000000068E8: DC690050 007C2D00
	s_mov_b32 exec_lo, s105                                    // 0000000068F0: BEFE0069
	v_readlane_b32 s59, v45, 27                                // 0000000068F4: D760003B 0001372D
	s_branch 118                                               // 0000000068FC: BFA00076 <r_3_3_3_8_8_8+0x54d8>
	s_mov_b32 s1, -1                                           // 000000006900: BE8100C1
	s_mov_b32 s16, 0                                           // 000000006904: BE900080
	v_writelane_b32 v44, s0, 28                                // 000000006908: D761002C 00013800
	v_writelane_b32 v44, s1, 29                                // 000000006910: D761002C 00013A01
	v_writelane_b32 v44, s2, 30                                // 000000006918: D761002C 00013C02
	v_writelane_b32 v44, s3, 31                                // 000000006920: D761002C 00013E03
	s_or_saveexec_b32 s105, -1                                 // 000000006928: BEE922C1
	scratch_store_b32 off, v44, off offset:80                  // 00000000692C: DC690050 007C2C00
	s_mov_b32 exec_lo, s105                                    // 000000006934: BEFE0069
	v_writelane_b32 v43, s4, 0                                 // 000000006938: D761002B 00010004
	v_writelane_b32 v43, s5, 1                                 // 000000006940: D761002B 00010205
	v_writelane_b32 v43, s6, 2                                 // 000000006948: D761002B 00010406
	v_writelane_b32 v43, s7, 3                                 // 000000006950: D761002B 00010607
	v_writelane_b32 v43, s8, 4                                 // 000000006958: D761002B 00010808
	v_writelane_b32 v43, s9, 5                                 // 000000006960: D761002B 00010A09
	v_writelane_b32 v43, s10, 6                                // 000000006968: D761002B 00010C0A
	v_writelane_b32 v43, s11, 7                                // 000000006970: D761002B 00010E0B
	v_writelane_b32 v43, s12, 8                                // 000000006978: D761002B 0001100C
	v_writelane_b32 v43, s13, 9                                // 000000006980: D761002B 0001120D
	v_writelane_b32 v43, s14, 10                               // 000000006988: D761002B 0001140E
	v_writelane_b32 v43, s15, 11                               // 000000006990: D761002B 0001160F
	v_writelane_b32 v43, s0, 12                                // 000000006998: D761002B 00011800
	v_writelane_b32 v43, s1, 13                                // 0000000069A0: D761002B 00011A01
	v_writelane_b32 v43, s2, 14                                // 0000000069A8: D761002B 00011C02
	v_writelane_b32 v43, s3, 15                                // 0000000069B0: D761002B 00011E03
	v_writelane_b32 v43, s4, 16                                // 0000000069B8: D761002B 00012004
	v_writelane_b32 v43, s5, 17                                // 0000000069C0: D761002B 00012205
	v_writelane_b32 v43, s6, 18                                // 0000000069C8: D761002B 00012406
	v_writelane_b32 v43, s7, 19                                // 0000000069D0: D761002B 00012607
	v_writelane_b32 v43, s0, 20                                // 0000000069D8: D761002B 00012800
	v_writelane_b32 v37, s12, 0                                // 0000000069E0: D7610025 0001000C
	v_writelane_b32 v43, s1, 21                                // 0000000069E8: D761002B 00012A01
	v_writelane_b32 v37, s13, 1                                // 0000000069F0: D7610025 0001020D
	v_writelane_b32 v43, s2, 22                                // 0000000069F8: D761002B 00012C02
	v_writelane_b32 v37, s14, 2                                // 000000006A00: D7610025 0001040E
	v_writelane_b32 v43, s3, 23                                // 000000006A08: D761002B 00012E03
	v_writelane_b32 v37, s15, 3                                // 000000006A10: D7610025 0001060F
	v_writelane_b32 v43, s4, 24                                // 000000006A18: D761002B 00013004
	v_writelane_b32 v43, s5, 25                                // 000000006A20: D761002B 00013205
	v_writelane_b32 v43, s6, 26                                // 000000006A28: D761002B 00013406
	v_writelane_b32 v43, s7, 27                                // 000000006A30: D761002B 00013607
	v_writelane_b32 v37, s0, 4                                 // 000000006A38: D7610025 00010800
	v_writelane_b32 v43, s8, 28                                // 000000006A40: D761002B 00013808
	v_writelane_b32 v37, s1, 5                                 // 000000006A48: D7610025 00010A01
	v_writelane_b32 v43, s9, 29                                // 000000006A50: D761002B 00013A09
	v_writelane_b32 v37, s2, 6                                 // 000000006A58: D7610025 00010C02
	v_writelane_b32 v43, s10, 30                               // 000000006A60: D761002B 00013C0A
	v_writelane_b32 v37, s3, 7                                 // 000000006A68: D7610025 00010E03
	v_writelane_b32 v43, s11, 31                               // 000000006A70: D761002B 00013E0B
	v_writelane_b32 v37, s4, 8                                 // 000000006A78: D7610025 00011004
	v_writelane_b32 v37, s5, 9                                 // 000000006A80: D7610025 00011205
	v_writelane_b32 v37, s6, 10                                // 000000006A88: D7610025 00011406
	v_writelane_b32 v37, s7, 11                                // 000000006A90: D7610025 00011607
	v_writelane_b32 v37, s4, 12                                // 000000006A98: D7610025 00011804
	v_writelane_b32 v37, s5, 13                                // 000000006AA0: D7610025 00011A05
	v_writelane_b32 v37, s6, 14                                // 000000006AA8: D7610025 00011C06
	v_writelane_b32 v37, s7, 15                                // 000000006AB0: D7610025 00011E07
	v_writelane_b32 v37, s8, 16                                // 000000006AB8: D7610025 00012008
	v_writelane_b32 v37, s9, 17                                // 000000006AC0: D7610025 00012209
	v_writelane_b32 v37, s10, 18                               // 000000006AC8: D7610025 0001240A
	v_writelane_b32 v37, s11, 19                               // 000000006AD0: D7610025 0001260B
	s_mov_b32 s13, s31                                         // 000000006AD8: BE8D001F
	v_writelane_b32 v37, s12, 20                               // 000000006ADC: D7610025 0001280C
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000006AE4: 8B6A017E
	v_writelane_b32 v37, s13, 21                               // 000000006AE8: D7610025 00012A0D
	s_cbranch_vccz 180                                         // 000000006AF0: BFA300B4 <r_3_3_3_8_8_8+0x57c4>
	s_or_saveexec_b32 s105, -1                                 // 000000006AF4: BEE922C1
	scratch_load_b32 v45, off, off offset:12                   // 000000006AF8: DC51000C 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000006B00: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000006B04: BF8903F7
	v_readlane_b32 s0, v45, 24                                 // 000000006B08: D7600000 0001312D
	v_readlane_b32 s1, v45, 25                                 // 000000006B10: D7600001 0001332D
	v_readlane_b32 s2, v45, 26                                 // 000000006B18: D7600002 0001352D
	s_mov_b32 s101, s100                                       // 000000006B20: BEE50064
	s_mov_b32 s102, s100                                       // 000000006B24: BEE60064
	s_add_u32 s0, s0, s28                                      // 000000006B28: 80001C00
	s_addc_u32 s1, s1, s29                                     // 000000006B2C: 82011D01
	s_add_u32 s2, s2, s28                                      // 000000006B30: 80021C02
	s_load_b256 s[20:27], s[0:1], null                         // 000000006B34: F40C0500 F8000000
	v_readlane_b32 s0, v45, 27                                 // 000000006B3C: D7600000 0001372D
	s_mov_b32 s103, s100                                       // 000000006B44: BEE70064
	s_mov_b64 s[64:65], s[100:101]                             // 000000006B48: BEC00164
	s_mov_b64 s[60:61], s[100:101]                             // 000000006B4C: BEBC0164
	s_mov_b32 s16, -1                                          // 000000006B50: BE9000C1
	s_addc_u32 s3, s0, s29                                     // 000000006B54: 82031D00
	v_readlane_b32 s0, v45, 30                                 // 000000006B58: D7600000 00013D2D
	s_mov_b64 s[88:89], s[100:101]                             // 000000006B60: BED80164
	s_mov_b64 s[66:67], s[102:103]                             // 000000006B64: BEC20166
	s_mov_b64 s[62:63], s[102:103]                             // 000000006B68: BEBE0166
	s_mov_b64 s[96:97], s[100:101]                             // 000000006B6C: BEE00164
	s_add_u32 s18, s0, s28                                     // 000000006B70: 80121C00
	v_readlane_b32 s0, v45, 31                                 // 000000006B74: D7600000 00013F2D
	s_mov_b64 s[8:9], s[100:101]                               // 000000006B7C: BE880164
	s_mov_b64 s[84:85], s[100:101]                             // 000000006B80: BED40164
	s_mov_b64 s[92:93], s[100:101]                             // 000000006B84: BEDC0164
	s_mov_b64 s[98:99], s[102:103]                             // 000000006B88: BEE20166
	s_addc_u32 s19, s0, s29                                    // 000000006B8C: 82131D00
	s_clause 0x1                                               // 000000006B90: BF850001
	s_load_b256 s[0:7], s[2:3], null                           // 000000006B94: F40C0001 F8000000
	s_load_b256 s[52:59], s[18:19], 0x1100                     // 000000006B9C: F40C0D09 F8001100
	s_mov_b64 s[10:11], s[102:103]                             // 000000006BA4: BE8A0166
	s_mov_b64 s[90:91], s[102:103]                             // 000000006BA8: BEDA0166
	s_mov_b64 s[86:87], s[102:103]                             // 000000006BAC: BED60166
	s_mov_b64 s[94:95], s[102:103]                             // 000000006BB0: BEDE0166
	s_waitcnt lgkmcnt(0)                                       // 000000006BB4: BF89FC07
	v_writelane_b32 v37, s0, 12                                // 000000006BB8: D7610025 00011800
	v_writelane_b32 v37, s1, 13                                // 000000006BC0: D7610025 00011A01
	v_writelane_b32 v37, s2, 14                                // 000000006BC8: D7610025 00011C02
	v_writelane_b32 v37, s3, 15                                // 000000006BD0: D7610025 00011E03
	v_writelane_b32 v37, s4, 16                                // 000000006BD8: D7610025 00012004
	v_writelane_b32 v37, s5, 17                                // 000000006BE0: D7610025 00012205
	v_writelane_b32 v37, s6, 18                                // 000000006BE8: D7610025 00012406
	v_writelane_b32 v37, s7, 19                                // 000000006BF0: D7610025 00012607
	s_mov_b64 s[0:1], s[100:101]                               // 000000006BF8: BE800164
	s_mov_b64 s[4:5], s[100:101]                               // 000000006BFC: BE840164
	s_mov_b64 s[2:3], s[102:103]                               // 000000006C00: BE820166
	s_mov_b64 s[6:7], s[102:103]                               // 000000006C04: BE860166
	v_writelane_b32 v37, s52, 4                                // 000000006C08: D7610025 00010834
	v_writelane_b32 v37, s53, 5                                // 000000006C10: D7610025 00010A35
	v_writelane_b32 v37, s54, 6                                // 000000006C18: D7610025 00010C36
	v_writelane_b32 v37, s55, 7                                // 000000006C20: D7610025 00010E37
	v_writelane_b32 v37, s56, 8                                // 000000006C28: D7610025 00011038
	v_writelane_b32 v37, s57, 9                                // 000000006C30: D7610025 00011239
	v_writelane_b32 v37, s58, 10                               // 000000006C38: D7610025 0001143A
	v_writelane_b32 v37, s59, 11                               // 000000006C40: D7610025 0001163B
	s_mov_b64 s[56:57], s[100:101]                             // 000000006C48: BEB80164
	s_mov_b64 s[52:53], s[100:101]                             // 000000006C4C: BEB40164
	s_mov_b64 s[58:59], s[102:103]                             // 000000006C50: BEBA0166
	s_mov_b64 s[54:55], s[102:103]                             // 000000006C54: BEB60166
	v_writelane_b32 v43, s52, 12                               // 000000006C58: D761002B 00011834
	v_writelane_b32 v43, s53, 13                               // 000000006C60: D761002B 00011A35
	v_writelane_b32 v43, s54, 14                               // 000000006C68: D761002B 00011C36
	v_writelane_b32 v43, s55, 15                               // 000000006C70: D761002B 00011E37
	v_writelane_b32 v43, s56, 16                               // 000000006C78: D761002B 00012038
	v_writelane_b32 v43, s57, 17                               // 000000006C80: D761002B 00012239
	v_writelane_b32 v43, s58, 18                               // 000000006C88: D761002B 0001243A
	v_writelane_b32 v43, s59, 19                               // 000000006C90: D761002B 0001263B
	s_or_saveexec_b32 s105, -1                                 // 000000006C98: BEE922C1
	scratch_load_b32 v45, off, off offset:80                   // 000000006C9C: DC510050 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 000000006CA4: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000006CA8: BF8903F7
	v_writelane_b32 v45, s84, 28                               // 000000006CAC: D761002D 00013854
	v_writelane_b32 v45, s85, 29                               // 000000006CB4: D761002D 00013A55
	v_writelane_b32 v45, s86, 30                               // 000000006CBC: D761002D 00013C56
	v_writelane_b32 v45, s87, 31                               // 000000006CC4: D761002D 00013E57
	s_or_saveexec_b32 s105, -1                                 // 000000006CCC: BEE922C1
	scratch_store_b32 off, v45, off offset:80                  // 000000006CD0: DC690050 007C2D00
	s_mov_b32 exec_lo, s105                                    // 000000006CD8: BEFE0069
	v_writelane_b32 v43, s88, 0                                // 000000006CDC: D761002B 00010058
	s_mov_b64 s[12:13], s[100:101]                             // 000000006CE4: BE8C0164
	s_mov_b64 s[14:15], s[102:103]                             // 000000006CE8: BE8E0166
	v_writelane_b32 v43, s89, 1                                // 000000006CEC: D761002B 00010259
	v_writelane_b32 v43, s90, 2                                // 000000006CF4: D761002B 0001045A
	v_writelane_b32 v43, s91, 3                                // 000000006CFC: D761002B 0001065B
	v_writelane_b32 v43, s92, 4                                // 000000006D04: D761002B 0001085C
	v_writelane_b32 v43, s93, 5                                // 000000006D0C: D761002B 00010A5D
	v_writelane_b32 v43, s94, 6                                // 000000006D14: D761002B 00010C5E
	v_writelane_b32 v43, s95, 7                                // 000000006D1C: D761002B 00010E5F
	v_writelane_b32 v43, s96, 8                                // 000000006D24: D761002B 00011060
	v_writelane_b32 v43, s97, 9                                // 000000006D2C: D761002B 00011261
	v_writelane_b32 v43, s98, 10                               // 000000006D34: D761002B 00011462
	v_writelane_b32 v43, s99, 11                               // 000000006D3C: D761002B 00011663
	v_writelane_b32 v43, s0, 20                                // 000000006D44: D761002B 00012800
	v_writelane_b32 v37, s12, 0                                // 000000006D4C: D7610025 0001000C
	v_writelane_b32 v43, s1, 21                                // 000000006D54: D761002B 00012A01
	v_writelane_b32 v37, s13, 1                                // 000000006D5C: D7610025 0001020D
	v_writelane_b32 v43, s2, 22                                // 000000006D64: D761002B 00012C02
	v_writelane_b32 v37, s14, 2                                // 000000006D6C: D7610025 0001040E
	v_writelane_b32 v43, s3, 23                                // 000000006D74: D761002B 00012E03
	v_writelane_b32 v37, s15, 3                                // 000000006D7C: D7610025 0001060F
	v_writelane_b32 v43, s4, 24                                // 000000006D84: D761002B 00013004
	v_writelane_b32 v43, s5, 25                                // 000000006D8C: D761002B 00013205
	v_writelane_b32 v43, s6, 26                                // 000000006D94: D761002B 00013406
	v_writelane_b32 v43, s7, 27                                // 000000006D9C: D761002B 00013607
	v_writelane_b32 v43, s8, 28                                // 000000006DA4: D761002B 00013808
	v_writelane_b32 v43, s9, 29                                // 000000006DAC: D761002B 00013A09
	v_writelane_b32 v43, s10, 30                               // 000000006DB4: D761002B 00013C0A
	v_writelane_b32 v43, s11, 31                               // 000000006DBC: D761002B 00013E0B
	v_writelane_b32 v37, s60, 22                               // 000000006DC4: D7610025 00012C3C
	v_writelane_b32 v37, s61, 23                               // 000000006DCC: D7610025 00012E3D
	v_writelane_b32 v37, s62, 24                               // 000000006DD4: D7610025 0001303E
	v_writelane_b32 v37, s63, 25                               // 000000006DDC: D7610025 0001323F
	v_writelane_b32 v37, s64, 26                               // 000000006DE4: D7610025 00013440
	v_writelane_b32 v37, s65, 27                               // 000000006DEC: D7610025 00013641
	v_writelane_b32 v37, s66, 28                               // 000000006DF4: D7610025 00013842
	v_writelane_b32 v37, s67, 29                               // 000000006DFC: D7610025 00013A43
	s_or_saveexec_b32 s105, -1                                 // 000000006E04: BEE922C1
	scratch_store_b32 off, v41, off offset:92                  // 000000006E08: DC69005C 007C2900
	s_mov_b32 exec_lo, s105                                    // 000000006E10: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000006E14: BEE922C1
	v_mov_b32_e32 v41, v38                                     // 000000006E18: 7E520326
	s_mov_b32 exec_lo, s105                                    // 000000006E1C: BEFE0069
	s_mov_b32 s0, s88                                          // 000000006E20: BE800058
	s_mov_b32 s8, s88                                          // 000000006E24: BE880058
	s_mov_b32 s9, s88                                          // 000000006E28: BE890058
	s_mov_b32 s10, s88                                         // 000000006E2C: BE8A0058
	s_mov_b32 s11, s88                                         // 000000006E30: BE8B0058
	s_mov_b32 s1, s88                                          // 000000006E34: BE810058
	s_mov_b32 s2, s88                                          // 000000006E38: BE820058
	s_mov_b32 s3, s88                                          // 000000006E3C: BE830058
	s_mov_b32 s4, s88                                          // 000000006E40: BE840058
	s_mov_b32 s5, s88                                          // 000000006E44: BE850058
	s_mov_b32 s6, s88                                          // 000000006E48: BE860058
	s_mov_b32 s7, s88                                          // 000000006E4C: BE870058
	s_mov_b32 s12, s88                                         // 000000006E50: BE8C0058
	s_mov_b32 s13, s88                                         // 000000006E54: BE8D0058
	s_mov_b32 s14, s88                                         // 000000006E58: BE8E0058
	s_mov_b32 s15, s88                                         // 000000006E5C: BE8F0058
	v_writelane_b32 v37, s0, 30                                // 000000006E60: D7610025 00013C00
	v_writelane_b32 v36, s2, 0                                 // 000000006E68: D7610024 00010002
	s_mov_b32 s89, s88                                         // 000000006E70: BED90058
	s_mov_b32 s90, s88                                         // 000000006E74: BEDA0058
	s_mov_b32 s91, s88                                         // 000000006E78: BEDB0058
	s_mov_b32 s60, s88                                         // 000000006E7C: BEBC0058
	v_writelane_b32 v36, s3, 1                                 // 000000006E80: D7610024 00010203
	s_and_not1_b32 vcc_lo, exec_lo, s16                        // 000000006E88: 916A107E
	s_mov_b32 s61, s88                                         // 000000006E8C: BEBD0058
	s_mov_b32 s62, s88                                         // 000000006E90: BEBE0058
	s_mov_b32 s63, s88                                         // 000000006E94: BEBF0058
	v_writelane_b32 v36, s4, 2                                 // 000000006E98: D7610024 00010404
	s_mov_b32 s52, s88                                         // 000000006EA0: BEB40058
	s_mov_b32 s53, s88                                         // 000000006EA4: BEB50058
	s_mov_b32 s54, s88                                         // 000000006EA8: BEB60058
	s_mov_b32 s55, s88                                         // 000000006EAC: BEB70058
	v_writelane_b32 v36, s5, 3                                 // 000000006EB0: D7610024 00010605
	s_mov_b32 s56, s88                                         // 000000006EB8: BEB80058
	s_mov_b32 s57, s88                                         // 000000006EBC: BEB90058
	s_mov_b32 s58, s88                                         // 000000006EC0: BEBA0058
	s_mov_b32 s59, s88                                         // 000000006EC4: BEBB0058
	v_writelane_b32 v36, s6, 4                                 // 000000006EC8: D7610024 00010806
	s_mov_b32 s84, s88                                         // 000000006ED0: BED40058
	s_mov_b32 s85, s88                                         // 000000006ED4: BED50058
	s_mov_b32 s86, s88                                         // 000000006ED8: BED60058
	s_mov_b32 s87, s88                                         // 000000006EDC: BED70058
	v_writelane_b32 v36, s7, 5                                 // 000000006EE0: D7610024 00010A07
	s_mov_b32 s64, s88                                         // 000000006EE8: BEC00058
	s_mov_b32 s65, s88                                         // 000000006EEC: BEC10058
	s_mov_b32 s66, s88                                         // 000000006EF0: BEC20058
	s_mov_b32 s67, s88                                         // 000000006EF4: BEC30058
	v_writelane_b32 v36, s8, 6                                 // 000000006EF8: D7610024 00010C08
	v_writelane_b32 v37, s1, 31                                // 000000006F00: D7610025 00013E01
	v_writelane_b32 v36, s9, 7                                 // 000000006F08: D7610024 00010E09
	v_writelane_b32 v36, s10, 8                                // 000000006F10: D7610024 0001100A
	v_writelane_b32 v36, s11, 9                                // 000000006F18: D7610024 0001120B
	v_writelane_b32 v36, s12, 10                               // 000000006F20: D7610024 0001140C
	v_writelane_b32 v36, s13, 11                               // 000000006F28: D7610024 0001160D
	v_writelane_b32 v36, s14, 12                               // 000000006F30: D7610024 0001180E
	v_writelane_b32 v36, s15, 13                               // 000000006F38: D7610024 00011A0F
	s_cbranch_vccnz 40                                         // 000000006F40: BFA40028 <r_3_3_3_8_8_8+0x59e4>
	s_clause 0x2                                               // 000000006F44: BF850002
	s_load_b512 s[52:67], s[18:19], 0x2500                     // 000000006F48: F4100D09 F8002500
	s_load_b256 s[84:91], s[18:19], 0x23e0                     // 000000006F50: F40C1509 F80023E0
	s_load_b512 s[0:15], s[18:19], 0x2640                      // 000000006F58: F4100009 F8002640
	s_waitcnt lgkmcnt(0)                                       // 000000006F60: BF89FC07
	v_writelane_b32 v37, s0, 30                                // 000000006F64: D7610025 00013C00
	v_writelane_b32 v36, s2, 0                                 // 000000006F6C: D7610024 00010002
	v_writelane_b32 v37, s1, 31                                // 000000006F74: D7610025 00013E01
	v_writelane_b32 v36, s3, 1                                 // 000000006F7C: D7610024 00010203
	v_writelane_b32 v36, s4, 2                                 // 000000006F84: D7610024 00010404
	v_writelane_b32 v36, s5, 3                                 // 000000006F8C: D7610024 00010605
	v_writelane_b32 v36, s6, 4                                 // 000000006F94: D7610024 00010806
	v_writelane_b32 v36, s7, 5                                 // 000000006F9C: D7610024 00010A07
	v_writelane_b32 v36, s8, 6                                 // 000000006FA4: D7610024 00010C08
	v_writelane_b32 v36, s9, 7                                 // 000000006FAC: D7610024 00010E09
	v_writelane_b32 v36, s10, 8                                // 000000006FB4: D7610024 0001100A
	v_writelane_b32 v36, s11, 9                                // 000000006FBC: D7610024 0001120B
	v_writelane_b32 v36, s12, 10                               // 000000006FC4: D7610024 0001140C
	v_writelane_b32 v36, s13, 11                               // 000000006FCC: D7610024 0001160D
	v_writelane_b32 v36, s14, 12                               // 000000006FD4: D7610024 0001180E
	v_writelane_b32 v36, s15, 13                               // 000000006FDC: D7610024 00011A0F
	s_load_b256 s[0:7], s[18:19], 0x1120                       // 000000006FE4: F40C0009 F8001120
	s_and_b32 vcc_lo, exec_lo, s34                             // 000000006FEC: 8B6A227E
	s_waitcnt lgkmcnt(0)                                       // 000000006FF0: BF89FC07
	v_writelane_b32 v36, s0, 14                                // 000000006FF4: D7610024 00011C00
	v_writelane_b32 v36, s1, 15                                // 000000006FFC: D7610024 00011E01
	v_writelane_b32 v36, s2, 16                                // 000000007004: D7610024 00012002
	v_writelane_b32 v36, s3, 17                                // 00000000700C: D7610024 00012203
	v_writelane_b32 v36, s4, 18                                // 000000007014: D7610024 00012404
	v_writelane_b32 v36, s5, 19                                // 00000000701C: D7610024 00012605
	v_writelane_b32 v36, s6, 20                                // 000000007024: D7610024 00012806
	v_writelane_b32 v36, s7, 21                                // 00000000702C: D7610024 00012A07
	s_load_b256 s[0:7], s[18:19], 0xfe0                        // 000000007034: F40C0009 F8000FE0
	s_waitcnt lgkmcnt(0)                                       // 00000000703C: BF89FC07
	v_writelane_b32 v36, s0, 22                                // 000000007040: D7610024 00012C00
	v_writelane_b32 v36, s1, 23                                // 000000007048: D7610024 00012E01
	v_writelane_b32 v36, s2, 24                                // 000000007050: D7610024 00013002
	v_writelane_b32 v36, s3, 25                                // 000000007058: D7610024 00013203
	v_writelane_b32 v36, s4, 26                                // 000000007060: D7610024 00013404
	v_writelane_b32 v36, s5, 27                                // 000000007068: D7610024 00013605
	v_writelane_b32 v36, s6, 28                                // 000000007070: D7610024 00013806
	v_writelane_b32 v36, s7, 29                                // 000000007078: D7610024 00013A07
	s_load_b512 s[0:15], s[18:19], 0x1240                      // 000000007080: F4100009 F8001240
	s_waitcnt lgkmcnt(0)                                       // 000000007088: BF89FC07
	v_writelane_b32 v36, s0, 30                                // 00000000708C: D7610024 00013C00
	v_writelane_b32 v38, s2, 0                                 // 000000007094: D7610026 00010002
	v_writelane_b32 v36, s1, 31                                // 00000000709C: D7610024 00013E01
	v_writelane_b32 v38, s3, 1                                 // 0000000070A4: D7610026 00010203
	v_writelane_b32 v38, s4, 2                                 // 0000000070AC: D7610026 00010404
	v_writelane_b32 v38, s5, 3                                 // 0000000070B4: D7610026 00010605
	v_writelane_b32 v38, s6, 4                                 // 0000000070BC: D7610026 00010806
	v_writelane_b32 v38, s7, 5                                 // 0000000070C4: D7610026 00010A07
	v_writelane_b32 v38, s8, 6                                 // 0000000070CC: D7610026 00010C08
	v_writelane_b32 v38, s9, 7                                 // 0000000070D4: D7610026 00010E09
	v_writelane_b32 v38, s10, 8                                // 0000000070DC: D7610026 0001100A
	v_writelane_b32 v38, s11, 9                                // 0000000070E4: D7610026 0001120B
	v_writelane_b32 v38, s12, 10                               // 0000000070EC: D7610026 0001140C
	v_writelane_b32 v38, s13, 11                               // 0000000070F4: D7610026 0001160D
	v_writelane_b32 v38, s14, 12                               // 0000000070FC: D7610026 0001180E
	v_writelane_b32 v38, s15, 13                               // 000000007104: D7610026 00011A0F
	v_writelane_b32 v38, s20, 14                               // 00000000710C: D7610026 00011C14
	v_writelane_b32 v38, s21, 15                               // 000000007114: D7610026 00011E15
	v_writelane_b32 v38, s22, 16                               // 00000000711C: D7610026 00012016
	v_writelane_b32 v38, s23, 17                               // 000000007124: D7610026 00012217
	v_writelane_b32 v38, s24, 18                               // 00000000712C: D7610026 00012418
	v_writelane_b32 v38, s25, 19                               // 000000007134: D7610026 00012619
	v_writelane_b32 v38, s26, 20                               // 00000000713C: D7610026 0001281A
	v_writelane_b32 v38, s27, 21                               // 000000007144: D7610026 00012A1B
	s_cbranch_vccnz 4502                                       // 00000000714C: BFA41196 <r_3_3_3_8_8_8+0xa1a8>
	s_or_saveexec_b32 s105, -1                                 // 000000007150: BEE922C1
	scratch_load_b32 v44, off, off                             // 000000007154: DC510000 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000715C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000007160: BF8903F7
	v_readlane_b32 s18, v44, 26                                // 000000007164: D7600012 0001352C
	s_or_saveexec_b32 s105, -1                                 // 00000000716C: BEE922C1
	scratch_store_b32 off, v42, off offset:36                  // 000000007170: DC690024 007C2A00
	s_mov_b32 exec_lo, s105                                    // 000000007178: BEFE0069
	s_add_u32 s20, s18, s28                                    // 00000000717C: 80141C12
	v_readlane_b32 s18, v44, 27                                // 000000007180: D7600012 0001372C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007188: BF870001
	s_addc_u32 s21, s18, s29                                   // 00000000718C: 82151D12
	s_or_saveexec_b32 s105, -1                                 // 000000007190: BEE922C1
	scratch_load_b32 v44, off, off offset:8                    // 000000007194: DC510008 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000719C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 0000000071A0: BF8903F7
	v_readlane_b32 s18, v44, 0                                 // 0000000071A4: D7600012 0001012C
	v_readlane_b32 s19, v44, 1                                 // 0000000071AC: D7600013 0001032C
	s_mov_b32 s101, s100                                       // 0000000071B4: BEE50064
	s_mov_b32 s102, s100                                       // 0000000071B8: BEE60064
	s_mov_b32 s103, s100                                       // 0000000071BC: BEE70064
	s_add_u32 s18, s18, s28                                    // 0000000071C0: 80121C12
	s_mov_b64 s[4:5], s[100:101]                               // 0000000071C4: BE840164
	s_mov_b64 s[0:1], s[100:101]                               // 0000000071C8: BE800164
	s_addc_u32 s19, s19, s29                                   // 0000000071CC: 82131D13
	s_and_b32 vcc_lo, exec_lo, s33                             // 0000000071D0: 8B6A217E
	s_mov_b64 s[6:7], s[102:103]                               // 0000000071D4: BE860166
	s_mov_b64 s[2:3], s[102:103]                               // 0000000071D8: BE820166
	s_cbranch_vccnz 9                                          // 0000000071DC: BFA40009 <r_3_3_3_8_8_8+0x5c04>
	v_readlane_b32 s23, v44, 2                                 // 0000000071E0: D7600017 0001052C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000071E8: BF8700A1
	s_add_u32 s24, s23, s28                                    // 0000000071EC: 80181C17
	v_readlane_b32 s23, v44, 3                                 // 0000000071F0: D7600017 0001072C
	s_addc_u32 s25, s23, s29                                   // 0000000071F8: 82191D17
	s_load_b256 s[0:7], s[24:25], null                         // 0000000071FC: F40C000C F8000000
	s_load_b256 s[8:15], s[20:21], null                        // 000000007204: F40C020A F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000720C: BF89FC07
	v_writelane_b32 v45, s8, 30                                // 000000007210: D761002D 00013C08
	v_writelane_b32 v42, s10, 0                                // 000000007218: D761002A 0001000A
	v_writelane_b32 v45, s9, 31                                // 000000007220: D761002D 00013E09
	v_writelane_b32 v42, s11, 1                                // 000000007228: D761002A 0001020B
	v_writelane_b32 v42, s12, 2                                // 000000007230: D761002A 0001040C
	v_writelane_b32 v42, s13, 3                                // 000000007238: D761002A 0001060D
	v_writelane_b32 v42, s14, 4                                // 000000007240: D761002A 0001080E
	v_writelane_b32 v42, s15, 5                                // 000000007248: D761002A 00010A0F
	s_or_saveexec_b32 s105, -1                                 // 000000007250: BEE922C1
	scratch_store_b32 off, v42, off offset:32                  // 000000007254: DC690020 007C2A00
	s_mov_b32 exec_lo, s105                                    // 00000000725C: BEFE0069
	s_load_b256 s[8:15], s[18:19], null                        // 000000007260: F40C0209 F8000000
	v_readlane_b32 s20, v44, 4                                 // 000000007268: D7600014 0001092C
	v_readlane_b32 s21, v44, 5                                 // 000000007270: D7600015 00010B2C
	s_mov_b32 s96, 0                                           // 000000007278: BEE00080
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000727C: BF870092
	s_add_u32 s20, s20, s28                                    // 000000007280: 80141C14
	s_addc_u32 s21, s21, s29                                   // 000000007284: 82151D15
	s_waitcnt lgkmcnt(0)                                       // 000000007288: BF89FC07
	v_writelane_b32 v45, s8, 22                                // 00000000728C: D761002D 00012C08
	v_writelane_b32 v45, s9, 23                                // 000000007294: D761002D 00012E09
	v_writelane_b32 v45, s10, 24                               // 00000000729C: D761002D 0001300A
	v_writelane_b32 v45, s11, 25                               // 0000000072A4: D761002D 0001320B
	v_writelane_b32 v45, s12, 26                               // 0000000072AC: D761002D 0001340C
	v_writelane_b32 v45, s13, 27                               // 0000000072B4: D761002D 0001360D
	v_writelane_b32 v45, s14, 28                               // 0000000072BC: D761002D 0001380E
	v_writelane_b32 v45, s15, 29                               // 0000000072C4: D761002D 00013A0F
	s_load_b512 s[8:23], s[20:21], null                        // 0000000072CC: F410020A F8000000
	s_waitcnt lgkmcnt(0)                                       // 0000000072D4: BF89FC07
	v_writelane_b32 v38, s8, 22                                // 0000000072D8: D7610026 00012C08
	v_writelane_b32 v45, s18, 0                                // 0000000072E0: D761002D 00010012
	v_writelane_b32 v38, s9, 23                                // 0000000072E8: D7610026 00012E09
	v_writelane_b32 v45, s19, 1                                // 0000000072F0: D761002D 00010213
	v_writelane_b32 v38, s10, 24                               // 0000000072F8: D7610026 0001300A
	v_writelane_b32 v45, s20, 2                                // 000000007300: D761002D 00010414
	v_writelane_b32 v38, s11, 25                               // 000000007308: D7610026 0001320B
	v_writelane_b32 v45, s21, 3                                // 000000007310: D761002D 00010615
	v_writelane_b32 v38, s12, 26                               // 000000007318: D7610026 0001340C
	v_writelane_b32 v45, s22, 4                                // 000000007320: D761002D 00010816
	v_writelane_b32 v38, s13, 27                               // 000000007328: D7610026 0001360D
	v_writelane_b32 v45, s23, 5                                // 000000007330: D761002D 00010A17
	v_readlane_b32 s18, v44, 6                                 // 000000007338: D7600012 00010D2C
	v_writelane_b32 v38, s14, 28                               // 000000007340: D7610026 0001380E
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000007348: BF870132
	s_add_u32 s34, s18, s28                                    // 00000000734C: 80221C12
	v_readlane_b32 s18, v44, 11                                // 000000007350: D7600012 0001172C
	v_writelane_b32 v38, s15, 29                               // 000000007358: D7610026 00013A0F
	s_addc_u32 s35, s18, s29                                   // 000000007360: 82231D12
	s_add_u32 s18, s34, 0xfffffbe0                             // 000000007364: 8012FF22 FFFFFBE0
	s_load_b256 s[8:15], s[34:35], 0x1100                      // 00000000736C: F40C0211 F8001100
	s_addc_u32 s19, s35, -1                                    // 000000007374: 8213C123
	v_writelane_b32 v38, s16, 30                               // 000000007378: D7610026 00013C10
	v_writelane_b32 v38, s17, 31                               // 000000007380: D7610026 00013E11
	s_waitcnt lgkmcnt(0)                                       // 000000007388: BF89FC07
	v_writelane_b32 v45, s8, 14                                // 00000000738C: D761002D 00011C08
	v_writelane_b32 v45, s9, 15                                // 000000007394: D761002D 00011E09
	v_writelane_b32 v45, s10, 16                               // 00000000739C: D761002D 0001200A
	v_writelane_b32 v45, s11, 17                               // 0000000073A4: D761002D 0001220B
	v_writelane_b32 v45, s12, 18                               // 0000000073AC: D761002D 0001240C
	v_writelane_b32 v45, s13, 19                               // 0000000073B4: D761002D 0001260D
	v_writelane_b32 v45, s14, 20                               // 0000000073BC: D761002D 0001280E
	v_writelane_b32 v45, s15, 21                               // 0000000073C4: D761002D 00012A0F
	s_load_b256 s[8:15], s[18:19], null                        // 0000000073CC: F40C0209 F8000000
	s_waitcnt lgkmcnt(0)                                       // 0000000073D4: BF89FC07
	v_writelane_b32 v45, s8, 6                                 // 0000000073D8: D761002D 00010C08
	v_writelane_b32 v45, s9, 7                                 // 0000000073E0: D761002D 00010E09
	v_writelane_b32 v45, s10, 8                                // 0000000073E8: D761002D 0001100A
	v_writelane_b32 v45, s11, 9                                // 0000000073F0: D761002D 0001120B
	v_writelane_b32 v45, s12, 10                               // 0000000073F8: D761002D 0001140C
	v_writelane_b32 v45, s13, 11                               // 000000007400: D761002D 0001160D
	v_writelane_b32 v45, s14, 12                               // 000000007408: D761002D 0001180E
	v_writelane_b32 v45, s15, 13                               // 000000007410: D761002D 00011A0F
	s_or_saveexec_b32 s105, -1                                 // 000000007418: BEE922C1
	scratch_load_b32 v42, off, off offset:36                   // 00000000741C: DC510024 2A7C0000
	s_mov_b32 exec_lo, s105                                    // 000000007424: BEFE0069
	s_branch 158                                               // 000000007428: BFA0009E <r_3_3_3_8_8_8+0x60a4>
	s_or_saveexec_b32 s105, -1                                 // 00000000742C: BEE922C1
	scratch_load_b32 v44, off, off offset:8                    // 000000007430: DC510008 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 000000007438: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000743C: BF8903F7
	v_readlane_b32 s18, v44, 0                                 // 000000007440: D7600012 0001012C
	v_readlane_b32 s19, v44, 1                                 // 000000007448: D7600013 0001032C
	v_readlane_b32 s20, v44, 2                                 // 000000007450: D7600014 0001052C
	v_readlane_b32 s16, v44, 11                                // 000000007458: D7600010 0001172C
	s_mov_b32 s101, s100                                       // 000000007460: BEE50064
	s_add_u32 s18, s18, s28                                    // 000000007464: 80121C12
	s_addc_u32 s19, s19, s29                                   // 000000007468: 82131D13
	s_add_u32 s20, s20, s28                                    // 00000000746C: 80141C14
	s_load_b256 s[0:7], s[18:19], null                         // 000000007470: F40C0009 F8000000
	v_readlane_b32 s18, v44, 3                                 // 000000007478: D7600012 0001072C
	s_mov_b32 s102, s100                                       // 000000007480: BEE60064
	s_mov_b32 s103, s100                                       // 000000007484: BEE70064
	s_mov_b64 s[24:25], s[100:101]                             // 000000007488: BE980164
	s_mov_b64 s[26:27], s[102:103]                             // 00000000748C: BE9A0166
	s_addc_u32 s21, s18, s29                                   // 000000007490: 82151D12
	v_readlane_b32 s18, v44, 6                                 // 000000007494: D7600012 00010D2C
	s_mov_b32 vcc_hi, -1                                       // 00000000749C: BEEB00C1
	s_mov_b64 s[96:97], s[100:101]                             // 0000000074A0: BEE00164
	s_mov_b64 s[98:99], s[102:103]                             // 0000000074A4: BEE20166
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000074A8: BF870001
	s_add_u32 s34, s18, s28                                    // 0000000074AC: 80221C12
	s_addc_u32 s35, s16, s29                                   // 0000000074B0: 82231D10
	s_mov_b64 s[28:29], s[100:101]                             // 0000000074B4: BE9C0164
	s_mov_b64 s[30:31], s[102:103]                             // 0000000074B8: BE9E0166
	s_waitcnt lgkmcnt(0)                                       // 0000000074BC: BF89FC07
	v_writelane_b32 v45, s0, 22                                // 0000000074C0: D761002D 00012C00
	v_writelane_b32 v45, s1, 23                                // 0000000074C8: D761002D 00012E01
	v_writelane_b32 v45, s2, 24                                // 0000000074D0: D761002D 00013002
	v_writelane_b32 v45, s3, 25                                // 0000000074D8: D761002D 00013203
	v_writelane_b32 v45, s4, 26                                // 0000000074E0: D761002D 00013404
	v_writelane_b32 v45, s5, 27                                // 0000000074E8: D761002D 00013605
	v_writelane_b32 v45, s6, 28                                // 0000000074F0: D761002D 00013806
	v_writelane_b32 v45, s7, 29                                // 0000000074F8: D761002D 00013A07
	s_clause 0x1                                               // 000000007500: BF850001
	s_load_b256 s[0:7], s[20:21], null                         // 000000007504: F40C000A F8000000
	s_load_b256 s[8:15], s[34:35], 0x1100                      // 00000000750C: F40C0211 F8001100
	s_mov_b64 s[20:21], s[100:101]                             // 000000007514: BE940164
	s_mov_b64 s[22:23], s[102:103]                             // 000000007518: BE960166
	s_waitcnt lgkmcnt(0)                                       // 00000000751C: BF89FC07
	v_writelane_b32 v45, s8, 14                                // 000000007520: D761002D 00011C08
	v_writelane_b32 v45, s9, 15                                // 000000007528: D761002D 00011E09
	v_writelane_b32 v45, s10, 16                               // 000000007530: D761002D 0001200A
	v_writelane_b32 v45, s11, 17                               // 000000007538: D761002D 0001220B
	v_writelane_b32 v45, s12, 18                               // 000000007540: D761002D 0001240C
	v_writelane_b32 v45, s13, 19                               // 000000007548: D761002D 0001260D
	v_writelane_b32 v45, s14, 20                               // 000000007550: D761002D 0001280E
	v_writelane_b32 v45, s15, 21                               // 000000007558: D761002D 00012A0F
	s_mov_b64 s[12:13], s[100:101]                             // 000000007560: BE8C0164
	s_mov_b64 s[8:9], s[100:101]                               // 000000007564: BE880164
	s_mov_b64 s[14:15], s[102:103]                             // 000000007568: BE8E0166
	s_mov_b64 s[10:11], s[102:103]                             // 00000000756C: BE8A0166
	v_writelane_b32 v45, s24, 30                               // 000000007570: D761002D 00013C18
	v_writelane_b32 v45, s25, 31                               // 000000007578: D761002D 00013E19
	s_or_saveexec_b32 s105, -1                                 // 000000007580: BEE922C1
	scratch_load_b32 v44, off, off offset:32                   // 000000007584: DC510020 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000758C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000007590: BF8903F7
	v_writelane_b32 v44, s26, 0                                // 000000007594: D761002C 0001001A
	v_writelane_b32 v44, s27, 1                                // 00000000759C: D761002C 0001021B
	v_writelane_b32 v44, s28, 2                                // 0000000075A4: D761002C 0001041C
	v_writelane_b32 v44, s29, 3                                // 0000000075AC: D761002C 0001061D
	v_writelane_b32 v44, s30, 4                                // 0000000075B4: D761002C 0001081E
	v_writelane_b32 v44, s31, 5                                // 0000000075BC: D761002C 00010A1F
	s_or_saveexec_b32 s105, -1                                 // 0000000075C4: BEE922C1
	scratch_store_b32 off, v44, off offset:32                  // 0000000075C8: DC690020 007C2C00
	s_mov_b32 exec_lo, s105                                    // 0000000075D0: BEFE0069
	s_mov_b64 s[92:93], s[100:101]                             // 0000000075D4: BEDC0164
	s_mov_b64 s[94:95], s[102:103]                             // 0000000075D8: BEDE0166
	v_writelane_b32 v45, s92, 6                                // 0000000075DC: D761002D 00010C5C
	s_mov_b64 s[16:17], s[100:101]                             // 0000000075E4: BE900164
	s_mov_b64 s[18:19], s[102:103]                             // 0000000075E8: BE920166
	v_writelane_b32 v38, s8, 22                                // 0000000075EC: D7610026 00012C08
	v_writelane_b32 v45, s93, 7                                // 0000000075F4: D761002D 00010E5D
	v_writelane_b32 v38, s9, 23                                // 0000000075FC: D7610026 00012E09
	v_writelane_b32 v45, s94, 8                                // 000000007604: D761002D 0001105E
	v_writelane_b32 v38, s10, 24                               // 00000000760C: D7610026 0001300A
	v_writelane_b32 v45, s95, 9                                // 000000007614: D761002D 0001125F
	v_writelane_b32 v38, s11, 25                               // 00000000761C: D7610026 0001320B
	v_writelane_b32 v45, s96, 10                               // 000000007624: D761002D 00011460
	v_writelane_b32 v38, s12, 26                               // 00000000762C: D7610026 0001340C
	v_writelane_b32 v45, s97, 11                               // 000000007634: D761002D 00011661
	v_writelane_b32 v38, s13, 27                               // 00000000763C: D7610026 0001360D
	v_writelane_b32 v45, s98, 12                               // 000000007644: D761002D 00011862
	v_writelane_b32 v38, s14, 28                               // 00000000764C: D7610026 0001380E
	v_writelane_b32 v45, s99, 13                               // 000000007654: D761002D 00011A63
	v_writelane_b32 v38, s15, 29                               // 00000000765C: D7610026 00013A0F
	v_writelane_b32 v45, s18, 0                                // 000000007664: D761002D 00010012
	v_writelane_b32 v38, s16, 30                               // 00000000766C: D7610026 00013C10
	v_writelane_b32 v45, s19, 1                                // 000000007674: D761002D 00010213
	v_writelane_b32 v38, s17, 31                               // 00000000767C: D7610026 00013E11
	v_writelane_b32 v45, s20, 2                                // 000000007684: D761002D 00010414
	v_writelane_b32 v45, s21, 3                                // 00000000768C: D761002D 00010615
	v_writelane_b32 v45, s22, 4                                // 000000007694: D761002D 00010816
	v_writelane_b32 v45, s23, 5                                // 00000000769C: D761002D 00010A17
	s_or_saveexec_b32 s105, -1                                 // 0000000076A4: BEE922C1
	scratch_load_b32 v44, off, off offset:32                   // 0000000076A8: DC510020 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 0000000076B0: BEFE0069
	s_waitcnt vmcnt(0)                                         // 0000000076B4: BF8903F7
	v_writelane_b32 v44, s0, 6                                 // 0000000076B8: D761002C 00010C00
	v_writelane_b32 v44, s1, 7                                 // 0000000076C0: D761002C 00010E01
	v_writelane_b32 v44, s2, 8                                 // 0000000076C8: D761002C 00011002
	v_writelane_b32 v44, s3, 9                                 // 0000000076D0: D761002C 00011203
	v_writelane_b32 v44, s4, 10                                // 0000000076D8: D761002C 00011404
	v_writelane_b32 v44, s5, 11                                // 0000000076E0: D761002C 00011605
	v_writelane_b32 v44, s6, 12                                // 0000000076E8: D761002C 00011806
	v_writelane_b32 v44, s7, 13                                // 0000000076F0: D761002C 00011A07
	s_or_saveexec_b32 s105, -1                                 // 0000000076F8: BEE922C1
	scratch_store_b32 off, v44, off offset:32                  // 0000000076FC: DC690020 007C2C00
	s_mov_b32 exec_lo, s105                                    // 000000007704: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000007708: BEE922C1
	scratch_store_b32 off, v40, off offset:76                  // 00000000770C: DC69004C 007C2800
	s_mov_b32 exec_lo, s105                                    // 000000007714: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000007718: BEE922C1
	scratch_store_b32 off, v39, off offset:104                 // 00000000771C: DC690068 007C2700
	s_mov_b32 exec_lo, s105                                    // 000000007724: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000007728: BEE922C1
	scratch_store_b32 off, v37, off offset:84                  // 00000000772C: DC690054 007C2500
	s_mov_b32 exec_lo, s105                                    // 000000007734: BEFE0069
	s_mov_b32 s97, s96                                         // 000000007738: BEE10060
	s_mov_b32 s98, s96                                         // 00000000773C: BEE20060
	s_mov_b32 s99, s96                                         // 000000007740: BEE30060
	s_mov_b32 s24, s96                                         // 000000007744: BE980060
	s_mov_b32 s25, s96                                         // 000000007748: BE990060
	s_mov_b32 s26, s96                                         // 00000000774C: BE9A0060
	s_mov_b32 s27, s96                                         // 000000007750: BE9B0060
	s_mov_b32 s16, s96                                         // 000000007754: BE900060
	s_and_not1_b32 vcc_lo, exec_lo, vcc_hi                     // 000000007758: 916A6B7E
	s_mov_b32 s17, s96                                         // 00000000775C: BE910060
	s_mov_b32 s18, s96                                         // 000000007760: BE920060
	s_mov_b32 s19, s96                                         // 000000007764: BE930060
	s_mov_b32 s20, s96                                         // 000000007768: BE940060
	s_mov_b32 s21, s96                                         // 00000000776C: BE950060
	s_mov_b32 s22, s96                                         // 000000007770: BE960060
	s_mov_b32 s23, s96                                         // 000000007774: BE970060
	s_mov_b32 s92, s96                                         // 000000007778: BEDC0060
	s_mov_b32 s93, s96                                         // 00000000777C: BEDD0060
	s_mov_b32 s94, s96                                         // 000000007780: BEDE0060
	s_mov_b32 s95, s96                                         // 000000007784: BEDF0060
	s_mov_b32 s28, s96                                         // 000000007788: BE9C0060
	s_mov_b32 s29, s96                                         // 00000000778C: BE9D0060
	s_mov_b32 s30, s96                                         // 000000007790: BE9E0060
	s_mov_b32 s31, s96                                         // 000000007794: BE9F0060
	s_cbranch_vccnz 5                                          // 000000007798: BFA40005 <r_3_3_3_8_8_8+0x61b0>
	s_clause 0x1                                               // 00000000779C: BF850001
	s_load_b256 s[92:99], s[34:35], 0x23e0                     // 0000000077A0: F40C1711 F80023E0
	s_load_b512 s[16:31], s[34:35], 0x2500                     // 0000000077A8: F4100411 F8002500
	s_or_saveexec_b32 s105, -1                                 // 0000000077B0: BEE922C1
	scratch_load_b32 v44, off, off offset:8                    // 0000000077B4: DC510008 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 0000000077BC: BEFE0069
	s_waitcnt vmcnt(0)                                         // 0000000077C0: BF8903F7
	v_readlane_b32 s0, v44, 20                                 // 0000000077C4: D7600000 0001292C
	v_readlane_b32 s1, v44, 21                                 // 0000000077CC: D7600001 00012B2C
	v_readlane_b32 s2, v44, 22                                 // 0000000077D4: D7600002 00012D2C
	v_readlane_b32 s3, v44, 23                                 // 0000000077DC: D7600003 00012F2C
	v_readlane_b32 s4, v44, 24                                 // 0000000077E4: D7600004 0001312C
	v_readlane_b32 s5, v44, 25                                 // 0000000077EC: D7600005 0001332C
	v_readlane_b32 s6, v44, 26                                 // 0000000077F4: D7600006 0001352C
	v_readlane_b32 s7, v44, 27                                 // 0000000077FC: D7600007 0001372C
	v_readlane_b32 s8, v44, 28                                 // 000000007804: D7600008 0001392C
	v_readlane_b32 s9, v44, 29                                 // 00000000780C: D7600009 00013B2C
	v_readlane_b32 s10, v44, 30                                // 000000007814: D760000A 00013D2C
	v_readlane_b32 s11, v44, 31                                // 00000000781C: D760000B 00013F2C
	s_or_saveexec_b32 s105, -1                                 // 000000007824: BEE922C1
	scratch_load_b32 v44, off, off offset:72                   // 000000007828: DC510048 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 000000007830: BEFE0069
	v_dual_add_f32 v24, s4, v24 :: v_dual_add_f32 v25, s9, v25 // 000000007834: C9083004 18183209
	v_add_f32_e32 v26, s0, v26                                 // 00000000783C: 06343400
	s_waitcnt vmcnt(0)                                         // 000000007840: BF8903F7
	v_readlane_b32 s12, v44, 0                                 // 000000007844: D760000C 0001012C
	v_readlane_b32 s13, v44, 1                                 // 00000000784C: D760000D 0001032C
	v_readlane_b32 s14, v44, 2                                 // 000000007854: D760000E 0001052C
	v_dual_add_f32 v25, s10, v25 :: v_dual_add_f32 v24, s5, v24// 00000000785C: C908320A 19183005
	v_readlane_b32 s15, v44, 3                                 // 000000007864: D760000F 0001072C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000786C: BF870092
	v_dual_add_f32 v25, s11, v25 :: v_dual_add_f32 v26, s1, v26// 000000007870: C908320B 191A3401
	v_dual_add_f32 v25, s12, v25 :: v_dual_add_f32 v24, s6, v24// 000000007878: C908320C 19183006
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007880: BF870091
	v_dual_add_f32 v25, s13, v25 :: v_dual_add_f32 v26, s2, v26// 000000007884: C908320D 191A3402
	v_dual_add_f32 v25, s14, v25 :: v_dual_add_f32 v24, s7, v24// 00000000788C: C908320E 19183007
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007894: BF870111
	v_dual_add_f32 v26, s3, v26 :: v_dual_add_f32 v25, s15, v25// 000000007898: C9083403 1A18320F
	v_add_f32_e32 v24, s8, v24                                 // 0000000078A0: 06303008
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000078A4: BF870112
	v_add_f32_e32 v26, s4, v26                                 // 0000000078A8: 06343404
	v_add_f32_e32 v24, s9, v24                                 // 0000000078AC: 06303009
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000078B0: BF870112
	v_add_f32_e32 v26, s5, v26                                 // 0000000078B4: 06343405
	v_add_f32_e32 v24, s10, v24                                // 0000000078B8: 0630300A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000078BC: BF870112
	v_add_f32_e32 v26, s6, v26                                 // 0000000078C0: 06343406
	v_add_f32_e32 v24, s11, v24                                // 0000000078C4: 0630300B
	v_readlane_b32 s0, v44, 4                                  // 0000000078C8: D7600000 0001092C
	v_readlane_b32 s4, v44, 8                                  // 0000000078D0: D7600004 0001112C
	v_readlane_b32 s5, v44, 9                                  // 0000000078D8: D7600005 0001132C
	v_readlane_b32 s9, v44, 13                                 // 0000000078E0: D7600009 00011B2C
	v_readlane_b32 s1, v44, 5                                  // 0000000078E8: D7600001 00010B2C
	v_readlane_b32 s6, v44, 10                                 // 0000000078F0: D7600006 0001152C
	v_add_f32_e32 v19, s4, v19                                 // 0000000078F8: 06262604
	v_readlane_b32 s10, v44, 14                                // 0000000078FC: D760000A 00011D2C
	v_add_f32_e32 v23, s0, v23                                 // 000000007904: 062E2E00
	v_add_f32_e32 v21, s9, v21                                 // 000000007908: 062A2A09
	v_readlane_b32 s2, v44, 6                                  // 00000000790C: D7600002 00010D2C
	v_add_f32_e32 v19, s5, v19                                 // 000000007914: 06262605
	v_readlane_b32 s7, v44, 11                                 // 000000007918: D7600007 0001172C
	v_readlane_b32 s11, v44, 15                                // 000000007920: D760000B 00011F2C
	v_add_f32_e32 v23, s1, v23                                 // 000000007928: 062E2E01
	v_add_f32_e32 v21, s10, v21                                // 00000000792C: 062A2A0A
	v_add_f32_e32 v19, s6, v19                                 // 000000007930: 06262606
	v_readlane_b32 s3, v44, 7                                  // 000000007934: D7600003 00010F2C
	v_readlane_b32 s8, v44, 12                                 // 00000000793C: D7600008 0001192C
	v_add_f32_e32 v23, s2, v23                                 // 000000007944: 062E2E02
	v_readlane_b32 s12, v44, 16                                // 000000007948: D760000C 0001212C
	v_add_f32_e32 v19, s7, v19                                 // 000000007950: 06262607
	v_add_f32_e32 v21, s11, v21                                // 000000007954: 062A2A0B
	v_readlane_b32 s13, v44, 17                                // 000000007958: D760000D 0001232C
	v_add_f32_e32 v23, s3, v23                                 // 000000007960: 062E2E03
	v_readlane_b32 s14, v44, 18                                // 000000007964: D760000E 0001252C
	v_add_f32_e32 v19, s8, v19                                 // 00000000796C: 06262608
	v_add_f32_e32 v21, s12, v21                                // 000000007970: 062A2A0C
	v_readlane_b32 s15, v44, 19                                // 000000007974: D760000F 0001272C
	v_add_f32_e32 v23, s4, v23                                 // 00000000797C: 062E2E04
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000007980: BF870214
	v_add_f32_e32 v19, s9, v19                                 // 000000007984: 06262609
	v_add_f32_e32 v21, s13, v21                                // 000000007988: 062A2A0D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000798C: BF870193
	v_add_f32_e32 v23, s5, v23                                 // 000000007990: 062E2E05
	v_add_f32_e32 v19, s10, v19                                // 000000007994: 0626260A
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007998: BF870193
	v_add_f32_e32 v21, s14, v21                                // 00000000799C: 062A2A0E
	v_add_f32_e32 v23, s6, v23                                 // 0000000079A0: 062E2E06
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000079A4: BF870193
	v_add_f32_e32 v19, s11, v19                                // 0000000079A8: 0626260B
	v_add_f32_e32 v21, s15, v21                                // 0000000079AC: 062A2A0F
	s_or_saveexec_b32 s105, -1                                 // 0000000079B0: BEE922C1
	scratch_load_b32 v40, off, off offset:88                   // 0000000079B4: DC510058 287C0000
	s_mov_b32 exec_lo, s105                                    // 0000000079BC: BEFE0069
	s_waitcnt vmcnt(0)                                         // 0000000079C0: BF8903F7
	v_readlane_b32 s0, v40, 4                                  // 0000000079C4: D7600000 00010928
	v_readlane_b32 s4, v40, 8                                  // 0000000079CC: D7600004 00011128
	v_readlane_b32 s5, v40, 9                                  // 0000000079D4: D7600005 00011328
	v_readlane_b32 s9, v40, 13                                 // 0000000079DC: D7600009 00011B28
	v_readlane_b32 s1, v40, 5                                  // 0000000079E4: D7600001 00010B28
	v_readlane_b32 s6, v40, 10                                 // 0000000079EC: D7600006 00011528
	v_add_f32_e32 v18, s4, v18                                 // 0000000079F4: 06242404
	v_readlane_b32 s10, v40, 14                                // 0000000079F8: D760000A 00011D28
	v_add_f32_e32 v22, s0, v22                                 // 000000007A00: 062C2C00
	v_add_f32_e32 v20, s9, v20                                 // 000000007A04: 06282809
	v_readlane_b32 s2, v40, 6                                  // 000000007A08: D7600002 00010D28
	v_add_f32_e32 v18, s5, v18                                 // 000000007A10: 06242405
	v_readlane_b32 s7, v40, 11                                 // 000000007A14: D7600007 00011728
	v_readlane_b32 s11, v40, 15                                // 000000007A1C: D760000B 00011F28
	v_add_f32_e32 v22, s1, v22                                 // 000000007A24: 062C2C01
	v_add_f32_e32 v20, s10, v20                                // 000000007A28: 0628280A
	v_add_f32_e32 v18, s6, v18                                 // 000000007A2C: 06242406
	v_readlane_b32 s3, v40, 7                                  // 000000007A30: D7600003 00010F28
	v_readlane_b32 s8, v40, 12                                 // 000000007A38: D7600008 00011928
	v_readlane_b32 s12, v40, 16                                // 000000007A40: D760000C 00012128
	v_add_f32_e32 v22, s2, v22                                 // 000000007A48: 062C2C02
	v_add_f32_e32 v18, s7, v18                                 // 000000007A4C: 06242407
	v_add_f32_e32 v20, s11, v20                                // 000000007A50: 0628280B
	v_readlane_b32 s13, v40, 17                                // 000000007A54: D760000D 00012328
	v_readlane_b32 s14, v40, 18                                // 000000007A5C: D760000E 00012528
	v_add_f32_e32 v22, s3, v22                                 // 000000007A64: 062C2C03
	v_add_f32_e32 v18, s8, v18                                 // 000000007A68: 06242408
	v_add_f32_e32 v20, s12, v20                                // 000000007A6C: 0628280C
	v_readlane_b32 s15, v40, 19                                // 000000007A70: D760000F 00012728
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000007A78: BF870214
	v_add_f32_e32 v22, s4, v22                                 // 000000007A7C: 062C2C04
	v_add_f32_e32 v18, s9, v18                                 // 000000007A80: 06242409
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007A84: BF870194
	v_add_f32_e32 v20, s13, v20                                // 000000007A88: 0628280D
	v_add_f32_e32 v22, s5, v22                                 // 000000007A8C: 062C2C05
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007A90: BF870193
	v_add_f32_e32 v18, s10, v18                                // 000000007A94: 0624240A
	v_add_f32_e32 v20, s14, v20                                // 000000007A98: 0628280E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007A9C: BF870193
	v_add_f32_e32 v22, s6, v22                                 // 000000007AA0: 062C2C06
	v_add_f32_e32 v18, s11, v18                                // 000000007AA4: 0624240B
	s_delay_alu instid0(VALU_DEP_3)                            // 000000007AA8: BF870003
	v_add_f32_e32 v20, s15, v20                                // 000000007AAC: 0628280F
	s_or_saveexec_b32 s105, -1                                 // 000000007AB0: BEE922C1
	scratch_load_b32 v39, off, off offset:60                   // 000000007AB4: DC51003C 277C0000
	s_mov_b32 exec_lo, s105                                    // 000000007ABC: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000007AC0: BF8903F7
	v_readlane_b32 s0, v39, 20                                 // 000000007AC4: D7600000 00012927
	v_readlane_b32 s1, v39, 21                                 // 000000007ACC: D7600001 00012B27
	v_readlane_b32 s2, v39, 22                                 // 000000007AD4: D7600002 00012D27
	v_readlane_b32 s3, v39, 23                                 // 000000007ADC: D7600003 00012F27
	v_readlane_b32 s4, v39, 24                                 // 000000007AE4: D7600004 00013127
	v_readlane_b32 s5, v39, 25                                 // 000000007AEC: D7600005 00013327
	v_readlane_b32 s6, v39, 26                                 // 000000007AF4: D7600006 00013527
	v_readlane_b32 s7, v39, 27                                 // 000000007AFC: D7600007 00013727
	v_readlane_b32 s8, v39, 28                                 // 000000007B04: D7600008 00013927
	v_readlane_b32 s9, v39, 29                                 // 000000007B0C: D7600009 00013B27
	v_readlane_b32 s10, v39, 30                                // 000000007B14: D760000A 00013D27
	v_readlane_b32 s11, v39, 31                                // 000000007B1C: D760000B 00013F27
	s_or_saveexec_b32 s105, -1                                 // 000000007B24: BEE922C1
	scratch_load_b32 v39, off, off offset:16                   // 000000007B28: DC510010 277C0000
	s_mov_b32 exec_lo, s105                                    // 000000007B30: BEFE0069
	v_dual_add_f32 v16, s4, v16 :: v_dual_add_f32 v13, s0, v13 // 000000007B34: C9082004 100C1A00
	s_waitcnt vmcnt(0)                                         // 000000007B3C: BF8903F7
	v_readlane_b32 s12, v39, 0                                 // 000000007B40: D760000C 00010127
	v_readlane_b32 s13, v39, 1                                 // 000000007B48: D760000D 00010327
	v_readlane_b32 s14, v39, 2                                 // 000000007B50: D760000E 00010527
	v_dual_add_f32 v16, s5, v16 :: v_dual_add_f32 v17, s9, v17 // 000000007B58: C9082005 10102209
	v_add_f32_e32 v13, s1, v13                                 // 000000007B60: 061A1A01
	v_readlane_b32 s15, v39, 3                                 // 000000007B64: D760000F 00010727
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007B6C: BF870093
	v_dual_add_f32 v16, s6, v16 :: v_dual_add_f32 v17, s10, v17// 000000007B70: C9082006 1010220A
	v_dual_add_f32 v16, s7, v16 :: v_dual_add_f32 v13, s2, v13 // 000000007B78: C9082007 100C1A02
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007B80: BF870091
	v_dual_add_f32 v16, s8, v16 :: v_dual_add_f32 v17, s11, v17// 000000007B84: C9082008 1010220B
	v_dual_add_f32 v16, s9, v16 :: v_dual_add_f32 v13, s3, v13 // 000000007B8C: C9082009 100C1A03
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007B94: BF870091
	v_dual_add_f32 v16, s10, v16 :: v_dual_add_f32 v17, s12, v17// 000000007B98: C908200A 1010220C
	v_dual_add_f32 v13, s4, v13 :: v_dual_add_f32 v16, s11, v16// 000000007BA0: C9081A04 0D10200B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007BA8: BF870112
	v_add_f32_e32 v17, s13, v17                                // 000000007BAC: 0622220D
	v_add_f32_e32 v13, s5, v13                                 // 000000007BB0: 061A1A05
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007BB4: BF870112
	v_add_f32_e32 v17, s14, v17                                // 000000007BB8: 0622220E
	v_add_f32_e32 v13, s6, v13                                 // 000000007BBC: 061A1A06
	s_delay_alu instid0(VALU_DEP_2)                            // 000000007BC0: BF870002
	v_add_f32_e32 v17, s15, v17                                // 000000007BC4: 0622220F
	v_readlane_b32 s0, v44, 20                                 // 000000007BC8: D7600000 0001292C
	v_readlane_b32 s4, v44, 24                                 // 000000007BD0: D7600004 0001312C
	v_readlane_b32 s5, v44, 25                                 // 000000007BD8: D7600005 0001332C
	v_readlane_b32 s1, v44, 21                                 // 000000007BE0: D7600001 00012B2C
	v_readlane_b32 s6, v44, 26                                 // 000000007BE8: D7600006 0001352C
	v_readlane_b32 s9, v44, 29                                 // 000000007BF0: D7600009 00013B2C
	v_add_f32_e32 v14, s4, v14                                 // 000000007BF8: 061C1C04
	v_add_f32_e32 v12, s0, v12                                 // 000000007BFC: 06181800
	v_readlane_b32 s2, v44, 22                                 // 000000007C00: D7600002 00012D2C
	v_readlane_b32 s7, v44, 27                                 // 000000007C08: D7600007 0001372C
	v_readlane_b32 s10, v44, 30                                // 000000007C10: D760000A 00013D2C
	v_add_f32_e32 v14, s5, v14                                 // 000000007C18: 061C1C05
	v_add_f32_e32 v12, s1, v12                                 // 000000007C1C: 06181801
	v_readlane_b32 s3, v44, 23                                 // 000000007C20: D7600003 00012F2C
	v_readlane_b32 s11, v44, 31                                // 000000007C28: D760000B 00013F2C
	v_readlane_b32 s8, v44, 28                                 // 000000007C30: D7600008 0001392C
	v_add_f32_e32 v14, s6, v14                                 // 000000007C38: 061C1C06
	v_dual_add_f32 v12, s2, v12 :: v_dual_add_f32 v15, s9, v15 // 000000007C3C: C9081802 0C0E1E09
	v_readlane_b32 s12, v40, 0                                 // 000000007C44: D760000C 00010128
	v_readlane_b32 s13, v40, 1                                 // 000000007C4C: D760000D 00010328
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000007C54: BF870214
	v_add_f32_e32 v14, s7, v14                                 // 000000007C58: 061C1C07
	v_dual_add_f32 v12, s3, v12 :: v_dual_add_f32 v15, s10, v15// 000000007C5C: C9081803 0C0E1E0A
	v_readlane_b32 s14, v40, 2                                 // 000000007C64: D760000E 00010528
	v_readlane_b32 s15, v40, 3                                 // 000000007C6C: D760000F 00010728
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000007C74: BF870214
	v_add_f32_e32 v14, s8, v14                                 // 000000007C78: 061C1C08
	v_dual_add_f32 v12, s4, v12 :: v_dual_add_f32 v15, s11, v15// 000000007C7C: C9081804 0C0E1E0B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007C84: BF870091
	v_dual_add_f32 v15, s12, v15 :: v_dual_add_f32 v14, s9, v14// 000000007C88: C9081E0C 0F0E1C09
	v_dual_add_f32 v15, s13, v15 :: v_dual_add_f32 v12, s5, v12// 000000007C90: C9081E0D 0F0C1805
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007C98: BF870091
	v_dual_add_f32 v15, s14, v15 :: v_dual_add_f32 v14, s10, v14// 000000007C9C: C9081E0E 0F0E1C0A
	v_dual_add_f32 v12, s6, v12 :: v_dual_add_f32 v15, s15, v15// 000000007CA4: C9081806 0C0E1E0F
	s_delay_alu instid0(VALU_DEP_2)                            // 000000007CAC: BF870002
	v_add_f32_e32 v14, s11, v14                                // 000000007CB0: 061C1C0B
	v_readlane_b32 s0, v40, 20                                 // 000000007CB4: D7600000 00012928
	v_readlane_b32 s1, v40, 21                                 // 000000007CBC: D7600001 00012B28
	v_readlane_b32 s2, v40, 22                                 // 000000007CC4: D7600002 00012D28
	v_readlane_b32 s3, v40, 23                                 // 000000007CCC: D7600003 00012F28
	v_readlane_b32 s4, v40, 24                                 // 000000007CD4: D7600004 00013128
	v_readlane_b32 s5, v40, 25                                 // 000000007CDC: D7600005 00013328
	v_readlane_b32 s6, v40, 26                                 // 000000007CE4: D7600006 00013528
	v_readlane_b32 s7, v40, 27                                 // 000000007CEC: D7600007 00013728
	v_readlane_b32 s8, v40, 28                                 // 000000007CF4: D7600008 00013928
	v_readlane_b32 s9, v40, 29                                 // 000000007CFC: D7600009 00013B28
	v_readlane_b32 s10, v40, 30                                // 000000007D04: D760000A 00013D28
	v_readlane_b32 s11, v40, 31                                // 000000007D0C: D760000B 00013F28
	s_or_saveexec_b32 s105, -1                                 // 000000007D14: BEE922C1
	scratch_load_b32 v44, off, off offset:44                   // 000000007D18: DC51002C 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 000000007D20: BEFE0069
	v_dual_add_f32 v24, s4, v24 :: v_dual_add_f32 v25, s9, v25 // 000000007D24: C9083004 18183209
	v_add_f32_e32 v26, s0, v26                                 // 000000007D2C: 06343400
	s_waitcnt vmcnt(0)                                         // 000000007D30: BF8903F7
	v_readlane_b32 s12, v44, 0                                 // 000000007D34: D760000C 0001012C
	v_readlane_b32 s13, v44, 1                                 // 000000007D3C: D760000D 0001032C
	v_readlane_b32 s14, v44, 2                                 // 000000007D44: D760000E 0001052C
	v_dual_add_f32 v25, s10, v25 :: v_dual_add_f32 v24, s5, v24// 000000007D4C: C908320A 19183005
	v_readlane_b32 s15, v44, 3                                 // 000000007D54: D760000F 0001072C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007D5C: BF870092
	v_dual_add_f32 v25, s11, v25 :: v_dual_add_f32 v26, s1, v26// 000000007D60: C908320B 191A3401
	v_dual_add_f32 v25, s12, v25 :: v_dual_add_f32 v24, s6, v24// 000000007D68: C908320C 19183006
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007D70: BF870091
	v_dual_add_f32 v25, s13, v25 :: v_dual_add_f32 v26, s2, v26// 000000007D74: C908320D 191A3402
	v_dual_add_f32 v25, s14, v25 :: v_dual_add_f32 v24, s7, v24// 000000007D7C: C908320E 19183007
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007D84: BF870111
	v_dual_add_f32 v26, s3, v26 :: v_dual_add_f32 v25, s15, v25// 000000007D88: C9083403 1A18320F
	v_add_f32_e32 v24, s8, v24                                 // 000000007D90: 06303008
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007D94: BF870112
	v_add_f32_e32 v26, s4, v26                                 // 000000007D98: 06343404
	v_add_f32_e32 v24, s9, v24                                 // 000000007D9C: 06303009
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007DA0: BF870112
	v_add_f32_e32 v26, s5, v26                                 // 000000007DA4: 06343405
	v_add_f32_e32 v24, s10, v24                                // 000000007DA8: 0630300A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007DAC: BF870112
	v_add_f32_e32 v26, s6, v26                                 // 000000007DB0: 06343406
	v_add_f32_e32 v24, s11, v24                                // 000000007DB4: 0630300B
	s_or_saveexec_b32 s105, -1                                 // 000000007DB8: BEE922C1
	scratch_load_b32 v40, off, off offset:48                   // 000000007DBC: DC510030 287C0000
	s_mov_b32 exec_lo, s105                                    // 000000007DC4: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000007DC8: BF8903F7
	v_readlane_b32 s0, v40, 20                                 // 000000007DCC: D7600000 00012928
	v_readlane_b32 s1, v40, 21                                 // 000000007DD4: D7600001 00012B28
	v_readlane_b32 s2, v40, 22                                 // 000000007DDC: D7600002 00012D28
	v_readlane_b32 s3, v40, 23                                 // 000000007DE4: D7600003 00012F28
	v_readlane_b32 s4, v40, 24                                 // 000000007DEC: D7600004 00013128
	v_readlane_b32 s5, v40, 25                                 // 000000007DF4: D7600005 00013328
	v_readlane_b32 s6, v40, 26                                 // 000000007DFC: D7600006 00013528
	v_readlane_b32 s7, v40, 27                                 // 000000007E04: D7600007 00013728
	v_readlane_b32 s8, v40, 28                                 // 000000007E0C: D7600008 00013928
	v_readlane_b32 s9, v40, 29                                 // 000000007E14: D7600009 00013B28
	v_readlane_b32 s10, v40, 30                                // 000000007E1C: D760000A 00013D28
	v_readlane_b32 s11, v40, 31                                // 000000007E24: D760000B 00013F28
	s_or_saveexec_b32 s105, -1                                 // 000000007E2C: BEE922C1
	scratch_load_b32 v40, off, off offset:40                   // 000000007E30: DC510028 287C0000
	s_mov_b32 exec_lo, s105                                    // 000000007E38: BEFE0069
	v_add_f32_e32 v19, s4, v19                                 // 000000007E3C: 06262604
	v_add_f32_e32 v23, s0, v23                                 // 000000007E40: 062E2E00
	v_add_f32_e32 v21, s9, v21                                 // 000000007E44: 062A2A09
	s_waitcnt vmcnt(0)                                         // 000000007E48: BF8903F7
	v_readlane_b32 s12, v40, 0                                 // 000000007E4C: D760000C 00010128
	v_readlane_b32 s13, v40, 1                                 // 000000007E54: D760000D 00010328
	v_add_f32_e32 v19, s5, v19                                 // 000000007E5C: 06262605
	v_add_f32_e32 v23, s1, v23                                 // 000000007E60: 062E2E01
	v_add_f32_e32 v21, s10, v21                                // 000000007E64: 062A2A0A
	v_readlane_b32 s14, v40, 2                                 // 000000007E68: D760000E 00010528
	v_readlane_b32 s15, v40, 3                                 // 000000007E70: D760000F 00010728
	v_add_f32_e32 v19, s6, v19                                 // 000000007E78: 06262606
	v_add_f32_e32 v23, s2, v23                                 // 000000007E7C: 062E2E02
	v_add_f32_e32 v21, s11, v21                                // 000000007E80: 062A2A0B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007E84: BF870193
	v_add_f32_e32 v19, s7, v19                                 // 000000007E88: 06262607
	v_add_f32_e32 v23, s3, v23                                 // 000000007E8C: 062E2E03
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007E90: BF870193
	v_add_f32_e32 v21, s12, v21                                // 000000007E94: 062A2A0C
	v_add_f32_e32 v19, s8, v19                                 // 000000007E98: 06262608
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007E9C: BF870193
	v_add_f32_e32 v23, s4, v23                                 // 000000007EA0: 062E2E04
	v_add_f32_e32 v21, s13, v21                                // 000000007EA4: 062A2A0D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007EA8: BF870193
	v_add_f32_e32 v19, s9, v19                                 // 000000007EAC: 06262609
	v_add_f32_e32 v23, s5, v23                                 // 000000007EB0: 062E2E05
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007EB4: BF870193
	v_add_f32_e32 v21, s14, v21                                // 000000007EB8: 062A2A0E
	v_add_f32_e32 v19, s10, v19                                // 000000007EBC: 0626260A
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007EC0: BF870193
	v_add_f32_e32 v23, s6, v23                                 // 000000007EC4: 062E2E06
	v_add_f32_e32 v21, s15, v21                                // 000000007EC8: 062A2A0F
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000007ECC: BF8704A3
	v_add_f32_e32 v19, s11, v19                                // 000000007ED0: 0626260B
	s_or_saveexec_b32 s105, -1                                 // 000000007ED4: BEE922C1
	s_mov_b32 exec_lo, s105                                    // 000000007ED8: BEFE0069
	v_readlane_b32 s0, v44, 4                                  // 000000007EDC: D7600000 0001092C
	v_readlane_b32 s4, v44, 8                                  // 000000007EE4: D7600004 0001112C
	v_readlane_b32 s5, v44, 9                                  // 000000007EEC: D7600005 0001132C
	v_readlane_b32 s9, v44, 13                                 // 000000007EF4: D7600009 00011B2C
	v_readlane_b32 s1, v44, 5                                  // 000000007EFC: D7600001 00010B2C
	v_readlane_b32 s6, v44, 10                                 // 000000007F04: D7600006 0001152C
	v_add_f32_e32 v18, s4, v18                                 // 000000007F0C: 06242404
	v_readlane_b32 s10, v44, 14                                // 000000007F10: D760000A 00011D2C
	v_add_f32_e32 v22, s0, v22                                 // 000000007F18: 062C2C00
	v_add_f32_e32 v20, s9, v20                                 // 000000007F1C: 06282809
	v_readlane_b32 s2, v44, 6                                  // 000000007F20: D7600002 00010D2C
	v_add_f32_e32 v18, s5, v18                                 // 000000007F28: 06242405
	v_readlane_b32 s7, v44, 11                                 // 000000007F2C: D7600007 0001172C
	v_readlane_b32 s11, v44, 15                                // 000000007F34: D760000B 00011F2C
	v_add_f32_e32 v22, s1, v22                                 // 000000007F3C: 062C2C01
	v_add_f32_e32 v20, s10, v20                                // 000000007F40: 0628280A
	v_add_f32_e32 v18, s6, v18                                 // 000000007F44: 06242406
	v_readlane_b32 s3, v44, 7                                  // 000000007F48: D7600003 00010F2C
	v_readlane_b32 s8, v44, 12                                 // 000000007F50: D7600008 0001192C
	v_readlane_b32 s12, v44, 16                                // 000000007F58: D760000C 0001212C
	v_add_f32_e32 v22, s2, v22                                 // 000000007F60: 062C2C02
	v_add_f32_e32 v18, s7, v18                                 // 000000007F64: 06242407
	v_add_f32_e32 v20, s11, v20                                // 000000007F68: 0628280B
	v_readlane_b32 s13, v44, 17                                // 000000007F6C: D760000D 0001232C
	v_readlane_b32 s14, v44, 18                                // 000000007F74: D760000E 0001252C
	v_add_f32_e32 v22, s3, v22                                 // 000000007F7C: 062C2C03
	v_add_f32_e32 v18, s8, v18                                 // 000000007F80: 06242408
	v_add_f32_e32 v20, s12, v20                                // 000000007F84: 0628280C
	v_readlane_b32 s15, v44, 19                                // 000000007F88: D760000F 0001272C
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000007F90: BF870214
	v_add_f32_e32 v22, s4, v22                                 // 000000007F94: 062C2C04
	v_add_f32_e32 v18, s9, v18                                 // 000000007F98: 06242409
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007F9C: BF870194
	v_add_f32_e32 v20, s13, v20                                // 000000007FA0: 0628280D
	v_add_f32_e32 v22, s5, v22                                 // 000000007FA4: 062C2C05
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007FA8: BF870193
	v_add_f32_e32 v18, s10, v18                                // 000000007FAC: 0624240A
	v_add_f32_e32 v20, s14, v20                                // 000000007FB0: 0628280E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007FB4: BF870193
	v_add_f32_e32 v22, s6, v22                                 // 000000007FB8: 062C2C06
	v_add_f32_e32 v18, s11, v18                                // 000000007FBC: 0624240B
	s_delay_alu instid0(VALU_DEP_3)                            // 000000007FC0: BF870003
	v_add_f32_e32 v20, s15, v20                                // 000000007FC4: 0628280F
	s_or_saveexec_b32 s105, -1                                 // 000000007FC8: BEE922C1
	scratch_load_b32 v44, off, off offset:64                   // 000000007FCC: DC510040 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 000000007FD4: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000007FD8: BF8903F7
	v_readlane_b32 s0, v44, 4                                  // 000000007FDC: D7600000 0001092C
	v_readlane_b32 s1, v44, 5                                  // 000000007FE4: D7600001 00010B2C
	v_readlane_b32 s2, v44, 6                                  // 000000007FEC: D7600002 00010D2C
	v_readlane_b32 s3, v44, 7                                  // 000000007FF4: D7600003 00010F2C
	v_readlane_b32 s4, v44, 8                                  // 000000007FFC: D7600004 0001112C
	v_readlane_b32 s5, v44, 9                                  // 000000008004: D7600005 0001132C
	v_readlane_b32 s6, v44, 10                                 // 00000000800C: D7600006 0001152C
	v_readlane_b32 s7, v44, 11                                 // 000000008014: D7600007 0001172C
	v_readlane_b32 s8, v44, 12                                 // 00000000801C: D7600008 0001192C
	v_readlane_b32 s9, v44, 13                                 // 000000008024: D7600009 00011B2C
	v_readlane_b32 s10, v44, 14                                // 00000000802C: D760000A 00011D2C
	v_readlane_b32 s11, v44, 15                                // 000000008034: D760000B 00011F2C
	v_readlane_b32 s12, v44, 16                                // 00000000803C: D760000C 0001212C
	v_readlane_b32 s13, v44, 17                                // 000000008044: D760000D 0001232C
	v_readlane_b32 s14, v44, 18                                // 00000000804C: D760000E 0001252C
	v_readlane_b32 s15, v44, 19                                // 000000008054: D760000F 0001272C
	s_or_saveexec_b32 s105, -1                                 // 00000000805C: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000008060: BF8700A9
	s_mov_b32 exec_lo, s105                                    // 000000008064: BEFE0069
	v_dual_add_f32 v16, s4, v16 :: v_dual_add_f32 v13, s0, v13 // 000000008068: C9082004 100C1A00
	v_dual_add_f32 v17, s9, v17 :: v_dual_add_f32 v16, s5, v16 // 000000008070: C9082209 11102005
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008078: BF870091
	v_dual_add_f32 v13, s1, v13 :: v_dual_add_f32 v16, s6, v16 // 00000000807C: C9081A01 0D102006
	v_dual_add_f32 v17, s10, v17 :: v_dual_add_f32 v16, s7, v16// 000000008084: C908220A 11102007
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000808C: BF870091
	v_dual_add_f32 v13, s2, v13 :: v_dual_add_f32 v16, s8, v16 // 000000008090: C9081A02 0D102008
	v_dual_add_f32 v17, s11, v17 :: v_dual_add_f32 v16, s9, v16// 000000008098: C908220B 11102009
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000080A0: BF870111
	v_dual_add_f32 v13, s3, v13 :: v_dual_add_f32 v16, s10, v16// 0000000080A4: C9081A03 0D10200A
	v_add_f32_e32 v17, s12, v17                                // 0000000080AC: 0622220C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000080B0: BF870112
	v_dual_add_f32 v13, s4, v13 :: v_dual_add_f32 v16, s11, v16// 0000000080B4: C9081A04 0D10200B
	v_add_f32_e32 v17, s13, v17                                // 0000000080BC: 0622220D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000080C0: BF870112
	v_add_f32_e32 v13, s5, v13                                 // 0000000080C4: 061A1A05
	v_add_f32_e32 v17, s14, v17                                // 0000000080C8: 0622220E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000080CC: BF870112
	v_add_f32_e32 v13, s6, v13                                 // 0000000080D0: 061A1A06
	v_add_f32_e32 v17, s15, v17                                // 0000000080D4: 0622220F
	s_or_saveexec_b32 s105, -1                                 // 0000000080D8: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000080DC: BF870009
	s_mov_b32 exec_lo, s105                                    // 0000000080E0: BEFE0069
	v_readlane_b32 s0, v40, 12                                 // 0000000080E4: D7600000 00011928
	v_readlane_b32 s4, v40, 16                                 // 0000000080EC: D7600004 00012128
	v_readlane_b32 s9, v40, 21                                 // 0000000080F4: D7600009 00012B28
	v_readlane_b32 s10, v40, 22                                // 0000000080FC: D760000A 00012D28
	v_readlane_b32 s5, v40, 17                                 // 000000008104: D7600005 00012328
	v_readlane_b32 s11, v40, 23                                // 00000000810C: D760000B 00012F28
	s_delay_alu instid0(VALU_DEP_4)                            // 000000008114: BF870004
	v_dual_add_f32 v14, s4, v14 :: v_dual_add_f32 v15, s9, v15 // 000000008118: C9081C04 0E0E1E09
	v_add_f32_e32 v12, s0, v12                                 // 000000008120: 06181800
	v_readlane_b32 s1, v40, 13                                 // 000000008124: D7600001 00011B28
	v_readlane_b32 s12, v40, 24                                // 00000000812C: D760000C 00013128
	v_readlane_b32 s6, v40, 18                                 // 000000008134: D7600006 00012528
	v_dual_add_f32 v15, s10, v15 :: v_dual_add_f32 v14, s5, v14// 00000000813C: C9081E0A 0F0E1C05
	v_readlane_b32 s13, v40, 25                                // 000000008144: D760000D 00013328
	v_readlane_b32 s2, v40, 14                                 // 00000000814C: D7600002 00011D28
	v_readlane_b32 s7, v40, 19                                 // 000000008154: D7600007 00012728
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 00000000815C: BF870244
	v_dual_add_f32 v15, s11, v15 :: v_dual_add_f32 v12, s1, v12// 000000008160: C9081E0B 0F0C1801
	v_readlane_b32 s14, v40, 26                                // 000000008168: D760000E 00013528
	v_readlane_b32 s3, v40, 15                                 // 000000008170: D7600003 00011F28
	v_readlane_b32 s8, v40, 20                                 // 000000008178: D7600008 00012928
	v_dual_add_f32 v15, s12, v15 :: v_dual_add_f32 v14, s6, v14// 000000008180: C9081E0C 0F0E1C06
	v_readlane_b32 s15, v40, 27                                // 000000008188: D760000F 00013728
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008190: BF870092
	v_dual_add_f32 v15, s13, v15 :: v_dual_add_f32 v12, s2, v12// 000000008194: C9081E0D 0F0C1802
	v_dual_add_f32 v15, s14, v15 :: v_dual_add_f32 v14, s7, v14// 00000000819C: C9081E0E 0F0E1C07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000081A4: BF870111
	v_dual_add_f32 v12, s3, v12 :: v_dual_add_f32 v15, s15, v15// 0000000081A8: C9081803 0C0E1E0F
	v_add_f32_e32 v14, s8, v14                                 // 0000000081B0: 061C1C08
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000081B4: BF870112
	v_add_f32_e32 v12, s4, v12                                 // 0000000081B8: 06181804
	v_add_f32_e32 v14, s9, v14                                 // 0000000081BC: 061C1C09
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000081C0: BF870112
	v_add_f32_e32 v12, s5, v12                                 // 0000000081C4: 06181805
	v_add_f32_e32 v14, s10, v14                                // 0000000081C8: 061C1C0A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000081CC: BF870112
	v_add_f32_e32 v12, s6, v12                                 // 0000000081D0: 06181806
	v_add_f32_e32 v14, s11, v14                                // 0000000081D4: 061C1C0B
	s_or_saveexec_b32 s105, -1                                 // 0000000081D8: BEE922C1
	scratch_load_b32 v40, off, off offset:52                   // 0000000081DC: DC510034 287C0000
	s_mov_b32 exec_lo, s105                                    // 0000000081E4: BEFE0069
	s_waitcnt vmcnt(0)                                         // 0000000081E8: BF8903F7
	v_readlane_b32 s0, v40, 4                                  // 0000000081EC: D7600000 00010928
	v_readlane_b32 s1, v40, 5                                  // 0000000081F4: D7600001 00010B28
	v_readlane_b32 s2, v40, 6                                  // 0000000081FC: D7600002 00010D28
	v_readlane_b32 s3, v40, 7                                  // 000000008204: D7600003 00010F28
	v_readlane_b32 s4, v40, 8                                  // 00000000820C: D7600004 00011128
	v_readlane_b32 s5, v40, 9                                  // 000000008214: D7600005 00011328
	v_readlane_b32 s6, v40, 10                                 // 00000000821C: D7600006 00011528
	v_readlane_b32 s7, v40, 11                                 // 000000008224: D7600007 00011728
	v_readlane_b32 s8, v40, 12                                 // 00000000822C: D7600008 00011928
	v_readlane_b32 s9, v40, 13                                 // 000000008234: D7600009 00011B28
	v_readlane_b32 s10, v40, 14                                // 00000000823C: D760000A 00011D28
	v_readlane_b32 s11, v40, 15                                // 000000008244: D760000B 00011F28
	v_readlane_b32 s12, v40, 16                                // 00000000824C: D760000C 00012128
	v_readlane_b32 s13, v40, 17                                // 000000008254: D760000D 00012328
	v_readlane_b32 s14, v40, 18                                // 00000000825C: D760000E 00012528
	s_or_saveexec_b32 s105, -1                                 // 000000008264: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 000000008268: BF8701C9
	s_mov_b32 exec_lo, s105                                    // 00000000826C: BEFE0069
	v_dual_add_f32 v24, s4, v24 :: v_dual_add_f32 v25, s9, v25 // 000000008270: C9083004 18183209
	v_add_f32_e32 v26, s0, v26                                 // 000000008278: 06343400
	v_readlane_b32 s15, v40, 19                                // 00000000827C: D760000F 00012728
	v_dual_add_f32 v25, s10, v25 :: v_dual_add_f32 v24, s5, v24// 000000008284: C908320A 19183005
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000828C: BF870091
	v_dual_add_f32 v25, s11, v25 :: v_dual_add_f32 v26, s1, v26// 000000008290: C908320B 191A3401
	v_dual_add_f32 v25, s12, v25 :: v_dual_add_f32 v24, s6, v24// 000000008298: C908320C 19183006
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000082A0: BF870091
	v_dual_add_f32 v25, s13, v25 :: v_dual_add_f32 v26, s2, v26// 0000000082A4: C908320D 191A3402
	v_dual_add_f32 v25, s14, v25 :: v_dual_add_f32 v24, s7, v24// 0000000082AC: C908320E 19183007
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000082B4: BF870111
	v_dual_add_f32 v26, s3, v26 :: v_dual_add_f32 v25, s15, v25// 0000000082B8: C9083403 1A18320F
	v_add_f32_e32 v24, s8, v24                                 // 0000000082C0: 06303008
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000082C4: BF870112
	v_add_f32_e32 v26, s4, v26                                 // 0000000082C8: 06343404
	v_add_f32_e32 v24, s9, v24                                 // 0000000082CC: 06303009
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000082D0: BF870112
	v_add_f32_e32 v26, s5, v26                                 // 0000000082D4: 06343405
	v_add_f32_e32 v24, s10, v24                                // 0000000082D8: 0630300A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000082DC: BF870112
	v_add_f32_e32 v26, s6, v26                                 // 0000000082E0: 06343406
	v_add_f32_e32 v24, s11, v24                                // 0000000082E4: 0630300B
	s_or_saveexec_b32 s105, -1                                 // 0000000082E8: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000082EC: BF870009
	s_mov_b32 exec_lo, s105                                    // 0000000082F0: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 0000000082F4: BEE922C1
	scratch_store_b32 off, v39, off offset:16                  // 0000000082F8: DC690010 007C2700
	s_mov_b32 exec_lo, s105                                    // 000000008300: BEFE0069
	v_readlane_b32 s0, v39, 20                                 // 000000008304: D7600000 00012927
	v_readlane_b32 s1, v39, 21                                 // 00000000830C: D7600001 00012B27
	v_readlane_b32 s2, v39, 22                                 // 000000008314: D7600002 00012D27
	v_readlane_b32 s3, v39, 23                                 // 00000000831C: D7600003 00012F27
	v_readlane_b32 s4, v39, 24                                 // 000000008324: D7600004 00013127
	v_readlane_b32 s5, v39, 25                                 // 00000000832C: D7600005 00013327
	v_readlane_b32 s6, v39, 26                                 // 000000008334: D7600006 00013527
	v_readlane_b32 s7, v39, 27                                 // 00000000833C: D7600007 00013727
	v_readlane_b32 s8, v39, 28                                 // 000000008344: D7600008 00013927
	v_readlane_b32 s9, v39, 29                                 // 00000000834C: D7600009 00013B27
	v_readlane_b32 s10, v39, 30                                // 000000008354: D760000A 00013D27
	v_readlane_b32 s11, v39, 31                                // 00000000835C: D760000B 00013F27
	s_or_saveexec_b32 s105, -1                                 // 000000008364: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000008368: BF870009
	s_mov_b32 exec_lo, s105                                    // 00000000836C: BEFE0069
	v_add_f32_e32 v19, s4, v19                                 // 000000008370: 06262604
	v_add_f32_e32 v23, s0, v23                                 // 000000008374: 062E2E00
	v_add_f32_e32 v21, s9, v21                                 // 000000008378: 062A2A09
	v_readlane_b32 s12, v44, 0                                 // 00000000837C: D760000C 0001012C
	v_readlane_b32 s13, v44, 1                                 // 000000008384: D760000D 0001032C
	v_add_f32_e32 v19, s5, v19                                 // 00000000838C: 06262605
	v_add_f32_e32 v23, s1, v23                                 // 000000008390: 062E2E01
	v_add_f32_e32 v21, s10, v21                                // 000000008394: 062A2A0A
	v_readlane_b32 s14, v44, 2                                 // 000000008398: D760000E 0001052C
	v_readlane_b32 s15, v44, 3                                 // 0000000083A0: D760000F 0001072C
	v_add_f32_e32 v19, s6, v19                                 // 0000000083A8: 06262606
	v_add_f32_e32 v23, s2, v23                                 // 0000000083AC: 062E2E02
	v_add_f32_e32 v21, s11, v21                                // 0000000083B0: 062A2A0B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000083B4: BF870193
	v_add_f32_e32 v19, s7, v19                                 // 0000000083B8: 06262607
	v_add_f32_e32 v23, s3, v23                                 // 0000000083BC: 062E2E03
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000083C0: BF870193
	v_add_f32_e32 v21, s12, v21                                // 0000000083C4: 062A2A0C
	v_add_f32_e32 v19, s8, v19                                 // 0000000083C8: 06262608
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000083CC: BF870193
	v_add_f32_e32 v23, s4, v23                                 // 0000000083D0: 062E2E04
	v_add_f32_e32 v21, s13, v21                                // 0000000083D4: 062A2A0D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000083D8: BF870193
	v_add_f32_e32 v19, s9, v19                                 // 0000000083DC: 06262609
	v_add_f32_e32 v23, s5, v23                                 // 0000000083E0: 062E2E05
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000083E4: BF870193
	v_add_f32_e32 v21, s14, v21                                // 0000000083E8: 062A2A0E
	v_add_f32_e32 v19, s10, v19                                // 0000000083EC: 0626260A
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000083F0: BF870193
	v_add_f32_e32 v23, s6, v23                                 // 0000000083F4: 062E2E06
	v_add_f32_e32 v21, s15, v21                                // 0000000083F8: 062A2A0F
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000083FC: BF8704A3
	v_add_f32_e32 v19, s11, v19                                // 000000008400: 0626260B
	s_or_saveexec_b32 s105, -1                                 // 000000008404: BEE922C1
	s_mov_b32 exec_lo, s105                                    // 000000008408: BEFE0069
	v_readlane_b32 s0, v40, 28                                 // 00000000840C: D7600000 00013928
	v_readlane_b32 s1, v40, 29                                 // 000000008414: D7600001 00013B28
	v_readlane_b32 s2, v40, 30                                 // 00000000841C: D7600002 00013D28
	v_readlane_b32 s3, v40, 31                                 // 000000008424: D7600003 00013F28
	s_or_saveexec_b32 s105, -1                                 // 00000000842C: BEE922C1
	scratch_load_b32 v44, off, off offset:56                   // 000000008430: DC510038 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 000000008438: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000843C: BF8903F7
	v_readlane_b32 s4, v44, 0                                  // 000000008440: D7600004 0001012C
	v_readlane_b32 s5, v44, 1                                  // 000000008448: D7600005 0001032C
	v_readlane_b32 s9, v44, 5                                  // 000000008450: D7600009 00010B2C
	v_readlane_b32 s6, v44, 2                                  // 000000008458: D7600006 0001052C
	v_readlane_b32 s10, v44, 6                                 // 000000008460: D760000A 00010D2C
	v_add_f32_e32 v18, s4, v18                                 // 000000008468: 06242404
	v_add_f32_e32 v22, s0, v22                                 // 00000000846C: 062C2C00
	v_add_f32_e32 v20, s9, v20                                 // 000000008470: 06282809
	v_readlane_b32 s7, v44, 3                                  // 000000008474: D7600007 0001072C
	v_readlane_b32 s11, v44, 7                                 // 00000000847C: D760000B 00010F2C
	v_add_f32_e32 v18, s5, v18                                 // 000000008484: 06242405
	v_add_f32_e32 v22, s1, v22                                 // 000000008488: 062C2C01
	v_add_f32_e32 v20, s10, v20                                // 00000000848C: 0628280A
	v_readlane_b32 s8, v44, 4                                  // 000000008490: D7600008 0001092C
	v_readlane_b32 s12, v44, 8                                 // 000000008498: D760000C 0001112C
	v_add_f32_e32 v18, s6, v18                                 // 0000000084A0: 06242406
	v_add_f32_e32 v22, s2, v22                                 // 0000000084A4: 062C2C02
	v_add_f32_e32 v20, s11, v20                                // 0000000084A8: 0628280B
	v_readlane_b32 s13, v44, 9                                 // 0000000084AC: D760000D 0001132C
	v_readlane_b32 s14, v44, 10                                // 0000000084B4: D760000E 0001152C
	v_add_f32_e32 v18, s7, v18                                 // 0000000084BC: 06242407
	v_add_f32_e32 v22, s3, v22                                 // 0000000084C0: 062C2C03
	v_add_f32_e32 v20, s12, v20                                // 0000000084C4: 0628280C
	v_readlane_b32 s15, v44, 11                                // 0000000084C8: D760000F 0001172C
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000084D0: BF870214
	v_add_f32_e32 v18, s8, v18                                 // 0000000084D4: 06242408
	v_add_f32_e32 v22, s4, v22                                 // 0000000084D8: 062C2C04
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000084DC: BF870194
	v_add_f32_e32 v20, s13, v20                                // 0000000084E0: 0628280D
	v_add_f32_e32 v18, s9, v18                                 // 0000000084E4: 06242409
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000084E8: BF870193
	v_add_f32_e32 v22, s5, v22                                 // 0000000084EC: 062C2C05
	v_add_f32_e32 v20, s14, v20                                // 0000000084F0: 0628280E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000084F4: BF870193
	v_add_f32_e32 v18, s10, v18                                // 0000000084F8: 0624240A
	v_add_f32_e32 v22, s6, v22                                 // 0000000084FC: 062C2C06
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000008500: BF870193
	v_add_f32_e32 v20, s15, v20                                // 000000008504: 0628280F
	v_add_f32_e32 v18, s11, v18                                // 000000008508: 0624240B
	s_or_saveexec_b32 s105, -1                                 // 00000000850C: BEE922C1
	scratch_load_b32 v44, off, off offset:4                    // 000000008510: DC510004 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 000000008518: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000851C: BF8903F7
	v_readlane_b32 s0, v44, 12                                 // 000000008520: D7600000 0001192C
	v_readlane_b32 s4, v44, 16                                 // 000000008528: D7600004 0001212C
	v_readlane_b32 s5, v44, 17                                 // 000000008530: D7600005 0001232C
	v_readlane_b32 s1, v44, 13                                 // 000000008538: D7600001 00011B2C
	v_readlane_b32 s6, v44, 18                                 // 000000008540: D7600006 0001252C
	v_readlane_b32 s9, v44, 21                                 // 000000008548: D7600009 00012B2C
	v_dual_add_f32 v16, s4, v16 :: v_dual_add_f32 v13, s0, v13 // 000000008550: C9082004 100C1A00
	v_readlane_b32 s7, v44, 19                                 // 000000008558: D7600007 0001272C
	v_readlane_b32 s10, v44, 22                                // 000000008560: D760000A 00012D2C
	v_readlane_b32 s2, v44, 14                                 // 000000008568: D7600002 00011D2C
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000008570: BF870244
	v_dual_add_f32 v16, s5, v16 :: v_dual_add_f32 v17, s9, v17 // 000000008574: C9082005 10102209
	v_add_f32_e32 v13, s1, v13                                 // 00000000857C: 061A1A01
	v_readlane_b32 s8, v44, 20                                 // 000000008580: D7600008 0001292C
	v_readlane_b32 s11, v44, 23                                // 000000008588: D760000B 00012F2C
	v_dual_add_f32 v16, s6, v16 :: v_dual_add_f32 v17, s10, v17// 000000008590: C9082006 1010220A
	v_readlane_b32 s3, v44, 15                                 // 000000008598: D7600003 00011F2C
	v_readlane_b32 s12, v44, 24                                // 0000000085A0: D760000C 0001312C
	v_readlane_b32 s13, v44, 25                                // 0000000085A8: D760000D 0001332C
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 0000000085B0: BF8701B4
	v_dual_add_f32 v16, s7, v16 :: v_dual_add_f32 v13, s2, v13 // 0000000085B4: C9082007 100C1A02
	v_readlane_b32 s14, v44, 26                                // 0000000085BC: D760000E 0001352C
	v_readlane_b32 s15, v44, 27                                // 0000000085C4: D760000F 0001372C
	v_dual_add_f32 v16, s8, v16 :: v_dual_add_f32 v17, s11, v17// 0000000085CC: C9082008 1010220B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000085D4: BF870091
	v_dual_add_f32 v16, s9, v16 :: v_dual_add_f32 v13, s3, v13 // 0000000085D8: C9082009 100C1A03
	v_dual_add_f32 v16, s10, v16 :: v_dual_add_f32 v17, s12, v17// 0000000085E0: C908200A 1010220C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000085E8: BF870111
	v_dual_add_f32 v13, s4, v13 :: v_dual_add_f32 v16, s11, v16// 0000000085EC: C9081A04 0D10200B
	v_add_f32_e32 v17, s13, v17                                // 0000000085F4: 0622220D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000085F8: BF870112
	v_add_f32_e32 v13, s5, v13                                 // 0000000085FC: 061A1A05
	v_add_f32_e32 v17, s14, v17                                // 000000008600: 0622220E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008604: BF870112
	v_add_f32_e32 v13, s6, v13                                 // 000000008608: 061A1A06
	v_add_f32_e32 v17, s15, v17                                // 00000000860C: 0622220F
	s_or_saveexec_b32 s105, -1                                 // 000000008610: BEE922C1
	scratch_load_b32 v40, off, off offset:28                   // 000000008614: DC51001C 287C0000
	s_mov_b32 exec_lo, s105                                    // 00000000861C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000008620: BF8903F7
	v_readlane_b32 s0, v40, 12                                 // 000000008624: D7600000 00011928
	v_readlane_b32 s4, v40, 16                                 // 00000000862C: D7600004 00012128
	v_readlane_b32 s9, v40, 21                                 // 000000008634: D7600009 00012B28
	v_readlane_b32 s10, v40, 22                                // 00000000863C: D760000A 00012D28
	v_readlane_b32 s5, v40, 17                                 // 000000008644: D7600005 00012328
	v_readlane_b32 s11, v40, 23                                // 00000000864C: D760000B 00012F28
	s_delay_alu instid0(VALU_DEP_4)                            // 000000008654: BF870004
	v_dual_add_f32 v14, s4, v14 :: v_dual_add_f32 v15, s9, v15 // 000000008658: C9081C04 0E0E1E09
	v_add_f32_e32 v12, s0, v12                                 // 000000008660: 06181800
	v_readlane_b32 s1, v40, 13                                 // 000000008664: D7600001 00011B28
	v_readlane_b32 s12, v40, 24                                // 00000000866C: D760000C 00013128
	v_readlane_b32 s6, v40, 18                                 // 000000008674: D7600006 00012528
	v_dual_add_f32 v15, s10, v15 :: v_dual_add_f32 v14, s5, v14// 00000000867C: C9081E0A 0F0E1C05
	v_readlane_b32 s13, v40, 25                                // 000000008684: D760000D 00013328
	v_readlane_b32 s2, v40, 14                                 // 00000000868C: D7600002 00011D28
	v_readlane_b32 s7, v40, 19                                 // 000000008694: D7600007 00012728
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 00000000869C: BF870244
	v_dual_add_f32 v15, s11, v15 :: v_dual_add_f32 v12, s1, v12// 0000000086A0: C9081E0B 0F0C1801
	v_readlane_b32 s14, v40, 26                                // 0000000086A8: D760000E 00013528
	v_readlane_b32 s3, v40, 15                                 // 0000000086B0: D7600003 00011F28
	v_readlane_b32 s8, v40, 20                                 // 0000000086B8: D7600008 00012928
	v_dual_add_f32 v15, s12, v15 :: v_dual_add_f32 v14, s6, v14// 0000000086C0: C9081E0C 0F0E1C06
	v_readlane_b32 s15, v40, 27                                // 0000000086C8: D760000F 00013728
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000086D0: BF870092
	v_dual_add_f32 v15, s13, v15 :: v_dual_add_f32 v12, s2, v12// 0000000086D4: C9081E0D 0F0C1802
	v_dual_add_f32 v15, s14, v15 :: v_dual_add_f32 v14, s7, v14// 0000000086DC: C9081E0E 0F0E1C07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000086E4: BF870111
	v_dual_add_f32 v12, s3, v12 :: v_dual_add_f32 v15, s15, v15// 0000000086E8: C9081803 0C0E1E0F
	v_add_f32_e32 v14, s8, v14                                 // 0000000086F0: 061C1C08
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000086F4: BF870112
	v_add_f32_e32 v12, s4, v12                                 // 0000000086F8: 06181804
	v_add_f32_e32 v14, s9, v14                                 // 0000000086FC: 061C1C09
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008700: BF870112
	v_add_f32_e32 v12, s5, v12                                 // 000000008704: 06181805
	v_add_f32_e32 v14, s10, v14                                // 000000008708: 061C1C0A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000870C: BF870112
	v_add_f32_e32 v12, s6, v12                                 // 000000008710: 06181806
	v_add_f32_e32 v14, s11, v14                                // 000000008714: 061C1C0B
	v_readlane_b32 s0, v44, 28                                 // 000000008718: D7600000 0001392C
	v_readlane_b32 s1, v44, 29                                 // 000000008720: D7600001 00013B2C
	v_readlane_b32 s2, v44, 30                                 // 000000008728: D7600002 00013D2C
	v_readlane_b32 s3, v44, 31                                 // 000000008730: D7600003 00013F2C
	s_or_saveexec_b32 s105, -1                                 // 000000008738: BEE922C1
	scratch_load_b32 v40, off, off offset:20                   // 00000000873C: DC510014 287C0000
	s_mov_b32 exec_lo, s105                                    // 000000008744: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000008748: BF8903F7
	v_readlane_b32 s4, v40, 0                                  // 00000000874C: D7600004 00010128
	v_readlane_b32 s5, v40, 1                                  // 000000008754: D7600005 00010328
	v_readlane_b32 s6, v40, 2                                  // 00000000875C: D7600006 00010528
	v_readlane_b32 s7, v40, 3                                  // 000000008764: D7600007 00010728
	v_readlane_b32 s8, v40, 4                                  // 00000000876C: D7600008 00010928
	v_readlane_b32 s9, v40, 5                                  // 000000008774: D7600009 00010B28
	v_readlane_b32 s10, v40, 6                                 // 00000000877C: D760000A 00010D28
	v_readlane_b32 s11, v40, 7                                 // 000000008784: D760000B 00010F28
	v_readlane_b32 s12, v40, 8                                 // 00000000878C: D760000C 00011128
	v_readlane_b32 s13, v40, 9                                 // 000000008794: D760000D 00011328
	v_readlane_b32 s14, v40, 10                                // 00000000879C: D760000E 00011528
	s_or_saveexec_b32 s105, -1                                 // 0000000087A4: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 0000000087A8: BF8701C9
	s_mov_b32 exec_lo, s105                                    // 0000000087AC: BEFE0069
	v_dual_add_f32 v24, s4, v24 :: v_dual_add_f32 v25, s9, v25 // 0000000087B0: C9083004 18183209
	v_add_f32_e32 v26, s0, v26                                 // 0000000087B8: 06343400
	v_readlane_b32 s15, v40, 11                                // 0000000087BC: D760000F 00011728
	v_dual_add_f32 v25, s10, v25 :: v_dual_add_f32 v24, s5, v24// 0000000087C4: C908320A 19183005
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000087CC: BF870091
	v_dual_add_f32 v25, s11, v25 :: v_dual_add_f32 v26, s1, v26// 0000000087D0: C908320B 191A3401
	v_dual_add_f32 v25, s12, v25 :: v_dual_add_f32 v24, s6, v24// 0000000087D8: C908320C 19183006
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000087E0: BF870091
	v_dual_add_f32 v25, s13, v25 :: v_dual_add_f32 v26, s2, v26// 0000000087E4: C908320D 191A3402
	v_dual_add_f32 v25, s14, v25 :: v_dual_add_f32 v24, s7, v24// 0000000087EC: C908320E 19183007
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000087F4: BF870111
	v_dual_add_f32 v26, s3, v26 :: v_dual_add_f32 v25, s15, v25// 0000000087F8: C9083403 1A18320F
	v_add_f32_e32 v24, s8, v24                                 // 000000008800: 06303008
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008804: BF870112
	v_add_f32_e32 v26, s4, v26                                 // 000000008808: 06343404
	v_add_f32_e32 v24, s9, v24                                 // 00000000880C: 06303009
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008810: BF870112
	v_add_f32_e32 v26, s5, v26                                 // 000000008814: 06343405
	v_add_f32_e32 v24, s10, v24                                // 000000008818: 0630300A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000881C: BF870112
	v_add_f32_e32 v26, s6, v26                                 // 000000008820: 06343406
	v_add_f32_e32 v24, s11, v24                                // 000000008824: 0630300B
	s_or_saveexec_b32 s105, -1                                 // 000000008828: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000882C: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000008830: BEFE0069
	v_readlane_b32 s0, v41, 4                                  // 000000008834: D7600000 00010929
	v_readlane_b32 s4, v41, 8                                  // 00000000883C: D7600004 00011129
	v_readlane_b32 s5, v41, 9                                  // 000000008844: D7600005 00011329
	v_readlane_b32 s9, v41, 13                                 // 00000000884C: D7600009 00011B29
	v_readlane_b32 s1, v41, 5                                  // 000000008854: D7600001 00010B29
	v_readlane_b32 s6, v41, 10                                 // 00000000885C: D7600006 00011529
	v_add_f32_e32 v19, s4, v19                                 // 000000008864: 06262604
	v_readlane_b32 s10, v41, 14                                // 000000008868: D760000A 00011D29
	v_add_f32_e32 v23, s0, v23                                 // 000000008870: 062E2E00
	v_add_f32_e32 v21, s9, v21                                 // 000000008874: 062A2A09
	v_readlane_b32 s2, v41, 6                                  // 000000008878: D7600002 00010D29
	v_add_f32_e32 v19, s5, v19                                 // 000000008880: 06262605
	v_readlane_b32 s7, v41, 11                                 // 000000008884: D7600007 00011729
	v_readlane_b32 s11, v41, 15                                // 00000000888C: D760000B 00011F29
	v_add_f32_e32 v23, s1, v23                                 // 000000008894: 062E2E01
	v_add_f32_e32 v21, s10, v21                                // 000000008898: 062A2A0A
	v_add_f32_e32 v19, s6, v19                                 // 00000000889C: 06262606
	v_readlane_b32 s3, v41, 7                                  // 0000000088A0: D7600003 00010F29
	v_readlane_b32 s8, v41, 12                                 // 0000000088A8: D7600008 00011929
	v_readlane_b32 s12, v41, 16                                // 0000000088B0: D760000C 00012129
	v_add_f32_e32 v23, s2, v23                                 // 0000000088B8: 062E2E02
	v_add_f32_e32 v19, s7, v19                                 // 0000000088BC: 06262607
	v_add_f32_e32 v21, s11, v21                                // 0000000088C0: 062A2A0B
	v_readlane_b32 s13, v41, 17                                // 0000000088C4: D760000D 00012329
	v_readlane_b32 s14, v41, 18                                // 0000000088CC: D760000E 00012529
	v_add_f32_e32 v23, s3, v23                                 // 0000000088D4: 062E2E03
	v_add_f32_e32 v19, s8, v19                                 // 0000000088D8: 06262608
	v_add_f32_e32 v21, s12, v21                                // 0000000088DC: 062A2A0C
	v_readlane_b32 s15, v41, 19                                // 0000000088E0: D760000F 00012729
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000088E8: BF870214
	v_add_f32_e32 v23, s4, v23                                 // 0000000088EC: 062E2E04
	v_add_f32_e32 v19, s9, v19                                 // 0000000088F0: 06262609
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000088F4: BF870194
	v_add_f32_e32 v21, s13, v21                                // 0000000088F8: 062A2A0D
	v_add_f32_e32 v23, s5, v23                                 // 0000000088FC: 062E2E05
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000008900: BF870193
	v_add_f32_e32 v19, s10, v19                                // 000000008904: 0626260A
	v_add_f32_e32 v21, s14, v21                                // 000000008908: 062A2A0E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000890C: BF870193
	v_add_f32_e32 v23, s6, v23                                 // 000000008910: 062E2E06
	v_add_f32_e32 v19, s11, v19                                // 000000008914: 0626260B
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000008918: BF8704A3
	v_add_f32_e32 v21, s15, v21                                // 00000000891C: 062A2A0F
	s_or_saveexec_b32 s105, -1                                 // 000000008920: BEE922C1
	s_mov_b32 exec_lo, s105                                    // 000000008924: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000008928: BEE922C1
	scratch_store_b32 off, v40, off offset:20                  // 00000000892C: DC690014 007C2800
	s_mov_b32 exec_lo, s105                                    // 000000008934: BEFE0069
	v_readlane_b32 s0, v40, 20                                 // 000000008938: D7600000 00012928
	v_readlane_b32 s1, v40, 21                                 // 000000008940: D7600001 00012B28
	v_readlane_b32 s2, v40, 22                                 // 000000008948: D7600002 00012D28
	v_readlane_b32 s3, v40, 23                                 // 000000008950: D7600003 00012F28
	v_readlane_b32 s4, v40, 24                                 // 000000008958: D7600004 00013128
	v_readlane_b32 s5, v40, 25                                 // 000000008960: D7600005 00013328
	v_readlane_b32 s6, v40, 26                                 // 000000008968: D7600006 00013528
	v_readlane_b32 s7, v40, 27                                 // 000000008970: D7600007 00013728
	v_readlane_b32 s8, v40, 28                                 // 000000008978: D7600008 00013928
	v_readlane_b32 s9, v40, 29                                 // 000000008980: D7600009 00013B28
	v_readlane_b32 s10, v40, 30                                // 000000008988: D760000A 00013D28
	v_readlane_b32 s11, v40, 31                                // 000000008990: D760000B 00013F28
	s_or_saveexec_b32 s105, -1                                 // 000000008998: BEE922C1
	scratch_load_b32 v44, off, off offset:24                   // 00000000899C: DC510018 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 0000000089A4: BEFE0069
	v_add_f32_e32 v18, s4, v18                                 // 0000000089A8: 06242404
	v_add_f32_e32 v22, s0, v22                                 // 0000000089AC: 062C2C00
	v_add_f32_e32 v20, s9, v20                                 // 0000000089B0: 06282809
	s_waitcnt vmcnt(0)                                         // 0000000089B4: BF8903F7
	v_readlane_b32 s12, v44, 0                                 // 0000000089B8: D760000C 0001012C
	v_readlane_b32 s13, v44, 1                                 // 0000000089C0: D760000D 0001032C
	v_add_f32_e32 v18, s5, v18                                 // 0000000089C8: 06242405
	v_add_f32_e32 v22, s1, v22                                 // 0000000089CC: 062C2C01
	v_add_f32_e32 v20, s10, v20                                // 0000000089D0: 0628280A
	v_readlane_b32 s14, v44, 2                                 // 0000000089D4: D760000E 0001052C
	v_readlane_b32 s15, v44, 3                                 // 0000000089DC: D760000F 0001072C
	v_add_f32_e32 v18, s6, v18                                 // 0000000089E4: 06242406
	v_add_f32_e32 v22, s2, v22                                 // 0000000089E8: 062C2C02
	v_add_f32_e32 v20, s11, v20                                // 0000000089EC: 0628280B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000089F0: BF870193
	v_add_f32_e32 v18, s7, v18                                 // 0000000089F4: 06242407
	v_add_f32_e32 v22, s3, v22                                 // 0000000089F8: 062C2C03
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000089FC: BF870193
	v_add_f32_e32 v20, s12, v20                                // 000000008A00: 0628280C
	v_add_f32_e32 v18, s8, v18                                 // 000000008A04: 06242408
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000008A08: BF870193
	v_add_f32_e32 v22, s4, v22                                 // 000000008A0C: 062C2C04
	v_add_f32_e32 v20, s13, v20                                // 000000008A10: 0628280D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000008A14: BF870193
	v_add_f32_e32 v18, s9, v18                                 // 000000008A18: 06242409
	v_add_f32_e32 v22, s5, v22                                 // 000000008A1C: 062C2C05
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000008A20: BF870193
	v_add_f32_e32 v20, s14, v20                                // 000000008A24: 0628280E
	v_add_f32_e32 v18, s10, v18                                // 000000008A28: 0624240A
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000008A2C: BF870193
	v_add_f32_e32 v22, s6, v22                                 // 000000008A30: 062C2C06
	v_add_f32_e32 v20, s15, v20                                // 000000008A34: 0628280F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000008A38: BF870003
	v_add_f32_e32 v18, s11, v18                                // 000000008A3C: 0624240B
	s_or_saveexec_b32 s105, -1                                 // 000000008A40: BEE922C1
	scratch_load_b32 v44, off, off offset:68                   // 000000008A44: DC510044 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 000000008A4C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000008A50: BF8903F7
	v_readlane_b32 s0, v44, 28                                 // 000000008A54: D7600000 0001392C
	v_readlane_b32 s4, v42, 0                                  // 000000008A5C: D7600004 0001012A
	v_readlane_b32 s5, v42, 1                                  // 000000008A64: D7600005 0001032A
	v_readlane_b32 s1, v44, 29                                 // 000000008A6C: D7600001 00013B2C
	v_readlane_b32 s6, v42, 2                                  // 000000008A74: D7600006 0001052A
	v_readlane_b32 s9, v42, 5                                  // 000000008A7C: D7600009 00010B2A
	v_dual_add_f32 v16, s4, v16 :: v_dual_add_f32 v13, s0, v13 // 000000008A84: C9082004 100C1A00
	v_readlane_b32 s7, v42, 3                                  // 000000008A8C: D7600007 0001072A
	v_readlane_b32 s10, v42, 6                                 // 000000008A94: D760000A 00010D2A
	v_readlane_b32 s2, v44, 30                                 // 000000008A9C: D7600002 00013D2C
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000008AA4: BF870244
	v_dual_add_f32 v16, s5, v16 :: v_dual_add_f32 v17, s9, v17 // 000000008AA8: C9082005 10102209
	v_add_f32_e32 v13, s1, v13                                 // 000000008AB0: 061A1A01
	v_readlane_b32 s8, v42, 4                                  // 000000008AB4: D7600008 0001092A
	v_readlane_b32 s11, v42, 7                                 // 000000008ABC: D760000B 00010F2A
	v_dual_add_f32 v16, s6, v16 :: v_dual_add_f32 v17, s10, v17// 000000008AC4: C9082006 1010220A
	v_readlane_b32 s3, v44, 31                                 // 000000008ACC: D7600003 00013F2C
	v_readlane_b32 s12, v42, 8                                 // 000000008AD4: D760000C 0001112A
	v_readlane_b32 s13, v42, 9                                 // 000000008ADC: D760000D 0001132A
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 000000008AE4: BF8701B4
	v_dual_add_f32 v16, s7, v16 :: v_dual_add_f32 v13, s2, v13 // 000000008AE8: C9082007 100C1A02
	v_readlane_b32 s14, v42, 10                                // 000000008AF0: D760000E 0001152A
	v_readlane_b32 s15, v42, 11                                // 000000008AF8: D760000F 0001172A
	v_dual_add_f32 v16, s8, v16 :: v_dual_add_f32 v17, s11, v17// 000000008B00: C9082008 1010220B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008B08: BF870091
	v_dual_add_f32 v16, s9, v16 :: v_dual_add_f32 v13, s3, v13 // 000000008B0C: C9082009 100C1A03
	v_dual_add_f32 v16, s10, v16 :: v_dual_add_f32 v17, s12, v17// 000000008B14: C908200A 1010220C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008B1C: BF870111
	v_dual_add_f32 v13, s4, v13 :: v_dual_add_f32 v16, s11, v16// 000000008B20: C9081A04 0D10200B
	v_add_f32_e32 v17, s13, v17                                // 000000008B28: 0622220D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008B2C: BF870112
	v_add_f32_e32 v13, s5, v13                                 // 000000008B30: 061A1A05
	v_add_f32_e32 v17, s14, v17                                // 000000008B34: 0622220E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008B38: BF870112
	v_add_f32_e32 v13, s6, v13                                 // 000000008B3C: 061A1A06
	v_add_f32_e32 v17, s15, v17                                // 000000008B40: 0622220F
	v_readlane_b32 s0, v42, 12                                 // 000000008B44: D7600000 0001192A
	v_readlane_b32 s4, v42, 16                                 // 000000008B4C: D7600004 0001212A
	v_readlane_b32 s5, v42, 17                                 // 000000008B54: D7600005 0001232A
	v_readlane_b32 s1, v42, 13                                 // 000000008B5C: D7600001 00011B2A
	v_readlane_b32 s6, v42, 18                                 // 000000008B64: D7600006 0001252A
	v_readlane_b32 s9, v42, 21                                 // 000000008B6C: D7600009 00012B2A
	v_add_f32_e32 v14, s4, v14                                 // 000000008B74: 061C1C04
	v_add_f32_e32 v12, s0, v12                                 // 000000008B78: 06181800
	v_readlane_b32 s2, v42, 14                                 // 000000008B7C: D7600002 00011D2A
	v_readlane_b32 s7, v42, 19                                 // 000000008B84: D7600007 0001272A
	v_readlane_b32 s10, v42, 22                                // 000000008B8C: D760000A 00012D2A
	v_add_f32_e32 v14, s5, v14                                 // 000000008B94: 061C1C05
	v_add_f32_e32 v12, s1, v12                                 // 000000008B98: 06181801
	v_readlane_b32 s3, v42, 15                                 // 000000008B9C: D7600003 00011F2A
	v_readlane_b32 s11, v42, 23                                // 000000008BA4: D760000B 00012F2A
	v_readlane_b32 s8, v42, 20                                 // 000000008BAC: D7600008 0001292A
	v_add_f32_e32 v14, s6, v14                                 // 000000008BB4: 061C1C06
	v_dual_add_f32 v12, s2, v12 :: v_dual_add_f32 v15, s9, v15 // 000000008BB8: C9081802 0C0E1E09
	v_readlane_b32 s12, v42, 24                                // 000000008BC0: D760000C 0001312A
	v_readlane_b32 s13, v42, 25                                // 000000008BC8: D760000D 0001332A
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000008BD0: BF870214
	v_add_f32_e32 v14, s7, v14                                 // 000000008BD4: 061C1C07
	v_dual_add_f32 v12, s3, v12 :: v_dual_add_f32 v15, s10, v15// 000000008BD8: C9081803 0C0E1E0A
	v_readlane_b32 s14, v42, 26                                // 000000008BE0: D760000E 0001352A
	v_readlane_b32 s15, v42, 27                                // 000000008BE8: D760000F 0001372A
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000008BF0: BF870214
	v_add_f32_e32 v14, s8, v14                                 // 000000008BF4: 061C1C08
	v_dual_add_f32 v12, s4, v12 :: v_dual_add_f32 v15, s11, v15// 000000008BF8: C9081804 0C0E1E0B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008C00: BF870091
	v_dual_add_f32 v15, s12, v15 :: v_dual_add_f32 v14, s9, v14// 000000008C04: C9081E0C 0F0E1C09
	v_dual_add_f32 v15, s13, v15 :: v_dual_add_f32 v12, s5, v12// 000000008C0C: C9081E0D 0F0C1805
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008C14: BF870091
	v_dual_add_f32 v15, s14, v15 :: v_dual_add_f32 v14, s10, v14// 000000008C18: C9081E0E 0F0E1C0A
	v_dual_add_f32 v12, s6, v12 :: v_dual_add_f32 v15, s15, v15// 000000008C20: C9081806 0C0E1E0F
	s_delay_alu instid0(VALU_DEP_2)                            // 000000008C28: BF870002
	v_add_f32_e32 v14, s11, v14                                // 000000008C2C: 061C1C0B
	v_readlane_b32 s0, v42, 28                                 // 000000008C30: D7600000 0001392A
	v_readlane_b32 s1, v42, 29                                 // 000000008C38: D7600001 00013B2A
	v_readlane_b32 s2, v42, 30                                 // 000000008C40: D7600002 00013D2A
	v_readlane_b32 s3, v42, 31                                 // 000000008C48: D7600003 00013F2A
	s_or_saveexec_b32 s105, -1                                 // 000000008C50: BEE922C1
	scratch_load_b32 v44, off, off offset:76                   // 000000008C54: DC51004C 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 000000008C5C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000008C60: BF8903F7
	v_readlane_b32 s4, v44, 0                                  // 000000008C64: D7600004 0001012C
	v_readlane_b32 s9, v44, 5                                  // 000000008C6C: D7600009 00010B2C
	v_readlane_b32 s10, v44, 6                                 // 000000008C74: D760000A 00010D2C
	v_readlane_b32 s5, v44, 1                                  // 000000008C7C: D7600005 0001032C
	v_readlane_b32 s11, v44, 7                                 // 000000008C84: D760000B 00010F2C
	s_delay_alu instid0(VALU_DEP_4)                            // 000000008C8C: BF870004
	v_dual_add_f32 v24, s4, v24 :: v_dual_add_f32 v25, s9, v25 // 000000008C90: C9083004 18183209
	v_add_f32_e32 v26, s0, v26                                 // 000000008C98: 06343400
	v_readlane_b32 s12, v44, 8                                 // 000000008C9C: D760000C 0001112C
	v_readlane_b32 s6, v44, 2                                  // 000000008CA4: D7600006 0001052C
	v_readlane_b32 s13, v44, 9                                 // 000000008CAC: D760000D 0001132C
	v_dual_add_f32 v25, s10, v25 :: v_dual_add_f32 v24, s5, v24// 000000008CB4: C908320A 19183005
	v_readlane_b32 s7, v44, 3                                  // 000000008CBC: D7600007 0001072C
	v_readlane_b32 s14, v44, 10                                // 000000008CC4: D760000E 0001152C
	v_readlane_b32 s8, v44, 4                                  // 000000008CCC: D7600008 0001092C
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000008CD4: BF870124
	v_dual_add_f32 v25, s11, v25 :: v_dual_add_f32 v26, s1, v26// 000000008CD8: C908320B 191A3401
	v_readlane_b32 s15, v44, 11                                // 000000008CE0: D760000F 0001172C
	v_dual_add_f32 v25, s12, v25 :: v_dual_add_f32 v24, s6, v24// 000000008CE8: C908320C 19183006
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008CF0: BF870091
	v_dual_add_f32 v25, s13, v25 :: v_dual_add_f32 v26, s2, v26// 000000008CF4: C908320D 191A3402
	v_dual_add_f32 v25, s14, v25 :: v_dual_add_f32 v24, s7, v24// 000000008CFC: C908320E 19183007
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008D04: BF870111
	v_dual_add_f32 v26, s3, v26 :: v_dual_add_f32 v25, s15, v25// 000000008D08: C9083403 1A18320F
	v_add_f32_e32 v24, s8, v24                                 // 000000008D10: 06303008
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008D14: BF870112
	v_add_f32_e32 v26, s4, v26                                 // 000000008D18: 06343404
	v_add_f32_e32 v24, s9, v24                                 // 000000008D1C: 06303009
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008D20: BF870112
	v_add_f32_e32 v26, s5, v26                                 // 000000008D24: 06343405
	v_add_f32_e32 v24, s10, v24                                // 000000008D28: 0630300A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008D2C: BF870112
	v_add_f32_e32 v26, s6, v26                                 // 000000008D30: 06343406
	v_add_f32_e32 v24, s11, v24                                // 000000008D34: 0630300B
	s_or_saveexec_b32 s105, -1                                 // 000000008D38: BEE922C1
	scratch_load_b32 v40, off, off offset:96                   // 000000008D3C: DC510060 287C0000
	s_mov_b32 exec_lo, s105                                    // 000000008D44: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000008D48: BF8903F7
	v_readlane_b32 s0, v40, 4                                  // 000000008D4C: D7600000 00010928
	v_readlane_b32 s4, v40, 8                                  // 000000008D54: D7600004 00011128
	v_readlane_b32 s5, v40, 9                                  // 000000008D5C: D7600005 00011328
	v_readlane_b32 s9, v40, 13                                 // 000000008D64: D7600009 00011B28
	v_readlane_b32 s1, v40, 5                                  // 000000008D6C: D7600001 00010B28
	v_readlane_b32 s6, v40, 10                                 // 000000008D74: D7600006 00011528
	v_add_f32_e32 v19, s4, v19                                 // 000000008D7C: 06262604
	v_readlane_b32 s10, v40, 14                                // 000000008D80: D760000A 00011D28
	v_add_f32_e32 v23, s0, v23                                 // 000000008D88: 062E2E00
	v_add_f32_e32 v21, s9, v21                                 // 000000008D8C: 062A2A09
	v_readlane_b32 s2, v40, 6                                  // 000000008D90: D7600002 00010D28
	v_add_f32_e32 v19, s5, v19                                 // 000000008D98: 06262605
	v_readlane_b32 s7, v40, 11                                 // 000000008D9C: D7600007 00011728
	v_readlane_b32 s11, v40, 15                                // 000000008DA4: D760000B 00011F28
	v_add_f32_e32 v23, s1, v23                                 // 000000008DAC: 062E2E01
	v_add_f32_e32 v21, s10, v21                                // 000000008DB0: 062A2A0A
	v_add_f32_e32 v19, s6, v19                                 // 000000008DB4: 06262606
	v_readlane_b32 s3, v40, 7                                  // 000000008DB8: D7600003 00010F28
	v_readlane_b32 s8, v40, 12                                 // 000000008DC0: D7600008 00011928
	v_readlane_b32 s12, v40, 16                                // 000000008DC8: D760000C 00012128
	v_add_f32_e32 v23, s2, v23                                 // 000000008DD0: 062E2E02
	v_add_f32_e32 v19, s7, v19                                 // 000000008DD4: 06262607
	v_add_f32_e32 v21, s11, v21                                // 000000008DD8: 062A2A0B
	v_readlane_b32 s13, v40, 17                                // 000000008DDC: D760000D 00012328
	v_readlane_b32 s14, v40, 18                                // 000000008DE4: D760000E 00012528
	v_add_f32_e32 v23, s3, v23                                 // 000000008DEC: 062E2E03
	v_add_f32_e32 v19, s8, v19                                 // 000000008DF0: 06262608
	v_add_f32_e32 v21, s12, v21                                // 000000008DF4: 062A2A0C
	v_readlane_b32 s15, v40, 19                                // 000000008DF8: D760000F 00012728
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000008E00: BF870214
	v_add_f32_e32 v23, s4, v23                                 // 000000008E04: 062E2E04
	v_add_f32_e32 v19, s9, v19                                 // 000000008E08: 06262609
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000008E0C: BF870194
	v_add_f32_e32 v21, s13, v21                                // 000000008E10: 062A2A0D
	v_add_f32_e32 v23, s5, v23                                 // 000000008E14: 062E2E05
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000008E18: BF870193
	v_add_f32_e32 v19, s10, v19                                // 000000008E1C: 0626260A
	v_add_f32_e32 v21, s14, v21                                // 000000008E20: 062A2A0E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000008E24: BF870193
	v_add_f32_e32 v23, s6, v23                                 // 000000008E28: 062E2E06
	v_add_f32_e32 v19, s11, v19                                // 000000008E2C: 0626260B
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000008E30: BF8704A3
	v_add_f32_e32 v21, s15, v21                                // 000000008E34: 062A2A0F
	s_or_saveexec_b32 s105, -1                                 // 000000008E38: BEE922C1
	s_mov_b32 exec_lo, s105                                    // 000000008E3C: BEFE0069
	v_readlane_b32 s0, v44, 12                                 // 000000008E40: D7600000 0001192C
	v_readlane_b32 s4, v44, 16                                 // 000000008E48: D7600004 0001212C
	v_readlane_b32 s5, v44, 17                                 // 000000008E50: D7600005 0001232C
	v_readlane_b32 s9, v44, 21                                 // 000000008E58: D7600009 00012B2C
	v_readlane_b32 s1, v44, 13                                 // 000000008E60: D7600001 00011B2C
	v_readlane_b32 s6, v44, 18                                 // 000000008E68: D7600006 0001252C
	v_add_f32_e32 v18, s4, v18                                 // 000000008E70: 06242404
	v_readlane_b32 s10, v44, 22                                // 000000008E74: D760000A 00012D2C
	v_add_f32_e32 v22, s0, v22                                 // 000000008E7C: 062C2C00
	v_add_f32_e32 v20, s9, v20                                 // 000000008E80: 06282809
	v_readlane_b32 s2, v44, 14                                 // 000000008E84: D7600002 00011D2C
	v_add_f32_e32 v18, s5, v18                                 // 000000008E8C: 06242405
	v_readlane_b32 s7, v44, 19                                 // 000000008E90: D7600007 0001272C
	v_readlane_b32 s11, v44, 23                                // 000000008E98: D760000B 00012F2C
	v_add_f32_e32 v22, s1, v22                                 // 000000008EA0: 062C2C01
	v_add_f32_e32 v20, s10, v20                                // 000000008EA4: 0628280A
	v_add_f32_e32 v18, s6, v18                                 // 000000008EA8: 06242406
	v_readlane_b32 s3, v44, 15                                 // 000000008EAC: D7600003 00011F2C
	v_readlane_b32 s8, v44, 20                                 // 000000008EB4: D7600008 0001292C
	v_readlane_b32 s12, v44, 24                                // 000000008EBC: D760000C 0001312C
	v_add_f32_e32 v22, s2, v22                                 // 000000008EC4: 062C2C02
	v_add_f32_e32 v18, s7, v18                                 // 000000008EC8: 06242407
	v_add_f32_e32 v20, s11, v20                                // 000000008ECC: 0628280B
	v_readlane_b32 s13, v44, 25                                // 000000008ED0: D760000D 0001332C
	v_readlane_b32 s14, v44, 26                                // 000000008ED8: D760000E 0001352C
	v_add_f32_e32 v22, s3, v22                                 // 000000008EE0: 062C2C03
	v_add_f32_e32 v18, s8, v18                                 // 000000008EE4: 06242408
	v_add_f32_e32 v20, s12, v20                                // 000000008EE8: 0628280C
	v_readlane_b32 s15, v44, 27                                // 000000008EEC: D760000F 0001372C
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000008EF4: BF870214
	v_add_f32_e32 v22, s4, v22                                 // 000000008EF8: 062C2C04
	v_add_f32_e32 v18, s9, v18                                 // 000000008EFC: 06242409
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000008F00: BF870194
	v_add_f32_e32 v20, s13, v20                                // 000000008F04: 0628280D
	v_add_f32_e32 v22, s5, v22                                 // 000000008F08: 062C2C05
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000008F0C: BF870193
	v_add_f32_e32 v18, s10, v18                                // 000000008F10: 0624240A
	v_add_f32_e32 v20, s14, v20                                // 000000008F14: 0628280E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000008F18: BF870193
	v_add_f32_e32 v22, s6, v22                                 // 000000008F1C: 062C2C06
	v_add_f32_e32 v18, s11, v18                                // 000000008F20: 0624240B
	s_delay_alu instid0(VALU_DEP_3)                            // 000000008F24: BF870003
	v_add_f32_e32 v20, s15, v20                                // 000000008F28: 0628280F
	s_or_saveexec_b32 s105, -1                                 // 000000008F2C: BEE922C1
	scratch_load_b32 v39, off, off offset:100                  // 000000008F30: DC510064 277C0000
	s_mov_b32 exec_lo, s105                                    // 000000008F38: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000008F3C: BF8903F7
	v_readlane_b32 s0, v39, 20                                 // 000000008F40: D7600000 00012927
	v_readlane_b32 s4, v39, 24                                 // 000000008F48: D7600004 00013127
	v_readlane_b32 s5, v39, 25                                 // 000000008F50: D7600005 00013327
	v_readlane_b32 s1, v39, 21                                 // 000000008F58: D7600001 00012B27
	v_readlane_b32 s6, v39, 26                                 // 000000008F60: D7600006 00013527
	v_readlane_b32 s9, v39, 29                                 // 000000008F68: D7600009 00013B27
	v_dual_add_f32 v16, s4, v16 :: v_dual_add_f32 v13, s0, v13 // 000000008F70: C9082004 100C1A00
	v_readlane_b32 s7, v39, 27                                 // 000000008F78: D7600007 00013727
	v_readlane_b32 s10, v39, 30                                // 000000008F80: D760000A 00013D27
	v_readlane_b32 s2, v39, 22                                 // 000000008F88: D7600002 00012D27
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000008F90: BF870244
	v_dual_add_f32 v16, s5, v16 :: v_dual_add_f32 v17, s9, v17 // 000000008F94: C9082005 10102209
	v_add_f32_e32 v13, s1, v13                                 // 000000008F9C: 061A1A01
	v_readlane_b32 s8, v39, 28                                 // 000000008FA0: D7600008 00013927
	v_readlane_b32 s11, v39, 31                                // 000000008FA8: D760000B 00013F27
	v_dual_add_f32 v16, s6, v16 :: v_dual_add_f32 v17, s10, v17// 000000008FB0: C9082006 1010220A
	v_readlane_b32 s3, v39, 23                                 // 000000008FB8: D7600003 00012F27
	v_readlane_b32 s12, v41, 0                                 // 000000008FC0: D760000C 00010129
	v_readlane_b32 s13, v41, 1                                 // 000000008FC8: D760000D 00010329
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 000000008FD0: BF8701B4
	v_dual_add_f32 v16, s7, v16 :: v_dual_add_f32 v13, s2, v13 // 000000008FD4: C9082007 100C1A02
	v_readlane_b32 s14, v41, 2                                 // 000000008FDC: D760000E 00010529
	v_readlane_b32 s15, v41, 3                                 // 000000008FE4: D760000F 00010729
	v_dual_add_f32 v16, s8, v16 :: v_dual_add_f32 v17, s11, v17// 000000008FEC: C9082008 1010220B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008FF4: BF870091
	v_dual_add_f32 v16, s9, v16 :: v_dual_add_f32 v13, s3, v13 // 000000008FF8: C9082009 100C1A03
	v_dual_add_f32 v16, s10, v16 :: v_dual_add_f32 v17, s12, v17// 000000009000: C908200A 1010220C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009008: BF870111
	v_dual_add_f32 v13, s4, v13 :: v_dual_add_f32 v16, s11, v16// 00000000900C: C9081A04 0D10200B
	v_add_f32_e32 v17, s13, v17                                // 000000009014: 0622220D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009018: BF870112
	v_add_f32_e32 v13, s5, v13                                 // 00000000901C: 061A1A05
	v_add_f32_e32 v17, s14, v17                                // 000000009020: 0622220E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009024: BF870112
	v_add_f32_e32 v13, s6, v13                                 // 000000009028: 061A1A06
	v_add_f32_e32 v17, s15, v17                                // 00000000902C: 0622220F
	v_readlane_b32 s0, v40, 20                                 // 000000009030: D7600000 00012928
	v_readlane_b32 s4, v40, 24                                 // 000000009038: D7600004 00013128
	v_readlane_b32 s5, v40, 25                                 // 000000009040: D7600005 00013328
	v_readlane_b32 s1, v40, 21                                 // 000000009048: D7600001 00012B28
	v_readlane_b32 s6, v40, 26                                 // 000000009050: D7600006 00013528
	v_readlane_b32 s9, v40, 29                                 // 000000009058: D7600009 00013B28
	v_add_f32_e32 v14, s4, v14                                 // 000000009060: 061C1C04
	v_add_f32_e32 v12, s0, v12                                 // 000000009064: 06181800
	v_readlane_b32 s2, v40, 22                                 // 000000009068: D7600002 00012D28
	v_readlane_b32 s7, v40, 27                                 // 000000009070: D7600007 00013728
	v_readlane_b32 s10, v40, 30                                // 000000009078: D760000A 00013D28
	v_add_f32_e32 v14, s5, v14                                 // 000000009080: 061C1C05
	v_add_f32_e32 v12, s1, v12                                 // 000000009084: 06181801
	v_readlane_b32 s3, v40, 23                                 // 000000009088: D7600003 00012F28
	v_readlane_b32 s11, v40, 31                                // 000000009090: D760000B 00013F28
	v_readlane_b32 s8, v40, 28                                 // 000000009098: D7600008 00013928
	v_add_f32_e32 v14, s6, v14                                 // 0000000090A0: 061C1C06
	v_dual_add_f32 v12, s2, v12 :: v_dual_add_f32 v15, s9, v15 // 0000000090A4: C9081802 0C0E1E09
	v_readlane_b32 s12, v39, 0                                 // 0000000090AC: D760000C 00010127
	v_readlane_b32 s13, v39, 1                                 // 0000000090B4: D760000D 00010327
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000090BC: BF870214
	v_add_f32_e32 v14, s7, v14                                 // 0000000090C0: 061C1C07
	v_dual_add_f32 v12, s3, v12 :: v_dual_add_f32 v15, s10, v15// 0000000090C4: C9081803 0C0E1E0A
	v_readlane_b32 s14, v39, 2                                 // 0000000090CC: D760000E 00010527
	v_readlane_b32 s15, v39, 3                                 // 0000000090D4: D760000F 00010727
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000090DC: BF870214
	v_add_f32_e32 v14, s8, v14                                 // 0000000090E0: 061C1C08
	v_dual_add_f32 v12, s4, v12 :: v_dual_add_f32 v15, s11, v15// 0000000090E4: C9081804 0C0E1E0B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000090EC: BF870091
	v_dual_add_f32 v15, s12, v15 :: v_dual_add_f32 v14, s9, v14// 0000000090F0: C9081E0C 0F0E1C09
	v_dual_add_f32 v15, s13, v15 :: v_dual_add_f32 v12, s5, v12// 0000000090F8: C9081E0D 0F0C1805
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009100: BF870091
	v_dual_add_f32 v15, s14, v15 :: v_dual_add_f32 v14, s10, v14// 000000009104: C9081E0E 0F0E1C0A
	v_dual_add_f32 v12, s6, v12 :: v_dual_add_f32 v15, s15, v15// 00000000910C: C9081806 0C0E1E0F
	s_delay_alu instid0(VALU_DEP_2)                            // 000000009114: BF870002
	v_add_f32_e32 v14, s11, v14                                // 000000009118: 061C1C0B
	v_readlane_b32 s0, v41, 20                                 // 00000000911C: D7600000 00012929
	v_readlane_b32 s1, v41, 21                                 // 000000009124: D7600001 00012B29
	v_readlane_b32 s2, v41, 22                                 // 00000000912C: D7600002 00012D29
	v_readlane_b32 s3, v41, 23                                 // 000000009134: D7600003 00012F29
	v_readlane_b32 s4, v41, 24                                 // 00000000913C: D7600004 00013129
	v_readlane_b32 s5, v41, 25                                 // 000000009144: D7600005 00013329
	v_readlane_b32 s6, v41, 26                                 // 00000000914C: D7600006 00013529
	v_readlane_b32 s7, v41, 27                                 // 000000009154: D7600007 00013729
	v_readlane_b32 s8, v41, 28                                 // 00000000915C: D7600008 00013929
	v_readlane_b32 s9, v41, 29                                 // 000000009164: D7600009 00013B29
	v_readlane_b32 s10, v41, 30                                // 00000000916C: D760000A 00013D29
	v_readlane_b32 s11, v41, 31                                // 000000009174: D760000B 00013F29
	s_or_saveexec_b32 s105, -1                                 // 00000000917C: BEE922C1
	scratch_load_b32 v42, off, off offset:104                  // 000000009180: DC510068 2A7C0000
	s_mov_b32 exec_lo, s105                                    // 000000009188: BEFE0069
	v_dual_add_f32 v24, s4, v24 :: v_dual_add_f32 v25, s9, v25 // 00000000918C: C9083004 18183209
	v_add_f32_e32 v26, s0, v26                                 // 000000009194: 06343400
	s_waitcnt vmcnt(0)                                         // 000000009198: BF8903F7
	v_readlane_b32 s12, v42, 0                                 // 00000000919C: D760000C 0001012A
	v_readlane_b32 s13, v42, 1                                 // 0000000091A4: D760000D 0001032A
	v_readlane_b32 s14, v42, 2                                 // 0000000091AC: D760000E 0001052A
	v_dual_add_f32 v25, s10, v25 :: v_dual_add_f32 v24, s5, v24// 0000000091B4: C908320A 19183005
	v_readlane_b32 s15, v42, 3                                 // 0000000091BC: D760000F 0001072A
	v_add_f32_e32 v19, s72, v19                                // 0000000091C4: 06262648
	v_add_f32_e32 v23, s68, v23                                // 0000000091C8: 062E2E44
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000091CC: BF870224
	v_dual_add_f32 v25, s11, v25 :: v_dual_add_f32 v26, s1, v26// 0000000091D0: C908320B 191A3401
	v_add_f32_e32 v21, s77, v21                                // 0000000091D8: 062A2A4D
	v_add_f32_e32 v19, s73, v19                                // 0000000091DC: 06262649
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000091E0: BF870214
	v_add_f32_e32 v23, s69, v23                                // 0000000091E4: 062E2E45
	v_dual_add_f32 v25, s12, v25 :: v_dual_add_f32 v24, s6, v24// 0000000091E8: C908320C 19183006
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000091F0: BF870214
	v_add_f32_e32 v21, s78, v21                                // 0000000091F4: 062A2A4E
	v_add_f32_e32 v19, s74, v19                                // 0000000091F8: 0626264A
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000091FC: BF870214
	v_add_f32_e32 v23, s70, v23                                // 000000009200: 062E2E46
	v_dual_add_f32 v25, s13, v25 :: v_dual_add_f32 v26, s2, v26// 000000009204: C908320D 191A3402
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000920C: BF870091
	v_dual_add_f32 v25, s14, v25 :: v_dual_add_f32 v24, s7, v24// 000000009210: C908320E 19183007
	v_dual_add_f32 v26, s3, v26 :: v_dual_add_f32 v25, s15, v25// 000000009218: C9083403 1A18320F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009220: BF870112
	v_add_f32_e32 v24, s8, v24                                 // 000000009224: 06303008
	v_add_f32_e32 v26, s4, v26                                 // 000000009228: 06343404
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000922C: BF870112
	v_add_f32_e32 v24, s9, v24                                 // 000000009230: 06303009
	v_add_f32_e32 v26, s5, v26                                 // 000000009234: 06343405
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009238: BF870112
	v_add_f32_e32 v24, s10, v24                                // 00000000923C: 0630300A
	v_add_f32_e32 v26, s6, v26                                 // 000000009240: 06343406
	s_delay_alu instid0(VALU_DEP_2)                            // 000000009244: BF870002
	v_add_f32_e32 v24, s11, v24                                // 000000009248: 0630300B
	v_readlane_b32 s0, v42, 12                                 // 00000000924C: D7600000 0001192A
	v_readlane_b32 s4, v42, 16                                 // 000000009254: D7600004 0001212A
	v_add_f32_e32 v19, s75, v19                                // 00000000925C: 0626264B
	v_readlane_b32 s9, v42, 21                                 // 000000009260: D7600009 00012B2A
	v_readlane_b32 s7, v42, 19                                 // 000000009268: D7600007 0001272A
	v_readlane_b32 s5, v42, 17                                 // 000000009270: D7600005 0001232A
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000009278: BF870214
	v_dual_add_f32 v18, s4, v18 :: v_dual_add_f32 v19, s76, v19// 00000000927C: C9082404 1212264C
	v_dual_add_f32 v20, s9, v20 :: v_dual_add_f32 v23, s71, v23// 000000009284: C9082809 14162E47
	v_readlane_b32 s10, v42, 22                                // 00000000928C: D760000A 00012D2A
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000009294: BF8701A3
	v_dual_add_f32 v18, s5, v18 :: v_dual_add_f32 v19, s77, v19// 000000009298: C9082405 1212264D
	v_readlane_b32 s12, v42, 24                                // 0000000092A0: D760000C 0001312A
	v_dual_add_f32 v23, s72, v23 :: v_dual_add_f32 v20, s10, v20// 0000000092A8: C9082E48 1714280A
	v_readlane_b32 s6, v42, 18                                 // 0000000092B0: D7600006 0001252A
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)// 0000000092B8: BF870254
	v_add_f32_e32 v19, s78, v19                                // 0000000092BC: 0626264E
	v_add_f32_e32 v21, s79, v21                                // 0000000092C0: 062A2A4F
	v_readlane_b32 s11, v42, 23                                // 0000000092C4: D760000B 00012F2A
	v_readlane_b32 s8, v42, 20                                 // 0000000092CC: D7600008 0001292A
	v_add_f32_e32 v18, s6, v18                                 // 0000000092D4: 06242406
	v_dual_add_f32 v28, s79, v19 :: v_dual_add_f32 v21, s80, v21// 0000000092D8: C908264F 1C142A50
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000092E0: BF870224
	v_add_f32_e32 v20, s11, v20                                // 0000000092E4: 0628280B
	v_readlane_b32 s13, v42, 25                                // 0000000092E8: D760000D 0001332A
	v_dual_add_f32 v18, s7, v18 :: v_dual_add_f32 v23, s73, v23// 0000000092F0: C9082407 12162E49
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 0000000092F8: BF870233
	v_dual_add_f32 v21, s81, v21 :: v_dual_add_f32 v20, s12, v20// 0000000092FC: C9082A51 1514280C
	v_readlane_b32 s1, v42, 13                                 // 000000009304: D7600001 00011B2A
	v_readlane_b32 s14, v42, 26                                // 00000000930C: D760000E 0001352A
	v_dual_add_f32 v18, s8, v18 :: v_dual_add_f32 v27, s74, v23// 000000009314: C9082408 121A2E4A
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 00000000931C: BF870244
	v_dual_add_f32 v20, s13, v20 :: v_dual_add_f32 v19, s0, v22// 000000009320: C908280D 14122C00
	v_readlane_b32 s2, v42, 14                                 // 000000009328: D7600002 00011D2A
	v_readlane_b32 s15, v42, 27                                // 000000009330: D760000F 0001372A
	v_readlane_b32 s3, v42, 15                                 // 000000009338: D7600003 00011F2A
	v_dual_add_f32 v20, s14, v20 :: v_dual_add_f32 v19, s1, v19// 000000009340: C908280E 14122601
	v_dual_add_f32 v21, s82, v21 :: v_dual_add_f32 v18, s9, v18// 000000009348: C9082A52 15122409
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009350: BF870112
	v_dual_add_f32 v20, s15, v20 :: v_dual_add_f32 v19, s2, v19// 000000009354: C908280F 14122602
	v_dual_add_f32 v21, s83, v21 :: v_dual_add_f32 v18, s10, v18// 00000000935C: C9082A53 1512240A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009364: BF870112
	v_add_f32_e32 v19, s3, v19                                 // 000000009368: 06262603
	v_add_f32_e32 v29, s11, v18                                // 00000000936C: 063A240B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009370: BF870092
	v_add_f32_e32 v19, s4, v19                                 // 000000009374: 06262604
	v_add_f32_e32 v19, s5, v19                                 // 000000009378: 06262605
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000937C: BF870001
	v_add_f32_e32 v22, s6, v19                                 // 000000009380: 062C2606
	s_or_saveexec_b32 s105, -1                                 // 000000009384: BEE922C1
	scratch_load_b32 v37, off, off offset:80                   // 000000009388: DC510050 257C0000
	s_mov_b32 exec_lo, s105                                    // 000000009390: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000009394: BF8903F7
	v_readlane_b32 s0, v37, 4                                  // 000000009398: D7600000 00010925
	v_readlane_b32 s4, v37, 8                                  // 0000000093A0: D7600004 00011125
	v_readlane_b32 s5, v37, 9                                  // 0000000093A8: D7600005 00011325
	v_readlane_b32 s1, v37, 5                                  // 0000000093B0: D7600001 00010B25
	v_readlane_b32 s6, v37, 10                                 // 0000000093B8: D7600006 00011525
	v_readlane_b32 s9, v37, 13                                 // 0000000093C0: D7600009 00011B25
	v_dual_add_f32 v16, s4, v16 :: v_dual_add_f32 v13, s0, v13 // 0000000093C8: C9082004 100C1A00
	v_readlane_b32 s7, v37, 11                                 // 0000000093D0: D7600007 00011725
	v_readlane_b32 s10, v37, 14                                // 0000000093D8: D760000A 00011D25
	v_readlane_b32 s2, v37, 6                                  // 0000000093E0: D7600002 00010D25
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 0000000093E8: BF870244
	v_dual_add_f32 v16, s5, v16 :: v_dual_add_f32 v17, s9, v17 // 0000000093EC: C9082005 10102209
	v_add_f32_e32 v13, s1, v13                                 // 0000000093F4: 061A1A01
	v_readlane_b32 s8, v37, 12                                 // 0000000093F8: D7600008 00011925
	v_readlane_b32 s11, v37, 15                                // 000000009400: D760000B 00011F25
	v_dual_add_f32 v16, s6, v16 :: v_dual_add_f32 v17, s10, v17// 000000009408: C9082006 1010220A
	v_readlane_b32 s3, v37, 7                                  // 000000009410: D7600003 00010F25
	v_readlane_b32 s12, v37, 16                                // 000000009418: D760000C 00012125
	v_readlane_b32 s13, v37, 17                                // 000000009420: D760000D 00012325
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000009428: BF870244
	v_dual_add_f32 v16, s7, v16 :: v_dual_add_f32 v13, s2, v13 // 00000000942C: C9082007 100C1A02
	v_readlane_b32 s14, v37, 18                                // 000000009434: D760000E 00012525
	v_readlane_b32 s15, v37, 19                                // 00000000943C: D760000F 00012725
	v_add_f32_e32 v18, s61, v21                                // 000000009444: 06242A3D
	v_dual_add_f32 v16, s8, v16 :: v_dual_add_f32 v17, s11, v17// 000000009448: C9082008 1010220B
	v_add_f32_e32 v12, s36, v12                                // 000000009450: 06181824
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000009454: BF870193
	v_add_f32_e32 v18, s62, v18                                // 000000009458: 0624243E
	v_dual_add_f32 v16, s9, v16 :: v_dual_add_f32 v13, s3, v13 // 00000000945C: C9082009 100C1A03
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000009464: BF870193
	v_dual_add_f32 v17, s12, v17 :: v_dual_add_f32 v12, s37, v12// 000000009468: C908220C 110C1825
	v_add_f32_e32 v18, s63, v18                                // 000000009470: 0624243F
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000009474: BF870193
	v_dual_add_f32 v16, s10, v16 :: v_dual_add_f32 v13, s4, v13// 000000009478: C908200A 100C1A04
	v_dual_add_f32 v17, s13, v17 :: v_dual_add_f32 v12, s38, v12// 000000009480: C908220D 110C1826
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000009488: BF870192
	v_dual_add_f32 v18, s64, v18 :: v_dual_add_f32 v31, s11, v16// 00000000948C: C9082440 121E200B
	v_add_f32_e32 v13, s5, v13                                 // 000000009494: 061A1A05
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000009498: BF870193
	v_dual_add_f32 v17, s14, v17 :: v_dual_add_f32 v12, s39, v12// 00000000949C: C908220E 110C1827
	v_add_f32_e32 v18, s65, v18                                // 0000000094A4: 06242441
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000094A8: BF870193
	v_add_f32_e32 v30, s6, v13                                 // 0000000094AC: 063C1A06
	v_dual_add_f32 v32, s15, v17 :: v_dual_add_f32 v13, s40, v14// 0000000094B0: C908220F 200C1C28
	v_add_f32_e32 v14, s45, v15                                // 0000000094B8: 061C1E2D
	v_readlane_b32 s0, v37, 28                                 // 0000000094BC: D7600000 00013925
	v_readlane_b32 s4, v43, 0                                  // 0000000094C4: D7600004 0001012B
	v_readlane_b32 s1, v37, 29                                 // 0000000094CC: D7600001 00013B25
	v_readlane_b32 s5, v43, 1                                  // 0000000094D4: D7600005 0001032B
	v_dual_add_f32 v14, s46, v14 :: v_dual_add_f32 v13, s41, v13// 0000000094DC: C9081C2E 0E0C1A29
	v_readlane_b32 s2, v37, 30                                 // 0000000094E4: D7600002 00013D25
	v_readlane_b32 s9, v43, 5                                  // 0000000094EC: D7600009 00010B2B
	v_readlane_b32 s3, v37, 31                                 // 0000000094F4: D7600003 00013F25
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 0000000094FC: BF870244
	v_dual_add_f32 v14, s47, v14 :: v_dual_add_f32 v13, s42, v13// 000000009500: C9081C2F 0E0C1A2A
	v_readlane_b32 s6, v43, 2                                  // 000000009508: D7600006 0001052B
	v_readlane_b32 s10, v43, 6                                 // 000000009510: D760000A 00010D2B
	v_readlane_b32 s7, v43, 3                                  // 000000009518: D7600007 0001072B
	v_dual_add_f32 v14, s48, v14 :: v_dual_add_f32 v13, s43, v13// 000000009520: C9081C30 0E0C1A2B
	v_add_f32_e32 v12, s40, v12                                // 000000009528: 06181828
	v_readlane_b32 s11, v43, 7                                 // 00000000952C: D760000B 00010F2B
	v_readlane_b32 s12, v43, 8                                 // 000000009534: D760000C 0001112B
	v_readlane_b32 s13, v43, 9                                 // 00000000953C: D760000D 0001132B
	v_dual_add_f32 v13, s44, v13 :: v_dual_add_f32 v14, s49, v14// 000000009544: C9081A2C 0D0E1C31
	v_readlane_b32 s8, v43, 4                                  // 00000000954C: D7600008 0001092B
	v_readlane_b32 s14, v43, 10                                // 000000009554: D760000E 0001152B
	v_readlane_b32 s15, v43, 11                                // 00000000955C: D760000F 0001172B
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000009564: BF870244
	v_dual_add_f32 v13, s45, v13 :: v_dual_add_f32 v12, s41, v12// 000000009568: C9081A2D 0D0C1829
	v_add_f32_e32 v15, s0, v26                                 // 000000009570: 061E3400
	v_add_f32_e32 v17, s52, v27                                // 000000009574: 06223634
	v_add_f32_e32 v21, s66, v18                                // 000000009578: 062A2442
	v_dual_add_f32 v13, s46, v13 :: v_dual_add_f32 v14, s50, v14// 00000000957C: C9081A2E 0D0E1C32
	v_add_f32_e32 v19, s42, v12                                // 000000009584: 0626182A
	v_dual_add_f32 v15, s1, v15 :: v_dual_add_f32 v16, s9, v25 // 000000009588: C9081E01 0F103209
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009590: BF870113
	v_dual_add_f32 v13, s47, v13 :: v_dual_add_f32 v12, s51, v14// 000000009594: C9081A2F 0D0C1C33
	v_dual_add_f32 v14, s4, v24 :: v_dual_add_f32 v15, s2, v15 // 00000000959C: C9083004 0E0E1E02
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000095A4: BF870111
	v_dual_add_f32 v17, s53, v17 :: v_dual_add_f32 v14, s5, v14// 0000000095A8: C9082235 110E1C05
	v_dual_add_f32 v15, s3, v15 :: v_dual_add_f32 v16, s10, v16// 0000000095B0: C9081E03 0F10200A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000095B8: BF870112
	v_dual_add_f32 v17, s54, v17 :: v_dual_add_f32 v14, s6, v14// 0000000095BC: C9082236 110E1C06
	v_add_f32_e32 v15, s4, v15                                 // 0000000095C4: 061E1E04
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000095C8: BF870112
	v_dual_add_f32 v17, s55, v17 :: v_dual_add_f32 v14, s7, v14// 0000000095CC: C9082237 110E1C07
	v_dual_add_f32 v15, s5, v15 :: v_dual_add_f32 v16, s11, v16// 0000000095D4: C9081E05 0F10200B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000095DC: BF870112
	v_add_f32_e32 v17, s56, v17                                // 0000000095E0: 06222238
	v_dual_add_f32 v23, s6, v15 :: v_dual_add_f32 v16, s12, v16// 0000000095E4: C9081E06 1710200C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000095EC: BF870091
	v_dual_add_f32 v17, s57, v17 :: v_dual_add_f32 v16, s13, v16// 0000000095F0: C9082239 1110200D
	v_add_f32_e32 v24, s58, v17                                // 0000000095F8: 0630223A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000095FC: BF870092
	v_dual_add_f32 v17, s67, v21 :: v_dual_add_f32 v16, s14, v16// 000000009600: C9082A43 1110200E
	v_dual_add_f32 v14, s8, v14 :: v_dual_add_f32 v15, s15, v16// 000000009608: C9081C08 0E0E200F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000009610: BF870121
	v_add_f32_e32 v14, s9, v14                                 // 000000009614: 061C1C09
	v_add_f32_e32 v16, s56, v28                                // 000000009618: 06203838
	v_add_f32_e32 v14, s10, v14                                // 00000000961C: 061C1C0A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009620: BF870112
	v_add_f32_e32 v16, s57, v16                                // 000000009624: 06202039
	v_add_f32_e32 v14, s11, v14                                // 000000009628: 061C1C0B
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000962C: BF870002
	v_add_f32_e32 v16, s58, v16                                // 000000009630: 0620203A
	v_readlane_b32 s0, v43, 20                                 // 000000009634: D7600000 0001292B
	v_readlane_b32 s1, v43, 21                                 // 00000000963C: D7600001 00012B2B
	v_readlane_b32 s2, v43, 22                                 // 000000009644: D7600002 00012D2B
	v_readlane_b32 s3, v43, 23                                 // 00000000964C: D7600003 00012F2B
	v_readlane_b32 s4, v43, 24                                 // 000000009654: D7600004 0001312B
	v_readlane_b32 s5, v43, 25                                 // 00000000965C: D7600005 0001332B
	v_readlane_b32 s6, v43, 26                                 // 000000009664: D7600006 0001352B
	v_readlane_b32 s7, v43, 27                                 // 00000000966C: D7600007 0001372B
	v_readlane_b32 s8, v43, 28                                 // 000000009674: D7600008 0001392B
	v_readlane_b32 s9, v43, 29                                 // 00000000967C: D7600009 00013B2B
	v_readlane_b32 s10, v43, 30                                // 000000009684: D760000A 00013D2B
	v_readlane_b32 s11, v43, 31                                // 00000000968C: D760000B 00013F2B
	v_add_f32_e32 v16, s59, v16                                // 000000009694: 0620203B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009698: BF870091
	v_add_f32_e32 v16, s60, v16                                // 00000000969C: 0620203C
	v_add_f32_e32 v16, s61, v16                                // 0000000096A0: 0620203D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000096A4: BF870091
	v_add_f32_e32 v16, s62, v16                                // 0000000096A8: 0620203E
	v_add_f32_e32 v18, s63, v16                                // 0000000096AC: 0624203F
	s_or_saveexec_b32 s105, -1                                 // 0000000096B0: BEE922C1
	scratch_load_b32 v44, off, off offset:84                   // 0000000096B4: DC510054 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 0000000096BC: BEFE0069
	v_dual_add_f32 v16, s4, v29 :: v_dual_add_f32 v21, s0, v22 // 0000000096C0: C9083A04 10142C00
	s_waitcnt vmcnt(0)                                         // 0000000096C8: BF8903F7
	v_readlane_b32 s12, v44, 0                                 // 0000000096CC: D760000C 0001012C
	v_readlane_b32 s13, v44, 1                                 // 0000000096D4: D760000D 0001032C
	v_readlane_b32 s14, v44, 2                                 // 0000000096DC: D760000E 0001052C
	v_dual_add_f32 v16, s5, v16 :: v_dual_add_f32 v21, s1, v21 // 0000000096E4: C9082005 10142A01
	v_add_f32_e32 v20, s9, v20                                 // 0000000096EC: 06282809
	v_readlane_b32 s15, v44, 3                                 // 0000000096F0: D760000F 0001072C
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000096F8: BF870093
	v_dual_add_f32 v21, s2, v21 :: v_dual_add_f32 v16, s6, v16 // 0000000096FC: C9082A02 15102006
	v_dual_add_f32 v21, s3, v21 :: v_dual_add_f32 v20, s10, v20// 000000009704: C9082A03 1514280A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000970C: BF870091
	v_dual_add_f32 v21, s4, v21 :: v_dual_add_f32 v16, s7, v16 // 000000009710: C9082A04 15102007
	v_dual_add_f32 v20, s11, v20 :: v_dual_add_f32 v21, s5, v21// 000000009718: C908280B 14142A05
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009720: BF870112
	v_add_f32_e32 v16, s8, v16                                 // 000000009724: 06202008
	v_add_f32_e32 v20, s12, v20                                // 000000009728: 0628280C
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000972C: BF870193
	v_add_f32_e32 v22, s6, v21                                 // 000000009730: 062C2A06
	v_add_f32_e32 v16, s9, v16                                 // 000000009734: 06202009
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009738: BF870113
	v_add_f32_e32 v20, s13, v20                                // 00000000973C: 0628280D
	v_add_f32_e32 v16, s10, v16                                // 000000009740: 0620200A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009744: BF870112
	v_add_f32_e32 v20, s14, v20                                // 000000009748: 0628280E
	v_add_f32_e32 v16, s11, v16                                // 00000000974C: 0620200B
	s_delay_alu instid0(VALU_DEP_2)                            // 000000009750: BF870002
	v_add_f32_e32 v20, s15, v20                                // 000000009754: 0628280F
	v_readlane_b32 s0, v36, 30                                 // 000000009758: D7600000 00013D24
	v_readlane_b32 s4, v38, 2                                  // 000000009760: D7600004 00010526
	v_readlane_b32 s1, v36, 31                                 // 000000009768: D7600001 00013F24
	v_readlane_b32 s5, v38, 3                                  // 000000009770: D7600005 00010726
	v_readlane_b32 s2, v38, 0                                  // 000000009778: D7600002 00010126
	v_add_f32_e32 v25, s0, v30                                 // 000000009780: 06323C00
	v_add_f32_e32 v21, s4, v31                                 // 000000009784: 062A3E04
	v_readlane_b32 s6, v38, 4                                  // 000000009788: D7600006 00010926
	v_readlane_b32 s3, v38, 1                                  // 000000009790: D7600003 00010326
	v_readlane_b32 s7, v38, 5                                  // 000000009798: D7600007 00010B26
	v_add_f32_e32 v25, s1, v25                                 // 0000000097A0: 06323201
	v_add_f32_e32 v21, s5, v21                                 // 0000000097A4: 062A2A05
	v_readlane_b32 s9, v38, 7                                  // 0000000097A8: D7600009 00010F26
	v_readlane_b32 s8, v38, 6                                  // 0000000097B0: D7600008 00010D26
	v_readlane_b32 s10, v38, 8                                 // 0000000097B8: D760000A 00011126
	v_add_f32_e32 v25, s2, v25                                 // 0000000097C0: 06323202
	v_add_f32_e32 v21, s6, v21                                 // 0000000097C4: 062A2A06
	v_readlane_b32 s11, v38, 9                                 // 0000000097C8: D760000B 00011326
	v_readlane_b32 s12, v38, 10                                // 0000000097D0: D760000C 00011526
	v_readlane_b32 s13, v38, 11                                // 0000000097D8: D760000D 00011726
	v_dual_add_f32 v25, s3, v25 :: v_dual_add_f32 v26, s9, v32 // 0000000097E0: C9083203 191A4009
	v_add_f32_e32 v21, s7, v21                                 // 0000000097E8: 062A2A07
	v_readlane_b32 s14, v38, 12                                // 0000000097EC: D760000E 00011926
	v_readlane_b32 s15, v38, 13                                // 0000000097F4: D760000F 00011B26
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000097FC: BF870214
	v_add_f32_e32 v25, s4, v25                                 // 000000009800: 06323204
	v_dual_add_f32 v21, s8, v21 :: v_dual_add_f32 v26, s10, v26// 000000009804: C9082A08 151A340A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000980C: BF870112
	v_add_f32_e32 v25, s5, v25                                 // 000000009810: 06323205
	v_dual_add_f32 v21, s9, v21 :: v_dual_add_f32 v26, s11, v26// 000000009814: C9082A09 151A340B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000981C: BF870112
	v_add_f32_e32 v25, s6, v25                                 // 000000009820: 06323206
	v_dual_add_f32 v21, s10, v21 :: v_dual_add_f32 v26, s12, v26// 000000009824: C9082A0A 151A340C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000982C: BF870091
	v_dual_add_f32 v21, s11, v21 :: v_dual_add_f32 v26, s13, v26// 000000009830: C9082A0B 151A340D
	v_add_f32_e32 v26, s14, v26                                // 000000009838: 0634340E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000983C: BF870001
	v_add_f32_e32 v27, s15, v26                                // 000000009840: 0636340F
	s_or_saveexec_b32 s105, -1                                 // 000000009844: BEE922C1
	scratch_load_b32 v41, off, off offset:8                    // 000000009848: DC510008 297C0000
	s_mov_b32 exec_lo, s105                                    // 000000009850: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000009854: BF8903F7
	v_readlane_b32 s36, v41, 12                                // 000000009858: D7600024 00011929
	v_readlane_b32 s37, v41, 13                                // 000000009860: D7600025 00011B29
	v_readlane_b32 s38, v41, 14                                // 000000009868: D7600026 00011D29
	v_readlane_b32 s39, v41, 15                                // 000000009870: D7600027 00011F29
	v_readlane_b32 s40, v41, 16                                // 000000009878: D7600028 00012129
	v_add_f32_e32 v11, s36, v11                                // 000000009880: 06161624
	v_readlane_b32 s41, v41, 17                                // 000000009884: D7600029 00012329
	v_readlane_b32 s42, v41, 18                                // 00000000988C: D760002A 00012529
	v_readlane_b32 s43, v41, 19                                // 000000009894: D760002B 00012729
	v_add_f32_e32 v9, s40, v9                                  // 00000000989C: 06121228
	v_add_f32_e32 v11, s37, v11                                // 0000000098A0: 06161625
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000098A4: BF870112
	v_add_f32_e32 v9, s41, v9                                  // 0000000098A8: 06121229
	v_add_f32_e32 v11, s38, v11                                // 0000000098AC: 06161626
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000098B0: BF870112
	v_add_f32_e32 v9, s42, v9                                  // 0000000098B4: 0612122A
	v_add_f32_e32 v11, s39, v11                                // 0000000098B8: 06161627
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000098BC: BF870112
	v_add_f32_e32 v9, s43, v9                                  // 0000000098C0: 0612122B
	v_add_f32_e32 v11, s40, v11                                // 0000000098C4: 06161628
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000098C8: BF870091
	v_add_f32_e32 v11, s41, v11                                // 0000000098CC: 06161629
	v_add_f32_e32 v11, s42, v11                                // 0000000098D0: 0616162A
	s_or_saveexec_b32 s105, -1                                 // 0000000098D4: BEE922C1
	scratch_load_b32 v41, off, off offset:60                   // 0000000098D8: DC51003C 297C0000
	s_mov_b32 exec_lo, s105                                    // 0000000098E0: BEFE0069
	s_waitcnt vmcnt(0)                                         // 0000000098E4: BF8903F7
	v_readlane_b32 s36, v41, 12                                // 0000000098E8: D7600024 00011929
	v_readlane_b32 s37, v41, 13                                // 0000000098F0: D7600025 00011B29
	v_readlane_b32 s38, v41, 14                                // 0000000098F8: D7600026 00011D29
	v_readlane_b32 s39, v41, 15                                // 000000009900: D7600027 00011F29
	v_readlane_b32 s40, v41, 16                                // 000000009908: D7600028 00012129
	v_readlane_b32 s41, v41, 17                                // 000000009910: D7600029 00012329
	v_readlane_b32 s42, v41, 18                                // 000000009918: D760002A 00012529
	v_readlane_b32 s43, v41, 19                                // 000000009920: D760002B 00012729
	s_or_saveexec_b32 s105, -1                                 // 000000009928: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 00000000992C: BF870139
	s_mov_b32 exec_lo, s105                                    // 000000009930: BEFE0069
	v_dual_add_f32 v10, s37, v10 :: v_dual_add_f32 v9, s36, v9 // 000000009934: C9081425 0A081224
	v_add_f32_e32 v8, 0, v8                                    // 00000000993C: 06101080
	v_dual_add_f32 v10, s38, v10 :: v_dual_add_f32 v9, s37, v9 // 000000009940: C9081426 0A081225
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009948: BF870091
	v_dual_add_f32 v10, s39, v10 :: v_dual_add_f32 v9, s38, v9 // 00000000994C: C9081427 0A081226
	v_dual_add_f32 v10, s40, v10 :: v_dual_add_f32 v9, s39, v9 // 000000009954: C9081428 0A081227
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000995C: BF870091
	v_add_f32_e32 v10, s41, v10                                // 000000009960: 06141429
	v_add_f32_e32 v10, s42, v10                                // 000000009964: 0614142A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000009968: BF870001
	v_add_f32_e32 v10, s43, v10                                // 00000000996C: 0614142B
	s_or_saveexec_b32 s105, -1                                 // 000000009970: BEE922C1
	scratch_load_b32 v44, off, off offset:44                   // 000000009974: DC51002C 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000997C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000009980: BF8903F7
	v_readlane_b32 s36, v44, 28                                // 000000009984: D7600024 0001392C
	v_readlane_b32 s37, v44, 29                                // 00000000998C: D7600025 00013B2C
	v_readlane_b32 s38, v44, 30                                // 000000009994: D7600026 00013D2C
	v_readlane_b32 s39, v44, 31                                // 00000000999C: D7600027 00013F2C
	s_or_saveexec_b32 s105, -1                                 // 0000000099A4: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000099A8: BF870009
	s_mov_b32 exec_lo, s105                                    // 0000000099AC: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 0000000099B0: BEE922C1
	scratch_store_b32 off, v38, off offset:36                  // 0000000099B4: DC690024 007C2600
	s_mov_b32 exec_lo, s105                                    // 0000000099BC: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 0000000099C0: BEE922C1
	scratch_load_b32 v38, off, off offset:48                   // 0000000099C4: DC510030 267C0000
	s_mov_b32 exec_lo, s105                                    // 0000000099CC: BEFE0069
	v_dual_add_f32 v8, s36, v8 :: v_dual_add_f32 v5, 0, v5     // 0000000099D0: C9081024 08040A80
	s_waitcnt vmcnt(0)                                         // 0000000099D8: BF8903F7
	v_readlane_b32 s40, v38, 0                                 // 0000000099DC: D7600028 00010126
	v_readlane_b32 s41, v38, 1                                 // 0000000099E4: D7600029 00010326
	v_readlane_b32 s42, v38, 2                                 // 0000000099EC: D760002A 00010526
	v_add_f32_e32 v8, s37, v8                                  // 0000000099F4: 06101025
	v_readlane_b32 s43, v38, 3                                 // 0000000099F8: D760002B 00010726
	v_dual_add_f32 v5, s40, v5 :: v_dual_add_f32 v6, 0, v6     // 000000009A00: C9080A28 05060C80
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000009A08: BF8701A3
	v_dual_add_f32 v7, 0, v7 :: v_dual_add_f32 v8, s38, v8     // 000000009A0C: C9080E80 07081026
	v_add_f32_e32 v1, 0, v1                                    // 000000009A14: 06020280
	v_dual_add_f32 v5, s41, v5 :: v_dual_add_f32 v2, 0, v2     // 000000009A18: C9080A29 05020480
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009A20: BF870091
	v_dual_add_f32 v8, s39, v8 :: v_dual_add_f32 v5, s42, v5   // 000000009A24: C9081027 08040A2A
	v_dual_add_f32 v8, s40, v8 :: v_dual_add_f32 v5, s43, v5   // 000000009A2C: C9081028 08040A2B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009A34: BF870091
	v_add_f32_e32 v8, s41, v8                                  // 000000009A38: 06101029
	v_add_f32_e32 v8, s42, v8                                  // 000000009A3C: 0610102A
	v_readlane_b32 s36, v38, 12                                // 000000009A40: D7600024 00011926
	v_readlane_b32 s37, v38, 13                                // 000000009A48: D7600025 00011B26
	v_readlane_b32 s38, v38, 14                                // 000000009A50: D7600026 00011D26
	v_readlane_b32 s39, v38, 15                                // 000000009A58: D7600027 00011F26
	v_readlane_b32 s40, v38, 16                                // 000000009A60: D7600028 00012126
	v_add_f32_e32 v6, s36, v6                                  // 000000009A68: 060C0C24
	v_readlane_b32 s41, v38, 17                                // 000000009A6C: D7600029 00012326
	v_readlane_b32 s42, v38, 18                                // 000000009A74: D760002A 00012526
	v_readlane_b32 s43, v38, 19                                // 000000009A7C: D760002B 00012726
	v_add_f32_e32 v2, s40, v2                                  // 000000009A84: 06040428
	v_add_f32_e32 v6, s37, v6                                  // 000000009A88: 060C0C25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009A8C: BF870112
	v_add_f32_e32 v2, s41, v2                                  // 000000009A90: 06040429
	v_add_f32_e32 v6, s38, v6                                  // 000000009A94: 060C0C26
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009A98: BF870112
	v_add_f32_e32 v2, s42, v2                                  // 000000009A9C: 0604042A
	v_add_f32_e32 v6, s39, v6                                  // 000000009AA0: 060C0C27
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009AA4: BF870112
	v_add_f32_e32 v2, s43, v2                                  // 000000009AA8: 0604042B
	v_add_f32_e32 v6, s40, v6                                  // 000000009AAC: 060C0C28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009AB0: BF870091
	v_add_f32_e32 v6, s41, v6                                  // 000000009AB4: 060C0C29
	v_add_f32_e32 v6, s42, v6                                  // 000000009AB8: 060C0C2A
	v_readlane_b32 s36, v38, 4                                 // 000000009ABC: D7600024 00010926
	v_readlane_b32 s37, v38, 5                                 // 000000009AC4: D7600025 00010B26
	v_readlane_b32 s38, v38, 6                                 // 000000009ACC: D7600026 00010D26
	v_readlane_b32 s39, v38, 7                                 // 000000009AD4: D7600027 00010F26
	v_readlane_b32 s40, v38, 8                                 // 000000009ADC: D7600028 00011126
	v_add_f32_e32 v7, s36, v7                                  // 000000009AE4: 060E0E24
	v_readlane_b32 s41, v38, 9                                 // 000000009AE8: D7600029 00011326
	v_readlane_b32 s42, v38, 10                                // 000000009AF0: D760002A 00011526
	v_readlane_b32 s43, v38, 11                                // 000000009AF8: D760002B 00011726
	v_add_f32_e32 v1, s40, v1                                  // 000000009B00: 06020228
	v_add_f32_e32 v7, s37, v7                                  // 000000009B04: 060E0E25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009B08: BF870112
	v_add_f32_e32 v1, s41, v1                                  // 000000009B0C: 06020229
	v_add_f32_e32 v7, s38, v7                                  // 000000009B10: 060E0E26
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009B14: BF870112
	v_add_f32_e32 v1, s42, v1                                  // 000000009B18: 0602022A
	v_add_f32_e32 v7, s39, v7                                  // 000000009B1C: 060E0E27
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009B20: BF870112
	v_add_f32_e32 v1, s43, v1                                  // 000000009B24: 0602022B
	v_add_f32_e32 v7, s40, v7                                  // 000000009B28: 060E0E28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009B2C: BF870091
	v_add_f32_e32 v7, s41, v7                                  // 000000009B30: 060E0E29
	v_add_f32_e32 v7, s42, v7                                  // 000000009B34: 060E0E2A
	s_or_saveexec_b32 s105, -1                                 // 000000009B38: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000009B3C: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000009B40: BEFE0069
	v_readlane_b32 s36, v44, 20                                // 000000009B44: D7600024 0001292C
	v_readlane_b32 s37, v44, 21                                // 000000009B4C: D7600025 00012B2C
	v_readlane_b32 s38, v44, 22                                // 000000009B54: D7600026 00012D2C
	v_readlane_b32 s39, v44, 23                                // 000000009B5C: D7600027 00012F2C
	v_readlane_b32 s40, v44, 24                                // 000000009B64: D7600028 0001312C
	v_dual_add_f32 v11, s36, v11 :: v_dual_add_f32 v4, 0, v4   // 000000009B6C: C9081624 0B040880
	v_readlane_b32 s41, v44, 25                                // 000000009B74: D7600029 0001332C
	v_readlane_b32 s42, v44, 26                                // 000000009B7C: D760002A 0001352C
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000009B84: BF870214
	v_add_f32_e32 v9, s40, v9                                  // 000000009B88: 06121228
	v_add_f32_e32 v11, s37, v11                                // 000000009B8C: 06161625
	v_readlane_b32 s43, v44, 27                                // 000000009B90: D760002B 0001372C
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000009B98: BF870193
	v_add_f32_e32 v9, s41, v9                                  // 000000009B9C: 06121229
	v_add_f32_e32 v11, s38, v11                                // 000000009BA0: 06161626
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009BA4: BF870112
	v_add_f32_e32 v9, s42, v9                                  // 000000009BA8: 0612122A
	v_add_f32_e32 v11, s39, v11                                // 000000009BAC: 06161627
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009BB0: BF870112
	v_add_f32_e32 v9, s43, v9                                  // 000000009BB4: 0612122B
	v_add_f32_e32 v11, s40, v11                                // 000000009BB8: 06161628
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009BBC: BF870091
	v_add_f32_e32 v11, s41, v11                                // 000000009BC0: 06161629
	v_add_f32_e32 v11, s42, v11                                // 000000009BC4: 0616162A
	s_or_saveexec_b32 s105, -1                                 // 000000009BC8: BEE922C1
	scratch_load_b32 v44, off, off offset:40                   // 000000009BCC: DC510028 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 000000009BD4: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000009BD8: BF8903F7
	v_readlane_b32 s36, v44, 28                                // 000000009BDC: D7600024 0001392C
	v_readlane_b32 s37, v44, 29                                // 000000009BE4: D7600025 00013B2C
	v_readlane_b32 s38, v44, 30                                // 000000009BEC: D7600026 00013D2C
	v_readlane_b32 s39, v44, 31                                // 000000009BF4: D7600027 00013F2C
	s_or_saveexec_b32 s105, -1                                 // 000000009BFC: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000009C00: BF870009
	s_mov_b32 exec_lo, s105                                    // 000000009C04: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000009C08: BEE922C1
	scratch_load_b32 v38, off, off offset:52                   // 000000009C0C: DC510034 267C0000
	s_mov_b32 exec_lo, s105                                    // 000000009C14: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000009C18: BF8903F7
	v_readlane_b32 s40, v38, 0                                 // 000000009C1C: D7600028 00010126
	v_readlane_b32 s41, v38, 1                                 // 000000009C24: D7600029 00010326
	v_readlane_b32 s42, v38, 2                                 // 000000009C2C: D760002A 00010526
	v_readlane_b32 s43, v38, 3                                 // 000000009C34: D760002B 00010726
	s_or_saveexec_b32 s105, -1                                 // 000000009C3C: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000009C40: BF8700A9
	s_mov_b32 exec_lo, s105                                    // 000000009C44: BEFE0069
	v_dual_add_f32 v4, s37, v4 :: v_dual_add_f32 v5, s36, v5   // 000000009C48: C9080825 04040A24
	v_dual_add_f32 v3, 0, v3 :: v_dual_add_f32 v4, s38, v4     // 000000009C50: C9080680 03040826
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009C58: BF870091
	v_dual_add_f32 v5, s37, v5 :: v_dual_add_f32 v4, s39, v4   // 000000009C5C: C9080A25 05040827
	v_dual_add_f32 v5, s38, v5 :: v_dual_add_f32 v4, s40, v4   // 000000009C64: C9080A26 05040828
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009C6C: BF870091
	v_dual_add_f32 v5, s39, v5 :: v_dual_add_f32 v4, s41, v4   // 000000009C70: C9080A27 05040829
	v_add_f32_e32 v4, s42, v4                                  // 000000009C78: 0608082A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000009C7C: BF870001
	v_add_f32_e32 v4, s43, v4                                  // 000000009C80: 0608082B
	s_or_saveexec_b32 s105, -1                                 // 000000009C84: BEE922C1
	scratch_store_b32 off, v43, off offset:44                  // 000000009C88: DC69002C 007C2B00
	s_mov_b32 exec_lo, s105                                    // 000000009C90: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 000000009C94: BEE922C1
	scratch_load_b32 v43, off, off offset:16                   // 000000009C98: DC510010 2B7C0000
	s_mov_b32 exec_lo, s105                                    // 000000009CA0: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000009CA4: BF8903F7
	v_readlane_b32 s36, v43, 12                                // 000000009CA8: D7600024 0001192B
	v_readlane_b32 s37, v43, 13                                // 000000009CB0: D7600025 00011B2B
	v_readlane_b32 s38, v43, 14                                // 000000009CB8: D7600026 00011D2B
	v_readlane_b32 s39, v43, 15                                // 000000009CC0: D7600027 00011F2B
	v_readlane_b32 s40, v43, 16                                // 000000009CC8: D7600028 0001212B
	v_readlane_b32 s41, v43, 17                                // 000000009CD0: D7600029 0001232B
	v_readlane_b32 s42, v43, 18                                // 000000009CD8: D760002A 0001252B
	v_readlane_b32 s43, v43, 19                                // 000000009CE0: D760002B 0001272B
	s_or_saveexec_b32 s105, -1                                 // 000000009CE8: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000009CEC: BF8700A9
	s_mov_b32 exec_lo, s105                                    // 000000009CF0: BEFE0069
	v_dual_add_f32 v3, s37, v3 :: v_dual_add_f32 v2, s36, v2   // 000000009CF4: C9080625 03020424
	v_dual_add_f32 v0, 0, v0 :: v_dual_add_f32 v3, s38, v3     // 000000009CFC: C9080080 00020626
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009D04: BF870091
	v_dual_add_f32 v2, s37, v2 :: v_dual_add_f32 v3, s39, v3   // 000000009D08: C9080425 02020627
	v_dual_add_f32 v2, s38, v2 :: v_dual_add_f32 v3, s40, v3   // 000000009D10: C9080426 02020628
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009D18: BF870091
	v_dual_add_f32 v2, s39, v2 :: v_dual_add_f32 v3, s41, v3   // 000000009D1C: C9080427 02020629
	v_add_f32_e32 v3, s42, v3                                  // 000000009D24: 0606062A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000009D28: BF870001
	v_add_f32_e32 v3, s43, v3                                  // 000000009D2C: 0606062B
	v_readlane_b32 s36, v44, 4                                 // 000000009D30: D7600024 0001092C
	v_readlane_b32 s37, v44, 5                                 // 000000009D38: D7600025 00010B2C
	v_readlane_b32 s38, v44, 6                                 // 000000009D40: D7600026 00010D2C
	v_readlane_b32 s39, v44, 7                                 // 000000009D48: D7600027 00010F2C
	v_readlane_b32 s40, v44, 8                                 // 000000009D50: D7600028 0001112C
	v_readlane_b32 s41, v44, 9                                 // 000000009D58: D7600029 0001132C
	v_add_f32_e32 v0, s37, v0                                  // 000000009D60: 06000025
	v_readlane_b32 s42, v44, 10                                // 000000009D64: D760002A 0001152C
	v_readlane_b32 s43, v44, 11                                // 000000009D6C: D760002B 0001172C
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009D74: BF870093
	v_dual_add_f32 v1, s36, v1 :: v_dual_add_f32 v0, s38, v0   // 000000009D78: C9080224 01000026
	v_dual_add_f32 v1, s37, v1 :: v_dual_add_f32 v0, s39, v0   // 000000009D80: C9080225 01000027
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009D88: BF870091
	v_dual_add_f32 v1, s38, v1 :: v_dual_add_f32 v0, s40, v0   // 000000009D8C: C9080226 01000028
	v_dual_add_f32 v1, s39, v1 :: v_dual_add_f32 v0, s41, v0   // 000000009D94: C9080227 01000029
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009D9C: BF870091
	v_add_f32_e32 v0, s42, v0                                  // 000000009DA0: 0600002A
	v_add_f32_e32 v0, s43, v0                                  // 000000009DA4: 0600002B
	v_readlane_b32 s36, v43, 4                                 // 000000009DA8: D7600024 0001092B
	v_readlane_b32 s37, v43, 5                                 // 000000009DB0: D7600025 00010B2B
	v_readlane_b32 s38, v43, 6                                 // 000000009DB8: D7600026 00010D2B
	v_readlane_b32 s39, v43, 7                                 // 000000009DC0: D7600027 00010F2B
	v_readlane_b32 s40, v43, 8                                 // 000000009DC8: D7600028 0001112B
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000009DD0: BF870244
	v_dual_add_f32 v9, s36, v9 :: v_dual_add_f32 v10, s37, v10 // 000000009DD4: C9081224 090A1425
	v_readlane_b32 s41, v43, 9                                 // 000000009DDC: D7600029 0001132B
	v_readlane_b32 s42, v43, 10                                // 000000009DE4: D760002A 0001152B
	v_readlane_b32 s43, v43, 11                                // 000000009DEC: D760002B 0001172B
	v_dual_add_f32 v9, s37, v9 :: v_dual_add_f32 v10, s38, v10 // 000000009DF4: C9081225 090A1426
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009DFC: BF870091
	v_dual_add_f32 v9, s38, v9 :: v_dual_add_f32 v10, s39, v10 // 000000009E00: C9081226 090A1427
	v_dual_add_f32 v9, s39, v9 :: v_dual_add_f32 v10, s40, v10 // 000000009E08: C9081227 090A1428
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009E10: BF870091
	v_add_f32_e32 v10, s41, v10                                // 000000009E14: 06141429
	v_add_f32_e32 v10, s42, v10                                // 000000009E18: 0614142A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000009E1C: BF870001
	v_add_f32_e32 v10, s43, v10                                // 000000009E20: 0614142B
	s_or_saveexec_b32 s105, -1                                 // 000000009E24: BEE922C1
	scratch_load_b32 v44, off, off offset:56                   // 000000009E28: DC510038 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 000000009E30: BEFE0069
	s_waitcnt vmcnt(0)                                         // 000000009E34: BF8903F7
	v_readlane_b32 s36, v44, 28                                // 000000009E38: D7600024 0001392C
	v_readlane_b32 s37, v44, 29                                // 000000009E40: D7600025 00013B2C
	v_readlane_b32 s40, v41, 0                                 // 000000009E48: D7600028 00010129
	v_readlane_b32 s38, v44, 30                                // 000000009E50: D7600026 00013D2C
	v_readlane_b32 s41, v41, 1                                 // 000000009E58: D7600029 00010329
	v_add_f32_e32 v8, s36, v8                                  // 000000009E60: 06101024
	v_readlane_b32 s39, v44, 31                                // 000000009E64: D7600027 00013F2C
	v_add_f32_e32 v5, s40, v5                                  // 000000009E6C: 060A0A28
	v_readlane_b32 s42, v41, 2                                 // 000000009E70: D760002A 00010529
	v_readlane_b32 s43, v41, 3                                 // 000000009E78: D760002B 00010729
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009E80: BF870093
	v_dual_add_f32 v8, s37, v8 :: v_dual_add_f32 v5, s41, v5   // 000000009E84: C9081025 08040A29
	v_dual_add_f32 v8, s38, v8 :: v_dual_add_f32 v5, s42, v5   // 000000009E8C: C9081026 08040A2A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009E94: BF870091
	v_dual_add_f32 v8, s39, v8 :: v_dual_add_f32 v5, s43, v5   // 000000009E98: C9081027 08040A2B
	v_add_f32_e32 v8, s40, v8                                  // 000000009EA0: 06101028
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009EA4: BF870091
	v_add_f32_e32 v8, s41, v8                                  // 000000009EA8: 06101029
	v_add_f32_e32 v8, s42, v8                                  // 000000009EAC: 0610102A
	v_readlane_b32 s36, v44, 20                                // 000000009EB0: D7600024 0001292C
	v_readlane_b32 s37, v44, 21                                // 000000009EB8: D7600025 00012B2C
	v_readlane_b32 s38, v44, 22                                // 000000009EC0: D7600026 00012D2C
	v_readlane_b32 s39, v44, 23                                // 000000009EC8: D7600027 00012F2C
	v_readlane_b32 s40, v44, 24                                // 000000009ED0: D7600028 0001312C
	v_readlane_b32 s41, v44, 25                                // 000000009ED8: D7600029 0001332C
	v_readlane_b32 s42, v44, 26                                // 000000009EE0: D760002A 0001352C
	v_readlane_b32 s43, v44, 27                                // 000000009EE8: D760002B 0001372C
	s_or_saveexec_b32 s105, -1                                 // 000000009EF0: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000009EF4: BF870139
	s_mov_b32 exec_lo, s105                                    // 000000009EF8: BEFE0069
	v_add_f32_e32 v6, s36, v6                                  // 000000009EFC: 060C0C24
	v_add_f32_e32 v2, s40, v2                                  // 000000009F00: 06040428
	v_add_f32_e32 v6, s37, v6                                  // 000000009F04: 060C0C25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009F08: BF870112
	v_add_f32_e32 v2, s41, v2                                  // 000000009F0C: 06040429
	v_add_f32_e32 v6, s38, v6                                  // 000000009F10: 060C0C26
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009F14: BF870112
	v_add_f32_e32 v2, s42, v2                                  // 000000009F18: 0604042A
	v_add_f32_e32 v6, s39, v6                                  // 000000009F1C: 060C0C27
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009F20: BF870112
	v_add_f32_e32 v2, s43, v2                                  // 000000009F24: 0604042B
	v_add_f32_e32 v6, s40, v6                                  // 000000009F28: 060C0C28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009F2C: BF870091
	v_add_f32_e32 v6, s41, v6                                  // 000000009F30: 060C0C29
	v_add_f32_e32 v6, s42, v6                                  // 000000009F34: 060C0C2A
	v_readlane_b32 s36, v41, 4                                 // 000000009F38: D7600024 00010929
	v_readlane_b32 s37, v41, 5                                 // 000000009F40: D7600025 00010B29
	v_readlane_b32 s38, v41, 6                                 // 000000009F48: D7600026 00010D29
	v_readlane_b32 s39, v41, 7                                 // 000000009F50: D7600027 00010F29
	v_readlane_b32 s40, v41, 8                                 // 000000009F58: D7600028 00011129
	v_add_f32_e32 v7, s36, v7                                  // 000000009F60: 060E0E24
	v_readlane_b32 s41, v41, 9                                 // 000000009F64: D7600029 00011329
	v_readlane_b32 s42, v41, 10                                // 000000009F6C: D760002A 00011529
	v_readlane_b32 s43, v41, 11                                // 000000009F74: D760002B 00011729
	v_add_f32_e32 v1, s40, v1                                  // 000000009F7C: 06020228
	v_add_f32_e32 v7, s37, v7                                  // 000000009F80: 060E0E25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009F84: BF870112
	v_add_f32_e32 v1, s41, v1                                  // 000000009F88: 06020229
	v_add_f32_e32 v7, s38, v7                                  // 000000009F8C: 060E0E26
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009F90: BF870112
	v_add_f32_e32 v1, s42, v1                                  // 000000009F94: 0602022A
	v_add_f32_e32 v7, s39, v7                                  // 000000009F98: 060E0E27
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009F9C: BF870112
	v_add_f32_e32 v1, s43, v1                                  // 000000009FA0: 0602022B
	v_add_f32_e32 v7, s40, v7                                  // 000000009FA4: 060E0E28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009FA8: BF870091
	v_add_f32_e32 v7, s41, v7                                  // 000000009FAC: 060E0E29
	v_add_f32_e32 v7, s42, v7                                  // 000000009FB0: 060E0E2A
	v_readlane_b32 s36, v44, 12                                // 000000009FB4: D7600024 0001192C
	v_readlane_b32 s37, v44, 13                                // 000000009FBC: D7600025 00011B2C
	v_readlane_b32 s38, v44, 14                                // 000000009FC4: D7600026 00011D2C
	v_readlane_b32 s39, v44, 15                                // 000000009FCC: D7600027 00011F2C
	v_readlane_b32 s40, v44, 16                                // 000000009FD4: D7600028 0001212C
	v_add_f32_e32 v11, s36, v11                                // 000000009FDC: 06161624
	v_readlane_b32 s41, v44, 17                                // 000000009FE0: D7600029 0001232C
	v_readlane_b32 s42, v44, 18                                // 000000009FE8: D760002A 0001252C
	v_readlane_b32 s43, v44, 19                                // 000000009FF0: D760002B 0001272C
	v_add_f32_e32 v9, s40, v9                                  // 000000009FF8: 06121228
	v_add_f32_e32 v11, s37, v11                                // 000000009FFC: 06161625
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A000: BF870112
	v_add_f32_e32 v9, s41, v9                                  // 00000000A004: 06121229
	v_add_f32_e32 v11, s38, v11                                // 00000000A008: 06161626
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A00C: BF870112
	v_add_f32_e32 v9, s42, v9                                  // 00000000A010: 0612122A
	v_add_f32_e32 v11, s39, v11                                // 00000000A014: 06161627
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A018: BF870112
	v_add_f32_e32 v9, s43, v9                                  // 00000000A01C: 0612122B
	v_add_f32_e32 v11, s40, v11                                // 00000000A020: 06161628
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A024: BF870091
	v_add_f32_e32 v11, s41, v11                                // 00000000A028: 06161629
	v_add_f32_e32 v11, s42, v11                                // 00000000A02C: 0616162A
	s_or_saveexec_b32 s105, -1                                 // 00000000A030: BEE922C1
	v_mov_b32_e32 v41, v38                                     // 00000000A034: 7E520326
	s_mov_b32 exec_lo, s105                                    // 00000000A038: BEFE0069
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000A03C: BF870001
	v_readlane_b32 s36, v41, 20                                // 00000000A040: D7600024 00012929
	v_readlane_b32 s37, v41, 21                                // 00000000A048: D7600025 00012B29
	v_readlane_b32 s38, v41, 22                                // 00000000A050: D7600026 00012D29
	v_readlane_b32 s39, v41, 23                                // 00000000A058: D7600027 00012F29
	v_readlane_b32 s40, v41, 24                                // 00000000A060: D7600028 00013129
	v_readlane_b32 s41, v41, 25                                // 00000000A068: D7600029 00013329
	v_dual_add_f32 v4, s37, v4 :: v_dual_add_f32 v5, s36, v5   // 00000000A070: C9080825 04040A24
	v_readlane_b32 s42, v41, 26                                // 00000000A078: D760002A 00013529
	v_readlane_b32 s43, v41, 27                                // 00000000A080: D760002B 00013729
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A088: BF870093
	v_dual_add_f32 v4, s38, v4 :: v_dual_add_f32 v5, s37, v5   // 00000000A08C: C9080826 04040A25
	v_dual_add_f32 v4, s39, v4 :: v_dual_add_f32 v5, s38, v5   // 00000000A094: C9080827 04040A26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A09C: BF870091
	v_dual_add_f32 v4, s40, v4 :: v_dual_add_f32 v5, s39, v5   // 00000000A0A0: C9080828 04040A27
	v_add_f32_e32 v4, s41, v4                                  // 00000000A0A8: 06080829
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A0AC: BF870091
	v_add_f32_e32 v4, s42, v4                                  // 00000000A0B0: 0608082A
	v_add_f32_e32 v4, s43, v4                                  // 00000000A0B4: 0608082B
	s_or_saveexec_b32 s105, -1                                 // 00000000A0B8: BEE922C1
	scratch_load_b32 v44, off, off offset:4                    // 00000000A0BC: DC510004 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000A0C4: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000A0C8: BF8903F7
	v_readlane_b32 s36, v44, 4                                 // 00000000A0CC: D7600024 0001092C
	v_readlane_b32 s37, v44, 5                                 // 00000000A0D4: D7600025 00010B2C
	v_readlane_b32 s38, v44, 6                                 // 00000000A0DC: D7600026 00010D2C
	v_readlane_b32 s39, v44, 7                                 // 00000000A0E4: D7600027 00010F2C
	v_readlane_b32 s40, v44, 8                                 // 00000000A0EC: D7600028 0001112C
	v_readlane_b32 s41, v44, 9                                 // 00000000A0F4: D7600029 0001132C
	v_readlane_b32 s42, v44, 10                                // 00000000A0FC: D760002A 0001152C
	v_readlane_b32 s43, v44, 11                                // 00000000A104: D760002B 0001172C
	s_or_saveexec_b32 s105, -1                                 // 00000000A10C: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000A110: BF8700A9
	s_mov_b32 exec_lo, s105                                    // 00000000A114: BEFE0069
	v_dual_add_f32 v3, s37, v3 :: v_dual_add_f32 v2, s36, v2   // 00000000A118: C9080625 03020424
	v_dual_add_f32 v3, s38, v3 :: v_dual_add_f32 v2, s37, v2   // 00000000A120: C9080626 03020425
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A128: BF870091
	v_dual_add_f32 v3, s39, v3 :: v_dual_add_f32 v2, s38, v2   // 00000000A12C: C9080627 03020426
	v_dual_add_f32 v3, s40, v3 :: v_dual_add_f32 v2, s39, v2   // 00000000A134: C9080628 03020427
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A13C: BF870091
	v_add_f32_e32 v3, s41, v3                                  // 00000000A140: 06060629
	v_add_f32_e32 v3, s42, v3                                  // 00000000A144: 0606062A
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000A148: BF870001
	v_add_f32_e32 v3, s43, v3                                  // 00000000A14C: 0606062B
	s_or_saveexec_b32 s105, -1                                 // 00000000A150: BEE922C1
	scratch_load_b32 v41, off, off offset:64                   // 00000000A154: DC510040 297C0000
	s_mov_b32 exec_lo, s105                                    // 00000000A15C: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000A160: BF8903F7
	v_readlane_b32 s36, v41, 20                                // 00000000A164: D7600024 00012929
	v_readlane_b32 s37, v41, 21                                // 00000000A16C: D7600025 00012B29
	v_readlane_b32 s38, v41, 22                                // 00000000A174: D7600026 00012D29
	v_readlane_b32 s39, v41, 23                                // 00000000A17C: D7600027 00012F29
	v_readlane_b32 s40, v41, 24                                // 00000000A184: D7600028 00013129
	v_readlane_b32 s41, v41, 25                                // 00000000A18C: D7600029 00013329
	v_readlane_b32 s42, v41, 26                                // 00000000A194: D760002A 00013529
	v_readlane_b32 s43, v41, 27                                // 00000000A19C: D760002B 00013729
	s_or_saveexec_b32 s105, -1                                 // 00000000A1A4: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000A1A8: BF8700A9
	s_mov_b32 exec_lo, s105                                    // 00000000A1AC: BEFE0069
	v_dual_add_f32 v0, s37, v0 :: v_dual_add_f32 v1, s36, v1   // 00000000A1B0: C9080025 00000224
	v_dual_add_f32 v0, s38, v0 :: v_dual_add_f32 v1, s37, v1   // 00000000A1B8: C9080026 00000225
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A1C0: BF870091
	v_dual_add_f32 v0, s39, v0 :: v_dual_add_f32 v1, s38, v1   // 00000000A1C4: C9080027 00000226
	v_dual_add_f32 v0, s40, v0 :: v_dual_add_f32 v1, s39, v1   // 00000000A1CC: C9080028 00000227
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A1D4: BF870091
	v_add_f32_e32 v0, s41, v0                                  // 00000000A1D8: 06000029
	v_add_f32_e32 v0, s42, v0                                  // 00000000A1DC: 0600002A
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000A1E0: BF870001
	v_add_f32_e32 v0, s43, v0                                  // 00000000A1E4: 0600002B
	v_readlane_b32 s36, v41, 28                                // 00000000A1E8: D7600024 00013929
	v_readlane_b32 s37, v41, 29                                // 00000000A1F0: D7600025 00013B29
	v_readlane_b32 s38, v41, 30                                // 00000000A1F8: D7600026 00013D29
	v_readlane_b32 s39, v41, 31                                // 00000000A200: D7600027 00013F29
	v_readlane_b32 s40, v44, 0                                 // 00000000A208: D7600028 0001012C
	v_add_f32_e32 v9, s36, v9                                  // 00000000A210: 06121224
	v_readlane_b32 s41, v44, 1                                 // 00000000A214: D7600029 0001032C
	v_readlane_b32 s42, v44, 2                                 // 00000000A21C: D760002A 0001052C
	v_readlane_b32 s43, v44, 3                                 // 00000000A224: D760002B 0001072C
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A22C: BF870094
	v_add_f32_e32 v9, s37, v9                                  // 00000000A230: 06121225
	v_dual_add_f32 v9, s38, v9 :: v_dual_add_f32 v10, s37, v10 // 00000000A234: C9081226 090A1425
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A23C: BF870091
	v_dual_add_f32 v9, s39, v9 :: v_dual_add_f32 v10, s38, v10 // 00000000A240: C9081227 090A1426
	v_add_f32_e32 v10, s39, v10                                // 00000000A248: 06141427
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A24C: BF870091
	v_add_f32_e32 v10, s40, v10                                // 00000000A250: 06141428
	v_add_f32_e32 v10, s41, v10                                // 00000000A254: 06141429
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A258: BF870091
	v_add_f32_e32 v10, s42, v10                                // 00000000A25C: 0614142A
	v_add_f32_e32 v10, s43, v10                                // 00000000A260: 0614142B
	s_or_saveexec_b32 s105, -1                                 // 00000000A264: BEE922C1
	scratch_load_b32 v43, off, off offset:24                   // 00000000A268: DC510018 2B7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000A270: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000A274: BF8903F7
	v_readlane_b32 s0, v43, 20                                 // 00000000A278: D7600000 0001292B
	v_readlane_b32 s1, v43, 21                                 // 00000000A280: D7600001 00012B2B
	v_readlane_b32 s4, v43, 24                                 // 00000000A288: D7600004 0001312B
	v_readlane_b32 s2, v43, 22                                 // 00000000A290: D7600002 00012D2B
	v_readlane_b32 s5, v43, 25                                 // 00000000A298: D7600005 0001332B
	v_add_f32_e32 v8, s0, v8                                   // 00000000A2A0: 06101000
	v_readlane_b32 s3, v43, 23                                 // 00000000A2A4: D7600003 00012F2B
	v_add_f32_e32 v5, s4, v5                                   // 00000000A2AC: 060A0A04
	v_readlane_b32 s6, v43, 26                                 // 00000000A2B0: D7600006 0001352B
	v_readlane_b32 s7, v43, 27                                 // 00000000A2B8: D7600007 0001372B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A2C0: BF870093
	v_dual_add_f32 v8, s1, v8 :: v_dual_add_f32 v5, s5, v5     // 00000000A2C4: C9081001 08040A05
	v_dual_add_f32 v8, s2, v8 :: v_dual_add_f32 v5, s6, v5     // 00000000A2CC: C9081002 08040A06
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A2D4: BF870091
	v_dual_add_f32 v8, s3, v8 :: v_dual_add_f32 v5, s7, v5     // 00000000A2D8: C9081003 08040A07
	v_add_f32_e32 v8, s4, v8                                   // 00000000A2E0: 06101004
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A2E4: BF870091
	v_add_f32_e32 v8, s5, v8                                   // 00000000A2E8: 06101005
	v_add_f32_e32 v8, s6, v8                                   // 00000000A2EC: 06101006
	v_readlane_b32 s0, v43, 12                                 // 00000000A2F0: D7600000 0001192B
	v_readlane_b32 s1, v43, 13                                 // 00000000A2F8: D7600001 00011B2B
	v_readlane_b32 s2, v43, 14                                 // 00000000A300: D7600002 00011D2B
	v_readlane_b32 s3, v43, 15                                 // 00000000A308: D7600003 00011F2B
	v_readlane_b32 s4, v43, 16                                 // 00000000A310: D7600004 0001212B
	v_readlane_b32 s5, v43, 17                                 // 00000000A318: D7600005 0001232B
	v_readlane_b32 s6, v43, 18                                 // 00000000A320: D7600006 0001252B
	v_readlane_b32 s7, v43, 19                                 // 00000000A328: D7600007 0001272B
	s_or_saveexec_b32 s105, -1                                 // 00000000A330: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000A334: BF870009
	s_mov_b32 exec_lo, s105                                    // 00000000A338: BEFE0069
	v_add_f32_e32 v6, s0, v6                                   // 00000000A33C: 060C0C00
	v_add_f32_e32 v2, s4, v2                                   // 00000000A340: 06040404
	v_readlane_b32 s36, v43, 28                                // 00000000A344: D7600024 0001392B
	v_readlane_b32 s37, v43, 29                                // 00000000A34C: D7600025 00013B2B
	v_readlane_b32 s38, v43, 30                                // 00000000A354: D7600026 00013D2B
	v_add_f32_e32 v6, s1, v6                                   // 00000000A35C: 060C0C01
	v_add_f32_e32 v2, s5, v2                                   // 00000000A360: 06040405
	v_readlane_b32 s39, v43, 31                                // 00000000A364: D7600027 00013F2B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000A36C: BF870193
	v_add_f32_e32 v6, s2, v6                                   // 00000000A370: 060C0C02
	v_add_f32_e32 v2, s6, v2                                   // 00000000A374: 06040406
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A378: BF870112
	v_add_f32_e32 v6, s3, v6                                   // 00000000A37C: 060C0C03
	v_add_f32_e32 v2, s7, v2                                   // 00000000A380: 06040407
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A384: BF870092
	v_add_f32_e32 v6, s4, v6                                   // 00000000A388: 060C0C04
	v_add_f32_e32 v6, s5, v6                                   // 00000000A38C: 060C0C05
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000A390: BF870001
	v_add_f32_e32 v6, s6, v6                                   // 00000000A394: 060C0C06
	s_or_saveexec_b32 s105, -1                                 // 00000000A398: BEE922C1
	scratch_load_b32 v44, off, off offset:68                   // 00000000A39C: DC510044 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000A3A4: BEFE0069
	v_readlane_b32 s0, v43, 4                                  // 00000000A3A8: D7600000 0001092B
	v_add_f32_e32 v7, s36, v7                                  // 00000000A3B0: 060E0E24
	v_readlane_b32 s1, v43, 5                                  // 00000000A3B4: D7600001 00010B2B
	v_readlane_b32 s2, v43, 6                                  // 00000000A3BC: D7600002 00010D2B
	s_waitcnt vmcnt(0)                                         // 00000000A3C4: BF8903F7
	v_readlane_b32 s40, v44, 0                                 // 00000000A3C8: D7600028 0001012C
	v_add_f32_e32 v11, s0, v11                                 // 00000000A3D0: 06161600
	v_add_f32_e32 v7, s37, v7                                  // 00000000A3D4: 060E0E25
	v_readlane_b32 s3, v43, 7                                  // 00000000A3D8: D7600003 00010F2B
	v_readlane_b32 s4, v43, 8                                  // 00000000A3E0: D7600004 0001112B
	v_readlane_b32 s41, v44, 1                                 // 00000000A3E8: D7600029 0001032C
	v_add_f32_e32 v11, s1, v11                                 // 00000000A3F0: 06161601
	v_add_f32_e32 v7, s38, v7                                  // 00000000A3F4: 060E0E26
	v_add_f32_e32 v1, s40, v1                                  // 00000000A3F8: 06020228
	v_readlane_b32 s5, v43, 9                                  // 00000000A3FC: D7600005 0001132B
	v_add_f32_e32 v9, s4, v9                                   // 00000000A404: 06121204
	v_add_f32_e32 v11, s2, v11                                 // 00000000A408: 06161602
	v_add_f32_e32 v7, s39, v7                                  // 00000000A40C: 060E0E27
	v_readlane_b32 s42, v44, 2                                 // 00000000A410: D760002A 0001052C
	v_add_f32_e32 v1, s41, v1                                  // 00000000A418: 06020229
	v_readlane_b32 s6, v43, 10                                 // 00000000A41C: D7600006 0001152B
	v_add_f32_e32 v11, s3, v11                                 // 00000000A424: 06161603
	v_add_f32_e32 v7, s40, v7                                  // 00000000A428: 060E0E28
	v_add_f32_e32 v9, s5, v9                                   // 00000000A42C: 06121205
	v_readlane_b32 s43, v44, 3                                 // 00000000A430: D760002B 0001072C
	v_add_f32_e32 v1, s42, v1                                  // 00000000A438: 0602022A
	v_add_f32_e32 v11, s4, v11                                 // 00000000A43C: 06161604
	v_add_f32_e32 v7, s41, v7                                  // 00000000A440: 060E0E29
	v_readlane_b32 s7, v43, 11                                 // 00000000A444: D7600007 0001172B
	v_add_f32_e32 v9, s6, v9                                   // 00000000A44C: 06121206
	v_add_f32_e32 v1, s43, v1                                  // 00000000A450: 0602022B
	v_add_f32_e32 v11, s5, v11                                 // 00000000A454: 06161605
	v_add_f32_e32 v7, s42, v7                                  // 00000000A458: 060E0E2A
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000A45C: BF870194
	v_add_f32_e32 v9, s7, v9                                   // 00000000A460: 06121207
	v_add_f32_e32 v11, s6, v11                                 // 00000000A464: 06161606
	s_or_saveexec_b32 s105, -1                                 // 00000000A468: BEE922C1
	scratch_load_b32 v43, off, off offset:20                   // 00000000A46C: DC510014 2B7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000A474: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000A478: BF8903F7
	v_readlane_b32 s0, v43, 12                                 // 00000000A47C: D7600000 0001192B
	v_readlane_b32 s1, v43, 13                                 // 00000000A484: D7600001 00011B2B
	v_readlane_b32 s2, v43, 14                                 // 00000000A48C: D7600002 00011D2B
	v_readlane_b32 s3, v43, 15                                 // 00000000A494: D7600003 00011F2B
	v_readlane_b32 s4, v43, 16                                 // 00000000A49C: D7600004 0001212B
	v_readlane_b32 s5, v43, 17                                 // 00000000A4A4: D7600005 0001232B
	v_dual_add_f32 v4, s1, v4 :: v_dual_add_f32 v5, s0, v5     // 00000000A4AC: C9080801 04040A00
	v_readlane_b32 s6, v43, 18                                 // 00000000A4B4: D7600006 0001252B
	v_readlane_b32 s7, v43, 19                                 // 00000000A4BC: D7600007 0001272B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A4C4: BF870093
	v_dual_add_f32 v4, s2, v4 :: v_dual_add_f32 v5, s1, v5     // 00000000A4C8: C9080802 04040A01
	v_dual_add_f32 v4, s3, v4 :: v_dual_add_f32 v5, s2, v5     // 00000000A4D0: C9080803 04040A02
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A4D8: BF870091
	v_dual_add_f32 v4, s4, v4 :: v_dual_add_f32 v5, s3, v5     // 00000000A4DC: C9080804 04040A03
	v_add_f32_e32 v4, s5, v4                                   // 00000000A4E4: 06080805
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A4E8: BF870091
	v_add_f32_e32 v4, s6, v4                                   // 00000000A4EC: 06080806
	v_add_f32_e32 v4, s7, v4                                   // 00000000A4F0: 06080807
	v_readlane_b32 s0, v44, 20                                 // 00000000A4F4: D7600000 0001292C
	v_readlane_b32 s1, v44, 21                                 // 00000000A4FC: D7600001 00012B2C
	v_readlane_b32 s2, v44, 22                                 // 00000000A504: D7600002 00012D2C
	v_readlane_b32 s3, v44, 23                                 // 00000000A50C: D7600003 00012F2C
	v_readlane_b32 s4, v44, 24                                 // 00000000A514: D7600004 0001312C
	v_readlane_b32 s5, v44, 25                                 // 00000000A51C: D7600005 0001332C
	v_add_f32_e32 v3, s1, v3                                   // 00000000A524: 06060601
	v_readlane_b32 s6, v44, 26                                 // 00000000A528: D7600006 0001352C
	v_readlane_b32 s7, v44, 27                                 // 00000000A530: D7600007 0001372C
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A538: BF870093
	v_add_f32_e32 v3, s2, v3                                   // 00000000A53C: 06060602
	v_dual_add_f32 v3, s3, v3 :: v_dual_add_f32 v2, s0, v2     // 00000000A540: C9080603 03020400
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A548: BF870091
	v_dual_add_f32 v3, s4, v3 :: v_dual_add_f32 v2, s1, v2     // 00000000A54C: C9080604 03020401
	v_dual_add_f32 v3, s5, v3 :: v_dual_add_f32 v2, s2, v2     // 00000000A554: C9080605 03020402
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A55C: BF870091
	v_dual_add_f32 v3, s6, v3 :: v_dual_add_f32 v2, s3, v2     // 00000000A560: C9080606 03020403
	v_add_f32_e32 v3, s7, v3                                   // 00000000A568: 06060607
	v_readlane_b32 s0, v44, 4                                  // 00000000A56C: D7600000 0001092C
	v_readlane_b32 s1, v44, 5                                  // 00000000A574: D7600001 00010B2C
	v_readlane_b32 s2, v44, 6                                  // 00000000A57C: D7600002 00010D2C
	v_readlane_b32 s3, v44, 7                                  // 00000000A584: D7600003 00010F2C
	v_readlane_b32 s4, v44, 8                                  // 00000000A58C: D7600004 0001112C
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 00000000A594: BF870244
	v_dual_add_f32 v1, s0, v1 :: v_dual_add_f32 v0, s1, v0     // 00000000A598: C9080200 01000001
	v_readlane_b32 s5, v44, 9                                  // 00000000A5A0: D7600005 0001132C
	v_readlane_b32 s6, v44, 10                                 // 00000000A5A8: D7600006 0001152C
	v_readlane_b32 s7, v44, 11                                 // 00000000A5B0: D7600007 0001172C
	v_dual_add_f32 v1, s1, v1 :: v_dual_add_f32 v0, s2, v0     // 00000000A5B8: C9080201 01000002
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A5C0: BF870091
	v_dual_add_f32 v1, s2, v1 :: v_dual_add_f32 v0, s3, v0     // 00000000A5C4: C9080202 01000003
	v_dual_add_f32 v1, s3, v1 :: v_dual_add_f32 v0, s4, v0     // 00000000A5CC: C9080203 01000004
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A5D4: BF870091
	v_add_f32_e32 v0, s5, v0                                   // 00000000A5D8: 06000005
	v_add_f32_e32 v0, s6, v0                                   // 00000000A5DC: 06000006
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000A5E0: BF870001
	v_add_f32_e32 v0, s7, v0                                   // 00000000A5E4: 06000007
	v_readlane_b32 s0, v44, 12                                 // 00000000A5E8: D7600000 0001192C
	v_readlane_b32 s1, v44, 13                                 // 00000000A5F0: D7600001 00011B2C
	v_readlane_b32 s2, v44, 14                                 // 00000000A5F8: D7600002 00011D2C
	v_readlane_b32 s3, v44, 15                                 // 00000000A600: D7600003 00011F2C
	v_readlane_b32 s4, v44, 16                                 // 00000000A608: D7600004 0001212C
	v_add_f32_e32 v9, s0, v9                                   // 00000000A610: 06121200
	v_readlane_b32 s5, v44, 17                                 // 00000000A614: D7600005 0001232C
	v_readlane_b32 s6, v44, 18                                 // 00000000A61C: D7600006 0001252C
	v_readlane_b32 s7, v44, 19                                 // 00000000A624: D7600007 0001272C
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A62C: BF870094
	v_add_f32_e32 v9, s1, v9                                   // 00000000A630: 06121201
	v_dual_add_f32 v9, s2, v9 :: v_dual_add_f32 v10, s1, v10   // 00000000A634: C9081202 090A1401
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A63C: BF870091
	v_dual_add_f32 v9, s3, v9 :: v_dual_add_f32 v10, s2, v10   // 00000000A640: C9081203 090A1402
	v_add_f32_e32 v10, s3, v10                                 // 00000000A648: 06141403
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A64C: BF870091
	v_add_f32_e32 v10, s4, v10                                 // 00000000A650: 06141404
	v_add_f32_e32 v10, s5, v10                                 // 00000000A654: 06141405
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A658: BF870091
	v_add_f32_e32 v10, s6, v10                                 // 00000000A65C: 06141406
	v_add_f32_e32 v10, s7, v10                                 // 00000000A660: 06141407
	s_or_saveexec_b32 s105, -1                                 // 00000000A664: BEE922C1
	scratch_load_b32 v44, off, off offset:92                   // 00000000A668: DC51005C 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000A670: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000A674: BF8903F7
	v_readlane_b32 s0, v44, 12                                 // 00000000A678: D7600000 0001192C
	v_readlane_b32 s1, v44, 13                                 // 00000000A680: D7600001 00011B2C
	v_readlane_b32 s4, v44, 16                                 // 00000000A688: D7600004 0001212C
	v_readlane_b32 s2, v44, 14                                 // 00000000A690: D7600002 00011D2C
	v_readlane_b32 s5, v44, 17                                 // 00000000A698: D7600005 0001232C
	v_add_f32_e32 v8, s0, v8                                   // 00000000A6A0: 06101000
	v_readlane_b32 s3, v44, 15                                 // 00000000A6A4: D7600003 00011F2C
	v_add_f32_e32 v5, s4, v5                                   // 00000000A6AC: 060A0A04
	v_readlane_b32 s6, v44, 18                                 // 00000000A6B0: D7600006 0001252C
	v_readlane_b32 s7, v44, 19                                 // 00000000A6B8: D7600007 0001272C
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A6C0: BF870093
	v_dual_add_f32 v8, s1, v8 :: v_dual_add_f32 v5, s5, v5     // 00000000A6C4: C9081001 08040A05
	v_dual_add_f32 v8, s2, v8 :: v_dual_add_f32 v5, s6, v5     // 00000000A6CC: C9081002 08040A06
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A6D4: BF870091
	v_dual_add_f32 v8, s3, v8 :: v_dual_add_f32 v5, s7, v5     // 00000000A6D8: C9081003 08040A07
	v_add_f32_e32 v8, s4, v8                                   // 00000000A6E0: 06101004
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A6E4: BF870091
	v_add_f32_e32 v8, s5, v8                                   // 00000000A6E8: 06101005
	v_add_f32_e32 v8, s6, v8                                   // 00000000A6EC: 06101006
	v_readlane_b32 s0, v44, 4                                  // 00000000A6F0: D7600000 0001092C
	v_readlane_b32 s1, v44, 5                                  // 00000000A6F8: D7600001 00010B2C
	v_readlane_b32 s2, v44, 6                                  // 00000000A700: D7600002 00010D2C
	v_readlane_b32 s3, v44, 7                                  // 00000000A708: D7600003 00010F2C
	v_readlane_b32 s4, v44, 8                                  // 00000000A710: D7600004 0001112C
	v_add_f32_e32 v6, s0, v6                                   // 00000000A718: 060C0C00
	v_readlane_b32 s5, v44, 9                                  // 00000000A71C: D7600005 0001132C
	v_readlane_b32 s6, v44, 10                                 // 00000000A724: D7600006 0001152C
	v_readlane_b32 s7, v44, 11                                 // 00000000A72C: D7600007 0001172C
	v_add_f32_e32 v2, s4, v2                                   // 00000000A734: 06040404
	v_add_f32_e32 v6, s1, v6                                   // 00000000A738: 060C0C01
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A73C: BF870112
	v_add_f32_e32 v2, s5, v2                                   // 00000000A740: 06040405
	v_add_f32_e32 v6, s2, v6                                   // 00000000A744: 060C0C02
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A748: BF870112
	v_add_f32_e32 v2, s6, v2                                   // 00000000A74C: 06040406
	v_add_f32_e32 v6, s3, v6                                   // 00000000A750: 060C0C03
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A754: BF870112
	v_add_f32_e32 v2, s7, v2                                   // 00000000A758: 06040407
	v_add_f32_e32 v6, s4, v6                                   // 00000000A75C: 060C0C04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A760: BF870091
	v_add_f32_e32 v6, s5, v6                                   // 00000000A764: 060C0C05
	v_add_f32_e32 v6, s6, v6                                   // 00000000A768: 060C0C06
	v_readlane_b32 s0, v44, 20                                 // 00000000A76C: D7600000 0001292C
	v_readlane_b32 s1, v44, 21                                 // 00000000A774: D7600001 00012B2C
	v_readlane_b32 s2, v44, 22                                 // 00000000A77C: D7600002 00012D2C
	v_readlane_b32 s3, v44, 23                                 // 00000000A784: D7600003 00012F2C
	v_readlane_b32 s4, v44, 24                                 // 00000000A78C: D7600004 0001312C
	v_add_f32_e32 v7, s0, v7                                   // 00000000A794: 060E0E00
	v_readlane_b32 s5, v44, 25                                 // 00000000A798: D7600005 0001332C
	v_readlane_b32 s6, v44, 26                                 // 00000000A7A0: D7600006 0001352C
	v_readlane_b32 s7, v44, 27                                 // 00000000A7A8: D7600007 0001372C
	v_add_f32_e32 v1, s4, v1                                   // 00000000A7B0: 06020204
	v_add_f32_e32 v7, s1, v7                                   // 00000000A7B4: 060E0E01
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A7B8: BF870112
	v_add_f32_e32 v1, s5, v1                                   // 00000000A7BC: 06020205
	v_add_f32_e32 v7, s2, v7                                   // 00000000A7C0: 060E0E02
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A7C4: BF870112
	v_add_f32_e32 v1, s6, v1                                   // 00000000A7C8: 06020206
	v_add_f32_e32 v7, s3, v7                                   // 00000000A7CC: 060E0E03
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A7D0: BF870112
	v_add_f32_e32 v1, s7, v1                                   // 00000000A7D4: 06020207
	v_add_f32_e32 v7, s4, v7                                   // 00000000A7D8: 060E0E04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A7DC: BF870091
	v_add_f32_e32 v7, s5, v7                                   // 00000000A7E0: 060E0E05
	v_add_f32_e32 v7, s6, v7                                   // 00000000A7E4: 060E0E06
	s_or_saveexec_b32 s105, -1                                 // 00000000A7E8: BEE922C1
	scratch_load_b32 v43, off, off offset:76                   // 00000000A7EC: DC51004C 2B7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000A7F4: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000A7F8: BF8903F7
	v_readlane_b32 s0, v43, 28                                 // 00000000A7FC: D7600000 0001392B
	v_readlane_b32 s1, v43, 29                                 // 00000000A804: D7600001 00013B2B
	v_readlane_b32 s2, v43, 30                                 // 00000000A80C: D7600002 00013D2B
	v_readlane_b32 s3, v43, 31                                 // 00000000A814: D7600003 00013F2B
	v_readlane_b32 s4, v44, 0                                  // 00000000A81C: D7600004 0001012C
	v_add_f32_e32 v11, s0, v11                                 // 00000000A824: 06161600
	v_readlane_b32 s5, v44, 1                                  // 00000000A828: D7600005 0001032C
	v_readlane_b32 s6, v44, 2                                  // 00000000A830: D7600006 0001052C
	v_readlane_b32 s7, v44, 3                                  // 00000000A838: D7600007 0001072C
	v_add_f32_e32 v9, s4, v9                                   // 00000000A840: 06121204
	v_add_f32_e32 v11, s1, v11                                 // 00000000A844: 06161601
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A848: BF870112
	v_add_f32_e32 v9, s5, v9                                   // 00000000A84C: 06121205
	v_add_f32_e32 v11, s2, v11                                 // 00000000A850: 06161602
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A854: BF870112
	v_add_f32_e32 v9, s6, v9                                   // 00000000A858: 06121206
	v_add_f32_e32 v11, s3, v11                                 // 00000000A85C: 06161603
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000A860: BF870112
	v_add_f32_e32 v9, s7, v9                                   // 00000000A864: 06121207
	v_add_f32_e32 v11, s4, v11                                 // 00000000A868: 06161604
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A86C: BF870091
	v_add_f32_e32 v11, s5, v11                                 // 00000000A870: 06161605
	v_add_f32_e32 v11, s6, v11                                 // 00000000A874: 06161606
	v_readlane_b32 s0, v44, 28                                 // 00000000A878: D7600000 0001392C
	v_readlane_b32 s1, v44, 29                                 // 00000000A880: D7600001 00013B2C
	v_readlane_b32 s2, v44, 30                                 // 00000000A888: D7600002 00013D2C
	v_readlane_b32 s3, v44, 31                                 // 00000000A890: D7600003 00013F2C
	v_readlane_b32 s4, v40, 0                                  // 00000000A898: D7600004 00010128
	v_readlane_b32 s5, v40, 1                                  // 00000000A8A0: D7600005 00010328
	v_dual_add_f32 v4, s1, v4 :: v_dual_add_f32 v5, s0, v5     // 00000000A8A8: C9080801 04040A00
	v_readlane_b32 s6, v40, 2                                  // 00000000A8B0: D7600006 00010528
	v_readlane_b32 s7, v40, 3                                  // 00000000A8B8: D7600007 00010728
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A8C0: BF870093
	v_dual_add_f32 v4, s2, v4 :: v_dual_add_f32 v5, s1, v5     // 00000000A8C4: C9080802 04040A01
	v_dual_add_f32 v4, s3, v4 :: v_dual_add_f32 v5, s2, v5     // 00000000A8CC: C9080803 04040A02
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A8D4: BF870091
	v_dual_add_f32 v4, s4, v4 :: v_dual_add_f32 v5, s3, v5     // 00000000A8D8: C9080804 04040A03
	v_add_f32_e32 v4, s5, v4                                   // 00000000A8E0: 06080805
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A8E4: BF870091
	v_add_f32_e32 v4, s6, v4                                   // 00000000A8E8: 06080806
	v_add_f32_e32 v4, s7, v4                                   // 00000000A8EC: 06080807
	v_readlane_b32 s0, v39, 12                                 // 00000000A8F0: D7600000 00011927
	v_readlane_b32 s1, v39, 13                                 // 00000000A8F8: D7600001 00011B27
	v_readlane_b32 s2, v39, 14                                 // 00000000A900: D7600002 00011D27
	v_readlane_b32 s3, v39, 15                                 // 00000000A908: D7600003 00011F27
	v_readlane_b32 s4, v39, 16                                 // 00000000A910: D7600004 00012127
	v_readlane_b32 s5, v39, 17                                 // 00000000A918: D7600005 00012327
	v_dual_add_f32 v3, s1, v3 :: v_dual_add_f32 v2, s0, v2     // 00000000A920: C9080601 03020400
	v_readlane_b32 s6, v39, 18                                 // 00000000A928: D7600006 00012527
	v_readlane_b32 s7, v39, 19                                 // 00000000A930: D7600007 00012727
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A938: BF870093
	v_dual_add_f32 v3, s2, v3 :: v_dual_add_f32 v2, s1, v2     // 00000000A93C: C9080602 03020401
	v_dual_add_f32 v3, s3, v3 :: v_dual_add_f32 v2, s2, v2     // 00000000A944: C9080603 03020402
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A94C: BF870091
	v_dual_add_f32 v3, s4, v3 :: v_dual_add_f32 v2, s3, v2     // 00000000A950: C9080604 03020403
	v_add_f32_e32 v3, s5, v3                                   // 00000000A958: 06060605
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A95C: BF870091
	v_add_f32_e32 v3, s6, v3                                   // 00000000A960: 06060606
	v_add_f32_e32 v3, s7, v3                                   // 00000000A964: 06060607
	s_or_saveexec_b32 s105, -1                                 // 00000000A968: BEE922C1
	scratch_load_b32 v41, off, off offset:28                   // 00000000A96C: DC51001C 297C0000
	s_mov_b32 exec_lo, s105                                    // 00000000A974: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000A978: BF8903F7
	v_readlane_b32 s0, v41, 28                                 // 00000000A97C: D7600000 00013929
	v_readlane_b32 s1, v41, 29                                 // 00000000A984: D7600001 00013B29
	v_readlane_b32 s2, v41, 30                                 // 00000000A98C: D7600002 00013D29
	v_readlane_b32 s3, v41, 31                                 // 00000000A994: D7600003 00013F29
	s_or_saveexec_b32 s105, -1                                 // 00000000A99C: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000A9A0: BF870009
	s_mov_b32 exec_lo, s105                                    // 00000000A9A4: BEFE0069
	s_or_saveexec_b32 s105, -1                                 // 00000000A9A8: BEE922C1
	scratch_load_b32 v44, off, off offset:108                  // 00000000A9AC: DC51006C 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000A9B4: BEFE0069
	v_dual_add_f32 v0, s1, v0 :: v_dual_add_f32 v1, s0, v1     // 00000000A9B8: C9080001 00000200
	s_waitcnt vmcnt(0)                                         // 00000000A9C0: BF8903F7
	v_readlane_b32 s4, v44, 0                                  // 00000000A9C4: D7600004 0001012C
	v_readlane_b32 s5, v44, 1                                  // 00000000A9CC: D7600005 0001032C
	v_readlane_b32 s6, v44, 2                                  // 00000000A9D4: D7600006 0001052C
	v_dual_add_f32 v0, s2, v0 :: v_dual_add_f32 v1, s1, v1     // 00000000A9DC: C9080002 00000201
	v_readlane_b32 s7, v44, 3                                  // 00000000A9E4: D7600007 0001072C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000A9EC: BF870092
	v_dual_add_f32 v0, s3, v0 :: v_dual_add_f32 v1, s2, v1     // 00000000A9F0: C9080003 00000202
	v_dual_add_f32 v0, s4, v0 :: v_dual_add_f32 v1, s3, v1     // 00000000A9F8: C9080004 00000203
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AA00: BF870091
	v_add_f32_e32 v0, s5, v0                                   // 00000000AA04: 06000005
	v_add_f32_e32 v0, s6, v0                                   // 00000000AA08: 06000006
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000AA0C: BF870001
	v_add_f32_e32 v0, s7, v0                                   // 00000000AA10: 06000007
	v_readlane_b32 s0, v39, 4                                  // 00000000AA14: D7600000 00010927
	v_readlane_b32 s1, v39, 5                                  // 00000000AA1C: D7600001 00010B27
	v_readlane_b32 s2, v39, 6                                  // 00000000AA24: D7600002 00010D27
	v_readlane_b32 s3, v39, 7                                  // 00000000AA2C: D7600003 00010F27
	v_readlane_b32 s4, v39, 8                                  // 00000000AA34: D7600004 00011127
	v_add_f32_e32 v9, s0, v9                                   // 00000000AA3C: 06121200
	v_readlane_b32 s5, v39, 9                                  // 00000000AA40: D7600005 00011327
	v_readlane_b32 s6, v39, 10                                 // 00000000AA48: D7600006 00011527
	v_readlane_b32 s7, v39, 11                                 // 00000000AA50: D7600007 00011727
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AA58: BF870094
	v_add_f32_e32 v9, s1, v9                                   // 00000000AA5C: 06121201
	v_dual_add_f32 v9, s2, v9 :: v_dual_add_f32 v10, s1, v10   // 00000000AA60: C9081202 090A1401
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AA68: BF870091
	v_dual_add_f32 v9, s3, v9 :: v_dual_add_f32 v10, s2, v10   // 00000000AA6C: C9081203 090A1402
	v_add_f32_e32 v10, s3, v10                                 // 00000000AA74: 06141403
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AA78: BF870091
	v_add_f32_e32 v10, s4, v10                                 // 00000000AA7C: 06141404
	v_add_f32_e32 v10, s5, v10                                 // 00000000AA80: 06141405
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AA84: BF870091
	v_add_f32_e32 v10, s6, v10                                 // 00000000AA88: 06141406
	v_add_f32_e32 v10, s7, v10                                 // 00000000AA8C: 06141407
	v_readlane_b32 s0, v41, 4                                  // 00000000AA90: D7600000 00010929
	v_readlane_b32 s1, v41, 5                                  // 00000000AA98: D7600001 00010B29
	v_readlane_b32 s2, v41, 6                                  // 00000000AAA0: D7600002 00010D29
	v_readlane_b32 s3, v41, 7                                  // 00000000AAA8: D7600003 00010F29
	v_readlane_b32 s4, v41, 8                                  // 00000000AAB0: D7600004 00011129
	v_add_f32_e32 v8, s0, v8                                   // 00000000AAB8: 06101000
	v_readlane_b32 s5, v41, 9                                  // 00000000AABC: D7600005 00011329
	v_readlane_b32 s6, v41, 10                                 // 00000000AAC4: D7600006 00011529
	v_readlane_b32 s7, v41, 11                                 // 00000000AACC: D7600007 00011729
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AAD4: BF870094
	v_dual_add_f32 v5, s4, v5 :: v_dual_add_f32 v8, s1, v8     // 00000000AAD8: C9080A04 05081001
	v_dual_add_f32 v5, s5, v5 :: v_dual_add_f32 v8, s2, v8     // 00000000AAE0: C9080A05 05081002
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AAE8: BF870091
	v_dual_add_f32 v5, s6, v5 :: v_dual_add_f32 v8, s3, v8     // 00000000AAEC: C9080A06 05081003
	v_dual_add_f32 v5, s7, v5 :: v_dual_add_f32 v8, s4, v8     // 00000000AAF4: C9080A07 05081004
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AAFC: BF870091
	v_add_f32_e32 v8, s5, v8                                   // 00000000AB00: 06101005
	v_add_f32_e32 v8, s6, v8                                   // 00000000AB04: 06101006
	v_readlane_b32 s0, v44, 4                                  // 00000000AB08: D7600000 0001092C
	v_readlane_b32 s1, v44, 5                                  // 00000000AB10: D7600001 00010B2C
	v_readlane_b32 s2, v44, 6                                  // 00000000AB18: D7600002 00010D2C
	v_readlane_b32 s3, v44, 7                                  // 00000000AB20: D7600003 00010F2C
	v_readlane_b32 s4, v44, 8                                  // 00000000AB28: D7600004 0001112C
	v_add_f32_e32 v6, s0, v6                                   // 00000000AB30: 060C0C00
	v_readlane_b32 s5, v44, 9                                  // 00000000AB34: D7600005 0001132C
	v_readlane_b32 s6, v44, 10                                 // 00000000AB3C: D7600006 0001152C
	v_readlane_b32 s7, v44, 11                                 // 00000000AB44: D7600007 0001172C
	v_add_f32_e32 v2, s4, v2                                   // 00000000AB4C: 06040404
	v_add_f32_e32 v6, s1, v6                                   // 00000000AB50: 060C0C01
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000AB54: BF870112
	v_add_f32_e32 v2, s5, v2                                   // 00000000AB58: 06040405
	v_add_f32_e32 v6, s2, v6                                   // 00000000AB5C: 060C0C02
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000AB60: BF870112
	v_add_f32_e32 v2, s6, v2                                   // 00000000AB64: 06040406
	v_add_f32_e32 v6, s3, v6                                   // 00000000AB68: 060C0C03
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000AB6C: BF870112
	v_add_f32_e32 v2, s7, v2                                   // 00000000AB70: 06040407
	v_add_f32_e32 v6, s4, v6                                   // 00000000AB74: 060C0C04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AB78: BF870091
	v_add_f32_e32 v6, s5, v6                                   // 00000000AB7C: 060C0C05
	v_add_f32_e32 v6, s6, v6                                   // 00000000AB80: 060C0C06
	v_readlane_b32 s0, v37, 20                                 // 00000000AB84: D7600000 00012925
	v_readlane_b32 s1, v37, 21                                 // 00000000AB8C: D7600001 00012B25
	v_readlane_b32 s2, v37, 22                                 // 00000000AB94: D7600002 00012D25
	v_readlane_b32 s3, v37, 23                                 // 00000000AB9C: D7600003 00012F25
	v_readlane_b32 s4, v37, 24                                 // 00000000ABA4: D7600004 00013125
	v_add_f32_e32 v7, s0, v7                                   // 00000000ABAC: 060E0E00
	v_readlane_b32 s5, v37, 25                                 // 00000000ABB0: D7600005 00013325
	v_readlane_b32 s6, v37, 26                                 // 00000000ABB8: D7600006 00013525
	v_readlane_b32 s7, v37, 27                                 // 00000000ABC0: D7600007 00013725
	v_add_f32_e32 v1, s4, v1                                   // 00000000ABC8: 06020204
	v_add_f32_e32 v7, s1, v7                                   // 00000000ABCC: 060E0E01
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000ABD0: BF870112
	v_add_f32_e32 v1, s5, v1                                   // 00000000ABD4: 06020205
	v_add_f32_e32 v7, s2, v7                                   // 00000000ABD8: 060E0E02
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000ABDC: BF870112
	v_add_f32_e32 v1, s6, v1                                   // 00000000ABE0: 06020206
	v_add_f32_e32 v7, s3, v7                                   // 00000000ABE4: 060E0E03
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000ABE8: BF870112
	v_add_f32_e32 v1, s7, v1                                   // 00000000ABEC: 06020207
	v_add_f32_e32 v7, s4, v7                                   // 00000000ABF0: 060E0E04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000ABF4: BF870091
	v_add_f32_e32 v7, s5, v7                                   // 00000000ABF8: 060E0E05
	v_add_f32_e32 v7, s6, v7                                   // 00000000ABFC: 060E0E06
	v_readlane_b32 s0, v42, 28                                 // 00000000AC00: D7600000 0001392A
	v_readlane_b32 s1, v42, 29                                 // 00000000AC08: D7600001 00013B2A
	v_readlane_b32 s2, v42, 30                                 // 00000000AC10: D7600002 00013D2A
	v_readlane_b32 s3, v42, 31                                 // 00000000AC18: D7600003 00013F2A
	v_readlane_b32 s4, v41, 0                                  // 00000000AC20: D7600004 00010129
	v_add_f32_e32 v11, s0, v11                                 // 00000000AC28: 06161600
	v_readlane_b32 s5, v41, 1                                  // 00000000AC2C: D7600005 00010329
	v_readlane_b32 s6, v41, 2                                  // 00000000AC34: D7600006 00010529
	v_readlane_b32 s7, v41, 3                                  // 00000000AC3C: D7600007 00010729
	v_add_f32_e32 v9, s4, v9                                   // 00000000AC44: 06121204
	v_add_f32_e32 v11, s1, v11                                 // 00000000AC48: 06161601
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000AC4C: BF870112
	v_add_f32_e32 v9, s5, v9                                   // 00000000AC50: 06121205
	v_add_f32_e32 v11, s2, v11                                 // 00000000AC54: 06161602
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000AC58: BF870112
	v_add_f32_e32 v9, s6, v9                                   // 00000000AC5C: 06121206
	v_add_f32_e32 v11, s3, v11                                 // 00000000AC60: 06161603
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000AC64: BF870112
	v_add_f32_e32 v9, s7, v9                                   // 00000000AC68: 06121207
	v_add_f32_e32 v11, s4, v11                                 // 00000000AC6C: 06161604
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AC70: BF870091
	v_add_f32_e32 v11, s5, v11                                 // 00000000AC74: 06161605
	v_add_f32_e32 v11, s6, v11                                 // 00000000AC78: 06161606
	v_readlane_b32 s0, v42, 4                                  // 00000000AC7C: D7600000 0001092A
	v_readlane_b32 s1, v42, 5                                  // 00000000AC84: D7600001 00010B2A
	v_readlane_b32 s2, v42, 6                                  // 00000000AC8C: D7600002 00010D2A
	v_readlane_b32 s3, v42, 7                                  // 00000000AC94: D7600003 00010F2A
	v_readlane_b32 s4, v42, 8                                  // 00000000AC9C: D7600004 0001112A
	v_readlane_b32 s5, v42, 9                                  // 00000000ACA4: D7600005 0001132A
	v_add_f32_e32 v4, s1, v4                                   // 00000000ACAC: 06080801
	v_readlane_b32 s6, v42, 10                                 // 00000000ACB0: D7600006 0001152A
	v_readlane_b32 s7, v42, 11                                 // 00000000ACB8: D7600007 0001172A
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000ACC0: BF870093
	v_dual_add_f32 v5, s0, v5 :: v_dual_add_f32 v4, s2, v4     // 00000000ACC4: C9080A00 05040802
	v_dual_add_f32 v5, s1, v5 :: v_dual_add_f32 v4, s3, v4     // 00000000ACCC: C9080A01 05040803
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000ACD4: BF870091
	v_dual_add_f32 v5, s2, v5 :: v_dual_add_f32 v4, s4, v4     // 00000000ACD8: C9080A02 05040804
	v_dual_add_f32 v5, s3, v5 :: v_dual_add_f32 v4, s5, v4     // 00000000ACE0: C9080A03 05040805
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000ACE8: BF870091
	v_add_f32_e32 v4, s6, v4                                   // 00000000ACEC: 06080806
	v_add_f32_e32 v4, s7, v4                                   // 00000000ACF0: 06080807
	v_readlane_b32 s0, v44, 28                                 // 00000000ACF4: D7600000 0001392C
	v_readlane_b32 s1, v44, 29                                 // 00000000ACFC: D7600001 00013B2C
	v_readlane_b32 s2, v44, 30                                 // 00000000AD04: D7600002 00013D2C
	v_readlane_b32 s3, v44, 31                                 // 00000000AD0C: D7600003 00013F2C
	v_readlane_b32 s4, v37, 0                                  // 00000000AD14: D7600004 00010125
	v_readlane_b32 s5, v37, 1                                  // 00000000AD1C: D7600005 00010325
	v_readlane_b32 s6, v37, 2                                  // 00000000AD24: D7600006 00010525
	v_readlane_b32 s7, v37, 3                                  // 00000000AD2C: D7600007 00010725
	s_or_saveexec_b32 s105, -1                                 // 00000000AD34: BEE922C1
	scratch_load_b32 v42, off, off offset:84                   // 00000000AD38: DC510054 2A7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000AD40: BEFE0069
	v_dual_add_f32 v3, s1, v3 :: v_dual_add_f32 v2, s0, v2     // 00000000AD44: C9080601 03020400
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AD4C: BF870091
	v_dual_add_f32 v3, s2, v3 :: v_dual_add_f32 v2, s1, v2     // 00000000AD50: C9080602 03020401
	v_dual_add_f32 v3, s3, v3 :: v_dual_add_f32 v2, s2, v2     // 00000000AD58: C9080603 03020402
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AD60: BF870091
	v_dual_add_f32 v3, s4, v3 :: v_dual_add_f32 v2, s3, v2     // 00000000AD64: C9080604 03020403
	v_add_f32_e32 v3, s5, v3                                   // 00000000AD6C: 06060605
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AD70: BF870091
	v_add_f32_e32 v3, s6, v3                                   // 00000000AD74: 06060606
	v_add_f32_e32 v3, s7, v3                                   // 00000000AD78: 06060607
	v_readlane_b32 s0, v44, 12                                 // 00000000AD7C: D7600000 0001192C
	v_readlane_b32 s1, v44, 13                                 // 00000000AD84: D7600001 00011B2C
	v_readlane_b32 s2, v44, 14                                 // 00000000AD8C: D7600002 00011D2C
	v_readlane_b32 s3, v44, 15                                 // 00000000AD94: D7600003 00011F2C
	v_readlane_b32 s4, v44, 16                                 // 00000000AD9C: D7600004 0001212C
	v_readlane_b32 s5, v44, 17                                 // 00000000ADA4: D7600005 0001232C
	v_add_f32_e32 v0, s1, v0                                   // 00000000ADAC: 06000001
	v_readlane_b32 s6, v44, 18                                 // 00000000ADB0: D7600006 0001252C
	v_readlane_b32 s7, v44, 19                                 // 00000000ADB8: D7600007 0001272C
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000ADC0: BF870093
	v_add_f32_e32 v0, s2, v0                                   // 00000000ADC4: 06000002
	v_dual_add_f32 v0, s3, v0 :: v_dual_add_f32 v1, s0, v1     // 00000000ADC8: C9080003 00000200
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000ADD0: BF870091
	v_dual_add_f32 v0, s4, v0 :: v_dual_add_f32 v1, s1, v1     // 00000000ADD4: C9080004 00000201
	v_dual_add_f32 v0, s5, v0 :: v_dual_add_f32 v1, s2, v1     // 00000000ADDC: C9080005 00000202
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000ADE4: BF870091
	v_dual_add_f32 v0, s6, v0 :: v_dual_add_f32 v1, s3, v1     // 00000000ADE8: C9080006 00000203
	v_add_f32_e32 v0, s7, v0                                   // 00000000ADF0: 06000007
	v_readlane_b32 s0, v44, 20                                 // 00000000ADF4: D7600000 0001292C
	v_readlane_b32 s1, v44, 21                                 // 00000000ADFC: D7600001 00012B2C
	v_readlane_b32 s2, v44, 22                                 // 00000000AE04: D7600002 00012D2C
	v_readlane_b32 s3, v44, 23                                 // 00000000AE0C: D7600003 00012F2C
	v_readlane_b32 s4, v44, 24                                 // 00000000AE14: D7600004 0001312C
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 00000000AE1C: BF870244
	v_dual_add_f32 v9, s0, v9 :: v_dual_add_f32 v10, s1, v10   // 00000000AE20: C9081200 090A1401
	v_readlane_b32 s5, v44, 25                                 // 00000000AE28: D7600005 0001332C
	v_readlane_b32 s6, v44, 26                                 // 00000000AE30: D7600006 0001352C
	v_readlane_b32 s7, v44, 27                                 // 00000000AE38: D7600007 0001372C
	v_dual_add_f32 v9, s1, v9 :: v_dual_add_f32 v10, s2, v10   // 00000000AE40: C9081201 090A1402
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AE48: BF870091
	v_dual_add_f32 v9, s2, v9 :: v_dual_add_f32 v10, s3, v10   // 00000000AE4C: C9081202 090A1403
	v_dual_add_f32 v9, s3, v9 :: v_dual_add_f32 v10, s4, v10   // 00000000AE54: C9081203 090A1404
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AE5C: BF870091
	v_add_f32_e32 v10, s5, v10                                 // 00000000AE60: 06141405
	v_add_f32_e32 v10, s6, v10                                 // 00000000AE64: 06141406
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000AE68: BF870001
	v_add_f32_e32 v10, s7, v10                                 // 00000000AE6C: 06141407
	s_waitcnt vmcnt(0)                                         // 00000000AE70: BF8903F7
	v_readlane_b32 s0, v42, 22                                 // 00000000AE74: D7600000 00012D2A
	v_readlane_b32 s4, v42, 26                                 // 00000000AE7C: D7600004 0001352A
	v_readlane_b32 s5, v42, 27                                 // 00000000AE84: D7600005 0001372A
	v_readlane_b32 s6, v42, 28                                 // 00000000AE8C: D7600006 0001392A
	v_readlane_b32 s1, v42, 23                                 // 00000000AE94: D7600001 00012F2A
	v_readlane_b32 s7, v42, 29                                 // 00000000AE9C: D7600007 00013B2A
	v_add_f32_e32 v5, s4, v5                                   // 00000000AEA4: 060A0A04
	v_readlane_b32 s2, v42, 24                                 // 00000000AEA8: D7600002 0001312A
	v_readlane_b32 s3, v42, 25                                 // 00000000AEB0: D7600003 0001332A
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AEB8: BF870093
	v_add_f32_e32 v5, s5, v5                                   // 00000000AEBC: 060A0A05
	v_dual_add_f32 v5, s6, v5 :: v_dual_add_f32 v8, s0, v8     // 00000000AEC0: C9080A06 05081000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AEC8: BF870091
	v_dual_add_f32 v5, s7, v5 :: v_dual_add_f32 v8, s1, v8     // 00000000AECC: C9080A07 05081001
	v_add_f32_e32 v8, s2, v8                                   // 00000000AED4: 06101002
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AED8: BF870091
	v_add_f32_e32 v8, s3, v8                                   // 00000000AEDC: 06101003
	v_add_f32_e32 v8, s4, v8                                   // 00000000AEE0: 06101004
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AEE4: BF870091
	v_add_f32_e32 v8, s5, v8                                   // 00000000AEE8: 06101005
	v_add_f32_e32 v8, s6, v8                                   // 00000000AEEC: 06101006
	s_or_saveexec_b32 s105, -1                                 // 00000000AEF0: BEE922C1
	scratch_load_b32 v43, off, off offset:36                   // 00000000AEF4: DC510024 2B7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000AEFC: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000AF00: BF8903F7
	v_readlane_b32 s0, v43, 14                                 // 00000000AF04: D7600000 00011D2B
	v_readlane_b32 s1, v43, 15                                 // 00000000AF0C: D7600001 00011F2B
	v_readlane_b32 s2, v43, 16                                 // 00000000AF14: D7600002 0001212B
	v_readlane_b32 s3, v43, 17                                 // 00000000AF1C: D7600003 0001232B
	v_readlane_b32 s4, v43, 18                                 // 00000000AF24: D7600004 0001252B
	v_readlane_b32 s5, v43, 19                                 // 00000000AF2C: D7600005 0001272B
	v_readlane_b32 s6, v43, 20                                 // 00000000AF34: D7600006 0001292B
	v_readlane_b32 s7, v43, 21                                 // 00000000AF3C: D7600007 00012B2B
	s_or_saveexec_b32 s105, -1                                 // 00000000AF44: BEE922C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 00000000AF48: BF870139
	s_mov_b32 exec_lo, s105                                    // 00000000AF4C: BEFE0069
	v_add_f32_e32 v6, s0, v6                                   // 00000000AF50: 060C0C00
	v_add_f32_e32 v2, s4, v2                                   // 00000000AF54: 06040404
	v_add_f32_e32 v6, s1, v6                                   // 00000000AF58: 060C0C01
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000AF5C: BF870112
	v_add_f32_e32 v2, s5, v2                                   // 00000000AF60: 06040405
	v_add_f32_e32 v6, s2, v6                                   // 00000000AF64: 060C0C02
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000AF68: BF870112
	v_add_f32_e32 v2, s6, v2                                   // 00000000AF6C: 06040406
	v_add_f32_e32 v6, s3, v6                                   // 00000000AF70: 060C0C03
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000AF74: BF870112
	v_add_f32_e32 v2, s7, v2                                   // 00000000AF78: 06040407
	v_add_f32_e32 v6, s4, v6                                   // 00000000AF7C: 060C0C04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AF80: BF870091
	v_add_f32_e32 v6, s5, v6                                   // 00000000AF84: 060C0C05
	v_add_f32_e32 v6, s6, v6                                   // 00000000AF88: 060C0C06
	v_readlane_b32 s0, v42, 12                                 // 00000000AF8C: D7600000 0001192A
	v_readlane_b32 s1, v42, 13                                 // 00000000AF94: D7600001 00011B2A
	v_readlane_b32 s2, v42, 14                                 // 00000000AF9C: D7600002 00011D2A
	v_readlane_b32 s3, v42, 15                                 // 00000000AFA4: D7600003 00011F2A
	v_readlane_b32 s4, v42, 16                                 // 00000000AFAC: D7600004 0001212A
	v_add_f32_e32 v7, s0, v7                                   // 00000000AFB4: 060E0E00
	v_readlane_b32 s5, v42, 17                                 // 00000000AFB8: D7600005 0001232A
	v_readlane_b32 s6, v42, 18                                 // 00000000AFC0: D7600006 0001252A
	v_readlane_b32 s7, v42, 19                                 // 00000000AFC8: D7600007 0001272A
	v_add_f32_e32 v1, s4, v1                                   // 00000000AFD0: 06020204
	v_add_f32_e32 v7, s1, v7                                   // 00000000AFD4: 060E0E01
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000AFD8: BF870112
	v_add_f32_e32 v1, s5, v1                                   // 00000000AFDC: 06020205
	v_add_f32_e32 v7, s2, v7                                   // 00000000AFE0: 060E0E02
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000AFE4: BF870112
	v_add_f32_e32 v1, s6, v1                                   // 00000000AFE8: 06020206
	v_add_f32_e32 v7, s3, v7                                   // 00000000AFEC: 060E0E03
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000AFF0: BF870112
	v_add_f32_e32 v1, s7, v1                                   // 00000000AFF4: 06020207
	v_add_f32_e32 v7, s4, v7                                   // 00000000AFF8: 060E0E04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000AFFC: BF870091
	v_add_f32_e32 v7, s5, v7                                   // 00000000B000: 060E0E05
	v_add_f32_e32 v7, s6, v7                                   // 00000000B004: 060E0E06
	v_readlane_b32 s0, v42, 4                                  // 00000000B008: D7600000 0001092A
	v_readlane_b32 s1, v42, 5                                  // 00000000B010: D7600001 00010B2A
	v_readlane_b32 s2, v42, 6                                  // 00000000B018: D7600002 00010D2A
	v_readlane_b32 s3, v42, 7                                  // 00000000B020: D7600003 00010F2A
	v_readlane_b32 s4, v42, 8                                  // 00000000B028: D7600004 0001112A
	v_add_f32_e32 v11, s0, v11                                 // 00000000B030: 06161600
	v_readlane_b32 s5, v42, 9                                  // 00000000B034: D7600005 0001132A
	v_readlane_b32 s6, v42, 10                                 // 00000000B03C: D7600006 0001152A
	v_readlane_b32 s7, v42, 11                                 // 00000000B044: D7600007 0001172A
	v_add_f32_e32 v9, s4, v9                                   // 00000000B04C: 06121204
	v_add_f32_e32 v11, s1, v11                                 // 00000000B050: 06161601
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000B054: BF870112
	v_add_f32_e32 v9, s5, v9                                   // 00000000B058: 06121205
	v_add_f32_e32 v11, s2, v11                                 // 00000000B05C: 06161602
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000B060: BF870112
	v_add_f32_e32 v9, s6, v9                                   // 00000000B064: 06121206
	v_add_f32_e32 v11, s3, v11                                 // 00000000B068: 06161603
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000B06C: BF870112
	v_add_f32_e32 v9, s7, v9                                   // 00000000B070: 06121207
	v_add_f32_e32 v11, s4, v11                                 // 00000000B074: 06161604
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000B078: BF870091
	v_add_f32_e32 v11, s5, v11                                 // 00000000B07C: 06161605
	v_add_f32_e32 v11, s6, v11                                 // 00000000B080: 06161606
	s_or_saveexec_b32 s105, -1                                 // 00000000B084: BEE922C1
	scratch_load_b32 v44, off, off offset:44                   // 00000000B088: DC51002C 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000B090: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000B094: BF8903F7
	v_readlane_b32 s0, v44, 12                                 // 00000000B098: D7600000 0001192C
	v_readlane_b32 s1, v44, 13                                 // 00000000B0A0: D7600001 00011B2C
	v_readlane_b32 s2, v44, 14                                 // 00000000B0A8: D7600002 00011D2C
	v_readlane_b32 s3, v44, 15                                 // 00000000B0B0: D7600003 00011F2C
	v_readlane_b32 s4, v44, 16                                 // 00000000B0B8: D7600004 0001212C
	v_readlane_b32 s5, v44, 17                                 // 00000000B0C0: D7600005 0001232C
	v_dual_add_f32 v4, s1, v4 :: v_dual_add_f32 v5, s0, v5     // 00000000B0C8: C9080801 04040A00
	v_readlane_b32 s6, v44, 18                                 // 00000000B0D0: D7600006 0001252C
	v_readlane_b32 s7, v44, 19                                 // 00000000B0D8: D7600007 0001272C
	v_dual_add_f32 v0, s85, v0 :: v_dual_add_f32 v1, s84, v1   // 00000000B0E0: C9080055 00000254
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 00000000B0E8: BF8701A4
	v_dual_add_f32 v4, s2, v4 :: v_dual_add_f32 v5, s1, v5     // 00000000B0EC: C9080802 04040A01
	v_readlane_b32 s68, v45, 30                                // 00000000B0F4: D7600044 00013D2D
	v_dual_add_f32 v0, s86, v0 :: v_dual_add_f32 v1, s85, v1   // 00000000B0FC: C9080056 00000255
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 00000000B104: BF8701A3
	v_dual_add_f32 v4, s3, v4 :: v_dual_add_f32 v5, s2, v5     // 00000000B108: C9080803 04040A02
	v_readlane_b32 s69, v45, 31                                // 00000000B110: D7600045 00013F2D
	v_dual_add_f32 v0, s87, v0 :: v_dual_add_f32 v1, s86, v1   // 00000000B118: C9080057 00000256
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000B120: BF870113
	v_dual_add_f32 v4, s4, v4 :: v_dual_add_f32 v5, s3, v5     // 00000000B124: C9080804 04040A03
	v_dual_add_f32 v0, s88, v0 :: v_dual_add_f32 v1, s87, v1   // 00000000B12C: C9080058 00000257
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000B134: BF870092
	v_add_f32_e32 v4, s5, v4                                   // 00000000B138: 06080805
	v_add_f32_e32 v4, s6, v4                                   // 00000000B13C: 06080806
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000B140: BF870001
	v_add_f32_e32 v4, s7, v4                                   // 00000000B144: 06080807
	v_readlane_b32 s0, v36, 22                                 // 00000000B148: D7600000 00012D24
	v_readlane_b32 s1, v36, 23                                 // 00000000B150: D7600001 00012F24
	v_readlane_b32 s2, v36, 24                                 // 00000000B158: D7600002 00013124
	v_readlane_b32 s3, v36, 25                                 // 00000000B160: D7600003 00013324
	v_readlane_b32 s4, v36, 26                                 // 00000000B168: D7600004 00013524
	v_readlane_b32 s5, v36, 27                                 // 00000000B170: D7600005 00013724
	v_add_f32_e32 v3, s1, v3                                   // 00000000B178: 06060601
	v_readlane_b32 s6, v36, 28                                 // 00000000B17C: D7600006 00013924
	v_readlane_b32 s7, v36, 29                                 // 00000000B184: D7600007 00013B24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000B18C: BF870093
	v_add_f32_e32 v3, s2, v3                                   // 00000000B190: 06060602
	v_dual_add_f32 v3, s3, v3 :: v_dual_add_f32 v2, s0, v2     // 00000000B194: C9080603 03020400
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000B19C: BF870091
	v_dual_add_f32 v3, s4, v3 :: v_dual_add_f32 v2, s1, v2     // 00000000B1A0: C9080604 03020401
	v_dual_add_f32 v3, s5, v3 :: v_dual_add_f32 v2, s2, v2     // 00000000B1A8: C9080605 03020402
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000B1B0: BF870091
	v_dual_add_f32 v3, s6, v3 :: v_dual_add_f32 v2, s3, v2     // 00000000B1B4: C9080606 03020403
	v_add_f32_e32 v3, s7, v3                                   // 00000000B1BC: 06060607
	v_readlane_b32 s0, v36, 14                                 // 00000000B1C0: D7600000 00011D24
	v_add_f32_e32 v0, s89, v0                                  // 00000000B1C8: 06000059
	v_readlane_b32 s1, v36, 15                                 // 00000000B1CC: D7600001 00011F24
	v_readlane_b32 s2, v36, 16                                 // 00000000B1D4: D7600002 00012124
	v_readlane_b32 s3, v36, 17                                 // 00000000B1DC: D7600003 00012324
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 00000000B1E4: BF870244
	v_dual_add_f32 v9, s0, v9 :: v_dual_add_f32 v0, s90, v0    // 00000000B1E8: C9081200 0900005A
	v_readlane_b32 s4, v36, 18                                 // 00000000B1F0: D7600004 00012524
	v_readlane_b32 s5, v36, 19                                 // 00000000B1F8: D7600005 00012724
	v_readlane_b32 s6, v36, 20                                 // 00000000B200: D7600006 00012924
	v_dual_add_f32 v9, s1, v9 :: v_dual_add_f32 v0, s91, v0    // 00000000B208: C9081201 0900005B
	v_readlane_b32 s7, v36, 21                                 // 00000000B210: D7600007 00012B24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000B218: BF870092
	v_dual_add_f32 v9, s2, v9 :: v_dual_add_f32 v10, s1, v10   // 00000000B21C: C9081202 090A1401
	v_dual_add_f32 v9, s3, v9 :: v_dual_add_f32 v10, s2, v10   // 00000000B224: C9081203 090A1402
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000B22C: BF870091
	v_add_f32_e32 v10, s3, v10                                 // 00000000B230: 06141403
	v_add_f32_e32 v10, s4, v10                                 // 00000000B234: 06141404
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000B238: BF870091
	v_add_f32_e32 v10, s5, v10                                 // 00000000B23C: 06141405
	v_add_f32_e32 v10, s6, v10                                 // 00000000B240: 06141406
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000B244: BF870001
	v_add_f32_e32 v10, s7, v10                                 // 00000000B248: 06141407
	v_readlane_b32 s0, v42, 30                                 // 00000000B24C: D7600000 00013D2A
	v_readlane_b32 s1, v42, 31                                 // 00000000B254: D7600001 00013F2A
	v_readlane_b32 s2, v36, 0                                  // 00000000B25C: D7600002 00010124
	v_readlane_b32 s4, v36, 2                                  // 00000000B264: D7600004 00010524
	v_readlane_b32 s3, v36, 1                                  // 00000000B26C: D7600003 00010324
	v_add_f32_e32 v19, s0, v19                                 // 00000000B274: 06262600
	v_readlane_b32 s5, v36, 3                                  // 00000000B278: D7600005 00010724
	v_readlane_b32 s6, v36, 4                                  // 00000000B280: D7600006 00010924
	v_add_f32_e32 v13, s4, v13                                 // 00000000B288: 061A1A04
	v_readlane_b32 s9, v36, 7                                  // 00000000B28C: D7600009 00010F24
	v_add_f32_e32 v19, s1, v19                                 // 00000000B294: 06262601
	v_readlane_b32 s7, v36, 5                                  // 00000000B298: D7600007 00010B24
	v_readlane_b32 s8, v36, 6                                  // 00000000B2A0: D7600008 00010D24
	v_add_f32_e32 v13, s5, v13                                 // 00000000B2A8: 061A1A05
	v_readlane_b32 s10, v36, 8                                 // 00000000B2AC: D760000A 00011124
	v_add_f32_e32 v19, s2, v19                                 // 00000000B2B4: 06262602
	v_readlane_b32 s11, v36, 9                                 // 00000000B2B8: D760000B 00011324
	v_readlane_b32 s12, v36, 10                                // 00000000B2C0: D760000C 00011524
	v_add_f32_e32 v13, s6, v13                                 // 00000000B2C8: 061A1A06
	v_readlane_b32 s13, v36, 11                                // 00000000B2CC: D760000D 00011724
	v_dual_add_f32 v19, s3, v19 :: v_dual_add_f32 v12, s9, v12 // 00000000B2D4: C9082603 130C1809
	v_readlane_b32 s14, v36, 12                                // 00000000B2DC: D760000E 00011924
	v_readlane_b32 s15, v36, 13                                // 00000000B2E4: D760000F 00011B24
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000B2EC: BF870003
	v_add_f32_e32 v19, s4, v19                                 // 00000000B2F0: 06262604
	s_or_saveexec_b32 s105, -1                                 // 00000000B2F4: BEE922C1
	scratch_load_b32 v44, off, off offset:32                   // 00000000B2F8: DC510020 2C7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000B300: BEFE0069
	v_dual_add_f32 v8, s68, v8 :: v_dual_add_f32 v19, s5, v19  // 00000000B304: C9081044 08122605
	v_dual_add_f32 v12, s10, v12 :: v_dual_add_f32 v13, s7, v13// 00000000B30C: C908180A 0C0C1A07
	s_waitcnt vmcnt(0)                                         // 00000000B314: BF8903F7
	v_readlane_b32 s36, v44, 6                                 // 00000000B318: D7600024 00010D2C
	v_readlane_b32 s70, v44, 0                                 // 00000000B320: D7600046 0001012C
	v_readlane_b32 s56, v45, 22                                // 00000000B328: D7600038 00012D2D
	v_dual_add_f32 v8, s69, v8 :: v_dual_add_f32 v13, s8, v13  // 00000000B330: C9081045 080C1A08
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 00000000B338: BF8701C4
	v_dual_add_f32 v12, s11, v12 :: v_dual_add_f32 v7, s36, v7 // 00000000B33C: C908180B 0C060E24
	v_readlane_b32 s37, v44, 7                                 // 00000000B344: D7600025 00010F2C
	v_readlane_b32 s57, v45, 23                                // 00000000B34C: D7600039 00012F2D
	v_dual_add_f32 v6, s56, v6 :: v_dual_add_f32 v19, s6, v19  // 00000000B354: C9080C38 06122606
	v_dual_add_f32 v8, s70, v8 :: v_dual_add_f32 v7, s37, v7   // 00000000B35C: C9081046 08060E25
	v_readlane_b32 s38, v44, 8                                 // 00000000B364: D7600026 0001112C
	v_readlane_b32 s71, v44, 1                                 // 00000000B36C: D7600047 0001032C
	v_readlane_b32 s58, v45, 24                                // 00000000B374: D760003A 0001312D
	v_dual_add_f32 v6, s57, v6 :: v_dual_add_f32 v13, s9, v13  // 00000000B37C: C9080C39 060C1A09
	v_readlane_b32 s39, v44, 9                                 // 00000000B384: D7600027 0001132C
	v_dual_add_f32 v12, s12, v12 :: v_dual_add_f32 v7, s38, v7 // 00000000B38C: C908180C 0C060E26
	v_readlane_b32 s59, v45, 25                                // 00000000B394: D760003B 0001332D
	v_readlane_b32 s40, v44, 10                                // 00000000B39C: D7600028 0001152C
	v_dual_add_f32 v6, s58, v6 :: v_dual_add_f32 v13, s10, v13 // 00000000B3A4: C9080C3A 060C1A0A
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)// 00000000B3AC: BF8701D4
	v_dual_add_f32 v8, s71, v8 :: v_dual_add_f32 v7, s39, v7   // 00000000B3B0: C9081047 08060E27
	v_add_f32_e32 v12, s13, v12                                // 00000000B3B8: 0618180D
	v_readlane_b32 s72, v44, 2                                 // 00000000B3BC: D7600048 0001052C
	v_readlane_b32 s60, v45, 26                                // 00000000B3C4: D760003C 0001352D
	s_load_b256 s[0:7], s[34:35], 0xfe0                        // 00000000B3CC: F40C0011 F8000FE0
	v_dual_add_f32 v7, s40, v7 :: v_dual_add_f32 v12, s14, v12 // 00000000B3D4: C9080E28 070C180E
	v_add_f32_e32 v6, s59, v6                                  // 00000000B3DC: 060C0C3B
	v_readlane_b32 s73, v44, 3                                 // 00000000B3E0: D7600049 0001072C
	v_readlane_b32 s61, v45, 27                                // 00000000B3E8: D760003D 0001372D
	v_add_f32_e32 v8, s72, v8                                  // 00000000B3F0: 06101048
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 00000000B3F4: BF870234
	v_dual_add_f32 v29, s15, v12 :: v_dual_add_f32 v6, s60, v6 // 00000000B3F8: C908180F 1D060C3C
	v_readlane_b32 s74, v44, 4                                 // 00000000B400: D760004A 0001092C
	v_readlane_b32 s62, v45, 28                                // 00000000B408: D760003E 0001392D
	v_add_f32_e32 v8, s73, v8                                  // 00000000B410: 06101049
	v_readlane_b32 s75, v44, 5                                 // 00000000B414: D760004B 00010B2C
	v_add_f32_e32 v6, s61, v6                                  // 00000000B41C: 060C0C3D
	v_readlane_b32 s63, v45, 29                                // 00000000B420: D760003F 00013B2D
	v_readlane_b32 s41, v44, 11                                // 00000000B428: D7600029 0001172C
	v_add_f32_e32 v8, s74, v8                                  // 00000000B430: 0610104A
	v_readlane_b32 s42, v44, 12                                // 00000000B434: D760002A 0001192C
	v_add_f32_e32 v6, s62, v6                                  // 00000000B43C: 060C0C3E
	v_readlane_b32 s43, v44, 13                                // 00000000B440: D760002B 00011B2C
	v_add_f32_e32 v28, s11, v13                                // 00000000B448: 06381A0B
	s_or_saveexec_b32 s105, -1                                 // 00000000B44C: BEE922C1
	v_mov_b32_e32 v44, v43                                     // 00000000B450: 7E58032B
	s_mov_b32 exec_lo, s105                                    // 00000000B454: BEFE0069
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000B458: BF870001
	v_readlane_b32 s44, v44, 22                                // 00000000B45C: D760002C 00012D2C
	v_readlane_b32 s45, v44, 23                                // 00000000B464: D760002D 00012F2C
	v_readlane_b32 s64, v45, 14                                // 00000000B46C: D7600040 00011D2D
	v_readlane_b32 s46, v44, 24                                // 00000000B474: D760002E 0001312C
	v_readlane_b32 s47, v44, 25                                // 00000000B47C: D760002F 0001332C
	s_waitcnt lgkmcnt(0)                                       // 00000000B484: BF89FC07
	v_dual_add_f32 v12, s44, v23 :: v_dual_add_f32 v13, s16, v24// 00000000B488: C9082E2C 0C0C3010
	v_add_f32_e32 v11, s64, v11                                // 00000000B490: 06161640
	v_readlane_b32 s65, v45, 15                                // 00000000B494: D7600041 00011F2D
	v_dual_add_f32 v5, s72, v5 :: v_dual_add_f32 v2, s60, v2   // 00000000B49C: C9080A48 0502043C
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 00000000B4A4: BF870234
	v_dual_add_f32 v12, s45, v12 :: v_dual_add_f32 v7, s41, v7 // 00000000B4A8: C908182D 0C060E29
	v_readlane_b32 s48, v44, 26                                // 00000000B4B0: D7600030 0001352C
	v_readlane_b32 s66, v45, 16                                // 00000000B4B8: D7600042 0001212D
	v_dual_add_f32 v13, s17, v13 :: v_dual_add_f32 v2, s61, v2 // 00000000B4C0: C9081A11 0D02043D
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 00000000B4C8: BF870234
	v_dual_add_f32 v12, s46, v12 :: v_dual_add_f32 v11, s65, v11// 00000000B4CC: C908182E 0C0A1641
	v_readlane_b32 s49, v44, 27                                // 00000000B4D4: D7600031 0001372C
	v_readlane_b32 s67, v45, 17                                // 00000000B4DC: D7600043 0001232D
	v_dual_add_f32 v5, s73, v5 :: v_dual_add_f32 v2, s62, v2   // 00000000B4E4: C9080A49 0502043E
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 00000000B4EC: BF870234
	v_dual_add_f32 v12, s47, v12 :: v_dual_add_f32 v7, s42, v7 // 00000000B4F0: C908182F 0C060E2A
	v_readlane_b32 s50, v44, 28                                // 00000000B4F8: D7600032 0001392C
	v_readlane_b32 s68, v45, 18                                // 00000000B500: D7600044 0001252D
	v_dual_add_f32 v13, s18, v13 :: v_dual_add_f32 v2, s63, v2 // 00000000B508: C9081A12 0D02043F
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 00000000B510: BF8701B4
	v_dual_add_f32 v12, s48, v12 :: v_dual_add_f32 v11, s66, v11// 00000000B514: C9081830 0C0A1642
	s_load_b256 s[8:15], s[34:35], 0x1120                      // 00000000B51C: F40C0211 F8001120
	v_readlane_b32 s69, v45, 19                                // 00000000B524: D7600045 0001272D
	v_dual_add_f32 v13, s19, v13 :: v_dual_add_f32 v22, 0, v22 // 00000000B52C: C9081A13 0D162C80
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 00000000B534: BF8701B3
	v_dual_add_f32 v12, s49, v12 :: v_dual_add_f32 v5, s74, v5 // 00000000B538: C9081831 0C040A4A
	v_dual_add_f32 v11, s67, v11 :: v_dual_add_f32 v2, s0, v2  // 00000000B540: C9081643 0B020400
	v_readlane_b32 s70, v45, 20                                // 00000000B548: D7600046 0001292D
	v_dual_add_f32 v26, s50, v12 :: v_dual_add_f32 v5, s75, v5 // 00000000B550: C9081832 1A040A4B
	v_add_f32_e32 v12, 0, v19                                  // 00000000B558: 06182680
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 00000000B55C: BF870244
	v_dual_add_f32 v11, s68, v11 :: v_dual_add_f32 v2, s1, v2  // 00000000B560: C9081644 0B020401
	v_readlane_b32 s72, v45, 6                                 // 00000000B568: D7600048 00010D2D
	v_dual_add_f32 v13, s20, v13 :: v_dual_add_f32 v14, s48, v14// 00000000B570: C9081A14 0D0E1C30
	v_readlane_b32 s73, v45, 7                                 // 00000000B578: D7600049 00010F2D
	v_dual_add_f32 v11, s69, v11 :: v_dual_add_f32 v18, s20, v18// 00000000B580: C9081645 0B122414
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 00000000B588: BF870244
	v_dual_add_f32 v5, s72, v5 :: v_dual_add_f32 v2, s2, v2    // 00000000B58C: C9080A48 05020402
	v_readlane_b32 s53, v44, 31                                // 00000000B594: D7600035 00013F2C
	v_readlane_b32 s74, v45, 8                                 // 00000000B59C: D760004A 0001112D
	v_dual_add_f32 v13, s21, v13 :: v_dual_add_f32 v14, s49, v14// 00000000B5A4: C9081A15 0D0E1C31
	v_dual_add_f32 v5, s73, v5 :: v_dual_add_f32 v2, s3, v2    // 00000000B5AC: C9080A49 05020403
	v_dual_add_f32 v11, s70, v11 :: v_dual_add_f32 v18, s21, v18// 00000000B5B4: C9081646 0B122415
	v_readlane_b32 s51, v44, 29                                // 00000000B5BC: D7600033 00013B2C
	v_readlane_b32 s75, v45, 9                                 // 00000000B5C4: D760004B 0001132D
	v_dual_add_f32 v23, s22, v13 :: v_dual_add_f32 v14, s50, v14// 00000000B5CC: C9081A16 170E1C32
	v_dual_add_f32 v5, s74, v5 :: v_dual_add_f32 v4, s73, v4   // 00000000B5D4: C9080A4A 05040849
	v_dual_add_f32 v13, 0, v25 :: v_dual_add_f32 v18, s22, v18 // 00000000B5DC: C9083280 0D122416
	v_add_f32_e32 v1, s40, v1                                  // 00000000B5E4: 06020228
	v_add_f32_e32 v9, s68, v9                                  // 00000000B5E8: 06121244
	v_dual_add_f32 v3, s1, v3 :: v_dual_add_f32 v0, s93, v0    // 00000000B5EC: C9080601 0300005D
	s_waitcnt lgkmcnt(0)                                       // 00000000B5F4: BF89FC07
	v_dual_add_f32 v15, s53, v15 :: v_dual_add_f32 v10, s9, v10// 00000000B5F8: C9081E35 0F0A1409
	v_readlane_b32 s52, v44, 30                                // 00000000B600: D7600034 00013D2C
	v_readlane_b32 s54, v45, 0                                 // 00000000B608: D7600036 0001012D
	v_dual_add_f32 v5, s75, v5 :: v_dual_add_f32 v14, s51, v14 // 00000000B610: C9080A4B 050E1C33
	v_dual_add_f32 v4, s74, v4 :: v_dual_add_f32 v1, s41, v1   // 00000000B618: C908084A 04000229
	v_dual_add_f32 v18, s23, v18 :: v_dual_add_f32 v9, s69, v9 // 00000000B620: C9082417 12081245
	v_dual_add_f32 v17, s25, v17 :: v_dual_add_f32 v0, s94, v0 // 00000000B628: C9082219 1100005E
	v_dual_add_f32 v3, s2, v3 :: v_dual_add_f32 v10, s10, v10  // 00000000B630: C9080602 030A140A
	v_readlane_b32 s55, v45, 1                                 // 00000000B638: D7600037 0001032D
	v_readlane_b32 s71, v45, 21                                // 00000000B640: D7600047 00012B2D
	v_dual_add_f32 v14, s52, v14 :: v_dual_add_f32 v1, s42, v1 // 00000000B648: C9081C34 0E00022A
	v_add_f32_e32 v18, s24, v18                                // 00000000B650: 06242418
	v_dual_add_f32 v4, s75, v4 :: v_dual_add_f32 v9, s70, v9   // 00000000B654: C908084B 04081246
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000B65C: BF870003
	v_dual_add_f32 v14, s53, v14 :: v_dual_add_f32 v15, s54, v15// 00000000B660: C9081C35 0E0E1E36
	v_dual_add_f32 v0, s95, v0 :: v_dual_add_f32 v17, s26, v17 // 00000000B668: C908005F 0010221A
	v_add_f32_e32 v10, s11, v10                                // 00000000B670: 0614140B
	v_readlane_b32 s56, v45, 2                                 // 00000000B674: D7600038 0001052D
	v_readlane_b32 s76, v45, 10                                // 00000000B67C: D760004C 0001152D
	v_add_f32_e32 v1, s43, v1                                  // 00000000B684: 0602022B
	v_dual_add_f32 v9, s71, v9 :: v_dual_add_f32 v18, s25, v18 // 00000000B688: C9081247 09122419
	v_dual_add_f32 v3, s3, v3 :: v_dual_add_f32 v0, s96, v0    // 00000000B690: C9080603 03000060
	v_dual_add_f32 v15, s55, v15 :: v_dual_add_f32 v10, s12, v10// 00000000B698: C9081E37 0F0A140C
	v_add_f32_e32 v17, s27, v17                                // 00000000B6A0: 0622221B
	v_readlane_b32 s57, v45, 3                                 // 00000000B6A4: D7600039 0001072D
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000B6AC: BF870004
	v_add_f32_e32 v0, s97, v0                                  // 00000000B6B0: 06000061
	v_readlane_b32 s77, v45, 11                                // 00000000B6B4: D760004D 0001172D
	v_dual_add_f32 v4, s76, v4 :: v_dual_add_f32 v1, s92, v1   // 00000000B6BC: C908084C 0400025C
	v_dual_add_f32 v14, s54, v14 :: v_dual_add_f32 v9, s8, v9  // 00000000B6C4: C9081C36 0E081208
	v_dual_add_f32 v3, s4, v3 :: v_dual_add_f32 v10, s13, v10  // 00000000B6CC: C9080604 030A140D
	v_dual_add_f32 v15, s56, v15 :: v_dual_add_f32 v0, s98, v0 // 00000000B6D4: C9081E38 0F000062
	v_add_f32_e32 v17, s28, v17                                // 00000000B6DC: 0622221C
	v_readlane_b32 s58, v45, 4                                 // 00000000B6E0: D760003A 0001092D
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000B6E8: BF870004
	v_add_f32_e32 v10, s14, v10                                // 00000000B6EC: 0614140E
	v_readlane_b32 s78, v45, 12                                // 00000000B6F0: D760004E 0001192D
	v_dual_add_f32 v1, s93, v1 :: v_dual_add_f32 v18, s26, v18 // 00000000B6F8: C908025D 0112241A
	v_dual_add_f32 v4, s77, v4 :: v_dual_add_f32 v9, s9, v9    // 00000000B700: C908084D 04081209
	v_dual_add_f32 v24, s55, v14 :: v_dual_add_f32 v3, s5, v3  // 00000000B708: C9081C37 18020605
	v_dual_add_f32 v0, s99, v0 :: v_dual_add_f32 v15, s57, v15 // 00000000B710: C9080063 000E1E39
	v_dual_add_f32 v10, s15, v10 :: v_dual_add_f32 v17, s29, v17// 00000000B718: C908140F 0A10221D
	v_add_f32_e32 v20, 0, v20                                  // 00000000B720: 06282880
	v_readlane_b32 s59, v45, 5                                 // 00000000B724: D760003B 00010B2D
	v_readlane_b32 s79, v45, 13                                // 00000000B72C: D760004F 00011B2D
	v_dual_add_f32 v1, s94, v1 :: v_dual_add_f32 v4, s78, v4   // 00000000B734: C908025E 0104084E
	v_add_f32_e32 v9, s10, v9                                  // 00000000B73C: 0612120A
	v_add_f32_e32 v3, s6, v3                                   // 00000000B740: 06060606
	v_add_f32_e32 v15, s58, v15                                // 00000000B744: 061E1E3A
	v_add_f32_e32 v17, s30, v17                                // 00000000B748: 0622221E
	v_dual_add_f32 v1, s95, v1 :: v_dual_add_f32 v14, 0, v28   // 00000000B74C: C908025F 010E3880
	v_dual_add_f32 v19, s27, v18 :: v_dual_add_f32 v18, 0, v16 // 00000000B754: C908241B 13122080
	v_add_f32_e32 v16, 0, v21                                  // 00000000B75C: 06202A80
	v_dual_add_f32 v9, s11, v9 :: v_dual_add_f32 v4, s79, v4   // 00000000B760: C908120B 0904084F
	v_add_f32_e32 v3, s7, v3                                   // 00000000B768: 06060607
	v_add_f32_e32 v25, s59, v15                                // 00000000B76C: 06321E3B
	v_add_f32_e32 v21, s31, v17                                // 00000000B770: 062A221F
	v_add_f32_e32 v17, 0, v27                                  // 00000000B774: 06223680
	v_add_f32_e32 v15, 0, v29                                  // 00000000B778: 061E3A80
	s_add_i32 s104, s104, 1                                    // 00000000B77C: 81688168
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000B780: BF870009
	s_cmp_eq_u32 s104, 8                                       // 00000000B784: BF068868
	s_cbranch_scc1 110                                         // 00000000B788: BFA2006E <r_3_3_3_8_8_8+0xa344>
	v_readlane_b32 s12, v42, 20                                // 00000000B78C: D760000C 0001292A
	v_readlane_b32 s13, v42, 21                                // 00000000B794: D760000D 00012B2A
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000B79C: BF870001
	s_mov_b32 s31, s13                                         // 00000000B7A0: BE9F000D
	s_branch 55531                                             // 00000000B7A4: BFA0D8EB <r_3_3_3_8_8_8+0x554>
	v_writelane_b32 v38, s0, 22                                // 00000000B7A8: D7610026 00012C00
	v_writelane_b32 v45, s10, 0                                // 00000000B7B0: D761002D 0001000A
	s_mov_b32 vcc_hi, 0                                        // 00000000B7B8: BEEB0080
	v_writelane_b32 v38, s1, 23                                // 00000000B7BC: D7610026 00012E01
	v_writelane_b32 v45, s11, 1                                // 00000000B7C4: D761002D 0001020B
	v_writelane_b32 v38, s2, 24                                // 00000000B7CC: D7610026 00013002
	v_writelane_b32 v45, s12, 2                                // 00000000B7D4: D761002D 0001040C
	v_writelane_b32 v38, s3, 25                                // 00000000B7DC: D7610026 00013203
	v_writelane_b32 v45, s13, 3                                // 00000000B7E4: D761002D 0001060D
	v_writelane_b32 v38, s4, 26                                // 00000000B7EC: D7610026 00013404
	v_writelane_b32 v45, s14, 4                                // 00000000B7F4: D761002D 0001080E
	v_writelane_b32 v38, s5, 27                                // 00000000B7FC: D7610026 00013605
	v_writelane_b32 v45, s15, 5                                // 00000000B804: D761002D 00010A0F
	v_writelane_b32 v38, s6, 28                                // 00000000B80C: D7610026 00013806
	v_writelane_b32 v38, s7, 29                                // 00000000B814: D7610026 00013A07
	v_writelane_b32 v45, s0, 6                                 // 00000000B81C: D761002D 00010C00
	v_writelane_b32 v38, s8, 30                                // 00000000B824: D7610026 00013C08
	v_writelane_b32 v45, s1, 7                                 // 00000000B82C: D761002D 00010E01
	v_writelane_b32 v38, s9, 31                                // 00000000B834: D7610026 00013E09
	v_writelane_b32 v45, s2, 8                                 // 00000000B83C: D761002D 00011002
	v_writelane_b32 v45, s3, 9                                 // 00000000B844: D761002D 00011203
	v_writelane_b32 v45, s4, 10                                // 00000000B84C: D761002D 00011404
	v_writelane_b32 v45, s5, 11                                // 00000000B854: D761002D 00011605
	v_writelane_b32 v45, s6, 12                                // 00000000B85C: D761002D 00011806
	v_writelane_b32 v45, s7, 13                                // 00000000B864: D761002D 00011A07
	v_writelane_b32 v45, s0, 14                                // 00000000B86C: D761002D 00011C00
	v_writelane_b32 v45, s1, 15                                // 00000000B874: D761002D 00011E01
	v_writelane_b32 v45, s2, 16                                // 00000000B87C: D761002D 00012002
	v_writelane_b32 v45, s3, 17                                // 00000000B884: D761002D 00012203
	v_writelane_b32 v45, s4, 18                                // 00000000B88C: D761002D 00012404
	v_writelane_b32 v45, s5, 19                                // 00000000B894: D761002D 00012605
	v_writelane_b32 v45, s6, 20                                // 00000000B89C: D761002D 00012806
	v_writelane_b32 v45, s7, 21                                // 00000000B8A4: D761002D 00012A07
	v_writelane_b32 v45, s0, 22                                // 00000000B8AC: D761002D 00012C00
	v_writelane_b32 v45, s1, 23                                // 00000000B8B4: D761002D 00012E01
	v_writelane_b32 v45, s2, 24                                // 00000000B8BC: D761002D 00013002
	v_writelane_b32 v45, s3, 25                                // 00000000B8C4: D761002D 00013203
	v_writelane_b32 v45, s4, 26                                // 00000000B8CC: D761002D 00013404
	v_writelane_b32 v45, s5, 27                                // 00000000B8D4: D761002D 00013605
	v_writelane_b32 v45, s6, 28                                // 00000000B8DC: D761002D 00013806
	v_writelane_b32 v45, s7, 29                                // 00000000B8E4: D761002D 00013A07
	v_writelane_b32 v45, s0, 30                                // 00000000B8EC: D761002D 00013C00
	v_writelane_b32 v44, s2, 0                                 // 00000000B8F4: D761002C 00010002
	v_writelane_b32 v45, s1, 31                                // 00000000B8FC: D761002D 00013E01
	v_writelane_b32 v44, s3, 1                                 // 00000000B904: D761002C 00010203
	v_writelane_b32 v44, s4, 2                                 // 00000000B90C: D761002C 00010404
	v_writelane_b32 v44, s5, 3                                 // 00000000B914: D761002C 00010605
	v_writelane_b32 v44, s6, 4                                 // 00000000B91C: D761002C 00010806
	v_writelane_b32 v44, s7, 5                                 // 00000000B924: D761002C 00010A07
	s_or_saveexec_b32 s105, -1                                 // 00000000B92C: BEE922C1
	scratch_store_b32 off, v44, off offset:32                  // 00000000B930: DC690020 007C2C00
	s_mov_b32 exec_lo, s105                                    // 00000000B938: BEFE0069
	s_cbranch_execnz 61115                                     // 00000000B93C: BFA6EEBB <r_3_3_3_8_8_8+0x5e2c>
	s_branch 61272                                             // 00000000B940: BFA0EF58 <r_3_3_3_8_8_8+0x60a4>
	v_dual_mov_b32 v35, 0 :: v_dual_mul_f32 v28, 0x3b272f05, v5// 00000000B944: CA060080 231C0AFF 3B272F05
	v_mul_f32_e32 v27, 0x3b3f112b, v8                          // 00000000B950: 103610FF 3B3F112B
	v_mul_f32_e32 v29, 0x3b3f112b, v4                          // 00000000B958: 103A08FF 3B3F112B
	v_mul_f32_e32 v30, 0x3b272f05, v26                         // 00000000B960: 103C34FF 3B272F05
	v_mul_f32_e32 v31, 0x3b124925, v24                         // 00000000B968: 103E30FF 3B124925
	v_mul_f32_e32 v32, 0x3b272f05, v25                         // 00000000B970: 104032FF 3B272F05
	v_mul_f32_e32 v33, 0x3b3f112b, v22                         // 00000000B978: 10422CFF 3B3F112B
	v_mul_f32_e32 v34, 0x3b272f05, v18                         // 00000000B980: 104424FF 3B272F05
	s_or_saveexec_b32 s105, -1                                 // 00000000B988: BEE922C1
	scratch_load_b32 v45, off, off offset:8                    // 00000000B98C: DC510008 2D7C0000
	s_mov_b32 exec_lo, s105                                    // 00000000B994: BEFE0069
	s_waitcnt vmcnt(0)                                         // 00000000B998: BF8903F7
	v_readlane_b32 s0, v45, 7                                  // 00000000B99C: D7600000 00010F2D
	v_readlane_b32 s1, v45, 8                                  // 00000000B9A4: D7600001 0001112D
	v_mul_f32_e32 v24, 0x3b3f112b, v20                         // 00000000B9AC: 103028FF 3B3F112B
	v_mul_f32_e32 v25, 0x3b272f05, v6                          // 00000000B9B4: 10320CFF 3B272F05
	v_mul_f32_e32 v26, 0x3b124925, v2                          // 00000000B9BC: 103404FF 3B124925
	v_mul_f32_e32 v2, 0x3b124925, v11                          // 00000000B9C4: 100416FF 3B124925
	s_clause 0x1                                               // 00000000B9CC: BF850001
	global_store_b128 v35, v[27:30], s[0:1]                    // 00000000B9D0: DC760000 00001B23
	global_store_b128 v35, v[31:34], s[0:1] offset:16          // 00000000B9D8: DC760010 00001F23
	v_mul_f32_e32 v27, 0x3b272f05, v3                          // 00000000B9E0: 103606FF 3B272F05
	v_mul_f32_e32 v3, 0x3b000000, v9                           // 00000000B9E8: 100612FF 3B000000
	v_mul_f32_e32 v4, 0x3b124925, v10                          // 00000000B9F0: 100814FF 3B124925
	v_mul_f32_e32 v5, 0x3b272f05, v13                          // 00000000B9F8: 100A1AFF 3B272F05
	v_mul_f32_e32 v8, 0x3b124925, v16                          // 00000000BA00: 101020FF 3B124925
	v_mul_f32_e32 v9, 0x3b272f05, v17                          // 00000000BA08: 101222FF 3B272F05
	v_mul_f32_e32 v10, 0x3b3f112b, v7                          // 00000000BA10: 10140EFF 3B3F112B
	v_mul_f32_e32 v11, 0x3b272f05, v1                          // 00000000BA18: 101602FF 3B272F05
	v_mul_f32_e32 v16, 0x3b3f112b, v0                          // 00000000BA20: 102000FF 3B3F112B
	v_mul_f32_e32 v17, 0x3b272f05, v23                         // 00000000BA28: 10222EFF 3B272F05
	v_mul_f32_e32 v18, 0x3b124925, v19                         // 00000000BA30: 102426FF 3B124925
	v_mul_f32_e32 v19, 0x3b272f05, v21                         // 00000000BA38: 10262AFF 3B272F05
	v_mul_f32_e32 v12, 0x3b3f112b, v12                         // 00000000BA40: 101818FF 3B3F112B
	v_mul_f32_e32 v13, 0x3b272f05, v14                         // 00000000BA48: 101A1CFF 3B272F05
	v_mul_f32_e32 v14, 0x3b3f112b, v15                         // 00000000BA50: 101C1EFF 3B3F112B
	v_readlane_b32 s2, v45, 9                                  // 00000000BA58: D7600002 0001132D
	v_readlane_b32 s3, v45, 10                                 // 00000000BA60: D7600003 0001152D
	s_clause 0x4                                               // 00000000BA68: BF850004
	global_store_b128 v35, v[24:27], s[0:1] offset:32          // 00000000BA6C: DC760020 00001823
	global_store_b128 v35, v[2:5], s[0:1] offset:48            // 00000000BA74: DC760030 00000223
	global_store_b128 v35, v[8:11], s[0:1] offset:64           // 00000000BA7C: DC760040 00000823
	global_store_b128 v35, v[16:19], s[0:1] offset:80          // 00000000BA84: DC760050 00001023
	global_store_b96 v35, v[12:14], s[0:1] offset:96           // 00000000BA8C: DC720060 00000C23
	s_endpgm                                                   // 00000000BA94: BFB00000

.rodata
.amdhsa_kernel kernel
  .amdhsa_enable_private_segment 1
  .amdhsa_private_segment_fixed_size 116
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel
