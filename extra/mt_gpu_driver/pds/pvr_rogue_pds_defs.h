/*
 * Copyright © 2022 Imagination Technologies Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef PVR_ROGUE_PDS_DEFS_H
#define PVR_ROGUE_PDS_DEFS_H

#include <stdint.h>

/* Instruction type C */
#define PVR_ROGUE_PDSINST_OPCODEC_MASK (0x0000000FU)
/* 64 bit add*/
#define PVR_ROGUE_PDSINST_OPCODEC_ADD64 UINT32_C(0x00000008)
/* 32 bit add*/
#define PVR_ROGUE_PDSINST_OPCODEC_ADD32 UINT32_C(0x00000009)
/* Shift and/or Logic Operation (64 bit)*/
#define PVR_ROGUE_PDSINST_OPCODEC_SFTLP64 UINT32_C(0x0000000a)
/* Compare and set predicate*/
#define PVR_ROGUE_PDSINST_OPCODEC_CMP UINT32_C(0x0000000b)
/* Branch and/or select predicate*/
#define PVR_ROGUE_PDSINST_OPCODEC_BRA UINT32_C(0x0000000c)
/* Umbrella OpcodeSP instructions*/
#define PVR_ROGUE_PDSINST_OPCODEC_SP UINT32_C(0x0000000d)
/* Multiply Accumulate with DOUD*/
#define PVR_ROGUE_PDSINST_OPCODEC_DDMAD UINT32_C(0x0000000e)
/* DOUT Command*/
#define PVR_ROGUE_PDSINST_OPCODEC_DOUT UINT32_C(0x0000000f)

/* Logical Operation */
#define PVR_ROGUE_PDSINST_LOP_MASK (0x00000007U)
#define PVR_ROGUE_PDSINST_LOP_NONE (0x00000000U)
#define PVR_ROGUE_PDSINST_LOP_NOT (0x00000001U)
#define PVR_ROGUE_PDSINST_LOP_AND (0x00000002U)
#define PVR_ROGUE_PDSINST_LOP_OR (0x00000003U)
#define PVR_ROGUE_PDSINST_LOP_XOR (0x00000004U)
#define PVR_ROGUE_PDSINST_LOP_XNOR (0x00000005U)
#define PVR_ROGUE_PDSINST_LOP_NAND (0x00000006U)
#define PVR_ROGUE_PDSINST_LOP_NOR (0x00000007U)

/* 64-bit Source Temps and Persistent Temps. */
#define PVR_ROGUE_PDSINST_REGS64TP_MASK (0x0000001FU)
#define PVR_ROGUE_PDSINST_REGS64TP_TEMP64 (0U)
#define PVR_ROGUE_PDSINST_REGS64TP_TEMP64_LOWER (0U)
#define PVR_ROGUE_PDSINST_REGS64TP_TEMP64_UPPER (15U)
#define PVR_ROGUE_PDSINST_REGS64TP_PTEMP64 (1U)
#define PVR_ROGUE_PDSINST_REGS64TP_PTEMP64_LOWER (16U)
#define PVR_ROGUE_PDSINST_REGS64TP_PTEMP64_UPPER (31U)

/* 32-bit Registers - 32-bit aligned. */
#define PVR_ROGUE_PDSINST_REGS32_MASK (0x000000FFU)
#define PVR_ROGUE_PDSINST_REGS32_CONST32 (0U)
#define PVR_ROGUE_PDSINST_REGS32_CONST32_LOWER (0U)
#define PVR_ROGUE_PDSINST_REGS32_CONST32_UPPER (127U)
#define PVR_ROGUE_PDSINST_REGS32_TEMP32 (1U)
#define PVR_ROGUE_PDSINST_REGS32_TEMP32_LOWER (128U)
#define PVR_ROGUE_PDSINST_REGS32_TEMP32_UPPER (159U)
#define PVR_ROGUE_PDSINST_REGS32_PTEMP32 (2U)
#define PVR_ROGUE_PDSINST_REGS32_PTEMP32_LOWER (192U)
#define PVR_ROGUE_PDSINST_REGS32_PTEMP32_UPPER (223U)

/* cc ? if im then
 * cc ?     dst = (*src0 lop *src1) << src2
 * cc ? else
 * cc ?     dst = (*src0 lop *src1) << *src2
 *
 * Take the logical operation of the 2 sources, and shift to a 64 bit result.
 * For unary operator NOT, *src0 is taken as the logical operand; for operator
 * NONE, an unmodified *src0 is shifted. If IM is set use SFT as a direct shift
 * value, otherwise use an address to obtain the shift value. The shift value
 * (SRC2) is treated as a 2's complement encoded signed value. A negative value
 * encodes a right shift. Values are clamped to the range [-63,63].
 */
#define PVR_ROGUE_PDSINST_SFTLP64_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_SFTLP64_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_SFTLP64_OPCODE_DEFAULT (0xA0000000U) /* SFTLP64 */
#define PVR_ROGUE_PDSINST_SFTLP64_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_SFTLP64_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_SFTLP64_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_SFTLP64_LOP_SHIFT (24U)
#define PVR_ROGUE_PDSINST_SFTLP64_IM_SHIFT (23U)
#define PVR_ROGUE_PDSINST_SFTLP64_IM_ENABLE (0x00800000U)
#define PVR_ROGUE_PDSINST_SFTLP64_SRC0_SHIFT (18U)
#define PVR_ROGUE_PDSINST_SFTLP64_SRC1_SHIFT (13U)
#define PVR_ROGUE_PDSINST_SFTLP64_SRC2_SHIFT (5U)
#define PVR_ROGUE_PDSINST_SFTLP64_DST_SHIFT (0U)

/* Instruction type B */
#define PVR_ROGUE_PDSINST_OPCODEB_MASK (0x00000007U)
/* Shift and/or Logic Operation (32 bit) */
#define PVR_ROGUE_PDSINST_OPCODEB_SFTLP32 UINT32_C(0x00000002)
/* Vertex Stream Out DMA Command */
#define PVR_ROGUE_PDSINST_OPCODEB_STM UINT32_C(0x00000003)

/* 32-bit Source Temps. */
#define PVR_ROGUE_PDSINST_REGS32T_MASK (0x0000001FU)
#define PVR_ROGUE_PDSINST_REGS32T_TEMP32 (0U)
#define PVR_ROGUE_PDSINST_REGS32T_TEMP32_LOWER (0U)
#define PVR_ROGUE_PDSINST_REGS32T_TEMP32_UPPER (31U)

/* 32-bit Source Temps and Persistent Temps. */
#define PVR_ROGUE_PDSINST_REGS32TP_MASK (0x0000003FU)
#define PVR_ROGUE_PDSINST_REGS32TP_TEMP32 (0U)
#define PVR_ROGUE_PDSINST_REGS32TP_TEMP32_LOWER (0U)
#define PVR_ROGUE_PDSINST_REGS32TP_TEMP32_UPPER (31U)
#define PVR_ROGUE_PDSINST_REGS32TP_PTEMP32 (1U)
#define PVR_ROGUE_PDSINST_REGS32TP_PTEMP32_LOWER (32U)
#define PVR_ROGUE_PDSINST_REGS32TP_PTEMP32_UPPER (63U)

/* cc ? if im then
 * cc ?     dst = (*src0 lop *src1) << src2
 * cc ? else
 * cc ?     dst = (*src0 lop *src1) << *src2
 *
 * Take the logical operation of the 2 sources, and shift to a 32 bit result.
 * For unary operator NOT, *src0 is taken as the logical operand; for operator
 * NONE, an unmodified *src0 is shifted.If IM is set, use the shift value SFT
 * (SRC2) as a direct shift value, otherwise use an address to obtain the shift
 * value. SFT (SRC2) is treated as a 2's complement encoded signed value. A
 * negative value encodes a right shift. Values are clamped to the range
 * [-31,31].
 */
#define PVR_ROGUE_PDSINST_SFTLP32_OPCODE_SHIFT (29U)
#define PVR_ROGUE_PDSINST_SFTLP32_OPCODE_CLRMSK (0x1FFFFFFFU)
#define PVR_ROGUE_PDSINST_SFTLP32_OPCODE_DEFAULT (0x40000000U) /* SFTLP32 */
#define PVR_ROGUE_PDSINST_SFTLP32_IM_SHIFT (28U)
#define PVR_ROGUE_PDSINST_SFTLP32_IM_ENABLE (0x10000000U)
#define PVR_ROGUE_PDSINST_SFTLP32_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_SFTLP32_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_SFTLP32_LOP_SHIFT (24U)
#define PVR_ROGUE_PDSINST_SFTLP32_SRC0_SHIFT (19U)
#define PVR_ROGUE_PDSINST_SFTLP32_SRC1_SHIFT (11U)
#define PVR_ROGUE_PDSINST_SFTLP32_SRC2_SHIFT (5U)
#define PVR_ROGUE_PDSINST_SFTLP32_DST_SHIFT (0U)

/* The stream being processed within the vertex, selects 1 of 4 streams. */
#define PVR_ROGUE_PDSINST_SO_MASK (0x00000003U)

/* An instruction to enable the 'Streaming Out' of data to memory.
 *
 * This instruction can only be used when called from a Stream Output Program
 * (see
 *
 * Stream output configuration words, as it reads its source data from unified
 * vertex store within the TA.
 *
 * Stream Out programs use the vertex data master, but are called from the TA.
 * They do not execute on the USC. If synchronization is required with the
 * control stream to the next draw call, a DOUTV command must be used when
 * stream out finishes for the current draw call. The VDM must have a
 * corresponding entry in the control stream indicating when it should wait for
 * the PDS.
 *
 * As SRC0, SRC1 needs to be held from program to program it is assumed these
 * are in persistent temps. There are 32 (dword) persistent temps, 8 of which
 * are required to support 4 streams. The driver needs to manage the allocation
 * of these. If the value needs to be carried from one geometry job to another,
 * it will need to be loaded from memory at the start of the geometry job, and
 * stored at the end of it (using a state program in the input control stream).
 *
 * When a new buffer is altered which was in use, the driver will need to fence
 * in order to make sure that the preceding operation have completed before the
 * persistent temps are updated.
 *
 * It is assumed that the USC compiler will optimize the stream order to keep
 * data which is contiguous in the output vertex (going to memory)
 * together. This will enable multiple words to be streamed out in a single
 * DMA. This will reduce the processing load on the TA.
 *
 * The sources are read from within the constant, temporary stores of the PDS,
 * and have the following meaning.
 *
 * If the buffer is being appended to then persistent constants need to be
 * stored to memory at the end of the geometry job, and reloaded at the start
 * of the next job (as another context may be run).
 *
 * ccs ? if (so_address + (so_vosize * so_primtype)) <= so_limit then
 *
 * dma the data from the vbg, and write it into memory. so_vioff is
 * an offset into the current vertex.
 * ccs ?      for (so_vertex=0 ; so_vertex < so_primtype; so_vertex++)
 * ccs ?         for (i=0 ; i < so_dmasize; i++)
 * ccs ?            *(so_address + so_vooff + i + (so_vertex * so_vosize)) =
 * 				        readvertexvbg(so_vioff + i + (so_vertex * stream_size))
 *
 * ccs ?     if so_eop then
 * ccs ?         so_address = so_address + (so_vosize * so_primtype)
 * ccs ?         so_primwritten = so_primwritten + 1
 * ccs ?
 * end if
 *
 * ccs ? else
 *
 * ccs ?     setp(so_overflow_predicate[so])
 * ccs ?     [so_overflow_predicate[global]]
 *
 * ccs ? end if
 *
 * if so_eop then
 * 	so_primneeded = so_primneeded + 1
 * end if
 *
 * The VBG presents a stream when outputted from the shader. A bit is set in the
 * input register indicating which stream is present. The PDS is called on a per
 * primitive basis. In simple geometry this is per input triangle, strip etc.,
 * in geometry shader land this is per output primitive from the geometry
 * shader. Primitives are unraveled to remove vertex sharing. The PDS is called
 * in submission order. The PDS program needs to be written for the primitive
 * which is being emitted.
 *
 * Example
 *
 * Data is actually going into three buffers (this is defined elsewhere).
 * SO_VERTEX0.Pos.XY -> buffer0
 * SO_VERTEX0.Mult.XY -> buffer0
 * SO_VERTEX1.Add.XY -> buffer1
 *
 * SO_VERTEX0.Pos.ZW -> buffer2
 *
 * Persistent temps:
 * pt0 = Buffer0 start address;
 * pt1 = Buffer1 start address;
 * pt2 = Buffer2 start address;
 * pt3 = 0 (buffer0 primwritten/needed)
 * pt4 = 0 (buffer1 primwritten/needed)
 * pt5 = 0 (buffer2 primwritten/needed)
 *
 * Constants:
 * c0 = Buffer 0 top
 * c1 = Buffer 1 top
 * c2 = Buffer 2 top
 * c3 = SRC2,3 for Pos.XY: VOOFF = 0, DMASIZE = 2, SO_VIOFF = 0, EOP = 0
 * c4 = SRC2,3 for Mult: VOSIZE = 4, VOOFF = 2, DMASIZE = 2, SO_VIOFF = 2, EOP =
 * 1 c5 = SRC2,3 for Pos.ZW: VOSIZE=2, VOOFF = 0, DMASIZE = 2, SO_VIOFF = 0, EOP
 * = 1 c6 = SRC2,3 for Add: VOSIZE=2, VOOFF = 0, DMASIZE = 2, SO_VIOFF = 0, EOP
 * = 1
 *
 * ifstream0 {
 *
 *  # Write Pos.XY
 *  STM SO=0, SRC3=c0, SRC2=c3, SRC1=pt3, SRC0=pt0
 *  STM SO=0, SRC3=c0, SRC2=c4, SRC1=pt3, SRC0=pt0
 *  #Write Pos.ZW to buffer 1 and advance
 *  STM SO=0, SRC3=c2, SRC2=c5, SRC1=pt5, SRC0=pt2
 *
 * }
 *
 * else if stream1 {
 *
 *  #Write Add to buffer 1 and advance
 *  STM S0=1, SRC3=c1, SRC2=c6, SRC1=pt4, SRC0=pt1
 *
 * }
 */
#define PVR_ROGUE_PDSINST_STM_OPCODE_SHIFT (29U)
#define PVR_ROGUE_PDSINST_STM_CCS_CCS_GLOBAL_SHIFT (28U)
#define PVR_ROGUE_PDSINST_STM_CCS_CCS_SO_SHIFT (27U)
#define PVR_ROGUE_PDSINST_STM_CCS_CCS_CC_SHIFT (26U)
#define PVR_ROGUE_PDSINST_STM_SO_TST_SHIFT (25U)
#define PVR_ROGUE_PDSINST_STM_SO_SHIFT (23U)
#define PVR_ROGUE_PDSINST_STM_SO_SRC0_SHIFT (18U)
#define PVR_ROGUE_PDSINST_STM_SO_SRC1_SHIFT (13U)
#define PVR_ROGUE_PDSINST_STM_SO_SRC2_SHIFT (5U)
#define PVR_ROGUE_PDSINST_STM_SO_SRC3_SHIFT (0U)

/* Multiple Accumulate */
#define PVR_ROGUE_PDSINST_OPCODEA_MAD UINT32_C(0x00000000)

/* ALU Mode */

/* ALU will perform unsigned math.*/
#define PVR_ROGUE_PDSINST_ALUM_UNSIGNED (0x00000000U)

/* 64-bit Registers - 64-bit aligned */
#define PVR_ROGUE_PDSINST_REGS64_MASK (0x0000007FU)
#define PVR_ROGUE_PDSINST_REGS64_CONST64 (0U)
#define PVR_ROGUE_PDSINST_REGS64_CONST64_LOWER (0U)
#define PVR_ROGUE_PDSINST_REGS64_CONST64_UPPER (63U)
#define PVR_ROGUE_PDSINST_REGS64_TEMP64 (1U)
#define PVR_ROGUE_PDSINST_REGS64_TEMP64_LOWER (64U)
#define PVR_ROGUE_PDSINST_REGS64_TEMP64_UPPER (79U)
#define PVR_ROGUE_PDSINST_REGS64_PTEMP64 (2U)
#define PVR_ROGUE_PDSINST_REGS64_PTEMP64_LOWER (96U)
#define PVR_ROGUE_PDSINST_REGS64_PTEMP64_UPPER (111U)

/* 64-bit Temps 0-15 Destination */
#define PVR_ROGUE_PDSINST_REGS64T_MASK (0x0000000FU)
#define PVR_ROGUE_PDSINST_REGS64T_TEMP64 (0U)
#define PVR_ROGUE_PDSINST_REGS64T_TEMP64_LOWER (0U)
#define PVR_ROGUE_PDSINST_REGS64T_TEMP64_UPPER (15U)

/* cc ? dst = (src0 * src1) + (src2 * -1sna) + cin
 *
 * Multiply 2 source 32 bit numbers to generate a 64 bit result, then add or
 * subtract a third source. Conditionally takes in a carry in. Always generates
 * a carry out which is held in the status register.
 */
#define PVR_ROGUE_PDSINST_MAD_OPCODE_SHIFT (30U)
#define PVR_ROGUE_PDSINST_MAD_SNA_SHIFT (29U)
#define PVR_ROGUE_PDSINST_MAD_SNA_ADD (0x00000000U)
#define PVR_ROGUE_PDSINST_MAD_SNA_SUB (0x20000000U)
#define PVR_ROGUE_PDSINST_MAD_ALUM_SHIFT (28U)
#define PVR_ROGUE_PDSINST_MAD_ALUM_SIGNED (0x10000000U)
#define PVR_ROGUE_PDSINST_MAD_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_MAD_CC_ENABLE (0x08000000U)
/* 32-bit source to multiply - 32-bit range. */
#define PVR_ROGUE_PDSINST_MAD_SRC0_SHIFT (19U)
/* 32-bit source to multiply - 32-bit range */
#define PVR_ROGUE_PDSINST_MAD_SRC1_SHIFT (11U)
/* 64-bit source to add - 64-bit range */
#define PVR_ROGUE_PDSINST_MAD_SRC2_SHIFT (4U)
#define PVR_ROGUE_PDSINST_MAD_DST_SHIFT (0U)

/* cc ? dst = src0 + (src1 * -1sna) + cin
 *
 * Add or subtract 2 64 bit numbers. Conditionally takes in a carry in. Always
 * generates a carry out which is held in the status register.
 */
#define PVR_ROGUE_PDSINST_ADD64_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_ADD64_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_ADD64_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_ADD64_ALUM_SHIFT (26U)
#define PVR_ROGUE_PDSINST_ADD64_ALUM_SIGNED (0x04000000U)
#define PVR_ROGUE_PDSINST_ADD64_SNA_SHIFT (24U)
#define PVR_ROGUE_PDSINST_ADD64_SNA_SUB (0x01000000U)

/* 64-bit source to add. */
#define PVR_ROGUE_PDSINST_ADD64_SRC0_SHIFT (12U)

/* 64-bit source to add */
#define PVR_ROGUE_PDSINST_ADD64_SRC1_SHIFT (5U)

/* 64-bit temp or persistent temp */
#define PVR_ROGUE_PDSINST_ADD64_DST_SHIFT (0U)
/* cc ? dst = src0 + (src1 * -1sna) + cin
 *
 * Add or subtract 2 32 bit numbers. Conditionally takes in a carry in. Always
 * generates a carry out which is held in the status register.
 */
#define PVR_ROGUE_PDSINST_ADD32_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_ADD32_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_ADD32_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_ADD32_ALUM_SHIFT (26U)
#define PVR_ROGUE_PDSINST_ADD32_ALUM_SIGNED (0x04000000U)
#define PVR_ROGUE_PDSINST_ADD32_SNA_SHIFT (24U)
#define PVR_ROGUE_PDSINST_ADD32_SNA_SUB (0x01000000U)
/* 32-bit source to add */
#define PVR_ROGUE_PDSINST_ADD32_SRC0_SHIFT (14U)
#define PVR_ROGUE_PDSINST_ADD32_SRC0_CLRMSK (0xFFC03FFFU)
/* 32-bit source to add */
#define PVR_ROGUE_PDSINST_ADD32_SRC1_SHIFT (6U)
#define PVR_ROGUE_PDSINST_ADD32_SRC1_CLRMSK (0xFFFFC03FU)
/* 32-bit temp or persistent temp */
#define PVR_ROGUE_PDSINST_ADD32_DST_SHIFT (0U)
#define PVR_ROGUE_PDSINST_ADD32_DST_CLRMSK (0xFFFFFFC0U)

/* Comparison Operation */
#define PVR_ROGUE_PDSINST_COP_MASK (0x00000003U)

/* = */
#define PVR_ROGUE_PDSINST_COP_EQ (0x00000000U)

/* > */
#define PVR_ROGUE_PDSINST_COP_GT (0x00000001U)

/* < */
#define PVR_ROGUE_PDSINST_COP_LT (0x00000002U)

/* != */
#define PVR_ROGUE_PDSINST_COP_NE (0x00000003U)

/* Compare Instruction with 2 sources (IM=0)
 *
 * im = 0;
 * cc ? dst = src0 op src1
 *
 * Test source 0 against source 1. The result is written to the destination
 * predicate (P0). All arguments are treated as unsigned.
 */
#define PVR_ROGUE_PDSINST_CMP_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_CMP_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_CMP_OPCODE_DEFAULT (0xB0000000U) /* CMP */
#define PVR_ROGUE_PDSINST_CMP_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_CMP_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_CMP_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_CMP_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_CMP_COP_SHIFT (25U)
#define PVR_ROGUE_PDSINST_CMP_COP_CLRMSK (0xF9FFFFFFU)
#define PVR_ROGUE_PDSINST_CMP_COP_EQ (0x00000000U)
#define PVR_ROGUE_PDSINST_CMP_COP_GT (0x02000000U)
#define PVR_ROGUE_PDSINST_CMP_COP_LT (0x04000000U)
#define PVR_ROGUE_PDSINST_CMP_COP_NE (0x06000000U)
#define PVR_ROGUE_PDSINST_CMP_SETCP_SHIFT (24U)
#define PVR_ROGUE_PDSINST_CMP_SETCP_CLRMSK (0xFEFFFFFFU)
#define PVR_ROGUE_PDSINST_CMP_SETCP_EN (0x01000000U)
#define PVR_ROGUE_PDSINST_CMP_IM_SHIFT (23U)
#define PVR_ROGUE_PDSINST_CMP_IM_CLRMSK (0xFF7FFFFFU)
#define PVR_ROGUE_PDSINST_CMP_IM_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_CMP_IM_ENABLE (0x00800000U)
#define PVR_ROGUE_PDSINST_CMP_SRC0_SHIFT (18U)
#define PVR_ROGUE_PDSINST_CMP_SRC0_CLRMSK (0xFF83FFFFU)
#define PVR_ROGUE_PDSINST_CMP_SRC1_SHIFT (2U)
#define PVR_ROGUE_PDSINST_CMP_SRC1_CLRMSK (0xFFFFFE03U)

/* 16-bit signed immediate. */
#define PVR_ROGUE_PDSINST_IMM16_MASK (0x0000FFFFU)

/* Compare Instruction with Immediate (IM=1)
 *
 * im = 1;
 * cc ? dst = src0 op imm16
 *
 * Test source 0 against an immediate. The result is written to the destination
 * predicate (P0). All arguments are treated as unsigned.
 */
#define PVR_ROGUE_PDSINST_CMPI_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_CMPI_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_CMPI_OPCODE_DEFAULT (0xB0000000U) /* CMP */
#define PVR_ROGUE_PDSINST_CMPI_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_CMPI_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_CMPI_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_CMPI_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_CMPI_COP_SHIFT (25U)
#define PVR_ROGUE_PDSINST_CMPI_COP_CLRMSK (0xF9FFFFFFU)
#define PVR_ROGUE_PDSINST_CMPI_COP_EQ (0x00000000U)
#define PVR_ROGUE_PDSINST_CMPI_COP_GT (0x02000000U)
#define PVR_ROGUE_PDSINST_CMPI_COP_LT (0x04000000U)
#define PVR_ROGUE_PDSINST_CMPI_COP_NE (0x06000000U)
#define PVR_ROGUE_PDSINST_CMPI_SETCP_SHIFT (24U)
#define PVR_ROGUE_PDSINST_CMPI_SETCP_CLRMSK (0xFEFFFFFFU)
#define PVR_ROGUE_PDSINST_CMPI_SETCP_EN (0x01000000U)
#define PVR_ROGUE_PDSINST_CMPI_IM_SHIFT (23U)
#define PVR_ROGUE_PDSINST_CMPI_IM_CLRMSK (0xFF7FFFFFU)
#define PVR_ROGUE_PDSINST_CMPI_IM_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_CMPI_IM_ENABLE (0x00800000U)
#define PVR_ROGUE_PDSINST_CMPI_SRC0_SHIFT (18U)
#define PVR_ROGUE_PDSINST_CMPI_SRC0_CLRMSK (0xFF83FFFFU)
#define PVR_ROGUE_PDSINST_CMPI_IM16_SHIFT (2U)
#define PVR_ROGUE_PDSINST_CMPI_IM16_CLRMSK (0xFFFC0003U)

/* Condition codes */
#define PVR_ROGUE_PDSINST_PREDICATE_MASK (0x0000000FU)

/* Use programmable predicate 0 */
#define PVR_ROGUE_PDSINST_PREDICATE_P0 (0x00000000U)
/* Input Predicate 0 - When DM Pixel Start/End Program End of Tile, When DM
 * Pixel State Program indicates load Uniforms, When DM Vertex Last Vertex In
 * Task, When DM Compute indicates shared or kernel task (compute thread barrier
 * mode) or Last In Task (normal mode), When DM Tessellator TBD.
 */
#define PVR_ROGUE_PDSINST_PREDICATE_IF0 (0x00000001U)
/* Input Predicate 1 - When DM Pixel Start/End Program End Render, When DM Pixel
 * State Program indicates load Texture, When DM vertex First In Task, When DM
 * Compute indicates synchronization task (compute thread barrier mode) or First
 * In Task (normal mode), When DM Tessellator TBD.
 */
#define PVR_ROGUE_PDSINST_PREDICATE_IF1 (0x00000002U)
/* Stream 0 Out has overflowed. Note this is per stream not per buffer. */
#define PVR_ROGUE_PDSINST_PREDICATE_SO_OVERFLOW_PREDICATE_0 (0x00000003U)
/* Stream 1 Out has overflowed. Note this is per stream not per buffer. */
#define PVR_ROGUE_PDSINST_PREDICATE_SO_OVERFLOW_PREDICATE_1 (0x00000004U)
/* Stream 2 Out has overflowed. Note this is per stream not per buffer. */
#define PVR_ROGUE_PDSINST_PREDICATE_SO_OVERFLOW_PREDICATE_2 (0x00000005U)
/* Stream 3 Out has overflowed. Note this is per stream not per buffer. */
#define PVR_ROGUE_PDSINST_PREDICATE_SO_OVERFLOW_PREDICATE_3 (0x00000006U)
/* A Stream Out has overflowed. Note this is per stream not per buffer. */
#define PVR_ROGUE_PDSINST_PREDICATE_SO_OVERFLOW_PREDICATE_GLOBAL (0x00000007U)
/* For SETC Don't set a new predicate, KEEP the existing one. For BRA
 * instruction where this is the source predicate, KEEP the instruction, don't
 * predicate it out.
 */
#define PVR_ROGUE_PDSINST_PREDICATE_KEEP (0x00000008U)
/* DMA Out of Bounds predicate - set by DDMAT instruction when DMA is out of
 * bounds.
 */
#define PVR_ROGUE_PDSINST_PREDICATE_OOB (0x00000009U)

/* Negate condition. */

/* Do not negate condition. */
#define PVR_ROGUE_PDSINST_NEG_DISABLE (0x00000000U)
/* Negate condition. */
#define PVR_ROGUE_PDSINST_NEG_ENABLE (0x00000001U)

/* Branch Address. */
#define PVR_ROGUE_PDSINST_BRAADDR_MASK (0x0007FFFFU)

/* Branch and Set Selected Predicate Instruction
 *
 * im = 1;
 * cc xor neg ? pc = dst;
 *
 * Conditionally branch to an address (ADDR), depending upon the predicate. The
 * meaning of the predicate can be negated using NEG. This instruction also
 * allows the current predicate referenced by other instructions to be set by
 * the SETC field. The current predicate is available by all instructions. This
 * is a signed offset from the current PC. BRA ADDR=0 would be an infinite loop
 * of the instruction.
 */

#define PVR_ROGUE_PDSINST_BRA_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_BRA_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_BRA_OPCODE_DEFAULT (0xC0000000U) /* BRA */
#define PVR_ROGUE_PDSINST_BRA_SRCC_SHIFT (24U)
#define PVR_ROGUE_PDSINST_BRA_SRCC_CLRMSK (0xF0FFFFFFU)
#define PVR_ROGUE_PDSINST_BRA_SRCC_P0 (0x00000000U)
#define PVR_ROGUE_PDSINST_BRA_SRCC_IF0 (0x01000000U)
#define PVR_ROGUE_PDSINST_BRA_SRCC_IF1 (0x02000000U)
#define PVR_ROGUE_PDSINST_BRA_SRCC_SO_OVERFLOW_PREDICATE_0 (0x03000000U)
#define PVR_ROGUE_PDSINST_BRA_SRCC_SO_OVERFLOW_PREDICATE_1 (0x04000000U)
#define PVR_ROGUE_PDSINST_BRA_SRCC_SO_OVERFLOW_PREDICATE_2 (0x05000000U)
#define PVR_ROGUE_PDSINST_BRA_SRCC_SO_OVERFLOW_PREDICATE_3 (0x06000000U)
#define PVR_ROGUE_PDSINST_BRA_SRCC_SO_OVERFLOW_PREDICATE_GLOBAL (0x07000000U)
#define PVR_ROGUE_PDSINST_BRA_SRCC_KEEP (0x08000000U)
#define PVR_ROGUE_PDSINST_BRA_SRCC_OOB (0x09000000U)
#define PVR_ROGUE_PDSINST_BRA_NEG_SHIFT (23U)
#define PVR_ROGUE_PDSINST_BRA_NEG_CLRMSK (0xFF7FFFFFU)
#define PVR_ROGUE_PDSINST_BRA_NEG_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_BRA_NEG_ENABLE (0x00800000U)
#define PVR_ROGUE_PDSINST_BRA_SETC_SHIFT (19U)
#define PVR_ROGUE_PDSINST_BRA_SETC_CLRMSK (0xFF87FFFFU)
#define PVR_ROGUE_PDSINST_BRA_SETC_P0 (0x00000000U)
#define PVR_ROGUE_PDSINST_BRA_SETC_IF0 (0x00080000U)
#define PVR_ROGUE_PDSINST_BRA_SETC_IF1 (0x00100000U)
#define PVR_ROGUE_PDSINST_BRA_SETC_SO_OVERFLOW_PREDICATE_0 (0x00180000U)
#define PVR_ROGUE_PDSINST_BRA_SETC_SO_OVERFLOW_PREDICATE_1 (0x00200000U)
#define PVR_ROGUE_PDSINST_BRA_SETC_SO_OVERFLOW_PREDICATE_2 (0x00280000U)
#define PVR_ROGUE_PDSINST_BRA_SETC_SO_OVERFLOW_PREDICATE_3 (0x00300000U)
#define PVR_ROGUE_PDSINST_BRA_SETC_SO_OVERFLOW_PREDICATE_GLOBAL (0x00380000U)
#define PVR_ROGUE_PDSINST_BRA_SETC_KEEP (0x00400000U)
#define PVR_ROGUE_PDSINST_BRA_SETC_OOB (0x00480000U)
#define PVR_ROGUE_PDSINST_BRA_ADDR_SHIFT (0U)
#define PVR_ROGUE_PDSINST_BRA_ADDR_CLRMSK (0xFFF80000U)

/* SLC_MODE_LD   SLC Cache Policy for loads. */
#define PVR_ROGUE_PDSINST_SLC_MODE_LD_MASK (0x00000003U)
/* Bypass Policy */
#define PVR_ROGUE_PDSINST_SLC_MODE_LD_BYPASS (0x00000000U)
/* Standard Cached Read */
#define PVR_ROGUE_PDSINST_SLC_MODE_LD_CACHED (0x00000001U)
/* Cached Read no allocate */
#define PVR_ROGUE_PDSINST_SLC_MODE_LD_CACHED_RD_NA (0x00000003U)

/* CMODE_LD   MCU (SLC) Cache Mode for Loads. */
#define PVR_ROGUE_PDSINST_CMODE_LD_MASK (0x00000003U)

/* Normal cache operation. */
#define PVR_ROGUE_PDSINST_CMODE_LD_CACHED (0x00000000U)

/* Bypass L0 and L1. */
#define PVR_ROGUE_PDSINST_CMODE_LD_BYPASS (0x00000001U)

/* Force line fill of L0 and L1. */
#define PVR_ROGUE_PDSINST_CMODE_LD_FORCE_LINE_FILL (0x00000002U)

/* ld: Number of 64 bit words to load. */
#define PVR_ROGUE_PDSINST_LD_COUNT8_MASK (0x00000007U)

/* Source Base Address for memory fetch in DWORDS - MUST BE 128 BIT ALIGNED. */
#define PVR_ROGUE_PDSINST_LD_SRCADD_MASK (UINT64_C(0x0000003FFFFFFFFF))

/* Load Instruction DMA : Src0 */

/* SLC cache policy. */
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_SLCMODE_SHIFT (62U)
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_SLCMODE_CLRMSK \
   (UINT64_C(0x3FFFFFFFFFFFFFFF))
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_SLCMODE_BYPASS \
   (UINT64_C(0x0000000000000000))
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_SLCMODE_CACHED \
   (UINT64_C(0x4000000000000000))
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_SLCMODE_CACHED_RD_NA \
   (UINT64_C(0xc000000000000000))

/* The destination address in the temps (persistent or not) for the read data -
 * MUST BE 128 BIT ALIGNED.
 */
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_DEST_SHIFT (47U)
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_DEST_CLRMSK (UINT64_C(0xFFF07FFFFFFFFFFF))

/* Cache Mode */
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_CMODE_SHIFT (44U)
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_CMODE_CLRMSK (UINT64_C(0xFFFFCFFFFFFFFFFF))
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_CMODE_CACHED (UINT64_C(0x0000000000000000))
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_CMODE_BYPASS (UINT64_C(0x0000100000000000))
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_CMODE_FORCE_LINE_FILL \
   (UINT64_C(0x0000200000000000))

/* ld: Number of 64 bit words to load. */
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_COUNT8_SHIFT (41U)
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_COUNT8_CLRMSK \
   (UINT64_C(0xFFFFF1FFFFFFFFFF))

/* Source Base Address for memory fetch - MUST BE 128 BIT ALIGNED. */
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_SRCADD_SHIFT (2U)
#define PVR_ROGUE_PDSINST_LD_LD_SRC0_SRCADD_CLRMSK \
   (UINT64_C(0xFFFFFF0000000003))

/* Special Instructions Op-code. */
#define PVR_ROGUE_PDSINST_OPCODESP_MASK (0x0000000FU)

/* Data Load from memory. */
#define PVR_ROGUE_PDSINST_OPCODESP_LD UINT32_C(0x00000000)

/* Data Store to memory. */
#define PVR_ROGUE_PDSINST_OPCODESP_ST UINT32_C(0x00000001)

/* Wait read or write data operations to complete. */
#define PVR_ROGUE_PDSINST_OPCODESP_WDF UINT32_C(0x00000002)

/* Load 16 bit immediate. */
#define PVR_ROGUE_PDSINST_OPCODESP_LIMM UINT32_C(0x00000003)

/* Lock the execute so only this instance can execute for this data master. */
#define PVR_ROGUE_PDSINST_OPCODESP_LOCK UINT32_C(0x00000004)

/* Release the lock taken by lock. */
#define PVR_ROGUE_PDSINST_OPCODESP_RELEASE UINT32_C(0x00000005)

/* Halt execution (program termination). */
#define PVR_ROGUE_PDSINST_OPCODESP_HALT UINT32_C(0x00000006)

/* Clear stream out predicate. */
#define PVR_ROGUE_PDSINST_OPCODESP_STMC UINT32_C(0x00000007)

/* Parallel Stream Out. */
#define PVR_ROGUE_PDSINST_OPCODESP_STMP UINT32_C(0x00000008)

/* Integer Divide. */
#define PVR_ROGUE_PDSINST_OPCODESP_IDIV UINT32_C(0x00000009)

/* Atomic Access. */
#define PVR_ROGUE_PDSINST_OPCODESP_AA UINT32_C(0x0000000a)

/* Issue Data Fence. */
#define PVR_ROGUE_PDSINST_OPCODESP_IDF UINT32_C(0x0000000b)

/* Issue Data Fence. */
#define PVR_ROGUE_PDSINST_OPCODESP_POL (0x0000000cU)

/*No Operation. */
#define PVR_ROGUE_PDSINST_OPCODESP_NOP (0x0000000fU)

/* Data Load Instruction (Opcode SP)
 *
 * for (i=0; i < count;i++) {
 * cc ? *(src0 + i) = mem(src1 + i)
 * }
 *
 * Load count 32 bit words from memory to the temporaries reading from the
 * address in memory pointed to by SRCADD. If the final destination address
 * (DEST + COUNT - 1) exceeds the amount of temps available the entire load is
 * discarded.
 */
#define PVR_ROGUE_PDSINST_LD_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_LD_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_LD_OPCODE_DEFAULT (0xD0000000U) /* SP */
#define PVR_ROGUE_PDSINST_LD_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_LD_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_LD_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_LD_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_LD_OP_SHIFT (23U)
#define PVR_ROGUE_PDSINST_LD_OP_CLRMSK (0xF87FFFFFU)
#define PVR_ROGUE_PDSINST_LD_OP_DEFAULT (0x00000000U) /* ld */
#define PVR_ROGUE_PDSINST_LD_SRC0_SHIFT (0U)
#define PVR_ROGUE_PDSINST_LD_SRC0_CLRMSK (0xFFFFFF80U)

/* CMODE_ST   MCU (SLC) Cache Mode for stores. */
#define PVR_ROGUE_PDSINST_CMODE_ST_MASK (0x00000003U)

/* Write-through Policy */
#define PVR_ROGUE_PDSINST_CMODE_ST_WRITE_THROUGH (0x00000000U)

/* Write-back Policy. */
#define PVR_ROGUE_PDSINST_CMODE_ST_WRITE_BACK (0x00000001U)

/* Lazy write-back policy. */
#define PVR_ROGUE_PDSINST_CMODE_ST_LAZY_WRITE_BACK (0x00000002U)

/* ST: Number of 32 bit Words to store. */
#define PVR_ROGUE_PDSINST_ST_COUNT4_MASK (0x0000000FU)

/* Source Base Address for memory fetch in DWORDS. */
#define PVR_ROGUE_PDSINST_ST_SRCADD_MASK (UINT64_C(0x0000003FFFFFFFFF))

/* Store Instruction DMA : Src0 */

/* SLC cache policy. */
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_SLCMODE_SHIFT (62U)
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_SLCMODE_CLRMSK \
   (UINT64_C(0x3FFFFFFFFFFFFFFF))
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_SLCMODE_WRITE_THROUGH \
   (UINT64_C(0x0000000000000000))
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_SLCMODE_WRITE_BACK \
   (UINT64_C(0x4000000000000000))

#define PVR_ROGUE_PDSINST_ST_ST_SRC0_SRC_SHIFT (46U)
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_SRC_CLRMSK (UINT64_C(0xFFF03FFFFFFFFFFF))

/* Cache Mode. */
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_CMODE_SHIFT (44U)
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_CMODE_CLRMSK (UINT64_C(0xFFFFCFFFFFFFFFFF))
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_CMODE_WRITE_THROUGH \
   (UINT64_C(0x0000000000000000))
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_CMODE_WRITE_BACK \
   (UINT64_C(0x0000100000000000))
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_CMODE_LAZY_WRITE_BACK \
   (UINT64_C(0x0000200000000000))

/* ST: Number of 32 bit Words to store. */
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_COUNT4_SHIFT (40U)
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_COUNT4_CLRMSK \
   (UINT64_C(0xFFFFF0FFFFFFFFFF))

/* Destination Base Address for memory write. */
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_DSTADD_SHIFT (2U)
#define PVR_ROGUE_PDSINST_ST_ST_SRC0_DSTADD_CLRMSK \
   (UINT64_C(0xFFFFFF0000000003))

/* Data Store Instruction (Opcode SP)
 *
 * for (i=0; i < count;i++) {
 * cc ? mem(src1 + i) = *(src0 + i)
 * }
 *
 * Store count 64 bit words from temporaries to memory (memory address starts at
 * src1). If the instruction attempts to read data (in temps) outside of it's
 * allocated region the entire store is discarded.
 */
#define PVR_ROGUE_PDSINST_ST_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_ST_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_ST_OPCODE_DEFAULT (0xD0000000U) /* SP */
#define PVR_ROGUE_PDSINST_ST_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_ST_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_ST_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_ST_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_ST_OP_SHIFT (23U)
#define PVR_ROGUE_PDSINST_ST_OP_CLRMSK (0xF87FFFFFU)
#define PVR_ROGUE_PDSINST_ST_OP_DEFAULT (0x00800000U) /* ST */
#define PVR_ROGUE_PDSINST_ST_SRC0_SHIFT (0U)
#define PVR_ROGUE_PDSINST_ST_SRC0_CLRMSK (0xFFFFFF80U)

/* Data Fence Instruction (Opcode SP)
 *
 * Cc ? wdf
 *
 * The data fence instruction gives the ability to track the return of dependent
 * read data and to determine when data written from the core has made it to the
 * MCU. This is required on reads as there is no implicit synchronization
 * between read accesses to the primary attribute bank and data returned by
 * dependent reads. For writes it is required where the program is enforcing
 * synchronization with another program (which could be on the PDS or any other
 * processor in the system). Note, this only guarantees order within the
 * PDS. For order elsewhere reads need to be issued, and flush commands may have
 * to be issued to the MCU
 *
 * The fence mechanism takes the form of a counter that is incremented whenever
 * a read (ld) or write (ST) instruction is encountered by the instruction fetch
 * decoder. When the read or write instruction returns, or writes all its data
 * the counter is decremented. There is 1 counter per thread. Prior to accessing
 * return data a WDF instruction must be issued, when this is seen by the
 * instruction decoder it will check the current count value and will suspend
 * execution if it is currently non zero, execution being resumed as soon as the
 * counter reaches zero, and a slot is available.
 *
 * Example
 *  Do a dependent read for data
 *
 * ldr0,#2,r3        Issue read
 * ...               Try and do some other stuff
 * wdf               Make sure read data has come back
 * add32 r2,r1,r0    And use the returned result
 *
 */
#define PVR_ROGUE_PDSINST_WDF_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_WDF_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_WDF_OPCODE_DEFAULT (0xD0000000U) /* SP */
#define PVR_ROGUE_PDSINST_WDF_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_WDF_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_WDF_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_WDF_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_WDF_OP_SHIFT (23U)
#define PVR_ROGUE_PDSINST_WDF_OP_CLRMSK (0xF87FFFFFU)
#define PVR_ROGUE_PDSINST_WDF_OP_DEFAULT (0x01000000U) /* WDF */

/* PDS Global Register access control */

/* Disable global register access */
#define PVR_ROGUE_PDSINST_GR_DISABLE (0x00000000U)

/* Enable global register access, global register specified by IMM16.*/
#define PVR_ROGUE_PDSINST_GR_ENABLE (0x00000001U)

/* Load Immediate (Opcode SP)
 *
 * cc ? GR = DISABLE : *src1 = src0
 * cc ? GR = ENABLE  : *src1 = greg[IMM16]
 *
 * Load an immediate value (src0) into the temporary registers. If the GR flag
 * is set, the PDS global register specified by IMM16 will be loaded instead.
 * greg[0] = cluster number greg[1] = instance number
 *
 */
#define PVR_ROGUE_PDSINST_LIMM_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_LIMM_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_LIMM_OPCODE_DEFAULT (0xD0000000U) /* SP */
#define PVR_ROGUE_PDSINST_LIMM_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_LIMM_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_LIMM_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_LIMM_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_LIMM_OP_SHIFT (23U)
#define PVR_ROGUE_PDSINST_LIMM_OP_CLRMSK (0xF87FFFFFU)
#define PVR_ROGUE_PDSINST_LIMM_OP_DEFAULT (0x01800000U) /* LIMM */
#define PVR_ROGUE_PDSINST_LIMM_SRC1_SHIFT (18U)
#define PVR_ROGUE_PDSINST_LIMM_SRC1_CLRMSK (0xFF83FFFFU)
#define PVR_ROGUE_PDSINST_LIMM_SRC0_SHIFT (2U)
#define PVR_ROGUE_PDSINST_LIMM_SRC0_CLRMSK (0xFFFC0003U)
#define PVR_ROGUE_PDSINST_LIMM_GR_SHIFT (1U)
#define PVR_ROGUE_PDSINST_LIMM_GR_CLRMSK (0xFFFFFFFDU)
#define PVR_ROGUE_PDSINST_LIMM_GR_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_LIMM_GR_ENABLE (0x00000002U)

/* Lock Instruction (Opcode SP)
 *
 * cc ? lock
 *
 * The hardware contains an internal mutex per data master. When the lock
 * instruction is issued, the thread will attempt to take control of the mutex
 * (for the current data master). If it is already taken by another thread, then
 * the thread is descheduled until it is available.
 *
 * The purpose of the lock (and release) instructions is to allow critical
 * sections of code to execute serially to other code for the same data
 * master. This is particularly useful when accessing the persistent (cross
 * thread) temporaries. Note that there is no communication possible across data
 * masters.
 *
 * It is illegal to place a DOUT instruction inside a LOCK, RELEASE section of
 * code.
 */
#define PVR_ROGUE_PDSINST_LOCK_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_LOCK_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_LOCK_OPCODE_DEFAULT (0xD0000000U) /* SP */
#define PVR_ROGUE_PDSINST_LOCK_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_LOCK_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_LOCK_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_LOCK_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_LOCK_OP_SHIFT (23U)
#define PVR_ROGUE_PDSINST_LOCK_OP_CLRMSK (0xF87FFFFFU)
#define PVR_ROGUE_PDSINST_LOCK_OP_DEFAULT (0x02000000U) /* LOCK */

/* Release Lock (Opcode SP)
 *
 * cc ? release
 *
 * The hardware contains an internal mutex per data master. If a thread has
 * issued a lock instruction, then a release instruction must be issued to
 * release the lock. See the corresponding lock instruction for more details
 *
 * It is illegal to place a DOUT instruction inside a LOCK, RELEASE section of
 * code.
 */
#define PVR_ROGUE_PDSINST_RELEASE_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_RELEASE_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_RELEASE_OPCODE_DEFAULT (0xD0000000U) /* SP */
#define PVR_ROGUE_PDSINST_RELEASE_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_RELEASE_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_RELEASE_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_RELEASE_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_RELEASE_OP_SHIFT (23U)
#define PVR_ROGUE_PDSINST_RELEASE_OP_CLRMSK (0xF87FFFFFU)
#define PVR_ROGUE_PDSINST_RELEASE_OP_DEFAULT (0x02800000U) /* RELEASE */

/* Special instruction - Halt
 * Halt Execution (Opcode SP)
 *
 * cc ? halt
 *
 * The last instruction in a program must always be a halt instruction, or a
 * DOUT/DDMAD instruction with the END flag set. This is required in order to
 * indicate the end of the program.
 */
#define PVR_ROGUE_PDSINST_HALT_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_HALT_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_HALT_OPCODE_DEFAULT (0xD0000000U) /* SP */
#define PVR_ROGUE_PDSINST_HALT_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_HALT_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_HALT_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_HALT_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_HALT_OP_SHIFT (23U)
#define PVR_ROGUE_PDSINST_HALT_OP_CLRMSK (0xF87FFFFFU)
#define PVR_ROGUE_PDSINST_HALT_OP_DEFAULT (0x03000000U) /* HALT */

/* Special instruction - Nop
 * No Operation (Opcode SP)
 *
 * cc ? NOP
 *
 * This instruction does no operation, and introduces a wait cycle into the
 * pipeline.
 *
 */
#define PVR_ROGUE_PDSINST_NOP_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_NOP_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_NOP_OPCODE_DEFAULT (0xD0000000U) /* SP */
#define PVR_ROGUE_PDSINST_NOP_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_NOP_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_NOP_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_NOP_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_NOP_OP_SHIFT (23U)
#define PVR_ROGUE_PDSINST_NOP_OP_CLRMSK (0xF87FFFFFU)
#define PVR_ROGUE_PDSINST_NOP_OP_DEFAULT (0x07800000U) /* NOP */

/* The SO bits to clear 0-3 streams 0-3, bit 4-global */
#define PVR_ROGUE_PDSINST_SOMASK_MASK (0x0000001FU)

/* Special instruction - Stream out predicate clear
 *  (Opcode SP)
 *
 * cc ? NOP
 *
 * This instruction clears the stream out predicates to 0, according to the
 * clear bits.
 *
 */
#define PVR_ROGUE_PDSINST_STMC_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_STMC_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_STMC_OPCODE_DEFAULT (0xD0000000U) /* SP */
#define PVR_ROGUE_PDSINST_STMC_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_STMC_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_STMC_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_STMC_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_STMC_OP_SHIFT (23U)
#define PVR_ROGUE_PDSINST_STMC_OP_CLRMSK (0xF87FFFFFU)
#define PVR_ROGUE_PDSINST_STMC_OP_DEFAULT (0x03800000U) /* STMC */
#define PVR_ROGUE_PDSINST_STMC_SOMASK_SHIFT (0U)
#define PVR_ROGUE_PDSINST_STMC_SOMASK_CLRMSK (0xFFFFFFE0U)

/* A 1 TB address, with byte granularity. Address must be dword aligned when
 * repeat is 0.
 */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_ADDRESS_MASK \
   (UINT64_C(0x000000FFFFFFFFFF))

/* SLC cache policy */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRCADD_SLCMODE_SHIFT (62U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRCADD_SLCMODE_CLRMSK \
   (UINT64_C(0x3FFFFFFFFFFFFFFF))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRCADD_SLCMODE_BYPASS \
   (UINT64_C(0x0000000000000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRCADD_SLCMODE_CACHED \
   (UINT64_C(0x4000000000000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRCADD_SLCMODE_CACHED_RD_NA \
   (UINT64_C(0xc000000000000000))

#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRCADD_ADDRESS_SHIFT (0U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRCADD_ADDRESS_CLRMSK \
   (UINT64_C(0xFFFFFF0000000000))

/* Size of external memory buffer in bytes (0 is 0 bytes) */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_MSIZE_MASK (0x7FFFFFFFU)

/* When repeat is enabled the size of the DMA in bytes */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_WORDSIZE_MASK (0x00000003U)
/* DMA of 1 byte */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_WORDSIZE_ONE (0x00000000U)
/* DMA of 2 byte */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_WORDSIZE_TWO (0x00000001U)
/* DMA of 3 byte */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_WORDSIZE_THREE (0x00000002U)
/* DMA of 4 byte */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_WORDSIZE_FOUR (0x00000003U)

/* DMA to unified store */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_DEST_UNIFIED_STORE (0x00000000U)

/* DMA to common store */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_DEST_COMMON_STORE (0x00000001U)

/* Primary instance data offset in 32 bit words (offset into the current
 * instance).
 */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_AO_MASK (0x00001FFFU)

/* Only applies to unified store DMAs, must be clear for common store.
 *
 * DMA is issued natively, in its entirety.
 */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_REPEAT_NOREPEAT (0x00000000U)
/* BSIZE is the number of times the DMA is repeated. Word size is the size of
 * the DMA. The DMA is expanded into BSIZE DMAs.
 */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_REPEAT_REPEAT (0x00000001U)

/* Size of fetch in dwords (0 is 0 dwords). */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_BSIZE_MASK (0x00000FFFU)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_BSIZE_RANGE (0U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_BSIZE_LOWER (0U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_BSIZE_UPPER (255U)

/* Size of external buffer in bytes. */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_MSIZE_SHIFT (33U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_MSIZE_CLRMSK \
   (UINT64_C(0x00000001FFFFFFFF))

/* Perform OOB checking. */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_TEST_SHIFT (32U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_TEST_CLRMSK \
   (UINT64_C(0xFFFFFFFEFFFFFFFF))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_TEST_EN \
   (UINT64_C(0x0000000100000000))

/* Last DMA in program. */
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_LAST_SHIFT (31U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_LAST_CLRMSK \
   (UINT64_C(0xFFFFFFFF7FFFFFFF))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_LAST_EN \
   (UINT64_C(0x0000000080000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_WORDSIZE_SHIFT (29U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_WORDSIZE_CLRMSK \
   (UINT64_C(0xFFFFFFFF9FFFFFFF))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_WORDSIZE_ONE \
   (UINT64_C(0x0000000000000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_WORDSIZE_TWO \
   (UINT64_C(0x0000000020000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_WORDSIZE_THREE \
   (UINT64_C(0x0000000040000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_WORDSIZE_FOUR \
   (UINT64_C(0x0000000060000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_DEST_SHIFT (28U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_DEST_CLRMSK \
   (UINT64_C(0xFFFFFFFFEFFFFFFF))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_DEST_UNIFIED_STORE \
   (UINT64_C(0x0000000000000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_DEST_COMMON_STORE \
   (UINT64_C(0x0000000010000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_CMODE_SHIFT (26U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_CMODE_CLRMSK \
   (UINT64_C(0xFFFFFFFFF3FFFFFF))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_CMODE_CACHED \
   (UINT64_C(0x0000000000000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_CMODE_BYPASS \
   (UINT64_C(0x0000000004000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_CMODE_FORCE_LINE_FILL \
   (UINT64_C(0x0000000008000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_AO_SHIFT (13U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_AO_CLRMSK \
   (UINT64_C(0xFFFFFFFFFC001FFF))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_REPEAT_SHIFT (12U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_REPEAT_CLRMSK \
   (UINT64_C(0xFFFFFFFFFFFFEFFF))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_REPEAT_NOREPEAT \
   (UINT64_C(0x0000000000000000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_REPEAT_REPEAT \
   (UINT64_C(0x0000000000001000))
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_BSIZE_SHIFT (0U)
#define PVR_ROGUE_PDSINST_DDMAD_FIELDS_SRC3_BSIZE_CLRMSK \
   (UINT64_C(0xFFFFFFFFFFFFF000))

/* Stop execution flag
 *
 * Continue execution after this instruction.
 */
#define PVR_ROGUE_PDSINST_END_DISABLE (0x00000000U)

/* Halt execution after this instruction. */
#define PVR_ROGUE_PDSINST_END_ENABLE (0x00000001U)

/* 64-bit Consts 0-63 Destination. */
#define PVR_ROGUE_PDSINST_REGS64C_MASK (0x0000003FU)
#define PVR_ROGUE_PDSINST_REGS64C_CONST64 (0U)
#define PVR_ROGUE_PDSINST_REGS64C_CONST64_LOWER (0U)
#define PVR_ROGUE_PDSINST_REGS64C_CONST64_UPPER (63U)

/* Multiply-add then send to DOUTD (Opcode SP). Optionally perform out-of-bounds
 * checking (DDMAD(T)). Multiply-add then send to DOUTD (Opcode SP).
 *
 * cc ?  if ( test == 1 ) then
 * cc ?  if ( ((src0 * src1) + src2)[39:0] + (src3[11:0]<<2) <= src2[39:0] +
 * src3[63:33] ) then cc ?  OOB = 0 cc ?  doutd = (src0 * src1) + src2, src3 cc
 * ? else cc ?  OOB = 1 cc ?  endif cc ?  else cc ?  doutd = (src0 * src1) +
 * src2 src3 cc ?  endif
 *
 * cc ?  doutd = (src0 * src1) + src2, src3
 *
 * This instruction performs a 32 bit multiply, followed by a 64 bit add. This
 * result is combined with a 4th source and used to create the data for an DOUTD
 * emit. A DOUTD is a command to a DMA engine, which reads data from memory and
 * writes it into the USC Unified or Common Store.
 *
 * Additionally the DDMAD performs an out-of-bounds check on the DMA when the
 * test flag is set . If a buffer overflow is predicated, the DMA is skipped and
 * the OOB (DMA out of bounds) predicate is set.
 */
#define PVR_ROGUE_PDSINST_DDMAD_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_DDMAD_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_DDMAD_OPCODE_DEFAULT (0xE0000000U) /* DDMAD */
#define PVR_ROGUE_PDSINST_DDMAD_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_DDMAD_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_DDMAD_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_DDMAD_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_DDMAD_END_SHIFT (26U)
#define PVR_ROGUE_PDSINST_DDMAD_END_CLRMSK (0xFBFFFFFFU)
#define PVR_ROGUE_PDSINST_DDMAD_END_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_DDMAD_END_ENABLE (0x04000000U)

/* 32-bit source to multiply - 32-bit range. */
#define PVR_ROGUE_PDSINST_DDMAD_SRC0_SHIFT (18U)
#define PVR_ROGUE_PDSINST_DDMAD_SRC0_CLRMSK (0xFC03FFFFU)

/* 32-bit source to multiply - 32-bit range. */
#define PVR_ROGUE_PDSINST_DDMAD_SRC1_SHIFT (13U)
#define PVR_ROGUE_PDSINST_DDMAD_SRC1_CLRMSK (0xFFFC1FFFU)

/* 64-bit source to add - 64-bit range */
#define PVR_ROGUE_PDSINST_DDMAD_SRC2_SHIFT (6U)
#define PVR_ROGUE_PDSINST_DDMAD_SRC2_CLRMSK (0xFFFFE03FU)

/* 64-bit constant register destination */
#define PVR_ROGUE_PDSINST_DDMAD_SRC3_SHIFT (0U)
#define PVR_ROGUE_PDSINST_DDMAD_SRC3_CLRMSK (0xFFFFFFC0U)

/* When DOUTU_SAMPLE_RATE is INSTANCE or SELECTIVE - 32 bit temps per instance
 * at 4 word granularity. When DOUTU_SAMPLE_RATE is FULL - 32 bit temps per
 * sample at 4 word granularity.
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_TEMPS_MASK (0x0000003FU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_TEMPS_ALIGNSHIFT (2U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_TEMPS_ALIGNSIZE (4U)

/* Sample rate */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SAMPLE_RATE_MASK (0x00000003U)

/* Instance rate */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SAMPLE_RATE_INSTANCE (0x00000000U)

/* Selective sample rate */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SAMPLE_RATE_SELECTIVE (0x00000001U)

/* Full sample rate */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SAMPLE_RATE_FULL (0x00000002U)

/* Code base address (4 byte alignment). */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_EXE_OFF_MASK (0x3FFFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_EXE_OFF_ALIGNSHIFT (2U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_EXE_OFF_ALIGNSIZE (4U)

/* Use Interface doutu : Src0 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_DUAL_PHASE_SHIFT (41U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_DUAL_PHASE_CLRMSK \
   (UINT64_C(0xFFFFFDFFFFFFFFFF))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_DUAL_PHASE_EN \
   (UINT64_C(0x0000020000000000))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_TEMPS_SHIFT (35U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_TEMPS_CLRMSK \
   (UINT64_C(0xFFFFFE07FFFFFFFF))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_TEMPS_ALIGNSHIFT (2U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_TEMPS_ALIGNSIZE (4U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_SAMPLE_RATE_SHIFT (33U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_SAMPLE_RATE_CLRMSK \
   (UINT64_C(0xFFFFFFF9FFFFFFFF))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_SAMPLE_RATE_INSTANCE \
   (UINT64_C(0x0000000000000000))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_SAMPLE_RATE_SELECTIVE \
   (UINT64_C(0x0000000200000000))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_SAMPLE_RATE_FULL \
   (UINT64_C(0x0000000400000000))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_EXE_OFF_SHIFT (2U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_EXE_OFF_CLRMSK \
   (UINT64_C(0xFFFFFFFF00000003))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_EXE_OFF_ALIGNSHIFT (2U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTU_SRC0_EXE_OFF_ALIGNSIZE (4U)

/* Use Interface doutu : Src1. */

/* Secondary instance data offset in 32 bit words (offset of the instance). */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_DOFFSET_MASK (0x00001FFFU)

/* Source Base Address for memory fetch. Address must be dword aligned when
 * repeat is 0.
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SBASE_MASK \
   (UINT64_C(0x000000FFFFFFFFFF))

/* DMA Interface DOutD : Src0 */

/* SLC cache policy */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC0_SLCMODE_SHIFT (62U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC0_SLCMODE_CLRMSK \
   (UINT64_C(0x3FFFFFFFFFFFFFFF))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC0_SLCMODE_BYPASS \
   (UINT64_C(0x0000000000000000))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC0_SLCMODE_CACHED \
   (UINT64_C(0x4000000000000000))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC0_SLCMODE_CACHED_RD_NA \
   (UINT64_C(0xc000000000000000))

/* Secondary instance data offset in 32 bit words (offset of the instance). */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC0_DOFFSET_SHIFT (40U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC0_DOFFSET_CLRMSK \
   (UINT64_C(0xFFE000FFFFFFFFFF))

/* Source Base Address for memory fetch. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC0_SBASE_SHIFT (0U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC0_SBASE_CLRMSK \
   (UINT64_C(0xFFFFFF0000000000))

/* When repeat is enabled the size of the DMA in bytes. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_WORDSIZE_MASK (0x00000003U)

/* DMA of 1 byte */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_WORDSIZE_ONE (0x00000000U)

/* DMA of 2 byte */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_WORDSIZE_TWO (0x00000001U)

/* DMA of 3 byte */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_WORDSIZE_THREE (0x00000002U)

/* DMA of 4 byte */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_WORDSIZE_FOUR (0x00000003U)

/* Unified Store */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_DEST_UNIFIED_STORE (0x00000000U)

/* Common Store */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_DEST_COMMON_STORE (0x00000001U)

/* Primary instance data offset in 32 bit words (offset into the current
 * instance).
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_AO_MASK (0x00001FFFU)

/* Only applies to unified store DMAs, ignore for common store. */

/* DMA is issued natively, in its entirety. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_REPEAT_NOREPEAT (0x00000000U)

/* BSIZE is the number of times the DMA is repeated. Word size is the size of
 * the DMA. The DMA is expanded into BSIZE DMAs.
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_REPEAT_REPEAT (0x00000001U)

/* Size of fetch in dwords (0 means don't DMA, 1=1 etc.) */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_BSIZE_MASK (0x00000FFFU)

/* DMA Interface DOutD : Src1 */

/* Last Write or DMA in program (This needs to only be set once on with the last
 * DMA or last direct write, which ever is last).
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_LAST_SHIFT (31U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_LAST_CLRMSK (0x7FFFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_LAST_EN (0x80000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_WORDSIZE_SHIFT (29U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_WORDSIZE_CLRMSK (0x9FFFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_WORDSIZE_ONE (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_WORDSIZE_TWO (0x20000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_WORDSIZE_THREE (0x40000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_WORDSIZE_FOUR (0x60000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_DEST_SHIFT (28U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_DEST_CLRMSK (0xEFFFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_DEST_UNIFIED_STORE \
   (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_DEST_COMMON_STORE (0x10000000U)

/* CMODE   Cache Mode */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_CMODE_SHIFT (26U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_CMODE_CLRMSK (0xF3FFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_CMODE_CACHED (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_CMODE_BYPASS (0x04000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_CMODE_FORCE_LINE_FILL \
   (0x08000000U)

/* Primary instance data offset in 32 bit words (offset into the current
 * instance).
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_AO_SHIFT (13U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_AO_CLRMSK (0xFC001FFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_REPEAT_SHIFT (12U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_REPEAT_CLRMSK (0xFFFFEFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_REPEAT_NOREPEAT (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_REPEAT_REPEAT (0x00001000U)

/* Size of fetch in dwords (0 means don't DMA, 1=1 etc.) */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_BSIZE_SHIFT (0U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_BSIZE_CLRMSK (0xFFFFF000U)

/* Lower 64-bit (63:0) data to be written. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SBASE0_MASK \
   (UINT64_C(0xFFFFFFFFFFFFFFFF))

/* Direct Write Interface doutw : Src0. */

/* Lower 64-bit (63:0) data to be written */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC0_DATA_SHIFT (0U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC0_DATA_CLRMSK \
   (UINT64_C(0x0000000000000000))

/* Unified Store */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_DEST_UNIFIED_STORE (0x00000000U)

/* Common Store */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_DEST_COMMON_STORE (0x00000001U)

/* Primary instance data offset in 128 bit words (offset into the current
 * instance).
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_AO_MASK (0x00001FFFU)

/* DMA Interface DOutD : Src1. */

/* Last Write or DMA in program (This needs to only be set once on with the last
 * DMA or last direct write, which ever is last).
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_LAST_SHIFT (31U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_LAST_CLRMSK (0x7FFFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_LAST_EN (0x80000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_DEST_SHIFT (28U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_DEST_CLRMSK (0xEFFFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_DEST_UNIFIED_STORE \
   (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_DEST_COMMON_STORE (0x10000000U)

/* CMODE   Cache Mode */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_CMODE_SHIFT (26U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_CMODE_CLRMSK (0xF3FFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_CMODE_CACHED (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_CMODE_BYPASS (0x04000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_CMODE_FORCE_LINE_FILL \
   (0x08000000U)

/* Primary instance data offset in 32 bit words (offset into the current
 * instance). For 64 bit writes the address needs to be 64 bit aligned.
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_AO_SHIFT (13U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_AO_CLRMSK (0xFC001FFFU)

/* 2-bit dword write mask. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_BSIZE_SHIFT (0U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_BSIZE_CLRMSK (0xFFFFFFFCU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_BSIZE_LOWER (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_BSIZE_UPPER (0x00000001U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_BSIZE_ALL64 (0x00000002U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_BSIZE_NONE (0x00000003U)

/* VDM Writeback Interface Doutv : Src0 */

/* Number of Indices to use in Draw Indirect (0 = 0) */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTV_SBASE_MASK (0xFFFFFFFFU)

/* VDM Writeback Interface Doutv : Src1 */

/* Number of Indices to use in Draw Indirect (0 = 0) */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTV_SRC1_SBASE_SHIFT (0U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTV_SRC1_SBASE_CLRMSK (0x00000000U)

/* Shade Model Control */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SHADEMODEL_MASK (0x00000003U)

/* Vertex 0 is the flat shaded color source. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SHADEMODEL_FLAT_VERTEX0 \
   (0x00000000U)

/* Vertex 1 is the flat shaded color source. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SHADEMODEL_FLAT_VERTEX1 \
   (0x00000001U)

/* Vertex 2 is the flat shaded color source. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SHADEMODEL_FLAT_VERTEX2 \
   (0x00000002U)

/* Gouraud shaded. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SHADEMODEL_GOURAUD (0x00000003U)

#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SIZE_MASK (0x00000003U)

/* 1 Dimension (U) */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SIZE_1D (0x00000000U)

/* 2 Dimension (UV) */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SIZE_2D (0x00000001U)

/* 3 Dimension (UVS) */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SIZE_3D (0x00000002U)

/* 4 Dimension (UVST) */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SIZE_4D (0x00000003U)

/* This issue is perspective correct. */

/* No W */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_PERSPECTIVE_DISABLE (0x00000000U)

/* Use W */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_PERSPECTIVE_ENABLE (0x00000001U)

/* The offset within the vertex if all data is treated as F32 (even if submitted
 * as F16).
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_F32_OFFSET_MASK (0x000000FFU)

/* The offset within vertex taking into account the F16s and F32s present. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_F16_OFFSET_MASK (0x000000FFU)

/* TSP Parameter Fetch Interface DOutI, This command is only legal in a
 * coefficient loading program.
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_MASK (0x1FFFFFFFU)

/* Apply depth bias to this layer. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_DEPTHBIAS_SHIFT (27U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_DEPTHBIAS_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_DEPTHBIAS_EN (0x08000000U)

/* Ignore the F16 and F32 offsets, and the WMODE and send the primitive id
 * instead.
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_PRIMITIVEID_SHIFT (26U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_PRIMITIVEID_CLRMSK (0xFBFFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_PRIMITIVEID_EN (0x04000000U)

/* Shade Model for Layer. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_SHADEMODEL_SHIFT (24U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_SHADEMODEL_CLRMSK (0xFCFFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_SHADEMODEL_FLAT_VERTEX0 \
   (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_SHADEMODEL_FLAT_VERTEX1 \
   (0x01000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_SHADEMODEL_FLAT_VERTEX2 \
   (0x02000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_SHADEMODEL_GOURAUD (0x03000000U)

/* Point sprite Forced. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_POINTSPRITE_SHIFT (23U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_POINTSPRITE_CLRMSK (0xFF7FFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_POINTSPRITE_EN (0x00800000U)

/* Wrap S Coordinate. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_WRAPS_SHIFT (22U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_WRAPS_CLRMSK (0xFFBFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_WRAPS_EN (0x00400000U)

/* Wrap V Coordinate. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_WRAPV_SHIFT (21U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_WRAPV_CLRMSK (0xFFDFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_WRAPV_EN (0x00200000U)

/* Wrap U Coordinate. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_WRAPU_SHIFT (20U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_WRAPU_CLRMSK (0xFFEFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_WRAPU_EN (0x00100000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_SIZE_SHIFT (18U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_SIZE_CLRMSK (0xFFF3FFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_SIZE_1D (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_SIZE_2D (0x00040000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_SIZE_3D (0x00080000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_SIZE_4D (0x000C0000U)

/* Issue is for F16 precision values. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_F16_SHIFT (17U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_F16_CLRMSK (0xFFFDFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_F16_EN (0x00020000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_PERSPECTIVE_SHIFT (16U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_PERSPECTIVE_CLRMSK (0xFFFEFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_PERSPECTIVE_DISABLE \
   (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_PERSPECTIVE_ENABLE (0x00010000U)
/* The offset within the vertex if all data is treated as F32 (even if submitted
 * as F16).
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_F32_OFFSET_SHIFT (8U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_F32_OFFSET_CLRMSK (0xFFFF00FFU)

/* The offset within vertex taking into account the F16s and F32s present. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_F16_OFFSET_SHIFT (0U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC_F16_OFFSET_CLRMSK (0xFFFFFF00U)

/* The starting address to write the data into the common store allocation, in
 * 128 bit words. Each 32 bit value consumes 128 bit words in the common store.
 * The issues are pack, Issue 0, followed by Issue 1.
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_DEST_MASK (0x000000FFU)

/* TSP Parameter Fetch Interface DOutI : Src0 */

/* This is the last issue for the triangle. */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE_SHIFT (63U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE_CLRMSK \
   (UINT64_C(0x7FFFFFFFFFFFFFFF))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE_EN \
   (UINT64_C(0x8000000000000000))

/* The starting address to write the data into the common store allocation, in
 * 128 bit words. Each 32 bit value consumes 128 bit words in the common store.
 * The issues are pack, Issue 0, followed by Issue 1.
 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_DEST_SHIFT (54U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_DEST_CLRMSK \
   (UINT64_C(0xC03FFFFFFFFFFFFF))

/* Issue 0 */
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_SHIFT (0U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_CLRMSK \
   (UINT64_C(0xFFFFFFFFE0000000))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_DEPTHBIAS_SHIFT (27U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_DEPTHBIAS_CLRMSK \
   (UINT64_C(0xfffffffff7ffffff))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_PRIMITIVEID_SHIFT (26U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_PRIMITIVEID_CLRMSK \
   (UINT64_C(0xfffffffffbffffff))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_SHADEMODEL_SHIFT (24U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_SHADEMODEL_CLRMSK \
   (UINT64_C(0xfffffffffcffffff))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_POINTSPRITE_SHIFT (23U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_POINTSPRITE_CLRMSK \
   (UINT64_C(0xffffffffff7fffff))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_WRAPS_SHIFT (22U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_WRAPS_CLRMSK \
   (UINT64_C(0xffffffffffbfffff))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_WRAPV_SHIFT (21U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_WRAPV_CLRMSK \
   (UINT64_C(0xffffffffffdfffff))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_WRAPU_SHIFT (20U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_WRAPU_CLRMSK \
   (UINT64_C(0xffffffffffefffff))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_SIZE_SHIFT (18U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_SIZE_CLRMSK \
   (UINT64_C(0xfffffffffff3ffff))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_F16_SHIFT (17U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_F16_CLRMSK \
   (UINT64_C(0xfffffffffffdffff))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_PERSPECTIVE_SHIFT (16U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_PERSPECTIVE_CLRMSK \
   (UINT64_C(0xfffffffffffeffff))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_F32_OFFSET_SHIFT (8U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_F32_OFFSET_CLRMSK \
   (UINT64_C(0xffffffffffff00ff))
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_F16_OFFSET_SHIFT (0U)
#define PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTI_SRC0_ISSUE0_F16_OFFSET_CLRMSK \
   (UINT64_C(0xffffffffffffff00))

/* TSP Parameter Fetch Interface DOutI : Src1 */

/* 32-bit Temp or DOUT. */
#define PVR_ROGUE_PDSINST_DSTDOUT_MASK (0x00000007U)

/* DMA data from memory to the USC. */
#define PVR_ROGUE_PDSINST_DSTDOUT_DOUTD (0x00000000U)

/* Write a value directly to the USC. */
#define PVR_ROGUE_PDSINST_DSTDOUT_DOUTW (0x00000001U)

/* Start a USC program. */
#define PVR_ROGUE_PDSINST_DSTDOUT_DOUTU (0x00000002U)

/* Issue a fence back to the VDM (with value). */
#define PVR_ROGUE_PDSINST_DSTDOUT_DOUTV (0x00000003U)

/* Issue a command to the TSP Parameter Fetch and FPU to calculate and load
 * coefficients to USC.
 */
#define PVR_ROGUE_PDSINST_DSTDOUT_DOUTI (0x00000004U)

/* Issue a fence back to the CDM. Used if compute is enabled. */
#define PVR_ROGUE_PDSINST_DSTDOUT_DOUTC (0x00000005U)

/* Issue DOUT to external devices (Opcode SP)
 *
 * cc ? dst = src0, src1
 *
 * PDS programs have to send data somewhere. This is primary function of the
 * PDS. All programs must therefore execute some one of DOUT, DDMAD, STM
 * commands. There are the following program types
 *
 * Vertex Shader, Geometry Shader, Hull Shader Programs These programs load data
 * into memory. These will use the DOUTD or DDMAD commands. Ideally the DDMAD
 * command is used as the most typical operation Src Address = Index * Stride +
 * Base, and then DMA from this address. They also schedule the execution of the
 * USSE and will issue a DOUTU command. This would normally be the last
 * instruction in the program.
 *
 * Obviously the shader programs must not overflow their allocated memory.
 * However, the USC will do cache look-aheads and so could attempt to fetch
 * shader code from beyond the end of the program. This could cause a page fault
 * if the last program instructions are very close to the end of the last valid
 * memory page.
 *
 * To avoid this happening always ensure that the start address of the last
 * instruction of a shader program does not occur in the last 26 bytes of a
 * page.
 *
 * State/Uniform Loading Programs
 * These programs load data into memory. These will use the typically use the
 * DOUTD command
 *
 * Coefficient Loading Programs
 * These programs run once per triangle. They load the A,B,C Coefficient for the
 * iteration of the varyings into the USC. These programs issue DOUTI
 * commands. These programs must not do any other sort of DOUT command
 * (DOUTW/DOUTD/DOUTU).
 *
 * Pixel Shader Programs
 * These programs once per group of pixels, schedule the execution of a pixel
 * shader on the USC for a group of pixels. This program issues a DOUTU (and
 * that is all).
 */

#define PVR_ROGUE_PDSINST_DOUT_OPCODE_SHIFT (28U)
#define PVR_ROGUE_PDSINST_DOUT_OPCODE_CLRMSK (0x0FFFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_OPCODE_DEFAULT (0xF0000000U) /* DOUT */
#define PVR_ROGUE_PDSINST_DOUT_CC_SHIFT (27U)
#define PVR_ROGUE_PDSINST_DOUT_CC_CLRMSK (0xF7FFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_CC_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_CC_ENABLE (0x08000000U)
#define PVR_ROGUE_PDSINST_DOUT_END_SHIFT (26U)
#define PVR_ROGUE_PDSINST_DOUT_END_CLRMSK (0xFBFFFFFFU)
#define PVR_ROGUE_PDSINST_DOUT_END_DISABLE (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_END_ENABLE (0x04000000U)

/* 32-bit source */
#define PVR_ROGUE_PDSINST_DOUT_SRC1_SHIFT (16U)
#define PVR_ROGUE_PDSINST_DOUT_SRC1_CLRMSK (0xFF00FFFFU)

/* 64-bit source */
#define PVR_ROGUE_PDSINST_DOUT_SRC0_SHIFT (8U)
#define PVR_ROGUE_PDSINST_DOUT_SRC0_CLRMSK (0xFFFF80FFU)

/* DOUT Destination */
#define PVR_ROGUE_PDSINST_DOUT_DST_SHIFT (0U)
#define PVR_ROGUE_PDSINST_DOUT_DST_CLRMSK (0xFFFFFFF8U)
#define PVR_ROGUE_PDSINST_DOUT_DST_DOUTD (0x00000000U)
#define PVR_ROGUE_PDSINST_DOUT_DST_DOUTW (0x00000001U)
#define PVR_ROGUE_PDSINST_DOUT_DST_DOUTU (0x00000002U)
#define PVR_ROGUE_PDSINST_DOUT_DST_DOUTV (0x00000003U)
#define PVR_ROGUE_PDSINST_DOUT_DST_DOUTI (0x00000004U)
#if defined(ROGUE_FEATURE_COMPUTE)
#   define PVR_ROGUE_PDSINST_DOUT_DST_DOUTC (0x00000005U)
#endif /* ROGUE_FEATURE_COMPUTE */

/* Shift */

#endif /* PVR_ROGUE_PDS_DEFS_H */
