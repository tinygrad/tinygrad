/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef VBIOS_H
#define VBIOS_H

#include "gpu/vbios/bios_types.h"
#define FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER     (0x4)

typedef struct
{
    NvU32 signature;
    NvU16 version;
    NvU16 size;
    NvU32 cmd_in_buffer_offset;
    NvU32 cmd_in_buffer_size;
    NvU32 cmd_out_buffer_offset;
    NvU32 cmd_out_buffer_size;
    NvU32 nvf_img_data_buffer_offset;
    NvU32 nvf_img_data_buffer_size;
    NvU32 printfBufferHdr;
    NvU32 ucode_build_time_stamp;
    NvU32 ucode_signature;
    NvU32 init_cmd;
    NvU32 ucode_feature;
    NvU32 ucode_cmd_mask0;
    NvU32 ucode_cmd_mask1;
    NvU32 multiTgtTbl;
} FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3;

#define FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_FRTS (0x15)
#define FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_SB   (0x19)


#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_FLAGS_VERSION                0:0
#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_FLAGS_VERSION_UNAVAILABLE    0x00
#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_FLAGS_VERSION_AVAILABLE      0x01
#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_FLAGS_RESERVED               1:1
#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_FLAGS_ENCRYPTED              2:2
#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_RESERVED                     7:3
#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION                      15:8
#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V1                   0x01
#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V2                   0x02
#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V3                   0x03
#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V4                   0x04
#define NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_SIZE                         31:16

typedef struct
{
    unsigned int vDesc;
} FALCON_UCODE_DESC_HEADER;
#define FALCON_UCODE_DESC_HEADER_FORMAT   "1d"

typedef struct {
    FALCON_UCODE_DESC_HEADER Hdr;
    unsigned int StoredSize;
    unsigned int PKCDataOffset;
    unsigned int InterfaceOffset;
    unsigned int IMEMPhysBase;
    unsigned int IMEMLoadSize;
    unsigned int IMEMVirtBase;
    unsigned int DMEMPhysBase;
    unsigned int DMEMLoadSize;
    unsigned short EngineIdMask;
    unsigned char UcodeId;
    unsigned char SignatureCount;
    unsigned short SignatureVersions;
    unsigned short Reserved;
} FALCON_UCODE_DESC_V3;

#define FALCON_UCODE_DESC_V3_SIZE_44    44
#define FALCON_UCODE_DESC_V3_44_FMT     "9d1w2b2w"
#define BCRT30_RSA3K_SIG_SIZE 384

typedef struct
{
    NvU32 version;
    NvU32 size;
    NvU64 gfwImageOffset;
    NvU32 gfwImageSize;
    NvU32 flags;
} FWSECLIC_READ_VBIOS_DESC;

#define FWSECLIC_READ_VBIOS_STRUCT_FLAGS (2)

typedef struct
{
    NvU32 version;
    NvU32 size;
    NvU32 frtsRegionOffset4K;
    NvU32 frtsRegionSize;
    NvU32 frtsRegionMediaType;
} FWSECLIC_FRTS_REGION_DESC;

#define FWSECLIC_FRTS_REGION_MEDIA_FB (2)
#define FWSECLIC_FRTS_REGION_SIZE_1MB_IN_4K (0x100)

typedef struct
{
    FWSECLIC_READ_VBIOS_DESC readVbiosDesc;
    FWSECLIC_FRTS_REGION_DESC frtsRegionDesc;
} FWSECLIC_FRTS_CMD;

#endif /* VBIOS_H */
