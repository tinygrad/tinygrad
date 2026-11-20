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

/* Enums, structures and pack functions for ROGUE.
 *
 * This file has been generated, do not hand edit.
 */

#ifndef ROGUE_CDM_H
#define ROGUE_CDM_H

#include "csbgen/pvr_packet_helpers.h"


#define ROGUE_CDMCTRL_GUARD_SIZE_DEFAULT         64

enum ROGUE_CDMCTRL_BLOCK_TYPE {
    ROGUE_CDMCTRL_BLOCK_TYPE_COMPUTE_KERNEL =      0,
    ROGUE_CDMCTRL_BLOCK_TYPE_STREAM_LINK =      1,
    ROGUE_CDMCTRL_BLOCK_TYPE_STREAM_TERMINATE =      2,
};

static const char *
ROGUE_CDMCTRL_BLOCK_TYPE_to_str(const enum ROGUE_CDMCTRL_BLOCK_TYPE value)
{
    switch (value) {
    case ROGUE_CDMCTRL_BLOCK_TYPE_COMPUTE_KERNEL: return "COMPUTE_KERNEL";
    case ROGUE_CDMCTRL_BLOCK_TYPE_STREAM_LINK: return "STREAM_LINK";
    case ROGUE_CDMCTRL_BLOCK_TYPE_STREAM_TERMINATE: return "STREAM_TERMINATE";
    default: return NULL;
    }
}

enum ROGUE_CDMCTRL_USC_TARGET {
    ROGUE_CDMCTRL_USC_TARGET_ALL         =      0,
    ROGUE_CDMCTRL_USC_TARGET_ANY         =      1,
};

static const char *
ROGUE_CDMCTRL_USC_TARGET_to_str(const enum ROGUE_CDMCTRL_USC_TARGET value)
{
    switch (value) {
    case ROGUE_CDMCTRL_USC_TARGET_ALL: return "ALL";
    case ROGUE_CDMCTRL_USC_TARGET_ANY: return "ANY";
    default: return NULL;
    }
}

enum ROGUE_CDMCTRL_SD_TYPE {
    ROGUE_CDMCTRL_SD_TYPE_NONE           =      0,
    ROGUE_CDMCTRL_SD_TYPE_PDS            =      1,
    ROGUE_CDMCTRL_SD_TYPE_USC            =      2,
};

static const char *
ROGUE_CDMCTRL_SD_TYPE_to_str(const enum ROGUE_CDMCTRL_SD_TYPE value)
{
    switch (value) {
    case ROGUE_CDMCTRL_SD_TYPE_NONE: return "NONE";
    case ROGUE_CDMCTRL_SD_TYPE_PDS: return "PDS";
    case ROGUE_CDMCTRL_SD_TYPE_USC: return "USC";
    default: return NULL;
    }
}

#define ROGUE_CDMCTRL_KERNEL0_length           1
#define ROGUE_CDMCTRL_KERNEL0_header            \
    .block_type                          = ROGUE_CDMCTRL_BLOCK_TYPE_COMPUTE_KERNEL

/* Helper macros for usc_common_size */
#define ROGUE_CDMCTRL_KERNEL0_USC_COMMON_SIZE_UNIT_SIZE 64
#define ROGUE_CDMCTRL_KERNEL0_USC_COMMON_SIZE_MAX_SIZE 256

/* Helper macros for usc_unified_size */
#define ROGUE_CDMCTRL_KERNEL0_USC_UNIFIED_SIZE_UNIT_SIZE 16

/* Helper macros for pds_temp_size */
#define ROGUE_CDMCTRL_KERNEL0_PDS_TEMP_SIZE_UNIT_SIZE 16

/* Helper macros for pds_data_size */
#define ROGUE_CDMCTRL_KERNEL0_PDS_DATA_SIZE_UNIT_SIZE 16

struct ROGUE_CDMCTRL_KERNEL0 {
    enum ROGUE_CDMCTRL_BLOCK_TYPE        block_type;
    bool                                 indirect_present;
    bool                                 global_offsets_present;
    bool                                 event_object_present;
    uint32_t                             usc_common_size;
    uint32_t                             usc_unified_size;
    uint32_t                             pds_temp_size;
    uint32_t                             pds_data_size;
    enum ROGUE_CDMCTRL_USC_TARGET        usc_target;
    bool                                 fence;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL0_pack(__attribute__((unused)) void * restrict dst,
                           __attribute__((unused)) const struct ROGUE_CDMCTRL_KERNEL0 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    dw[0] =
      __pvr_uint(values->block_type, 30, 31) |
      __pvr_uint(values->indirect_present, 29, 29) |
      __pvr_uint(values->global_offsets_present, 28, 28) |
      __pvr_uint(values->event_object_present, 27, 27) |
      __pvr_uint(values->usc_common_size, 18, 26) |
      __pvr_uint(values->usc_unified_size, 12, 17) |
      __pvr_uint(values->pds_temp_size, 8, 11) |
      __pvr_uint(values->pds_data_size, 2, 7) |
      __pvr_uint(values->usc_target, 1, 1) |
      __pvr_uint(values->fence, 0, 0);
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL0_unpack(__attribute__((unused)) const void * restrict src,
                             __attribute__((unused)) struct ROGUE_CDMCTRL_KERNEL0 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->block_type = __pvr_uint_unpack(dw[0], 30, 31);
    values->indirect_present = __pvr_uint_unpack(dw[0], 29, 29);
    values->global_offsets_present = __pvr_uint_unpack(dw[0], 28, 28);
    values->event_object_present = __pvr_uint_unpack(dw[0], 27, 27);
    values->usc_common_size = __pvr_uint_unpack(dw[0], 18, 26);
    values->usc_unified_size = __pvr_uint_unpack(dw[0], 12, 17);
    values->pds_temp_size = __pvr_uint_unpack(dw[0], 8, 11);
    values->pds_data_size = __pvr_uint_unpack(dw[0], 2, 7);
    values->usc_target = __pvr_uint_unpack(dw[0], 1, 1);
    values->fence = __pvr_uint_unpack(dw[0], 0, 0);
}

#define ROGUE_CDMCTRL_KERNEL1_length           1
#define ROGUE_CDMCTRL_KERNEL1_header            \


/* Helper macros for data_addr */
#define ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT 16

struct ROGUE_CDMCTRL_KERNEL1 {
    __pvr_address_type                   data_addr;
    enum ROGUE_CDMCTRL_SD_TYPE           sd_type;
    bool                                 usc_common_shared;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL1_pack(__attribute__((unused)) void * restrict dst,
                           __attribute__((unused)) const struct ROGUE_CDMCTRL_KERNEL1 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    const uint32_t v0 =
      __pvr_uint(values->sd_type, 2, 3) |
      __pvr_uint(values->usc_common_shared, 1, 1);
    dw[0] = __pvr_address(values->data_addr, 4, 4, 31) | v0;
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL1_unpack(__attribute__((unused)) const void * restrict src,
                             __attribute__((unused)) struct ROGUE_CDMCTRL_KERNEL1 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->data_addr = __pvr_address_unpack(dw[0], 4, 4, 31);
    values->sd_type = __pvr_uint_unpack(dw[0], 2, 3);
    values->usc_common_shared = __pvr_uint_unpack(dw[0], 1, 1);
}

#define ROGUE_CDMCTRL_KERNEL2_length           1
#define ROGUE_CDMCTRL_KERNEL2_header            \


/* Helper macros for code_addr */
#define ROGUE_CDMCTRL_KERNEL2_CODE_ADDR_ALIGNMENT 16

struct ROGUE_CDMCTRL_KERNEL2 {
    __pvr_address_type                   code_addr;
    bool                                 one_wg_per_task;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL2_pack(__attribute__((unused)) void * restrict dst,
                           __attribute__((unused)) const struct ROGUE_CDMCTRL_KERNEL2 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    const uint32_t v0 =
      __pvr_uint(values->one_wg_per_task, 0, 0);
    dw[0] = __pvr_address(values->code_addr, 4, 4, 31) | v0;
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL2_unpack(__attribute__((unused)) const void * restrict src,
                             __attribute__((unused)) struct ROGUE_CDMCTRL_KERNEL2 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->code_addr = __pvr_address_unpack(dw[0], 4, 4, 31);
    values->one_wg_per_task = __pvr_uint_unpack(dw[0], 0, 0);
}

#define ROGUE_CDMCTRL_KERNEL3_length           1
#define ROGUE_CDMCTRL_KERNEL3_header            \


struct ROGUE_CDMCTRL_KERNEL3 {
    uint32_t                             workgroup_x;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL3_pack(__attribute__((unused)) void * restrict dst,
                           __attribute__((unused)) const struct ROGUE_CDMCTRL_KERNEL3 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    dw[0] =
      __pvr_uint(values->workgroup_x, 0, 31);
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL3_unpack(__attribute__((unused)) const void * restrict src,
                             __attribute__((unused)) struct ROGUE_CDMCTRL_KERNEL3 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->workgroup_x = __pvr_uint_unpack(dw[0], 0, 31);
}

#define ROGUE_CDMCTRL_KERNEL4_length           1
#define ROGUE_CDMCTRL_KERNEL4_header            \


struct ROGUE_CDMCTRL_KERNEL4 {
    uint32_t                             workgroup_y;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL4_pack(__attribute__((unused)) void * restrict dst,
                           __attribute__((unused)) const struct ROGUE_CDMCTRL_KERNEL4 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    dw[0] =
      __pvr_uint(values->workgroup_y, 0, 31);
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL4_unpack(__attribute__((unused)) const void * restrict src,
                             __attribute__((unused)) struct ROGUE_CDMCTRL_KERNEL4 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->workgroup_y = __pvr_uint_unpack(dw[0], 0, 31);
}

#define ROGUE_CDMCTRL_KERNEL5_length           1
#define ROGUE_CDMCTRL_KERNEL5_header            \


struct ROGUE_CDMCTRL_KERNEL5 {
    uint32_t                             workgroup_z;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL5_pack(__attribute__((unused)) void * restrict dst,
                           __attribute__((unused)) const struct ROGUE_CDMCTRL_KERNEL5 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    dw[0] =
      __pvr_uint(values->workgroup_z, 0, 31);
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL5_unpack(__attribute__((unused)) const void * restrict src,
                             __attribute__((unused)) struct ROGUE_CDMCTRL_KERNEL5 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->workgroup_z = __pvr_uint_unpack(dw[0], 0, 31);
}

#define ROGUE_CDMCTRL_KERNEL6_length           1
#define ROGUE_CDMCTRL_KERNEL6_header            \


/* Helper macros for indirect_addrmsb */
#define ROGUE_CDMCTRL_KERNEL6_INDIRECT_ADDRMSB_ALIGNMENT 4294967296

struct ROGUE_CDMCTRL_KERNEL6 {
    __pvr_address_type                   indirect_addrmsb;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL6_pack(__attribute__((unused)) void * restrict dst,
                           __attribute__((unused)) const struct ROGUE_CDMCTRL_KERNEL6 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    dw[0] = __pvr_address(values->indirect_addrmsb, 32, 0, 7) | 0;
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL6_unpack(__attribute__((unused)) const void * restrict src,
                             __attribute__((unused)) struct ROGUE_CDMCTRL_KERNEL6 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->indirect_addrmsb = __pvr_address_unpack(dw[0], 32, 0, 7);
}

#define ROGUE_CDMCTRL_KERNEL7_length           1
#define ROGUE_CDMCTRL_KERNEL7_header            \


/* Helper macros for indirect_addrlsb */
#define ROGUE_CDMCTRL_KERNEL7_INDIRECT_ADDRLSB_ALIGNMENT 4

struct ROGUE_CDMCTRL_KERNEL7 {
    __pvr_address_type                   indirect_addrlsb;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL7_pack(__attribute__((unused)) void * restrict dst,
                           __attribute__((unused)) const struct ROGUE_CDMCTRL_KERNEL7 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    dw[0] = __pvr_address(values->indirect_addrlsb, 2, 2, 31) | 0;
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL7_unpack(__attribute__((unused)) const void * restrict src,
                             __attribute__((unused)) struct ROGUE_CDMCTRL_KERNEL7 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->indirect_addrlsb = __pvr_address_unpack(dw[0], 2, 2, 31);
}

#define ROGUE_CDMCTRL_KERNEL8_length           1
#define ROGUE_CDMCTRL_KERNEL8_header            \


/* Helper macros for max_instances */
#define ROGUE_CDMCTRL_KERNEL8_MAX_INSTANCES_MAX_SIZE 31

struct ROGUE_CDMCTRL_KERNEL8 {
    uint32_t                             max_instances;
    uint32_t                             workgroup_size_x;
    uint32_t                             workgroup_size_y;
    uint32_t                             workgroup_size_z;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL8_pack(__attribute__((unused)) void * restrict dst,
                           __attribute__((unused)) const struct ROGUE_CDMCTRL_KERNEL8 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    dw[0] =
      __pvr_uint(values->max_instances, 27, 31) |
      __pvr_uint(values->workgroup_size_x, 18, 26) |
      __pvr_uint(values->workgroup_size_y, 9, 17) |
      __pvr_uint(values->workgroup_size_z, 0, 8);
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL8_unpack(__attribute__((unused)) const void * restrict src,
                             __attribute__((unused)) struct ROGUE_CDMCTRL_KERNEL8 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->max_instances = __pvr_uint_unpack(dw[0], 27, 31);
    values->workgroup_size_x = __pvr_uint_unpack(dw[0], 18, 26);
    values->workgroup_size_y = __pvr_uint_unpack(dw[0], 9, 17);
    values->workgroup_size_z = __pvr_uint_unpack(dw[0], 0, 8);
}

#define ROGUE_CDMCTRL_KERNEL9_length           1
#define ROGUE_CDMCTRL_KERNEL9_header            \


struct ROGUE_CDMCTRL_KERNEL9 {
    uint32_t                             global_offset_x;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL9_pack(__attribute__((unused)) void * restrict dst,
                           __attribute__((unused)) const struct ROGUE_CDMCTRL_KERNEL9 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    dw[0] =
      __pvr_uint(values->global_offset_x, 0, 31);
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL9_unpack(__attribute__((unused)) const void * restrict src,
                             __attribute__((unused)) struct ROGUE_CDMCTRL_KERNEL9 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->global_offset_x = __pvr_uint_unpack(dw[0], 0, 31);
}

#define ROGUE_CDMCTRL_KERNEL10_length          1
#define ROGUE_CDMCTRL_KERNEL10_header           \


struct ROGUE_CDMCTRL_KERNEL10 {
    uint32_t                             global_offset_y;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL10_pack(__attribute__((unused)) void * restrict dst,
                            __attribute__((unused)) const struct ROGUE_CDMCTRL_KERNEL10 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    dw[0] =
      __pvr_uint(values->global_offset_y, 0, 31);
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL10_unpack(__attribute__((unused)) const void * restrict src,
                              __attribute__((unused)) struct ROGUE_CDMCTRL_KERNEL10 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->global_offset_y = __pvr_uint_unpack(dw[0], 0, 31);
}

#define ROGUE_CDMCTRL_KERNEL11_length          1
#define ROGUE_CDMCTRL_KERNEL11_header           \


struct ROGUE_CDMCTRL_KERNEL11 {
    uint32_t                             global_offset_z;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL11_pack(__attribute__((unused)) void * restrict dst,
                            __attribute__((unused)) const struct ROGUE_CDMCTRL_KERNEL11 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    dw[0] =
      __pvr_uint(values->global_offset_z, 0, 31);
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_KERNEL11_unpack(__attribute__((unused)) const void * restrict src,
                              __attribute__((unused)) struct ROGUE_CDMCTRL_KERNEL11 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->global_offset_z = __pvr_uint_unpack(dw[0], 0, 31);
}

#define ROGUE_CDMCTRL_STREAM_LINK0_length      1
#define ROGUE_CDMCTRL_STREAM_LINK0_header       \
    .block_type                          = ROGUE_CDMCTRL_BLOCK_TYPE_STREAM_LINK

/* Helper macros for link_addrmsb */
#define ROGUE_CDMCTRL_STREAM_LINK0_LINK_ADDRMSB_ALIGNMENT 4294967296

struct ROGUE_CDMCTRL_STREAM_LINK0 {
    enum ROGUE_CDMCTRL_BLOCK_TYPE        block_type;
    __pvr_address_type                   link_addrmsb;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_STREAM_LINK0_pack(__attribute__((unused)) void * restrict dst,
                                __attribute__((unused)) const struct ROGUE_CDMCTRL_STREAM_LINK0 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    const uint32_t v0 =
      __pvr_uint(values->block_type, 30, 31);
    dw[0] = __pvr_address(values->link_addrmsb, 32, 0, 7) | v0;
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_STREAM_LINK0_unpack(__attribute__((unused)) const void * restrict src,
                                  __attribute__((unused)) struct ROGUE_CDMCTRL_STREAM_LINK0 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->block_type = __pvr_uint_unpack(dw[0], 30, 31);
    values->link_addrmsb = __pvr_address_unpack(dw[0], 32, 0, 7);
}

#define ROGUE_CDMCTRL_STREAM_LINK1_length      1
#define ROGUE_CDMCTRL_STREAM_LINK1_header       \


/* Helper macros for link_addrlsb */
#define ROGUE_CDMCTRL_STREAM_LINK1_LINK_ADDRLSB_ALIGNMENT 4

struct ROGUE_CDMCTRL_STREAM_LINK1 {
    __pvr_address_type                   link_addrlsb;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_STREAM_LINK1_pack(__attribute__((unused)) void * restrict dst,
                                __attribute__((unused)) const struct ROGUE_CDMCTRL_STREAM_LINK1 * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    dw[0] = __pvr_address(values->link_addrlsb, 2, 2, 31) | 0;
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_STREAM_LINK1_unpack(__attribute__((unused)) const void * restrict src,
                                  __attribute__((unused)) struct ROGUE_CDMCTRL_STREAM_LINK1 * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->link_addrlsb = __pvr_address_unpack(dw[0], 2, 2, 31);
}

#define ROGUE_CDMCTRL_STREAM_TERMINATE_length      1
#define ROGUE_CDMCTRL_STREAM_TERMINATE_header   \
    .block_type                          = ROGUE_CDMCTRL_BLOCK_TYPE_STREAM_TERMINATE

struct ROGUE_CDMCTRL_STREAM_TERMINATE {
    enum ROGUE_CDMCTRL_BLOCK_TYPE        block_type;
};

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_STREAM_TERMINATE_pack(__attribute__((unused)) void * restrict dst,
                                    __attribute__((unused)) const struct ROGUE_CDMCTRL_STREAM_TERMINATE * restrict values)
{
    uint32_t * restrict dw = (uint32_t * restrict) dst;

    dw[0] =
      __pvr_uint(values->block_type, 30, 31);
}

static inline __attribute__((always_inline)) void
ROGUE_CDMCTRL_STREAM_TERMINATE_unpack(__attribute__((unused)) const void * restrict src,
                                      __attribute__((unused)) struct ROGUE_CDMCTRL_STREAM_TERMINATE * restrict values)
{
    const uint32_t * restrict dw = (const uint32_t * restrict) src;
    values->block_type = __pvr_uint_unpack(dw[0], 30, 31);
}

#endif /* ROGUE_CDM_H */
