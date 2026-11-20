/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: MTFW utilization data struct
 *
 */
#ifndef __MTFW_FWIF_UTIL_H__
#define __MTFW_FWIF_UTIL_H__

#include "mtfw_fwif_types.h"
#include "mtfw_fwif_dm_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
    CFG_GPU_UTIL_ALL_DISABLE    = 0x0, /* disable calculate utilization. */
    CFG_GPU_UTIL_OVERALL_ENABLE = 0x1, /* calculate overall utilization. */
    CFG_GPU_UTIL_ALL_ENABLE     = 0x2, /* calculate all specific dm utilization. */
} MTFW_FWIF_GPU_UTIL_CFG_VALUE;

typedef enum
{
    GPU_UTIL_STATE_IDX_IDLE   = 0x0,
    GPU_UTIL_STATE_IDX_ACTIVE = 0x1,
    GPU_UTIL_STATE_IDX_MAX,
} MTFW_FWIF_GPU_UTIL_STATE;

typedef enum
{
    GPU_UTIL_ITEM_IDX_OVERALL_UTIL = 0x0,
    GPU_UTIL_ITEM_IDX_TDM          = 0x1,
    GPU_UTIL_ITEM_IDX_GEOM         = 0x2,
    GPU_UTIL_ITEM_IDX_3D           = 0x3,
    GPU_UTIL_ITEM_IDX_CDM          = 0x4,
    GPU_UTIL_ITEM_IDX_CE           = 0x5,
    GPU_UTIL_ITEM_IDX_MAX,
} MTFW_FWIF_GPU_UTIL_ITEM_INDEX;

typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_FWIF_GPU_UTIL_CFG_VALUE gpuUtilCfg;
} MTFW_FWIF_GPU_UTIL_CONFIG;

typedef struct MTFW_ALIGN_BYTES_8
{
    /* the state of latest dm update */
    uint64_t latestUpdateState;

    /* the timestamp of latest dm state update */
    uint64_t latestUpdateTimestamp;

    /* Counters for the amount of time the GPU DM was active/idle */
    uint64_t statsCounters[GPU_UTIL_STATE_IDX_MAX];
} MTFW_FWIF_GPU_UTIL_ITEM;

typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_FWIF_GPU_UTIL_ITEM gpuDmUtil[GPU_UTIL_ITEM_IDX_MAX];

    uint32_t fwScheCounterBuf[256]; /* Reserve 1k for schedule counter. layout is MTFW_FW_SCH_COUNTER */
} MTFW_FWIF_GPU_UTIL_BUF;

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_UTIL_H__ */
