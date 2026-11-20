/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: MTFW PFM data struct
 *
 */
#ifndef __MTFW_FWIF_PCS_H__
#define __MTFW_FWIF_PCS_H__

#include "mtfw_fwif_types.h"
#include "mtfw_fwif_submission.h"

#ifdef __cplusplus
extern "C"
{
#endif

// =====================================================================================================================
// Entrance of the overall pc sampler cfg
typedef struct MTFW_ALIGN_BYTES_8
{
    union
    {
        struct
        {
            uint32_t enable      : 1; // Set to enable PC Sampler, otherwise to disable
            uint32_t counterSize : 1; // Set for 32-bit counters, otherwise 64-bit counters
            uint32_t random2     : 1; // Using random 2 mode ; otherwise random 1 mode
            uint32_t stallReason : 19;
            uint32_t periodLevel : 5; // Use for calculations samplePeriod = (2 ^ (periodLevel + 5)) cycles
            uint32_t rsrv        : 5;
        } bits;

        uint32_t value;
    } cfg;

    fw_va dropCntBufVa;    // Base va of the buffer to store drop count which struct is MTFW_FWIF_DROP_CNT

    // TODO: delete below variable after driver not use
    fw_va dropCntCdmBufVa; // Base va of the buffer to store drop count of CDM
    fw_va dropCntGfxBufVa; // Base va of the buffer to store drop count of GFX
} MTFW_FWIF_PC_SAMPLER_CFG;

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_PCS_H__ */
