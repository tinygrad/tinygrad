/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: Defines hardware context setting related data structures.
 *
 */
#ifndef __MTFW_FWIF_HWCONTEXT_H__
#define __MTFW_FWIF_HWCONTEXT_H__

#include "mtfw_fwif_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

// =====================================================================================================================
typedef struct MTFW_ALIGN_BYTES_8
{
    uint64_t taRegDceCmd0;
    uint32_t taRegDceCmd1;
    uint32_t taRegDceWrite;
    uint64_t taRegDceDraw0;
    uint64_t taRegDceDraw1;
    uint32_t taRegGtaSoPrim[4];
    uint16_t taCurrentIdx;
} MTFW_GEOM_CTX_STATE;

// =====================================================================================================================
typedef struct MTFW_ALIGN_BYTES_8
{
    // FW-accessible ISP state which must be written out to memory on context store
    uint32_t ispSTORE[32];
    uint32_t pmDeallocMaskStatus;
    uint32_t pmPdsMtileFreeStatus;
} MTFW_PIXEL_CTX_STATE;

// =====================================================================================================================
// Context store configuration for TA
typedef struct MTFW_ALIGN_BYTES_8
{
    // 3D ctx buffer, used for partial render
    uint64_t vdmContextStateBaseAddr;
    uint64_t geContextStateBaseAddr;
    // vdm store
    uint64_t dceContextStoreTaskVDM0;
    uint64_t dceContextStoreTaskVDM1;
    uint64_t dceContextStoreTaskVDM2;
    // vdm resume
    uint64_t dceContextResumeTaskVDM0;
    uint64_t dceContextResumeTaskVDM1;
    uint64_t dceContextResumeTaskVDM2;
    // ddm store
    uint64_t dceContextStoreTaskDDM0;
    uint64_t dceContextStoreTaskDDM1;
    uint64_t dceContextStoreTaskDDM2;
    uint64_t dceContextStoreTaskXFB;
    // ddm resume
    uint64_t dceContextResumeTaskDDM0;
    uint64_t dceContextResumeTaskDDM1;
    uint64_t dceContextResumeTaskDDM2;
    uint64_t dceContextResumeTaskXFB;
} MTFW_GEOM_CTX_REGS;

// =====================================================================================================================
// Context store configuration for Compute
typedef struct MTFW_ALIGN_BYTES_8
{
    uint64_t cdmContextPds0;
    uint64_t cdmContextPds1;
    uint64_t cdmTerminatePds0;
    uint64_t cdmTerminatePds1;

    uint64_t cdmResumePds0;
    uint64_t cdmResumePds1;
} MTFW_COMPUTE_CTX_REGS;

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_HWCONTEXT_H__ */
