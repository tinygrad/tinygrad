/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: MT GPU Firware ABI generation.
 *
 */
#ifndef __MTFW_FWIF_GEN_1_H__
#define __MTFW_FWIF_GEN_1_H__

#include "mtfw_fwif_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define MTFW_FWIF_GEN_VER_1 1

#ifndef MTFW_FWIF_GEN_VER
#define MTFW_FWIF_GEN_VER MTFW_FWIF_GEN_VER_1
#endif

/**
 * Reserved pages for PM to support MCG
 * Define this value with max multiKick nums(256) * MMU_VCE/TE/ALIST(3) * node(2)
 */
#define MTFW_RSRV_PM_PAGE_CNT 1536

#define MTFW_SCG_PB_COUNT  2
#define MTFW_MCG2_PB_COUNT 4
#define MTFW_MCG3_PB_COUNT 6
#define MTFW_MCG4_PB_COUNT 8
#define MTFW_MCG5_PB_COUNT 10
#define MTFW_MCG6_PB_COUNT 12
#define MTFW_MCG7_PB_COUNT 14
#define MTFW_MCG8_PB_COUNT 16

typedef struct MTFW_ALIGN_BYTES_8
{
    uint32_t version;

    union
    {
        struct
        {
            uint32_t computeOnly : 1; // Indicate the env only for musa or not.
            uint32_t disableHwr  : 1; // 1: disable HWR
        } field;
        uint32_t value;
    } flags;

    gpu_va flStateAddrsSCG[MTFW_SCG_PB_COUNT];
    gpu_va flStateAddrsMCG2[MTFW_MCG2_PB_COUNT];
    gpu_va flStateAddrsMCG3[MTFW_MCG3_PB_COUNT];
    gpu_va flStateAddrsMCG4[MTFW_MCG4_PB_COUNT];
    gpu_va flStateAddrsMCG5[MTFW_MCG5_PB_COUNT];
    gpu_va flStateAddrsMCG6[MTFW_MCG6_PB_COUNT];
    gpu_va flStateAddrsMCG7[MTFW_MCG7_PB_COUNT];
    gpu_va flStateAddrsMCG8[MTFW_MCG8_PB_COUNT];

    uint64_t yuvCoefficientAddr;
    uint64_t pdsHeapBaseAddr;
    uint64_t uscHeapBaseAddr;
    uint64_t fbcdcHeapBaseAddr;
    uint64_t fbcdcLargeHeapBaseAddr;
    uint64_t texStateHeapBaseAddr;
    uint64_t componentCtrlHeapBaseAddr;
    uint32_t dmKillShaderOffset; /* Dm kill shader offset, base on USC heap  */
} MTFW_FWIF_CONFIG_GEN1;

#ifndef MTFW_FWIF_CONFIG
#define MTFW_FWIF_CONFIG MTFW_FWIF_CONFIG_GEN1
#endif

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_GEN_1_H__ */
