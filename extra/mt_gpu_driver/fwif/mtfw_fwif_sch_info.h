/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: Firmware schedule info for error or CSW.
 *
 */
#ifndef __MTFW_FWIF_SCH_INFO_H__
#define __MTFW_FWIF_SCH_INFO_H__

#include "mtfw_fwif_types.h"
#include "mtfw_fwif_submission.h"
#include "mtfw_fwif_node.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct MTFW_ALIGN_BYTES_8
{
    union
    {
        struct
        {
            uint32_t validFlag        : 1;
            uint32_t validRunningMask : 8;
        } field;
        uint32_t value;
    } flags;
    uint32_t running[8]; /* cmdIdx kick to DM */

    uint32_t kicked;   /* Last cmdIdx kick into ready queue */
    uint32_t finished; /* Last finished cmdIdx */
} MTFW_FW_SCH_KICK_BRIEF;

/* Each Submission has a MTFW_FWIF_SCH_BRIEF to store schedule info */
typedef struct MTFW_ALIGN_BYTES_8
{
    uint64_t ccbItemToken;
    gpu_pa   pageTableRootAddr;

    uint32_t processId;

    MTFW_SUBMISSION_CMD_TYPE submissionCmd;

    /* Each kick type use one schCounter. */
    MTFW_FW_SCH_KICK_BRIEF schCounter[MTFW_SUB_REGION_MAX_KICK_TYPE + 1];
} MTFW_FWIF_SCH_CTX;

typedef union MTFW_ALIGN_BYTES_8
{
    MTFW_FWIF_SCH_CTX schCtx;

    uint8_t schBuffer[4096];
} MTFW_FW_SCH_BUFFER;

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_SCH_DUMP_H__ */
