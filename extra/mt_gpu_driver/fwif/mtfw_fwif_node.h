/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: This is firmware Node data struct definitions.
 *
 */
#ifndef __MTFW_FWIF_NODE_H__
#define __MTFW_FWIF_NODE_H__

#include "mtfw_fwif_types.h"
#include "mtfw_fwif_version.h"
#include "mtfw_fwif_submission.h"
#include "mtfw_fwif_perf.h"
#include "mtfw_fwif_log.h"
#include "mtfw_fwif_pcs.h"
#include "mtfw_fwif_utilization.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
    MTFW_NODE_TYPE_GP = 0,
    MTFW_NODE_TYPE_TDM,
    MTFW_NODE_TYPE_UQ,
    MTFW_NODE_TYPE_CDM,
    MTFW_NODE_TYPE_CE,
    MTFW_NODE_TYPE_KMD_CE,
    MTFW_NODE_TYPE_MAX_NUM,
} MTFW_FWIF_NODE_TYPE;

typedef struct MTFW_ALIGN_BYTES_8
{
    gpu_pa   pageTableRootAddr;
    uint64_t pageTableGeneration;

    gpu_va   submission; /* This is submission description address used to In-Submission schedule. */
    uint32_t submissionSize;
    uint32_t submissionQueueId;
} MTFW_NODE_CMD_SUBMISSION;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint32_t wrOffset;
} MTFW_NODE_CTRL_PREEMPTION;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint32_t osid;
    uint64_t qosLimit;

    union
    {
        uint32_t scheTime;   /* This is the time to reset vm schedule info */
        uint64_t sysMemSize; /* This is the system memory size to qos */
    } scheInfo;
} MTFW_NODE_CMD_VM_SCH_INFO;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint64_t haltReason; /* haltReason reserve for host. This value fill into MTFW_FWIF_DUMP_INFO_DESC.hostReason */
} MTFW_FWIF_FIRMWARE_HALT;

typedef union MTFW_ALIGN_BYTES_8 _MTFW_NODE_ITEM_PARA_
{
    MTFW_NODE_CMD_SUBMISSION  submissionData; /* for GPU submission info. */
    MTFW_NODE_CTRL_PREEMPTION preemptionData; /* for Preemption CMD. */

    MTFW_FWIF_TRACE_LOG_CONFIG logCfg;      /* for Log group config. */
    MTFW_FWIF_PERF_REQUEST     perfRequest; /* for PERF config. */

    MTFW_FWIF_GPU_UTIL_CONFIG gpuUtilCfg; /* for gpu util config. */

    MTFW_NODE_CMD_VM_SCH_INFO vmInfo; /* for vm schedule info */

    MTFW_FWIF_FIRMWARE_HALT haltCfg;

    MTFW_FWIF_PC_SAMPLER_CFG pcSamplerCfg;  /* for PC sampling configuration */
} MTFW_NODE_ITEM_PARA;

/**
 * Note: The MTFW_NODE_ITEM size suggest cache line align.
 */
typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_FWIF_HEAD flags;

    uint32_t priority; /* 0 is the lowest priority */

    MTFW_SUBMISSION_ATTRIBUTE attribute;

    MTFW_SUBMISSION_CMD_TYPE submissionCmd;

    uint64_t MTFW_ALIGN_BYTES_8 ccbItemToken;

    uint64_t hostPrivateData;

    MTFW_NODE_ITEM_PARA itemPara;

    uint32_t sequenceNums; /* As same as WDDM fence. */

    /* Extern information for debug and perf trace */
    uint32_t processId; /* This is OS process Id */
} MTFW_NODE_ITEM;

typedef enum
{
    MTFW_NODE_RESP_DONE = 0, /* Subimssion all finish */

    MTFW_NODE_RESP_CANCEL,        /* Submission cancel, none kick did. */
    MTFW_NODE_RESP_BREAK,         /* Submission break and partial kick finish, can NOT resume */
    MTFW_NODE_RESP_PARTIAL,       /* Submission partial kick finish, can resume, need MTFW_SUBMISSION_CTX_BUFFER */
    MTFW_NODE_RESP_STALL,         /* Submission break because STALL */
    MTFW_NODE_RESP_ENGINE_SYNC,   /* Submission break becasse EngineSync */
    MTFW_NODE_RESP_PREEMPTION,    /* Preemption CMD finish */
    MTFW_NODE_RESP_QSEMP_TIMEOUT, /* Submission queue semaphore wait timeout. */
    MTFW_NODE_RESP_STALL_TIMEOUT, /* Submission break because STALL timeout*/
    MTFW_NODE_RESP_ENGINE_SYNC_TIMEOUT, /* Submission break because Enginesync timeout */

    /* Follow enum are event from FW to FEC/host */
    MTFW_NODE_EVENT_START = 0x50,
    MTFW_NODE_EVENT_PERF  = MTFW_NODE_EVENT_START,

    /*Follow response value for error handle */
    MTFW_NODE_RESP_ERROR_HANDLE_START = 0x100,
    MTFW_NODE_RESP_INVALID = MTFW_NODE_RESP_ERROR_HANDLE_START, /* Submission cmd or any parameters INVALID */
    MTFW_NODE_RESP_FAULT, /* Submission FAULT and META Halt, fill MTFW_FWIF_FAULT_INFO */
    MTFW_NODE_RESP_ERROR, /* Submission ERROR, submission abort, fill MTFW_FWIF_ERROR_INFO */

    /* Here is RESP end */
    MTFW_NODE_RESP_VALUE_INVALID,
} MTFW_NODE_RESP_VALUE;

typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_FWIF_HEAD flags;

    MTFW_NODE_RESP_VALUE respValue;

    uint32_t sequenceNums; /* This is fenceId in WDDM scenario */

    uint32_t runTime;
    uint64_t startTimestamp;
} MTFW_NODE_RESP;

#define MTFW_NODE_QUEUE_LENGTH 64

typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_NODE_ITEM reqNodeQ[MTFW_PRIORITY_LEVEL_NUMS][MTFW_NODE_QUEUE_LENGTH];
    MTFW_NODE_RESP respNodeQ[MTFW_NODE_QUEUE_LENGTH];

    MTFW_RING_CTRL reqNodeCtrl[MTFW_PRIORITY_LEVEL_NUMS];
    MTFW_RING_CTRL respNodeCtrl;
} MTFW_NODE_QUEUE, MTFW_LEVEL_QUEUE;

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_NODE_H__ */
