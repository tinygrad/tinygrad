/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: GPU firmware CCB data struct definition.
 *
 */
#ifndef __MTFW_FWIF_CCB_H__
#define __MTFW_FWIF_CCB_H__

#include "mtfw_fwif_types.h"
#include "mtfw_fwif_version.h"
#include "mtfw_fwif_submission.h"
#include "mtfw_fwif_node.h"
#include "mtfw_fwif_perf.h"
#include "mtfw_fwif_pcs.h"
#include "mtfw_fwif_log.h"
#include "mtfw_fwif_utilization.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
    MTFW_CCB_TYPE_GP = 0,
    MTFW_CCB_TYPE_UQ,
    MTFW_CCB_TYPE_TDM,
    MTFW_CCB_TYPE_CDM,
    MTFW_CCB_TYPE_CE,
    MTFW_CCB_TYPE_MAX_NUM,
} MTFW_FWIF_CCB_TYPE;

/**
 * The MTFW_CCB_ITEM size suggest cache line align.
 */

#define MTFW_DEPEND_Q_SEM_NUMS 32
#define MTFW_FINISH_Q_SEM_NUMS 8

typedef struct MTFW_ALIGN_BYTES_8
{
    /* Firmware enqueue submission queue depend pageTableRootAddr and submissionQueueId */
    gpu_pa   pageTableRootAddr;
    uint64_t pageTableGeneration;
    uint32_t submissionQueueId;

    union
    {
        struct
        {
            uint32_t qSempOutofOrder : 1; /* 0: This submission will block all others follow it. */
            uint32_t withDoorbell    : 1; /* doorbellId is valid when withDoorbell is 1 */
        } field;
        uint32_t value;
    } flags;

    uint64_t dependQSemValidMask;
    uint64_t finishQSemValidMask;

    MTFW_Q_SEMAPHORE dependQSem[MTFW_DEPEND_Q_SEM_NUMS];
    MTFW_Q_SEMAPHORE finishQSem[MTFW_FINISH_Q_SEM_NUMS];

    gpu_va   submissionAddr; /* This is submission description address used to In-Submission schedule. */
    uint32_t submissionSize;

    uint32_t doorbellId;
} MTFW_SUBMISSION_PARA;

typedef struct MTFW_ALIGN_BYTES_8
{
    gpu_pa   pageTableRootAddr;
    uint64_t pageTableGeneration;

    uint64_t           ccbItemToken;
    MTFW_FWIF_CCB_TYPE ccbType;
    uint32_t           submissionQueueId;
} MTFW_SUBMISSION_SCHEDULE_PARA;

typedef struct MTFW_ALIGN_BYTES_8
{
    gpu_pa   pageTableRootAddr;
    uint64_t pageTableGeneration;

    MTFW_Q_SEMAPHORE writeQSem;
} MTFW_SUBMISSION_Q_SEM_PARA;

typedef struct MTFW_ALIGN_BYTES_8
{
    gpu_pa   pageTableRootAddr;
    uint64_t pageTableGeneration;

    uint64_t dependQSemValidMask;
    uint64_t finishQSemValidMask;

    MTFW_Q_SEMAPHORE dependQSem[MTFW_DEPEND_Q_SEM_NUMS];
    MTFW_Q_SEMAPHORE finishQSem[MTFW_FINISH_Q_SEM_NUMS];

    /* Firmware will invald all TLB if invaidAddr or invalidSize is 0 */
    gpu_va   invalidAddr; /* invalid address MUST 4k bytes align. */
    uint64_t invalidSize;

} MTFW_SUBMISSION_INVALID_TLB;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint64_t qosLimit;
    union
    {
        uint64_t sysMemSize; /* This is the system memory size to qos */
        uint32_t scheTime;   /* This is the time to reset vm schedule info */
    } scheInfo;

    uint32_t osid;
} MTFW_FWIF_VM_SCH_INFO;

typedef union MTFW_ALIGN_BYTES_8 _MTFW_CCB_ITEM_PARA_
{
    /* GPU submission parameters. */
    MTFW_SUBMISSION_PARA submissionPara;

    /* MTFW_SUBMISSION_CMD_INIT_Q_SEMAPHORE, MTFW_SUBMISSION_CMD_WRITE_Q_SEMAPHORE */
    MTFW_SUBMISSION_Q_SEM_PARA qSempPara;

    /* MTFW_SUBMISSION_CMD_CONTINUE, MTFW_SUBMISSION_CMD_CANCEL */
    MTFW_SUBMISSION_SCHEDULE_PARA schPara;

    /* Follow para for control cmd */
    MTFW_FWIF_TRACE_LOG_CONFIG logCfg;      /* for Log group config. */
    MTFW_FWIF_PERF_REQUEST     perfRequest; /* for PERF config. */
    MTFW_FWIF_GPU_UTIL_CONFIG  gpuUtilCfg;  /* for gpu util config. */

    MTFW_FWIF_VM_SCH_INFO vmInfo; /* for vm schedule info */

    MTFW_FWIF_FIRMWARE_HALT haltCfg;

    MTFW_SUBMISSION_INVALID_TLB invalidTlbCfg;

    MTFW_FWIF_PC_SAMPLER_CFG pcSamplerCfg;  /* for PC sampling configuration */
} MTFW_CCB_ITEM_PARA;

typedef union MTFW_ALIGN_BYTES_8
{
    struct
    {
        MTFW_SUBMISSION_STATE submState;
    } field;
    uint64_t buffer[8]; /* Reserve for FW */
} MTFW_FW_CONTEXT;

typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_FWIF_HEAD flags;

    MTFW_SUBMISSION_ATTRIBUTE attribute;

    MTFW_SUBMISSION_CMD_TYPE submissionCmd;

    uint32_t priority; /* 0 is the lowest priority */

    uint64_t ccbItemToken;
    uint64_t hostPrivateData;

    uint64_t frameId;
    uint64_t cbId;      /* Cmd Buffer id */
    uint32_t processId; /* This is OS process Id */

    MTFW_CCB_ITEM_PARA ccbItemPara;

    MTFW_FW_CONTEXT fwBuffer; /* Reserve for FW private data */
} MTFW_CCB_ITEM;

typedef enum
{
    MTFW_CCB_RESP_DONE = 0,

    MTFW_CCB_RESP_CANCEL        = MTFW_NODE_RESP_CANCEL,
    MTFW_CCB_RESP_BREAK         = MTFW_NODE_RESP_BREAK,
    MTFW_CCB_RESP_PARTIAL       = MTFW_NODE_RESP_PARTIAL,
    MTFW_CCB_RESP_QSEMP_TIMEOUT = MTFW_NODE_RESP_QSEMP_TIMEOUT,
    MTFW_CCB_RESP_STALL_TIMEOUT       = MTFW_NODE_RESP_STALL_TIMEOUT,
    MTFW_CCB_RESP_ENGINE_SYNC_TIMEOUT = MTFW_NODE_RESP_ENGINE_SYNC_TIMEOUT,

    /* Follow enum are event from FW to host */
    MTFW_CCB_EVENT_START = MTFW_NODE_EVENT_START,
    MTFW_CCB_EVENT_PERF  = MTFW_CCB_EVENT_START,

    /* Follow response value for error handle */
    MTFW_CCB_RESP_ERROR_HANDLE_START = MTFW_NODE_RESP_ERROR_HANDLE_START,
    MTFW_CCB_RESP_INVALID            = MTFW_NODE_RESP_INVALID, /* Submission cmd or any parameters INVALID */
    MTFW_CCB_RESP_FAULT              = MTFW_NODE_RESP_FAULT,   /* FW fault and HWR buffer has detail info. */
    MTFW_CCB_RESP_ERROR              = MTFW_NODE_RESP_ERROR,   /* FW got some ERROR */

    /* Here is RESP end */
    MTFW_CCB_RESP_VALUE_INVALID = MTFW_NODE_RESP_VALUE_INVALID,
} MTFW_CCB_RESP_VALUE;

typedef union
{
    uint64_t extInfo;
} MTFW_CCB_RESP_EXT_INFO;

typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_FWIF_HEAD flags;

    MTFW_CCB_RESP_VALUE respValue;

    uint64_t ccbItemToken;
    uint64_t hostPrivateData;

    uint64_t startTimestamp;
    uint32_t runTime;

    MTFW_CCB_RESP_EXT_INFO extInfo;
} MTFW_CCB_RESP;

#define MTFW_CCB_QUEUE_LENGTH 256

typedef struct MTFW_ALIGN_BYTES_8
{
    fw_va ccbItemAddr; /* Point to MTFW_CCB_ITEM object */
} MTFW_CCB_Q_CELL;

/**
 * MTFW_CCB_QUEUE start address suggest CacheLine align.
 */
typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_CCB_Q_CELL reqCcbQ[MTFW_PRIORITY_LEVEL_NUMS][MTFW_CCB_QUEUE_LENGTH];
    MTFW_CCB_RESP   respCcbQ[MTFW_CCB_QUEUE_LENGTH];

    MTFW_RING_CTRL reqCcbCtrl[MTFW_PRIORITY_LEVEL_NUMS];
    MTFW_RING_CTRL respCcbCtrl;
} MTFW_CCB_QUEUE;

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_CCB_H__ */
