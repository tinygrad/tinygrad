/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: This is Submission common definitions.
 *
 */
#ifndef __MTFW_FWIF_SUBMISSION_H__
#define __MTFW_FWIF_SUBMISSION_H__

#include "mtfw_fwif_types.h"
#include "mtfw_fwif_version.h"
#include "mtfw_fwif_hwcontext.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define MTFW_SUBM_RETRY_DEFAULT_COUNT 3       /* Submission idle timeout retry count. */
#define MTFW_KICK_TIMEOUT_DEFAULT_SEC 200     /* Kick run on GPU DM total time. Unit is second. */
#define MTFW_KICK_RETRY_TIMEOUT_US    1000000 /* Kick retry timeout when STALL/EngineSync ...  */

/**
 * Q Semaphore is uint64 value, and Q-Semaphore addr MUST 64bit align.
 * Q-Semaphore address low 3bits is Q-Semaphore control.
 * bit[0:1] : Q-Semaphore type.
 * bit[2]   : shadow or set zero.
 */
#define MTFW_Q_SEM_ATTR_MASK    (7ULL) /* Q Semaphore addr MUST 64bit align. */
#define MTFW_Q_SEM_ADDRESS_MASK (~(MTFW_Q_SEM_ATTR_MASK))

#define MTFW_Q_SEM_ATTR_WITH_SHADOW (1ULL << 2) /* Finis Q-Semaphore will write shadow address. */

#define MTFW_Q_SEM_ATTR_ERR_DIFFISON (1ULL << 1) /* Dependent Q-Semaphore will diffuse error. */
#define MTFW_Q_SEM_ATTR_SET_ZERO     (1ULL << 2) /* Dependent Q-Semaphore will write zero to semaphore address. */

/* 2bits Q-Semaphore type, MTFW_Q_SEM_FINISH_TYPE or MTFW_Q_SEM_DEPENDENT_TYPE */
#define MTFW_Q_SEM_TYPE_MASK (3ULL)

typedef enum
{
    MTFW_Q_SEM_FINISH_TYPE_INC    = 0, /* This finish Q-Semaphore value is continue incease. */
    MTFW_Q_SEM_FINISH_TYPE_BINARY = 1, /* This finish Q-Semaphore value is MTFW_Q_SEMAPHORE.value */
    MTFW_Q_SEM_FINISH_TYPE_SIGNAL = 2, /* This finish Q-Semaphore value is 1 ONLY */
    MTFW_Q_SEM_FINISH_TYPE_EXTEND = 3, /* Depend on MTFW_Q_SEMAPHORE.attribute */
} MTFW_Q_SEM_FINISH_TYPE;

typedef enum
{
    MTFW_Q_SEM_DEPENDENT_TYPE_NORMAL = 0, /* Q-Semaphore depend on value is MTFW_Q_SEMAPHORE.value */
    MTFW_Q_SEM_DEPENDENT_TYPE_SIGNAL = 1, /* Q-Seamphore depend on value more than 0 */
} MTFW_Q_SEM_DEPENDENT_TYPE;

#define MTFW_Q_SEM_VALUE_INVALID 0x8000000000000000ULL /* Q-Semaphore is INVALID if submission is unsuccessful. */

#define MTFW_Q_SEM_ARRAY_SIZE 3

typedef struct MTFW_ALIGN_BYTES_8
{
    /**
     * Point to uint64_t[MTFW_Q_SEM_ARRAY_SIZE] array. addr MUST 8bytes align.
     * addr[0] is Queue Semaphore current value.
     * Others reserve for firmware.
     *
     * shadowAddr is system memory for FW sync semaphore value to host when it is NOT null.
     */
    volatile union
    {
        fw_va    addr;
        uint32_t addr32;
    };

    volatile union
    {
        fw_va    shadowAddr;
        uint32_t shadowAddr32;
    };

    union
    {
        uint64_t value;
        uint32_t value32;
    };

    union
    {
        uint64_t revert;
    } attribute;
} MTFW_Q_SEMAPHORE;

typedef enum
{
    MTFW_SUBMISSION_STATE_INIT = 0,
    MTFW_SUBMISSION_STATE_BLOCK,    /* Wait Queue Semaphore */
    MTFW_SUBMISSION_STATE_READY,    /* Wait submit to In-Submission schedule */
    MTFW_SUBMISSION_STATE_RUNNING,  /* Submit into In-Submissiont schedule */
    MTFW_SUBMISSION_STATE_SUSPEND,  /* STALL and wait CONTINUE cmd */
    MTFW_SUBMISSION_STATE_FINISHED, /* Finish from In-Submission schedule */
    MTFW_SUBMISSION_STATE_NUM,
    MTFW_SUBMISSION_STATE_INVALID = MTFW_SUBMISSION_STATE_NUM
} MTFW_SUBMISSION_STATE;

/**
 * Submission Description
 *
 * MTFW_SUBMISSION_HEAD
 * MTFW_SUBMISSION_REGION_DESC [n]
 * Submission Region [m] There are different kick array.
 *
 * Note:
 * 1. Submission Description start address and size MUST 32bytes align for DMA copy.
 * 2. MTFW_SUBMISSION_REGION_DESC number MUST more or equel Submission Region numbers.
 *
 */

typedef enum
{
    MTFW_SUBMISSION_CMD_NOP = 0,
    MTFW_SUBMISSION_CMD_CONTINUE,           /* Parameter is MTFW_SUBMISSION_SCHEDULE_PARA */
    MTFW_SUBMISSION_CMD_CANCEL,             /* Parameter is MTFW_SUBMISSION_SCHEDULE_PARA */
    MTFW_SUBMISSION_CMD_TERMINATED,         /* Parameter is MTFW_SUBMISSION_SCHEDULE_PARA */
    MTFW_SUBMISSION_CMD_CANCEL_SUBM_Q,      /* Parameter is MTFW_SUBMISSION_SCHEDULE_PARA */
    MTFW_SUBMISSION_CMD_TERMINATED_SUBM_Q,  /* Parameter is MTFW_SUBMISSION_SCHEDULE_PARA */
    MTFW_SUBMISSION_CMD_INIT_Q_SEMAPHORE,   /* Parameter is MTFW_SUBMISSION_Q_SEM_PARA */
    MTFW_SUBMISSION_CMD_WRITE_Q_SEMAPHORE,  /* Parameter is MTFW_SUBMISSION_Q_SEM_PARA */
    MTFW_SUBMISSION_CMD_SCH_RESUME,         /* Resume FW schedule, parameter is void */
    MTFW_SUBMISSION_CMD_SCH_SKIP,           /* Parameter is MTFW_SUBMISSION_PARA */
    MTFW_SUBMISSION_CMD_SET_LOG_GROUP,      /* Parameter is MTFW_FWIF_TRACE_LOG_CONFIG */
    MTFW_SUBMISSION_CMD_SET_LOG_GROUP_FEC,  /* Parameter is MTFW_FWIF_TRACE_LOG_CONFIG */
    MTFW_SUBMISSION_CMD_SET_PERF_REQUESET,  /* Parameter is MTFW_FWIF_PERF_REQUEST */
    MTFW_SUBMISSION_CMD_FLUSH_MMUCACHE,     /* Parameter is MTFW_SUBMISSION_INVALID_TLB */
    MTFW_SUBMISSION_CMD_FLUSH_ALL_MMUCACHE, /* Parameter is void */
    MTFW_SUBMISSION_CMD_SET_GPU_UTIL,       /* Parameter is MTFW_FWIF_GPU_UTIL_CONFIG */
    MTFW_SUBMISSION_CMD_SET_HALT,           /* Parameter is MTFW_FWIF_FIRMWARE_HALT */
    MTFW_SUBMISSION_CMD_SCH_ABORT,          /* Parameter is MTFW_SUBMISSION_PARA */
    MTFW_SUBMISSION_CMD_TRIG_PC_SAMPLER,    /* Parameter is MTFW_FWIF_PC_SAMPLER_CFG */

    /* Only used for VGPU */
    MTFW_SUBMISSION_CMD_VM_INIT_CHECK = 70, /* Parameter is MTFW_NODE_CMD_VM_SCH_INFO */
    MTFW_SUBMISSION_CMD_VM_OFFLINE,         /* Parameter is MTFW_NODE_CMD_VM_SCH_INFO */
    MTFW_SUBMISSION_CMD_VM_CREATE,          /* Parameter is MTFW_NODE_CMD_VM_SCH_INFO */
    MTFW_SUBMISSION_CMD_VM_SCHEDULE_UPDATE, /* Parameter is MTFW_NODE_CMD_VM_SCH_INFO */
    MTFW_SUBMISSION_CMD_VM_QOS_UPDATE,      /* Parameter is MTFW_NODE_CMD_VM_SCH_INFO */
    MTFW_SUBMISSION_CMD_VM_REONLINE,        /* Parameter is MTFW_NODE_CMD_VM_SCH_INFO */

    /* Submission before MTFW_SUBMISSION_CMD_GPU_START excute on FEC. */
    MTFW_SUBMISSION_CMD_GPU_START = 100,
    MTFW_SUBMISSION_CMD_GPU_NOP   = MTFW_SUBMISSION_CMD_GPU_START,
    MTFW_SUBMISSION_CMD_GPU_PREEMPTION, /* Drop all submission in this NODE */
    MTFW_SUBMISSION_CMD_GPU_UQ,
    MTFW_SUBMISSION_CMD_GPU_TDM,
    MTFW_SUBMISSION_CMD_GPU_CDM,
    MTFW_SUBMISSION_CMD_GPU_CE,
    MTFW_SUBMISSION_CMD_GPU_CDM_STREAM,
    MTFW_SUBMISSION_CMD_GPU_GFX,

    /* Invalid from MTFW_SUBMISSION_CMD_MAX_NUMS */
    MTFW_SUBMISSION_CMD_MAX_NUMS
} MTFW_SUBMISSION_CMD_TYPE;

/* Submission name for version control. */
#define MTFW_SUBMISSION_NAME_NOP        "S-NOP"
#define MTFW_SUBMISSION_NAME_PREEMPTION "S-PREE"
#define MTFW_SUBMISSION_NAME_UQ         "S-UQ"
#define MTFW_SUBMISSION_NAME_TDM        "S-TDM"
#define MTFW_SUBMISSION_NAME_CDM        "S-CDM"
#define MTFW_SUBMISSION_NAME_CE         "S-CE"
#define MTFW_SUBMISSION_NAME_CDM_STREAM "S-CDMS"
#define MTFW_SUBMISSION_NAME_GFX        "S-GFX"

/**
 * GPU support submission CMD number
 */
#define MTFW_SUBMISSION_GPU_CMD_NUMS (MTFW_SUBMISSION_CMD_MAX_NUMS - MTFW_SUBMISSION_CMD_GPU_START)

typedef enum
{
    MTFW_SUB_REGION_START = 0,
    MTFW_SUB_REGION_NOP,
    MTFW_SUB_REGION_CDM,
    MTFW_SUB_REGION_TDM,
    MTFW_SUB_REGION_CE,
    MTFW_SUB_REGION_GFX,
    MTFW_SUB_REGION_SPR,
    /* Region kick enum befor MTFW_SUB_REGION_KICK_MAX_TYPE are GPU kick type. */
    MTFW_SUB_REGION_MAX_KICK_TYPE,

    MTFW_SUB_REGION_K_SEM_BUFFER,       /* Kick Semaphore buffer */
    MTFW_SUB_REGION_CSW_BUFFER,         /* This is a META va point MTFW_SUBMISSION_CTX_BUFFER object */
    MTFW_SUB_REGION_ERROR_BUFFER,       /* This is a META va point MTFW_FWIF_ERROR_INFO object */
    MTFW_SUB_REGION_TRACE_INFO,         /* Point to a MTFW_SUBMISSION_TRACE object */
    MTFW_SUB_REGION_PFM_CFG,            /* Point to a MTFW_FWIF_PFM_SETTINGS object */
    MTFW_SUB_REGION_GEOM_IDX_FETCH_CFG, /* Point to a MTFW_FWIF_GEOM_IDX_FETCH_CFG object */
    MTFW_SUB_REGION_SCHEDULE_TABLE,     /* Point to a MTFW_KICK_SCHEDULE_TABLE object, use
                                           MTFW_SUBMISSION_REGION_DESC.regionStart.value*/
    MTFW_SUB_REGION_DBG_CSW_CFG,        /* Point to a MTFW_FWIF_CSW_VALIDATION_CFG */

    MTFW_SUB_REGION_SPACE,

    /* Follow enum for program */
    MTFW_SUB_REGION_NUMS,
    MTFW_SUB_REGION_INVALID = MTFW_SUB_REGION_NUMS,
} MTFW_SUBMISSION_REGION_TYPE;

/* Region name for version control */
#define MTFW_KICK_NAME_NOP   "R-NOP"
#define MTFW_KICK_NAME_CDM   "R-CDM"
#define MTFW_KICK_NAME_TDM   "R-TDM"
#define MTFW_KICK_NAME_CE    "R-CE"
#define MTFW_KICK_NAME_GFX   "R-GFX"
#define MTFW_KICK_NAME_SPR   "R-SPR"
#define MTFW_KICK_NAME_SPACE "R-SPACE"
#define MTFW_KICK_NAME_PFM   "R-PFM"
// #define MTFW_KICK_NAME_PCS "R-PCS"

typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_FWIF_HEAD flags;

    /**
     * CmdFlags is per submission. Same cfg in MTFW_KICK_HEAD will override submission value.
     */
    union
    {
        struct
        {
            uint32_t rsv : 1;
        } field;
        uint32_t value;
    } cmdFlags;

    gpu_va timestampAddr; /* Point a UINT64 buffer for submission end timestamp */
    gpu_va fwCswBufer;    /* Point to MTFW_SUBMISSION_CTX_BUFFER object for submission context switch. */

    uint32_t priority; /* 0 is the lowest priority */

    MTFW_SUBMISSION_CMD_TYPE submissionCmd;

    uint32_t submissionQueueId;

    uint32_t retryCount; /* Submission retry count, 0 is default value MTFW_SUBM_RETRY_DEFAULT_COUNT  */

    /**
     * MTFW_SUBMISSION_REGION_DESC array start offset base from this submission description start.
     */
    uint32_t subRegionDescStartOffset;
    uint32_t subRegionDescNums;

    /* Extern information */
    uint64_t MTFW_ALIGN_BYTES_8 frameId;
    uint64_t                    cbId;      /* Cmd Buffer id */
    uint32_t                    processId; /* This is OS process Id */

    uint32_t reserve[4];
} MTFW_SUBMISSION_HEAD;

typedef union
{
    struct
    {
        uint32_t errorPause : 1; /* FW schedule pause if this submission has ERROR. */
        uint32_t userPause  : 1; /* FW schedule pause when this submission finish. */
        uint32_t reSubmit   : 1; /* This submission resubmit because of csw. */
    } field;
    uint32_t value;
} MTFW_SUBMISSION_ATTRIBUTE;

typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_SUBMISSION_REGION_TYPE regionType;

    /**
     * regionStart is a union. offset or addr dependent regionType.
     */
    union
    {
        uint32_t offset;  /* This offset MUST base on submission description start. */
        gpu_va   gpuAddr; /* point to extend buffer */
        fw_va    fwAddr;
        uint64_t value; /* value for simple config */
    } regionStart;

    uint32_t regionSize;
} MTFW_SUBMISSION_REGION_DESC;

#define MTFW_KICK_SEMAPHORE_MAX_NUMS 8

/* MTFW_KICK_SEMAPHORE_UQ is kick semaphore index for UQ submission. */
typedef enum
{
    KICK_SEMAPHORE_IDX_TDM   = 0x0,
    KICK_SEMAPHORE_IDX_SPACE = KICK_SEMAPHORE_IDX_TDM,
    KICK_SEMAPHORE_IDX_GEOM  = 0x1,
    KICK_SEMAPHORE_IDX_SPR   = KICK_SEMAPHORE_IDX_GEOM,
    KICK_SEMAPHORE_IDX_PIXEL = 0x2,
    KICK_SEMAPHORE_IDX_CDM   = 0x3,
    KICK_SEMAPHORE_IDX_CE    = 0x4,
    KICK_SEMAPHORE_IDX_MAX   = 0x5,
} MTFW_KICK_SEMAPHORE_UQ;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint64_t dependentValue[MTFW_KICK_SEMAPHORE_MAX_NUMS];
    uint64_t finishValue;
    uint8_t  finishValueMask; /* If finishValueMask bit is 1, write finsiValue to memroy[bitIdx],  */
    uint8_t  reserve[3];      /* reserve for uint64_t align. */
} MTFW_KICK_SEMAPHORE;

typedef struct MTFW_ALIGN_BYTES_8
{
    volatile gpu_va addr; /* Point to Kick Event address */
    uint64_t        value;
} MTFW_KICK_EVENT;

typedef struct MTFW_ALIGN_BYTES_8
{
    gpu_va dependKickEventArray; /* Point to MTFW_KICK_EVENT array start address */
    gpu_va finishKickEventArray; /* Point to MTFW_KICK_EVENT array start address */

    uint32_t dependKickEventNum;
    uint32_t finishKickEventNum;

    uint32_t dependKickPollTimeoutSec; /* If this value is 0, timeout time is MTFW_KICK_TIMEOUT_DEFAULT_SEC */
} MTFW_KICK_EVENT_PARA;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint64_t startTimestamp;
    uint32_t runTime;

    union
    {
        struct
        {
            uint32_t rtaNums    : 8; /* How many RTAs in this kick */
            uint32_t oomCounter : 8; /* Partial render times */
        } gfxField;
        uint32_t value;
    } Info;
} MTFW_KICK_TRACE, MTFW_SUBMISSION_TRACE;

typedef struct MTFW_ALIGN_BYTES_8
{
    /* This flag is general flags for every type kick */
    MTFW_FWIF_HEAD flags;

    union
    {
        struct
        {
            uint32_t mmuInvalid : 1;
        } field;
        uint32_t value;
    } kickCtrl;

    uint32_t kickSize; /* UMD fill it to tell FW the size of each kick */

    uint32_t kickTimeoutMs;  /* If this value is 0, timeout time is MTFW_KICK_TIMEOUT_DEFAULT_SEC */
    uint32_t retryTimeoutUs; /* If this value is 0, timeout time is MTFW_KICK_RETRY_TIMEOUT_US */

    MTFW_KICK_SEMAPHORE  kickSemaphore;
    MTFW_KICK_EVENT_PARA kickEvent;

    gpu_va MTFW_ALIGN_BYTES_8 traceAddr;     /* Point to MTFW_KICK_TRACE object. This address MUST align to 8Bytes. */
    gpu_va MTFW_ALIGN_BYTES_8 timeStampAddr; /* Point a UINT64 buffer for kick end timestamp used by query feature */

    uint64_t optionalRegCfg; /* This is optional register config. Referent MTFW_KICK_OPT_REG_CFG_METHORD */

    uint64_t reserve[4];
} MTFW_KICK_HEAD;

typedef union
{
    struct
    {
        uint32_t submissionSerial : 1; /* Submission belong to same context can NOT overlap. */
        uint32_t contextSerial    : 1; /* Submission belong to different context can NOT overlap. */
        uint32_t globalSerial     : 1; /* All submission can NOT overlap. */
        uint32_t submQueueSerial  : 1; /* Submission belong to same submission Queue(CMD Queue) can NOT overlap. */
        uint32_t forceInOrder     : 1; /* Kick MUST in-order */
        uint32_t kickExclusive    : 1; /* All kick exclusive. */
        uint32_t inDmExclusive    : 1; /* Kick exclusive in same DM */

        uint32_t issueNum : 8; /* Howmany kick can kick to DM same time. 0 is default by FW. */
        /**
         * kickMode for different kinds of node has different meanings, use macros prefixed with MTFW_FWIF_SCH_MODE
         */
        uint32_t kickMode : 4;

        uint32_t geomAheadNum : 8; /* Howmany geom can ahead 3D. 0 is default by FW. */

        uint32_t fromKmd : 1; /* Cmd is from kmd*/

        uint32_t kickEvent : 1; /* Submission need to check kick event. */

        uint32_t dmExclusive : 1; /* Submission want to occupy dm alone */
    } field;
    uint32_t value;
    uint32_t reserve[4];
} MTFW_KICK_SCH_CTRL;

/*
 * Defines for the Circular Buffer Read/Write Pointers.
 * 1TB range 32-bit granular Write/Read offsets stored in two adjacent 64bit values.
 */
typedef struct MTFW_ALIGN_BYTES_8
{
    uint64_t writeOffset;
    uint64_t readOffset;
} MTFW_CBUFFER_CTRL;

typedef enum
{
    MTFW_KICK_OPT_REG_CFG_IGNORE = 0ull, /* Do NOT config this register. */
    MTFW_KICK_OPT_REG_CFG_USER   = 1ull, /* Use user value config this register. */
    MTFW_KICK_OPT_REG_CFG_FW     = 2ull, /* FW auto config this register. */
    MTFW_KICK_OPT_REG_CFG_EXTEND = 3ull, /* Reserve for extend. */
} MTFW_KICK_OPT_REG_CFG_METHOD;

/* Every optional register config use 2bits. */
#define MTFW_KICK_OPT_REG_CFG_MASK                   0x3ull
#define MTFW_KICK_OPT_REG_CFG_SET(cfg, cfgIdx)       (((cfg) & MTFW_KICK_OPT_REG_CFG_MASK) << (2 * (cfgIdx)))
#define MTFW_KICK_OPT_REG_CFG_GET(optCfg, cfgIdx)    ((optCfg) >> (2 * (cfgIdx)) & MTFW_KICK_OPT_REG_CFG_MASK)
#define MTFW_KICK_OPT_REG_BIT_IDX_TO_CFG_IDX(bitIdx) ((bitIdx) >> 1)

#define MTFW_SCHEDULE_TABLE_MAX_NUM 8
typedef union MTFW_ALIGN_BYTES_8
{
    uint64_t value;
    // Valid value MTFW_SUBMISSION_REGION_TYPE  from MTFW_SUB_REGION_START   to MTFW_SUB_REGION_MAX_KICK_TYPE,
    // and MTFW_SUB_REGION_START represent finish.
    uint8_t schTable[MTFW_SCHEDULE_TABLE_MAX_NUM];
} MTFW_KICK_SCHEDULE_TABLE;

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_SUBMISSION_H__ */
