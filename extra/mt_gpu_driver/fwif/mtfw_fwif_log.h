/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: MTFW log and assert data struct.
 *
 */
#ifndef MTFW_FWIF_LOG_H
#define MTFW_FWIF_LOG_H

#include "mtfw_fwif_types.h"

// Available log groups
#define MTFW_LOG_GROUP_NAME    \
    X(MTFW_GROUP_NULL, NULL)   \
    X(MTFW_GROUP_ERR, ERR)     \
    X(MTFW_GROUP_DBG, DBG)     \
    X(MTFW_GROUP_FUNC, FUNC)   \
    X(MTFW_GROUP_REG, REG)     \
    X(MTFW_GROUP_MMU, MMU)     \
    X(MTFW_GROUP_INT, INT)     \
    X(MTFW_GROUP_CORE, CORE)   \
    X(MTFW_GROUP_DRV, DRV)     \
    X(MTFW_GROUP_SCH, SCH)     \
    X(MTFW_GROUP_NODE, NODE)   \
    X(MTFW_GROUP_WORK, WORK)   \
    X(MTFW_GROUP_GPU, GPU)     \
    X(MTFW_GROUP_PERF, PERF)   \
    X(MTFW_GROUP_PFM, PFM)     \
    X(MTFW_GROUP_POW, POW)     \
    X(MTFW_GROUP_CSW, CSW)     \
    X(MTFW_GROUP_CCB, CCB)     \
    X(MTFW_GROUP_CTX, CTX)     \
    X(MTFW_GROUP_SUBM, SUBM)   \
    X(MTFW_GROUP_FSCH, FSCH)   \
    X(MTFW_GROUP_MAIN, MAIN)   \
    X(MTFW_GROUP_VGPU, VGPU)   \
    X(MTFW_GROUP_QSEMP, QSEMP) \
    X(MTFW_GROUP_RSV24, RSV24) \
    X(MTFW_GROUP_RSV25, RSV25) \
    X(MTFW_GROUP_RSV26, RSV26) \
    X(MTFW_GROUP_RSV27, RSV27) \
    X(MTFW_GROUP_RSV28, RSV28) \
    X(MTFW_GROUP_RSV29, RSV29) \
    X(MTFW_GROUP_RSV30, RSV30) \
    X(MTFW_GROUP_RSV31, RSV31)
/* NOTE: LOG GROUP NUM MUST <= 32 */
enum MTFW_LOG_SFGROUPS
{
#define X(A, B) A,
    MTFW_LOG_GROUP_NAME
#undef X
        MTFW_LOG_NUMBER_OF_GROUPS
};

/* Trace Buffer */
#define MTFW_FWIF_TRACE_BUFFER_SIZE        24000
#define MTFW_FWIF_TRACE_BUFFER_ASSERT_SIZE 200
#define MTFW_FWIF_THREAD_NUM               1
#define MTFW_FWIF_ASSERT_EXTINFO_SIZE      256

typedef struct MTFW_ALIGN_BYTES_8
{
    char     path[MTFW_FWIF_TRACE_BUFFER_ASSERT_SIZE];
    char     info[MTFW_FWIF_TRACE_BUFFER_ASSERT_SIZE];
    uint32_t lineNum;
} MTFW_FWIF_ASSERTBUF;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint32_t            tracePointer;
    bool                bWrap;
    uint32_t            traceBuffer[MTFW_FWIF_TRACE_BUFFER_SIZE];
    MTFW_FWIF_ASSERTBUF assertBuf;
    uint32_t            assertEXTInfo[MTFW_FWIF_ASSERT_EXTINFO_SIZE];
} MTFW_FWIF_TRACEBUF_SPACE;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint32_t                 logGroups[MTFW_LOG_NUMBER_OF_GROUPS];
    MTFW_FWIF_TRACEBUF_SPACE traceBuf[MTFW_FWIF_THREAD_NUM];
} MTFW_FWIF_TRACEBUF;

typedef struct MTFW_ALIGN_BYTES_8
{
    fw_va    bufferAddr;     // Ptr to Signature Buffer memory
    uint32_t leftSizeInRegs; // Amount of space left for storing regs in the buffer
} MTFW_FWIF_SIGBUF_CTL;

/***********************************************************************************************************/

typedef struct MTFW_ALIGN_BYTES_8
{
    uint32_t logGroup; // log groups
    uint8_t  fwIndex;
} MTFW_FWIF_TRACE_LOG_CONFIG;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint64_t baseAddr;    /* address of log entries section */
    uint64_t dataLength;  /* amount of bytes following this header */
    uint64_t dataOffset;  /* offset to first entry in this file */
} MTFW_FWIF_LOG_HEADER;

typedef struct
{
    uint32_t groups;
    uint32_t paramsNum;
    uint32_t lineIdx;
    uint32_t fileNameLen;
    uint32_t textLen;
} MTFW_FWIF_LOG_ENTRY;

#endif /* RGX_FWIF_LOG_H */
