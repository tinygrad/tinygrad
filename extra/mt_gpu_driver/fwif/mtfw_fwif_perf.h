/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: MTFW performance event data struct.
 *
 */
#ifndef __MTFW_FWIF_PERF_H__
#define __MTFW_FWIF_PERF_H__

#include "mtfw_fwif_submission.h"

#ifdef __cplusplus
extern "C"
{
#endif

#if !defined(PDUMP)
#define MTFW_PERF_EVENT_MAX (5120)
#else
// change this value if you want to measure performance of large amount of RTs
#define MTFW_PERF_EVENT_MAX (256)
#endif

/***********************************************************************************************************/
#ifndef RGX_HWPERF_EVENT_MASK_VALUE
#define RGX_HWPERF_EVENT_MASK_VALUE(e) (UINT64_C(1) << (e))
#endif
/***********************************************************************************************************/

// Now use uint32 as perf filter bitmask, so NO.0-31 events correspond to NOo.32-63 events.
// For example, BG kick event is 0x01, then BG finish event is 0x21. These two events should be one group.
// One filter bit will control a group rather than an event.
// So, there are 32 group at most. Can extend to 64 groups, if use uint64 as filter bitmask.
#define PERF_EVENT_FINISH_OFFSET 0x20
/***********************************************************************************************************/
typedef enum
{
    // RGX_HWPERF_INVALID = 0x00,
    PERF_REPORTEVENT_RANGE_BEGIN = 0x01, // events checked by ReportEvent()

    // FW types
    PERF_EVENT_FW_BG_KICK    = PERF_REPORTEVENT_RANGE_BEGIN,
    PERF_EVENT_FW_BG_FINISH  = PERF_EVENT_FW_BG_KICK + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_FW_IRQ_KICK   = 0x02,
    PERF_EVENT_FW_IRQ_FINISH = PERF_EVENT_FW_IRQ_KICK + PERF_EVENT_FINISH_OFFSET,

    // FW schedule type
    PERF_EVENT_L1_SCH_START   = 0x03, /* Event data is MTFW_FWIF_PERF_EVENT_SCH_DATA object */
    PERF_EVENT_L1_SCH_END     = PERF_EVENT_L1_SCH_START + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_FAST_SCH_START = 0x04, /* Event data is MTFW_FWIF_PERF_EVENT_SCH_DATA object */
    PERF_EVENT_FAST_SCH_END   = PERF_EVENT_FAST_SCH_START + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_NODE_START     = 0x05,
    PERF_EVENT_NODE_END       = PERF_EVENT_NODE_START + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_WORK_SCH_START = 0x06,
    PERF_EVENT_WORK_SCH_END   = PERF_EVENT_WORK_SCH_START + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_DM_SCH_START   = 0x07,
    PERF_EVENT_DM_SCH_END     = PERF_EVENT_DM_SCH_START + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_DM_KICK_START  = 0x08,
    PERF_EVENT_DM_KICK_END    = PERF_EVENT_DM_KICK_START + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_PREEMPT_REQ    = 0x09,
    PERF_EVENT_PREEMPT_ACK    = PERF_EVENT_PREEMPT_REQ + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_FENCE_NOTIFY   = 0x0a,
    PERF_EVENT_SUBM_SCH_START = 0x0b, /* Event data is MTFW_FWIF_PERF_EVENT_SUBM_DATA */
    PERF_EVENT_SUBM_SCH_END   = PERF_EVENT_SUBM_SCH_START + PERF_EVENT_FINISH_OFFSET,

    // HW types
    PERF_EVENT_PMOOM_TAPAUSE  = 0x10,
    PERF_EVENT_PMOOM_TARESUME = PERF_EVENT_PMOOM_TAPAUSE + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_TA_KICK        = 0x11,
    PERF_EVENT_TA_FINISH      = PERF_EVENT_TA_KICK + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_3D_KICK        = 0x12,
    PERF_EVENT_3D_FINISH      = PERF_EVENT_3D_KICK + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_CDM_KICK       = 0x13,
    PERF_EVENT_CDM_FINISH     = PERF_EVENT_CDM_KICK + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_3DSPM_KICK     = 0x14,
    PERF_EVENT_3DSPM_FINISH   = PERF_EVENT_3DSPM_KICK + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_TDM_KICK       = 0x15,
    PERF_EVENT_TDM_FINISH     = PERF_EVENT_TDM_KICK + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_3DRTA_KICK     = 0x16,
    PERF_EVENT_3DRTA_FINISH   = PERF_EVENT_3DRTA_KICK + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_CE_KICK        = 0x17,
    PERF_EVENT_CE_FINISH      = PERF_EVENT_CE_KICK + PERF_EVENT_FINISH_OFFSET,

    PERF_EVENT_PFM_KICK   = 0x18,
    PERF_EVENT_PFM_FINISH = PERF_EVENT_PFM_KICK + PERF_EVENT_FINISH_OFFSET,

    // Debug types
    PERF_EVENT_DEBUG_A_KICK   = 0x19,
    PERF_EVENT_DEBUG_A_FINISH = PERF_EVENT_DEBUG_A_KICK + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_DEBUG_B_KICK   = 0x1a,
    PERF_EVENT_DEBUG_B_FINISH = PERF_EVENT_DEBUG_B_KICK + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_DEBUG_C_KICK   = 0x1b,
    PERF_EVENT_DEBUG_C_FINISH = PERF_EVENT_DEBUG_C_KICK + PERF_EVENT_FINISH_OFFSET,
    PERF_EVENT_DEBUG_D_KICK   = 0x1c,
    PERF_EVENT_DEBUG_D_FINISH = PERF_EVENT_DEBUG_D_KICK + PERF_EVENT_FINISH_OFFSET,

    PERF_EVENT_GROUP_LIMIT =
        0x1F, // Now use uint32 as filter bitmask, so limit 32 groups, kick event should not beyond this.
    PERF_REPORTEVENT_RANGE_END = PERF_EVENT_DEBUG_D_FINISH,

    PERF_EVENT_FORCE_DWORD = 0x7FFFFFFF // Force enum to be 32-bits wide
} MTFW_FWIF_PERF_EVENT_TYPE;

// =====================================================================================================================
typedef enum
{
    PERF_FILTER_INVALID_PACKET       = 0x00,
    PERF_FILTER_VALID_FW_PACKET      = 0x01,
    PERF_FILTER_VALID_HW_PACKET      = 0x03,
    PERF_FILTER_VALID_COUNTER_PACKET = 0x04,
    PERF_FILTER_SCH_DATA_PACKET      = 0x05,
    PERF_FILTER_SUBM_DATA_PACKET     = 0x06,

    PERF_FILTER_FORCE_DWORD = 0x7FFFFFFF // Force enum to be 32-bits wide
} MTFW_FWIF_PERF_FILTER;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint32_t domain;
    uint32_t fwThreadId;
    uint32_t fenceId;
    uint32_t id; // Use for FW schedule type, show nodeid, dmtype, coreid, etc.

    uint64_t frameId;
    uint64_t cbId;
    uint32_t processId;
    uint32_t queueId; /* Hw queue id */
} MTFW_FWIF_PERF_EVENT_FW_DATA;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint64_t frameId;
    uint64_t cbId; /* Cmd Buffer ID */
    uint64_t ccbItemToken;
    uint32_t fenceId;
    uint32_t processId;

    uint32_t coreId;
    uint32_t coreMask;
    uint32_t queueId; /* Hw queue id */
    uint32_t commandFlags;

    uint64_t MTFW_ALIGN_BYTES_8 kickSemaphoreBufferAddr;
    struct MTFW_ALIGN_BYTES_8
    {
        uint64_t dependentValue[MTFW_KICK_SEMAPHORE_MAX_NUMS];
        uint64_t finishValue;
    } kickSemaphore;
} MTFW_FWIF_PERF_EVENT_HW_DATA;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint32_t processId;
    uint32_t fenceId;
    uint64_t ccbItemToken;
    uint64_t frameId;
    uint64_t cbId; /* Cmd Buffer ID */

    uint64_t qSempWaitMask;
    uint64_t qSempKickedMask;

    uint32_t submFlags;
    uint32_t submAttribute;
    uint32_t submCmd;
    uint32_t submState; /* Value is MTFW_SUBMISSION_STATE */
} MTFW_FWIF_PERF_EVENT_SUBM_DATA;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint64_t schQCounter64[4];
    uint32_t schQCounter32[8];
    uint16_t schQCounter[16];

    uint32_t queueCtxflags;

    uint32_t queueId;
    uint32_t scheType;
} MTFW_FWIF_PERF_EVENT_SCH_DATA;

/* Signature ASCII pattern 'FWP0' */
#define MTFW_FWIF_PERF_EVENT_SIG 0x46575030

typedef struct MTFW_ALIGN_BYTES_8
{
    uint64_t timestamp;
    uint32_t signature; /* protocol signature, always the value MTFW_FWIF_PERF_EVENT_SIG */
    union
    {
        struct
        {
            uint32_t eventVersion : 7;
            uint32_t osid         : 4;
        } field;
        uint32_t value;
    } flags;

    uint32_t size; /* Size of MTFW_FWIF_PERF_EVENT */
    uint32_t ordinal;

    MTFW_FWIF_PERF_EVENT_TYPE event;
    MTFW_FWIF_PERF_FILTER     filter;
} MTFW_FWIF_PERF_EVENT_HEADER;

#define MTFW_FWIF_GET_PERF_EVENT_SIG(evAddr) (((MTFW_FWIF_PERF_EVENT_HEADER*)(evAddr))->signature)

#define MTFW_FWIF_PERF_EVENT_ISVALID(sig) ((sig) == MTFW_FWIF_PERF_EVENT_SIG)

typedef union MTFW_ALIGN_BYTES_8
{
    MTFW_FWIF_PERF_EVENT_HW_DATA hardware;
    MTFW_FWIF_PERF_EVENT_FW_DATA firmware;

    MTFW_FWIF_PERF_EVENT_SCH_DATA  schData;
    MTFW_FWIF_PERF_EVENT_SUBM_DATA submData;
} MTFW_FWIF_PERF_EVENT_DATA;

typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_FWIF_PERF_EVENT_HEADER header;
    uint64_t                    timer;   /* will be deprecated */
    uint32_t                    ordinal; /* will be deprecated */
    uint32_t                    osid;    /* will be deprecated */
    MTFW_FWIF_PERF_EVENT_TYPE   event;   /* will be deprecated */
    MTFW_FWIF_PERF_FILTER       filter;  /* will be deprecated */
    MTFW_FWIF_PERF_EVENT_DATA   data;
} MTFW_FWIF_PERF_EVENT;

#define MTFW_FWIF_PERF_EVENT_BUFFER_SIZE (sizeof(MTFW_FWIF_PERF_EVENT) * MTFW_PERF_EVENT_MAX)

/* Perf event buffer  */
typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_FWIF_PERF_EVENT perfEvents[MTFW_PERF_EVENT_MAX];
    volatile uint32_t    perfEventWriteOffset;
    volatile uint32_t    perfEventReadOffset;
    volatile uint32_t    perfFilter;
    volatile uint64_t    hwPerfFilter;
} MTFW_FWIF_PERF_EVENT_BUF_CTL;

/***********************************************************************************************************/

typedef struct MTFW_ALIGN_BYTES_8
{
    uint16_t baseAddr;
    uint16_t selectAddr;
    uint64_t selectValue;
    uint16_t indirectAddr;
    uint8_t  indirectUnitNum;
    uint8_t  counterSelect;
    uint16_t counterAddr;
} MTFW_FWIF_PERF_SELECTOR_RECORD;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint32_t                       ordinal;
    uint32_t                       order;
    uint32_t                       count;
    MTFW_FWIF_PERF_SELECTOR_RECORD selectors;
} MTFW_FWIF_PERF_SELECTOR;

typedef enum
{
    PERF_REQUEST_PING            = 0,
    PERF_REQUEST_ENABLE_EVENTS   = 1, // Enable perf events
    PERF_REQUEST_DISABLE_EVENTS  = 2, // Disable perf events
    PERF_REQUEST_CONFIG_COUNTERS = 3, // Configure perf events
    PERF_REQUEST_GET_INFO        = 4,
    PERF_REQUEST_GET_DATA        = 5,
#if defined(PDUMP)
    PERF_REQUEST_ENABLE_PDUMP = 9, // Enable perf events in pdump
#endif
    PERF_REQUEST_FORCE_DWORD = 0x7FFFFFFF // Force enum to be 32-bits wide
} MTFW_PERF_REQUEST_TYPE;

/* This is how Host sends requests down to the FW.  */
typedef struct MTFW_ALIGN_BYTES_8
{
    MTFW_PERF_REQUEST_TYPE requestType;
    union MTFW_ALIGN_BYTES_8
    {
        MTFW_FWIF_PERF_SELECTOR select;
        uint32_t                param;
        struct
        {
            uint32_t addr;
            uint64_t val;
        } reg;
    } u;
} MTFW_FWIF_PERF_REQUEST;

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_PERF_H__ */
