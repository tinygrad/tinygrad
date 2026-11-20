/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: This is Firmware API top-level data struct.
 *
 */
#ifndef __MTFW_FWIF_H__
#define __MTFW_FWIF_H__

#include "mtfw_fwif_types.h"
#include "mtfw_fwif_version.h"
#include "mtfw_fwif_log.h"
#include "mtfw_fwif_ccb.h"
#include "mtfw_fwif_node.h"
#include "mtfw_fwif_gen.h"
#include "mtfw_fwif_queue.h"

#ifdef __cplusplus
extern "C"
{
#endif

// FW config bit
#define MTFWIF_INIT_CFG_DISABLE_CE     0x00000001 // fwCfg[1] bit0, CE disable flag
#define MTFWIF_INIT_CFG_PREFETCH_SHIFT 1
#define MTFWIF_INIT_CFG_PREFETCH_BITMASK                                          \
    (0x00000003 << MTFWIF_INIT_CFG_PREFETCH_SHIFT) /* fwCfg[1] bit[2:1],          \
                                                    * 00: disable prefetch,       \
                                                    * 01: prefetch 1 line ahead,  \
                                                    * 10: prefetch 3 lines ahead, \
                                                    * 11: prefetch 5 lines ahead. \
                                                    */

/* pm-reserved pages occupies 1536 dword for asics before gen3, and there are other data*/
#define MTFWIF_INIT_CFG_SIZE 2048

#define MTFWIF_FW_BUILD_VERSION_SIZE 8

#define MTFW_COREMEM_SIZE (0x18000)

/*
 * Virtualisation definitions
 */

/* The default number of OSID is 1, higher number implies VZ enabled firmware */
#if defined(PVRSRV_VZ_NUM_OSID) && (PVRSRV_VZ_NUM_OSID + 1U > 1U)
#define MTFW_NUM_OS PVRSRV_VZ_NUM_OSID
#else
#define MTFW_NUM_OS 1U
#endif

#define MTFW_HOST_OS          (0U)

#define MTFW_FW_HEAP_SIZE (1 << 23) /* 8 MB */

#define MTFW_EXT_OSID_CCB_SIZE (128U)
#define MTFW_EXT_OSID_SIZE     (8U)

#define MTFW_VIRTUALISATION_REG_SIZE_PER_OS (0x10000)

/* Firmware's connection state */
typedef enum
{
    MTFW_CONNECTION_FW_OFFLINE = 0, /*!< Firmware is offline */
    MTFW_CONNECTION_FW_READY,       /*!< Firmware is initialised */
    MTFW_CONNECTION_FW_ACTIVE,      /*!< Firmware connection is fully established */
    MTFW_CONNECTION_FW_OFFLOADING,  /*!< Firmware is clearing up connection data */
    MTFW_CONNECTION_FW_DECLINE,     /*!< Firmware decline to connection */
    MTFW_CONNECTION_FW_STATE_COUNT
} MTFWIF_CONNECTION_FW_STATE;

/* OS' connection state */
typedef enum
{
    MTFW_CONNECTION_OS_OFFLINE = 0, /*!< OS is offline */
    MTFW_CONNECTION_OS_READY,       /*!< OS's KM driver is setup and waiting */
    MTFW_CONNECTION_OS_ACTIVE,      /*!< OS connection is fully established */
    MTFW_CONNECTION_OS_STATE_COUNT
} MTFWIF_CONNECTION_OS_STATE;

typedef union
{
    struct
    {
        uint32_t magicValue : 8; /* 0x4d */
        uint32_t threadNum  : 1; /* 0: One thread, 1: Tow threads. */
    } field;
    uint32_t value;
} MTFW_FWIF_STARTUP_INFO;

typedef struct
{
    uint32_t t0WatchDogCounter;
    uint32_t t1WatchDogCounter;

    uint32_t t0State; /* 0 is T0 schedule is idle. */
    uint32_t t1State; /* 0 is T1 schedule is idle. */
} MTFW_FWIF_WATCHDOG;

typedef struct
{
    union {
        uint64_t vgpuFWVersion; /* deprecated */
        uint64_t vgpuFlag;
    };
    uint32_t osIdStart;
    uint32_t osIdCount;
    uint32_t vgpuSchedulingPolicy;
    uint32_t vgpuTimeSlicedValue;
    uint32_t vgpuUtil;
    uint32_t vgpuQosEnable;
    fw_va    vgpuSharedInfoAddr;
    uint64_t vgpuReleaseVersion; /* struct vgpu_release_version, for example: major.minor.patch[-update] */
} MTFW_FWIF_VGPU;

typedef struct MTFW_ALIGN(32) MTFW_FWIF_TAG
{
    union
    {
        struct
        {
            uint32_t version : 4;
        } field;
        uint32_t value;
    } flags;

    volatile uint32_t fwStarted; /* value is MTFW_FWIF_STARTUP_INFO */

    uint8_t fwBuildVer[MTFWIF_FW_BUILD_VERSION_SIZE]; /* FW build version for KMD debug */

    fw_va watchDog; /* point to MTFW_FWIF_WATCHDOG object */

    /* Level 1 schedule import queues. Point to MTFW_CCB_QUEUE array start address. */
    fw_va ccbQueueArrayAddr;
    /* Level 1 fast schedule import queues. Point to MTFW_CCB_QUEUE array start address. */
    fw_va ccbFastQueueArrayAddr;
    /* Soft-Queue between FEC and META. Point to MTFW_LEVEL_QUEUE array start address. */
    fw_va levelQueueArrayFecAddr;
    fw_va levelQueueArrayMetaAddr;
    /* Level 2 schedule import queues. Point to MTFW_NODE_QUEUE array start address. */
    fw_va nodeQueueArrayAddr;

    /* Queue array number. */
    uint32_t ccbQueueArrayNum;
    uint32_t ccbFastQueueArrayNum;
    uint32_t nodeQueueArrayNum;
    uint32_t levelQueueArrayNum;

    uint32_t startUpCounter; /* Host inc this value when GPU reset happen. */

    /**
     * CCB between Host and Firmware Submission Level schedule.
     */
    MTFW_CCB_QUEUE ccbQueueArray[MTFW_CCB_TYPE_MAX_NUM];

    /**
     * Node between firmware SubmissionLevel schedule and In-Submission sedule.
     */
    MTFW_NODE_QUEUE nodeQueueArray[MTFW_NODE_TYPE_MAX_NUM];

    /**
     * DFX config and buffer. perf, log ...
     */
    fw_va traceBufAddr;    /* point to LOG buffer for META. MTFW_FWIF_TRACEBUF object */
    fw_va traceBufAddrFEC; /* point to LOG buffer for FEC. MTFW_FWIF_TRACEBUF object */

    fw_va    metaPerfEventBufCtrlArray; /* Meta perf event buffer array for BG, IRQ ... MTFW_FWIF_PERF_EVENT_BUF_CTL */
    fw_va    fecPerfEventBufCtrlArray;  /* FEC perf event buffer array ... MTFW_FWIF_PERF_EVENT_BUF_CTL */
    uint32_t metaPerfEventBufCtrlNum;
    uint32_t fecPerfEventBufCtrlNum;

    fw_va faultDumpBuffer; /* point to Meta fault dump buffer. MTFW_FWIF_FAULT_INFO_CTRL */
    fw_va gpuUtilBufAddr;  /* point to GPU Utilization buffer. MTFW_FWIF_GPU_UTIL_BUF */

    /* for VDI */
    union {
        uint64_t vgpuFWVersion; /* deprecated */
        uint64_t vgpuFlag;
    };
    uint32_t osIdStart;
    uint32_t osIdCount;
    uint32_t vgpuSchedulingPolicy;
    uint32_t vgpuTimeSlicedValue;
    uint32_t vgpuUtil;
    uint32_t vgpuQosEnable;
    fw_va    vgpuSharedInfoAddr;
    uint64_t vgpuReleaseVersion; /* struct vgpu_release_version, for example: major.minor.patch[-update] */

    uint32_t socTimerFreq; /* SOC timer register frequence. 0 is FW default. */
    uint32_t gpuFrequence; /* GPU frequence for META timer. 0 is FW default. */

    uint64_t multicoreClusterMask; /* active cluster mask for each core; TODO: remove the entry when SMC is ready */

    union MTFW_ALIGN_BYTES_8
    {
        MTFW_FWIF_CONFIG cfg;
        uint32_t         value[MTFWIF_INIT_CFG_SIZE];
    } fwCfgs;

    // Align to DMA block size (32)
    MTFW_ALIGN(32) uint8_t firmwareCoreMemory[MTFW_COREMEM_SIZE];
} MTFW_FWIF;

typedef struct MTFW_ALIGN_BYTES_8
{
    // total hw execution time
    uint32_t cmdExecutedTime;

    // completed or preempted cmd
    uint32_t maxCmdTime;

    // count of completed cmd
    uint8_t completeCmdCnt;

    // count of preempted cmd
    uint8_t preemptedCmdCnt;

    uint8_t swContextSwitchCnt;
    uint8_t hwContextSwitchCnt;

    uint8_t preemptionReqCnt;
    uint8_t blockedCmdCnt;
    uint8_t scheduleWeight;
    uint8_t heavyCmdBlockCount;
} MTFW_VGPU_SCHEDULE_INFO;

typedef struct MTFW_ALIGN_BYTES_8
{
    uint32_t magic;
    /* Every 8 bits represents timeout time for triggering one DM context switch when DM executes command */
    uint64_t cswTime;
    uint64_t SysMemSize[MTFW_NUM_OS]; // For vgpu qos node get sysmem info

    // Record active osids: each bit represents an osid online/offline status
    uint32_t ActiveOSid;

    //  HWR related osid info
    uint32_t RunningOSid[MTFW_FWIF_QUEUE_MAX];
    uint32_t NeedHWROSid[MTFW_FWIF_QUEUE_MAX];

    uint32_t                costTime;
    uint32_t                quotaOsidMask[MTFW_FWIF_QUEUE_MAX];
    MTFW_VGPU_SCHEDULE_INFO sche_info[MTFW_FWIF_QUEUE_MAX - 1][MTFW_NUM_OS];
    uint32_t                qosCredit[MTFW_FWIF_QUEUE_MAX];

    uint64_t qosLimit;
    uint64_t qosAvr;
    uint64_t qosStandard;

    uint32_t reOnlineOsid;
} MTFW_VGPU_SHARED_INFO;

#if defined(SUPPORT_SW_OSID_EXTENSION)
typedef struct MTFW_ALIGN_BYTES_8
{
    // Host BG Kick Array
    uint32_t hostBgKickCCB[MTFW_EXT_OSID_CCB_SIZE];
    uint32_t hostBgKickReadOffset;
    uint32_t hostBgKickWriteOffset;

    // FW BG Kick Array
    uint32_t fwBgKickCCB[MTFW_EXT_OSID_CCB_SIZE];
    uint32_t fwBgKickReadOffset;
    uint32_t fwBgKickWriteOffset;

    // IRQ Kick Array
    uint32_t irqKickCCB[MTFW_EXT_OSID_CCB_SIZE];
    uint32_t irqKickReadOffset;
    uint32_t irqKickWriteOffset;

    uint32_t scratch0[MTFW_EXT_OSID_SIZE];
    uint32_t scratch1[MTFW_EXT_OSID_SIZE];
    uint32_t scratch2[MTFW_EXT_OSID_SIZE];
    uint32_t scratch3[MTFW_EXT_OSID_SIZE];

    uint32_t irqStatus[MTFW_EXT_OSID_SIZE];
} MTFW_VGPU_EXT_OSID_INFO;
#endif

/**
 * For vdi load firmware
 * pls make sure this value > sizeof(MTFW_FWIF)
 * and is the same as the value in gr-kmd
 */
#define MTFW_VGPU_INIT_FIXED_SIZE 450000 // 367136 * 1.2 = 440564 < 450000

// For vdi fw load
// pls make sure this value == sizeof(MTFW_FWIF_TRACEBUF)
// and is the same as the value in gr-kmd
// If there are compilation errors, please contact the VDI team for handling.
#define MTFW_VGPU_TRACEBUF_FIXED_SIZE 97560

#define MTFW_MAX_DM_NUM 6 /* Used for VGPU, TDM, UQ-TDM, UQ-TA, UQ-3D, UQ-CDM, CDM */

typedef struct _MTFW_FWIF_FEC_BOOT_CFG_
{
    uint32_t fwifFwVaLow;
    uint32_t fwifFwVaHi;
    uint32_t fwifHeapSize;

    uint32_t fwvaPageTableRootLow;
    uint32_t fwvaPageTableRootHi;

    uint32_t reservePPageAddrLow;
    uint32_t reservePPageAddrHi;
    uint32_t reservePPageNum; /* Physical Page size is 4K */
} MTFW_FWIF_FEC_BOOT_CFG;

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FW_IF_H__ */
