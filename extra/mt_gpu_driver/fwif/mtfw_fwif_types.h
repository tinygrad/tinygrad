/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: GPU firmware basic type definitions.
 *
 */
#ifndef __MTFW_FWIF_TYPES_H__
#define __MTFW_FWIF_TYPES_H__

#if defined(_MSC_VER)
#include <stddef.h>
#elif !defined(__KERNEL__)
#include <stdint.h>
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#if defined(__GNUC__) || (defined(__linux__) && defined(__KERNEL__))
#define MTFW_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#define MTFW_ALIGN(n) __declspec(align(n))
#pragma warning(disable : 4324)
#else
#error "Align MACROS need to be defined for this compiler"
#endif

#define MTFW_ALIGN_BYTES_8   MTFW_ALIGN(8)
#define MTFW_ALIGN_BYTES_32  MTFW_ALIGN(32)
#define MTFW_ALIGN_CACHELINE MTFW_ALIGN(64)

#ifndef RGXFW_ALIGN_UINT64
#define RGXFW_ALIGN_UINT64 MTFW_ALIGN_BYTES_8
#endif

#ifndef RGXFW_ALIGN
#define RGXFW_ALIGN(n) MTFW_ALIGN(n)
#endif

/**
 * This is GPU va in UserDomain. If want to access MUST use app pagetable map it befor.
 */
typedef uint64_t gpu_va;

/**
 * This is firmware VA in SubmissionDomain. Firmware access it directly.
 * NOTE: FEC is 64bit system, but Meta is 32bit system.
 */
typedef uint64_t fw_va;

#define MTFW_FW_VA_TO_POINTER(va) ((void*)((unsigned long)(va)))

/**
 * vmem PA.
 */
typedef uint64_t gpu_pa;

typedef union
{
    struct
    {
        uint32_t version  : 12;
        uint32_t crc      : 16;
    } field;
    uint32_t value;
} MTFW_FWIF_HEAD;

typedef struct MTFW_ALIGN_BYTES_8
{
    union
    {
        volatile uint32_t wrIdx;
        volatile fw_va    wrIdxAddr; /* point to Host address for write index */
    };
    volatile uint32_t rdIdx;

    uint32_t reserve;
} MTFW_RING_CTRL;

typedef enum
{
    MTFW_PRIORITY_LOWEST = 0,
    MTFW_PRIORITY_0      = MTFW_PRIORITY_LOWEST,
    MTFW_PRIORITY_1,
    MTFW_PRIORITY_HIGHEST = MTFW_PRIORITY_1,
    MTFW_PRIORITY_LEVEL_NUMS,
} MTFW_FWIF_PRIORITY;

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_TYPES_H__ */
