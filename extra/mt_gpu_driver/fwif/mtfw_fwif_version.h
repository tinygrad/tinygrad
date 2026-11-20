/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: GPU firmware ABI version.
 *
 */
#ifndef __MTFW_FWIF_VERSION_H__
#define __MTFW_FWIF_VERSION_H__

#include "mtfw_fwif_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
    MTFW_FWIF_ABI_TYPE_RESERVE = 0,
    MTFW_FWIF_ABI_TYPE_GEN,        /* This is fwif generation */
    MTFW_FWIF_ABI_TYPE_SUBMISSION, /* This ABI object is MTFW_SUBMISSION_CMD_TYPE */
    MTFW_FWIF_ABI_TYPE_KICK,       /* This ABI object is MTFW_SUBMISSION_REGION_TYPE */
} MTFW_FWIF_ABI_TYPE;

#define MTFW_FWIF_VERSION_VALUE(abiType, minVer, maxVer) \
    (((abiType)&0xff) | (((minVer)&0xfff) << 8) | (((maxVer)&0xfff) << 20))

/**
 * $fwif_version type data will place in $fwif_version segment.
 * If $fwif_version.field.value is 0, this version item is INVALID.
 */
typedef struct
{
    union
    {
        struct
        {
            uint32_t abiType : 8; /* This is MTFW_FWIF_ABI_TYPE */
            uint32_t minVer  : 12;
            uint32_t maxVer  : 12;
        } field;
        uint32_t value;
    } version;

    uint32_t abiEnum;
    uint8_t  abiName[8]; /* This is ABI name string */
} MTFW_FWIF_VERSION;

#define MTWF_FWIF_NAME_GEN "F-GEN"

/**
 * Max number MTFW_FWIF_VERSION in $fwif_version section..
 */
#define MTFW_FWIF_VERSION_ITEM_MAX_NUM 256

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_VERSION_H__ */
