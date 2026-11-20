/**
 *
 * Copyright (C) 2025 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: GPU firmware queue definition.
 *
 */
#ifndef __MTFW_FWIF_QUEUE_H__
#define __MTFW_FWIF_QUEUE_H__

#include "mtfw_fwif_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define MTFW_FWIF_QUEUE_TYPE_INVALID UINT32_MAX

#define MTFW_FWIF_QUEUE_MAX 8 /* HW allows up to 8 submission queues in maximum */

typedef enum
{
    MTFW_FWIF_QUEUE_TYPE_GP = 0,
    MTFW_FWIF_QUEUE_TYPE_UQ,
    MTFW_FWIF_QUEUE_TYPE_TDM,
    MTFW_FWIF_QUEUE_TYPE_CDM,
    MTFW_FWIF_QUEUE_TYPE_CE,
    MTFW_FWIF_QUEUE_TYPE_KMD_CE,

    MTFW_FWIF_QUEUE_TYPE_MAX,
} MTFW_FWIF_QUEUE_TYPE;

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_QUEUE_H__ */
