/**
 *
 * Copyright (C) 2024 - 2024 Moore Threads Ltd. All Rights Reserved.
 *
 * Descript: MTFW GPU DM type enum.
 *
 */
#ifndef __MTFW_FWIF_DM_TYPES_H__
#define __MTFW_FWIF_DM_TYPES_H__

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
    MTFW_FWIF_DM_GP   = 0,
    MTFW_FWIF_DM_TDM  = 1,
    MTFW_FWIF_DM_GEOM = 2,
    MTFW_FWIF_DM_3D   = 3,
    MTFW_FWIF_DM_CDM  = 4,
    MTFW_FWIF_DM_CE   = 5,
} MTFW_FWIF_GPU_DM_TYPE;

#ifdef __cplusplus
}
#endif

#endif /* __MTFW_FWIF_DM_TYPES_H__ */
