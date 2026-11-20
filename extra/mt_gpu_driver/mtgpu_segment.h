/*
 * @Copyright   Copyright (c) Moorethreads Technologies Ltd. All Rights Reserved
 * @License     Dual MIT/GPLv2
 */

#ifndef _MTGPU_SEGMENT_H_
#define _MTGPU_SEGMENT_H_

#include "linux-types.h"

#define MTGPU_SEGMENT_ID_DEFAULT	0
#define MTGPU_SEGMENT_ID_KERNEL		1
/* segment id 2/3/4 used by vpu */
#define MTGPU_SEGMENT_ID_USER		2
#define MTGPU_SEGMENT_ID_VPU_GUEST	2
#define MTGPU_SEGMENT_ID_VPU_GUEST_VCPU   3
#define MTGPU_SEGMENT_ID_VPU_GUEST_ENCODE 5
#define MTGPU_SEGMENT_ID_PB               6
#define MTGPU_SEGMENT_ID_PB_VGPU          7
#define MTGPU_SEGMENT_ID_VPU_HOST       0xff

#define MTGPU_SEGMENT_STEP_SIZE         0x400000000ULL
#define MTGPU_MAX_SEGMENT               9
#define MTGPU_MAX_NORMAL_SEGMENT	5
#define MTVPU_MAX_SEGMENT               4
#define MTGPU_SEGMENT_NORMAL_SIZE       0x100000000ULL
#define MTGPU_VPU_GROUP			2
#define MTGPU_VPU_FRONT_RESERVE		0x204000ULL
#define MTGPU_VPU_BACK_RESERVE		0x11000ULL
#define MTGPU_VPU_TOTAL_RESERVE		0x215000ULL
#define MTGPU_VPU_GUEST_SHM_SIZE        0x8000ULL

struct mtgpu_device;
struct drm_device;

enum mtgpu_segment_flag {
	SEGMENT_FLAG_VPU_VISIBLE = 0x1ULL,
	SEGMENT_FLAG_VPU_VCPU = 0x2ULL,
	SEGMENT_FLAG_DISP_VISIBLE = 0x4ULL,
	SEGMENT_FLAG_PB = 0x8ULL,
	SEGMENT_FLAG_VPU_VCORE = 0x10ULL,
	SEGMENT_FLAG_VPU_ENCODE = 0x20ULL,
	SEGMENT_FLAG_VPU_DECODE_DEDICATE = 0x40ULL,
	SEGMENT_FLAG_VPU_DECODE_SHARED_WITH_VGPU = 0x80ULL,
};

struct mtgpu_segment {
	u64 start;
	u64 size;
	/* 0 is the highest priority */
	int priority;
	int flag;
	int segment_id;
};

struct mtgpu_segment_info {
	int segment_cnt;
	struct mtgpu_segment *segments;
};

struct mtgpu_segment_stats {
	u64 start;
	u64 size;
	u64 freed;
};

#if !defined(NO_HARDWARE)
int mtgpu_vram_get_segment_for_display(struct drm_device *ddev);
int mtgpu_generate_mem_segments(struct mtgpu_device *mtdev);
void mtgpu_destroy_mem_segments(struct mtgpu_device *mtdev);
#else
static int mtgpu_vram_get_segment_for_display(struct drm_device *ddev)
{
	return 0;
}
#endif

#endif /* _MTGPU_SEGMENT_H_ */
