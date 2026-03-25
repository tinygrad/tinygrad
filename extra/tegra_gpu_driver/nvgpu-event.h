/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 * Event session
 *
 * NVGPU_GPU_IOCTL_GET_EVENT_FD opens an event session.
 * Below ioctls can be used on these sessions fds.
 */

#ifndef _UAPI__LINUX_NVGPU_EVENT_H__
#define _UAPI__LINUX_NVGPU_EVENT_H__

#include "nvgpu-uapi-common.h"

#define NVGPU_EVENT_IOCTL_MAGIC	'E'

/* Normal events (POLLIN) */
/* Event associated to a VF update */
#define NVGPU_GPU_EVENT_VF_UPDATE				0

/* Recoverable alarms (POLLPRI) */
/* Alarm when target frequency on any session is not possible */
#define NVGPU_GPU_EVENT_ALARM_TARGET_VF_NOT_POSSIBLE		1
/* Alarm when target frequency on current session is not possible */
#define NVGPU_GPU_EVENT_ALARM_LOCAL_TARGET_VF_NOT_POSSIBLE	2
/* Alarm when Clock Arbiter failed */
#define NVGPU_GPU_EVENT_ALARM_CLOCK_ARBITER_FAILED		3
/* Alarm when VF table update failed */
#define NVGPU_GPU_EVENT_ALARM_VF_TABLE_UPDATE_FAILED		4
/* Alarm on thermal condition */
#define NVGPU_GPU_EVENT_ALARM_THERMAL_ABOVE_THRESHOLD		5
/* Alarm on power condition */
#define NVGPU_GPU_EVENT_ALARM_POWER_ABOVE_THRESHOLD		6

/* Non recoverable alarm (POLLHUP) */
/* Alarm on GPU shutdown/fall from bus */
#define NVGPU_GPU_EVENT_ALARM_GPU_LOST				7

#define NVGPU_GPU_EVENT_LAST	NVGPU_GPU_EVENT_ALARM_GPU_LOST

struct nvgpu_gpu_event_info {
	__u32 event_id;		/* NVGPU_GPU_EVENT_* */
	__u32 reserved;
	__u64 timestamp;	/* CPU timestamp (in nanoseconds) */
};

struct nvgpu_gpu_set_event_filter_args {

	/* in: Flags (not currently used). */
	__u32 flags;

	/* in: Size of event filter in 32-bit words */
	__u32 size;

	/* in: Address of buffer containing bit mask of events.
	 * Bit #n is set if event #n should be monitored.
	 */
	__u64 buffer;
};

#define NVGPU_EVENT_IOCTL_SET_FILTER \
	_IOW(NVGPU_EVENT_IOCTL_MAGIC, 1, struct nvgpu_gpu_set_event_filter_args)
#define NVGPU_EVENT_IOCTL_LAST		\
	_IOC_NR(NVGPU_EVENT_IOCTL_SET_FILTER)
#define NVGPU_EVENT_IOCTL_MAX_ARG_SIZE	\
	sizeof(struct nvgpu_gpu_set_event_filter_args)

#endif
