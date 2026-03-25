/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 */

#ifndef _UAPI__LINUX_NVGPU_NVS_H
#define _UAPI__LINUX_NVGPU_NVS_H

#include "nvgpu-uapi-common.h"

#define NVGPU_NVS_IOCTL_MAGIC 'N'
#define NVGPU_NVS_CTRL_FIFO_IOCTL_MAGIC 'F'

/**
 * Domain parameters to pass to the kernel.
 */
struct nvgpu_nvs_ioctl_domain {
	/*
	 * Human readable null-terminated name for this domain.
	 */
	char name[32];

	/*
	 * Scheduling parameters: specify how long this domain should be scheduled
	 * for and what the grace period the scheduler should give this domain when
	 * preempting. A value of zero is treated as an infinite timeslice or an
	 * infinite grace period, respectively.
	 */
	__u64 timeslice_ns;
	__u64 preempt_grace_ns;

	/*
	 * Pick which subscheduler to use. These will be implemented by the kernel
	 * as needed. There'll always be at least one, which is the host HW built in
	 * round-robin scheduler.
	 */
	__u32 subscheduler;

/*
 * GPU host hardware round-robin.
 */
#define NVGPU_SCHED_IOCTL_SUBSCHEDULER_HOST_HW_RR 0x0

	/*
	 * Populated by the IOCTL when created: unique identifier. User space
	 * must set this to 0.
	 */
	__u64 dom_id;

	/* Must be 0. */
	__u64 reserved1;
	/* Must be 0. */
	__u64 reserved2;
};

/**
 * NVGPU_NVS_IOCTL_CREATE_DOMAIN
 *
 * Create a domain - essentially a group of GPU contexts. Applications
 * can be bound into this domain on request for each TSG.
 *
 * The domain ID is returned in dom_id; this id is _not_ secure. The
 * nvsched device needs to have restricted permissions such that only a
 * single user, or group of users, has permissions to modify the
 * scheduler.
 *
 * It's fine to allow read-only access to the device node for other
 * users; this lets other users query scheduling information that may be
 * of interest to them.
 */
struct nvgpu_nvs_ioctl_create_domain {
	/*
	 * In/out: domain parameters that userspace configures.
	 *
	 * The domain ID is returned here.
	 */
	struct nvgpu_nvs_ioctl_domain domain_params;

	/* Must be 0. */
	__u64 reserved1;
};

/**
 * NVGPU_NVS_IOCTL_REMOVE_DOMAIN
 *
 * Remove a domain that has been previously created.
 *
 * The domain must be empty; it must have no TSGs bound to it. The domain's
 * device node must not be open by anyone.
 */
struct nvgpu_nvs_ioctl_remove_domain {
	/*
	 * In: a domain_id to remove.
	 */
	__u64 dom_id;

	/* Must be 0. */
	__u64 reserved1;
};

/**
 * NVGPU_NVS_IOCTL_QUERY_DOMAINS
 *
 * Query the current list of domains in the scheduler. This is a two
 * part IOCTL.
 *
 * If domains is 0, then this IOCTL will populate nr with the number
 * of present domains.
 *
 * If domains is nonzero, then this IOCTL will treat domains as a pointer to an
 * array of nvgpu_nvs_ioctl_domain and will write up to nr domains into that
 * array. The nr field will be updated with the number of present domains,
 * which may be more than the number of entries written.
 */
struct nvgpu_nvs_ioctl_query_domains {
	/*
	 * In/Out: If 0, leave untouched. If nonzero, then write up to nr
	 * elements of nvgpu_nvs_ioctl_domain into where domains points to.
	 */
	__u64 domains;

	/*
	 * - In: the capacity of the domains array if domais is not 0.
	 * - Out: populate with the number of domains present.
	 */
	__u32 nr;

	/* Must be 0. */
	__u32 reserved0;

	/* Must be 0. */
	__u64 reserved1;
};

#define NVGPU_NVS_IOCTL_CREATE_DOMAIN			\
	_IOWR(NVGPU_NVS_IOCTL_MAGIC, 1,			\
	      struct nvgpu_nvs_ioctl_create_domain)
#define NVGPU_NVS_IOCTL_REMOVE_DOMAIN			\
	_IOW(NVGPU_NVS_IOCTL_MAGIC, 2,			\
	      struct nvgpu_nvs_ioctl_remove_domain)
#define NVGPU_NVS_IOCTL_QUERY_DOMAINS			\
	_IOWR(NVGPU_NVS_IOCTL_MAGIC, 3,			\
	      struct nvgpu_nvs_ioctl_query_domains)

#define NVGPU_NVS_IOCTL_LAST				\
	_IOC_NR(NVGPU_NVS_IOCTL_QUERY_DOMAINS)
#define NVGPU_NVS_IOCTL_MAX_ARG_SIZE			\
	sizeof(struct nvgpu_nvs_ioctl_create_domain)

/* Request for a Control Queue. */
#define NVGPU_NVS_CTRL_FIFO_QUEUE_NUM_CONTROL 1U
/* Request for an Event queue.  */
#define NVGPU_NVS_CTRL_FIFO_QUEUE_NUM_EVENT 2U

/* Direction of the requested queue is from CLIENT(producer)
 * to SCHEDULER(consumer).
 */
#define NVGPU_NVS_CTRL_FIFO_QUEUE_DIRECTION_CLIENT_TO_SCHEDULER 0

/* Direction of the requested queue is from SCHEDULER(producer)
 * to CLIENT(consumer).
 */
#define NVGPU_NVS_CTRL_FIFO_QUEUE_DIRECTION_SCHEDULER_TO_CLIENT 1

#define NVGPU_NVS_CTRL_FIFO_QUEUE_ACCESS_TYPE_EXCLUSIVE 1
#define NVGPU_NVS_CTRL_FIFO_QUEUE_ACCESS_TYPE_NON_EXCLUSIVE 0

/**
 * NVGPU_NVS_CTRL_FIFO_IOCTL_CREATE_QUEUE
 *
 * Create shared queues for domain scheduler's control fifo.
 *
 * 'queue_num' is set by UMD to NVS_CTRL_FIFO_QUEUE_NUM_CONTROL
 * for Send/Receive queues and NVGPU_NVS_CTRL_FIFO_QUEUE_NUM_EVENT
 * for Event Queue.
 *
 * 'direction' is set by UMD to NVGPU_NVS_CTRL_FIFO_QUEUE_DIRECTION_CLIENT_TO_SCHEDULER
 * for Send Queue and NVGPU_NVS_CTRL_FIFO_QUEUE_DIRECTION_SCHEDULER_TO_CLIENT
 * for Receive/Event Queue.
 *
 * The parameter 'queue_size' is set by KMD.
 *
 * Initially, all clients are setup as non-exclusive. The first client to successfully
 * request an exclusive access is internally marked as an exclusive client. It remains
 * so until the client closes the control-fifo device node.
 *
 * Clients that require exclusive access shall set 'access_type'
 * to NVGPU_NVS_CTRL_FIFO_QUEUE_ACCESS_TYPE_EXCLUSIVE, otherwise set it to
 * NVGPU_NVS_CTRL_FIFO_QUEUE_ACCESS_TYPE_NON_EXCLUSIVE.
 *
 * Note, queues of NVGPU_NVS_CTRL_FIFO_QUEUE_NUM_EVENT has shared read-only
 * access irrespective of the type of client.
 *
 * 'dmabuf_fd' is populated by the KMD for the success case, else its set to -1.
 */
struct nvgpu_nvs_ctrl_fifo_ioctl_create_queue_args {
	/* - In: Denote the queue num. */
	__u32 queue_num;

	/* - In: Denote the direction of producer => consumer */
	__u8 direction;

	/* - In: Denote the type of request */
	__u8 access_type;

	/* Must be 0. */
	__u16 reserve0;

	/* - Out: Size of the queue in bytes. Multiple of 4 bytes */
	__u32 queue_size;

	/* - Out: dmabuf file descriptor(FD) of the shared queue exposed via the KMD.
	 * - This field is expected to be populated by the KMD.
	 * - UMD is expected to close the FD.
	 *
	 * - mmap() is used to access the queue.
	 * - MAP_SHARED must be specified.
	 * - Exclusive access clients may map with read-write access (PROT_READ | PROT_WRITE).
	 *   Shared access clients may map only with read-only access (PROT_READ)
	 *
	 * - Cpu Caching Mode
	 * - cached-coherent memory type is used when the system supports this between the client and scheduler.
	 * - non-cached memory type otherwise.
	 *
	 * - On Tegra:
	 *   Normal cacheable (inner shareable) on T194/T234 with the KMD scheduler.
	 *   Normal cacheable (outer shareable, I/O coherency enabled) for T234 with the GSP scheduler.
	 *
	 * - On generic ARM:
	 *   Normal cacheable (inner shareable) with the KMD scheduler.
	 *   Normal non-cacheable write-combining with the GSP scheduler.
	 */
	__s32 dmabuf_fd;
};

/**
 * NVGPU_NVS_CTRL_FIFO_IOCTL_RELEASE_QUEUE
 *
 * Release a domain scheduler's queue.
 *
 * 'queue_num' is set by UMD to NVS_CTRL_FIFO_QUEUE_NUM_CONTROL
 * for Send/Receive queues and NVGPU_NVS_CTRL_FIFO_QUEUE_NUM_EVENT
 * for Event Queue.
 *
 * 'direction' is set by UMD to NVGPU_NVS_CTRL_FIFO_QUEUE_DIRECTION_CLIENT_TO_SCHEDULER
 * for Send Queue and NVGPU_NVS_CTRL_FIFO_QUEUE_DIRECTION_SCHEDULER_TO_CLIENT
 * for Receive/Event Queue.
 *
 * Returns an error if queues of type NVS_CTRL_FIFO_QUEUE_NUM_CONTROL
 * have an active mapping.
 *
 * Mapped buffers are removed immediately for queues of type
 * NVS_CTRL_FIFO_QUEUE_NUM_CONTROL while those of type NVGPU_NVS_CTRL_FIFO_QUEUE_NUM_EVENT
 * are removed when the last user releases the control device node.
 *
 * User must ensure to invoke this IOCTL after invoking munmap on
 * the mmapped address. Otherwise, accessing the buffer could lead to segfaults.
 *
 */
struct nvgpu_nvs_ctrl_fifo_ioctl_release_queue_args {
	/* - In: Denote the queue num. */
	__u32 queue_num;

	/* - In: Denote the direction of producer => consumer */
	__u8 direction;

	/* Must be 0. */
	__u8 reserve0;

	/* Must be 0. */
	__u16 reserve1;


	/* Must be 0. */
	__u64 reserve2;
};

struct nvgpu_nvs_ctrl_fifo_ioctl_event {
/* Enable Fault Detection Event */
#define NVGPU_NVS_CTRL_FIFO_EVENT_FAULTDETECTED 1LLU
/* Enable Fault Recovery Detection Event */
#define NVGPU_NVS_CTRL_FIFO_EVENT_FAULTRECOVERY 2LLU
	__u64 event_mask;

	/* Must be 0. */
	__u64 reserve0;
};

/**
 * NVGPU_NVS_CTRL_FIFO_IOCTL_QUERY_SCHEDULER_CHARACTERISTICS
 *
 * Query the characteristics of the domain scheduler.
 * For R/W user, available_queues is set to
 * NVGPU_NVS_CTRL_FIFO_QUEUE_NUM_CONTROL | NVGPU_NVS_CTRL_FIFO_QUEUE_NUM_EVENT
 *
 * For Non-Exclusive users(can be multiple), available_queues is set to
 * NVGPU_NVS_CTRL_FIFO_QUEUE_NUM_EVENT.
 *
 * Note that, even for multiple R/W users, only one user at a time
 * can exist as an exclusive user. Only exclusive users can create/destroy
 * queues of type 'NVGPU_NVS_CTRL_FIFO_QUEUE_NUM_CONTROL'
 */
struct nvgpu_nvs_ctrl_fifo_ioctl_query_scheduler_characteristics_args {
	/*
	 * Invalid domain scheduler.
	 * The value of 'domain_scheduler_implementation'
	 * when 'has_domain_scheduler_control_fifo' is 0.
	 */
#define NVGPU_NVS_DOMAIN_SCHED_INVALID 0U
	/*
	 * CPU based scheduler implementation. Intended use is mainly
	 * for debug and testing purposes. Doesn't meet latency requirements.
	 * Implementation will be supported in the initial versions and eventually
	 * discarded.
	 */
#define NVGPU_NVS_DOMAIN_SCHED_KMD 1U
	/*
	 * GSP based scheduler implementation that meets latency requirements.
	 * This implementation will eventually replace NVGPU_NVS_DOMAIN_SCHED_KMD.
	 */
#define NVGPU_NVS_DOMAIN_SCHED_GSP 2U
	/*
	 * - Out: Value is expected to be among the above available flags.
	 */
	__u8 domain_scheduler_implementation;

	/* Must be 0 */
	__u8 reserved0;

	/* Must be 0 */
	__u16 reserved1;

	/* - Out: Mask of supported queue nums. */
	__u32 available_queues;

	/* Must be 0. */
	__u64 reserved2[8];
};

#define NVGPU_NVS_CTRL_FIFO_IOCTL_CREATE_QUEUE		\
	_IOWR(NVGPU_NVS_CTRL_FIFO_IOCTL_MAGIC, 1,	\
	       struct nvgpu_nvs_ctrl_fifo_ioctl_create_queue_args)
#define NVGPU_NVS_CTRL_FIFO_IOCTL_RELEASE_QUEUE		\
	_IOWR(NVGPU_NVS_CTRL_FIFO_IOCTL_MAGIC, 2,	\
	       struct nvgpu_nvs_ctrl_fifo_ioctl_release_queue_args)
#define NVGPU_NVS_CTRL_FIFO_IOCTL_ENABLE_EVENT		\
	_IOW(NVGPU_NVS_CTRL_FIFO_IOCTL_MAGIC, 3,	\
	       struct nvgpu_nvs_ctrl_fifo_ioctl_event)
#define NVGPU_NVS_CTRL_FIFO_IOCTL_QUERY_SCHEDULER_CHARACTERISTICS	\
	_IOR(NVGPU_NVS_CTRL_FIFO_IOCTL_MAGIC, 4,	\
	       struct nvgpu_nvs_ctrl_fifo_ioctl_query_scheduler_characteristics_args)
#define NVGPU_NVS_CTRL_FIFO_IOCTL_LAST				\
	_IOC_NR(NVGPU_NVS_CTRL_FIFO_IOCTL_QUERY_SCHEDULER_CHARACTERISTICS)
#define NVGPU_NVS_CTRL_FIFO_IOCTL_MAX_ARG_SIZE			\
	sizeof(struct nvgpu_nvs_ctrl_fifo_ioctl_query_scheduler_characteristics_args)

#endif
