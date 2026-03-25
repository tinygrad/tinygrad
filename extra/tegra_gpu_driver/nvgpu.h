/*
 * NVGPU Public Interface Header
 *
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef _UAPI__LINUX_NVGPU_IOCTL_H
#define _UAPI__LINUX_NVGPU_IOCTL_H

#include "nvgpu-ctrl.h"
#include "nvgpu-event.h"
#include "nvgpu-as.h"

#include "nvgpu-nvs.h"

/*
 * /dev/nvhost-tsg-gpu device
 *
 * Opening a '/dev/nvhost-tsg-gpu' device node creates a way to
 * bind/unbind a channel to/from TSG group
 */

#define NVGPU_TSG_IOCTL_MAGIC 'T'

struct nvgpu_tsg_bind_channel_ex_args {
	/* in: channel fd */
	__s32 channel_fd;

	/* in: VEID in Volta */
	__u32 subcontext_id;
	__u8 reserved[16];
};

struct nvgpu_tsg_bind_scheduling_domain_args {
	/* in: id of the domain this tsg will be bound to */
	__s32 domain_fd;
	/* Must be set to 0 */
	__s32 reserved0;
	/* Must be set to 0 */
	__u64 reserved[3];
};

/*
 * This struct helps to report the SM error state of a single SM.
 * This acts upon the currently resident TSG context.
 * Global Error status register
 * Warp Error status register
 * Warp Error status register PC
 * Global Error status register Report Mask
 * Warp Error status register Report Mask
 */
struct nvgpu_tsg_sm_error_state_record {
	__u32 global_esr;
	__u32 warp_esr;
	__u64 warp_esr_pc;
	__u32 global_esr_report_mask;
	__u32 warp_esr_report_mask;
};

/*
 * This struct helps to read the SM error state.
 */
struct nvgpu_tsg_read_single_sm_error_state_args {
	/* Valid SM ID */
	__u32 sm_id;
	__u32 reserved;
	/*
	 * This is pointer to the struct nvgpu_tsg_sm_error_state_record
	 */
	__u64 record_mem;
	/* size of the record size to read */
	__u64 record_size;
};

/*
 * This struct helps to read SM error states for all the SMs
 */
struct nvgpu_tsg_read_all_sm_error_state_args {
	/*
	 * in: Number of SM error states to be returned. Must be equal to the number of SMs.
	 */
	__u32 num_sm;
	/*
	 * Padding to make KMD UAPI compatible with both 32-bit and 64-bit callers.
	 */
	__u32 reserved;
	/*
	 * out: This points to an array of nvgpu_tsg_read_single_sm_error_state_args.
	 */
	__u64 buffer_mem;
	/* in: size of the buffer to store error states */
	__u64 buffer_size;
};

/*
 * This struct is used to read and configure l2 max evict_last
 * setting.
 */
struct nvgpu_tsg_l2_max_ways_evict_last_args {
	/*
	 * Maximum number of ways in a l2 cache set that can be allocated
	 * with eviction_policy=EVICT_LAST
	 */
	__u32 max_ways;
	__u32 reserved;
};

/*
 * This struct contains the parameter for configuring L2 sector promotion.
 * It supports 3 valid options:-
 *  - PROMOTE_NONE(1): cache-miss doens't get promoted.
 *  - PROMOTE_64B(2): cache-miss gets promoted to 64 bytes if less than 64 bytes.
 *  - PROMOTE_128B(4): cache-miss gets promoted to 128 bytes if less than 128 bytes.
 */
#define NVGPU_GPU_IOCTL_TSG_L2_SECTOR_PROMOTE_FLAG_NONE		(1U << 0U)
#define NVGPU_GPU_IOCTL_TSG_L2_SECTOR_PROMOTE_FLAG_64B		(1U << 1U)
#define NVGPU_GPU_IOCTL_TSG_L2_SECTOR_PROMOTE_FLAG_128B		(1U << 2U)
struct nvgpu_tsg_set_l2_sector_promotion_args {
	/* Valid promotion flag */
	__u32 promotion_flag;
	__u32 reserved;
};

/*
 * NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT - create a subcontext within a TSG.
 *
 * This IOCTL creates a subcontext of specified type within TSG if available
 * and returns VEID for it. Subcontext is associated with the user supplied
 * address space.
 *
 * return 0 on success, -1 on error.
 * retval EINVAL if invalid parameters are specified.
 * retval ENOSPC if subcontext of specified type can't be allocated.
 */

/* Subcontext types */

/*
 * Synchronous subcontext. Subcontext of this type may hold the
 * graphics channel, and multiple copy engine and compute channels.
 */
#define NVGPU_TSG_SUBCONTEXT_TYPE_SYNC               (0x0U)

/*
 * Asynchronous subcontext. Asynchronous subcontext is for compute
 * and copy engine channels only.
 */
#define NVGPU_TSG_SUBCONTEXT_TYPE_ASYNC              (0x1U)

/* Arguments for NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT */
struct nvgpu_tsg_create_subcontext_args {
	/* in: subcontext type */
	__u32 type;

	/* in: address space fd */
	__s32 as_fd;

	/*
	 * out: VEID for the subcontext
	 *      This is HW specific constant. For Volta+ chips (up to T234
	 *      Legacy mode), the HW-specific range is [0..63]. Of these,
	 *      0th subcontext is SYNC subcontext and remaining are
	 *      ASYNC subcontexts.
	 *      For SMC case, less number of VEIDs are supported per GPU
	 *      instance. This value is SMC engine relative VEID.
	 */
	__u32 veid;
	__u32 reserved;
};

/*
 * NVGPU_TSG_IOCTL_DELETE_SUBCONTEXT - delete a subcontext within a TSG.
 *
 * This IOCTL deletes a subcontext with specified subcontext id.
 *
 * return 0 on success, -1 on error.
 * retval EINVAL if invalid parameters are specified.
 */

/* Arguments for NVGPU_TSG_IOCTL_DELETE_SUBCONTEXT */
struct nvgpu_tsg_delete_subcontext_args {
	/* in: VEID for the subcontext */
	__u32 veid;
	__u32 reserved;
};

/*
 * NVGPU_TSG_IOCTL_GET_SHARE_TOKEN - get TSG share token ioctl.
 *
 * This IOCTL creates a TSG share token. The TSG share token may be
 * specified when opening new TSG fds with NVGPU_GPU_IOCTL_OPEN_TSG.
 * In this case, the underlying HW TSG will be shared.
 *
 * The source_device_instance_id is checked to be authorized while
 * creating share token.
 *
 * TBD: TSG share tokens have an expiration time. This is controlled
 * by /sys/kernel/debug/<gpu>/tsg_share_token_timeout_ms. The default
 * timeout is 30000 ms on silicon platforms and 0 (= no timeout) on
 * pre-silicon platforms.
 *
 * When the TSG is closed, all share tokens on the TSG will be
 * invalidated. Share token can also be invalidated by calling
 * the ioctl NVGPU_TSG_IOCTL_REVOKE_SHARE_TOKEN.
 *
 * When creating a TSG fd with share token, it is ensured that
 * target_device_instance_id specified here matches with the
 * device instance id that is used to call the ioctl
 * NVGPU_GPU_IOCTL_OPEN_TSG. This will secure the sharing of
 * the TSG within trusted parties.
 *
 * Note: share token is unique in the context of the source and
 * target device instance ids for a TSG.
 *
 * return 0 on success, -1 on error.
 * retval EINVAL if invalid parameters are specified.
 *               (if any of the device id argument is zero or
 *                unauthorized device).
 */

/* Arguments for NVGPU_TSG_IOCTL_GET_SHARE_TOKEN */
struct nvgpu_tsg_get_share_token_args {
	/* in: Source (exporter) device id */
	__u64 source_device_instance_id;

	/* in: Target (importer) device id */
	__u64 target_device_instance_id;

	/* out: Share token */
	__u64 share_token;
};

/*
 * NVGPU_TSG_IOCTL_REVOKE_SHARE_TOKEN - revoke TSG share token ioctl.
 *
 * This IOCTL revokes a TSG share token. It is intended to be called
 * whe the share token data exchange with the other end point is
 * unsuccessful.
 *
 * return 0 on success, -1 on error.
 * retval EINVAL if invalid parameters are specified.
 *               (if any of the argument is zero or unauthorized device).
 * retval EINVAL if share token doesn't exist or is expired.
 */

/* Arguments for NVGPU_TSG_IOCTL_REVOKE_SHARE_TOKEN */
struct nvgpu_tsg_revoke_share_token_args {
	/* in: Source (exporter) device id */
	__u64 source_device_instance_id;

	/* in: Target (importer) device id */
	__u64 target_device_instance_id;

	/* in: Share token */
	__u64 share_token;
};

#define NVGPU_TSG_IOCTL_BIND_CHANNEL \
	_IOW(NVGPU_TSG_IOCTL_MAGIC, 1, int)
#define NVGPU_TSG_IOCTL_UNBIND_CHANNEL \
	_IOW(NVGPU_TSG_IOCTL_MAGIC, 2, int)
#define NVGPU_IOCTL_TSG_ENABLE \
	_IO(NVGPU_TSG_IOCTL_MAGIC, 3)
#define NVGPU_IOCTL_TSG_DISABLE \
	_IO(NVGPU_TSG_IOCTL_MAGIC, 4)
#define NVGPU_IOCTL_TSG_PREEMPT \
	_IO(NVGPU_TSG_IOCTL_MAGIC, 5)
#define NVGPU_IOCTL_TSG_EVENT_ID_CTRL \
	_IOWR(NVGPU_TSG_IOCTL_MAGIC, 7, struct nvgpu_event_id_ctrl_args)
#define NVGPU_IOCTL_TSG_SET_RUNLIST_INTERLEAVE \
	_IOW(NVGPU_TSG_IOCTL_MAGIC, 8, struct nvgpu_runlist_interleave_args)
#define NVGPU_IOCTL_TSG_SET_TIMESLICE \
	_IOW(NVGPU_TSG_IOCTL_MAGIC, 9, struct nvgpu_timeslice_args)
#define NVGPU_IOCTL_TSG_GET_TIMESLICE \
	_IOR(NVGPU_TSG_IOCTL_MAGIC, 10, struct nvgpu_timeslice_args)
#define NVGPU_TSG_IOCTL_BIND_CHANNEL_EX \
	_IOWR(NVGPU_TSG_IOCTL_MAGIC, 11, struct nvgpu_tsg_bind_channel_ex_args)
#define NVGPU_TSG_IOCTL_READ_SINGLE_SM_ERROR_STATE \
	_IOWR(NVGPU_TSG_IOCTL_MAGIC, 12, \
			struct nvgpu_tsg_read_single_sm_error_state_args)
#define NVGPU_TSG_IOCTL_SET_L2_MAX_WAYS_EVICT_LAST \
	_IOW(NVGPU_TSG_IOCTL_MAGIC, 13, \
			struct nvgpu_tsg_l2_max_ways_evict_last_args)
#define NVGPU_TSG_IOCTL_GET_L2_MAX_WAYS_EVICT_LAST \
	_IOR(NVGPU_TSG_IOCTL_MAGIC, 14, \
			struct nvgpu_tsg_l2_max_ways_evict_last_args)
#define NVGPU_TSG_IOCTL_SET_L2_SECTOR_PROMOTION \
	_IOW(NVGPU_TSG_IOCTL_MAGIC, 15, \
			struct nvgpu_tsg_set_l2_sector_promotion_args)
#define NVGPU_TSG_IOCTL_BIND_SCHEDULING_DOMAIN \
	_IOW(NVGPU_TSG_IOCTL_MAGIC, 16, \
			struct nvgpu_tsg_bind_scheduling_domain_args)
#define NVGPU_TSG_IOCTL_READ_ALL_SM_ERROR_STATES \
	_IOWR(NVGPU_TSG_IOCTL_MAGIC, 17, \
			struct nvgpu_tsg_read_all_sm_error_state_args)
#define NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT \
	_IOWR(NVGPU_TSG_IOCTL_MAGIC, 18, \
			struct nvgpu_tsg_create_subcontext_args)
#define NVGPU_TSG_IOCTL_DELETE_SUBCONTEXT \
	_IOW(NVGPU_TSG_IOCTL_MAGIC, 19, \
			struct nvgpu_tsg_delete_subcontext_args)
#define NVGPU_TSG_IOCTL_GET_SHARE_TOKEN \
	_IOWR(NVGPU_TSG_IOCTL_MAGIC, 20, \
			struct nvgpu_tsg_get_share_token_args)
#define NVGPU_TSG_IOCTL_REVOKE_SHARE_TOKEN \
	_IOW(NVGPU_TSG_IOCTL_MAGIC, 21, \
			struct nvgpu_tsg_revoke_share_token_args)
#define NVGPU_TSG_IOCTL_MAX_ARG_SIZE	\
		sizeof(struct nvgpu_tsg_bind_scheduling_domain_args)

#define NVGPU_TSG_IOCTL_LAST		\
	_IOC_NR(NVGPU_TSG_IOCTL_REVOKE_SHARE_TOKEN)

/*
 * /dev/nvhost-dbg-gpu device
 *
 * Opening a '/dev/nvhost-dbg-gpu' device node creates a new debugger
 * session.  nvgpu channels (for the same module) can then be bound to such a
 * session.
 *
 * One nvgpu channel can also be bound to multiple debug sessions
 *
 * As long as there is an open device file to the session, or any bound
 * nvgpu channels it will be valid.  Once all references to the session
 * are removed the session is deleted.
 *
 */

#define NVGPU_DBG_GPU_IOCTL_MAGIC 'D'

/*
 * Binding/attaching a debugger session to an nvgpu channel
 *
 * The 'channel_fd' given here is the fd used to allocate the
 * gpu channel context.
 *
 */
struct nvgpu_dbg_gpu_bind_channel_args {
	__u32 channel_fd; /* in */
	__u32 _pad0[1];
};

#define NVGPU_DBG_GPU_IOCTL_BIND_CHANNEL				\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 1, struct nvgpu_dbg_gpu_bind_channel_args)

/*
 * Register operations
 * All operations are targeted towards first channel
 * attached to debug session
 */
/* valid op values */
#define NVGPU_DBG_GPU_REG_OP_READ_32                             (0x00000000)
#define NVGPU_DBG_GPU_REG_OP_WRITE_32                            (0x00000001)
#define NVGPU_DBG_GPU_REG_OP_READ_64                             (0x00000002)
#define NVGPU_DBG_GPU_REG_OP_WRITE_64                            (0x00000003)
/* note: 8b ops are unsupported */
#define NVGPU_DBG_GPU_REG_OP_READ_08                             (0x00000004)
#define NVGPU_DBG_GPU_REG_OP_WRITE_08                            (0x00000005)

/* valid type values */
#define NVGPU_DBG_GPU_REG_OP_TYPE_GLOBAL                         (0x00000000)
#define NVGPU_DBG_GPU_REG_OP_TYPE_GR_CTX                         (0x00000001)
#define NVGPU_DBG_GPU_REG_OP_TYPE_GR_CTX_TPC                     (0x00000002)
#define NVGPU_DBG_GPU_REG_OP_TYPE_GR_CTX_SM                      (0x00000004)
#define NVGPU_DBG_GPU_REG_OP_TYPE_GR_CTX_CROP                    (0x00000008)
#define NVGPU_DBG_GPU_REG_OP_TYPE_GR_CTX_ZROP                    (0x00000010)
/*#define NVGPU_DBG_GPU_REG_OP_TYPE_FB                           (0x00000020)*/
#define NVGPU_DBG_GPU_REG_OP_TYPE_GR_CTX_QUAD                    (0x00000040)

/* valid status values */
#define NVGPU_DBG_GPU_REG_OP_STATUS_SUCCESS                      (0x00000000)
#define NVGPU_DBG_GPU_REG_OP_STATUS_INVALID_OP                   (0x00000001)
#define NVGPU_DBG_GPU_REG_OP_STATUS_INVALID_TYPE                 (0x00000002)
#define NVGPU_DBG_GPU_REG_OP_STATUS_INVALID_OFFSET               (0x00000004)
#define NVGPU_DBG_GPU_REG_OP_STATUS_UNSUPPORTED_OP               (0x00000008)
#define NVGPU_DBG_GPU_REG_OP_STATUS_INVALID_MASK                 (0x00000010)

struct nvgpu_dbg_gpu_reg_op {
	__u8    op;
	__u8    type;
	__u8    status;
	__u8    quad;
	__u32   group_mask;
	__u32   sub_group_mask;
	__u32   offset;
	__u32   value_lo;
	__u32   value_hi;
	__u32   and_n_mask_lo;
	__u32   and_n_mask_hi;
};

struct nvgpu_dbg_gpu_exec_reg_ops_args {
	__u64 ops; /* pointer to nvgpu_reg_op operations */
	__u32 num_ops;
	__u32 gr_ctx_resident;
};

#define NVGPU_DBG_GPU_IOCTL_REG_OPS					\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 2, struct nvgpu_dbg_gpu_exec_reg_ops_args)

/* Enable/disable/clear event notifications */
struct nvgpu_dbg_gpu_events_ctrl_args {
	__u32 cmd; /* in */
	__u32 _pad0[1];
};

/* valid event ctrl values */
#define NVGPU_DBG_GPU_EVENTS_CTRL_CMD_DISABLE                    (0x00000000)
#define NVGPU_DBG_GPU_EVENTS_CTRL_CMD_ENABLE                     (0x00000001)
#define NVGPU_DBG_GPU_EVENTS_CTRL_CMD_CLEAR                      (0x00000002)

#define NVGPU_DBG_GPU_IOCTL_EVENTS_CTRL				\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 3, struct nvgpu_dbg_gpu_events_ctrl_args)


/* Powergate/Unpowergate control */

#define NVGPU_DBG_GPU_POWERGATE_MODE_ENABLE                                 1
#define NVGPU_DBG_GPU_POWERGATE_MODE_DISABLE                                2

struct nvgpu_dbg_gpu_powergate_args {
	__u32 mode;
};

#define NVGPU_DBG_GPU_IOCTL_POWERGATE					\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 4, struct nvgpu_dbg_gpu_powergate_args)


/* SMPC Context Switch Mode */
#define NVGPU_DBG_GPU_SMPC_CTXSW_MODE_NO_CTXSW               (0x00000000)
#define NVGPU_DBG_GPU_SMPC_CTXSW_MODE_CTXSW                  (0x00000001)

struct nvgpu_dbg_gpu_smpc_ctxsw_mode_args {
	__u32 mode;
};

#define NVGPU_DBG_GPU_IOCTL_SMPC_CTXSW_MODE \
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 5, struct nvgpu_dbg_gpu_smpc_ctxsw_mode_args)

/* Suspend /Resume SM control */
#define NVGPU_DBG_GPU_SUSPEND_ALL_SMS 1
#define NVGPU_DBG_GPU_RESUME_ALL_SMS  2

struct nvgpu_dbg_gpu_suspend_resume_all_sms_args {
	__u32 mode;
};

#define NVGPU_DBG_GPU_IOCTL_SUSPEND_RESUME_ALL_SMS			\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 6, struct nvgpu_dbg_gpu_suspend_resume_all_sms_args)

struct nvgpu_dbg_gpu_perfbuf_map_args {
	__u32 dmabuf_fd;	/* in */
	__u32 reserved;
	__u64 mapping_size;	/* in, size of mapped buffer region */
	__u64 offset;		/* out, virtual address of the mapping */
};

struct nvgpu_dbg_gpu_perfbuf_unmap_args {
	__u64 offset;
};

#define NVGPU_DBG_GPU_IOCTL_PERFBUF_MAP \
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 7, struct nvgpu_dbg_gpu_perfbuf_map_args)
#define NVGPU_DBG_GPU_IOCTL_PERFBUF_UNMAP \
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 8, struct nvgpu_dbg_gpu_perfbuf_unmap_args)

/* Enable/disable PC Sampling */
struct nvgpu_dbg_gpu_pc_sampling_args {
	__u32 enable;
	__u32 _pad0[1];
};

#define NVGPU_DBG_GPU_IOCTL_PC_SAMPLING_DISABLE	0
#define NVGPU_DBG_GPU_IOCTL_PC_SAMPLING_ENABLE	1

#define NVGPU_DBG_GPU_IOCTL_PC_SAMPLING \
	_IOW(NVGPU_DBG_GPU_IOCTL_MAGIC,  9, struct nvgpu_dbg_gpu_pc_sampling_args)

/* Enable/Disable timeouts */
#define NVGPU_DBG_GPU_IOCTL_TIMEOUT_ENABLE                                   1
#define NVGPU_DBG_GPU_IOCTL_TIMEOUT_DISABLE                                  0

struct nvgpu_dbg_gpu_timeout_args {
	__u32 enable;
	__u32 padding;
};

#define NVGPU_DBG_GPU_IOCTL_TIMEOUT \
	_IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 10, struct nvgpu_dbg_gpu_timeout_args)

#define NVGPU_DBG_GPU_IOCTL_GET_TIMEOUT \
	_IOR(NVGPU_DBG_GPU_IOCTL_MAGIC, 11, struct nvgpu_dbg_gpu_timeout_args)


struct nvgpu_dbg_gpu_set_next_stop_trigger_type_args {
	__u32 broadcast;
	__u32 reserved;
};

#define NVGPU_DBG_GPU_IOCTL_SET_NEXT_STOP_TRIGGER_TYPE			\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 12, struct nvgpu_dbg_gpu_set_next_stop_trigger_type_args)


/* PM Context Switch Mode */
/*This mode says that the pms are not to be context switched. */
#define NVGPU_DBG_GPU_HWPM_CTXSW_MODE_NO_CTXSW               (0x00000000)
/* This mode says that the pms in Mode-B are to be context switched */
#define NVGPU_DBG_GPU_HWPM_CTXSW_MODE_CTXSW                  (0x00000001)
/* This mode says that the pms in Mode-E (stream out) are to be context switched. */
#define NVGPU_DBG_GPU_HWPM_CTXSW_MODE_STREAM_OUT_CTXSW       (0x00000002)

struct nvgpu_dbg_gpu_hwpm_ctxsw_mode_args {
	__u32 mode;
	__u32 reserved;
};

#define NVGPU_DBG_GPU_IOCTL_HWPM_CTXSW_MODE \
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 13, struct nvgpu_dbg_gpu_hwpm_ctxsw_mode_args)


struct nvgpu_dbg_gpu_sm_error_state_record {
	__u32 hww_global_esr;
	__u32 hww_warp_esr;
	__u64 hww_warp_esr_pc;
	__u32 hww_global_esr_report_mask;
	__u32 hww_warp_esr_report_mask;

	/*
	 * Notes
	 * - This struct can be safely appended with new fields. However, always
	 *   keep the structure size multiple of 8 and make sure that the binary
	 *   layout does not change between 32-bit and 64-bit architectures.
	 */
};

struct nvgpu_dbg_gpu_read_single_sm_error_state_args {
	__u32 sm_id;
	__u32 padding;
	__u64 sm_error_state_record_mem;
	__u64 sm_error_state_record_size;
};

#define NVGPU_DBG_GPU_IOCTL_READ_SINGLE_SM_ERROR_STATE			\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 14, struct nvgpu_dbg_gpu_read_single_sm_error_state_args)


struct nvgpu_dbg_gpu_clear_single_sm_error_state_args {
	__u32 sm_id;
	__u32 padding;
};

#define NVGPU_DBG_GPU_IOCTL_CLEAR_SINGLE_SM_ERROR_STATE			\
	_IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 15, struct nvgpu_dbg_gpu_clear_single_sm_error_state_args)

/*
 * Unbinding/detaching a debugger session from a nvgpu channel
 *
 * The 'channel_fd' given here is the fd used to allocate the
 * gpu channel context.
 */
struct nvgpu_dbg_gpu_unbind_channel_args {
	__u32 channel_fd; /* in */
	__u32 _pad0[1];
};

#define NVGPU_DBG_GPU_IOCTL_UNBIND_CHANNEL				\
	_IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 17, struct nvgpu_dbg_gpu_unbind_channel_args)


#define NVGPU_DBG_GPU_SUSPEND_ALL_CONTEXTS	1
#define NVGPU_DBG_GPU_RESUME_ALL_CONTEXTS	2

struct nvgpu_dbg_gpu_suspend_resume_contexts_args {
	__u32 action;
	__u32 is_resident_context;
	__s32 resident_context_fd;
	__u32 padding;
};

#define NVGPU_DBG_GPU_IOCTL_SUSPEND_RESUME_CONTEXTS			\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 18, struct nvgpu_dbg_gpu_suspend_resume_contexts_args)


#define NVGPU_DBG_GPU_IOCTL_ACCESS_FB_MEMORY_CMD_READ	1
#define NVGPU_DBG_GPU_IOCTL_ACCESS_FB_MEMORY_CMD_WRITE	2

struct nvgpu_dbg_gpu_access_fb_memory_args {
	__u32 cmd;       /* in: either read or write */

	__s32 dmabuf_fd; /* in: dmabuf fd of the buffer in FB */
	__u64 offset;    /* in: offset within buffer in FB, should be 4B aligned */

	__u64 buffer;    /* in/out: temp buffer to read/write from */
	__u64 size;      /* in: size of the buffer i.e. size of read/write in bytes, should be 4B aligned */
};

#define NVGPU_DBG_GPU_IOCTL_ACCESS_FB_MEMORY	\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 19, struct nvgpu_dbg_gpu_access_fb_memory_args)

struct nvgpu_dbg_gpu_profiler_obj_mgt_args {
	__u32 profiler_handle;
	__u32 reserved;
};

#define NVGPU_DBG_GPU_IOCTL_PROFILER_ALLOCATE	\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 20, struct nvgpu_dbg_gpu_profiler_obj_mgt_args)

#define NVGPU_DBG_GPU_IOCTL_PROFILER_FREE	\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 21, struct nvgpu_dbg_gpu_profiler_obj_mgt_args)

struct nvgpu_dbg_gpu_profiler_reserve_args {
	__u32 profiler_handle;
	__u32 acquire;
};

#define NVGPU_DBG_GPU_IOCTL_PROFILER_RESERVE			\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 22, struct nvgpu_dbg_gpu_profiler_reserve_args)

/*
 * This struct helps to set the exception mask. If mask is not set
 * or set to NVGPU_DBG_GPU_IOCTL_SET_SM_EXCEPTION_TYPE_MASK_NONE
 * then kernel code will follow recovery path on sm exception.
 * If mask is set to NVGPU_DBG_GPU_IOCTL_SET_SM_EXCEPTION_TYPE_MASK_FATAL, then
 * kernel code will skip recovery path on sm exception.
 */
struct nvgpu_dbg_gpu_set_sm_exception_type_mask_args {
#define NVGPU_DBG_GPU_IOCTL_SET_SM_EXCEPTION_TYPE_MASK_NONE	(0x0U)
#define NVGPU_DBG_GPU_IOCTL_SET_SM_EXCEPTION_TYPE_MASK_FATAL	(0x1U << 0U)
	/* exception type mask value */
	__u32 exception_type_mask;
	__u32 reserved;
};

#define NVGPU_DBG_GPU_IOCTL_SET_SM_EXCEPTION_TYPE_MASK \
	_IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 23, \
			struct nvgpu_dbg_gpu_set_sm_exception_type_mask_args)

struct nvgpu_dbg_gpu_cycle_stats_args {
	__u32 dmabuf_fd;
	__u32 reserved;
};

#define NVGPU_DBG_GPU_IOCTL_CYCLE_STATS	\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 24, struct nvgpu_dbg_gpu_cycle_stats_args)

/* cycle stats snapshot buffer support for mode E */
struct nvgpu_dbg_gpu_cycle_stats_snapshot_args {
	__u32 cmd;		/* in: command to handle     */
	__u32 dmabuf_fd;	/* in: dma buffer handler    */
	__u32 extra;		/* in/out: extra payload e.g.*/
				/*    counter/start perfmon  */
	__u32 reserved;
};

/* valid commands to control cycle stats shared buffer */
#define NVGPU_DBG_GPU_IOCTL_CYCLE_STATS_SNAPSHOT_CMD_FLUSH   0
#define NVGPU_DBG_GPU_IOCTL_CYCLE_STATS_SNAPSHOT_CMD_ATTACH  1
#define NVGPU_DBG_GPU_IOCTL_CYCLE_STATS_SNAPSHOT_CMD_DETACH  2

#define NVGPU_DBG_GPU_IOCTL_CYCLE_STATS_SNAPSHOT	\
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 25, struct nvgpu_dbg_gpu_cycle_stats_snapshot_args)

/* MMU Debug Mode */
#define NVGPU_DBG_GPU_CTX_MMU_DEBUG_MODE_DISABLED	0
#define NVGPU_DBG_GPU_CTX_MMU_DEBUG_MODE_ENABLED	1

struct nvgpu_dbg_gpu_set_ctx_mmu_debug_mode_args {
	__u32 mode;
	__u32 reserved;
};
#define NVGPU_DBG_GPU_IOCTL_SET_CTX_MMU_DEBUG_MODE	\
	_IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 26, \
	struct nvgpu_dbg_gpu_set_ctx_mmu_debug_mode_args)

/* Get gr context size */
struct nvgpu_dbg_gpu_get_gr_context_size_args {
	__u32 size;
	__u32 reserved;
};

#define NVGPU_DBG_GPU_IOCTL_GET_GR_CONTEXT_SIZE \
	_IOR(NVGPU_DBG_GPU_IOCTL_MAGIC, 27, \
	struct nvgpu_dbg_gpu_get_gr_context_size_args)

/* Get gr context */
struct nvgpu_dbg_gpu_get_gr_context_args {
	__u64 buffer;    /* in/out: the output buffer containing contents of the gr context.
						buffer address is given by the user */
	__u32 size;      /* in: size of the context buffer */
	__u32 reserved;
};

#define NVGPU_DBG_GPU_IOCTL_GET_GR_CONTEXT \
	_IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 28, \
	struct nvgpu_dbg_gpu_get_gr_context_args)

#define NVGPU_DBG_GPU_IOCTL_TSG_SET_TIMESLICE \
	_IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 29, \
	struct nvgpu_timeslice_args)

#define NVGPU_DBG_GPU_IOCTL_TSG_GET_TIMESLICE \
	_IOR(NVGPU_DBG_GPU_IOCTL_MAGIC, 30, \
	struct nvgpu_timeslice_args)

struct nvgpu_dbg_gpu_get_mappings_entry {
	/* out: start of GPU VA for this mapping */
	__u64 gpu_va;
	/* out: size in bytes of this mapping */
	__u64 size;
};

struct nvgpu_dbg_gpu_get_mappings_args {
	/* in: lower VA range, inclusive */
	__u64 va_lo;
	/* in: upper VA range, exclusive */
	__u64 va_hi;
	/* in: Pointer to the struct nvgpu_dbg_gpu_get_mappings_entry. */
	__u64 ops_buffer;
	/*
	 * in: maximum number of the entries that ops_buffer may hold.
	 * out: number of entries written to ops_buffer.
	 * When ops_buffer is zero:
	 * out: number of mapping entries in range [va_lo, va_hi).
	 */
	__u32 count;
	/* out: Has more valid mappings in this range than count */
	__u8 has_more;
	__u8 reserved[3];
};

/* Maximum read/write ops supported in a single call */
#define NVGPU_DBG_GPU_IOCTL_ACCESS_GPUVA_CMD_READ 1U
#define NVGPU_DBG_GPU_IOCTL_ACCESS_GPUVA_CMD_WRITE 2U
struct nvgpu_dbg_gpu_va_access_entry {
	/* in: gpu_va address */
	__u64 gpu_va;
	/* in/out: Pointer to buffer through which data needs to be read/written */
	__u64 data;
	/* in: Access size in bytes */
	__u32 size;
	/* out: Whether the GpuVA is accessible */
	__u8 valid;
	__u8 reserved[3];
};

struct nvgpu_dbg_gpu_va_access_args {
	/* in/out: Pointer to the struct nvgpu_dbg_gpu_va_access_entry */
	__u64 ops_buf;
	/* in: Number of buffer ops */
	__u32 count;
	/* in: Access cmd Read/Write */
	__u8 cmd;
	__u8 reserved[3];
};

#define NVGPU_DBG_GPU_IOCTL_GET_MAPPINGS \
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 31, struct nvgpu_dbg_gpu_get_mappings_args)
#define NVGPU_DBG_GPU_IOCTL_ACCESS_GPU_VA \
	_IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 32, struct nvgpu_dbg_gpu_va_access_args)

/* Implicit ERRBAR Mode */
#define NVGPU_DBG_GPU_SCHED_EXIT_WAIT_FOR_ERRBAR_DISABLED	0
#define NVGPU_DBG_GPU_SCHED_EXIT_WAIT_FOR_ERRBAR_ENABLED	1

struct nvgpu_sched_exit_wait_for_errbar_args {
	__u32 enable; /* enable 1, disable 0*/
};

#define NVGPU_DBG_GPU_IOCTL_SET_SCHED_EXIT_WAIT_FOR_ERRBAR \
	_IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 33, \
	struct nvgpu_sched_exit_wait_for_errbar_args)

#define NVGPU_DBG_GPU_IOCTL_LAST		\
	_IOC_NR(NVGPU_DBG_GPU_IOCTL_SET_SCHED_EXIT_WAIT_FOR_ERRBAR)

#define NVGPU_DBG_GPU_IOCTL_MAX_ARG_SIZE		\
	sizeof(struct nvgpu_dbg_gpu_access_fb_memory_args)


/*
 * /dev/nvhost-prof-dev-gpu and /dev/nvhost-prof-ctx-gpu devices
 *
 * Opening a '/dev/nvhost-prof-*' device node creates a way to
 * open and manage a profiler object.
 */

#define NVGPU_PROFILER_IOCTL_MAGIC 'P'

struct nvgpu_profiler_bind_context_args {
	__s32 tsg_fd;		/* in: TSG file descriptor */
	__u32 reserved;
};

#define NVGPU_PROFILER_PM_RESOURCE_ARG_HWPM_LEGACY	0U
#define NVGPU_PROFILER_PM_RESOURCE_ARG_SMPC		1U
#define NVGPU_PROFILER_PM_RESOURCE_ARG_PC_SAMPLER	2U
#define NVGPU_PROFILER_PM_RESOURCE_ARG_HES_CWD		3U

struct nvgpu_profiler_reserve_pm_resource_args {
	__u32 resource; 	/* in: NVGPU_PROFILER_PM_RESOURCE_ARG_* resource to be reserved */

/* in: if ctxsw should be enabled for resource */
#define NVGPU_PROFILER_RESERVE_PM_RESOURCE_ARG_FLAG_CTXSW	(1 << 0)
	__u32 flags;

	__u32 reserved[2];
};

struct nvgpu_profiler_release_pm_resource_args {
	__u32 resource; 	/* in: NVGPU_PROFILER_PM_RESOURCE_ARG_* resource to be released */
	__u32 reserved;
};

struct nvgpu_profiler_alloc_pma_stream_args {
	__u64 pma_buffer_map_size;			/* in: PMA stream buffer size */
	__u64 pma_buffer_offset;			/* in: offset of PMA stream buffer */
	__u64 pma_buffer_va;				/* out: PMA stream buffer virtual address */
	__s32 pma_buffer_fd;				/* in: PMA stream buffer fd */

	__s32 pma_bytes_available_buffer_fd;		/* in: PMA available bytes buffer fd */

/* in: if ctxsw should be enabled for PMA channel */
#define NVGPU_PROFILER_ALLOC_PMA_STREAM_ARG_FLAG_CTXSW		(1 << 0)
	__u32 flags;

	__u32 pma_channel_id;				/* out: PMA hardware stream id */

	__u32 reserved[2];
};

struct nvgpu_profiler_free_pma_stream_args {
	__u32 pma_channel_id;				/* in: PMA hardware stream id */
	__u32 reserved[2];
};

struct nvgpu_profiler_pma_stream_update_get_put_args {
	__u64 bytes_consumed;		/* in: total bytes consumed by user since last update */
	__u64 bytes_available;		/* out: available bytes in PMA buffer for user to consume */

	__u64 put_ptr;			/* out: current PUT pointer to be returned */

/* in: if available bytes buffer should be updated */
#define NVGPU_PROFILER_PMA_STREAM_UPDATE_GET_PUT_ARG_FLAG_UPDATE_AVAILABLE_BYTES	(1 << 0)
/* in: if need to wait for available bytes buffer to get updated */
#define NVGPU_PROFILER_PMA_STREAM_UPDATE_GET_PUT_ARG_FLAG_WAIT_FOR_UPDATE		(1 << 1)
/* in: if current PUT pointer should be returned */
#define NVGPU_PROFILER_PMA_STREAM_UPDATE_GET_PUT_ARG_FLAG_RETURN_PUT_PTR		(1 << 2)
/* out: if PMA stream buffer overflow was triggered */
#define NVGPU_PROFILER_PMA_STREAM_UPDATE_GET_PUT_ARG_FLAG_OVERFLOW_TRIGGERED		(1 << 3)
	__u32 flags;

	__u32 pma_channel_id;			/* in: PMA channel index */

	__u32 reserved[2];
};

/*
 * MODE_ALL_OR_NONE
 * Reg_ops execution will bail out if any of the reg_op is not valid
 * or if there is any other error such as failure to access context image.
 * Subsequent reg_ops will not be executed and nvgpu_profiler_reg_op.status
 * will not be populated for them.
 * IOCTL will always return error for all of the errors.
 */
#define NVGPU_PROFILER_EXEC_REG_OPS_ARG_MODE_ALL_OR_NONE	0U
/*
 * MODE_CONTINUE_ON_ERROR
 * This mode allows continuing reg_ops execution even if some of the
 * reg_ops are not valid. Invalid reg_ops will be skipped and valid
 * ones will be executed.
 * IOCTL will return error only if there is some other severe failure
 * such as failure to access context image.
 * If any of the reg_op is invalid, or if didn't pass, it will be
 * reported via NVGPU_PROFILER_EXEC_REG_OPS_ARG_FLAG_ALL_PASSED flag.
 * IOCTL will return success in such cases.
 */
#define NVGPU_PROFILER_EXEC_REG_OPS_ARG_MODE_CONTINUE_ON_ERROR	1U

struct nvgpu_profiler_reg_op {
	__u8    op;		/* Operation in the form NVGPU_DBG_GPU_REG_OP_READ/WRITE_* */
	__u8    status;		/* Status in the form NVGPU_DBG_GPU_REG_OP_STATUS_* */
	__u32   offset;
	__u64   value;
	__u64   and_n_mask;
};

struct nvgpu_profiler_exec_reg_ops_args {
	__u32 mode;		/* in: operation mode NVGPU_PROFILER_EXEC_REG_OPS_ARG_MODE_* */

	__u32 count;		/* in: number of reg_ops operations,
				 *     upper limit nvgpu_gpu_characteristics.reg_ops_limit
				 */

	__u64 ops;		/* in/out: pointer to actual operations nvgpu_profiler_reg_op */

/* out: if all reg_ops passed, valid only for MODE_CONTINUE_ON_ERROR */
#define NVGPU_PROFILER_EXEC_REG_OPS_ARG_FLAG_ALL_PASSED		(1 << 0)
/* out: if the operations were performed directly on HW or in context image */
#define NVGPU_PROFILER_EXEC_REG_OPS_ARG_FLAG_DIRECT_OPS		(1 << 1)
	__u32 flags;

	__u32 reserved[3];
};

struct nvgpu_profiler_vab_range_checker {

	/*
	 * in: starting physical address. Must be aligned by
	 *     1 << (granularity_shift + bitmask_size_shift) where
	 *     bitmask_size_shift is a HW specific constant.
	 */
	__u64 start_phys_addr;

	/* in: log2 of coverage granularity per bit */
	__u8  granularity_shift;

	__u8  reserved[7];
};

/* Range checkers track all accesses (read and write) */
#define NVGPU_PROFILER_VAB_RANGE_CHECKER_MODE_ACCESS  1U
/* Range checkers track writes (writes and read-modify-writes) */
#define NVGPU_PROFILER_VAB_RANGE_CHECKER_MODE_DIRTY   2U

struct nvgpu_profiler_vab_reserve_args {

	/* in: range checker mode */
	__u8 vab_mode;

	__u8 reserved[3];

	/* in: number of range checkers, must match with the HW */
	__u32 num_range_checkers;

	/*
	 * in: range checker parameters. Pointer to array of
	 *     nvgpu_profiler_vab_range_checker elements
	 */
	__u64 range_checkers_ptr;
};

struct nvgpu_profiler_vab_flush_state_args {
	__u64 buffer_ptr;	/* in: usermode pointer to receive the
				 * VAB state buffer */
	__u64 buffer_size;	/* in: VAB buffer size. Must match
				 * with the hardware VAB state size */
};

#define NVGPU_PROFILER_IOCTL_BIND_CONTEXT \
	_IOW(NVGPU_PROFILER_IOCTL_MAGIC, 1, struct nvgpu_profiler_bind_context_args)
#define NVGPU_PROFILER_IOCTL_RESERVE_PM_RESOURCE \
	_IOW(NVGPU_PROFILER_IOCTL_MAGIC, 2, struct nvgpu_profiler_reserve_pm_resource_args)
#define NVGPU_PROFILER_IOCTL_RELEASE_PM_RESOURCE \
	_IOW(NVGPU_PROFILER_IOCTL_MAGIC, 3, struct nvgpu_profiler_release_pm_resource_args)
#define NVGPU_PROFILER_IOCTL_ALLOC_PMA_STREAM \
	_IOWR(NVGPU_PROFILER_IOCTL_MAGIC, 4, struct nvgpu_profiler_alloc_pma_stream_args)
#define NVGPU_PROFILER_IOCTL_FREE_PMA_STREAM \
	_IOW(NVGPU_PROFILER_IOCTL_MAGIC, 5, struct nvgpu_profiler_free_pma_stream_args)
#define NVGPU_PROFILER_IOCTL_BIND_PM_RESOURCES \
	_IO(NVGPU_PROFILER_IOCTL_MAGIC, 6)
#define NVGPU_PROFILER_IOCTL_UNBIND_PM_RESOURCES \
	_IO(NVGPU_PROFILER_IOCTL_MAGIC, 7)
#define NVGPU_PROFILER_IOCTL_PMA_STREAM_UPDATE_GET_PUT \
	_IOWR(NVGPU_PROFILER_IOCTL_MAGIC, 8, struct nvgpu_profiler_pma_stream_update_get_put_args)
#define NVGPU_PROFILER_IOCTL_EXEC_REG_OPS \
	_IOWR(NVGPU_PROFILER_IOCTL_MAGIC, 9, struct nvgpu_profiler_exec_reg_ops_args)
#define NVGPU_PROFILER_IOCTL_UNBIND_CONTEXT \
	_IO(NVGPU_PROFILER_IOCTL_MAGIC, 10)
#define NVGPU_PROFILER_IOCTL_VAB_RESERVE \
	_IOW(NVGPU_PROFILER_IOCTL_MAGIC, 11, struct nvgpu_profiler_vab_reserve_args)
#define NVGPU_PROFILER_IOCTL_VAB_RELEASE \
	_IO(NVGPU_PROFILER_IOCTL_MAGIC, 12)
#define NVGPU_PROFILER_IOCTL_VAB_FLUSH_STATE \
	_IOW(NVGPU_PROFILER_IOCTL_MAGIC, 13, struct nvgpu_profiler_vab_flush_state_args)
#define NVGPU_PROFILER_IOCTL_MAX_ARG_SIZE	\
		sizeof(struct nvgpu_profiler_alloc_pma_stream_args)
#define NVGPU_PROFILER_IOCTL_LAST		\
	_IOC_NR(NVGPU_PROFILER_IOCTL_VAB_FLUSH_STATE)


/*
 * /dev/nvhost-gpu device
 */

#define NVGPU_IOCTL_MAGIC 'H'
#define NVGPU_NO_TIMEOUT ((__u32)~0U)
#define NVGPU_TIMEOUT_FLAG_DISABLE_DUMP		0

/* this is also the hardware memory format */
struct nvgpu_gpfifo {
	__u32 entry0; /* first word of gpfifo entry */
	__u32 entry1; /* second word of gpfifo entry */
};

struct nvgpu_get_param_args {
	__u32 value;
};

struct nvgpu_channel_open_args {
	union {
		__s32 channel_fd; /* deprecated: use out.channel_fd instead */
		struct {
			 /* runlist_id is the runlist for the
			  * channel. Basically, the runlist specifies the target
			  * engine(s) for which the channel is
			  * opened. Runlist_id -1 is synonym for the primary
			  * graphics runlist. */
			__s32 runlist_id;
		} in;
		struct {
			__s32 channel_fd;
		} out;
	};
};

struct nvgpu_set_nvmap_fd_args {
	__u32 fd;
};

#define NVGPU_ALLOC_OBJ_FLAGS_LOCKBOOST_ZERO	(1 << 0)
/* Flags in nvgpu_alloc_obj_ctx_args.flags */
#define NVGPU_ALLOC_OBJ_FLAGS_GFXP		(1 << 1)
#define NVGPU_ALLOC_OBJ_FLAGS_CILP		(1 << 2)

struct nvgpu_alloc_obj_ctx_args {
	__u32 class_num; /* kepler3d, 2d, compute, etc       */
	__u32 flags;     /* input, output */
	__u64 obj_id;    /* output, used to free later       */
};

/* Deprecated. Use the SETUP_BIND IOCTL instead. */
struct nvgpu_alloc_gpfifo_ex_args {
	__u32 num_entries;
	__u32 num_inflight_jobs;
#define NVGPU_ALLOC_GPFIFO_EX_FLAGS_VPR_ENABLED		(1 << 0)
#define NVGPU_ALLOC_GPFIFO_EX_FLAGS_DETERMINISTIC	(1 << 1)
	__u32 flags;
	__u32 reserved[5];
};

/*
 * Setup the channel and bind it (enable).
 */
struct nvgpu_channel_setup_bind_args {
/*
 * Must be power of 2. Max value U32_MAX/8 (size of gpfifo entry) rounded of to
 * nearest lower power of 2 i.e. 2^28. The lower limit is due to the fact that
 * the last entry of gpfifo is kept empty and used to determine buffer empty or
 * full condition. Additionally, kmd submit uses pre/post sync which needs
 * another extra entry.
 * Range: 2, 4, 8, ..., 2^28 when
 *        NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_SUPPORT is set.
 * Range: 4, 8, 16, ..., 2^28 otherwise.
 */
	__u32 num_gpfifo_entries;
	__u32 num_inflight_jobs;
/* Set owner channel of this gpfifo as a vpr channel. */
#define NVGPU_CHANNEL_SETUP_BIND_FLAGS_VPR_ENABLED		(1 << 0)
/*
 * Channel shall exhibit deterministic behavior in the submit path.
 *
 * NOTE: as an exception, VPR resize may still cause the GPU to reset at any
 * time, which is not deterministic behavior. If this is not acceptable, the
 * user has to make sure that VPR resize does not occur.
 *
 * With this flag, any submits with in-kernel job tracking also require that
 * num_inflight_jobs is nonzero, and additionally that
 * NVGPU_GPU_FLAGS_SUPPORT_DETERMINISTIC_SUBMIT_FULL is found in gpu
 * characteristics.flags.
 *
 * Note that fast submits (with no in-kernel job tracking) are also
 * deterministic and are supported if the characteristics flags contain
 * NVGPU_GPU_FLAGS_SUPPORT_DETERMINISTIC_SUBMIT_NO_JOBTRACKING; this flag or
 * num_inflight_jobs are not necessary in that case.
 */
#define NVGPU_CHANNEL_SETUP_BIND_FLAGS_DETERMINISTIC	(1 << 1)
/* enable replayable gmmu faults for this channel */
#define NVGPU_CHANNEL_SETUP_BIND_FLAGS_REPLAYABLE_FAULTS_ENABLE   (1 << 2)
/*
 * Enable usermode submits on this channel.
 *
 * Submits in usermode are supported in some environments. If supported and
 * this flag is set + USERD and GPFIFO buffers are provided here, a submit
 * token is passed back to be written in the doorbell register in the usermode
 * region to notify the GPU for new work on this channel. Usermode and
 * kernelmode submit modes are mutually exclusive; by passing this flag, the
 * SUBMIT_GPFIFO IOCTL cannot be used.
 */
#define NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_SUPPORT	(1 << 3)
/*
 * Map usermode resources to GPU and return GPU VAs to userspace.
 */
#define NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_GPU_MAP_RESOURCES_SUPPORT	(1 << 4)
	__u32 flags;
	__s32 userd_dmabuf_fd;	/* in */
	__s32 gpfifo_dmabuf_fd;	/* in */
	__u32 work_submit_token; /* out */
	__u64 userd_dmabuf_offset; /* in */
	__u64 gpfifo_dmabuf_offset; /* in */
	__u64 gpfifo_gpu_va; /* out */
	__u64 userd_gpu_va; /* out */
	__u64 usermode_mmio_gpu_va; /* out */
	__u32 reserved[9];
};

struct nvgpu_fence {
	__u32 id;        /* syncpoint id or sync fence fd */
	__u32 value;     /* syncpoint value (discarded when using sync fence) */
};

/* insert a wait on the fence before submitting gpfifo */
#define NVGPU_SUBMIT_GPFIFO_FLAGS_FENCE_WAIT	(1 << 0)
/* insert a fence update after submitting gpfifo and
   return the new fence for others to wait on */
#define NVGPU_SUBMIT_GPFIFO_FLAGS_FENCE_GET	(1 << 1)
/* choose between different gpfifo entry formats */
#define NVGPU_SUBMIT_GPFIFO_FLAGS_HW_FORMAT	(1 << 2)
/* interpret fence as a sync fence fd instead of raw syncpoint fence */
#define NVGPU_SUBMIT_GPFIFO_FLAGS_SYNC_FENCE	(1 << 3)
/* suppress WFI before fence trigger */
#define NVGPU_SUBMIT_GPFIFO_FLAGS_SUPPRESS_WFI	(1 << 4)
/* skip buffer refcounting during submit */
#define NVGPU_SUBMIT_GPFIFO_FLAGS_SKIP_BUFFER_REFCOUNTING	(1 << 5)

struct nvgpu_submit_gpfifo_args {
	__u64 gpfifo;
	__u32 num_entries;
	__u32 flags;
	struct nvgpu_fence fence;
};

struct nvgpu_wait_args {
#define NVGPU_WAIT_TYPE_NOTIFIER	0x0
#define NVGPU_WAIT_TYPE_SEMAPHORE	0x1
	__u32 type;
	__u32 timeout;
	union {
		struct {
			/* handle and offset for notifier memory */
			__u32 dmabuf_fd;
			__u32 offset;
			__u32 padding1;
			__u32 padding2;
		} notifier;
		struct {
			/* handle and offset for semaphore memory */
			__u32 dmabuf_fd;
			__u32 offset;
			/* semaphore payload to wait for */
			__u32 payload;
			__u32 padding;
		} semaphore;
	} condition; /* determined by type field */
};

struct nvgpu_set_timeout_args {
	__u32 timeout;
};

struct nvgpu_set_timeout_ex_args {
	__u32 timeout;
	__u32 flags;
};

#define NVGPU_ZCULL_MODE_GLOBAL		0
#define NVGPU_ZCULL_MODE_NO_CTXSW		1
#define NVGPU_ZCULL_MODE_SEPARATE_BUFFER	2
#define NVGPU_ZCULL_MODE_PART_OF_REGULAR_BUF	3

struct nvgpu_zcull_bind_args {
	__u64 gpu_va;
	__u32 mode;
	__u32 padding;
};

struct nvgpu_set_error_notifier {
	__u64 offset;
	__u64 size;
	__u32 mem;
	__u32 padding;
};

struct nvgpu_notification {
	struct {			/* 0000- */
		__u32 nanoseconds[2];	/* nanoseconds since Jan. 1, 1970 */
	} time_stamp;			/* -0007 */
	__u32 info32;	/* info returned depends on method 0008-000b */
#define	NVGPU_CHANNEL_FIFO_ERROR_IDLE_TIMEOUT		8
#define	NVGPU_CHANNEL_GR_ERROR_SW_METHOD		12
#define	NVGPU_CHANNEL_GR_ERROR_SW_NOTIFY		13
#define	NVGPU_CHANNEL_GR_EXCEPTION			13
#define	NVGPU_CHANNEL_GR_SEMAPHORE_TIMEOUT		24
#define	NVGPU_CHANNEL_GR_ILLEGAL_NOTIFY			25
#define	NVGPU_CHANNEL_FIFO_ERROR_MMU_ERR_FLT		31
#define	NVGPU_CHANNEL_PBDMA_ERROR			32
#define NVGPU_CHANNEL_FECS_ERR_UNIMP_FIRMWARE_METHOD	37
#define	NVGPU_CHANNEL_RESETCHANNEL_VERIF_ERROR		43
#define	NVGPU_CHANNEL_PBDMA_PUSHBUFFER_CRC_MISMATCH	80
	__u16 info16;	/* info returned depends on method 000c-000d */
	__u16 status;	/* user sets bit 15, NV sets status 000e-000f */
#define	NVGPU_CHANNEL_SUBMIT_TIMEOUT		1
};

/* configure watchdog per-channel */
struct nvgpu_channel_wdt_args {
	__u32 wdt_status;
	__u32 timeout_ms;
};
#define NVGPU_IOCTL_CHANNEL_DISABLE_WDT		  (1 << 0)
#define NVGPU_IOCTL_CHANNEL_ENABLE_WDT		  (1 << 1)
#define NVGPU_IOCTL_CHANNEL_WDT_FLAG_SET_TIMEOUT  (1 << 2)
#define NVGPU_IOCTL_CHANNEL_WDT_FLAG_DISABLE_DUMP (1 << 3)

/*
 * Interleaving channels in a runlist is an approach to improve
 * GPU scheduling by allowing certain channels to appear multiple
 * times on the runlist. The number of times a channel appears is
 * governed by the following levels:
 *
 * low (L)   : appears once
 * medium (M): if L, appears L times
 *             else, appears once
 * high (H)  : if L, appears (M + 1) x L times
 *             else if M, appears M times
 *             else, appears once
 */
struct nvgpu_runlist_interleave_args {
	__u32 level;
	__u32 reserved;
};
#define NVGPU_RUNLIST_INTERLEAVE_LEVEL_LOW	0
#define NVGPU_RUNLIST_INTERLEAVE_LEVEL_MEDIUM	1
#define NVGPU_RUNLIST_INTERLEAVE_LEVEL_HIGH	2
#define NVGPU_RUNLIST_INTERLEAVE_NUM_LEVELS	3

/* controls how long a channel occupies an engine uninterrupted */
struct nvgpu_timeslice_args {
	__u32 timeslice_us;
	__u32 reserved;
};

struct nvgpu_event_id_ctrl_args {
	__u32 cmd; /* in */
	__u32 event_id; /* in */
	__s32 event_fd; /* out */
	__u32 padding;
};
#define NVGPU_IOCTL_CHANNEL_EVENT_ID_BPT_INT		0
#define NVGPU_IOCTL_CHANNEL_EVENT_ID_BPT_PAUSE		1
#define NVGPU_IOCTL_CHANNEL_EVENT_ID_BLOCKING_SYNC	2
#define NVGPU_IOCTL_CHANNEL_EVENT_ID_CILP_PREEMPTION_STARTED	3
#define NVGPU_IOCTL_CHANNEL_EVENT_ID_CILP_PREEMPTION_COMPLETE	4
#define NVGPU_IOCTL_CHANNEL_EVENT_ID_GR_SEMAPHORE_WRITE_AWAKEN	5
#define NVGPU_IOCTL_CHANNEL_EVENT_ID_MAX		6

#define NVGPU_IOCTL_CHANNEL_EVENT_ID_CMD_ENABLE		1

struct nvgpu_preemption_mode_args {
/* only one should be enabled at a time */
#define NVGPU_GRAPHICS_PREEMPTION_MODE_WFI              (1 << 0)
#define NVGPU_GRAPHICS_PREEMPTION_MODE_GFXP		(1 << 1)
	__u32 graphics_preempt_mode; /* in */

/* only one should be enabled at a time */
#define NVGPU_COMPUTE_PREEMPTION_MODE_WFI               (1 << 0)
#define NVGPU_COMPUTE_PREEMPTION_MODE_CTA               (1 << 1)
#define NVGPU_COMPUTE_PREEMPTION_MODE_CILP		(1 << 2)

	__u32 compute_preempt_mode; /* in */
};

struct nvgpu_boosted_ctx_args {
#define NVGPU_BOOSTED_CTX_MODE_NORMAL			(0U)
#define NVGPU_BOOSTED_CTX_MODE_BOOSTED_EXECUTION	(1U)
	__u32 boost;
	__u32 padding;
};

struct nvgpu_get_user_syncpoint_args {
	__u64 gpu_va;		/* out */
	__u32 syncpoint_id;	/* out */
	__u32 syncpoint_max;	/* out */
};

struct nvgpu_reschedule_runlist_args {
#define NVGPU_RESCHEDULE_RUNLIST_PREEMPT_NEXT           (1 << 0)
	__u32 flags;
};

#define NVGPU_IOCTL_CHANNEL_SET_NVMAP_FD	\
	_IOW(NVGPU_IOCTL_MAGIC, 5, struct nvgpu_set_nvmap_fd_args)
#define NVGPU_IOCTL_CHANNEL_SET_TIMEOUT	\
	_IOW(NVGPU_IOCTL_MAGIC, 11, struct nvgpu_set_timeout_args)
#define NVGPU_IOCTL_CHANNEL_GET_TIMEDOUT	\
	_IOR(NVGPU_IOCTL_MAGIC, 12, struct nvgpu_get_param_args)
#define NVGPU_IOCTL_CHANNEL_SET_TIMEOUT_EX	\
	_IOWR(NVGPU_IOCTL_MAGIC, 18, struct nvgpu_set_timeout_ex_args)
#define NVGPU_IOCTL_CHANNEL_WAIT		\
	_IOWR(NVGPU_IOCTL_MAGIC, 102, struct nvgpu_wait_args)
#define NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO	\
	_IOWR(NVGPU_IOCTL_MAGIC, 107, struct nvgpu_submit_gpfifo_args)
#define NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX	\
	_IOWR(NVGPU_IOCTL_MAGIC, 108, struct nvgpu_alloc_obj_ctx_args)
#define NVGPU_IOCTL_CHANNEL_ZCULL_BIND		\
	_IOWR(NVGPU_IOCTL_MAGIC, 110, struct nvgpu_zcull_bind_args)
#define NVGPU_IOCTL_CHANNEL_SET_ERROR_NOTIFIER  \
	_IOWR(NVGPU_IOCTL_MAGIC, 111, struct nvgpu_set_error_notifier)
#define NVGPU_IOCTL_CHANNEL_OPEN	\
	_IOR(NVGPU_IOCTL_MAGIC,  112, struct nvgpu_channel_open_args)
#define NVGPU_IOCTL_CHANNEL_ENABLE	\
	_IO(NVGPU_IOCTL_MAGIC,  113)
#define NVGPU_IOCTL_CHANNEL_DISABLE	\
	_IO(NVGPU_IOCTL_MAGIC,  114)
#define NVGPU_IOCTL_CHANNEL_PREEMPT	\
	_IO(NVGPU_IOCTL_MAGIC,  115)
#define NVGPU_IOCTL_CHANNEL_FORCE_RESET	\
	_IO(NVGPU_IOCTL_MAGIC,  116)
#define NVGPU_IOCTL_CHANNEL_EVENT_ID_CTRL \
	_IOWR(NVGPU_IOCTL_MAGIC, 117, struct nvgpu_event_id_ctrl_args)
#define NVGPU_IOCTL_CHANNEL_WDT \
	_IOW(NVGPU_IOCTL_MAGIC, 119, struct nvgpu_channel_wdt_args)
#define NVGPU_IOCTL_CHANNEL_SET_RUNLIST_INTERLEAVE \
	_IOW(NVGPU_IOCTL_MAGIC, 120, struct nvgpu_runlist_interleave_args)
#define NVGPU_IOCTL_CHANNEL_SET_PREEMPTION_MODE \
	_IOW(NVGPU_IOCTL_MAGIC, 122, struct nvgpu_preemption_mode_args)
#define NVGPU_IOCTL_CHANNEL_ALLOC_GPFIFO_EX	\
	_IOW(NVGPU_IOCTL_MAGIC, 123, struct nvgpu_alloc_gpfifo_ex_args)
#define NVGPU_IOCTL_CHANNEL_SET_BOOSTED_CTX	\
	_IOW(NVGPU_IOCTL_MAGIC, 124, struct nvgpu_boosted_ctx_args)
#define NVGPU_IOCTL_CHANNEL_GET_USER_SYNCPOINT \
	_IOR(NVGPU_IOCTL_MAGIC, 126, struct nvgpu_get_user_syncpoint_args)
#define NVGPU_IOCTL_CHANNEL_RESCHEDULE_RUNLIST	\
	_IOW(NVGPU_IOCTL_MAGIC, 127, struct nvgpu_reschedule_runlist_args)
#define NVGPU_IOCTL_CHANNEL_SETUP_BIND	\
	_IOWR(NVGPU_IOCTL_MAGIC, 128, struct nvgpu_channel_setup_bind_args)

#define NVGPU_IOCTL_CHANNEL_LAST	\
	_IOC_NR(NVGPU_IOCTL_CHANNEL_SETUP_BIND)
#define NVGPU_IOCTL_CHANNEL_MAX_ARG_SIZE \
	sizeof(struct nvgpu_channel_setup_bind_args)

/*
 * /dev/nvhost-ctxsw-gpu device
 *
 * Opening a '/dev/nvhost-ctxsw-gpu' device node creates a way to trace
 * context switches on GR engine
 */

#define NVGPU_CTXSW_IOCTL_MAGIC 'C'

#define NVGPU_CTXSW_TAG_SOF			0x00
#define NVGPU_CTXSW_TAG_CTXSW_REQ_BY_HOST	0x01
#define NVGPU_CTXSW_TAG_FE_ACK			0x02
#define NVGPU_CTXSW_TAG_FE_ACK_WFI		0x0a
#define NVGPU_CTXSW_TAG_FE_ACK_GFXP		0x0b
#define NVGPU_CTXSW_TAG_FE_ACK_CTAP		0x0c
#define NVGPU_CTXSW_TAG_FE_ACK_CILP		0x0d
#define NVGPU_CTXSW_TAG_SAVE_END		0x03
#define NVGPU_CTXSW_TAG_RESTORE_START		0x04
#define NVGPU_CTXSW_TAG_CONTEXT_START		0x05
#define NVGPU_CTXSW_TAG_ENGINE_RESET		0xfe
#define NVGPU_CTXSW_TAG_INVALID_TIMESTAMP	0xff
#define NVGPU_CTXSW_TAG_LAST			\
	NVGPU_CTXSW_TAG_INVALID_TIMESTAMP

struct nvgpu_ctxsw_trace_entry {
	__u8 tag;
	__u8 vmid;
	__u16 seqno;		/* sequence number to detect drops */
	__u32 context_id;	/* context_id as allocated by FECS */
	__u64 pid;		/* 64-bit is max bits of different OS pid */
	__u64 timestamp;	/* 64-bit time */
};

#define NVGPU_CTXSW_RING_HEADER_MAGIC 0x7000fade
#define NVGPU_CTXSW_RING_HEADER_VERSION 0

struct nvgpu_ctxsw_ring_header {
	__u32 magic;
	__u32 version;
	__u32 num_ents;
	__u32 ent_size;
	volatile __u32 drop_count;	/* excluding filtered out events */
	volatile __u32 write_seqno;
	volatile __u32 write_idx;
	volatile __u32 read_idx;
};

struct nvgpu_ctxsw_ring_setup_args {
	__u32 size;	/* [in/out] size of ring buffer in bytes (including
			   header). will be rounded page size. this parameter
			   is updated with actual allocated size. */
};

#define NVGPU_CTXSW_FILTER_SIZE	(NVGPU_CTXSW_TAG_LAST + 1)
#define NVGPU_CTXSW_FILTER_SET(n, p) \
	((p)->tag_bits[(n) / 64] |=  (1 << ((n) & 63)))
#define NVGPU_CTXSW_FILTER_CLR(n, p) \
	((p)->tag_bits[(n) / 64] &= ~(1 << ((n) & 63)))
#define NVGPU_CTXSW_FILTER_ISSET(n, p) \
	((p)->tag_bits[(n) / 64] &   (1 << ((n) & 63)))
#define NVGPU_CTXSW_FILTER_CLR_ALL(p) \
	((void) memset((void *)(p), 0, sizeof(*(p))))
#define NVGPU_CTXSW_FILTER_SET_ALL(p) \
	((void) memset((void *)(p), ~0, sizeof(*(p))))

struct nvgpu_ctxsw_trace_filter {
	__u64 tag_bits[(NVGPU_CTXSW_FILTER_SIZE + 63) / 64];
};

struct nvgpu_ctxsw_trace_filter_args {
	struct nvgpu_ctxsw_trace_filter filter;
};

#define NVGPU_CTXSW_IOCTL_TRACE_ENABLE \
	_IO(NVGPU_CTXSW_IOCTL_MAGIC, 1)
#define NVGPU_CTXSW_IOCTL_TRACE_DISABLE \
	_IO(NVGPU_CTXSW_IOCTL_MAGIC, 2)
#define NVGPU_CTXSW_IOCTL_RING_SETUP \
	_IOWR(NVGPU_CTXSW_IOCTL_MAGIC, 3, struct nvgpu_ctxsw_ring_setup_args)
#define NVGPU_CTXSW_IOCTL_SET_FILTER \
	_IOW(NVGPU_CTXSW_IOCTL_MAGIC, 4, struct nvgpu_ctxsw_trace_filter_args)
#define NVGPU_CTXSW_IOCTL_GET_FILTER \
	_IOR(NVGPU_CTXSW_IOCTL_MAGIC, 5, struct nvgpu_ctxsw_trace_filter_args)
#define NVGPU_CTXSW_IOCTL_POLL \
	_IO(NVGPU_CTXSW_IOCTL_MAGIC, 6)

#define NVGPU_CTXSW_IOCTL_LAST            \
	_IOC_NR(NVGPU_CTXSW_IOCTL_POLL)

#define NVGPU_CTXSW_IOCTL_MAX_ARG_SIZE	\
	sizeof(struct nvgpu_ctxsw_trace_filter_args)

/*
 * /dev/nvhost-sched-gpu device
 *
 * Opening a '/dev/nvhost-sched-gpu' device node creates a way to control
 * GPU scheduling parameters.
 */

#define NVGPU_SCHED_IOCTL_MAGIC 'S'

/*
 * When the app manager receives a NVGPU_SCHED_STATUS_TSG_OPEN notification,
 * it is expected to query the list of recently opened TSGs using
 * NVGPU_SCHED_IOCTL_GET_RECENT_TSGS. The kernel driver maintains a bitmap
 * of recently opened TSGs. When the app manager queries the list, it
 * atomically clears the bitmap. This way, at each invocation of
 * NVGPU_SCHED_IOCTL_GET_RECENT_TSGS, app manager only receives the list of
 * TSGs that have been opened since last invocation.
 *
 * If the app manager needs to re-synchronize with the driver, it can use
 * NVGPU_SCHED_IOCTL_GET_TSGS to retrieve the complete list of TSGs. The
 * recent TSG bitmap will be cleared in that case too.
 */
struct nvgpu_sched_get_tsgs_args {
	/* in: size of buffer in bytes */
	/* out: actual size of size of TSG bitmap. if user-provided size is too
	 * small, ioctl will return -ENOSPC, and update this field, allowing
	 * application to discover required number of bytes and allocate
	 * a buffer accordingly.
	 */
	__u32 size;

	/* in: address of 64-bit aligned buffer */
	/* out: buffer contains a TSG bitmap.
	 * Bit #n will be set in the bitmap if TSG #n is present.
	 * When using NVGPU_SCHED_IOCTL_GET_RECENT_TSGS, the first time you use
	 * this command, it will return the opened TSGs and subsequent calls
	 * will only return the delta (ie. each invocation clears bitmap)
	 */
	__u64 buffer;
};

struct nvgpu_sched_get_tsgs_by_pid_args {
	/* in: process id for which we want to retrieve TSGs */
	__u64 pid;

	/* in: size of buffer in bytes */
	/* out: actual size of size of TSG bitmap. if user-provided size is too
	 * small, ioctl will return -ENOSPC, and update this field, allowing
	 * application to discover required number of bytes and allocate
	 * a buffer accordingly.
	 */
	__u32 size;

	/* in: address of 64-bit aligned buffer */
	/* out: buffer contains a TSG bitmap. */
	__u64 buffer;
};

struct nvgpu_sched_tsg_get_params_args {
	__u32 tsgid;		/* in: TSG identifier */
	__u32 timeslice;	/* out: timeslice in usecs */
	__u32 runlist_interleave;
	__u32 graphics_preempt_mode;
	__u32 compute_preempt_mode;
	__u64 pid;		/* out: process identifier of TSG owner */
};

struct nvgpu_sched_tsg_timeslice_args {
	__u32 tsgid;                    /* in: TSG identifier */
	__u32 timeslice;                /* in: timeslice in usecs */
};

struct nvgpu_sched_tsg_runlist_interleave_args {
	__u32 tsgid;			/* in: TSG identifier */

		/* in: see NVGPU_RUNLIST_INTERLEAVE_LEVEL_ */
	__u32 runlist_interleave;
};

struct nvgpu_sched_api_version_args {
	__u32 version;
};

struct nvgpu_sched_tsg_refcount_args {
	__u32 tsgid;                    /* in: TSG identifier */
};

#define NVGPU_SCHED_IOCTL_GET_TSGS					\
	_IOWR(NVGPU_SCHED_IOCTL_MAGIC, 1,				\
		struct nvgpu_sched_get_tsgs_args)
#define NVGPU_SCHED_IOCTL_GET_RECENT_TSGS				\
	_IOWR(NVGPU_SCHED_IOCTL_MAGIC, 2,				\
		struct nvgpu_sched_get_tsgs_args)
#define NVGPU_SCHED_IOCTL_GET_TSGS_BY_PID				\
	_IOWR(NVGPU_SCHED_IOCTL_MAGIC, 3,				\
		struct nvgpu_sched_get_tsgs_by_pid_args)
#define NVGPU_SCHED_IOCTL_TSG_GET_PARAMS				\
	_IOWR(NVGPU_SCHED_IOCTL_MAGIC, 4,				\
		struct nvgpu_sched_tsg_get_params_args)
#define NVGPU_SCHED_IOCTL_TSG_SET_TIMESLICE				\
	_IOW(NVGPU_SCHED_IOCTL_MAGIC, 5,				\
		struct nvgpu_sched_tsg_timeslice_args)
#define NVGPU_SCHED_IOCTL_TSG_SET_RUNLIST_INTERLEAVE			\
	_IOW(NVGPU_SCHED_IOCTL_MAGIC, 6,				\
		struct nvgpu_sched_tsg_runlist_interleave_args)
#define NVGPU_SCHED_IOCTL_LOCK_CONTROL					\
	_IO(NVGPU_SCHED_IOCTL_MAGIC, 7)
#define NVGPU_SCHED_IOCTL_UNLOCK_CONTROL				\
	_IO(NVGPU_SCHED_IOCTL_MAGIC, 8)
#define NVGPU_SCHED_IOCTL_GET_API_VERSION				\
	_IOR(NVGPU_SCHED_IOCTL_MAGIC, 9,				\
	    struct nvgpu_sched_api_version_args)
#define NVGPU_SCHED_IOCTL_GET_TSG					\
	_IOW(NVGPU_SCHED_IOCTL_MAGIC, 10,				\
		struct nvgpu_sched_tsg_refcount_args)
#define NVGPU_SCHED_IOCTL_PUT_TSG					\
	_IOW(NVGPU_SCHED_IOCTL_MAGIC, 11,				\
		struct nvgpu_sched_tsg_refcount_args)
#define NVGPU_SCHED_IOCTL_LAST						\
	_IOC_NR(NVGPU_SCHED_IOCTL_PUT_TSG)

#define NVGPU_SCHED_IOCTL_MAX_ARG_SIZE					\
	sizeof(struct nvgpu_sched_tsg_get_params_args)


#define NVGPU_SCHED_SET(n, bitmap)	\
	(((__u64 *)(bitmap))[(n) / 64] |=  (1ULL << (((__u64)n) & 63)))
#define NVGPU_SCHED_CLR(n, bitmap)	\
	(((__u64 *)(bitmap))[(n) / 64] &= ~(1ULL << (((__u64)n) & 63)))
#define NVGPU_SCHED_ISSET(n, bitmap)	\
	(((__u64 *)(bitmap))[(n) / 64] & (1ULL << (((__u64)n) & 63)))

#define NVGPU_SCHED_STATUS_TSG_OPEN	(1ULL << 0)

struct nvgpu_sched_event_arg {
	__u64 reserved;
	__u64 status;
};

#define NVGPU_SCHED_API_VERSION		1

#endif
