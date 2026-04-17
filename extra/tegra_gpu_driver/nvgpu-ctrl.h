/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * Legacy node:   /dev/nvhost-ctrl-gpu device
 * New hierarchy: /dev/nvgpu/igpu0/ctrl
 *
 * This device serves as the core control node for NvGPU. From this node
 * one can query GPU device information, instantiate GPU device objects
 * (TSGs, address spaces, etc), and do various other non-context specific
 * things.
 */

#ifndef _UAPI__LINUX_NVGPU_CTRL_H__
#define _UAPI__LINUX_NVGPU_CTRL_H__

#include "nvgpu-uapi-common.h"

#define NVGPU_GPU_IOCTL_MAGIC 'G'

/* return zcull ctx size */
struct nvgpu_gpu_zcull_get_ctx_size_args {
	__u32 size;
};

/* return zcull info */
struct nvgpu_gpu_zcull_get_info_args {
	__u32 width_align_pixels;
	__u32 height_align_pixels;
	__u32 pixel_squares_by_aliquots;
	__u32 aliquot_total;
	__u32 region_byte_multiplier;
	__u32 region_header_size;
	__u32 subregion_header_size;
	__u32 subregion_width_align_pixels;
	__u32 subregion_height_align_pixels;
	__u32 subregion_count;
};

#define NVGPU_ZBC_COLOR_VALUE_SIZE	4
#define NVGPU_ZBC_TYPE_INVALID		0
#define NVGPU_ZBC_TYPE_COLOR		1
#define NVGPU_ZBC_TYPE_DEPTH		2
#define NVGPU_ZBC_TYPE_STENCIL		3

struct nvgpu_gpu_zbc_set_table_args {
	__u32 color_ds[NVGPU_ZBC_COLOR_VALUE_SIZE];
	__u32 color_l2[NVGPU_ZBC_COLOR_VALUE_SIZE];
	__u32 depth;
	__u32 stencil;
	__u32 format;
	__u32 type;	/* color, depth or stencil */
};

struct nvgpu_gpu_zbc_query_table_args {
	__u32 color_ds[NVGPU_ZBC_COLOR_VALUE_SIZE];
	__u32 color_l2[NVGPU_ZBC_COLOR_VALUE_SIZE];
	__u32 depth;
	__u32 stencil;
	__u32 ref_cnt;
	__u32 format;
	__u32 type;		/* color, depth or stencil */
	__u32 index_size;	/* [out] size, [in] index */
};


/* This contains the minimal set by which the userspace can
   determine all the properties of the GPU */
#define NVGPU_GPU_ARCH_GK100	0x000000E0
#define NVGPU_GPU_ARCH_GM200	0x00000120
#define NVGPU_GPU_ARCH_GP100	0x00000130
#define NVGPU_GPU_ARCH_GV110	0x00000150
#define NVGPU_GPU_ARCH_GV100	0x00000140

#define NVGPU_GPU_IMPL_GK20A	0x0000000A
#define NVGPU_GPU_IMPL_GM204	0x00000004
#define NVGPU_GPU_IMPL_GM206	0x00000006
#define NVGPU_GPU_IMPL_GM20B	0x0000000B
#define NVGPU_GPU_IMPL_GM20B_B	0x0000000E
#define NVGPU_GPU_IMPL_GP104	0x00000004
#define NVGPU_GPU_IMPL_GP106	0x00000006
#define NVGPU_GPU_IMPL_GP10B	0x0000000B
#define NVGPU_GPU_IMPL_GV11B	0x0000000B
#define NVGPU_GPU_IMPL_GV100	0x00000000

#define NVGPU_GPU_BUS_TYPE_NONE         0
#define NVGPU_GPU_BUS_TYPE_AXI         32

#define NVGPU_GPU_FLAGS_HAS_SYNCPOINTS			(1ULL << 0)
/* MAP_BUFFER_EX with sparse allocations */
#define NVGPU_GPU_FLAGS_SUPPORT_SPARSE_ALLOCS		(1ULL << 2)
/* sync fence FDs are available in, e.g., submit_gpfifo */
#define NVGPU_GPU_FLAGS_SUPPORT_SYNC_FENCE_FDS		(1ULL << 3)
/* NVGPU_DBG_GPU_IOCTL_CYCLE_STATS is available */
#define NVGPU_GPU_FLAGS_SUPPORT_CYCLE_STATS		(1ULL << 4)
/* NVGPU_DBG_GPU_IOCTL_CYCLE_STATS_SNAPSHOT is available */
#define NVGPU_GPU_FLAGS_SUPPORT_CYCLE_STATS_SNAPSHOT	(1ULL << 6)
/* Both gpu driver and device support TSG */
#define NVGPU_GPU_FLAGS_SUPPORT_TSG			(1ULL << 8)
/* Clock control support */
#define NVGPU_GPU_FLAGS_SUPPORT_CLOCK_CONTROLS		(1ULL << 9)
/* NVGPU_GPU_IOCTL_GET_VOLTAGE is available */
#define NVGPU_GPU_FLAGS_SUPPORT_GET_VOLTAGE		(1ULL << 10)
/* NVGPU_GPU_IOCTL_GET_CURRENT is available */
#define NVGPU_GPU_FLAGS_SUPPORT_GET_CURRENT		(1ULL << 11)
/* NVGPU_GPU_IOCTL_GET_POWER is available */
#define NVGPU_GPU_FLAGS_SUPPORT_GET_POWER		(1ULL << 12)
/* NVGPU_GPU_IOCTL_GET_TEMPERATURE is available */
#define NVGPU_GPU_FLAGS_SUPPORT_GET_TEMPERATURE		(1ULL << 13)
/* NVGPU_GPU_IOCTL_SET_THERM_ALERT_LIMIT is available */
#define NVGPU_GPU_FLAGS_SUPPORT_SET_THERM_ALERT_LIMIT	(1ULL << 14)
/* NVGPU_GPU_IOCTL_GET_EVENT_FD is available */
#define NVGPU_GPU_FLAGS_SUPPORT_DEVICE_EVENTS		(1ULL << 15)
/* FECS context switch tracing is available */
#define NVGPU_GPU_FLAGS_SUPPORT_FECS_CTXSW_TRACE	(1ULL << 16)
/* NVGPU_AS_IOCTL_MAP_BUFFER_COMPBITS is available */
#define NVGPU_GPU_FLAGS_SUPPORT_MAP_COMPBITS		(1ULL << 17)
/* Fast deterministic submits with no job tracking are supported */
#define NVGPU_GPU_FLAGS_SUPPORT_DETERMINISTIC_SUBMIT_NO_JOBTRACKING (1ULL << 18)
/* Deterministic submits are supported even with job tracking */
#define NVGPU_GPU_FLAGS_SUPPORT_DETERMINISTIC_SUBMIT_FULL (1ULL << 19)
/* IO coherence support is available */
#define NVGPU_GPU_FLAGS_SUPPORT_IO_COHERENCE		(1ULL << 20)
/* NVGPU_IOCTL_CHANNEL_RESCHEDULE_RUNLIST is available */
#define NVGPU_GPU_FLAGS_SUPPORT_RESCHEDULE_RUNLIST	(1ULL << 21)
/*  subcontexts are available */
#define NVGPU_GPU_FLAGS_SUPPORT_TSG_SUBCONTEXTS         (1ULL << 22)
/* NVGPU_GPU_IOCTL_SET_DETERMINISTIC_OPTS is available */
#define NVGPU_GPU_FLAGS_SUPPORT_DETERMINISTIC_OPTS	(1ULL << 24)
/* SCG support is available */
#define NVGPU_GPU_FLAGS_SUPPORT_SCG			(1ULL << 25)
/* GPU_VA address of a syncpoint is supported */
#define NVGPU_GPU_FLAGS_SUPPORT_SYNCPOINT_ADDRESS	(1ULL << 26)
/* VPR is supported */
#define NVGPU_GPU_FLAGS_SUPPORT_VPR			(1ULL << 27)
/* Allocating per-channel syncpoint in user space is supported */
#define NVGPU_GPU_FLAGS_SUPPORT_USER_SYNCPOINT		(1ULL << 28)
/* Railgating (powering the GPU off completely) is supported and enabled */
#define NVGPU_GPU_FLAGS_CAN_RAILGATE			(1ULL << 29)
/* Usermode submit is available */
#define NVGPU_GPU_FLAGS_SUPPORT_USERMODE_SUBMIT		(1ULL << 30)
/* Reduced profile is enabled */
#define NVGPU_GPU_FLAGS_DRIVER_REDUCED_PROFILE		(1ULL << 31)
/* Set MMU debug mode is available */
#define NVGPU_GPU_FLAGS_SUPPORT_SET_CTX_MMU_DEBUG_MODE	(1ULL << 32)
/* Fault recovery is enabled */
#define NVGPU_GPU_FLAGS_SUPPORT_FAULT_RECOVERY		(1ULL << 33)
/* Mapping modify is enabled */
#define NVGPU_GPU_FLAGS_SUPPORT_MAPPING_MODIFY		(1ULL << 34)
/* Remap is enabled */
#define NVGPU_GPU_FLAGS_SUPPORT_REMAP			(1ULL << 35)
/* Compression is enabled */
#define NVGPU_GPU_FLAGS_SUPPORT_COMPRESSION		(1ULL << 36)
/* SM TTU is enabled */
#define NVGPU_GPU_FLAGS_SUPPORT_SM_TTU			(1ULL << 37)
/* Compression PLC is enabled */
#define NVGPU_GPU_FLAGS_SUPPORT_POST_L2_COMPRESSION	(1ULL << 38)
/** GMMU map access type available */
#define NVGPU_GPU_FLAGS_SUPPORT_MAP_ACCESS_TYPE		(1ULL << 39)
/* Flag to indicate whether 2d operations are supported */
#define NVGPU_GPU_FLAGS_SUPPORT_2D			(1ULL << 40)
/* Flag to indicate whether 3d graphics operations are supported */
#define NVGPU_GPU_FLAGS_SUPPORT_3D			(1ULL << 41)
/* Flag to indicate whether compute operations are supported */
#define NVGPU_GPU_FLAGS_SUPPORT_COMPUTE			(1ULL << 42)
/* Flag to indicate whether inline methods are supported */
#define NVGPU_GPU_FLAGS_SUPPORT_I2M			(1ULL << 43)
/* Flag to indicate whether zbc classes are supported */
#define NVGPU_GPU_FLAGS_SUPPORT_ZBC			(1ULL << 44)
/* Profiler V2 device objects are supported */
#define NVGPU_GPU_FLAGS_SUPPORT_PROFILER_V2_DEVICE	(1ULL << 46)
/* Profiler V2 context objects are supported */
#define NVGPU_GPU_FLAGS_SUPPORT_PROFILER_V2_CONTEXT	(1ULL << 47)
/* Profiling SMPC in global mode is supported */
#define NVGPU_GPU_FLAGS_SUPPORT_SMPC_GLOBAL_MODE	(1ULL << 48)
/* Retrieving contents of graphics context is supported */
#define NVGPU_GPU_FLAGS_SUPPORT_GET_GR_CONTEXT	    (1ULL << 49)
/*
 * Note: Additional buffer metadata association support. This feature is only
 * for supporting legacy userspace APIs and for compatibility with desktop
 * RM behavior. Usage of this feature should be avoided.
 */
#define NVGPU_GPU_FLAGS_SUPPORT_BUFFER_METADATA		(1ULL << 50)
/* Flag to indicate whether configuring L2_MAXEVICTLAST_WAYS is supported */
#define NVGPU_GPU_FLAGS_L2_MAX_WAYS_EVICT_LAST_ENABLED	(1ULL << 51)
/* Vidmem access bits feature is supported */
#define NVGPU_GPU_FLAGS_SUPPORT_VAB		(1ULL << 52)
/* The NVS scheduler interface is usable */
#define NVGPU_GPU_FLAGS_SUPPORT_NVS		(1ULL << 53)
/* Flag to indicate whether implicit ERRBAR is supported */
#define NVGPU_GPU_FLAGS_SCHED_EXIT_WAIT_FOR_ERRBAR_SUPPORTED    (1ULL << 55)
/* Flag to indicate whether multi-process TSG sharing is supported */
#define NVGPU_GPU_FLAGS_MULTI_PROCESS_TSG_SHARING    (1ULL << 56)
/* SM LRF ECC is enabled */
#define NVGPU_GPU_FLAGS_ECC_ENABLED_SM_LRF	(1ULL << 60)
/* Flag to indicate GPU MMIO support */
#define NVGPU_GPU_FLAGS_SUPPORT_GPU_MMIO	(1ULL << 57)
/* SM SHM ECC is enabled */
#define NVGPU_GPU_FLAGS_ECC_ENABLED_SM_SHM	(1ULL << 61)
/* TEX ECC is enabled */
#define NVGPU_GPU_FLAGS_ECC_ENABLED_TEX		(1ULL << 62)
/* L2 ECC is enabled */
#define NVGPU_GPU_FLAGS_ECC_ENABLED_LTC		(1ULL << 63)
/* All types of ECC are enabled */
#define NVGPU_GPU_FLAGS_ALL_ECC_ENABLED	\
				(NVGPU_GPU_FLAGS_ECC_ENABLED_SM_LRF |	\
				NVGPU_GPU_FLAGS_ECC_ENABLED_SM_SHM |	\
				NVGPU_GPU_FLAGS_ECC_ENABLED_TEX    |	\
				NVGPU_GPU_FLAGS_ECC_ENABLED_LTC)

struct nvgpu_gpu_characteristics {
	__u32 arch;
	__u32 impl;
	__u32 rev;
	__u32 num_gpc;

	/*
	 * Specifies the NUMA domain for the GPU device
	 * A value of "-1" specifies no NUMA domain info.
	 */
#define NVGPU_GPU_CHARACTERISTICS_NO_NUMA_INFO (-1)
	__s32 numa_domain_id;

	__u64 L2_cache_size;               /* bytes */
	__u64 on_board_video_memory_size;  /* bytes */

	__u32 num_tpc_per_gpc; /* the architectural maximum */
	__u32 bus_type;

	__u32 big_page_size; /* the default big page size */
	__u32 compression_page_size;

	__u32 pde_coverage_bit_count;

	/* bit N set ==> big page size 2^N is available in
	   NVGPU_GPU_IOCTL_ALLOC_AS. The default big page size is
	   always available regardless of this field. */
	__u32 available_big_page_sizes;

	__u64 flags;

	__u32 twod_class;
	__u32 threed_class;
	__u32 compute_class;
	__u32 gpfifo_class;
	__u32 inline_to_memory_class;
	__u32 dma_copy_class;

	__u32 gpc_mask; /* enabled GPCs */

	__u32 sm_arch_sm_version; /* sm version */
	__u32 sm_arch_spa_version; /* sm instruction set */
	__u32 sm_arch_warp_count;

	/* IOCTL interface levels by service. -1 if not supported */
	__s16 gpu_ioctl_nr_last;
	__s16 tsg_ioctl_nr_last;
	__s16 dbg_gpu_ioctl_nr_last;
	__s16 ioctl_channel_nr_last;
	__s16 as_ioctl_nr_last;

	__u8 gpu_va_bit_count;
	__u8 reserved;

	__u32 max_fbps_count;
	__u32 fbp_en_mask;
	__u32 emc_en_mask;
	__u32 max_ltc_per_fbp;
	__u32 max_lts_per_ltc;
	__u32 max_tex_per_tpc;
	__u32 max_gpc_count;
	/* mask of Rop_L2 for each FBP */
	__u32 rop_l2_en_mask_DEPRECATED[2];


	__u8 chipname[8];

	__u64 gr_compbit_store_base_hw;
	__u32 gr_gobs_per_comptagline_per_slice;
	__u32 num_ltc;
	__u32 lts_per_ltc;
	__u32 cbc_cache_line_size;
	__u32 cbc_comptags_per_line;

	/* MAP_BUFFER_BATCH: the upper limit for num_unmaps and
	 * num_maps */
	__u32 map_buffer_batch_limit;

	__u64 max_freq;

	/* supported preemption modes */
	__u32 graphics_preemption_mode_flags; /* NVGPU_GRAPHICS_PREEMPTION_MODE_* */
	__u32 compute_preemption_mode_flags; /* NVGPU_COMPUTE_PREEMPTION_MODE_* */
	/* default preemption modes */
	__u32 default_graphics_preempt_mode; /* NVGPU_GRAPHICS_PREEMPTION_MODE_* */
	__u32 default_compute_preempt_mode; /* NVGPU_COMPUTE_PREEMPTION_MODE_* */

	__u64 local_video_memory_size; /* in bytes, non-zero only for dGPUs */

	/* These are meaningful only for PCI devices */
	__u16 pci_vendor_id, pci_device_id;
	__u16 pci_subsystem_vendor_id, pci_subsystem_device_id;
	__u16 pci_class;
	__u8  pci_revision;
	__u8  vbios_oem_version;
	__u32 vbios_version;

	/* NVGPU_DBG_GPU_IOCTL_REG_OPS: the upper limit for the number
	 * of regops */
	__u32 reg_ops_limit;
	__u32 reserved1;

	__s16 event_ioctl_nr_last;
	__u16 pad;

	__u32 max_css_buffer_size;

	__s16 ctxsw_ioctl_nr_last;
	__s16 prof_ioctl_nr_last;
	__s16 nvs_ioctl_nr_last;
	__u8 reserved2[2];

	__u32 max_ctxsw_ring_buffer_size;
	__u32 reserved3;

	__u64 per_device_identifier;

	__u32 num_ppc_per_gpc;
	__u32 max_veid_count_per_tsg;

	__u32 num_sub_partition_per_fbpa;
	__u32 gpu_instance_id;

	__u32 gr_instance_id;

	/** Max gpfifo entries allowed by nvgpu-rm. */
	__u32 max_gpfifo_entries;

	__u32 max_dbg_tsg_timeslice;
	__u32 reserved5;

	/*
	 * Instance id of the opened ctrl node. Unique number over the
	 * nvgpu driver's lifetime (probe to unload).
	 */
	__u64 device_instance_id;

	/* Notes:
	   - This struct can be safely appended with new fields. However, always
	     keep the structure size multiple of 8 and make sure that the binary
	     layout does not change between 32-bit and 64-bit architectures.
	   - If the last field is reserved/padding, it is not
	     generally safe to repurpose the field in future revisions.
	*/
};

struct nvgpu_gpu_get_characteristics {
	/* [in]  size reserved by the user space. Can be 0.
	   [out] full buffer size by kernel */
	__u64 gpu_characteristics_buf_size;

	/* [in]  address of nvgpu_gpu_characteristics buffer. Filled with field
	   values by exactly MIN(buf_size_in, buf_size_out) bytes. Ignored, if
	   buf_size_in is zero.  */
	__u64 gpu_characteristics_buf_addr;
};

#define NVGPU_GPU_COMPBITS_NONE		0
#define NVGPU_GPU_COMPBITS_GPU		(1 << 0)
#define NVGPU_GPU_COMPBITS_CDEH		(1 << 1)
#define NVGPU_GPU_COMPBITS_CDEV		(1 << 2)

struct nvgpu_gpu_prepare_compressible_read_args {
	__u32 handle;			/* in, dmabuf fd */
	union {
		__u32 request_compbits;	/* in */
		__u32 valid_compbits;	/* out */
	};
	__u64 offset;			/* in, within handle */
	__u64 compbits_hoffset;		/* in, within handle */
	__u64 compbits_voffset;		/* in, within handle */
	__u32 width;			/* in, in pixels */
	__u32 height;			/* in, in pixels */
	__u32 block_height_log2;	/* in */
	__u32 submit_flags;		/* in (NVGPU_SUBMIT_GPFIFO_FLAGS_) */
	union {
		struct {
			__u32 syncpt_id;
			__u32 syncpt_value;
		};
		__s32 fd;
	} fence;			/* in/out */
	__u32 zbc_color;		/* out */
	__u32 reserved;		/* must be zero */
	__u64 scatterbuffer_offset;	/* in, within handle */
	__u32 reserved2[2];		/* must be zero */
};

struct nvgpu_gpu_mark_compressible_write_args {
	__u32 handle;			/* in, dmabuf fd */
	__u32 valid_compbits;		/* in */
	__u64 offset;			/* in, within handle */
	__u32 zbc_color;		/* in */
	__u32 reserved[3];		/* must be zero */
};

struct nvgpu_alloc_as_args {
	__u32 big_page_size;	/* zero for no big pages for this VA */
	__s32 as_fd;

/*
 * The GPU address space will be managed by the userspace. This has
 * the following changes in functionality:
 *   1. All non-fixed-offset user mappings are rejected (i.e.,
 *      fixed-offset only)
 *   2. Address space does not need to be allocated for fixed-offset
 *      mappings, except to mark sparse address space areas.
 *   3. Maps and unmaps are immediate. In particular, mapping ref
 *      increments at kickoffs and decrements at job completion are
 *      bypassed.
 */
#define NVGPU_GPU_IOCTL_ALLOC_AS_FLAGS_UNIFIED_VA	 	(1 << 1)
	__u32 flags;
	__u32 reserved;		/* must be zero */
	__u64 va_range_start;	/* in: starting VA (aligned by PDE) */
	__u64 va_range_end;	/* in: ending VA (aligned by PDE) */
	__u64 va_range_split;	/* in: small/big page split (aligned by PDE,
				 * must be zero if UNIFIED_VA is set) */
	__u32 padding[6];
};

/*
 * NVGPU_GPU_IOCTL_OPEN_TSG - create/share TSG ioctl.
 *
 * This IOCTL allocates one of the available TSG for user when called
 * without share token specified. When called with share token specified,
 * fd is created for already allocated TSG for sharing the TSG under
 * different device/CTRL object hierarchies in different processes.
 *
 * Source device is specified in the arguments and target device is
 * implied from the caller. Share token is unique for a TSG.
 *
 * When the TSG is successfully created first time or is opened with share
 * token, the device instance id associated with the CTRL fd will be added
 * to the TSG private data structure as authorized device instance ids.
 * This is used for a security check when creating a TSG share token with
 * nvgpu_tsg_get_share_token.
 *
 * return 0 on success, -1 on error.
 * retval EINVAL if invalid parameters are specified (if TSG_FLAGS_SHARE
 *               is set but source_device_instance_id and/or share token
 *               are zero or TSG_FLAGS_SHARE is not set but other
 *               arguments are non-zero).
 * retval EINVAL if share token doesn't exist or is expired.
 */

/*
 * Specify that the newly created TSG fd will map to existing hardware
 * TSG resources.
 */
#define NVGPU_GPU_IOCTL_OPEN_TSG_FLAGS_SHARE	((__u32)1U << 0U)

/* Arguments for NVGPU_GPU_IOCTL_OPEN_TSG */
struct nvgpu_gpu_open_tsg_args {
	__u32 tsg_fd;		/* out: tsg fd */
	__u32 flags;		/* in: NVGPU_GPU_IOCTL_OPEN_TSG_FLAGS_* */

	__u64 source_device_instance_id; /*
					  * in: source device instance id
					  * that created the token. Ignored when
					  * NVGPU_GPU_IOCTL_OPEN_TSG_FLAGS_SHARE
					  * is unset.
					  */

	__u64 share_token;	/*
				 * in: share token obtained from
				 * NVGPU_TSG_IOCTL_GET_SHARE_TOKEN. Ignored when
				 * NVGPU_GPU_IOCTL_OPEN_TSG_FLAGS_SHARE
				 * is unset.
				 */
};

struct nvgpu_gpu_get_tpc_masks_args {
	/* [in]  TPC mask buffer size reserved by userspace. Should be
		 at least sizeof(__u32) * fls(gpc_mask) to receive TPC
		 mask for each GPC.
	   [out] full kernel buffer size
	*/
	__u32 mask_buf_size;
	__u32 reserved;

	/* [in]  pointer to TPC mask buffer. It will receive one
		 32-bit TPC mask per GPC or 0 if GPC is not enabled or
		 not present. This parameter is ignored if
		 mask_buf_size is 0. */
	__u64 mask_buf_addr;
};

struct nvgpu_gpu_get_gpc_physical_map_args {
	/* [in] GPC logical-map-buffer size. It must be
	 * sizeof(__u32) * popcnt(gpc_mask)
	 */
	__u32 map_buf_size;
	__u32 reserved;

	/* [out] pointer to array of u32 entries.
	 * For each entry, index=local gpc index and value=physical gpc index.
	 */
	__u64 physical_gpc_buf_addr;
};

struct nvgpu_gpu_get_gpc_logical_map_args {
	/* [in] GPC logical-map-buffer size. It must be
	 * sizeof(__u32) * popcnt(gpc_mask)
	 */
	__u32 map_buf_size;
	__u32 reserved;

	/* [out] pointer to array of u32 entries.
	 * For each entry, index=local gpc index and value=logical gpc index.
	 */
	__u64 logical_gpc_buf_addr;
};

struct nvgpu_gpu_open_channel_args {
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

/* L2 cache writeback, optionally invalidate clean lines and flush fb */
struct nvgpu_gpu_l2_fb_args {
	__u32 l2_flush:1;
	__u32 l2_invalidate:1;
	__u32 fb_flush:1;
	__u32 reserved;
} __packed;

struct nvgpu_gpu_mmu_debug_mode_args {
	__u32 state;
	__u32 reserved;
};

struct nvgpu_gpu_sm_debug_mode_args {
	int channel_fd;
	__u32 enable;
	__u64 sms;
};

struct warpstate {
	__u64 valid_warps[2];
	__u64 trapped_warps[2];
	__u64 paused_warps[2];
};

struct nvgpu_gpu_wait_pause_args {
	__u64 pwarpstate;
};

struct nvgpu_gpu_tpc_exception_en_status_args {
	__u64 tpc_exception_en_sm_mask;
};

struct nvgpu_gpu_num_vsms {
	__u32 num_vsms;
	__u32 reserved;
};

struct nvgpu_gpu_vsms_mapping_entry {
	/* Logical GPC index */
	__u8 gpc_logical_index;

	/* Virtual GPC index */
	__u8 gpc_virtual_index;

	/* Local Logical index in TPC */
	__u8 tpc_local_logical_index;

	/* Global Logical index in GPU */
	__u8 tpc_global_logical_index;

	/* Local SM index in TPC */
	__u8 sm_local_id;

	/* Migratable TPC index */
	__u8 tpc_migratable_index;
};

struct nvgpu_gpu_vsms_mapping {
	__u64 vsms_map_buf_addr;
};

/*
 * get buffer information ioctl.
 *
 * Note: Additional metadata is available with the buffer only for supporting
 * legacy userspace APIs and for compatibility with desktop RM. Usage of this
 * API should be avoided.
 *
 * This ioctl returns information about buffer to libnvrm_gpu. This information
 * includes buffer registration status, comptags allocation status, size of the
 * buffer, copy of the metadata blob associated with the buffer during
 * registration based on input size and size of the metadata blob
 * registered.
 *
 * return 0 on success, < 0 in case of failure. Note that If the buffer
 *         has no privdata allocated or if it is not registered, this
 *         devctl returns 0 with only size.
 * retval -EINVAL if the enabled flag NVGPU_SUPPORT_BUFFER_METADATA isn't
 *                set or invalid params.
 * retval -EFAULT if the metadata blob copy fails.
 */

/*
 * If the buffer registration is done, this flag is set in the output flags in
 * the buffer info query ioctl.
 */
#define NVGPU_GPU_BUFFER_INFO_FLAGS_METADATA_REGISTERED		(1ULL << 0)

/*
 * If the comptags are allocated and enabled for the buffer, this flag is set
 * in the output flags in the buffer info query ioctl.
 */
#define NVGPU_GPU_BUFFER_INFO_FLAGS_COMPTAGS_ALLOCATED		(1ULL << 1)

/*
 * If the metadata state (blob and comptags) of the buffer can be redefined,
 * this flag is set in the output flags in the buffer info query ioctl.
 */
#define NVGPU_GPU_BUFFER_INFO_FLAGS_MUTABLE_METADATA		(1ULL << 2)

/*
 * get buffer info ioctl arguments struct.
 *
 * Note: Additional metadata is available with the buffer only for supporting
 * legacy userspace APIs and for compatibility with desktop RM. Usage of this
 * API should be avoided.
 */
struct nvgpu_gpu_get_buffer_info_args {
	union {
		struct {
			/* [in] dma-buf fd */
			__s32 dmabuf_fd;
			/* [in] size reserved by the user space. */
			__u32 metadata_size;
			/* [in]  Pointer to receive the buffer metadata. */
			__u64 metadata_addr;
		} in;
		struct {
			/* [out] buffer information flags. */
			__u64 flags;

			/*
			 * [out] buffer metadata size registered.
			 *       this is always 0 for unregistered buffers.
			 */
			__u32 metadata_size;
			__u32 reserved;

			/* [out] allocated size of the buffer */
			__u64 size;
		} out;
	};
};

#define NVGPU_GPU_GET_CPU_TIME_CORRELATION_INFO_MAX_COUNT		16
#define NVGPU_GPU_GET_CPU_TIME_CORRELATION_INFO_SRC_ID_TSC		1
#define NVGPU_GPU_GET_CPU_TIME_CORRELATION_INFO_SRC_ID_OSTIME	2

struct nvgpu_gpu_get_cpu_time_correlation_sample {
	/* gpu timestamp value */
	__u64 cpu_timestamp;
	/* raw GPU counter (PTIMER) value */
	__u64 gpu_timestamp;
};

struct nvgpu_gpu_get_cpu_time_correlation_info_args {
	/* timestamp pairs */
	struct nvgpu_gpu_get_cpu_time_correlation_sample samples[
		NVGPU_GPU_GET_CPU_TIME_CORRELATION_INFO_MAX_COUNT];
	/* number of pairs to read */
	__u32 count;
	/* cpu clock source id */
	__u32 source_id;
};

struct nvgpu_gpu_get_gpu_time_args {
	/* raw GPU counter (PTIMER) value */
	__u64 gpu_timestamp;

	/* reserved for future extensions */
	__u64 reserved;
};

struct nvgpu_gpu_get_engine_info_item {

#define NVGPU_GPU_ENGINE_ID_GR 0
#define NVGPU_GPU_ENGINE_ID_GR_COPY 1
#define NVGPU_GPU_ENGINE_ID_ASYNC_COPY 2
#define NVGPU_GPU_ENGINE_ID_NVENC 5
#define NVGPU_GPU_ENGINE_ID_OFA 6
#define NVGPU_GPU_ENGINE_ID_NVDEC 7
#define NVGPU_GPU_ENGINE_ID_NVJPG 8
	__u32 engine_id;

	__u32 engine_instance;

	/* runlist id for opening channels to the engine, or -1 if
	 * channels are not supported */
	__s32 runlist_id;

	__u32 reserved;
};

struct nvgpu_gpu_get_engine_info_args {
	/* [in]  Buffer size reserved by userspace.
	 *
	 * [out] Full kernel buffer size. Multiple of sizeof(struct
	 *       nvgpu_gpu_get_engine_info_item)
	*/
	__u32 engine_info_buf_size;
	__u32 reserved;
	__u64 engine_info_buf_addr;
};

#define NVGPU_GPU_ALLOC_VIDMEM_FLAG_CONTIGUOUS		(1U << 0)

/* CPU access and coherency flags (3 bits). Use CPU access with care,
 * BAR resources are scarce. */
#define NVGPU_GPU_ALLOC_VIDMEM_FLAG_CPU_NOT_MAPPABLE	(0U << 1)
#define NVGPU_GPU_ALLOC_VIDMEM_FLAG_CPU_WRITE_COMBINE	(1U << 1)
#define NVGPU_GPU_ALLOC_VIDMEM_FLAG_CPU_CACHED		(2U << 1)
#define NVGPU_GPU_ALLOC_VIDMEM_FLAG_CPU_MASK		(7U << 1)

#define NVGPU_GPU_ALLOC_VIDMEM_FLAG_VPR			(1U << 4)

/* Allocation of device-specific local video memory. Returns dmabuf fd
 * on success. */
struct nvgpu_gpu_alloc_vidmem_args {
	union {
		struct {
			/* Size for allocation. Must be a multiple of
			 * small page size. */
			__u64 size;

			/* NVGPU_GPU_ALLOC_VIDMEM_FLAG_* */
			__u32 flags;

			/* Informational mem tag for resource usage
			 * tracking. */
			__u16 memtag;

			__u16 reserved0;

			/* GPU-visible physical memory alignment in
			 * bytes.
			 *
			 * Alignment must be a power of two. Minimum
			 * alignment is the small page size, which 0
			 * also denotes.
			 *
			 * For contiguous and non-contiguous
			 * allocations, the start address of the
			 * physical memory allocation will be aligned
			 * by this value.
			 *
			 * For non-contiguous allocations, memory is
			 * internally allocated in round_up(size /
			 * alignment) contiguous blocks. The start
			 * address of each block is aligned by the
			 * alignment value. If the size is not a
			 * multiple of alignment (which is ok), the
			 * last allocation block size is (size %
			 * alignment).
			 *
			 * By specifying the big page size here and
			 * allocation size that is a multiple of big
			 * pages, it will be guaranteed that the
			 * allocated buffer is big page size mappable.
			 */
			__u32 alignment;

			__u32 reserved1[3];
		} in;

		struct {
			__s32 dmabuf_fd;
		} out;
	};
};

/* Memory clock */
#define NVGPU_GPU_CLK_DOMAIN_MCLK                                (0)
/* Main graphics core clock */
#define NVGPU_GPU_CLK_DOMAIN_GPCCLK	                         (1)

struct nvgpu_gpu_clk_range {

	/* Flags (not currently used) */
	__u32 flags;

	/* NVGPU_GPU_CLK_DOMAIN_* */
	__u32 clk_domain;
	__u64 min_hz;
	__u64 max_hz;
};

/* Request on specific clock domains */
#define NVGPU_GPU_CLK_FLAG_SPECIFIC_DOMAINS		(1UL << 0)

struct nvgpu_gpu_clk_range_args {

	/* Flags. If NVGPU_GPU_CLK_FLAG_SPECIFIC_DOMAINS the request will
	   apply only to domains specified in clock entries. In this case
	   caller must set clock domain in each entry. Otherwise, the
	   ioctl will return all clock domains.
	*/
	__u32 flags;

	__u16 pad0;

	/* in/out: Number of entries in clk_range_entries buffer. If zero,
	   NVGPU_GPU_IOCTL_CLK_GET_RANGE will return 0 and
	   num_entries will be set to number of clock domains.
	 */
	__u16 num_entries;

	/* in: Pointer to clock range entries in the caller's address space.
	   size must be >= max_entries * sizeof(struct nvgpu_gpu_clk_range)
	 */
	__u64 clk_range_entries;
};

struct nvgpu_gpu_clk_vf_point {
	__u64 freq_hz;
};

struct nvgpu_gpu_clk_vf_points_args {

	/* in: Flags (not currently used) */
	__u32 flags;

	/* in: NVGPU_GPU_CLK_DOMAIN_* */
	__u32 clk_domain;

	/* in/out: max number of nvgpu_gpu_clk_vf_point entries in
	   clk_vf_point_entries.  If max_entries is zero,
	   NVGPU_GPU_IOCTL_CLK_GET_VF_POINTS will return 0 and max_entries will
	   be set to the max number of VF entries for this clock domain. If
	   there are more entries than max_entries, then ioctl will return
	   -EINVAL.
	*/
	__u16 max_entries;

	/* out: Number of nvgpu_gpu_clk_vf_point entries returned in
	   clk_vf_point_entries. Number of entries might vary depending on
	   thermal conditions.
	*/
	__u16 num_entries;

	__u32 reserved;

	/* in: Pointer to clock VF point entries in the caller's address space.
	   size must be >= max_entries * sizeof(struct nvgpu_gpu_clk_vf_point).
	 */
	__u64 clk_vf_point_entries;
};

/* Target clock requested by application*/
#define NVGPU_GPU_CLK_TYPE_TARGET	1
/* Actual clock frequency for the domain.
   May deviate from desired target frequency due to PLL constraints. */
#define NVGPU_GPU_CLK_TYPE_ACTUAL	2
/* Effective clock, measured from hardware */
#define NVGPU_GPU_CLK_TYPE_EFFECTIVE	3

struct nvgpu_gpu_clk_info {

	/* Flags (not currently used) */
	__u16 flags;

	/* in: When NVGPU_GPU_CLK_FLAG_SPECIFIC_DOMAINS set, indicates
	   the type of clock info to be returned for this entry. It is
	   allowed to have several entries with different clock types in
	   the same request (for instance query both target and actual
	   clocks for a given clock domain). This field is ignored for a
	   SET operation. */
	__u16 clk_type;

	/* NVGPU_GPU_CLK_DOMAIN_xxx */
	__u32 clk_domain;

	__u64 freq_hz;
};

struct nvgpu_gpu_clk_get_info_args {

	/* Flags. If NVGPU_GPU_CLK_FLAG_SPECIFIC_DOMAINS the request will
	   apply only to domains specified in clock entries. In this case
	   caller must set clock domain in each entry. Otherwise, the
	   ioctl will return all clock domains.
	*/
	__u32 flags;

	/* in: indicates which type of clock info to be returned (see
	   NVGPU_GPU_CLK_TYPE_xxx). If NVGPU_GPU_CLK_FLAG_SPECIFIC_DOMAINS
	   is defined, clk_type is specified in each clock info entry instead.
	 */
	__u16 clk_type;

	/* in/out: Number of clock info entries contained in clk_info_entries.
	   If zero, NVGPU_GPU_IOCTL_CLK_GET_INFO will return 0 and
	   num_entries will be set to number of clock domains. Also,
	   last_req_nr will be updated, which allows checking if a given
	   request has completed. If there are more entries than max_entries,
	   then ioctl will return -EINVAL.
	 */
	__u16 num_entries;

	/* in: Pointer to nvgpu_gpu_clk_info entries in the caller's address
	   space. Buffer size must be at least:
		num_entries * sizeof(struct nvgpu_gpu_clk_info)
	   If NVGPU_GPU_CLK_FLAG_SPECIFIC_DOMAINS is set, caller should set
	   clk_domain to be queried in  each entry. With this flag,
	   clk_info_entries passed to an NVGPU_GPU_IOCTL_CLK_SET_INFO,
	   can be re-used on completion for a NVGPU_GPU_IOCTL_CLK_GET_INFO.
	   This allows checking actual_mhz.
	 */
	__u64 clk_info_entries;

};

struct nvgpu_gpu_clk_set_info_args {

	/* in: Flags (not currently used). */
	__u32 flags;

	__u16 pad0;

	/* Number of clock info entries contained in clk_info_entries.
	   Must be > 0.
	 */
	__u16 num_entries;

	/* Pointer to clock info entries in the caller's address space. Buffer
	   size must be at least
		num_entries * sizeof(struct nvgpu_gpu_clk_info)
	 */
	__u64 clk_info_entries;

	/* out: File descriptor for request completion. Application can poll
	   this file descriptor to determine when the request has completed.
	   The fd must be closed afterwards.
	 */
	__s32 completion_fd;
};

struct nvgpu_gpu_get_event_fd_args {

	/* in: Flags (not currently used). */
	__u32 flags;

	/* out: File descriptor for events, e.g. clock update.
	 * On successful polling of this event_fd, application is
	 * expected to read status (nvgpu_gpu_event_info),
	 * which provides detailed event information
	 * For a poll operation, alarms will be reported with POLLPRI,
	 * and GPU shutdown will be reported with POLLHUP.
	 */
	__s32 event_fd;
};

struct nvgpu_gpu_get_memory_state_args {
	/*
	 * Current free space for this device; may change even when any
	 * kernel-managed metadata (e.g., page tables or channels) is allocated
	 * or freed. For an idle gpu, an allocation of this size would succeed.
	 */
	__u64 total_free_bytes;

	/* For future use; must be set to 0. */
	__u64 reserved[4];
};

struct nvgpu_gpu_get_fbp_l2_masks_args {
	/* [in]  L2 mask buffer size reserved by userspace. Should be
		 at least sizeof(__u32) * fls(fbp_en_mask) to receive LTC
		 mask for each FBP.
	   [out] full kernel buffer size
	*/
	__u32 mask_buf_size;
	__u32 reserved;

	/* [in]  pointer to L2 mask buffer. It will receive one
		 32-bit L2 mask per FBP or 0 if FBP is not enabled or
		 not present. This parameter is ignored if
		 mask_buf_size is 0. */
	__u64 mask_buf_addr;
};

#define NVGPU_GPU_VOLTAGE_CORE		1
#define NVGPU_GPU_VOLTAGE_SRAM		2
#define NVGPU_GPU_VOLTAGE_BUS		3	/* input to regulator */

struct nvgpu_gpu_get_voltage_args {
	__u64 reserved;
	__u32 which;		/* in: NVGPU_GPU_VOLTAGE_* */
	__u32 voltage;		/* uV */
};

struct nvgpu_gpu_get_current_args {
	__u32 reserved[3];
	__u32 currnt;		/* mA */
};

struct nvgpu_gpu_get_power_args {
	__u32 reserved[3];
	__u32 power;		/* mW */
};

struct nvgpu_gpu_get_temperature_args {
	__u32 reserved[3];
	/* Temperature in signed fixed point format SFXP24.8
	 *    Celsius = temp_f24_8 / 256.
	 */
	__s32 temp_f24_8;
};

struct nvgpu_gpu_set_therm_alert_limit_args {
	__u32 reserved[3];
	/* Temperature in signed fixed point format SFXP24.8
	 *    Celsius = temp_f24_8 / 256.
	 */
	__s32 temp_f24_8;
};

/*
 * Adjust options of deterministic channels in channel batches.
 *
 * This supports only one option currently: relax railgate blocking by
 * "disabling" the channel.
 *
 * Open deterministic channels do not allow the GPU to railgate by default. It
 * may be preferable to hold preopened channel contexts open and idle and still
 * railgate the GPU, taking the channels back into use dynamically in userspace
 * as an optimization. This ioctl allows to drop or reacquire the requirement
 * to hold GPU power on for individual channels. If allow_railgate is set on a
 * channel, no work can be submitted to it.
 *
 * num_channels is updated to signify how many channels were updated
 * successfully. It can be used to test which was the first update to fail.
 */
struct nvgpu_gpu_set_deterministic_opts_args {
	__u32 num_channels; /* in/out */
/*
 * Set or unset the railgating reference held by deterministic channels. If
 * the channel status is already the same as the flag, this is a no-op. Both
 * of these flags cannot be set at the same time. If none are set, the state
 * is left as is.
 */
#define NVGPU_GPU_SET_DETERMINISTIC_OPTS_FLAGS_ALLOW_RAILGATING    (1 << 0)
#define NVGPU_GPU_SET_DETERMINISTIC_OPTS_FLAGS_DISALLOW_RAILGATING (1 << 1)
	__u32 flags;        /* in */
	/*
	 * This is a pointer to an array of size num_channels.
	 *
	 * The channels have to be valid fds and be previously set as
	 * deterministic.
	 */
	__u64 channels; /* in */
};

/*
 * register buffer information ioctl.
 *
 * Note: Additional metadata is associated with the buffer only for supporting
 * legacy userspace APIs and for compatibility with desktop RM. Usage of this
 * API should be avoided.
 *
 * This ioctl allocates comptags for the buffer if requested/required
 * by libnvrm_gpu and associates metadata blob sent by libnvrm_gpu
 * with the buffer in the buffer privdata.
 *
 * return 0 on success, < 0 in case of failure.
 * retval -EINVAL if the enabled flag NVGPU_SUPPORT_BUFFER_METADATA
 *               isn't set or invalid params.
 * retval -EINVAL if the enabled flag NVGPU_SUPPORT_COMPRESSION
 *               isn't set and comptags are required.
 * retval -ENOMEM in case of sufficient memory is not available for
 *                privdata or comptags.
 * retval -EFAULT if the metadata blob copy fails.
 */

/*
 * NVGPU_GPU_COMPTAGS_ALLOC_NONE: Specified to not allocate comptags
 * for the buffer.
 */
#define NVGPU_GPU_COMPTAGS_ALLOC_NONE			0U

/*
 * NVGPU_GPU_COMPTAGS_ALLOC_REQUESTED: Specified to attempt comptags
 * allocation for the buffer. If comptags are not available, the
 * register buffer call will not fail and userspace can fallback
 * to no compression.
 */
#define NVGPU_GPU_COMPTAGS_ALLOC_REQUESTED		1U

/*
 * NVGPU_GPU_COMPTAGS_ALLOC_REQUIRED: Specified to allocate comptags
 * for the buffer when userspace can't fallback to no compression.
 * If comptags are not available, the register buffer call will fail.
 */
#define NVGPU_GPU_COMPTAGS_ALLOC_REQUIRED		2U

/*
 * If the comptags are allocated for the buffer, this flag is set in the output
 * flags in the register buffer ioctl.
 */
#define NVGPU_GPU_REGISTER_BUFFER_FLAGS_COMPTAGS_ALLOCATED	(1U << 0)

 /*
  * Specify buffer registration as mutable. This allows modifying the buffer
  * attributes by calling this IOCTL again with NVGPU_GPU_REGISTER_BUFFER_FLAGS_MODIFY.
  *
  * Mutable registration is intended for private buffers where the physical
  * memory allocation may be recycled. Buffers intended for interoperability
  * should be specified without this flag.
  */
#define NVGPU_GPU_REGISTER_BUFFER_FLAGS_MUTABLE			(1U << 1)

 /*
  * Re-register the buffer. When this flag is set, the buffer comptags state,
  * metadata binary blob, and other attributes are re-defined.
  *
  * This flag may be set only when the buffer was previously registered as
  * mutable. This flag is ignored when the buffer is registered for the
  * first time.
  *
  * If the buffer previously had comptags and the re-registration also specifies
  * comptags, the associated comptags are not cleared.
  *
  */
#define NVGPU_GPU_REGISTER_BUFFER_FLAGS_MODIFY			(1U << 2)

/* Maximum size of the user supplied buffer metadata */
#define NVGPU_GPU_REGISTER_BUFFER_METADATA_MAX_SIZE	256U

/*
 * register buffer ioctl arguments struct.
 *
 * Note: Additional metadata is associated with the buffer only for supporting
 * legacy userspace APIs and for compatibility with desktop RM. Usage of this
 * API should be avoided.
 */
struct nvgpu_gpu_register_buffer_args {
	/* [in] dmabuf fd */
	__s32 dmabuf_fd;

	/*
	 * [in] Compression tags allocation control.
	 *
	 * Set to one of the NVGPU_GPU_COMPTAGS_ALLOC_* values. See the
	 * description of the values for semantics of this field.
	 */
	__u8 comptags_alloc_control;
	__u8 reserved0;
	__u16 reserved1;

	/*
	 * [in] Pointer to buffer metadata.
	 *
	 * This is a binary blob populated by nvrm_gpu that will be associated
	 * with the dmabuf.
	 */
	__u64 metadata_addr;

	/* [in] buffer metadata size */
	__u32 metadata_size;

	/*
	 * [in/out] flags.
	 *
	 * See description of NVGPU_GPU_REGISTER_BUFFER_FLAGS_* for semantics
	 * of this field.
	 */
	__u32 flags;
};

#define NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE \
	_IOR(NVGPU_GPU_IOCTL_MAGIC, 1, struct nvgpu_gpu_zcull_get_ctx_size_args)
#define NVGPU_GPU_IOCTL_ZCULL_GET_INFO \
	_IOR(NVGPU_GPU_IOCTL_MAGIC, 2, struct nvgpu_gpu_zcull_get_info_args)
#define NVGPU_GPU_IOCTL_ZBC_SET_TABLE	\
	_IOW(NVGPU_GPU_IOCTL_MAGIC, 3, struct nvgpu_gpu_zbc_set_table_args)
#define NVGPU_GPU_IOCTL_ZBC_QUERY_TABLE	\
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 4, struct nvgpu_gpu_zbc_query_table_args)
#define NVGPU_GPU_IOCTL_GET_CHARACTERISTICS   \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 5, struct nvgpu_gpu_get_characteristics)
#define NVGPU_GPU_IOCTL_PREPARE_COMPRESSIBLE_READ \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 6, struct nvgpu_gpu_prepare_compressible_read_args)
#define NVGPU_GPU_IOCTL_MARK_COMPRESSIBLE_WRITE \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 7, struct nvgpu_gpu_mark_compressible_write_args)
#define NVGPU_GPU_IOCTL_ALLOC_AS \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 8, struct nvgpu_alloc_as_args)
#define NVGPU_GPU_IOCTL_OPEN_TSG \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 9, struct nvgpu_gpu_open_tsg_args)
#define NVGPU_GPU_IOCTL_GET_TPC_MASKS \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 10, struct nvgpu_gpu_get_tpc_masks_args)
#define NVGPU_GPU_IOCTL_OPEN_CHANNEL \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 11, struct nvgpu_gpu_open_channel_args)
#define NVGPU_GPU_IOCTL_FLUSH_L2 \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 12, struct nvgpu_gpu_l2_fb_args)
#define NVGPU_GPU_IOCTL_SET_MMUDEBUG_MODE \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 14, struct nvgpu_gpu_mmu_debug_mode_args)
#define NVGPU_GPU_IOCTL_SET_SM_DEBUG_MODE \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 15, struct nvgpu_gpu_sm_debug_mode_args)
#define NVGPU_GPU_IOCTL_WAIT_FOR_PAUSE \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 16, struct nvgpu_gpu_wait_pause_args)
#define NVGPU_GPU_IOCTL_GET_TPC_EXCEPTION_EN_STATUS \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 17, struct nvgpu_gpu_tpc_exception_en_status_args)
#define NVGPU_GPU_IOCTL_NUM_VSMS \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 18, struct nvgpu_gpu_num_vsms)
#define NVGPU_GPU_IOCTL_VSMS_MAPPING \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 19, struct nvgpu_gpu_vsms_mapping)
#define NVGPU_GPU_IOCTL_RESUME_FROM_PAUSE \
	_IO(NVGPU_GPU_IOCTL_MAGIC, 21)
#define NVGPU_GPU_IOCTL_TRIGGER_SUSPEND \
	_IO(NVGPU_GPU_IOCTL_MAGIC, 22)
#define NVGPU_GPU_IOCTL_CLEAR_SM_ERRORS \
	_IO(NVGPU_GPU_IOCTL_MAGIC, 23)
#define NVGPU_GPU_IOCTL_GET_CPU_TIME_CORRELATION_INFO \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 24, \
			struct nvgpu_gpu_get_cpu_time_correlation_info_args)
#define NVGPU_GPU_IOCTL_GET_GPU_TIME \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 25, \
			struct nvgpu_gpu_get_gpu_time_args)
#define NVGPU_GPU_IOCTL_GET_ENGINE_INFO \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 26, \
			struct nvgpu_gpu_get_engine_info_args)
#define NVGPU_GPU_IOCTL_ALLOC_VIDMEM \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 27, \
			struct nvgpu_gpu_alloc_vidmem_args)
#define NVGPU_GPU_IOCTL_CLK_GET_RANGE \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 28, struct nvgpu_gpu_clk_range_args)
#define NVGPU_GPU_IOCTL_CLK_GET_VF_POINTS \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 29, struct nvgpu_gpu_clk_vf_points_args)
#define NVGPU_GPU_IOCTL_CLK_GET_INFO \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 30, struct nvgpu_gpu_clk_get_info_args)
#define NVGPU_GPU_IOCTL_CLK_SET_INFO \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 31, struct nvgpu_gpu_clk_set_info_args)
#define NVGPU_GPU_IOCTL_GET_EVENT_FD \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 32, struct nvgpu_gpu_get_event_fd_args)
#define NVGPU_GPU_IOCTL_GET_MEMORY_STATE \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 33, \
			struct nvgpu_gpu_get_memory_state_args)
#define NVGPU_GPU_IOCTL_GET_VOLTAGE \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 34, struct nvgpu_gpu_get_voltage_args)
#define NVGPU_GPU_IOCTL_GET_CURRENT \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 35, struct nvgpu_gpu_get_current_args)
#define NVGPU_GPU_IOCTL_GET_POWER \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 36, struct nvgpu_gpu_get_power_args)
#define NVGPU_GPU_IOCTL_GET_TEMPERATURE \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 37, struct nvgpu_gpu_get_temperature_args)
#define NVGPU_GPU_IOCTL_GET_FBP_L2_MASKS \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 38, struct nvgpu_gpu_get_fbp_l2_masks_args)
#define NVGPU_GPU_IOCTL_SET_THERM_ALERT_LIMIT \
		_IOWR(NVGPU_GPU_IOCTL_MAGIC, 39, \
			struct nvgpu_gpu_set_therm_alert_limit_args)
#define NVGPU_GPU_IOCTL_SET_DETERMINISTIC_OPTS \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 40, \
			struct nvgpu_gpu_set_deterministic_opts_args)
#define NVGPU_GPU_IOCTL_REGISTER_BUFFER	\
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 41, struct nvgpu_gpu_register_buffer_args)
#define NVGPU_GPU_IOCTL_GET_BUFFER_INFO	\
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 42, struct nvgpu_gpu_get_buffer_info_args)
#define NVGPU_GPU_IOCTL_GET_GPC_LOCAL_TO_PHYSICAL_MAP\
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 43, struct nvgpu_gpu_get_gpc_physical_map_args)
#define NVGPU_GPU_IOCTL_GET_GPC_LOCAL_TO_LOGICAL_MAP\
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 44, struct nvgpu_gpu_get_gpc_logical_map_args)
#define NVGPU_GPU_IOCTL_LAST		\
	_IOC_NR(NVGPU_GPU_IOCTL_GET_GPC_LOCAL_TO_LOGICAL_MAP)
#define NVGPU_GPU_IOCTL_MAX_ARG_SIZE	\
	sizeof(struct nvgpu_gpu_get_cpu_time_correlation_info_args)

#endif /* _UAPI__LINUX_NVGPU_CTRL_H__ */
