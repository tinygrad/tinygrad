/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * Copyright (c) 2009-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * structure declarations for nvmem and nvmap user-space ioctls
 */

#include <linux/ioctl.h>
#include <linux/types.h>

#ifndef __UAPI_LINUX_NVMAP_H
#define __UAPI_LINUX_NVMAP_H

/*
 * DOC: NvMap Userspace API
 *
 * create a client by opening /dev/nvmap
 * most operations handled via following ioctls
 *
 */
enum {
	NVMAP_HANDLE_PARAM_SIZE = 1,
	NVMAP_HANDLE_PARAM_ALIGNMENT,
	NVMAP_HANDLE_PARAM_BASE,
	NVMAP_HANDLE_PARAM_HEAP,
	NVMAP_HANDLE_PARAM_KIND,
	NVMAP_HANDLE_PARAM_COMPR, /* ignored, to be removed */
};

enum {
	NVMAP_CACHE_OP_WB = 0,
	NVMAP_CACHE_OP_INV,
	NVMAP_CACHE_OP_WB_INV,
};

enum {
	NVMAP_PAGES_UNRESERVE = 0,
	NVMAP_PAGES_RESERVE,
	NVMAP_INSERT_PAGES_ON_UNRESERVE,
	NVMAP_PAGES_PROT_AND_CLEAN,
};

#define NVMAP_ELEM_SIZE_U64 (1 << 31)

struct nvmap_create_handle {
	union {
		struct {
			union {
				/* size will be overwritten */
				__u32 size;	/* CreateHandle */
				__s32 fd;	/* DmaBufFd or FromFd */
			};
			__u32 handle;		/* returns nvmap handle */
		};
		struct {
			/* one is input parameter, and other is output parameter
			 * since its a union please note that input parameter
			 * will be overwritten once ioctl returns
			 */
			union {
				__u64 ivm_id;	 /* CreateHandle from ivm*/
				__u32 ivm_handle;/* Get ivm_id from handle */
			};
		};
		struct {
			union {
				/* size64 will be overwritten */
				__u64 size64; /* CreateHandle */
				__u32 handle64; /* returns nvmap handle */
			};
		};
	};
};

struct nvmap_create_handle_from_va {
	__u64 va;		/* FromVA*/
	__u32 size;		/* non-zero for partial memory VMA. zero for end of VMA */
	__u32 flags;		/* wb/wc/uc/iwb, tag etc. */
	union {
		__u32 handle;		/* returns nvmap handle */
		__u64 size64;		/* used when size is 0 */
	};
};

struct nvmap_gup_test {
	__u64 va;		/* FromVA*/
	__u32 handle;		/* returns nvmap handle */
	__u32 result;		/* result=1 for pass, result=-err for failure */
};

struct nvmap_alloc_handle {
	__u32 handle;		/* nvmap handle */
	__u32 heap_mask;	/* heaps to allocate from */
	__u32 flags;		/* wb/wc/uc/iwb etc. */
	__u32 align;		/* min alignment necessary */
	__s32 numa_nid;		/* NUMA node id */
};

struct nvmap_alloc_ivm_handle {
	__u32 handle;		/* nvmap handle */
	__u32 heap_mask;	/* heaps to allocate from */
	__u32 flags;		/* wb/wc/uc/iwb etc. */
	__u32 align;		/* min alignment necessary */
	__u32 peer;		/* peer with whom handle must be shared. Used
				 *  only for NVMAP_HEAP_CARVEOUT_IVM
				 */
};

struct nvmap_rw_handle {
	__u64 addr;		/* user pointer*/
	__u32 handle;		/* nvmap handle */
	__u64 offset;		/* offset into hmem */
	__u64 elem_size;	/* individual atom size */
	__u64 hmem_stride;	/* delta in bytes between atoms in hmem */
	__u64 user_stride;	/* delta in bytes between atoms in user */
	__u64 count;		/* number of atoms to copy */
};

#ifdef __KERNEL__
#ifdef CONFIG_COMPAT
struct nvmap_rw_handle_32 {
	__u32 addr;		/* user pointer */
	__u32 handle;		/* nvmap handle */
	__u32 offset;		/* offset into hmem */
	__u32 elem_size;	/* individual atom size */
	__u32 hmem_stride;	/* delta in bytes between atoms in hmem */
	__u32 user_stride;	/* delta in bytes between atoms in user */
	__u32 count;		/* number of atoms to copy */
};
#endif /* CONFIG_COMPAT */
#endif /* __KERNEL__ */

struct nvmap_handle_param {
	__u32 handle;		/* nvmap handle */
	__u32 param;		/* size/align/base/heap etc. */
	unsigned long result;	/* returns requested info*/
};

#ifdef __KERNEL__
#ifdef CONFIG_COMPAT
struct nvmap_handle_param_32 {
	__u32 handle;		/* nvmap handle */
	__u32 param;		/* size/align/base/heap etc. */
	__u32 result;		/* returns requested info*/
};
#endif /* CONFIG_COMPAT */
#endif /* __KERNEL__ */

struct nvmap_cache_op {
	unsigned long addr;	/* user pointer*/
	__u32 handle;		/* nvmap handle */
	__u32 len;		/* bytes to flush */
	__s32 op;		/* wb/wb_inv/inv */
};

struct nvmap_cache_op_64 {
	unsigned long addr;	/* user pointer*/
	__u32 handle;		/* nvmap handle */
	__u64 len;		/* bytes to flush */
	__s32 op;		/* wb/wb_inv/inv */
};

#ifdef __KERNEL__
#ifdef CONFIG_COMPAT
struct nvmap_cache_op_32 {
	__u32 addr;		/* user pointer*/
	__u32 handle;		/* nvmap handle */
	__u32 len;		/* bytes to flush */
	__s32 op;		/* wb/wb_inv/inv */
};
#endif /* CONFIG_COMPAT */
#endif /* __KERNEL__ */

struct nvmap_cache_op_list {
	__u64 handles;		/* Ptr to u32 type array, holding handles */
	__u64 offsets;		/* Ptr to u32 type array, holding offsets
				 * into handle mem */
	__u64 sizes;		/* Ptr to u32 type array, holindg sizes of memory
				 * regions within each handle */
	__u32 nr;		/* Number of handles */
	__s32 op;		/* wb/wb_inv/inv */
};

struct nvmap_debugfs_handles_header {
	__u8 version;
};

struct nvmap_debugfs_handles_entry {
	__u64 base;
	__u64 size;
	__u32 flags;
	__u32 share_count;
	__u64 mapped_size;
};

struct nvmap_set_tag_label {
	__u32 tag;
	__u32 len;		/* in: label length
				   out: number of characters copied */
	__u64 addr;		/* in: pointer to label or NULL to remove */
};

struct nvmap_available_heaps {
	__u64 heaps;		/* heaps bitmask */
};

struct nvmap_heap_size {
	__u32 heap;
	__u64 size;
};

struct nvmap_sciipc_map {
	__u64 auth_token;    /* AuthToken */
	__u32 flags;       /* Exporter permission flags */
	__u64 sci_ipc_id;  /* FromImportId */
	__u32 handle;      /* Nvmap handle */
};

struct nvmap_handle_parameters {
    __u8 contig;
    __u32 import_id;
    __u32 handle;
    __u32 heap_number;
    __u32 access_flags;
    __u64 heap;
    __u64 align;
    __u64 coherency;
    __u64 size;
    __u64 offset;
    __u64 serial_id;
};

/**
 * Struct used while querying heap parameters
 */
struct nvmap_query_heap_params {
	__u32 heap_mask;
	__u32 flags;
	__u8 contig;
	__u64 total;
	__u64 free;
	__u64 largest_free_block;
	__u32 granule_size;
};

/**
 * Struct used while duplicating memory handle
 */
struct nvmap_duplicate_handle {
	__u32 handle;
	__u32 access_flags;
	__u32 dup_handle;
};

/**
 * Struct used while duplicating memory handle
 */
struct nvmap_fd_for_range_from_list {
	__u32 *handles;  /* Head of handles list */
	__u32 num_handles; /* Number of handles in the list */
	__u64 offset; /* Offset aligned by page size in the buffers */
	__u64 size; /* Size of the sub buffer for which fd to be returned */
	__s32 fd; /* Sub range Dma Buf fd to be returned*/
};

#define NVMAP_IOC_MAGIC 'N'

/* Creates a new memory handle. On input, the argument is the size of the new
 * handle; on return, the argument is the name of the new handle
 */
#define NVMAP_IOC_CREATE  _IOWR(NVMAP_IOC_MAGIC, 0, struct nvmap_create_handle)
#define NVMAP_IOC_CREATE_64 \
	_IOWR(NVMAP_IOC_MAGIC, 1, struct nvmap_create_handle)
#define NVMAP_IOC_FROM_ID _IOWR(NVMAP_IOC_MAGIC, 2, struct nvmap_create_handle)

/* Actually allocates memory for the specified handle */
#define NVMAP_IOC_ALLOC    _IOW(NVMAP_IOC_MAGIC, 3, struct nvmap_alloc_handle)

/* Frees a memory handle, unpinning any pinned pages and unmapping any mappings
 */
#define NVMAP_IOC_FREE       _IO(NVMAP_IOC_MAGIC, 4)

/* Reads/writes data (possibly strided) from a user-provided buffer into the
 * hmem at the specified offset */
#define NVMAP_IOC_WRITE      _IOW(NVMAP_IOC_MAGIC, 6, struct nvmap_rw_handle)
#define NVMAP_IOC_READ       _IOW(NVMAP_IOC_MAGIC, 7, struct nvmap_rw_handle)
#ifdef __KERNEL__
#ifdef CONFIG_COMPAT
#define NVMAP_IOC_WRITE_32   _IOW(NVMAP_IOC_MAGIC, 6, struct nvmap_rw_handle_32)
#define NVMAP_IOC_READ_32    _IOW(NVMAP_IOC_MAGIC, 7, struct nvmap_rw_handle_32)
#endif /* CONFIG_COMPAT */
#endif /* __KERNEL__ */

#define NVMAP_IOC_PARAM _IOWR(NVMAP_IOC_MAGIC, 8, struct nvmap_handle_param)
#ifdef __KERNEL__
#ifdef CONFIG_COMPAT
#define NVMAP_IOC_PARAM_32 _IOWR(NVMAP_IOC_MAGIC, 8, struct nvmap_handle_param_32)
#endif /* CONFIG_COMPAT */
#endif /* __KERNEL__ */

#define NVMAP_IOC_CACHE      _IOW(NVMAP_IOC_MAGIC, 12, struct nvmap_cache_op)
#define NVMAP_IOC_CACHE_64   _IOW(NVMAP_IOC_MAGIC, 12, struct nvmap_cache_op_64)
#ifdef __KERNEL__
#ifdef CONFIG_COMPAT
#define NVMAP_IOC_CACHE_32  _IOW(NVMAP_IOC_MAGIC, 12, struct nvmap_cache_op_32)
#endif /* CONFIG_COMPAT */
#endif /* __KERNEL__ */

/* Returns a global ID usable to allow a remote process to create a handle
 * reference to the same handle */
#define NVMAP_IOC_GET_ID  _IOWR(NVMAP_IOC_MAGIC, 13, struct nvmap_create_handle)

/* Returns a file id that allows a remote process to create a handle
 * reference to the same handle */
#define NVMAP_IOC_GET_FD  _IOWR(NVMAP_IOC_MAGIC, 15, struct nvmap_create_handle)

/* Create a new memory handle from file id passed */
#define NVMAP_IOC_FROM_FD _IOWR(NVMAP_IOC_MAGIC, 16, struct nvmap_create_handle)

/* Perform cache maintenance on a list of handles. */
#define NVMAP_IOC_CACHE_LIST _IOW(NVMAP_IOC_MAGIC, 17,	\
				  struct nvmap_cache_op_list)

#define NVMAP_IOC_FROM_IVC_ID _IOWR(NVMAP_IOC_MAGIC, 19, struct nvmap_create_handle)
#define NVMAP_IOC_GET_IVC_ID _IOWR(NVMAP_IOC_MAGIC, 20, struct nvmap_create_handle)
#define NVMAP_IOC_GET_IVM_HEAPS _IOR(NVMAP_IOC_MAGIC, 21, unsigned int)

/* Create a new memory handle from VA passed */
#define NVMAP_IOC_FROM_VA _IOWR(NVMAP_IOC_MAGIC, 22, struct nvmap_create_handle_from_va)

#define NVMAP_IOC_GUP_TEST _IOWR(NVMAP_IOC_MAGIC, 23, struct nvmap_gup_test)

/* Define a label for allocation tag */
#define NVMAP_IOC_SET_TAG_LABEL	_IOW(NVMAP_IOC_MAGIC, 24, struct nvmap_set_tag_label)

#define NVMAP_IOC_GET_AVAILABLE_HEAPS \
	_IOR(NVMAP_IOC_MAGIC, 25, struct nvmap_available_heaps)

#define NVMAP_IOC_GET_HEAP_SIZE \
	_IOR(NVMAP_IOC_MAGIC, 26, struct nvmap_heap_size)

#define NVMAP_IOC_PARAMETERS \
	_IOR(NVMAP_IOC_MAGIC, 27, struct nvmap_handle_parameters)

/* START of T124 IOCTLS */
/* Actually allocates memory from IVM heaps */
#define NVMAP_IOC_ALLOC_IVM _IOW(NVMAP_IOC_MAGIC, 101, struct nvmap_alloc_ivm_handle)

/* Allocate seperate memory for VPR */
#define NVMAP_IOC_VPR_FLOOR_SIZE _IOW(NVMAP_IOC_MAGIC, 102, __u32)

/* Get SCI_IPC_ID tied up with nvmap_handle */
#define NVMAP_IOC_GET_SCIIPCID _IOR(NVMAP_IOC_MAGIC, 103, \
		struct nvmap_sciipc_map)

/* Get Nvmap handle from SCI_IPC_ID */
#define NVMAP_IOC_HANDLE_FROM_SCIIPCID _IOR(NVMAP_IOC_MAGIC, 104, \
		struct nvmap_sciipc_map)

/* Get heap parameters such as total and frre size */
#define NVMAP_IOC_QUERY_HEAP_PARAMS _IOR(NVMAP_IOC_MAGIC, 105, \
		struct nvmap_query_heap_params)

/* Duplicate NvRmMemHandle with same/reduced permission */
#define NVMAP_IOC_DUP_HANDLE _IOWR(NVMAP_IOC_MAGIC, 106, \
		struct nvmap_duplicate_handle)

/*  Get for range from list of NvRmMemHandles */
#define NVMAP_IOC_GET_FD_FOR_RANGE_FROM_LIST _IOR(NVMAP_IOC_MAGIC, 107, \
		struct nvmap_fd_for_range_from_list)

#define NVMAP_IOC_MAXNR (_IOC_NR(NVMAP_IOC_GET_FD_FOR_RANGE_FROM_LIST))

#endif /* __UAPI_LINUX_NVMAP_H */
