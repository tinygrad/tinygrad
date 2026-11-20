// SPDX-License-Identifier: GPL-2.0

#ifndef _LINUX_TYPES_H
#define _LINUX_TYPES_H

#define BITS_PER_BYTE			8
#define __KERNEL_DIV_ROUND_UP(n, d)	(((n) + (d) - 1) / (d))
#define DIV_ROUND_UP			__KERNEL_DIV_ROUND_UP
#define BITS_PER_TYPE(type)		(sizeof(type) * BITS_PER_BYTE)
#define BITS_TO_LONGS(nr)		DIV_ROUND_UP(nr, BITS_PER_TYPE(long))
#define DECLARE_BITMAP(name, bits)\
	unsigned long name[BITS_TO_LONGS(bits)]

#define offsetof(TYPE, MEMBER)		((size_t)&((TYPE *)0)->MEMBER)
#define NULL				((void *)0)
#define __FD_SETSIZE			1024

enum {
	false   = 0,
	true    = 1
};

typedef struct {
	unsigned long fds_bits[__FD_SETSIZE / (8 * sizeof(long))];
} __kernel_fd_set;
typedef __kernel_fd_set			fd_set;

typedef unsigned short			umode_t;
typedef _Bool				bool;
typedef unsigned long			uintptr_t;

#ifndef __kernel_long_t
typedef long				__kernel_long_t;
typedef unsigned long			__kernel_ulong_t;
#endif

#ifndef __kernel_uid32_t
typedef unsigned int			__kernel_uid32_t;
typedef unsigned int			__kernel_gid32_t;
#endif

#ifndef __kernel_ino_t
typedef __kernel_ulong_t		__kernel_ino_t;
#endif
typedef __kernel_ino_t			ino_t;

typedef int				__kernel_key_t;
typedef __kernel_key_t			key_t;

typedef int				__kernel_clockid_t;
typedef __kernel_clockid_t		clockid_t;

typedef int				__kernel_timer_t;
typedef __kernel_timer_t		timer_t;

typedef __kernel_uid32_t		uid_t;
typedef __kernel_gid32_t		gid_t;

typedef __kernel_long_t			__kernel_off_t;
typedef __kernel_off_t			off_t;

#ifndef __kernel_pid_t
typedef int				__kernel_pid_t;
#endif
typedef __kernel_pid_t			pid_t;

typedef long long			__kernel_loff_t;
#if defined(__GNUC__)
typedef __kernel_loff_t			loff_t;
#endif

#ifndef __kernel_ino_t
typedef __kernel_ulong_t		__kernel_ino_t;
#endif

typedef __kernel_ulong_t		__kernel_size_t;
typedef __kernel_long_t			__kernel_ssize_t;
typedef __kernel_long_t			__kernel_ptrdiff_t;
typedef __kernel_long_t			__kernel_clock_t;
typedef __kernel_long_t			__kernel_time_t;

#ifndef _SIZE_T
#define _SIZE_T
typedef __kernel_size_t			size_t;
#endif

#ifndef _SSIZE_T
#define _SSIZE_T
typedef __kernel_ssize_t		ssize_t;
#endif

#ifndef _PTRDIFF_T
#define _PTRDIFF_T
typedef __kernel_ptrdiff_t		ptrdiff_t;
#endif

#ifndef _CLOCK_T
#define _CLOCK_T
typedef __kernel_clock_t		clock_t;
#endif

#ifndef _TIME_T
#define _TIME_T
typedef __kernel_time_t			time_t;
#endif

typedef unsigned char			unchar;
typedef unsigned short			ushort;
typedef unsigned int			uint;
typedef unsigned long			ulong;

typedef signed char			__s8;
typedef unsigned char			__u8;

typedef signed short			__s16;
typedef unsigned short			__u16;

typedef signed int			__s32;
typedef unsigned int			__u32;

#ifdef __GNUC__
typedef signed long long		__s64;
typedef unsigned long long		__u64;
#endif

typedef __s8				s8;
typedef __u8				u8;
typedef __s16				s16;
typedef __u16				u16;
typedef __s32				s32;
typedef __u32				u32;
typedef __s64				s64;
typedef __u64				u64;

typedef s8				int8_t;
typedef s16				int16_t;
typedef s32				int32_t;

typedef u8				uint8_t;
typedef u16				uint16_t;
typedef u32				uint32_t;

#if defined(__GNUC__)
typedef u64				uint64_t;
typedef u64				u_int64_t;
typedef s64				int64_t;
#endif

#define S8_C(x)				x
#define U8_C(x)				x ## U
#define S16_C(x)			x
#define U16_C(x)			x ## U
#define S32_C(x)			x
#define U32_C(x)			x ## U
#define S64_C(x)			x ## LL
#define U64_C(x)			x ## ULL

/* this is a special 64bit data type that is 8-byte aligned */
#define aligned_u64			__aligned_u64
#define aligned_be64			__aligned_be64
#define aligned_le64			__aligned_le64

/*
 * The type of an index into the pagecache.
 */
#define pgoff_t				unsigned long


typedef u64				dma_addr_t;

typedef unsigned int			gfp_t;
typedef unsigned int			slab_flags_t;
typedef unsigned int			fmode_t;
typedef unsigned			__poll_t;

typedef u32				__kernel_dev_t;
typedef __kernel_dev_t			dev_t;
typedef u64				sector_t;
typedef u64				blkcnt_t;
typedef u64				phys_addr_t;

typedef phys_addr_t			resource_size_t;

typedef unsigned long			irq_hw_number_t;

typedef struct {
	int counter;
} atomic_t;

typedef struct {
	s64 counter;
} atomic64_t;

struct list_head {
	struct list_head *next, *prev;
};

struct hlist_head {
	struct hlist_node *first;
};

struct hlist_node {
	struct hlist_node *next, **pprev;
};

struct callback_head {
	struct callback_head *next;
	void (*func)(struct callback_head *head);
} __attribute__((aligned(sizeof(void *))));
#define rcu_head callback_head

typedef void (*rcu_callback_t)(struct rcu_head *head);
typedef void (*call_rcu_func_t)(struct rcu_head *head, rcu_callback_t func);

#endif /* _LINUX_TYPES_H */
