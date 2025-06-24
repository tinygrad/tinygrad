#ifndef TDMA_H
#define TDMA_H

#include <linux/types.h>
#include <linux/ioctl.h>

struct tdma_ioctl_usge {
  __u64 offset;
  __u64 size;
};

struct tdma_ioctl {
  __u16 domain;
  __u8 bus;
  __u8 device;
  __u8 function;
  __u8 bar;
  __u16 usgl_size;
  struct tdma_ioctl_usge* usgl;
  __s32 out_fd;
};

#define TDMA_GET_DMABUF   _IOWR('T', 0x01, struct tdma_ioctl)

#endif
