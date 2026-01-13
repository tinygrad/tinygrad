// tinygpu shim - A signed helper binary for Python to access PCI BARs
// Compile: clang -framework IOKit -framework CoreFoundation -o shim shim.c
// Usage: ./shim <socket_path>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>

// Protocol commands
enum {
  CMD_MAP_BAR = 1,      // Map BAR, returns info
  CMD_MAP_SYSMEM = 2,   // Allocate DMA memory
  CMD_CFG_READ = 3,     // PCI config read
  CMD_CFG_WRITE = 4,    // PCI config write
  CMD_RESET = 5,        // Device reset
  CMD_MMIO_READ = 6,    // MMIO read from BAR
  CMD_MMIO_WRITE = 7,   // MMIO write to BAR
  CMD_MMIO_BULK_READ = 8,   // Bulk MMIO read
  CMD_MMIO_BULK_WRITE = 9,  // Bulk MMIO write
  CMD_SYSMEM_BULK_READ = 10,  // Bulk sysmem read
  CMD_SYSMEM_BULK_WRITE = 11, // Bulk sysmem write
  CMD_QUIT = 255,
};

// Response codes
enum {
  RESP_OK = 0,
  RESP_ERR = 1,
};

// Message structures
typedef struct {
  uint8_t cmd;
  uint8_t bar;          // for MAP_BAR, MMIO
  uint64_t offset;      // for CFG_READ/WRITE, MMIO offset (64-bit for large BARs)
  uint64_t size;        // for CFG_READ/WRITE or MAP_SYSMEM, MMIO width
  uint64_t value;       // for CFG_WRITE, MMIO_WRITE
} __attribute__((packed)) request_t;

typedef struct {
  uint8_t status;
  uint64_t value;       // for CFG_READ result or mapped size or MMIO_READ
  uint64_t addr;        // mapped virtual address (for info)
} __attribute__((packed)) response_t;

// Static buffer for bulk transfers (no malloc)
#define BULK_BUF_SIZE (16 * 1024 * 1024)  // 16MB
static uint8_t g_bulk_buf[BULK_BUF_SIZE];

// Global state
static io_connect_t g_conn = IO_OBJECT_NULL;

// BAR mappings (up to 6 BARs)
#define MAX_BARS 6
static struct {
  mach_vm_address_t addr;
  mach_vm_size_t size;
  int mapped;
} g_bars[MAX_BARS];

// Sysmem mappings (up to 64 allocations)
#define MAX_SYSMEM 64
static struct {
  mach_vm_address_t addr;
  mach_vm_size_t size;
  uint64_t mem_type;  // used to unmap
  int mapped;
} g_sysmem[MAX_SYSMEM];
static int g_sysmem_count = 0;

static io_connect_t open_tinygpu(void) {
  io_service_t svc = IOServiceGetMatchingService(kIOMasterPortDefault,
                                                  IOServiceNameMatching("tinygpu"));
  if (!svc) {
    fprintf(stderr, "shim: tinygpu service not found\n");
    return IO_OBJECT_NULL;
  }

  io_connect_t conn = IO_OBJECT_NULL;
  kern_return_t kr = IOServiceOpen(svc, mach_task_self(), 0, &conn);
  IOObjectRelease(svc);

  if (kr != KERN_SUCCESS) {
    fprintf(stderr, "shim: IOServiceOpen failed: 0x%x\n", kr);
    return IO_OBJECT_NULL;
  }
  return conn;
}

static int map_bar(uint32_t bar, mach_vm_address_t *out_addr, mach_vm_size_t *out_size) {
  if (bar >= MAX_BARS) return -1;

  // Return cached mapping if already mapped
  if (g_bars[bar].mapped) {
    *out_addr = g_bars[bar].addr;
    *out_size = g_bars[bar].size;
    return 0;
  }

  mach_vm_address_t addr = 0;
  mach_vm_size_t size = 0;
  kern_return_t kr = IOConnectMapMemory64(g_conn, bar, mach_task_self(),
                                          &addr, &size, kIOMapAnywhere);
  if (kr != KERN_SUCCESS) {
    fprintf(stderr, "shim: IOConnectMapMemory64(bar=%u) failed: 0x%x\n", bar, kr);
    return -1;
  }

  g_bars[bar].addr = addr;
  g_bars[bar].size = size;
  g_bars[bar].mapped = 1;

  *out_addr = addr;
  *out_size = size;
  printf("shim: BAR%u mapped at 0x%llx size 0x%llx\n", bar, addr, size);
  return 0;
}

static int map_sysmem(uint64_t size, mach_vm_address_t *out_addr, mach_vm_size_t *out_size, int *out_idx) {
  if (g_sysmem_count >= MAX_SYSMEM) {
    fprintf(stderr, "shim: max sysmem allocations reached\n");
    return -1;
  }

  // type >= 6 is interpreted as size for DMA allocation
  mach_vm_address_t addr = 0;
  mach_vm_size_t mapped_size = 0;
  kern_return_t kr = IOConnectMapMemory64(g_conn, size, mach_task_self(),
                                          &addr, &mapped_size, kIOMapAnywhere);
  if (kr != KERN_SUCCESS) {
    fprintf(stderr, "shim: IOConnectMapMemory64(sysmem size=%llu) failed: 0x%x\n", size, kr);
    return -1;
  }

  int idx = g_sysmem_count++;
  g_sysmem[idx].addr = addr;
  g_sysmem[idx].size = mapped_size;
  g_sysmem[idx].mem_type = size;
  g_sysmem[idx].mapped = 1;

  *out_addr = addr;
  *out_size = mapped_size;
  *out_idx = idx;
  printf("shim: sysmem[%d] mapped at 0x%llx size 0x%llx\n", idx, addr, mapped_size);
  return 0;
}

// MMIO read from mapped BAR
static int mmio_read(uint8_t bar, uint32_t offset, uint32_t width, uint64_t *out_val) {
  if (bar >= MAX_BARS || !g_bars[bar].mapped) return -1;
  if (offset + width > g_bars[bar].size) return -1;

  volatile void *ptr = (volatile void*)(g_bars[bar].addr + offset);
  switch (width) {
    case 1: *out_val = *(volatile uint8_t*)ptr; break;
    case 2: *out_val = *(volatile uint16_t*)ptr; break;
    case 4: *out_val = *(volatile uint32_t*)ptr; break;
    case 8: *out_val = *(volatile uint64_t*)ptr; break;
    default: return -1;
  }
  return 0;
}

// MMIO write to mapped BAR
static int mmio_write(uint8_t bar, uint32_t offset, uint32_t width, uint64_t value) {
  if (bar >= MAX_BARS || !g_bars[bar].mapped) return -1;
  if (offset + width > g_bars[bar].size) return -1;

  volatile void *ptr = (volatile void*)(g_bars[bar].addr + offset);
  switch (width) {
    case 1: *(volatile uint8_t*)ptr = (uint8_t)value; break;
    case 2: *(volatile uint16_t*)ptr = (uint16_t)value; break;
    case 4: *(volatile uint32_t*)ptr = (uint32_t)value; break;
    case 8: *(volatile uint64_t*)ptr = value; break;
    default: return -1;
  }
  return 0;
}

static int cfg_read(uint64_t offset, uint64_t size, uint64_t *out_val) {
  uint64_t in_scalars[2] = {offset, size};
  uint64_t out_scalars[16];
  uint32_t out_cnt = 16;

  kern_return_t kr = IOConnectCallMethod(g_conn, 0 /* ReadCfg */,
                                          in_scalars, 2, NULL, 0,
                                          out_scalars, &out_cnt, NULL, NULL);
  if (kr != KERN_SUCCESS) {
    fprintf(stderr, "shim: cfg_read failed: 0x%x\n", kr);
    return -1;
  }
  *out_val = out_scalars[0];
  return 0;
}

static int cfg_write(uint64_t offset, uint64_t size, uint64_t value) {
  uint64_t in_scalars[3] = {offset, size, value};
  uint64_t out_scalars[16];
  uint32_t out_cnt = 16;

  kern_return_t kr = IOConnectCallMethod(g_conn, 1 /* WriteCfg */,
                                          in_scalars, 3, NULL, 0,
                                          out_scalars, &out_cnt, NULL, NULL);
  if (kr != KERN_SUCCESS) {
    fprintf(stderr, "shim: cfg_write failed: 0x%x\n", kr);
    return -1;
  }
  return 0;
}

static int device_reset(void) {
  uint64_t out_scalars[16];
  uint32_t out_cnt = 16;

  kern_return_t kr = IOConnectCallMethod(g_conn, 2 /* Reset */,
                                          NULL, 0, NULL, 0,
                                          out_scalars, &out_cnt, NULL, NULL);
  if (kr != KERN_SUCCESS) {
    fprintf(stderr, "shim: reset failed: 0x%x\n", kr);
    return -1;
  }
  return 0;
}

// Send response with optional file descriptor
static int send_response(int client_fd, response_t *resp, int send_fd) {
  struct msghdr msg = {0};
  struct iovec iov;
  char cmsgbuf[CMSG_SPACE(sizeof(int))];

  iov.iov_base = resp;
  iov.iov_len = sizeof(*resp);
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  if (send_fd >= 0) {
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &send_fd, sizeof(int));
  }

  return sendmsg(client_fd, &msg, 0) > 0 ? 0 : -1;
}

static void handle_client(int client_fd) {
  request_t req;
  response_t resp;

  // Increase socket buffer sizes for faster bulk transfers
  int bufsize = 16 * 1024 * 1024;  // 16MB
  setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
  setsockopt(client_fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));

  printf("shim: client connected\n");

  while (1) {
    ssize_t n = recv(client_fd, &req, sizeof(req), 0);
    if (n <= 0) break;
    if (n != sizeof(req)) continue;

    memset(&resp, 0, sizeof(resp));

    switch (req.cmd) {
      case CMD_MAP_BAR: {
        mach_vm_address_t addr;
        mach_vm_size_t size;
        if (map_bar(req.bar, &addr, &size) == 0) {
          resp.status = RESP_OK;
          resp.addr = addr;
          resp.value = size;
          // For BAR access, Python can't directly mmap IOKit memory
          // Instead, we provide the address and size - Python uses this shim for MMIO
          send_response(client_fd, &resp, -1);
        } else {
          resp.status = RESP_ERR;
          send_response(client_fd, &resp, -1);
        }
        break;
      }

      case CMD_MAP_SYSMEM: {
        mach_vm_address_t addr;
        mach_vm_size_t mapped_size;
        int idx;
        if (map_sysmem(req.size, &addr, &mapped_size, &idx) == 0) {
          resp.status = RESP_OK;
          resp.addr = idx;   // return index for bulk operations
          resp.value = mapped_size;
          send_response(client_fd, &resp, -1);
        } else {
          resp.status = RESP_ERR;
          send_response(client_fd, &resp, -1);
        }
        break;
      }

      case CMD_CFG_READ: {
        uint64_t val;
        if (cfg_read(req.offset, req.size, &val) == 0) {
          resp.status = RESP_OK;
          resp.value = val;
        } else {
          resp.status = RESP_ERR;
        }
        send_response(client_fd, &resp, -1);
        break;
      }

      case CMD_CFG_WRITE: {
        if (cfg_write(req.offset, req.size, req.value) == 0) {
          resp.status = RESP_OK;
        } else {
          resp.status = RESP_ERR;
        }
        send_response(client_fd, &resp, -1);
        break;
      }

      case CMD_RESET: {
        if (device_reset() == 0) {
          resp.status = RESP_OK;
        } else {
          resp.status = RESP_ERR;
        }
        send_response(client_fd, &resp, -1);
        break;
      }

      case CMD_MMIO_READ: {
        uint64_t val;
        if (mmio_read(req.bar, req.offset, req.size, &val) == 0) {
          resp.status = RESP_OK;
          resp.value = val;
        } else {
          resp.status = RESP_ERR;
        }
        send_response(client_fd, &resp, -1);
        break;
      }

      case CMD_MMIO_WRITE: {
        if (mmio_write(req.bar, req.offset, req.size, req.value) == 0) {
          resp.status = RESP_OK;
        } else {
          resp.status = RESP_ERR;
        }
        send_response(client_fd, &resp, -1);
        break;
      }

      case CMD_MMIO_BULK_READ: {
        // req.bar = BAR, req.offset = start offset, req.size = byte count
        uint8_t bar = req.bar;
        uint64_t off = req.offset;
        uint64_t len = req.size;

        if (bar >= MAX_BARS || !g_bars[bar].mapped || off + len > g_bars[bar].size || len > BULK_BUF_SIZE) {
          resp.status = RESP_ERR;
          send_response(client_fd, &resp, -1);
          break;
        }

        // MMIO requires aligned volatile accesses - can't use memcpy/memmove
        volatile uint32_t *src = (volatile uint32_t *)(g_bars[bar].addr + off);
        uint32_t *dst = (uint32_t *)g_bulk_buf;
        uint64_t words = len / 4;
        for (uint64_t i = 0; i < words; i++) dst[i] = src[i];

        // Handle remaining bytes with byte accesses
        if (len % 4) {
          volatile uint8_t *src_bytes = (volatile uint8_t *)(g_bars[bar].addr + off + words * 4);
          uint8_t *dst_bytes = g_bulk_buf + words * 4;
          for (uint64_t i = 0; i < len % 4; i++) dst_bytes[i] = src_bytes[i];
        }

        resp.status = RESP_OK;
        resp.value = len;
        send_response(client_fd, &resp, -1);
        send(client_fd, g_bulk_buf, len, 0);
        break;
      }

      case CMD_MMIO_BULK_WRITE: {
        // req.bar = BAR, req.offset = start offset, req.size = byte count
        // NO RESPONSE - fire and forget for speed
        uint8_t bar = req.bar;
        uint64_t off = req.offset;
        uint64_t len = req.size;

        if (bar >= MAX_BARS || !g_bars[bar].mapped || off + len > g_bars[bar].size || len > BULK_BUF_SIZE) {
          // drain the data even on error
          size_t remaining = len;
          while (remaining > 0) {
            ssize_t r = recv(client_fd, g_bulk_buf, remaining > BULK_BUF_SIZE ? BULK_BUF_SIZE : remaining, 0);
            if (r <= 0) break;
            remaining -= r;
          }
          break;
        }

        // recv all data first
        size_t received = 0;
        while (received < len) {
          ssize_t r = recv(client_fd, g_bulk_buf + received, len - received, 0);
          if (r <= 0) break;
          received += r;
        }

        // MMIO requires aligned volatile accesses - can't use memcpy/memmove
        volatile uint32_t *dst = (volatile uint32_t *)(g_bars[bar].addr + off);
        uint32_t *src = (uint32_t *)g_bulk_buf;
        uint64_t words = len / 4;
        for (uint64_t i = 0; i < words; i++) dst[i] = src[i];

        // Handle remaining bytes with byte accesses
        if (len % 4) {
          volatile uint8_t *dst_bytes = (volatile uint8_t *)(g_bars[bar].addr + off + words * 4);
          uint8_t *src_bytes = g_bulk_buf + words * 4;
          for (uint64_t i = 0; i < len % 4; i++) dst_bytes[i] = src_bytes[i];
        }
        // no response
        break;
      }

      case CMD_SYSMEM_BULK_READ: {
        // req.bar = sysmem index, req.offset = start offset, req.size = byte count
        uint8_t idx = req.bar;
        uint64_t off = req.offset;
        uint64_t len = req.size;

        if (idx >= g_sysmem_count || !g_sysmem[idx].mapped || off + len > g_sysmem[idx].size || len > BULK_BUF_SIZE) {
          resp.status = RESP_ERR;
          send_response(client_fd, &resp, -1);
          break;
        }

        resp.status = RESP_OK;
        resp.value = len;
        send_response(client_fd, &resp, -1);
        send(client_fd, (void *)(g_sysmem[idx].addr + off), len, 0);
        break;
      }

      case CMD_SYSMEM_BULK_WRITE: {
        // req.bar = sysmem index, req.offset = start offset, req.size = byte count
        // NO RESPONSE - fire and forget for speed
        uint8_t idx = req.bar;
        uint64_t off = req.offset;
        uint64_t len = req.size;

        if (idx >= g_sysmem_count || !g_sysmem[idx].mapped || off + len > g_sysmem[idx].size || len > BULK_BUF_SIZE) {
          // drain the data even on error
          size_t remaining = len;
          while (remaining > 0) {
            ssize_t r = recv(client_fd, g_bulk_buf, remaining > BULK_BUF_SIZE ? BULK_BUF_SIZE : remaining, 0);
            if (r <= 0) break;
            remaining -= r;
          }
          break;
        }

        // recv all data first
        size_t received = 0;
        while (received < len) {
          ssize_t r = recv(client_fd, (void *)(g_sysmem[idx].addr + off + received), len - received, 0);
          if (r <= 0) break;
          received += r;
        }

        // memcpy to sysmem
        // memcpy((void *)(g_sysmem[idx].addr + off), g_bulk_buf, len);
        // no response
        break;
      }

      case CMD_QUIT:
        printf("shim: quit requested\n");
        resp.status = RESP_OK;
        send_response(client_fd, &resp, -1);
        return;

      default:
        resp.status = RESP_ERR;
        send_response(client_fd, &resp, -1);
    }
  }

  printf("shim: client disconnected\n");

  // Cleanup sysmem allocations when client disconnects
  int cleaned = g_sysmem_count;
  for (int i = 0; i < g_sysmem_count; i++) {
    if (g_sysmem[i].mapped) {
      IOConnectUnmapMemory64(g_conn, g_sysmem[i].mem_type, mach_task_self(), g_sysmem[i].addr);
      g_sysmem[i].mapped = 0;
    }
  }
  g_sysmem_count = 0;
  printf("shim: cleaned up %d sysmem allocations\n", cleaned);
}

static void cleanup(void) {
  // Unmap all BARs
  for (int i = 0; i < MAX_BARS; i++) {
    if (g_bars[i].mapped) {
      IOConnectUnmapMemory64(g_conn, i, mach_task_self(), g_bars[i].addr);
      g_bars[i].mapped = 0;
    }
  }

  // Unmap all sysmem
  for (int i = 0; i < g_sysmem_count; i++) {
    if (g_sysmem[i].mapped) {
      IOConnectUnmapMemory64(g_conn, g_sysmem[i].mem_type, mach_task_self(), g_sysmem[i].addr);
      g_sysmem[i].mapped = 0;
    }
  }
  g_sysmem_count = 0;

  if (g_conn != IO_OBJECT_NULL) {
    IOServiceClose(g_conn);
    g_conn = IO_OBJECT_NULL;
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <socket_path>\n", argv[0]);
    return 1;
  }

  const char *sock_path = argv[1];

  // Connect to tinygpu driver
  g_conn = open_tinygpu();
  if (g_conn == IO_OBJECT_NULL) {
    fprintf(stderr, "shim: failed to connect to tinygpu\n");
    return 1;
  }

  printf("shim: connected to tinygpu\n");

  // Create Unix socket
  int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd < 0) {
    perror("socket");
    cleanup();
    return 1;
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

  unlink(sock_path);  // Remove if exists

  if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    perror("bind");
    close(server_fd);
    cleanup();
    return 1;
  }

  if (listen(server_fd, 1) < 0) {
    perror("listen");
    close(server_fd);
    cleanup();
    return 1;
  }

  printf("shim: listening on %s\n", sock_path);

  // Accept one client at a time
  while (1) {
    int client_fd = accept(server_fd, NULL, NULL);
    if (client_fd < 0) {
      if (errno == EINTR) continue;
      perror("accept");
      break;
    }
    handle_client(client_fd);
    close(client_fd);
  }

  close(server_fd);
  unlink(sock_path);
  cleanup();
  return 0;
}
