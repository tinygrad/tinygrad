#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <dispatch/dispatch.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOMessage.h>
#include <mach/mach.h>

// Protocol

enum {
  CMD_PROBE = 0,          // probe devices, returns count
  CMD_MAP_BAR = 1,        // map PCI BAR, returns size
  CMD_MAP_SYSMEM_FD = 2,  // alloc DMA memory, returns fd via SCM_RIGHTS
  CMD_CFG_READ = 3,       // read PCI config space
  CMD_CFG_WRITE = 4,      // write PCI config space
  CMD_RESET = 5,          // reset device
  CMD_MMIO_READ = 6,      // bulk read from BAR
  CMD_MMIO_WRITE = 7,     // bulk write to BAR
  CMD_MAP_SYSMEM = 8,     // map system memory
  CMD_SYSMEM_READ = 9,    // bulk read from system memory
  CMD_SYSMEM_WRITE = 10,  // bulk write to system memory
  CMD_RESIZE_BAR = 11,    // resize bar (noop)
  CMD_PING = 12,          // x1476 fork: returns server version (resp0) and dext version (resp1, 0 = upstream dext)
  RESP_OK = 0, RESP_ERR = 1,
};

// x1476 fork: CMD_RESET is synchronous. Request: arg0 = PCIe reset type (0 = the dext picks:
// hot reset for NVIDIA Blackwell, FLR otherwise), arg1 = reset options, arg2 = timeout in ms
// (0 = 30 s). tinygrad 0.14.0 sends zeros for all three. Response: resp0 = the dext's status
// word (TinyGPUResetOutcome in TinyGPUDriverUserClient.iig; low byte 0 = device ready),
// resp1 = RESET_FLAG_* below. A reset that cannot be completed answers RESP_ERR with a message,
// which tinygrad raises as RuntimeError("RPC failed: ...").
#define RESET_FLAG_REENUMERATED 1   // the device was re-probed; the service was reopened and BARs remapped
#define RESET_FLAG_UPSTREAM_DEXT 2  // the dext has no ResetWait; fell back to its FLR-first Reset + our own wait

#define TINYGPU_SERVER_VERSION 0x00010000u
#define RESET_DEFAULT_TIMEOUT_MS 30000
#define RESET_REOPEN_POLL_MS 100

// Dext user-client selectors (TinyGPURPC in TinyGPUDriverUserClient.iig)
enum { SEL_READ_CFG = 0, SEL_WRITE_CFG = 1, SEL_RESET = 2, SEL_PREPARE_DMA = 3, SEL_RESET_WAIT = 4, SEL_PING = 5 };
enum { RESET_OUTCOME_READY = 0, RESET_OUTCOME_TIMEOUT = 1, RESET_OUTCOME_VENDOR_MISMATCH = 2, RESET_OUTCOME_FAILED = 3, RESET_OUTCOME_TERMINATED = 4 };

typedef struct { uint8_t cmd; uint32_t dev_id, bar; uint64_t arg0, arg1, arg2; } __attribute__((packed)) request_t;
typedef struct { uint8_t status; uint64_t resp0, resp1; } __attribute__((packed)) response_t;

// Constants and state

#define BULK_BUF_SIZE (64 << 20)
#define MAX_BARS 6
#define MAX_SYSMEM 128

static uint8_t g_bulk_buf[BULK_BUF_SIZE];
static io_connect_t g_conn = IO_OBJECT_NULL;
static int g_client_active = 0;
// x1476 fork: set from the IOKit interest notification when the dext's service terminates
// (the GPU dropped off the bus, or a reset re-probed it). Upstream _exit(0)'d here; this
// server keeps running so a reset can reopen the service and the client gets an error
// instead of a dead socket.
static volatile int g_service_gone = 0;
static int g_resetting = 0;

static struct { mach_vm_address_t addr; mach_vm_size_t size; int mapped; } g_bars[MAX_BARS];
static struct { mach_vm_address_t addr; mach_vm_size_t size; int shm_fd; char shm_name[32]; } g_sysmem[MAX_SYSMEM];
static int g_sysmem_count = 0;

// Utilities

static void recvall(int fd, void *buf, size_t len) {
  for (size_t off = 0; off < len; ) {
    ssize_t r = recv(fd, (uint8_t*)buf + off, len - off, 0);
    if (r <= 0) break;
    off += r;
  }
}

// MMIO requires 32-bit aligned volatile accesses
static void mmio_copy(void *dst, void *src, size_t len) {
  volatile uint32_t *d = dst, *s = src;
  for (size_t i = 0; i < len / 4; i++) d[i] = s[i];
  for (size_t i = len & ~3; i < len; i++) ((volatile uint8_t*)dst)[i] = ((volatile uint8_t*)src)[i];
}

static int send_response(int fd, response_t *resp, int send_fd) {
  char cmsgbuf[CMSG_SPACE(sizeof(int))];
  struct iovec iov = {resp, sizeof(*resp)};
  struct msghdr msg = {.msg_iov = &iov, .msg_iovlen = 1};

  if (send_fd >= 0) {
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    *cmsg = (struct cmsghdr){.cmsg_level = SOL_SOCKET, .cmsg_type = SCM_RIGHTS, .cmsg_len = CMSG_LEN(sizeof(int))};
    memcpy(CMSG_DATA(cmsg), &send_fd, sizeof(int));
  }
  return sendmsg(fd, &msg, 0) > 0 ? 0 : -1;
}

static void send_error(int fd, const char *msg) {
  response_t resp = {.status = RESP_ERR, .resp0 = strlen(msg)};
  send_response(fd, &resp, -1);
  send(fd, msg, strlen(msg), 0);
}

// Driver interface

static void on_disconnect(void *refcon, io_service_t svc, uint32_t msg, void *arg) {
  if (msg != kIOMessageServiceIsTerminated) return;
  g_service_gone = 1;
  fprintf(stderr, "tinygpu: device service terminated%s\n", g_resetting ? " (during reset)" : "");
}

static io_connect_t open_tinygpu(void) {
  static IONotificationPortRef port;
  static io_object_t notif;
  io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceNameMatching("tinygpu"));
  if (!svc) return IO_OBJECT_NULL;

  // One interest notification per service instance: a re-probed GPU is a new service, and
  // the notification registered on the old one never fires again.
  if (!port) {
    port = IONotificationPortCreate(kIOMainPortDefault);
    IONotificationPortSetDispatchQueue(port, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0));
  }
  if (notif) { IOObjectRelease(notif); notif = IO_OBJECT_NULL; }
  IOServiceAddInterestNotification(port, svc, kIOGeneralInterest, on_disconnect, NULL, &notif);

  io_connect_t conn;
  kern_return_t kr = IOServiceOpen(svc, mach_task_self(), 0, &conn);
  IOObjectRelease(svc);
  if (kr == KERN_SUCCESS) g_service_gone = 0;
  return kr == KERN_SUCCESS ? conn : IO_OBJECT_NULL;
}

// The device is gone (or is being re-probed): drop the BAR mappings and the connection so
// nothing touches device memory that no longer exists. Which BARs were mapped is remembered
// so a reopen can map them again at the same indices.
static uint32_t cfg_read32(uint32_t off);

static void teardown_device(void) {
  for (int i = 0; i < MAX_BARS; i++)
    if (g_bars[i].addr) { IOConnectUnmapMemory64(g_conn, i, mach_task_self(), g_bars[i].addr); g_bars[i].addr = 0; g_bars[i].size = 0; }
  if (g_conn != IO_OBJECT_NULL) { IOServiceClose(g_conn); g_conn = IO_OBJECT_NULL; }
}

// Wait for a tinygpu service to be available again (the dext is relaunched for the re-probed
// device), open it and map the BARs the client had mapped, at the same indices.
static int reopen_device(uint32_t timeout_ms) {
  teardown_device();
  for (uint32_t waited = 0; ; waited += RESET_REOPEN_POLL_MS) {
    g_conn = open_tinygpu();
    if (g_conn != IO_OBJECT_NULL) {
      // Right after a re-probe the terminating old service and the new one can both match;
      // only a service whose device answers configuration reads is the one to keep.
      uint32_t vd = cfg_read32(0);
      if (vd != 0 && vd != 0xffffffffu && vd != 0xffff0001u) break;
      IOServiceClose(g_conn); g_conn = IO_OBJECT_NULL;
    }
    if (waited >= timeout_ms) { fprintf(stderr, "tinygpu: no live device service after %u ms\n", waited); return -1; }
    usleep(RESET_REOPEN_POLL_MS * 1000);
  }
  for (int i = 0; i < MAX_BARS; i++) {
    if (!g_bars[i].mapped) continue;
    if (IOConnectMapMemory64(g_conn, i, mach_task_self(), &g_bars[i].addr, &g_bars[i].size, kIOMapAnywhere)) {
      fprintf(stderr, "tinygpu: remapping BAR %d after reopen failed\n", i);
      g_bars[i].addr = 0; g_bars[i].size = 0;
      return -1;
    }
  }
  fprintf(stderr, "tinygpu: device service reopened, BARs remapped\n");
  return 0;
}

static int dext_ping(uint64_t *dext_version) {
  uint64_t out[2] = {0, 0};
  uint32_t out_cnt = 2;
  if (IOConnectCallMethod(g_conn, SEL_PING, NULL, 0, NULL, 0, out, &out_cnt, NULL, NULL) != KERN_SUCCESS) return -1;
  *dext_version = out[0];
  return 0;
}

static uint32_t cfg_read32(uint32_t off) {
  uint64_t in[2] = {off, 4}, out[2] = {0, 0};
  uint32_t out_cnt = 2;
  if (IOConnectCallMethod(g_conn, SEL_READ_CFG, in, 2, NULL, 0, out, &out_cnt, NULL, NULL) != KERN_SUCCESS) return 0xffffffffu;
  return (uint32_t)out[0];
}

// The whole reset: ask the dext to reset and wait, then handle the two ways the device can
// come back — as the same service (nothing to remap) or as a re-probed one (reopen + remap).
// Returns 0 with the status word and flags filled in, or -1 with a message for the client.
static int reset_device(uint32_t type, uint32_t options, uint32_t timeout_ms, uint64_t *status, uint64_t *flags, const char **msg) {
  *status = 0; *flags = 0; *msg = NULL;
  if (timeout_ms == 0) timeout_ms = RESET_DEFAULT_TIMEOUT_MS;
  if (g_sysmem_count > 0) { *msg = "reset refused: system-memory DMA mappings are live; reset before allocating"; return -1; }
  if (g_conn == IO_OBJECT_NULL || g_service_gone) {
    // The device already vanished (e.g. it dropped off the bus): try to pick the re-probed one up.
    if (reopen_device(timeout_ms)) { *msg = "reset failed: the device is gone and did not come back"; return -1; }
    *flags |= RESET_FLAG_REENUMERATED;
  }

  g_resetting = 1;
  uint64_t in[3] = {type, options, timeout_ms}, out[2] = {0, 0};
  uint32_t out_cnt = 2;
  kern_return_t kr = IOConnectCallMethod(g_conn, SEL_RESET_WAIT, in, 3, NULL, 0, out, &out_cnt, NULL, NULL);
  if (kr == kIOReturnUnsupported || kr == kIOReturnBadArgument) {
    // Upstream dext: only the FLR-first Reset exists. Use it, then wait for the device ourselves.
    *flags |= RESET_FLAG_UPSTREAM_DEXT;
    kr = IOConnectCallMethod(g_conn, SEL_RESET, NULL, 0, NULL, 0, NULL, NULL, NULL, NULL);
    uint32_t waited = 0, vd = 0xffffffffu;
    while (kr == KERN_SUCCESS && !g_service_gone && waited < timeout_ms) {
      vd = cfg_read32(0);
      if (vd != 0 && vd != 0xffffffffu && vd != 0xffff0001u) break;
      usleep(10 * 1000); waited += 10;
    }
    out[0] = (vd != 0 && vd != 0xffffffffu && vd != 0xffff0001u) ? (RESET_OUTCOME_READY | ((uint64_t)waited << 16)) : (RESET_OUTCOME_TIMEOUT | ((uint64_t)waited << 16));
  }
  fprintf(stderr, "tinygpu: reset type=%u options=%u -> kr=0x%x status=0x%llx gone=%d\n", type, options, kr, (unsigned long long)out[0], g_service_gone);

  if (g_service_gone || kr == MACH_SEND_INVALID_DEST || kr == kIOReturnNotAttached || kr == kIOReturnNoDevice ||
      (out[0] & 0xff) == RESET_OUTCOME_TERMINATED) {
    // Case B: the reset re-probed the device (or the link dropped and macOS re-enumerated it).
    if (reopen_device(timeout_ms)) { g_resetting = 0; *msg = "reset failed: the device did not come back after re-enumeration"; return -1; }
    *flags |= RESET_FLAG_REENUMERATED;
    uint32_t vd = cfg_read32(0);
    out[0] = (vd != 0 && vd != 0xffffffffu && vd != 0xffff0001u) ? RESET_OUTCOME_READY : RESET_OUTCOME_TIMEOUT;
  } else if (kr != KERN_SUCCESS) {
    g_resetting = 0; *msg = "reset failed: the driver extension rejected the reset"; return -1;
  }
  g_resetting = 0;
  *status = out[0];
  if ((out[0] & 0xff) != RESET_OUTCOME_READY) {
    static char buf[128];
    snprintf(buf, sizeof(buf), "reset did not bring the device back (outcome %llu after %llu ms)",
             (unsigned long long)(out[0] & 0xff), (unsigned long long)((out[0] >> 16) & 0xffff));
    *msg = buf;
    return -1;
  }
  return 0;
}

static int dext_rpc(uint32_t sel, uint64_t *in, uint32_t in_cnt, uint64_t *out_val) {
  uint64_t out[2];
  uint32_t out_cnt = 2;
  if (IOConnectCallMethod(g_conn, sel, in, in_cnt, NULL, 0, out, &out_cnt, NULL, NULL) != KERN_SUCCESS) return -1;
  if (out_val) *out_val = out[0];
  return 0;
}

static int map_bar(uint32_t bar, response_t *resp) {
  if (bar >= MAX_BARS) return -1;
  if (g_service_gone) return -1;
  if (!g_bars[bar].addr && IOConnectMapMemory64(g_conn, bar, mach_task_self(), &g_bars[bar].addr, &g_bars[bar].size, kIOMapAnywhere)) return -1;
  g_bars[bar].mapped = 1;
  resp->resp0 = g_bars[bar].addr;
  resp->resp1 = g_bars[bar].size;
  return 0;
}

static int map_sysmem_fd(uint64_t size, int contiguous, response_t *resp, int *out_fd) {
  if (g_sysmem_count >= MAX_SYSMEM) return -1;
  int idx = g_sysmem_count;
  int fd = -1;
  void *ptr = MAP_FAILED;
  char shm_name[32];

  // page-align, min 16KB for IOMemoryDescriptor
  size_t alloc_sz = (size + 0xfff) & ~0xfff;
  if (alloc_sz < 0x4000) alloc_sz = 0x4000;

  snprintf(shm_name, sizeof(shm_name), "/tinygpu_%d", idx);
  shm_unlink(shm_name);
  if ((fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600)) < 0) goto fail;
  if (ftruncate(fd, alloc_sz) < 0) goto fail;
  if ((ptr = mmap(NULL, alloc_sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED) goto fail;

  // PrepareDMA writes physical addresses to output buffer, copy to shared mem
  uint8_t paddr_buf[8192] = {0};
  size_t out_sz = sizeof(paddr_buf);
  if (IOConnectCallStructMethod(g_conn, 3, ptr, alloc_sz, paddr_buf, &out_sz) != KERN_SUCCESS) goto fail;
  memcpy(ptr, paddr_buf, out_sz);

  g_sysmem[idx] = (typeof(g_sysmem[idx])){.addr = (mach_vm_address_t)ptr, .size = alloc_sz, .shm_fd = fd};
  strncpy(g_sysmem[idx].shm_name, shm_name, sizeof(g_sysmem[idx].shm_name));
  g_sysmem_count++;

  *resp = (response_t){.resp0 = alloc_sz, .resp1 = idx};
  *out_fd = fd;
  return 0;

fail:
  if (ptr != MAP_FAILED) munmap(ptr, alloc_sz);
  if (fd >= 0) close(fd);
  shm_unlink(shm_name);
  return -1;
}

static int validate_bar(uint8_t bar, uint64_t off, uint64_t sz) {
  // A device that left the bus takes its BAR mappings with it: touching them would fault the
  // server, so the moment the service is gone every MMIO request is refused.
  if (g_service_gone) return -1;
  return (bar < MAX_BARS && g_bars[bar].addr && off + sz <= g_bars[bar].size && sz <= BULK_BUF_SIZE) ? 0 : -1;
}

static void cleanup(void) {
  for (int i = 0; i < MAX_BARS; i++)
    if (g_bars[i].addr) { IOConnectUnmapMemory64(g_conn, i, mach_task_self(), g_bars[i].addr); g_bars[i].addr = 0; g_bars[i].size = 0; }
  for (int i = 0; i < MAX_BARS; i++) g_bars[i].mapped = 0;

  for (int i = 0; i < g_sysmem_count; i++) {
    munmap((void*)g_sysmem[i].addr, g_sysmem[i].size);
    close(g_sysmem[i].shm_fd);
    shm_unlink(g_sysmem[i].shm_name);
  }

  g_sysmem_count = 0;
  if (g_conn != IO_OBJECT_NULL) { IOServiceClose(g_conn); g_conn = IO_OBJECT_NULL; }
}

static void handle_client(int fd) {
  int bufsize = BULK_BUF_SIZE;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
  printf("client connected\n");

  g_conn = open_tinygpu();
  if (g_conn == IO_OBJECT_NULL) {
    fprintf(stderr, "failed to connect to tinygpu driver\n");
    request_t req; recv(fd, &req, sizeof(req), 0);
    send_error(fd, "Driver not available. Check: System Report > PCI for GPU, System Settings > Privacy & Security.");
    return;
  }

  request_t req;
  response_t resp;
  while (recv(fd, &req, sizeof(req), 0) == sizeof(req)) {
    resp = (response_t){0};

    switch (req.cmd) {
    case CMD_MAP_BAR:
      resp.status = map_bar(req.bar, &resp) ? 1 : 0;
      break;

    case CMD_MAP_SYSMEM_FD: {
      int shm_fd = -1;
      resp.status = map_sysmem_fd(req.arg0, (int)req.arg1, &resp, &shm_fd) ? 1 : 0;
      send_response(fd, &resp, shm_fd);
      continue;
    }

    case CMD_CFG_READ: {
      if (g_service_gone) { send_error(fd, "device lost: the GPU left the bus (reset or power-cycle the enclosure)"); continue; }
      uint64_t in[2] = {req.arg0, req.arg1};
      resp.status = dext_rpc(SEL_READ_CFG, in, 2, &resp.resp0) ? 1 : 0;
      break;
    }

    case CMD_CFG_WRITE: {
      if (g_service_gone) { send_error(fd, "device lost: the GPU left the bus (reset or power-cycle the enclosure)"); continue; }
      uint64_t in[3] = {req.arg0, req.arg1, req.arg2};
      resp.status = dext_rpc(SEL_WRITE_CFG, in, 3, NULL) ? 1 : 0;
      break;
    }

    case CMD_RESIZE_BAR:
      break;

    case CMD_RESET: {
      uint64_t status = 0, flags = 0;
      const char *msg = NULL;
      if (reset_device((uint32_t)req.arg0, (uint32_t)req.arg1, (uint32_t)req.arg2, &status, &flags, &msg)) {
        send_error(fd, msg ? msg : "reset failed");
        continue;
      }
      resp.resp0 = status;
      resp.resp1 = flags;
      break;
    }

    case CMD_PING: {
      uint64_t dext_version = 0;
      if (g_conn != IO_OBJECT_NULL && !g_service_gone) dext_ping(&dext_version);
      resp.resp0 = TINYGPU_SERVER_VERSION;
      resp.resp1 = dext_version;
      break;
    }

    case CMD_MMIO_READ:
      if (validate_bar(req.bar, req.arg0, req.arg1)) {
        send_error(fd, g_service_gone ? "device lost: the GPU left the bus (reset or power-cycle the enclosure)" : "invalid BAR access");
        continue;
      }
      mmio_copy(g_bulk_buf, (void*)(g_bars[req.bar].addr + req.arg0), req.arg1);
      resp.resp0 = req.arg1;
      send_response(fd, &resp, -1);
      send(fd, g_bulk_buf, req.arg1, 0);
      continue;

    case CMD_MMIO_WRITE:
      recvall(fd, g_bulk_buf, req.arg1);
      if (!validate_bar(req.bar, req.arg0, req.arg1))
        mmio_copy((void*)(g_bars[req.bar].addr + req.arg0), g_bulk_buf, req.arg1);
      continue;

    default:
      resp.status = 1;
    }
    send_response(fd, &resp, -1);
  }

  printf("client disconnected\n");
  cleanup();
}

int run_server(const char *sock_path) {
  int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd < 0) { perror("socket"); return 1; }

  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
  unlink(sock_path);

  if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); close(server_fd); return 1; }
  if (listen(server_fd, 1) < 0) { perror("listen"); close(server_fd); return 1; }
  printf("listening on %s\n", sock_path);

  while (1) {
    int client_fd = accept(server_fd, NULL, NULL);
    if (client_fd < 0) { if (errno == EINTR) continue; perror("accept"); break; }
    if (g_client_active) { printf("rejected: client already connected\n"); close(client_fd); continue; }
    g_client_active = 1;
    handle_client(client_fd);
    g_client_active = 0;
    close(client_fd);
  }

  close(server_fd);
  unlink(sock_path);
  cleanup();
  return 0;
}
